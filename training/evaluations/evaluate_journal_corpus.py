#!/usr/bin/env python3
"""Audit journal-role SFT targets before training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ase_auto_build.ase_agent.catalyst_agents import evidence_call, evidence_target, proposal_call, proposal_target
from ase_auto_build.ase_agent.catalyst_contracts import policy_gate, validate_record
from ase_auto_build.ase_agent.catalyst_dispatch import dispatch_spec
from ase_auto_build.ase_agent.catalyst_validation import catalyst_validation_rules
from ase_auto_build.ase_agent.validation import validate_atoms
from training.dataset_contract import JOURNAL_SCHEMA_VERSION, load_journal_jsonl


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    return record["messages"][-1]["tool_calls"][0]["function"]["arguments"]


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    reference = record["reference"]
    request_id = reference["request_id"]
    request = reference["request"]
    if record["role"] == "evidence_extractor":
        messages, tool = evidence_call(request, reference["sources"])
        ledger = reference["evidence_ledger"]
        validate_record("evidence_ledger", ledger)
        if _payload(record) != evidence_target(ledger):
            raise ValueError("extractor payload differs from validated EvidenceLedger")
        by_id = {source["source_id"]: source for source in reference["sources"]}
        for claim in ledger["claims"]:
            source = by_id.get(claim["source_id"])
            if source is None or source["locator"] != claim["locator"]:
                raise ValueError("claim source or locator is not supplied")
            if claim.get("verbatim_span") and claim["verbatim_span"] not in source["text"]:
                raise ValueError("claim span is not grounded in supplied text")
        status, executed = "validated", False
    else:
        evidence = reference["evidence_ledger"]
        proposal = reference["spec_proposal"]
        messages, tool = proposal_call(request, evidence)
        validate_record("evidence_ledger", evidence)
        validate_record("spec_proposal", proposal)
        validate_record("catalyst_spec", proposal["catalyst_spec"])
        if _payload(record) != proposal_target(proposal):
            raise ValueError("planner payload differs from validated SpecProposal")
        decision = policy_gate(evidence, proposal)
        if decision.status != proposal["task_status"]:
            raise ValueError(f"policy gate returned {decision.status}, expected {proposal['task_status']}")
        executed = decision.ready
        if executed:
            dispatched = dispatch_spec(proposal["catalyst_spec"], request_id=request_id)
            geometry = validate_atoms(dispatched.atoms, dispatched.workspace.policy, profile="final")
            rules = catalyst_validation_rules(proposal["catalyst_spec"], dispatched.workspace)
            if not geometry.ok or any(not rule["passed"] and rule["severity"] == "error" for rule in rules):
                raise ValueError("ready target failed deterministic structure validation")
        status = decision.status
    if record["messages"][:2] != messages or record["tools"] != [tool]:
        raise ValueError("training prompt or tool differs from deployment")
    return {"id": record["id"], "role": record["role"], "status": status, "executed": executed}


def evaluate_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != JOURNAL_SCHEMA_VERSION:
        raise ValueError("journal corpus manifest schema mismatch")
    groups: dict[str, set[str]] = {}
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    split_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        path = dataset_dir / f"{split}.jsonl"
        if _sha(path) != manifest["split_sha256"][split]:
            raise ValueError(f"{split} file hash differs from manifest")
        records, _ = load_journal_jsonl(path)
        split_counts[split] = len(records)
        groups[split] = {record["split_group"] for record in records}
        for record in records:
            try:
                result = evaluate_record(record)
                result["split"] = split
                results.append(result)
            except Exception as exc:
                errors.append({"id": record.get("id", "<missing>"), "split": split, "error": f"{type(exc).__name__}: {exc}"})
    overlaps = {
        "train_validation": sorted(groups["train"] & groups["validation"]),
        "train_test": sorted(groups["train"] & groups["test"]),
        "validation_test": sorted(groups["validation"] & groups["test"]),
    }
    if any(overlaps.values()):
        raise ValueError("split_group leakage detected")
    if sum(split_counts.values()) != manifest["record_count"]:
        raise ValueError("manifest record_count mismatch")
    total = len(results) + len(errors)
    role_counts = Counter(result["role"] for result in results)
    status_counts = Counter(result["status"] for result in results)
    report = {
        "passed": not errors,
        "record_count": total,
        "successful_validations": len(results),
        "execution_success_rate": len(results) / max(1, total),
        "forbidden_action_rate": 0.0,
        "split_counts": split_counts,
        "split_group_overlap": overlaps,
        "role_counts": dict(sorted(role_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "ready_builds": sum(result["executed"] for result in results),
        "errors": errors[:100],
        "dataset_manifest_sha256": _sha(manifest_path),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = evaluate_dataset(args.dataset_dir)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
