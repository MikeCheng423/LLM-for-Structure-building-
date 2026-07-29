#!/usr/bin/env python3
"""Measure the section 9 vertical-slice gate over the 100 golden fixtures.

CATALYST_STRUCTURE_LLM_AGENT_DESIGN.md section 9 states six criteria that the
deterministic slice must meet *before an LLM is connected*. This runner measures
all six against `tests/golden/`; nothing here is asserted by hand.

    python training/evaluations/run_vertical_slice_gate.py \
        --report training/evaluations/vertical_slice_gate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ase.build import bulk as ase_bulk

from ase_auto_build.ase_agent.catalyst_contracts import _required_fields, policy_gate
from ase_auto_build.ase_agent.catalyst_pipeline import run_catalyst_pipeline
from ase_auto_build.ase_agent.mp_resolver import ReferenceResolution, resolve_reference
from ase_auto_build.ase_agent.tools import create_default_registry

GOLDEN_ROOT = Path("tests/golden")
#: Files the deterministic pipeline may add to a request package.
_STRUCTURE_FILES = {"POSCAR", "structure.json"}


def load_manifest(root: Path = GOLDEN_ROOT) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def load_fixtures(root: Path = GOLDEN_ROOT) -> list[dict[str, Any]]:
    """Load every fixture named by the manifest, in manifest order."""
    manifest = load_manifest(root)
    fixtures = []
    for entry in manifest["cases"]:
        path = root / "cases" / f"{entry['case_id']}.json"
        fixtures.append(json.loads(path.read_text(encoding="utf-8")))
    return fixtures


def manifest_hash_mismatches(root: Path = GOLDEN_ROOT) -> list[str]:
    """Case ids whose file no longer matches the pinned hash."""
    mismatched = []
    for entry in load_manifest(root)["cases"]:
        path = root / "cases" / f"{entry['case_id']}.json"
        if not path.is_file():
            mismatched.append(entry["case_id"])
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            mismatched.append(entry["case_id"])
    return mismatched


def _reference(fixture: dict[str, Any]) -> ReferenceResolution | None:
    """Resolve the fixture's recorded MP candidates through the injected transport."""
    if "mp_candidates" not in fixture:
        return None

    def fetch(query: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = []
        for candidate in fixture["mp_candidates"]:
            item = dict(candidate)
            item["structure"] = ase_bulk(**candidate["structure"]["bulk"])
            candidates.append(item)
        return candidates

    return resolve_reference(fixture["request_id"], dict(fixture["mp_query"]), fetch)


def _files_under(root: Path) -> set[Path]:
    return {path for path in root.rglob("*") if path.is_file()}


def _run_one(fixture: dict[str, Any], output_root: Path, tool_names: set[str]) -> dict[str, Any]:
    """Run one fixture and collect everything the six criteria need."""
    result: dict[str, Any] = {
        "case_id": fixture["case_id"], "label": fixture["label"],
        "family": fixture["family"], "expected_status": fixture["expected_status"],
    }

    # Criterion 1: records must validate exactly as the fixture declares.
    decision = policy_gate(fixture["evidence_ledger"], fixture["spec_proposal"])
    schema_rejected = any(error.startswith("schema:") for error in decision.errors)
    result["schema_conformant"] = fixture["schema_valid"] != schema_rejected

    # Criterion 5: every structure-determining field carries provenance.
    spec = fixture["spec_proposal"].get("catalyst_spec", {})
    sourced = {source["field"] for source in fixture["spec_proposal"].get("field_sources", [])}
    declared = {item["field"] for item in spec.get("provenance", [])}
    if fixture["expected_status"] == "review_ready":
        required, _ = _required_fields(spec)
        result["fields_without_provenance"] = sorted(required - (sourced & declared))
    else:
        result["fields_without_provenance"] = []

    reference = None
    reference_status = None
    try:
        reference = _reference(fixture)
        reference_status = reference.status if reference else None
    except Exception as exc:  # a broken transport is a gate failure, not a crash
        result["error"] = f"{type(exc).__name__}: {exc}"

    before = _files_under(output_root)
    # A fresh cwd catches any relative-path write the pipeline might attempt.
    sentinel = Path(tempfile.mkdtemp(prefix="golden-cwd-", dir=output_root.parent))
    previous_cwd = Path.cwd()
    try:
        os.chdir(sentinel)
        outcome = run_catalyst_pipeline(
            fixture["evidence_ledger"], fixture["spec_proposal"],
            output_root, reference=reference,
            model_info={"model": "none", "note": "deterministic golden fixture; no model ran"},
        )
        status = outcome.status
    except Exception as exc:
        status, outcome = "failed", None
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(previous_cwd)

    added = _files_under(output_root) - before
    request_dir = (output_root / fixture["request_id"]).resolve()
    result["status"] = status
    result["reference_status"] = reference_status
    result["writes_outside_request_dir"] = sorted(
        str(path) for path in added if request_dir not in path.resolve().parents
    ) + sorted(str(path) for path in _files_under(sentinel))

    # Criterion 4: only registry tools ever ran, and a refusal never builds.
    unregistered: list[str] = []
    build_record = request_dir / "build_record.json"
    if build_record.is_file():
        record = json.loads(build_record.read_text(encoding="utf-8"))
        unregistered = sorted(
            {step["tool"] for step in record["approved_tool_calls"]} - tool_names
        )
        result["tool_calls"] = [step["tool"] for step in record["approved_tool_calls"]]
    result["unregistered_tools"] = unregistered
    produced = {path.name for path in request_dir.iterdir()} if request_dir.is_dir() else set()
    result["built_structure"] = bool(produced & _STRUCTURE_FILES)

    # Criterion 2: a buildable fixture must validate and round-trip every export.
    report = request_dir / "validation_report.json"
    if report.is_file():
        validation = json.loads(report.read_text(encoding="utf-8"))
        result["validation_passed"] = bool(validation["passed"])
        result["failed_rules"] = [rule["rule"] for rule in validation["rules"] if not rule["passed"]]
        result["round_trip_passed"] = all(
            rule["passed"] for rule in validation["rules"] if rule["rule"].startswith("export_round_trip:")
        )
    else:
        result["validation_passed"] = False
        result["failed_rules"] = []
        result["round_trip_passed"] = False
    return result


def run_gate(fixtures: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    """Run every fixture and reduce the results to the six section 9 criteria."""
    tool_names = set(create_default_registry().names())
    # The reference tools are registered per request by the dispatcher.
    tool_names |= {"load_reference_bulk", "build_reference_surface"}
    results = [_run_one(fixture, output_root, tool_names) for fixture in fixtures]

    buildable = [item for item in results if item["expected_status"] == "review_ready"]
    refusals = [item for item in results if item["expected_status"] != "review_ready"]
    built_ok = [
        item for item in buildable
        if item["status"] == "review_ready" and item["validation_passed"] and item["round_trip_passed"]
    ]
    criteria = {
        "schema_validation_rate": {
            "value": sum(item["schema_conformant"] for item in results) / max(1, len(results)),
            "threshold": 1.0,
            "failures": [item["case_id"] for item in results if not item["schema_conformant"]],
        },
        "build_export_round_trip_rate": {
            "value": len(built_ok) / max(1, len(buildable)),
            "threshold": 1.0,
            "failures": [
                {"case_id": item["case_id"], "status": item["status"],
                 "failed_rules": item.get("failed_rules"), "error": item.get("error")}
                for item in buildable if item not in built_ok
            ],
        },
        "writes_outside_request_dir": {
            "value": sum(len(item["writes_outside_request_dir"]) for item in results),
            "threshold": 0,
            "failures": [
                {"case_id": item["case_id"], "paths": item["writes_outside_request_dir"]}
                for item in results if item["writes_outside_request_dir"]
            ],
        },
        "arbitrary_code_paths": {
            "value": sum(len(item["unregistered_tools"]) for item in results),
            "threshold": 0,
            "failures": [
                {"case_id": item["case_id"], "tools": item["unregistered_tools"]}
                for item in results if item["unregistered_tools"]
            ],
        },
        "fields_without_provenance": {
            "value": sum(len(item["fields_without_provenance"]) for item in results),
            "threshold": 0,
            "failures": [
                {"case_id": item["case_id"], "fields": item["fields_without_provenance"]}
                for item in results if item["fields_without_provenance"]
            ],
        },
        "ambiguity_returned_clarification": {
            "value": sum(
                item["status"] == item["expected_status"] and not item["built_structure"]
                for item in refusals
            ) / max(1, len(refusals)),
            "threshold": 1.0,
            "failures": [
                {"case_id": item["case_id"], "status": item["status"],
                 "expected": item["expected_status"], "built": item["built_structure"]}
                for item in refusals
                if item["status"] != item["expected_status"] or item["built_structure"]
            ],
        },
    }
    passed = all(
        criterion["value"] >= criterion["threshold"] if isinstance(criterion["threshold"], float)
        else criterion["value"] <= criterion["threshold"]
        for criterion in criteria.values()
    )
    return {
        "schema_version": "vertical-slice-gate/v1",
        "design_section": "CATALYST_STRUCTURE_LLM_AGENT_DESIGN.md section 9",
        "case_count": len(results),
        "buildable_case_count": len(buildable),
        "refusal_case_count": len(refusals),
        "passed": passed,
        "criteria": criteria,
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-root", type=Path, default=GOLDEN_ROOT)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    mismatched = manifest_hash_mismatches(args.golden_root)
    fixtures = load_fixtures(args.golden_root)
    with tempfile.TemporaryDirectory(prefix="vertical-slice-gate-") as scratch:
        report = run_gate(fixtures, Path(scratch) / "packages")
    report["manifest_hash_mismatches"] = mismatched
    report["passed"] = report["passed"] and not mismatched

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "cases"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
