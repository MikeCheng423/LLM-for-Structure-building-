#!/usr/bin/env python3
"""Fail-closed promotion decision for the journal-role adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _live_smoke(smoke_dir: Path) -> dict:
    """Read the free-running CLI package; offline metrics alone cannot promote.

    journal-role-r1 scored 100% schema-valid on the teacher-forced harness and
    still failed the first live request, so the smoke is a promotion check.
    """
    result = {"directory": str(smoke_dir), "passed": False, "reason": "smoke package is missing"}
    if not smoke_dir.is_dir():
        return result
    packages = [path for path in sorted(smoke_dir.iterdir()) if path.is_dir()]
    if len(packages) != 1:
        result["reason"] = f"expected exactly one request package, found {len(packages)}"
        return result
    package = packages[0]
    result["package"] = str(package)
    if (package / "failure_record.json").is_file():
        failure = json.loads((package / "failure_record.json").read_text(encoding="utf-8"))
        result["reason"] = f"{failure.get('stage')}: {failure.get('message')}"
        return result
    packet_path = package / "review_packet.json"
    if not packet_path.is_file():
        result["reason"] = "no review_packet.json"
        return result
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    result["status"] = packet.get("status")
    if packet.get("status") != "review_ready":
        result["reason"] = f"review packet status is {packet.get('status')!r}"
        return result
    return {**result, "passed": True, "reason": "live request reached review_ready"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--smoke-dir", type=Path, required=True,
        help="Request package written by a live ASE_catalyst_build run with this adapter.",
    )
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    adapter = json.loads(args.adapter.read_text(encoding="utf-8"))
    base = baseline.get("summary", {})
    tuned = adapter.get("summary", {})
    same_records = baseline.get("evaluation_ids_sha256") == adapter.get("evaluation_ids_sha256")
    smoke = _live_smoke(args.smoke_dir)
    checks = {
        "reports_complete": baseline.get("complete") is True and adapter.get("complete") is True,
        "matched_records": same_records,
        "minimum_300_records": tuned.get("record_count", 0) >= 300,
        "schema_valid_at_least_98_percent": tuned.get("schema_valid_rate", 0.0) >= 0.98,
        "executable_or_safe_at_least_95_percent": tuned.get("executable_or_safe_rate", 0.0) >= 0.95,
        "negative_safety_100_percent": tuned.get("negative_safety_rate", 0.0) == 1.0,
        "zero_forbidden_actions": tuned.get("forbidden_action_rate", 1.0) == 0.0,
        # A hardcoded 0.0 must not be able to satisfy the check above.
        "forbidden_actions_measured": tuned.get("forbidden_action_measured") is True,
        "live_smoke_passed": smoke["passed"],
        "schema_no_regression": tuned.get("schema_valid_rate", 0.0) >= base.get("schema_valid_rate", 0.0),
        "execution_no_regression": tuned.get("executable_or_safe_rate", 0.0) >= base.get("executable_or_safe_rate", 0.0),
        "field_recall_improved": tuned.get("field_recall", 0.0) > base.get("field_recall", 0.0),
        "provenance_precision_at_least_95_percent": tuned.get("provenance_precision", 0.0) >= 0.95,
        "provenance_recall_at_least_95_percent": tuned.get("provenance_recall", 0.0) >= 0.95,
    }
    report = {
        "promoted": all(checks.values()), "checks": checks,
        "baseline_report": str(args.baseline), "adapter_report": str(args.adapter),
        "baseline_summary": base, "adapter_summary": tuned, "live_smoke": smoke,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["promoted"]:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("corpus_contract") != "journal":
            raise SystemExit("refusing to mark a non-journal adapter as journal-role ready")
        expected_adapter = (args.manifest.parent / "adapter").resolve()
        if Path(tuned.get("adapter", "")).resolve() != expected_adapter:
            raise SystemExit("adapter report does not match the run manifest")
        manifest["journal_role_ready"] = True
        manifest["journal_role_promotion"] = {
            "report": str(args.output), "evaluation_ids_sha256": adapter["evaluation_ids_sha256"],
            "checks": checks,
        }
        temporary = args.manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["promoted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
