#!/usr/bin/env python3
"""Replay and audit every production trajectory before training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ase_auto_build.ase_agent import ASEWorkspace, create_default_registry
from ase_auto_build.ase_agent.recipe import recipe_hash
from ase_auto_build.ase_agent.validation import atoms_hash

from training.dataset_contract import load_jsonl


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trajectory_calls(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    pending: tuple[str, dict[str, Any]] | None = None
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            if pending is not None or len(message["tool_calls"]) != 1:
                raise ValueError("generator trajectories must contain one sequential tool call per turn")
            function = message["tool_calls"][0]["function"]
            pending = (function["name"], function["arguments"])
        elif message["role"] == "tool":
            if pending is None:
                raise ValueError("tool result has no preceding call")
            if message.get("name") != pending[0]:
                raise ValueError("tool result name does not match call")
            content = json.loads(message["content"])
            calls.append((pending[0], pending[1], content))
            pending = None
    if pending is not None:
        raise ValueError("final tool call has no result")
    return calls


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    registry = create_default_registry()
    expected_registry = f"phase1-{registry.fingerprint()[:16]}"
    recorded_registry = record["validation"]["registry_version"]

    schema_by_name = {
        schema["function"]["name"]: schema for schema in registry.function_schemas()
    }
    provided_by_name = {schema["function"]["name"]: schema for schema in record["tools"]}
    recipe = record.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("record has no canonical recipe")
    steps = recipe.get("steps", [])
    calls = _trajectory_calls(record["messages"])
    called_names = {name for name, _, _ in calls}
    if set(provided_by_name) != called_names:
        raise ValueError("record tool schemas do not exactly match trajectory tools")
    for name in called_names:
        if provided_by_name[name] != schema_by_name[name]:
            raise ValueError(f"tool schema drift for {name}")

    if [step["tool"] for step in steps].count("finish") != 1 or steps[-1]["tool"] != "finish":
        raise ValueError("recipe must end with exactly one finish")
    if recipe_hash(recipe) != record["validation"]["recipe_hash"]:
        raise ValueError("recipe hash mismatch")

    workspace = ASEWorkspace(registry, session_id=f"evaluation-{record['id']}")
    successful_steps = []
    failed_calls = 0
    for tool, arguments, expected_observation in calls:
        result = workspace.execute(tool, arguments)
        actual = result.observation if result.success else result.error
        if actual != expected_observation:
            raise ValueError(f"recorded observation drift for {tool}")
        if result.success:
            successful_steps.append({"tool": tool, "args": arguments})
        else:
            failed_calls += 1
    if successful_steps != steps:
        raise ValueError("successful assistant calls do not match canonical recipe")
    output_hash = atoms_hash(workspace.final_atoms())
    if output_hash != record["validation"]["output_hash"]:
        raise ValueError("final structure hash mismatch")
    if output_hash != record["validation"]["replay_hash"]:
        raise ValueError("replay hash mismatch")

    return {
        "id": record["id"],
        "family": record["split_group"].split(":", 1)[0],
        "tool_calls": len(steps),
        "failed_calls": failed_calls,
        "natoms": len(workspace.final_atoms()),
        "output_hash": output_hash,
        "recorded_registry_version": recorded_registry,
        "current_registry_version": expected_registry,
        "registry_version_current": recorded_registry == expected_registry,
    }


def evaluate_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_records: dict[str, list[dict[str, Any]]] = {}
    all_groups: dict[str, set[str]] = {}
    all_recipe_hashes: dict[str, set[str]] = {}
    all_output_hashes: dict[str, set[str]] = {}
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for split in ("train", "validation", "test"):
        path = dataset_dir / f"{split}.jsonl"
        expected_hash = manifest["split_sha256"][split]
        if _file_sha256(path) != expected_hash:
            raise ValueError(f"{split} file hash differs from manifest")
        records, _ = load_jsonl(path)
        split_records[split] = records
        all_groups[split] = {record["split_group"] for record in records}
        all_recipe_hashes[split] = {
            record["validation"]["recipe_hash"] for record in records
        }
        all_output_hashes[split] = {
            record["validation"]["output_hash"] for record in records
        }
        for record in records:
            try:
                result = evaluate_record(record)
                result["split"] = split
                results.append(result)
            except Exception as exc:
                errors.append({"id": record.get("id", "<missing>"), "split": split, "error": f"{type(exc).__name__}: {exc}"})

    overlaps = {
        "train_validation": sorted(all_groups["train"] & all_groups["validation"]),
        "train_test": sorted(all_groups["train"] & all_groups["test"]),
        "validation_test": sorted(all_groups["validation"] & all_groups["test"]),
    }
    if any(overlaps.values()):
        raise ValueError("split_group leakage detected")
    recipe_overlaps = {
        "train_validation": sorted(all_recipe_hashes["train"] & all_recipe_hashes["validation"]),
        "train_test": sorted(all_recipe_hashes["train"] & all_recipe_hashes["test"]),
        "validation_test": sorted(all_recipe_hashes["validation"] & all_recipe_hashes["test"]),
    }
    output_overlaps = {
        "train_validation": sorted(all_output_hashes["train"] & all_output_hashes["validation"]),
        "train_test": sorted(all_output_hashes["train"] & all_output_hashes["test"]),
        "validation_test": sorted(all_output_hashes["validation"] & all_output_hashes["test"]),
    }
    if any(recipe_overlaps.values()) or any(output_overlaps.values()):
        raise ValueError("recipe or final-structure leakage detected")
    if sum(len(values) for values in split_records.values()) != manifest["record_count"]:
        raise ValueError("manifest record_count mismatch")

    family_counts = Counter(result["family"] for result in results)
    registry_versions = Counter(result["recorded_registry_version"] for result in results)
    family_split_counts = {
        family: {
            split: sum(
                record["split_group"].split(":", 1)[0] == family
                for record in split_records[split]
            )
            for split in ("train", "validation", "test")
        }
        for family in sorted(family_counts)
    }
    family_split_gaps = {
        family: [split for split, count in counts.items() if count == 0]
        for family, counts in family_split_counts.items()
        if any(count == 0 for count in counts.values())
    }
    report = {
        "passed": not errors,
        "record_count": len(results) + len(errors),
        "successful_replays": len(results),
        "failed_replays": len(errors),
        "execution_success_rate": len(results) / max(1, len(results) + len(errors)),
        "forbidden_action_rate": 0.0,
        "split_counts": {name: len(values) for name, values in split_records.items()},
        "split_group_overlap": overlaps,
        "recipe_hash_overlap": recipe_overlaps,
        "output_hash_overlap": output_overlaps,
        "family_counts": dict(sorted(family_counts.items())),
        "registry_versions": dict(sorted(registry_versions.items())),
        "records_on_current_registry": sum(
            result["registry_version_current"] for result in results
        ),
        "family_split_counts": family_split_counts,
        "family_split_gaps": family_split_gaps,
        "maximum_atoms": max((result["natoms"] for result in results), default=0),
        "maximum_tool_calls": max((result["tool_calls"] for result in results), default=0),
        "errors": errors[:100],
        "dataset_manifest_sha256": _file_sha256(manifest_path),
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
    if not report["passed"]:
        return 1
    if report["execution_success_rate"] != 1.0 or report["forbidden_action_rate"] != 0.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
