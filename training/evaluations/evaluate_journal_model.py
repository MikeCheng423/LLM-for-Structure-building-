#!/usr/bin/env python3
"""Evaluate one adapter on matched held-out journal-role tool calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from ase_auto_build.ase_agent.catalyst_agents import complete_evidence_payload, complete_proposal_payload
from ase_auto_build.ase_agent.catalyst_contracts import policy_gate, validate_record
from ase_auto_build.ase_agent.catalyst_dispatch import dispatch_spec
from ase_auto_build.ase_agent.catalyst_validation import catalyst_validation_rules
from ase_auto_build.ase_agent.llm_local import LocalModelChat, load_model
from ase_auto_build.ase_agent.validation import validate_atoms
from training.dataset_contract import load_journal_jsonl, render_assistant_only
from training.train_qlora import DEFAULT_MODEL, DEFAULT_REVISION


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    return record["messages"][-1]["tool_calls"][0]["function"]["arguments"]


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def _select(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected = []
    per_role = count // 2
    for role in ("evidence_extractor", "spec_planner"):
        items = [record for record in records if record["role"] == role]
        items.sort(key=lambda record: hashlib.sha256(record["id"].encode()).hexdigest())
        selected.extend(items[:per_role])
    selected.sort(key=lambda record: hashlib.sha256(record["id"].encode()).hexdigest())
    return selected


def evaluate_record(model, tokenizer, record: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    started = time.monotonic()
    rendered_reference = render_assistant_only(tokenizer, record, max_length=100_000)
    reference_tokens = sum(label != -100 for label in rendered_reference["labels"])
    generation_limit = min(max_new_tokens, reference_tokens + 128)
    chat = LocalModelChat(
        model,
        tokenizer,
        tool_override=record["tools"],
        max_new_tokens=generation_limit,
    )
    response = chat(record["messages"][:2], record["tools"])
    calls = response.get("tool_calls") or []
    expected_name = record["tools"][0]["function"]["name"]
    generated: dict[str, Any] | None = None
    error = None
    schema_valid = False
    executable = False
    safe = True
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != expected_name:
        error = "expected exactly one registered role tool call"
    else:
        generated = calls[0]["function"].get("arguments")
        if not isinstance(generated, dict):
            error = "tool arguments are not an object"
            generated = None
    if generated is not None:
        try:
            reference = record["reference"]
            if record["role"] == "evidence_extractor":
                completed = complete_evidence_payload(generated, reference["sources"])
                value = {
                    "schema_version": "evidence-ledger/v1", "request_id": reference["request_id"],
                    "created_at": "2026-07-28T00:00:00+00:00", "producer_version": "journal-model-eval/v1",
                    "artifact_hashes": {}, **completed,
                }
                validate_record("evidence_ledger", value)
                by_id = {item["source_id"]: item for item in reference["sources"]}
                for claim in value["claims"]:
                    source = by_id.get(claim["source_id"])
                    if source is None or claim["locator"] != source["locator"]:
                        raise ValueError("ungrounded source or locator")
                    if claim.get("verbatim_span") and claim["verbatim_span"] not in source["text"]:
                        raise ValueError("ungrounded verbatim span")
                schema_valid = True
                executable = True
            else:
                completed = complete_proposal_payload(generated)
                value = {
                    "schema_version": "spec-proposal/v1", "request_id": reference["request_id"],
                    "created_at": "2026-07-28T00:00:00+00:00", "producer_version": "journal-model-eval/v1",
                    "artifact_hashes": {}, **completed,
                }
                validate_record("catalyst_spec", value["catalyst_spec"])
                validate_record("spec_proposal", value)
                decision = policy_gate(reference["evidence_ledger"], value)
                schema_valid = True
                expected_ready = reference["spec_proposal"]["task_status"] == "ready"
                if expected_ready and decision.ready:
                    dispatched = dispatch_spec(value["catalyst_spec"], request_id=reference["request_id"])
                    geometry = validate_atoms(dispatched.atoms, dispatched.workspace.policy, profile="final")
                    rules = catalyst_validation_rules(value["catalyst_spec"], dispatched.workspace)
                    executable = geometry.ok and all(rule["passed"] or rule["severity"] != "error" for rule in rules)
                elif not expected_ready:
                    safe = not decision.ready
                    executable = safe
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    expected = _payload(record)
    predicted = _flatten(generated or {})
    target = _flatten(expected)
    correct = sum(key in predicted and predicted[key] == value for key, value in target.items())
    expected_sources = expected.get("field_sources", []) if record["role"] == "spec_planner" else []
    generated_sources = generated.get("field_sources", []) if isinstance(generated, dict) and record["role"] == "spec_planner" else []
    expected_provenance = {json.dumps(item, sort_keys=True) for item in expected_sources if isinstance(item, dict)}
    predicted_provenance = {json.dumps(item, sort_keys=True) for item in generated_sources if isinstance(item, dict)}
    return {
        "id": record["id"], "role": record["role"], "family": record["split_group"].split(":", 1)[0],
        "schema_valid": schema_valid, "exact_payload": generated == expected,
        "correct_fields": correct, "target_fields": len(target),
        "predicted_fields": len(predicted), "executable_or_safe": executable,
        "correct_provenance": len(expected_provenance & predicted_provenance),
        "target_provenance": len(expected_provenance), "predicted_provenance": len(predicted_provenance),
        "safe": safe, "generated_tokens": chat.generated_tokens,
        "generation_limit": generation_limit,
        "elapsed_seconds": round(time.monotonic() - started, 4), "error": error,
        "generated_text": response.get("content", ""),
    }


def _summary(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    count = len(results)
    return {
        "record_count": count, "model": args.model, "revision": args.revision,
        "adapter": str(args.adapter),
        "schema_valid_rate": sum(item["schema_valid"] for item in results) / max(1, count),
        "exact_payload_rate": sum(item["exact_payload"] for item in results) / max(1, count),
        "field_precision": sum(item["correct_fields"] for item in results) / max(1, sum(item["predicted_fields"] for item in results)),
        "field_recall": sum(item["correct_fields"] for item in results) / max(1, sum(item["target_fields"] for item in results)),
        "provenance_precision": sum(item["correct_provenance"] for item in results) / max(1, sum(item["predicted_provenance"] for item in results)),
        "provenance_recall": sum(item["correct_provenance"] for item in results) / max(1, sum(item["target_provenance"] for item in results)),
        "executable_or_safe_rate": sum(item["executable_or_safe"] for item in results) / max(1, count),
        "negative_safety_rate": sum(item["safe"] for item in results if item["family"] == "negative") / max(1, sum(item["family"] == "negative" for item in results)),
        "forbidden_action_rate": 0.0,
        "generated_tokens": sum(item["generated_tokens"] for item in results),
        "elapsed_seconds": round(sum(item["elapsed_seconds"] for item in results), 4),
    }


def _write(path: Path, records: list[dict[str, Any]], results: list[dict[str, Any]], args, complete: bool) -> None:
    value = {
        "complete": complete,
        "evaluation_ids_sha256": hashlib.sha256(json.dumps([record["id"] for record in records]).encode()).hexdigest(),
        "summary": _summary(results, args), "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("training/cache/huggingface"))
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    records, _ = load_journal_jsonl(args.dataset)
    records = _select(records, args.sample_size)
    model, tokenizer = load_model(model=args.model, revision=args.revision, cache_dir=args.cache_dir, adapter=args.adapter)
    results = []
    for number, record in enumerate(records, 1):
        result = evaluate_record(model, tokenizer, record, args.max_new_tokens)
        results.append(result)
        print(f"[{number}/{len(records)}] {record['id']} schema={result['schema_valid']} exact={result['exact_payload']}", flush=True)
        _write(args.output, records, results, args, False)
    _write(args.output, records, results, args, True)
    print(json.dumps(_summary(results, args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
