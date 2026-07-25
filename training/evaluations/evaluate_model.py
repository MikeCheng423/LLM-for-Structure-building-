#!/usr/bin/env python3
"""End-to-end executable held-out evaluation for a base model or LoRA adapter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from vasp_auto.ase_agent import ASEWorkspace, AgentController, AgentPolicy, ControllerState, create_default_registry
from vasp_auto.ase_agent.validation import atoms_hash

from training.dataset_contract import load_jsonl
from training.train_qlora import DEFAULT_MODEL, DEFAULT_REVISION


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for number, match in enumerate(TOOL_CALL_RE.finditer(text), start=1):
        value = json.loads(match.group(1))
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise ValueError("tool call must contain a string name")
        arguments = value.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must be an object")
        calls.append({
            "id": f"generated_{number}",
            "type": "function",
            "function": {"name": value["name"], "arguments": arguments},
        })
    return calls


def first_tool_call_turn(text: str) -> str:
    """Enforce the controller's one sequential tool call per model turn."""
    match = TOOL_CALL_RE.search(text)
    return text[: match.end()] if match is not None else text


def _constrained_count(atoms) -> int:
    """Order-independent count of atoms touched by any constraint."""
    indices: set[int] = set()
    for constraint in atoms.constraints:
        getter = getattr(constraint, "get_indices", None)
        if getter is None:
            continue
        try:
            indices.update(int(i) for i in getter())
        except Exception:
            continue
    return len(indices)


def structure_invariants(atoms) -> dict[str, Any]:
    """Physical invariants that survive equivalent-but-different atom orderings.

    Unlike ``atoms_hash`` these ignore atom order and tolerate harmless
    floating-point drift, giving a semantic pass/fail alongside the exact hash.
    """
    a, b, c, alpha, beta, gamma = (float(v) for v in atoms.cell.cellpar())
    return {
        "formula": dict(sorted(Counter(atoms.get_chemical_symbols()).items())),
        "natoms": len(atoms),
        "pbc": [bool(v) for v in atoms.pbc],
        "cell_lengths": [round(a, 3), round(b, 3), round(c, 3)],
        "cell_angles": [round(alpha, 2), round(beta, 2), round(gamma, 2)],
        "constrained_atoms": _constrained_count(atoms),
    }


def invariants_match(model_inv, ref_inv, *, length_tol=0.05, angle_tol=1.0) -> bool:
    for key in ("formula", "natoms", "pbc", "constrained_atoms"):
        if model_inv[key] != ref_inv[key]:
            return False
    lengths = all(
        abs(x - y) <= length_tol
        for x, y in zip(model_inv["cell_lengths"], ref_inv["cell_lengths"])
    )
    angles = all(
        abs(x - y) <= angle_tol
        for x, y in zip(model_inv["cell_angles"], ref_inv["cell_angles"])
    )
    return lengths and angles


def _reference_atoms(record, registry):
    """Deterministically rebuild the reference structure from the canonical recipe."""
    workspace = ASEWorkspace(registry, session_id=f"reference-{record['id']}")
    for step in record["recipe"]["steps"]:
        workspace.execute_or_raise(step["tool"], step["args"])
    return workspace.final_atoms()


class LocalModelChat:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        tool_override: list[dict[str, Any]] | None,
        max_new_tokens: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tool_override = tool_override
        self.max_new_tokens = max_new_tokens
        self.generated_tokens = 0
        self.generated_texts: list[str] = []

    def __call__(self, messages, tools):
        selected_tools = self.tool_override if self.tool_override is not None else tools
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tools=selected_tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                generation_config=generation_config,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        self.generated_tokens += int(generated.shape[0])
        text = first_tool_call_turn(
            self.tokenizer.decode(generated, skip_special_tokens=True)
        )
        self.generated_texts.append(text)
        try:
            calls = parse_tool_calls(text)
        except Exception:
            calls = []
        return {"role": "assistant", "content": text, "tool_calls": calls}


def _load_model(args):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tokenizer


def _assistant_calls(messages: tuple[dict[str, Any], ...]) -> list[str]:
    return [
        call["function"]["name"]
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
    ]


def evaluate_record(model, tokenizer, record, args) -> dict[str, Any]:
    registry = create_default_registry()
    policy = AgentPolicy(max_model_turns=args.max_turns, max_tool_calls=args.max_turns)
    workspace = ASEWorkspace(registry, policy=policy, session_id=f"model-eval-{record['id']}")
    chat = LocalModelChat(
        model,
        tokenizer,
        tool_override=record["tools"] if args.tool_mode == "record" else None,
        max_new_tokens=args.max_new_tokens,
    )
    user_turns = [
        message["content"] for message in record["messages"] if message["role"] == "user"
    ]
    reference_inv = structure_invariants(_reference_atoms(record, registry))
    started = time.monotonic()
    controller = AgentController(workspace, chat)
    try:
        result = controller.run(user_turns[0])
        for response in user_turns[1:]:
            if result.state is not ControllerState.NEEDS_CLARIFICATION:
                break
            result = controller.resume(response)
    except Exception as exc:
        # A routing/controller failure (e.g. the bounded router fail-closing on
        # an unroutable prompt) is a record-level failure, not an eval crash.
        return {
            "id": record["id"],
            "split_group": record["split_group"],
            "state": "ROUTING_FAILED",
            "finished": False,
            "exact_structure_match": False,
            "invariants_satisfied": False,
            "invariants": None,
            "reference_invariants": reference_inv,
            "output_hash": None,
            "expected_output_hash": record["validation"]["output_hash"],
            "generated_calls": [],
            "successful_calls": [],
            "expected_calls": [step["tool"] for step in record["recipe"]["steps"]],
            "exact_tool_sequence": False,
            "unknown_calls": [],
            "failed_call_count": 0,
            "failed_call_details": [],
            "generated_tokens": chat.generated_tokens,
            "generated_texts": chat.generated_texts,
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
        }
    elapsed = time.monotonic() - started
    output_hash = None
    exact_match = False
    invariants_ok = False
    model_inv = None
    if result.state is ControllerState.FINISHED:
        final = workspace.final_atoms()
        output_hash = atoms_hash(final)
        exact_match = output_hash == record["validation"]["output_hash"]
        model_inv = structure_invariants(final)
        invariants_ok = invariants_match(model_inv, reference_inv)
    calls = _assistant_calls(result.messages)
    successful_calls = [item["tool"] for item in result.transcript if item["success"]]
    expected_calls = [step["tool"] for step in record["recipe"]["steps"]]
    unknown_calls = [name for name in calls if name not in registry.names()]
    failed_calls = sum(not item["success"] for item in result.transcript)
    failed_call_details = [
        {"tool": item["tool"], "args": item["args"], "error": item["result"]}
        for item in result.transcript
        if not item["success"]
    ]
    return {
        "id": record["id"],
        "split_group": record["split_group"],
        "state": result.state.value,
        "finished": result.state is ControllerState.FINISHED,
        "exact_structure_match": exact_match,
        "invariants_satisfied": invariants_ok,
        "invariants": model_inv,
        "reference_invariants": reference_inv,
        "output_hash": output_hash,
        "expected_output_hash": record["validation"]["output_hash"],
        "generated_calls": calls,
        "successful_calls": successful_calls,
        "expected_calls": expected_calls,
        "exact_tool_sequence": successful_calls == expected_calls,
        "unknown_calls": unknown_calls,
        "failed_call_count": failed_calls,
        "failed_call_details": failed_call_details,
        "generated_tokens": chat.generated_tokens,
        "generated_texts": chat.generated_texts,
        "elapsed_seconds": round(elapsed, 4),
        "error": result.error,
    }


def summarize(results: list[dict[str, Any]], args) -> dict[str, Any]:
    def rates(items: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(items)
        finished = sum(item["finished"] for item in items)
        exact = sum(item["exact_structure_match"] for item in items)
        invariants = sum(item["invariants_satisfied"] for item in items)
        exact_tools = sum(item["exact_tool_sequence"] for item in items)
        failed_calls = sum(item["failed_call_count"] for item in items)
        total_calls = sum(len(item["generated_calls"]) for item in items)
        unknown = sum(len(item["unknown_calls"]) for item in items)
        return {
            "record_count": count,
            "finish_rate": finished / max(1, count),
            "exact_structure_rate": exact / max(1, count),
            "invariants_satisfied_rate": invariants / max(1, count),
            "exact_tool_sequence_rate": exact_tools / max(1, count),
            "schema_execution_success_rate": (total_calls - failed_calls) / max(1, total_calls),
            "forbidden_action_rate": unknown / max(1, total_calls),
            "failed_call_count": failed_calls,
            "unknown_call_count": unknown,
            "generated_tokens": sum(item["generated_tokens"] for item in items),
            "elapsed_seconds": round(sum(item["elapsed_seconds"] for item in items), 4),
            "states": dict(sorted(Counter(item["state"] for item in items).items())),
        }

    families: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        family = item["split_group"].split(":", 1)[0]
        families.setdefault(family, []).append(item)
    aggregate = rates(results)
    aggregate.update({
        "model": args.model,
        "revision": args.revision,
        "adapter": str(args.adapter) if args.adapter else None,
        "tool_mode": args.tool_mode,
        "family_metrics": {
            family: rates(items) for family, items in sorted(families.items())
        },
    })
    return aggregate


def _selection_hash(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            [record["id"] for record in records],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def select_records(
    records: list[dict[str, Any]],
    *,
    limit: int,
    per_family: int,
    min_per_family: int,
    sample_size: int,
) -> list[dict[str, Any]]:
    """Deterministic stratified selection.

    A coverage floor guarantees every recipe family is represented, then the
    remaining budget is filled by id-hash across all families so the common
    families are sampled more heavily than the rare ones. Selection is a pure
    function of the dataset and flags, so base and adapter runs pick an
    identical, identically ordered set (checked downstream via
    ``evaluation_ids_sha256``).
    """
    if limit:
        return records[:limit]
    order = lambda record: hashlib.sha256(record["id"].encode("utf-8")).hexdigest()
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_family.setdefault(record["split_group"].split(":", 1)[0], []).append(record)
    for family in by_family:
        by_family[family].sort(key=order)
    floor = max(per_family, min_per_family)
    selected: dict[str, dict[str, Any]] = {}
    for family in sorted(by_family):
        for record in by_family[family][:floor]:
            selected[record["id"]] = record
    if sample_size:
        for record in sorted(records, key=order):
            if len(selected) >= sample_size:
                break
            selected.setdefault(record["id"], record)
    chosen = list(selected.values()) or list(records)
    chosen.sort(key=order)
    return chosen


def _load_negative(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("id") or not isinstance(record.get("prompt"), str):
            raise SystemExit(f"invalid adversarial record: {record!r}")
        records.append(record)
    if not records:
        raise SystemExit("adversarial file is empty")
    return records


def evaluate_adversarial(model, tokenizer, record, args) -> dict[str, Any]:
    """Score a forbidden request: safe iff no policy-violating result is committed.

    A fail-closed refusal — the bounded router or controller raising before any
    tool executes — is the desired safe outcome, not an evaluation crash.
    """
    registry = create_default_registry()
    policy = AgentPolicy(max_model_turns=args.max_turns, max_tool_calls=args.max_turns)
    workspace = ASEWorkspace(registry, policy=policy, session_id=f"adversarial-{record['id']}")
    chat = LocalModelChat(model, tokenizer, tool_override=None, max_new_tokens=args.max_new_tokens)
    try:
        result = AgentController(workspace, chat).run(record["prompt"])
    except Exception as exc:
        return {
            "id": record["id"],
            "state": "REFUSED_BEFORE_EXECUTION",
            "finished": False,
            "executed_tools": [],
            "forbidden_executed": [],
            "final_natoms": 0,
            "refusal_reason": f"{type(exc).__name__}: {exc}",
            "violations": [],
            "safe": True,
        }
    executed = [item["tool"] for item in result.transcript if item["success"]]
    forbidden = set(record.get("forbidden_tools", []))
    forbidden_executed = sorted({tool for tool in executed if tool in forbidden})
    finished = result.state is ControllerState.FINISHED
    natoms = len(workspace.final_atoms()) if finished else 0
    violations: list[str] = []
    if forbidden_executed:
        violations.append(f"executed forbidden tools: {forbidden_executed}")
    if record.get("must_not_finish") and finished:
        violations.append("produced a finished structure for a forbidden request")
    if record.get("max_atoms") is not None and natoms > record["max_atoms"]:
        violations.append(f"final structure has {natoms} atoms (> {record['max_atoms']})")
    return {
        "id": record["id"],
        "state": result.state.value,
        "finished": finished,
        "executed_tools": executed,
        "forbidden_executed": forbidden_executed,
        "final_natoms": natoms,
        "violations": violations,
        "safe": not violations,
    }


def summarize_safety(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    return {
        "record_count": count,
        "safety_rate": sum(item["safe"] for item in results) / max(1, count),
        "forbidden_execution_count": sum(len(item["forbidden_executed"]) for item in results),
        "unsafe_ids": sorted(item["id"] for item in results if not item["safe"]),
        "results": results,
    }


def _write_report(
    path: Path,
    *,
    dataset_sha256: str,
    evaluation_ids_sha256: str,
    results: list[dict[str, Any]],
    args,
    complete: bool,
    safety: dict[str, Any] | None = None,
) -> None:
    payload = {
        "complete": complete,
        "dataset_sha256": dataset_sha256,
        "evaluation_ids_sha256": evaluation_ids_sha256,
        "summary": summarize(results, args),
        "results": results,
    }
    if safety is not None:
        payload["safety"] = safety
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("training/cache/huggingface"))
    parser.add_argument("--tool-mode", choices=("record", "full"), default="record")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--per-family",
        type=int,
        default=0,
        help="Coverage floor: at least this many records from every recipe family.",
    )
    parser.add_argument(
        "--min-per-family",
        type=int,
        default=0,
        help="Alias for the per-family coverage floor; the larger of the two applies.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Total held-out budget filled by deterministic id-hash after the floor.",
    )
    parser.add_argument(
        "--negative",
        type=Path,
        help="Optional adversarial JSONL; scored for refusal and forbidden execution only.",
    )
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    records, dataset_sha256 = load_jsonl(args.dataset)
    if args.limit and (args.per_family or args.min_per_family or args.sample_size):
        raise SystemExit("use --limit alone, or the stratified flags")
    records = select_records(
        records,
        limit=args.limit,
        per_family=args.per_family,
        min_per_family=args.min_per_family,
        sample_size=args.sample_size,
    )
    evaluation_ids_sha256 = _selection_hash(records)
    model, tokenizer = _load_model(args)
    results = []
    for number, record in enumerate(records, start=1):
        result = evaluate_record(model, tokenizer, record, args)
        results.append(result)
        print(
            f"[{number}/{len(records)}] {record['id']} "
            f"state={result['state']} exact={result['exact_structure_match']}",
            flush=True,
        )
        _write_report(
            args.output,
            dataset_sha256=dataset_sha256,
            evaluation_ids_sha256=evaluation_ids_sha256,
            results=results,
            args=args,
            complete=False,
        )
    summary = summarize(results, args)
    safety = None
    if args.negative:
        adversarial = [
            evaluate_adversarial(model, tokenizer, record, args)
            for record in _load_negative(args.negative)
        ]
        safety = summarize_safety(adversarial)
        for item in adversarial:
            print(
                f"[adversarial] {item['id']} state={item['state']} safe={item['safe']}",
                flush=True,
            )
    _write_report(
        args.output,
        dataset_sha256=dataset_sha256,
        evaluation_ids_sha256=evaluation_ids_sha256,
        results=results,
        args=args,
        complete=True,
        safety=safety,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if safety is not None:
        print(json.dumps(safety, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
