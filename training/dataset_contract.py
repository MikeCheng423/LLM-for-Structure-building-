"""Validation and rendering for versioned ASE-agent training trajectories."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "ase-agent-trajectory/v1"
JOURNAL_SCHEMA_VERSION = "journal-agent-trajectory/v1"

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"Authorization:\s*Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"(?:^|/)POTCAR(?:$|/)", re.IGNORECASE),
    re.compile(r"/home/[^/\s]+/\.ssh/"),
)


class DatasetContractError(ValueError):
    """Raised when a trajectory cannot enter training."""


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _require(condition: bool, record_id: str, message: str) -> None:
    if not condition:
        raise DatasetContractError(f"{record_id}: {message}")


def validate_record(record: dict[str, Any], *, allow_smoke: bool = False) -> None:
    """Reject unsafe, incomplete, or unvalidated trajectory records."""

    record_id = record.get("id", "<missing-id>")
    _require(isinstance(record_id, str) and bool(record_id.strip()), record_id, "invalid id")
    _require(record.get("schema_version") == SCHEMA_VERSION, record_id, "wrong schema_version")
    _require(isinstance(record.get("split_group"), str), record_id, "missing split_group")

    tools = record.get("tools")
    _require(isinstance(tools, list) and bool(tools), record_id, "tools must be a non-empty list")
    tool_names: set[str] = set()
    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name")
        _require(tool.get("type") == "function", record_id, "only function tools are allowed")
        _require(isinstance(name, str) and bool(name), record_id, "tool is missing a name")
        _require(name not in tool_names, record_id, f"duplicate tool {name}")
        _require(isinstance(function.get("parameters"), dict), record_id, f"{name} lacks parameters")
        tool_names.add(name)

    messages = record.get("messages")
    _require(isinstance(messages, list) and len(messages) >= 3, record_id, "too few messages")
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    _require(len(roles) == len(messages), record_id, "message must be an object")
    _require(roles[0] == "system", record_id, "first message must be system")
    _require("user" in roles and "assistant" in roles, record_id, "user/assistant turn missing")
    _require(set(roles) <= {"system", "user", "assistant", "tool"}, record_id, "unknown role")

    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls", [])
        _require(isinstance(tool_calls, list), record_id, "tool_calls must be a list")
        for call in tool_calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = function.get("name")
            _require(name in tool_names, record_id, f"call to unregistered tool {name!r}")
            _require(isinstance(function.get("arguments"), dict), record_id, "tool arguments must be an object")

    provenance = record.get("provenance", {})
    _require(provenance.get("sanitized") is True, record_id, "record is not marked sanitized")
    _require(
        provenance.get("contains_private_structure") is False,
        record_id,
        "private structures are forbidden",
    )

    smoke_only = record.get("smoke_only") is True
    if smoke_only:
        _require(allow_smoke, record_id, "smoke-only data is forbidden in production training")
    else:
        validation = record.get("validation", {})
        _require(validation.get("schema_valid") is True, record_id, "schema validation missing")
        _require(validation.get("executed") is True, record_id, "recipe was not executed")
        _require(validation.get("invariants_passed") is True, record_id, "invariants did not pass")
        _require(validation.get("forbidden_action_count") == 0, record_id, "forbidden action present")
        recipe_hash = validation.get("recipe_hash", "")
        _require(
            isinstance(recipe_hash, str) and re.fullmatch(r"[0-9a-f]{64}", recipe_hash) is not None,
            record_id,
            "invalid recipe_hash",
        )
        _require(isinstance(validation.get("registry_version"), str), record_id, "registry version missing")

    for value in _strings(record):
        for pattern in _SECRET_PATTERNS:
            _require(pattern.search(value) is None, record_id, "possible secret or private file detected")


def validate_journal_record(record: dict[str, Any]) -> None:
    """Reject journal-role targets that lack deterministic validation proof."""
    record_id = record.get("id", "<missing-id>")
    _require(isinstance(record_id, str) and bool(record_id.strip()), record_id, "invalid id")
    _require(record.get("schema_version") == JOURNAL_SCHEMA_VERSION, record_id, "wrong schema_version")
    _require(isinstance(record.get("split_group"), str), record_id, "missing split_group")
    role = record.get("role")
    expected_tool = {
        "evidence_extractor": "submit_evidence_ledger",
        "spec_planner": "submit_spec_proposal",
    }.get(role)
    _require(expected_tool is not None, record_id, "invalid journal role")

    tools = record.get("tools")
    _require(isinstance(tools, list) and len(tools) == 1, record_id, "exactly one tool is required")
    function = tools[0].get("function", {}) if isinstance(tools[0], dict) else {}
    _require(tools[0].get("type") == "function", record_id, "only function tools are allowed")
    _require(function.get("name") == expected_tool, record_id, "role/tool mismatch")
    _require(isinstance(function.get("parameters"), dict), record_id, "tool lacks parameters")

    messages = record.get("messages")
    _require(isinstance(messages, list) and len(messages) == 3, record_id, "expected one system/user/assistant turn")
    _require([item.get("role") for item in messages] == ["system", "user", "assistant"], record_id, "invalid message sequence")
    calls = messages[-1].get("tool_calls", [])
    _require(isinstance(calls, list) and len(calls) == 1, record_id, "exactly one target tool call is required")
    call_function = calls[0].get("function", {}) if isinstance(calls[0], dict) else {}
    _require(call_function.get("name") == expected_tool, record_id, "target tool mismatch")
    _require(isinstance(call_function.get("arguments"), dict), record_id, "tool arguments must be an object")

    provenance = record.get("provenance", {})
    _require(provenance.get("sanitized") is True, record_id, "record is not marked sanitized")
    _require(provenance.get("contains_private_structure") is False, record_id, "private structures are forbidden")
    validation = record.get("validation", {})
    _require(validation.get("schema_valid") is True, record_id, "schema validation missing")
    _require(validation.get("policy_valid") is True, record_id, "policy validation missing")
    _require(validation.get("forbidden_action_count") == 0, record_id, "forbidden action present")
    _require(isinstance(validation.get("registry_version"), str), record_id, "registry version missing")
    payload_hash = validation.get("payload_hash", "")
    _require(isinstance(payload_hash, str) and re.fullmatch(r"[0-9a-f]{64}", payload_hash) is not None, record_id, "invalid payload_hash")
    expected_hash = hashlib.sha256(json.dumps(call_function["arguments"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _require(payload_hash == expected_hash, record_id, "payload_hash mismatch")
    _require(isinstance(record.get("reference"), dict), record_id, "validated reference is missing")

    for value in _strings(record):
        for pattern in _SECRET_PATTERNS:
            _require(pattern.search(value) is None, record_id, "possible secret or private file detected")


def load_jsonl(path: Path, *, allow_smoke: bool = False) -> tuple[list[dict[str, Any]], str]:
    """Load and validate a JSONL file, returning records and its SHA-256."""

    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetContractError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise DatasetContractError(f"line {line_number}: record must be an object")
        validate_record(record, allow_smoke=allow_smoke)
        records.append(record)
    if not records:
        raise DatasetContractError("dataset is empty")
    return records, hashlib.sha256(raw).hexdigest()


def load_journal_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load a validated journal-role JSONL split."""
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetContractError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise DatasetContractError(f"line {line_number}: record must be an object")
        validate_journal_record(record)
        records.append(record)
    if not records:
        raise DatasetContractError("dataset is empty")
    return records, hashlib.sha256(raw).hexdigest()


def render_assistant_only(
    tokenizer: Any,
    record: dict[str, Any],
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Render a tool trajectory and mask all non-assistant tokens from loss."""

    messages = record["messages"]
    tools = record["tools"]
    full_ids = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
    )
    if len(full_ids) > max_length:
        raise DatasetContractError(
            f"{record['id']}: rendered length {len(full_ids)} exceeds max_length {max_length}"
        )

    labels = [-100] * len(full_ids)
    previous_ids: list[int] = []
    for index, message in enumerate(messages):
        prefix_ids = tokenizer.apply_chat_template(
            messages[: index + 1],
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise DatasetContractError(f"{record['id']}: chat template prefixes are unstable")
        if message["role"] == "assistant":
            labels[len(previous_ids) : len(prefix_ids)] = full_ids[len(previous_ids) : len(prefix_ids)]
        previous_ids = prefix_ids

    if all(label == -100 for label in labels):
        raise DatasetContractError(f"{record['id']}: no assistant tokens available for loss")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }
