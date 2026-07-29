from __future__ import annotations

import copy
import hashlib

import pytest

from training.dataset_contract import (
    DatasetContractError, SCHEMA_VERSION, validate_journal_record, validate_record,
)
from training.generators.build_journal_corpus import _negative_records


def valid_record() -> dict:
    return {
        "id": "bulk-al-1",
        "schema_version": SCHEMA_VERSION,
        "split_group": "bulk-fcc-al",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "build_bulk",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "Use registered tools."},
            {"role": "user", "content": "Build bulk Al."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "build_bulk", "arguments": {"element": "Al"}},
                    }
                ],
            },
        ],
        "provenance": {
            "source": "canonical_generator",
            "sanitized": True,
            "contains_private_structure": False,
        },
        "validation": {
            "schema_valid": True,
            "executed": True,
            "invariants_passed": True,
            "forbidden_action_count": 0,
            "recipe_hash": hashlib.sha256(b"recipe").hexdigest(),
            "registry_version": "v1",
        },
    }


def test_valid_production_record_passes() -> None:
    validate_record(valid_record())


@pytest.mark.parametrize("field", ["schema_valid", "executed", "invariants_passed"])
def test_missing_validation_proof_is_rejected(field: str) -> None:
    record = valid_record()
    record["validation"][field] = False
    with pytest.raises(DatasetContractError):
        validate_record(record)


def test_smoke_record_requires_explicit_permission() -> None:
    record = valid_record()
    record["smoke_only"] = True
    with pytest.raises(DatasetContractError, match="smoke-only"):
        validate_record(record)
    validate_record(record, allow_smoke=True)


def test_unregistered_tool_call_is_rejected() -> None:
    record = valid_record()
    record["messages"][2]["tool_calls"][0]["function"]["name"] = "shell"
    with pytest.raises(DatasetContractError, match="unregistered"):
        validate_record(record)


def test_secret_marker_is_rejected() -> None:
    record = copy.deepcopy(valid_record())
    record["messages"][1]["content"] = "token sk-abcdefghijklmnopqrstuvwxyz123456"
    with pytest.raises(DatasetContractError, match="possible secret"):
        validate_record(record)


def test_journal_contract_rejects_role_tool_mismatch() -> None:
    record = _negative_records(0)[0]
    validate_journal_record(record)
    record["role"] = "spec_planner"
    with pytest.raises(DatasetContractError, match="role/tool mismatch"):
        validate_journal_record(record)
