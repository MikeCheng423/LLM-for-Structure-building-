"""Regression test for the journal-role prompt/train key-order mismatch.

`training/generators/build_journal_corpus.py` writes corpus records with
`json.dumps(item, sort_keys=True, ...)`, so every stored `tools` entry has
alphabetically ordered keys once reloaded. `tokenizer.apply_chat_template`
renders that tool JSON schema into the prompt verbatim, so the model only ever
trained on prompts whose tool properties were sorted. At inference,
`evidence_call()` / `proposal_call()` built the tool dict in Python insertion
order and with top-level keys `["type", "function"]` -- a prompt shape the
model never trained on. It then mirrored that unfamiliar order in its own
output and dropped fields such as `catalyst_spec.modifications`.

The fix canonicalizes the tool dict returned by both call-builders with
`json.loads(json.dumps(value, sort_keys=True))` before it is handed back, so
the rendered prompt is byte-identical to what the corpus baked in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ase_auto_build.ase_agent.catalyst_agents import evidence_call, proposal_call

DATASET_TEST_FILE = Path("training/datasets/journal_roles_v1/test.jsonl")


def _assert_sorted_order(value, path: str = "$") -> None:
    """Recursively assert every dict along the way has alphabetically sorted keys."""
    if isinstance(value, dict):
        keys = list(value.keys())
        assert keys == sorted(keys), f"keys not sorted at {path}: {keys}"
        for key, sub in value.items():
            _assert_sorted_order(sub, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_sorted_order(item, f"{path}[{index}]")


def test_evidence_call_tool_has_sorted_key_order() -> None:
    _, tool = evidence_call("Build it", [{
        "source_id": "fixture-1", "locator": "Methods, p. 1", "text": "fcc Cu bulk cell.",
    }])
    _assert_sorted_order(tool)
    assert list(tool.keys()) == sorted(tool.keys())
    assert list(tool["function"]["parameters"]["properties"].keys()) == sorted(
        tool["function"]["parameters"]["properties"].keys()
    )


def test_proposal_call_tool_has_sorted_key_order() -> None:
    evidence = {
        "claims": [{
            "field": "material.formula", "value": "Cu", "evidence_type": "user_supplied",
            "source_id": "fixture-1",
        }],
        "contradictions": [], "unresolved_fields": [],
    }
    _, tool = proposal_call("Build it", evidence)
    _assert_sorted_order(tool)
    assert list(tool.keys()) == sorted(tool.keys())
    assert list(tool["function"]["parameters"]["properties"].keys()) == sorted(
        tool["function"]["parameters"]["properties"].keys()
    )


def _first_record(role: str) -> dict | None:
    if not DATASET_TEST_FILE.is_file():
        return None
    with DATASET_TEST_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("role") == role:
                return record
    return None


def test_proposal_call_matches_stored_training_tool_byte_for_byte() -> None:
    record = _first_record("spec_planner")
    if record is None:
        pytest.skip("training/datasets/journal_roles_v1/test.jsonl is not present (git-ignored)")
    reference = record["reference"]
    _, built_tool = proposal_call(reference["request"], reference["evidence_ledger"])
    stored_tool = record["tools"][0]
    assert built_tool == stored_tool
    assert json.dumps(built_tool, sort_keys=False) == json.dumps(stored_tool, sort_keys=False)


def test_evidence_call_matches_stored_training_tool_byte_for_byte() -> None:
    record = _first_record("evidence_extractor")
    if record is None:
        pytest.skip("training/datasets/journal_roles_v1/test.jsonl is not present (git-ignored)")
    reference = record["reference"]
    _, built_tool = evidence_call(reference["request"], reference["sources"])
    stored_tool = record["tools"][0]
    assert built_tool == stored_tool
    assert json.dumps(built_tool, sort_keys=False) == json.dumps(stored_tool, sort_keys=False)
