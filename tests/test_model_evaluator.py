from __future__ import annotations

import argparse

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import training.evaluations.evaluate_model as em
from training.evaluations.evaluate_model import (
    evaluate_adversarial,
    first_tool_call_turn,
    invariants_match,
    parse_tool_calls,
    select_records,
)


def _record(record_id: str, family: str):
    return {"id": record_id, "split_group": f"{family}:comp:{record_id}"}


def _invariants(formula, natoms, cell_lengths, cell_angles=(90.0, 90.0, 90.0), constrained=0):
    return {
        "formula": dict(formula),
        "natoms": natoms,
        "pbc": [True, True, True],
        "cell_lengths": list(cell_lengths),
        "cell_angles": list(cell_angles),
        "constrained_atoms": constrained,
    }


def test_parse_qwen_tool_call() -> None:
    text = '<tool_call>\n{"name":"build_bulk","arguments":{"name":"al","element":"Al"}}\n</tool_call>'
    calls = parse_tool_calls(text)
    assert calls[0]["function"]["name"] == "build_bulk"
    assert calls[0]["function"]["arguments"]["element"] == "Al"


def test_parse_multiple_tool_calls() -> None:
    text = (
        '<tool_call>{"name":"inspect_structure","arguments":{"name":"x"}}</tool_call>'
        '<tool_call>{"name":"finish","arguments":{"name":"x"}}</tool_call>'
    )
    assert [call["function"]["name"] for call in parse_tool_calls(text)] == [
        "inspect_structure", "finish"
    ]


def test_parse_rejects_non_object_arguments() -> None:
    with pytest.raises(ValueError, match="arguments"):
        parse_tool_calls('<tool_call>{"name":"finish","arguments":"x"}</tool_call>')


def test_local_turn_stops_after_first_complete_tool_call() -> None:
    text = (
        '<tool_call>{"name":"build_bulk","arguments":{"name":"al","element":"Al"}}</tool_call>'
        '<tool_call>{"name":"finish","arguments":{"name":"al"}}</tool_call>'
    )
    truncated = first_tool_call_turn(text)
    assert [call["function"]["name"] for call in parse_tool_calls(truncated)] == [
        "build_bulk"
    ]


def test_invariants_tolerate_small_drift_but_reject_formula_change() -> None:
    reference = _invariants({"Cu": 3, "O": 1}, 4, (2.55, 2.55, 28.17))
    close = _invariants({"Cu": 3, "O": 1}, 4, (2.56, 2.55, 28.18))
    assert invariants_match(close, reference) is True
    wrong_formula = _invariants({"Cu": 3}, 3, (2.55, 2.55, 28.17))
    assert invariants_match(wrong_formula, reference) is False
    wrong_cell = _invariants({"Cu": 3, "O": 1}, 4, (2.55, 2.55, 30.0))
    assert invariants_match(wrong_cell, reference) is False


def test_select_records_covers_every_family_then_fills_budget() -> None:
    records = (
        [_record(f"bulk-{i}", "bulk") for i in range(10)]
        + [_record(f"molecule-{i}", "molecule") for i in range(10)]
        + [_record("clar-0", "clarification")]
    )
    chosen = select_records(records, limit=0, per_family=1, min_per_family=0, sample_size=8)
    families = {record["split_group"].split(":", 1)[0] for record in chosen}
    assert families == {"bulk", "molecule", "clarification"}  # coverage floor met
    assert len(chosen) == 8  # filled up to the budget
    # Deterministic: identical inputs and flags select the identical ordered set.
    assert select_records(records, limit=0, per_family=1, min_per_family=0, sample_size=8) == chosen


def test_select_records_limit_takes_precedence() -> None:
    records = [_record(f"bulk-{i}", "bulk") for i in range(5)]
    assert select_records(records, limit=2, per_family=0, min_per_family=0, sample_size=0) == records[:2]


def test_adversarial_router_refusal_scored_safe(monkeypatch) -> None:
    class _RefusingController:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, prompt):  # noqa: ANN001
            raise ValueError("unsupported structure request")

    monkeypatch.setattr(em, "AgentController", _RefusingController)
    args = argparse.Namespace(max_turns=6, max_new_tokens=64)
    record = {"id": "adv-x", "prompt": "do something forbidden", "must_not_finish": True}
    out = evaluate_adversarial(None, None, record, args)
    assert out["safe"] is True
    assert out["state"] == "REFUSED_BEFORE_EXECUTION"
    assert out["forbidden_executed"] == []
