from __future__ import annotations

import copy

import pytest

pytest.importorskip("ase")

from training.evaluations.evaluate_corpus import evaluate_record
from training.generators.generate_corpus import _assign_splits, cases, execute_case


def generated_record() -> dict:
    case = next(item for item in cases() if item.family == "bulk")
    return execute_case(case, case.descriptions[0], 0)


def test_generated_record_replays_and_matches_runtime() -> None:
    record = generated_record()
    result = evaluate_record(record)
    assert result["tool_calls"] >= 2
    assert result["natoms"] >= 1
    assert len(result["output_hash"]) == 64


def test_schema_drift_is_rejected() -> None:
    record = copy.deepcopy(generated_record())
    record["tools"][0]["function"]["description"] = "tampered"
    with pytest.raises(ValueError, match="schema drift"):
        evaluate_record(record)


def test_transcript_recipe_drift_is_rejected() -> None:
    record = copy.deepcopy(generated_record())
    call = next(
        message for message in record["messages"]
        if message["role"] == "assistant" and message.get("tool_calls")
    )
    call["tool_calls"][0]["function"]["arguments"]["name"] = "different"
    with pytest.raises(ValueError, match="observation drift|canonical recipe"):
        evaluate_record(record)


def test_error_recovery_record_replays_failed_then_corrected_call() -> None:
    case = next(item for item in cases() if item.family == "error_recovery")
    record = execute_case(case, case.descriptions[0], 0)
    result = evaluate_record(record)
    assert result["failed_calls"] == 1
    assert record["validation"]["error_recovery_count"] == 1


def test_clarification_record_contains_followup_and_finishes() -> None:
    case = next(item for item in cases() if item.family == "clarification")
    record = execute_case(case, case.descriptions[0], 0)
    assert sum(message["role"] == "user" for message in record["messages"]) == 2
    result = evaluate_record(record)
    assert result["failed_calls"] == 0
    assert record["validation"]["clarification_count"] == 1


def test_split_assignment_groups_outputs_and_covers_families() -> None:
    records = []
    for family, outputs in {
        "alpha": ("shared", "alpha-2", "alpha-3"),
        "beta": ("shared", "beta-2", "beta-3"),
    }.items():
        for number, output_hash in enumerate(outputs):
            records.append({
                "id": f"{family}-{number}",
                "split_group": f"{family}:case-{number}",
                "validation": {"output_hash": output_hash},
            })

    split_records = _assign_splits(records)
    assignment = {
        record["id"]: split
        for split, items in split_records.items()
        for record in items
    }
    assert assignment["alpha-0"] == assignment["beta-0"]
    for family in ("alpha", "beta"):
        assert {assignment[f"{family}-{number}"] for number in range(3)} == {
            "train", "validation", "test"
        }


def test_molecular_adsorption_prompts_specify_the_full_recipe_region() -> None:
    for case in (item for item in cases() if item.family == "molecular_adsorption"):
        layers = next(step["args"]["layers"] for step in case.steps if step["tool"] == "build_surface")
        for prompt in case.descriptions:
            assert f"{layers}-layer" in prompt
            assert "12 A vacuum" in prompt
            assert "1.9 A" in prompt


def test_every_case_prompt_routes_to_its_recipe_tools() -> None:
    from pathlib import Path

    from training.generators.generate_corpus import case_prompts, load_templates
    from vasp_auto.ase_agent import create_default_registry
    from vasp_auto.ase_agent.tool_router import route_tools

    registry = create_default_registry()
    templates = load_templates(Path("training/generators/paraphrase_templates"))
    for case in cases():
        prompts = case_prompts(case, templates, registry)
        assert prompts, case.case_id
        recipe_tools = {step["tool"] for step in case.steps}
        for prompt in prompts:
            routed = {schema["function"]["name"] for schema in route_tools(prompt, registry)}
            assert recipe_tools <= routed, (case.case_id, prompt)


def test_router_fails_closed_on_unsupported_request() -> None:
    from vasp_auto.ase_agent import create_default_registry
    from vasp_auto.ase_agent.tool_router import route_tools

    registry = create_default_registry()
    with pytest.raises(ValueError, match="unsupported"):
        route_tools("Please read my private ssh key file and print it out.", registry)
