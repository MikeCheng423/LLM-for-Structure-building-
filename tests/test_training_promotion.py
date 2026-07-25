from __future__ import annotations

from training.evaluations.promote_adapter import promotion_decision


def report(
    exact: float,
    finish: float = 1.0,
    execution: float = 1.0,
    forbidden: float = 0.0,
    invariant: float = 0.95,
    safety_rate: float = 1.0,
    forbidden_exec: int = 0,
):
    return {
        "complete": True,
        "dataset_sha256": "abc",
        "evaluation_ids_sha256": "records-abc",
        "summary": {
            "tool_mode": "full",
            "record_count": 30,
            "exact_structure_rate": exact,
            "invariants_satisfied_rate": invariant,
            "finish_rate": finish,
            "schema_execution_success_rate": execution,
            "forbidden_action_rate": forbidden,
            "family_metrics": {
                "bulk": {"exact_structure_rate": exact, "invariants_satisfied_rate": invariant},
                "surface": {"exact_structure_rate": exact, "invariants_satisfied_rate": invariant},
            },
        },
        "safety": {
            "record_count": 5,
            "safety_rate": safety_rate,
            "forbidden_execution_count": forbidden_exec,
            "unsafe_ids": [],
        },
    }


def test_promotes_only_non_regressing_safe_adapter() -> None:
    decision = promotion_decision({"passed": True}, report(0.4), report(0.6))
    assert decision["promoted"] is True
    assert all(decision["checks"].values())


def test_forbidden_action_always_blocks_promotion() -> None:
    decision = promotion_decision({"passed": True}, report(0.4), report(0.8, forbidden=0.01))
    assert decision["promoted"] is False
    assert decision["checks"]["zero_forbidden_actions"] is False


def test_dataset_mismatch_blocks_promotion() -> None:
    adapter = report(0.8)
    adapter["dataset_sha256"] = "different"
    decision = promotion_decision({"passed": True}, report(0.4), adapter)
    assert decision["promoted"] is False
    assert decision["checks"]["same_evaluation_dataset"] is False


def test_recipe_family_regression_blocks_promotion() -> None:
    base = report(0.5)
    adapter = report(0.7)
    base["summary"]["family_metrics"]["surface"]["exact_structure_rate"] = 0.8
    adapter["summary"]["family_metrics"]["surface"]["exact_structure_rate"] = 0.6
    decision = promotion_decision({"passed": True}, base, adapter)
    assert decision["promoted"] is False
    assert decision["checks"]["family_exact_rates_not_regressed"] is False


def test_different_record_subset_blocks_promotion() -> None:
    adapter = report(0.8)
    adapter["evaluation_ids_sha256"] = "other-records"
    decision = promotion_decision({"passed": True}, report(0.4), adapter)
    assert decision["promoted"] is False
    assert decision["checks"]["same_ordered_evaluation_records"] is False


def test_oracle_record_mode_cannot_promote() -> None:
    base = report(0.8)
    adapter = report(0.9)
    base["summary"]["tool_mode"] = "record"
    adapter["summary"]["tool_mode"] = "record"
    decision = promotion_decision({"passed": True}, base, adapter)
    assert decision["promoted"] is False
    assert decision["checks"]["deployment_tool_mode"] is False


def test_incomplete_evaluation_cannot_promote() -> None:
    adapter = report(0.9)
    adapter["complete"] = False
    decision = promotion_decision({"passed": True}, report(0.8), adapter)
    assert decision["promoted"] is False
    assert decision["checks"]["adapter_evaluation_complete"] is False


def test_low_invariant_rate_blocks_promotion() -> None:
    decision = promotion_decision({"passed": True}, report(0.4), report(0.6, invariant=0.5))
    assert decision["promoted"] is False
    assert decision["checks"]["minimum_invariant_rate"] is False


def test_invariant_regression_blocks_promotion() -> None:
    decision = promotion_decision(
        {"passed": True}, report(0.4, invariant=0.95), report(0.6, invariant=0.9)
    )
    assert decision["promoted"] is False
    assert decision["checks"]["invariant_rate_not_regressed"] is False


def test_missing_safety_block_cannot_promote() -> None:
    adapter = report(0.9)
    del adapter["safety"]
    decision = promotion_decision({"passed": True}, report(0.4), adapter)
    assert decision["promoted"] is False
    assert decision["checks"]["adversarial_safety_evaluated"] is False


def test_forbidden_execution_blocks_promotion() -> None:
    decision = promotion_decision(
        {"passed": True}, report(0.4), report(0.9, forbidden_exec=1, safety_rate=0.8)
    )
    assert decision["promoted"] is False
    assert decision["checks"]["zero_forbidden_executions"] is False
    assert decision["checks"]["minimum_adversarial_safety_rate"] is False
