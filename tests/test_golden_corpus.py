"""The golden corpus and the section 9 vertical-slice gate.

CATALYST_STRUCTURE_LLM_AGENT_DESIGN.md section 9 requires the deterministic
slice to pass six criteria before an LLM is connected. These tests keep the gate
a standing barrier rather than a one-off report, and the negative controls prove
each newly added rule can actually fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from ase_auto_build.ase_agent.catalyst_contracts import policy_gate
from ase_auto_build.ase_agent.catalyst_dispatch import dispatch_spec
from ase_auto_build.ase_agent.catalyst_validation import catalyst_validation_rules
from ase_auto_build.ase_agent.policy import AgentPolicy
from ase_auto_build.ase_agent.validation import validate_atoms

from training.evaluations.run_vertical_slice_gate import (
    load_fixtures,
    load_manifest,
    manifest_hash_mismatches,
    run_gate,
)

GOLDEN_ROOT = Path(__file__).resolve().parents[1] / "tests" / "golden"

#: Section 7.2's required composition.
EXPECTED_FAMILIES = {"bulk": 30, "slab": 30, "defect": 15, "adsorbate": 15, "refusal": 10}


@pytest.fixture(scope="module")
def fixtures() -> list[dict]:
    return load_fixtures(GOLDEN_ROOT)


@pytest.fixture(scope="module")
def gate_report(fixtures, tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("vertical-slice-gate") / "packages"
    return run_gate(fixtures, root)


def test_corpus_has_the_composition_section_7_2_requires():
    manifest = load_manifest(GOLDEN_ROOT)
    # Section 7.2's 100 cases are counted separately from section 14's examples.
    assert manifest["section_counts"]["7.2"] == 100
    assert manifest["case_count"] == sum(manifest["section_counts"].values())
    assert manifest["family_counts"] == EXPECTED_FAMILIES
    # physical_reference_golden needs licensed data; it is empty on purpose.
    assert manifest["label_counts"]["physical_reference_golden"] == 0
    assert manifest["notes"]["physical_reference_golden"]


def test_fixtures_are_immutable():
    assert manifest_hash_mismatches(GOLDEN_ROOT) == []


def test_every_fixture_records_that_domain_review_is_still_pending(fixtures):
    # Deterministic and reviewed-by-code is not the same as section 7.1 review.
    assert {fixture["review"]["domain_review"] for fixture in fixtures} == {"pending"}


def test_numeric_fields_vary_independently(fixtures):
    """Section 7.2 forbids one constant vacuum, thickness, height or supercell."""
    slabs = [f["spec_proposal"]["catalyst_spec"]["model"] for f in fixtures if f["family"] == "slab"]
    assert len({model["vacuum_angstrom"] for model in slabs}) >= 5
    assert len({model["layers"] for model in slabs}) >= 5
    assert len({tuple(model["supercell"]) for model in slabs}) >= 4
    heights = {
        modification["height_angstrom"]
        for fixture in fixtures if fixture["family"] == "adsorbate"
        for modification in fixture["spec_proposal"]["catalyst_spec"]["modifications"]
    }
    assert len(heights) >= 5


@pytest.mark.parametrize("case_id", [entry["case_id"] for entry in load_manifest(GOLDEN_ROOT)["cases"]])
def test_policy_gate_agrees_with_the_declared_outcome(case_id):
    fixture = json.loads((GOLDEN_ROOT / "cases" / f"{case_id}.json").read_text(encoding="utf-8"))
    decision = policy_gate(fixture["evidence_ledger"], fixture["spec_proposal"])
    if fixture["expected_status"] == "review_ready":
        assert decision.ready, (case_id, decision.status, decision.errors)
    elif fixture["expected_status"] == "unsupported":
        assert decision.status == "unsupported", (case_id, decision.errors)
    elif "mp_candidates" not in fixture:
        assert not decision.ready, case_id


def test_vertical_slice_gate_passes(gate_report):
    failures = {
        name: criterion["failures"]
        for name, criterion in gate_report["criteria"].items() if criterion["failures"]
    }
    assert gate_report["passed"], json.dumps(failures, indent=2, sort_keys=True)
    assert gate_report["case_count"] == 106
    assert gate_report["buildable_case_count"] == 95


@pytest.mark.parametrize("criterion", [
    "schema_validation_rate", "build_export_round_trip_rate", "writes_outside_request_dir",
    "arbitrary_code_paths", "fields_without_provenance", "ambiguity_returned_clarification",
])
def test_each_section_9_criterion_is_measured(gate_report, criterion):
    measured = gate_report["criteria"][criterion]
    if isinstance(measured["threshold"], float):
        assert measured["value"] >= measured["threshold"], measured["failures"]
    else:
        assert measured["value"] <= measured["threshold"], measured["failures"]


def test_the_injection_fixture_never_builds_or_names_a_tool(gate_report):
    case = next(item for item in gate_report["cases"] if item["case_id"] == "refuse-injection")
    assert case["status"] == "needs_clarification"
    assert case["built_structure"] is False
    assert "tool_calls" not in case


def test_exported_sidecar_does_not_claim_validation_before_it_ran(gate_report):
    # controller_state used to say VALIDATED at export time, 19 lines early.
    assert all(item["validation_passed"] for item in gate_report["cases"] if item["status"] == "review_ready")


# --------------------------------------------------------------------------
# Negative controls: each rule must be able to fail.
# --------------------------------------------------------------------------


def test_overlap_scan_finds_a_violation_hidden_behind_a_closer_legal_pair():
    """H-H at 0.30 A is legal; Pt-Pt at 0.90 A is not, but sits further apart.

    Comparing only the globally closest pair reported no issue here.
    """
    atoms = Atoms(
        "H2Pt2",
        positions=[[0, 0, 0], [0.30, 0, 0], [5.0, 0, 0], [5.90, 0, 0]],
        cell=np.eye(3) * 20.0, pbc=False,
    )
    report = validate_atoms(atoms, AgentPolicy(), profile="final")
    overlaps = [issue for issue in report.issues if issue.code == "atom_overlap"]
    assert overlaps, "a Pt-Pt contact below its own threshold must be reported"
    assert "3 and 4" in overlaps[0].message


def test_overlap_scan_accepts_a_legal_small_pair():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.80, 0, 0]], cell=np.eye(3) * 20.0, pbc=False)
    report = validate_atoms(atoms, AgentPolicy(), profile="final")
    assert not [issue for issue in report.issues if issue.code == "atom_overlap"]


def _dispatch_golden(case_id: str):
    fixture = json.loads((GOLDEN_ROOT / "cases" / f"{case_id}.json").read_text(encoding="utf-8"))
    spec = fixture["spec_proposal"]["catalyst_spec"]
    return spec, dispatch_spec(spec, request_id=fixture["request_id"])


def _rule(rules: list[dict], name: str) -> dict:
    return next(item for item in rules if item["rule"] == name)


def test_slab_layer_count_rule_fails_on_a_wrong_layer_count():
    spec, dispatched = _dispatch_golden("slab-00")
    assert _rule(catalyst_validation_rules(spec, dispatched.workspace), "slab_layer_count")["passed"]
    tampered = {**spec, "model": {**spec["model"], "layers": spec["model"]["layers"] + 1}}
    assert not _rule(catalyst_validation_rules(tampered, dispatched.workspace), "slab_layer_count")["passed"]


def test_atom_count_rule_fails_when_a_modification_did_not_run():
    spec, dispatched = _dispatch_golden("slab-01")
    assert _rule(catalyst_validation_rules(spec, dispatched.workspace), "atom_count")["passed"]
    tampered = {**spec, "modifications": [{
        "operation": "add_adsorbate", "species": "CO", "anchor": 1,
        "site": "ontop", "site_index": 1, "height_angstrom": 1.9,
    }]}
    assert not _rule(catalyst_validation_rules(tampered, dispatched.workspace), "atom_count")["passed"]


def test_host_stoichiometry_rule_fails_on_a_mismatched_formula():
    spec, dispatched = _dispatch_golden("slab-02")
    assert _rule(catalyst_validation_rules(spec, dispatched.workspace), "host_stoichiometry")["passed"]
    tampered = {**spec, "material": {**spec["material"], "formula": "PtO2"}}
    assert not _rule(catalyst_validation_rules(tampered, dispatched.workspace), "host_stoichiometry")["passed"]
