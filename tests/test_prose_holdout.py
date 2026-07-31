"""`prose_holdout` -- Phase 1 of `training/JOURNAL_ROLE_R3_SCHEDULE.md`.

The defining property of this set, distinguishing it from `template_holdout`,
is that its source text contains no JSON literal of any target value -- see
`build_prose_holdout._assert_no_leak`. These tests check that guard actually
fires, that the generated dataset never leaks a literal, that every generated
target still passes the deterministic gate/dispatch, and that the numeric
fields vary the way section 3 of the schedule requires.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ase_auto_build.ase_agent.catalyst_contracts import policy_gate, validate_record
from ase_auto_build.ase_agent.catalyst_dispatch import dispatch_spec

from training.generators import build_prose_holdout as prose

DATASET = Path("training/datasets/journal_holdout_prose/test.jsonl")


def _dataset_records() -> list[dict]:
    if not DATASET.exists():
        pytest.skip("training/datasets/journal_holdout_prose/test.jsonl is not present (git-ignored)")
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------
# The guard itself
# --------------------------------------------------------------------------

def test_leak_guard_fires_on_a_quoted_json_literal():
    with pytest.raises(RuntimeError, match="leaks JSON literal"):
        prose._assert_no_leak("probe-case", "model.atom_ordering", "ase_default", 'The reported value is "ase_default".')


def test_leak_guard_fires_on_a_bare_enum_literal():
    with pytest.raises(RuntimeError, match="leaks bare literal"):
        prose._assert_no_leak("probe-case", "model.atom_ordering", "ase_default", "Atoms follow ase_default ordering.")


def test_leak_guard_fires_on_a_bare_boolean_literal():
    with pytest.raises(RuntimeError, match="leaks bare literal"):
        prose._assert_no_leak("probe-case", "model.center", True, "The centering flag is true.")


def test_leak_guard_fires_on_a_bare_int_literal():
    with pytest.raises(RuntimeError, match="leaks bare literal"):
        prose._assert_no_leak("probe-case", "model.layers", 4, "The slab has 4 layers.")


def test_leak_guard_fires_on_a_list_literal():
    with pytest.raises(RuntimeError, match="leaks JSON literal"):
        prose._assert_no_leak("probe-case", "model.supercell", [2, 2, 1], "The repeat is [2, 2, 1].")


def test_leak_guard_passes_genuine_prose():
    # No RuntimeError -- these are exactly the forms the generator produces.
    prose._assert_no_leak("probe-case", "model.atom_ordering", "ase_default", "ASE-default atom ordering was used.")
    prose._assert_no_leak("probe-case", "model.layers", 4, "A four-layer slab was built.")
    prose._assert_no_leak("probe-case", "model.center", True, "The slab was centred in the cell.")
    prose._assert_no_leak("probe-case", "material.formula", "Al", "Aluminum is the material of interest.")
    prose._assert_no_leak("probe-case", "material.formula", "Ni", "Nickel is the host metal.")


def test_leak_guard_exempts_floats():
    # Physical measurements: prose and JSON necessarily render the same
    # digits (JOURNAL_ROLE_R3_SCHEDULE.md's own worked example writes
    # "1.85 A above the surface"), so this is not the copy-paste failure
    # mode the guard exists to catch.
    prose._assert_no_leak("probe-case", "modifications[0].height_angstrom", 1.85, "1.85 A above the surface.")


def test_boundary_matching_does_not_flag_a_prefix_of_a_longer_word():
    # "Al" must not fire just because "Aluminum" starts with those two
    # letters -- there is no word boundary after the match inside the word.
    assert not prose._bare_leak("Aluminum is the material of interest.", "Al")
    assert not prose._bare_leak("Nickel is the host metal.", "Ni")
    assert prose._bare_leak("Al is the material.", "Al")


# --------------------------------------------------------------------------
# Generator-level invariants (probe a handful of freshly built cases so this
# does not depend on the git-ignored dataset being present).
# --------------------------------------------------------------------------

def test_generated_case_covers_every_field_and_grounds_no_literal():
    records = prose._bulk_case("probe-bulk", 0, "Pt", "fcc", [2, 2, 1])
    assert len(records) == 2
    text = records[0]["reference"]["sources"][0]["text"]
    assert "platinum" in text.lower()
    for claim in records[0]["reference"]["evidence_ledger"]["claims"]:
        prose._assert_no_leak("probe-bulk", claim["field"], claim["value"], text)


def test_field_coverage_mismatch_fails_closed():
    spec = prose._spec("bulk", {"formula": "Pt", "crystal_structure": "fcc"}, {
        "kind": "bulk", "supercell": [1, 1, 1],
        "periodic_boundary_conditions": [True, True, True],
        "center": True, "atom_ordering": "ase_default",
    }, [])
    with pytest.raises(RuntimeError, match="field coverage mismatch"):
        prose._build_case("probe-incomplete", "bulk", spec, "Build it.", {}, {})


@pytest.mark.parametrize("builder,args", [
    ("_bulk_case", ("probe-bulk", 0, "Cu", "fcc", [2, 1, 1])),
    ("_surface_case", ("probe-surface", 0, "Ni", "fcc", [1, 1, 1], 4, 12.0, [2, 2, 1])),
    ("_defect_case", ("probe-defect", 0, "Fe", "bcc", [1, 0, 0], 4, 12.0, [2, 2, 1], "substitute")),
])
def test_each_family_builder_passes_the_gate_and_dispatches(builder, args):
    records = getattr(prose, builder)(*args)
    assert len(records) == 2
    reference = records[0]["reference"]
    validate_record("evidence_ledger", reference["evidence_ledger"])
    decision = policy_gate(reference["evidence_ledger"], reference["spec_proposal"])
    assert decision.ready, decision.errors
    dispatched = dispatch_spec(reference["spec_proposal"]["catalyst_spec"], request_id=reference["request_id"])
    assert len(dispatched.atoms) > 0


def test_adsorbate_case_passes_the_gate_and_dispatches():
    records = prose._adsorbate_case(
        "probe-adsorbate", 0, "Pt", "fcc", [1, 1, 1], 4, 14.0, [2, 2, 1],
        None, "CO", 2, 1.85,
    )
    reference = records[0]["reference"]
    decision = policy_gate(reference["evidence_ledger"], reference["spec_proposal"])
    assert decision.ready, decision.errors
    dispatched = dispatch_spec(reference["spec_proposal"]["catalyst_spec"], request_id=reference["request_id"])
    assert len(dispatched.atoms) > 0


# --------------------------------------------------------------------------
# Dataset-level checks (git-ignored; skip if not built).
# --------------------------------------------------------------------------

def test_no_source_text_contains_a_target_literal():
    for record in _dataset_records():
        if record["role"] != "evidence_extractor":
            continue
        text = record["reference"]["sources"][0]["text"]
        for claim in record["reference"]["evidence_ledger"]["claims"]:
            prose._assert_no_leak(record["id"], claim["field"], claim["value"], text)


def test_every_generated_target_passes_policy_gate_and_dispatch():
    records = _dataset_records()
    planner_records = [record for record in records if record["role"] == "spec_planner"][:10]
    assert planner_records
    for record in planner_records:
        reference = record["reference"]
        validate_record("evidence_ledger", reference["evidence_ledger"])
        validate_record("spec_proposal", reference["spec_proposal"])
        decision = policy_gate(reference["evidence_ledger"], reference["spec_proposal"])
        assert decision.ready, (record["id"], decision.errors)
        dispatched = dispatch_spec(reference["spec_proposal"]["catalyst_spec"], request_id=reference["request_id"])
        assert len(dispatched.atoms) > 0


def test_numeric_fields_vary_across_the_dataset():
    records = _dataset_records()
    vacuums = {
        record["reference"]["spec_proposal"]["catalyst_spec"]["model"]["vacuum_angstrom"]
        for record in records if record["role"] == "spec_planner"
        and record["reference"]["spec_proposal"]["catalyst_spec"]["model"]["kind"] == "surface"
    }
    assert len(vacuums) >= 4

    layers = {
        record["reference"]["spec_proposal"]["catalyst_spec"]["model"]["layers"]
        for record in records if record["role"] == "spec_planner"
        and record["reference"]["spec_proposal"]["catalyst_spec"]["model"]["kind"] == "surface"
    }
    assert len(layers) >= 3

    heights = {
        modification["height_angstrom"]
        for record in records if record["role"] == "spec_planner"
        for modification in record["reference"]["spec_proposal"]["catalyst_spec"]["modifications"]
        if modification["operation"] == "add_adsorbate"
    }
    assert len(heights) >= 4


def test_all_four_families_are_present():
    records = _dataset_records()
    families = {record["split_group"].split(":", 1)[0] for record in records}
    assert families == {"bulk", "surface", "adsorbate", "defect"}


def test_manifest_declares_evaluation_only_and_no_literals():
    manifest_path = DATASET.parent / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("training/datasets/journal_holdout_prose/manifest.json is not present (git-ignored)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["set_name"] == "prose_holdout"
    assert manifest["evaluation_only"] is True
    assert manifest["no_literal_values"] is True
    assert "train" not in manifest["split_sha256"]
    assert 40 <= sum(manifest["family_counts"].values()) // 2 <= 60
