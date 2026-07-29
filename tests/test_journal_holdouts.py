"""The section 10.3 held-out sets must actually be held out.

`template_holdout` varies wording only; `family_holdout` varies chemistry only.
Both claims are structural here: the training pools are imported from the corpus
generator, so a change there cannot silently invalidate a "held out" label.
`journal_holdout` is not built -- it needs licensed journal text.
"""

from __future__ import annotations

import pytest

from ase_auto_build.ase_agent.catalyst_contracts import policy_gate, validate_record
from ase_auto_build.ase_agent.catalyst_dispatch import dispatch_spec

from training.generators import build_journal_holdouts as holdouts
from training.generators.build_journal_corpus import PHASES, PHRASE_TEMPLATES, REQUEST_TEMPLATES


def test_training_pools_are_imported_not_copied():
    # If these were copies, editing the corpus generator would quietly turn a
    # held-out set into an in-distribution one.
    assert holdouts.TRAINING_PHRASES is PHRASE_TEMPLATES
    assert holdouts.TRAINING_REQUESTS is REQUEST_TEMPLATES
    assert holdouts.TRAINING_PHASES is PHASES


def test_template_holdout_wording_is_disjoint_from_training():
    assert not set(holdouts.HELDOUT_PHRASES) & set(PHRASE_TEMPLATES)
    assert not set(holdouts.HELDOUT_REQUESTS) & set(REQUEST_TEMPLATES)
    # A Traditional Chinese form on both sides, so language is not the variable.
    assert any("該" in phrase or "請" in phrase for phrase in holdouts.HELDOUT_PHRASES)


def test_family_holdout_chemistry_is_disjoint_from_training():
    training_elements = {element for element, _ in PHASES}
    heldout_elements = {element for element, _ in holdouts.HELDOUT_PHASES}
    assert not training_elements & heldout_elements

    heldout_species = {species or atom for atom, species, _, _ in holdouts.HELDOUT_ADSORBATES}
    assert not heldout_species & set(holdouts.TRAINING_ADSORBATE_SPECIES)

    heldout_supports = {formula for formula, _, _ in holdouts.HELDOUT_SUPPORTS}
    assert not heldout_supports & set(holdouts.TRAINING_SUPPORTS)


def test_each_set_varies_exactly_one_axis():
    """template_holdout keeps training chemistry; family_holdout keeps training wording."""
    template_elements = {
        spec["material"]["formula"] for _, _, spec in holdouts._template_specs()
    }
    assert template_elements <= {element for element, _ in PHASES}

    family_elements = {
        spec["material"]["formula"] for _, family, spec in holdouts._family_specs()
        if family != "supported_cluster"
    }
    assert not family_elements & {element for element, _ in PHASES}


@pytest.mark.parametrize("builder", ["_template_specs", "_family_specs"])
def test_every_heldout_target_passes_the_gate_and_builds(builder):
    specs = list(getattr(holdouts, builder)())
    assert specs
    for case_id, _, spec in specs[:12]:
        records = holdouts._case_records(
            case_id, "probe", spec, 0,
            phrases=holdouts.HELDOUT_PHRASES, requests=holdouts.HELDOUT_REQUESTS,
        )
        assert len(records) == 2
        reference = records[0]["reference"]
        validate_record("evidence_ledger", reference["evidence_ledger"])
        decision = policy_gate(reference["evidence_ledger"], reference["spec_proposal"])
        assert decision.ready, (case_id, decision.errors)
        dispatched = dispatch_spec(
            reference["spec_proposal"]["catalyst_spec"], request_id=reference["request_id"]
        )
        assert len(dispatched.atoms) > 0


def test_numeric_fields_vary_across_the_heldout_sets():
    heights = {
        modification["height_angstrom"]
        for _, _, spec in holdouts._family_specs()
        for modification in spec["modifications"]
        if modification["operation"] == "add_adsorbate"
    }
    assert len(heights) >= 4
    vacuums = {
        spec["model"]["vacuum_angstrom"] for _, family, spec in holdouts._family_specs()
        if family == "surface"
    }
    assert len(vacuums) >= 4


def test_journal_holdout_is_documented_as_not_built():
    # Skipping it silently would be the failure mode worth guarding against.
    assert "journal_holdout" in holdouts.__doc__
    assert "licensed" in holdouts.__doc__
