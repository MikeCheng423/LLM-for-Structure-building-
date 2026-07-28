"""The guided request builder must compose slot-complete, in-distribution prompts.

The point of `guided` is that a composed request cannot be missing a structural
determinant. These tests pin that claim from three directions: the question list
matches the corpus rule, every composed prompt satisfies the runtime pre-flight
check, and the phrasings really are the corpus phrasings rather than invented
wording that would drift out of the adapter's training distribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ase_auto_build.ase_agent import guided
from ase_auto_build.ase_agent.request_check import (
    FAMILY_REQUIRED,
    check_request,
    stated_values,
)

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / (
    "training/generators/paraphrase_templates_r6"
)

# One fully-specified answer set per region, in the parsed form fill_slots yields.
SAMPLES: dict[str, dict] = {
    "bulk": {"element": "W", "phase": "bcc", "repeat": (2, 2, 1),
             "convention": "conventional cubic"},
    "surface": {"element": "Cu", "facet": "111", "layers": 4, "vacuum": 12.0,
                "phase": "fcc", "repeat": (2, 2, 1)},
    "surface_constraint": {"element": "Cu", "facet": "100", "layers": 4,
                           "vacuum": 12.0, "phase": None, "repeat": (2, 2, 1),
                           "freeze_side": "bottom", "freeze_count": 2},
    "atomic_adsorption": {"element": "Cu", "facet": "100", "layers": 5,
                          "vacuum": 12.0, "phase": None, "repeat": (2, 2, 1),
                          "adsorbate": "O", "site": "ontop", "height": 2.5},
    "molecular_adsorption": {"element": "Fe", "facet": "110", "layers": 5,
                             "vacuum": 12.0, "phase": None, "repeat": (2, 2, 1),
                             "adsorbate": "CO", "site": "ontop", "height": 1.9,
                             "anchor": "carbon"},
    "molecule": {"species": "H2O", "box": 12.0},
    "nanotube": {"chirality": "(6,3)", "length": 2},
    "prototype": {"prototype": "hBN"},
    "vacancy": {"element": "Fe", "phase": "bcc", "defect_site": "first",
                "repeat": (2, 1, 1), "convention": None},
    "substitution": {"element": "Cu", "phase": "fcc", "defect_site": "first",
                     "dopant": "Au", "repeat": (2, 2, 1), "convention": None},
}


def test_every_region_has_a_sample():
    assert set(SAMPLES) == set(guided.REGION_SLOTS)


# --------------------------------------------------------------------------- #
# The wizard asks for exactly what the corpus rule requires
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("region", sorted(guided.REGION_SLOTS))
def test_required_questions_match_the_corpus_rule(region):
    """The drift guard: FAMILY_REQUIRED is the authority for the question list."""
    asked = {slot.rule_slot for slot in guided.REGION_SLOTS[region]
             if slot.required and slot.rule_slot}
    assert asked == set(FAMILY_REQUIRED[region])


def test_regions_match_the_rule_exactly():
    expected = {name for name in FAMILY_REQUIRED if name != "clarification"}
    assert set(guided.REGION_SLOTS) == expected
    assert set(guided.REGION_ORDER) == expected
    assert set(guided.TEMPLATES) == expected
    assert set(guided.REGION_BLURBS) == expected


def test_optional_slots_are_never_rule_slots():
    """An optional question must not be how a required determinant gets stated."""
    for region, slots in guided.REGION_SLOTS.items():
        for slot in slots:
            if not slot.required:
                assert slot.rule_slot is None, (region, slot.key)


# --------------------------------------------------------------------------- #
# Composed prompts pass the pre-flight check -- the module's whole promise
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("region", sorted(SAMPLES))
def test_composed_request_states_every_required_slot(region):
    advisory = check_request(guided.compose(region, SAMPLES[region]))
    assert advisory.region == region
    assert advisory.missing == ()
    assert advisory.ok


@pytest.mark.parametrize("region", sorted(SAMPLES))
def test_composed_request_is_slot_complete_without_optional_answers(region):
    """Leaving every optional question blank must still satisfy the rule."""
    optional = {slot.key for slot in guided.REGION_SLOTS[region] if not slot.required}
    values = {key: value for key, value in SAMPLES[region].items()
              if key not in optional}
    values.update({slot.key: slot.default for slot in guided.REGION_SLOTS[region]
                   if not slot.required})
    advisory = check_request(guided.compose(region, values))
    assert advisory.region == region
    assert advisory.missing == ()


@pytest.mark.parametrize("region", ["atomic_adsorption", "molecular_adsorption"])
def test_adsorption_height_uses_the_measured_phrasing(region):
    """The one phrasing that survived Cu(100)/O/ontop on r5; see the module docstring.

    Both halves matter and were measured: 'at a height of N' *and* the unit spelled
    'angstrom'. The same sentence with 'Å' loses the height again, so this test
    guards the unit token as much as the clause.
    """
    request = guided.compose(region, SAMPLES[region])
    assert "at a height of" in request
    assert "at a height of {height} Å" not in guided.TEMPLATES[region]
    assert "angstrom" in request.split("at a height of")[1]
    assert stated_values(request)["height"] == SAMPLES[region]["height"]


@pytest.mark.parametrize("region", sorted(CORPUS_HEIGHT_REGIONS := {
    "atomic_adsorption", "molecular_adsorption"}))
def test_non_default_height_is_flagged_as_extrapolation(region):
    """The corpus held one height per tool; the form should say so, not hide it."""
    values = dict(SAMPLES[region], height=guided.CORPUS_HEIGHTS[region] + 1.0)
    assert any("extrapolation" in w for w in guided.cross_check(region, values))


@pytest.mark.parametrize("region", sorted({"atomic_adsorption", "molecular_adsorption"}))
def test_trained_height_is_not_flagged(region):
    values = dict(SAMPLES[region], height=guided.CORPUS_HEIGHTS[region])
    assert not any("extrapolation" in w for w in guided.cross_check(region, values))


@pytest.mark.parametrize("region", sorted(SAMPLES))
def test_stated_numbers_are_recoverable_from_the_composed_text(region):
    """Whatever the form collected, the post-build check must be able to read back."""
    values, request = SAMPLES[region], guided.compose(region, SAMPLES[region])
    recovered = stated_values(request)
    for slot in ("layers", "vacuum", "box", "height"):
        if values.get(slot) is not None:
            assert recovered.get(slot) == pytest.approx(float(values[slot])), slot


# --------------------------------------------------------------------------- #
# The phrasings are corpus phrasings
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not TEMPLATE_DIR.is_dir(), reason="training/ not in this checkout")
@pytest.mark.parametrize("region", sorted(guided.TEMPLATES))
def test_templates_come_from_the_training_corpus(region):
    """Unadapted templates are byte-identical to their corpus source."""
    if region in guided.ADAPTED_TEMPLATES:
        pytest.skip("adapted; covered by test_adapted_templates_declare_their_source")
    corpus = json.loads((TEMPLATE_DIR / f"{region}.json").read_text())
    assert guided.TEMPLATES[region] in corpus, (
        f"{region} template is not in {TEMPLATE_DIR.name}"
    )


def test_only_adsorption_templates_are_adapted():
    assert set(guided.ADAPTED_TEMPLATES) == {
        "atomic_adsorption", "molecular_adsorption"
    }


@pytest.mark.skipif(not TEMPLATE_DIR.is_dir(), reason="training/ not in this checkout")
@pytest.mark.parametrize("region", sorted(guided.ADAPTED_TEMPLATES))
def test_adapted_templates_declare_their_source(region):
    """An adapted template must still be a corpus template's skeleton, not new prose."""
    source_index, why = guided.ADAPTED_TEMPLATES[region]
    corpus = json.loads((TEMPLATE_DIR / f"{region}.json").read_text())
    source = corpus[source_index]
    assert why.strip(), f"{region} must document why it deviates"
    # The unchanged tail -- everything the rewrite did not touch -- is corpus text.
    tail = "{suffix} {element}({facet})"
    assert tail in source and tail in guided.TEMPLATES[region]


@pytest.mark.skipif(not TEMPLATE_DIR.is_dir(), reason="training/ not in this checkout")
def test_molecular_adsorption_fragments_are_corpus_text():
    corpus = json.loads((TEMPLATE_DIR / "molecular_adsorption.json").read_text())
    assert "at a height of {height} angstrom" in corpus[8]
    assert "anchored through its {anchor}" in corpus[4]


# --------------------------------------------------------------------------- #
# Value parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw,expected", [
    ("cu", "Cu"), ("CU", "Cu"), ("Fe", "Fe"), (" au ", "Au"),
])
def test_parse_element_normalises_case(raw, expected):
    assert guided.parse_element(raw) == expected


@pytest.mark.parametrize("raw", ["Xx", "copper", "", "12"])
def test_parse_element_rejects_non_symbols(raw):
    with pytest.raises(guided.SlotError):
        guided.parse_element(raw)


@pytest.mark.parametrize("raw,expected", [
    ("111", "111"), ("(100)", "100"), ("1,1,1", "111"), ("1 0 0", "100"),
    ("0001", "0001"), ("(1,-1,0)", "110"),
])
def test_parse_facet_normalises_to_corpus_form(raw, expected):
    assert guided.parse_facet(raw) == expected


@pytest.mark.parametrize("raw", ["11", "11111", "abc", ""])
def test_parse_facet_rejects_malformed(raw):
    with pytest.raises(guided.SlotError):
        guided.parse_facet(raw)


@pytest.mark.parametrize("raw,expected", [
    ("2", (2, 1, 1)), ("2x2", (2, 2, 1)), ("2x2x1", (2, 2, 1)),
    ("3 3 2", (3, 3, 2)), ("2,2,1", (2, 2, 1)), ("2X2", (2, 2, 1)),
])
def test_parse_repeat(raw, expected):
    assert guided.parse_repeat(raw) == expected


@pytest.mark.parametrize("raw", ["0", "2x0", "-1", "2x2x2x2", ""])
def test_parse_repeat_rejects_bad_factors(raw):
    with pytest.raises(guided.SlotError):
        guided.parse_repeat(raw)


@pytest.mark.parametrize("raw,expected", [
    ("ontop", "ontop"), ("on-top", "ontop"), ("on top", "ontop"),
    ("bridge", "bridge"), ("hollow", "hollow"), ("fcc hollow", "hollow"),
    ("ontop site", "ontop"),
])
def test_parse_site(raw, expected):
    assert guided.parse_site(raw) == expected


def test_parse_site_rejects_unknown():
    with pytest.raises(guided.SlotError):
        guided.parse_site("atop-ish")


@pytest.mark.parametrize("raw,expected", [("C", "carbon"), ("carbon", "carbon"),
                                          ("O", "oxygen"), ("n", "nitrogen")])
def test_parse_anchor_uses_words(raw, expected):
    assert guided.parse_anchor(raw) == expected


@pytest.mark.parametrize("raw,expected", [("6,3", "(6,3)"), ("(6,3)", "(6,3)"),
                                          ("6 3", "(6,3)"), ("(10, 0)", "(10,0)")])
def test_parse_chirality(raw, expected):
    assert guided.parse_chirality(raw) == expected


@pytest.mark.parametrize("raw", ["6", "6,3,1", "a,b"])
def test_parse_chirality_rejects_malformed(raw):
    with pytest.raises(guided.SlotError):
        guided.parse_chirality(raw)


@pytest.mark.parametrize("raw,expected", [("1", "first"), ("first", "first"),
                                          ("3", "index 3")])
def test_parse_defect_site(raw, expected):
    assert guided.parse_defect_site(raw) == expected


def test_parse_prototype_resolves_aliases():
    assert guided.parse_prototype("bn") == "hBN"
    assert guided.parse_prototype("graphene") == "graphene"


def test_parse_prototype_rejects_unknown():
    with pytest.raises(guided.SlotError):
        guided.parse_prototype("perovskite-of-my-dreams")


def test_parse_species_forgives_case():
    assert guided.parse_species("h2o") == "H2O"


def test_parse_species_rejects_unknown():
    with pytest.raises(guided.SlotError):
        guided.parse_species("unobtainium")


def test_blank_required_slot_is_refused():
    slot = guided.REGION_SLOTS["surface"][0]
    assert slot.required
    with pytest.raises(guided.SlotError):
        slot.ask("   ")


def test_blank_optional_slot_takes_the_default():
    optional = [s for s in guided.REGION_SLOTS["surface"] if not s.required]
    assert optional, "surface should have optional slots"
    for slot in optional:
        assert slot.ask("") == slot.default


# --------------------------------------------------------------------------- #
# Formatting matches the corpus generator's surface forms
# --------------------------------------------------------------------------- #


def test_repeat_defaults_match_template_fill():
    """An unstated repeat renders 1x1 / 1x1x1, as training/template_fill did."""
    formatted = guided.format_dict("surface", {"element": "Cu"})
    assert formatted["suffix"] == "1x1"
    assert formatted["repeat3"] == "1x1x1"


def test_optional_phase_rides_on_the_element_for_surfaces():
    formatted = guided.format_dict("surface", {"element": "Cu", "phase": "fcc"})
    assert formatted["element"] == "fcc Cu"
    assert "crystalline" not in formatted


def test_cell_convention_is_folded_into_the_phase_for_bulk():
    formatted = guided.format_dict(
        "bulk", {"element": "W", "phase": "bcc", "convention": "conventional cubic"}
    )
    assert formatted["crystalline"] == "conventional cubic bcc"
    assert formatted["element"] == "W"


def test_conventional_cubic_reaches_the_composed_request():
    """The USER_MANUAL section 6 disambiguation must survive composition."""
    request = guided.compose("bulk", SAMPLES["bulk"])
    assert "conventional cubic" in request


def test_whole_numbers_lose_their_decimal_point():
    formatted = guided.format_dict("surface", {"element": "Cu", "vacuum": 12.0})
    assert formatted["vacuum"] == "12"


def test_fractional_numbers_are_kept():
    formatted = guided.format_dict("atomic_adsorption", {"height": 2.5})
    assert formatted["height"] == "2.5"


# --------------------------------------------------------------------------- #
# Cross-slot checks
# --------------------------------------------------------------------------- #


def test_freezing_more_layers_than_exist_warns():
    warnings = guided.cross_check(
        "surface_constraint", {"layers": 2, "freeze_count": 4}
    )
    assert warnings and "whole" in warnings[0]


def test_sane_freeze_does_not_warn():
    assert guided.cross_check(
        "surface_constraint", {"layers": 4, "freeze_count": 2}
    ) == ()


def test_vacancy_in_a_single_cell_warns():
    warnings = guided.cross_check(
        "vacancy", {"defect_site": "first", "repeat": (1, 1, 1)}
    )
    assert warnings


# --------------------------------------------------------------------------- #
# The wizard loop, driven through the console seam
# --------------------------------------------------------------------------- #


def _console(answers):
    """A console that replays scripted answers and records what was printed."""
    queue = list(answers)

    def read(_label):
        if not queue:
            raise EOFError
        return queue.pop(0)

    console = guided._Console(read=read, write=lambda _text: None)
    return console


def test_wizard_composes_a_surface_constraint_request():
    console = _console([
        "3",          # surface_constraint
        "Cu", "100", "4", "12", "", "2x2", "bottom", "2",
        "",           # accept
    ])
    request = guided.run_wizard(console)
    assert request == (
        "Build a 2x2 Cu(100) slab (4 layers, 12 Å vacuum) and freeze the "
        "2 bottom layers."
    )
    assert check_request(request).missing == ()


def test_wizard_accepts_a_region_by_name():
    console = _console(["molecule", "H2O", "12", ""])
    assert guided.run_wizard(console) == (
        "Build an isolated H2O molecule in a 12 Å cubic box."
    )


def test_wizard_reasks_after_a_bad_answer():
    console = _console(["molecule", "unobtainium", "H2O", "12", ""])
    assert "H2O" in guided.run_wizard(console)


def test_wizard_back_command_reasks_the_previous_slot():
    console = _console(["molecule", "H2O", "b", "CO2", "12", ""])
    assert "CO2" in guided.run_wizard(console)


def test_wizard_cancels_on_q():
    console = _console(["molecule", "q"])
    assert guided.run_wizard(console) is None


def test_wizard_cancels_on_eof():
    assert guided.run_wizard(_console([])) is None


def test_wizard_region_argument_skips_the_menu():
    console = _console(["H2O", "12", ""])
    assert guided.run_wizard(console, region="molecule").startswith("Build an isolated")


def test_wizard_reports_slot_coverage():
    console = _console(["molecule", "H2O", "12", ""])
    guided.run_wizard(console)
    printed = "\n".join(console.lines)
    assert "[ok] pre-flight" in printed
    assert "composed request:" in printed


def test_wizard_surfaces_cross_check_warnings():
    console = _console([
        "surface_constraint", "Cu", "100", "2", "12", "", "", "bottom", "4", "",
    ])
    guided.run_wizard(console)
    assert any("[warn]" in line for line in console.lines)


# --------------------------------------------------------------------------- #
# Compound substrates
# --------------------------------------------------------------------------- #


COMPOUND_SAMPLES: dict[str, dict] = {
    "surface": {"element": "rocksalt-MgO", "facet": "100", "layers": 4,
                "vacuum": 12.0, "phase": None, "repeat": (2, 2, 1)},
    "surface_constraint": {"element": "rocksalt-MgO", "facet": "100", "layers": 4,
                           "vacuum": 12.0, "phase": None, "repeat": (2, 2, 1),
                           "freeze_side": "bottom", "freeze_count": 2},
    "atomic_adsorption": {"element": "wurtzite-ZnO", "facet": "001", "layers": 4,
                          "vacuum": 15.0, "phase": None, "repeat": (2, 2, 1),
                          "adsorbate": "H", "site": "ontop", "height": 1.5},
    "vacancy": {"element": "rocksalt-MgO", "phase": None, "defect_site": "first",
                "repeat": (2, 2, 1), "convention": None},
    "substitution": {"element": "rocksalt-MgO", "phase": None,
                     "defect_site": "first", "dopant": "Ca",
                     "repeat": (2, 2, 1), "convention": None},
    "bulk": {"element": "fluorite-CeO2", "phase": None, "repeat": (2, 2, 1),
             "convention": None},
}


@pytest.mark.parametrize("region", sorted(COMPOUND_SAMPLES))
def test_a_compound_request_is_still_slot_complete(region):
    """A compound names its own crystal family, so 'crystalline' stays covered."""
    advisory = check_request(guided.compose(region, COMPOUND_SAMPLES[region]))
    assert advisory.region == region
    assert advisory.missing == ()


@pytest.mark.parametrize("region", sorted(COMPOUND_SAMPLES))
def test_a_compound_request_keeps_the_hyphenated_name(region):
    """'rocksalt-MgO' is what r5 maps straight onto a prototype; a bare formula
    makes it guess the family and retry."""
    request = guided.compose(region, COMPOUND_SAMPLES[region])
    assert COMPOUND_SAMPLES[region]["element"] in request


@pytest.mark.parametrize("region", sorted(COMPOUND_SAMPLES))
def test_a_compound_request_has_no_double_spaces(region):
    """The phase placeholder is blanked for compounds; whitespace must collapse."""
    assert "  " not in guided.compose(region, COMPOUND_SAMPLES[region])


@pytest.mark.parametrize("raw,expected", [
    ("Cu", "Cu"), ("cu", "Cu"), ("Fe", "Fe"),
    ("MgO", "rocksalt-MgO"), ("mgo", "rocksalt-MgO"),
    ("rocksalt-NiO", "rocksalt-NiO"), ("rocksalt NiO", "rocksalt-NiO"),
    ("CeO2", "fluorite-CeO2"), ("ZnO", "wurtzite-ZnO"),
])
def test_parse_substrate_accepts_elements_and_compounds(raw, expected):
    assert guided.parse_substrate(raw) == expected


@pytest.mark.parametrize("raw", ["Xx", "unobtainium", ""])
def test_parse_substrate_rejects_nonsense(raw):
    with pytest.raises(guided.SlotError):
        guided.parse_substrate(raw)


def test_parse_substrate_forwards_the_ambiguity_message():
    with pytest.raises(ValueError, match="ambiguous"):
        guided.parse_substrate("ZnS")


def test_a_compound_substrate_skips_the_phase_question():
    """The family already names the phase; asking again would be redundant."""
    console = _console(["surface", "MgO", "100", "4", "12", "2x2", ""])
    request = guided.run_wizard(console)
    assert request == (
        "Build a 2x2 rocksalt-MgO(100) slab with 4 layers and 12 Å vacuum."
    )


def test_an_elemental_substrate_still_asks_for_the_phase():
    console = _console(["surface", "Cu", "111", "4", "12", "fcc", "2x2", ""])
    assert "fcc Cu(111)" in guided.run_wizard(console)


def test_a_required_phase_is_skipped_for_a_compound():
    """vacancy requires 'crystalline', but 'rocksalt-MgO' already states it."""
    console = _console(["vacancy", "MgO", "1", "2x2x1", "", ""])
    request = guided.run_wizard(console)
    assert request == (
        "Build a 2x2x1 rocksalt-MgO supercell and remove the first atom."
    )
    assert check_request(request).missing == ()


def test_the_phase_question_returns_when_stepping_back_to_an_element():
    """'b' back to the substrate and answering an element must re-ask the phase."""
    console = _console(["surface", "MgO", "100", "b", "b", "Cu", "111", "4", "12",
                        "fcc", "2x2", ""])
    assert "fcc Cu(111)" in guided.run_wizard(console)


def test_wizard_edit_reasks_then_builds():
    console = _console(["molecule", "H2O", "12", "e", "CO2", "12", ""])
    assert "CO2" in guided.run_wizard(console)
