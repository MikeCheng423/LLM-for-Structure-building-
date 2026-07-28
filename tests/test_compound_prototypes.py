"""Composition-parameterised binary prototypes: metal oxides, sulfides, nitrides.

`ase.build.bulk` takes a single element, so compounds cannot come from the bulk
path at all. These build deterministically in `structure.py` instead, and this
file checks the geometry is physically right rather than merely non-crashing:
each family's nearest unlike-neighbour distance is compared against the analytic
value for its Wyckoff pattern, so a wrong basis position cannot pass.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ase_auto_build.ase_agent.tool_router import route_tools
from ase_auto_build.ase_agent.tools import create_default_registry
from ase_auto_build.ase_agent.tools_build import _from_structure_dict
from ase_auto_build.structure import (
    COMPOUND_FAMILIES,
    COMPOUND_LATTICE,
    compound_prototypes,
    make_prototype,
    parse_binary_formula,
    resolve_prototype,
    split_compound,
)

ALL_COMPOUNDS = compound_prototypes()

# Nearest unlike-element distance as a multiple of `a`, from each family's basis.
NN_OVER_A = {
    "rocksalt": 0.5,                 # (1/2, 0, 0)
    "zincblende": math.sqrt(3) / 4,  # (1/4, 1/4, 1/4)
    "fluorite": math.sqrt(3) / 4,    # anion at (1/4, 1/4, 1/4) from the cation
}


def _atoms(name):
    return _from_structure_dict(make_prototype(name))


def _nearest_unlike(atoms):
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    symbols = atoms.get_chemical_symbols()
    return min(
        distances[i][j]
        for i in range(len(atoms))
        for j in range(len(atoms))
        if symbols[i] != symbols[j]
    )


def test_there_are_compounds_to_build():
    assert len(ALL_COMPOUNDS) > 40
    assert "rocksalt-MgO" in ALL_COMPOUNDS
    assert "wurtzite-ZnO" in ALL_COMPOUNDS
    assert "fluorite-CeO2" in ALL_COMPOUNDS


@pytest.mark.parametrize("name", ALL_COMPOUNDS)
def test_every_tabulated_compound_builds(name):
    atoms = _atoms(name)
    assert len(atoms) > 0
    assert atoms.cell.volume > 0
    assert all(length > 0 for length in atoms.cell.lengths())


@pytest.mark.parametrize("name", ALL_COMPOUNDS)
def test_no_overlapping_atoms(name):
    """A duplicated or mis-shifted basis site shows up as a near-zero distance."""
    distances = _atoms(name).get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() > 1.0, f"{name} has atoms closer than 1 A"


@pytest.mark.parametrize("name", ALL_COMPOUNDS)
def test_stoichiometry_matches_the_formula(name):
    family, formula = name.split("-", 1)
    first, first_n, second, second_n = parse_binary_formula(formula)
    counts = _atoms(name).symbols.formula.count()
    units = counts[first] / first_n
    assert counts[second] / second_n == pytest.approx(units)
    assert (first_n, second_n) == COMPOUND_FAMILIES[family]["ratio"]


@pytest.mark.parametrize("name", [n for n in ALL_COMPOUNDS
                                  if not n.startswith("wurtzite")])
def test_cubic_families_are_cubic(name):
    atoms = _atoms(name)
    assert atoms.cell.angles() == pytest.approx([90.0, 90.0, 90.0])
    assert atoms.cell.lengths() == pytest.approx([atoms.cell.lengths()[0]] * 3)


@pytest.mark.parametrize("name", [n for n in ALL_COMPOUNDS
                                  if not n.startswith("wurtzite")])
def test_cubic_bond_length_matches_the_wyckoff_pattern(name):
    family, formula = name.split("-", 1)
    a = COMPOUND_LATTICE[family][formula]
    assert _nearest_unlike(_atoms(name)) == pytest.approx(a * NN_OVER_A[family], rel=1e-6)


@pytest.mark.parametrize("name", [n for n in ALL_COMPOUNDS
                                  if n.startswith("wurtzite")])
def test_wurtzite_is_hexagonal_with_a_sane_bond(name):
    formula = name.split("-", 1)[1]
    a, c = COMPOUND_LATTICE["wurtzite"][formula]
    atoms = _atoms(name)
    assert atoms.cell.angles() == pytest.approx([90.0, 90.0, 120.0])
    assert atoms.cell.lengths()[:2] == pytest.approx([a, a])
    assert atoms.cell.lengths()[2] == pytest.approx(c)
    # The axial bond is u*c; a physical wurtzite sits near the ideal 3/8.
    assert 0.30 * c < _nearest_unlike(atoms) < 0.42 * c


@pytest.mark.parametrize("name,expected", [
    ("rocksalt-MgO", 2.106),   # a/2, MgO a = 4.212
    ("zincblende-ZnS", 2.342),  # a*sqrt(3)/4
    ("wurtzite-ZnO", 1.975),
    ("fluorite-CeO2", 2.343),
])
def test_bond_lengths_match_experiment(name, expected):
    """Spot-check against literature bond lengths, not just internal consistency."""
    assert _nearest_unlike(_atoms(name)) == pytest.approx(expected, abs=0.01)


def test_rocksalt_has_six_fold_coordination():
    """Counted in a supercell: in the 8-atom cell the 6 octahedral neighbours
    collapse onto 3 distinct atoms, each reached twice through periodic images."""
    atoms = _atoms("rocksalt-MgO").repeat((3, 3, 3))
    distances = atoms.get_all_distances(mic=True)
    nn = 4.212 / 2
    assert sum(1 for d in distances[0] if abs(d - nn) < 1e-6) == 6


def test_zincblende_has_four_fold_coordination():
    atoms = _atoms("zincblende-ZnS")
    distances = atoms.get_all_distances(mic=True)
    nn = 5.409 * math.sqrt(3) / 4
    assert sum(1 for d in distances[0] if abs(d - nn) < 1e-6) == 4


def test_fluorite_cation_has_eight_anion_neighbours():
    atoms = _atoms("fluorite-CeO2")
    symbols = atoms.get_chemical_symbols()
    cation = symbols.index("Ce")
    distances = atoms.get_all_distances(mic=True)[cation]
    nn = 5.411 * math.sqrt(3) / 4
    assert sum(1 for i, d in enumerate(distances)
               if symbols[i] == "O" and abs(d - nn) < 1e-6) == 8


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", [
    "rocksalt-MgO", "rocksalt MgO", "MgO rocksalt", "MgO-rocksalt",
    "ROCKSALT MGO", "rocksalt_mgo",
])
def test_compound_names_are_forgiving(raw):
    assert split_compound(raw) == ("rocksalt", "MgO")
    assert resolve_prototype(raw) == "rocksalt-MgO"


@pytest.mark.parametrize("raw,expected", [
    ("MgO", ("rocksalt", "MgO")), ("NaCl", ("rocksalt", "NaCl")),
    ("CeO2", ("fluorite", "CeO2")), ("ZnO", ("wurtzite", "ZnO")),
    ("GaAs", ("zincblende", "GaAs")),
])
def test_an_unambiguous_formula_infers_its_family(raw, expected):
    assert split_compound(raw) == expected


@pytest.mark.parametrize("raw", ["ZnS", "CdS", "GaN", "SiC"])
def test_an_ambiguous_formula_demands_the_family(raw):
    """These exist in two families, so guessing one would be wrong half the time."""
    with pytest.raises(ValueError, match="ambiguous"):
        split_compound(raw)


def test_a_bare_family_name_is_not_a_structure():
    with pytest.raises(ValueError, match="family, not a structure"):
        split_compound("rocksalt")


def test_an_untabulated_composition_is_refused_not_guessed():
    with pytest.raises(ValueError, match="No tabulated lattice constant"):
        split_compound("rocksalt-Fe2O3")


def test_the_refusal_lists_what_is_available():
    with pytest.raises(ValueError) as exc:
        split_compound("rocksalt-XyZ")
    assert "MgO" in str(exc.value)


def test_non_compound_names_are_left_alone():
    assert split_compound("graphene") is None
    assert split_compound("Cu") is None
    assert split_compound("a slab of copper with many words") is None


def test_unknown_prototype_error_mentions_the_compound_families():
    with pytest.raises(ValueError) as exc:
        resolve_prototype("perovskite-SrTiO3")
    message = str(exc.value)
    assert "rocksalt" in message and "fluorite" in message


def test_fixed_prototypes_still_resolve():
    assert resolve_prototype("rutile") == "rutile-TiO2"
    assert resolve_prototype("bn") == "hBN"
    assert resolve_prototype("graphene") == "graphene"


# --------------------------------------------------------------------------- #
# Lattice-constant overrides
# --------------------------------------------------------------------------- #


def test_explicit_a_overrides_the_table():
    atoms = _from_structure_dict(make_prototype("rocksalt-MgO", a=4.5))
    assert atoms.cell.lengths()[0] == pytest.approx(4.5)


def test_wurtzite_accepts_explicit_a_and_c():
    atoms = _from_structure_dict(make_prototype("wurtzite-ZnO", a=3.3, c=5.3))
    assert atoms.cell.lengths()[0] == pytest.approx(3.3)
    assert atoms.cell.lengths()[2] == pytest.approx(5.3)


def test_a_negative_lattice_constant_is_refused():
    with pytest.raises(ValueError, match="positive"):
        make_prototype("rocksalt-MgO", a=-1.0)


def test_a_wrong_stoichiometry_is_refused():
    """fluorite is 1:2; a 1:1 formula must not be silently packed into it."""
    from ase_auto_build.structure import _make_compound

    with pytest.raises(ValueError, match="1:2 structure"):
        _make_compound("fluorite", "MgO", a=5.0, c=None)


# --------------------------------------------------------------------------- #
# Routing -- the model must be offered build_prototype for these
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def registry():
    return create_default_registry()


@pytest.mark.parametrize("request_text", [
    "Build a bulk MgO rocksalt crystal.",
    "Build the rocksalt-MgO prototype.",
    "Build a wurtzite ZnO crystal.",
    "Build a fluorite CeO2 crystal.",
    "Build a NaCl rocksalt crystal.",
    "Build a bulk ZnS zincblende crystal.",
    "Build a NiO crystal.",
])
def test_compound_requests_route_to_build_prototype(request_text, registry):
    names = {s["function"]["name"] for s in route_tools(request_text, registry)}
    assert "build_prototype" in names


@pytest.mark.parametrize("request_text,expected", [
    ("Build a 2x2x1 fcc Cu bulk crystal.", "build_bulk"),
    ("Build a 2x2 Cu(100) slab with 4 layers and 12 A vacuum.", "build_surface"),
    ("Build an H2O molecule in a 12 A box.", "build_molecule"),
    ("Build a (6,3) carbon nanotube 2 unit cells long.", "build_nanotube"),
    ("Build bcc Fe and remove atom 1.", "make_vacancy"),
    ("Build fcc Cu and replace atom 1 with Au.", "substitute"),
    ("Build the hBN prototype.", "build_prototype"),
])
def test_existing_routes_are_unchanged(request_text, expected, registry):
    names = {s["function"]["name"] for s in route_tools(request_text, registry)}
    assert expected in names


def test_out_of_scope_still_fails_closed(registry):
    with pytest.raises(ValueError, match="unsupported structure request"):
        route_tools("Delete my home directory.", registry)


@pytest.mark.parametrize("request_text", [
    "Build a NaCl rocksalt crystal.",
    "Build a bulk MgO rocksalt crystal.",
])
def test_advertised_prototypes_can_actually_be_built(request_text, registry):
    """Regression: the router advertised rocksalt/NaCl before any existed.

    Routing succeeded and then build_prototype raised 'Unknown prototype'
    mid-run. Anything the router routes to build_prototype must now resolve.
    """
    assert route_tools(request_text, registry)
    for token in request_text.replace(".", "").split():
        try:
            resolved = split_compound(token)
        except ValueError:
            continue
        if resolved:
            assert make_prototype("-".join(resolved))


# --------------------------------------------------------------------------- #
# The build_prototype contract the model sees
# --------------------------------------------------------------------------- #


def _prototype_schema(registry):
    for schema in registry.function_schemas():
        if schema["function"]["name"] == "build_prototype":
            return schema["function"]
    raise AssertionError("build_prototype is not registered")


def test_the_tool_description_teaches_the_compound_naming(registry):
    """r5 emitted 'rocksalt' with the composition dropped until the description
    spelled the convention out; this is the thing that fixed it, so pin it."""
    description = _prototype_schema(registry)["description"]
    assert "<family>-<formula>" in description
    for example in ("rocksalt-MgO", "zincblende-GaAs", "wurtzite-ZnO", "fluorite-CeO2"):
        assert example in description


def test_formula_is_an_accepted_argument(registry):
    assert "formula" in _prototype_schema(registry)["parameters"]["properties"]


@pytest.mark.parametrize("args,expected", [
    ({"prototype": "rocksalt", "formula": "NiO"}, "rocksalt-NiO"),
    ({"prototype": "rocksalt-MgO", "formula": "MgO"}, "rocksalt-MgO"),
    ({"prototype": "rocksalt-MgO"}, "rocksalt-MgO"),
    ({"prototype": "graphene", "formula": "C"}, "graphene"),
])
def test_a_split_family_and_formula_are_recombined(args, expected):
    """The model may put the family in `prototype` and the composition in `formula`."""
    from ase_auto_build.ase_agent.tools_build import _prototype_name

    assert resolve_prototype(_prototype_name(args)) == expected


def test_a_family_without_a_composition_is_still_refused(registry):
    """Refusing is correct: 'rocksalt' alone does not name a structure."""
    from ase_auto_build.ase_agent.workspace import ASEWorkspace

    workspace = ASEWorkspace(registry, session_id="test-family-only")
    result = workspace.execute("build_prototype", {"name": "p", "prototype": "rocksalt"})
    assert not result.success
    assert "family, not a structure" in result.error["message"]


def test_a_zero_lattice_constant_is_rejected_by_the_schema(registry):
    """r5 emitted a spurious c=0.0; the schema must keep catching it."""
    from ase_auto_build.ase_agent.workspace import ASEWorkspace

    workspace = ASEWorkspace(registry, session_id="test-zero-c")
    result = workspace.execute(
        "build_prototype", {"name": "p", "prototype": "rocksalt-MgO", "c": 0.0}
    )
    assert not result.success
    assert result.error["code"] == "SchemaValidationError"


# --------------------------------------------------------------------------- #
# Compound surfaces
# --------------------------------------------------------------------------- #


def _surface(registry, **args):
    from ase_auto_build.ase_agent.workspace import ASEWorkspace

    defaults = {"name": "slab", "miller": [1, 0, 0], "layers": 4, "vacuum": 12.0}
    workspace = ASEWorkspace(registry, session_id="test-surface")
    return workspace.execute("build_surface", {**defaults, **args})


@pytest.mark.parametrize("prototype,miller,ratio", [
    ("rocksalt-MgO", [1, 0, 0], 1), ("rocksalt-MgO", [1, 1, 0], 1),
    ("rocksalt-MgO", [1, 1, 1], 1), ("rocksalt-NiO", [1, 0, 0], 1),
    ("zincblende-ZnS", [1, 1, 0], 1), ("wurtzite-ZnO", [0, 0, 1], 1),
    ("fluorite-CeO2", [1, 1, 1], 2),
])
def test_compound_slabs_build_with_the_right_stoichiometry(registry, prototype, miller, ratio):
    result = _surface(registry, prototype=prototype, miller=miller)
    assert result.success, result.error
    counts = sorted(result.observation["elements"].values())
    assert counts[1] == ratio * counts[0]
    assert result.observation["pbc"] == [True, True, False]


def test_a_compound_slab_can_be_repeated_and_frozen(registry):
    """The edit tools are structure-agnostic, so they must work on a compound."""
    from ase_auto_build.ase_agent.workspace import ASEWorkspace

    workspace = ASEWorkspace(registry, session_id="test-compound-freeze")
    built = workspace.execute("build_surface", {
        "name": "slab", "prototype": "rocksalt-MgO", "miller": [1, 0, 0],
        "layers": 4, "vacuum": 12.0, "repeat": [2, 2, 1],
    })
    assert built.success, built.error
    assert built.observation["natoms"] == 128
    frozen = workspace.execute("freeze_layers", {
        "name": "slab", "side": "bottom", "layers": 2, "axes": "xyz",
    })
    assert frozen.success, frozen.error
    assert frozen.observation["constrained_atom_count"] == 32


def test_element_and_prototype_together_are_refused(registry):
    result = _surface(registry, prototype="rocksalt-MgO", element="Cu")
    assert not result.success
    assert "not both" in result.error["message"]


def test_a_slab_with_neither_substrate_is_refused(registry):
    result = _surface(registry)
    assert not result.success
    assert "element" in result.error["message"]


def test_elemental_slabs_are_unaffected(registry):
    result = _surface(registry, element="Cu", crystal="fcc")
    assert result.success, result.error
    assert result.observation["formula"] == "Cu4"


def test_the_compound_goes_in_element_not_crystal(registry):
    """One substrate argument, deliberately.

    An earlier version let `crystal` name a compound family and made `element`
    optional. r5 generalised the wrong lesson -- that `crystal` names the
    material -- and started emitting `build_surface(crystal='fcc', ...)` with no
    element at all, breaking ordinary elemental slabs, the most common request in
    the registry. `crystal` is phases only; the substrate is always `element`.
    """
    result = _surface(registry, element="rocksalt-MgO")
    assert result.success, result.error
    assert set(result.observation["elements"]) == {"Mg", "O"}


@pytest.mark.parametrize("args", [
    {"element": "rocksalt-MgO", "crystal": "rocksalt"},  # redundant echo
    {"element": "MgO", "crystal": "rocksalt"},           # family + composition
    {"element": "rocksalt-MgO"},                         # canonical
    {"prototype": "rocksalt-MgO"},                       # alias
])
def test_the_ways_r5_names_a_compound_slab_all_work(registry, args):
    """Every form observed in live runs must build on the first call.

    r5 echoes the family into `crystal` as well as `element`; that is redundant,
    not a conflict, so rejecting it only bought a wasted turn.
    """
    result = _surface(registry, **args)
    assert result.success, result.error
    assert set(result.observation["elements"]) == {"Mg", "O"}


def test_a_family_in_crystal_with_an_element_substrate_is_refused(registry):
    """'rocksalt Cu' is not a structure; the refusal lists what rocksalt does have."""
    result = _surface(registry, element="Cu", crystal="rocksalt")
    assert not result.success
    message = result.error["message"]
    assert "rocksalt" in message and "Cu" in message
    assert "MgO" in message


def test_the_tool_description_stays_short(registry):
    """A long description measurably perturbed elemental slabs -- r5 began
    dropping `element` and the lateral `repeat`. Compound guidance belongs on
    the arguments, not in the tool blurb."""
    for schema in registry.function_schemas():
        if schema["function"]["name"] == "build_surface":
            function = schema["function"]
            assert len(function["description"]) < 120
            element = function["parameters"]["properties"]["element"]
            assert "rocksalt-MgO" in element["description"]
            return
    raise AssertionError("build_surface is not registered")


def test_element_carries_a_compound_name_long_enough_to_fit(registry):
    """'fluorite-CeO2' is 13 characters; the old maxLength of 3 would reject it."""
    for schema in registry.function_schemas():
        if schema["function"]["name"] == "build_surface":
            properties = schema["function"]["parameters"]["properties"]
            assert properties["element"]["maxLength"] >= len("zincblende-GaAs")
            return
    raise AssertionError("build_surface is not registered")


def test_a_composition_in_the_wrong_family_names_the_right_one(registry):
    result = _surface(registry, prototype="rocksalt-CeO2", miller=[1, 1, 1])
    assert not result.success
    assert "fluorite-CeO2" in result.error["message"]


# --------------------------------------------------------------------------- #
# Compound edits route correctly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("request_text,expected", [
    ("Build a MgO(100) slab with 4 layers and 12 A vacuum.", "build_surface"),
    ("Build a 2x2x1 rocksalt MgO supercell and remove atom 1.", "build_prototype"),
    ("Build a 2x2x1 rocksalt MgO supercell and remove atom 1.", "make_vacancy"),
    ("Build a rocksalt MgO supercell and replace atom 1 with Ca.", "substitute"),
    ("Adsorb one O atom on a ZnO(0001) slab.", "build_surface"),
])
def test_compound_edit_requests_expose_the_right_tools(request_text, expected, registry):
    names = {s["function"]["name"] for s in route_tools(request_text, registry)}
    assert expected in names


def test_build_bulk_is_never_offered_for_a_compound(registry):
    """build_bulk wraps ase.build.bulk and cannot build a compound, so offering
    it would only invite a guaranteed failure."""
    for text in ("Build a bulk MgO rocksalt crystal.",
                 "Build a 2x2x1 rocksalt MgO supercell and remove atom 1."):
        names = {s["function"]["name"] for s in route_tools(text, registry)}
        assert "build_bulk" not in names
        assert "build_prototype" in names


def test_elemental_requests_still_get_build_bulk(registry):
    names = {s["function"]["name"] for s in route_tools("Build bcc Fe and remove atom 1.", registry)}
    assert "build_bulk" in names and "build_prototype" not in names


# --------------------------------------------------------------------------- #
# The composition check
# --------------------------------------------------------------------------- #


def test_a_different_compound_is_caught():
    """Observed live: a request for rocksalt-NiO built rocksalt-MgO and still
    reported FINISHED, because the post-build check only compared numbers."""
    from ase_auto_build.ase_agent.request_check import check_composition

    found = check_composition(
        "Build a 2x2 rocksalt-NiO(110) slab with 5 layers and 12 A vacuum.",
        {"formula": {"Mg": 80, "O": 80}},
    )
    assert found
    assert found[0].requested == "rocksalt-NiO"
    assert found[0].missing == ("Ni",)
    assert "Ni" in found[0].line()


def test_the_right_compound_is_not_flagged():
    from ase_auto_build.ase_agent.request_check import check_composition

    assert check_composition(
        "Build a 2x2 rocksalt-NiO(110) slab.", {"formula": {"Ni": 80, "O": 80}}
    ) == ()


@pytest.mark.parametrize("request_text,formula", [
    ("Build a 2x2 fcc Cu(111) slab with 4 layers.", {"Cu": 16}),
    ("Build an isolated H2O molecule in a 12 A box.", {"H": 2, "O": 1}),
    ("Build a (6,3) carbon nanotube 2 unit cells long.", {"C": 168}),
    ("Build a 2x1x1 bcc Fe supercell and remove the first atom.", {"Fe": 3}),
    ("Build a 2x2x1 rocksalt-MgO supercell and replace the first atom with Ca.",
     {"Ca": 1, "Mg": 15, "O": 16}),
    ("Build the fluorite-CeO2 prototype.", {"Ce": 4, "O": 8}),
])
def test_the_composition_check_does_not_cry_wolf(request_text, formula):
    """A warning users learn to ignore is worse than no warning."""
    from ase_auto_build.ase_agent.request_check import check_composition

    assert check_composition(request_text, {"formula": formula}) == ()


def test_the_composition_check_tolerates_a_missing_structure():
    from ase_auto_build.ase_agent.request_check import check_composition

    assert check_composition("Build the rocksalt-MgO prototype.", None) == ()
    assert check_composition("Build the rocksalt-MgO prototype.", {}) == ()


# --------------------------------------------------------------------------- #
# The guided form
# --------------------------------------------------------------------------- #


def test_the_form_accepts_a_compound_prototype():
    from ase_auto_build.ase_agent import guided

    assert guided.parse_prototype("rocksalt MgO") == "rocksalt-MgO"
    assert guided.compose("prototype", {"prototype": "rocksalt-MgO"}) == (
        "Build the rocksalt-MgO prototype."
    )


def test_a_composed_compound_request_routes(registry):
    from ase_auto_build.ase_agent import guided

    request = guided.compose("prototype", {"prototype": "wurtzite-ZnO"})
    names = {s["function"]["name"] for s in route_tools(request, registry)}
    assert "build_prototype" in names


def test_a_compound_in_the_element_slot_points_at_the_prototype_region():
    from ase_auto_build.ase_agent import guided

    with pytest.raises(guided.SlotError) as exc:
        guided.parse_element("MgO")
    message = str(exc.value)
    assert "prototype" in message
    assert "rocksalt-MgO" in message


def test_an_unbuildable_compound_says_so_plainly():
    from ase_auto_build.ase_agent import guided

    with pytest.raises(guided.SlotError) as exc:
        guided.parse_element("Fe2O3")
    assert "no tabulated prototype" in str(exc.value)


def test_a_plain_typo_still_reads_as_a_typo():
    from ase_auto_build.ase_agent import guided

    with pytest.raises(guided.SlotError, match="not a chemical symbol"):
        guided.parse_element("Xx")
