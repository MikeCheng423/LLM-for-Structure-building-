from __future__ import annotations

import pytest

pytest.importorskip("ase")

from ase_auto_build.ase_agent import ASEWorkspace, AgentPolicy, create_default_registry
from ase_auto_build.ase_agent.registry import SchemaValidationError
from ase_auto_build.ase_agent.selectors import geometric_layers, select_atoms
from ase_auto_build.ase_agent.validation import atoms_hash


def workspace() -> ASEWorkspace:
    return ASEWorkspace(create_default_registry(), session_id="test-session")


def test_registry_exports_one_schema_per_tool() -> None:
    registry = create_default_registry()
    assert len(registry) >= 25
    schemas = registry.function_schemas()
    assert len(schemas) == len(registry)
    assert len({item["function"]["name"] for item in schemas}) == len(schemas)
    assert all(item["function"]["parameters"]["additionalProperties"] is False for item in schemas)


def test_schema_rejects_unknown_fields_and_wrong_types() -> None:
    registry = create_default_registry()
    with pytest.raises(SchemaValidationError, match="unknown fields"):
        registry.validate_arguments("build_bulk", {"name": "al", "element": "Al", "shell": "id"})
    with pytest.raises(SchemaValidationError, match="must be integer"):
        registry.validate_arguments("repeat", {"name": "al", "repeat": [2, 2, 1.5]})


def test_transaction_rollback_preserves_active_revision() -> None:
    ws = workspace()
    ws.execute_or_raise("build_bulk", {"name": "al", "element": "Al", "crystal": "fcc", "cubic": True})
    before = ws.active_revision("al")
    failed = ws.execute("add_atom", {"name": "al", "element": "Al", "position": [0.0, 0.0, 0.0]})
    assert not failed.success
    assert failed.error["code"] == "ToolExecutionError"
    assert ws.active_revision("al").revision_id == before.revision_id
    assert atoms_hash(ws.require_atoms("al")) == before.output_hash


def test_recipe_replay_is_hash_equivalent() -> None:
    ws = workspace()
    ws.execute_or_raise("build_bulk", {
        "name": "pt", "element": "Pt", "crystal": "fcc", "a": 3.92,
        "cubic": True,
    })
    ws.execute_or_raise("repeat", {"name": "pt", "repeat": [2, 2, 1]})
    ws.execute_or_raise("substitute", {
        "name": "pt", "selector": {"indices": [1]}, "element": "Au",
    })
    ws.execute_or_raise("finish", {"name": "pt"})
    expected_hash = atoms_hash(ws.final_atoms())
    recipe = ws.recipe()

    replay = workspace()
    for step in recipe["steps"]:
        replay.execute_or_raise(step["tool"], step["args"])
    assert replay.recipe_hash() == ws.recipe_hash()
    assert atoms_hash(replay.final_atoms()) == expected_hash


def test_adsorbate_site_selectors_are_mutually_exclusive() -> None:
    """``xy`` overrides ``site``/``site_index`` silently, so reject the combination.

    Passing both used to build a valid structure at the wrong place: the r7 adapter
    emitted the correct call plus a spurious ``xy``, displacing CO ~4 A laterally
    while invariants still passed.
    """
    registry = create_default_registry()
    for tool, base in (
        ("add_atomic_adsorbate", {"name": "slab", "element": "O"}),
        ("add_molecular_adsorbate", {"name": "slab", "species": "CO", "anchor": 2}),
    ):
        # Either selector alone stays valid.
        registry.validate_arguments(tool, {**base, "site": "ontop", "site_index": 1})
        registry.validate_arguments(tool, {**base, "xy": [1.0, 2.0]})
        for conflicting in ({"site": "ontop"}, {"site_index": 1}):
            with pytest.raises(SchemaValidationError, match="mutually exclusive"):
                registry.validate_arguments(tool, {**base, "xy": [0.0, 0.0], **conflicting})


def test_surface_adsorbate_and_bottom_layer_constraint() -> None:
    ws = workspace()
    ws.execute_or_raise("build_surface", {
        "name": "slab", "element": "Pt", "crystal": "fcc",
        "miller": [1, 1, 1], "layers": 4, "vacuum": 12.0,
        "repeat": [2, 2, 1],
    })
    before = len(ws.require_atoms("slab"))
    ws.execute_or_raise("add_atomic_adsorbate", {
        "name": "slab", "element": "O", "site": "ontop", "height": 1.8,
    })
    ws.execute_or_raise("freeze_layers", {
        "name": "slab", "side": "bottom", "layers": 2, "axes": "xyz",
    })
    result = ws.execute_or_raise("finish", {"name": "slab"})
    atoms = ws.final_atoms()
    assert len(atoms) == before + 1
    assert atoms.get_chemical_symbols().count("O") == 1
    assert atoms.constraints
    assert result.observation["finished"] is True


def test_typed_layer_and_element_selection_intersect() -> None:
    ws = workspace()
    ws.execute_or_raise("build_surface", {
        "name": "slab", "element": "Cu", "crystal": "fcc",
        "miller": [1, 0, 0], "layers": 3, "vacuum": 10.0,
    })
    atoms = ws.require_atoms("slab")
    assert len(geometric_layers(atoms)) == 3
    selected = select_atoms(atoms, {"element": ["Cu"], "layer": {"side": "bottom", "count": 1}})
    assert selected
    assert len(selected) < len(atoms)


def test_read_only_tools_do_not_create_revisions() -> None:
    ws = workspace()
    ws.execute_or_raise("build_molecule", {"name": "water", "species": "H2O", "box": 10.0})
    revision = ws.active_revision("water").revision_id
    result = ws.execute_or_raise("inspect_structure", {"name": "water"})
    assert result.observation["formula"] == "H2O"
    assert ws.active_revision("water").revision_id == revision


def test_hash_canonicalizes_signed_zero_and_blas_noise() -> None:
    ws = workspace()
    ws.execute_or_raise("build_bulk", {"name": "al", "element": "Al", "crystal": "fcc"})
    left = ws.require_atoms("al")
    right = left.copy()
    cell = right.cell.array.copy()
    cell[0, 0] = -3.4e-17
    right.set_cell(cell, scale_atoms=False)
    assert atoms_hash(left) == atoms_hash(right)


def test_structure_name_cannot_be_a_path() -> None:
    ws = workspace()
    result = ws.execute(
        "build_bulk",
        {"name": "../../escape", "element": "Al", "crystal": "fcc"},
    )
    assert result.success is False
    assert ws.structures == {}
    assert "path-free identifier" in result.error["message"]


def test_repeat_atom_budget_is_checked_before_allocation() -> None:
    ws = ASEWorkspace(
        create_default_registry(),
        policy=AgentPolicy(max_atoms=8),
        session_id="atom-budget-test",
    )
    ws.execute_or_raise(
        "build_bulk",
        {"name": "al", "element": "Al", "crystal": "fcc", "cubic": True},
    )
    revision = ws.active_revision("al")
    result = ws.execute("repeat", {"name": "al", "repeat": [2, 2, 2]})
    assert result.success is False
    assert "would create 32 atoms" in result.error["message"]
    assert ws.active_revision("al").revision_id == revision.revision_id


def test_nanotube_atom_budget_is_checked_before_build() -> None:
    ws = workspace()
    result = ws.execute(
        "build_nanotube",
        {"name": "tube", "n": 100, "m": 99, "length": 50},
    )
    assert result.success is False
    assert "limit is 2000" in result.error["message"]
    assert ws.structures == {}


def test_omitted_name_defaults_to_active_structure() -> None:
    ws = workspace()
    built = ws.execute(
        "build_surface",
        {"crystal": "fcc", "element": "Cu", "layers": 3, "miller": [1, 0, 0], "vacuum": 12.0},
    )
    assert built.success is True
    assert ws.active_name == "structure"
    adsorbed = ws.execute("add_atomic_adsorbate", {"element": "O", "site": "ontop", "height": 1.8})
    assert adsorbed.success is True
    assert adsorbed.observation["formula"].startswith("Cu")
    assert ws.execute("finish", {}).success is True
    assert ws.result_name == "structure"


def test_explicit_name_still_honored_alongside_default() -> None:
    ws = workspace()
    ws.execute_or_raise("build_bulk", {"name": "al", "element": "Al", "crystal": "fcc", "cubic": True})
    # An omitted name resolves to the active structure, not a new default.
    result = ws.execute("repeat", {"repeat": [2, 2, 1]})
    assert result.success is True
    assert result.observation["name"] == "al"
