from __future__ import annotations

import pytest

pytest.importorskip("ase")

from ase_auto_build.ase_agent import create_default_registry
from ase_auto_build.ase_agent.tool_router import route_tools
from training.evaluations.evaluate_corpus import evaluate_record
from training.generators.build_catalyst_supplement import catalyst_cases
from training.generators.generate_corpus import execute_case


def test_catalyst_supplement_cases_are_routable_and_replayable() -> None:
    registry = create_default_registry()
    families = {}
    for case in catalyst_cases():
        routed = {item["function"]["name"] for item in route_tools(case.descriptions[0], registry)}
        assert {step["tool"] for step in case.steps} <= routed
        families.setdefault(case.family, case)

    assert set(families) == {
        "catalyst_adsorption_site", "catalyst_high_index", "catalyst_nanoparticle",
        "catalyst_oxide_surface", "catalyst_supported_cluster",
        "catalyst_surface_substitution", "catalyst_surface_vacancy",
    }
    for case in families.values():
        result = evaluate_record(execute_case(case, case.descriptions[0], 0))
        assert result["output_hash"]


@pytest.mark.parametrize("shape,shells,natoms", [
    ("icosahedron", 2, 13),
    ("cuboctahedron", 2, 55),
    ("octahedron", 1, 19),
])
def test_nanoparticle_builder_is_deterministic(shape, shells, natoms) -> None:
    from ase_auto_build.ase_agent import ASEWorkspace

    workspace = ASEWorkspace(create_default_registry(), session_id=f"test-{shape}")
    result = workspace.execute("build_nanoparticle", {
        "name": "particle", "element": "Pt", "shape": shape,
        "shells": shells, "vacuum": 8.0,
    })
    assert result.success, result.error
    assert result.observation["natoms"] == natoms
    assert result.observation["pbc"] == [False, False, False]


@pytest.mark.parametrize("word,edit", [("vacancy", "make_vacancy"), ("substitute", "substitute")])
def test_surface_defect_router_composes_builder_and_edit(word, edit) -> None:
    routed = {
        item["function"]["name"]
        for item in route_tools(
            f"Build a Pt(111) slab and {word} the first atom in the top layer.",
            create_default_registry(),
        )
    }
    assert routed == {"build_surface", edit, "finish"}


def test_supported_cluster_router_exposes_only_the_composed_recipe() -> None:
    routed = {
        item["function"]["name"]
        for item in route_tools(
            "Build a two-shell icosahedral Pt cluster supported on a fluorite-CeO2(111) slab.",
            create_default_registry(),
        )
    }
    assert routed == {"build_surface", "build_nanoparticle", "combine", "finish"}


@pytest.mark.parametrize("request_text", [
    "Build a Pt cluster supported on CeO2.",
    "Build a Pt surface vacancy.",
])
def test_underspecified_new_families_can_clarify(request_text) -> None:
    routed = {
        item["function"]["name"]
        for item in route_tools(request_text, create_default_registry())
    }
    assert "ask_clarification" in routed
