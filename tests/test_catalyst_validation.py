from __future__ import annotations

from copy import deepcopy

from ase_auto_build.ase_agent.catalyst_dispatch import dispatch_spec
from ase_auto_build.ase_agent.catalyst_validation import catalyst_validation_rules
from tests.test_catalyst_contracts import records


def surface_spec():
    _, proposal = records()
    spec = deepcopy(proposal["catalyst_spec"])
    spec["model"] = {
        "kind": "surface", "miller_indices": [1, 1, 1],
        "supercell": [2, 2, 1], "layers": 4, "vacuum_angstrom": 12.0,
        "periodic_boundary_conditions": [True, True, False],
        "fixed_layers_from_bottom": 2, "center": True,
        "atom_ordering": "ase_default",
    }
    spec["modifications"] = [{
        "operation": "add_adsorbate", "element": "H", "site": "ontop",
        "site_index": 1, "height_angstrom": 1.8,
    }]
    return spec


def test_spec_aware_surface_validation_passes() -> None:
    spec = surface_spec()
    result = dispatch_spec(spec, request_id="validation-test")
    rules = catalyst_validation_rules(spec, result.workspace)
    assert all(rule["passed"] for rule in rules)
    assert {rule["rule"] for rule in rules} >= {
        "periodic_boundary_conditions", "allowed_composition", "slab_vacuum_angstrom",
        "slab_centering", "fixed_layer_atom_count", "adsorbate_1_anchor_height",
        "adsorbate_1_support_separation", "adsorbate_1_site_identity",
    }


def test_spec_aware_validation_detects_wrong_claimed_height() -> None:
    built_spec = surface_spec()
    result = dispatch_spec(built_spec, request_id="validation-test")
    claimed = deepcopy(built_spec)
    claimed["modifications"][0]["height_angstrom"] = 2.5
    failed = {rule["rule"] for rule in catalyst_validation_rules(claimed, result.workspace) if not rule["passed"]}
    assert "adsorbate_1_anchor_height" in failed
