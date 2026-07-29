"""Deterministically map a policy-approved CatalystSpec to registered tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import reduce
from math import gcd, prod
from typing import Any

from ase.build import surface as ase_surface
from ase.formula import Formula

from .mp_resolver import ReferenceResolution
from .schemas import ToolOutcome, ToolSpec
from .tools import create_default_registry
from .validation import atoms_hash
from .workspace import ASEWorkspace, ToolExecutionError


class CatalystDispatchError(ValueError):
    pass


@dataclass(frozen=True)
class DispatchResult:
    workspace: ASEWorkspace
    result_name: str

    @property
    def atoms(self):
        return self.workspace.final_atoms()


def _element_count(formula: str) -> int:
    return len(re.findall(r"[A-Z][a-z]?", formula))


def _formula_key(value: str) -> tuple[tuple[str, int], ...]:
    counts = Formula(value).count()
    divisor = reduce(gcd, counts.values())
    return tuple(sorted((element, count // divisor) for element, count in counts.items()))


def _reference_tools(registry, reference: ReferenceResolution) -> None:
    assert reference.record is not None and reference.structure is not None
    expected_id = reference.record["material_id"]
    parent = reference.structure.copy()
    common = {
        "name": {"type": "string", "minLength": 1, "maxLength": 64},
        "reference_id": {"type": "string", "minLength": 1, "maxLength": 64},
        "repeat": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {"type": "integer", "minimum": 1, "maximum": 12},
        },
    }

    def load_bulk(workspace, args):
        if args["reference_id"] != expected_id:
            raise ValueError("reference id does not match the resolved structure")
        atoms = parent.repeat(tuple(args["repeat"]))
        if len(atoms) > workspace.policy.max_atoms:
            raise ValueError("reference supercell exceeds the atom budget")
        atoms.pbc = (True, True, True)
        atoms.info["ase_agent"] = {"kind": "bulk", "reference_id": expected_id}
        return ToolOutcome(args["name"], atoms)

    registry.register(ToolSpec(
        "load_reference_bulk", "Load one policy-resolved bulk reference.",
        {"type": "object", "properties": common,
         "required": ["name", "reference_id", "repeat"], "additionalProperties": False},
        "mutates_structure", load_bulk,
    ))

    surface_properties = {
        **common,
        "miller": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "integer"}},
        "layers": {"type": "integer", "minimum": 1, "maximum": 40},
        "vacuum": {"type": "number", "minimum": 3.0, "maximum": 100.0},
    }

    def build_surface(workspace, args):
        if args["reference_id"] != expected_id:
            raise ValueError("reference id does not match the resolved structure")
        repeats = tuple(args["repeat"])
        predicted = len(parent) * args["layers"] * prod(repeats)
        if predicted > workspace.policy.max_atoms:
            raise ValueError("reference slab exceeds the atom budget")
        miller = tuple(args["miller"])
        atoms = ase_surface(parent, miller, args["layers"], vacuum=None).repeat(repeats)
        atoms.pbc = (True, True, False)
        atoms.center(vacuum=args["vacuum"], axis=2)
        atoms.info["ase_agent"] = {
            "kind": "surface", "reference_id": expected_id,
            "miller": list(miller), "layers": args["layers"],
        }
        return ToolOutcome(args["name"], atoms)

    registry.register(ToolSpec(
        "build_reference_surface", "Cut a slab from one policy-resolved bulk reference.",
        {"type": "object", "properties": surface_properties,
         "required": ["name", "reference_id", "repeat", "miller", "layers", "vacuum"],
         "additionalProperties": False},
        "mutates_structure", build_surface,
    ))


#: Phases ase.build.bulk understands; anything else names a registered prototype.
ELEMENTAL_PHASES = {"fcc", "bcc", "hcp", "diamond", "sc"}


def dispatch_spec(
    spec: dict[str, Any], *, request_id: str,
    reference: ReferenceResolution | None = None,
) -> DispatchResult:
    """Execute only values represented by the typed CatalystSpec schema."""
    material = spec["material"]
    model = spec["model"]
    kind = model["kind"]
    formula = material["formula"]
    crystal = material.get("crystal_structure")
    lattice = material.get("lattice_parameters_angstrom", {})

    if model.get("center") is not True or model.get("atom_ordering") != "ase_default":
        raise CatalystDispatchError("only centered ASE-default atom ordering is registered")
    expected_pbc = [False, False, False] if kind == "nanoparticle" else [True, True, kind == "bulk"]
    if model["periodic_boundary_conditions"] != expected_pbc:
        raise CatalystDispatchError(f"{kind} requires PBC {expected_pbc}")

    reference_id = material.get("reference_id")
    if reference_id:
        if (
            reference is None or reference.status != "resolved"
            or reference.record is None or reference.structure is None
        ):
            raise CatalystDispatchError("a resolved bulk reference is required")
        if reference.record["material_id"] != reference_id:
            raise CatalystDispatchError("CatalystSpec reference_id does not match ReferenceRecord")
        if reference.record["request_id"] != request_id:
            raise CatalystDispatchError("request_id does not match ReferenceRecord")
        if atoms_hash(reference.structure) != reference.record["structure_sha256"]:
            raise CatalystDispatchError("resolved structure hash does not match ReferenceRecord")
        if _formula_key(reference.record["formula"]) != _formula_key(formula):
            raise CatalystDispatchError("CatalystSpec formula does not match ReferenceRecord")

    registry = create_default_registry()
    if reference_id:
        assert reference is not None
        _reference_tools(registry, reference)
    workspace = ASEWorkspace(registry, session_id=request_id)
    name = "candidate"
    if reference_id and kind == "bulk":
        workspace.execute_or_raise("load_reference_bulk", {
            "name": name, "reference_id": reference_id, "repeat": model["supercell"],
        })
    elif reference_id and kind == "surface":
        workspace.execute_or_raise("build_reference_surface", {
            "name": name, "reference_id": reference_id,
            "miller": model["miller_indices"], "layers": model["layers"],
            "repeat": model["supercell"], "vacuum": model["vacuum_angstrom"],
        })
    elif kind == "nanoparticle":
        args = {
            "name": name, "element": formula, "shape": model["shape"],
            "shells": model["shells"], "vacuum": model["vacuum_angstrom"],
        }
        if "lattice_constant_angstrom" in model:
            args["lattice_constant"] = model["lattice_constant_angstrom"]
        workspace.execute_or_raise("build_nanoparticle", args)
    elif kind == "bulk":
        if _element_count(formula) != 1 or crystal not in {"fcc", "bcc", "hcp", "diamond", "sc"}:
            raise CatalystDispatchError("bulk dispatch currently requires a registered elemental phase")
        args = {
            "name": name, "element": formula, "crystal": crystal, "cubic": True,
            "repeat": model["supercell"],
        }
        for key in ("a", "c"):
            if key in lattice:
                args[key] = lattice[key]
        workspace.execute_or_raise("build_bulk", args)
    elif kind == "surface":
        compound = _element_count(formula) > 1
        if compound and model.get("termination") != "ase_default":
            raise CatalystDispatchError("compound surfaces require the registered ase_default termination")
        if compound:
            # `crystal` only names the family here, and the builder treats it as
            # redundant with the prototype; passing both trips the phase enum.
            element, phase = f"{crystal}-{formula}", None
        elif crystal in ELEMENTAL_PHASES:
            element, phase = formula, crystal
        else:
            # A registered single-element prototype, e.g. graphene or graphite.
            element, phase = crystal, None
        args = {
            "name": name, "element": element,
            "miller": model["miller_indices"], "layers": model["layers"],
            "repeat": model["supercell"], "vacuum": model["vacuum_angstrom"],
        }
        if phase:
            args["crystal"] = phase
        for key in ("a", "c"):
            if key in lattice:
                args[key] = lattice[key]
        workspace.execute_or_raise("build_surface", args)
    else:
        raise CatalystDispatchError(f"unregistered CatalystSpec kind: {kind}")

    fixed = model.get("fixed_layers_from_bottom", 0)
    if fixed:
        workspace.execute_or_raise("freeze_layers", {
            "name": name, "side": "bottom", "layers": fixed, "axes": "xyz",
        })

    for modification in spec["modifications"]:
        operation = modification["operation"]
        if operation == "make_vacancy":
            workspace.execute_or_raise("make_vacancy", {"name": name, "selector": modification["selector"]})
        elif operation == "substitute":
            workspace.execute_or_raise("substitute", {
                "name": name, "selector": modification["selector"], "element": modification["element"],
            })
        elif operation == "add_adsorbate":
            unsupported = set(modification) & {"coverage_monolayer", "orientation"}
            if unsupported:
                raise CatalystDispatchError(f"unregistered adsorbate controls: {sorted(unsupported)}")
            common = {
                "name": name, "site": modification["site"],
                "height": modification["height_angstrom"],
            }
            if "site_index" in modification:
                common["site_index"] = modification["site_index"]
            if "species" in modification:
                common["species"] = modification["species"]
                if "anchor" in modification:
                    common["anchor"] = modification["anchor"]
                workspace.execute_or_raise("add_molecular_adsorbate", common)
            else:
                common["element"] = modification["element"]
                workspace.execute_or_raise("add_atomic_adsorbate", common)
        elif operation == "add_supported_cluster":
            cluster = "supported_cluster"
            cluster_args = {
                "name": cluster, "element": modification["element"],
                "shape": modification["shape"], "shells": modification["shells"],
                "vacuum": modification["vacuum_angstrom"],
            }
            if "lattice_constant_angstrom" in modification:
                cluster_args["lattice_constant"] = modification["lattice_constant_angstrom"]
            workspace.execute_or_raise("build_nanoparticle", cluster_args)
            workspace.execute_or_raise("combine", {
                "name": name, "host": name, "guest": cluster,
                "mode": "stack", "gap": modification["gap_angstrom"],
                "vacuum": modification["vacuum_angstrom"],
            })
        else:  # pragma: no cover - schema and policy gate reject this first
            raise CatalystDispatchError(f"unregistered modification: {operation}")

    workspace.execute_or_raise("finish", {"name": name})
    try:
        workspace.final_atoms()
    except ToolExecutionError as exc:
        raise CatalystDispatchError(str(exc)) from exc
    return DispatchResult(workspace, name)


__all__ = ["CatalystDispatchError", "DispatchResult", "dispatch_spec"]
