"""Deterministic ASE construction tools."""

from __future__ import annotations

import math
from typing import Any

from ase import Atoms
from ase.build import bulk, molecule, nanotube, surface
from ase.cluster import Icosahedron, Octahedron
from ase.constraints import FixAtoms, FixCartesian
from ase.spacegroup import crystal

from ase_auto_build.structure import (
    COMPOUND_FAMILIES,
    make_prototype,
    resolve_prototype,
    split_compound,
)

from .registry import ToolRegistry
from .schemas import ToolOutcome, ToolSpec


NAME = {"type": "string", "minLength": 1, "maxLength": 64}
REPEAT = {
    "type": "array",
    "items": {"type": "integer", "minimum": 1, "maximum": 12},
    "minItems": 3,
    "maxItems": 3,
}


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _metadata(atoms: Atoms, **values: Any) -> Atoms:
    atoms.info["ase_agent"] = {**atoms.info.get("ase_agent", {}), **values}
    return atoms


def _ensure_atom_budget(workspace, predicted_atoms: int) -> None:
    if predicted_atoms > workspace.policy.max_atoms:
        raise ValueError(
            f"operation would create {predicted_atoms} atoms; limit is {workspace.policy.max_atoms}"
        )


def _from_structure_dict(structure: dict[str, Any]) -> Atoms:
    symbols: list[str] = []
    for symbol, count in zip(structure["elements"], structure["counts"]):
        symbols.extend([symbol] * int(count))
    atoms = Atoms(symbols=symbols, cell=structure["lattice"], pbc=True)
    if structure.get("cartesian"):
        atoms.set_positions(structure["coords"])
    else:
        atoms.set_scaled_positions(structure["coords"])
    if structure.get("selective"):
        full_fixed: list[int] = []
        for index, flags in enumerate(structure.get("flags", [])):
            fixed = tuple(str(value).upper() == "F" for value in flags)
            if fixed == (True, True, True):
                full_fixed.append(index)
            elif any(fixed):
                atoms.constraints.append(FixCartesian(index, mask=fixed))
        if full_fixed:
            atoms.constraints.append(FixAtoms(indices=full_fixed))
    return atoms


def _build_bulk(workspace, args: dict[str, Any]) -> ToolOutcome:
    kwargs: dict[str, Any] = {"cubic": bool(args.get("cubic", False))}
    if "crystal" in args:
        kwargs["crystalstructure"] = args["crystal"]
    if "a" in args:
        kwargs["a"] = float(args["a"])
    if "c" in args:
        kwargs["c"] = float(args["c"])
    atoms = bulk(args["element"], **kwargs)
    if "repeat" in args:
        _ensure_atom_budget(workspace, len(atoms) * math.prod(args["repeat"]))
        atoms = atoms.repeat(tuple(args["repeat"]))
    return ToolOutcome(args["name"], _metadata(atoms, kind="bulk"))


def _surface_substrate(args: dict[str, Any]) -> Atoms:
    """The bulk a slab is cut from, named by the single `element` argument.

    `element` carries either a symbol ('Cu') or a compound prototype
    ('rocksalt-MgO'); `ase.build.surface` cuts from any Atoms, so a compound slab
    only needs the right substrate. Keeping it to *one* required argument is
    deliberate: an earlier version made `element` optional and added a parallel
    `prototype` argument, and r5 promptly started omitting `element` on ordinary
    elemental slabs -- a regression on the most common request in the registry.
    `prototype` survives as an accepted alias because the model also uses it.
    """
    substrate = args.get("prototype") or args.get("element")
    if not substrate:
        raise ValueError(
            "missing required fields ['element'] -- name the substrate: "
            "element='Cu' for an elemental slab, or element='rocksalt-MgO' "
            "(equivalently prototype=...) for a compound slab"
        )
    if args.get("prototype") and args.get("element"):
        # Both named is ambiguous about which one the facet was cut from, unless
        # they agree -- the model sometimes echoes the compound into both.
        if split_compound(args["element"]) != split_compound(args["prototype"]):
            raise ValueError(
                "give either 'element' or 'prototype', not both -- got "
                f"element={args['element']!r} and prototype={args['prototype']!r}"
            )
    crystal = args.get("crystal")
    if crystal in COMPOUND_FAMILIES and not split_compound(substrate):
        # A family in `crystal` with only a composition in `element`: combine them
        # rather than reject a call whose intent is unambiguous.
        substrate = f"{crystal}-{substrate}"
    compound = split_compound(substrate)  # raises with guidance on a near miss
    if compound is not None:
        # A `crystal` echoing the family is redundant, not a conflict; ignore it.
        return _from_structure_dict(make_prototype(
            f"{compound[0]}-{compound[1]}",
            a=float(args["a"]) if "a" in args else None,
            c=float(args["c"]) if "c" in args else None,
        ))
    if crystal in COMPOUND_FAMILIES:
        raise ValueError(
            f"crystal={crystal!r} is a compound family but element={substrate!r} "
            f"is not a composition; use element='{crystal}-MgO' style"
        )
    kwargs: dict[str, Any] = {}
    if crystal:
        kwargs["crystalstructure"] = crystal
    if "a" in args:
        kwargs["a"] = float(args["a"])
    return bulk(substrate, **kwargs)


def _build_surface(workspace, args: dict[str, Any]) -> ToolOutcome:
    base = _surface_substrate(args)
    miller = tuple(int(value) for value in args["miller"])
    layers = int(args.get("layers", 4))
    repeats = args.get("repeat", [1, 1, 1])
    _ensure_atom_budget(workspace, len(base) * layers * math.prod(repeats))
    atoms = surface(base, miller, layers, vacuum=None)
    if "repeat" in args:
        atoms = atoms.repeat(tuple(args["repeat"]))
    atoms.pbc = (True, True, False)
    atoms.center(vacuum=float(args.get("vacuum", 12.0)), axis=2)
    return ToolOutcome(
        args["name"],
        _metadata(atoms, kind="surface", miller=list(miller), layers=layers),
    )


def _build_molecule(workspace, args: dict[str, Any]) -> ToolOutcome:
    atoms = molecule(args["species"])
    box = float(args.get("box", 12.0))
    atoms.set_cell([box, box, box])
    atoms.pbc = (False, False, False)
    atoms.center()
    atoms.info["charge"] = int(args.get("charge", 0))
    if "multiplicity" in args:
        atoms.info["multiplicity"] = int(args["multiplicity"])
    return ToolOutcome(args["name"], _metadata(atoms, kind="molecule", species=args["species"]))


def _build_crystal(workspace, args: dict[str, Any]) -> ToolOutcome:
    cell = [
        float(args["a"]),
        float(args.get("b", args["a"])),
        float(args.get("c", args["a"])),
        float(args.get("alpha", 90.0)),
        float(args.get("beta", 90.0)),
        float(args.get("gamma", 90.0)),
    ]
    atoms = crystal(
        symbols=args["symbols"],
        basis=args["basis"],
        spacegroup=int(args["spacegroup"]),
        cellpar=cell,
    )
    return ToolOutcome(args["name"], _metadata(atoms, kind="crystal", spacegroup=args["spacegroup"]))


def _build_nanotube(workspace, args: dict[str, Any]) -> ToolOutcome:
    n = int(args["n"])
    m = int(args["m"])
    length = int(args.get("length", 1))
    unit_atoms = 4 * (n * n + n * m + m * m) // math.gcd(2 * n + m, 2 * m + n)
    _ensure_atom_budget(workspace, unit_atoms * length)
    kwargs: dict[str, Any] = {
        "length": length,
        "symbol": args.get("element", "C"),
    }
    if "bond" in args:
        kwargs["bond"] = float(args["bond"])
    atoms = nanotube(n, m, **kwargs)
    atoms.center(vacuum=float(args.get("vacuum", 10.0)), axis=(0, 1))
    atoms.pbc = (False, False, True)
    return ToolOutcome(args["name"], _metadata(atoms, kind="nanotube"))


def _build_nanoparticle(workspace, args: dict[str, Any]) -> ToolOutcome:
    """Build one of ASE's deterministic closed-shell metal clusters."""
    element = args["element"]
    shape = args["shape"]
    shells = int(args["shells"])
    lattice_constant = float(args["lattice_constant"]) if "lattice_constant" in args else None
    if shape == "icosahedron":
        atoms = Icosahedron(element, shells, latticeconstant=lattice_constant)
    else:
        length = 2 * shells + 1
        cutoff = shells if shape == "cuboctahedron" else 0
        atoms = Octahedron(
            element, length, cutoff=cutoff, latticeconstant=lattice_constant
        )
    _ensure_atom_budget(workspace, len(atoms))
    atoms.center(vacuum=float(args.get("vacuum", 8.0)))
    atoms.pbc = (False, False, False)
    return ToolOutcome(
        args["name"],
        _metadata(atoms, kind="nanoparticle", shape=shape, shells=shells),
    )


def _prototype_name(args: dict[str, Any]) -> str:
    """Combine a family and a separate composition, e.g. 'rocksalt' + 'NiO'.

    A compound prototype is named '<family>-<formula>', but the model tends to put
    the family in `prototype` and drop the composition. `formula` gives it a place
    to put it; a `prototype` that already resolves on its own is left alone.
    """
    requested = str(args["prototype"])
    formula = args.get("formula")
    if not formula:
        return requested
    try:
        resolve_prototype(requested)
    except ValueError:
        return f"{requested}-{formula}"
    return requested


def _build_prototype(workspace, args: dict[str, Any]) -> ToolOutcome:
    structure = make_prototype(
        _prototype_name(args),
        a=float(args["a"]) if "a" in args else None,
        c=float(args["c"]) if "c" in args else None,
        vacuum=float(args["vacuum"]) if "vacuum" in args else None,
    )
    return ToolOutcome(args["name"], _metadata(_from_structure_dict(structure), kind="prototype"))


def _clone(workspace, args: dict[str, Any]) -> ToolOutcome:
    return ToolOutcome(args["name"], workspace.require_atoms(args["source"]), (args["source"],))


def register(registry: ToolRegistry) -> None:
    registry.register(ToolSpec("build_bulk", "Build an elemental bulk crystal.", _object({
        "name": NAME, "element": {"type": "string", "minLength": 1, "maxLength": 3},
        "crystal": {"type": "string", "enum": ["fcc", "bcc", "hcp", "diamond", "sc"]},
        "a": {"type": "number", "exclusiveMinimum": 0.0},
        "c": {"type": "number", "exclusiveMinimum": 0.0},
        "cubic": {"type": "boolean"}, "repeat": REPEAT,
    }, ["name", "element", "crystal", "cubic"]), "mutates_structure", _build_bulk))
    # The tool description stays terse on purpose. A long one measurably perturbed
    # elemental slabs -- r5 began dropping `element` and the lateral `repeat` --
    # so the compound guidance lives on the arguments it concerns instead.
    registry.register(ToolSpec("build_surface", "Build an elemental or compound surface slab.", _object({
        "name": NAME,
        "element": {
            "type": "string", "minLength": 1, "maxLength": 24,
            "description": "Required. A chemical symbol for an elemental slab "
                           "('Cu'), or a compound prototype for a compound slab "
                           "('rocksalt-MgO', 'wurtzite-ZnO', 'fluorite-CeO2').",
        },
        "prototype": {"type": "string", "minLength": 1, "maxLength": 64,
                      "description": "Alias for a compound 'element'."},
        "miller": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
        "layers": {"type": "integer", "minimum": 1, "maximum": 40},
        "vacuum": {"type": "number", "minimum": 3.0, "maximum": 100.0},
        "crystal": {
            "type": "string",
            "enum": ["fcc", "bcc", "hcp", "diamond", "sc", *sorted(COMPOUND_FAMILIES)],
            "description": "Phase of an elemental slab. Redundant when 'element' "
                           "already names a compound prototype, and then ignored.",
        },
        "a": {"type": "number", "exclusiveMinimum": 0.0},
        "c": {"type": "number", "exclusiveMinimum": 0.0}, "repeat": REPEAT,
    }, ["name", "miller"]), "mutates_structure", _build_surface))
    registry.register(ToolSpec("build_molecule", "Build an ASE molecule in an isolated box.", _object({
        "name": NAME, "species": {"type": "string", "minLength": 1, "maxLength": 32},
        "box": {"type": "number", "minimum": 3.0, "maximum": 100.0},
        "charge": {"type": "integer", "minimum": -10, "maximum": 10},
        "multiplicity": {"type": "integer", "minimum": 1, "maximum": 20},
    }, ["name", "species"]), "mutates_structure", _build_molecule))
    registry.register(ToolSpec("build_crystal", "Build a space-group crystal from fractional basis sites.", _object({
        "name": NAME,
        "symbols": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 64},
        "basis": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "minItems": 1, "maxItems": 64},
        "spacegroup": {"type": "integer", "minimum": 1, "maximum": 230},
        "a": {"type": "number", "exclusiveMinimum": 0.0},
        "b": {"type": "number", "exclusiveMinimum": 0.0}, "c": {"type": "number", "exclusiveMinimum": 0.0},
        "alpha": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 180.0},
        "beta": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 180.0},
        "gamma": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 180.0},
    }, ["name", "symbols", "basis", "spacegroup", "a"]), "mutates_structure", _build_crystal))
    registry.register(ToolSpec("build_nanotube", "Build a periodic single-wall nanotube.", _object({
        "name": NAME, "element": {"type": "string", "minLength": 1, "maxLength": 3},
        "n": {"type": "integer", "minimum": 1, "maximum": 100},
        "m": {"type": "integer", "minimum": 0, "maximum": 100},
        "length": {"type": "integer", "minimum": 1, "maximum": 50},
        "bond": {"type": "number", "exclusiveMinimum": 0.0},
        "vacuum": {"type": "number", "minimum": 3.0, "maximum": 100.0},
    }, ["name", "n", "m"]), "mutates_structure", _build_nanotube))
    registry.register(ToolSpec(
        "build_nanoparticle",
        "Build a closed-shell icosahedral, octahedral, or cuboctahedral nanoparticle.",
        _object({
            "name": NAME,
            "element": {"type": "string", "minLength": 1, "maxLength": 3},
            "shape": {
                "type": "string",
                "enum": ["icosahedron", "octahedron", "cuboctahedron"],
            },
            "shells": {"type": "integer", "minimum": 1, "maximum": 6},
            "lattice_constant": {"type": "number", "exclusiveMinimum": 0.0},
            "vacuum": {"type": "number", "minimum": 3.0, "maximum": 100.0},
        }, ["name", "element", "shape", "shells"]),
        "mutates_structure",
        _build_nanoparticle,
    ))
    registry.register(ToolSpec("build_prototype", (
        "Build a named prototype. Either a fixed name (graphene, graphite, hBN, "
        "rutile-TiO2, anatase-TiO2) or a binary compound as '<family>-<formula>' "
        "in one string: rocksalt-MgO, zincblende-GaAs, wurtzite-ZnO, "
        "fluorite-CeO2. Lattice constants are tabulated; omit 'a' and 'c'."
    ), _object({
        "name": NAME, "prototype": {"type": "string", "minLength": 1, "maxLength": 64},
        "formula": {"type": "string", "minLength": 1, "maxLength": 16},
        "a": {"type": "number", "exclusiveMinimum": 0.0}, "c": {"type": "number", "exclusiveMinimum": 0.0},
        "vacuum": {"type": "number", "minimum": 3.0, "maximum": 100.0},
    }, ["name", "prototype"]), "mutates_structure", _build_prototype))
    registry.register(ToolSpec("clone", "Clone a named structure into a new branch.", _object({
        "source": NAME, "name": NAME,
    }, ["source", "name"]), "mutates_structure", _clone))
