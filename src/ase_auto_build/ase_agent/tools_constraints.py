"""Constraint operations using typed selectors."""

from __future__ import annotations

from ase.constraints import FixAtoms, FixCartesian

from .registry import ToolRegistry
from .schemas import ToolOutcome, ToolSpec
from .selectors import SELECTOR_SCHEMA, select_atoms
from .tools_build import NAME, _object


def _freeze(workspace, args):
    name = args["name"]
    atoms = workspace.require_atoms(name)
    indices = select_atoms(atoms, args["selector"])
    axes = args.get("axes", "xyz").lower()
    if set(axes) == {"x", "y", "z"}:
        atoms.constraints = list(atoms.constraints) + [FixAtoms(indices=indices)]
    else:
        mask = tuple(axis in axes for axis in "xyz")
        atoms.constraints = list(atoms.constraints) + [
            FixCartesian(index, mask=mask) for index in indices
        ]
    return ToolOutcome(name, atoms, (name,), {"selected_count": len(indices), "selected_indices": [index + 1 for index in indices[:32]]})


def _clear(workspace, args):
    name = args["name"]
    atoms = workspace.require_atoms(name)
    atoms.set_constraint()
    return ToolOutcome(name, atoms, (name,))


def _list(workspace, args):
    atoms = workspace.require_atoms(args["name"])
    values = []
    for constraint in atoms.constraints:
        try:
            indices = [int(index) + 1 for index in constraint.get_indices()]
        except Exception:
            indices = []
        values.append({"type": type(constraint).__name__, "indices": indices[:64], "count": len(indices)})
    return ToolOutcome(data={"name": args["name"], "constraints": values})


def register(registry: ToolRegistry) -> None:
    registry.register(ToolSpec("freeze_atoms", "Freeze selected atom axes.", _object({
        "name": NAME, "selector": SELECTOR_SCHEMA,
        "axes": {"type": "string", "enum": ["x", "y", "z", "xy", "xz", "yz", "xyz"]},
    }, ["name", "selector"]), "mutates_structure", _freeze))
    registry.register(ToolSpec("freeze_layers", "Freeze a number of top or bottom geometric layers.", _object({
        "name": NAME,
        "side": {"type": "string", "enum": ["bottom", "top"]},
        "layers": {"type": "integer", "minimum": 1, "maximum": 40},
        "axes": {"type": "string", "enum": ["x", "y", "z", "xy", "xz", "yz", "xyz"]},
        "tolerance": {"type": "number", "exclusiveMinimum": 0.0},
    }, ["name", "side", "layers"]), "mutates_structure", lambda ws, args: _freeze(ws, {
        "name": args["name"], "axes": args.get("axes", "xyz"),
        "selector": {"layer": {"side": args["side"], "count": args["layers"], "tolerance": args.get("tolerance", 0.35)}},
    })))
    registry.register(ToolSpec("clear_constraints", "Remove all atom constraints.", _object({"name": NAME}, ["name"]), "mutates_structure", _clear))
    registry.register(ToolSpec("list_constraints", "List constraint types and bounded indices.", _object({"name": NAME}, ["name"]), "read_only", _list))

