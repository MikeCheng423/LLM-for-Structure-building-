"""Safe structure mutation tools."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from ase import Atom

from .registry import ToolRegistry
from .schemas import ToolOutcome, ToolSpec
from .selectors import SELECTOR_SCHEMA, select_atoms
from .tools_build import NAME, REPEAT, _ensure_atom_budget, _object


VECTOR = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}


def _same(workspace, args, operation) -> ToolOutcome:
    name = args["name"]
    atoms = workspace.require_atoms(name)
    operation(atoms)
    return ToolOutcome(name, atoms, (name,))


def _repeat(workspace, args):
    name = args["name"]
    source = workspace.require_atoms(name)
    _ensure_atom_budget(workspace, len(source) * math.prod(args["repeat"]))
    atoms = source.repeat(tuple(args["repeat"]))
    return ToolOutcome(name, atoms, (name,))


def _scale_cell(workspace, args):
    factors = np.asarray(args["factors"], dtype=float)
    return _same(workspace, args, lambda atoms: atoms.set_cell(atoms.cell.array * factors[:, None], scale_atoms=True))


def _set_cell(workspace, args):
    return _same(workspace, args, lambda atoms: atoms.set_cell(args["cell"], scale_atoms=bool(args.get("scale_atoms", True))))


def _center(workspace, args):
    axes = tuple({"x": 0, "y": 1, "z": 2}[axis] for axis in args.get("axes", ["x", "y", "z"]))
    return _same(workspace, args, lambda atoms: atoms.center(vacuum=float(args.get("vacuum", 0.0)), axis=axes))


def _wrap(workspace, args):
    return _same(workspace, args, lambda atoms: atoms.wrap())


def _vacancy(workspace, args):
    def operation(atoms):
        indices = select_atoms(atoms, args["selector"])
        del atoms[indices]
    return _same(workspace, args, operation)


def _substitute(workspace, args):
    def operation(atoms):
        for index in select_atoms(atoms, args["selector"]):
            atoms[index].symbol = args["element"]
    return _same(workspace, args, operation)


def _add_atom(workspace, args):
    def operation(atoms):
        position = np.asarray(args["position"], dtype=float)
        if args.get("fractional"):
            position = position @ atoms.cell.array
        atoms.append(Atom(args["element"], position=position))
    return _same(workspace, args, operation)


def _delete_atoms(workspace, args):
    return _vacancy(workspace, args)


def _move_atoms(workspace, args):
    def operation(atoms):
        indices = select_atoms(atoms, args["selector"])
        vector = np.asarray(args["vector"], dtype=float)
        if args.get("fractional"):
            vector = vector @ atoms.cell.array
        atoms.positions[indices] += vector
    return _same(workspace, args, operation)


def _translate(workspace, args):
    return _same(workspace, args, lambda atoms: atoms.translate(args["vector"]))


def _rotate(workspace, args):
    def operation(atoms):
        indices = select_atoms(atoms, args["selector"])
        subset = atoms[indices]
        center: Any = args.get("center", "COM")
        subset.rotate(float(args["angle"]), args["axis"], center=center, rotate_cell=False)
        atoms.positions[indices] = subset.positions
    return _same(workspace, args, operation)


def _combine(workspace, args):
    host = workspace.require_atoms(args["host"])
    guest = workspace.require_atoms(args["guest"])
    _ensure_atom_budget(workspace, len(host) + len(guest))
    if args.get("mode", "stack") == "stack":
        host_center = host.cell.array[0] * 0.5 + host.cell.array[1] * 0.5
        guest_xy = np.mean(guest.positions[:, :2], axis=0)
        target_xy = host_center[:2]
        guest.positions[:, :2] += target_xy - guest_xy
        guest.positions[:, 2] += float(np.max(host.positions[:, 2]) + args.get("gap", 2.0) - np.min(guest.positions[:, 2]))
        new_c = max(float(host.cell.lengths()[2]), float(np.max(guest.positions[:, 2]) + args.get("vacuum", 10.0)))
        cell = host.cell.array.copy()
        current_c = float(np.linalg.norm(cell[2]))
        if current_c <= 1e-12:
            raise ValueError("host c vector is singular")
        cell[2] *= new_c / current_c
        host.set_cell(cell, scale_atoms=False)
    host.extend(guest)
    return ToolOutcome(args["name"], host, (args["host"], args["guest"]))


def register(registry: ToolRegistry) -> None:
    specs = [
        ToolSpec("repeat", "Repeat a structure along all three lattice axes.", _object({"name": NAME, "repeat": REPEAT}, ["name", "repeat"]), "mutates_structure", _repeat),
        ToolSpec("scale_cell", "Scale lattice vectors and atom positions.", _object({"name": NAME, "factors": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0.0}, "minItems": 3, "maxItems": 3}}, ["name", "factors"]), "mutates_structure", _scale_cell),
        ToolSpec("set_cell", "Set a complete 3x3 cell.", _object({"name": NAME, "cell": {"type": "array", "items": VECTOR, "minItems": 3, "maxItems": 3}, "scale_atoms": {"type": "boolean"}}, ["name", "cell"]), "mutates_structure", _set_cell),
        ToolSpec("center", "Center atoms with optional vacuum.", _object({"name": NAME, "vacuum": {"type": "number", "minimum": 0.0, "maximum": 100.0}, "axes": {"type": "array", "items": {"type": "string", "enum": ["x", "y", "z"]}, "minItems": 1, "maxItems": 3, "uniqueItems": True}}, ["name"]), "mutates_structure", _center),
        ToolSpec("wrap", "Wrap atoms into the periodic cell.", _object({"name": NAME}, ["name"]), "mutates_structure", _wrap),
        ToolSpec("make_vacancy", "Delete atoms selected by a typed selector.", _object({"name": NAME, "selector": SELECTOR_SCHEMA}, ["name", "selector"]), "mutates_structure", _vacancy),
        ToolSpec("substitute", "Replace selected atoms with another element.", _object({"name": NAME, "selector": SELECTOR_SCHEMA, "element": {"type": "string", "minLength": 1, "maxLength": 3}}, ["name", "selector", "element"]), "mutates_structure", _substitute),
        ToolSpec("add_atom", "Add one atom at a Cartesian or fractional position.", _object({"name": NAME, "element": {"type": "string", "minLength": 1, "maxLength": 3}, "position": VECTOR, "fractional": {"type": "boolean"}}, ["name", "element", "position"]), "mutates_structure", _add_atom),
        ToolSpec("delete_atoms", "Delete atoms selected by a typed selector.", _object({"name": NAME, "selector": SELECTOR_SCHEMA}, ["name", "selector"]), "mutates_structure", _delete_atoms),
        ToolSpec("move_atoms", "Translate selected atoms by a vector.", _object({"name": NAME, "selector": SELECTOR_SCHEMA, "vector": VECTOR, "fractional": {"type": "boolean"}}, ["name", "selector", "vector"]), "mutates_structure", _move_atoms),
        ToolSpec("translate", "Translate an entire structure.", _object({"name": NAME, "vector": VECTOR}, ["name", "vector"]), "mutates_structure", _translate),
        ToolSpec("rotate", "Rotate selected atoms without rotating the cell.", _object({"name": NAME, "selector": SELECTOR_SCHEMA, "axis": VECTOR, "angle": {"type": "number", "minimum": -360.0, "maximum": 360.0}, "center": VECTOR}, ["name", "selector", "axis", "angle"]), "mutates_structure", _rotate),
        ToolSpec("combine", "Combine a host and guest, optionally stacking the guest above the host.", _object({"name": NAME, "host": NAME, "guest": NAME, "mode": {"type": "string", "enum": ["stack", "insert"]}, "gap": {"type": "number", "minimum": 0.5, "maximum": 20.0}, "vacuum": {"type": "number", "minimum": 3.0, "maximum": 100.0}}, ["name", "host", "guest"]), "mutates_structure", _combine),
    ]
    for spec in specs:
        registry.register(spec)
