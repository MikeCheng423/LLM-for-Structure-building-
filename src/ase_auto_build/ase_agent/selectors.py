"""Typed, non-executable atom and layer selectors."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms


SELECTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "indices": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "minItems": 1,
            "uniqueItems": True,
            "description": "One-based atom indices.",
        },
        "element": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 3},
            "minItems": 1,
            "uniqueItems": True,
        },
        "region": {
            "type": "object",
            "properties": {
                "axis": {"type": "string", "enum": ["x", "y", "z"]},
                "op": {"type": "string", "enum": ["<", "<=", ">", ">=", "=="]},
                "value": {"type": "number"},
                "fractional": {"type": "boolean"},
                "tolerance": {"type": "number", "minimum": 0.0},
            },
            "required": ["axis", "op", "value"],
            "additionalProperties": False,
        },
        "layer": {
            "type": "object",
            "properties": {
                "side": {"type": "string", "enum": ["bottom", "top"]},
                "count": {"type": "integer", "minimum": 1},
                "tolerance": {"type": "number", "exclusiveMinimum": 0.0},
            },
            "required": ["side", "count"],
            "additionalProperties": False,
        },
        "ordinal": {
            "type": "integer",
            "minimum": 1,
            "description": "One-based position after all other selector filters.",
        },
    },
    "additionalProperties": False,
}


def geometric_layers(atoms: Atoms, *, tolerance: float = 0.35) -> list[list[int]]:
    if tolerance <= 0:
        raise ValueError("layer tolerance must be positive")
    if not len(atoms):
        return []
    order = sorted(range(len(atoms)), key=lambda index: float(atoms.positions[index, 2]))
    layers: list[list[int]] = []
    centers: list[float] = []
    for index in order:
        z = float(atoms.positions[index, 2])
        if not layers or abs(z - centers[-1]) > tolerance:
            layers.append([index])
            centers.append(z)
        else:
            layers[-1].append(index)
            centers[-1] = float(np.mean([atoms.positions[item, 2] for item in layers[-1]]))
    return layers


def select_atoms(atoms: Atoms, selector: dict[str, Any]) -> list[int]:
    if not selector:
        raise ValueError("selector must contain at least one condition")
    selected = set(range(len(atoms)))

    if "indices" in selector:
        indices = {int(index) - 1 for index in selector["indices"]}
        if any(index < 0 or index >= len(atoms) for index in indices):
            raise IndexError("selector atom index is out of range")
        selected &= indices

    if "element" in selector:
        allowed = set(selector["element"])
        selected &= {
            index for index, symbol in enumerate(atoms.get_chemical_symbols()) if symbol in allowed
        }

    if "region" in selector:
        region = selector["region"]
        axis = {"x": 0, "y": 1, "z": 2}[region["axis"]]
        coordinates = atoms.get_scaled_positions(wrap=False) if region.get("fractional") else atoms.positions
        value = float(region["value"])
        tolerance = float(region.get("tolerance", 1e-8))
        op = region["op"]

        def matches(number: float) -> bool:
            if op == "<":
                return number < value
            if op == "<=":
                return number <= value
            if op == ">":
                return number > value
            if op == ">=":
                return number >= value
            return abs(number - value) <= tolerance

        selected &= {index for index in range(len(atoms)) if matches(float(coordinates[index, axis]))}

    if "layer" in selector:
        layer = selector["layer"]
        layers = geometric_layers(atoms, tolerance=float(layer.get("tolerance", 0.35)))
        count = int(layer["count"])
        if count > len(layers):
            raise ValueError(f"requested {count} layers but structure has {len(layers)}")
        chosen = layers[:count] if layer["side"] == "bottom" else layers[-count:]
        selected &= {index for indices in chosen for index in indices}

    result = sorted(selected)
    if not result:
        raise ValueError("selector matched no atoms")
    if "ordinal" in selector:
        ordinal = int(selector["ordinal"])
        if ordinal > len(result):
            raise IndexError(
                f"selector ordinal {ordinal} exceeds {len(result)} matched atoms"
            )
        result = [result[ordinal - 1]]
    return result
