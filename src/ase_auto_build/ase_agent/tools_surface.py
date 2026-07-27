"""Layer, site, and adsorbate operations."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, molecule

from .registry import ToolRegistry
from .schemas import ToolOutcome, ToolSpec
from .selectors import geometric_layers
from .tools_build import NAME, _object


def _top_layer(atoms: Atoms, tolerance: float = 0.35) -> list[int]:
    layers = geometric_layers(atoms, tolerance=tolerance)
    if not layers:
        raise ValueError("surface has no atoms")
    return layers[-1]


def adsorption_sites(atoms: Atoms, tolerance: float = 0.35) -> list[dict[str, Any]]:
    top = _top_layer(atoms, tolerance)
    sites: list[dict[str, Any]] = []
    for index in top:
        sites.append({"kind": "ontop", "indices": [index + 1], "xy": atoms.positions[index, :2].tolist()})
    pairs = []
    for i, j in itertools.combinations(top, 2):
        distance = float(atoms.get_distance(i, j, mic=True))
        pairs.append((distance, i, j))
    if pairs:
        cutoff = min(distance for distance, _, _ in pairs) * 1.15
        for distance, i, j in sorted(pairs):
            if distance <= cutoff:
                xy = ((atoms.positions[i, :2] + atoms.positions[j, :2]) / 2).tolist()
                sites.append({"kind": "bridge", "indices": [i + 1, j + 1], "xy": xy})
    if len(top) >= 3:
        nearest = sorted(top, key=lambda index: float(np.linalg.norm(atoms.positions[index, :2] - atoms.positions[top[0], :2])))[:3]
        xy = np.mean(atoms.positions[nearest, :2], axis=0).tolist()
        sites.append({"kind": "hollow", "indices": [index + 1 for index in nearest], "xy": xy})
    for number, site in enumerate(sites, start=1):
        site["site_id"] = f"{site['kind']}-{number}"
        site["xy"] = [round(float(value), 6) for value in site["xy"]]
    return sites


def _choose_site(atoms: Atoms, args: dict[str, Any]) -> tuple[float, float]:
    if "xy" in args:
        return float(args["xy"][0]), float(args["xy"][1])
    requested = args.get("site", "ontop")
    candidates = [site for site in adsorption_sites(atoms) if site["kind"] == requested]
    if not candidates:
        raise ValueError(f"no {requested} adsorption site found")
    index = int(args.get("site_index", 1)) - 1
    if index < 0 or index >= len(candidates):
        raise IndexError(f"site_index {index + 1} out of range for {len(candidates)} sites")
    return tuple(candidates[index]["xy"])


def _add(workspace, args, *, molecular: bool):
    name = args["name"]
    atoms = workspace.require_atoms(name)
    position = _choose_site(atoms, args)
    if molecular:
        adsorbate = molecule(args["species"])
        add_adsorbate(atoms, adsorbate, float(args.get("height", 1.8)), position=position, mol_index=int(args.get("anchor", 1)) - 1)
    else:
        add_adsorbate(atoms, args["element"], float(args.get("height", 1.8)), position=position)
    return ToolOutcome(name, atoms, (name,), {"adsorption_site": args.get("site", "custom" if "xy" in args else "ontop")})


def _list_layers(workspace, args):
    atoms = workspace.require_atoms(args["name"])
    layers = geometric_layers(atoms, tolerance=float(args.get("tolerance", 0.35)))
    return ToolOutcome(data={
        "name": args["name"],
        "layer_count": len(layers),
        "layers": [{"number": number, "count": len(indices), "indices": [index + 1 for index in indices[:64]]} for number, indices in enumerate(layers, start=1)],
    })


def _list_sites(workspace, args):
    sites = adsorption_sites(workspace.require_atoms(args["name"]), float(args.get("tolerance", 0.35)))
    return ToolOutcome(data={"name": args["name"], "site_count": len(sites), "sites": sites[:128]})


def _site_fields() -> dict[str, Any]:
    return {
        "site": {"type": "string", "enum": ["ontop", "bridge", "hollow"]},
        "site_index": {"type": "integer", "minimum": 1, "maximum": 1000},
        "xy": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "height": {"type": "number", "minimum": 0.5, "maximum": 20.0},
    }


# ``_choose_site`` short-circuits on ``xy`` before it consults ``site``/``site_index``,
# so passing both silently discards the named site and places the adsorbate somewhere
# else entirely. Reject that combination instead of building the wrong structure.
_SITE_SELECTOR_EXCLUSIONS = [["site", "xy"], ["site_index", "xy"]]

_SITE_SELECTION_DOC = (
    " Choose the site either by name (site, optionally site_index) or by explicit xy"
    " coordinates, never both."
)


def register(registry: ToolRegistry) -> None:
    registry.register(ToolSpec("list_layers", "List geometric z layers.", _object({
        "name": NAME, "tolerance": {"type": "number", "exclusiveMinimum": 0.0},
    }, ["name"]), "read_only", _list_layers))
    registry.register(ToolSpec("list_adsorption_sites", "List deterministic ontop, bridge, and hollow sites.", _object({
        "name": NAME, "tolerance": {"type": "number", "exclusiveMinimum": 0.0},
    }, ["name"]), "read_only", _list_sites))
    registry.register(ToolSpec("add_atomic_adsorbate", "Place one atomic adsorbate at a bounded surface site." + _SITE_SELECTION_DOC, _object({
        "name": NAME, "element": {"type": "string", "minLength": 1, "maxLength": 3}, **_site_fields(),
    }, ["name", "element"], _SITE_SELECTOR_EXCLUSIONS), "mutates_structure", lambda ws, args: _add(ws, args, molecular=False)))
    registry.register(ToolSpec("add_molecular_adsorbate", "Place an ASE molecule using a chosen anchor atom." + _SITE_SELECTION_DOC, _object({
        "name": NAME, "species": {"type": "string", "minLength": 1, "maxLength": 32},
        "anchor": {"type": "integer", "minimum": 1, "maximum": 100}, **_site_fields(),
    }, ["name", "species"], _SITE_SELECTOR_EXCLUSIONS), "mutates_structure", lambda ws, args: _add(ws, args, molecular=True)))

