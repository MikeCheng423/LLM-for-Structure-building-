"""Compact model-facing structure summaries; never expose full coordinates."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from ase import Atoms


def structure_observation(
    name: str,
    atoms: Atoms,
    *,
    revision_id: str | None = None,
    warnings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    counts = Counter(atoms.get_chemical_symbols())
    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    constrained: set[int] = set()
    for constraint in atoms.constraints:
        try:
            constrained.update(int(index) + 1 for index in constraint.get_indices())
        except Exception:
            continue
    minimum_distance = None
    if len(atoms) > 1:
        distances = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))
        np.fill_diagonal(distances, np.inf)
        minimum_distance = round(float(np.min(distances)), 6)
    result: dict[str, Any] = {
        "ok": True,
        "name": name,
        "formula": atoms.get_chemical_formula(mode="hill"),
        "natoms": len(atoms),
        "elements": dict(sorted(counts.items())),
        "cell_lengths": [round(float(value), 6) for value in lengths],
        "cell_angles": [round(float(value), 6) for value in angles],
        "pbc": [bool(value) for value in atoms.pbc],
        "constraint_count": len(atoms.constraints),
        "constrained_atom_count": len(constrained),
        "minimum_distance": minimum_distance,
    }
    if revision_id is not None:
        result["revision_id"] = revision_id
    if warnings:
        result["warnings"] = warnings
    return result

