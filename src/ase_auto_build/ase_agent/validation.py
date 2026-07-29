"""Deterministic structural validation and stable content hashing."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

from .policy import AgentPolicy
from .schemas import ValidationIssue, ValidationReport

HASH_DECIMALS = 8


def _rounded_list(value: Any) -> list[Any]:
    array = np.round(np.asarray(value, dtype=float), HASH_DECIMALS)
    array[np.abs(array) < 0.5 * 10 ** (-HASH_DECIMALS)] = 0.0
    return array.tolist()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def atoms_payload(atoms: Atoms) -> dict[str, Any]:
    constraints = []
    for constraint in atoms.constraints:
        try:
            constraints.append(_jsonable(constraint.todict()))
        except Exception:
            constraints.append({"name": type(constraint).__name__, "repr": repr(constraint)})
    return {
        "symbols": atoms.get_chemical_symbols(),
        # 1e-8 A/degree-independent canonicalization removes signed zero and
        # harmless BLAS differences while remaining far below structural
        # validation tolerances.
        "positions": _rounded_list(atoms.get_positions()),
        "cell": _rounded_list(atoms.cell.array),
        "pbc": [bool(value) for value in atoms.pbc],
        "constraints": constraints,
        "initial_magmoms": _rounded_list(atoms.get_initial_magnetic_moments()),
        "initial_charges": _rounded_list(atoms.get_initial_charges()),
    }


def atoms_hash(atoms: Atoms) -> str:
    raw = json.dumps(atoms_payload(atoms), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _constrained_count(atoms: Atoms) -> int:
    """Order-independent count of atoms touched by any constraint."""
    indices: set[int] = set()
    for constraint in atoms.constraints:
        getter = getattr(constraint, "get_indices", None)
        if getter is None:
            continue
        try:
            indices.update(int(i) for i in getter())
        except Exception:
            continue
    return len(indices)


def structure_invariants(atoms: Atoms) -> dict[str, Any]:
    """Physical invariants that survive equivalent-but-different atom orderings.

    Unlike ``atoms_hash`` these ignore atom order and tolerate harmless
    floating-point drift, giving a semantic pass/fail alongside the exact hash.
    Lives here (not in ``llm_local``) so the export path needs no torch.
    """
    a, b, c, alpha, beta, gamma = (float(v) for v in atoms.cell.cellpar())
    return {
        "formula": dict(sorted(Counter(atoms.get_chemical_symbols()).items())),
        "natoms": len(atoms),
        "pbc": [bool(v) for v in atoms.pbc],
        "cell_lengths": [round(a, 3), round(b, 3), round(c, 3)],
        "cell_angles": [round(alpha, 2), round(beta, 2), round(gamma, 2)],
        "constrained_atoms": _constrained_count(atoms),
    }


def pair_hard_limits(numbers: np.ndarray) -> np.ndarray:
    """Element-pair minimum-distance table derived from covalent radii."""
    radii = covalent_radii[np.asarray(numbers, dtype=int)]
    return np.maximum(0.25, 0.35 * (radii[:, None] + radii[None, :]))


def _worst_overlap(atoms: Atoms) -> tuple[float, int, int, float, int] | None:
    """Return the pair that most violates *its own* threshold, plus a count.

    Comparing only the globally closest pair hides a violation whenever some
    other pair of larger atoms happens to sit closer in absolute terms, so the
    scan has to be per-pair like the threshold already is.
    """
    if len(atoms) < 2:
        return None
    distances = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))
    np.fill_diagonal(distances, np.inf)
    limits = pair_hard_limits(atoms.numbers)
    slack = distances - limits
    np.fill_diagonal(slack, np.inf)
    flat = int(np.argmin(slack))
    i, j = np.unravel_index(flat, slack.shape)
    if float(slack[i, j]) >= 0.0:
        return None
    violations = int(np.count_nonzero(slack < 0.0) // 2)
    return float(distances[i, j]), int(i), int(j), float(limits[i, j]), violations


def validate_atoms(
    atoms: Atoms,
    policy: AgentPolicy,
    *,
    profile: str = "intermediate",
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    if len(atoms) < 1:
        issues.append(ValidationIssue("empty_structure", "structure has no atoms", "error"))
    if len(atoms) > policy.max_atoms:
        issues.append(
            ValidationIssue(
                "atom_limit",
                f"structure has {len(atoms)} atoms; limit is {policy.max_atoms}",
                "error",
            )
        )

    symbols = atoms.get_chemical_symbols()
    invalid = sorted({symbol for symbol in symbols if symbol not in atomic_numbers})
    if invalid:
        issues.append(ValidationIssue("invalid_symbol", f"invalid symbols: {invalid}", "error"))

    positions = atoms.get_positions()
    cell = atoms.cell.array
    if not np.isfinite(positions).all():
        issues.append(ValidationIssue("nonfinite_positions", "positions must be finite", "error"))
    if not np.isfinite(cell).all():
        issues.append(ValidationIssue("nonfinite_cell", "cell must be finite", "error"))

    lengths = atoms.cell.lengths()
    if any(length > policy.max_cell_length for length in lengths):
        issues.append(ValidationIssue("cell_too_large", "cell exceeds policy limit", "error"))
    if np.any(atoms.pbc):
        volume = float(abs(np.linalg.det(cell)))
        if not math.isfinite(volume) or volume <= 1e-8:
            issues.append(ValidationIssue("singular_cell", "periodic structure needs positive volume", "error"))
        for axis, periodic in enumerate(atoms.pbc):
            if periodic and lengths[axis] < policy.min_cell_length:
                issues.append(ValidationIssue("short_cell", f"periodic axis {axis} is too short", "error"))

    if len(atoms) <= policy.max_atoms and np.isfinite(positions).all():
        overlap = _worst_overlap(atoms)
        if overlap is not None:
            distance, i, j, hard_limit, violations = overlap
            issues.append(
                ValidationIssue(
                    "atom_overlap",
                    f"atoms {i + 1} and {j + 1} are {distance:.3f} A apart "
                    f"(hard limit {hard_limit:.3f} A); {violations} pair(s) below limit",
                    "error",
                )
            )

    for constraint in atoms.constraints:
        try:
            indices = np.asarray(constraint.get_indices(), dtype=int)
        except Exception:
            continue
        if np.any(indices < 0) or np.any(indices >= len(atoms)):
            issues.append(ValidationIssue("invalid_constraint", "constraint index out of range", "error"))

    if len(atoms) and bool(atoms.pbc[0]) and bool(atoms.pbc[1]) and not bool(atoms.pbc[2]):
        z_span = float(np.ptp(positions[:, 2])) if len(atoms) > 1 else 0.0
        vacuum = float(lengths[2] - z_span)
        if vacuum < 6.0:
            severity = "error" if profile == "final" and vacuum < 3.0 else "warning"
            issues.append(
                ValidationIssue("low_vacuum", f"nonperiodic z vacuum is only {vacuum:.2f} A", severity)
            )

    return ValidationReport(tuple(issues))
