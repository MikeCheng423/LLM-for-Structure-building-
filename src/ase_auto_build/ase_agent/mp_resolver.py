"""Credential-free deterministic selection over an injected Materials Project transport."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import reduce
from math import gcd
from typing import Any, Callable, Iterable, Literal, Mapping

from ase import Atoms
from ase.formula import Formula

from .catalyst_contracts import validate_record
from .validation import atoms_hash, atoms_payload

_QUERY_FIELDS = {"material_id", "formula", "elements", "crystal_system", "spacegroup"}


class ReferenceResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class ReferenceResolution:
    status: Literal["resolved", "needs_clarification", "not_found"]
    candidate_material_ids: tuple[str, ...]
    record: dict[str, Any] | None = None
    structure: Atoms | None = None


def _formula_key(value: str) -> tuple[tuple[str, int], ...]:
    counts = Formula(value).count()
    divisor = reduce(gcd, counts.values())
    return tuple(sorted((element, count // divisor) for element, count in counts.items()))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Atoms):
        return atoms_payload(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ReferenceResolutionError(f"unsupported MP response value: {type(value).__name__}")


def _sha(value: Any) -> str:
    raw = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _candidate_formula(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("formula") or candidate.get("formula_pretty")
    if not isinstance(value, str):
        raise ReferenceResolutionError("MP candidate has no formula")
    return value


def _candidate_crystal_system(candidate: Mapping[str, Any]) -> str | None:
    symmetry = candidate.get("symmetry")
    if isinstance(symmetry, Mapping):
        return symmetry.get("crystal_system")
    return candidate.get("crystal_system")


def _candidate_spacegroup(candidate: Mapping[str, Any]) -> Any:
    symmetry = candidate.get("symmetry")
    if isinstance(symmetry, Mapping):
        return symmetry.get("number") or symmetry.get("symbol")
    return candidate.get("spacegroup")


def _calculation_method(candidate: Mapping[str, Any]) -> str:
    value = str(candidate.get("calculation_method") or candidate.get("run_type") or "unknown")
    aliases = {"GGA+U": "GGA_U", "GGA_U": "GGA_U", "GGA": "GGA", "r2SCAN": "r2SCAN"}
    return aliases.get(value, "unknown")


def resolve_reference(
    request_id: str,
    query: dict[str, Any],
    fetch: Callable[[dict[str, Any]], Iterable[dict[str, Any]]],
) -> ReferenceResolution:
    """Resolve one bulk parent or return an ordered clarification set."""
    extra = set(query) - _QUERY_FIELDS
    if extra:
        raise ReferenceResolutionError(f"unsupported MP query fields: {sorted(extra)}")
    if not query.get("material_id") and not query.get("formula"):
        raise ReferenceResolutionError("material_id or formula is required")
    if any("key" in field.lower() or "token" in field.lower() for field in query):
        raise ReferenceResolutionError("credentials are forbidden in resolver queries")

    candidates = list(fetch(dict(query)))
    if any(not isinstance(candidate, Mapping) for candidate in candidates):
        raise ReferenceResolutionError("MP transport returned a non-object candidate")
    material_ids = [candidate.get("material_id") for candidate in candidates]
    if any(not isinstance(value, str) or not value for value in material_ids):
        raise ReferenceResolutionError("every MP candidate requires a material_id")
    if len(set(material_ids)) != len(material_ids):
        raise ReferenceResolutionError("MP response contains duplicate material_id values")
    response_sha = _sha(candidates)
    material_id = query.get("material_id")
    if material_id:
        candidates = [item for item in candidates if item.get("material_id") == material_id]
    else:
        wanted_formula = _formula_key(str(query["formula"]))
        candidates = [item for item in candidates if _formula_key(_candidate_formula(item)) == wanted_formula]
        wanted_elements = set(query.get("elements") or ())
        if wanted_elements:
            candidates = [item for item in candidates if set(item.get("elements") or Formula(_candidate_formula(item)).count()) == wanted_elements]
        if query.get("crystal_system"):
            candidates = [item for item in candidates if _candidate_crystal_system(item) == query["crystal_system"]]
        if query.get("spacegroup") is not None:
            candidates = [item for item in candidates if _candidate_spacegroup(item) == query["spacegroup"]]

    candidates.sort(key=lambda item: (not bool(item.get("is_stable")), str(item.get("material_id", ""))))
    ids = tuple(str(item.get("material_id")) for item in candidates)
    if not candidates:
        return ReferenceResolution("not_found", ())
    if not material_id:
        preferred = [item for item in candidates if item.get("is_stable")]
        meaningful = preferred or candidates
        if len(meaningful) > 1:
            return ReferenceResolution("needs_clarification", tuple(str(item["material_id"]) for item in meaningful))
        selected = meaningful[0]
    else:
        if len(candidates) != 1:
            return ReferenceResolution("needs_clarification", ids)
        selected = candidates[0]

    structure = selected.get("structure")
    if not isinstance(structure, Atoms):
        raise ReferenceResolutionError("transport must convert the selected structure to ase.Atoms")
    if _formula_key(structure.get_chemical_formula()) != _formula_key(_candidate_formula(selected)):
        raise ReferenceResolutionError("candidate formula does not match its structure")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = {
        "schema_version": "reference-record/v1",
        "request_id": request_id,
        "created_at": now,
        "producer_version": "ASE_auto_build/0.9.2",
        "artifact_hashes": {"mp_response": response_sha},
        "source": "materials_project",
        "material_id": str(selected["material_id"]),
        "retrieved_at": now,
        "structure_role": "bulk_parent",
        "formula": _candidate_formula(selected),
        "structure_sha256": atoms_hash(structure),
        "calculation_method": _calculation_method(selected),
        "selection_rule": "user-specified mp-id" if material_id else "deterministic ranked query",
        "query_arguments": dict(query),
        "response_sha256": response_sha,
        "candidate_material_ids": list(ids),
        "citation_required": True,
    }
    if selected.get("task_id"):
        record["task_id"] = str(selected["task_id"])
    validate_record("reference_record", record)
    return ReferenceResolution("resolved", ids, record, structure.copy())


__all__ = ["ReferenceResolution", "ReferenceResolutionError", "resolve_reference"]
