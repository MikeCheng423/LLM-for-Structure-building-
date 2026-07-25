"""Canonical replayable recipe serialization and hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


RECIPE_SCHEMA_VERSION = 1


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_json(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [normalize_json(item) for item in value]
    if isinstance(value, float):
        return float(f"{value:.12g}")
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"recipe contains unsupported value {type(value).__name__}")


def canonical_recipe(steps: Iterable[dict[str, Any]]) -> dict[str, Any]:
    normalized = [
        {"tool": str(step["tool"]), "args": normalize_json(step.get("args", {}))}
        for step in steps
    ]
    return {"schema_version": RECIPE_SCHEMA_VERSION, "steps": normalized}


def recipe_bytes(recipe: dict[str, Any]) -> bytes:
    return json.dumps(normalize_json(recipe), sort_keys=True, separators=(",", ":")).encode("utf-8")


def recipe_hash(recipe: dict[str, Any]) -> str:
    return hashlib.sha256(recipe_bytes(recipe)).hexdigest()

