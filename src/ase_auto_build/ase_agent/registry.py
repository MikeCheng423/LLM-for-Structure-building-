"""Single declarative registry and dependency-free JSON-schema subset validator."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

from .recipe import normalize_json
from .schemas import ToolSpec


class SchemaValidationError(ValueError):
    pass


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate(value: Any, schema: dict[str, Any], path: str) -> None:
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path} must be one of {schema['enum']!r}")
    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(f"{path} must equal {schema['const']!r}")

    expected = schema.get("type")
    valid = True
    if expected == "object":
        valid = isinstance(value, dict)
    elif expected == "array":
        valid = isinstance(value, list)
    elif expected == "string":
        valid = isinstance(value, str)
    elif expected == "integer":
        valid = _is_integer(value)
    elif expected == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    elif expected == "boolean":
        valid = isinstance(value, bool)
    elif expected == "null":
        valid = value is None
    if not valid:
        raise SchemaValidationError(f"{path} must be {expected}")

    if expected == "object":
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise SchemaValidationError(f"{path} missing required fields {missing}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise SchemaValidationError(f"{path} has unknown fields {extra}")
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key], f"{path}.{key}")
    elif expected == "array":
        if len(value) < schema.get("minItems", 0):
            raise SchemaValidationError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaValidationError(f"{path} has too many items")
        if schema.get("uniqueItems") and len({repr(item) for item in value}) != len(value):
            raise SchemaValidationError(f"{path} items must be unique")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{index}]")
    elif expected in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(f"{path} is below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaValidationError(f"{path} is above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise SchemaValidationError(f"{path} must exceed {schema['exclusiveMinimum']}")
    elif expected == "string":
        if len(value) < schema.get("minLength", 0):
            raise SchemaValidationError(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise SchemaValidationError(f"{path} is too long")


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool {spec.name!r}")
        if spec.parameters.get("type") != "object":
            raise ValueError(f"tool {spec.name!r} parameters must be an object schema")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool {name!r}") from exc

    def validate_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise SchemaValidationError("arguments must be an object")
        spec = self.get(name)
        _validate(arguments, spec.parameters, "arguments")
        return normalize_json(arguments)

    def function_schemas(self) -> list[dict[str, Any]]:
        return [self._specs[name].function_schema() for name in sorted(self._specs)]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def fingerprint(self) -> str:
        payload = json.dumps(self.function_schemas(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self._specs)
