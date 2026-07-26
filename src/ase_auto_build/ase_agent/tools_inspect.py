"""Read-only measurements, validation, recipe preview, and finish."""

from __future__ import annotations

import copy

from .observations import structure_observation
from .registry import ToolRegistry
from .schemas import ToolOutcome, ToolSpec
from .tools_build import NAME, _object
from .validation import validate_atoms


INDEX = {"type": "integer", "minimum": 1}


def _inspect(workspace, args):
    return ToolOutcome(data=structure_observation(args["name"], workspace.require_atoms(args["name"])))


def _distance(workspace, args):
    atoms = workspace.require_atoms(args["name"])
    i, j = int(args["a"]) - 1, int(args["b"]) - 1
    if min(i, j) < 0 or max(i, j) >= len(atoms):
        raise IndexError("measurement atom index out of range")
    return ToolOutcome(data={"name": args["name"], "a": i + 1, "b": j + 1, "distance": round(float(atoms.get_distance(i, j, mic=bool(args.get("mic", True)))), 8)})


def _angle(workspace, args):
    atoms = workspace.require_atoms(args["name"])
    indices = [int(args[key]) - 1 for key in ("a", "b", "c")]
    if min(indices) < 0 or max(indices) >= len(atoms):
        raise IndexError("measurement atom index out of range")
    return ToolOutcome(data={"name": args["name"], "indices": [index + 1 for index in indices], "angle": round(float(atoms.get_angle(*indices, mic=bool(args.get("mic", True)))), 8)})


def _validate(workspace, args):
    report = validate_atoms(workspace.require_atoms(args["name"]), workspace.policy, profile=args.get("profile", "final"))
    return ToolOutcome(data={
        "name": args["name"], "valid": report.ok,
        "issues": [{"code": issue.code, "severity": issue.severity, "message": issue.message} for issue in report.issues],
    })


def _preview(workspace, args):
    recipe = workspace.recipe()
    return ToolOutcome(data={"recipe": copy.deepcopy(recipe), "recipe_hash": workspace.recipe_hash()})


def _finish(workspace, args):
    atoms = workspace.require_atoms(args["name"])
    report = validate_atoms(atoms, workspace.policy, profile="final")
    if not report.ok:
        raise ValueError("; ".join(issue.message for issue in report.errors))
    return ToolOutcome(data={"finished": True}, result_name=args["name"])


def _ask_clarification(workspace, args):
    request = {
        "question": args["question"],
        "choices": list(args.get("choices", [])),
        "field": args.get("field"),
    }
    workspace.pending_clarification = request
    return ToolOutcome(data={"needs_clarification": True, **request})


def register(registry: ToolRegistry) -> None:
    registry.register(ToolSpec("inspect_structure", "Return a compact structure summary.", _object({"name": NAME}, ["name"]), "read_only", _inspect))
    registry.register(ToolSpec("measure_distance", "Measure a bounded atom-pair distance.", _object({"name": NAME, "a": INDEX, "b": INDEX, "mic": {"type": "boolean"}}, ["name", "a", "b"]), "read_only", _distance))
    registry.register(ToolSpec("measure_angle", "Measure an a-b-c atom angle.", _object({"name": NAME, "a": INDEX, "b": INDEX, "c": INDEX, "mic": {"type": "boolean"}}, ["name", "a", "b", "c"]), "read_only", _angle))
    registry.register(ToolSpec("validate_structure", "Run deterministic intermediate or final validation.", _object({"name": NAME, "profile": {"type": "string", "enum": ["intermediate", "final"]}}, ["name"]), "read_only", _validate))
    registry.register(ToolSpec("preview_recipe", "Return the canonical recipe and SHA-256.", _object({}, []), "read_only", _preview))
    registry.register(ToolSpec("ask_clarification", "Pause for a material structure-building choice.", _object({
        "question": {"type": "string", "minLength": 1, "maxLength": 500},
        "choices": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 200}, "minItems": 2, "maxItems": 8},
        "field": {"type": "string", "minLength": 1, "maxLength": 64},
    }, ["question"]), "control", _ask_clarification))
    registry.register(ToolSpec("finish", "Validate and select exactly one final structure.", _object({"name": NAME}, ["name"]), "control", _finish))
