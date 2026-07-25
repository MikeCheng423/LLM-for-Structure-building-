"""Transactional named ASE structures with immutable revision history."""

from __future__ import annotations

import copy
import re
import time
import uuid
from typing import Any

from ase import Atoms

from .observations import structure_observation
from .policy import AgentPolicy, DEFAULT_POLICY
from .recipe import canonical_recipe, recipe_hash
from .registry import ToolRegistry
from .schemas import Revision, SessionBudget, StructureHistory, ToolOutcome, ToolResult
from .validation import atoms_hash, validate_atoms


class ToolExecutionError(RuntimeError):
    pass


_SAFE_STRUCTURE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")


class ASEWorkspace:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        policy: AgentPolicy = DEFAULT_POLICY,
        session_id: str | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.session_id = session_id or str(uuid.uuid4())
        self.structures: dict[str, StructureHistory] = {}
        self.active_name: str | None = None
        self.result_name: str | None = None
        self.pending_clarification: dict[str, Any] | None = None
        self.pending_confirmation: dict[str, Any] | None = None
        self.budget = SessionBudget()
        self.steps: list[dict[str, Any]] = []
        self.transcript: list[dict[str, Any]] = []
        self._revision_counter = 0

    def require_atoms(self, name: str) -> Atoms:
        history = self.structures.get(name)
        if history is None or history.active_revision_id is None:
            raise KeyError(f"no structure named {name!r}")
        return history.revisions[history.active_revision_id].atoms.copy()

    def active_revision(self, name: str) -> Revision:
        history = self.structures.get(name)
        if history is None or history.active_revision_id is None:
            raise KeyError(f"no structure named {name!r}")
        return history.revisions[history.active_revision_id]

    def _commit(self, tool: str, arguments: dict[str, Any], outcome: ToolOutcome) -> dict[str, Any]:
        if outcome.atoms is None or outcome.structure_name is None:
            raise ToolExecutionError(f"mutating tool {tool!r} returned no structure")
        name = outcome.structure_name
        if _SAFE_STRUCTURE_NAME.fullmatch(name) is None:
            raise ToolExecutionError(
                "structure name must be a path-free identifier beginning with a letter"
            )

        parents: list[Revision] = []
        for parent_name in outcome.parent_names:
            parents.append(self.active_revision(parent_name))
        if name in self.structures and not outcome.parent_names:
            raise ToolExecutionError(f"structure {name!r} already exists")
        if name not in self.structures and len(self.structures) >= self.policy.max_named_structures:
            raise ToolExecutionError("named-structure limit reached")

        atoms = outcome.atoms.copy()
        report = validate_atoms(atoms, self.policy)
        if not report.ok:
            message = "; ".join(issue.message for issue in report.errors)
            raise ToolExecutionError(message)

        history = self.structures.setdefault(name, StructureHistory(name=name))
        if len(history.revisions) >= self.policy.max_revisions_per_structure:
            raise ToolExecutionError(f"revision limit reached for {name!r}")
        self._revision_counter += 1
        revision_id = f"r{self._revision_counter:06d}"
        revision = Revision(
            revision_id=revision_id,
            parent_revision_ids=tuple(parent.revision_id for parent in parents),
            atoms=atoms,
            tool=tool,
            arguments=copy.deepcopy(arguments),
            input_hashes=tuple(parent.output_hash for parent in parents),
            output_hash=atoms_hash(atoms),
            created_at=time.time(),
            warnings=report.warnings,
        )
        history.revisions[revision_id] = revision
        history.active_revision_id = revision_id
        history.redo_stack.clear()
        self.active_name = name
        warning_data = [
            {"code": warning.code, "message": warning.message}
            for warning in report.warnings
        ]
        observation = structure_observation(
            name,
            atoms,
            revision_id=revision_id,
            warnings=warning_data,
        )
        observation.update(copy.deepcopy(outcome.data))
        return observation

    def _with_default_name(self, spec, arguments: dict[str, Any]) -> dict[str, Any]:
        """Default an omitted `name` to the active structure.

        The `name` is a workspace bookkeeping handle a user never speaks; forcing
        the model to invent it is brittle. When a tool accepts `name` and the call
        omits it, resolve it to the active structure (or "structure" for the first
        build). Matching compares atoms_hash, not the handle, so this is neutral.
        """
        properties = spec.parameters.get("properties", {})
        if (
            "name" in properties
            and isinstance(arguments, dict)
            and "name" not in arguments
        ):
            return {**arguments, "name": self.active_name or "structure"}
        return arguments

    def execute(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        self.budget.tool_calls += 1
        normalized: dict[str, Any] = {}
        if self.budget.tool_calls > self.policy.max_tool_calls:
            return ToolResult(
                False,
                tool,
                {},
                error={"code": "budget_exhausted", "message": "tool-call limit reached"},
            )
        try:
            spec = self.registry.get(tool)
            arguments = self._with_default_name(spec, arguments)
            normalized = self.registry.validate_arguments(tool, arguments)
            outcome = spec.executor(self, copy.deepcopy(normalized))
            if spec.classification == "mutates_structure":
                observation = self._commit(tool, normalized, outcome)
            else:
                observation = {"ok": True, **copy.deepcopy(outcome.data)}
            if outcome.result_name is not None:
                self.require_atoms(outcome.result_name)
                self.result_name = outcome.result_name
                observation["result_name"] = outcome.result_name
            self.steps.append({"tool": tool, "args": copy.deepcopy(normalized)})
            result = ToolResult(True, tool, normalized, observation=observation)
        except Exception as exc:
            self.budget.errors += 1
            result = ToolResult(
                False,
                tool,
                normalized or copy.deepcopy(arguments),
                error={"code": type(exc).__name__, "message": str(exc)},
            )
        self.transcript.append(
            {
                "tool": result.tool,
                "args": copy.deepcopy(result.arguments),
                "result": copy.deepcopy(result.observation if result.success else result.error),
                "success": result.success,
            }
        )
        return result

    def execute_or_raise(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        result = self.execute(tool, arguments)
        if not result.success:
            assert result.error is not None
            raise ToolExecutionError(f"{result.error['code']}: {result.error['message']}")
        return result

    def final_atoms(self) -> Atoms:
        if self.result_name is None:
            raise ToolExecutionError("finish has not selected a result")
        atoms = self.require_atoms(self.result_name)
        report = validate_atoms(atoms, self.policy, profile="final")
        if not report.ok:
            raise ToolExecutionError("; ".join(issue.message for issue in report.errors))
        return atoms

    def recipe(self) -> dict[str, Any]:
        return canonical_recipe(self.steps)

    def recipe_hash(self) -> str:
        return recipe_hash(self.recipe())
