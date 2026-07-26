"""Shared typed records for the safe ASE tool runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from ase import Atoms
    from .workspace import ASEWorkspace


ToolClass = Literal["read_only", "mutates_structure", "proposes_compute", "control"]


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: Literal["warning", "error"]


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")


@dataclass
class ToolOutcome:
    """Executor output; the workspace performs validation and commit."""

    structure_name: str | None = None
    atoms: Atoms | None = None
    parent_names: tuple[str, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)
    result_name: str | None = None


ToolExecutor = Callable[["ASEWorkspace", dict[str, Any]], ToolOutcome]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    classification: ToolClass
    executor: ToolExecutor
    schema_version: int = 1
    estimated_cost: Literal["low", "medium", "high"] = "low"
    confirmation_required: bool = False

    def function_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class Revision:
    revision_id: str
    parent_revision_ids: tuple[str, ...]
    atoms: Atoms
    tool: str
    arguments: dict[str, Any]
    input_hashes: tuple[str, ...]
    output_hash: str
    created_at: float
    warnings: tuple[ValidationIssue, ...] = ()


@dataclass
class StructureHistory:
    name: str
    revisions: dict[str, Revision] = field(default_factory=dict)
    active_revision_id: str | None = None
    redo_stack: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ToolResult:
    success: bool
    tool: str
    arguments: dict[str, Any]
    observation: dict[str, Any] = field(default_factory=dict)
    error: dict[str, str] | None = None


@dataclass
class SessionBudget:
    model_turns: int = 0
    tool_calls: int = 0
    errors: int = 0

