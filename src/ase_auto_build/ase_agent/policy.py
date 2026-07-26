"""Server-side ceilings for interactive structure construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentPolicy:
    max_atoms: int = 2_000
    max_model_turns: int = 24
    max_tool_calls: int = 64
    max_errors: int = 8
    max_revisions_per_structure: int = 20
    max_named_structures: int = 16
    max_repeat_axis: int = 12
    max_cell_length: float = 500.0
    min_cell_length: float = 0.1
    max_import_bytes: int = 20 * 1024 * 1024
    max_optimization_steps_without_confirmation: int = 200
    max_md_steps_without_confirmation: int = 10_000

    def __post_init__(self) -> None:
        positive = (
            "max_atoms",
            "max_model_turns",
            "max_tool_calls",
            "max_errors",
            "max_revisions_per_structure",
            "max_named_structures",
            "max_repeat_axis",
        )
        for field_name in positive:
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be positive")


DEFAULT_POLICY = AgentPolicy()

