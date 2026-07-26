"""Construction of the one authoritative phase-one tool registry."""

from __future__ import annotations

from .registry import ToolRegistry
from . import tools_build, tools_constraints, tools_edit, tools_inspect, tools_surface


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for module in (tools_build, tools_edit, tools_surface, tools_constraints, tools_inspect):
        module.register(registry)
    return registry

