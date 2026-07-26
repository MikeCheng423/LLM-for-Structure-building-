"""Provider-neutral, bounded tool-calling state machine."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .workspace import ASEWorkspace, ToolExecutionError
from .tool_router import route_tools


class ControllerState(str, Enum):
    NEW = "NEW"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


SYSTEM_PROMPT = (
    "Build structures only with the registered deterministic ASE tools. "
    "Use typed selectors, inspect tool results, and call finish exactly once. "
    "Use cubic bulk cells before repeat and omit optional arguments the user did not specify. "
    "Never emit Python, shell commands, file paths, or atomic coordinates."
)


ChatCallable = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]


@dataclass(frozen=True)
class ControllerResult:
    state: ControllerState
    result_name: str | None
    messages: tuple[dict[str, Any], ...]
    transcript: tuple[dict[str, Any], ...]
    error: str | None = None


def _message_from_response(response: dict[str, Any]) -> dict[str, Any]:
    if "choices" in response:
        return response["choices"][0]["message"]
    return response


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        decoded = json.loads(value or "{}")
        if isinstance(decoded, dict):
            return decoded
    raise ValueError("tool arguments must decode to an object")


class AgentController:
    def __init__(self, workspace: ASEWorkspace, chat: ChatCallable) -> None:
        self.workspace = workspace
        self.chat = chat
        self.state = ControllerState.NEW
        self._messages: list[dict[str, Any]] = []
        self._repeated_failures: dict[str, int] = {}
        self._tools: list[dict[str, Any]] = []

    def run(self, request: str) -> ControllerResult:
        if not request or not request.strip():
            raise ValueError("structure request must not be empty")
        if self.state is not ControllerState.NEW:
            raise RuntimeError("controller has already started; use resume after a pause")
        self._tools = route_tools(request, self.workspace.registry)
        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.strip()},
        ]
        self.state = ControllerState.PLANNING
        return self._advance()

    def resume(self, response: str) -> ControllerResult:
        """Continue a paused session without losing model or tool context."""
        if self.state not in {
            ControllerState.NEEDS_CLARIFICATION,
            ControllerState.NEEDS_CONFIRMATION,
        }:
            raise RuntimeError("controller is not waiting for user input")
        if not response or not response.strip():
            raise ValueError("user response must not be empty")
        if self.state is ControllerState.NEEDS_CLARIFICATION:
            self.workspace.pending_clarification = None
        else:
            self.workspace.pending_confirmation = None
        self._messages.append({"role": "user", "content": response.strip()})
        self.state = ControllerState.PLANNING
        return self._advance()

    def _advance(self) -> ControllerResult:
        remaining_turns = (
            self.workspace.policy.max_model_turns - self.workspace.budget.model_turns
        )
        if remaining_turns <= 0:
            self.state = ControllerState.BUDGET_EXHAUSTED
            return self._result("model-turn budget exhausted")

        for _ in range(remaining_turns):
            self.workspace.budget.model_turns += 1
            try:
                message = _message_from_response(
                    self.chat(
                        copy.deepcopy(self._messages),
                        copy.deepcopy(self._tools),
                    )
                )
            except Exception as exc:
                self.state = ControllerState.FAILED
                return self._result(f"model call failed: {exc}")

            tool_calls = message.get("tool_calls") or []
            assistant_message = {
                "role": "assistant",
                "content": message.get("content") or "",
            }
            if tool_calls:
                assistant_message["tool_calls"] = copy.deepcopy(tool_calls)
            self._messages.append(assistant_message)
            if not tool_calls:
                self.state = ControllerState.FAILED
                return self._result("model stopped without calling finish")

            self.state = ControllerState.EXECUTING
            for tool_call in tool_calls:
                call_id = str(tool_call.get("id") or f"call_{self.workspace.budget.tool_calls + 1}")
                try:
                    function = tool_call["function"]
                    tool = str(function["name"])
                    arguments = _arguments(function.get("arguments", {}))
                    spec = self.workspace.registry.get(tool)
                    if len(tool_calls) > 1 and spec.classification != "read_only":
                        raise ValueError("parallel mutating calls are forbidden")
                    result = self.workspace.execute(tool, arguments)
                except Exception as exc:
                    tool = str(tool_call.get("function", {}).get("name", "<invalid>"))
                    arguments = {}
                    result_data = {"code": type(exc).__name__, "message": str(exc)}
                    success = False
                else:
                    result_data = result.observation if result.success else result.error
                    success = result.success

                signature = json.dumps(
                    {"tool": tool, "args": arguments}, sort_keys=True, separators=(",", ":")
                )
                if not success:
                    self._repeated_failures[signature] = (
                        self._repeated_failures.get(signature, 0) + 1
                    )
                    if self._repeated_failures[signature] > 2:
                        self.state = ControllerState.FAILED
                        self._messages.append({
                            "role": "tool", "tool_call_id": call_id, "name": tool,
                            "content": json.dumps(result_data, sort_keys=True),
                        })
                        return self._result("same failed tool call repeated more than twice")

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool,
                    "content": json.dumps(result_data, sort_keys=True, separators=(",", ":")),
                })
                if success and isinstance(result_data, dict) and result_data.get("needs_clarification"):
                    self.state = ControllerState.NEEDS_CLARIFICATION
                    return self._result(None)
                if self.workspace.result_name is not None:
                    try:
                        self.workspace.final_atoms()
                    except ToolExecutionError as exc:
                        self.state = ControllerState.FAILED
                        return self._result(str(exc))
                    self.state = ControllerState.FINISHED
                    return self._result(None)

            if self.workspace.budget.errors > self.workspace.policy.max_errors:
                self.state = ControllerState.BUDGET_EXHAUSTED
                return self._result("error budget exhausted")

        self.state = ControllerState.BUDGET_EXHAUSTED
        return self._result("model-turn budget exhausted")

    def _result(self, error: str | None) -> ControllerResult:
        sanitized = tuple(
            {
                "tool": item["tool"],
                "args": copy.deepcopy(item["args"]),
                "result": copy.deepcopy(item["result"]),
                "success": item["success"],
            }
            for item in self.workspace.transcript
        )
        return ControllerResult(
            self.state,
            self.workspace.result_name,
            tuple(copy.deepcopy(self._messages)),
            sanitized,
            error,
        )
