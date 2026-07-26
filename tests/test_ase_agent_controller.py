from __future__ import annotations

import json

import pytest

pytest.importorskip("ase")

from ase_auto_build.ase_agent import (
    ASEWorkspace,
    AgentController,
    ControllerState,
    create_default_registry,
)


def call(name: str, arguments: dict, number: int = 1) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": f"call_{number}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


class ScriptedChat:
    def __init__(self, messages: list[dict]) -> None:
        self.messages = list(messages)
        self.calls = 0

    def __call__(self, messages, tools):
        assert messages[0]["role"] == "system"
        assert tools
        self.calls += 1
        return self.messages.pop(0)


class ToolRecordingChat(ScriptedChat):
    def __call__(self, messages, tools):
        self.tools = [tool["function"]["name"] for tool in tools]
        return super().__call__(messages, tools)


def controller(script: list[dict]) -> AgentController:
    return AgentController(
        ASEWorkspace(create_default_registry(), session_id="controller-test"),
        ScriptedChat(script),
    )


def test_controller_builds_and_requires_finish() -> None:
    worker = controller([
        call("build_bulk", {"name": "al", "element": "Al", "crystal": "fcc"}),
        call("finish", {"name": "al"}, 2),
    ])
    result = worker.run("Build fcc aluminium.")
    assert result.state is ControllerState.FINISHED
    assert result.result_name == "al"
    assert [item["tool"] for item in result.transcript] == ["build_bulk", "finish"]


def test_controller_feeds_error_back_then_recovers() -> None:
    worker = controller([
        call("build_bulk", {"name": "al", "element": "Al", "shell": "id"}),
        call("build_bulk", {"name": "al", "element": "Al", "crystal": "fcc"}, 2),
        call("finish", {"name": "al"}, 3),
    ])
    result = worker.run("Build fcc aluminium.")
    assert result.state is ControllerState.FINISHED
    assert result.transcript[0]["success"] is False
    assert result.transcript[1]["success"] is True
    tool_error = next(
        message for message in result.messages
        if message["role"] == "tool" and "unknown fields" in message["content"]
    )
    assert tool_error


def test_controller_rejects_text_only_completion() -> None:
    worker = controller([{"role": "assistant", "content": "Done."}])
    result = worker.run("Build aluminium.")
    assert result.state is ControllerState.FAILED
    assert "without calling finish" in result.error


def test_controller_stops_repeated_failed_call() -> None:
    bad = call("finish", {"name": "missing"})
    worker = controller([bad, bad, bad])
    result = worker.run("Finish a structure that does not exist.")
    assert result.state is ControllerState.FAILED
    assert "repeated" in result.error


def test_controller_pauses_on_clarification() -> None:
    worker = controller([
        call("ask_clarification", {
            "question": "Which surface?", "choices": ["(111)", "(100)"],
            "field": "miller",
        }),
    ])
    result = worker.run("Build a platinum surface.")
    assert result.state is ControllerState.NEEDS_CLARIFICATION
    assert result.error is None
    assert worker.workspace.pending_clarification["field"] == "miller"


def test_controller_resumes_after_clarification_with_full_context() -> None:
    scripted = ScriptedChat([
        call("ask_clarification", {
            "question": "Which surface?", "choices": ["(111)", "(100)"],
            "field": "miller",
        }),
        call("build_surface", {
            "name": "slab", "element": "Pt", "crystal": "fcc",
            "miller": [1, 1, 1], "layers": 4, "vacuum": 12.0,
        }, 2),
        call("finish", {"name": "slab"}, 3),
    ])
    worker = AgentController(
        ASEWorkspace(create_default_registry(), session_id="resume-test"),
        scripted,
    )

    paused = worker.run("Build a platinum surface.")
    result = worker.resume("Use Pt(111), four layers, and 12 A vacuum.")

    assert paused.state is ControllerState.NEEDS_CLARIFICATION
    assert result.state is ControllerState.FINISHED
    assert worker.workspace.pending_clarification is None
    assert any(
        message["role"] == "user" and message["content"].startswith("Use Pt(111)")
        for message in result.messages
    )


def test_controller_rejects_resume_when_not_paused() -> None:
    worker = controller([call("build_bulk", {"name": "al", "element": "Al"})])
    with pytest.raises(RuntimeError, match="not waiting"):
        worker.resume("continue")


def test_controller_routes_supported_request_to_relevant_tools() -> None:
    chat = ToolRecordingChat([
        call("build_nanotube", {"name": "tube", "n": 4, "m": 3}),
        call("finish", {"name": "tube"}, 2),
    ])
    worker = AgentController(
        ASEWorkspace(create_default_registry(), session_id="router-test"), chat
    )

    assert worker.run("Build a (4,3) carbon nanotube.").state is ControllerState.FINISHED
    assert chat.tools == ["build_nanotube", "finish"]


def test_controller_rejects_request_outside_supported_region() -> None:
    worker = controller([])
    with pytest.raises(ValueError, match="unsupported structure request"):
        worker.run("Tell me a random sentence about the weather.")
