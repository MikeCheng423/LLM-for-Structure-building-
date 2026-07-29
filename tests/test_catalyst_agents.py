from __future__ import annotations

from copy import deepcopy

import pytest

from ase_auto_build.ase_agent.catalyst_agents import CatalystAgentError, extract_evidence, plan_spec
from ase_auto_build.ase_agent.catalyst_pipeline import run_journal_request
from tests.test_catalyst_contracts import records


class ScriptedChat:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments
        self.calls = 0

    def __call__(self, messages, tools):
        self.calls += 1
        assert [tool["function"]["name"] for tool in tools] == [self.name]
        return {"role": "assistant", "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }]}


class SequenceChat:
    def __init__(self, calls):
        self.calls = list(calls)
        self.seen = []

    def __call__(self, messages, tools):
        expected, arguments = self.calls.pop(0)
        assert tools[0]["function"]["name"] == expected
        self.seen.append(expected)
        return {"role": "assistant", "tool_calls": [{
            "id": f"call-{len(self.seen)}", "type": "function",
            "function": {"name": expected, "arguments": arguments},
        }]}


def test_bounded_agents_create_valid_handoffs() -> None:
    source = {
        "source_id": "paper-1", "locator": "Methods, p. 4",
        "text": "A conventional 2 x 2 x 2 fcc Pt bulk cell was used.",
    }
    claims = [{
        "field": "material.formula", "value": "Pt", "evidence_type": "reported",
        "source_id": "paper-1", "locator": "Methods, p. 4",
        "verbatim_span": "fcc Pt", "confidence": "high",
    }]
    ledger = extract_evidence("request-1", "Build it", [source], ScriptedChat(
        "submit_evidence_ledger", {"claims": claims, "contradictions": [], "unresolved_fields": []},
    ))
    _, template = records()
    payload = {key: deepcopy(template[key]) for key in (
        "task_status", "catalyst_spec", "field_sources", "clarification_questions", "agent_warnings",
    )}
    proposal = plan_spec("request-1", "Build it", ledger, ScriptedChat("submit_spec_proposal", payload))
    assert ledger["schema_version"] == "evidence-ledger/v1"
    assert proposal["schema_version"] == "spec-proposal/v1"


def test_private_evidence_needs_external_authorization() -> None:
    source = {"source_id": "private-1", "locator": "local", "text": "secret", "private": True}
    chat = ScriptedChat("submit_evidence_ledger", {})
    with pytest.raises(CatalystAgentError, match="explicit"):
        extract_evidence("request-1", "Build it", [source], chat, model_location="external")
    assert chat.calls == 0


def test_extractor_rejects_ungrounded_verbatim_span() -> None:
    source = {"source_id": "paper-1", "locator": "p. 1", "text": "Pt slab"}
    payload = {"claims": [{
        "field": "model.layers", "value": 4, "evidence_type": "reported",
        "source_id": "paper-1", "locator": "p. 1", "verbatim_span": "four layers",
        "confidence": "high",
    }], "contradictions": [], "unresolved_fields": []}
    with pytest.raises(CatalystAgentError, match="absent"):
        extract_evidence("request-1", "Build it", [source], ScriptedChat("submit_evidence_ledger", payload))


def test_full_journal_request_runs_without_interaction(tmp_path) -> None:
    evidence, proposal = records()
    evidence_payload = {key: deepcopy(evidence[key]) for key in (
        "claims", "contradictions", "unresolved_fields",
    )}
    proposal_payload = {key: deepcopy(proposal[key]) for key in (
        "task_status", "catalyst_spec", "field_sources", "clarification_questions", "agent_warnings",
    )}
    chat = SequenceChat([
        ("submit_evidence_ledger", evidence_payload),
        ("submit_spec_proposal", proposal_payload),
    ])
    result = run_journal_request(
        "request-1", "Build a conventional 2 x 2 x 2 fcc Pt bulk cell.",
        [{
            "source_id": "user-evidence-1", "locator": "request",
            "text": "Build a conventional 2 x 2 x 2 fcc Pt bulk cell.",
        }],
        chat, tmp_path,
    )
    assert result.status == "review_ready"
    assert (result.request_dir / "supplied_evidence.json").is_file()
    assert chat.seen == ["submit_evidence_ledger", "submit_spec_proposal"]
    assert not chat.calls


def test_full_auto_agent_failure_leaves_a_reviewable_record(tmp_path) -> None:
    def wrong_tool(messages, tools):
        return {"role": "assistant", "tool_calls": [{
            "id": "bad", "type": "function",
            "function": {"name": "run_shell", "arguments": {}},
        }]}

    result = run_journal_request(
        "failed-request", "Build Pt", [{
            "source_id": "user-1", "locator": "request", "text": "Build Pt",
        }], wrong_tool, tmp_path,
    )
    assert result.status == "failed"
    assert (result.request_dir / "failure_record.json").is_file()
    assert (result.request_dir / "review_packet.json").is_file()
    assert not (result.request_dir / "POSCAR").exists()
