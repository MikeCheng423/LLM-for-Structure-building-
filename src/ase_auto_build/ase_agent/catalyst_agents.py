"""Bounded LLM adapters for evidence extraction and CatalystSpec planning."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from .catalyst_contracts import load_schema, validate_record
from .controller import _arguments, _message_from_response

ChatCallable = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]
PRODUCER_VERSION = "ASE_auto_build/0.9.2"


class CatalystAgentError(ValueError):
    pass


def _sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _metadata(request_id: str, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer_version": PRODUCER_VERSION,
        "artifact_hashes": hashes,
    }


def _canonical_json(value: Any) -> Any:
    """Present JSON exactly as the training corpus baked it: sorted keys.

    The corpus JSONL is written with sort_keys=True, so every prompt the model
    saw had alphabetically ordered tool properties. apply_chat_template renders
    the tool schema into the prompt verbatim, so an insertion-ordered tool is a
    prompt shape the model never trained on -- it then mirrors that order in its
    output and drops fields such as catalyst_spec.modifications.
    """
    return json.loads(json.dumps(value, sort_keys=True))


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": properties, "required": required,
        },
    }}


def _call_one(chat: ChatCallable, messages: list[dict[str, Any]], tool: dict[str, Any]) -> dict[str, Any]:
    message = _message_from_response(chat(messages, [tool]))
    calls = message.get("tool_calls") or []
    expected = tool["function"]["name"]
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != expected:
        raise CatalystAgentError(f"agent must call {expected!r} exactly once")
    try:
        return _arguments(calls[0]["function"].get("arguments", {}))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CatalystAgentError(f"invalid {expected} payload: {exc}") from exc


def evidence_call(request: str, sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the exact deployment prompt and tool used by the extractor."""
    schema = load_schema("evidence_ledger")["properties"]
    claims = deepcopy(schema["claims"])
    claims["items"]["properties"].pop("locator")
    claims["items"]["properties"].pop("confidence")
    claims["items"]["required"].remove("locator")
    claims["items"]["required"].remove("confidence")
    tool = _tool(
        "submit_evidence_ledger",
        "Submit only claims explicitly grounded in the supplied source text.",
        {"claims": claims, **{key: schema[key] for key in ("contradictions", "unresolved_fields")}},
        ["claims", "contradictions", "unresolved_fields"],
    )
    messages = [
        {"role": "system", "content": (
            "Extract evidence only. Never invent a value, citation, or locator. "
            "Do not produce code, paths, coordinates, or a structure. Call "
            "submit_evidence_ledger once with claims, contradictions, and "
            "unresolved_fields. Include [] for either array when it is empty. "
            "Use exact CatalystSpec paths in claim.field, including material.formula, "
            "material.crystal_structure, model.kind, model.supercell, "
            "model.periodic_boundary_conditions, model.center, model.atom_ordering, "
            "and requested_outputs. Preserve arrays, numbers, and booleans as typed values."
        )},
        {"role": "user", "content": json.dumps({"request": request, "sources": sources}, sort_keys=True)},
    ]
    return messages, _canonical_json(tool)


def evidence_target(ledger: dict[str, Any]) -> dict[str, Any]:
    claims = [{key: value for key, value in claim.items() if key not in {"locator", "confidence"}}
              for claim in ledger["claims"]]
    return {"claims": claims, "contradictions": ledger["contradictions"],
            "unresolved_fields": ledger["unresolved_fields"]}


def complete_evidence_payload(payload: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {source["source_id"]: source for source in sources}
    conflicted = {item.get("field") for item in payload.get("contradictions", []) if isinstance(item, dict)}
    claims = []
    for claim in payload["claims"]:
        source = by_id.get(claim.get("source_id"))
        if source is None:
            raise CatalystAgentError("claim references an unknown source_id")
        claims.append({**claim, "locator": source["locator"],
                       "confidence": "low" if claim.get("field") in conflicted else "high"})
    return {**payload, "claims": claims}


def proposal_call(request: str, evidence: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the exact deployment prompt and tool used by the planner."""
    proposal_schema = load_schema("spec_proposal")["properties"]
    properties = {key: proposal_schema[key] for key in (
        "task_status", "catalyst_spec", "field_sources",
        "clarification_questions", "agent_warnings",
    )}
    tool = _tool(
        "submit_spec_proposal",
        "Submit a provenance-linked CatalystSpec or explicit clarification/refusal.",
        properties, list(properties),
    )
    planner_evidence = {
        "claims": [{key: claim[key] for key in ("field", "value", "evidence_type", "source_id")}
                   for claim in evidence["claims"]],
        "contradictions": evidence["contradictions"],
        "unresolved_fields": evidence["unresolved_fields"],
    }
    messages = [
        {"role": "system", "content": (
            "Propose CatalystSpec JSON from the EvidenceLedger. Link every reported "
            "field to a claim. Never execute code or silently choose topology-changing values. "
            "Call submit_spec_proposal once with task_status, catalyst_spec, field_sources, "
            "clarification_questions, and agent_warnings. Include [] for empty arrays. "
            "Return the tool call without commentary."
        )},
        {"role": "user", "content": json.dumps({"request": request, "evidence_ledger": planner_evidence}, sort_keys=True)},
    ]
    return messages, _canonical_json(tool)


def proposal_target(proposal: dict[str, Any]) -> dict[str, Any]:
    """Return the compact model payload; deterministic code owns duplicated fields."""
    spec = deepcopy(proposal["catalyst_spec"])
    spec.pop("provenance", None)
    spec.pop("task_status", None)
    spec.pop("clarification_questions", None)
    return {
        "task_status": proposal["task_status"], "catalyst_spec": spec,
        "field_sources": proposal["field_sources"],
        "clarification_questions": proposal["clarification_questions"],
        "agent_warnings": proposal["agent_warnings"],
    }


def complete_proposal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    spec = deepcopy(payload["catalyst_spec"])
    spec["task_status"] = payload["task_status"]
    spec["clarification_questions"] = payload["clarification_questions"]
    provenance = []
    for source in payload["field_sources"]:
        value: Any = spec
        try:
            for key, index in re.findall(r"([A-Za-z_]+)(?:\[(\d+)\])?", source["field"]):
                value = value[key]
                if index:
                    value = value[int(index)]
        except (KeyError, IndexError, TypeError) as exc:
            raise CatalystAgentError(f"field source references an absent field: {source['field']}") from exc
        item = {key: source[key] for key in ("field", "evidence_type", "claim_index", "reason") if key in source}
        item["value"] = value
        provenance.append(item)
    spec["provenance"] = provenance
    return {**payload, "catalyst_spec": spec}


def extract_evidence(
    request_id: str,
    request: str,
    sources: list[dict[str, Any]],
    chat: ChatCallable,
    *,
    model_location: Literal["local", "external"] = "local",
    authorize_private_external: bool = False,
) -> dict[str, Any]:
    """Extract claims once; deterministic code stamps and grounds the ledger."""
    if model_location == "external" and any(source.get("private") for source in sources) and not authorize_private_external:
        raise CatalystAgentError("private evidence requires explicit external-provider authorization")
    public_sources = [{
        "source_id": source["source_id"], "text": source["text"],
        "locator": source["locator"],
    } for source in sources]
    if len({source["source_id"] for source in public_sources}) != len(public_sources):
        raise CatalystAgentError("source_id values must be unique")
    messages, tool = evidence_call(request, public_sources)
    payload = complete_evidence_payload(_call_one(chat, messages, tool), public_sources)
    by_id = {source["source_id"]: source for source in public_sources}
    for claim in payload.get("claims", []):
        source = by_id.get(claim.get("source_id"))
        if source is None:
            raise CatalystAgentError("claim references an unknown source_id")
        if claim.get("locator") != source["locator"]:
            raise CatalystAgentError("claim locator does not match its supplied source")
        span = claim.get("verbatim_span")
        if span and span not in source["text"]:
            raise CatalystAgentError("verbatim_span is absent from its supplied source")
    record = {
        **_metadata(request_id, {"supplied_evidence": _sha(public_sources)}),
        "schema_version": "evidence-ledger/v1", **payload,
    }
    return validate_record("evidence_ledger", record)


def plan_spec(
    request_id: str,
    request: str,
    evidence: dict[str, Any],
    chat: ChatCallable,
) -> dict[str, Any]:
    """Propose a spec once; policy_gate remains the authority to build."""
    validate_record("evidence_ledger", evidence)
    messages, tool = proposal_call(request, evidence)
    payload = complete_proposal_payload(_call_one(chat, messages, tool))
    record = {
        **_metadata(request_id, {"evidence_ledger": _sha(evidence)}),
        "schema_version": "spec-proposal/v1", **payload,
    }
    validate_record("catalyst_spec", record["catalyst_spec"])
    return validate_record("spec_proposal", record)


__all__ = [
    "CatalystAgentError", "complete_evidence_payload", "complete_proposal_payload",
    "evidence_call", "evidence_target", "extract_evidence", "plan_spec", "proposal_call", "proposal_target",
]
