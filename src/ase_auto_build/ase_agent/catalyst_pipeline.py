"""Non-interactive policy → build → validate → export → review pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import ase
from ase.io import read as ase_read

from .catalyst_contracts import GateDecision, policy_gate, validate_record
from .catalyst_agents import ChatCallable, extract_evidence, plan_spec
from .catalyst_dispatch import dispatch_spec
from .catalyst_validation import catalyst_validation_rules
from .export import write_bundle
from .mp_resolver import ReferenceResolution
from .validation import atoms_hash, structure_invariants, validate_atoms

PRODUCER_VERSION = "ASE_auto_build/0.9.2"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def _sha(value: bytes | dict[str, Any]) -> str:
    if isinstance(value, dict):
        value = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def _spec_value(spec: dict[str, Any], path: str) -> Any:
    value: Any = spec
    for key, index in re.findall(r"([A-Za-z_]+)(?:\[(\d+)\])?", path):
        value = value[key]
        if index:
            value = value[int(index)]
    return value


def _metadata(request_id: str, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer_version": PRODUCER_VERSION,
        "artifact_hashes": hashes,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class CatalystPipelineResult:
    status: str
    request_dir: Path
    gate: GateDecision | None
    files: tuple[Path, ...] = ()


def _review_markdown(packet: dict[str, Any]) -> str:
    lines = [
        f"# Catalyst reconstruction {packet['request_id']}", "",
        f"Status: {packet['status']}", "",
        "## Structure summary", "",
        "```json", json.dumps(packet["structure_summary"], indent=2, sort_keys=True), "```", "",
        "## Source-backed facts", "",
    ]
    lines.extend(
        f"- `{fact['field']}` = `{json.dumps(fact['value'], sort_keys=True)}` "
        f"({fact['evidence_type']}; {fact['source_id']}, {fact['locator']})"
        for fact in packet["source_backed_facts"]
    )
    if not packet["source_backed_facts"]:
        lines.append("- None")
    lines += ["", "## Assumptions and derived values", ""]
    lines.extend(
        f"- `{item['field']}` = `{json.dumps(item['value'], sort_keys=True)}` "
        f"({item['evidence_type']}; {item['reason']})"
        for item in packet["assumptions"]
    )
    if not packet["assumptions"]:
        lines.append("- None")
    lines += ["", "## Warnings", ""]
    lines.extend(f"- {warning}" for warning in packet["warnings"])
    if not packet["warnings"]:
        lines.append("- None")
    lines += ["", "## Files", ""]
    lines.extend(f"- `{path}`" for path in packet["files"])
    lines += [
        "", "## Reproduction", "", "```json",
        json.dumps(packet["reproduction"], indent=2, sort_keys=True), "```",
    ]
    return "\n".join(lines) + "\n"


def _prepare_request_dir(
    output_root: Path,
    request_id: str,
    *,
    allowed_existing: set[str] | frozenset[str] = frozenset(),
) -> Path:
    if _SAFE_ID.fullmatch(request_id) is None:
        raise ValueError("request_id must be a path-free identifier")
    root = Path(output_root).resolve()
    request_dir = (root / request_id).resolve()
    if request_dir.parent != root:
        raise ValueError("request output escaped its root")
    existing = {path.name for path in request_dir.iterdir()} if request_dir.exists() else set()
    if existing - allowed_existing:
        raise FileExistsError(f"refusing to overwrite request package: {request_dir}")
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir


def _failure_result(request_id: str, request_dir: Path, stage: str, exc: Exception) -> CatalystPipelineResult:
    existing = sorted(path.name for path in request_dir.iterdir() if path.is_file())
    hashes = {name: _sha((request_dir / name).read_bytes()) for name in existing}
    failure = {
        **_metadata(request_id, hashes),
        "schema_version": "failure-record/v1", "stage": stage,
        "error_code": type(exc).__name__, "message": str(exc)[:1000] or type(exc).__name__,
    }
    validate_record("failure_record", failure)
    _write_json(request_dir / "failure_record.json", failure)
    packet = {
        **_metadata(request_id, {**hashes, "failure_record": _sha(failure)}),
        "schema_version": "review-packet/v1", "status": "failed",
        "structure_summary": {}, "source_backed_facts": [], "assumptions": [],
        "warnings": [failure["message"]], "files": [*existing, "failure_record.json"],
        "reproduction": {"failed_stage": stage, "error_code": failure["error_code"]},
    }
    validate_record("review_packet", packet)
    _write_json(request_dir / "review_packet.json", packet)
    (request_dir / "review.md").write_text(_review_markdown(packet), encoding="utf-8")
    return CatalystPipelineResult("failed", request_dir, None, tuple(sorted(request_dir.iterdir())))


def _run_catalyst_pipeline(
    evidence: dict[str, Any],
    proposal: dict[str, Any],
    output_root: Path,
    *,
    reference: ReferenceResolution | None = None,
    _allow_existing_inputs: bool = False,
) -> CatalystPipelineResult:
    """Run one immutable request package without any interactive fallback."""
    request_id = str(proposal.get("request_id", ""))
    allowed = {
        "supplied_evidence.json", "evidence_ledger.json", "spec_proposal.json",
    } if _allow_existing_inputs else set()
    request_dir = _prepare_request_dir(output_root, request_id, allowed_existing=allowed)

    input_hashes = {"evidence_ledger": _sha(evidence), "spec_proposal": _sha(proposal)}
    if (request_dir / "supplied_evidence.json").is_file():
        input_hashes["supplied_evidence"] = _sha((request_dir / "supplied_evidence.json").read_bytes())
    _write_json(request_dir / "evidence_ledger.json", evidence)
    _write_json(request_dir / "spec_proposal.json", proposal)
    gate = policy_gate(evidence, proposal)
    reference_id = proposal.get("catalyst_spec", {}).get("material", {}).get("reference_id")
    if gate.ready and reference_id:
        if reference is None:
            gate = GateDecision("needs_clarification", ("bulk reference resolution is required",))
        elif reference.status == "needs_clarification":
            candidates = ", ".join(reference.candidate_material_ids)
            gate = GateDecision("needs_clarification", (f"select one bulk reference: {candidates}",))
        elif reference.status != "resolved" or reference.record is None:
            gate = GateDecision("unsupported", ("no matching bulk reference was found",))
    if not gate.ready:
        extra_files: list[str] = []
        if gate.status == "needs_clarification":
            questions = (
                proposal.get("clarification_questions") or list(gate.errors) or
                ["Please resolve the structure-determining fields listed in reasons."]
            )
            clarification = {
                **_metadata(request_id, input_hashes),
                "schema_version": "clarification-request/v1",
                "questions": questions, "reasons": list(gate.errors) or questions,
            }
            validate_record("clarification_request", clarification)
            _write_json(request_dir / "clarification_request.json", clarification)
            extra_files.append("clarification_request.json")
        packet = {
            **_metadata(request_id, input_hashes),
            "schema_version": "review-packet/v1",
            "status": gate.status,
            "structure_summary": {},
            "source_backed_facts": [],
            "assumptions": [],
            "warnings": list(gate.errors),
            "files": [
                *(["supplied_evidence.json"] if "supplied_evidence" in input_hashes else []),
                "evidence_ledger.json", "spec_proposal.json", *extra_files,
            ],
            "reproduction": {"policy_gate": gate.status},
        }
        validate_record("review_packet", packet)
        _write_json(request_dir / "review_packet.json", packet)
        (request_dir / "review.md").write_text(_review_markdown(packet), encoding="utf-8")
        return CatalystPipelineResult(gate.status, request_dir, gate, tuple(request_dir.iterdir()))

    spec = proposal["catalyst_spec"]
    package_inputs = [
        *(["supplied_evidence.json"] if "supplied_evidence" in input_hashes else []),
        "evidence_ledger.json", "spec_proposal.json",
    ]
    if reference_id:
        assert reference is not None and reference.record is not None
        validate_record("reference_record", reference.record)
        _write_json(request_dir / "reference_record.json", reference.record)
        input_hashes["reference_record"] = _sha(reference.record)
        package_inputs.append("reference_record.json")
    dispatched = dispatch_spec(spec, request_id=request_id, reference=reference)
    atoms = dispatched.atoms
    recipe = dispatched.workspace.recipe()
    recipe_digest = dispatched.workspace.recipe_hash()
    requested = spec["requested_outputs"]
    bundle = write_bundle(
        atoms, request_dir, request="CatalystSpec reconstruction", recipe=recipe,
        recipe_hash=recipe_digest, formats=requested,
        tool_sequence=[step["tool"] for step in recipe["steps"]],
        controller_state="VALIDATED",
    )

    output_paths = [path.relative_to(request_dir).as_posix() for path in bundle.paths]
    build = {
        **_metadata(request_id, input_hashes),
        "schema_version": "build-record/v1",
        "registry_version": f"phase1-{dispatched.workspace.registry.fingerprint()[:16]}",
        "ase_version": ase.__version__,
        "approved_tool_calls": recipe["steps"],
        "input_hashes": input_hashes,
        "output_hashes": {"atoms": atoms_hash(atoms), **{
            path.relative_to(request_dir).as_posix(): _sha(path.read_bytes()) for path in bundle.paths
        }},
        "output_paths": output_paths,
    }
    validate_record("build_record", build)
    _write_json(request_dir / "build_record.json", build)

    geometry = validate_atoms(atoms, dispatched.workspace.policy, profile="final")
    rules = [{
        "rule": issue.code, "measured": issue.message, "threshold": "runtime policy",
        "severity": issue.severity, "passed": issue.severity != "error", "message": issue.message,
    } for issue in geometry.issues]
    rules.extend(catalyst_validation_rules(spec, dispatched.workspace))
    for path in bundle.paths[:-1]:
        try:
            restored = ase_read(path)
            passed = len(restored) == len(atoms) and restored.get_chemical_formula() == atoms.get_chemical_formula()
            message = "atom count and composition preserved" if passed else "round-trip changed atom count or composition"
        except Exception as exc:
            passed, message = False, str(exc)
        rules.append({
            "rule": f"export_round_trip:{path.name}", "measured": message,
            "threshold": "same atom count and composition", "severity": "error",
            "passed": passed, "message": message,
        })
    validation = {
        **_metadata(request_id, {**input_hashes, "build_record": _sha(build)}),
        "schema_version": "validation-report/v1", "profile": "final",
        "passed": geometry.ok and all(rule["passed"] for rule in rules),
        "rules": rules, "invariants": structure_invariants(atoms),
    }
    validate_record("validation_report", validation)
    _write_json(request_dir / "validation_report.json", validation)

    sources = proposal["field_sources"]
    source_backed_facts = []
    assumptions = []
    for source in sources:
        if source["evidence_type"] in {"reported", "user_supplied"}:
            claim = evidence["claims"][source["claim_index"]]
            source_backed_facts.append({
                "field": source["field"], "value": claim["value"],
                "evidence_type": source["evidence_type"],
                "source_id": claim["source_id"], "locator": claim["locator"],
                "claim_index": source["claim_index"],
            })
        elif source["evidence_type"] in {"assumed_default", "derived"}:
            assumptions.append({
                "field": source["field"], "value": _spec_value(spec, source["field"]),
                "evidence_type": source["evidence_type"], "reason": source["reason"],
            })
    packet = {
        **_metadata(request_id, {
            **input_hashes, "build_record": _sha(build), "validation_report": _sha(validation),
        }),
        "schema_version": "review-packet/v1",
        "status": "review_ready" if validation["passed"] else "failed",
        "structure_summary": validation["invariants"],
        "source_backed_facts": source_backed_facts,
        "assumptions": assumptions,
        "warnings": list(proposal["agent_warnings"]) + [
            rule["message"] for rule in rules if not rule["passed"] or rule["severity"] == "warning"
        ],
        "files": [
            *package_inputs, *output_paths,
            "build_record.json", "validation_report.json",
        ],
        "reproduction": {
            "catalyst_spec_schema": spec["schema_version"],
            "registry_version": build["registry_version"],
            "recipe_hash": recipe_digest,
            "atoms_hash": build["output_hashes"]["atoms"],
        },
    }
    validate_record("review_packet", packet)
    _write_json(request_dir / "review_packet.json", packet)
    (request_dir / "review.md").write_text(_review_markdown(packet), encoding="utf-8")
    files = tuple(sorted(request_dir.iterdir()))
    return CatalystPipelineResult(packet["status"], request_dir, gate, files)


def run_catalyst_pipeline(
    evidence: dict[str, Any],
    proposal: dict[str, Any],
    output_root: Path,
    *,
    reference: ReferenceResolution | None = None,
    _allow_existing_inputs: bool = False,
) -> CatalystPipelineResult:
    """Run the deterministic pipeline and retain a failure record after setup."""
    try:
        return _run_catalyst_pipeline(
            evidence, proposal, output_root, reference=reference,
            _allow_existing_inputs=_allow_existing_inputs,
        )
    except FileExistsError:
        raise
    except Exception as exc:
        request_id = str(proposal.get("request_id", ""))
        root = Path(output_root).resolve()
        request_dir = (root / request_id).resolve()
        if request_dir.parent != root or not request_dir.is_dir():
            raise
        return _failure_result(request_id, request_dir, "deterministic_pipeline", exc)


def run_journal_request(
    request_id: str,
    request: str,
    sources: list[dict[str, Any]],
    chat: ChatCallable,
    output_root: Path,
    *,
    reference: ReferenceResolution | None = None,
    model_location: Literal["local", "external"] = "local",
    authorize_private_external: bool = False,
) -> CatalystPipelineResult:
    """Run the complete two-agent workflow without an interactive pause."""
    request_dir = _prepare_request_dir(output_root, request_id)
    supplied = {
        **_metadata(request_id, {"request": _sha(request.encode())}),
        "schema_version": "supplied-evidence/v1", "request": request,
        "sources": sources, "model_location": model_location,
        "private_external_authorized": bool(authorize_private_external),
    }
    validate_record("supplied_evidence", supplied)
    _write_json(request_dir / "supplied_evidence.json", supplied)
    try:
        evidence = extract_evidence(
            request_id, request, sources, chat,
            model_location=model_location,
            authorize_private_external=authorize_private_external,
        )
        _write_json(request_dir / "evidence_ledger.json", evidence)
    except Exception as exc:
        return _failure_result(request_id, request_dir, "evidence_extraction", exc)
    try:
        proposal = plan_spec(request_id, request, evidence, chat)
        _write_json(request_dir / "spec_proposal.json", proposal)
    except Exception as exc:
        return _failure_result(request_id, request_dir, "spec_planning", exc)
    try:
        return run_catalyst_pipeline(
            evidence, proposal, output_root, reference=reference,
            _allow_existing_inputs=True,
        )
    except Exception as exc:
        return _failure_result(request_id, request_dir, "deterministic_pipeline", exc)


__all__ = ["CatalystPipelineResult", "run_catalyst_pipeline", "run_journal_request"]
