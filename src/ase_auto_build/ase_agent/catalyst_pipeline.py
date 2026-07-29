"""Non-interactive policy → build → validate → export → review pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import ase
import numpy as np
from ase.geometry import find_mic
from ase.io import read as ase_read

from .catalyst_contracts import GateDecision, policy_gate, validate_record
from .catalyst_agents import ChatCallable, extract_evidence, plan_spec
from .catalyst_dispatch import dispatch_spec
from .catalyst_validation import catalyst_validation_rules
from .export import _extension_for, write_bundle
from .mp_resolver import ReferenceResolution, resolve_reference
from .validation import _constrained_count, atoms_hash, structure_invariants, validate_atoms

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


#: Formats that carry selective dynamics / constraints through a write-read cycle.
_CONSTRAINT_AWARE = {"POSCAR", ".traj"}
_POSITION_TOLERANCE = 1e-2
_CELL_TOLERANCE = 1e-2


def _missing_formats(requested: list[str], produced: set[str]) -> list[str]:
    """Requested output formats that did not actually produce a file."""
    missing = []
    for fmt in requested:
        name = str(fmt).strip().lower()
        if name in {"vasp", "poscar"}:
            if "POSCAR" not in produced:
                missing.append(name)
        elif f"structure.{_extension_for(name)}" not in produced:
            missing.append(name)
    return missing


def _round_trip_mismatch(path: Path, atoms) -> str | None:
    """Return the first way `path` fails to reproduce `atoms`, or None."""
    restored = ase_read(path)
    if len(restored) != len(atoms):
        return f"atom count changed {len(atoms)} -> {len(restored)}"
    if restored.get_chemical_symbols() != atoms.get_chemical_symbols():
        return "chemical symbols or their order changed"
    if len(atoms):
        delta = restored.get_positions() - atoms.get_positions()
        if np.any(atoms.pbc) and atoms.cell.rank == 3:
            delta, _ = find_mic(delta, atoms.cell, atoms.pbc)
        drift = float(np.max(np.linalg.norm(delta, axis=1)))
        if drift > _POSITION_TOLERANCE:
            return f"positions drifted by {drift:.4f} A"
    if restored.cell.rank == 3 and atoms.cell.rank == 3:
        for measured, expected, label in (
            (restored.cell.lengths(), atoms.cell.lengths(), "lengths"),
            (restored.cell.angles(), atoms.cell.angles(), "angles"),
        ):
            if float(np.max(np.abs(np.asarray(measured) - np.asarray(expected)))) > _CELL_TOLERANCE:
                return f"cell {label} changed"
    if path.name in _CONSTRAINT_AWARE or path.suffix in _CONSTRAINT_AWARE:
        if _constrained_count(restored) != _constrained_count(atoms):
            return (
                f"constrained atom count changed "
                f"{_constrained_count(atoms)} -> {_constrained_count(restored)}"
            )
    return None


def _model_summary(spec: dict[str, Any]) -> dict[str, Any]:
    """Describe the surface and its adsorbates, as section 11 requires."""
    model = spec["model"]
    summary: dict[str, Any] = {
        "kind": model["kind"],
        "formula": spec["material"]["formula"],
        "crystal_structure": spec["material"].get("crystal_structure"),
        "supercell": model.get("supercell"),
    }
    if model["kind"] == "surface":
        summary.update({
            "miller_indices": model.get("miller_indices"),
            "layers": model.get("layers"),
            "vacuum_angstrom": model.get("vacuum_angstrom"),
            "termination": model.get("termination", "ase_default"),
            "fixed_layers_from_bottom": model.get("fixed_layers_from_bottom", 0),
        })
    elif model["kind"] == "nanoparticle":
        summary.update({"shape": model.get("shape"), "shells": model.get("shells")})
    adsorbates, defects, clusters = [], [], []
    for modification in spec["modifications"]:
        operation = modification["operation"]
        if operation == "add_adsorbate":
            adsorbates.append({
                "species": modification.get("species") or modification["element"],
                "site": modification["site"], "site_index": modification.get("site_index", 1),
                "height_angstrom": modification["height_angstrom"],
                "anchor": modification.get("anchor"),
            })
        elif operation == "add_supported_cluster":
            clusters.append({
                "element": modification["element"], "shape": modification["shape"],
                "shells": modification["shells"], "gap_angstrom": modification["gap_angstrom"],
            })
        else:
            defects.append({
                "operation": operation, "element": modification.get("element"),
                "selector": modification.get("selector"),
            })
    summary["adsorbates"] = adsorbates
    summary["defects"] = defects
    summary["supported_clusters"] = clusters
    return summary


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
    model_info: dict[str, Any] | None = None,
    paper: dict[str, Any] | None = None,
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
        model_info=dict(model_info or {}),
        # Truthful at export time; validation has not run yet. The final state
        # lives in validation_report.passed and review_packet.status.
        controller_state="BUILT",
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
            mismatch = _round_trip_mismatch(path, atoms)
        except Exception as exc:
            mismatch = f"{type(exc).__name__}: {exc}"
        rules.append({
            "rule": f"export_round_trip:{path.name}",
            "measured": mismatch or "symbols, positions, cell and constraints preserved",
            "threshold": (
                f"positions within {_POSITION_TOLERANCE} A and cell within {_CELL_TOLERANCE}"
            ),
            "severity": "error", "passed": mismatch is None,
            "message": mismatch or f"{path.name} round-trips to the built structure",
        })
    produced = {path.name for path in bundle.paths}
    missing_formats = sorted(_missing_formats(requested, produced))
    rules.append({
        "rule": "requested_output_formats", "measured": sorted(produced),
        "threshold": sorted(requested), "severity": "error",
        "passed": not missing_formats,
        "message": (
            f"requested formats without a file: {missing_formats}" if missing_formats
            else "every requested output format produced a file"
        ),
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
                **({"doi": paper["doi"]} if paper and paper.get("doi") else {}),
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
        "structure_summary": {**validation["invariants"], "model": _model_summary(spec)},
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
            **({"paper": paper} if paper else {}),
            "catalyst_spec_schema": spec["schema_version"],
            "registry_version": build["registry_version"],
            "recipe_hash": recipe_digest,
            "atoms_hash": build["output_hashes"]["atoms"],
            "producer_version": PRODUCER_VERSION,
            "ase_version": ase.__version__,
            "model": dict(model_info or {}),
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
    model_info: dict[str, Any] | None = None,
    paper: dict[str, Any] | None = None,
    _allow_existing_inputs: bool = False,
) -> CatalystPipelineResult:
    """Run the deterministic pipeline and retain a failure record after setup."""
    try:
        return _run_catalyst_pipeline(
            evidence, proposal, output_root, reference=reference,
            model_info=model_info, paper=paper,
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


@dataclass(frozen=True)
class CandidateSetResult:
    """One package per scientifically reasonable reconstruction."""
    record: dict[str, Any]
    results: tuple[CatalystPipelineResult, ...]


def _candidate_proposal(
    proposal: dict[str, Any], material_id: str, index: int, total: int, request_id: str
) -> dict[str, Any]:
    """Point one proposal at one candidate parent, labelled `derived`.

    Section 5 forbids presenting a system choice as reported, so the selected
    material id carries a reason naming its position in the candidate set.
    """
    spec = deepcopy(proposal["catalyst_spec"])
    spec["material"]["reference_id"] = material_id
    reason = (
        f"Ambiguous bulk parent: candidate {index} of {total} ({material_id}). "
        "Each candidate is built separately rather than one being chosen silently."
    )
    entry = {
        "field": "material.reference_id", "value": material_id,
        "evidence_type": "derived", "reason": reason,
    }
    spec["provenance"] = [
        item for item in spec["provenance"] if item["field"] != "material.reference_id"
    ] + [entry]
    sources = [
        item for item in proposal["field_sources"] if item["field"] != "material.reference_id"
    ] + [{"field": "material.reference_id", "evidence_type": "derived", "reason": reason}]
    return {
        **proposal, "request_id": request_id, "catalyst_spec": spec, "field_sources": sources,
    }


def run_candidate_set(
    evidence: dict[str, Any],
    proposal: dict[str, Any],
    output_root: Path,
    *,
    query: dict[str, Any],
    fetch: Callable[[dict[str, Any]], Any],
    model_info: dict[str, Any] | None = None,
    paper: dict[str, Any] | None = None,
) -> CandidateSetResult:
    """Build every meaningful bulk parent when the resolver cannot choose one.

    CATALYST_STRUCTURE_LLM.md section 7: when several scientifically reasonable
    reconstructions exist, generate separately named candidates rather than
    silently selecting one.
    """
    base_request_id = str(proposal["request_id"])
    probe = resolve_reference(base_request_id, dict(query), fetch)
    if probe.status != "needs_clarification":
        raise ValueError("candidate sets apply only to an unresolved bulk parent")
    material_ids = list(probe.candidate_material_ids)

    results: list[CatalystPipelineResult] = []
    entries: list[dict[str, Any]] = []
    for index, material_id in enumerate(material_ids, start=1):
        request_id = f"{base_request_id}-{material_id}"
        if _SAFE_ID.fullmatch(request_id) is None:
            raise ValueError(f"candidate request_id is not path-free: {request_id}")
        resolution = resolve_reference(request_id, {**query, "material_id": material_id}, fetch)
        if resolution.status != "resolved":
            raise ValueError(f"candidate {material_id} did not resolve to one entry")
        candidate_evidence = {**evidence, "request_id": request_id}
        candidate_proposal = _candidate_proposal(
            proposal, material_id, index, len(material_ids), request_id
        )
        result = run_catalyst_pipeline(
            candidate_evidence, candidate_proposal, output_root,
            reference=resolution, model_info=model_info, paper=paper,
        )
        results.append(result)
        entry = {
            "candidate_id": f"candidate-{index}", "request_id": request_id,
            "value": material_id, "status": result.status,
            "directory": result.request_dir.name,
        }
        build_record = result.request_dir / "build_record.json"
        if build_record.is_file():
            entry["atoms_hash"] = json.loads(
                build_record.read_text(encoding="utf-8")
            )["output_hashes"]["atoms"]
        entries.append(entry)

    record = {
        **_metadata(base_request_id, {"spec_proposal": _sha(proposal)}),
        "schema_version": "candidate-set/v1",
        "ambiguous_field": "material.reference_id",
        "reason": (
            f"The bulk reference query matched {len(material_ids)} meaningful entries; "
            "each is built as a separate named candidate."
        ),
        "candidates": entries,
    }
    validate_record("candidate_set", record)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / f"{base_request_id}-candidate_set.json", record)
    return CandidateSetResult(record, tuple(results))


def run_journal_request(
    request_id: str,
    request: str,
    sources: list[dict[str, Any]],
    chat: ChatCallable,
    output_root: Path,
    *,
    reference: ReferenceResolution | None = None,
    model_info: dict[str, Any] | None = None,
    paper: dict[str, Any] | None = None,
    model_location: Literal["local", "external"] = "local",
    authorize_private_external: bool = False,
) -> CatalystPipelineResult:
    """Run the complete two-agent workflow without an interactive pause."""
    request_dir = _prepare_request_dir(output_root, request_id)
    supplied = {
        **_metadata(request_id, {"request": _sha(request.encode())}),
        "schema_version": "supplied-evidence/v1", "request": request,
        "sources": sources, "model_location": model_location,
        **({"paper": paper} if paper else {}),
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
            model_info=model_info, paper=paper, _allow_existing_inputs=True,
        )
    except Exception as exc:
        return _failure_result(request_id, request_dir, "deterministic_pipeline", exc)


__all__ = [
    "CandidateSetResult", "CatalystPipelineResult", "run_candidate_set",
    "run_catalyst_pipeline", "run_journal_request",
]
