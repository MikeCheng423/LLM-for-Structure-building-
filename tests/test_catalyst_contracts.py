from __future__ import annotations

from copy import deepcopy

import pytest
from ase.build import bulk

from ase_auto_build.ase_agent.catalyst_contracts import policy_gate, validate_record
from ase_auto_build.ase_agent.catalyst_dispatch import CatalystDispatchError, dispatch_spec
from ase_auto_build.ase_agent.catalyst_pipeline import run_catalyst_pipeline
from ase_auto_build.ase_agent.catalyst_validation import catalyst_validation_rules
from ase_auto_build.ase_agent.mp_resolver import ReferenceResolution, resolve_reference


def records():
    metadata = {
        "request_id": "request-1",
        "created_at": "2026-07-28T00:00:00Z",
        "producer_version": "test",
        "artifact_hashes": {},
    }
    fields = {
        "material.formula": "Pt",
        "material.crystal_structure": "fcc",
        "model.kind": "bulk",
        "model.supercell": [2, 2, 2],
        "model.periodic_boundary_conditions": [True, True, True],
        "model.center": True,
        "model.atom_ordering": "ase_default",
        "requested_outputs": ["cif", "vasp"],
    }
    evidence = {
        **metadata,
        "schema_version": "evidence-ledger/v1",
        "claims": [{
            "field": field,
            "value": value,
            "evidence_type": "user_supplied",
            "source_id": "user-evidence-1",
            "locator": "request",
            "confidence": "high",
        } for field, value in fields.items()],
        "contradictions": [],
        "unresolved_fields": [],
    }
    spec = {
        "schema_version": "0.1",
        "task_status": "ready",
        "material": {"formula": "Pt", "crystal_structure": "fcc"},
        "model": {
            "kind": "bulk",
            "supercell": [2, 2, 2],
            "periodic_boundary_conditions": [True, True, True],
            "center": True,
            "atom_ordering": "ase_default",
        },
        "modifications": [],
        "provenance": [{"field": field, "value": value, "evidence_type": "user_supplied"} for field, value in fields.items()],
        "clarification_questions": [],
        "requested_outputs": ["cif", "vasp"],
    }
    proposal = {
        **metadata,
        "schema_version": "spec-proposal/v1",
        "task_status": "ready",
        "catalyst_spec": spec,
        "field_sources": [
            {"field": field, "evidence_type": "user_supplied", "claim_index": index}
            for index, field in enumerate(fields)
        ],
        "clarification_questions": [],
        "agent_warnings": [],
    }
    return evidence, proposal


def source_value(evidence, proposal, field, value) -> None:
    for source in proposal["field_sources"]:
        if source["field"] == field:
            evidence["claims"][source["claim_index"]]["value"] = value
            next(item for item in proposal["catalyst_spec"]["provenance"] if item["field"] == field)["value"] = value
            return
    evidence["claims"].append({
        "field": field, "value": value, "evidence_type": "user_supplied",
        "source_id": "user-evidence-1", "locator": "request", "confidence": "high",
    })
    proposal["field_sources"].append({
        "field": field, "evidence_type": "user_supplied",
        "claim_index": len(evidence["claims"]) - 1,
    })
    proposal["catalyst_spec"]["provenance"].append({
        "field": field, "value": value, "evidence_type": "user_supplied",
    })


def test_policy_gate_accepts_fully_sourced_bulk_spec() -> None:
    evidence, proposal = records()
    assert validate_record("evidence_ledger", evidence)["request_id"] == "request-1"
    decision = policy_gate(evidence, proposal)
    assert decision.ready
    assert decision.tool_family == "bulk"


def test_policy_gate_rejects_unlinked_reported_value() -> None:
    evidence, proposal = records()
    proposal["field_sources"][0] = {
        "field": "material.formula", "evidence_type": "reported", "claim_index": 0
    }
    decision = policy_gate(evidence, proposal)
    assert not decision.ready
    assert any("wrong claim" in error for error in decision.errors)


def test_policy_gate_requires_surface_topology() -> None:
    evidence, proposal = records()
    proposal = deepcopy(proposal)
    proposal["catalyst_spec"]["model"]["kind"] = "surface"
    proposal["catalyst_spec"]["model"]["periodic_boundary_conditions"] = [True, True, False]
    decision = policy_gate(evidence, proposal)
    assert decision.status == "needs_clarification"
    assert any("model.miller_indices" in error for error in decision.errors)


def test_policy_gate_and_dispatch_support_simple_nanoparticle() -> None:
    evidence, proposal = records()
    model = proposal["catalyst_spec"]["model"]
    model.update({
        "kind": "nanoparticle", "supercell": [1, 1, 1],
        "periodic_boundary_conditions": [False, False, False],
        "shape": "icosahedron", "shells": 2, "vacuum_angstrom": 8.0,
    })
    for field, value in {
        "model.kind": "nanoparticle", "model.supercell": [1, 1, 1],
        "model.periodic_boundary_conditions": [False, False, False],
        "model.shape": "icosahedron", "model.shells": 2,
        "model.vacuum_angstrom": 8.0,
    }.items():
        source_value(evidence, proposal, field, value)
    decision = policy_gate(evidence, proposal)
    assert decision.ready
    result = dispatch_spec(proposal["catalyst_spec"], request_id="request-1")
    assert len(result.atoms) == 13
    assert list(result.atoms.pbc) == [False, False, False]
    rules = catalyst_validation_rules(proposal["catalyst_spec"], result.workspace)
    assert all(rule["passed"] for rule in rules)
    assert {rule["rule"] for rule in rules} >= {
        "nanoparticle_vacuum_angstrom", "nanoparticle_centering",
    }


def test_approved_spec_dispatches_without_model_authored_code() -> None:
    evidence, proposal = records()
    assert policy_gate(evidence, proposal).ready
    result = dispatch_spec(proposal["catalyst_spec"], request_id="request-1")
    assert len(result.atoms) == 32
    assert result.workspace.recipe()["steps"][-1] == {"tool": "finish", "args": {"name": "candidate"}}


def test_dispatch_rejects_unregistered_control() -> None:
    _, proposal = records()
    proposal["catalyst_spec"]["model"]["center"] = False
    assert policy_gate(records()[0], proposal).status == "unsupported"
    try:
        dispatch_spec(proposal["catalyst_spec"], request_id="request-1")
    except CatalystDispatchError as exc:
        assert "centered" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unregistered control was accepted")


def test_policy_gate_requires_provenance_for_optional_structure_values() -> None:
    evidence, proposal = records()
    proposal["catalyst_spec"]["material"]["lattice_parameters_angstrom"] = {"a": 3.92}
    decision = policy_gate(evidence, proposal)
    assert decision.status == "needs_clarification"
    assert "missing field provenance: material.lattice_parameters_angstrom.a" in decision.errors


def test_noninteractive_pipeline_writes_auditable_request_package(tmp_path) -> None:
    evidence, proposal = records()
    result = run_catalyst_pipeline(evidence, proposal, tmp_path)
    assert result.status == "review_ready"
    assert result.request_dir.parent == tmp_path.resolve()
    assert {path.name for path in result.files} >= {
        "POSCAR", "structure.cif", "structure.json", "evidence_ledger.json",
        "spec_proposal.json", "build_record.json", "validation_report.json",
        "review_packet.json", "review.md",
    }
    import json
    validation = json.loads((result.request_dir / "validation_report.json").read_text())
    assert validation["passed"] is True
    assert all(rule["passed"] for rule in validation["rules"])
    packet = json.loads((result.request_dir / "review_packet.json").read_text())
    formula = next(item for item in packet["source_backed_facts"] if item["field"] == "material.formula")
    assert formula == {
        "field": "material.formula", "value": "Pt",
        "evidence_type": "user_supplied", "source_id": "user-evidence-1",
        "locator": "request", "claim_index": 0,
    }
    review = (result.request_dir / "review.md").read_text()
    assert "## Source-backed facts" in review
    assert "## Reproduction" in review

    try:
        run_catalyst_pipeline(evidence, proposal, tmp_path)
    except FileExistsError:
        pass
    else:  # pragma: no cover
        raise AssertionError("immutable request package was overwritten")


def test_contradiction_produces_clarification_without_structure(tmp_path) -> None:
    evidence, proposal = records()
    evidence["contradictions"] = [{"fields": ["material.formula"], "reason": "Pt versus Pd"}]
    result = run_catalyst_pipeline(evidence, proposal, tmp_path)
    assert result.status == "needs_clarification"
    assert (result.request_dir / "clarification_request.json").is_file()
    assert not (result.request_dir / "POSCAR").exists()


def test_deterministic_build_failure_leaves_failure_record(tmp_path) -> None:
    evidence, proposal = records()
    proposal["catalyst_spec"]["model"]["supercell"] = [12, 12, 12]
    source_value(evidence, proposal, "model.supercell", [12, 12, 12])
    result = run_catalyst_pipeline(evidence, proposal, tmp_path)
    assert result.status == "failed"
    assert (result.request_dir / "failure_record.json").is_file()
    assert (result.request_dir / "review_packet.json").is_file()


def test_resolved_bulk_reference_is_recorded_and_dispatched(tmp_path) -> None:
    evidence, proposal = records()
    field = "material.reference_id"
    value = "mp-1"
    proposal["catalyst_spec"]["material"]["reference_id"] = value
    proposal["catalyst_spec"]["provenance"].append({
        "field": field, "value": value, "evidence_type": "user_supplied",
    })
    evidence["claims"].append({
        "field": field, "value": value, "evidence_type": "user_supplied",
        "source_id": "user-evidence-1", "locator": "request", "confidence": "high",
    })
    proposal["field_sources"].append({
        "field": field, "evidence_type": "user_supplied",
        "claim_index": len(evidence["claims"]) - 1,
    })
    row = {
        "material_id": value, "formula_pretty": "Pt", "elements": ["Pt"],
        "is_stable": True, "run_type": "GGA", "structure": bulk("Pt", "fcc", cubic=True),
    }
    reference = resolve_reference("request-1", {"material_id": value}, lambda query: [row])

    result = run_catalyst_pipeline(evidence, proposal, tmp_path, reference=reference)
    assert result.status == "review_ready"
    assert (result.request_dir / "reference_record.json").is_file()
    import json
    build = json.loads((result.request_dir / "build_record.json").read_text())
    assert build["approved_tool_calls"][0]["tool"] == "load_reference_bulk"


def test_missing_bulk_reference_stops_before_build(tmp_path) -> None:
    evidence, proposal = records()
    field = "material.reference_id"
    proposal["catalyst_spec"]["material"]["reference_id"] = "mp-1"
    proposal["catalyst_spec"]["provenance"].append({
        "field": field, "value": "mp-1", "evidence_type": "assumed_default",
    })
    proposal["field_sources"].append({
        "field": field, "evidence_type": "assumed_default", "reason": "resolver required",
    })
    result = run_catalyst_pipeline(evidence, proposal, tmp_path)
    assert result.status == "needs_clarification"
    assert not (result.request_dir / "build_record.json").exists()


def test_dispatch_rejects_reference_structure_hash_mismatch() -> None:
    evidence, proposal = records()
    proposal["catalyst_spec"]["material"]["reference_id"] = "mp-1"
    row = {
        "material_id": "mp-1", "formula_pretty": "Pt", "elements": ["Pt"],
        "is_stable": True, "run_type": "GGA", "structure": bulk("Pt", "fcc", cubic=True),
    }
    reference = resolve_reference("request-1", {"material_id": "mp-1"}, lambda query: [row])
    changed = reference.structure.copy()
    changed.positions[0, 0] += 0.1
    tampered = ReferenceResolution(
        reference.status, reference.candidate_material_ids, reference.record, changed,
    )
    with pytest.raises(CatalystDispatchError, match="hash"):
        dispatch_spec(proposal["catalyst_spec"], request_id="request-1", reference=tampered)


def test_supported_cluster_pipeline_uses_composed_registered_tools(tmp_path) -> None:
    evidence, proposal = records()
    model = proposal["catalyst_spec"]["model"]
    model.update({
        "kind": "surface", "supercell": [3, 3, 1],
        "periodic_boundary_conditions": [True, True, False],
        "miller_indices": [1, 1, 1], "layers": 4, "vacuum_angstrom": 14.0,
    })
    modification = {
        "operation": "add_supported_cluster", "element": "Pt",
        "shape": "icosahedron", "shells": 2,
        "gap_angstrom": 2.2, "vacuum_angstrom": 12.0,
    }
    proposal["catalyst_spec"]["modifications"] = [modification]
    fields = {
        "model.kind": "surface", "model.supercell": [3, 3, 1],
        "model.periodic_boundary_conditions": [True, True, False],
        "model.miller_indices": [1, 1, 1], "model.layers": 4,
        "model.vacuum_angstrom": 14.0,
        **{f"modifications[0].{key}": value for key, value in modification.items()},
    }
    for field, value in fields.items():
        source_value(evidence, proposal, field, value)

    result = run_catalyst_pipeline(evidence, proposal, tmp_path)
    assert result.status == "review_ready"
    import json
    build = json.loads((result.request_dir / "build_record.json").read_text())
    assert [step["tool"] for step in build["approved_tool_calls"]] == [
        "build_surface", "build_nanoparticle", "combine", "finish",
    ]
    validation = json.loads((result.request_dir / "validation_report.json").read_text())
    gap = next(rule for rule in validation["rules"] if "interface_gap" in rule["rule"])
    assert gap["passed"] is True
    assert gap["measured"] == 2.2
