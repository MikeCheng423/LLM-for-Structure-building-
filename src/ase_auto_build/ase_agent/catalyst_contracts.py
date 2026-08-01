"""Versioned catalyst hand-off validation and the deterministic policy gate."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ase.data import atomic_numbers
from ase.formula import Formula

from .registry import SchemaValidationError, validate_schema


def _schema_dir() -> Path:
    installed = Path(sys.prefix) / "share" / "ase_auto_build" / "schemas"
    if installed.is_dir():
        return installed
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "schemas"
        if candidate.is_dir():
            return candidate
    raise RuntimeError("catalyst JSON schemas are not installed")


def load_schema(name: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z_]+", name):
        raise ValueError("invalid schema name")
    return json.loads((_schema_dir() / f"{name}.schema.json").read_text(encoding="utf-8"))


SCHEMA_NAMES = (
    "build_record", "candidate_set", "catalyst_spec", "clarification_request", "evidence_ledger",
    "failure_record",
    "reference_record", "relaxation_record", "review_packet", "spec_proposal",
    "supplied_evidence", "validation_report",
)


def validate_record(name: str, value: dict[str, Any]) -> dict[str, Any]:
    if name not in SCHEMA_NAMES:
        raise ValueError(f"unknown catalyst schema {name!r}")
    return validate_schema(value, load_schema(name), path=name)


@dataclass(frozen=True)
class GateDecision:
    status: Literal["ready", "needs_clarification", "unsupported"]
    errors: tuple[str, ...] = ()
    tool_family: str | None = None

    @property
    def ready(self) -> bool:
        return self.status == "ready"


def _field_paths(spec: dict[str, Any]) -> set[str]:
    paths: set[str] = set()

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            if prefix:
                paths.add(prefix)
            for key, item in value.items():
                visit(item, f"{prefix}.{key}" if prefix else key)
        elif isinstance(value, list):
            if prefix == "modifications":
                for index, item in enumerate(value):
                    visit(item, f"{prefix}[{index}]")
            else:
                paths.add(prefix)
        else:
            paths.add(prefix)

    visit(spec, "")
    return paths


def _field_value(spec: dict[str, Any], path: str) -> Any:
    value: Any = spec
    for key, index in re.findall(r"([A-Za-z_]+)(?:\[(\d+)\])?", path):
        value = value[key]
        if index:
            value = value[int(index)]
    return value


def _valid_element(value: Any) -> bool:
    return isinstance(value, str) and atomic_numbers.get(value, 0) > 0


def _required_fields(spec: dict[str, Any]) -> tuple[set[str], str | None]:
    material = spec.get("material", {})
    model = spec.get("model", {})
    kind = model.get("kind")
    common = {
        "material.formula", "model.kind", "model.supercell",
        "model.periodic_boundary_conditions", "model.center",
        "model.atom_ordering", "requested_outputs",
    }
    try:
        host_elements = set(Formula(str(material.get("formula", ""))).count())
    except (KeyError, ValueError):
        host_elements = set()
    if not host_elements or not all(_valid_element(element) for element in host_elements):
        return common, "material formula contains an invalid element"
    if material.get("reference_id"):
        common.add("material.reference_id")
    elif kind != "nanoparticle":
        common.add("material.crystal_structure")
    lattice = set(material.get("lattice_parameters_angstrom", {}))
    compound = len(re.findall(r"[A-Z][a-z]?", str(material.get("formula", "")))) > 1
    if material.get("reference_id") or kind == "nanoparticle":
        supported_lattice = set()
    elif kind == "bulk" or (kind == "surface" and compound):
        supported_lattice = {"a", "c"}
    else:
        supported_lattice = {"a"}
    ignored_lattice = sorted(lattice - supported_lattice)
    if ignored_lattice:
        return common, f"unregistered lattice controls: {ignored_lattice}"
    if kind == "surface":
        common |= {"model.miller_indices", "model.layers", "model.vacuum_angstrom"}
        # Multi-element surfaces can have inequivalent terminations.
        if len(re.findall(r"[A-Z][a-z]?", str(material.get("formula", "")))) > 1:
            common.add("model.termination")
            if model.get("termination") not in {None, "ase_default"}:
                return common, "requested compound termination is not registered"
    elif kind == "nanoparticle":
        common |= {"model.shape", "model.shells", "model.vacuum_angstrom"}
        if material.get("reference_id"):
            return common, "nanoparticle dispatch does not use a bulk reference"
        if len(re.findall(r"[A-Z][a-z]?", str(material.get("formula", "")))) != 1:
            return common, "nanoparticle dispatch requires one element"
        if model.get("supercell") not in (None, [1, 1, 1]):
            return common, "nanoparticle supercell must be [1, 1, 1]"
        if model.get("fixed_layers_from_bottom", 0):
            return common, "nanoparticle fixed layers are not registered"
    elif kind != "bulk":
        return common, f"unsupported model kind {kind!r}"
    elif (
        len(re.findall(r"[A-Z][a-z]?", str(material.get("formula", "")))) != 1
        and not material.get("reference_id")
    ):
        return common, "compound bulk CatalystSpec dispatch requires a reference resolver"

    expected_pbc = [False, False, False] if kind == "nanoparticle" else [True, True, kind == "bulk"]
    if model.get("periodic_boundary_conditions") not in (None, expected_pbc):
        return common, f"{kind} requires PBC {expected_pbc}"
    if model.get("center") not in {None, True} or model.get("atom_ordering") not in {None, "ase_default"}:
        return common, "requested centering or atom ordering is not registered"
    if kind != "surface" and model.get("fixed_layers_from_bottom", 0):
        return common, "fixed bottom layers require a surface model"

    cluster_count = 0
    for index, modification in enumerate(spec.get("modifications", [])):
        prefix = f"modifications[{index}]"
        operation = modification.get("operation")
        common.add(f"{prefix}.operation")
        if operation == "make_vacancy":
            common.add(f"{prefix}.selector")
        elif operation == "substitute":
            common |= {f"{prefix}.selector", f"{prefix}.element"}
            if not _valid_element(modification.get("element")):
                return common, "substitution contains an invalid element"
        elif operation == "add_adsorbate":
            controls = sorted(set(modification) & {"coverage_monolayer", "orientation"})
            if controls:
                return common, f"requested adsorbate controls are not registered: {controls}"
            common |= {f"{prefix}.site", f"{prefix}.height_angstrom"}
            common.add(f"{prefix}.species" if "species" in modification else f"{prefix}.element")
            if "element" in modification and not _valid_element(modification["element"]):
                return common, "adsorbate contains an invalid element"
            if "species" in modification and len(re.findall(r"[A-Z][a-z]?", modification["species"])) > 1:
                common.add(f"{prefix}.anchor")
        elif operation == "add_supported_cluster":
            cluster_count += 1
            if cluster_count > 1:
                return common, "only one supported cluster is registered per candidate"
            if kind != "surface":
                return common, "supported clusters require a surface host"
            common |= {
                f"{prefix}.element", f"{prefix}.shape", f"{prefix}.shells",
                f"{prefix}.gap_angstrom", f"{prefix}.vacuum_angstrom",
            }
            if not _valid_element(modification.get("element")):
                return common, "supported cluster contains an invalid element"
        else:
            return common, f"unsupported modification {operation!r}"
    for field in _field_paths(spec):
        if not field.startswith(("material.", "model.", "modifications[")):
            continue
        if ".selector." in field or isinstance(_field_value(spec, field), dict):
            continue
        common.add(field)
    return common, None


def policy_gate(evidence: dict[str, Any], proposal: dict[str, Any]) -> GateDecision:
    """Fail closed before retrieval or ASE execution."""
    try:
        validate_record("evidence_ledger", evidence)
        validate_record("spec_proposal", proposal)
        spec = validate_record("catalyst_spec", proposal["catalyst_spec"])
    except (SchemaValidationError, KeyError, TypeError, ValueError) as exc:
        return GateDecision("needs_clarification", (f"schema: {exc}",))

    if evidence["request_id"] != proposal["request_id"]:
        return GateDecision("unsupported", ("request_id mismatch",))
    if proposal["task_status"] != spec["task_status"]:
        return GateDecision("unsupported", ("proposal/spec task_status mismatch",))
    if proposal["task_status"] != "ready":
        if proposal["task_status"] == "needs_clarification" and not proposal["clarification_questions"]:
            return GateDecision("needs_clarification", ("clarification status has no question",))
        return GateDecision(proposal["task_status"], tuple(proposal["clarification_questions"]))

    errors: list[str] = []
    if evidence["contradictions"]:
        errors.append("supplied evidence contains unresolved contradictions")
    if proposal["clarification_questions"]:
        errors.append("proposal still contains clarification questions")
    claims = evidence["claims"]
    sources: dict[str, dict[str, Any]] = {}
    for source in proposal["field_sources"]:
        field = source["field"]
        if field in sources:
            errors.append(f"duplicate field source: {field}")
            continue
        sources[field] = source
        if source["evidence_type"] in {"reported", "user_supplied"}:
            index = source.get("claim_index")
            if not isinstance(index, int) or index >= len(claims):
                errors.append(f"source-backed field has no EvidenceLedger claim: {field}")
            elif claims[index]["field"] != field or claims[index]["evidence_type"] != source["evidence_type"]:
                errors.append(f"source-backed field links the wrong claim: {field}")
            elif field in _field_paths(spec) and claims[index]["value"] != _field_value(spec, field):
                errors.append(f"source-backed field value differs from its claim: {field}")
        elif source["evidence_type"] in {"assumed_default", "derived"} and not source.get("reason"):
            errors.append(f"{source['evidence_type']} field has no reason: {field}")

    required, unsupported = _required_fields(spec)
    if unsupported:
        return GateDecision("unsupported", (unsupported,))
    present = _field_paths(spec)
    spec_sources: dict[str, str] = {}
    for item in spec["provenance"]:
        field = item["field"]
        if field in spec_sources:
            errors.append(f"duplicate CatalystSpec provenance: {field}")
            continue
        spec_sources[field] = item["evidence_type"]
        if field not in present:
            errors.append(f"CatalystSpec provenance references an absent field: {field}")
        elif item["value"] != _field_value(spec, field):
            errors.append(f"CatalystSpec provenance value differs from its field: {field}")
    errors.extend(
        f"CatalystSpec provenance disagrees with field source: {field}"
        for field, source in sources.items()
        if spec_sources.get(field) != source["evidence_type"]
    )
    errors.extend(f"missing structure field: {field}" for field in sorted(required - present))
    errors.extend(f"missing field provenance: {field}" for field in sorted(required - sources.keys()))
    errors.extend(
        f"ambiguous structure field: {field}"
        for field in sorted(required)
        if sources.get(field, {}).get("evidence_type") == "ambiguous"
    )
    unresolved = set(evidence["unresolved_fields"])
    for index, modification in enumerate(spec["modifications"]):
        if modification["operation"] in {"make_vacancy", "substitute"} and not modification.get("selector"):
            errors.append(f"empty atom selector: modifications[{index}].selector")
    errors.extend(f"unresolved structure field: {field}" for field in sorted(required & unresolved))
    if errors:
        return GateDecision("needs_clarification", tuple(errors))
    return GateDecision("ready", tool_family=str(spec["model"]["kind"]))


__all__ = ["GateDecision", "SCHEMA_NAMES", "load_schema", "policy_gate", "validate_record"]
