#!/usr/bin/env python3
"""Generate executed, replay-verified ASE tool trajectories."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ase_auto_build.ase_agent import ASEWorkspace, create_default_registry
from ase_auto_build.ase_agent.controller import SYSTEM_PROMPT
from ase_auto_build.ase_agent.tool_router import route_tools
from ase_auto_build.ase_agent.validation import atoms_hash

from training.dataset_contract import SCHEMA_VERSION, validate_record
from training.generators.request_rule import missing_slots, slot_values
from training.generators.template_fill import fill_templates


SYSTEM_MESSAGE = SYSTEM_PROMPT

# Regions whose prompts are hand-authored and not templated (clarification
# deliberately withholds a slot; error_recovery has no required slots).
CANONICAL_ONLY = {"clarification", "error_recovery"}
MAX_PROMPTS_PER_CASE = 6


def load_templates(templates_dir: Path) -> dict[str, list[str]]:
    """Load agent-authored `<family>.json` template lists, if present."""
    templates: dict[str, list[str]] = {}
    if not templates_dir.is_dir():
        return templates
    for path in sorted(templates_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise RuntimeError(f"{path}: template file must be a JSON list of strings")
        templates[path.stem] = data
    return templates


def _routable(prompt: str, recipe_tools: set[str], registry) -> bool:
    """A prompt is usable only if the bounded router exposes all its recipe's tools."""
    try:
        routed = {schema["function"]["name"] for schema in route_tools(prompt, registry)}
    except Exception:
        return False
    return recipe_tools <= routed


def case_prompts(case: RecipeCase, templates: dict[str, list[str]], registry) -> list[str]:
    """Rule-conforming, router-deployable prompts: filled agent templates, else canonical."""
    steps = [{"tool": s["tool"], "args": s["args"]} for s in case.steps]
    recipe_tools = {s["tool"] for s in case.steps}
    if case.family in CANONICAL_ONLY or case.family not in templates:
        candidates = list(case.descriptions)
    else:
        filled = fill_templates(case.family, templates[case.family], steps)
        if filled:
            candidates = filled
        else:
            values = slot_values(steps)
            candidates = [d for d in case.descriptions if not missing_slots(case.family, values, d)]
    routable = [prompt for prompt in candidates if _routable(prompt, recipe_tools, registry)]
    if not routable:
        raise RuntimeError(
            f"{case.case_id}: no rule-conforming, router-deployable prompt available"
        )
    return routable[:MAX_PROMPTS_PER_CASE]


@dataclass(frozen=True)
class RecipeCase:
    case_id: str
    family: str
    composition: str
    descriptions: tuple[str, ...]
    steps: tuple[dict[str, Any], ...]

    @property
    def split_group(self) -> str:
        return f"{self.family}:{self.composition}:{self.case_id}"


def _step(
    tool: str,
    *,
    expect_error: bool = False,
    followup_user: str | None = None,
    **args: Any,
) -> dict[str, Any]:
    step = {"tool": tool, "args": args}
    if expect_error:
        step["expect_error"] = True
    if followup_user is not None:
        step["followup_user"] = followup_user
    return step


def _bulk_cases() -> Iterable[RecipeCase]:
    materials = (
        ("Al", "fcc"), ("Cu", "fcc"), ("Ni", "fcc"), ("Pt", "fcc"),
        ("Au", "fcc"), ("Fe", "bcc"), ("W", "bcc"), ("Si", "diamond"),
    )
    repeats = ((1, 1, 1), (2, 2, 1), (2, 2, 2))
    for (element, crystal_name), repeat in itertools.product(materials, repeats):
        suffix = "x".join(str(value) for value in repeat)
        base = (
            _step("build_bulk", name="bulk", element=element, crystal=crystal_name, cubic=True),
            *(() if repeat == (1, 1, 1) else (_step("repeat", name="bulk", repeat=list(repeat)),)),
        )
        yield RecipeCase(
            f"{element.lower()}-{crystal_name}-{suffix}", "bulk", element,
            (
                f"Build a {suffix} {crystal_name} {element} bulk crystal.",
                f"Create crystalline {element} in the {crystal_name} phase and repeat it {suffix}.",
                f"I need a {element} {crystal_name} bulk supercell with repetitions {list(repeat)}.",
            ),
            (*base, _step("finish", name="bulk")),
        )
        if repeat != (1, 1, 1):
            yield RecipeCase(
                f"{element.lower()}-{crystal_name}-{suffix}-vacancy", "vacancy", element,
                (
                    f"Make a {suffix} {crystal_name} {element} cell and remove atom 1.",
                    f"Create a one-vacancy {element} supercell in the {crystal_name} phase; delete the first site.",
                    f"Build {element} {crystal_name}, repeat {list(repeat)}, then introduce a vacancy at index 1.",
                ),
                (*base, _step("make_vacancy", name="bulk", selector={"indices": [1]}), _step("finish", name="bulk")),
            )
            dopant = "Au" if element != "Au" else "Cu"
            yield RecipeCase(
                f"{element.lower()}-{crystal_name}-{suffix}-{dopant.lower()}", "substitution", f"{element}-{dopant}",
                (
                    f"Build {suffix} {crystal_name} {element} and replace atom 1 with {dopant}.",
                    f"Create a substitutional {dopant} defect on site 1 of a {element} {suffix} supercell.",
                    f"Prepare {element} {crystal_name} repeated {list(repeat)} with the first atom changed to {dopant}.",
                ),
                (*base, _step("substitute", name="bulk", selector={"indices": [1]}, element=dopant), _step("finish", name="bulk")),
            )


def _surface_cases() -> Iterable[RecipeCase]:
    materials = (("Pt", "fcc"), ("Cu", "fcc"), ("Ni", "fcc"), ("Fe", "bcc"))
    millers = ((1, 1, 1), (1, 0, 0), (1, 1, 0))
    for (element, crystal_name), miller, layers, repeat in itertools.product(
        materials, millers, (3, 4, 5), ((1, 1, 1), (2, 2, 1))
    ):
        face = "".join(str(value) for value in miller)
        suffix = "x".join(str(value) for value in repeat[:2])
        build = _step(
            "build_surface", name="slab", element=element, crystal=crystal_name,
            miller=list(miller), layers=layers, vacuum=12.0, repeat=list(repeat),
        )
        yield RecipeCase(
            f"{element.lower()}-{face}-{layers}-{suffix}", "surface", element,
            (
                f"Build a {suffix} {element}({face}) slab with {layers} layers and 12 angstrom vacuum.",
                f"Create a {layers}-layer {crystal_name} {element} surface along ({face}), repeated {suffix} in plane.",
                f"I need {element} ({face}), {layers} atomic layers, {suffix} lateral repeat, and 12 A vacuum.",
            ),
            (build, _step("finish", name="slab")),
        )
        yield RecipeCase(
            f"{element.lower()}-{face}-{layers}-{suffix}-frozen", "surface_constraint", element,
            (
                f"Build that {suffix} {element}({face}) {layers}-layer slab and freeze its bottom two layers.",
                f"Create {element} ({face}) with {layers} layers; constrain the lowest two layers in xyz.",
                f"Prepare a {suffix} {element}({face}) slab with 12 A vacuum and fix two bottom layers.",
            ),
            (build, _step("freeze_layers", name="slab", side="bottom", layers=2, axes="xyz"), _step("finish", name="slab")),
        )
        for adsorbate in ("O", "H"):
            yield RecipeCase(
                f"{element.lower()}-{face}-{layers}-{suffix}-{adsorbate.lower()}-ontop", "atomic_adsorption", f"{element}-{adsorbate}",
                (
                    f"Put one {adsorbate} atom 1.8 A above an ontop site of a {suffix} {element}({face}) {layers}-layer slab.",
                    f"Build {element} ({face}), {layers} layers and {suffix}, then adsorb atomic {adsorbate} at the first ontop site.",
                    f"Create an {adsorbate}/{element}({face}) slab using an ontop adsorption height of 1.8 angstrom.",
                ),
                (build, _step("add_atomic_adsorbate", name="slab", element=adsorbate, site="ontop", height=1.8), _step("finish", name="slab")),
            )
        if repeat == (2, 2, 1):
            yield RecipeCase(
                f"{element.lower()}-{face}-{layers}-{suffix}-co", "molecular_adsorption", f"{element}-CO",
                (
                    f"Adsorb CO at the first ontop site of a {suffix} {element}({face}) {layers}-layer slab with 12 A vacuum, carbon anchored 1.9 A high.",
                    f"Build a {suffix} {layers}-layer {element} ({face}) surface with 12 A vacuum and place a CO molecule ontop through atom 1 at 1.9 A.",
                    f"Prepare CO on a {suffix} {element}({face}) {layers}-layer slab with 12 A vacuum, anchored through carbon at the first ontop site 1.9 A high.",
                ),
                (build, _step("add_molecular_adsorbate", name="slab", species="CO", anchor=1, site="ontop", height=1.9), _step("finish", name="slab")),
            )


def _molecule_cases() -> Iterable[RecipeCase]:
    for species in ("H2", "N2", "O2", "CO", "CO2", "H2O", "NH3", "CH4"):
        for box in (10.0, 12.0, 15.0):
            yield RecipeCase(
                f"{species.lower()}-{int(box)}", "molecule", species,
                (
                    f"Build an isolated {species} molecule in a {box:g} angstrom cubic box.",
                    f"Create molecular {species} centered in a nonperiodic {box:g} A cell.",
                    f"I need {species} with {box:g} angstrom of box size for an isolated calculation.",
                ),
                (_step("build_molecule", name="molecule", species=species, box=box), _step("finish", name="molecule")),
            )


def _prototype_cases() -> Iterable[RecipeCase]:
    for prototype in ("graphene", "graphite", "rutile-TiO2", "anatase-TiO2", "hBN"):
        yield RecipeCase(
            prototype.lower(), "prototype", prototype,
            (
                f"Build the {prototype} prototype.",
                f"Create a standard unit cell of {prototype}.",
                f"Use the named prototype builder for {prototype} and finish that structure.",
            ),
            (_step("build_prototype", name="prototype", prototype=prototype), _step("finish", name="prototype")),
        )


def _nanotube_cases() -> Iterable[RecipeCase]:
    for n, m, length in itertools.product((4, 5, 6, 8), (0, 3, 5), (1, 2)):
        if m > n:
            continue
        yield RecipeCase(
            f"cnt-{n}-{m}-{length}", "nanotube", "C",
            (
                f"Build a ({n},{m}) carbon nanotube {length} unit cells long with 10 A radial vacuum.",
                f"Create a length-{length} C nanotube of chirality ({n}, {m}).",
                f"I need an ({n},{m}) CNT, length {length}, isolated by 10 angstrom in x and y.",
            ),
            (_step("build_nanotube", name="tube", element="C", n=n, m=m, length=length, vacuum=10.0), _step("finish", name="tube")),
        )


def _recovery_cases() -> Iterable[RecipeCase]:
    for element, crystal_name in (("Al", "fcc"), ("Pt", "fcc"), ("Fe", "bcc"), ("Si", "diamond")):
        yield RecipeCase(
            f"{element.lower()}-invalid-index-recovery", "error_recovery", element,
            (
                f"Build a 2x1x1 {crystal_name} {element} cell with a vacancy at atom 1.",
                f"Create a {element} {crystal_name} supercell and remove its first site; recover if an atom index is invalid.",
                f"Prepare one vacancy in 2x1x1 {element}, using site 1 after correcting any failed selection.",
            ),
            (
                _step("build_bulk", name="bulk", element=element, crystal=crystal_name, cubic=True),
                _step("repeat", name="bulk", repeat=[2, 1, 1]),
                _step("make_vacancy", expect_error=True, name="bulk", selector={"indices": [9999]}),
                _step("make_vacancy", name="bulk", selector={"indices": [1]}),
                _step("finish", name="bulk"),
            ),
        )


def _clarification_cases() -> Iterable[RecipeCase]:
    yield RecipeCase(
        "iron-surface-miller", "clarification", "Fe",
        (
            "Build an iron surface.",
            "Make a slab of Fe, but ask me which facet before choosing one.",
            "I need an iron slab; clarify the missing Miller indices first.",
        ),
        (
            _step(
                "ask_clarification",
                question="Which iron surface facet should be built?",
                choices=["(110)", "(100)", "(111)"],
                field="miller",
                followup_user="Use the bcc Fe(110) facet with four layers and 12 A vacuum.",
            ),
            _step("build_surface", name="slab", element="Fe", crystal="bcc", miller=[1, 1, 0], layers=4, vacuum=12.0),
            _step("finish", name="slab"),
        ),
    )
    yield RecipeCase(
        "carbon-form", "clarification", "C",
        (
            "Build a carbon structure.",
            "Create carbon, but ask whether I mean bulk, molecule, sheet, or tube.",
            "I need a carbon material; clarify its structural form before building.",
        ),
        (
            _step(
                "ask_clarification",
                question="Which form of carbon should be built?",
                choices=["diamond bulk", "graphene sheet", "carbon nanotube", "isolated molecule"],
                field="structure_type",
                followup_user="Use a graphene sheet.",
            ),
            _step("build_prototype", name="sheet", prototype="graphene"),
            _step("finish", name="sheet"),
        ),
    )
    yield RecipeCase(
        "sodium-chloride-form", "clarification", "Na-Cl",
        (
            "Build sodium chloride.",
            "Create NaCl, but clarify whether I mean an isolated pair or a crystal.",
            "I need sodium chloride; ask which structural form before building it.",
        ),
        (
            _step(
                "ask_clarification",
                question="Should sodium chloride be an isolated pair or a rocksalt crystal?",
                choices=["isolated NaCl pair", "rocksalt crystal"],
                field="structure_type",
                followup_user="Build the rocksalt crystal with a 5.64 A cubic lattice.",
            ),
            _step(
                "build_crystal",
                name="rocksalt",
                symbols=["Na", "Cl"],
                basis=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                spacegroup=225,
                a=5.64,
            ),
            _step("finish", name="rocksalt"),
        ),
    )


def cases() -> list[RecipeCase]:
    return sorted(
        [
            *_bulk_cases(), *_surface_cases(), *_molecule_cases(), *_prototype_cases(),
            *_nanotube_cases(), *_recovery_cases(), *_clarification_cases(),
        ],
        key=lambda item: (item.family, item.case_id),
    )


def _split(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def _assign_splits(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Keep identical outputs together while covering every viable family."""
    output_groups: dict[str, list[dict[str, Any]]] = {}
    family_groups: dict[str, set[str]] = {}
    for record in records:
        output_hash = record["validation"]["output_hash"]
        family = record["split_group"].split(":", 1)[0]
        output_groups.setdefault(output_hash, []).append(record)
        family_groups.setdefault(family, set()).add(output_hash)

    assignments: dict[str, str] = {}
    for family, hashes in sorted(family_groups.items(), key=lambda item: (len(item[1]), item[0])):
        ordered = sorted(
            hashes,
            key=lambda value: hashlib.sha256(f"{family}:{value}".encode("utf-8")).hexdigest(),
        )
        required = ("train", "validation", "test") if len(ordered) >= 3 else ("train", "test")
        present = {assignments[value] for value in ordered if value in assignments}
        for split in required:
            if split in present:
                continue
            candidate = next((value for value in ordered if value not in assignments), None)
            if candidate is None:
                raise RuntimeError(f"cannot provide {split} coverage for family {family}")
            assignments[candidate] = split
            present.add(split)

    for output_hash in output_groups:
        assignments.setdefault(output_hash, _split(output_hash))

    split_records = {name: [] for name in ("train", "validation", "test")}
    for output_hash, grouped_records in output_groups.items():
        split_records[assignments[output_hash]].extend(grouped_records)
    return split_records


def _schemas_for(registry, names: set[str]) -> list[dict[str, Any]]:
    return [schema for schema in registry.function_schemas() if schema["function"]["name"] in names]


def execute_case(case: RecipeCase, prompt: str, paraphrase_index: int) -> dict[str, Any]:
    registry = create_default_registry()
    workspace = ASEWorkspace(registry, session_id=f"generator-{case.case_id}-{paraphrase_index}")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    for number, step in enumerate(case.steps, start=1):
        call_id = f"call_{number:03d}"
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": step["tool"], "arguments": step["args"]},
            }],
        })
        result = workspace.execute(step["tool"], step["args"])
        expect_error = bool(step.get("expect_error"))
        if expect_error and result.success:
            raise RuntimeError(f"{case.case_id}: expected {step['tool']} to fail")
        if not expect_error and not result.success:
            raise RuntimeError(f"{case.case_id}: {step['tool']} failed: {result.error}")
        result_content = result.observation if result.success else result.error
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": step["tool"],
            "content": json.dumps(result_content, sort_keys=True, separators=(",", ":")),
        })
        if step.get("followup_user"):
            messages.append({"role": "user", "content": step["followup_user"]})
    messages.append({"role": "assistant", "content": "The requested structure is complete and validated."})

    final_hash = atoms_hash(workspace.final_atoms())
    replay = ASEWorkspace(registry, session_id=f"replay-{case.case_id}")
    for step in workspace.recipe()["steps"]:
        replay.execute_or_raise(step["tool"], step["args"])
    replay_hash = atoms_hash(replay.final_atoms())
    if replay_hash != final_hash or replay.recipe_hash() != workspace.recipe_hash():
        raise RuntimeError(f"replay mismatch for {case.case_id}")

    tool_names = {step["tool"] for step in case.steps}
    record = {
        "id": f"{case.case_id}-p{paraphrase_index + 1}",
        "schema_version": SCHEMA_VERSION,
        "split_group": case.split_group,
        "tools": _schemas_for(registry, tool_names),
        "messages": messages,
        "recipe": workspace.recipe(),
        "provenance": {
            "source": "canonical_generator",
            "sanitized": True,
            "contains_private_structure": False,
        },
        "validation": {
            "schema_valid": True,
            "executed": True,
            "invariants_passed": True,
            "forbidden_action_count": 0,
            "recipe_hash": workspace.recipe_hash(),
            "registry_version": f"phase1-{registry.fingerprint()[:16]}",
            "output_hash": final_hash,
            "replay_hash": replay_hash,
            "error_recovery_count": sum(bool(step.get("expect_error")) for step in case.steps),
            "clarification_count": sum(step["tool"] == "ask_clarification" for step in case.steps),
        },
    }
    validate_record(record)
    return record


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    raw = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("training/datasets/pilot"))
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=Path("training/generators/paraphrase_templates"),
        help="Directory of agent-authored <family>.json prompt-template lists.",
    )
    args = parser.parse_args()

    templates = load_templates(args.templates_dir)
    routing_registry = create_default_registry()

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for case in cases():
        for paraphrase_index, prompt in enumerate(case_prompts(case, templates, routing_registry)):
            if args.max_records and len(records) >= args.max_records:
                break
            try:
                records.append(execute_case(case, prompt, paraphrase_index))
            except Exception as exc:
                failures.append({"case_id": case.case_id, "error": f"{type(exc).__name__}: {exc}"})
        if args.max_records and len(records) >= args.max_records:
            break

    if failures:
        preview = "; ".join(f"{item['case_id']}: {item['error']}" for item in failures[:10])
        raise RuntimeError(f"{len(failures)} generated cases failed validation: {preview}")
    if not records:
        raise RuntimeError("no records generated")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_records = _assign_splits(records)
    if not args.max_records and any(not values for values in split_records.values()):
        raise RuntimeError("grouped split produced an empty partition")

    hashes = {
        name: _write_jsonl(args.output_dir / f"{name}.jsonl", values)
        for name, values in split_records.items()
    }
    registry = create_default_registry()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "registry_version": f"phase1-{registry.fingerprint()[:16]}",
        "split_strategy": "stratified_family_coverage_grouped_by_output_hash/v1",
        "record_count": len(records),
        "split_counts": {name: len(values) for name, values in split_records.items()},
        "split_sha256": hashes,
        "failures": failures,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
