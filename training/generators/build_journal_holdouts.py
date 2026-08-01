#!/usr/bin/env python3
"""Build the section 10.3 held-out evaluation sets for the journal roles.

`CATALYST_STRUCTURE_LLM.md` section 10.3 asks for three held-out sets. Two are
buildable from deterministic fixtures and are produced here:

- `template_holdout` -- familiar chemistry, wording the training pool never used.
  Every phrase template and both request sentences are disjoint from
  `build_journal_corpus.py`, and generation fails closed if any rendered source
  text also appears in the training corpus.
- `family_holdout` -- unseen catalyst families and supports. Elements, adsorbate
  species and oxide supports are all absent from training; the wording stays
  in-distribution so chemistry is the only variable.

`journal_holdout` is deliberately not produced: it requires complete papers never
used in data preparation, and this repository contains no licensed journal text.

Both sets are evaluation-only -- one `test.jsonl` each, no train split.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from ase_auto_build.ase_agent.catalyst_contracts import policy_gate, validate_record
from training.dataset_contract import JOURNAL_SCHEMA_VERSION
from training.generators.build_journal_corpus import (
    CREATED_AT,
    PHASES as TRAINING_PHASES,
    PHRASE_TEMPLATES as TRAINING_PHRASES,
    REQUEST_TEMPLATES as TRAINING_REQUESTS,
    _paths as provenance_paths,
    _role_records,
    _spec,
)

PRODUCER = "journal-holdout-generator/v1"

# --------------------------------------------------------------------------
# Wording pools
# --------------------------------------------------------------------------

# TRAINING_PHRASES, TRAINING_REQUESTS and TRAINING_PHASES are imported above
# straight from the corpus generator, so a change to the training pool cannot
# silently invalidate the "held out" claim made here.

#: Disjoint from every string above; four English forms plus one Traditional Chinese.
HELDOUT_PHRASES = (
    "{label_capitalized} was set to {value} in the published procedure.",
    "We adopted {value} as the {label}.",
    "According to the methods section, {label} came to {value}.",
    "Reported {label} — {value}.",
    "該研究以 {value} 作為{label}。",
)
HELDOUT_REQUESTS = (
    "Rebuild the catalyst exactly as this excerpt describes.",
    "請重建本段文字所描述的觸媒結構。",
)

_LABELS = {
    "material.formula": "chemical formula",
    "material.crystal_structure": "crystal structure",
    "model.kind": "model type",
    "model.miller_indices": "Miller indices",
    "model.supercell": "supercell repeat",
    "model.layers": "slab layer count",
    "model.vacuum_angstrom": "vacuum in angstrom",
    "model.periodic_boundary_conditions": "periodic boundary conditions",
    "model.fixed_layers_from_bottom": "fixed bottom layers",
    "model.center": "centering flag",
    "model.atom_ordering": "atom ordering",
    "model.termination": "surface termination",
    "model.shape": "cluster shape",
    "model.shells": "cluster shell count",
    "requested_outputs": "requested output formats",
}


# --------------------------------------------------------------------------
# Chemistry pools
# --------------------------------------------------------------------------

TRAINING_ADSORBATE_SPECIES = ("O", "H", "CO", "NH3", "H2O")
TRAINING_SUPPORTS = ("MgO",)

HELDOUT_PHASES = (
    ("Rh", "fcc"), ("Ir", "fcc"), ("Pb", "fcc"), ("Mo", "bcc"), ("Ta", "bcc"),
)
#: (element, species, anchor, height) -- none of these species appear in training.
HELDOUT_ADSORBATES = (
    ("N", None, None, 1.45), ("C", None, None, 1.35),
    (None, "NO", 1, 1.95), (None, "O2", 1, 1.75), (None, "OH", 1, 1.85),
)
HELDOUT_SUPPORTS = (
    ("CeO2", "fluorite", [1, 1, 1]),
    ("CaO", "rocksalt", [1, 0, 0]),
)

_FACETS = ([1, 1, 1], [1, 0, 0], [1, 1, 0])
_REPEATS = ([1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1])


def _render(field: str, value: Any, phrases: Iterable[str], ordinal: int) -> str:
    label = _LABELS.get(field, field.replace("_", " ").replace(".", " "))
    pool = tuple(phrases)
    return pool[ordinal % len(pool)].format(
        label=label, label_capitalized=label.capitalize(),
        value=json.dumps(value, sort_keys=True),
    )


def _case_records(
    case_id: str,
    family: str,
    spec: dict[str, Any],
    ordinal: int,
    *,
    phrases: tuple[str, ...],
    requests: tuple[str, ...],
) -> list[dict[str, Any]]:
    """One extractor and one planner record, grounded in the rendered source."""
    request_id = f"journal-{case_id}"
    locator = f"Held-out fixture {case_id}"
    fields = provenance_paths(spec)
    lines = [_render(field, value, phrases, ordinal + index) for index, (field, value) in enumerate(fields)]
    claims = [{
        "field": field, "value": value, "evidence_type": "user_supplied",
        "source_id": "fixture-1", "locator": locator, "confidence": "high",
    } for field, value in fields]
    evidence = {
        "schema_version": "evidence-ledger/v1", "request_id": request_id,
        "created_at": CREATED_AT, "producer_version": PRODUCER, "artifact_hashes": {},
        "claims": claims, "contradictions": [], "unresolved_fields": [],
    }
    spec = {**spec, "provenance": [{
        "field": claim["field"], "value": claim["value"],
        "evidence_type": "user_supplied", "claim_index": index,
    } for index, claim in enumerate(claims)]}
    proposal = {
        "schema_version": "spec-proposal/v1", "request_id": request_id,
        "created_at": CREATED_AT, "producer_version": PRODUCER, "artifact_hashes": {},
        "task_status": "ready", "catalyst_spec": spec,
        "field_sources": [{"field": claim["field"], "evidence_type": "user_supplied", "claim_index": index}
                          for index, claim in enumerate(claims)],
        "clarification_questions": [], "agent_warnings": [],
    }
    validate_record("evidence_ledger", evidence)
    validate_record("spec_proposal", proposal)
    decision = policy_gate(evidence, proposal)
    if not decision.ready:
        raise RuntimeError(f"{case_id}: held-out target did not pass the policy gate: {decision.errors}")
    source = {"source_id": "fixture-1", "locator": locator, "text": "\n".join(lines)}
    return _role_records(
        case_id, family, requests[ordinal % len(requests)], [source], evidence, proposal
    )


def _surface(element: str, crystal: str, facet, layers, vacuum, repeat, modifications=()):
    return _spec("surface", {"formula": element, "crystal_structure": crystal}, {
        "kind": "surface", "miller_indices": facet, "supercell": repeat,
        "layers": layers, "vacuum_angstrom": vacuum,
        "periodic_boundary_conditions": [True, True, False],
        "fixed_layers_from_bottom": min(2, layers - 1), "center": True,
        "atom_ordering": "ase_default",
    }, list(modifications))


def _template_specs() -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Familiar chemistry only -- every element and adsorbate is a training one."""
    for index, ((element, crystal), repeat) in enumerate(itertools.islice(
        itertools.cycle(itertools.product(TRAINING_PHASES, _REPEATS)), 40
    )):
        yield f"tmpl-bulk-{index:03d}", "bulk", _spec(
            "bulk", {"formula": element, "crystal_structure": crystal}, {
                "kind": "bulk", "supercell": repeat,
                "periodic_boundary_conditions": [True, True, True],
                "center": True, "atom_ordering": "ase_default",
            }, [])

    for index, ((element, crystal), facet, layers, vacuum) in enumerate(itertools.islice(
        itertools.cycle(itertools.product(TRAINING_PHASES[:9], _FACETS, (3, 4, 5, 6), (10.0, 12.0, 14.0, 16.0))), 40
    )):
        yield f"tmpl-surface-{index:03d}", "surface", _surface(
            element, crystal, facet, layers, vacuum, _REPEATS[index % 4])

    training_adsorbates = (
        ("O", None, None, 1.6), ("H", None, None, 1.2), (None, "CO", 2, 1.9),
        (None, "NH3", 1, 2.0), (None, "H2O", 1, 1.9),
    )
    for index, ((element, crystal), facet, adsorbate) in enumerate(itertools.islice(
        itertools.cycle(itertools.product(TRAINING_PHASES[:5], _FACETS, training_adsorbates)), 20
    )):
        atom, species, anchor, height = adsorbate
        modification: dict[str, Any] = {
            "operation": "add_adsorbate", "site": "ontop", "site_index": 1,
            "height_angstrom": height,
        }
        modification.update({"species": species, "anchor": anchor} if species else {"element": atom})
        yield f"tmpl-adsorbate-{index:03d}", "adsorbate", _surface(
            element, crystal, facet, 4, 14.0, [2, 2, 1], [modification])


def _family_specs() -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Unseen elements, adsorbate species and oxide supports."""
    for index, ((element, crystal), repeat) in enumerate(itertools.islice(
        itertools.cycle(itertools.product(HELDOUT_PHASES, _REPEATS)), 30
    )):
        yield f"fam-bulk-{index:03d}", "bulk", _spec(
            "bulk", {"formula": element, "crystal_structure": crystal}, {
                "kind": "bulk", "supercell": repeat,
                "periodic_boundary_conditions": [True, True, True],
                "center": True, "atom_ordering": "ase_default",
            }, [])

    for index, ((element, crystal), facet, layers, vacuum) in enumerate(itertools.islice(
        itertools.cycle(itertools.product(HELDOUT_PHASES, _FACETS, (3, 4, 5, 6), (11.0, 13.0, 15.0, 17.0))), 30
    )):
        yield f"fam-surface-{index:03d}", "surface", _surface(
            element, crystal, facet, layers, vacuum, _REPEATS[index % 4])

    for index, ((element, crystal), facet, adsorbate) in enumerate(itertools.islice(
        itertools.cycle(itertools.product(HELDOUT_PHASES, _FACETS, HELDOUT_ADSORBATES)), 25
    )):
        atom, species, anchor, height = adsorbate
        modification: dict[str, Any] = {
            "operation": "add_adsorbate", "site": "ontop", "site_index": 1,
            "height_angstrom": height,
        }
        modification.update({"species": species, "anchor": anchor} if species else {"element": atom})
        yield f"fam-adsorbate-{index:03d}", "adsorbate", _surface(
            element, crystal, facet, 4, 14.0, [2, 2, 1], [modification])

    for index, ((formula, crystal, facet), element, gap) in enumerate(itertools.islice(
        itertools.cycle(itertools.product(HELDOUT_SUPPORTS, ("Rh", "Ir", "Mo"), (2.1, 2.4))), 15
    )):
        spec = _spec("surface", {"formula": formula, "crystal_structure": crystal}, {
            "kind": "surface", "miller_indices": facet, "supercell": [2, 2, 1],
            "layers": 4, "vacuum_angstrom": 14.0, "termination": "ase_default",
            "periodic_boundary_conditions": [True, True, False], "center": True,
            "atom_ordering": "ase_default",
        }, [{"operation": "add_supported_cluster", "element": element, "shape": "icosahedron",
             "shells": 1, "gap_angstrom": gap, "vacuum_angstrom": 8.0}])
        yield f"fam-supported-{index:03d}", "supported_cluster", spec


def _training_signature(train_path: Path) -> tuple[set[str], set[str]]:
    """Source texts and chemistry tuples the training corpus already contains."""
    texts: set[str] = set()
    chemistry: set[str] = set()
    for line in train_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        reference = record["reference"]
        for source in reference["sources"]:
            texts.add(source["text"])
        spec = reference["spec_proposal"]["catalyst_spec"]
        chemistry.add(spec["material"]["formula"])
        for modification in spec["modifications"]:
            species = modification.get("species") or modification.get("element")
            if species:
                chemistry.add(species)
    return texts, chemistry


def _write(directory: Path, name: str, records: list[dict[str, Any]], notes: dict[str, Any]) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in records)
    (directory / "test.jsonl").write_text(raw, encoding="utf-8")
    families: dict[str, int] = {}
    for record in records:
        family = record["split_group"].split(":", 1)[0]
        families[family] = families.get(family, 0) + 1
    manifest = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "generator": "training/generators/build_journal_holdouts.py",
        "set_name": name, "evaluation_only": True,
        "record_count": len(records), "family_counts": families,
        "split_sha256": {"test": hashlib.sha256(raw.encode()).hexdigest()},
        "source_policy": "deterministic synthetic fixtures; no journal text",
        **notes,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("training/datasets/journal_roles_v1/train.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("training/datasets"))
    args = parser.parse_args()

    train_texts, train_chemistry = _training_signature(args.train)

    template_records: list[dict[str, Any]] = []
    for ordinal, (case_id, family, spec) in enumerate(_template_specs()):
        template_records.extend(_case_records(
            case_id, family, spec, ordinal,
            phrases=HELDOUT_PHRASES, requests=HELDOUT_REQUESTS,
        ))
    family_records: list[dict[str, Any]] = []
    for ordinal, (case_id, family, spec) in enumerate(_family_specs()):
        family_records.extend(_case_records(
            case_id, family, spec, ordinal,
            phrases=TRAINING_PHRASES, requests=TRAINING_REQUESTS,
        ))

    # Fail closed rather than ship a "held-out" set that leaks.
    leaked_text = [
        record["id"] for record in template_records
        if any(source["text"] in train_texts for source in record["reference"]["sources"])
    ]
    if leaked_text:
        raise RuntimeError(f"template_holdout wording appears in training: {leaked_text[:5]}")
    if set(HELDOUT_PHRASES) & set(TRAINING_PHRASES) or set(HELDOUT_REQUESTS) & set(TRAINING_REQUESTS):
        raise RuntimeError("template_holdout must not reuse a training phrase")

    leaked_chemistry = sorted({
        element for element, _ in HELDOUT_PHASES if element in train_chemistry
    } | {
        (species or atom) for atom, species, _, _ in HELDOUT_ADSORBATES
        if (species or atom) in train_chemistry
    } | {
        formula for formula, _, _ in HELDOUT_SUPPORTS if formula in train_chemistry
    })
    if leaked_chemistry:
        raise RuntimeError(f"family_holdout chemistry appears in training: {leaked_chemistry}")

    manifests = {
        "template_holdout": _write(
            args.output_root / "journal_holdout_template", "template_holdout", template_records,
            {
                "design_section": "CATALYST_STRUCTURE_LLM.md section 10.3 template_holdout",
                "variable": "wording only; chemistry is drawn from the training pool",
                "phrase_templates": list(HELDOUT_PHRASES),
                "request_templates": list(HELDOUT_REQUESTS),
                "disjoint_from_training_wording": True,
            },
        ),
        "family_holdout": _write(
            args.output_root / "journal_holdout_family", "family_holdout", family_records,
            {
                "design_section": "CATALYST_STRUCTURE_LLM.md section 10.3 family_holdout",
                "variable": "chemistry only; wording is drawn from the training pool",
                "heldout_elements": [element for element, _ in HELDOUT_PHASES],
                "heldout_adsorbates": [species or atom for atom, species, _, _ in HELDOUT_ADSORBATES],
                "heldout_supports": [formula for formula, _, _ in HELDOUT_SUPPORTS],
                "disjoint_from_training_chemistry": True,
            },
        ),
    }
    print(json.dumps(manifests, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
