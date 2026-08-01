#!/usr/bin/env python3
"""Build `prose_holdout` -- Phase 1 of `training/JOURNAL_ROLE_R3_SCHEDULE.md`.

`build_journal_corpus.py` grounds every training claim in a source sentence of
the form `The reported {label} is {json.dumps(value)}.` -- the extractor never
has to *interpret* prose, only copy a literal that is already sitting in the
text. The live smoke that motivated this schedule showed the gap directly:
real requests read like

    "...centered geometry, ASE-default atom ordering, and CIF plus VASP
    outputs."

and the r2 extractor, trained only on literal-echo sentences, answered
`"ase-default"` and `center: "geometry"` -- both schema violations.

This generator builds an evaluation-only set with the same target schema and
the same deterministic dispatch guarantee as the training corpus, but whose
source text is genuine prose: numbers are spelled out or written in natural
units, booleans and enums are described rather than quoted, and every
sentence requires mapping onto a canonical value rather than copying one.
`_assert_no_leak` enforces this as a fail-closed generation-time invariant --
see the schedule doc section 3, Phase 1, for the specific prose forms this is
built to cover.

Evaluation-only: one `test.jsonl`, no train split.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterator

from ase_auto_build.ase_agent.catalyst_contracts import policy_gate, validate_record
from ase_auto_build.ase_agent.catalyst_dispatch import dispatch_spec
from training.dataset_contract import JOURNAL_SCHEMA_VERSION
from training.generators.build_journal_corpus import (
    CREATED_AT,
    PHASES as TRAINING_PHASES,
    TRAINING_PROSE_TEMPLATES,
    _paths as provenance_paths,
    _role_records,
    _spec,
)
from training.generators.prose_leak_guard import _assert_no_leak, _bare_leak, _literal_tokens

PRODUCER = "prose-holdout-generator/v1"

# --------------------------------------------------------------------------
# Chemistry pool -- deliberately in-distribution (family_holdout already
# measures unseen chemistry; this set isolates the prose-vs-literal axis).
# Silicon/diamond is dropped: "diamond cubic" cannot be phrased in English
# without writing the word "diamond", which *is* the crystal_structure
# literal -- every other phase's element name and crystal-family name are
# safely disjoint from their own schema tokens (checked by
# `test_no_source_text_contains_a_target_literal`).
# --------------------------------------------------------------------------
PHASES = tuple(phase for phase in TRAINING_PHASES if phase[0] != "Si")

ELEMENT_NAMES = {
    "Al": "aluminum", "Cu": "copper", "Ni": "nickel", "Pt": "platinum",
    "Au": "gold", "Ag": "silver", "Pd": "palladium", "Fe": "iron", "W": "tungsten",
    "O": "oxygen", "H": "hydrogen",
}
#: CRYSTAL_NAMES (per-family noun phrases) is defined further down, next to
#: the other prose-template pools it must stay disjoint from -- see
#: `HOLDOUT_PROSE_TEMPLATES` and the cross-pool assertion below it.
_MOLECULE_NAMES = {"CO": "carbon monoxide", "NH3": "ammonia", "H2O": "water"}
_ANCHOR_ATOM = {"CO": "carbon", "NH3": "nitrogen", "H2O": "oxygen"}
_LAYER_WORDS = {4: "four", 5: "five", 6: "six"}

PROSE_REQUESTS = (
    "Build the catalyst candidate described in this excerpt.",
    "Reconstruct the structure reported in the passage below.",
    "From the following description, construct the corresponding candidate.",
    "Rebuild the catalyst exactly as this excerpt describes.",
)

#: Reasons for fields the deterministic generator marks `evidence_type: "derived"`
#: rather than grounding in a rendered sentence -- see the module docstring on
#: `_build_case` for why these specific fields are exempt from prose rendering.
DERIVED_REASONS = {
    "model.kind": "the model kind is implied by which fields (Miller index, layers, vacuum) are present in the description",
    "model.fixed_layers_from_bottom": "freezing the bottom two layers is the standard slab-relaxation convention used throughout this study",
    "modifications[0].operation": "the modification type is implied by the described atomic-level change",
    "modifications[0].site_index": "the site index defaults to the sole reported equivalent adsorption site",
    "modifications[0].anchor": "the anchor atom is the one named as binding the surface in the description",
    "modifications[0].selector": "the selector targets the sole outermost surface position described",
}


# --------------------------------------------------------------------------
# The literal-leakage guard -- the property that distinguishes this set from
# template_holdout, which varies sentence *frames* but keeps the embedded
# JSON literal (JOURNAL_ROLE_R3_SCHEDULE.md section 1). Implementation lives
# in `prose_leak_guard.py`, shared with `build_journal_corpus.py`'s training
# prose pool, so the two generators cannot silently diverge on what counts as
# a leak; re-imported here (`_assert_no_leak` etc. above) so `prose._assert_no_leak`
# keeps working for existing callers/tests.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Per-field prose templates. Every raw template string below is collected into
# `HOLDOUT_PROSE_TEMPLATES` and asserted disjoint from
# `build_journal_corpus.TRAINING_PROSE_TEMPLATES` -- see the assertion at the
# end of this section. Every sentence rendered from these templates must also
# independently satisfy `_assert_no_leak` for the field(s) it grounds --
# exercised at generation time for every case, not just spot-checked.
# --------------------------------------------------------------------------

_FORMULA_BULK_TEMPLATES = ("The catalyst is {name}.", "{name_cap} is the material of interest.", "The bulk phase is {name}.")
_FORMULA_SURFACE_TEMPLATES = ("The host metal is {name}.", "{name_cap} was used as the substrate.", "The catalyst surface is {name}.")

#: Noun phrases per crystal family -- disjoint from build_journal_corpus's own
#: pool (which uses "face-centred cubic"/"an fcc lattice" wording instead).
CRYSTAL_NAMES = {
    "fcc": ("a cubic close-packed lattice", "the ccp structure", "cubic close packing"),
    "bcc": ("a body-centered cubic lattice", "the body-centered cubic form", "body-centered cubic packing"),
}
_CRYSTAL_SENTENCE_TEMPLATES = ("It crystallizes in {phrase}.", "The lattice is {phrase}.", "{phrase_cap} describes the structure.")

_SUPERCELL_SQUARE_TEMPLATES = ("A {a} x {a} surface cell was used.", "A {a}×{a} supercell was built.", "The cell was repeated {a} by {a}.")
_SUPERCELL_GENERAL_TEMPLATES = ("A {a}×{b}×{c} repeat of the cell was used.", "The cell was repeated {a} by {b} by {c}.")

_PBC_BULK_TEMPLATES = ("The cell is periodic in all three directions.", "Periodic boundary conditions apply along every axis.")
_PBC_SURFACE_TEMPLATES = (
    "The system is periodic in x and y, with vacuum separating slabs along z.",
    "Periodicity is enforced only in the surface plane, with a vacuum gap normal to it.",
)

_CENTER_TEMPLATES = ("The slab was centred in the simulation cell.", "The structure was centered within the cell.", "Atoms were centered in the box before export.")
_ORDERING_TEMPLATES = ("Atoms were kept in ASE-default order.", "The atom ordering follows ASE's default sequence.", "Atoms are listed in the ASE default order.")
_OUTPUTS_TEMPLATES = ("CIF and VASP outputs are requested.", "Please export CIF plus POSCAR files.", "Provide both a CIF file and a VASP POSCAR output.")

#: Miller-index phrasing avoids the parenthesised "(111)" style reserved for
#: build_journal_corpus's training pool.
_MILLER_TEMPLATES = ("A {idx}-oriented surface was modeled.", "The {{{idx}}} facet was examined.", "The surface orientation is {idx}.")

_LAYERS_TEMPLATES = ("The slab has {word} atomic layers.", "A {word}-layer slab was built.", "{word_cap} layers make up the slab.")
_VACUUM_TEMPLATES = ("{n} Å of vacuum separates the slabs.", "About {n} angstroms of vacuum were added.", "{article} {n} angstrom vacuum gap was used.")
_ELEMENT_SUB_TEMPLATES = ("The topmost atom was substituted with a {name} atom.", "A surface atom was replaced by {name}.", "{name_cap} was substituted for the outermost atom.")

_ADSORBATE_MOLECULE_TEMPLATES = (
    "A {name} molecule was bound through its {atom} atom at an atop site, {height} Å above the surface.",
    "{name_cap} adsorbs via its {atom} atom on the on-top site, {height} Å from the surface.",
    "The {name} molecule sits at the atop adsorption site, anchored through {atom}, {height} Å above the top layer.",
)
_ADSORBATE_ATOM_TEMPLATES = (
    "An atomic {name} adsorbate occupies the atop site, {height} Å above the surface.",
    "A single {name} atom was placed on the on-top site, {height} Å from the surface.",
    "{name_cap} adsorbs at the atop position, {height} Å above the top layer.",
)

#: Every raw template string this module renders from, flattened for the
#: cross-pool disjointness assertion below.
HOLDOUT_PROSE_TEMPLATES: tuple[str, ...] = (
    _FORMULA_BULK_TEMPLATES + _FORMULA_SURFACE_TEMPLATES
    + tuple(phrase for phrases in CRYSTAL_NAMES.values() for phrase in phrases)
    + _CRYSTAL_SENTENCE_TEMPLATES
    + _SUPERCELL_SQUARE_TEMPLATES + _SUPERCELL_GENERAL_TEMPLATES
    + _PBC_BULK_TEMPLATES + _PBC_SURFACE_TEMPLATES
    + _CENTER_TEMPLATES + _ORDERING_TEMPLATES + _OUTPUTS_TEMPLATES
    + _MILLER_TEMPLATES + _LAYERS_TEMPLATES + _VACUUM_TEMPLATES + _ELEMENT_SUB_TEMPLATES
    + _ADSORBATE_MOLECULE_TEMPLATES + _ADSORBATE_ATOM_TEMPLATES
)


def _assert_disjoint_prose_forms(training: tuple[str, ...], holdout: tuple[str, ...]) -> None:
    """Fail closed if this module's own wording reuses a training surface
    form for any field. Called at import time below (real pools); a negative
    control that proves this fires on a deliberately-shared pair lives in
    `tests/test_prose_holdout.py`."""
    shared = set(training) & set(holdout)
    if shared:
        raise RuntimeError(f"prose_holdout reuses training prose surface forms: {sorted(shared)[:5]}")


_assert_disjoint_prose_forms(TRAINING_PROSE_TEMPLATES, HOLDOUT_PROSE_TEMPLATES)


def _formula_sentence(value: str, ordinal: int, family: str) -> str:
    name = ELEMENT_NAMES[value]
    pool = _FORMULA_BULK_TEMPLATES if family == "bulk" else _FORMULA_SURFACE_TEMPLATES
    return pool[ordinal % len(pool)].format(name=name, name_cap=name.capitalize())


def _crystal_sentence(value: str, ordinal: int) -> str:
    phrase = CRYSTAL_NAMES[value][ordinal % len(CRYSTAL_NAMES[value])]
    pool = _CRYSTAL_SENTENCE_TEMPLATES
    return pool[ordinal % len(pool)].format(phrase=phrase, phrase_cap=phrase.capitalize())


def _supercell_sentence(value: list[int], ordinal: int) -> str:
    a, b, c = value
    if a == b and c == 1 and a in (2, 3):
        pool = _SUPERCELL_SQUARE_TEMPLATES
        return pool[ordinal % len(pool)].format(a=a)
    pool = _SUPERCELL_GENERAL_TEMPLATES
    return pool[ordinal % len(pool)].format(a=a, b=b, c=c)


def _pbc_sentence(bulk: bool, ordinal: int) -> str:
    pool = _PBC_BULK_TEMPLATES if bulk else _PBC_SURFACE_TEMPLATES
    return pool[ordinal % len(pool)]


def _center_sentence(ordinal: int) -> str:
    pool = _CENTER_TEMPLATES
    return pool[ordinal % len(pool)]


def _ordering_sentence(ordinal: int) -> str:
    pool = _ORDERING_TEMPLATES
    return pool[ordinal % len(pool)]


def _outputs_sentence(ordinal: int) -> str:
    pool = _OUTPUTS_TEMPLATES
    return pool[ordinal % len(pool)]


def _miller_sentence(value: list[int], ordinal: int) -> str:
    idx = "".join(str(i) for i in value)
    pool = _MILLER_TEMPLATES
    return pool[ordinal % len(pool)].format(idx=idx)


def _layers_sentence(value: int, ordinal: int) -> str:
    word = _LAYER_WORDS[value]
    pool = _LAYERS_TEMPLATES
    return pool[ordinal % len(pool)].format(word=word, word_cap=word.capitalize())


#: Spoken forms of the vacuum pool that start with a vowel sound ("eleven",
#: "eighteen"), so "A {n} angstrom" reads as "An {n} angstrom".
_VOWEL_SOUND_VACUUM = {11, 18}


def _vacuum_sentence(value: float, ordinal: int) -> str:
    n = int(value)
    article = "An" if n in _VOWEL_SOUND_VACUUM else "A"
    pool = _VACUUM_TEMPLATES
    return pool[ordinal % len(pool)].format(n=n, article=article)


def _element_sentence(value: str, ordinal: int) -> str:
    name = ELEMENT_NAMES[value]
    pool = _ELEMENT_SUB_TEMPLATES
    return pool[ordinal % len(pool)].format(name=name, name_cap=name.capitalize())


def _adsorbate_sentence(species: str | None, element: str | None, height: float, ordinal: int) -> str:
    if species:
        name, atom = _MOLECULE_NAMES[species], _ANCHOR_ATOM[species]
        pool = _ADSORBATE_MOLECULE_TEMPLATES
        return pool[ordinal % len(pool)].format(name=name, name_cap=name.capitalize(), atom=atom, height=height)
    name = ELEMENT_NAMES[element]
    pool = _ADSORBATE_ATOM_TEMPLATES
    return pool[ordinal % len(pool)].format(name=name, name_cap=name.capitalize(), height=height)


# --------------------------------------------------------------------------
# Case assembly -- ties prose to `provenance_paths(spec)` field-for-field.
# --------------------------------------------------------------------------

def _build_case(
    case_id: str, family: str, spec: dict[str, Any], request: str,
    sentence_by_field: dict[str, str], derived_reason_by_field: dict[str, str],
) -> list[dict[str, Any]]:
    """Ground one case in prose. `sentence_by_field` supplies the rendered
    text for fields backed by the source; `derived_reason_by_field` supplies
    a `reason` for fields the CatalystSpec provenance marks `"derived"`
    instead -- required fields never grounded in a claim (e.g. `model.kind`,
    the internal tool `operation` name) because English prose about them
    would either be nonsensical or force exactly the token collision this
    generator exists to avoid (see the module docstring).

    Fails closed if the two dicts do not together cover exactly the fields
    `provenance_paths(spec)` -- the deterministic generator's own field
    walk -- reports, so this can never silently under- or over-ground a case.
    """
    fields = dict(provenance_paths(spec))
    covered = set(sentence_by_field) | set(derived_reason_by_field)
    if covered != set(fields):
        raise RuntimeError(
            f"{case_id}: field coverage mismatch -- missing={sorted(set(fields) - covered)}, "
            f"extra={sorted(covered - set(fields))}"
        )

    request_id = f"journal-{case_id}"
    locator = f"Held-out prose fixture {case_id}"

    ordered_sentences: list[str] = []
    for field in fields:
        sentence = sentence_by_field.get(field)
        if sentence is not None and sentence not in ordered_sentences:
            ordered_sentences.append(sentence)
    text = " ".join(ordered_sentences)

    claims: list[dict[str, Any]] = []
    claim_index_by_field: dict[str, int] = {}
    for field, sentence in sentence_by_field.items():
        value = fields[field]
        _assert_no_leak(case_id, field, value, text)
        claim_index_by_field[field] = len(claims)
        claims.append({
            "field": field, "value": value, "evidence_type": "user_supplied",
            "source_id": "fixture-1", "locator": locator, "confidence": "high",
        })
    evidence = {
        "schema_version": "evidence-ledger/v1", "request_id": request_id,
        "created_at": CREATED_AT, "producer_version": PRODUCER, "artifact_hashes": {},
        "claims": claims, "contradictions": [], "unresolved_fields": [],
    }

    provenance: list[dict[str, Any]] = []
    field_sources: list[dict[str, Any]] = []
    for field, value in fields.items():
        if field in sentence_by_field:
            index = claim_index_by_field[field]
            provenance.append({"field": field, "value": value, "evidence_type": "user_supplied", "claim_index": index})
            field_sources.append({"field": field, "evidence_type": "user_supplied", "claim_index": index})
        else:
            reason = derived_reason_by_field[field]
            provenance.append({"field": field, "value": value, "evidence_type": "derived", "reason": reason})
            field_sources.append({"field": field, "evidence_type": "derived", "reason": reason})

    new_spec = {**spec, "provenance": provenance}
    proposal = {
        "schema_version": "spec-proposal/v1", "request_id": request_id,
        "created_at": CREATED_AT, "producer_version": PRODUCER, "artifact_hashes": {},
        "task_status": "ready", "catalyst_spec": new_spec, "field_sources": field_sources,
        "clarification_questions": [], "agent_warnings": [],
    }
    validate_record("evidence_ledger", evidence)
    validate_record("spec_proposal", proposal)
    decision = policy_gate(evidence, proposal)
    if not decision.ready:
        raise RuntimeError(f"{case_id}: prose target did not pass the policy gate: {decision.errors}")
    try:
        dispatch_spec(new_spec, request_id=request_id)
    except Exception as exc:  # noqa: BLE001 -- fail closed naming the case
        raise RuntimeError(f"{case_id}: prose target failed dispatch_spec: {exc}") from exc

    source = {"source_id": "fixture-1", "locator": locator, "text": text}
    return _role_records(case_id, family, request, [source], evidence, proposal)


def _surface_spec(element: str, crystal: str, facet: list[int], layers: int, vacuum: float, repeat: list[int], modifications: list[dict[str, Any]]) -> dict[str, Any]:
    return _spec("surface", {"formula": element, "crystal_structure": crystal}, {
        "kind": "surface", "miller_indices": facet, "supercell": repeat,
        "layers": layers, "vacuum_angstrom": vacuum,
        "periodic_boundary_conditions": [True, True, False],
        "fixed_layers_from_bottom": min(2, layers - 1), "center": True,
        "atom_ordering": "ase_default",
    }, modifications)


def _bulk_case(case_id: str, ordinal: int, element: str, crystal: str, repeat: list[int]) -> list[dict[str, Any]]:
    spec = _spec("bulk", {"formula": element, "crystal_structure": crystal}, {
        "kind": "bulk", "supercell": repeat,
        "periodic_boundary_conditions": [True, True, True],
        "center": True, "atom_ordering": "ase_default",
    }, [])
    sentence_by_field = {
        "material.formula": _formula_sentence(element, ordinal, "bulk"),
        "material.crystal_structure": _crystal_sentence(crystal, ordinal + 1),
        "model.supercell": _supercell_sentence(repeat, ordinal + 2),
        "model.periodic_boundary_conditions": _pbc_sentence(True, ordinal + 3),
        "model.center": _center_sentence(ordinal + 4),
        "model.atom_ordering": _ordering_sentence(ordinal + 5),
        "requested_outputs": _outputs_sentence(ordinal + 6),
    }
    derived = {"model.kind": DERIVED_REASONS["model.kind"]}
    request = PROSE_REQUESTS[ordinal % len(PROSE_REQUESTS)]
    return _build_case(case_id, "bulk", spec, request, sentence_by_field, derived)


def _surface_case(case_id: str, ordinal: int, element: str, crystal: str, facet: list[int], layers: int, vacuum: float, repeat: list[int]) -> list[dict[str, Any]]:
    spec = _surface_spec(element, crystal, facet, layers, vacuum, repeat, [])
    sentence_by_field = {
        "material.formula": _formula_sentence(element, ordinal, "surface"),
        "material.crystal_structure": _crystal_sentence(crystal, ordinal + 1),
        "model.miller_indices": _miller_sentence(facet, ordinal + 2),
        "model.supercell": _supercell_sentence(repeat, ordinal + 3),
        "model.layers": _layers_sentence(layers, ordinal + 4),
        "model.vacuum_angstrom": _vacuum_sentence(vacuum, ordinal + 5),
        "model.periodic_boundary_conditions": _pbc_sentence(False, ordinal + 6),
        "model.center": _center_sentence(ordinal + 7),
        "model.atom_ordering": _ordering_sentence(ordinal + 8),
        "requested_outputs": _outputs_sentence(ordinal + 9),
    }
    derived = {
        "model.kind": DERIVED_REASONS["model.kind"],
        "model.fixed_layers_from_bottom": DERIVED_REASONS["model.fixed_layers_from_bottom"],
    }
    request = PROSE_REQUESTS[ordinal % len(PROSE_REQUESTS)]
    return _build_case(case_id, "surface", spec, request, sentence_by_field, derived)


def _adsorbate_case(
    case_id: str, ordinal: int, element: str, crystal: str, facet: list[int],
    layers: int, vacuum: float, repeat: list[int],
    atom: str | None, species: str | None, anchor: int | None, height: float,
) -> list[dict[str, Any]]:
    modification: dict[str, Any] = {"operation": "add_adsorbate", "site": "ontop", "site_index": 1, "height_angstrom": height}
    if species:
        modification.update({"species": species, "anchor": anchor})
    else:
        modification["element"] = atom
    spec = _surface_spec(element, crystal, facet, layers, vacuum, repeat, [modification])

    adsorbate_line = _adsorbate_sentence(species, atom, height, ordinal + 10)
    sentence_by_field = {
        "material.formula": _formula_sentence(element, ordinal, "surface"),
        "material.crystal_structure": _crystal_sentence(crystal, ordinal + 1),
        "model.miller_indices": _miller_sentence(facet, ordinal + 2),
        "model.supercell": _supercell_sentence(repeat, ordinal + 3),
        "model.layers": _layers_sentence(layers, ordinal + 4),
        "model.vacuum_angstrom": _vacuum_sentence(vacuum, ordinal + 5),
        "model.periodic_boundary_conditions": _pbc_sentence(False, ordinal + 6),
        "model.center": _center_sentence(ordinal + 7),
        "model.atom_ordering": _ordering_sentence(ordinal + 8),
        "requested_outputs": _outputs_sentence(ordinal + 9),
        "modifications[0].site": adsorbate_line,
        "modifications[0].height_angstrom": adsorbate_line,
    }
    if species:
        sentence_by_field["modifications[0].species"] = adsorbate_line
    else:
        sentence_by_field["modifications[0].element"] = adsorbate_line

    derived = {
        "model.kind": DERIVED_REASONS["model.kind"],
        "model.fixed_layers_from_bottom": DERIVED_REASONS["model.fixed_layers_from_bottom"],
        "modifications[0].operation": DERIVED_REASONS["modifications[0].operation"],
        "modifications[0].site_index": DERIVED_REASONS["modifications[0].site_index"],
    }
    if species:
        derived["modifications[0].anchor"] = DERIVED_REASONS["modifications[0].anchor"]
    request = PROSE_REQUESTS[ordinal % len(PROSE_REQUESTS)]
    return _build_case(case_id, "adsorbate", spec, request, sentence_by_field, derived)


def _defect_case(case_id: str, ordinal: int, element: str, crystal: str, facet: list[int], layers: int, vacuum: float, repeat: list[int], operation: str) -> list[dict[str, Any]]:
    modification: dict[str, Any] = {
        "operation": operation, "selector": {"layer": {"side": "top", "count": 1}, "ordinal": 1},
    }
    substitute_element = None
    if operation == "substitute":
        substitute_element = "Au" if element != "Au" else "Cu"
        modification["element"] = substitute_element
    spec = _surface_spec(element, crystal, facet, layers, vacuum, repeat, [modification])

    sentence_by_field = {
        "material.formula": _formula_sentence(element, ordinal, "surface"),
        "material.crystal_structure": _crystal_sentence(crystal, ordinal + 1),
        "model.miller_indices": _miller_sentence(facet, ordinal + 2),
        "model.supercell": _supercell_sentence(repeat, ordinal + 3),
        "model.layers": _layers_sentence(layers, ordinal + 4),
        "model.vacuum_angstrom": _vacuum_sentence(vacuum, ordinal + 5),
        "model.periodic_boundary_conditions": _pbc_sentence(False, ordinal + 6),
        "model.center": _center_sentence(ordinal + 7),
        "model.atom_ordering": _ordering_sentence(ordinal + 8),
        "requested_outputs": _outputs_sentence(ordinal + 9),
    }
    if substitute_element:
        sentence_by_field["modifications[0].element"] = _element_sentence(substitute_element, ordinal + 10)

    derived = {
        "model.kind": DERIVED_REASONS["model.kind"],
        "model.fixed_layers_from_bottom": DERIVED_REASONS["model.fixed_layers_from_bottom"],
        "modifications[0].operation": DERIVED_REASONS["modifications[0].operation"],
        "modifications[0].selector": DERIVED_REASONS["modifications[0].selector"],
    }
    request = PROSE_REQUESTS[ordinal % len(PROSE_REQUESTS)]
    return _build_case(case_id, "defect", spec, request, sentence_by_field, derived)


# --------------------------------------------------------------------------
# Pools. Layers are drawn from {4, 5, 6} and square supercells only ever
# render the digits {2, 3} -- disjoint ranges so a layer-count sentence
# ("four-layer slab", never a digit) can never coincide with a supercell
# digit that happens to equal a *different* case's layer count, and vice
# versa. Vacuum stays >= 10 (rendered as a two-digit whole number, and in
# any case exempt as a float) so it never collides either. This is what
# keeps `_assert_no_leak` from false-positiving on an unrelated field's
# digits appearing elsewhere in the same rendered paragraph.
# --------------------------------------------------------------------------
_FACETS = ([1, 1, 1], [1, 0, 0], [1, 1, 0])
_BULK_REPEATS = ([1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 2], [3, 1, 1])
_SURFACE_REPEATS = ([2, 2, 1], [3, 3, 1], [2, 3, 1], [3, 2, 1])
_LAYERS = (4, 5, 6)
_VACUUM = (10.0, 11.0, 12.0, 13.0, 15.0, 17.0, 18.0, 20.0)
_ADSORBATES = (
    ("O", None, None, 1.35), ("H", None, None, 1.15),
    (None, "CO", 2, 1.85), (None, "NH3", 1, 2.05), (None, "H2O", 1, 1.65),
)
_DEFECT_OPERATIONS = ("make_vacancy", "substitute")


def _generate() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for index, ((element, crystal), repeat) in enumerate(itertools.islice(
        itertools.cycle(itertools.product(PHASES, _BULK_REPEATS)), 12
    )):
        records.extend(_bulk_case(f"prose-bulk-{index:03d}", index, element, crystal, repeat))

    for index, ((element, crystal), facet, layers, vacuum) in enumerate(itertools.islice(
        itertools.cycle(itertools.product(PHASES, _FACETS, _LAYERS, _VACUUM)), 15
    )):
        records.extend(_surface_case(
            f"prose-surface-{index:03d}", index, element, crystal, facet, layers, vacuum,
            _SURFACE_REPEATS[index % len(_SURFACE_REPEATS)],
        ))

    for index, ((element, crystal), facet, adsorbate) in enumerate(itertools.islice(
        itertools.cycle(itertools.product(PHASES, _FACETS, _ADSORBATES)), 15
    )):
        atom, species, anchor, height = adsorbate
        vacuum = _VACUUM[index % len(_VACUUM)]
        records.extend(_adsorbate_case(
            f"prose-adsorbate-{index:03d}", index, element, crystal, facet, 4, vacuum,
            [2, 2, 1], atom, species, anchor, height,
        ))

    for index, ((element, crystal), facet, layers, operation) in enumerate(itertools.islice(
        itertools.cycle(itertools.product(PHASES, _FACETS, _LAYERS, _DEFECT_OPERATIONS)), 10
    )):
        vacuum = _VACUUM[index % len(_VACUUM)]
        records.extend(_defect_case(
            f"prose-defect-{index:03d}", index, element, crystal, facet, layers, vacuum,
            _SURFACE_REPEATS[index % len(_SURFACE_REPEATS)], operation,
        ))

    return records


def _write(directory: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in records)
    (directory / "test.jsonl").write_text(raw, encoding="utf-8")
    families: dict[str, int] = {}
    for record in records:
        family = record["split_group"].split(":", 1)[0]
        families[family] = families.get(family, 0) + 1
    manifest = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "generator": "training/generators/build_prose_holdout.py",
        "set_name": "prose_holdout", "evaluation_only": True,
        "record_count": len(records), "family_counts": families,
        "split_sha256": {"test": hashlib.sha256(raw.encode()).hexdigest()},
        "source_policy": "deterministic synthetic fixtures; no journal text",
        "design_section": "training/JOURNAL_ROLE_R3_SCHEDULE.md section 3, Phase 1",
        "variable": "value representation: prose requiring interpretation, no JSON literals",
        "no_literal_values": True,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("training/datasets/journal_holdout_prose"))
    args = parser.parse_args()
    records = _generate()
    manifest = _write(args.output_dir, records)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
