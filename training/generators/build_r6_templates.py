#!/usr/bin/env python3
"""Assemble the r6 phrasing pools and prove every template is rule-conforming.

Two disjoint pools are written:

* ``paraphrase_templates_r6/``          -- TRAIN phrasing. The proven r5 templates
  plus a handful of new paraphrase forms per family (extends training diversity).
* ``paraphrase_templates_r6_heldout/``  -- HELD-OUT phrasing. Entirely new surface
  forms, never used in training, used only to build the novel-phrasing eval set.

Both pools obey the structured-request rule: for every real case in each family
we fill the template and assert ``request_rule.missing_slots`` is empty. A template
that cannot be filled for a case (missing placeholder) is allowed -- the generator
simply skips it there -- but a template that fills yet drops a required slot is a
bug and fails this build. Templates only rephrase; they never author a tool call.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from training.generators import generate_corpus as gc
from training.generators.template_fill import case_slots, format_dict
from training.generators import request_rule as rr

HERE = Path(__file__).resolve().parent
R5_DIR = HERE / "paraphrase_templates"
TRAIN_DIR = HERE / "paraphrase_templates_r6"
HELDOUT_DIR = HERE / "paraphrase_templates_r6_heldout"

# --- New TRAIN paraphrase forms (added on top of the proven r5 templates). ---
TRAIN_NEW: dict[str, list[str]] = {
    "bulk": [
        "Fashion a {repeat3} bulk cell of {element} in the {crystalline} structure.",
        "Replicate {crystalline} {element} across a {repeat3} bulk supercell.",
        "Model bulk {element} ({crystalline} phase) as a {repeat3} repeat.",
        "Create a {repeat3} periodic bulk block of {crystalline} {element}.",
    ],
    "vacancy": [
        "Remove the {defect_ordinal} atom from {repeat3} {crystalline} {element} to seed a vacancy.",
        "Form a monovacancy in a {repeat3} {crystalline} {element} supercell at the {defect_ordinal} site.",
        "Build {repeat3} {crystalline} {element} and delete its {defect_ordinal} atom.",
        "Knock the {defect_ordinal} atom out of a {repeat3} {crystalline} {element} cell.",
    ],
    "substitution": [
        "Substitute {dopant} for the {defect_ordinal} atom in {repeat3} {crystalline} {element}.",
        "Replace the {defect_ordinal} atom of a {repeat3} {crystalline} {element} supercell with {dopant}.",
        "Dope a {repeat3} {crystalline} {element} block, putting {dopant} at the {defect_ordinal} atom.",
        "Turn the {defect_ordinal} atom of {repeat3} {crystalline} {element} into a {dopant} atom.",
    ],
    "surface": [
        "Build a {suffix} {layers}-layer {element}({facet}) slab with {vacuum} Å vacuum.",
        "Prepare the {element}({facet}) surface as a {suffix} slab, {layers} layers, {vacuum} angstrom vacuum.",
        "Generate {element}({facet}) with {layers} layers and {vacuum} Å of vacuum in a {suffix} cell.",
        "Cut a {suffix} {element}({facet}) slab, {layers} layers thick, {vacuum} A vacuum.",
    ],
    "surface_constraint": [
        "Build a {suffix} {element}({facet}) slab ({layers} layers, {vacuum} Å vacuum) and freeze the {freeze_count} {freeze_side} layers.",
        "Prepare {element}({facet}), {suffix}, {layers} layers, {vacuum} angstrom vacuum, then fix its {freeze_side} {freeze_count} layers.",
        "Make a {layers}-layer {suffix} {element}({facet}) slab with {vacuum} Å vacuum and constrain the {freeze_count} {freeze_side} layers.",
        "Generate a {suffix} {element}({facet}) surface, {layers} layers, {vacuum} A vacuum, holding the {freeze_side} {freeze_count} layers fixed.",
    ],
    "atomic_adsorption": [
        "Adsorb one {adsorbate} atom at the {site} site {height} Å above a {suffix} {element}({facet}) slab of {layers} layers with {vacuum} Å vacuum.",
        "Build {suffix} {element}({facet}), {layers} layers, {vacuum} angstrom vacuum, and place {adsorbate} at the {site} site {height} angstrom high.",
        "Put a single {adsorbate} atom on the {site} site, {height} Å above a {layers}-layer {suffix} {element}({facet}) slab with {vacuum} Å vacuum.",
        "On a {suffix} {element}({facet}) surface ({layers} layers, {vacuum} A vacuum), add {adsorbate} at the {site} site {height} A above.",
    ],
    "molecular_adsorption": [
        "Adsorb {adsorbate} at the {site} site of a {suffix} {element}({facet}) slab ({layers} layers, {vacuum} Å vacuum), anchored through its {anchor} {height} Å above.",
        "Build {suffix} {element}({facet}), {layers} layers, {vacuum} angstrom vacuum, and place {adsorbate} at the {site} site through the {anchor} {height} angstrom up.",
        "Put {adsorbate} on the {site} site of a {layers}-layer {suffix} {element}({facet}) slab with {vacuum} Å vacuum, anchored via {anchor} {height} Å high.",
        "On a {suffix} {element}({facet}) slab ({layers} layers, {vacuum} A vacuum), adsorb {adsorbate} at the {site} site with its {anchor} {height} A above.",
    ],
    "molecule": [
        "Create an isolated {species} molecule within a {box} Å cubic box.",
        "Place {species} alone in a cubic box of {box} angstrom.",
        "Set up gas-phase {species} in a {box} Å cubic cell.",
        "Model an isolated {species} molecule using a {box}-angstrom cube.",
    ],
    "nanotube": [
        "Construct a {chirality} carbon nanotube spanning {length} unit cells.",
        "Create a {chirality} single-walled CNT, {length} unit cells long.",
        "Generate a carbon nanotube of chirality {chirality}, {length} unit cells in length.",
        "Make a {length}-unit-cell {chirality} carbon nanotube.",
    ],
    "prototype": [
        "Build the {prototype} prototype structure.",
        "Construct the conventional cell of the {prototype} prototype.",
        "Generate a unit cell for the {prototype} prototype.",
        "Set up the {prototype} prototype.",
    ],
}

# --- HELD-OUT paraphrase forms: novel surface forms, never used in training. ---
HELDOUT: dict[str, list[str]] = {
    "bulk": [
        "Tile {crystalline} {element} into a {repeat3} bulk supercell.",
        "Give me bulk {element} ({crystalline}) replicated {repeat3}.",
        "Lay out a {repeat3} periodic cell of {crystalline}-structured {element}.",
        "I want the {crystalline} allotrope of {element} as a {repeat3} bulk block.",
        "Stack {crystalline} {element} into a {repeat3} crystalline supercell.",
        "Produce bulk crystalline {element} in the {crystalline} lattice, tiled {repeat3}.",
        "Expand a {crystalline} {element} unit cell to {repeat3} for a bulk model.",
        "Deliver a {repeat3} bulk arrangement of {element} with {crystalline} packing.",
    ],
    "vacancy": [
        "Carve a vacancy into {repeat3} {crystalline} {element} by deleting its {defect_ordinal} atom.",
        "From a {repeat3} {crystalline} {element} supercell, pull out the {defect_ordinal} atom.",
        "Introduce a single vacancy at the {defect_ordinal} site of {repeat3} {crystalline} {element}.",
        "Take {crystalline} {element}, build it {repeat3}, and vacate the {defect_ordinal} atomic position.",
        "Excise the {defect_ordinal} atom from a {repeat3} supercell of {crystalline} {element}.",
        "Assemble {repeat3} {crystalline} {element} and leave a hole where the {defect_ordinal} atom was.",
        "In a {repeat3} {crystalline} {element} cell, drop the {defect_ordinal} atom to make a vacancy.",
        "Prepare a vacancy-bearing {repeat3} {crystalline} {element} supercell missing its {defect_ordinal} atom.",
    ],
    "substitution": [
        "Swap the {defect_ordinal} atom of {repeat3} {crystalline} {element} for a {dopant} atom.",
        "In {repeat3} {crystalline} {element}, put {dopant} where the {defect_ordinal} atom sits.",
        "Dope {crystalline} {element} (tiled {repeat3}) by replacing its {defect_ordinal} atom with {dopant}.",
        "Build {repeat3} {crystalline} {element} and convert the {defect_ordinal} atom into {dopant}.",
        "Introduce a {dopant} substituent at the {defect_ordinal} site of a {repeat3} {crystalline} {element} supercell.",
        "Make {repeat3} {crystalline} {element}, then trade the {defect_ordinal} atom for {dopant}.",
        "Replace the very {defect_ordinal} atom of {repeat3} {crystalline} {element} with {dopant}.",
        "Prepare a {dopant}-doped {repeat3} {crystalline} {element} cell where the {defect_ordinal} atom becomes {dopant}.",
    ],
    "surface": [
        "Cleave a {suffix} {element}({facet}) surface {layers} layers deep with {vacuum} Å of vacuum.",
        "Slab out {element} along ({facet}): {layers} layers, {suffix} in plane, {vacuum} angstrom vacuum.",
        "I need a {element}({facet}) slab — {layers} atomic layers, {suffix} lateral cell, {vacuum} Å vacuum spacing.",
        "Terminate {element} on its ({facet}) face as a {suffix} slab of {layers} layers under {vacuum} A vacuum.",
        "Construct a {suffix}, {layers}-layer {element}({facet}) surface separated by {vacuum} angstrom of vacuum.",
        "Prepare {element}({facet}), {layers} layers thick, {suffix} supercell, capped with {vacuum} Å vacuum.",
        "Give me a {vacuum}-angstrom-vacuum {element}({facet}) slab, {layers} layers, repeated {suffix}.",
        "Set up the ({facet}) surface of {element} with {layers} layers, {suffix} periodicity, and {vacuum} Å vacuum.",
    ],
    "surface_constraint": [
        "Cleave a {suffix} {element}({facet}) slab, {layers} layers, {vacuum} Å vacuum, and immobilize the {freeze_count} {freeze_side} layers.",
        "Build {element}({facet}) as a {suffix} slab ({layers} layers, {vacuum} angstrom vacuum) and hold the {freeze_side} {freeze_count} layers fixed.",
        "A {layers}-layer {suffix} {element}({facet}) surface with {vacuum} Å vacuum — pin its {freeze_count} {freeze_side} layers.",
        "Set up {element}({facet}), {suffix}, {layers} layers, {vacuum} A vacuum, then lock the {freeze_count} layers at the {freeze_side}.",
        "Prepare a constrained {suffix} {element}({facet}) slab of {layers} layers with {vacuum} Å vacuum, freezing the {freeze_side} {freeze_count} layers.",
        "Make {element}({facet}) with {layers} layers, {suffix} cell, {vacuum} angstrom vacuum, and clamp the {freeze_count} {freeze_side} layers.",
        "I want a {suffix} {element}({facet}) slab ({layers} layers, {vacuum} Å vacuum) whose {freeze_count} {freeze_side} layers are frozen.",
        "Cleave {element}({facet}) into a {suffix}, {layers}-layer slab with {vacuum} A vacuum and constrain its {freeze_side} {freeze_count} layers.",
    ],
    "atomic_adsorption": [
        "Adsorb a lone {adsorbate} atom at the {site} site, {height} Å above a {suffix} {element}({facet}) slab of {layers} layers with {vacuum} Å vacuum.",
        "On a {suffix} {element}({facet}) slab ({layers} layers, {vacuum} angstrom vacuum), perch one {adsorbate} atom {height} angstrom over the {site} site.",
        "Place {adsorbate} at height {height} Å on the {site} site of a {layers}-layer {suffix} {element}({facet}) surface with {vacuum} Å vacuum.",
        "Build {element}({facet}), {suffix}, {layers} layers, {vacuum} A vacuum, and sit an {adsorbate} atom {height} A above the {site} site.",
        "Decorate a {suffix} {element}({facet}) slab ({layers} layers, {vacuum} Å vacuum) with a single {adsorbate} at the {site} site {height} Å up.",
        "I need {adsorbate} bound {height} angstrom above the {site} site of a {suffix} {element}({facet}) slab, {layers} layers, {vacuum} angstrom vacuum.",
        "Put an {adsorbate} adatom on the {site} site — {height} Å above — of a {layers}-layer {suffix} {element}({facet}) slab with {vacuum} Å vacuum.",
        "For a {suffix} {element}({facet}) surface of {layers} layers and {vacuum} A vacuum, seat {adsorbate} at the {site} site {height} A high.",
    ],
    "molecular_adsorption": [
        "Dock {adsorbate} at the {site} site of a {suffix} {element}({facet}) slab ({layers} layers, {vacuum} Å vacuum), tethered through its {anchor} {height} Å above the surface.",
        "On a {layers}-layer {suffix} {element}({facet}) slab with {vacuum} angstrom vacuum, bind {adsorbate} at the {site} site via its {anchor}, {height} angstrom up.",
        "Attach {adsorbate} to a {suffix} {element}({facet}) surface ({layers} layers, {vacuum} Å vacuum) at the {site} site, its {anchor} anchored {height} Å high.",
        "Build {element}({facet}), {suffix}, {layers} layers, {vacuum} A vacuum, then seat {adsorbate} at the {site} site through the {anchor} atom {height} A above.",
        "Place the {adsorbate} molecule at the {site} site of a {suffix} {element}({facet}) slab of {layers} layers with {vacuum} Å vacuum, anchored via {anchor} {height} Å over the top layer.",
        "I want {adsorbate} chemisorbed at the {site} site of a {suffix} {element}({facet}) slab ({layers} layers, {vacuum} angstrom vacuum), its {anchor} {height} angstrom above.",
        "Adsorb {adsorbate} through {anchor} at the {site} site, {height} Å above a {layers}-layer {suffix} {element}({facet}) surface with {vacuum} Å vacuum.",
        "For a {suffix} {element}({facet}) slab ({layers} layers, {vacuum} A vacuum), coordinate {adsorbate} at the {site} site with the {anchor} atom held {height} A above.",
    ],
    "molecule": [
        "Isolate a {species} molecule inside a cubic {box} Å box.",
        "Drop one {species} molecule into a {box}-angstrom cubic cell.",
        "Model gas-phase {species} in a cube measuring {box} Å per edge.",
        "Enclose a single {species} molecule in a {box} angstrom cubic box.",
        "Set {species} alone at the center of a {box} Å cubic cell.",
        "I need isolated {species} within a cubic simulation box of {box} angstrom.",
        "Build one {species} molecule surrounded by vacuum in a {box} Å cube cell.",
        "Put {species} by itself in a {box}-Å cubic box.",
    ],
    "nanotube": [
        "Roll a {chirality} carbon nanotube extending {length} unit cells.",
        "Generate a single-walled CNT of chirality {chirality}, length {length} unit cells.",
        "I want a {chirality} nanotube that runs {length} unit cells long.",
        "Build a carbon nanotube, chirality {chirality}, spanning {length} repeat cells.",
        "Make a {length}-unit-cell {chirality} single-walled carbon nanotube.",
        "Produce a {chirality} CNT with an axial length of {length} unit cells.",
        "Assemble tube chirality {chirality}, {length} unit cells in length.",
        "Construct a {chirality} carbon nanotube {length} cells long.",
    ],
    "prototype": [
        "Instantiate the {prototype} prototype cell.",
        "I need the standard {prototype} structure.",
        "Lay down the conventional cell of {prototype}.",
        "Produce a unit cell for {prototype} from the prototype library.",
        "Give me {prototype} as its prototype unit cell.",
        "Render {prototype} using the named-prototype builder.",
        "Generate the {prototype} prototype's atomic cell.",
        "Set up the crystalline unit cell of the {prototype} prototype.",
    ],
}


def _load_r5(family: str) -> list[str]:
    path = R5_DIR / f"{family}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _cases_by_family() -> dict[str, list[gc.RecipeCase]]:
    grouped: dict[str, list[gc.RecipeCase]] = defaultdict(list)
    for case in gc.cases():
        grouped[case.family].append(case)
    return grouped


def _validate(family: str, templates: list[str], cases: list[gc.RecipeCase]) -> int:
    """Every template that *fills* for a case must be rule-conforming. Return #fills."""
    fills = 0
    for case in cases:
        steps = [{"tool": s["tool"], "args": s["args"]} for s in case.steps]
        fmt = format_dict(case_slots(steps))
        values = rr.slot_values(steps)
        for template in templates:
            try:
                prompt = template.format(**fmt).strip()
            except (KeyError, IndexError):
                continue  # placeholder not provided by this case; generator skips it too
            fills += 1
            missing = rr.missing_slots(family, values, prompt)
            if missing:
                raise SystemExit(
                    f"[{family}] template drops required slot(s) {missing}:\n"
                    f"  template: {template!r}\n  filled:   {prompt!r}\n  case: {case.case_id}"
                )
    return fills


def _write_pool(out_dir: Path, pool: dict[str, list[str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for family, templates in pool.items():
        (out_dir / f"{family}.json").write_text(
            json.dumps(templates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def main() -> int:
    by_family = _cases_by_family()
    templatable = set(gc.FAMILY_PLACEHOLDERS) if hasattr(gc, "FAMILY_PLACEHOLDERS") else set(HELDOUT)

    train_pool: dict[str, list[str]] = {}
    heldout_pool: dict[str, list[str]] = {}
    for family in sorted(HELDOUT):
        r5 = _load_r5(family)
        new_train = TRAIN_NEW[family]
        # New forms first so a bumped per-case cap is guaranteed to include them.
        train = list(dict.fromkeys(new_train + r5))
        heldout = list(dict.fromkeys(HELDOUT[family]))

        overlap = set(train) & set(heldout)
        if overlap:
            raise SystemExit(f"[{family}] train/held-out template overlap: {overlap}")

        cases = by_family[family]
        n_train = _validate(family, train, cases)
        n_heldout = _validate(family, heldout, cases)
        print(
            f"{family:22s} train={len(train):2d} tmpl ({n_train:3d} fills)  "
            f"heldout={len(heldout):2d} tmpl ({n_heldout:3d} fills)  cases={len(cases)}"
        )
        train_pool[family] = train
        heldout_pool[family] = heldout

    _write_pool(TRAIN_DIR, train_pool)
    _write_pool(HELDOUT_DIR, heldout_pool)
    print(f"\nWrote TRAIN pool  -> {TRAIN_DIR}")
    print(f"Wrote HELD-OUT pool -> {HELDOUT_DIR}")
    print(f"Families: {len(train_pool)} (templatable set has {len(templatable)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
