#!/usr/bin/env python3
"""Build a small replay-verified catalyst supplement from supported tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase_auto_build.ase_agent import create_default_registry
from training.dataset_contract import SCHEMA_VERSION, validate_record
from training.generators.generate_corpus import (
    RecipeCase,
    _assign_splits,
    _step,
    _write_jsonl,
    case_prompts,
    execute_case,
)


UNSUPPORTED_CATEGORIES: dict[str, str] = {}


def catalyst_cases() -> list[RecipeCase]:
    cases: list[RecipeCase] = []

    for element, crystal, miller, layers in (
        ("Pt", "fcc", (2, 1, 1), 6),
        ("Pt", "fcc", (3, 1, 1), 6),
        ("Cu", "fcc", (2, 1, 0), 5),
        ("Ni", "fcc", (3, 1, 0), 5),
        ("Fe", "bcc", (2, 1, 1), 6),
        ("W", "bcc", (3, 1, 0), 5),
    ):
        face = "".join(map(str, miller))
        case_id = f"high-index-{element.lower()}-{face}"
        cases.append(RecipeCase(
            case_id, "catalyst_high_index", element,
            (f"Build a 2x2 {element}({face}) high-index slab with {layers} layers and 14 A vacuum.",),
            (
                _step(
                    "build_surface", name="slab", element=element, crystal=crystal,
                    miller=list(miller), layers=layers, vacuum=14.0, repeat=[2, 2, 1],
                ),
                _step("finish", name="slab"),
            ),
        ))

    for prototype, miller, layers in (
        ("rocksalt-MgO", (1, 0, 0), 4),
        ("rocksalt-NiO", (1, 1, 0), 5),
        ("rocksalt-CoO", (1, 1, 1), 4),
        ("wurtzite-ZnO", (0, 0, 1), 5),
        ("fluorite-CeO2", (1, 1, 1), 4),
    ):
        formula = prototype.split("-", 1)[-1]
        face = "".join(map(str, miller))
        case_id = f"oxide-surface-{prototype.lower()}-{face}"
        cases.append(RecipeCase(
            case_id, "catalyst_oxide_surface", formula,
            (f"Build a 2x2 {prototype}({face}) oxide slab with {layers} layers and 14 A vacuum.",),
            (
                _step(
                    "build_surface", name="slab", prototype=prototype,
                    miller=list(miller), layers=layers, vacuum=14.0, repeat=[2, 2, 1],
                ),
                _step("finish", name="slab"),
            ),
        ))

    for site, site_index, adsorbate, height in (
        ("ontop", 2, "O", 1.7),
        ("bridge", 1, "O", 1.6),
        ("bridge", 3, "H", 1.2),
        ("hollow", 1, "N", 1.5),
        ("hollow", 1, "H", 1.1),
    ):
        case_id = f"pt111-{adsorbate.lower()}-{site}-{site_index}"
        cases.append(RecipeCase(
            case_id, "catalyst_adsorption_site", f"Pt-{adsorbate}",
            (
                f"Build a 2x2 Pt(111) slab with 4 layers and 14 A vacuum, then adsorb "
                f"one {adsorbate} atom at {site} site {site_index}, {height:g} A above it.",
            ),
            (
                _step(
                    "build_surface", name="slab", element="Pt", crystal="fcc",
                    miller=[1, 1, 1], layers=4, vacuum=14.0, repeat=[2, 2, 1],
                ),
                _step(
                    "add_atomic_adsorbate", name="slab", element=adsorbate,
                    site=site, site_index=site_index, height=height,
                ),
                _step("finish", name="slab"),
            ),
        ))

    for site, site_index, species, anchor, height in (
        ("ontop", 3, "CO", 1, 1.9),
        ("bridge", 2, "CO", 1, 1.8),
        ("hollow", 1, "NH3", 1, 2.0),
    ):
        case_id = f"pt111-{species.lower()}-{site}-{site_index}"
        cases.append(RecipeCase(
            case_id, "catalyst_adsorption_site", f"Pt-{species}",
            (
                f"Build a 2x2 Pt(111) slab with 4 layers and 14 A vacuum, then adsorb "
                f"{species} at {site} site {site_index} through anchor atom {anchor}, "
                f"{height:g} A above it.",
            ),
            (
                _step(
                    "build_surface", name="slab", element="Pt", crystal="fcc",
                    miller=[1, 1, 1], layers=4, vacuum=14.0, repeat=[2, 2, 1],
                ),
                _step(
                    "add_molecular_adsorbate", name="slab", species=species,
                    anchor=anchor, site=site, site_index=site_index, height=height,
                ),
                _step("finish", name="slab"),
            ),
        ))

    for element, shape, shells in (
        ("Pt", "icosahedron", 2),
        ("Au", "cuboctahedron", 2),
        ("Ni", "octahedron", 1),
    ):
        cases.append(RecipeCase(
            f"nanoparticle-{element.lower()}-{shape}-{shells}",
            "catalyst_nanoparticle",
            element,
            (f"Build a {shells}-shell {shape} {element} nanoparticle with 8 A vacuum.",),
            (
                _step(
                    "build_nanoparticle", name="particle", element=element,
                    shape=shape, shells=shells, vacuum=8.0,
                ),
                _step("finish", name="particle"),
            ),
        ))

    for element, shape, shells, support, facet, layers, gap in (
        ("Pt", "icosahedron", 2, "fluorite-CeO2", (1, 1, 1), 4, 2.2),
        ("Au", "cuboctahedron", 1, "rocksalt-MgO", (1, 0, 0), 4, 2.5),
        ("Ni", "octahedron", 1, "rocksalt-NiO", (1, 0, 0), 4, 2.3),
    ):
        face = "".join(map(str, facet))
        cases.append(RecipeCase(
            f"supported-{element.lower()}-{shape}-{support.lower()}-{face}",
            "catalyst_supported_cluster",
            f"{element}-{support}",
            (
                f"Build a {shells}-shell {shape} {element} cluster supported on a "
                f"2x2 {support}({face}) slab with {layers} layers and 14 A vacuum, "
                f"using a {gap:g} A interface gap.",
            ),
            (
                _step(
                    "build_surface", name="support", element=support,
                    miller=list(facet), layers=layers, vacuum=14.0,
                    repeat=[2, 2, 1],
                ),
                _step(
                    "build_nanoparticle", name="cluster", element=element,
                    shape=shape, shells=shells, vacuum=8.0,
                ),
                _step(
                    "combine", name="supported", host="support", guest="cluster",
                    mode="stack", gap=gap, vacuum=12.0,
                ),
                _step("finish", name="supported"),
            ),
        ))

    surface = _step(
        "build_surface", name="slab", element="Pt", crystal="fcc",
        miller=[1, 1, 1], layers=4, vacuum=14.0, repeat=[2, 2, 1],
    )
    cases.extend((
        RecipeCase(
            "surface-vacancy-pt111-top-first", "catalyst_surface_vacancy", "Pt",
            ("Build a 2x2 four-layer Pt(111) slab with 14 A vacuum and remove the first atom in its top layer.",),
            (
                surface,
                _step(
                    "make_vacancy", name="slab",
                    selector={"layer": {"side": "top", "count": 1}, "ordinal": 1},
                ),
                _step("finish", name="slab"),
            ),
        ),
        RecipeCase(
            "surface-substitution-pt111-au-top-first",
            "catalyst_surface_substitution", "Pt-Au",
            ("Build a 2x2 four-layer Pt(111) slab with 14 A vacuum and substitute Au for the first atom in its top layer.",),
            (
                surface,
                _step(
                    "substitute", name="slab", element="Au",
                    selector={"layer": {"side": "top", "count": 1}, "ordinal": 1},
                ),
                _step("finish", name="slab"),
            ),
        ),
        RecipeCase(
            "surface-vacancy-ceo2-111-top-o-first",
            "catalyst_surface_vacancy", "Ce-O",
            ("Build a 2x2 four-layer fluorite-CeO2(111) slab with 14 A vacuum and remove the first O atom in its top layer.",),
            (
                _step(
                    "build_surface", name="slab", element="fluorite-CeO2",
                    miller=[1, 1, 1], layers=4, vacuum=14.0, repeat=[2, 2, 1],
                ),
                _step(
                    "make_vacancy", name="slab",
                    selector={
                        "element": ["O"],
                        "layer": {"side": "top", "count": 1},
                        "ordinal": 1,
                    },
                ),
                _step("finish", name="slab"),
            ),
        ),
    ))
    for element, crystal, facet, dopant in (
        ("Cu", "fcc", (1, 0, 0), "Ag"),
        ("Ni", "fcc", (1, 1, 0), "Co"),
    ):
        face = "".join(map(str, facet))
        build = _step(
            "build_surface", name="slab", element=element, crystal=crystal,
            miller=list(facet), layers=4, vacuum=14.0, repeat=[2, 2, 1],
        )
        cases.extend((
            RecipeCase(
                f"surface-vacancy-{element.lower()}{face}-top-first",
                "catalyst_surface_vacancy", element,
                (f"Build a 2x2 four-layer {element}({face}) slab with 14 A vacuum and remove the first atom in its top layer.",),
                (
                    build,
                    _step(
                        "make_vacancy", name="slab",
                        selector={"layer": {"side": "top", "count": 1}, "ordinal": 1},
                    ),
                    _step("finish", name="slab"),
                ),
            ),
            RecipeCase(
                f"surface-substitution-{element.lower()}{face}-{dopant.lower()}-top-first",
                "catalyst_surface_substitution", f"{element}-{dopant}",
                (f"Build a 2x2 four-layer {element}({face}) slab with 14 A vacuum and substitute {dopant} for the first atom in its top layer.",),
                (
                    build,
                    _step(
                        "substitute", name="slab", element=dopant,
                        selector={"layer": {"side": "top", "count": 1}, "ordinal": 1},
                    ),
                    _step("finish", name="slab"),
                ),
            ),
        ))

    return sorted(cases, key=lambda case: (case.family, case.case_id))


def build_records() -> list[dict]:
    registry = create_default_registry()
    records = []
    for case in catalyst_cases():
        prompt = case_prompts(case, {}, registry, max_prompts=1)[0]
        record = execute_case(case, prompt, 0)
        record["provenance"]["source"] = "catalyst_supplement_generator"
        validate_record(record)
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("training/datasets/catalyst_supplement_v2"),
    )
    args = parser.parse_args()

    records = build_records()
    split_records = _assign_splits(records)
    if any(not items for items in split_records.values()):
        raise RuntimeError("catalyst supplement produced an empty split")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        split: _write_jsonl(args.output_dir / f"{split}.jsonl", items)
        for split, items in split_records.items()
    }
    registry = create_default_registry()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": "training/generators/build_catalyst_supplement.py",
        "registry_version": f"phase1-{registry.fingerprint()[:16]}",
        "record_count": len(records),
        "split_counts": {split: len(items) for split, items in split_records.items()},
        "split_sha256": hashes,
        "unsupported_categories": UNSUPPORTED_CATEGORIES,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
