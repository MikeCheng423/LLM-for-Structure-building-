#!/usr/bin/env python3
"""Build the Catalyst ambiguity/Traditional-Chinese supplemental slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ase_auto_build.ase_agent import create_default_registry
from ase_auto_build.ase_agent.tool_router import route_tools

from training.dataset_contract import SCHEMA_VERSION, validate_record
from training.evaluations.evaluate_corpus import evaluate_record
from training.generators.generate_corpus import RecipeCase, _step, _write_jsonl, execute_case


def cases() -> dict[str, tuple[RecipeCase, ...]]:
    """Controller-valid cases with one material choice withheld, then supplied."""
    return {
        "train": (
            RecipeCase(
                "catalyst-supplement-fe-facet-zh", "clarification", "Fe",
                ("請建立鐵的 surface；facet 尚未指定，請先 clarify 再建立。",),
                (
                    _step("ask_clarification", question="要建立哪一個鐵表面？", choices=["bcc Fe(110)", "bcc Fe(100)"], field="miller", followup_user="請使用 bcc Fe(110)，四層，真空層 12 Å。"),
                    _step("build_surface", name="slab", element="Fe", crystal="bcc", miller=[1, 1, 0], layers=4, vacuum=12.0),
                    _step("finish", name="slab"),
                ),
            ),
            RecipeCase(
                "catalyst-supplement-carbon-form-zh", "clarification", "C",
                ("請建立碳材料；結構形式不明，請先 clarify 再 building。",),
                (
                    _step("ask_clarification", question="要建立哪一種碳結構？", choices=["diamond bulk", "graphene sheet", "carbon nanotube"], field="structure_type", followup_user="請建立 graphene sheet。"),
                    _step("build_prototype", name="sheet", prototype="graphene"),
                    _step("finish", name="sheet"),
                ),
            ),
            RecipeCase(
                "catalyst-supplement-nacl-form", "clarification", "Na-Cl",
                ("Create NaCl, but clarify whether it is an isolated pair or a crystal before building.",),
                (
                    _step("ask_clarification", question="Should NaCl be an isolated pair or a rocksalt crystal?", choices=["isolated pair", "rocksalt crystal"], field="structure_type", followup_user="Use a rocksalt crystal with a 5.64 A cubic lattice."),
                    _step("build_crystal", name="rocksalt", symbols=["Na", "Cl"], basis=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], spacegroup=225, a=5.64),
                    _step("finish", name="rocksalt"),
                ),
            ),
        ),
        "validation": (
            RecipeCase(
                "catalyst-supplement-cu-facet", "clarification", "Cu",
                ("Build a copper slab, but ask which facet before choosing it.",),
                (
                    _step("ask_clarification", question="Which copper facet should be built?", choices=["fcc Cu(100)", "fcc Cu(111)"], field="miller", followup_user="Use fcc Cu(100), three layers, and 10 A vacuum."),
                    _step("build_surface", name="slab", element="Cu", crystal="fcc", miller=[1, 0, 0], layers=3, vacuum=10.0),
                    _step("finish", name="slab"),
                ),
            ),
            RecipeCase(
                "catalyst-supplement-si-form-zh", "clarification", "Si",
                ("請建立矽結構；bulk 或其他形式尚未決定，請先 clarify。",),
                (
                    _step("ask_clarification", question="要使用哪一種矽結構？", choices=["diamond bulk", "isolated cluster"], field="structure_type", followup_user="請使用 diamond bulk 的 conventional cubic cell。"),
                    _step("build_bulk", name="bulk", element="Si", crystal="diamond", cubic=True),
                    _step("finish", name="bulk"),
                ),
            ),
            RecipeCase(
                "catalyst-supplement-cnt-chirality", "clarification", "C",
                ("Build a carbon nanotube, but clarify its chirality before building it.",),
                (
                    _step("ask_clarification", question="Which nanotube chirality should be used?", choices=["(6,0)", "(6,6)"], field="chirality", followup_user="Use a (6,0) tube, two unit cells long, with 10 A radial vacuum."),
                    _step("build_nanotube", name="tube", element="C", n=6, m=0, length=2, vacuum=10.0),
                    _step("finish", name="tube"),
                ),
            ),
        ),
        "test": (
            RecipeCase(
                "catalyst-supplement-pt-facet-zh", "clarification", "Pt",
                ("請建立鉑的 slab；表面方向有歧義，請先 clarify 再 building。",),
                (
                    _step("ask_clarification", question="要建立哪一個鉑表面？", choices=["fcc Pt(111)", "fcc Pt(100)"], field="miller", followup_user="請使用 fcc Pt(111)，五層，真空層 14 Å。"),
                    _step("build_surface", name="slab", element="Pt", crystal="fcc", miller=[1, 1, 1], layers=5, vacuum=14.0),
                    _step("finish", name="slab"),
                ),
            ),
            RecipeCase(
                "catalyst-supplement-au-form", "clarification", "Au",
                ("Create gold, but clarify its structural form before building.",),
                (
                    _step("ask_clarification", question="Which structural form of gold is required?", choices=["fcc bulk", "surface slab"], field="structure_type", followup_user="Use an fcc conventional cubic bulk cell."),
                    _step("build_bulk", name="bulk", element="Au", crystal="fcc", cubic=True),
                    _step("finish", name="bulk"),
                ),
            ),
            RecipeCase(
                "catalyst-supplement-al-form-zh", "clarification", "Al",
                ("請建立鋁材料；bulk 或 surface 尚未指定，請先 clarify。",),
                (
                    _step("ask_clarification", question="要建立鋁的 bulk 還是 surface？", choices=["fcc bulk", "surface slab"], field="structure_type", followup_user="請使用 fcc conventional cubic bulk，並重複為 2×2×1。"),
                    _step("build_bulk", name="bulk", element="Al", crystal="fcc", cubic=True),
                    _step("repeat", name="bulk", repeat=[2, 2, 1]),
                    _step("finish", name="bulk"),
                ),
            ),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("training/datasets/catalyst_supplement_ambiguity"))
    args = parser.parse_args()
    registry = create_default_registry()
    split_records: dict[str, list[dict]] = {}

    for split, split_cases in cases().items():
        records = []
        for case in split_cases:
            prompt = case.descriptions[0]
            routed = {item["function"]["name"] for item in route_tools(prompt, registry)}
            needed = {step["tool"] for step in case.steps}
            if not needed <= routed:
                raise RuntimeError(f"{case.case_id}: router omits {sorted(needed - routed)}")
            record = execute_case(case, prompt, 0)
            record["provenance"]["source"] = "catalyst_supplement_controller_generator"
            validate_record(record)
            evaluate_record(record)
            records.append(record)
        split_records[split] = records

    output_sets = {
        split: {record["validation"]["output_hash"] for record in records}
        for split, records in split_records.items()
    }
    if any(output_sets[a] & output_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("final-structure leakage across splits")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {split: _write_jsonl(args.output_dir / f"{split}.jsonl", records) for split, records in split_records.items()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "registry_version": f"phase1-{registry.fingerprint()[:16]}",
        "split_strategy": "curated_catalyst_supplement_no_output_overlap/v1",
        "record_count": sum(map(len, split_records.values())),
        "split_counts": {split: len(records) for split, records in split_records.items()},
        "split_sha256": hashes,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "failures": [],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
