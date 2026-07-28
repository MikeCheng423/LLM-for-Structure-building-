#!/usr/bin/env python3
"""Build the r5 replay corpus plus validated catalyst supplements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase_auto_build.ase_agent import create_default_registry
from training.dataset_contract import SCHEMA_VERSION
from training.generators.build_catalyst_supplement import catalyst_cases
from training.generators.catalyst_supplement_ambiguity import cases as ambiguity_cases
from training.generators.generate_corpus import (
    _assign_splits,
    _write_jsonl,
    case_prompts,
    cases as replay_cases,
    execute_case,
    load_templates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/datasets/catalyst_supplement_r2"),
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=Path("training/generators/paraphrase_templates"),
    )
    args = parser.parse_args()

    registry = create_default_registry()
    templates = load_templates(args.templates_dir)
    sources = {
        "r5_replay": replay_cases(),
        "catalyst_novel": catalyst_cases(),
        "clarification_multilingual": [
            case for split in ambiguity_cases().values() for case in split
        ],
    }
    records = []
    source_counts = {}
    for source, cases in sources.items():
        before = len(records)
        prompt_cap = 6 if source == "r5_replay" else 1
        for case in cases:
            prompts = case_prompts(case, templates, registry, max_prompts=prompt_cap)
            for index, prompt in enumerate(prompts):
                record = execute_case(case, prompt, index)
                record["provenance"]["source"] = source
                records.append(record)
        source_counts[source] = len(records) - before

    split_records = _assign_splits(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        split: _write_jsonl(args.output_dir / f"{split}.jsonl", items)
        for split, items in split_records.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "registry_version": f"phase1-{registry.fingerprint()[:16]}",
        "generator": "training/generators/build_catalyst_training_corpus.py",
        "scope": "bounded_catalyst_supplement_pilot",
        "record_count": len(records),
        "source_counts": source_counts,
        "split_strategy": "stratified_family_coverage_grouped_by_output_hash/v1",
        "split_counts": {split: len(items) for split, items in split_records.items()},
        "split_sha256": hashes,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
