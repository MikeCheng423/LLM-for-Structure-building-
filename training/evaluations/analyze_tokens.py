#!/usr/bin/env python3
"""Measure rendered trajectory lengths with the pinned training tokenizer."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from training.dataset_contract import load_journal_jsonl, load_jsonl, render_assistant_only
from training.train_qlora import DEFAULT_MODEL, DEFAULT_REVISION


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _stats(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "minimum": min(values, default=0),
        "mean": round(statistics.fmean(values), 3) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "maximum": max(values, default=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("training/cache/huggingface"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--corpus-contract", choices=("ase", "journal"), default="ase")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    by_split: dict[str, list[int]] = {}
    assistant_by_split: dict[str, list[int]] = {}
    ids_by_length: list[tuple[int, str]] = []
    for split in ("train", "validation", "test"):
        load_records = load_journal_jsonl if args.corpus_contract == "journal" else load_jsonl
        records, _ = load_records(args.dataset_dir / f"{split}.jsonl")
        lengths: list[int] = []
        assistant_lengths: list[int] = []
        for record in records:
            rendered = render_assistant_only(tokenizer, record, max_length=100_000)
            length = len(rendered["input_ids"])
            assistant_length = sum(label != -100 for label in rendered["labels"])
            lengths.append(length)
            assistant_lengths.append(assistant_length)
            ids_by_length.append((length, record["id"]))
        by_split[split] = lengths
        assistant_by_split[split] = assistant_lengths

    all_lengths = [value for values in by_split.values() for value in values]
    all_assistant = [value for values in assistant_by_split.values() for value in values]
    report = {
        "model": args.model,
        "revision": args.revision,
        "total_tokens": _stats(all_lengths),
        "assistant_loss_tokens": _stats(all_assistant),
        "by_split": {
            split: {
                "total_tokens": _stats(by_split[split]),
                "assistant_loss_tokens": _stats(assistant_by_split[split]),
            }
            for split in by_split
        },
        "over_limit": {
            str(limit): sum(length > limit for length in all_lengths)
            for limit in (512, 768, 1024, 1536, 2048, 4096)
        },
        "longest_records": [
            {"id": record_id, "tokens": length}
            for length, record_id in sorted(ids_by_length, reverse=True)[:20]
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
