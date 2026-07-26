#!/usr/bin/env python3
"""Build the novel-phrasing evaluation file from a held-out template corpus.

The novel-phrasing probe asks whether the adapter generalizes to phrasings it
never trained on. A corpus generated from the held-out template pool is *mostly*
novel, but the canonical-only families (clarification, error_recovery) are not
templated at all -- their prompts are hand-authored and identical in both pools,
so they leak verbatim into the "novel" set and would flatter the result.

This drops every record whose user prompt also appears in the training corpus,
leaving only genuinely unseen phrasings. Previously done ad hoc; kept here so the
eval input is reproducible from the two dataset directories alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _prompts(path: Path) -> set[str]:
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for message in json.loads(line)["messages"]:
            if message.get("role") == "user":
                seen.add(message["content"])
                break
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--novel-dir", type=Path, required=True,
                        help="Corpus built from the held-out template pool.")
    parser.add_argument("--train-dir", type=Path, required=True,
                        help="Corpus the adapter trained on; its prompts are excluded.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trained = set()
    for split in ("train", "validation", "test"):
        path = args.train_dir / f"{split}.jsonl"
        if path.is_file():
            trained |= _prompts(path)

    kept: list[str] = []
    dropped = 0
    for split in ("train", "validation", "test"):
        path = args.novel_dir / f"{split}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = next(m["content"] for m in record["messages"] if m.get("role") == "user")
            if prompt in trained:
                dropped += 1
                continue
            kept.append(json.dumps(record, sort_keys=True, separators=(",", ":")))

    if not kept:
        raise SystemExit("no novel-phrasing records survived the overlap filter")
    args.output.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    print(f"wrote {len(kept)} novel-phrasing records to {args.output} "
          f"({dropped} dropped as seen in {args.train_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
