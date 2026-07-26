#!/usr/bin/env python3
"""Tabulate the r6 seen-phrasing gate and novel-phrasing generalization reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EV = Path("training/evaluations")
METRICS = [
    ("exact_structure_rate", "exact"),
    ("invariants_satisfied_rate", "invariants"),
    ("finish_rate", "finish"),
    ("schema_execution_success_rate", "schema"),
    ("exact_tool_sequence_rate", "tool-seq"),
]


def load(tag: str, name: str):
    p = EV / f"{name}-{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def pct(x):
    return "  -  " if x is None else f"{100*x:5.1f}"


def row(label, summary):
    cells = " ".join(pct(summary.get(m)) if summary else "  -  " for m, _ in METRICS)
    n = summary.get("record_count") if summary else None
    return f"  {label:14s} n={str(n):>4}  {cells}"


def header():
    return "  " + " " * 14 + "        " + " ".join(f"{lbl:>5}" for _, lbl in METRICS)


def block(title, rows):
    print(f"\n{title}")
    print(header())
    for label, rep in rows:
        print(row(label, (rep or {}).get("summary")))


def family_table(title, reps):
    print(f"\n{title}  (exact_structure_rate per family)")
    fams = sorted({f for rep in reps.values() if rep for f in rep["summary"].get("family_metrics", {})})
    cols = list(reps)
    print("  " + f"{'family':22s}" + "".join(f"{c:>8}" for c in cols) + "    n")
    for fam in fams:
        cells = ""
        n = None
        for c in cols:
            fm = (reps[c] or {}).get("summary", {}).get("family_metrics", {}).get(fam) if reps[c] else None
            cells += f"{pct(fm.get('exact_structure_rate')) if fm else '  -  ':>8}"
            if fm:
                n = fm.get("record_count")
        print(f"  {fam:22s}{cells}    {n}")


def safety_line(rep):
    if not rep:
        return "  (none)"
    s = rep.get("safety", {})
    rc = s.get("record_count")
    fe = s.get("forbidden_execution_count")
    safe = sum(1 for r in s.get("results", []) if r.get("safe"))
    return f"  adversarial: {safe}/{rc} safe, forbidden_executions={fe}"


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "pf6-s120"
    base_seen = load(tag, "base-r7-seen")
    r6_seen = load(tag, "adapter-r7-seen")
    base_novel = load(tag, "base-r7-novel")
    r5_novel = load(tag, "adapter-r5-novel")
    r6_novel = load(tag, "adapter-r7-novel")
    promo = load(tag, "promotion-r7-seen")

    print("=" * 72)
    print(f"r6 EVALUATION SUMMARY  (tag={tag})")
    print("=" * 72)

    block("SEEN phrasing — pilot_r7/test (deployment gate; structures held out by hash)",
          [("frozen base", base_seen), ("r6 adapter", r6_seen)])
    print("\n  " + safety_line(r6_seen).strip())

    block("NOVEL phrasing — held-out templates never seen in training (structures seen)",
          [("frozen base", base_novel), ("r5 adapter", r5_novel), ("r6 adapter", r6_novel)])

    # Prove the novel trio scored identical records.
    ids = {name: (rep or {}).get("evaluation_ids_sha256") for name, rep in
           (("base", base_novel), ("r5", r5_novel), ("r6", r6_novel)) if rep}
    same = len(set(ids.values())) == 1 and None not in ids.values()
    print(f"\n  novel-set record identity (evaluation_ids_sha256 match across models): {same}")
    if ids:
        print(f"    {ids}")

    family_table("NOVEL phrasing per-family",
                 {"base": base_novel, "r5": r5_novel, "r6": r6_novel})

    if promo:
        print(f"\nPROMOTION GATE (seen phrasing): promoted={promo.get('promoted')}")
        checks = promo.get("checks") or promo.get("gate_checks") or {}
        if isinstance(checks, dict):
            failed = [k for k, v in checks.items() if v is False]
            print(f"  checks: {len(checks)} total, failed={failed or 'none'}")
        for k in ("failures", "reasons", "failed_checks"):
            if promo.get(k):
                print(f"  {k}: {promo[k]}")

    # Headline deltas.
    def ex(rep):
        return (rep or {}).get("summary", {}).get("exact_structure_rate")
    print("\nHEADLINE (exact_structure_rate):")
    print(f"  r6 seen  : {pct(ex(r6_seen))}%")
    print(f"  r6 novel : {pct(ex(r6_novel))}%   <- limitation #1 probe")
    print(f"  r5 novel : {pct(ex(r5_novel))}%")
    print(f"  base novel:{pct(ex(base_novel))}%")
    if ex(r6_novel) is not None and ex(r5_novel) is not None:
        print(f"  novel delta r7-r5 : {100*(ex(r6_novel)-ex(r5_novel)):+.1f} pts")
    if ex(r6_seen) is not None and ex(r6_novel) is not None:
        print(f"  r6 seen->novel gap: {100*(ex(r6_seen)-ex(r6_novel)):+.1f} pts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
