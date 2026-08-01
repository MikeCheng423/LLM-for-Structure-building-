#!/usr/bin/env python3
"""Shared literal-leakage guard for prose-rendered corpus records.

Both `build_journal_corpus.py` (training prose, half the corpus) and
`build_prose_holdout.py` (evaluation-only `prose_holdout`) render source text
that must require *interpretation* rather than *copying* -- a rendered
sentence must never contain a JSON literal (or bare enum/boolean spelling) of
the value it is meant to ground. This module is the single implementation of
that check so the two generators cannot drift, and so `build_journal_corpus`
can depend on it without importing `build_prose_holdout` (which itself imports
from `build_journal_corpus` -- a shared leaf module avoids the cycle).

See `training/JOURNAL_ROLE_R3_SCHEDULE.md` section 1 for the failure this
guards against, and `training/STATUS.md`'s "prose_holdout: the extractor
cannot read prose" entry for the measured failure modes.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _literal_tokens(value: Any) -> tuple[set[str], set[str]]:
    """Return (quoted_forms, bare_forms) that would let a model copy `value`
    out of the text instead of interpreting prose.

    - bool: only the JSON spelling ("true"/"false") is a risk.
    - list/dict: only the exact bracketed/braced JSON literal is a risk --
      English prose about a Miller index or a supercell repeat is full of
      digits that are not that literal (see `_bare_leak`'s word-boundary
      matching below).
    - float: excluded entirely. `"1.85 A above the surface"` is how English
      states a decimal quantity; `json.dumps(1.85)` renders the identical
      digits, so there is no alternate prose form to require. This is not
      the failure mode the schedule documents (enum/boolean/JSON-syntax
      copying) -- it is an unavoidable identity for continuous measurements.
    - str: a single-character token (bare element symbols "O"/"H"/"W") is
      excluded from the bare check because a lone capital letter appears
      constantly in ordinary English regardless of the target value; the
      quoted JSON form ('"O"') is still checked since prose never emits
      quote marks around a symbol.
    """
    if isinstance(value, bool):
        return set(), {"true" if value else "false"}
    if isinstance(value, (list, dict)):
        return {
            json.dumps(value, sort_keys=True),
            json.dumps(value, sort_keys=True, separators=(",", ":")),
        }, set()
    if isinstance(value, float):
        return set(), set()
    if isinstance(value, int):
        return set(), {str(value)}
    if isinstance(value, str):
        bare = {value} if len(value) > 1 else set()
        return {json.dumps(value)}, bare
    return set(), set()


def _bare_leak(text: str, token: str) -> bool:
    """Word-boundary match: "Al" must not flag "Aluminum" (no boundary after
    the match), but must flag a standalone "Al" token.

    "." is excluded from the boundary on both sides **for numeric tokens
    only**: an integer token like "1" must not match either digit of an
    unrelated decimal quantity such as "1.2" (a float claim is exempt for
    *its own* claim, but its digits still appear in the assembled text, and a
    `.` is not a conventional word boundary -- without this, "1.2 A above the
    surface" would falsely flag any co-occurring int-valued claim equal to 1
    or 2).

    For word tokens ("true", "fcc", "ase_default") a "." *is* a boundary, and
    the most common one: a sentence-final literal is exactly how a leak would
    read ("The centering flag is true."). Excluding "." for those tokens made
    every end-of-sentence literal invisible to this guard."""
    numeric = token.lstrip("-").isdigit()
    edge = r"[A-Za-z0-9_.]" if numeric else r"[A-Za-z0-9_]"
    pattern = re.compile(rf"(?<!{edge}){re.escape(token)}(?!{edge})")
    return pattern.search(text) is not None


def _assert_no_leak(case_id: str, field: str, value: Any, text: str) -> None:
    quoted, bare = _literal_tokens(value)
    for token in quoted:
        if token in text:
            raise RuntimeError(f"{case_id}: field {field!r} leaks JSON literal {token!r} into source text: {text!r}")
    for token in bare:
        if _bare_leak(text, token):
            raise RuntimeError(f"{case_id}: field {field!r} leaks bare literal {token!r} into source text: {text!r}")


__all__ = ["_assert_no_leak", "_bare_leak", "_literal_tokens"]
