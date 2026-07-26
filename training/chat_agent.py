#!/usr/bin/env python3
"""Deprecated shim: the entry point now lives in the runtime package.

The real front door is ``vasp_auto.ase_agent.cli`` (console script
``vasp-auto-build``), which does everything this REPL did *and* writes the
finished structure out as a ready-to-run ``vasp-auto`` case. This file stays so
existing commands, scripts, and notes keep working.

    # old
    .venv/bin/python training/chat_agent.py --prompt "..."
    # new
    .venv/bin/python -m vasp_auto.ase_agent.cli --prompt "..."
    # or, once installed:
    vasp-auto-build "..."

Flag compatibility: ``--prompt``, ``--adapter``, ``--base-only``, ``--model``,
``--revision``, ``--cache-dir``, ``--max-turns``, and ``--max-new-tokens`` all
mean exactly what they used to, because the new CLI defines the same names.
Bare invocation is still an interactive REPL.
"""

from __future__ import annotations

import sys

from vasp_auto.ase_agent.cli import main


def _forward(argv: list[str]) -> list[str]:
    """Preserve the old behaviour of not writing files unless asked to."""
    if "--no-write" in argv or any(
        item == "--out" or item == "-o" or item.startswith("--out=") for item in argv
    ):
        return argv
    return [*argv, "--no-write"]


if __name__ == "__main__":
    print(
        "training/chat_agent.py is deprecated; use `vasp-auto-build` "
        "(python -m vasp_auto.ase_agent.cli). Running it for you, without "
        "writing files -- add --out DIR to get a POSCAR case.",
        file=sys.stderr,
    )
    raise SystemExit(main(_forward(sys.argv[1:])))
