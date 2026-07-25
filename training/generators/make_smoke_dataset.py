#!/usr/bin/env python3
"""Create synthetic, non-production records for one-step pipeline testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.dataset_contract import SCHEMA_VERSION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    tool = {
        "type": "function",
        "function": {
            "name": "smoke_echo_structure_request",
            "description": "Synthetic tool used only to verify the training pipeline.",
            "parameters": {
                "type": "object",
                "properties": {"formula": {"type": "string"}},
                "required": ["formula"],
                "additionalProperties": False,
            },
        },
    }
    records = []
    for index, formula in enumerate(("Al", "Si"), start=1):
        records.append(
            {
                "id": f"smoke-{index}",
                "schema_version": SCHEMA_VERSION,
                "split_group": "synthetic-smoke-only",
                "smoke_only": True,
                "tools": [tool],
                "messages": [
                    {
                        "role": "system",
                        "content": "This is a synthetic pipeline test. Use only the registered tool.",
                    },
                    {"role": "user", "content": f"Echo a request for {formula}."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "smoke_echo_structure_request",
                                    "arguments": {"formula": formula},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "smoke_echo_structure_request",
                        "content": json.dumps({"ok": True, "formula": formula}),
                    },
                    {"role": "assistant", "content": "The synthetic request was echoed."},
                ],
                "provenance": {
                    "source": "synthetic_smoke",
                    "sanitized": True,
                    "contains_private_structure": False,
                },
                "validation": {
                    "schema_valid": False,
                    "executed": False,
                    "invariants_passed": False,
                    "forbidden_action_count": 0,
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"wrote {len(records)} smoke-only records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

