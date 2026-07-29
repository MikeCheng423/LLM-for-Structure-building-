"""Non-interactive journal evidence to CatalystSpec package command."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from .catalyst_pipeline import run_journal_request
from .cli import EntryPointError, default_cache_dir, prepare_environment, repo_root
from .llm_defaults import DEFAULT_MODEL, DEFAULT_REVISION

EXIT_CODES = {
    "review_ready": 0,
    "failed": 1,
    "unsupported": 3,
    "needs_clarification": 5,
}
JOURNAL_ADAPTER_ENV = "ASE_CATALYST_ADAPTER"
DEFAULT_JOURNAL_ADAPTER = Path("training/runs/pilot-qwen3-4b-journal-role-r1/adapter")


def resolve_journal_adapter(args) -> Path | None:
    if args.base_only:
        return None
    candidate = Path(args.adapter or os.environ.get(JOURNAL_ADAPTER_ENV) or repo_root() / DEFAULT_JOURNAL_ADAPTER)
    manifest_path = candidate.parent / "manifest.json"
    if not (candidate / "adapter_config.json").is_file() or not manifest_path.is_file():
        raise EntryPointError(f"journal adapter or manifest is missing: {candidate}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("corpus_contract") != "journal" or manifest.get("journal_role_ready") is not True:
        raise EntryPointError(f"adapter has not passed the journal-role promotion gate: {candidate}")
    return candidate


def load_request(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"request_id", "request", "sources"}:
        raise ValueError("input must contain only request_id, request, and sources")
    if not isinstance(value["request_id"], str) or not isinstance(value["request"], str):
        raise ValueError("request_id and request must be strings")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value["request_id"]) is None:
        raise ValueError("request_id must be a path-free identifier")
    if not value["request"] or len(value["request"]) > 10_000:
        raise ValueError("request must contain 1 to 10000 characters")
    if not isinstance(value["sources"], list):
        raise ValueError("sources must be an array")
    if not value["sources"]:
        raise ValueError("sources must contain at least one item")
    for source in value["sources"]:
        if not isinstance(source, dict) or not {"source_id", "locator", "text"} <= set(source):
            raise ValueError("each source requires source_id, locator, and text")
        if set(source) - {"source_id", "locator", "text", "private"}:
            raise ValueError("source contains an unsupported field")
        if not all(isinstance(source[key], str) and source[key] for key in ("source_id", "locator", "text")):
            raise ValueError("source_id, locator, and text must be nonempty strings")
        if "private" in source and not isinstance(source["private"], bool):
            raise ValueError("source private flag must be boolean")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Request JSON file.")
    parser.add_argument("--out", type=Path, required=True, help="Root for immutable request packages.")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        request = load_request(args.input)
        if args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be positive")
        args.cache_dir = args.cache_dir or default_cache_dir()
        prepare_environment(args.cache_dir, offline=not args.online)
        adapter = resolve_journal_adapter(args)
        from .llm_local import LocalModelChat, load_model

        model, tokenizer = load_model(
            model=args.model, revision=args.revision,
            cache_dir=args.cache_dir, adapter=adapter,
        )
        chat = LocalModelChat(
            model, tokenizer, tool_override=None,
            max_new_tokens=args.max_new_tokens,
        )
        result = run_journal_request(
            request["request_id"], request["request"], request["sources"],
            chat, args.out,
        )
    except Exception as exc:
        print(f"ASE_catalyst_build: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 6
    print(json.dumps({
        "status": result.status,
        "request_dir": str(result.request_dir),
        "files": [str(path) for path in result.files],
    }, sort_keys=True))
    return EXIT_CODES.get(result.status, 1)


if __name__ == "__main__":
    raise SystemExit(main())
