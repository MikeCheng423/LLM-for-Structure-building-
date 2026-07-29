from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ase_auto_build.ase_agent.catalyst_cli import build_parser, load_request, resolve_journal_adapter
from ase_auto_build.ase_agent.cli import EntryPointError


def test_catalyst_cli_loads_one_bounded_request(tmp_path) -> None:
    path = tmp_path / "request.json"
    value = {
        "request_id": "paper-1", "request": "Build Pt(111)",
        "sources": [{"source_id": "paper", "locator": "p. 1", "text": "Pt(111)"}],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    assert load_request(path) == value
    args = build_parser().parse_args(["--input", str(path), "--out", str(tmp_path / "out")])
    assert args.max_new_tokens == 2048


def test_catalyst_cli_rejects_extra_authority_fields(tmp_path) -> None:
    path = tmp_path / "request.json"
    path.write_text(json.dumps({
        "request_id": "paper-1", "request": "Build Pt", "sources": [],
        "shell": "rm -rf anything",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="only"):
        load_request(path)


def test_catalyst_cli_requires_role_promoted_adapter(tmp_path) -> None:
    adapter = tmp_path / "run" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    manifest = {"corpus_contract": "journal", "journal_role_ready": False}
    (adapter.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(base_only=False, adapter=adapter)
    with pytest.raises(EntryPointError, match="promotion gate"):
        resolve_journal_adapter(args)
    manifest["journal_role_ready"] = True
    (adapter.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert resolve_journal_adapter(args) == adapter
