"""The journal-role promotion gate must consume the live smoke.

journal-role-r1 scored 1.000 schema-valid on the teacher-forced offline harness
and failed its first free-running CLI request two minutes after the manifest was
flagged ready. Nothing in the promotion path read that result. These tests pin
the gate so an offline report alone can no longer promote.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ase_auto_build.ase_agent.catalyst_cli import model_info, resolve_journal_adapter
from ase_auto_build.ase_agent.cli import EntryPointError

from training.evaluations.promote_journal_adapter import _live_smoke


def _package(root, name: str):
    package = root / name
    package.mkdir(parents=True)
    return package


def test_live_smoke_fails_on_the_recorded_r1_failure(tmp_path) -> None:
    package = _package(tmp_path, "journal-agent-r1-smoke")
    (package / "failure_record.json").write_text(json.dumps({
        "stage": "spec_planning",
        "message": "catalyst_spec missing required fields ['modifications']",
    }), encoding="utf-8")
    smoke = _live_smoke(tmp_path)
    assert smoke["passed"] is False
    assert "spec_planning" in smoke["reason"]


def test_live_smoke_fails_when_the_packet_is_not_review_ready(tmp_path) -> None:
    package = _package(tmp_path, "smoke")
    (package / "review_packet.json").write_text(
        json.dumps({"status": "needs_clarification"}), encoding="utf-8"
    )
    assert _live_smoke(tmp_path)["passed"] is False


def test_live_smoke_passes_only_on_a_completed_reconstruction(tmp_path) -> None:
    package = _package(tmp_path, "smoke")
    (package / "review_packet.json").write_text(
        json.dumps({"status": "review_ready"}), encoding="utf-8"
    )
    assert _live_smoke(tmp_path)["passed"] is True


def test_live_smoke_fails_when_no_package_was_written(tmp_path) -> None:
    assert _live_smoke(tmp_path / "missing")["passed"] is False
    assert _live_smoke(tmp_path)["passed"] is False


def _candidate(tmp_path, *, ready: bool):
    adapter = tmp_path / "run" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter.parent / "manifest.json").write_text(json.dumps({
        "corpus_contract": "journal", "journal_role_ready": ready,
    }), encoding="utf-8")
    return adapter


def test_unpromoted_adapter_runs_only_with_an_explicit_bypass(tmp_path) -> None:
    adapter = _candidate(tmp_path, ready=False)
    args = SimpleNamespace(base_only=False, adapter=adapter, allow_unpromoted=False)
    with pytest.raises(EntryPointError, match="promotion gate"):
        resolve_journal_adapter(args)
    args.allow_unpromoted = True
    assert resolve_journal_adapter(args) == adapter


def test_a_bypassed_package_says_so_in_its_reproduction_record(tmp_path) -> None:
    adapter = _candidate(tmp_path, ready=False)
    args = SimpleNamespace(
        base_only=False, adapter=adapter, allow_unpromoted=True,
        model="Qwen/Qwen3-4B-Instruct-2507", revision="main", max_new_tokens=1200,
    )
    info = model_info(args, resolve_journal_adapter(args))
    assert info["promotion_gate_bypassed"] is True
    assert info["adapter_sha256"]
    assert info["decoding"] == {"do_sample": False, "seed": None, "max_new_tokens": 1200}
