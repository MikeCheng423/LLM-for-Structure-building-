"""Tests for the remote-machine setup helper (no real SSH)."""

from __future__ import annotations

import json

import yaml

from vasp_auto import remote_setup


def _machine():
    return {
        "host": "cluster.example.edu",
        "user": "me",
        "remote_root": "/scratch/me/jobs",
        "vasp_executable": "/opt/vasp/bin/vasp_std",
        "scheduler": "slurm",
        "scheduler_options": ["module load vasp"],
        "port": "",          # empty -> dropped
    }


def test_clean_entry_drops_empty_and_keeps_lists():
    entry = remote_setup._clean_entry(_machine())
    assert "port" not in entry
    assert entry["host"] == "cluster.example.edu"
    assert entry["scheduler_options"] == ["module load vasp"]


def test_save_to_config_appends_when_no_remotes(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("vasp_executable: vasp_std\njobs_root: /tmp/jobs\n", encoding="utf-8")

    remote_setup.save_to_config(_machine(), "cluster", config_path=cfg)

    text = cfg.read_text(encoding="utf-8")
    # original content preserved (comment-safe append path)
    assert "jobs_root: /tmp/jobs" in text
    data = yaml.safe_load(text)
    assert data["remotes"]["cluster"]["host"] == "cluster.example.edu"
    assert "port" not in data["remotes"]["cluster"]


def test_save_to_config_merges_existing_remotes(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump({"remotes": {"old": {"host": "h1", "remote_root": "/r",
                                            "vasp_executable": "v"}}}),
        encoding="utf-8",
    )
    remote_setup.save_to_config(_machine(), "cluster", config_path=cfg)
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert set(data["remotes"]) == {"old", "cluster"}


def test_save_to_ui_store_roundtrip(tmp_path):
    store = tmp_path / "remotes.json"
    remote_setup.save_to_ui_store(_machine(), "cluster", store_path=store)
    data = json.loads(store.read_text(encoding="utf-8"))
    assert data["cluster"]["name"] == "cluster"
    assert data["cluster"]["vasp_executable"] == "/opt/vasp/bin/vasp_std"


def test_ensure_local_key_reuses_existing(tmp_path, monkeypatch):
    home = tmp_path
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_ed25519").write_text("KEY", encoding="utf-8")
    monkeypatch.setattr(remote_setup.Path, "home", staticmethod(lambda: home))
    key = remote_setup.ensure_local_key()
    assert key == home / ".ssh" / "id_ed25519"
