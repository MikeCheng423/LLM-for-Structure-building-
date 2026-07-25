"""The analysis endpoints resolve remote jobs into a local cache dir, so the
Results-tab viewers (DOS, PDOS, bands, volume, animation, report) work on
remote results exactly like on local ones."""
from pathlib import Path

import pytest

from vasp_auto_ui import server as ui_server


def test_analysis_dir_local_passthrough(tmp_path):
    assert ui_server._analysis_dir(str(tmp_path)) == tmp_path
    assert ui_server._analysis_dir(str(tmp_path), "local") == tmp_path
    assert ui_server._analysis_dir(str(tmp_path), None) == tmp_path


def test_analysis_dir_remote_syncs_to_cache(monkeypatch, tmp_path):
    calls = []

    def fake_fetch(remote, remote_dir, local_dir, include_heavy=False):
        calls.append(remote_dir)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return {"local_dir": local_dir, "remote_dir": remote_dir, "transferred": True}

    monkeypatch.setattr(ui_server, "_resolve_remote", lambda name: {"name": name, "host": "x"})
    monkeypatch.setattr(ui_server, "fetch_remote_results", fake_fetch)
    monkeypatch.setattr(ui_server, "REMOTE_RESULT_CACHE", tmp_path)

    d = ui_server._analysis_dir("/remote/jobs/0007_H2O", "tlclab")
    assert calls == ["/remote/jobs/0007_H2O"]
    assert d.is_relative_to(tmp_path / "tlclab")
    assert d.name.startswith("0007_H2O-")
    # Same remote job -> same cache dir; same name elsewhere -> a different one.
    assert ui_server._analysis_dir("/remote/jobs/0007_H2O", "tlclab") == d
    assert ui_server._analysis_dir("/elsewhere/0007_H2O", "tlclab") != d


def test_slot_dir_string_and_dict(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ui_server, "_analysis_dir",
                        lambda path, machine=None, heavy=(): calls.append((path, machine, heavy)) or Path(path))
    ui_server._slot_dir(str(tmp_path))
    ui_server._slot_dir({"path": "/remote/j", "machine": "tlclab"}, heavy=("CHGCAR",))
    assert calls == [(str(tmp_path), None, ()), ("/remote/j", "tlclab", ("CHGCAR",))]


def _local_job(tmp_path, name, energy):
    job = tmp_path / name
    job.mkdir()
    (job / "OUTCAR").write_text(
        f"  aborting loop because EDIFF is reached\n  free  energy   TOTEN  = {energy:.8f} eV\n",
        encoding="utf-8",
    )
    return job


def test_api_adsorption_accepts_machine_slots(tmp_path):
    total = _local_job(tmp_path, "tot", -110.0)
    slab = _local_job(tmp_path, "slab", -100.0)
    mol = _local_job(tmp_path, "mol", -6.0)
    # dict slots with machine=local resolve straight through _analysis_dir.
    result = ui_server.api_adsorption(None, {
        "total": {"path": str(total), "machine": "local"},
        "slab": str(slab),  # bare string still works
        "molecule": {"path": str(mol), "machine": "local"},
        "scale": 0.5,
    })
    assert result["adsorption_energy_eV"] == pytest.approx(-110.0 + 100.0 + 3.0)
