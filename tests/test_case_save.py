"""Saving a structure by name must never clobber an existing case — the
server bumps to name_2, name_3, … (run auto-saves rely on this)."""
from vasp_auto_ui import server as ui_server

PAYLOAD = {
    "comment": "H2O",
    "lattice": [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
    "symbols": ["O", "H", "H"],
    "frac": [[0.5, 0.5, 0.5], [0.5, 0.55, 0.45], [0.5, 0.45, 0.45]],
}


def test_name_save_bumps_instead_of_overwriting(tmp_path):
    body = {"structure": PAYLOAD, "name": "H2O", "root": str(tmp_path)}
    first = ui_server.api_structure_save(None, dict(body))
    second = ui_server.api_structure_save(None, dict(body))
    third = ui_server.api_structure_save(None, dict(body))
    assert first["case"] == str(tmp_path / "H2O")
    assert second["case"] == str(tmp_path / "H2O_2")
    assert third["case"] == str(tmp_path / "H2O_3")
    assert (tmp_path / "H2O" / "POSCAR").exists()
    assert (tmp_path / "H2O_2" / "POSCAR").exists()


def test_explicit_dir_still_overwrites(tmp_path):
    target = tmp_path / "mycase"
    body = {"structure": PAYLOAD, "dir": str(target)}
    ui_server.api_structure_save(None, dict(body))
    res = ui_server.api_structure_save(None, dict(body))  # "Save over original"
    assert res["case"] == str(target)
    assert not (tmp_path / "mycase_2").exists()
