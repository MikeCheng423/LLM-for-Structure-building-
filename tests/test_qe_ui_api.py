"""QE analysis API tests that do not require binding the HTTP server."""
from pathlib import Path

import pytest

from vasp_auto_ui import server


POSCAR_FE = """Fe
1.0
3 0 0
0 3 0
0 0 3
Fe
1
Direct
0 0 0
"""


def _qe_job(path: Path) -> Path:
    path.mkdir()
    (path / ".engine").write_text("qe\n", encoding="utf-8")
    (path / "POSCAR").write_text(POSCAR_FE, encoding="utf-8")
    (path / "pw.out").write_text("the Fermi energy is 0.0 ev\n", encoding="utf-8")
    return path


def test_qe_pdos_and_dband_api(tmp_path):
    job = _qe_job(tmp_path / "dos")
    (job / "pdos.pdos_atm#1(Fe)_wfc#2(d)").write_text(
        "# E (eV) ldos(E) pdos(E)\n-2 1 1\n-1 1 1\n0 1 1\n",
        encoding="utf-8",
    )
    query = {"path": [str(job)], "atoms": ["1"]}
    pdos = server.api_pdos(query, {})
    d_band = server.api_dband(query, {})
    assert pdos["curves"][0]["shell"] == "d"
    assert d_band["d_band_center_eV"] == pytest.approx(-1.0)


def test_qe_cube_charge_difference_api(tmp_path):
    header = (
        "density\nQE pp.x\n 1 0 0 0\n 2 1 0 0\n 1 0 1 0\n 1 0 0 1\n"
        " 26 0 0 0 0\n"
    )
    total = _qe_job(tmp_path / "total")
    part = _qe_job(tmp_path / "part")
    (total / "charge-density.cube").write_text(header + " 3 5\n", encoding="utf-8")
    (part / "charge-density.cube").write_text(header + " 1 2\n", encoding="utf-8")
    result = server.api_chgdiff({}, {"total": str(total), "parts": [str(part)]})
    assert result["path"].endswith("charge-density-diff.cube")
    assert Path(result["path"]).exists()
