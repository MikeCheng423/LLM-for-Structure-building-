"""Tests for the plan.txt build-out: reports, trajectories, DOS parsing."""
import pytest

from vasp_auto.parser import parse_dos
from vasp_auto.report import build_job_report, write_job_report
from vasp_auto.trajectory import job_trajectory, read_xdatcar
from vasp_auto_ui import server as ui_server


XDATCAR_TEXT = """Al test
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
Al
2
Direct configuration=     1
0.00 0.00 0.00
0.50 0.50 0.50
Direct configuration=     2
0.00 0.00 0.10
0.50 0.50 0.60
"""

DOS_VASPRUN = """<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
 <calculation>
  <dos>
   <i name="efermi">  2.50000000 </i>
   <total>
    <array>
     <set>
      <set comment="spin 1">
       <r> -2.0 0.0 0.0 </r>
       <r>  0.0 1.5 1.0 </r>
       <r>  2.0 0.5 2.0 </r>
      </set>
      <set comment="spin 2">
       <r> -2.0 0.0 0.0 </r>
       <r>  0.0 1.2 1.0 </r>
       <r>  2.0 0.4 2.0 </r>
      </set>
     </set>
    </array>
   </total>
  </dos>
 </calculation>
</modeling>
"""


def test_read_xdatcar(tmp_path):
    (tmp_path / "XDATCAR").write_text(XDATCAR_TEXT, encoding="utf-8")
    data = read_xdatcar(tmp_path / "XDATCAR")
    assert data["symbols"] == ["Al", "Al"]
    assert len(data["frames"]) == 2
    assert data["frames"][1][0] == [0.0, 0.0, 0.10]


def test_job_trajectory_from_xdatcar(tmp_path):
    (tmp_path / "XDATCAR").write_text(XDATCAR_TEXT, encoding="utf-8")
    traj = job_trajectory(tmp_path)
    assert traj["kind"] == "relax"
    assert len(traj["frames"]) == 2
    # fractional 0.5,0.5,0.6 in a 4 Å cube → 2.0, 2.0, 2.4 Å
    assert traj["frames"][1][1] == [2.0, 2.0, 2.4]


def test_job_trajectory_neb(tmp_path):
    poscar = "img\n1.0\n4 0 0\n0 4 0\n0 0 4\nAl\n1\nDirect\n{x} 0.0 0.0\n"
    for i, x in enumerate(("0.0", "0.25", "0.5")):
        image = tmp_path / f"{i:02d}"
        image.mkdir()
        (image / "POSCAR").write_text(poscar.format(x=x), encoding="utf-8")
    traj = job_trajectory(tmp_path)
    assert traj["kind"] == "neb"
    assert len(traj["frames"]) == 3
    assert traj["frames"][2][0][0] == 2.0


def test_job_trajectory_none_for_empty(tmp_path):
    assert job_trajectory(tmp_path) is None


def test_parse_dos(tmp_path):
    vasprun = tmp_path / "vasprun.xml"
    vasprun.write_text(DOS_VASPRUN, encoding="utf-8")
    dos = parse_dos(vasprun)
    assert dos["efermi"] == 2.5
    assert dos["energies"] == [-2.0, 0.0, 2.0]
    assert len(dos["total"]) == 2
    assert dos["total"][0] == [0.0, 1.5, 0.5]


def test_parse_dos_missing(tmp_path):
    assert parse_dos(tmp_path / "vasprun.xml") is None


def test_build_job_report(finished_job):
    (finished_job / "INCAR").write_text("ENCUT = 450\nISMEAR = 0\nSIGMA = 0.05\n", encoding="utf-8")
    text = build_job_report(finished_job, case_name="Al")
    assert "# Calculation report — Al" in text
    assert "- ENCUT = 450" in text
    assert "- converged: yes" in text
    assert "- free energy (TOTEN): -12.500000 eV" in text
    assert "- ionic steps: 2" in text


def test_write_job_report_with_extra(finished_job):
    path = write_job_report(finished_job, extra={"selected_encut": 450})
    assert path.name == "report.md"
    assert "- selected_encut: 450" in path.read_text()


def test_report_includes_errors(tmp_path):
    (tmp_path / "run.log").write_text("ZBRENT: fatal error\n", encoding="utf-8")
    text = build_job_report(tmp_path)
    assert "## Detected problems" in text
    assert "ZBRENT" in text


# ---------------------------------------------------------------- UI endpoints

def test_api_trajectory(tmp_path):
    (tmp_path / "XDATCAR").write_text(XDATCAR_TEXT, encoding="utf-8")
    traj = ui_server.api_trajectory({"path": [str(tmp_path)]}, None)
    assert traj["kind"] == "relax" and len(traj["frames"]) == 2


def test_api_trajectory_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ui_server.api_trajectory({"path": [str(tmp_path)]}, None)


def test_api_dos(tmp_path):
    (tmp_path / "vasprun.xml").write_text(DOS_VASPRUN, encoding="utf-8")
    dos = ui_server.api_dos({"path": [str(tmp_path)]}, None)
    assert dos["efermi"] == 2.5


def test_api_report(finished_job):
    result = ui_server.api_report(None, {"job_dir": str(finished_job), "case": "Al"})
    assert "Calculation report — Al" in result["text"]
    assert (finished_job / "report.md").exists()


def test_api_build_tss(tmp_path, scf_case, monkeypatch):
    out = tmp_path / "neb_out"
    result = ui_server.api_build(None, {
        "action": "tss",
        "initial": str(scf_case),
        "final": str(scf_case / "POSCAR"),
        "output": str(out),
    })
    assert (out / "initial" / "POSCAR").exists()
    assert (out / "final" / "POSCAR").exists()
    assert result["case"] == str(out.resolve())
