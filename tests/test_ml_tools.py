"""Tests for the MLIP integration (ml_tools.py + the UI endpoint).

The real OMat24/UMA backend needs fairchem-core (not a test dependency);
these tests use ASE's built-in EMT potential, which exercises the identical
relax/report/XDATCAR path.
"""
from pathlib import Path

import pytest

ase = pytest.importorskip("ase")

from vasp_auto.ml_tools import get_ml_calculator, ml_energy, ml_relax_case
from vasp_auto.structure import read_poscar
from vasp_auto.trajectory import job_trajectory
from vasp_auto_ui import server as ui_server


# 2x1x1 fcc Al supercell with one atom nudged off its site so EMT has
# something to relax.
RATTLED_AL = """Al rattled
1.0
8.10 0.0 0.0
0.0 4.05 0.0
0.0 0.0 4.05
Al
8
Direct
0.03 0.02 0.00
0.25 0.50 0.50
0.25 0.00 0.50
0.00 0.50 0.00
0.50 0.00 0.00
0.75 0.50 0.50
0.75 0.00 0.50
0.50 0.50 0.00
"""


@pytest.fixture
def rattled_case(tmp_path):
    case = tmp_path / "Al_rattled"
    case.mkdir()
    (case / "POSCAR").write_text(RATTLED_AL, encoding="utf-8")
    return case


def test_get_ml_calculator_emt():
    calc = get_ml_calculator("emt")
    assert calc.__class__.__name__ == "EMT"


def test_get_ml_calculator_fairchem_missing_hint():
    pytest.importorskip("ase")
    try:
        import fairchem  # noqa: F401
        pytest.skip("fairchem installed; the missing-dependency hint cannot fire")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="fairchem-core"):
        get_ml_calculator("uma-s-1")


def test_ml_relax_case_writes_derived_case(rattled_case):
    result = ml_relax_case(rattled_case, model="emt", fmax=0.05, steps=100)
    out = Path(result["case_dir"])
    assert out.name == "Al_rattled_ml"
    assert (out / "POSCAR").exists()
    assert result["converged"] is True
    assert result["steps"] > 0
    assert result["max_force_eV_A"] <= 0.05
    # the original case is untouched
    assert read_poscar(rattled_case / "POSCAR")["coords"][0][0] == 0.03
    # relaxation lowered the energy below the starting single point
    start = ml_energy(rattled_case, model="emt")
    assert result["energy_eV"] < start["energy_eV"]


def test_ml_relax_writes_animatable_xdatcar(rattled_case):
    result = ml_relax_case(rattled_case, model="emt", fmax=0.05, steps=100)
    traj = job_trajectory(Path(result["case_dir"]))
    assert traj is not None and traj["kind"] == "relax"
    assert len(traj["frames"]) == result["steps"] + 1
    assert traj["symbols"] == ["Al"] * 8


def test_ml_energy_single_point(rattled_case):
    result = ml_energy(rattled_case, model="emt")
    assert isinstance(result["energy_eV"], float)
    assert result["max_force_eV_A"] > 0.05  # rattled structure is strained
    assert result["model"] == "emt"


def test_api_mlrelax_endpoint(rattled_case):
    result = ui_server.api_mlrelax(None, {"case": str(rattled_case), "model": "emt"})
    assert Path(result["case_dir"]).exists()
    assert result["converged"] is True


# ---------------------------------------------------------------- TASK 3: --ml-energy + /api/mlenergy

def test_ml_energy_returns_dict_with_all_keys(rattled_case):
    result = ml_energy(rattled_case, model="emt")
    assert "energy_eV" in result
    assert "max_force_eV_A" in result
    assert "model" in result
    assert result["model"] == "emt"


def test_ml_energy_poscar_file_path(rattled_case):
    """ml_energy should also accept a direct POSCAR file path."""
    result = ml_energy(rattled_case / "POSCAR", model="emt")
    assert isinstance(result["energy_eV"], float)


def test_api_mlenergy_endpoint(rattled_case):
    result = ui_server.api_mlenergy(None, {"case": str(rattled_case), "model": "emt"})
    assert "energy_eV" in result
    assert "max_force_eV_A" in result
    assert result["model"] == "emt"


def test_cli_ml_energy_prints_result(rattled_case, capsys):
    """_apply_ml_energy reads POSCAR and prints energy/force/model."""
    from vasp_auto.cli import _apply_ml_energy
    import argparse

    args = argparse.Namespace(
        ml_energy=str(rattled_case),
        ml_model="emt",
        ml_task=None,
        ml_checkpoint=None,
    )
    result = _apply_ml_energy(args, {})
    assert result is True
    captured = capsys.readouterr()
    assert "ML energy" in captured.out
    assert "eV" in captured.out
    assert "model" in captured.out.lower() or "model" in captured.out
