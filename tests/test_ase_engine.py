"""Tests for the generic ASE-calculator engine (engine: ase).

EMT needs no external binary, so the create/run/parse cycle is exercised for
real here, the same way the QE tests use a tiny fake pw.out.
"""
import json

import pytest

from vasp_auto import ase_engine
from vasp_auto.job_manager import create_job_from_case, make_case_info, preview_job_from_case
from vasp_auto.runner import run_ase
from vasp_auto.workflow import build_row, job_engine, should_retry_failed

# fcc Cu near the EMT equilibrium — a 2-atom cell so a relax has something to do.
POSCAR_CU = """Cu fcc
3.6
0.0 1.8 1.8
1.8 0.0 1.8
1.8 1.8 0.0
Cu
1
Direct
0.0 0.0 0.0
"""


@pytest.fixture
def ase_case(tmp_path):
    case = tmp_path / "Cu"
    case.mkdir()
    (case / "POSCAR").write_text(POSCAR_CU, encoding="utf-8")
    config = {"engine": "ase", "ase_calculator": "emt"}
    case_info = make_case_info(case, tmp_path / "jobs")
    return case, config, case_info


# ----------------------------------------------------------------- prepare

def test_preview_reports_calculator(ase_case):
    _, config, case_info = ase_case
    preview = preview_job_from_case(case_info, calc_type="scf", engine="ase", config=config)
    assert preview["engine"] == "ase"
    assert preview["calculator"] == "emt"
    spec = json.loads(preview["ase_calc.json"])
    assert spec["calc_type"] == "scf"


def test_create_writes_driver_and_marker(ase_case):
    _, config, case_info = ase_case
    job_dir = create_job_from_case(case_info=case_info, calc_type="scf", engine="ase", config=config)
    assert (job_dir / "run_ase.py").exists()
    assert (job_dir / "POSCAR").exists()
    assert (job_dir / ".engine").read_text().strip() == "ase"
    assert job_engine(job_dir) == "ase"


def test_unsupported_calc_type_rejected(ase_case):
    _, config, case_info = ase_case
    with pytest.raises(ValueError, match="does not support calc-type"):
        create_job_from_case(case_info=case_info, calc_type="dos", engine="ase", config=config)


def test_custom_calculator_params_carried(ase_case):
    case, _, case_info = ase_case
    config = {"engine": "ase", "ase_calculator": "lj", "ase_calc_params": {"epsilon": 0.5}}
    job_dir = create_job_from_case(case_info=case_info, calc_type="scf", engine="ase", config=config)
    spec = json.loads((job_dir / "ase_calc.json").read_text())
    assert spec["calculator"] == "lj"
    assert spec["params"] == {"epsilon": 0.5}


# ----------------------------------------------------------------- run (real EMT)

def test_scf_run_produces_energy(ase_case):
    _, config, case_info = ase_case
    job_dir = create_job_from_case(case_info=case_info, calc_type="scf", engine="ase", config=config)
    rc = run_ase(str(job_dir))
    assert rc == 0
    summary = ase_engine.parse_ase_output(job_dir)
    assert summary["engine"] == "ase"
    assert summary["calc_type"] == "scf"
    assert isinstance(summary["energy_eV"], float)
    assert summary["converged"] is True
    assert (job_dir / "CONTCAR").exists()


def test_relax_run_converges(ase_case):
    _, config, case_info = ase_case
    job_dir = create_job_from_case(case_info=case_info, calc_type="relax", engine="ase", config=config)
    rc = run_ase(str(job_dir))
    assert rc == 0
    summary = ase_engine.parse_ase_output(job_dir)
    assert summary["calc_type"] == "relax"
    assert summary["converged"] is True
    assert summary["max_force_eV_A"] <= 0.05


# ----------------------------------------------------------------- row + retry

def test_build_row_from_ase_results(ase_case):
    _, config, case_info = ase_case
    job_dir = create_job_from_case(case_info=case_info, calc_type="scf", engine="ase", config=config)
    run_ase(str(job_dir))
    row = build_row("proj", "single", case_info)
    assert row["engine"] == "ase"
    assert row["calculator"] == "emt"
    assert row["status"] == "done"
    assert row["energy_eV"] is not None


def test_should_retry_when_unrun(ase_case):
    _, config, case_info = ase_case
    create_job_from_case(case_info=case_info, calc_type="scf", engine="ase", config=config)
    # No run yet -> no ase_results.json -> should retry.
    assert should_retry_failed(case_info) is True
