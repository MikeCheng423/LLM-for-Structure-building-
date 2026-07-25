"""Tests for the 2026-06-12 build-out: spin, magmoms, auto-retry, SIGMA scan,
interstitials, HSE06 preset, and the friendlier UI server API."""
import pytest

from vasp_auto.calc_types import CALC_TYPE_INFO, CalcType, parse_calc_type
from vasp_auto.convergence import _select_sigma_trial, parse_sigma_values
from vasp_auto.incar import (
    apply_spin_to_incar,
    get_incar_value,
    magmom_line,
    parse_magmom_map,
    spin_incar_text,
)
from vasp_auto.job_manager import create_job_from_case, load_incar_template, make_case_info
from vasp_auto.structure import add_interstitial, per_atom_symbols, read_poscar
from vasp_auto.workflow import (
    VASP_ERROR_FIXES,
    apply_error_fixes,
    parse_magmoms,
    parse_outcar_summary,
    run_one_case,
)
from vasp_auto_ui import server as ui_server


SPIN_OUTCAR = """ vasp.6.4.2 mock output
   number of electron      16.0000000 magnetization       4.2000000
  free  energy   TOTEN  =       -20.00000000 eV
  energy  without entropy=      -19.99800000  energy(sigma->0) =      -19.99900000

 magnetization (x)

# of ion       s       p       d       tot
------------------------------------------
    1        0.030   0.010   2.500   2.540
    2        0.020   0.005   1.600   1.625
--------------------------------------------------
tot          0.050   0.015   4.100   4.165

  aborting loop because EDIFF is reached
"""


# ---------------------------------------------------------------- calc types

def test_hse06_calc_type():
    assert parse_calc_type("HSE06") is CalcType.HSE06
    assert load_incar_template("hse06").startswith("SYSTEM = vasp_auto_hse06")


def test_every_calc_type_has_a_description():
    assert set(CALC_TYPE_INFO) == set(CalcType)
    assert all(CALC_TYPE_INFO[t] for t in CalcType)


# ---------------------------------------------------------------- incar/spin

def test_parse_magmom_map():
    assert parse_magmom_map("Fe:5.0, O:0.6") == {"Fe": 5.0, "O": 0.6}
    assert parse_magmom_map(None) == {}
    with pytest.raises(ValueError):
        parse_magmom_map("Fe")


def test_magmom_line_uses_map_and_defaults():
    line = magmom_line(["Fe", "O", "Xx"], [2, 4, 1], {"O": 0.8})
    assert line == "2*5.0 4*0.8 1*0.6"


def test_spin_incar_text_adds_ispin_and_magmom(scf_case):
    text = spin_incar_text("ENCUT = 400\n", scf_case / "POSCAR")
    assert get_incar_value(text, "ISPIN") == "2"
    assert get_incar_value(text, "MAGMOM") == "1*0.6 1*0.6"


def test_spin_incar_text_keeps_user_magmom(scf_case):
    text = spin_incar_text("MAGMOM = 2*3.0\nISPIN = 1\n", scf_case / "POSCAR")
    assert get_incar_value(text, "ISPIN") == "2"
    assert get_incar_value(text, "MAGMOM") == "2*3.0"


def test_create_job_with_spin(scf_case, tmp_path, potcar_library):
    case_info = make_case_info(scf_case, tmp_path / "jobs", single_mode=True)
    job_dir = create_job_from_case(
        case_info, potcar_root=str(potcar_library), spin=True, magmom_map={"Al": 1.5}
    )
    incar = (job_dir / "INCAR").read_text()
    assert get_incar_value(incar, "ISPIN") == "2"
    assert get_incar_value(incar, "MAGMOM") == "1*1.5 1*0.6"


# ---------------------------------------------------------------- magmom parsing

def test_parse_outcar_summary_spin_fields(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(SPIN_OUTCAR, encoding="utf-8")
    summary = parse_outcar_summary(outcar)
    assert summary["energy_eV"] == -20.0
    assert summary["energy_without_entropy_eV"] == -19.998
    assert summary["magmom_total"] == 4.2
    assert summary["converged"] is True


def test_parse_magmoms_block(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(SPIN_OUTCAR, encoding="utf-8")
    assert parse_magmoms(outcar) == [2.540, 1.625]


def test_parse_magmoms_missing(tmp_path):
    assert parse_magmoms(tmp_path / "OUTCAR") is None


# ---------------------------------------------------------------- auto-retry

def test_apply_error_fixes(tmp_path):
    (tmp_path / "INCAR").write_text("ENCUT = 400\n", encoding="utf-8")
    applied = apply_error_fixes(tmp_path, [{"code": "EDDDAV", "file": "run.log", "hint": ""}])
    assert applied == ["ALGO = All"]
    assert "ALGO = All" in (tmp_path / "INCAR").read_text()


def test_fixes_only_for_known_codes():
    known_codes = {"ZBRENT", "EDDDAV", "RHOSYG", "SUBSPACE", "ZPOTRF", "PRICEL", "SGRCON"}
    assert set(VASP_ERROR_FIXES) == known_codes


def test_run_one_case_auto_retry(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "INCAR").write_text("ENCUT = 400\n", encoding="utf-8")
    calls = []

    def fake_run_vasp(job_path, exe, cpus=None, on_progress=None):
        calls.append(job_path)
        if len(calls) == 1:
            (job_dir / "run.log").write_text("Error EDDDAV: failure\n", encoding="utf-8")
            return 1
        (job_dir / "run.log").write_text("clean run\n", encoding="utf-8")
        (job_dir / "OUTCAR").write_text(
            " free  energy   TOTEN  =  -1.0 eV\n aborting loop because EDIFF is reached\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("vasp_auto.workflow.run_vasp", fake_run_vasp)
    case_info = {"case_name": "job", "job_dir": job_dir, "calculation_type": "scf"}
    row = run_one_case("proj", "single", case_info, "vasp_std", auto_retry=2)

    assert len(calls) == 2
    assert row["auto_retries"] == 1
    assert row["auto_fixes"] == "ALGO = All"
    assert row["return_code"] == 0
    assert "errors" not in row
    assert "ALGO = All" in (job_dir / "INCAR").read_text()


def test_run_one_case_no_retry_by_default(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "INCAR").write_text("ENCUT = 400\n", encoding="utf-8")
    calls = []

    def fake_run_vasp(job_path, exe, cpus=None, on_progress=None):
        calls.append(job_path)
        (job_dir / "run.log").write_text("Error EDDDAV: failure\n", encoding="utf-8")
        return 1

    monkeypatch.setattr("vasp_auto.workflow.run_vasp", fake_run_vasp)
    case_info = {"case_name": "job", "job_dir": job_dir, "calculation_type": "scf"}
    row = run_one_case("proj", "single", case_info, "vasp_std")

    assert len(calls) == 1
    assert "EDDDAV" in row["errors"]
    assert "auto_retries" not in row


# ---------------------------------------------------------------- SIGMA scan

def test_parse_sigma_values():
    assert parse_sigma_values("0.2, 0.1,0.05") == [0.2, 0.1, 0.05]
    with pytest.raises(ValueError):
        parse_sigma_values(None)


def _sigma_row(sigma, entropy, converged=True):
    return {
        "sigma": sigma,
        "energy_eV": -10.0,
        "entropy_eV_per_atom": entropy,
        "converged": converged,
    }


def test_select_sigma_picks_largest_within_tolerance():
    rows = [_sigma_row(0.2, 0.005), _sigma_row(0.1, 0.0008), _sigma_row(0.05, 0.0002)]
    assert _select_sigma_trial(rows, 1e-3)["sigma"] == 0.1


def test_select_sigma_falls_back_to_smallest_entropy():
    rows = [_sigma_row(0.2, 0.005), _sigma_row(0.1, 0.003)]
    assert _select_sigma_trial(rows, 1e-3)["sigma"] == 0.1


def test_select_sigma_no_energies():
    assert _select_sigma_trial([], 1e-3) is None


# ---------------------------------------------------------------- interstitial

def test_add_interstitial(scf_case):
    struct = read_poscar(scf_case / "POSCAR")
    edited = add_interstitial(struct, "H", (0.25, 0.25, 0.25))
    assert sum(edited["counts"]) == 3
    assert per_atom_symbols(edited) == ["Al", "O", "H"]
    assert edited["coords"][-1] == [0.25, 0.25, 0.25]
    # original untouched
    assert sum(struct["counts"]) == 2


def test_apply_spin_to_incar_roundtrip(tmp_path, scf_case):
    incar = tmp_path / "INCAR"
    incar.write_text("ENCUT = 400\n", encoding="utf-8")
    apply_spin_to_incar(incar, scf_case / "POSCAR", {"Al": 2.0})
    text = incar.read_text()
    assert get_incar_value(text, "ISPIN") == "2"
    assert get_incar_value(text, "MAGMOM") == "1*2.0 1*0.6"


# ---------------------------------------------------------------- UI server

def test_build_cli_args_new_flags():
    args = ui_server.build_cli_args(
        {
            "target": "inputs/Al",
            "mode": "run",
            "spin": True,
            "magmom": "Fe:5.0",
            "auto_retry": 2,
            "converge_sigma": "0.2,0.1",
            "sigma_tol": "1e-3",
            "reuse_wavecar": True,
        }
    )
    assert args[0] == "inputs/Al"
    assert "--spin" in args
    assert args[args.index("--magmom") + 1] == "Fe:5.0"
    assert args[args.index("--auto-retry") + 1] == "2"
    assert args[args.index("--converge-sigma") + 1] == "0.2,0.1"
    assert args[args.index("--sigma-tol") + 1] == "1e-3"
    assert "--reuse-wavecar" in args


# ---------------------------------------------------------------- TASK 6: solvation

def test_create_job_with_solvation(scf_case, tmp_path, potcar_library):
    """--solvation injects LSOL and EB_K into the INCAR."""
    from vasp_auto.incar import get_incar_value
    from vasp_auto.job_manager import DEFAULT_SOLVATION_EPS

    case_info = {
        "case_name": "scf",
        "case_dir": scf_case,
        "job_dir": tmp_path / "jobs" / "solvated",
        "calculation_type": "scf",
    }
    create_job_from_case(case_info, potcar_root=str(potcar_library), solvation=True)

    incar_text = (tmp_path / "jobs" / "solvated" / "INCAR").read_text(encoding="utf-8")
    assert get_incar_value(incar_text, "LSOL") == ".TRUE."
    eb_k_str = get_incar_value(incar_text, "EB_K")
    assert eb_k_str is not None
    assert abs(float(eb_k_str) - DEFAULT_SOLVATION_EPS) < 0.01


def test_create_job_with_solvation_custom_eps(scf_case, tmp_path, potcar_library):
    """--solvation-eps overrides the default dielectric constant."""
    from vasp_auto.incar import get_incar_value

    case_info = {
        "case_name": "scf",
        "case_dir": scf_case,
        "job_dir": tmp_path / "jobs" / "solvated_acetonitrile",
        "calculation_type": "scf",
    }
    create_job_from_case(
        case_info, potcar_root=str(potcar_library), solvation=True, solvation_eps=36.6
    )

    incar_text = (tmp_path / "jobs" / "solvated_acetonitrile" / "INCAR").read_text(encoding="utf-8")
    eb_k_str = get_incar_value(incar_text, "EB_K")
    assert eb_k_str is not None
    assert abs(float(eb_k_str) - 36.6) < 0.01


def test_create_job_without_solvation_has_no_lsol(scf_case, tmp_path, potcar_library):
    """Without --solvation, LSOL must not appear in the INCAR."""
    from vasp_auto.incar import get_incar_value

    case_info = {
        "case_name": "scf",
        "case_dir": scf_case,
        "job_dir": tmp_path / "jobs" / "no_sol",
        "calculation_type": "scf",
    }
    create_job_from_case(case_info, potcar_root=str(potcar_library), solvation=False)

    incar_text = (tmp_path / "jobs" / "no_sol" / "INCAR").read_text(encoding="utf-8")
    assert get_incar_value(incar_text, "LSOL") is None
