from pathlib import Path

import pytest
from conftest import OUTCAR_TEXT

from vasp_auto.chain import load_workflow_spec, parse_workflow_steps, run_workflow_case
from vasp_auto.calc_types import CalcType
from vasp_auto.job_manager import make_case_info


def fake_run(job_dir, exe, cpus=None):
    job_dir = Path(job_dir)
    (job_dir / "OUTCAR").write_text(OUTCAR_TEXT, encoding="utf-8")
    (job_dir / "CONTCAR").write_text((job_dir / "POSCAR").read_text() + "! relaxed\n", encoding="utf-8")
    (job_dir / "CHGCAR").write_text("charge density\n", encoding="utf-8")
    return 0


def test_parse_workflow_steps_from_string():
    steps = parse_workflow_steps("relax, scf,dos")
    assert [step["calc_type"] for step in steps] == [CalcType.RELAX, CalcType.SCF, CalcType.DOS]


def test_parse_workflow_steps_rejects_neb():
    with pytest.raises(ValueError):
        parse_workflow_steps("relax,neb")


def test_load_workflow_spec_priority(scf_case):
    (scf_case / "workflow.yaml").write_text(
        "steps:\n  - calc_type: relax\n  - calc_type: scf\n    incar:\n      ENCUT: 450\n",
        encoding="utf-8",
    )
    # CLI flag beats the file.
    steps = load_workflow_spec(scf_case, {}, "scf")
    assert len(steps) == 1

    steps = load_workflow_spec(scf_case, {}, None)
    assert len(steps) == 2
    assert steps[1]["incar"] == {"ENCUT": 450}


def test_run_workflow_case_chains_outputs(scf_case, potcar_library, tmp_path):
    case_info = make_case_info(scf_case, tmp_path / "jobs", single_mode=True)
    config = {"vasp_executable": "fake_vasp", "potcar_root": str(potcar_library)}

    rows = run_workflow_case(case_info, "relax,scf,dos", config, run_fn=fake_run)

    job_dir = Path(case_info["job_dir"])
    assert (job_dir / "01_relax" / "POSCAR").exists()
    # The scf step starts from the relaxed CONTCAR.
    scf_poscar = (job_dir / "02_scf" / "POSCAR").read_text()
    assert "! relaxed" in scf_poscar
    # The dos step receives the CHGCAR and a non-SCF INCAR.
    assert (job_dir / "03_dos" / "CHGCAR").exists()
    assert "ICHARG = 11" in (job_dir / "03_dos" / "INCAR").read_text()
    # POTCAR present in every step.
    assert all((job_dir / step / "POTCAR").exists() for step in ("01_relax", "02_scf", "03_dos"))

    assert [row["status"] for row in rows] == ["done", "done", "done"]
    assert all(row["converged"] for row in rows)
    assert rows[0]["case"].endswith(":01_relax")


def test_run_workflow_case_prepare_only(scf_case, potcar_library, tmp_path):
    case_info = make_case_info(scf_case, tmp_path / "jobs", single_mode=True)
    config = {"vasp_executable": "fake_vasp", "potcar_root": str(potcar_library)}

    rows = run_workflow_case(case_info, "relax,scf", config, prepare_only=True)

    job_dir = Path(case_info["job_dir"])
    assert (job_dir / "01_relax" / "INCAR").exists()
    assert not (job_dir / "01_relax" / "OUTCAR").exists()
    assert [row["status"] for row in rows] == ["prepared", "prepared"]


def test_run_workflow_step_kpoints_override(scf_case, potcar_library, tmp_path):
    case_info = make_case_info(scf_case, tmp_path / "jobs", single_mode=True)
    config = {"vasp_executable": "fake_vasp", "potcar_root": str(potcar_library)}
    steps = [{"calc_type": "scf"}, {"calc_type": "dos", "kpoints": "8x8x8"}]

    run_workflow_case(case_info, steps, config, prepare_only=True)

    kpoints = (Path(case_info["job_dir"]) / "02_dos" / "KPOINTS").read_text()
    assert "8 8 8" in kpoints


def test_run_workflow_step_reads_per_type_files(scf_case, potcar_library, tmp_path):
    # INCAR_<type>/KPOINTS_<type> saved in the case (the workflow editor) are the
    # base for that step, no workflow.yaml overrides needed.
    (scf_case / "KPOINTS_dos").write_text("Automatic mesh\n0\nGamma\n6 6 6\n0 0 0\n")
    (scf_case / "INCAR_dos").write_text("ISMEAR = -5\nNEDOS = 3001\n")
    case_info = make_case_info(scf_case, tmp_path / "jobs", single_mode=True)
    config = {"vasp_executable": "fake_vasp", "potcar_root": str(potcar_library)}

    run_workflow_case(case_info, "scf,dos", config, prepare_only=True)

    step = Path(case_info["job_dir"]) / "02_dos"
    assert "6 6 6" in (step / "KPOINTS").read_text()
    assert "NEDOS = 3001" in (step / "INCAR").read_text()


# ------------------------------------------------- convergence workflow step

def test_parse_workflow_steps_accepts_converge():
    steps = parse_workflow_steps("converge,relax,scf")
    assert steps[0]["calc_type"] == "converge" and steps[0]["converge"] is True
    assert steps[1]["calc_type"] == CalcType.RELAX


def test_converge_step_carries_settings_forward(scf_case, potcar_library, tmp_path, monkeypatch):
    # Fake VASP for the convergence trials (converge_scf_case calls run_vasp).
    import vasp_auto.convergence as conv

    def fake_conv_run(job_dir, exe, cpus=None):
        Path(job_dir, "OUTCAR").write_text(OUTCAR_TEXT, encoding="utf-8")
        return 0

    monkeypatch.setattr(conv, "run_vasp", fake_conv_run)

    case_info = make_case_info(scf_case, tmp_path / "jobs", single_mode=True)
    config = {"vasp_executable": "fake_vasp", "potcar_root": str(potcar_library)}
    steps = [
        {"calc_type": "converge", "encut": "400,450", "kpoints": "3,5", "scan_nelm": False},
        {"calc_type": "scf"},
    ]

    rows = run_workflow_case(case_info, steps, config, run_fn=fake_run)

    job_dir = Path(case_info["job_dir"])
    # The converge step ran a scan and reported selected settings.
    assert rows[0]["calculation_type"] == "converge"
    assert rows[0]["status"] == "done"
    assert (job_dir / "01_converge" / "scf_convergence").is_dir()
    assert rows[0]["selected_encut"] in (400, 450)

    # The scf step inherits the converged ENCUT and k-mesh.
    scf_incar = (job_dir / "02_scf" / "INCAR").read_text()
    assert f"ENCUT = {rows[0]['selected_encut']}" in scf_incar
    selected_mesh = rows[0]["selected_kpoints"].split()[0]
    assert selected_mesh in (job_dir / "02_scf" / "KPOINTS").read_text()
    # The scf step starts from the case POSCAR (a scan does not relax).
    assert "! relaxed" not in (job_dir / "02_scf" / "POSCAR").read_text()


def test_converge_step_prepare_only_skips_scan(scf_case, potcar_library, tmp_path):
    case_info = make_case_info(scf_case, tmp_path / "jobs", single_mode=True)
    config = {"vasp_executable": "fake_vasp", "potcar_root": str(potcar_library)}
    rows = run_workflow_case(case_info, "converge,scf", config, prepare_only=True)
    assert rows[0]["status"] == "prepared"
    assert not (Path(case_info["job_dir"]) / "01_converge" / "scf_convergence").exists()
