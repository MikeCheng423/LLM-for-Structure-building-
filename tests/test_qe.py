"""Tests for the Quantum ESPRESSO (pw.x) engine path."""
import pytest

from vasp_auto import qe_tools
from vasp_auto.config_loader import default_config, load_config
from vasp_auto.job_manager import create_job_from_case, make_case_info, preview_job_from_case
from vasp_auto.parser import parse_pw_final_structure, parse_pw_output
from vasp_auto.workflow import build_row, job_engine, should_retry_failed


POSCAR_SI = """Si bulk
5.43
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
Si
2
Direct
0.0 0.0 0.0
0.25 0.25 0.25
"""

PW_OUT = """
     Program PWSCF v.7.2 starts ...
     Self-consistent Calculation
     convergence has been achieved in  11 iterations
!    total energy              =     -15.83792000 Ry
     Total force =     0.000100     Total SCF correction =     0.000001
     number of bfgs steps      =      4

     JOB DONE.
"""


@pytest.fixture
def qe_case(tmp_path):
    case = tmp_path / "Si"
    case.mkdir()
    (case / "POSCAR").write_text(POSCAR_SI, encoding="utf-8")
    pseudo = tmp_path / "pseudo"
    pseudo.mkdir()
    (pseudo / "Si.pbe-n-kjpaw_psl.1.0.0.UPF").write_text("<UPF/>", encoding="utf-8")
    config = {"pseudo_dir": str(pseudo), "pseudo_map": {}, "qe_executable": "pw.x"}
    case_info = make_case_info(case, tmp_path / "jobs")
    return case, config, case_info


# ----------------------------------------------------------------- pseudos

def test_find_pseudo_auto(tmp_path):
    pseudo = tmp_path / "p"
    pseudo.mkdir()
    (pseudo / "Fe.pbe-spn-kjpaw.UPF").write_text("x")
    assert qe_tools.find_pseudo("Fe", pseudo) == "Fe.pbe-spn-kjpaw.UPF"


def test_find_pseudo_map_wins(tmp_path):
    pseudo = tmp_path / "p"
    pseudo.mkdir()
    (pseudo / "Fe.default.UPF").write_text("x")
    name = qe_tools.find_pseudo("Fe", pseudo, {"Fe": "Fe.special.UPF"})
    assert name == "Fe.special.UPF"


def test_find_pseudo_missing_raises(tmp_path):
    pseudo = tmp_path / "p"
    pseudo.mkdir()
    with pytest.raises(FileNotFoundError):
        qe_tools.find_pseudo("Au", pseudo)


# ----------------------------------------------------------------- input gen

def test_build_pw_input_namelists(qe_case):
    case, config, case_info = qe_case
    preview = preview_job_from_case(
        case_info, calc_type="scf",
        kpoints_spec={"mode": "gamma", "mesh": "4x4x4"},
        engine="qe", config=config,
    )
    text = preview["pw.in"]
    assert "calculation = 'scf'" in text
    assert "ibrav = 0" in text
    assert "nat = 2" in text and "ntyp = 1" in text
    assert "ATOMIC_SPECIES" in text
    assert "Si.pbe-n-kjpaw_psl.1.0.0.UPF" in text
    assert "CELL_PARAMETERS angstrom" in text
    assert "ATOMIC_POSITIONS crystal" in text
    assert "K_POINTS automatic" in text and "4 4 4" in text


def test_ecutwfc_from_config_override(qe_case):
    case, config, case_info = qe_case
    config = {**config, "qe_ecutwfc": 80.0, "qe_ecutrho": 640.0}
    preview = preview_job_from_case(case_info, calc_type="scf", engine="qe", config=config)
    assert "ecutwfc = 80" in preview["pw.in"]
    assert "ecutrho = 640" in preview["pw.in"]


def test_calc_type_mapping():
    assert qe_tools.QE_CALC_MAP["vcrelax"] == "vc-relax"
    assert qe_tools.QE_CALC_MAP["dos"] == "nscf"
    assert qe_tools.QE_CALC_MAP["bands"] == "bands"


def test_vcrelax_emits_cell_namelist(qe_case):
    case, config, case_info = qe_case
    preview = preview_job_from_case(case_info, calc_type="vcrelax", engine="qe", config=config)
    assert "calculation = 'vc-relax'" in preview["pw.in"]
    assert "&CELL" in preview["pw.in"]
    assert "&IONS" in preview["pw.in"]


def test_bands_needs_kpath(qe_case):
    case, config, case_info = qe_case
    with pytest.raises(ValueError):
        preview_job_from_case(case_info, calc_type="bands", engine="qe", config=config)


def test_bands_kpath_crystal_b(qe_case):
    case, config, case_info = qe_case
    preview = preview_job_from_case(
        case_info, calc_type="bands",
        kpoints_spec={"mode": "line", "kpath": "fcc", "divisions": 20},
        engine="qe", config=config,
    )
    assert "K_POINTS crystal_b" in preview["pw.in"]


def test_phonon_generates_native_stage(qe_case):
    case, config, case_info = qe_case
    preview = preview_job_from_case(case_info, calc_type="phonon", engine="qe", config=config)
    assert "&INPUTPH" in preview["ph.in"]
    assert [stage["program"] for stage in preview["stages"]] == ["pw.x", "ph.x"]


def test_spin_adds_starting_magnetization(qe_case):
    case, config, case_info = qe_case
    preview = preview_job_from_case(
        case_info, calc_type="scf", engine="qe", config=config,
        spin=True, magmom_map={"Si": 0.3},
    )
    assert "nspin = 2" in preview["pw.in"]
    assert "starting_magnetization(1) = 0.3" in preview["pw.in"]


# ----------------------------------------------------------------- job dir

def test_create_qe_job_writes_inputs(qe_case):
    case, config, case_info = qe_case
    job_dir = create_job_from_case(
        case_info, calc_type="relax",
        kpoints_spec={"mode": "mp", "mesh": "6x6x6"},
        engine="qe", config=config,
    )
    assert (job_dir / "pw.in").exists()
    assert (job_dir / ".engine").read_text().strip() == "qe"
    assert (job_dir / "POSCAR").exists()
    assert (job_dir / "pseudo" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF").exists()
    assert job_engine(job_dir) == "qe"


def test_user_pw_in_takes_precedence(qe_case):
    case, config, case_info = qe_case
    (case / "pw.in").write_text("&CONTROL\n  calculation = 'scf'\n/\nCUSTOM\n", encoding="utf-8")
    job_dir = create_job_from_case(case_info, calc_type="scf", engine="qe", config=config)
    assert "CUSTOM" in (job_dir / "pw.in").read_text()


def test_qe_tss_generates_neb_input(tmp_path):
    case = tmp_path / "tss"
    for end in ("initial", "final"):
        (case / end).mkdir(parents=True)
        (case / end / "POSCAR").write_text(POSCAR_SI, encoding="utf-8")
    (tmp_path / "Si.test.UPF").write_text("<UPF/>", encoding="utf-8")
    case_info = make_case_info(case, tmp_path / "jobs")
    job = create_job_from_case(
        case_info, engine="qe", config={"pseudo_dir": str(tmp_path)}, neb_images=5,
    )
    neb = (job / "neb.in").read_text(encoding="utf-8")
    assert "BEGIN_PATH_INPUT" in neb
    assert "num_of_images = 7" in neb
    assert "FIRST_IMAGE" in neb and "LAST_IMAGE" in neb
    stages = __import__("json").loads((job / "qe_stages.json").read_text())["stages"]
    assert stages[0]["program"] == "neb.x"


@pytest.mark.parametrize(
    ("calc_type", "programs", "input_fragment"),
    [
        ("dos", ["pw.x", "pw.x", "dos.x", "projwfc.x"], "&DOS"),
        ("bands", ["pw.x", "pw.x", "bands.x"], "&BANDS"),
        ("freq", ["pw.x", "ph.x", "dynmat.x"], "&INPUTPH"),
        ("optics", ["pw.x", "pw.x", "epsilon.x"], "&ENERGY_GRID"),
        ("charge", ["pw.x", "pp.x"], "plot_num = 0"),
        ("workfunction", ["pw.x", "pp.x", "average.x"], "plot_num = 11"),
    ],
)
def test_qe_companion_stage_plans(qe_case, calc_type, programs, input_fragment):
    _case, config, case_info = qe_case
    kwargs = {"kpoints_spec": {"kpath": "fcc"}} if calc_type == "bands" else {}
    preview = preview_job_from_case(case_info, calc_type=calc_type, engine="qe", config=config, **kwargs)
    assert [stage["program"] for stage in preview["stages"]] == programs
    assert any(input_fragment in value for key, value in preview.items()
               if key.endswith(".in") and isinstance(value, str))


def test_qe_md_and_hse_parameters(qe_case):
    _case, config, case_info = qe_case
    md = preview_job_from_case(case_info, calc_type="md", engine="qe", config=config)["pw.in"]
    assert "calculation = 'md'" in md and "ion_temperature = 'rescaling'" in md
    hse = preview_job_from_case(case_info, calc_type="hse06", engine="qe", config=config)["pw.in"]
    assert "input_dft = 'HSE'" in hse and "screening_parameter = 0.106" in hse


def test_qe_d_band_center_reads_projwfc_files(tmp_path):
    from vasp_auto.analysis import qe_d_band_center

    job = tmp_path / "dos"
    job.mkdir()
    (job / "pw.out").write_text("the Fermi energy is 0.0000 ev\n", encoding="utf-8")
    (job / "pdos.pdos_atm#1(Fe)_wfc#3(d)").write_text(
        "# E (eV) ldos(E) pdos(E)\n-2 1 1\n-1 1 1\n0 1 1\n",
        encoding="utf-8",
    )
    result = qe_d_band_center(job, [1])
    assert result["d_band_center_eV"] == pytest.approx(-1.0)
    assert result["d_band_width_eV"] == pytest.approx(0.5)


def test_qe_cube_charge_difference(tmp_path):
    from vasp_auto.qe_volumetric import cube_difference, read_cube

    header = (
        "total\nQE pp.x\n 1 0 0 0\n 2 1 0 0\n 1 0 1 0\n 1 0 0 1\n"
        " 14 0 0 0 0\n"
    )
    total = tmp_path / "total.cube"
    part = tmp_path / "part.cube"
    total.write_text(header + " 3.0 5.0\n", encoding="utf-8")
    part.write_text(header + " 1.0 2.0\n", encoding="utf-8")
    output = tmp_path / "diff.cube"
    cube_difference(total, [part], output)
    assert read_cube(output)["data"] == pytest.approx([2.0, 3.0])


def test_qe_upf_zval_formats(tmp_path):
    from vasp_auto.chgcar import _upf_zval

    xml = tmp_path / "xml.UPF"
    legacy = tmp_path / "legacy.UPF"
    xml.write_text('<PP_HEADER z_valence="8.000000"/>\n', encoding="utf-8")
    legacy.write_text("  4.000000 Z valence\n", encoding="utf-8")
    assert _upf_zval(xml) == 8.0
    assert _upf_zval(legacy) == 4.0


def test_aggregate_qe_pdos_spin_and_selection(tmp_path):
    from vasp_auto.parser import aggregate_qe_pdos

    job = tmp_path / "dos"
    job.mkdir()
    (job / "pw.out").write_text("the Fermi energy is 1.5 ev\n", encoding="utf-8")
    for atom, scale in ((1, 1.0), (2, 10.0)):
        (job / f"pdos.pdos_atm#{atom}(Fe)_wfc#2(d)").write_text(
            "# E (eV) ldosup(E) ldosdw(E)\n"
            f"0 {scale} {2 * scale}\n1 {3 * scale} {4 * scale}\n",
            encoding="utf-8",
        )
    result = aggregate_qe_pdos(job, ["Fe", "Fe"], atoms=[1])
    assert result["efermi"] == 1.5
    assert len(result["curves"]) == 2
    assert result["curves"][0]["values"] == [1.0, 3.0]
    assert result["curves"][1]["values"] == [2.0, 4.0]


def test_run_qe_executes_manifest_and_marks_completion(tmp_path, monkeypatch):
    import json
    from vasp_auto.runner import run_qe

    job = tmp_path / "job"
    binary = tmp_path / "bin"
    job.mkdir()
    binary.mkdir()
    for program in ("pw.x", "ph.x"):
        (binary / program).write_text("", encoding="utf-8")
    for name in ("pw.in", "ph.in"):
        (job / name).write_text("&INPUT\n/\n", encoding="utf-8")
    (job / "qe_stages.json").write_text(json.dumps({"stages": [
        {"program": "pw.x", "input": "pw.in", "output": "pw.out", "mpi": True},
        {"program": "ph.x", "input": "ph.in", "output": "ph.out", "mpi": True},
    ]}), encoding="utf-8")
    commands = []

    class Process:
        stdout = ["JOB DONE.\n"]

        def wait(self):
            return 0

    def fake_popen(command, **_kwargs):
        commands.append(command)
        return Process()

    monkeypatch.setattr("vasp_auto.runner.subprocess.Popen", fake_popen)
    assert run_qe(str(job), str(binary / "pw.x"), cpus=2) == 0
    assert [command[-3] for command in commands] == [str(binary / "pw.x"), str(binary / "ph.x")]
    assert (job / ".qe_complete").exists()


# ----------------------------------------------------------------- parsing

def test_parse_pw_output(tmp_path):
    pw_out = tmp_path / "pw.out"
    pw_out.write_text(PW_OUT, encoding="utf-8")
    summary = parse_pw_output(pw_out)
    assert summary["converged"] is True
    assert summary["ionic_steps"] == 4
    assert summary["energy_eV"] == pytest.approx(-15.83792 * qe_tools.RY_TO_EV)
    assert summary["max_force_eV_A"] is not None


def test_parse_pw_output_missing(tmp_path):
    assert parse_pw_output(tmp_path / "none.out") is None


DATA_FILE_SCHEMA = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
 <output>
  <band_structure>
   <fermi_energy>0.5</fermi_energy>
   <ks_energies>
    <k_point weight="0.5">0.0 0.0 0.0</k_point>
    <eigenvalues size="3">-0.1 0.0 0.2</eigenvalues>
    <occupations size="3">1.0 1.0 0.0</occupations>
   </ks_energies>
   <ks_energies>
    <k_point weight="0.5">0.5 0.0 0.0</k_point>
    <eigenvalues size="3">-0.05 0.1 0.3</eigenvalues>
    <occupations size="3">1.0 0.0 0.0</occupations>
   </ks_energies>
  </band_structure>
 </output>
</qes:espresso>
"""


def _qe_dos_job(tmp_path):
    from vasp_auto.parser import HARTREE_TO_EV

    job = tmp_path / "qe_dos"
    save = job / "tmp" / "vasp_auto.save"
    save.mkdir(parents=True)
    (job / ".engine").write_text("qe", encoding="utf-8")
    (save / "data-file-schema.xml").write_text(DATA_FILE_SCHEMA, encoding="utf-8")
    return job, HARTREE_TO_EV


def test_parse_qe_eigenvalues(tmp_path):
    from vasp_auto.parser import parse_qe_eigenvalues

    job, h = _qe_dos_job(tmp_path)
    data = parse_qe_eigenvalues(job)
    assert data["efermi"] == pytest.approx(0.5 * h)
    assert len(data["kpoints"]) == 2
    assert data["eigenvalues"][0][0] == pytest.approx(-0.1 * h)
    assert data["weights"] == pytest.approx([0.5, 0.5])


def test_parse_qe_dos(tmp_path):
    from vasp_auto.parser import parse_qe_dos

    job, h = _qe_dos_job(tmp_path)
    dos = parse_qe_dos(job)
    assert dos["efermi"] == pytest.approx(0.5 * h)
    assert len(dos["energies"]) == len(dos["total"][0])
    assert max(dos["total"][0]) > 0  # peaks where eigenvalues sit
    assert parse_qe_dos(tmp_path / "not_a_job") is None


def test_parse_qe_bands(tmp_path):
    from vasp_auto.parser import parse_qe_bands

    job, h = _qe_dos_job(tmp_path)
    bands = parse_qe_bands(job)
    assert bands["distances"] == pytest.approx([0.0, 0.5])
    assert len(bands["bands"][0]) == 3          # 3 bands, one spin channel
    assert len(bands["bands"][0][0]) == 2        # over 2 k-points


def test_parse_pw_final_structure(tmp_path):
    pw_out = tmp_path / "pw.out"
    pw_out.write_text(
        "Begin final coordinates\n"
        "CELL_PARAMETERS (angstrom)\n"
        "  3.0 0.0 0.0\n  0.0 3.0 0.0\n  0.0 0.0 3.0\n"
        "ATOMIC_POSITIONS (crystal)\n"
        "  Si 0.0 0.0 0.0\n  Si 0.5 0.5 0.5\n"
        "End final coordinates\n",
        encoding="utf-8",
    )
    struct = parse_pw_final_structure(pw_out)
    assert struct["elements"] == ["Si"]
    assert struct["counts"] == [2]
    assert struct["lattice"][0][0] == 3.0


# ----------------------------------------------------------------- rows

def test_build_row_qe(tmp_path):
    job_dir = tmp_path / "jobs" / "Si"
    job_dir.mkdir(parents=True)
    (job_dir / ".engine").write_text("qe\n")
    (job_dir / "pw.out").write_text(PW_OUT, encoding="utf-8")
    case = tmp_path / "Si"
    case.mkdir()
    (case / "POSCAR").write_text(POSCAR_SI, encoding="utf-8")
    case_info = make_case_info(case, tmp_path / "jobs")
    row = build_row("p", "project", case_info)
    assert row["engine"] == "qe"
    assert row["converged"] is True
    assert row["status"] == "done"
    assert should_retry_failed(case_info) is False


def test_should_retry_unconverged_qe(tmp_path):
    job_dir = tmp_path / "jobs" / "Si"
    job_dir.mkdir(parents=True)
    (job_dir / ".engine").write_text("qe\n")
    (job_dir / "pw.out").write_text("!    total energy = -1.0 Ry\n", encoding="utf-8")
    case = tmp_path / "Si"
    case.mkdir()
    (case / "POSCAR").write_text(POSCAR_SI, encoding="utf-8")
    case_info = make_case_info(case, tmp_path / "jobs")
    assert should_retry_failed(case_info) is True


# ----------------------------------------------------------------- config

def test_default_config_has_engine():
    config = default_config()
    assert config["engine"] == "vasp"
    assert config["qe_executable"] == "pw.x"


def test_load_config_engine_default():
    config = load_config()
    assert config.get("engine", "vasp") in ("vasp", "qe")
