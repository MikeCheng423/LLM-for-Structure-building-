import pytest

import vasp_auto.job_manager as job_manager
from vasp_auto.job_manager import (
    _interpolate_coords,
    create_job_from_case,
    load_incar_template,
    make_case_info,
    preview_job_from_case,
)


def test_load_incar_template_prefers_vasp_auto_root(tmp_path, monkeypatch):
    example = tmp_path / "example"
    example.mkdir()
    (example / "INCAR_scf").write_text("SYSTEM = from_env_root\n", encoding="utf-8")
    monkeypatch.setenv("VASP_AUTO_ROOT", str(tmp_path))

    assert load_incar_template("scf") == "SYSTEM = from_env_root\n"


def test_load_incar_template_reads_repo_example():
    text = load_incar_template("scf")
    assert "ENCUT" in text


def test_load_incar_template_falls_back_to_builtin(monkeypatch):
    # With no template directories available, the built-in default applies.
    monkeypatch.setattr(job_manager, "_template_search_dirs", lambda: [])
    text = load_incar_template("neb")
    assert "LCLIMB" in text
    assert "vasp_auto_neb" in text


def test_load_incar_template_rejects_unknown_type():
    with pytest.raises(ValueError):
        load_incar_template("not_a_type")


def test_interpolate_coords_wraps_periodic_boundary():
    initial = {"coord_mode": "Direct", "coords": [[0.9, 0.0, 0.0]]}
    final = {"coord_mode": "Direct", "coords": [[0.1, 0.0, 0.0]]}
    coords = _interpolate_coords(initial, final, step=1, total_steps=2)
    assert coords[0][0] == pytest.approx(0.0)


def test_make_case_info_rejects_unknown_case(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        make_case_info(empty, tmp_path / "jobs")


def test_create_job_from_case_scf(scf_case, potcar_library, tmp_path):
    case_info = make_case_info(scf_case, tmp_path / "jobs", single_mode=True)
    job_dir = create_job_from_case(case_info, potcar_root=str(potcar_library))

    assert (job_dir / "POSCAR").exists()
    assert "ENCUT" in (job_dir / "INCAR").read_text()
    assert (job_dir / "KPOINTS").exists()
    assert (job_dir / "POTCAR").read_text() == "POTCAR Al\nPOTCAR O\n"


def test_create_job_from_case_does_not_touch_case_dir(scf_case, potcar_library, tmp_path):
    case_info = make_case_info(scf_case, tmp_path / "jobs", single_mode=True)
    create_job_from_case(case_info, potcar_root=str(potcar_library))

    # Generated INCAR/KPOINTS/POTCAR must not leak into the user's input dir.
    assert sorted(p.name for p in scf_case.iterdir()) == ["POSCAR"]


def test_create_job_with_calc_type_and_kmesh(scf_case, potcar_library, tmp_path):
    case_info = make_case_info(scf_case, tmp_path / "jobs", single_mode=True)
    job_dir = create_job_from_case(
        case_info,
        potcar_root=str(potcar_library),
        calc_type="relax",
        kpoints_spec={"mode": "mp", "mesh": "4x4x1"},
    )

    incar = (job_dir / "INCAR").read_text()
    assert "IBRION = 2" in incar
    kpoints = (job_dir / "KPOINTS").read_text()
    assert "Monkhorst-Pack" in kpoints and "4 4 1" in kpoints


def test_preview_job_from_case(scf_case, potcar_library, tmp_path):
    case_info = make_case_info(scf_case, tmp_path / "jobs", single_mode=True)
    preview = preview_job_from_case(
        case_info,
        potcar_root=str(potcar_library),
        potcar_map={"O": "O_s"},
        calc_type="scf",
    )

    assert "ENCUT" in preview["INCAR"]
    assert preview["POTCAR"] == "Al, O_s"
    assert "Gamma" in preview["KPOINTS"]
    # Nothing was written.
    assert not (tmp_path / "jobs").exists() or not any((tmp_path / "jobs").iterdir())


def test_create_job_from_case_tss_interpolates_images(tss_case, potcar_library, tmp_path):
    case_info = make_case_info(tss_case, tmp_path / "jobs", single_mode=True)
    job_dir = create_job_from_case(
        case_info, potcar_root=str(potcar_library), neb_images=3
    )

    image_dirs = sorted(p.name for p in job_dir.iterdir() if p.is_dir())
    assert image_dirs == ["00", "01", "02", "03", "04"]
    assert "IMAGES = 3" in (job_dir / "INCAR").read_text()
    assert (job_dir / "POTCAR").exists()


# ----------------------------------------------------- numbered job folders

def test_make_case_info_new_allocates_sequential(scf_case, tmp_path):
    out = tmp_path / "jobs"
    name = scf_case.name
    first = make_case_info(scf_case, out, job_mode="new")["job_dir"]
    second = make_case_info(scf_case, out, job_mode="new")["job_dir"]
    assert first.name == f"0001_{name}"
    assert second.name == f"0002_{name}"
    # "new" claims the directory atomically, so reruns never overwrite.
    assert first.is_dir() and second.is_dir()
    assert first != second


def test_make_case_info_new_is_global_across_cases(scf_case, tmp_path):
    # A second case sharing the same output_root continues the same counter.
    other = tmp_path / "Other"
    other.mkdir()
    (other / "POSCAR").write_text((scf_case / "POSCAR").read_text(), encoding="utf-8")
    out = tmp_path / "jobs"
    a = make_case_info(scf_case, out, job_mode="new")["job_dir"]
    b = make_case_info(other, out, job_mode="new")["job_dir"]
    assert a.name == f"0001_{scf_case.name}"
    assert b.name == "0002_Other"


def test_make_case_info_latest_resolves_highest(scf_case, tmp_path):
    out = tmp_path / "jobs"
    make_case_info(scf_case, out, job_mode="new")          # 0001
    make_case_info(scf_case, out, job_mode="new")          # 0002
    latest = make_case_info(scf_case, out, job_mode="latest")["job_dir"]
    assert latest.name == f"0002_{scf_case.name}"


def test_make_case_info_preview_does_not_create(scf_case, tmp_path):
    out = tmp_path / "jobs"
    predicted = make_case_info(scf_case, out, job_mode="preview")["job_dir"]
    assert predicted.name == f"0001_{scf_case.name}"
    assert not predicted.exists()


def test_make_case_info_bare_is_unchanged(scf_case, tmp_path):
    out = tmp_path / "jobs"
    job_dir = make_case_info(scf_case, out, job_mode="bare")["job_dir"]
    assert job_dir == (out / scf_case.name)
