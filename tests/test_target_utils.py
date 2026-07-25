import pytest

from vasp_auto.target_utils import filter_case_dirs, get_case_type, inspect_target


def test_get_case_type_scf(scf_case):
    assert get_case_type(scf_case) == "scf"


def test_get_case_type_tss(tss_case):
    assert get_case_type(tss_case) == "tss"


def test_get_case_type_unknown(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert get_case_type(empty) is None


def test_get_case_type_ignores_poscar_named_subdir(tmp_path):
    # A project root holding a case folder literally named "POSCAR" must not be
    # mistaken for a case itself (POSCAR must be a *file*), or the whole case
    # list collapses to one bogus entry instead of listing the real cases.
    project = tmp_path / "project"
    (project / "POSCAR").mkdir(parents=True)        # a case folder named POSCAR
    (project / "POSCAR" / "POSCAR").write_text("x")  # ...whose POSCAR is a file
    (project / "Fe").mkdir()
    (project / "Fe" / "POSCAR").write_text("x")

    assert get_case_type(project) is None  # the parent is not itself a case

    info = inspect_target(project)
    assert info["mode"] == "project"
    assert set(info["case_types"]) == {"POSCAR", "Fe"}


def test_inspect_target_single_mode(scf_case):
    info = inspect_target(scf_case)
    assert info["mode"] == "single"
    assert info["project_name"] == scf_case.name
    assert info["case_dirs"] == [scf_case]


def test_inspect_target_project_mode(scf_case, tss_case):
    project_root = scf_case.parent
    info = inspect_target(project_root)
    assert info["mode"] == "project"
    assert set(info["case_types"].values()) == {"scf", "tss"}


def test_inspect_target_rejects_empty(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        inspect_target(empty)


def test_filter_case_dirs(scf_case, tss_case):
    cases = [scf_case, tss_case]
    assert filter_case_dirs(cases, None) == cases
    assert filter_case_dirs(cases, [tss_case.name]) == [tss_case]
