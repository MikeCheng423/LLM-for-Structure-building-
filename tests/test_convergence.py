import pytest

from vasp_auto.convergence import (
    _select_converged_trial,
    parse_encut_values,
    parse_kpoint_meshes,
    parse_nelm_values,
    set_incar_value,
    write_gamma_kpoints,
)


def test_parse_encut_values():
    assert parse_encut_values("400, 450,500") == [400, 450, 500]
    with pytest.raises(ValueError):
        parse_encut_values(None)
    with pytest.raises(ValueError):
        parse_encut_values(" , ")


def test_parse_nelm_values_default():
    assert parse_nelm_values(None) == [40, 60, 80, 100]


def test_parse_nelm_values_custom():
    assert parse_nelm_values("40, 80,120") == [40, 80, 120]


def test_parse_nelm_values_rejects_empty():
    with pytest.raises(ValueError):
        parse_nelm_values(" , ")


def test_parse_kpoint_meshes_scalar_and_triplet():
    assert parse_kpoint_meshes("3, 4x4x1") == [(3, 3, 3), (4, 4, 1)]


def test_parse_kpoint_meshes_rejects_pairs():
    with pytest.raises(ValueError):
        parse_kpoint_meshes("3x3")


def test_set_incar_value_replaces_and_appends(tmp_path):
    incar = tmp_path / "INCAR"
    incar.write_text("ENCUT = 400\nnelm = 40\n", encoding="utf-8")

    set_incar_value(incar, "NELM", 80)
    set_incar_value(incar, "ISMEAR", 0)

    text = incar.read_text()
    assert "NELM = 80" in text
    assert "nelm = 40" not in text
    assert "ENCUT = 400" in text
    assert "ISMEAR = 0" in text


def test_write_gamma_kpoints(tmp_path):
    kpoints = tmp_path / "KPOINTS"
    write_gamma_kpoints(kpoints, (4, 4, 1))
    lines = kpoints.read_text().splitlines()
    assert lines[2] == "Gamma"
    assert lines[3] == "4 4 1"


def _trial(nelm, energy, improvement, converged):
    return {
        "stage": "NELM",
        "nelm": nelm,
        "energy_eV": energy,
        "energy_improvement_eV": improvement,
        "converged": converged,
    }


def test_select_converged_trial_picks_cheapest_within_tolerance():
    rows = [
        _trial(40, -9.0, None, True),
        _trial(60, -10.0, 1.0, True),
        _trial(80, -10.00005, 0.00005, True),
        _trial(100, -10.5, 0.49995, True),
    ]
    selected = _select_converged_trial(rows, energy_tolerance=1e-4)
    assert selected["nelm"] == 80


def test_select_converged_trial_falls_back_to_last_converged():
    rows = [
        _trial(40, -9.0, None, False),
        _trial(60, -10.0, 1.0, True),
    ]
    selected = _select_converged_trial(rows, energy_tolerance=1e-6)
    assert selected["nelm"] == 60


def test_select_converged_trial_handles_no_energies():
    rows = [_trial(40, None, None, False)]
    assert _select_converged_trial(rows, energy_tolerance=1e-4) is None
