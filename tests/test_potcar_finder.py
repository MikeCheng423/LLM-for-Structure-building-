import pytest

from vasp_auto.potcar_finder import (
    build_potcar,
    clean_species_symbol,
    get_elements_from_poscar,
    map_potcar_dirs,
)


def test_clean_species_symbol_strips_vasp_contcar_decoration():
    # VASP 6.x writes "Au/<sha256-fragment>" into CONTCAR when the POTCAR has a
    # SHA256 line; reusing that CONTCAR as a POSCAR must still resolve to "Au".
    assert clean_species_symbol("Au/d0044ae04e2") == "Au"
    assert clean_species_symbol("Au") == "Au"
    assert clean_species_symbol("Fe_pv") == "Fe_pv"  # variant suffix preserved
    assert clean_species_symbol("O_s") == "O_s"


def test_get_elements_from_poscar_sanitizes_decorated_species(tmp_path):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "Au\n1.0\n10 0 0\n0 10 0\n0 0 10\n  Au/d0044ae04e2\n13\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    assert get_elements_from_poscar(poscar) == ["Au"]


def test_get_elements_from_poscar(scf_case):
    assert get_elements_from_poscar(scf_case / "POSCAR") == ["Al", "O"]


def test_get_elements_rejects_vasp4_poscar(tmp_path):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "no symbols\n1.0\n4 0 0\n0 4 0\n0 0 4\n1 1\nDirect\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        get_elements_from_poscar(poscar)


def test_map_potcar_dirs_defaults_to_symbol():
    assert map_potcar_dirs(["Fe", "O"]) == ["Fe", "O"]


def test_map_potcar_dirs_uses_variants():
    assert map_potcar_dirs(["Fe", "O"], {"Fe": "Fe_pv"}) == ["Fe_pv", "O"]


def test_build_potcar_concatenates_in_poscar_order(scf_case, potcar_library, tmp_path):
    output = tmp_path / "POTCAR_OUT"
    build_potcar(str(scf_case / "POSCAR"), str(potcar_library), str(output))
    assert output.read_text() == "POTCAR Al\nPOTCAR O\n"


def test_build_potcar_applies_potcar_map(tmp_path, potcar_library):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "Fe\n1.0\n3 0 0\n0 3 0\n0 0 3\nFe\n1\nDirect\n0 0 0\n", encoding="utf-8"
    )
    output = tmp_path / "POTCAR_OUT"
    build_potcar(str(poscar), str(potcar_library), str(output), potcar_map={"Fe": "Fe_pv"})
    assert output.read_text() == "POTCAR Fe_pv\n"


def test_build_potcar_reports_missing_elements(tmp_path, potcar_library):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "Zr\n1.0\n3 0 0\n0 3 0\n0 0 3\nZr\n1\nDirect\n0 0 0\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        build_potcar(str(poscar), str(potcar_library), str(tmp_path / "POTCAR_OUT"))
