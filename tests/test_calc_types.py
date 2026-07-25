import pytest

from vasp_auto.calc_types import CALC_TYPE_INFO, CHAIN_INPUTS, CalcType, parse_calc_type


def test_parse_calc_type_normalises():
    assert parse_calc_type(" SCF ") == CalcType.SCF
    assert parse_calc_type("relax") == CalcType.RELAX


def test_parse_calc_type_rejects_unknown():
    with pytest.raises(ValueError):
        parse_calc_type("gw")


def test_every_calc_type_has_chain_inputs():
    assert set(CHAIN_INPUTS) == set(CalcType)


def test_dos_chain_inputs_include_chgcar():
    assert CHAIN_INPUTS[CalcType.DOS]["CHGCAR"] == "CHGCAR"


# ---------------------------------------------------------------- TASK 2: phonon description

def test_phonon_description_says_dfpt():
    desc = CALC_TYPE_INFO[CalcType.PHONON]
    assert "finite displacements" not in desc.lower()
    # Should mention DFPT or IBRION=8
    assert "DFPT" in desc or "IBRION=8" in desc or "IBRION" in desc


def test_phonon_description_not_empty():
    assert len(CALC_TYPE_INFO[CalcType.PHONON]) > 5
