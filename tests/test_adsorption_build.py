"""Tests for the adsorption-oriented Build upgrade: cell resize, move/add
atom, selective-dynamics freezing, and the one-shot adsorbate placement."""
import pytest

from vasp_auto.cli import (
    _parse_adsorbate_spec,
    _parse_cell_spec,
    _parse_freeze_spec,
    _parse_move_spec,
)
from vasp_auto.structure import (
    add_adsorbate,
    cell_lengths,
    freeze_atoms,
    move_atom,
    parse_atom_selection,
    read_poscar,
    scale_cell,
    write_poscar,
)
from vasp_auto_ui import server as ui_server


@pytest.fixture
def slab():
    return {
        "comment": "Pt slab",
        "scale": 1.0,
        "lattice": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 20.0]],
        "elements": ["Pt"],
        "counts": [4],
        "selective": False,
        "cartesian": False,
        "coords": [[0, 0, 0.10], [0.5, 0.5, 0.15], [0, 0, 0.20], [0.5, 0.5, 0.25]],
        "flags": [[], [], [], []],
    }


# ------------------------------------------------------------------ cell size

def test_scale_cell_uniform_keeps_fractional(slab):
    out = scale_cell(slab, (1.05, 1.05, 1.0))
    assert out["lattice"][0][0] == pytest.approx(4.2)
    assert out["lattice"][2][2] == pytest.approx(20.0)
    assert out["coords"] == slab["coords"]  # atoms follow fractionally


def test_scale_cell_cartesian_mode_remaps_coordinates(slab):
    cart = {**slab, "cartesian": True, "coords": [[2.0, 2.0, 3.0]], "counts": [1],
            "flags": [[]]}
    out = scale_cell(cart, (2.0, 1.0, 1.0))
    assert out["coords"][0] == pytest.approx([4.0, 2.0, 3.0])


def test_parse_cell_spec_absolute_lengths(slab):
    factors = _parse_cell_spec(slab, "a=4.2,c=25")
    assert factors[0] == pytest.approx(1.05)
    assert factors[1] == pytest.approx(1.0)
    assert factors[2] == pytest.approx(1.25)
    assert cell_lengths(scale_cell(slab, factors))[2] == pytest.approx(25.0)


def test_parse_cell_spec_single_factor(slab):
    assert _parse_cell_spec(slab, "1.02") == (1.02, 1.02, 1.02)


# ------------------------------------------------------------------ move/add

def test_move_atom_translate_and_place(slab):
    shifted = move_atom(slab, 1, (0.0, 0.0, 0.05))
    assert shifted["coords"][0][2] == pytest.approx(0.15)
    placed = move_atom(slab, 1, (0.25, 0.25, 0.5), absolute=True)
    assert placed["coords"][0] == [0.25, 0.25, 0.5]
    assert slab["coords"][0][2] == 0.10  # input untouched


def test_parse_move_spec():
    assert _parse_move_spec("5@0.5,0.5,0.6") == (5, (0.5, 0.5, 0.6), True)
    assert _parse_move_spec("5+0,0,-0.1") == (5, (0.0, 0.0, -0.1), False)
    with pytest.raises(ValueError):
        _parse_move_spec("5:1,2,3")


def test_add_adsorbate_above_anchor(slab):
    out = add_adsorbate(slab, "O", 4, 2.0)
    assert out["elements"] == ["Pt", "O"]
    assert out["counts"] == [4, 1]
    # anchor at frac z=0.25 (5 Å) + 2 Å → frac z = 7/20 = 0.35, same x/y
    assert out["coords"][-1] == pytest.approx([0.5, 0.5, 0.35])


def test_add_adsorbate_molecule_places_all_atoms(slab):
    pytest.importorskip("ase")
    out = add_adsorbate(slab, "CO2", 4, 2.0)
    assert out["elements"] == ["Pt", "C", "O"]
    assert out["counts"] == [4, 1, 2]  # a whole CO2, not one "CO2" atom
    # lowest molecule atom sits at the anchor height: frac z = 7/20 = 0.35
    added = out["coords"][-3:]
    assert min(c[2] for c in added) == pytest.approx(0.35)


def test_parse_adsorbate_spec_defaults_height():
    assert _parse_adsorbate_spec("O@12+2.5") == ("O", 12, 2.5)
    assert _parse_adsorbate_spec("H@3") == ("H", 3, 2.0)


# ------------------------------------------------------------------- freezing

def test_parse_atom_selection_forms(slab):
    assert parse_atom_selection(slab, "1-2,4") == [1, 2, 4]
    assert parse_atom_selection(slab, "z<0.18") == [1, 2]
    assert parse_atom_selection(slab, "z>0.18") == [3, 4]
    with pytest.raises(ValueError):
        parse_atom_selection(slab, "9")


def test_freeze_atoms_sets_selective_dynamics(slab):
    out = freeze_atoms(slab, [1, 2], axes="XYZ")
    assert out["selective"] is True
    assert out["flags"][0] == ["F", "F", "F"]
    assert out["flags"][2] == ["T", "T", "T"]


def test_freeze_axes_compose(slab):
    out = freeze_atoms(freeze_atoms(slab, [1], axes="XYZ"), [2], axes="XY")
    assert out["flags"][0] == ["F", "F", "F"]
    assert out["flags"][1] == ["F", "F", "T"]  # z stays free


def test_freeze_survives_poscar_roundtrip(slab, tmp_path):
    out = freeze_atoms(slab, parse_atom_selection(slab, "z<0.18"))
    write_poscar(out, tmp_path / "POSCAR")
    text = (tmp_path / "POSCAR").read_text(encoding="utf-8")
    assert "Selective dynamics" in text
    back = read_poscar(tmp_path / "POSCAR")
    assert back["selective"] is True
    assert back["flags"][0] == ["F", "F", "F"]
    assert back["flags"][3] == ["T", "T", "T"]


def test_parse_freeze_spec():
    assert _parse_freeze_spec("z<0.3") == ("z<0.3", "XYZ")
    assert _parse_freeze_spec("z<0.3:XY") == ("z<0.3", "XY")
    assert _parse_freeze_spec("1-8:Z") == ("1-8", "Z")


# ------------------------------------------------------- UI edit action (API)

def test_api_build_adsorption_case(slab, tmp_path):
    case = tmp_path / "Pt_slab"
    case.mkdir()
    write_poscar(slab, case / "POSCAR")

    result = ui_server.api_build(None, {
        "action": "edit",
        "source": str(case),
        "adsorbate": ["O", 4, 2.0],
        "freeze": ["z<0.18", "XYZ"],
    })
    out = read_poscar(case.parent / "Pt_slab_adsO4_frz" / "POSCAR")
    assert out["elements"] == ["Pt", "O"]
    assert out["selective"] is True
    assert out["flags"][0] == ["F", "F", "F"]   # bottom layer frozen
    assert out["flags"][-1] == ["T", "T", "T"]  # adsorbate free
    assert result["case"].endswith("_adsO4_frz")


def test_api_build_move_and_cell(slab, tmp_path):
    case = tmp_path / "Pt_slab"
    case.mkdir()
    write_poscar(slab, case / "POSCAR")

    ui_server.api_build(None, {
        "action": "edit",
        "source": str(case),
        "move_atom": [1, [0.0, 0.0, 0.05], False],
        "scale_cell": "a=4.2",
    })
    out = read_poscar(case.parent / "Pt_slab_mv1_cell" / "POSCAR")
    assert out["coords"][0][2] == pytest.approx(0.15)
    assert out["lattice"][0][0] == pytest.approx(4.2)
