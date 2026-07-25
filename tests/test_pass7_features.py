"""Tests for the pass-7 features: prototype crystals, supercell matching,
adsorbate placement, PDOS aggregation, k-point labels and band parsing."""
import sys

import pytest

from vasp_auto import cli
from vasp_auto.parser import aggregate_pdos, parse_bands, read_kpoints_labels
from vasp_auto.structure import (
    add_adsorbate,
    build_struct,
    cell_parameters,
    frac_coords,
    make_prototype,
    match_supercells,
    per_atom_symbols,
    read_poscar,
    resolve_prototype,
    scaled_lattice,
)


# ------------------------------------------------------------- prototypes

def test_resolve_prototype_aliases():
    assert resolve_prototype("graphene") == "graphene"
    assert resolve_prototype("RUTILE") == "rutile-TiO2"
    assert resolve_prototype("anatase_tio2") == "anatase-TiO2"
    assert resolve_prototype("bn") == "hBN"


def test_resolve_prototype_unknown():
    with pytest.raises(ValueError, match="Available"):
        resolve_prototype("diamondium")


def test_make_prototype_graphene():
    struct = make_prototype("graphene")
    assert per_atom_symbols(struct) == ["C", "C"]
    params = cell_parameters(scaled_lattice(struct))
    assert params["a"] == pytest.approx(2.468)
    assert params["gamma"] == pytest.approx(120.0)
    assert params["c"] == pytest.approx(15.0)  # default vacuum box


def test_make_prototype_sheet_vacuum_override():
    struct = make_prototype("graphene", a=2.5, vacuum=18.0)
    params = cell_parameters(scaled_lattice(struct))
    assert params["a"] == pytest.approx(2.5)
    assert params["c"] == pytest.approx(18.0)


def test_make_prototype_rutile():
    struct = make_prototype("rutile")
    assert struct["elements"] == ["Ti", "O"]
    assert struct["counts"] == [2, 4]
    params = cell_parameters(scaled_lattice(struct))
    assert params["a"] == pytest.approx(4.593)
    assert params["c"] == pytest.approx(2.959)
    assert params["gamma"] == pytest.approx(90.0)


def test_make_prototype_rejects_bad_constants():
    with pytest.raises(ValueError, match="positive"):
        make_prototype("graphene", a=-1.0)


def test_cli_build_prototype(tmp_path, monkeypatch):
    out = tmp_path / "gr"
    monkeypatch.setattr(
        sys, "argv",
        ["vasp-auto", "--build-prototype", "graphene:a=2.46,vacuum=18",
         "--ase-output", str(out), "--build-only"],
    )
    cli.main()
    struct = read_poscar(out / "POSCAR")
    params = cell_parameters(scaled_lattice(struct))
    assert params["a"] == pytest.approx(2.46)
    assert params["c"] == pytest.approx(18.0)


# ------------------------------------------------------- supercell matching

def _sheet(a: float, gamma_deg: float = 90.0, element: str = "X") -> dict:
    import math
    ga = math.radians(gamma_deg)
    lattice = [
        [a, 0.0, 0.0],
        [a * math.cos(ga), a * math.sin(ga), 0.0],
        [0.0, 0.0, 15.0],
    ]
    return build_struct("sheet", lattice, [element], [[0.0, 0.0, 0.5]])


def test_match_supercells_exact():
    host = _sheet(2.0)
    guest = _sheet(3.0)
    matches = match_supercells(host, guest, max_repeat=6)
    assert matches, "3a_host = 2a_guest should match exactly"
    best = matches[0]
    assert best["host_repeat"] == (3, 3)
    assert best["guest_repeat"] == (2, 2)
    assert best["strain_pct"] == pytest.approx(0.0, abs=1e-9)
    assert best["host_atoms"] == 9
    assert best["guest_atoms"] == 4


def test_match_supercells_respects_strain_limit():
    host = _sheet(2.0)
    guest = _sheet(2.3)  # 15% off; only multi-cell combinations get close
    tight = match_supercells(host, guest, max_repeat=1, max_strain=0.05)
    assert tight == []
    loose = match_supercells(host, guest, max_repeat=1, max_strain=0.2)
    assert loose and loose[0]["strain_pct"] == pytest.approx(15.0)


def test_match_supercells_gamma_mismatch_rejected():
    host = _sheet(2.0, gamma_deg=90.0)
    guest = _sheet(2.0, gamma_deg=120.0)
    assert match_supercells(host, guest, gamma_tol=8.0) == []


def test_match_supercells_hexagonal_supplementary_angle():
    # 60 and 120 deg describe the same hexagonal sheet (b-vector convention).
    host = _sheet(2.0, gamma_deg=60.0)
    guest = _sheet(2.0, gamma_deg=120.0)
    matches = match_supercells(host, guest, gamma_tol=8.0)
    assert matches and matches[0]["gamma_mismatch_deg"] == pytest.approx(0.0)


def test_match_supercells_sorted_and_capped():
    host = _sheet(2.0)
    guest = _sheet(2.0)
    matches = match_supercells(host, guest, max_repeat=4, max_results=5)
    assert len(matches) <= 5
    assert matches[0]["host_repeat"] == (1, 1)  # smallest cell wins the tie
    strains = [m["strain_pct"] for m in matches]
    assert strains == sorted(strains)


def test_cli_match_cells(tmp_path, monkeypatch, capsys):
    from vasp_auto.structure import write_poscar

    host_dir = tmp_path / "host"
    guest_dir = tmp_path / "guest"
    write_poscar(_sheet(2.0, element="C"), host_dir / "POSCAR")
    write_poscar(_sheet(3.0, element="Ti"), guest_dir / "POSCAR")
    monkeypatch.setattr(
        sys, "argv",
        ["vasp-auto", str(host_dir), "--match-cells", str(guest_dir), "--build-only"],
    )
    cli.main()
    out = capsys.readouterr().out
    assert "3x3" in out and "2x2" in out
    assert "--combine" in out  # prints the follow-up commands


# ------------------------------------------------------------- adsorbates

def test_add_adsorbate_direct_mode(scf_case):
    struct = read_poscar(scf_case / "POSCAR")
    result = add_adsorbate(struct, "H", anchor_index=2, height=1.5)
    assert per_atom_symbols(result) == ["Al", "O", "H"]
    # O sits at cart (2,2,2) in the 4 Å cube; H goes 1.5 Å above it.
    assert frac_coords(result)[2] == pytest.approx([0.5, 0.5, 3.5 / 4.0])


def test_add_adsorbate_bad_anchor(scf_case):
    struct = read_poscar(scf_case / "POSCAR")
    with pytest.raises(ValueError, match="out of range"):
        add_adsorbate(struct, "H", anchor_index=9, height=2.0)


# --------------------------------------------------------- PDOS aggregation

PDOS_RESULT = {
    "efermi": 2.0,
    "energies": [0.0, 1.0],
    "fields": ["s", "dxy", "x2-y2"],
    "pdos": {
        1: [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]],   # Ti: s + two d channels
        2: [[[10.0, 10.0], [0.0, 0.0], [0.0, 0.0]]],  # O: s only
    },
}


def test_aggregate_pdos_sums_shells():
    result = aggregate_pdos(PDOS_RESULT, ["Ti", "O"])
    by_label = {curve["label"]: curve for curve in result["curves"]}
    assert by_label["Ti d"]["values"] == pytest.approx([5.0, 5.0])  # dxy + x2-y2
    assert by_label["Ti s"]["values"] == pytest.approx([1.0, 1.0])
    assert by_label["O s"]["values"] == pytest.approx([10.0, 10.0])
    assert result["efermi"] == pytest.approx(2.0)


def test_aggregate_pdos_atom_selection():
    result = aggregate_pdos(PDOS_RESULT, ["Ti", "O"], atoms=[1])
    labels = [curve["label"] for curve in result["curves"]]
    assert "O s" not in labels
    assert "Ti d" in labels


# --------------------------------------------------------- k-point labels

KPOINTS_LINE = """k-path
3
Line-mode
Reciprocal
0.0 0.0 0.0 ! G
0.5 0.0 0.0 ! X

0.5 0.0 0.0 ! X
0.5 0.5 0.0 ! M
"""


def test_read_kpoints_labels(tmp_path):
    path = tmp_path / "KPOINTS"
    path.write_text(KPOINTS_LINE, encoding="utf-8")
    labels = read_kpoints_labels(path)
    assert labels == [
        {"index": 0, "label": "G"},
        {"index": 2, "label": "X"},   # shared joint, same label: no duplicate
        {"index": 5, "label": "M"},
    ]


def test_read_kpoints_labels_discontinuous_path(tmp_path):
    text = KPOINTS_LINE.replace("0.5 0.0 0.0 ! X\n0.5 0.5 0.0 ! M", "0.5 0.5 0.5 ! K\n0.5 0.5 0.0 ! M")
    path = tmp_path / "KPOINTS"
    path.write_text(text, encoding="utf-8")
    labels = read_kpoints_labels(path)
    assert labels[1] == {"index": 2, "label": "X|K"}


def test_read_kpoints_labels_not_line_mode(tmp_path):
    path = tmp_path / "KPOINTS"
    path.write_text("mesh\n0\nGamma\n4 4 4\n", encoding="utf-8")
    assert read_kpoints_labels(path) is None


# ------------------------------------------------------------- band parsing

VASPRUN_BANDS = """<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
 <kpoints>
  <varray name="kpointlist">
   <v> 0.00 0.0 0.0 </v>
   <v> 0.25 0.0 0.0 </v>
   <v> 0.50 0.0 0.0 </v>
  </varray>
 </kpoints>
 <calculation>
  <eigenvalues><array><set>
   <set comment="spin 1">
    <set comment="kpoint 1"><r> 1.0 1.0 </r><r> 3.0 0.0 </r></set>
    <set comment="kpoint 2"><r> 1.5 1.0 </r><r> 3.5 0.0 </r></set>
    <set comment="kpoint 3"><r> 2.0 1.0 </r><r> 4.0 0.0 </r></set>
   </set>
  </set></array></eigenvalues>
  <dos><i name="efermi"> 2.0 </i></dos>
 </calculation>
</modeling>
"""


def test_parse_bands(tmp_path):
    vasprun = tmp_path / "vasprun.xml"
    vasprun.write_text(VASPRUN_BANDS, encoding="utf-8")
    result = parse_bands(vasprun)

    assert result["efermi"] == pytest.approx(2.0)
    assert len(result["bands"]) == 1          # one spin channel
    assert len(result["bands"][0]) == 2       # two bands
    assert result["bands"][0][0] == pytest.approx([1.0, 1.5, 2.0])
    assert result["bands"][0][1] == pytest.approx([3.0, 3.5, 4.0])
    # No rec_basis in the file -> identity basis, steps of 0.25.
    assert result["distances"] == pytest.approx([0.0, 0.25, 0.5])
    assert result["labels"] == []


def test_parse_bands_with_mismatched_kpoints_file(tmp_path):
    vasprun = tmp_path / "vasprun.xml"
    vasprun.write_text(VASPRUN_BANDS, encoding="utf-8")
    kpoints = tmp_path / "KPOINTS"
    kpoints.write_text(KPOINTS_LINE, encoding="utf-8")  # labels run to index 5
    result = parse_bands(vasprun, kpoints)
    assert result["labels"] == []  # 3-kpoint run cannot carry 6-point labels


def test_parse_bands_missing_eigenvalues(tmp_path):
    vasprun = tmp_path / "vasprun.xml"
    vasprun.write_text(
        "<modeling><calculation><energy><i name='e_fr_energy'>1</i></energy>"
        "</calculation></modeling>",
        encoding="utf-8",
    )
    assert parse_bands(vasprun) is None
