"""Tests for the ASE-backed space-group crystal and nanotube builders
and their CLI / UI-server wiring."""
import sys

import pytest

pytest.importorskip("ase")

from vasp_auto import cli
from vasp_auto.ase_tools import build_crystal_case, build_nanotube_case
from vasp_auto.structure import per_atom_symbols, read_poscar
from vasp_auto_ui import server as ui_server


# ----------------------------------------------------- space-group crystal

def test_build_crystal_rocksalt(tmp_path):
    case = tmp_path / "NaCl"
    poscar = build_crystal_case(
        symbols=["Na", "Cl"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
        spacegroup=225,
        case_dir=case,
        a=5.64,
    )
    struct = read_poscar(poscar)
    symbols = per_atom_symbols(struct)
    # Fm-3m rocksalt: 4 Na + 4 Cl in the conventional cell.
    assert symbols.count("Na") == 4
    assert symbols.count("Cl") == 4
    lattice = struct["lattice"]
    assert lattice[0][0] == pytest.approx(5.64, abs=1e-3)


def test_build_crystal_defaults_cubic_when_bc_omitted(tmp_path):
    poscar = build_crystal_case(
        symbols=["Po"], basis=[(0, 0, 0)], spacegroup=221,
        case_dir=tmp_path / "Po", a=3.35,
    )
    struct = read_poscar(poscar)
    a, b, c = (sum(v * v for v in row) ** 0.5 for row in struct["lattice"])
    assert a == pytest.approx(b) == pytest.approx(c)


def test_build_crystal_rejects_mismatched_basis(tmp_path):
    with pytest.raises(ValueError):
        build_crystal_case(
            symbols=["Na", "Cl"], basis=[(0, 0, 0)], spacegroup=225,
            case_dir=tmp_path / "bad", a=5.64,
        )


def test_build_crystal_accepts_string_symbols(tmp_path):
    poscar = build_crystal_case(
        symbols="Ti O", basis=[(0, 0, 0), (0.305, 0.305, 0)], spacegroup=136,
        case_dir=tmp_path / "TiO2", a=4.59, c=2.96,
    )
    symbols = per_atom_symbols(read_poscar(poscar))
    assert symbols.count("Ti") == 2 and symbols.count("O") == 4


# ----------------------------------------------------------------- nanotube

def test_build_nanotube_armchair(tmp_path):
    poscar = build_nanotube_case("C", 5, 5, tmp_path / "cnt", length=2)
    struct = read_poscar(poscar)
    symbols = per_atom_symbols(struct)
    assert symbols and set(symbols) == {"C"}
    # vacuum in a/b, tube periodic along c → a, b much larger than c
    a, b, c = (sum(v * v for v in row) ** 0.5 for row in struct["lattice"])
    assert a > c and b > c


# --------------------------------------------------------- UI build endpoint

def test_api_build_crystal_to_editor(tmp_path):
    out = ui_server.api_build({}, {
        "action": "crystal", "to_editor": True, "symbols": "Na Cl",
        "spacegroup": 225, "basis": [[0, 0, 0], [0.5, 0.5, 0.5]], "a": 5.64,
    })
    assert out["structure"]["counts"] == {"Cl": 4, "Na": 4}


def test_api_build_nanotube_to_editor(tmp_path):
    out = ui_server.api_build({}, {
        "action": "nanotube", "to_editor": True, "symbol": "C",
        "n": 6, "m": 0, "length": 1,
    })
    assert out["structure"]["natoms"] > 0
    assert set(out["structure"]["symbols"]) == {"C"}


def test_api_build_crystal_writes_case(tmp_path):
    out = ui_server.api_build({}, {
        "action": "crystal", "symbols": ["Po"], "spacegroup": 221,
        "basis": [[0, 0, 0]], "a": 3.35, "output": str(tmp_path / "Po"),
    })
    assert read_poscar(out["poscar"])


# ------------------------------------------------------------- CLI wiring

def test_cli_build_crystal(tmp_path, monkeypatch):
    out = tmp_path / "NaCl"
    monkeypatch.setattr(sys, "argv", [
        "vasp-auto", "--ase-build-crystal", "Na Cl", "--ase-spacegroup", "225",
        "--ase-basis", "0,0,0;0.5,0.5,0.5", "--ase-a", "5.64",
        "--ase-output", str(out), "--build-only",
    ])
    cli.main()
    symbols = per_atom_symbols(read_poscar(out / "POSCAR"))
    assert symbols.count("Na") == 4 and symbols.count("Cl") == 4


def test_cli_build_nanotube(tmp_path, monkeypatch):
    out = tmp_path / "cnt"
    monkeypatch.setattr(sys, "argv", [
        "vasp-auto", "--ase-build-nanotube", "C", "--ase-nt-n", "5",
        "--ase-nt-m", "5", "--ase-nt-length", "2",
        "--ase-output", str(out), "--build-only",
    ])
    cli.main()
    assert set(per_atom_symbols(read_poscar(out / "POSCAR"))) == {"C"}


def test_cli_build_crystal_requires_spacegroup(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "vasp-auto", "--ase-build-crystal", "Na Cl",
        "--ase-output", str(tmp_path / "x"), "--build-only",
    ])
    with pytest.raises(ValueError):
        cli.main()
