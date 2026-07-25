import math

import pytest

from vasp_auto.kpoints import (
    auto_kpath,
    guess_lattice_type,
    kpoints_text_from_spec,
    line_mode_text,
    mesh_from_spacing,
    mesh_kpoints_text,
    parse_kpath,
    parse_mesh,
)


def test_parse_mesh_scalar_and_triplet():
    assert parse_mesh("4") == (4, 4, 4)
    assert parse_mesh("4x4x1") == (4, 4, 1)
    assert parse_mesh("4 2 1") == (4, 2, 1)


def test_parse_mesh_rejects_pairs():
    with pytest.raises(ValueError):
        parse_mesh("4x4")


def test_mesh_kpoints_text_modes():
    gamma = mesh_kpoints_text((4, 4, 1), mode="gamma")
    mp = mesh_kpoints_text((4, 4, 1), mode="mp")
    assert "Gamma" in gamma and "4 4 1" in gamma
    assert "Monkhorst-Pack" in mp


def test_mesh_from_spacing_cubic(scf_case):
    # 4 Å cube: |b| = 2π/4 ≈ 1.5708 1/Å, so 0.5 1/Å spacing needs 4 divisions.
    assert mesh_from_spacing(scf_case / "POSCAR", 0.5) == (4, 4, 4)
    assert mesh_from_spacing(scf_case / "POSCAR", 2.0) == (1, 1, 1)


def test_parse_kpath_preset():
    points = parse_kpath("fcc")
    assert points[0][0] == "G"
    assert len(points) >= 4


def test_parse_kpath_custom_and_invalid():
    points = parse_kpath("G 0 0 0; X 0.5 0 0.5")
    assert points == [("G", (0.0, 0.0, 0.0)), ("X", (0.5, 0.0, 0.5))]
    with pytest.raises(ValueError):
        parse_kpath("nonsense")


def test_line_mode_text_pairs_segments():
    text = line_mode_text([("G", (0, 0, 0)), ("X", (0.5, 0, 0.5))], divisions=15)
    assert "Line-mode" in text
    assert "15" in text.splitlines()[1]
    assert "! G" in text and "! X" in text


def test_kpoints_text_from_spec_spacing(scf_case):
    text = kpoints_text_from_spec({"mode": "gamma", "spacing": 0.5}, poscar_path=scf_case / "POSCAR")
    assert "4 4 4" in text


def test_kpoints_text_from_spec_line_requires_kpath():
    with pytest.raises(ValueError):
        kpoints_text_from_spec({"mode": "line"})


# ---------------------------------------------------------------- TASK 5: auto k-path

def _cubic_lattice():
    """Simple cubic: a=b=c=4, 90/90/90."""
    return [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]


def _fcc_primitive_lattice():
    """FCC primitive: a=b=c, angles 60°."""
    a = 4.05 / math.sqrt(2)
    return [
        [0.0, a, a],
        [a, 0.0, a],
        [a, a, 0.0],
    ]


def _bcc_primitive_lattice():
    """BCC primitive: a=b=c, angles ≈109.47°."""
    a = 3.3 / 2.0
    return [
        [-a,  a,  a],
        [ a, -a,  a],
        [ a,  a, -a],
    ]


def _hex_lattice():
    """Hexagonal: a=b=2.46, c=6.7, α=β=90°, γ=120°.

    For γ=120° between a1 and a2:
        a1 = [a, 0, 0]
        a2 = [a*cos(120°), a*sin(120°), 0] = [-a/2, a*sqrt(3)/2, 0]
    """
    a, c = 2.46, 6.70
    return [
        [a, 0.0, 0.0],
        [a * math.cos(math.radians(120)), a * math.sin(math.radians(120)), 0.0],
        [0.0, 0.0, c],
    ]


def _generic_lattice():
    """Triclinic — no high-symmetry angles."""
    return [
        [3.0, 0.5, 0.2],
        [0.3, 4.1, 0.8],
        [0.1, 0.4, 5.5],
    ]


def test_guess_lattice_type_cubic():
    assert guess_lattice_type(_cubic_lattice()) == "cubic"


def test_guess_lattice_type_fcc():
    assert guess_lattice_type(_fcc_primitive_lattice()) == "fcc"


def test_guess_lattice_type_bcc():
    assert guess_lattice_type(_bcc_primitive_lattice()) == "bcc"


def test_guess_lattice_type_hex():
    assert guess_lattice_type(_hex_lattice()) == "hex"


def test_guess_lattice_type_generic():
    assert guess_lattice_type(_generic_lattice()) == "generic"


def test_auto_kpath_cubic(tmp_path):
    """POSCAR with a cubic cell → auto_kpath returns the cubic preset."""
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "Cubic test\n1.0\n4.0 0.0 0.0\n0.0 4.0 0.0\n0.0 0.0 4.0\n"
        "Al\n1\nDirect\n0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    path = auto_kpath(poscar)
    assert path[0][0] == "G"
    labels = [p[0] for p in path]
    assert "X" in labels


def test_auto_kpath_generic_raises(tmp_path):
    """Generic triclinic cell → auto_kpath raises ValueError."""
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "Triclinic\n1.0\n3.0 0.5 0.2\n0.3 4.1 0.8\n0.1 0.4 5.5\n"
        "Al\n1\nDirect\n0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generic"):
        auto_kpath(poscar)


def test_kpoints_text_from_spec_auto_kpath(tmp_path):
    """kpoints_text_from_spec with kpath='auto' triggers auto-detection."""
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "Cubic test\n1.0\n4.0 0.0 0.0\n0.0 4.0 0.0\n0.0 0.0 4.0\n"
        "Al\n1\nDirect\n0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    text = kpoints_text_from_spec({"mode": "line", "kpath": "auto"}, poscar_path=poscar)
    assert "Line-mode" in text
    assert "! G" in text
