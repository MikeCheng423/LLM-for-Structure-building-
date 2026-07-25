"""Tests for chgcar.py: volumetric IO, difference/sum, planar average, Bader."""
import pytest

from vasp_auto import chgcar

HEADER = """test cell
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 8.0
Al O
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""


def _write_chgcar(path, values, grid=(2, 1, 2)):
    nx, ny, nz = grid
    lines = HEADER + f"\n  {nx}  {ny}  {nz}\n"
    lines += " ".join(f"{v:.6E}" for v in values) + "\n"
    path.write_text(lines, encoding="utf-8")


def test_read_write_roundtrip(tmp_path):
    path = tmp_path / "CHGCAR"
    _write_chgcar(path, [1.0, 2.0, 3.0, 4.0])

    volume = chgcar.read_volumetric(path)
    assert volume["grid"] == (2, 1, 2)
    assert volume["data"] == [1.0, 2.0, 3.0, 4.0]

    out = tmp_path / "copy"
    chgcar.write_volumetric(volume, out)
    again = chgcar.read_volumetric(out)
    assert again["grid"] == volume["grid"]
    assert again["data"] == pytest.approx(volume["data"])


def test_read_truncated_grid_raises(tmp_path):
    path = tmp_path / "CHGCAR"
    path.write_text(HEADER + "\n  2  2  2\n 1.0 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="truncated"):
        chgcar.read_volumetric(path)


def test_charge_difference(tmp_path):
    total = tmp_path / "AB"
    part_a = tmp_path / "A"
    part_b = tmp_path / "B"
    _write_chgcar(total, [10.0, 10.0, 10.0, 10.0])
    _write_chgcar(part_a, [4.0, 4.0, 4.0, 4.0])
    _write_chgcar(part_b, [1.0, 2.0, 3.0, 4.0])

    output = tmp_path / "CHGCAR_diff"
    diff = chgcar.charge_difference(total, [part_a, part_b], output)
    assert diff["data"] == pytest.approx([5.0, 4.0, 3.0, 2.0])
    assert chgcar.read_volumetric(output)["data"] == pytest.approx([5.0, 4.0, 3.0, 2.0])


def test_charge_difference_grid_mismatch(tmp_path):
    total = tmp_path / "AB"
    part = tmp_path / "A"
    _write_chgcar(total, [1.0, 1.0, 1.0, 1.0], grid=(2, 1, 2))
    _write_chgcar(part, [1.0, 1.0], grid=(2, 1, 1))
    with pytest.raises(ValueError, match="grids differ"):
        chgcar.charge_difference(total, [part], tmp_path / "out")


def test_charge_sum(tmp_path):
    a = tmp_path / "AECCAR0"
    b = tmp_path / "AECCAR2"
    _write_chgcar(a, [1.0, 2.0, 3.0, 4.0])
    _write_chgcar(b, [10.0, 20.0, 30.0, 40.0])
    total = chgcar.charge_sum([a, b], tmp_path / "CHGCAR_sum")
    assert total["data"] == pytest.approx([11.0, 22.0, 33.0, 44.0])


def test_planar_average_along_c(tmp_path):
    path = tmp_path / "LOCPOT"
    # grid 2x1x2, x fastest: planes z=0 -> (1, 2), z=1 -> (3, 4)
    _write_chgcar(path, [1.0, 2.0, 3.0, 4.0])
    volume = chgcar.read_volumetric(path)
    assert chgcar.planar_average(volume, axis=2) == pytest.approx([1.5, 3.5])
    assert chgcar.planar_average(volume, axis=0) == pytest.approx([2.0, 3.0])


def test_zval_from_potcar(tmp_path):
    potcar = tmp_path / "POTCAR"
    potcar.write_text(
        "  PAW_PBE Al 04Jan2001\n   POMASS =   26.982; ZVAL   =    3.000    mass and valenz\n"
        "  PAW_PBE O 08Apr2002\n   POMASS =   16.000; ZVAL   =    6.000    mass and valenz\n",
        encoding="utf-8",
    )
    assert chgcar.zval_from_potcar(potcar) == [3.0, 6.0]


ACF_TEXT = """    #         X           Y           Z       CHARGE      MIN DIST   ATOMIC VOL
 --------------------------------------------------------------------------------
    1      0.0000      0.0000      0.0000     2.5000        0.3691     11.4221
    2      2.0000      2.0000      4.0000     6.5000        0.4000     12.0000
 --------------------------------------------------------------------------------
    VACUUM CHARGE:               0.0000
    NUMBER OF ELECTRONS:         9.0000
"""


def test_parse_acf(tmp_path):
    acf = tmp_path / "ACF.dat"
    acf.write_text(ACF_TEXT, encoding="utf-8")
    rows = chgcar.parse_acf(acf)
    assert len(rows) == 2
    assert rows[0]["electrons"] == pytest.approx(2.5)


def test_bader_net_charges(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "ACF.dat").write_text(ACF_TEXT, encoding="utf-8")
    (job / "POSCAR").write_text(HEADER, encoding="utf-8")
    (job / "POTCAR").write_text(
        "ZVAL   =    3.000\nZVAL   =    6.000\n", encoding="utf-8"
    )
    charges = chgcar.bader_net_charges(job)
    assert charges[0]["element"] == "Al"
    assert charges[0]["net_charge"] == pytest.approx(0.5)   # 3.0 - 2.5
    assert charges[1]["net_charge"] == pytest.approx(-0.5)  # 6.0 - 6.5


def test_run_bader_requires_binary(tmp_path, monkeypatch):
    job = tmp_path / "job"
    job.mkdir()
    (job / "CHGCAR").write_text("x", encoding="utf-8")
    monkeypatch.setattr(chgcar.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="PATH"):
        chgcar.run_bader(job)
