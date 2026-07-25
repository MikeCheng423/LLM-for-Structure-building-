"""Tests for the py4vasp analysis backend and the OSZICAR energy fallback."""
from pathlib import Path

from vasp_auto import py4vasp_tools
from vasp_auto.trajectory import oszicar_energies

OSZICAR = """\
       N       E                     dE             d eps       ncg     rms          rms(c)
DAV:   1     0.364839310725E+02    0.36484E+02   -0.18736E+03    32   0.421E+02
DAV:   2    -0.263242617545E+02   -0.62808E+02   -0.60767E+02    40   0.113E+02
   1 F= -.27864425E+02 E0= -.27864425E+02  d E =-.278644E+02
DAV:   1    -0.279105467973E+02   -0.46104E-01   -0.36041E+00    32   0.905E+00
   2 F= -.27910547E+02 E0= -.27910547E+02  d E =-.461041E-01
"""


def test_oszicar_energies(tmp_path):
    (tmp_path / "OSZICAR").write_text(OSZICAR, encoding="utf-8")
    assert oszicar_energies(tmp_path) == [-27.864425, -27.910547]


def test_oszicar_energies_missing_or_empty(tmp_path):
    assert oszicar_energies(tmp_path) is None
    (tmp_path / "OSZICAR").write_text("garbage\n", encoding="utf-8")
    assert oszicar_energies(tmp_path) is None


def test_py4vasp_returns_none_without_vaspout(tmp_path):
    # No vaspout.h5 -> every accessor reports "not available" instead of raising.
    assert py4vasp_tools.dos(tmp_path) is None
    assert py4vasp_tools.step_energies(tmp_path) is None


def test_py4vasp_survives_corrupt_vaspout(tmp_path):
    (tmp_path / "vaspout.h5").write_text("not an hdf5 file", encoding="utf-8")
    assert py4vasp_tools.dos(tmp_path) is None
    assert py4vasp_tools.step_energies(tmp_path) is None
