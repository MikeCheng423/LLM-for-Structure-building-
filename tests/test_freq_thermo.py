"""Frequency thermochemistry parser + harmonic corrections."""
from vasp_auto.freq_thermo import CM2EV, parse_frequencies, thermo

# Two real modes + one imaginary (f/i), VASP OUTCAR column layout.
_OUTCAR = """\
   1 f  =   48.960837 THz   307.6 2PiTHz 1633.157728 cm-1   202.485660 meV
   2 f  =   24.000000 THz   150.0 2PiTHz  800.000000 cm-1   100.000000 meV
 184 f/i=    0.657420 THz     4.1 2PiTHz    9.300000 cm-1     1.153200 meV
"""


def test_parse_splits_real_and_imaginary(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(_OUTCAR, encoding="utf-8")
    real, imag = parse_frequencies(outcar)
    assert len(real) == 2 and len(imag) == 1
    assert abs(real[0] - 0.20248566) < 1e-6   # meV -> eV
    assert abs(imag[0] / CM2EV - 9.3) < 0.1    # round-trips back to ~9.3 cm^-1


def test_thermo_gibbs_correction():
    # High-frequency modes: ZPE dominates, thermal/entropy terms ~0 at 298 K.
    real = [0.20248566, 0.10]
    t = thermo(real)
    assert abs(t["ZPE"] - sum(real) / 2.0) < 1e-9
    assert t["Hcorr"] < 1e-2 and t["TS"] < 1e-2
    assert abs(t["G_corr"] - (t["ZPE"] + t["Hcorr"] - t["TS"])) < 1e-12
