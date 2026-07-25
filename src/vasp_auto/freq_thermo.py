"""Harmonic thermochemistry from a VASP frequency run (IBRION = 5/6/7/8).

Parses the vibrational modes VASP prints in OUTCAR ('f=' real, 'f/i=' imaginary)
and returns the zero-point energy, thermal (enthalpy) correction and the -TS
vibrational entropy term at 298.15 K, so a frequency job's Gibbs correction
(ZPE + Hcorr - TS) can be added to E_DFT to get G. Imaginary modes are dropped;
low real modes are raised to CUT_CM (quasi-harmonic) to tame the 1/omega entropy
divergence. Consumed by job_log to print a THERMOCHEMISTRY section.

Runnable standalone (same interface as the original script):
    python -m vasp_auto.freq_thermo OUTCAR
"""
from __future__ import annotations

import math
from pathlib import Path

T = 298.15                # K
KB = 8.617333e-5          # eV/K
KT = KB * T
CM2EV = 1.239841984e-4    # 1 cm^-1 -> eV
CUT_CM = 50.0             # ponytail: raise real modes below this (cm^-1) to the
                          # cutoff; set 0 to disable. The quasi-harmonic knob.


def parse_frequencies(outcar) -> tuple[list[float], list[float]]:
    """(real, imaginary) mode energies in eV from OUTCAR's 'f=' / 'f/i=' lines."""
    real: list[float] = []
    imag: list[float] = []
    for ln in Path(outcar).read_text(encoding="utf-8", errors="ignore").splitlines():
        if "meV" not in ln:
            continue
        try:
            mev = float(ln.split()[-2])  # second-to-last field is the meV value
        except (ValueError, IndexError):
            continue
        (imag if "f/i" in ln else real).append(mev / 1000.0)
    return real, imag


def thermo(vib_eV) -> dict:
    """ZPE, Hcorr, TS and Gibbs correction (all eV) for the given real modes."""
    cut = CUT_CM * CM2EV
    eps = [max(e, cut) for e in vib_eV]
    zpe = sum(e / 2.0 for e in eps)
    hcorr = sum(e / (math.exp(e / KT) - 1.0) for e in eps)
    ts = KT * sum((e / KT) / (math.exp(e / KT) - 1.0) - math.log(1.0 - math.exp(-e / KT))
                  for e in eps)
    return {"ZPE": zpe, "Hcorr": hcorr, "TS": ts, "G_corr": zpe + hcorr - ts}


def demo():
    """Self-check on a two-mode synthetic OUTCAR snippet."""
    lines = [
        "   1 f  =   48.960837 THz   307.6 2PiTHz 1633.157728 cm-1   202.485660 meV",
        "   2 f/i=    0.657420 THz     4.1 2PiTHz    9.300000 cm-1     1.153200 meV",
    ]
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix="OUTCAR", delete=False) as fh:
        fh.write("\n".join(lines))
        path = fh.name
    real, imag = parse_frequencies(path)
    assert len(real) == 1 and len(imag) == 1, (real, imag)
    assert abs(real[0] - 0.20248566) < 1e-6, real
    t = thermo(real)
    # single 202.5 meV mode: ZPE = e/2, essentially no thermal population at 298 K.
    assert abs(t["ZPE"] - real[0] / 2.0) < 1e-9, t
    assert t["Hcorr"] < 1e-2 and t["TS"] < 1e-2, t
    print("freq_thermo demo OK:", {k: round(v, 4) for k, v in t.items()})


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
        raise SystemExit
    fn = sys.argv[1] if len(sys.argv) > 1 else "OUTCAR"
    real, imag = parse_frequencies(fn)
    print(f"real modes: {len(real)}   imaginary (dropped): {len(imag)}")
    if imag:
        print("  imaginary (cm^-1):", [round(e / CM2EV, 1) for e in imag])
    t = thermo(real)
    print(f"ZPE   = {t['ZPE']:.3f} eV")
    print(f"Hcorr = {t['Hcorr']:.3f} eV")
    print(f"TS    = {t['TS']:.3f} eV")
    print(f"ZPE + Hcorr - TS = {t['G_corr']:.3f} eV   (add to E_DFT = G)")
