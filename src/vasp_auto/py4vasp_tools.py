"""py4vasp-backed analysis of finished jobs (reads vaspout.h5).

Optional layer over the pure-Python parsers: every function returns None when
py4vasp is not installed, the job wrote no vaspout.h5, or the file lacks the
datum — VASP 6.4 writes DOS/eigenvalues to the h5 but no charge density and
no per-step history — so callers fall back to the vasprun.xml / XDATCAR /
OSZICAR parsers.
"""
from __future__ import annotations

from pathlib import Path


def _calc(job_dir: Path):
    """A py4vasp Calculation for the job dir, or None when unavailable."""
    if not (Path(job_dir) / "vaspout.h5").exists():
        return None
    try:
        from py4vasp import Calculation
    except ImportError:
        return None
    try:
        return Calculation.from_path(str(job_dir))
    except Exception:
        return None


def dos(job_dir: Path) -> dict | None:
    """Total DOS in the parse_dos payload shape: {efermi, energies, total}."""
    calc = _calc(job_dir)
    if calc is None:
        return None
    try:
        data = calc.dos.to_dict()
    except Exception:
        return None
    fermi = float(data.get("fermi_energy") or 0.0)
    # py4vasp reports energies relative to E_F; the UI expects absolute + efermi.
    energies = [float(e) + fermi for e in data["energies"]]
    if "up" in data and "down" in data:
        total = [[float(v) for v in data["up"]], [float(v) for v in data["down"]]]
    elif "total" in data:
        total = [[float(v) for v in data["total"]]]
    else:
        return None
    return {"efermi": fermi, "energies": energies, "total": total, "source": "py4vasp"}


def step_energies(job_dir: Path) -> list[float] | None:
    """Free energy (TOTEN) of every ionic step, for animation labels."""
    calc = _calc(job_dir)
    if calc is None:
        return None
    try:
        data = calc.energy[:].to_dict()
    except Exception:
        return None
    for key, values in data.items():
        if "TOTEN" in key:
            return [float(v) for v in values]
    return None
