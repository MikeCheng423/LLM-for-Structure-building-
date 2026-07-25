"""Human-readable per-job summary, written as ``job.log``.

Written automatically when a VASP job finishes or fails (see run_one_case), so
every job directory carries a readable result summary — parameters, energies,
cell, convergence, warnings and timing — next to the raw VASP output. The UI's
Results tab also offers it as a download.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from vasp_auto.incar import get_incar_value
from vasp_auto.parser import parse_vasprun
from vasp_auto.qe_tools import ATOMIC_MASSES
from vasp_auto.report import _kpoints_summary
from vasp_auto.structure import cell_parameters, read_poscar, scaled_lattice
from vasp_auto.workflow import (
    parse_energy_from_outcar,
    parse_outcar_summary,
    scan_vasp_errors,
)

_AVOGADRO = 6.02214076e23
_ANG3_TO_CM3 = 1.0e-24

ISMEAR_LABELS = {
    "-5": "tetrahedron method with Bloechl corrections",
    "-4": "tetrahedron method",
    "-1": "Fermi smearing",
    "0": "Gaussian smearing",
    "1": "first-order Methfessel-Paxton smearing",
    "2": "second-order Methfessel-Paxton smearing",
}

IBRION_LABELS = {
    "-1": "single-point energy",
    "0": "molecular dynamics",
    "1": "geometry optimization (RMM-DIIS / quasi-Newton)",
    "2": "geometry optimization (conjugate gradient)",
    "3": "geometry optimization (damped molecular dynamics)",
    "5": "vibrational frequencies (finite differences)",
    "6": "frequencies + elastic constants (finite differences)",
    "7": "vibrational frequencies (DFPT)",
    "8": "frequencies + dielectric (DFPT)",
}


def _fmt(value, unit="", nd=6):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{nd}f}{unit}"
    return f"{value}{unit}"


def _duration(seconds) -> str:
    if seconds is None:
        return "—"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{seconds} s ({h}:{m:02d}:{s:02d})"


def _formula(struct: dict) -> str:
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in zip(struct["elements"], struct["counts"]))


def _potcar_titles(job_dir: Path) -> list[str]:
    potcar = job_dir / "POTCAR"
    if not potcar.exists():
        return []
    titles = []
    for line in potcar.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "TITEL" in line:
            titles.append(line.split("=", 1)[1].strip())
    return titles


def _oszicar_energies(oszicar: Path) -> list[float]:
    """Free energy (F=) at each ionic step, in order."""
    if not oszicar.exists():
        return []
    energies = []
    for line in oszicar.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.search(r"\bF=\s*([-+.\dEe]+)", line)
        if match:
            try:
                energies.append(float(match.group(1)))
            except ValueError:
                pass
    return energies


def _walltime_seconds(job_dir: Path) -> float | None:
    outcar = job_dir / "OUTCAR"
    if not outcar.exists():
        return None
    value = None
    for line in outcar.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Elapsed time (sec):" in line:
            try:
                value = float(line.split(":")[-1])
            except ValueError:
                pass
    return value


def _functional(text: str) -> str:
    if (get_incar_value(text, "LHFCALC") or "").upper().startswith(".T"):
        return "hybrid (Hartree-Fock exchange mixing)"
    gga = get_incar_value(text, "GGA")
    return f"GGA ({gga})" if gga else "GGA-PBE"


def _cell_block(job_dir: Path):
    """Cell parameters + density from the final geometry (CONTCAR, else POSCAR)."""
    src = job_dir / "CONTCAR"
    if not src.exists() or src.stat().st_size == 0:
        src = job_dir / "POSCAR"
    if not src.exists():
        return None
    try:
        struct = read_poscar(src)
    except Exception:
        return None
    cp = cell_parameters(scaled_lattice(struct))
    mass = sum(ATOMIC_MASSES.get(el, 0.0) * n
               for el, n in zip(struct["elements"], struct["counts"]))
    density = None
    if cp["volume"] and mass:
        density = mass / (_AVOGADRO * cp["volume"] * _ANG3_TO_CM3)  # g/cm^3
    return {"src": src.name, "struct": struct, "density": density, **cp}


def _neb_image_energies(job_dir: Path) -> list[tuple[str, float]]:
    out = []
    for image_dir in sorted(p for p in job_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        energy = parse_energy_from_outcar(image_dir / "OUTCAR")
        if energy is not None:
            out.append((image_dir.name, energy))
    return out


def _section(title: str) -> list[str]:
    return ["", f"  {title}", "  " + "-" * 68]


def build_job_log(job_dir: Path, case_name: str | None = None,
                  calc_type: str | None = None, return_code: int | None = None) -> str:
    """Build the ``job.log`` text for one finished/failed job dir."""
    job_dir = Path(job_dir)
    case_name = case_name or job_dir.name
    incar_text = (job_dir / "INCAR").read_text(encoding="utf-8", errors="ignore") \
        if (job_dir / "INCAR").exists() else ""

    image_dirs = [p for p in job_dir.iterdir() if p.is_dir() and p.name.isdigit()] \
        if job_dir.exists() else []
    is_neb = bool(image_dirs)

    summary = parse_outcar_summary(job_dir / "OUTCAR")
    vasprun = parse_vasprun(job_dir / "vasprun.xml") or {}
    findings = scan_vasp_errors(job_dir)
    cell = _cell_block(job_dir)

    energy = summary["energy_eV"]
    if energy is None and is_neb:
        energy = next((e for _, e in _neb_image_energies(job_dir)), None)
    has_result = energy is not None
    status = "finished" if has_result else "failed"

    ibrion = get_incar_value(incar_text, "IBRION")
    nsw = get_incar_value(incar_text, "NSW")
    calc_label = IBRION_LABELS.get(str(ibrion), None)
    if is_neb:
        calc_label = "nudged elastic band (transition state)"
    elif calc_label is None:
        calc_label = "single-point energy" if (nsw in (None, "0")) else "geometry optimization"

    system = get_incar_value(incar_text, "SYSTEM")
    if not system and cell:
        system = _formula(cell["struct"])

    lines = [
        "=" * 72,
        "  vasp_auto — job summary",
        "=" * 72,
        f"  Case         : {case_name}",
        f"  System       : {system or '—'}",
        f"  Calculation  : {calc_label}" + (f"   [{calc_type}]" if calc_type else ""),
        f"  Status       : {status}",
        f"  Converged    : {'yes' if summary['converged'] else 'no'}",
        f"  Completed    : {datetime.now():%Y-%m-%d %H:%M}",
    ]
    if return_code is not None:
        lines.append(f"  VASP exit    : {return_code}")

    # Parameters
    lines += _section("PARAMETERS")
    ismear = get_incar_value(incar_text, "ISMEAR")
    sigma = get_incar_value(incar_text, "SIGMA")
    ispin = get_incar_value(incar_text, "ISPIN")
    lines += [
        f"  Functional   : {_functional(incar_text)}",
        f"  Precision    : {get_incar_value(incar_text, 'PREC') or '—'}",
        f"  Cutoff ENCUT : {_fmt(get_incar_value(incar_text, 'ENCUT'))} eV",
        f"  EDIFF        : {get_incar_value(incar_text, 'EDIFF') or '—'} eV",
        f"  Algorithm    : {get_incar_value(incar_text, 'ALGO') or '—'}",
        f"  Smearing     : {ISMEAR_LABELS.get(str(ismear), f'ISMEAR={ismear}')}"
        + (f", SIGMA = {sigma} eV" if sigma else ""),
        f"  Spin         : {'spin-polarised' if str(ispin) == '2' else 'non-magnetic'} (ISPIN={ispin or 1})",
        f"  k-points     : {_kpoints_summary(job_dir)}",
    ]
    titles = _potcar_titles(job_dir)
    if titles:
        lines.append("  Potentials   :")
        lines += [f"      {t}" for t in titles]

    # Results
    lines += _section("RESULTS")
    if is_neb:
        profile = _neb_image_energies(job_dir)
        if profile:
            values = [e for _, e in profile]
            lines += [f"  image {name} : {e:.6f} eV" for name, e in profile]
            lines += [
                f"  forward barrier  : {max(values) - values[0]:.6f} eV",
                f"  backward barrier : {max(values) - values[-1]:.6f} eV",
            ]
        else:
            lines.append("  No image energies parsed yet.")
    else:
        oszicar = _oszicar_energies(job_dir / "OSZICAR")
        lines.append(f"  Final energy        : {_fmt(energy, ' eV')}")
        if len(oszicar) >= 2:
            gained = oszicar[-1] - oszicar[0]
            lines += [
                f"  Initial energy      : {_fmt(oszicar[0], ' eV')}",
                f"  Relaxation gained   : {_fmt(gained, ' eV')} over {len(oszicar)} ionic steps",
            ]
        elif oszicar:
            lines.append(f"  Ionic steps         : {len(oszicar)}")
        lines += [
            f"  Maximum force       : {_fmt(vasprun.get('max_force_eV_A'), ' eV/Å', 3)}",
            f"  Fermi level         : {_fmt(vasprun.get('fermi_eV'), ' eV', 4)}",
            f"  Band gap            : {_fmt(vasprun.get('band_gap_eV'), ' eV', 4)}",
            f"  Pressure            : {_fmt(vasprun.get('pressure_kB'), ' kB', 3)}",
        ]
        if summary["magmom_total"] is not None:
            lines.append(f"  Total magnetisation : {_fmt(summary['magmom_total'], ' μB', 3)}")

    # Thermochemistry for frequency runs (IBRION 5-8): ZPE / Hcorr / -TS and the
    # Gibbs correction to add to E_DFT (see freq_thermo).
    if str(ibrion) in ("5", "6", "7", "8") and (job_dir / "OUTCAR").exists():
        from vasp_auto.freq_thermo import CM2EV, parse_frequencies, thermo
        real, imag = parse_frequencies(job_dir / "OUTCAR")
        if real:
            t = thermo(real)
            lines += _section("THERMOCHEMISTRY  (harmonic, 298.15 K)")
            lines += [
                f"  Real modes          : {len(real)}",
                f"  Imaginary (dropped) : {len(imag)}"
                + (f"   [{', '.join(f'{e / CM2EV:.1f}' for e in imag)} cm⁻¹]" if imag else ""),
                f"  Zero-point energy   : {t['ZPE']:.3f} eV",
                f"  Thermal corr. Hcorr : {t['Hcorr']:.3f} eV",
                f"  Entropy term  T·S   : {t['TS']:.3f} eV",
                f"  Gibbs correction    : {t['G_corr']:.3f} eV   (add to E_DFT for G)",
            ]
            # E_DFT for a frequency run is the undisplaced reference = first
            # OSZICAR step, not the last displaced geometry.
            osz = _oszicar_energies(job_dir / "OSZICAR")
            e_dft = osz[0] if osz else energy
            if e_dft is not None:
                lines.append(f"  G = E_DFT + corr    : {e_dft + t['G_corr']:.3f} eV")

    # Cell
    if cell and not is_neb:
        lines += _section(f"CELL  (from {cell['src']})")
        lines += [
            f"  a, b, c          : {cell['a']:.4f}, {cell['b']:.4f}, {cell['c']:.4f} Å",
            f"  alpha, beta, gamma : {cell['alpha']:.2f}, {cell['beta']:.2f}, {cell['gamma']:.2f} °",
            f"  Volume           : {cell['volume']:.3f} Å³",
            f"  Density          : {_fmt(cell['density'], ' g/cm³', 3)}",
        ]

    # Warnings / problems
    warnings = []
    if has_result and not summary["converged"]:
        if is_neb:
            warnings.append("The NEB calculation did not fully converge.")
        else:
            warnings.append(
                f"Geometry/electronic optimisation did not converge"
                + (f" (NSW = {nsw} reached or VASP stopped)." if nsw else "."))
    if not has_result:
        warnings.append("No final energy was produced — the job did not complete a VASP run.")
    if warnings:
        lines += _section("WARNINGS")
        lines += [f"  - {w}" for w in warnings]

    if findings:
        lines += _section("DETECTED PROBLEMS")
        lines += [f"  - {f['code']} (in {f['file']}): {f['hint']}" for f in findings]

    lines += ["", f"  Wall time    : {_duration(_walltime_seconds(job_dir))}",
              "=" * 72,
              f"Generated by vasp_auto on {datetime.now():%Y-%m-%d %H:%M}.", ""]
    return "\n".join(lines)


def write_job_log(job_dir: Path, case_name: str | None = None,
                  calc_type: str | None = None, return_code: int | None = None) -> Path | None:
    """Write ``job.log`` into the job directory. Never raises — a summary failure
    must not fail the run; returns the path, or None if it could not be written."""
    job_dir = Path(job_dir)
    try:
        text = build_job_log(job_dir, case_name, calc_type, return_code)
        path = job_dir / "job.log"
        path.write_text(text, encoding="utf-8")
        return path
    except Exception:
        return None
