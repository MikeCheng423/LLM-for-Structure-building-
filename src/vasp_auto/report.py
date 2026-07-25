"""Per-job calculation reports: a short Markdown summary of setup + results.

`vasp-auto ... --report` writes report.md into every job directory; the UI
shows the same text and offers it as a download.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from vasp_auto.incar import get_incar_value
from vasp_auto.parser import parse_vasprun
from vasp_auto.workflow import (
    count_ionic_steps,
    parse_energy_from_outcar,
    parse_magmoms,
    parse_outcar_summary,
    scan_vasp_errors,
)

# INCAR tags worth echoing in the "calculation details" section.
INCAR_REPORT_KEYS = (
    "SYSTEM", "ENCUT", "EDIFF", "EDIFFG", "ISMEAR", "SIGMA", "ISPIN",
    "MAGMOM", "IBRION", "NSW", "NELM", "ALGO", "LHFCALC", "IMAGES",
)


def _fmt(value, unit=""):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}{unit}"
    return f"{value}{unit}"


def _incar_details(job_dir: Path) -> list[str]:
    incar = job_dir / "INCAR"
    if not incar.exists():
        return ["- INCAR: not found"]
    text = incar.read_text(encoding="utf-8")
    lines = []
    for key in INCAR_REPORT_KEYS:
        value = get_incar_value(text, key)
        if value is not None:
            lines.append(f"- {key} = {value}")
    return lines or ["- INCAR: no recognised tags"]


def _kpoints_summary(job_dir: Path) -> str:
    kpoints = job_dir / "KPOINTS"
    if not kpoints.exists():
        return "—"
    lines = [l.strip() for l in kpoints.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) >= 4 and lines[2].lower().startswith(("g", "m")):
        return f"{lines[2]} {lines[3]}"
    if len(lines) >= 3 and lines[2].lower().startswith("l"):
        return f"line-mode, {lines[1]} points per segment"
    return lines[0] if lines else "—"


def _neb_section(job_dir: Path) -> list[str]:
    image_dirs = [p for p in sorted(job_dir.iterdir()) if p.is_dir() and p.name.isdigit()]
    if not image_dirs:
        return []
    energies = []
    for image_dir in image_dirs:
        energy = parse_energy_from_outcar(image_dir / "OUTCAR")
        if energy is not None:
            energies.append((image_dir.name, energy))
    if not energies:
        return []
    values = [e for _, e in energies]
    lines = ["", "## NEB energy profile", ""]
    lines += [f"- image {name}: {energy:.6f} eV" for name, energy in energies]
    lines += [
        "",
        f"- forward barrier: {max(values) - values[0]:.6f} eV",
        f"- backward barrier: {max(values) - values[-1]:.6f} eV",
    ]
    return lines


def build_job_report(job_dir: Path, case_name: str | None = None, extra: dict | None = None) -> str:
    """Build the Markdown report text for one job directory."""
    job_dir = Path(job_dir)
    case_name = case_name or job_dir.name
    summary = parse_outcar_summary(job_dir / "OUTCAR")
    vasprun = parse_vasprun(job_dir / "vasprun.xml") or {}
    magmoms = parse_magmoms(job_dir / "OUTCAR") if summary["magmom_total"] is not None else None

    lines = [
        f"# Calculation report — {case_name}",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by vasp_auto.",
        "",
        "## Calculation details",
        "",
        f"- job directory: `{job_dir}`",
        f"- k-points: {_kpoints_summary(job_dir)}",
        *_incar_details(job_dir),
        "",
        "## Results",
        "",
        f"- converged: {'yes' if summary['converged'] else 'no'}",
        f"- free energy (TOTEN): {_fmt(summary['energy_eV'], ' eV')}",
        f"- energy without entropy: {_fmt(summary['energy_without_entropy_eV'], ' eV')}",
        f"- Fermi level: {_fmt(vasprun.get('fermi_eV'), ' eV')}",
        f"- band gap: {_fmt(vasprun.get('band_gap_eV'), ' eV')}",
        f"- max force: {_fmt(vasprun.get('max_force_eV_A'), ' eV/Å')}",
        f"- pressure: {_fmt(vasprun.get('pressure_kB'), ' kB')}",
        f"- ionic steps: {_fmt(count_ionic_steps(job_dir / 'OSZICAR'))}",
    ]

    if summary["magmom_total"] is not None:
        lines.append(f"- total magnetisation: {_fmt(summary['magmom_total'], ' μB')}")
        if magmoms:
            lines.append("- per-atom moments (μB): " + " ".join(f"{m:.3f}" for m in magmoms))

    for key, value in (extra or {}).items():
        if value is not None:
            lines.append(f"- {key}: {value}")

    lines += _neb_section(job_dir)

    findings = scan_vasp_errors(job_dir)
    if findings:
        lines += ["", "## Detected problems", ""]
        lines += [f"- {f['code']} (in {f['file']}): {f['hint']}" for f in findings]

    return "\n".join(lines) + "\n"


def write_job_report(job_dir: Path, case_name: str | None = None, extra: dict | None = None) -> Path:
    """Write report.md into the job directory; returns its path."""
    job_dir = Path(job_dir)
    report_path = job_dir / "report.md"
    report_path.write_text(build_job_report(job_dir, case_name, extra), encoding="utf-8")
    return report_path
