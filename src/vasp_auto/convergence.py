from __future__ import annotations

import csv
import shutil
from pathlib import Path

from vasp_auto.incar import set_incar_value
from vasp_auto.runner import run_vasp, run_vasp_remote
from vasp_auto.workflow import parse_outcar_summary, report_vasp_errors


DEFAULT_NELM_VALUES = [40, 60, 80, 100]
DEFAULT_KPOINT_MESHES = [(2, 2, 1), (3, 3, 1), (4, 4, 1), (5, 5, 1)]
# Files worth carrying from one trial to the next to seed the SCF loop.
SEED_FILES = ("WAVECAR", "CHGCAR")

OUTPUT_FILES = {
    "CHG",
    "CHGCAR",
    "CONTCAR",
    "DOSCAR",
    "EIGENVAL",
    "IBZKPT",
    "OSZICAR",
    "OUTCAR",
    "PCDAT",
    "PROCAR",
    "REPORT",
    "WAVECAR",
    "XDATCAR",
    "vasprun.xml",
    "run.log",
}
WORK_DIRS = {"scf_convergence"}


def parse_nelm_values(text: str | None) -> list[int]:
    if not text:
        return list(DEFAULT_NELM_VALUES)
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("NELM values cannot be empty")
    return values


def parse_encut_values(text: str | None) -> list[int]:
    if not text:
        raise ValueError("ENCUT values cannot be empty, e.g. --converge-encut 400,450,500,550")
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("ENCUT values cannot be empty")
    return values


def parse_sigma_values(text: str | None) -> list[float]:
    if not text:
        raise ValueError("SIGMA values cannot be empty, e.g. --converge-sigma 0.2,0.1,0.05")
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("SIGMA values cannot be empty")
    return values


def parse_kpoint_meshes(text: str | None) -> list[tuple[int, int, int]]:
    if not text:
        return list(DEFAULT_KPOINT_MESHES)

    meshes = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.lower().replace("x", " ").split()
        if len(parts) == 1:
            value = int(parts[0])
            meshes.append((value, value, value))
        elif len(parts) == 3:
            meshes.append(tuple(int(part) for part in parts))
        else:
            raise ValueError(f"Invalid KPOINTS mesh: {item}")

    if not meshes:
        raise ValueError("KPOINTS values cannot be empty")
    return meshes


def _natoms_from_poscar(poscar_path: Path) -> int | None:
    try:
        lines = Path(poscar_path).read_text(encoding="utf-8").splitlines()
        return sum(int(x) for x in lines[6].split())
    except (OSError, IndexError, ValueError):
        return None


def write_gamma_kpoints(kpoints_path: Path, mesh: tuple[int, int, int]):
    kpoints_path.write_text(
        "\n".join(
            [
                "Automatic mesh",
                "0",
                "Gamma",
                f"{mesh[0]} {mesh[1]} {mesh[2]}",
                "0 0 0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _copy_clean_inputs(src_dir: Path, dst_dir: Path):
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        if item.name in OUTPUT_FILES or item.name in WORK_DIRS:
            continue
        target = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _run_trial(
    source_dir: Path,
    trial_dir: Path,
    stage: str,
    vasp_executable: str,
    cpus: int | None,
    nelm: int | None = None,
    encut: int | None = None,
    sigma: float | None = None,
    kpoints: tuple[int, int, int] | None = None,
    seed_dir: Path | None = None,
    remote: dict | None = None,
):
    _copy_clean_inputs(source_dir, trial_dir)
    if seed_dir is not None:
        for name in SEED_FILES:
            seed = seed_dir / name
            if seed.exists() and seed.stat().st_size > 0:
                shutil.copy2(seed, trial_dir / name)
    if nelm is not None:
        set_incar_value(trial_dir / "INCAR", "NELM", nelm)
    if encut is not None:
        set_incar_value(trial_dir / "INCAR", "ENCUT", encut)
    if sigma is not None:
        set_incar_value(trial_dir / "INCAR", "SIGMA", sigma)
    set_incar_value(trial_dir / "INCAR", "IBRION", -1)
    set_incar_value(trial_dir / "INCAR", "NSW", 0)
    if kpoints is not None:
        write_gamma_kpoints(trial_dir / "KPOINTS", kpoints)

    if remote is not None:
        # Mirror the local layout under remote_root so trials from different
        # cases (each with e.g. an "encut_400" dir) don't collide remotely.
        try:
            subdir = str(Path(trial_dir).resolve().relative_to(Path(source_dir).resolve().parent))
        except ValueError:
            subdir = Path(trial_dir).name
        return_code = run_vasp_remote(str(trial_dir), remote, cpus=cpus, remote_subdir=subdir)
    else:
        return_code = run_vasp(str(trial_dir), str(vasp_executable), cpus=cpus)
    summary = parse_outcar_summary(trial_dir / "OUTCAR")
    report_vasp_errors(trial_dir)

    entropy_per_atom = None
    if summary["energy_eV"] is not None and summary["energy_without_entropy_eV"] is not None:
        natoms = _natoms_from_poscar(trial_dir / "POSCAR")
        if natoms:
            entropy_per_atom = (summary["energy_eV"] - summary["energy_without_entropy_eV"]) / natoms

    return {
        "stage": stage,
        "trial_dir": str(trial_dir),
        "nelm": nelm if nelm is not None else "base",
        "encut": encut if encut is not None else "base",
        "sigma": sigma if sigma is not None else "base",
        "kpoints": " ".join(str(x) for x in kpoints) if kpoints else "base",
        "energy_eV": summary["energy_eV"],
        "entropy_eV_per_atom": entropy_per_atom,
        "converged": summary["converged"],
        "return_code": return_code,
    }


def _energy_improvement(previous: float | None, current: float | None):
    if previous is None or current is None:
        return None
    return previous - current


def _select_converged_trial(rows: list[dict], energy_tolerance: float):
    """Pick the cheapest setting whose energy stopped changing.

    Returns the first converged trial whose energy differs from the previous
    trial by no more than energy_tolerance — not the lowest-energy trial,
    which would always favour the most expensive setting. Falls back to the
    last converged trial, then the last trial with any energy.
    """
    valid = [row for row in rows if row["energy_eV"] is not None]
    if not valid:
        return None

    for row in valid:
        improvement = row.get("energy_improvement_eV")
        if row["converged"] and improvement is not None and abs(improvement) <= energy_tolerance:
            return row

    converged = [row for row in valid if row["converged"]]
    return converged[-1] if converged else valid[-1]


def _select_sigma_trial(rows: list[dict], entropy_tolerance: float):
    """Pick the largest SIGMA whose entropy term T*S stays below tolerance.

    The standard VASP guidance: choose the largest smearing for which the
    entropy contribution per atom is below ~1 meV. Falls back to the trial
    with the smallest entropy term, then the last trial with any energy.
    """
    valid = [row for row in rows if row["energy_eV"] is not None]
    if not valid:
        return None

    with_entropy = [
        row for row in valid
        if row["converged"] and row["entropy_eV_per_atom"] is not None
    ]
    acceptable = [row for row in with_entropy if abs(row["entropy_eV_per_atom"]) <= entropy_tolerance]
    if acceptable:
        return max(acceptable, key=lambda row: float(row["sigma"]))
    if with_entropy:
        return min(with_entropy, key=lambda row: abs(row["entropy_eV_per_atom"]))
    return valid[-1]


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "stage",
        "encut",
        "sigma",
        "nelm",
        "kpoints",
        "energy_eV",
        "energy_improvement_eV",
        "entropy_eV_per_atom",
        "converged",
        "return_code",
        "trial_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _write_markdown_report(path: Path, case_name: str, rows: list[dict], selected: dict[str, dict | None]):
    lines = [
        f"# SCF Convergence Report: {case_name}",
        "",
        "## Selected Settings",
        "",
    ]

    stage_keys = {"ENCUT": "encut", "SIGMA": "sigma", "NELM": "nelm", "KPOINTS": "kpoints"}
    for stage, best in selected.items():
        if best:
            value = best[stage_keys[stage]]
            lines.append(f"- {stage}: {value} with free energy {best['energy_eV']} eV")
        else:
            lines.append(f"- {stage}: no valid trial energy found")

    lines.extend(
        [
            "",
            "## Steps",
            "",
            "| Stage | ENCUT | SIGMA | NELM | KPOINTS | Free energy (eV) | Energy change (eV) | T*S /atom (eV) | Converged | Return code | Directory |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )

    for row in rows:
        energy = "" if row["energy_eV"] is None else f"{row['energy_eV']:.10f}"
        improvement = (
            ""
            if row.get("energy_improvement_eV") is None
            else f"{row['energy_improvement_eV']:.10f}"
        )
        entropy = (
            ""
            if row.get("entropy_eV_per_atom") is None
            else f"{row['entropy_eV_per_atom']:.6f}"
        )
        lines.append(
            "| {stage} | {encut} | {sigma} | {nelm} | {kpoints} | {energy} | {improvement} | {entropy} | {converged} | {return_code} | {trial_dir} |".format(
                stage=row["stage"],
                encut=row.get("encut", "base"),
                sigma=row.get("sigma", "base"),
                nelm=row["nelm"],
                kpoints=row["kpoints"],
                energy=energy,
                improvement=improvement,
                entropy=entropy,
                converged=row["converged"],
                return_code=row["return_code"],
                trial_dir=row["trial_dir"],
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scan_stage(
    stage: str,
    values: list,
    trial_dir_fn,
    trial_kwargs_fn,
    base_job_dir: Path,
    vasp_executable: str,
    cpus: int | None,
    energy_tolerance: float,
    stop_on_tol: bool = True,
    reuse_wavecar: bool = False,
    remote: dict | None = None,
) -> list[dict]:
    """Run trials for one stage, stopping once |ΔE| <= tolerance (when stop_on_tol)."""
    rows = []
    previous_energy = None
    seed_dir = None

    for value in values:
        trial_dir = trial_dir_fn(value)
        row = _run_trial(
            source_dir=base_job_dir,
            trial_dir=trial_dir,
            stage=stage,
            vasp_executable=vasp_executable,
            cpus=cpus,
            seed_dir=seed_dir,
            remote=remote,
            **trial_kwargs_fn(value),
        )
        row["energy_improvement_eV"] = _energy_improvement(previous_energy, row["energy_eV"])
        rows.append(row)
        if reuse_wavecar and row["return_code"] == 0:
            seed_dir = trial_dir

        if row["energy_eV"] is not None:
            if (
                stop_on_tol
                and row["converged"]
                and row["energy_improvement_eV"] is not None
                and abs(row["energy_improvement_eV"]) <= energy_tolerance
            ):
                break
            previous_energy = row["energy_eV"]

    return rows


def converge_scf_case(
    case_name: str,
    base_job_dir: Path,
    vasp_executable: str,
    cpus: int | None = None,
    nelm_values: list[int] | None = None,
    kpoint_meshes: list[tuple[int, int, int]] | None = None,
    encut_values: list[int] | None = None,
    sigma_values: list[float] | None = None,
    energy_tolerance: float = 1e-4,
    sigma_tolerance: float = 1e-3,
    scan_nelm: bool = True,
    scan_kpoints: bool = True,
    reuse_wavecar: bool = False,
    remote: dict | None = None,
):
    base_job_dir = Path(base_job_dir).resolve()
    nelm_values = nelm_values or list(DEFAULT_NELM_VALUES)
    kpoint_meshes = kpoint_meshes or list(DEFAULT_KPOINT_MESHES)

    convergence_dir = base_job_dir / "scf_convergence"
    convergence_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    selected = {}

    def scan(stage, values, dir_fn, kwargs_fn, stop_on_tol=True):
        stage_rows = _scan_stage(
            stage,
            values,
            dir_fn,
            kwargs_fn,
            base_job_dir,
            vasp_executable,
            cpus,
            energy_tolerance,
            stop_on_tol=stop_on_tol,
            reuse_wavecar=reuse_wavecar,
            remote=remote,
        )
        rows.extend(stage_rows)
        return stage_rows

    # ENCUT first: it changes the basis set every other stage depends on.
    best_encut = None
    selected_encut = None
    if encut_values:
        encut_rows = scan(
            "ENCUT",
            encut_values,
            lambda encut: convergence_dir / f"encut_{encut}",
            lambda encut: {"encut": encut},
        )
        best_encut = _select_converged_trial(encut_rows, energy_tolerance)
        selected_encut = int(best_encut["encut"]) if best_encut else int(encut_values[-1])
        selected["ENCUT"] = best_encut

    # SIGMA next: judged by the entropy term, so every trial must run.
    best_sigma = None
    selected_sigma = None
    if sigma_values:
        sigma_rows = scan(
            "SIGMA",
            sigma_values,
            lambda sigma: convergence_dir / f"sigma_{sigma}",
            lambda sigma: {"sigma": sigma, "encut": selected_encut},
            stop_on_tol=False,
        )
        best_sigma = _select_sigma_trial(sigma_rows, sigma_tolerance)
        selected_sigma = float(best_sigma["sigma"]) if best_sigma else float(sigma_values[-1])
        selected["SIGMA"] = best_sigma

    best_nelm = None
    selected_nelm = None
    if scan_nelm:
        nelm_rows = scan(
            "NELM",
            nelm_values,
            lambda nelm: convergence_dir / f"nelm_{nelm}",
            lambda nelm: {"nelm": nelm, "encut": selected_encut, "sigma": selected_sigma},
        )
        best_nelm = _select_converged_trial(nelm_rows, energy_tolerance)
        selected_nelm = int(best_nelm["nelm"]) if best_nelm else int(nelm_values[-1])
        selected["NELM"] = best_nelm

    best_kpoints = None
    if scan_kpoints:
        kpoints_rows = scan(
            "KPOINTS",
            kpoint_meshes,
            lambda mesh: convergence_dir / ("kpoints_" + "x".join(str(x) for x in mesh)),
            lambda mesh: {
                "nelm": selected_nelm,
                "encut": selected_encut,
                "sigma": selected_sigma,
                "kpoints": mesh,
            },
        )
        best_kpoints = _select_converged_trial(kpoints_rows, energy_tolerance)
        selected["KPOINTS"] = best_kpoints

    csv_path = convergence_dir / "scf_convergence_steps.csv"
    report_path = convergence_dir / "scf_convergence_report.md"
    _write_csv(csv_path, rows)
    _write_markdown_report(report_path, case_name, rows, selected)

    final = best_kpoints or best_nelm or best_sigma or best_encut

    return {
        "case": case_name,
        "job_dir": str(base_job_dir),
        "convergence_dir": str(convergence_dir),
        "selected_encut": selected_encut,
        "selected_sigma": selected_sigma,
        "selected_nelm": selected_nelm,
        "selected_kpoints": best_kpoints["kpoints"] if best_kpoints else None,
        "selected_energy_eV": final["energy_eV"] if final else None,
        "report_path": str(report_path),
        "csv_path": str(csv_path),
        "steps": rows,
    }
