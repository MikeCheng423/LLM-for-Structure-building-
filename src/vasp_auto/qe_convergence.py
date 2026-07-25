"""Quantum ESPRESSO cutoff, smearing, and k-point convergence scans."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

from vasp_auto.parser import parse_pw_output
from vasp_auto.qe_tools import create_qe_job
from vasp_auto.runner import run_qe, run_vasp_remote


def _selected(records: list[dict], key: str, tolerance: float):
    previous = None
    for record in records:
        energy = record.get("energy_eV")
        if energy is not None and previous is not None and abs(energy - previous) <= tolerance:
            return record[key], energy
        if energy is not None:
            previous = energy
    valid = [r for r in records if r.get("energy_eV") is not None]
    return (valid[-1][key], valid[-1]["energy_eV"]) if valid else (None, None)


def converge_qe_case(
    *,
    case_name: str,
    base_job_dir: str | Path,
    qe_executable: str,
    config: dict,
    cpus: int | None = None,
    encut_values: list[int] | None = None,
    sigma_values: list[float] | None = None,
    kpoint_meshes: list[tuple[int, int, int]] | None = None,
    energy_tolerance: float = 1e-4,
    sigma_tolerance: float = 1e-3,
    scan_kpoints: bool = True,
    remote: dict | None = None,
) -> dict:
    """Run QE-native ecutwfc (Ry), degauss (Ry), and k-mesh scans."""
    base = Path(base_job_dir).resolve()
    root = base / "qe_convergence"
    root.mkdir(parents=True, exist_ok=True)
    pseudo_dir = base / "pseudo"
    run_config = {**config, "pseudo_dir": str(pseudo_dir)}
    encuts = encut_values or [30, 40, 50, 60, 70]
    meshes = kpoint_meshes or [(3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6)]
    records: list[dict] = []

    def trial(kind: str, value, *, ecut: float, sigma: float | None, mesh):
        tag = str(value).replace(" ", "").replace("(", "").replace(")", "").replace(",", "x")
        directory = root / f"{kind}_{tag}"
        if directory.exists():
            shutil.rmtree(directory)
        cfg = {**run_config, "qe_ecutwfc": ecut}
        if sigma is not None:
            cfg["qe_degauss"] = sigma
        info = {"case_dir": str(base), "job_dir": str(directory), "case_name": case_name,
                "calculation_type": "scf"}
        create_qe_job(info, cfg, calc_type="scf", kpoints_spec={"mesh": "x".join(map(str, mesh))})
        if remote:
            rc = run_vasp_remote(str(directory), remote, cpus=cpus,
                                 remote_subdir=f"{base.name}/qe_convergence/{directory.name}", engine="qe")
        else:
            rc = run_qe(str(directory), qe_executable, cpus=cpus)
        summary = parse_pw_output(directory / "pw.out") or {}
        record = {"scan": kind, "value": value, "return_code": rc,
                  "energy_eV": summary.get("energy_eV"), "converged": summary.get("converged", False)}
        records.append(record)
        return record

    cutoff_records = [trial("ecutwfc_Ry", value, ecut=value, sigma=None, mesh=meshes[0])
                      for value in encuts]
    selected_encut, selected_energy = _selected(cutoff_records, "value", energy_tolerance)
    selected_encut = selected_encut or encuts[-1]

    selected_sigma = None
    if sigma_values:
        sigma_records = [trial("degauss_Ry", value, ecut=selected_encut, sigma=value, mesh=meshes[0])
                         for value in sigma_values]
        selected_sigma, selected_energy = _selected(sigma_records, "value", sigma_tolerance)

    selected_mesh = None
    if scan_kpoints:
        mesh_records = [trial("kmesh", "x".join(map(str, mesh)), ecut=selected_encut,
                              sigma=selected_sigma, mesh=mesh) for mesh in meshes]
        selected_mesh, selected_energy = _selected(mesh_records, "value", energy_tolerance)

    csv_path = root / "scf_convergence_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scan", "value", "return_code", "energy_eV", "converged"])
        writer.writeheader()
        writer.writerows(records)
    report_path = root / "scf_convergence_report.md"
    report_path.write_text(
        "# Quantum ESPRESSO SCF convergence\n\n"
        "> `ecutwfc` and `degauss` are in Rydberg. They are not VASP ENCUT/SIGMA values.\n\n"
        f"Selected ecutwfc: {selected_encut} Ry\n\n"
        f"Selected degauss: {selected_sigma if selected_sigma is not None else 'not scanned'} Ry\n\n"
        f"Selected k-mesh: {selected_mesh or 'not scanned'}\n",
        encoding="utf-8",
    )
    return {"selected_encut": selected_encut, "selected_sigma": selected_sigma,
            "selected_nelm": None, "selected_kpoints": selected_mesh,
            "selected_energy_eV": selected_energy, "report_path": str(report_path),
            "csv_path": str(csv_path), "records": records}
