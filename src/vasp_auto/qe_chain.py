"""Quantum ESPRESSO chained workflows, isolated from the VASP chain runner."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from vasp_auto.chain import CONVERGE_STEP, parse_workflow_steps
from vasp_auto.parser import parse_pw_final_structure, parse_pw_output
from vasp_auto.qe_tools import create_qe_job
from vasp_auto.runner import resolve_remote_run_mode, run_qe, run_vasp_remote
from vasp_auto.structure import read_poscar, write_poscar


def _step_config(config: dict, step: dict, carry: dict) -> dict:
    result = {**config, **carry}
    overrides = step.get("qe") or step.get("qe_params") or {}
    # Accept the existing workflow editor's generic override mapping too. QE
    # users can write {ecutwfc: 70}; VASP tags that have no QE meaning are ignored.
    if not overrides and isinstance(step.get("incar"), dict):
        overrides = {str(k).lower(): v for k, v in step["incar"].items()}
    for key, value in overrides.items():
        name = str(key).lower()
        if name in {"ecutwfc", "ecutrho", "conv_thr", "degauss", "smearing", "occupations",
                    "nbnd", "md_steps", "md_dt", "md_temperature", "exx_fraction",
                    "screening_parameter"}:
            result[f"qe_{name}"] = value
    return result


def _kpoints_spec(step: dict) -> dict | None:
    if step.get("kpath"):
        return {"mode": "line", "kpath": step["kpath"], "divisions": step.get("divisions", 20)}
    if step.get("kpoints"):
        return {"mesh": step["kpoints"]}
    return None


def _reuse_previous_state(previous: Path, current: Path) -> None:
    save = previous / "tmp"
    if save.exists():
        shutil.copytree(save, current / "tmp", dirs_exist_ok=True)
    manifest = current / "qe_stages.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    stages = data.get("stages", [])
    # DOS/bands/optics standalone jobs include their own SCF prerequisite. In a
    # chain the preceding save directory is the prerequisite, so do not repeat it.
    if stages and stages[0].get("input") == "scf.in":
        data["stages"] = stages[1:]
        manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run_qe_workflow_case(
    case_info: dict,
    steps: list[dict],
    config: dict,
    *,
    cpus: int | None = None,
    prepare_only: bool = False,
    remote: dict | None = None,
) -> list[dict]:
    case_dir = Path(case_info["case_dir"]).resolve()
    job_dir = Path(case_info["job_dir"]).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    current_struct = read_poscar(case_dir / "POSCAR")
    previous: Path | None = None
    rows: list[dict] = []
    carry: dict = {}
    machine = remote.get("name") or remote.get("host") if remote else None
    if remote and not prepare_only and resolve_remote_run_mode(remote) != "ssh":
        raise ValueError("QE chained workflows require direct SSH or detached offload execution")

    for index, step in enumerate(parse_workflow_steps(steps), start=1):
        calc_type = str(step["calc_type"])
        step_name = f"{index:02d}_{calc_type}"
        step_dir = job_dir / step_name
        cfg = _step_config(config, step, carry)
        row = {"project": case_info.get("project", ""),
               "case": f"{case_info['case_name']}:{step_name}", "engine": "qe",
               "calculation_type": calc_type, "step": step_name, "job_dir": str(step_dir)}

        with tempfile.TemporaryDirectory() as tmp:
            staged_case = Path(tmp)
            write_poscar(current_struct, staged_case / "POSCAR")
            info = {"case_dir": str(staged_case), "job_dir": str(step_dir),
                    "case_name": case_info["case_name"], "calculation_type": "scf"}
            if calc_type == CONVERGE_STEP:
                create_qe_job(info, cfg, calc_type="scf", kpoints_spec=_kpoints_spec(step),
                              spin=bool(step.get("spin") or cfg.get("spin")),
                              magmom_map=cfg.get("magmom_map"))
            else:
                create_qe_job(info, cfg, calc_type=calc_type, kpoints_spec=_kpoints_spec(step),
                              spin=bool(step.get("spin") or cfg.get("spin")),
                              magmom_map=cfg.get("magmom_map"))

        if previous is not None:
            _reuse_previous_state(previous, step_dir)

        if prepare_only:
            row.update({"status": "prepared", "converged": False})
            rows.append(row)
            if calc_type != CONVERGE_STEP:
                previous = step_dir
            continue

        if calc_type == CONVERGE_STEP:
            from vasp_auto.convergence import parse_encut_values, parse_kpoint_meshes, parse_sigma_values
            from vasp_auto.qe_convergence import converge_qe_case

            result = converge_qe_case(
                case_name=row["case"], base_job_dir=step_dir,
                qe_executable=str(cfg.get("qe_executable", "pw.x")), config=cfg, cpus=cpus,
                encut_values=parse_encut_values(str(step["encut"])) if step.get("encut") else None,
                sigma_values=parse_sigma_values(str(step["sigma"])) if step.get("sigma") else None,
                kpoint_meshes=parse_kpoint_meshes(str(step["kpoints"])) if step.get("kpoints") else None,
                energy_tolerance=float(step.get("energy_tol", 1e-4)),
                sigma_tolerance=float(step.get("sigma_tol", 1e-3)), remote=remote,
            )
            carry["qe_ecutwfc"] = result["selected_encut"]
            if result.get("selected_sigma") is not None:
                carry["qe_degauss"] = result["selected_sigma"]
            row.update({"status": "done", "converged": result.get("selected_energy_eV") is not None,
                        "energy_eV": result.get("selected_energy_eV"),
                        "selected_ecutwfc_Ry": result.get("selected_encut"),
                        "selected_degauss_Ry": result.get("selected_sigma"),
                        "selected_kpoints": result.get("selected_kpoints"),
                        "report_path": result.get("report_path")})
            rows.append(row)
            continue

        if remote:
            rc = run_vasp_remote(str(step_dir), remote, cpus=cpus,
                                 remote_subdir=f"{job_dir.name}/{step_name}", engine="qe")
        else:
            rc = run_qe(str(step_dir), str(cfg.get("qe_executable", "pw.x")), cpus=cpus)
        summary = parse_pw_output(step_dir / "pw.out") or {}
        row.update({"status": "done" if rc == 0 else "failed", "return_code": rc,
                    "energy_eV": summary.get("energy_eV"),
                    "converged": bool(summary.get("converged"))})
        if machine:
            row["machine"] = machine
        rows.append(row)
        final = parse_pw_final_structure(step_dir / "pw.out")
        if final:
            current_struct = final
        previous = step_dir
        if rc != 0:
            break
    return rows
