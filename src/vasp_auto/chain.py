"""Chained workflows: relax → scf → dos in one command.

Each step runs in its own subdirectory of the case job dir; outputs feed the
next step's inputs (CONTCAR → POSCAR, CHGCAR for ICHARG=11) either by the
per-type defaults in calc_types.CHAIN_INPUTS or an explicit `copy:` map.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from vasp_auto.calc_types import CHAIN_INPUTS, CalcType, parse_calc_type
from vasp_auto.incar import apply_spin_to_incar, set_incar_value
from vasp_auto.job_manager import DEFAULT_KPOINTS, load_incar_template
from vasp_auto.kpoints import kpoints_text_from_spec, mesh_kpoints_text, parse_mesh
from vasp_auto.potcar_finder import build_potcar
from vasp_auto.runner import resolve_remote_run_mode, run_vasp, run_vasp_remote
from vasp_auto.workflow import parse_outcar_summary, report_vasp_errors

# A workflow step name that runs an automatic SCF convergence scan instead of a
# single VASP calculation. The settings it finds (ENCUT/SIGMA/NELM/KPOINTS) are
# carried forward as overrides for every later step — the classic
# "convergence → optimisation → SCF → DOS" flow.
CONVERGE_STEP = "converge"


def parse_workflow_steps(spec) -> list[dict]:
    """Normalise a workflow spec into a list of step dicts.

    Accepts 'relax,scf,dos', a list of type names, or a list of dicts
    (from workflow.yaml) with keys: calc_type, incar, kpoints, kpath, copy.
    """
    if isinstance(spec, str):
        items = [item.strip() for item in spec.split(",") if item.strip()]
    elif isinstance(spec, dict):
        items = spec.get("steps", [])
    else:
        items = list(spec or [])

    steps = []
    for item in items:
        step = dict(item) if isinstance(item, dict) else {"calc_type": item}
        raw = str(step.get("calc_type") or step.get("step") or "").strip().lower()
        if raw == CONVERGE_STEP:
            step["calc_type"] = CONVERGE_STEP
            step["converge"] = True
            steps.append(step)
            continue
        calc_type = parse_calc_type(raw)
        if calc_type == CalcType.NEB:
            raise ValueError("NEB cannot be a chained workflow step; run it as a TSS case")
        step["calc_type"] = calc_type
        steps.append(step)

    if not steps:
        raise ValueError("Workflow has no steps")
    return steps


def load_workflow_spec(case_dir: Path, config: dict, cli_workflow: str | None) -> list[dict] | None:
    """Resolve the workflow for a case: CLI flag > case workflow.yaml > config."""
    if cli_workflow:
        return parse_workflow_steps(cli_workflow)

    case_file = Path(case_dir) / "workflow.yaml"
    if case_file.exists():
        data = yaml.safe_load(case_file.read_text(encoding="utf-8"))
        return parse_workflow_steps(data)

    if config.get("workflow"):
        return parse_workflow_steps(config["workflow"])

    return None


def _write_step_kpoints(
    step: dict, step_dir: Path, prev_dir: Path | None, case_dir: Path,
    carry_kpoints: str | None = None,
):
    target = step_dir / "KPOINTS"
    if step.get("kpath"):
        target.write_text(
            kpoints_text_from_spec(
                {"mode": "line", "kpath": step["kpath"], "divisions": step.get("divisions")},
            ),
            encoding="utf-8",
        )
    elif step.get("kpoints"):
        target.write_text(mesh_kpoints_text(parse_mesh(step["kpoints"])), encoding="utf-8")
    elif step.get("calc_type") and (case_dir / f"KPOINTS_{step['calc_type']}").exists():
        # Per-step KPOINTS saved from the workflow editor; an explicit user choice, so
        # it beats a converge scan's carried mesh (mirrors INCAR_<type>).
        # ponytail: type-keyed — two steps of the same type share one file.
        shutil.copy2(case_dir / f"KPOINTS_{step['calc_type']}", target)
    elif carry_kpoints:
        # k-mesh chosen by a preceding convergence step.
        target.write_text(mesh_kpoints_text(parse_mesh(carry_kpoints)), encoding="utf-8")
    elif (case_dir / "KPOINTS").exists():
        shutil.copy2(case_dir / "KPOINTS", target)
    elif prev_dir is not None and (prev_dir / "KPOINTS").exists():
        shutil.copy2(prev_dir / "KPOINTS", target)
    else:
        target.write_text(DEFAULT_KPOINTS, encoding="utf-8")


def _write_step_incar(
    step: dict, step_dir: Path, case_dir: Path, carry_incar: dict | None = None,
    calc_type=None,
):
    calc_type = calc_type or step["calc_type"]
    case_template = case_dir / f"INCAR_{calc_type}"
    if case_template.exists():
        text = case_template.read_text(encoding="utf-8")
    else:
        text = load_incar_template(str(calc_type))
    (step_dir / "INCAR").write_text(text, encoding="utf-8")

    # Settings from a preceding convergence step first, then the step's own
    # explicit incar: overrides win over them.
    for key, value in (carry_incar or {}).items():
        set_incar_value(step_dir / "INCAR", str(key), value)
    for key, value in (step.get("incar") or {}).items():
        set_incar_value(step_dir / "INCAR", str(key), value)


def _link_previous_outputs(step: dict, step_dir: Path, prev_dir: Path):
    copies = step.get("copy") or CHAIN_INPUTS.get(step["calc_type"], {"CONTCAR": "POSCAR"})
    for src_name, dst_name in copies.items():
        src = prev_dir / src_name
        if src.exists() and src.stat().st_size > 0:
            shutil.copy2(src, step_dir / dst_name)
        elif dst_name == "POSCAR" and (prev_dir / "POSCAR").exists():
            shutil.copy2(prev_dir / "POSCAR", step_dir / "POSCAR")
        else:
            print(f"Warning   : {src_name} missing in {prev_dir.name}; {dst_name} not staged")


def _converge_kwargs(step: dict) -> dict:
    """Build converge_scf_case keyword arguments from a converge step dict."""
    from vasp_auto.convergence import (
        parse_encut_values,
        parse_kpoint_meshes,
        parse_nelm_values,
        parse_sigma_values,
    )

    kwargs = {}
    if step.get("encut"):
        kwargs["encut_values"] = parse_encut_values(str(step["encut"]))
    if step.get("sigma"):
        kwargs["sigma_values"] = parse_sigma_values(str(step["sigma"]))

    scan_nelm = step.get("scan_nelm")
    scan_kpoints = step.get("scan_kpoints")
    if step.get("nelm"):
        kwargs["nelm_values"] = parse_nelm_values(str(step["nelm"]))
        scan_nelm = True if scan_nelm is None else scan_nelm
    if step.get("kpoints"):
        kwargs["kpoint_meshes"] = parse_kpoint_meshes(str(step["kpoints"]))
        scan_kpoints = True if scan_kpoints is None else scan_kpoints

    # Default: when nothing specific is asked for, do the NELM + k-mesh scan.
    kwargs["scan_nelm"] = True if scan_nelm is None else bool(scan_nelm)
    kwargs["scan_kpoints"] = True if scan_kpoints is None else bool(scan_kpoints)
    if step.get("energy_tol"):
        kwargs["energy_tolerance"] = float(step["energy_tol"])
    if step.get("sigma_tol"):
        kwargs["sigma_tolerance"] = float(step["sigma_tol"])
    if step.get("reuse_wavecar"):
        kwargs["reuse_wavecar"] = True
    return kwargs


def run_workflow_case(
    case_info,
    steps: list[dict],
    config: dict,
    cpus: int | None = None,
    prepare_only: bool = False,
    run_fn=run_vasp,
    remote: dict | None = None,
) -> list[dict]:
    case_dir = Path(case_info["case_dir"]).resolve()
    job_dir = Path(case_info["job_dir"]).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)

    if not (case_dir / "POSCAR").exists():
        raise FileNotFoundError(f"Workflow case requires POSCAR in {case_dir}")

    # Remote execution: each step runs on the remote machine over SSH (results are
    # pulled back so chaining CONTCAR -> POSCAR works as locally). Scheduler-mode
    # remotes aren't supported for chained workflows (they queue and return).
    machine = None
    if remote and not prepare_only:
        if resolve_remote_run_mode(remote) != "ssh":
            raise ValueError(
                "Chained workflows on a remote machine require its run mode to be "
                "'ssh' (direct mpirun); queue submission of a multi-step chain is "
                "not supported. Set run_mode: ssh on the machine, or run the "
                "workflow locally."
            )
        machine = remote.get("name") or remote.get("host")

        def run_fn(step_dir, _exe, cpus=None):
            step = Path(step_dir).resolve()
            try:
                subdir = str(step.relative_to(job_dir.parent))
            except ValueError:
                subdir = step.name
            return run_vasp_remote(str(step_dir), remote, cpus=cpus, remote_subdir=subdir)

    rows = []
    prev_dir = None
    potcar_source = case_dir / "POTCAR" if (case_dir / "POTCAR").exists() else None
    # Settings discovered by a converge step, applied to every later step.
    carry_incar: dict = {}
    carry_kpoints: str | None = None

    for index, step in enumerate(parse_workflow_steps(steps), start=1):
        calc_type = step["calc_type"]
        is_converge = bool(step.get("converge"))
        step_name = f"{index:02d}_{calc_type}"
        step_dir = job_dir / step_name
        step_dir.mkdir(parents=True, exist_ok=True)

        if prev_dir is None:
            shutil.copy2(case_dir / "POSCAR", step_dir / "POSCAR")
        else:
            _link_previous_outputs(step, step_dir, prev_dir)

        if potcar_source is None:
            build_potcar(
                poscar_path=str(step_dir / "POSCAR"),
                potcar_root=config.get("potcar_root"),
                output_path=str(step_dir / "POTCAR"),
                potcar_map=config.get("potcar_map"),
            )
            potcar_source = step_dir / "POTCAR"
        else:
            shutil.copy2(potcar_source, step_dir / "POTCAR")

        # A converge step scans inputs (as an SCF) instead of running once; its
        # encut/sigma/nelm/kpoints keys are scan ranges, not single-step values,
        # so strip them before staging the baseline INCAR/KPOINTS.
        incar_calc_type = CalcType.SCF if is_converge else calc_type
        stage_step = step
        if is_converge:
            stage_step = {k: v for k, v in step.items()
                          if k not in ("encut", "sigma", "nelm", "kpoints", "kpath")}
        _write_step_incar(stage_step, step_dir, case_dir, carry_incar, calc_type=incar_calc_type)
        if step.get("spin") or config.get("spin"):
            apply_spin_to_incar(step_dir / "INCAR", step_dir / "POSCAR", config.get("magmom_map"))
        _write_step_kpoints(stage_step, step_dir, prev_dir, case_dir, carry_kpoints)

        row = {
            "project": case_info.get("project", ""),
            "case": f"{case_info['case_name']}:{step_name}",
            "calculation_type": str(calc_type),
            "step": step_name,
            "job_dir": str(step_dir),
        }

        if is_converge:
            if prepare_only:
                row.update({"status": "prepared", "converged": False})
                rows.append(row)
                continue  # convergence cannot be staged without running
            from vasp_auto.convergence import converge_scf_case

            print(f"Step      : {step_name} (convergence scan)")
            result = converge_scf_case(
                case_name=f"{case_info['case_name']}:{step_name}",
                base_job_dir=step_dir,
                vasp_executable=str(config["vasp_executable"]),
                cpus=cpus,
                remote=remote,
                **_converge_kwargs(step),
            )
            if result.get("selected_encut") is not None:
                carry_incar["ENCUT"] = result["selected_encut"]
            if result.get("selected_sigma") is not None:
                carry_incar["SIGMA"] = result["selected_sigma"]
            if result.get("selected_nelm") is not None:
                carry_incar["NELM"] = result["selected_nelm"]
            if result.get("selected_kpoints"):
                carry_kpoints = result["selected_kpoints"]
            row.update({
                "status": "done",
                "energy_eV": result.get("selected_energy_eV"),
                "converged": result.get("selected_energy_eV") is not None,
                "selected_encut": result.get("selected_encut"),
                "selected_sigma": result.get("selected_sigma"),
                "selected_nelm": result.get("selected_nelm"),
                "selected_kpoints": result.get("selected_kpoints"),
                "report_path": result.get("report_path"),
            })
            if machine:
                row["machine"] = machine
            rows.append(row)
            continue  # a scan does not relax, so prev_dir is unchanged

        if prepare_only:
            row.update({"status": "prepared", "converged": False})
        else:
            print(f"Step      : {step_name}")
            return_code = run_fn(str(step_dir), str(config["vasp_executable"]), cpus=cpus)
            error_summary = report_vasp_errors(step_dir)
            summary = parse_outcar_summary(step_dir / "OUTCAR")
            row.update(
                {
                    "status": "done",
                    "energy_eV": summary["energy_eV"],
                    "converged": summary["converged"],
                    "return_code": return_code,
                }
            )
            if error_summary:
                row["errors"] = error_summary

        if machine:
            row["machine"] = machine
        rows.append(row)
        prev_dir = step_dir

    return rows
