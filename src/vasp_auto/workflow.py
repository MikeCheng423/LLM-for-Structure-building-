from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from vasp_auto.incar import set_incar_value
from vasp_auto.parser import parse_pw_output, parse_vasprun
from vasp_auto.runner import (
    resolve_remote_run_mode,
    run_ase,
    run_qe,
    run_vasp,
    run_vasp_remote,
    submit_job,
    submit_job_remote,
)

VASPRUN_ROW_KEYS = ("fermi_eV", "band_gap_eV", "max_force_eV_A", "pressure_kB")


CONVERGENCE_MARKERS = (
    "reached required accuracy",
    "aborting loop because EDIFF is reached",
)

# Known VASP failure signatures: (code, text to find in run.log/OUTCAR, suggested fix).
# Specific errors are checked first; generic fallback signatures are listed separately
# in VASP_GENERIC_ERROR_SIGNATURES and are only reported when no specific signature matched.
VASP_ERROR_SIGNATURES = (
    ("ZBRENT", "ZBRENT: fatal",
     "ionic step failed to bracket; copy CONTCAR->POSCAR and restart, or reduce POTIM"),
    ("EDDDAV", "Error EDDDAV", "Davidson diagonalisation failed; try ALGO = All or reduce NSIM"),
    ("RHOSYG", "RHOSYG", "charge-density symmetrisation failed; try ISYM = 0"),
    ("SUBSPACE", "Sub-Space-Matrix is not hermitian", "try ALGO = Normal, reduce AMIX, or set ISYM = 0"),
    ("ZPOTRF", "Routine ZPOTRF failed", "structure may be collapsing; reduce POTIM or check the starting geometry"),
    ("PRICEL", "internal error in subroutine PRICEL", "symmetry detection failed; try ISYM = 0 or adjust SYMPREC"),
    ("SGRCON", "internal error in subroutine SGRCON", "k-mesh/symmetry mismatch; adjust KPOINTS or SYMPREC"),
    ("TOO_FEW_BANDS", "TOO FEW BANDS", "increase NBANDS"),
)

# Generic VASP fatal banner — printed alongside many specific errors; only
# report it when no more-specific signature was found in the same scan.
VASP_GENERIC_ERROR_SIGNATURES = (
    ("SICK_JOB", "I REFUSE TO CONTINUE WITH THIS SICK JOB",
     "fatal input error; check INCAR/POSCAR/POTCAR consistency"),
)

# INCAR settings --auto-retry applies for each error code. Only fixes that are
# safe without knowing the system are listed; codes without an entry (e.g.
# TOO_FEW_BANDS, which needs a system-specific NBANDS) are never auto-fixed.
#
# Note on ZBRENT: changing IBRION alone from the same (unconverged) POSCAR
# will typically reproduce the same failure.  The primary remedy is to restart
# from CONTCAR (handled by apply_error_fixes via VASP_ERROR_RESTART_FROM_CONTCAR).
# A mild IBRION switch is kept as a secondary INCAR nudge only for cases where
# CONTCAR is absent or identical to POSCAR.
VASP_ERROR_FIXES: dict[str, dict[str, str | int | float]] = {
    "ZBRENT": {"IBRION": 1},
    "EDDDAV": {"ALGO": "All"},
    "RHOSYG": {"ISYM": 0},
    "SUBSPACE": {"ALGO": "Normal", "ISYM": 0},
    "ZPOTRF": {"POTIM": 0.1},
    "PRICEL": {"ISYM": 0},
    "SGRCON": {"SYMPREC": 1e-4},
}

# Error codes for which auto-retry should copy CONTCAR -> POSCAR before re-running.
# ZBRENT: the VASP error message itself says "copy CONTCAR to POSCAR and continue".
# ZPOTRF: structure is collapsing; restarting from the latest geometry is safer
#         than re-running from the original POSCAR after only an INCAR tweak.
VASP_ERROR_RESTART_FROM_CONTCAR: frozenset[str] = frozenset({"ZBRENT", "ZPOTRF"})


def parse_outcar_summary(outcar_path: Path) -> dict:
    """Read energy, convergence, entropy term, and total moment in one pass over OUTCAR."""
    summary = {
        "energy_eV": None,
        "converged": False,
        "energy_without_entropy_eV": None,
        "magmom_total": None,
    }
    if not outcar_path.exists():
        return summary

    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "free  energy   TOTEN" in line:
                parts = line.split()
                try:
                    summary["energy_eV"] = float(parts[4])
                except (IndexError, ValueError):
                    pass
            elif "energy  without entropy" in line:
                try:
                    summary["energy_without_entropy_eV"] = float(line.split("=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif "number of electron" in line and "magnetization" in line:
                parts = line.split()
                try:
                    summary["magmom_total"] = float(parts[-1])
                except (IndexError, ValueError):
                    pass
            elif not summary["converged"] and any(marker in line for marker in CONVERGENCE_MARKERS):
                summary["converged"] = True
    return summary


def parse_magmoms(outcar_path: Path) -> list[float] | None:
    """Per-atom magnetic moments from the last 'magnetization (x)' block in OUTCAR."""
    outcar_path = Path(outcar_path)
    if not outcar_path.exists():
        return None

    magmoms = None
    current = None
    in_table = False
    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "magnetization (x)" in line:
                current = []
                in_table = False
                continue
            if current is None:
                continue
            stripped = line.strip()
            if stripped.startswith("---"):
                if in_table:
                    # Table closed; keep this block (later blocks overwrite it).
                    if current:
                        magmoms = current
                    current = None
                else:
                    in_table = True
                continue
            if in_table:
                parts = stripped.split()
                if parts and parts[0].isdigit():
                    try:
                        current.append(float(parts[-1]))
                    except ValueError:
                        current = None
    return magmoms


def parse_energy_from_outcar(outcar_path: Path):
    return parse_outcar_summary(outcar_path)["energy_eV"]


def is_converged(outcar_path: Path):
    return parse_outcar_summary(outcar_path)["converged"]


def count_ionic_steps(oszicar_path: Path):
    if not oszicar_path.exists():
        return None

    count = 0
    with open(oszicar_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "F=" in line:
                count += 1
    return count


def scan_vasp_errors(job_dir: Path) -> list[dict]:
    """Scan run.log and OUTCAR for known VASP error signatures.

    Returns a list of {"code", "file", "hint"} dicts, one per distinct error.

    Generic signatures (SICK_JOB) are reported only when no more-specific
    signature matched — VASP prints "I REFUSE TO CONTINUE WITH THIS SICK JOB"
    alongside every specific error banner, so treating it as independent would
    cause duplicate, misleading reports.
    """
    job_dir = Path(job_dir)
    findings = []
    seen = set()
    generic_findings: list[dict] = []
    generic_seen: set[str] = set()

    for filename in ("run.log", "OUTCAR"):
        path = job_dir / filename
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for code, signature, hint in VASP_ERROR_SIGNATURES:
                    if code not in seen and signature in line:
                        seen.add(code)
                        findings.append({"code": code, "file": filename, "hint": hint})
                for code, signature, hint in VASP_GENERIC_ERROR_SIGNATURES:
                    if code not in generic_seen and signature in line:
                        generic_seen.add(code)
                        generic_findings.append({"code": code, "file": filename, "hint": hint})

    # Only append generic findings when no specific error was detected.
    if not findings:
        findings.extend(generic_findings)

    return findings


def geometry_converged(job_dir: Path) -> bool:
    """Return True when the geometry appears to have reached its minimum.

    Checks (in order):
    1. OUTCAR/run.log contains a CONVERGENCE_MARKER ("reached required accuracy"
       or "aborting loop because EDIFF is reached").
    2. The parsed max force from OUTCAR is at or below the force criterion read
       from INCAR (|EDIFFG| when EDIFFG < 0, default 0.02 eV/Å).

    Returns False when no outputs exist or neither condition is satisfied.
    """
    job_dir = Path(job_dir)

    # 1. Check for explicit convergence markers in run.log and OUTCAR.
    for filename in ("run.log", "OUTCAR"):
        path = job_dir / filename
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if any(marker in line for marker in CONVERGENCE_MARKERS):
                    return True

    # 2. Fallback: compare max force from vasprun.xml to EDIFFG force criterion.
    # Read EDIFFG from INCAR (negative value → force criterion).
    force_tol = 0.02  # eV/Å default (matches VASP default EDIFFG = -0.02)
    incar_path = job_dir / "INCAR"
    if incar_path.exists():
        with open(incar_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Strip inline comments (! or #) and match EDIFFG = <value>
                stripped = line.split("!")[0].split("#")[0].strip()
                if stripped.upper().startswith("EDIFFG"):
                    try:
                        val = float(stripped.split("=")[1].strip().split()[0])
                        if val < 0:
                            force_tol = abs(val)
                    except (IndexError, ValueError):
                        pass

    from vasp_auto.parser import parse_vasprun
    vasprun = parse_vasprun(job_dir / "vasprun.xml")
    if vasprun is not None and vasprun.get("max_force_eV_A") is not None:
        return vasprun["max_force_eV_A"] <= force_tol

    return False


def report_vasp_errors(job_dir: Path) -> str:
    """Print diagnostics for known VASP errors; return a summary string for Excel.

    For ZBRENT specifically: when the geometry appears already converged, the
    hint is adjusted to reflect that restarting from CONTCAR is the right action
    rather than changing POTIM or IBRION.
    """
    job_dir = Path(job_dir)
    findings = scan_vasp_errors(job_dir)
    for finding in findings:
        hint = finding["hint"]
        if finding["code"] == "ZBRENT" and geometry_converged(job_dir):
            hint = (
                "geometry appears already converged — VASP could not bracket a further "
                "step; accept the result or copy CONTCAR->POSCAR and restart. "
                "Reducing EDIFF/EDIFFG or POTIM only helps if forces are still large."
            )
        print(f"VASP error: {finding['code']} (in {finding['file']}) — {hint}")
    return "; ".join(f"{finding['code']}: {finding['hint']}" for finding in findings)


def apply_error_fixes(job_dir: Path, findings: list[dict]) -> list[str]:
    """Apply the known fix for each finding; returns action strings.

    For errors in VASP_ERROR_RESTART_FROM_CONTCAR, if a non-empty CONTCAR
    exists and differs from POSCAR, it is copied onto POSCAR as the primary
    fix.  Any INCAR edits from VASP_ERROR_FIXES are applied afterwards as a
    secondary nudge.
    """
    applied = []
    job_dir = Path(job_dir)
    incar_path = job_dir / "INCAR"
    contcar_path = job_dir / "CONTCAR"
    poscar_path = job_dir / "POSCAR"

    for finding in findings:
        code = finding["code"]

        # Primary fix: restart from CONTCAR when appropriate.
        if code in VASP_ERROR_RESTART_FROM_CONTCAR:
            if (
                contcar_path.exists()
                and contcar_path.stat().st_size > 0
                and contcar_path.read_bytes() != poscar_path.read_bytes()
            ):
                shutil.copy2(contcar_path, poscar_path)
                applied.append("restart from CONTCAR")

        # Secondary fix: INCAR edits (always applied when listed).
        for key, value in VASP_ERROR_FIXES.get(code, {}).items():
            set_incar_value(incar_path, key, value)
            applied.append(f"{key} = {value}")

    return applied


def _neb_image_dirs(job_dir: Path):
    return [p for p in sorted(job_dir.iterdir()) if p.is_dir() and p.name.isdigit()]


def _neb_image_energy(image_dir: Path):
    """Try OUTCAR, then OSZICAR (last F= token), then vasprun.xml for an image energy."""
    outcar = image_dir / "OUTCAR"
    energy = parse_energy_from_outcar(outcar)
    if energy is not None:
        return energy

    oszicar = image_dir / "OSZICAR"
    if oszicar.exists():
        last_energy = None
        with open(oszicar, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "F=" in line:
                    try:
                        idx = line.index("F=")
                        last_energy = float(line[idx + 2:].split()[0])
                    except (ValueError, IndexError):
                        pass
        if last_energy is not None:
            return last_energy

    vasprun = image_dir / "vasprun.xml"
    if vasprun.exists():
        from vasp_auto.parser import parse_vasprun
        result = parse_vasprun(vasprun)
        if result and result.get("energy_eV") is not None:
            return result["energy_eV"]

    return None


def _tail_has(path: Path, needle: str, tail: int = 4000) -> bool:
    """True if the last ``tail`` bytes of a text file contain ``needle``."""
    try:
        with open(path, "rb") as f:
            try:
                f.seek(-tail, 2)
            except OSError:
                f.seek(0)
            return needle.encode() in f.read()
    except OSError:
        return False


def job_finished(job_dir) -> bool:
    """True once a job has actually run to completion — engine-agnostic.

    Distinguishes a finished run from one whose OUTCAR/pw.out is still streaming
    (or was killed mid-run): the per-job ``job.log`` summary (VASP), the OUTCAR
    end-of-run footer (legacy VASP without job.log), QE's ``JOB DONE.``, or ASE's
    ``ase_results.json`` (only written at the end).
    """
    job_dir = Path(job_dir)
    if (job_dir / ".qe_complete").exists():
        return True
    if (job_dir / "job.log").exists() or (job_dir / "ase_results.json").exists():
        return True
    outcar = job_dir / "OUTCAR"
    if outcar.exists() and _tail_has(outcar, "General timing"):
        return True
    pw = job_dir / "pw.out"
    if pw.exists() and _tail_has(pw, "JOB DONE"):
        return True
    manifest = job_dir / "qe_stages.json"
    if manifest.exists():
        try:
            stages = json.loads(manifest.read_text(encoding="utf-8")).get("stages", [])
            output = job_dir / stages[-1]["output"]
            return output.exists() and _tail_has(output, "JOB DONE")
        except (ValueError, KeyError, IndexError, OSError, TypeError):
            return False
    return False


def pid_alive(pid: int) -> bool:
    """True if a process with this PID currently exists on this machine."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


def proc_cwds() -> set[str]:
    """Working directories of every process we can inspect (via /proc).

    The real "is something running here" signal: mpirun/vasp/pw.x execute with
    cwd inside the job directory, so a job dir with a live process under it is
    genuinely running whatever files it has or hasn't written yet.
    """
    cwds: set[str] = set()
    for link in Path("/proc").glob("[0-9]*/cwd"):
        try:
            cwds.add(os.readlink(link))
        except OSError:
            pass  # other users' processes aren't readable — fine
    return cwds


def dir_active(job_dir, cwds: set[str] | None = None) -> bool:
    """True if any live process is working in ``job_dir`` (or a subdir — NEB
    images and convergence trials run in nested folders)."""
    d = str(Path(job_dir))
    if cwds is None:
        cwds = proc_cwds()
    return any(c == d or c.startswith(d + "/") for c in cwds)


def pids_in_dir(job_dir) -> list[int]:
    """PIDs of every live process whose cwd is inside ``job_dir`` — the kill-side
    counterpart of :func:`dir_active`, so anything the board shows as "running"
    can also be terminated, even without a recorded ``.pid``."""
    d = str(Path(job_dir))
    pids: list[int] = []
    for link in Path("/proc").glob("[0-9]*/cwd"):
        try:
            c = os.readlink(link)
        except OSError:
            continue
        if c == d or c.startswith(d + "/"):
            pids.append(int(link.parent.name))
    return pids


def _read_pid(job_dir: Path) -> int | None:
    """The PID a running job recorded in ``.pid`` (None if absent/garbage)."""
    try:
        return int((Path(job_dir) / ".pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _write_pid(job_dir) -> None:
    try:
        (Path(job_dir) / ".pid").write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass


def _clear_pid(job_dir) -> None:
    try:
        (Path(job_dir) / ".pid").unlink()
    except OSError:
        pass


def terminal_status(job_dir, has_output: bool) -> str:
    """Coarse run state of a job dir (no scheduler override in play).

    Only "done" once the run has genuinely finished (:func:`job_finished`), and
    only "running" when something is *actually alive*: the PID recorded in
    ``.pid`` still exists, or a live process has its cwd inside the job dir.
    Output files without a live process read "stalled" — a file having been
    created is not proof the job is still running. A job stopped via the kill
    button/CLI leaves a ``.terminated`` marker. Mirrored by the shell in
    ``list_remote_jobs`` and by ``_local_job_rows`` so every tab agrees.
    (Liveness is only ever checked on the machine the job runs on — local dirs
    here, remote dirs inside the remote shell scan.)
    """
    job_dir = Path(job_dir)
    if (job_dir / ".terminated").exists():
        return "terminated"
    if job_finished(job_dir):
        return "done"
    pid = _read_pid(job_dir)
    alive = (pid is not None and pid_alive(pid)) or dir_active(job_dir)
    if alive:
        return "running"
    if pid is not None or has_output or (job_dir / "OSZICAR").exists():
        return "stalled"
    return "missing"


def _build_neb_row(project_name, mode, case_info, status_override=None):
    job_dir = Path(case_info["job_dir"])
    image_dirs = _neb_image_dirs(job_dir) if job_dir.exists() else []
    energies = []
    converged_flags = []
    ionic_steps = 0

    if image_dirs:
        numeric = sorted(image_dirs, key=lambda p: int(p.name))
        endpoints = {numeric[0].name, numeric[-1].name} if len(numeric) >= 2 else {"00"}
    else:
        endpoints = {"00"}

    for image_dir in image_dirs:
        is_endpoint = image_dir.name in endpoints
        outcar = image_dir / "OUTCAR"
        oszicar = image_dir / "OSZICAR"

        if is_endpoint:
            # Endpoints may have no OUTCAR; try all sources for their energy.
            energy = _neb_image_energy(image_dir)
            if energy is not None:
                energies.append((image_dir.name, energy))
            # Endpoints are excluded from convergence tracking.
        else:
            energy = _neb_image_energy(image_dir)
            if energy is not None:
                energies.append((image_dir.name, energy))
            if outcar.exists():
                converged_flags.append(is_converged(outcar))

        steps = count_ionic_steps(oszicar)
        if steps:
            ionic_steps += steps

    energy_values = [energy for _, energy in energies]
    barrier = max(energy_values) - min(energy_values) if energy_values else None

    # Forward/backward barriers: need endpoint energies to be meaningful.
    energy_map = dict(energies)

    if image_dirs and len(numeric) >= 2:
        e_initial = energy_map.get(numeric[0].name)
        e_final = energy_map.get(numeric[-1].name)
    else:
        e_initial = None
        e_final = None

    if energy_values and e_initial is not None:
        forward = max(energy_values) - e_initial
    else:
        forward = None

    if energy_values and e_final is not None:
        backward = max(energy_values) - e_final
    else:
        backward = None

    has_outputs = bool(energy_values or converged_flags)

    return {
        "project": project_name,
        "case": case_info["case_name"],
        "calculation_type": "tss",
        "job_dir": str(job_dir),
        "status": status_override or terminal_status(job_dir, has_outputs),
        "energy_eV": barrier,
        "neb_barrier_eV": barrier,
        "neb_forward_barrier_eV": forward,
        "neb_backward_barrier_eV": backward,
        "neb_image_energies_eV": "; ".join(f"{name}:{energy:.8f}" for name, energy in energies),
        "converged": all(converged_flags) if converged_flags else False,
        "ionic_steps": ionic_steps or None,
    }


def neb_energy_profile(job_dir, endpoint_energies=None):
    """Energy profile along an NEB/TSS path — the reaction-coordinate ("energy
    stage") plot usually shown in papers.

    ``endpoint_energies`` is an optional ``(initial_eV, final_eV)`` pair used
    for the first/last image directories when they have no energy files of
    their own — a standard NEB run never computes the endpoints, so their
    energies come from the endpoint relaxation runs. Either entry may be None.
    An energy file already inside 00/NN wins over the provided value.

    Returns ``None`` when there are fewer than two images with energies, else a
    dict:
      - images:            image indices (0, 1, …)
      - energies_eV:       absolute energy of each image
      - relative_eV:       energy of each image relative to the initial image
      - reaction_coord:    normalised 0→1 coordinate (cumulative path length in
                           configuration space when structures are available,
                           otherwise evenly spaced)
      - barrier_eV / forward_barrier_eV:  E(TS) − E(initial)
      - backward_barrier_eV:              E(TS) − E(final)
      - delta_e_eV:        reaction energy E(final) − E(initial)
      - ts_image / ts_index:  the highest-energy (transition-state) image
      - n_images
    """
    job_dir = Path(job_dir)
    if not job_dir.exists():
        return None
    image_dirs = _neb_image_dirs(job_dir)
    if len(image_dirs) < 2:
        return None
    numeric = sorted(image_dirs, key=lambda p: int(p.name))

    from vasp_auto.trajectory import _frac_to_cart, _poscar_frame

    fallback = {}
    if endpoint_energies:
        for image_dir, energy in zip((numeric[0], numeric[-1]), endpoint_energies):
            if energy is not None:
                fallback[image_dir.name] = float(energy)

    images, energies, carts = [], [], []
    for image_dir in numeric:
        energy = _neb_image_energy(image_dir)
        if energy is None:
            energy = fallback.get(image_dir.name)
        if energy is None:
            continue
        images.append(int(image_dir.name))
        energies.append(energy)
        poscar = image_dir / "CONTCAR"
        if not poscar.exists() or poscar.stat().st_size == 0:
            poscar = image_dir / "POSCAR"
        if poscar.exists():
            frame = _poscar_frame(poscar)
            cart = frame["cart"] or [_frac_to_cart(c, frame["lattice"]) for c in frame["frac"]]
            carts.append(cart)
        else:
            carts.append(None)

    if len(energies) < 2:
        return None

    e_initial = energies[0]
    relative = [e - e_initial for e in energies]

    # Reaction coordinate = cumulative configuration-space distance between
    # consecutive images, normalised to [0, 1]; fall back to even spacing.
    coord = [0.0]
    have_structs = all(c is not None for c in carts) and len({len(c) for c in carts}) == 1
    if have_structs:
        for a, b in zip(carts, carts[1:]):
            dist = sum((ai[k] - bi[k]) ** 2 for ai, bi in zip(a, b) for k in range(3)) ** 0.5
            coord.append(coord[-1] + dist)
        total = coord[-1]
        coord = ([c / total for c in coord] if total > 0
                 else [i / (len(energies) - 1) for i in range(len(energies))])
    else:
        coord = [i / (len(energies) - 1) for i in range(len(energies))]

    e_max = max(energies)
    ts_index = energies.index(e_max)
    return {
        "images": images,
        "energies_eV": energies,
        "relative_eV": relative,
        "reaction_coord": coord,
        "barrier_eV": e_max - e_initial,
        "forward_barrier_eV": e_max - e_initial,
        "backward_barrier_eV": e_max - energies[-1],
        "delta_e_eV": energies[-1] - e_initial,
        "ts_image": images[ts_index],
        "ts_index": ts_index,
        "n_images": len(energies),
    }


def read_remote_marker(job_dir) -> dict | None:
    """Return the .remote.json tag written when a case was submitted to a remote
    machine (host/name, remote_dir, job_id, scheduler), or None if it ran locally."""
    marker = Path(job_dir) / ".remote.json"
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _apply_remote_marker(row, job_dir):
    """Tag a result row with the remote machine it was submitted to, if any."""
    marker = read_remote_marker(job_dir)
    if not marker:
        return
    row["machine"] = marker.get("machine") or marker.get("host")
    row["remote_dir"] = marker.get("remote_dir")
    if marker.get("job_id"):
        row["job_id"] = marker["job_id"]
    if marker.get("scheduler"):
        row["remote_scheduler"] = marker["scheduler"]
    # A remote run leaves no local OUTCAR until results are fetched.
    if row.get("status") == "missing":
        row["status"] = "remote"


def job_engine(job_dir: Path) -> str:
    """Read the engine marker written when the job was prepared (default vasp).

    A ``.engine`` file holds "qe" for Quantum ESPRESSO jobs; absent it (or a
    bare pw.out from an older job) the engine is VASP.
    """
    job_dir = Path(job_dir)
    marker = job_dir / ".engine"
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip().lower() or "vasp"
    if (job_dir / "pw.out").exists() or (job_dir / "pw.in").exists():
        return "qe"
    return "vasp"


def _build_qe_row(project_name, mode, case_info, status_override=None):
    """Summary row for a Quantum ESPRESSO job (pw.out), mirroring build_row."""
    job_dir = Path(case_info["job_dir"])
    pw_out = job_dir / "pw.out"
    if case_info.get("calculation_type") == "tss":
        from vasp_auto.parser import parse_qe_neb
        profile = parse_qe_neb(job_dir) or {}
        neb_out = job_dir / "neb.out"
        row = {
            "project": project_name, "case": case_info["case_name"], "engine": "qe",
            "calculation_type": "tss", "job_dir": str(job_dir),
            "status": status_override or terminal_status(job_dir, neb_out.exists()),
            "converged": job_finished(job_dir),
            "barrier_forward_eV": profile.get("barrier_forward_eV"),
            "barrier_reverse_eV": profile.get("barrier_reverse_eV"),
            "delta_eV": profile.get("delta_eV"),
        }
        if status_override is None:
            _apply_remote_marker(row, job_dir)
        return row
    summary = parse_pw_output(pw_out) or {}

    row = {
        "project": project_name,
        "case": case_info["case_name"],
        "engine": "qe",
        "calculation_type": case_info.get("calculation_type", "scf"),
        "job_dir": str(job_dir),
        "status": status_override or terminal_status(job_dir, pw_out.exists()),
        "energy_eV": summary.get("energy_eV"),
        "converged": summary.get("converged", False),
        "ionic_steps": summary.get("ionic_steps", 0),
    }
    for key in ("max_force_eV_A", "pressure_kB"):
        if summary.get(key) is not None:
            row[key] = summary[key]
    if status_override is None:
        _apply_remote_marker(row, job_dir)
    return row


def _build_ase_row(project_name, mode, case_info, status_override=None):
    """Summary row for an ASE-engine job (ase_results.json), mirroring build_row."""
    from vasp_auto.ase_engine import parse_ase_output

    job_dir = Path(case_info["job_dir"])
    summary = parse_ase_output(job_dir) or {}
    row = {
        "project": project_name,
        "case": case_info["case_name"],
        "engine": "ase",
        "calculator": summary.get("calculator"),
        "calculation_type": case_info.get("calculation_type", "scf"),
        "job_dir": str(job_dir),
        "status": status_override or terminal_status(job_dir, bool(summary)),
        "energy_eV": summary.get("energy_eV"),
        "converged": summary.get("converged", False),
        "ionic_steps": summary.get("ionic_steps", 0),
    }
    if summary.get("max_force_eV_A") is not None:
        row["max_force_eV_A"] = summary["max_force_eV_A"]
    return row


def build_row(project_name, mode, case_info, status_override=None):
    engine_name = job_engine(Path(case_info["job_dir"]))
    if engine_name == "qe":
        return _build_qe_row(project_name, mode, case_info, status_override=status_override)
    if engine_name == "ase":
        return _build_ase_row(project_name, mode, case_info, status_override=status_override)
    if case_info.get("calculation_type") == "tss":
        return _build_neb_row(project_name, mode, case_info, status_override=status_override)

    job_dir = Path(case_info["job_dir"])
    outcar = job_dir / "OUTCAR"
    oszicar = job_dir / "OSZICAR"

    summary = parse_outcar_summary(outcar)
    ionic_steps = count_ionic_steps(oszicar)

    row = {
        "project": project_name,
        "case": case_info["case_name"],
        "calculation_type": case_info.get("calculation_type", "scf"),
        "job_dir": str(job_dir),
        "status": status_override or terminal_status(job_dir, outcar.exists()),
        "energy_eV": summary["energy_eV"],
        "converged": summary["converged"],
        "ionic_steps": ionic_steps,
    }

    if summary["magmom_total"] is not None:
        row["magmom_total"] = summary["magmom_total"]
        magmoms = parse_magmoms(outcar)
        if magmoms:
            row["magmoms"] = " ".join(f"{m:.3f}" for m in magmoms)

    vasprun = parse_vasprun(job_dir / "vasprun.xml")
    if vasprun:
        row.update({key: vasprun[key] for key in VASPRUN_ROW_KEYS if vasprun.get(key) is not None})

    if status_override is None:
        _apply_remote_marker(row, job_dir)
    return row


def parse_existing_job(project_name, mode, case_info):
    return build_row(project_name, mode, case_info)


def should_retry_failed(case_info):
    job_dir = Path(case_info["job_dir"])
    if case_info.get("calculation_type") == "tss":
        image_dirs = _neb_image_dirs(job_dir) if job_dir.exists() else []
        if not image_dirs:
            return True
        numeric = sorted(image_dirs, key=lambda p: int(p.name))
        endpoints = {numeric[0].name, numeric[-1].name} if len(numeric) >= 2 else {"00"}
        interior = [d for d in image_dirs if d.name not in endpoints]
        outcars = [image_dir / "OUTCAR" for image_dir in interior]
        if not outcars or any(not outcar.exists() for outcar in outcars):
            return True
        return not all(is_converged(outcar) for outcar in outcars)

    engine_name = job_engine(job_dir)
    if engine_name == "qe":
        if case_info.get("calculation_type") == "tss":
            return not job_finished(job_dir)
        summary = parse_pw_output(job_dir / "pw.out")
        if not summary:
            return True
        return not summary.get("converged", False)

    if engine_name == "ase":
        from vasp_auto.ase_engine import parse_ase_output
        summary = parse_ase_output(job_dir)
        if not summary:
            return True
        return not summary.get("converged", False)

    outcar = job_dir / "OUTCAR"
    if not outcar.exists():
        return True
    return not is_converged(outcar)


def run_one_case(
    project_name,
    mode,
    case_info,
    vasp_executable,
    cpus=None,
    scheduler="local",
    job_template=None,
    scheduler_options=None,
    on_progress=None,
    auto_retry=0,
    remote=None,
    engine="vasp",
    qe_executable=None,
    ase_python=None,
):
    job_dir = Path(case_info["job_dir"])
    executable = qe_executable if engine == "qe" else vasp_executable

    # Remote execution: ship every input file to another machine and run it there.
    if remote:
        if engine == "ase":
            raise ValueError(
                "Remote execution supports VASP and QE, not --engine ase."
            )

        # Direct SSH: run mpirun on the remote synchronously and pull results back
        # (for single-workstation machines with no working scheduler).
        if resolve_remote_run_mode(remote) == "ssh":
            _write_pid(job_dir)  # liveness marker while this wrapper drives the remote run
            try:
                return_code = run_vasp_remote(
                    str(job_dir), remote, cpus=cpus, on_progress=on_progress, engine=engine
                )
            finally:
                _clear_pid(job_dir)
            error_summary = None
            if engine == "vasp":
                # VASP error signatures / job.log don't apply to pw.x output.
                error_summary = report_vasp_errors(job_dir)
                # Write job.log before build_row so the row reads "done", not "running".
                from vasp_auto.job_log import write_job_log
                write_job_log(job_dir, case_info.get("case_name"),
                              case_info.get("calculation_type"), return_code)
            row = build_row(project_name, mode, case_info)
            row["return_code"] = return_code
            if error_summary:
                row["errors"] = error_summary
            machine = remote.get("name") or remote.get("host")
            print(f"Finished  : {case_info['case_name']} on {machine} "
                  f"({remote['remote_root'].rstrip('/')}/{job_dir.name})")
            return row

        # Scheduler submission: queue on the remote and exit, so the local host
        # can be turned off afterwards.
        submission = submit_job_remote(
            str(job_dir),
            remote,
            cpus=cpus,
            job_template=job_template,
            engine=engine,
        )
        row = build_row(project_name, mode, case_info, status_override="submitted")
        row["job_id"] = submission["job_id"]
        row["remote_dir"] = submission["remote_dir"]
        print(
            f"Submitted : {case_info['case_name']} -> {submission['host']} "
            f"{submission['scheduler']} job {submission['job_id']} "
            f"({submission['remote_dir']})"
        )
        return row

    if scheduler and scheduler != "local":
        if engine == "ase":
            raise ValueError(
                "Scheduler submission is not available for --engine ase yet; run it "
                "locally (scheduler: local)."
            )
        submission = submit_job(
            str(job_dir),
            str(executable),
            cpus=cpus,
            scheduler=scheduler,
            template_path=job_template,
            options=scheduler_options,
            engine=engine,
        )
        row = build_row(project_name, mode, case_info, status_override="submitted")
        row["job_id"] = submission["job_id"]
        print(f"Submitted : {case_info['case_name']} -> {scheduler} job {submission['job_id']}")
        return row

    # Local execution: record a .pid liveness marker for the running-job scans,
    # cleared as soon as the process returns (before build_row, since .pid holds
    # this still-alive wrapper's own PID).
    if engine == "qe":
        # Quantum ESPRESSO: run pw.x; VASP error-signature auto-retry does not
        # apply, so the row is built straight from pw.out.
        _write_pid(job_dir)
        try:
            return_code = run_qe(str(job_dir), str(executable), cpus=cpus, on_progress=on_progress)
        finally:
            _clear_pid(job_dir)
        row = build_row(project_name, mode, case_info)
        row["return_code"] = return_code
        return row

    if engine == "ase":
        # Generic ASE calculator: run the run_ase.py driver; results come from
        # ase_results.json. VASP error-signature auto-retry does not apply.
        _write_pid(job_dir)
        try:
            return_code = run_ase(str(job_dir), python_exe=ase_python, cpus=cpus,
                                  on_progress=on_progress)
        finally:
            _clear_pid(job_dir)
        row = build_row(project_name, mode, case_info)
        row["return_code"] = return_code
        return row

    _write_pid(job_dir)
    try:
        return_code = run_vasp(str(job_dir), str(vasp_executable), cpus=cpus, on_progress=on_progress)
        findings = scan_vasp_errors(job_dir)
        retries = 0
        all_fixes = []

        while findings and retries < auto_retry:
            fixes = apply_error_fixes(job_dir, findings)
            if not fixes:
                break  # nothing safe to change automatically for these errors
            retries += 1
            all_fixes.extend(fixes)
            print(
                f"Auto-retry {retries}/{auto_retry}: {case_info['case_name']} — "
                f"applied {', '.join(fixes)}"
            )
            return_code = run_vasp(str(job_dir), str(vasp_executable), cpus=cpus, on_progress=on_progress)
            findings = scan_vasp_errors(job_dir)
    finally:
        _clear_pid(job_dir)

    error_summary = report_vasp_errors(job_dir)
    # Drop a human-readable job.log summary next to the raw VASP output — before
    # build_row so the returned row reads "done" rather than "running".
    from vasp_auto.job_log import write_job_log
    write_job_log(job_dir, case_info.get("case_name"),
                  case_info.get("calculation_type"), return_code)
    row = build_row(project_name, mode, case_info)
    row["return_code"] = return_code
    if error_summary:
        row["errors"] = error_summary
    if retries:
        row["auto_retries"] = retries
        row["auto_fixes"] = "; ".join(all_fixes)
    return row


# Sequence prefix on a numbered job folder, e.g. "0004_Fe" -> case name "Fe".
_JOB_NUMBER_PREFIX = re.compile(r"^\d+_")

# VASP prints its rank count near the top of OUTCAR:
# v5 "running on    8 total cores", v6 "running   8 mpi-ranks, ...".
_RUN_RANKS = re.compile(r"running\s+(?:on\s+)?(\d+)\s+(?:total cores|mpi-ranks|cores)")


def _previous_run_ranks(job_dir: Path) -> int | None:
    """MPI rank count the previous run in ``job_dir`` used, read from the header
    VASP prints in OUTCAR (for NEB, the first image's OUTCAR). None if unknown."""
    for name in ("OUTCAR", "01/OUTCAR"):
        path = job_dir / name
        if not path.is_file():
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                head = fh.read(8192)
        except OSError:
            continue
        match = _RUN_RANKS.search(head)
        if match:
            return int(match.group(1))
    return None


def _ranks_for_images(job_dir: Path, cpus: int | None) -> int | None:
    """Clamp an MPI rank count to what a NEB job can actually use.

    VASP requires the rank count to be divisible by IMAGES (INCAR); anything
    else aborts immediately in M_divide / MPI_Cart_sub. Rounds ``cpus`` down to
    a multiple of IMAGES (minimum one rank per image). INCARs without an IMAGES
    tag return ``cpus`` unchanged.
    """
    try:
        text = (job_dir / "INCAR").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return cpus
    match = re.search(r"^[ \t]*IMAGES[ \t]*=[ \t]*(\d+)", text, re.M | re.I)
    if not match:
        return cpus
    images = int(match.group(1))
    if images <= 0:
        return cpus
    if not cpus or cpus < images:
        return images
    return cpus - cpus % images


def _seed_poscar_from_contcar(work_dir: Path) -> bool:
    """Advance POSCAR to the latest geometry VASP wrote (CONTCAR), in place.

    Copies a non-empty ``CONTCAR`` onto ``POSCAR`` (keeping the previous POSCAR as
    ``POSCAR.bak``) and returns True. Returns False when no usable CONTCAR exists —
    the run never started, so the existing POSCAR is already the newest geometry
    and is left untouched.
    """
    contcar = work_dir / "CONTCAR"
    poscar = work_dir / "POSCAR"
    if contcar.exists() and contcar.stat().st_size > 0:
        if poscar.exists():
            shutil.copy2(poscar, work_dir / "POSCAR.bak")
        shutil.copy2(contcar, poscar)
        return True
    return False


def resume_job(
    job_dir,
    vasp_executable,
    cpus=None,
    on_progress=None,
    project_name="",
    case_name=None,
    calculation_type=None,
    force=False,
):
    """Resume an unfinished VASP job *in place*, restarting from its newest CONTCAR.

    This never allocates a new ``NNNN_`` job number: it runs in ``job_dir`` itself
    (e.g. ``jobs/Fe/0004_Fe``) and reuses the INCAR, KPOINTS and POTCAR already
    sitting there — they are taken verbatim, never rebuilt from the original case
    directory. Only the geometry advances: the latest non-empty CONTCAR becomes the
    new POSCAR (the previous POSCAR is kept as ``POSCAR.bak``). Any WAVECAR/CHGCAR
    present are left untouched so VASP can warm-start from them. Results are written
    back into the same directory.

    NEB/TSS jobs (numeric image subdirs) are resumed per image: each image's
    CONTCAR seeds its POSCAR, and the shared INCAR/KPOINTS/POTCAR at the top level
    are reused.

    Unless ``force`` is set, a job that already converged is left alone (a row is
    returned without re-running). ``calculation_type`` is auto-detected from the
    directory layout when not given. Returns a summary row, mirroring
    :func:`run_one_case`.
    """
    job_dir = Path(job_dir).resolve()
    if not job_dir.is_dir():
        raise FileNotFoundError(f"job directory not found: {job_dir}")

    engine_name = job_engine(job_dir)
    if engine_name != "vasp":
        raise ValueError(
            f"resume_job supports VASP jobs only; {job_dir.name} looks like a "
            f"'{engine_name}' job. Re-run it from the case instead."
        )

    if case_name is None:
        case_name = _JOB_NUMBER_PREFIX.sub("", job_dir.name) or job_dir.name

    # NEB/TSS jobs keep one CONTCAR per numeric image subdir; a flat job keeps a
    # single CONTCAR at the top level. Detect the layout when not told.
    image_dirs = _neb_image_dirs(job_dir)
    is_neb = (calculation_type == "tss") or (
        calculation_type is None
        and len(image_dirs) >= 2
        and all((d / "POSCAR").exists() or (d / "CONTCAR").exists() for d in image_dirs)
    )
    calculation_type = "tss" if is_neb else (calculation_type or "scf")

    case_info = {
        "case_name": case_name,
        "job_dir": job_dir,
        "calculation_type": calculation_type,
        "project": project_name,
    }

    # Only resume jobs that actually need it, unless explicitly forced.
    if not force and not should_retry_failed(case_info):
        print(f"Skip      : {case_name} already finished ({job_dir.name})")
        return build_row(project_name, "single", case_info)

    # The inputs must already live in the job dir — we reuse them as-is and never
    # fall back to the original case directory.
    missing = [name for name in ("INCAR", "KPOINTS", "POTCAR") if not (job_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"cannot resume {job_dir.name}: missing {', '.join(missing)} in the job directory"
        )

    if is_neb:
        seeded = sum(_seed_poscar_from_contcar(d) for d in image_dirs)
        print(
            f"Resume    : {case_name} ({job_dir.name}) — kept INCAR/KPOINTS/POTCAR; "
            f"advanced {seeded}/{len(image_dirs)} image POSCAR(s) from CONTCAR"
        )
    else:
        if not (job_dir / "POSCAR").exists() and not (job_dir / "CONTCAR").exists():
            raise FileNotFoundError(
                f"cannot resume {job_dir.name}: no POSCAR or CONTCAR in the job directory"
            )
        seeded = _seed_poscar_from_contcar(job_dir)
        source = "CONTCAR" if seeded else "existing POSCAR (no CONTCAR yet)"
        print(f"Resume    : {case_name} ({job_dir.name}) — kept INCAR/KPOINTS/POTCAR; restart from {source}")

    if cpus is None:
        cpus = _previous_run_ranks(job_dir)
        if cpus:
            print(f"CPUs      : {cpus} (same as the previous run, from OUTCAR)")
    adjusted = _ranks_for_images(job_dir, cpus)
    if adjusted != cpus:
        print(f"CPUs      : {adjusted} (rounded to a multiple of IMAGES for NEB)")
        cpus = adjusted

    # Drop the stale summary: while the resume runs the job must not read
    # "finished", and the rebuilt job.log below reflects the current (possibly
    # edited) INCAR instead of the previous run's parameters.
    (job_dir / "job.log").unlink(missing_ok=True)

    return_code = run_vasp(str(job_dir), str(vasp_executable), cpus=cpus, on_progress=on_progress)
    error_summary = report_vasp_errors(job_dir)
    row = build_row(project_name, "single", case_info)
    row["return_code"] = return_code
    row["resumed"] = True
    if error_summary:
        row["errors"] = error_summary

    from vasp_auto.job_log import write_job_log
    write_job_log(job_dir, case_name, calculation_type, return_code)
    return row
