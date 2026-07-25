from __future__ import annotations

import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Commands used to poll job status for each scheduler.
# squeue: query by job ID (-h = no header, -j = job list).
# qstat: query by job ID (prints to stdout, may error when job is done).
POLL_COMMANDS: dict[str, list[str]] = {
    "slurm": ["squeue", "-h", "-j"],
    "pbs": ["qstat"],
}


DEFAULT_SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks={cpus}
#SBATCH --output=run.log
{extra}
{preamble}
cd "{job_dir}"
mpirun -np {cpus} "{exe}"
"""

DEFAULT_PBS_TEMPLATE = """#!/bin/bash
#PBS -N {job_name}
#PBS -l nodes=1:ppn={cpus}
#PBS -o run.log
#PBS -j oe
{extra}
{preamble}
cd "{job_dir}"
mpirun -np {cpus} "{exe}"
"""

DEFAULT_SLURM_QE_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks={cpus}
#SBATCH --output=run.log
{extra}
{preamble}
cd "{job_dir}"
{qe_commands}
"""

DEFAULT_PBS_QE_TEMPLATE = """#!/bin/bash
#PBS -N {job_name}
#PBS -l nodes=1:ppn={cpus}
#PBS -o run.log
#PBS -j oe
{extra}
{preamble}
cd "{job_dir}"
{qe_commands}
"""

SCHEDULER_COMMANDS = {"slurm": "sbatch", "pbs": "qsub"}


def _resolve_executable(vasp_executable: str) -> str:
    exe_path = Path(vasp_executable)
    if exe_path.parent == Path("."):
        resolved = shutil.which(str(exe_path))
        if resolved is None:
            raise FileNotFoundError(
                f"VASP executable not found: {vasp_executable}. "
                "Set vasp_executable in config.yaml or export VASP_EXECUTABLE."
            )
        return resolved
    if exe_path.exists():
        return str(exe_path)
    raise FileNotFoundError(f"VASP executable not found: {vasp_executable}")


def _local_core_counts() -> tuple[int, int]:
    """Return ``(physical_cores, logical_cpus)`` for this machine.

    Open MPI counts *physical cores* as slots by default, so ``-np N`` aborts
    with "not enough slots" once N exceeds the physical-core count — even when
    idle hardware threads (hyperthreads) remain. ``os.cpu_count()`` reports the
    logical count, so we read ``/proc/cpuinfo`` for the physical count (Linux),
    falling back to the logical count when that is unavailable.
    """
    logical = os.cpu_count() or 1
    physical = logical
    try:
        pairs: set[tuple[str | None, str | None]] = set()
        phys_id = core_id = None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("physical id"):
                phys_id = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":", 1)[1].strip()
                pairs.add((phys_id, core_id))
            elif not line.strip():
                phys_id = core_id = None
        if pairs:
            physical = len(pairs)
    except OSError:
        pass
    return physical, logical


def _mpi_command(launcher: str, mpi_ranks: int, exe: str, *exe_args: str):
    """Build the mpirun command, picking the right flag when ranks exceed slots.

    Open MPI aborts when ``-np N`` is larger than the number of slots (physical
    cores by default). We honour the requested rank count so the job still runs:

    * ranks within the hardware-thread count -> ``--use-hwthread-cpus`` (treats
      hyperthreads as slots; no oversubscription),
    * ranks beyond even the hardware threads -> ``--oversubscribe``.

    ``VASP_AUTO_OVERSUBSCRIBE`` overrides the choice: ``1`` always adds
    ``--oversubscribe``; ``0`` adds nothing (let Open MPI enforce its slot
    limit). The flags are OpenMPI-specific, so they are skipped for other
    launchers (e.g. ``mpiexec`` for Intel/MS-MPI, which have no slot concept).

    Returns ``(cmd, warning_or_None)``.
    """
    physical, logical = _local_core_counts()
    pref = os.environ.get("VASP_AUTO_OVERSUBSCRIBE")
    is_mpirun = Path(launcher).name == "mpirun"

    cmd = [launcher, "-np", str(mpi_ranks)]
    warning = None

    if pref == "1":
        if is_mpirun:
            cmd.append("--oversubscribe")
    elif pref == "0":
        if mpi_ranks > physical:
            warning = (
                f"[vasp_auto] requested {mpi_ranks} MPI ranks but this machine "
                f"has {physical} physical core(s); the run will fail unless you "
                f"lower CPU cores or unset VASP_AUTO_OVERSUBSCRIBE."
            )
    elif mpi_ranks > logical:
        if is_mpirun:
            cmd.append("--oversubscribe")
        warning = (
            f"[vasp_auto] requested {mpi_ranks} MPI ranks but this machine has "
            f"{logical} hardware thread(s); running with --oversubscribe "
            f"(slower — set CPU cores <= {physical} for best performance)."
        )
    elif mpi_ranks > physical:
        if is_mpirun:
            cmd.append("--use-hwthread-cpus")
        warning = (
            f"[vasp_auto] requested {mpi_ranks} MPI ranks but this machine has "
            f"{physical} physical core(s); using hardware threads "
            f"(--use-hwthread-cpus). Set CPU cores <= {physical} to avoid this."
        )

    cmd += [exe, *exe_args]
    return cmd, warning


def run_vasp(job_dir: str, vasp_executable: str, cpus: int | None = None, on_progress=None):
    """Run VASP locally via mpirun. on_progress(line) streams output lines."""
    job_dir = Path(job_dir)
    log_file = job_dir / "run.log"
    exe = _resolve_executable(vasp_executable)

    env = os.environ.copy()

    # MPI 版 VASP：cpus 當作 MPI ranks，用 mpirun 啟動
    mpi_ranks = cpus if cpus is not None else 1

    # 避免 BLAS / OMP 自己再亂開 threads
    env["OMP_NUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    # mpirun by default; VASP_AUTO_MPI=mpiexec covers MS-MPI / Intel MPI hosts.
    mpi_launcher = os.environ.get("VASP_AUTO_MPI", "mpirun")
    cmd, warning = _mpi_command(mpi_launcher, mpi_ranks, exe)

    with open(log_file, "w", encoding="utf-8") as log:
        if warning:
            log.write(warning + "\n")
            log.flush()  # land before the subprocess writes to the same fd
            if on_progress is not None:
                on_progress(warning)
        if on_progress is None:
            result = subprocess.run(
                cmd,
                cwd=job_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return result.returncode

        process = subprocess.Popen(
            cmd,
            cwd=job_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            log.write(line)
            on_progress(line.rstrip("\n"))
        return process.wait()


def run_qe(job_dir: str, qe_executable: str, cpus: int | None = None, on_progress=None):
    """Run Quantum ESPRESSO pw.x locally via mpirun.

    pw.x reads ``pw.in`` and writes its results to stdout, which we capture as
    ``pw.out`` (the file the QE parser reads). Output is also streamed to
    ``run.log`` so the progress callback and the UI log viewer behave exactly as
    they do for VASP. on_progress(line) streams output lines.
    """
    job_dir = Path(job_dir)
    log_file = job_dir / "run.log"
    exe = _resolve_executable(qe_executable)

    env = os.environ.copy()
    mpi_ranks = cpus if cpus is not None else 1
    env["OMP_NUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    mpi_launcher = os.environ.get("VASP_AUTO_MPI", "mpirun")
    stages = _load_qe_stages(job_dir)
    completion_marker = job_dir / ".qe_complete"
    completion_marker.unlink(missing_ok=True)

    with open(log_file, "w", encoding="utf-8") as log:
        for stage in stages:
            program = _resolve_qe_program(exe, stage["program"])
            if stage.get("mpi", True):
                cmd, warning = _mpi_command(
                    mpi_launcher, mpi_ranks, program, "-in", stage["input"]
                )
            else:
                cmd, warning = [program], None
            if warning:
                log.write(warning + "\n")
                if on_progress is not None:
                    on_progress(warning)
            label = f"[qe] {stage['program']} < {stage['input']}"
            log.write(label + "\n")
            if on_progress is not None:
                on_progress(label)
            output_path = job_dir / stage["output"]
            with open(job_dir / stage["input"], "r", encoding="utf-8") as inp, open(
                output_path, "w", encoding="utf-8"
            ) as out:
                process = subprocess.Popen(
                    cmd, cwd=job_dir, env=env, stdin=inp,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                for line in process.stdout:
                    out.write(line)
                    log.write(line)
                    if on_progress is not None:
                        on_progress(line.rstrip("\n"))
                return_code = process.wait()
            if return_code != 0:
                return return_code
            for src, dst in (stage.get("rename") or {}).items():
                source = job_dir / src
                if source.exists():
                    source.replace(job_dir / dst)
        completion_marker.write_text("complete\n", encoding="utf-8")
        return 0


def _load_qe_stages(job_dir: Path) -> list[dict]:
    """Read the QE stage manifest, with the legacy single-pw.x fallback."""
    manifest = Path(job_dir) / "qe_stages.json"
    if not manifest.exists():
        return [{"program": "pw.x", "input": "pw.in", "output": "pw.out", "mpi": True}]
    data = json.loads(manifest.read_text(encoding="utf-8"))
    stages = data.get("stages") if isinstance(data, dict) else None
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"invalid or empty QE stage manifest: {manifest}")
    for stage in stages:
        if not isinstance(stage, dict) or not all(stage.get(k) for k in ("program", "input", "output")):
            raise ValueError(f"invalid QE stage in {manifest}: {stage!r}")
        if not (Path(job_dir) / stage["input"]).exists():
            raise FileNotFoundError(f"QE stage input is missing: {Path(job_dir) / stage['input']}")
    return stages


def _resolve_qe_program(pw_executable: str, program: str) -> str:
    """Resolve a QE companion beside pw.x, or from PATH for a bare pw.x."""
    if program == "pw.x":
        return pw_executable
    pw_path = Path(pw_executable)
    candidate = pw_path.with_name(program)
    if pw_path.parent != Path(".") and candidate.exists():
        return str(candidate)
    resolved = shutil.which(program)
    if resolved:
        return resolved
    raise FileNotFoundError(
        f"Quantum ESPRESSO companion executable not found: {program}. "
        f"Install the full QE suite beside {pw_executable}, or put {program} on PATH."
    )


def _qe_shell_commands(job_dir: Path, qe_executable: str, ranks: int,
                       quote_program: bool = True) -> list[str]:
    """Shell commands for the same manifest used by local QE execution."""
    commands = ["rm -f .qe_complete", ": > run.log"]
    qe_dir = str(Path(qe_executable).parent)
    stages = _load_qe_stages(job_dir)
    for stage in stages:
        program = qe_executable if stage["program"] == "pw.x" else str(Path(qe_dir) / stage["program"])
        quoted_program = ('"' + program.replace('"', '\\"') + '"'
                          if quote_program else shlex.quote(program))
        inp = shlex.quote(stage["input"])
        out = shlex.quote(stage["output"])
        if stage.get("mpi", True):
            launch = f"mpirun -np {ranks} {quoted_program} -in {inp}"
        else:
            launch = f"{quoted_program} < {inp}"
        commands.append(
            f"echo {shlex.quote('[qe] ' + stage['program'] + ' < ' + stage['input'])} >> run.log"
        )
        commands.append(f"{launch} > {out} 2>&1 || {{ rc=$?; cat {out} >> run.log; exit $rc; }}")
        commands.append(f"cat {out} >> run.log")
        for src, dst in (stage.get("rename") or {}).items():
            commands.append(f"test ! -f {shlex.quote(src)} || mv -f {shlex.quote(src)} {shlex.quote(dst)}")
    if len(stages) == 1 and stages[0]["output"] == "pw.out":
        commands.append("cp -f pw.out run.log")
    commands.append("printf 'complete\\n' > .qe_complete")
    return commands


def run_ase(job_dir: str, python_exe: str | None = None, cpus: int | None = None,
            on_progress=None):
    """Run the ASE engine driver (run_ase.py) locally as a subprocess.

    The driver reads POSCAR + ase_calc.json, runs the chosen ASE calculator, and
    writes ase_results.json (the parse contract) + CONTCAR. Output is streamed to
    run.log so the progress callback and UI log viewer behave as for VASP/QE.
    ``cpus`` is exported as OMP_NUM_THREADS for threaded calculators (EMT and the
    like ignore it). on_progress(line) streams output lines.
    """
    job_dir = Path(job_dir)
    log_file = job_dir / "run.log"
    driver = job_dir / "run_ase.py"
    if not driver.exists():
        raise FileNotFoundError(f"no run_ase.py in {job_dir}; prepare the ASE job first")
    python = python_exe or sys.executable or "python3"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(cpus if cpus is not None else 1)

    with open(log_file, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [python, "run_ase.py"], cwd=job_dir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in process.stdout:
            log.write(line)
            if on_progress is not None:
                on_progress(line.rstrip("\n"))
        return process.wait()


def write_submit_script(
    job_dir: str,
    vasp_executable: str,
    cpus: int | None = None,
    scheduler: str = "slurm",
    template_path: str | None = None,
    options: list[str] | None = None,
    run_dir: str | None = None,
    engine: str = "vasp",
    preamble: list[str] | None = None,
) -> Path:
    """Write a scheduler submit script into the job directory.

    The executable path is embedded as given (compute nodes may resolve paths
    differently from the launch host). ``run_dir`` overrides the directory the
    script ``cd``s into at run time — set it to the *remote* job path when the
    script will execute on another machine; it defaults to the local job_dir.
    """
    if scheduler not in SCHEDULER_COMMANDS:
        raise ValueError(f"Unknown scheduler: {scheduler} (use slurm or pbs)")

    job_dir = Path(job_dir).resolve()
    if template_path:
        template = Path(template_path).read_text(encoding="utf-8")
    elif engine == "qe":
        template = DEFAULT_SLURM_QE_TEMPLATE if scheduler == "slurm" else DEFAULT_PBS_QE_TEMPLATE
    else:
        template = DEFAULT_SLURM_TEMPLATE if scheduler == "slurm" else DEFAULT_PBS_TEMPLATE

    qe_commands = "\n".join(_qe_shell_commands(job_dir, vasp_executable, cpus or 1)) \
        if engine == "qe" else ""
    has_preamble_placeholder = "{preamble}" in template
    preamble_text = "\n".join(preamble or [])
    script_text = template.format(
        job_name=job_dir.name,
        cpus=cpus if cpus is not None else 1,
        exe=vasp_executable,
        job_dir=run_dir if run_dir is not None else job_dir,
        extra="\n".join(options or []),
        preamble=preamble_text,
        qe_commands=qe_commands,
    )
    if preamble_text and not has_preamble_placeholder:
        # Existing custom templates predate the {preamble} placeholder. Insert
        # the gate after their shebang/header directives so the concurrency
        # rule still applies without invalidating #SBATCH/#PBS directives.
        lines = script_text.splitlines()
        insert_at = 1 if lines and lines[0].startswith("#!") else 0
        while insert_at < len(lines) and (
            not lines[insert_at].strip() or lines[insert_at].lstrip().startswith("#")
        ):
            insert_at += 1
        lines[insert_at:insert_at] = preamble_text.splitlines()
        script_text = "\n".join(lines) + ("\n" if script_text.endswith("\n") else "")

    script_path = job_dir / "submit.sh"
    script_path.write_text(script_text, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def submit_job(
    job_dir: str,
    vasp_executable: str,
    cpus: int | None = None,
    scheduler: str = "slurm",
    template_path: str | None = None,
    options: list[str] | None = None,
    engine: str = "vasp",
) -> dict:
    """Submit a job via sbatch/qsub; returns {"job_id", "script", "submit_output"}."""
    script_path = write_submit_script(
        job_dir, vasp_executable, cpus=cpus, scheduler=scheduler,
        template_path=template_path, options=options, engine=engine,
    )

    command = SCHEDULER_COMMANDS[scheduler]
    result = subprocess.run(
        [command, str(script_path)],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{command} failed: {result.stderr.strip() or result.stdout.strip()}")

    output = result.stdout.strip()
    # sbatch: "Submitted batch job 12345"; qsub: "12345.hostname"
    job_id = output.split()[-1] if scheduler == "slurm" else output.splitlines()[0].strip()

    return {"job_id": job_id, "script": str(script_path), "submit_output": output}


# ---------------------------------------------------------------- remote submit

def _ssh_target(remote: dict) -> str:
    """Build the ``user@host`` (or bare ``host``) destination from a remote config."""
    host = remote.get("host")
    if not host:
        raise ValueError("remote config needs a 'host' (the machine to submit to)")
    user = remote.get("user")
    return f"{user}@{host}" if user else host


def _ssh_options(remote: dict) -> list[str]:
    """ssh-style option flags (-p PORT, -i KEY, plus any extra ssh_options)."""
    opts: list[str] = []
    port = remote.get("port")
    if port:
        opts += ["-p", str(port)]
    key = remote.get("ssh_key")
    if key:
        opts += ["-i", str(Path(key).expanduser())]
    opts += list(remote.get("ssh_options") or [])
    return opts


def _run_checked(cmd: list[str], what: str) -> str:
    """Run a command, raising RuntimeError with stderr context on failure."""
    # Decode as UTF-8 (not the host locale) so non-ASCII paths/errors from the
    # remote survive; errors="replace" keeps a stray byte from crashing us.
    result = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{what} failed: {detail}")
    return result.stdout


def _transfer_dir(local_dir: Path, target: str, remote_dir: str, remote: dict) -> None:
    """Copy a whole job directory to the remote host (rsync if present, else scp)."""
    ssh_opts = _ssh_options(remote)
    if shutil.which("rsync"):
        cmd = ["rsync", "-az"]
        if ssh_opts:
            cmd += ["-e", "ssh " + " ".join(shlex.quote(o) for o in ssh_opts)]
        # trailing slash on source copies the contents into remote_dir
        cmd += [f"{local_dir}/", f"{target}:{remote_dir}/"]
        _run_checked(cmd, "rsync")
        return

    # scp fallback: it uses -P (uppercase) for the port, unlike ssh's -p.
    scp_opts: list[str] = []
    port = remote.get("port")
    if port:
        scp_opts += ["-P", str(port)]
    key = remote.get("ssh_key")
    if key:
        scp_opts += ["-i", str(Path(key).expanduser())]
    scp_opts += list(remote.get("ssh_options") or [])
    cmd = ["scp", "-r", *scp_opts, f"{local_dir}/.", f"{target}:{remote_dir}/"]
    _run_checked(cmd, "scp")


# Run modes that mean "execute directly over SSH" rather than submit to a queue.
SSH_RUN_MODES = {"ssh", "direct", "none", "local", ""}


def remote_run_mode(remote: dict) -> str:
    """How a remote machine runs jobs.

    Returns ``"ssh"`` for direct ``mpirun`` over SSH (no scheduler), or the
    scheduler name (``"slurm"``/``"pbs"``) for queue submission. An explicit
    ``run_mode`` wins; otherwise the ``scheduler`` field decides, with the
    pseudo-schedulers ``ssh``/``direct``/``none``/``local`` meaning direct SSH.
    """
    mode = (remote.get("run_mode") or "").strip().lower()
    if mode in {"ssh_detached", "detached", "offload"}:
        return "ssh_detached"
    if mode in {"ssh", "direct"}:
        return "ssh"
    if mode in SCHEDULER_COMMANDS:
        return mode
    scheduler = (remote.get("scheduler") or "slurm").strip().lower()
    if scheduler in SSH_RUN_MODES:
        return "ssh"
    return scheduler


def detect_remote_run_mode(remote: dict) -> str:
    """Probe a remote once for a batch scheduler and pick a run mode.

    ``sbatch`` present -> ``"slurm"``, ``qsub`` -> ``"pbs"``, neither (or the probe
    fails / times out) -> ``"ssh_detached"`` (offload): the safe default that works
    on any plain workstation and self-installs its engine on first run.
    """
    target = _ssh_target(remote)
    ssh_opts = _ssh_options(remote)
    cmd = ("command -v sbatch >/dev/null 2>&1 && echo slurm || "
           "{ command -v qsub >/dev/null 2>&1 && echo pbs || echo none; }")
    try:
        res = subprocess.run(["ssh", "-x", *ssh_opts, target, cmd],
                             capture_output=True, encoding="utf-8", errors="replace", timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        return "ssh_detached"
    tag = (res.stdout.strip().splitlines() or [""])[-1].strip()
    return tag if tag in SCHEDULER_COMMANDS else "ssh_detached"


def resolve_remote_run_mode(remote: dict) -> str:
    """Run mode for dispatch, auto-detecting when the machine pins nothing.

    An explicit ``run_mode`` or ``scheduler`` always wins (see
    :func:`remote_run_mode`), so a cluster user can pin ``slurm`` and a workstation
    user ``ssh``. With neither set, probe the machine (:func:`detect_remote_run_mode`)
    and cache the answer on ``remote`` so every dispatch site in one run agrees
    without re-SSHing. ponytail: per-run cache on the dict; fine because a machine's
    scheduler does not change mid-run.
    """
    if (remote.get("run_mode") or "").strip() or (remote.get("scheduler") or "").strip():
        return remote_run_mode(remote)
    cached = remote.get("_run_mode_detected")
    if not cached:
        cached = detect_remote_run_mode(remote)
        remote["_run_mode_detected"] = cached
    return cached


def _remote_vasp_exe(remote: dict) -> str:
    """The VASP binary path on the remote machine.

    If ``vasp_executable`` points at a directory (e.g. ``.../bin``) rather than a
    binary, fall back to ``<dir>/vasp_std`` so a common misconfiguration still
    works.
    """
    exe = remote.get("vasp_executable")
    if not exe:
        raise ValueError(
            "remote config needs a 'vasp_executable' (the VASP path on the remote machine)"
        )
    exe = exe.rstrip("/")
    base = exe.rsplit("/", 1)[-1].lower()
    if "vasp" not in base and "pw" not in base:
        exe = exe + "/vasp_std"
    return exe


def _remote_exe(remote: dict, engine: str = "vasp") -> str:
    """The DFT binary path on the remote machine for the given engine."""
    if engine == "qe":
        # QE is usually a distro package on PATH; a remote 'qe_executable' overrides.
        return remote.get("qe_executable") or "pw.x"
    return _remote_vasp_exe(remote)


def _remote_env_setup(remote: dict, engine: str = "vasp") -> str:
    """The shell snippet sourced before a remote run, per engine.

    ``env_setup`` is the machine's VASP toolchain (MKL/Intel MPI paths) and is
    never sourced for QE — an Intel ``mpirun`` first on PATH silently runs an
    OpenMPI-linked pw.x as N duplicate serial jobs. A machine whose QE build
    does need environment gets its own ``qe_env_setup``.
    """
    key = "qe_env_setup" if engine == "qe" else "env_setup"
    return (remote.get(key) or "").strip()


def remote_results_base(remote: dict) -> str:
    """Canonical results directory on a remote machine: ``<remote_root>/results``.

    Every run directory — synchronous mpirun, scheduler submit, and detached
    offload — lives here, so remote output is never mixed in with the
    ``<remote_root>/inputs`` cases and never doubles up when ``remote_root``
    itself already ends in e.g. ``jobs``.
    """
    root = (remote.get("remote_root") or "").rstrip("/")
    if not root:
        raise ValueError(
            "remote config needs a 'remote_root' (base directory on the remote machine)"
        )
    return f"{root}/results"


def remote_concurrency_limit(remote: dict) -> int | None:
    """Configured maximum number of simultaneous jobs on one remote machine.

    A missing value means unlimited, preserving the behaviour of existing
    remote configurations. Invalid values are also treated as unset; the UI
    normalizes saved values, while hand-written config files remain forgiving.
    """
    value = remote.get("max_jobs")
    if value in (None, ""):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _remote_slot_lines(remote: dict, state_file: str) -> list[str]:
    """Bash prologue that waits until one of the remote's job slots is free.

    ``flock`` locks are owned by the remote process and released by the kernel
    even if a job crashes or is killed.  This makes the limit independent of
    the submitting UI process and safe across separate CLI submissions.
    """
    limit = remote_concurrency_limit(remote)
    if limit is None:
        return []
    root = (remote.get("remote_root") or "").rstrip("/")
    slot_root = f"{root}/{ENGINE_SUBDIR}/job_slots"
    return [
        f"vasp_auto_slot_root={shlex.quote(slot_root)}",
        f"vasp_auto_state_file={shlex.quote(state_file)}",
        f"vasp_auto_slot_limit={limit}",
        'mkdir -p "$vasp_auto_slot_root"',
        'if ! command -v flock >/dev/null 2>&1; then '
        'printf "%s\\n" "vasp_auto: max jobs requires flock on the remote" >&2; exit 69; fi',
        'printf "%s\\n" pending > "$vasp_auto_state_file"',
        'while :; do',
        '  vasp_auto_slot_i=1',
        '  while [ "$vasp_auto_slot_i" -le "$vasp_auto_slot_limit" ]; do',
        '    exec 199>"$vasp_auto_slot_root/slot_${vasp_auto_slot_i}.lock"',
        '    if flock -n 199; then vasp_auto_slot_acquired=1; break 2; fi',
        '    exec 199>&-',
        '    vasp_auto_slot_i=$((vasp_auto_slot_i + 1))',
        '  done',
        '  sleep 2',
        'done',
        'printf "%s\\n" running > "$vasp_auto_state_file"',
    ]


def run_vasp_remote(
    job_dir: str,
    remote: dict,
    cpus: int | None = None,
    on_progress=None,
    fetch_heavy: bool = True,
    remote_subdir: str | None = None,
    engine: str = "vasp",
) -> int:
    """Run VASP (or QE pw.x, ``engine="qe"``) on a remote machine via ``mpirun`` over SSH.

    Unlike :func:`submit_job_remote` (fire-and-forget queue submission), this runs
    synchronously: it ships the prepared inputs to ``<remote_root>/results/<job name>``,
    runs ``mpirun`` there (sourcing the machine's ``env_setup`` first so MKL/MPI
    libraries are on the path), waits for it to finish, then copies the results
    back so the local parsers and viewers work unchanged. A ``.remote.json`` marker
    tags the job with the machine and remote directory. Returns the VASP exit code,
    mirroring :func:`run_vasp`.

    Useful when the remote machine has no working scheduler (e.g. a single
    workstation) but can run ``mpirun`` directly. Required remote keys: ``host``,
    ``remote_root``, ``vasp_executable``. Optional: ``user``, ``port``,
    ``ssh_key``, ``ssh_options``, ``env_setup`` (a shell snippet sourced before the
    run, e.g. ``source /opt/intel/oneapi/setvars.sh``).
    """
    job_dir = Path(job_dir).resolve()
    if not remote.get("remote_root"):
        raise ValueError(
            "remote config needs a 'remote_root' (base directory on the remote machine)"
        )
    target = _ssh_target(remote)
    ssh_opts = _ssh_options(remote)
    # remote_subdir keeps multi-case / multi-trial jobs from colliding on the
    # remote (e.g. two cases both with an "encut_400" convergence trial).
    subpath = (remote_subdir or job_dir.name).strip("/")
    remote_dir = remote_results_base(remote) + "/" + subpath
    exe = _remote_exe(remote, engine)
    ranks = cpus if cpus is not None else 1
    env_setup = _remote_env_setup(remote, engine)
    machine = remote.get("name") or remote.get("host")

    quoted_dir = shlex.quote(remote_dir)
    _run_checked(["ssh", "-x", *ssh_opts, target, f"mkdir -p {quoted_dir}"], "remote mkdir")
    _transfer_dir(job_dir, target, remote_dir, remote)

    # One non-interactive shell: set up the toolchain, then launch mpirun. Its
    # output goes to run.log on the remote (fetched back below).
    state_file = f"{remote_dir}/.vasp_auto_state"
    parts = ["unset DISPLAY", *_remote_slot_lines(remote, state_file)]
    if env_setup:
        parts.append(f"{{ {env_setup} ; }} >/dev/null 2>&1 || true")
    parts.append("ulimit -s unlimited 2>/dev/null || true")
    parts.append(f"cd {quoted_dir}")
    if engine == "qe":
        parts.extend(_qe_shell_commands(job_dir, exe, ranks, quote_program=False))
    else:
        parts.append(f"mpirun -np {ranks} {shlex.quote(exe)} > run.log 2>&1")
    remote_script = "\n".join(parts)

    if on_progress is not None:
        on_progress(f"[remote] running on {machine}: {remote_dir}")
    result = subprocess.run(
        ["ssh", "-x", *ssh_opts, target, f"bash -lc {shlex.quote(remote_script)}"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return_code = result.returncode

    # Bring the results home so build_row / the viewers see local files.
    try:
        fetch_remote_results(remote, remote_dir, str(job_dir), include_heavy=fetch_heavy)
    except (RuntimeError, OSError):
        pass

    marker = {
        "machine": machine,
        "host": remote.get("host"),
        "remote_dir": remote_dir,
        "scheduler": "ssh",
        "mode": "ssh",
        "ran_at": datetime.now().isoformat(timespec="seconds"),
    }
    (job_dir / ".remote.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return return_code


# Shell equivalent of workflow._previous_run_ranks: pull the MPI rank count the
# previous run used from the header VASP prints in OUTCAR ("running on N total
# cores" / "running N mpi-ranks"). Runs in the remote job dir; leaves $ranks
# empty when there is no OUTCAR yet (mpirun then falls back to 1 below).
_RESUME_RANKS_SH = (
    "ranks=$(grep -h -m1 -aoE 'running +(on +)?[0-9]+ +(total cores|mpi-ranks|cores)' "
    "OUTCAR 01/OUTCAR 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)"
)

# Shell equivalent of workflow._ranks_for_images: VASP requires the rank count
# to be divisible by IMAGES (INCAR) — anything else aborts in M_divide /
# MPI_Cart_sub. Round $ranks down to a multiple, minimum one rank per image.
_NEB_RANKS_SH = (
    "img=$(grep -aiE '^[[:space:]]*IMAGES[[:space:]]*=' INCAR 2>/dev/null "
    "| grep -oE '[0-9]+' | head -1); "
    'if [ -n "$img" ] && [ "$img" -gt 0 ]; then '
    'if [ "${ranks:-0}" -lt "$img" ]; then ranks=$img; '
    "else ranks=$(( ranks - ranks % img )); fi; fi"
)


def resume_job_remote(
    remote: dict,
    remote_job_dir: str,
    cpus: int | None = None,
    on_progress=None,
    local_job_dir: str | None = None,
    fetch_heavy: bool = True,
) -> int:
    """Resume an unfinished VASP job *in place* on a remote machine.

    The remote counterpart of :func:`vasp_auto.workflow.resume_job`. Unlike
    :func:`run_vasp_remote`, nothing is shipped to the remote: the job already
    lives there, so its own INCAR/KPOINTS/POTCAR are reused verbatim. In one SSH
    session it advances POSCAR to the latest geometry (CONTCAR -> POSCAR, keeping
    POSCAR.bak) for the job and for every NEB image subdir, then runs ``mpirun``
    in that directory (sourcing the machine's ``env_setup`` first). When
    ``local_job_dir`` is given the results are copied back so the local parsers
    and viewers work. Returns the VASP exit code, mirroring :func:`run_vasp_remote`.

    Required remote keys: ``host``, ``vasp_executable``. Optional: ``user``,
    ``port``, ``ssh_key``, ``ssh_options``, ``env_setup``.
    """
    if not remote_job_dir:
        raise ValueError("resume_job_remote needs the remote job directory to resume")
    target = _ssh_target(remote)
    ssh_opts = _ssh_options(remote)
    exe = _remote_vasp_exe(remote)
    env_setup = (remote.get("env_setup") or "").strip()
    machine = remote.get("name") or remote.get("host")
    remote_job_dir = remote_job_dir.rstrip("/")
    quoted_dir = shlex.quote(remote_job_dir)

    # One non-interactive shell: set up the toolchain, advance POSCAR to the
    # newest CONTCAR (the job itself, plus each NEB image subdir, keeping the
    # previous POSCAR as POSCAR.bak), then launch mpirun in place.
    state_file = f"{remote_job_dir}/.vasp_auto_state"
    parts = ["unset DISPLAY", *_remote_slot_lines(remote, state_file)]
    if env_setup:
        parts.append(f"{{ {env_setup} ; }} >/dev/null 2>&1 || true")
    parts.append("ulimit -s unlimited 2>/dev/null || true")
    parts.append(f"cd {quoted_dir}")
    parts.append(
        'seed() { if [ -s "$1/CONTCAR" ]; then '
        '[ -f "$1/POSCAR" ] && cp -f "$1/POSCAR" "$1/POSCAR.bak"; '
        'cp -f "$1/CONTCAR" "$1/POSCAR"; fi; }'
    )
    parts.append("seed .")
    parts.append('for d in [0-9][0-9]; do [ -d "$d" ] && seed "$d"; done')
    # Stale summary from the previous run: it would mark the job "finished" while
    # the resume runs, and its parameters no longer match the (possibly edited) INCAR.
    parts.append("rm -f job.log")
    if cpus is not None:
        parts.append(f"ranks={int(cpus)}")
    else:
        # No explicit count: reuse the rank count of the previous run (OUTCAR).
        parts.append(_RESUME_RANKS_SH)
    parts.append(_NEB_RANKS_SH)
    parts.append(f'mpirun -np "${{ranks:-1}}" {shlex.quote(exe)} > run.log 2>&1')
    remote_script = "\n".join(parts)

    if on_progress is not None:
        on_progress(f"[remote] resuming on {machine}: {remote_job_dir}")
    result = subprocess.run(
        ["ssh", "-x", *ssh_opts, target, f"bash -lc {shlex.quote(remote_script)}"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return_code = result.returncode

    if local_job_dir:
        try:
            fetch_remote_results(remote, remote_job_dir, str(local_job_dir), include_heavy=fetch_heavy)
        except (RuntimeError, OSError):
            pass
        # The remote no longer carries a job.log (removed at resume start), so
        # rebuild it here from the fetched files — with the resumed INCAR.
        if (Path(local_job_dir) / "OUTCAR").exists():
            from vasp_auto.job_log import write_job_log
            write_job_log(Path(local_job_dir), Path(local_job_dir).name,
                          return_code=return_code)
        marker = {
            "machine": machine,
            "host": remote.get("host"),
            "remote_dir": remote_job_dir,
            "scheduler": "ssh",
            "mode": "ssh",
            "resumed_at": datetime.now().isoformat(timespec="seconds"),
        }
        Path(local_job_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_job_dir) / ".remote.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return return_code


def resume_job_detached(
    remote: dict,
    remote_job_dir: str,
    cpus: int | None = None,
    local_job_dir: str | None = None,
    on_progress=None,
) -> dict:
    """Resume an unfinished VASP job *in place* on a remote machine, detached.

    The power-off-safe counterpart of :func:`resume_job_remote`: it advances
    POSCAR to the newest CONTCAR (the job itself plus every NEB image subdir,
    keeping POSCAR.bak) and re-runs ``mpirun`` in the remote job directory under
    ``setsid``, recording the PID and return code in a control dir so the run can
    be polled (and the local host powered off) after SSH disconnects. Nothing is
    shipped — the remote job's own INCAR/KPOINTS/POTCAR are reused.

    Like :func:`submit_job_detached`, the control dir lives under
    ``<remote_root>/.vasp_auto/runs/<job name>`` and holds ``pid``/``rc``/``job_dir``/
    ``run.sh``; :func:`poll_detached_job` and :func:`resolve_detached_job_dir` work
    against it unchanged. When ``local_job_dir`` is given a ``.remote.json`` marker
    (mode ``ssh_detached``) is written there so the UI's status/fetch buttons behave
    exactly as for a fresh offload. Returns
    {"machine","remote_dir","control_dir","pid","mode","scheduler"}.
    """
    if not remote_job_dir:
        raise ValueError("resume_job_detached needs the remote job directory to resume")
    paths = _remote_engine_paths(remote)
    target = _ssh_target(remote)
    ssh_opts = _ssh_options(remote)
    exe = _remote_vasp_exe(remote)
    env_setup = (remote.get("env_setup") or "").strip()
    machine = remote.get("name") or remote.get("host")
    remote_job_dir = remote_job_dir.rstrip("/")
    job_name = remote_job_dir.rsplit("/", 1)[-1] or "job"
    control_dir = f"{paths['runs']}/{job_name}"

    quoted_job = shlex.quote(remote_job_dir)
    pid_f = f"{control_dir}/pid"
    rc_f = f"{control_dir}/rc"
    jobdir_f = f"{control_dir}/job_dir"
    state_f = f"{control_dir}/state"
    env_line = env_setup if env_setup else "true"

    _run_checked(
        ["ssh", "-x", *ssh_opts, target, f"mkdir -p {shlex.quote(control_dir)}"],
        "remote mkdir control",
    )

    # run.sh: seed POSCAR<-CONTCAR (the job + each NEB image), then mpirun in the
    # job dir. The PID/rc files make it poll-able after the launcher disconnects.
    script = "\n".join([
        "#!/bin/bash",
        f"echo $$ > {shlex.quote(pid_f)}",
        f"echo {quoted_job} > {shlex.quote(jobdir_f)}",
        *_remote_slot_lines(remote, state_f),
        f"{{ {env_line} ; }} >/dev/null 2>&1 || true",
        "ulimit -s unlimited 2>/dev/null || true",
        f"cd {quoted_job}",
        'seed() { if [ -s "$1/CONTCAR" ]; then '
        '[ -f "$1/POSCAR" ] && cp -f "$1/POSCAR" "$1/POSCAR.bak"; '
        'cp -f "$1/CONTCAR" "$1/POSCAR"; fi; }',
        "seed .",
        'for d in [0-9][0-9]; do [ -d "$d" ] && seed "$d"; done',
        "rm -f job.log",
        # No explicit count: reuse the rank count of the previous run (OUTCAR).
        f"ranks={int(cpus)}" if cpus is not None else _RESUME_RANKS_SH,
        _NEB_RANKS_SH,
        f'mpirun -np "${{ranks:-1}}" {shlex.quote(exe)} > run.log 2>&1',
        f"echo $? > {shlex.quote(rc_f)}",
        "",
    ])
    with tempfile.TemporaryDirectory() as tmp:
        sh = Path(tmp) / "run.sh"
        sh.write_text(script, encoding="utf-8")
        _ship_file(sh, target, f"{control_dir}/run.sh", remote, ssh_opts)

    if on_progress is not None:
        on_progress(f"[remote] resuming detached on {machine}: {remote_job_dir}")
    launch = (
        f"rm -f {shlex.quote(rc_f)} {shlex.quote(pid_f)}; "
        f"setsid bash {shlex.quote(control_dir + '/run.sh')} </dev/null >/dev/null 2>&1 & "
        f"sleep 1; cat {shlex.quote(pid_f)} 2>/dev/null"
    )
    out = _run_checked(["ssh", "-x", *ssh_opts, target, launch], "remote resume launch")
    pid = out.strip().splitlines()[-1].strip() if out.strip() else ""

    result = {
        "machine": machine,
        "host": remote.get("host"),
        "remote_dir": remote_job_dir,
        "control_dir": control_dir,
        **({"state_file": state_f} if remote_concurrency_limit(remote) is not None else {}),
        "pid": pid,
        "scheduler": "ssh_detached",
        "mode": "ssh_detached",
    }
    if local_job_dir:
        marker = {**result, "resumed_at": datetime.now().isoformat(timespec="seconds")}
        Path(local_job_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_job_dir) / ".remote.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
        # Same as the remote side: the stale summary must not mark the mirror
        # "finished"; the fetch after the run rebuilds it from the resumed INCAR.
        (Path(local_job_dir) / "job.log").unlink(missing_ok=True)
    return result


def submit_job_remote(
    job_dir: str,
    remote: dict,
    cpus: int | None = None,
    job_template: str | None = None,
    engine: str = "vasp",
) -> dict:
    """Send a fully prepared job to a remote machine and submit it to its queue.

    Every input file VASP needs (INCAR, KPOINTS, POSCAR, POTCAR, submit.sh) is
    copied to the remote host first, then ``sbatch``/``qsub`` is invoked there
    over SSH. Once this returns the job is queued on the remote scheduler, so the
    local host can be powered off.

    ``remote`` is the config.yaml ``remote:`` mapping. Required keys: ``host`` and
    ``remote_root`` (a base directory on the remote machine). Recommended:
    ``vasp_executable`` (the VASP path *on the remote*). Optional: ``user``,
    ``port``, ``ssh_key``, ``ssh_options`` (list), ``scheduler`` (slurm|pbs,
    default slurm), ``scheduler_options`` (list of extra script lines).

    Returns {"job_id", "scheduler", "host", "remote_dir", "submit_output"}.
    """
    job_dir = Path(job_dir).resolve()
    remote_root = remote.get("remote_root")
    if not remote_root:
        raise ValueError(
            "remote config needs a 'remote_root' (base directory on the remote machine)"
        )
    scheduler = remote.get("scheduler", "slurm")
    if scheduler not in SCHEDULER_COMMANDS:
        raise ValueError(f"Unknown remote scheduler: {scheduler} (use slurm or pbs)")
    exe = _remote_exe(remote, engine)

    target = _ssh_target(remote)
    remote_dir = remote_results_base(remote) + "/" + job_dir.name
    template = job_template or remote.get("job_template")

    # Write the submit script with the remote run directory baked in.
    scheduler_options = list(remote.get("scheduler_options") or [])
    qe_env_setup = _remote_env_setup(remote, engine)
    if engine == "qe" and qe_env_setup:
        scheduler_options.append(qe_env_setup)
    state_file = f"{remote_dir}/.vasp_auto_state"
    slot_preamble = _remote_slot_lines(remote, state_file)
    write_submit_script(
        str(job_dir),
        str(exe),
        cpus=cpus,
        scheduler=scheduler,
        template_path=template,
        options=scheduler_options,
        run_dir=remote_dir,
        engine=engine,
        preamble=slot_preamble,
    )

    ssh_opts = _ssh_options(remote)
    quoted_dir = shlex.quote(remote_dir)

    # 1. make sure the destination exists, 2. copy everything, 3. submit there.
    _run_checked(
        ["ssh", *ssh_opts, target, f"mkdir -p {quoted_dir}"],
        "remote mkdir",
    )
    _transfer_dir(job_dir, target, remote_dir, remote)
    submit_cmd = SCHEDULER_COMMANDS[scheduler]
    output = _run_checked(
        ["ssh", *ssh_opts, target, f"cd {quoted_dir} && {submit_cmd} submit.sh"],
        f"remote {submit_cmd}",
    ).strip()

    # sbatch: "Submitted batch job 12345"; qsub: "12345.hostname"
    if scheduler == "slurm":
        job_id = output.split()[-1] if output else ""
    else:
        job_id = output.splitlines()[0].strip() if output else ""

    machine = remote.get("name") or remote.get("host")
    result = {
        "job_id": job_id,
        "scheduler": scheduler,
        "machine": machine,
        "host": remote.get("host"),
        "remote_dir": remote_dir,
        **({"state_file": state_file} if slot_preamble else {}),
        "submit_output": output,
    }

    # Tag the local job dir so the UI/results know this case ran on a remote
    # machine (the output files themselves stay on that machine).
    marker = {**result, "submitted_at": datetime.now().isoformat(timespec="seconds")}
    (job_dir / ".remote.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------- detached offload mode
#
# "Offload" runs install the full vasp_auto engine in a venv on the remote
# machine and drive the whole calculation there detached (setsid), so the local
# host can be powered off. Unlike run_vasp_remote (synchronous, local stays on)
# or submit_job_remote (needs a working scheduler), this works on a plain
# workstation and supports the iterative paths (convergence scans, workflows)
# because the engine itself runs remotely.

ENGINE_SUBDIR = ".vasp_auto"   # under remote_root: venv/ + runs/ control dirs


def _scp_options(remote: dict) -> list[str]:
    """scp option flags (-P PORT, -i KEY, extra ssh_options). scp uses -P, not -p."""
    opts: list[str] = []
    if remote.get("port"):
        opts += ["-P", str(remote["port"])]
    if remote.get("ssh_key"):
        opts += ["-i", str(Path(remote["ssh_key"]).expanduser())]
    opts += list(remote.get("ssh_options") or [])
    return opts


def _ship_file(local: Path, target: str, remote_path: str, remote: dict, ssh_opts: list[str]) -> None:
    """Copy a single local file to remote_path (rsync if present, else scp)."""
    if shutil.which("rsync"):
        cmd = ["rsync", "-az"]
        if ssh_opts:
            cmd += ["-e", "ssh " + " ".join(shlex.quote(o) for o in ssh_opts)]
        cmd += [str(local), f"{target}:{remote_path}"]
        _run_checked(cmd, "rsync file")
    else:
        _run_checked(["scp", *_scp_options(remote), str(local), f"{target}:{remote_path}"], "scp file")


def _remote_engine_paths(remote: dict) -> dict:
    """Standard locations of the remote-installed engine and its run state."""
    root = remote.get("remote_root")
    if not root:
        raise ValueError("remote config needs a 'remote_root' (base directory on the remote)")
    root = root.rstrip("/")
    home = f"{root}/{ENGINE_SUBDIR}"
    return {
        "root": root,
        "home": home,
        "venv": f"{home}/venv",
        "vasp_auto": f"{home}/venv/bin/vasp-auto",
        "runs": f"{home}/runs",
    }


def build_engine_wheel(dest_dir: str | Path) -> Path:
    """Build a vasp_auto wheel from the source repo; return the wheel path."""
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "pyproject.toml").exists():
        raise FileNotFoundError(
            f"cannot build a wheel: no pyproject.toml at {repo_root}. Remote engine "
            "setup needs vasp_auto installed from source."
        )
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    # setuptools uses <repo>/build even when wheels have separate destinations.
    # Serialize builds so concurrent setup requests for multiple remotes do not
    # delete or replace one another's intermediate files.
    lock_path = Path(tempfile.gettempdir()) / "vasp_auto-wheel-build.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _run_checked(
            [sys.executable, "-m", "pip", "wheel", str(repo_root), "-w", str(dest), "--no-deps"],
            "build wheel",
        )
    wheels = sorted(dest.glob("vasp_auto-*.whl"))
    if not wheels:
        raise RuntimeError("wheel build produced no vasp_auto-*.whl")
    return wheels[-1]


def setup_remote_engine(remote: dict, timeout: int = 600, on_progress=None) -> dict:
    """Install vasp_auto into a venv on the remote machine (one-time per machine).

    Builds a wheel locally, ships it, creates a venv (bootstrapping pip via
    get-pip.py when the distro lacks ensurepip / python3-venv), and pip-installs
    the wheel with its dependencies. After this, detached/offload runs
    (submit_job_detached) can drive the full engine on the remote so the local
    host can be powered off. Returns {"ok", "vasp_auto", "detail"}.
    """
    paths = _remote_engine_paths(remote)
    target = _ssh_target(remote)
    ssh_opts = _ssh_options(remote)

    def log(msg):
        if on_progress is not None:
            on_progress(msg)

    with tempfile.TemporaryDirectory() as tmp:
        log("building wheel…")
        wheel = build_engine_wheel(tmp)
        _run_checked(["ssh", "-x", *ssh_opts, target, f"mkdir -p {shlex.quote(paths['home'])}"],
                     "remote mkdir engine home")
        log(f"shipping {wheel.name}…")
        _ship_file(wheel, target, f"{paths['home']}/{wheel.name}", remote, ssh_opts)
        wheel_remote = f"{paths['home']}/{wheel.name}"

        log("creating venv + installing dependencies (may take a minute)…")
        script = "\n".join([
            "set -e",
            f"cd {shlex.quote(paths['home'])}",
            "rm -rf venv",
            "python3 -m venv --without-pip venv",
            "if [ ! -x venv/bin/pip ]; then "
            "(curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py "
            "|| wget -q https://bootstrap.pypa.io/get-pip.py -O get-pip.py); "
            "venv/bin/python get-pip.py --quiet; fi",
            f"venv/bin/pip install --quiet {shlex.quote(wheel_remote)}",
            # Verify only the lean engine: vasp_auto plus its sole core dep (PyYAML).
            # pandas/openpyxl live in the [results] extra and are intentionally NOT
            # installed here — the offload engine is lean (docs/INSTALL.md, "Why the
            # engine is lean"). Importing them in the check made setup always fail.
            "venv/bin/python -c 'import vasp_auto, yaml; print(\"ENGINE_OK\")'",
        ])
        res = subprocess.run(
            ["ssh", "-x", *ssh_opts, target, f"bash -lc {shlex.quote(script)}"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
        )

    ok = res.returncode == 0 and "ENGINE_OK" in res.stdout
    detail = (res.stdout + res.stderr).strip()
    return {"ok": ok, "vasp_auto": paths["vasp_auto"], "detail": detail[-2000:]}


def remote_engine_installed(remote: dict) -> bool:
    """True if the offload engine (vasp-auto) is present in the remote venv."""
    paths = _remote_engine_paths(remote)
    target = _ssh_target(remote)
    ssh_opts = _ssh_options(remote)
    try:
        res = subprocess.run(
            ["ssh", "-x", *ssh_opts, target,
             f"test -x {shlex.quote(paths['vasp_auto'])} && echo yes || echo no"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=20,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return "yes" in res.stdout


def submit_job_detached(
    case_dir: str,
    remote: dict,
    case_name: str,
    cpus: int | None,
    calc_flags: list[str],
    local_job_dir: str | None = None,
    on_progress=None,
    engine: str = "vasp",
) -> dict:
    """Offload a full calculation to the remote engine and return immediately.

    The remote-installed engine (see setup_remote_engine) runs the calculation
    detached via setsid, so the local machine can be powered off. Ships the case
    inputs (POSCAR + POTCAR + optional INCAR/KPOINTS) and a remote config.yaml,
    then launches ``vasp-auto inputs/<case> <calc_flags> -n <cpus>`` under setsid,
    records its PID, and writes a ``.remote.json`` marker into ``local_job_dir``.
    Results stay under ``<remote_root>/jobs/<case>`` until fetched. Returns
    {"machine","remote_dir","inputs_dir","control_dir","pid","log"}.
    """
    case_dir = Path(case_dir)
    paths = _remote_engine_paths(remote)
    target = _ssh_target(remote)
    ssh_opts = _ssh_options(remote)
    exe = _remote_exe(remote, engine)
    ranks = cpus if cpus is not None else 1
    env_setup = _remote_env_setup(remote, engine)
    machine = remote.get("name") or remote.get("host")

    # A dedicated results/ dir (absolute jobs_root) so output never lands in a
    # doubled path when remote_root itself already ends in e.g. "jobs".
    results_base = remote_results_base(remote)
    inputs_remote = f"{paths['root']}/inputs/{case_name}"
    jobs_remote = f"{results_base}/{case_name}"
    control_dir = f"{paths['runs']}/{case_name}"

    # First offload to a machine self-installs the engine (one-time, idempotent),
    # so no separate --remote-setup step is needed before a normal run.
    if not remote_engine_installed(remote):
        if on_progress is not None:
            on_progress(f"[remote] installing the offload engine on {machine} (one-time)…")
        setup = setup_remote_engine(remote, on_progress=on_progress)
        if not setup.get("ok"):
            raise RuntimeError(
                f"could not install the offload engine on {machine}. Detail:\n{setup.get('detail', '')}"
            )

    _run_checked(
        ["ssh", "-x", *ssh_opts, target,
         f"mkdir -p {shlex.quote(inputs_remote)} {shlex.quote(control_dir)} {shlex.quote(results_base)}"],
        "remote mkdir",
    )

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "config.yaml"
        cfg_lines = [f"jobs_root: {results_base}", "neb_images: 5"]
        if engine == "qe":
            cfg_lines.append(f"qe_executable: {exe}")
            # The machine's own UPF library, if it has one; otherwise the caller
            # ships the UPFs in the bundle and forwards a --pseudo-dir flag.
            pseudo_dir = (remote.get("pseudo_dir") or "").strip()
            if pseudo_dir:
                cfg_lines.append(f"pseudo_dir: {pseudo_dir}")
        else:
            cfg_lines.append(f"vasp_executable: {exe}")
            # When the machine has its own POTCAR library (remote 'potcar_root'), point the
            # remote engine at it so it builds the POTCAR there and no POTCAR needs shipping.
            # Without it, a pre-built POTCAR rides along in the inputs bundle (see
            # _run_detached_offload). ponytail: potcar_map (variant selection) isn't
            # forwarded — add it here if a config ever sets one.
            potcar_root = (remote.get("potcar_root") or "").strip()
            if potcar_root:
                cfg_lines.append(f"potcar_root: {potcar_root}")
        cfg.write_text("\n".join(cfg_lines) + "\n", encoding="utf-8")
        _ship_file(cfg, target, f"{paths['root']}/config.yaml", remote, ssh_opts)

    # Ship the INCAR templates (example/INCAR_*) so the remote engine can build
    # INCARs for every calc type (relax/dos/bands/freq/…), not only the built-in
    # scf/neb. VASP_AUTO_ROOT (set in run.sh below) points the loader at them.
    repo_example = Path(__file__).resolve().parents[2] / "example"
    templates_root = ""
    if repo_example.is_dir():
        _run_checked(
            ["ssh", "-x", *ssh_opts, target, f"mkdir -p {shlex.quote(paths['home'] + '/example')}"],
            "remote mkdir example",
        )
        _transfer_dir(repo_example, target, f"{paths['home']}/example", remote)
        templates_root = paths["home"]

    # Inputs bundle: POSCAR + a pre-built POTCAR (+ INCAR/KPOINTS if supplied) so
    # the remote engine never needs the POTCAR library.
    _transfer_dir(case_dir, target, inputs_remote, remote)

    flags_str = " ".join(shlex.quote(f) for f in calc_flags)
    env_line = env_setup if env_setup else "true"
    pid_f = f"{control_dir}/pid"
    rc_f = f"{control_dir}/rc"
    log_f = f"{control_dir}/run.log"
    jobdir_f = f"{control_dir}/job_dir"
    state_f = f"{control_dir}/state"
    script = "\n".join([
        "#!/bin/bash",
        f"echo $$ > {shlex.quote(pid_f)}",
        *_remote_slot_lines(remote, state_f),
        f"cd {shlex.quote(paths['root'])}",
        *([f"export VASP_AUTO_ROOT={shlex.quote(templates_root)}"] if templates_root else []),
        # The engine writes the real (numbered) job root here, so the submitting host
        # can later resolve exactly where the job lives on this machine.
        f"export VASP_AUTO_JOBDIR_FILE={shlex.quote(jobdir_f)}",
        f"{{ {env_line} ; }} >/dev/null 2>&1 || true",
        "ulimit -s unlimited 2>/dev/null || true",
        f"{shlex.quote(paths['vasp_auto'])} {shlex.quote('inputs/' + case_name)} "
        f"{flags_str} -n {ranks} > {shlex.quote(log_f)} 2>&1",
        f"echo $? > {shlex.quote(rc_f)}",
        "",
    ])
    with tempfile.TemporaryDirectory() as tmp:
        sh = Path(tmp) / "run.sh"
        sh.write_text(script, encoding="utf-8")
        _ship_file(sh, target, f"{control_dir}/run.sh", remote, ssh_opts)

    launch = (
        f"rm -f {shlex.quote(rc_f)} {shlex.quote(pid_f)}; "
        f"setsid bash {shlex.quote(control_dir + '/run.sh')} </dev/null >/dev/null 2>&1 & "
        f"sleep 1; cat {shlex.quote(pid_f)} 2>/dev/null"
    )
    out = _run_checked(["ssh", "-x", *ssh_opts, target, launch], "remote launch")
    pid = out.strip().splitlines()[-1].strip() if out.strip() else ""

    result = {
        "machine": machine,
        "host": remote.get("host"),
        "remote_dir": jobs_remote,
        "inputs_dir": inputs_remote,
        "control_dir": control_dir,
        **({"state_file": state_f} if remote_concurrency_limit(remote) is not None else {}),
        "pid": pid,
        "log": log_f,
        "scheduler": "ssh_detached",
        "mode": "ssh_detached",
    }
    if local_job_dir:
        marker = {**result, "submitted_at": datetime.now().isoformat(timespec="seconds")}
        Path(local_job_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_job_dir) / ".remote.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return result


def poll_detached_job(remote: dict, control_dir: str, pid: str | None = None) -> dict:
    """Status of a detached offload job from its control dir (pid + rc files).

    Returns {"state", "return_code", "raw"} with state running/completed/unknown.
    """
    target = _ssh_target(remote)
    ssh_opts = _ssh_options(remote)
    rc_f = f"{control_dir}/rc"
    state_f = f"{control_dir}/state"
    cmd = (
        f"if [ -f {shlex.quote(rc_f)} ]; then echo DONE; cat {shlex.quote(rc_f)}; "
        f"elif [ \"$(cat {shlex.quote(state_f)} 2>/dev/null)\" = pending ]; then echo PENDING; "
        f"elif [ -n {shlex.quote(pid or '')} ] && kill -0 {shlex.quote(pid or '0')} 2>/dev/null; "
        f"then echo RUNNING; else echo UNKNOWN; fi"
    )
    try:
        res = subprocess.run(["ssh", "-x", *ssh_opts, target, cmd],
                             capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"state": "unknown", "return_code": None, "raw": str(exc)}
    lines = res.stdout.strip().splitlines()
    if lines and lines[0] == "DONE":
        return {"state": "completed", "return_code": lines[1] if len(lines) > 1 else None,
                "raw": res.stdout.strip()}
    if lines and lines[0] == "RUNNING":
        return {"state": "running", "return_code": None, "raw": res.stdout.strip()}
    if lines and lines[0] == "PENDING":
        return {"state": "pending", "return_code": None, "raw": res.stdout.strip()}
    return {"state": "unknown", "return_code": None, "raw": res.stdout.strip() or res.stderr.strip()}


def kill_detached_job(remote: dict, control_dir: str, pid: str | None = None) -> dict:
    """Terminate a detached offload job by its recorded PID over SSH.

    Detached jobs are launched under ``setsid``, so the recorded PID leads its own
    process group — ``kill -TERM -<pid>`` takes down mpirun/vasp and the driver in
    one shot (with a bare-PID fallback). Stamps ``rc=143`` (128+SIGTERM) in the
    control dir so a later :func:`poll_detached_job` reads "completed", not "unknown".
    Returns ``{"killed", "raw"}``.
    """
    if not pid:
        return {"killed": False, "raw": "no PID recorded for this job"}
    target = _ssh_target(remote)
    ssh_opts = _ssh_options(remote)
    p = shlex.quote(str(pid))
    cd = shlex.quote(control_dir.rstrip("/")) if control_dir else ""
    # Stamp rc=143 (poll reads "completed") and drop a .terminated marker in the
    # job dir the control dir points at, so the Running tab reads "terminated".
    stamp = (f"echo 143 > {cd}/rc; jd=$(cat {cd}/job_dir 2>/dev/null); "
             f'[ -n "$jd" ] && echo terminated > "$jd/.terminated" 2>/dev/null; ') if control_dir else ""
    cmd = f"kill -TERM -{p} 2>/dev/null; kill -TERM {p} 2>/dev/null; {stamp}true"
    try:
        res = subprocess.run(["ssh", "-x", *ssh_opts, target, cmd],
                             capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"killed": False, "raw": str(exc)}
    return {"killed": res.returncode == 0, "raw": (res.stdout + res.stderr).strip()}


def kill_job_by_dir(remote: dict, remote_dir: str) -> dict:
    """Terminate the job running in a remote job directory: the process group of
    the PID recorded in ``<dir>/.pid``, plus any live process whose cwd is inside
    the dir — the same two liveness signals the Running board uses, so anything
    it shows as "running" can be stopped. Drops a ``.terminated`` marker so the
    listing reads "terminated". Works for any job on the machine, not just ones
    this host launched. Returns {"killed","raw"}.
    """
    rd = shlex.quote(remote_dir.rstrip("/"))
    cmd = (
        f'd={rd}; pid=$(cat "$d/.pid" 2>/dev/null); ok=0; '
        f'if [ -n "$pid" ]; then kill -TERM -"$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; ok=1; fi; '
        # Also processes working inside the dir (cwd match) — covers jobs with no
        # recorded .pid, exactly what the running scan counts as alive.
        'for l in /proc/[0-9]*/cwd; do c=$(readlink "$l" 2>/dev/null); '
        'case "$c" in "$d"|"$d"/*) p=${l#/proc/}; p=${p%/cwd}; '
        'kill -TERM "$p" 2>/dev/null && ok=1;; esac; done; '
        f'echo terminated > "$d/.terminated" 2>/dev/null; echo "$ok"'
    )
    try:
        res = remote_command(remote, cmd, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"killed": False, "raw": str(exc)}
    last = res.stdout.strip().splitlines()[-1:] if res.stdout.strip() else []
    return {"killed": last == ["1"], "raw": (res.stdout + res.stderr).strip()}


def delete_remote_dir(remote: dict, remote_dir: str) -> None:
    """``rm -rf`` a directory on the remote machine. The caller MUST have checked
    that ``remote_dir`` is inside the machine's remote_root — this does not."""
    res = remote_command(remote, f"rm -rf {shlex.quote(remote_dir)}", timeout=60)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or f"could not delete {remote_dir}")


def clear_remote_terminated(remote: dict, remote_dir: str) -> None:
    """Remove a stale ``.terminated`` marker before a remote resume (best effort)."""
    try:
        remote_command(remote, f"rm -f {shlex.quote(remote_dir.rstrip('/'))}/.terminated", timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        pass


def resolve_detached_job_dir(remote: dict, control_dir: str) -> str | None:
    """The real (numbered) job root a detached offload chose, or None if not yet known.

    The remote engine writes its allocated job dir to ``<control_dir>/job_dir`` at the
    start of a run (see the run.sh built by :func:`submit_job_detached`). Reading it back
    gives the submitting host the exact ``<remote_root>/results/<NNNN>_<case>`` path the
    job lives in, instead of the bare placeholder recorded at submit time.
    """
    if not control_dir:
        return None
    target = _ssh_target(remote)
    ssh_opts = _ssh_options(remote)
    jobdir_f = f"{control_dir.rstrip('/')}/job_dir"
    cmd = f"cat {shlex.quote(jobdir_f)} 2>/dev/null"
    try:
        res = subprocess.run(["ssh", "-x", *ssh_opts, target, cmd],
                             capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return None
    path = res.stdout.strip()
    return path or None


def poll_job_status(job_id: str, scheduler: str = "slurm") -> dict:
    """Query a scheduler for the status of a submitted job.

    Returns a dict with keys:
      - job_id: the queried job ID
      - scheduler: the scheduler used
      - state: one of "running", "pending", "completed", "unknown"
      - raw: the raw stdout from the poll command (empty string if unavailable)

    Gracefully returns state="unknown" when:
      - the scheduler binary is not on PATH
      - the scheduler is not in POLL_COMMANDS
      - the command fails (e.g. job ID no longer in the queue = completed)
    """
    if scheduler not in POLL_COMMANDS:
        return {"job_id": job_id, "scheduler": scheduler, "state": "unknown", "raw": ""}

    base_cmd = POLL_COMMANDS[scheduler]
    binary = base_cmd[0]

    if shutil.which(binary) is None:
        return {"job_id": job_id, "scheduler": scheduler, "state": "unknown", "raw": ""}

    cmd = base_cmd + [str(job_id)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return {"job_id": job_id, "scheduler": scheduler, "state": "unknown", "raw": ""}

    return _parse_poll_output(scheduler, job_id, result.returncode, result.stdout, result.stderr)


def _parse_poll_output(scheduler, job_id, returncode, stdout, stderr):
    """Turn squeue/qstat output into a {state, raw, ...} dict (local or remote)."""
    raw = (stdout + stderr).strip()

    if scheduler == "slurm":
        # squeue -h -j <id> prints nothing when the job is done/absent.
        if returncode != 0 or not stdout.strip():
            if "Invalid job id" in raw or not raw:
                return {"job_id": job_id, "scheduler": scheduler, "state": "completed", "raw": raw}
            return {"job_id": job_id, "scheduler": scheduler, "state": "unknown", "raw": raw}
        # With -h (no header) each line is: JOBID PARTITION NAME USER ST TIME NODES NODELIST
        fields = stdout.split()
        st = fields[4] if len(fields) > 4 else ""
        state_map = {"R": "running", "CG": "running", "PD": "pending", "CF": "pending"}
        return {"job_id": job_id, "scheduler": scheduler,
                "state": state_map.get(st, "unknown"), "raw": raw}

    if scheduler == "pbs":
        if returncode != 0:
            # qstat fails when the job is not found (finished and purged).
            return {"job_id": job_id, "scheduler": scheduler, "state": "completed", "raw": raw}
        # qstat output: job_id.host owner queue job_name session NDS TSK mem time status
        for line in stdout.splitlines():
            if str(job_id) in line:
                parts = line.split()
                st = parts[-2] if len(parts) >= 2 else ""
                state_map = {"R": "running", "E": "running", "Q": "pending",
                             "H": "pending", "C": "completed"}
                return {"job_id": job_id, "scheduler": scheduler,
                        "state": state_map.get(st, "unknown"), "raw": raw}
        return {"job_id": job_id, "scheduler": scheduler, "state": "unknown", "raw": raw}

    return {"job_id": job_id, "scheduler": scheduler, "state": "unknown", "raw": raw}


# ------------------------------------------------- remote connection / status / fetch

# Large binaries that are not worth pulling back across the network by default.
# "tmp" is QE's outdir (wavefunctions/charge under <prefix>.save).
HEAVY_OUTPUTS = ["WAVECAR", "CHG", "CHGCAR", "vaspout.h5", "AECCAR0", "AECCAR1", "AECCAR2", "tmp"]


def remote_command(remote: dict, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a shell command on the remote host over SSH and return the result."""
    target = _ssh_target(remote)
    ssh_opts = _ssh_options(remote)
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", *ssh_opts, target, command],
        capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
    )


def check_remote_connection(remote: dict) -> dict:
    """Check SSH reachability plus remote_root / VASP / scheduler availability.

    Returns {"ok", "host", "message", "checks": [{"name","ok","detail"}, ...]}.
    Never raises for a normal SSH failure — it reports it in the result so the
    UI button can show what went wrong.
    """
    checks: list[dict] = []
    try:
        target = _ssh_target(remote)
    except ValueError as exc:
        return {"ok": False, "host": remote.get("host"), "message": str(exc), "checks": checks}

    try:
        probe = remote_command(remote, "echo vasp_auto_ok", timeout=20)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"ok": False, "host": remote.get("host"),
                "message": f"SSH to {target} failed: {exc}", "checks": checks}

    if probe.returncode != 0 or "vasp_auto_ok" not in probe.stdout:
        return {"ok": False, "host": remote.get("host"),
                "message": probe.stderr.strip() or "SSH connection failed", "checks": checks}
    checks.append({"name": "ssh", "ok": True, "detail": f"connected to {target}"})

    root = remote.get("remote_root")
    if root:
        res = remote_command(remote, f"test -d {shlex.quote(root)} && echo yes || echo no")
        ok = "yes" in res.stdout
        checks.append({"name": "remote_root", "ok": True,
                       "detail": f"{root} exists" if ok else f"{root} will be created on submit"})

    if remote.get("vasp_executable"):
        exe = _remote_vasp_exe(remote)  # resolves a bin directory to <dir>/vasp_std
        res = remote_command(remote, f"test -x {shlex.quote(exe)} && echo yes || echo no")
        ok = "yes" in res.stdout
        checks.append({"name": "vasp_executable", "ok": ok,
                       "detail": f"{exe} found" if ok else f"{exe} not found or not executable"})

    if remote.get("qe_executable"):
        qe = remote["qe_executable"]
        res = remote_command(
            remote,
            f"{{ test -x {shlex.quote(qe)} || command -v {shlex.quote(qe)} >/dev/null 2>&1; }} "
            "&& echo yes || echo no",
        )
        ok = "yes" in res.stdout
        checks.append({"name": "qe_executable", "ok": ok,
                       "detail": f"{qe} found" if ok else f"{qe} not found or not executable"})
        if ok and "/" in qe:
            from vasp_auto.qe_tools import QE_COMPANION_PROGRAMS
            qe_dir = str(Path(qe).parent)
            names = " ".join(shlex.quote(name) for name in QE_COMPANION_PROGRAMS)
            script = (
                f"missing=''; for x in {names}; do test -x {shlex.quote(qe_dir)}/$x || "
                "missing=\"$missing $x\"; done; "
                "test -z \"$missing\" && echo yes || echo missing:$missing"
            )
            suite = remote_command(remote, script)
            suite_ok = suite.stdout.strip() == "yes"
            checks.append({"name": "qe_suite", "ok": suite_ok,
                           "detail": "all companion programs found" if suite_ok
                           else suite.stdout.strip()})

    if remote.get("pseudo_dir"):
        pseudo_dir = remote["pseudo_dir"]
        res = remote_command(
            remote, f"find {shlex.quote(pseudo_dir)} -maxdepth 1 -type f -iname '*.upf' -print -quit "
                    "2>/dev/null | grep -q . && echo yes || echo no",
        )
        ok = "yes" in res.stdout
        checks.append({"name": "qe_pseudos", "ok": ok,
                       "detail": f"UPF files found in {pseudo_dir}" if ok
                       else f"no UPF files found in {pseudo_dir}"})

    # ssh run mode launches mpirun directly; scheduler modes need their submit cmd.
    if remote_run_mode(remote) == "ssh":
        res = remote_command(remote, "command -v mpirun >/dev/null 2>&1 && echo yes || echo no")
        ok = "yes" in res.stdout
        checks.append({"name": "mpirun", "ok": ok,
                       "detail": "mpirun available" if ok else "mpirun not on PATH"})
    else:
        scheduler = remote.get("scheduler", "slurm")
        submit_cmd = SCHEDULER_COMMANDS.get(scheduler)
        if submit_cmd:
            res = remote_command(remote, f"command -v {submit_cmd} >/dev/null 2>&1 && echo yes || echo no")
            ok = "yes" in res.stdout
            checks.append({"name": scheduler, "ok": ok,
                           "detail": f"{submit_cmd} available" if ok else f"{submit_cmd} not on PATH"})

    if remote_concurrency_limit(remote) is not None:
        res = remote_command(remote, "command -v flock >/dev/null 2>&1 && echo yes || echo no")
        ok = "yes" in res.stdout
        checks.append({"name": "job limit", "ok": ok,
                       "detail": (f"up to {remote_concurrency_limit(remote)} jobs at once"
                                  if ok else "flock is required for the max-jobs setting")})

    # remote_root is allowed to be missing (created on submit); everything else must pass.
    overall = all(c["ok"] for c in checks if c["name"] != "remote_root")
    return {
        "ok": overall,
        "host": remote.get("host"),
        "message": "Connection OK" if overall else "Connected, but some checks failed",
        "checks": checks,
    }


def poll_remote_job(remote: dict, job_id: str, state_file: str | None = None) -> dict:
    """Query the remote scheduler for a job's status over SSH.

    When a configured concurrency limit holds an already-allocated scheduler
    job at the application gate, its state file takes precedence over the
    scheduler's coarse ``running`` state and reports ``pending`` to the UI.
    """
    scheduler = remote.get("scheduler", "slurm")
    if scheduler not in POLL_COMMANDS:
        return {"job_id": job_id, "scheduler": scheduler, "state": "unknown", "raw": ""}
    base_cmd = POLL_COMMANDS[scheduler] + [str(job_id)]
    command = " ".join(shlex.quote(part) for part in base_cmd)
    try:
        res = remote_command(remote, command, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return {"job_id": job_id, "scheduler": scheduler, "state": "unknown", "raw": ""}
    result = _parse_poll_output(scheduler, job_id, res.returncode, res.stdout, res.stderr)
    if state_file and result.get("state") == "running":
        try:
            gated = remote_command(
                remote, f"cat {shlex.quote(state_file)} 2>/dev/null", timeout=15
            ).stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            gated = ""
        if gated == "pending":
            result["state"] = "pending"
    return result


def fetch_remote_results(
    remote: dict,
    remote_dir: str,
    local_dir: str,
    include_heavy: bool = False,
) -> dict:
    """Copy result files from the remote job directory back to the local one.

    The files stay on the remote machine — this pulls a copy so the local
    analysis buttons (report, DOS, trajectory, …) work. Heavy binaries
    (WAVECAR/CHGCAR/…) are skipped unless ``include_heavy`` is set.

    Returns {"local_dir", "remote_dir", "transferred": bool}.
    """
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    target = _ssh_target(remote)
    ssh_opts = _ssh_options(remote)
    src = f"{target}:{remote_dir.rstrip('/')}/"

    if shutil.which("rsync"):
        cmd = ["rsync", "-az"]
        if ssh_opts:
            cmd += ["-e", "ssh " + " ".join(shlex.quote(o) for o in ssh_opts)]
        if not include_heavy:
            for name in HEAVY_OUTPUTS:
                cmd += ["--exclude", name]
        cmd += [src, f"{local}/"]
        _run_checked(cmd, "rsync fetch")
        return {"local_dir": str(local), "remote_dir": remote_dir, "transferred": True}

    # scp fallback: pull a known set of result files, ignoring any that are absent.
    wanted = ["OUTCAR", "CONTCAR", "OSZICAR", "vasprun.xml", "run.log", "job.log",
              "DOSCAR", "EIGENVAL", "XDATCAR", "INCAR", "KPOINTS", "POSCAR", "LOCPOT",
              "pw.in", "pw.out", ".engine"]
    if include_heavy:
        wanted += HEAVY_OUTPUTS
    scp_opts: list[str] = []
    port = remote.get("port")
    if port:
        scp_opts += ["-P", str(port)]
    key = remote.get("ssh_key")
    if key:
        scp_opts += ["-i", str(Path(key).expanduser())]
    scp_opts += list(remote.get("ssh_options") or [])
    # A single scp call pulling each wanted file; missing files just warn (rc may be !=0).
    subprocess.run(
        ["scp", *scp_opts, *[f"{target}:{remote_dir.rstrip('/')}/{n}" for n in wanted], f"{local}/"],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    return {"local_dir": str(local), "remote_dir": remote_dir, "transferred": True}


def list_remote_jobs(remote: dict, root: str) -> list[dict]:
    """List VASP job directories that live on a remote machine, newest first.

    Walks up to two levels under ``root`` — flat ``root/<case>`` and nested
    ``root/<project>/<case>`` — and keeps directories that hold VASP I/O. Each
    entry carries the newest output mtime (so the UI can sort by date) and a
    coarse status. One ``ssh`` round-trip; raises RuntimeError on failure.
    """
    root = (root or "").rstrip("/") or "/"
    rq = shlex.quote(root)
    # POSIX sh: a job dir has any VASP I/O; emit
    # "<mtime>\t<o><v><z>\t<type><converged>\t<energy>\t<path>" per job, where
    # type is n(eb)/c(onvergence)/s(cf) and energy is the last OSZICAR E0=
    # (or "-"). Mirrors the local _result_calc_type / CONVERGENCE_MARKERS /
    # parse_outcar_summary semantics so remote rows fill the same columns.
    script = (
        f"r={rq}; "
        # One snapshot of every visible process's cwd, shared by all dirs below
        # (only this user's /proc entries are readable — VASP runs as this user).
        'cw=$(readlink /proc/[0-9]*/cwd 2>/dev/null); '
        'isjob() { [ -e "$1/OUTCAR" ] || [ -e "$1/vasprun.xml" ] || '
        '[ -e "$1/OSZICAR" ] || [ -e "$1/INCAR" ] || [ -e "$1/POSCAR" ] || '
        '[ -e "$1/pw.in" ]; }; '
        'emit() { d=$1; n=0; '
        'for f in OUTCAR vasprun.xml OSZICAR run.log CONTCAR INCAR POSCAR pw.out; do '
        '[ -e "$d/$f" ] && { t=$(stat -c %Y "$d/$f" 2>/dev/null || echo 0); '
        '[ "$t" -gt "$n" ] && n=$t; }; done; '
        '[ "$n" = 0 ] && n=$(stat -c %Y "$d" 2>/dev/null || echo 0); '
        'o=0; { [ -e "$d/OUTCAR" ] || [ -e "$d/pw.out" ]; } && o=1; '
        'v=0; [ -e "$d/vasprun.xml" ] && v=1; '
        'z=0; [ -e "$d/OSZICAR" ] && z=1; '
        # g: run actually finished (job.log / OUTCAR footer / QE JOB DONE / ASE
        # results — mirrors job_finished); k: killed via the stop button.
        'g=0; { [ -e "$d/job.log" ] || [ -e "$d/ase_results.json" ] || '
        'grep -q "General timing" "$d/OUTCAR" 2>/dev/null || '
        'grep -q "JOB DONE" "$d/pw.out" 2>/dev/null; } && g=1; '
        'k=0; [ -e "$d/.terminated" ] && k=1; '
        # q: waiting at the per-machine concurrency gate. Scheduler jobs place
        # their state in the job dir; detached jobs are visible via their local
        # marker/status endpoint until the remote engine allocates this dir.
        'q=0; [ "$(cat "$d/.vasp_auto_state" 2>/dev/null)" = pending ] && q=1; '
        # p: a .pid is recorded; a: the job is actually alive right now — its
        # recorded PID exists, or some live process has its cwd inside the dir
        # (real liveness, not just "output file exists"). pd: PID for display.
        'p=0; a=0; pd=-; '
        'if [ -f "$d/.pid" ]; then pd=$(cat "$d/.pid" 2>/dev/null); p=1; '
        'kill -0 "$pd" 2>/dev/null && a=1; fi; '
        # ponytail: word-splits $cw, so job paths with spaces are missed here;
        # the .pid check above still covers those.
        'if [ "$a" = 0 ]; then for cwd_i in $cw; do case "$cwd_i" in '
        '"$d"|"$d"/*) a=1;; esac; done; fi; '
        # ponytail: NEB detected via initial/final or 00+01 image dirs (VASP
        # always numbers images with two digits); good enough for a listing.
        'y=s; [ -d "$d/scf_convergence" ] && y=c; '
        '{ [ -e "$d/initial/POSCAR" ] && [ -e "$d/final/POSCAR" ]; } && y=n; '
        '{ [ -d "$d/00" ] && [ -d "$d/01" ]; } && y=n; '
        'c=0; grep -q -e "reached required accuracy" '
        '-e "aborting loop because EDIFF is reached" "$d/OUTCAR" 2>/dev/null && c=1; '
        "e=$(awk '$4==\"E0=\"{x=$5} END{print x}' \"$d/OSZICAR\" 2>/dev/null); "
        '[ -n "$e" ] || e=-; '
        'printf "%s\\t%s%s%s%s%s%s%s%s\\t%s%s\\t%s\\t%s\\t%s\\n" "$n" "$o" "$v" "$z" "$g" "$k" "$p" "$a" "$q" "$y" "$c" "$e" "$pd" "$d"; }; '
        'for d in "$r"/*/; do d=${d%/}; [ -d "$d" ] || continue; '
        'if isjob "$d"; then emit "$d"; else '
        'for s in "$d"/*/; do s=${s%/}; [ -d "$s" ] || continue; '
        'isjob "$s" && emit "$s"; done; fi; done'
    )
    try:
        res = remote_command(remote, script, timeout=60)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(f"could not reach {remote.get('host')}: {exc}") from exc
    if res.returncode != 0 and not res.stdout.strip():
        raise RuntimeError(res.stderr.strip() or f"could not list {root} on {remote.get('host')}")
    rows: list[dict] = []
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 6:
            continue
        mt, flags, type_conv, energy, pid, path = parts
        flags = (flags + "00000000")[:8]
        has_out, has_vr, has_osz = flags[0] == "1", flags[1] == "1", flags[2] == "1"
        has_log, killed = flags[3] == "1", flags[4] == "1"
        has_pid, alive = flags[5] == "1", flags[6] == "1"
        pending = flags[7] == "1"
        type_conv = (type_conv + "s0")[:2]
        rel = path[len(root):].lstrip("/") if path.startswith(root) else path
        # Same rule as _local_job_rows: terminated > finished > actually alive
        # (live PID or a process working in the dir) > output-without-a-process
        # = stalled. A created file alone never counts as "running".
        status = ("terminated" if killed else "done" if has_log
                  else "pending" if pending
                  else "running" if alive
                  else "stalled" if (has_pid or has_osz or has_out)
                  else "prepared")
        row = {
            "path": path,
            "name": Path(path).name,
            "rel": rel or Path(path).name,
            "modified_ts": int(mt) if mt.isdigit() else 0,
            "status": status,
            "has_outcar": has_out,
            "has_vasprun": has_vr,
            "calculation_type": {"n": "tss", "c": "convergence"}.get(type_conv[0], "scf"),
            "converged": type_conv[1] == "1",
        }
        if pid and pid != "-" and pid.isdigit():
            row["pid"] = int(pid)
        try:
            row["energy_eV"] = float(energy)
        except ValueError:
            pass  # "-": no OSZICAR ionic step yet (or a QE job)
        rows.append(row)
    rows.sort(key=lambda r: r["modified_ts"], reverse=True)
    return rows


def _last_oszicar_e0(path: Path) -> float | None:
    """Last ionic step's E0 from an OSZICAR (matches the awk in list_remote_jobs)."""
    try:
        last = None
        for line in path.read_text(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[3] == "E0=":
                last = parts[4]
        return float(last) if last is not None else None
    except (OSError, ValueError):
        return None


def _local_job_rows(root: str) -> list[dict]:
    """Local mirror of list_remote_jobs: job dirs under ``root`` (up to two
    levels), newest first, same row shape (name/status/calculation_type/energy).

    ponytail: duplicates the remote shell walk in Python so the CLI and UI can
    scan the local machine without importing the UI's server-side scanner.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    from vasp_auto.workflow import proc_cwds
    cwds = proc_cwds()  # one /proc snapshot shared by every row's liveness check
    io = ("OUTCAR", "vasprun.xml", "OSZICAR", "INCAR", "POSCAR", "pw.in")
    stamp = ("OUTCAR", "vasprun.xml", "OSZICAR", "run.log", "CONTCAR",
             "INCAR", "POSCAR", "pw.out")

    def is_job(d: Path) -> bool:
        return any((d / f).exists() for f in io)

    def row_for(d: Path) -> dict:
        mt = max((int((d / f).stat().st_mtime) for f in stamp if (d / f).exists()),
                 default=int(d.stat().st_mtime))
        has_out = (d / "OUTCAR").exists() or (d / "pw.out").exists()
        has_osz = (d / "OSZICAR").exists()
        calc = "scf"
        if (d / "scf_convergence").is_dir():
            calc = "convergence"
        if ((d / "initial" / "POSCAR").exists() and (d / "final" / "POSCAR").exists()) \
                or ((d / "00").is_dir() and (d / "01").is_dir()):
            calc = "tss"
        # Only "done" once the run has actually finished (job_finished), and only
        # "running" when something is actually alive — the recorded .pid, or a
        # live process working inside the dir. Output without a live process is
        # "stalled": a created file is not proof the job still runs. Killed via
        # the app → .terminated.
        from vasp_auto.workflow import _read_pid, job_finished, pid_alive
        pid = _read_pid(d)
        alive = (pid is not None and pid_alive(pid)) or \
            any(c == str(d) or c.startswith(str(d) + "/") for c in cwds)
        if (d / ".terminated").exists():
            status = "terminated"
        elif job_finished(d):
            status = "done"
        elif alive:
            status = "running"
        elif pid is not None or has_out or has_osz:
            status = "stalled"
        else:
            status = "prepared"
        r = {
            "path": str(d), "name": d.name,
            "rel": str(d.relative_to(base)) if d != base else d.name,
            "modified_ts": mt,
            "status": status,
            "has_outcar": has_out, "calculation_type": calc,
        }
        if pid is not None:
            r["pid"] = pid
        e = _last_oszicar_e0(d / "OSZICAR")
        if e is not None:
            r["energy_eV"] = e
        return r

    dirs: list[Path] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if is_job(d):
            dirs.append(d)
        else:  # a project folder holding job dirs
            dirs += [s for s in sorted(d.iterdir()) if s.is_dir() and is_job(s)]
    rows = [row_for(d) for d in dirs]
    rows.sort(key=lambda r: r["modified_ts"], reverse=True)
    return rows


def list_running_jobs(config: dict, running_only: bool = True,
                      machine: str | None = None) -> dict:
    """Jobs across the local machine and every configured remote, newest first.

    "running" means a process is actually alive in the job dir (recorded PID or
    a live cwd match); output files without a live process read "stalled".
    Remotes come from ``config['remotes']`` (config.yaml + the UI's remotes.json,
    already merged by load_config) plus a single ``config['remote']``. An
    unreachable machine lands in ``errors`` instead of aborting the whole
    listing. ``machine`` narrows the scan to "local" or one remote name
    (None/"all" = everything).
    """
    want = machine or "all"
    jobs: list[dict] = []
    errors: list[dict] = []
    if want in ("all", "local"):
        for row in _local_job_rows(str(config.get("jobs_root") or "")):
            row["machine"] = "local"
            jobs.append(row)
    machines = dict(config.get("remotes") or {})
    if config.get("remote"):
        machines.setdefault("default", config["remote"])
    if want not in ("all", "local"):
        machines = {want: machines[want]} if want in machines else {}
        if not machines:
            errors.append({"machine": want, "error": "unknown machine"})
    elif want == "local":
        machines = {}
    for name, entry in machines.items():
        root = (entry.get("remote_root") or "").strip()
        if not root:
            errors.append({"machine": name, "error": "no remote_root set"})
            continue
        try:
            for row in list_remote_jobs({**entry, "name": name}, root):
                row["machine"] = name
                jobs.append(row)
        except Exception as exc:  # unreachable / SSH error — keep the other machines
            errors.append({"machine": name, "error": str(exc)})
    if running_only:
        # "stalled" = had a PID that has since died without finishing; surface it
        # too so a job that silently crashed doesn't just vanish from the board.
        jobs = [j for j in jobs if j.get("status") in ("pending", "running", "stalled")]
    jobs.sort(key=lambda j: j.get("modified_ts", 0), reverse=True)
    return {"jobs": jobs, "errors": errors}


def list_remote_dir(remote: dict, path: str) -> dict:
    """List the immediate entries (files + subdirs) of a remote directory.

    Backs the per-job file browser: each entry has name/path/is_dir/size/mtime.
    """
    p = (path or "").rstrip("/") or "/"
    pq = shlex.quote(p)
    script = (
        f"d={pq}; for e in \"$d\"/* \"$d\"/.[!.]*; do [ -e \"$e\" ] || continue; "
        'if [ -d "$e" ]; then '
        'printf "d\\t0\\t%s\\t%s\\n" "$(stat -c %Y "$e" 2>/dev/null || echo 0)" "$e"; '
        'else '
        'printf "f\\t%s\\t%s\\t%s\\n" "$(stat -c %s "$e" 2>/dev/null || echo 0)" '
        '"$(stat -c %Y "$e" 2>/dev/null || echo 0)" "$e"; fi; done'
    )
    try:
        res = remote_command(remote, script, timeout=45)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(f"could not reach {remote.get('host')}: {exc}") from exc
    if res.returncode != 0 and not res.stdout.strip():
        raise RuntimeError(res.stderr.strip() or f"could not list {p} on {remote.get('host')}")
    entries: list[dict] = []
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        kind, size, mt, epath = parts
        entries.append({
            "name": Path(epath).name,
            "path": epath,
            "is_dir": kind == "d",
            "size": int(size) if size.isdigit() else 0,
            "modified_ts": int(mt) if mt.isdigit() else 0,
        })
    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    parent = str(Path(p).parent) if str(Path(p).parent) != p else None
    return {"path": p, "parent": parent, "entries": entries}


def list_remote_cases(remote: dict, path: str) -> dict:
    """List VASP case directories under a remote directory (one SSH round-trip).

    A "case" is a directory with a POSCAR file (single) or initial/POSCAR plus
    final/POSCAR (TSS/NEB). If ``path`` itself is a case it is returned alone;
    otherwise its immediate subdirectories are scanned. Returns
    {"path", "cases": [{"name", "path", "type"}]}.
    """
    p = (path or "").rstrip("/") or "/"
    pq = shlex.quote(p)
    script = (
        f"r={pq}; "
        'emit() { if [ -f "$1/POSCAR" ]; then printf "scf\\t%s\\n" "$1"; '
        'elif [ -f "$1/initial/POSCAR" ] && [ -f "$1/final/POSCAR" ]; then '
        'printf "tss\\t%s\\n" "$1"; fi; }; '
        'if [ -f "$r/POSCAR" ] || { [ -f "$r/initial/POSCAR" ] && [ -f "$r/final/POSCAR" ]; }; '
        'then emit "$r"; '
        'else for d in "$r"/*/; do d="${d%/}"; [ -d "$d" ] && emit "$d"; done; fi'
    )
    try:
        res = remote_command(remote, script, timeout=45)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(f"could not reach {remote.get('host')}: {exc}") from exc
    cases: list[dict] = []
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        ctype, cpath = parts
        cases.append({"name": cpath.rstrip("/").rsplit("/", 1)[-1], "path": cpath, "type": ctype})
    cases.sort(key=lambda c: c["name"].lower())
    return {"path": p, "cases": cases}


def fetch_remote_file(remote: dict, remote_path: str, local_path) -> Path:
    """Copy one file off the remote machine (for a UI download). Returns the path."""
    local = Path(local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    target = _ssh_target(remote)
    scp_opts: list[str] = []
    port = remote.get("port")
    if port:
        scp_opts += ["-P", str(port)]
    key = remote.get("ssh_key")
    if key:
        scp_opts += ["-i", str(Path(key).expanduser())]
    scp_opts += list(remote.get("ssh_options") or [])
    # Single-quote the remote path so the remote shell takes it literally.
    cmd = ["scp", *scp_opts, f"{target}:{shlex.quote(remote_path)}", str(local)]
    _run_checked(cmd, "scp file")
    return local


def read_remote_text(remote: dict, remote_path: str, max_bytes: int = 200_000) -> dict:
    """Read up to ``max_bytes`` of a remote text file over SSH (for the UI viewer).

    Returns ``{"text", "size", "truncated"}``. One ``ssh`` round-trip: it prints
    the byte size, a marker, then a capped ``head`` of the file. Callers should
    only pass text files (binary content is mangled by text-mode decoding).
    """
    q = shlex.quote(remote_path)
    marker = "==VASP_AUTO_SPLIT=="
    script = f"wc -c < {q} 2>/dev/null || echo 0; echo {marker}; head -c {int(max_bytes)} {q} 2>/dev/null"
    try:
        res = remote_command(remote, script, timeout=45)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(f"could not reach {remote.get('host')}: {exc}") from exc
    out = res.stdout
    if (marker + "\n") in out:
        head, text = out.split(marker + "\n", 1)
    elif marker in out:
        head, text = out.split(marker, 1)
    else:
        head, text = "", out
    try:
        size = int(head.strip().splitlines()[-1])
    except (ValueError, IndexError):
        size = len(text.encode("utf-8", "replace"))
    return {"text": text, "size": size, "truncated": size > max_bytes}
