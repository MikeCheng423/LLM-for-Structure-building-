import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from vasp_auto.runner import (
    _local_core_counts,
    _mpi_command,
    _remote_engine_paths,
    _remote_vasp_exe,
    fetch_remote_file,
    fetch_remote_results,
    list_remote_dir,
    list_remote_jobs,
    list_running_jobs,
    poll_detached_job,
    read_remote_text,
    resolve_detached_job_dir,
    poll_job_status,
    poll_remote_job,
    detect_remote_run_mode,
    remote_run_mode,
    resolve_remote_run_mode,
    run_vasp_remote,
    submit_job_detached,
    submit_job_remote,
    check_remote_connection,
    write_submit_script,
)


# ---------------------------------------------------------------- mpi command / oversubscribe

def _cores(monkeypatch, physical, logical):
    """Pin the detected (physical, logical) core counts for a test."""
    monkeypatch.setattr("vasp_auto.runner._local_core_counts", lambda: (physical, logical))


def test_mpi_command_within_cores(monkeypatch):
    """Ranks <= physical cores: a plain mpirun command, no warning, no flags."""
    monkeypatch.delenv("VASP_AUTO_OVERSUBSCRIBE", raising=False)
    _cores(monkeypatch, 8, 16)
    cmd, warning = _mpi_command("mpirun", 4, "vasp_std")
    assert cmd == ["mpirun", "-np", "4", "vasp_std"]
    assert warning is None


def test_mpi_command_uses_hwthreads_past_physical(monkeypatch):
    """Ranks past physical cores but within threads -> --use-hwthread-cpus.

    This is the 8-core/16-thread machine asking for 16 ranks: it should run on
    the hardware threads without genuine oversubscription.
    """
    monkeypatch.delenv("VASP_AUTO_OVERSUBSCRIBE", raising=False)
    _cores(monkeypatch, 8, 16)
    cmd, warning = _mpi_command("mpirun", 16, "vasp_std")
    assert cmd == ["mpirun", "-np", "16", "--use-hwthread-cpus", "vasp_std"]
    assert warning and "hardware threads" in warning and "8 physical" in warning


def test_mpi_command_oversubscribes_past_threads(monkeypatch):
    """Ranks beyond even the hardware threads -> --oversubscribe and warns."""
    monkeypatch.delenv("VASP_AUTO_OVERSUBSCRIBE", raising=False)
    _cores(monkeypatch, 8, 16)
    cmd, warning = _mpi_command("mpirun", 32, "vasp_std")
    assert cmd == ["mpirun", "-np", "32", "--oversubscribe", "vasp_std"]
    assert warning and "--oversubscribe" in warning


def test_mpi_command_env_disables_flags(monkeypatch):
    """VASP_AUTO_OVERSUBSCRIBE=0 adds no flag even when over-committed, but warns."""
    monkeypatch.setenv("VASP_AUTO_OVERSUBSCRIBE", "0")
    _cores(monkeypatch, 8, 16)
    cmd, warning = _mpi_command("mpirun", 16, "vasp_std")
    assert cmd == ["mpirun", "-np", "16", "vasp_std"]
    assert warning and "lower CPU cores" in warning


def test_mpi_command_env_forces_oversubscribe(monkeypatch):
    """VASP_AUTO_OVERSUBSCRIBE=1 adds --oversubscribe even when ranks fit."""
    monkeypatch.setenv("VASP_AUTO_OVERSUBSCRIBE", "1")
    _cores(monkeypatch, 8, 16)
    cmd, warning = _mpi_command("mpirun", 4, "vasp_std")
    assert "--oversubscribe" in cmd
    assert warning is None  # not over-committed, so no warning


def test_mpi_command_passes_exe_args(monkeypatch):
    """Extra executable args (QE's -in pw.in) are appended after the exe."""
    monkeypatch.delenv("VASP_AUTO_OVERSUBSCRIBE", raising=False)
    _cores(monkeypatch, 8, 16)
    cmd, _ = _mpi_command("mpirun", 2, "pw.x", "-in", "pw.in")
    assert cmd == ["mpirun", "-np", "2", "pw.x", "-in", "pw.in"]


def test_mpi_command_skips_flag_for_non_mpirun(monkeypatch):
    """The OpenMPI flags are not added for mpiexec launchers, but it still warns."""
    monkeypatch.delenv("VASP_AUTO_OVERSUBSCRIBE", raising=False)
    _cores(monkeypatch, 4, 8)
    cmd, warning = _mpi_command("mpiexec", 16, "vasp_std")
    assert cmd == ["mpiexec", "-np", "16", "vasp_std"]
    assert warning is not None  # still warns about the over-request


def test_local_core_counts_returns_positive():
    """The detector returns sane (physical <= logical) positive counts."""
    physical, logical = _local_core_counts()
    assert physical >= 1 and logical >= 1
    assert physical <= logical


def test_write_submit_script_slurm(tmp_path):
    script = write_submit_script(str(tmp_path), "/opt/vasp/vasp_std", cpus=16, scheduler="slurm")
    text = script.read_text()
    assert script.name == "submit.sh"
    assert f"--job-name={tmp_path.name}" in text
    assert "--ntasks=16" in text
    assert 'mpirun -np 16 "/opt/vasp/vasp_std"' in text


def test_write_submit_script_pbs_with_options(tmp_path):
    script = write_submit_script(
        str(tmp_path),
        "vasp_std",
        cpus=8,
        scheduler="pbs",
        options=["#PBS -q standby", "module load vasp"],
    )
    text = script.read_text()
    assert "#PBS -N" in text
    assert "#PBS -q standby" in text
    assert "module load vasp" in text


def test_write_submit_script_custom_template(tmp_path):
    template = tmp_path / "template.sh"
    template.write_text("#!/bin/bash\n# {job_name} on {cpus} cores\nmpirun -np {cpus} {exe}\n")
    script = write_submit_script(
        str(tmp_path), "vasp_std", cpus=4, scheduler="slurm", template_path=str(template)
    )
    assert f"# {tmp_path.name} on 4 cores" in script.read_text()


def test_write_submit_script_rejects_unknown_scheduler(tmp_path):
    with pytest.raises(ValueError):
        write_submit_script(str(tmp_path), "vasp_std", scheduler="lsf")


# ---------------------------------------------------------------- TASK 4: poll_job_status

def _fake_run_slurm_running(cmd, **kwargs):
    """Simulate squeue output for a running job (R = running)."""
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=0,
        stdout="123456  debug  vasp  user  R  0:10  1  node01\n",
        stderr="",
    )


def _fake_run_slurm_pending(cmd, **kwargs):
    return subprocess.CompletedProcess(
        args=cmd, returncode=0,
        stdout="123456  debug  vasp  user  PD  0:00  1  (None)\n",
        stderr="",
    )


def _fake_run_slurm_absent(cmd, **kwargs):
    """squeue returns no output for a completed job."""
    return subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout="", stderr="",
    )


def _fake_run_pbs_queued(cmd, **kwargs):
    return subprocess.CompletedProcess(
        args=cmd, returncode=0,
        stdout="                                                            Req'd  Req'd   Elap\n"
               "Job ID          Username Queue    Jobname SessID NDS TSK Memory Time  S Time\n"
               "--------------- -------- -------- ------- ------ --- --- ------ ----- - -----\n"
               "456.cluster     user     standby  vasp    --     1   16  --     01:00 Q --\n",
        stderr="",
    )


def _fake_run_pbs_not_found(cmd, **kwargs):
    """qstat fails when job is gone (completed and purged)."""
    return subprocess.CompletedProcess(
        args=cmd, returncode=1,
        stdout="", stderr="qstat: Unknown Job Id 456.cluster\n",
    )


def test_poll_job_status_slurm_running():
    with patch("shutil.which", return_value="/usr/bin/squeue"), \
         patch("subprocess.run", side_effect=_fake_run_slurm_running):
        result = poll_job_status("123456", scheduler="slurm")
    assert result["state"] == "running"
    assert result["job_id"] == "123456"
    assert result["scheduler"] == "slurm"


def test_poll_job_status_slurm_pending():
    with patch("shutil.which", return_value="/usr/bin/squeue"), \
         patch("subprocess.run", side_effect=_fake_run_slurm_pending):
        result = poll_job_status("123456", scheduler="slurm")
    assert result["state"] == "pending"


def test_poll_job_status_slurm_completed():
    """Empty squeue output → completed."""
    with patch("shutil.which", return_value="/usr/bin/squeue"), \
         patch("subprocess.run", side_effect=_fake_run_slurm_absent):
        result = poll_job_status("123456", scheduler="slurm")
    assert result["state"] == "completed"


def test_poll_job_status_pbs_pending():
    with patch("shutil.which", return_value="/usr/bin/qstat"), \
         patch("subprocess.run", side_effect=_fake_run_pbs_queued):
        result = poll_job_status("456.cluster", scheduler="pbs")
    assert result["state"] == "pending"


def test_poll_job_status_pbs_completed():
    """qstat fails for a gone job → completed."""
    with patch("shutil.which", return_value="/usr/bin/qstat"), \
         patch("subprocess.run", side_effect=_fake_run_pbs_not_found):
        result = poll_job_status("456.cluster", scheduler="pbs")
    assert result["state"] == "completed"


def test_poll_job_status_binary_absent():
    """When squeue is not on PATH → state = unknown."""
    with patch("shutil.which", return_value=None):
        result = poll_job_status("123456", scheduler="slurm")
    assert result["state"] == "unknown"
    assert result["job_id"] == "123456"


def test_poll_job_status_unknown_scheduler():
    result = poll_job_status("99", scheduler="lsf")
    assert result["state"] == "unknown"


# ---------------------------------------------------------------- remote submit

def _make_job_dir(tmp_path):
    job = tmp_path / "Au13"
    job.mkdir()
    (job / "INCAR").write_text("ENCUT = 400\n")
    (job / "POSCAR").write_text("Au\n")
    (job / "KPOINTS").write_text("auto\n")
    (job / "POTCAR").write_text("PAW_PBE Au\n")
    return job


def test_write_submit_script_remote_run_dir(tmp_path):
    """run_dir overrides the cd path so the script targets the remote directory."""
    script = write_submit_script(
        str(tmp_path), "/opt/vasp/vasp_std", cpus=8, scheduler="slurm",
        run_dir="/scratch/me/jobs/Au13",
    )
    text = script.read_text()
    assert 'cd "/scratch/me/jobs/Au13"' in text
    assert str(tmp_path) not in text  # local path must not leak into the script


def test_submit_job_remote_validates_config(tmp_path):
    job = _make_job_dir(tmp_path)
    with pytest.raises(ValueError, match="host"):
        submit_job_remote(str(job), {"remote_root": "/x", "vasp_executable": "v"})
    with pytest.raises(ValueError, match="remote_root"):
        submit_job_remote(str(job), {"host": "h", "vasp_executable": "v"})
    with pytest.raises(ValueError, match="vasp_executable"):
        submit_job_remote(str(job), {"host": "h", "remote_root": "/x"})


def test_submit_job_remote_slurm(tmp_path):
    job = _make_job_dir(tmp_path)
    remote = {
        "host": "cluster.edu",
        "user": "me",
        "port": 2222,
        "remote_root": "/scratch/me/jobs",
        "vasp_executable": "/opt/vasp/vasp_std",
        "scheduler": "slurm",
    }
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out = "Submitted batch job 98765" if "sbatch" in " ".join(cmd) else ""
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=out, stderr="")

    # Force the scp fallback path so the test does not depend on rsync presence.
    with patch("shutil.which", return_value=None), \
         patch("subprocess.run", side_effect=fake_run):
        result = submit_job_remote(str(job), remote, cpus=16)

    assert result["job_id"] == "98765"
    assert result["scheduler"] == "slurm"
    assert result["host"] == "cluster.edu"
    assert result["remote_dir"] == "/scratch/me/jobs/results/Au13"

    joined = [" ".join(c) for c in calls]
    assert any(c.startswith("ssh") and "mkdir -p" in c for c in joined)
    assert any(c.startswith("scp") and "-P 2222" in c for c in joined)
    assert any("cd /scratch/me/jobs/results/Au13 && sbatch submit.sh" in c for c in joined)

    # The written submit.sh must cd into the remote dir, with the remote exe.
    script = (job / "submit.sh").read_text()
    assert 'cd "/scratch/me/jobs/results/Au13"' in script
    assert "/opt/vasp/vasp_std" in script


def test_submit_job_remote_uses_rsync_when_available(tmp_path):
    job = _make_job_dir(tmp_path)
    remote = {
        "host": "cluster.edu",
        "remote_root": "/work",
        "vasp_executable": "vasp_std",
    }
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out = "Submitted batch job 1" if "sbatch" in " ".join(cmd) else ""
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=out, stderr="")

    with patch("shutil.which", return_value="/usr/bin/rsync"), \
         patch("subprocess.run", side_effect=fake_run):
        submit_job_remote(str(job), remote)

    joined = [" ".join(c) for c in calls]
    assert any(c.startswith("rsync") for c in joined)
    # bare host (no user) → no '@' in the destination
    assert any("cluster.edu:/work/results/Au13" in c for c in joined)


def test_submit_job_remote_raises_on_failure(tmp_path):
    job = _make_job_dir(tmp_path)
    remote = {"host": "h", "remote_root": "/w", "vasp_executable": "v"}

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom")

    with patch("shutil.which", return_value=None), \
         patch("subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="boom"):
            submit_job_remote(str(job), remote)


def test_submit_job_remote_writes_marker(tmp_path):
    """A remote submission tags the local job dir with .remote.json."""
    import json as _json
    job = _make_job_dir(tmp_path)
    remote = {"host": "c.edu", "name": "cluster1", "remote_root": "/scratch",
              "vasp_executable": "vasp_std", "scheduler": "slurm"}

    def fake_run(cmd, **kwargs):
        out = "Submitted batch job 42" if "sbatch" in " ".join(cmd) else ""
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=out, stderr="")

    with patch("shutil.which", return_value="/usr/bin/rsync"), \
         patch("subprocess.run", side_effect=fake_run):
        submit_job_remote(str(job), remote)

    marker = _json.loads((job / ".remote.json").read_text())
    assert marker["machine"] == "cluster1"
    assert marker["job_id"] == "42"
    assert marker["remote_dir"] == "/scratch/results/Au13"


# ---------------------------------------------------------------- direct-SSH run mode

def test_remote_run_mode_resolution():
    assert remote_run_mode({"run_mode": "ssh", "scheduler": "slurm"}) == "ssh"
    assert remote_run_mode({"run_mode": "direct"}) == "ssh"
    assert remote_run_mode({"run_mode": "slurm"}) == "slurm"
    assert remote_run_mode({"run_mode": "ssh_detached"}) == "ssh_detached"
    assert remote_run_mode({"run_mode": "detached"}) == "ssh_detached"
    assert remote_run_mode({"run_mode": "offload"}) == "ssh_detached"
    assert remote_run_mode({"scheduler": "slurm"}) == "slurm"
    assert remote_run_mode({"scheduler": "pbs"}) == "pbs"
    assert remote_run_mode({"scheduler": "ssh"}) == "ssh"
    assert remote_run_mode({"scheduler": "none"}) == "ssh"
    assert remote_run_mode({}) == "slurm"  # safe default preserves prior behaviour


def test_resolve_remote_run_mode_honours_explicit():
    """An explicit run_mode/scheduler wins and never triggers an SSH probe."""
    with patch("vasp_auto.runner.detect_remote_run_mode",
               side_effect=AssertionError("must not probe when pinned")):
        assert resolve_remote_run_mode({"run_mode": "ssh"}) == "ssh"
        assert resolve_remote_run_mode({"run_mode": "ssh_detached"}) == "ssh_detached"
        assert resolve_remote_run_mode({"scheduler": "slurm"}) == "slurm"


def test_resolve_remote_run_mode_detects_and_caches():
    """With nothing pinned, probe once and reuse the cached answer."""
    remote = {"host": "wkstn"}
    with patch("vasp_auto.runner.detect_remote_run_mode", return_value="slurm") as det:
        assert resolve_remote_run_mode(remote) == "slurm"
        assert resolve_remote_run_mode(remote) == "slurm"  # cached, no second probe
    det.assert_called_once()
    assert remote["_run_mode_detected"] == "slurm"


def test_detect_remote_run_mode_reads_probe():
    remote = {"host": "wkstn"}

    def fake_run(cmd, **kwargs):
        out = "slurm\n" if "sbatch" in " ".join(cmd) else ""
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=out, stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        assert detect_remote_run_mode(remote) == "slurm"

    # No scheduler on the box -> offload; an unreachable host -> offload too.
    with patch("subprocess.run",
               return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="none\n", stderr="")):
        assert detect_remote_run_mode(remote) == "ssh_detached"
    with patch("subprocess.run", side_effect=OSError("no route to host")):
        assert detect_remote_run_mode(remote) == "ssh_detached"


def test_remote_engine_paths():
    paths = _remote_engine_paths({"remote_root": "/work/me/"})
    assert paths["root"] == "/work/me"
    assert paths["home"] == "/work/me/.vasp_auto"
    assert paths["vasp_auto"] == "/work/me/.vasp_auto/venv/bin/vasp-auto"
    assert paths["runs"] == "/work/me/.vasp_auto/runs"


def test_submit_job_detached_ships_launches_and_marks(tmp_path):
    """Offload submit: ships config + bundle + run.sh, launches setsid, writes marker."""
    bundle = tmp_path / "H2O"
    bundle.mkdir()
    (bundle / "POSCAR").write_text("H2O\n")
    (bundle / "POTCAR").write_text("PAW_PBE H\n")
    local_job = tmp_path / "jobs" / "H2O"
    remote = {"host": "wkstn", "name": "wkstn", "remote_root": "/work",
              "vasp_executable": "/opt/vasp/bin/vasp_std", "run_mode": "ssh_detached",
              "env_setup": "source /opt/intel/oneapi/setvars.sh"}
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(" ".join(cmd))
        # the launch command ends by cat-ing the pid file
        out = "44213" if "setsid bash" in " ".join(cmd) else ""
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=out, stderr="")

    import vasp_auto.runner as runner_mod
    shipped = {}
    real_ship = runner_mod._ship_file

    def cap_ship(local, target, remote_path, rem, ssh_opts):
        if remote_path.endswith("/run.sh"):
            shipped["run_sh"] = Path(local).read_text()
        real_ship(local, target, remote_path, rem, ssh_opts)

    with patch("shutil.which", return_value="/usr/bin/rsync"), \
         patch("subprocess.run", side_effect=fake_run), \
         patch("vasp_auto.runner._ship_file", side_effect=cap_ship), \
         patch("vasp_auto.runner.remote_engine_installed", return_value=True):
        result = submit_job_detached(
            case_dir=str(bundle), remote=remote, case_name="H2O", cpus=8,
            calc_flags=["--converge-encut", "300,400", "--energy-tol", "0.001"],
            local_job_dir=str(local_job),
        )

    assert result["pid"] == "44213"
    assert result["remote_dir"] == "/work/results/H2O"
    assert result["mode"] == "ssh_detached"
    joined = "\n".join(calls)
    # config.yaml + inputs bundle + run.sh shipped, and a detached launch happened
    assert "/work/config.yaml" in joined
    assert "wkstn:/work/inputs/H2O/" in joined
    assert "setsid bash" in joined
    # INCAR templates shipped + VASP_AUTO_ROOT exported so any calc type works remotely
    assert "/work/.vasp_auto/example" in joined
    assert "export VASP_AUTO_ROOT=/work/.vasp_auto" in shipped["run_sh"]
    assert "--converge-encut" in shipped["run_sh"]
    # the engine records its real (numbered) job root here for later resolution
    assert "export VASP_AUTO_JOBDIR_FILE=/work/.vasp_auto/runs/H2O/job_dir" in shipped["run_sh"]
    # marker tags the machine, control dir and pid for later polling
    marker = json.loads((local_job / ".remote.json").read_text())
    assert marker["machine"] == "wkstn"
    assert marker["control_dir"] == "/work/.vasp_auto/runs/H2O"
    assert marker["pid"] == "44213"


def test_submit_job_detached_raises_when_autosetup_fails(tmp_path):
    """A missing engine that cannot be auto-installed surfaces the failure detail."""
    bundle = tmp_path / "H2O"
    bundle.mkdir()
    (bundle / "POSCAR").write_text("H2O\n")
    remote = {"host": "wkstn", "remote_root": "/work",
              "vasp_executable": "/opt/vasp/bin/vasp_std", "run_mode": "ssh_detached"}
    with patch("vasp_auto.runner.remote_engine_installed", return_value=False), \
         patch("vasp_auto.runner.setup_remote_engine",
               return_value={"ok": False, "detail": "pip blew up"}):
        with pytest.raises(RuntimeError, match="pip blew up"):
            submit_job_detached(case_dir=str(bundle), remote=remote, case_name="H2O",
                                cpus=4, calc_flags=[])


def test_submit_job_detached_autoinstalls_engine(tmp_path):
    """First offload to a fresh machine self-installs the engine, no --remote-setup."""
    bundle = tmp_path / "H2O"
    bundle.mkdir()
    (bundle / "POSCAR").write_text("H2O\n")
    remote = {"host": "wkstn", "remote_root": "/work",
              "vasp_executable": "/opt/vasp/bin/vasp_std", "run_mode": "ssh_detached"}

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="44213", stderr="")

    with patch("vasp_auto.runner.remote_engine_installed", return_value=False), \
         patch("vasp_auto.runner.setup_remote_engine",
               return_value={"ok": True, "detail": ""}) as setup, \
         patch("shutil.which", return_value="/usr/bin/rsync"), \
         patch("subprocess.run", side_effect=fake_run):
        submit_job_detached(case_dir=str(bundle), remote=remote, case_name="H2O",
                            cpus=4, calc_flags=[])
    setup.assert_called_once()


def test_resolve_detached_job_dir_reads_control_file():
    """The real numbered job root is read back from <control_dir>/job_dir over SSH."""
    remote = {"host": "wkstn", "user": "u", "remote_root": "/work"}

    def fake_run(cmd, **kwargs):
        assert "cat" in " ".join(cmd) and "/work/.vasp_auto/runs/H2O/job_dir" in " ".join(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0,
                                           stdout="/work/results/0003_H2O\n", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        got = resolve_detached_job_dir(remote, "/work/.vasp_auto/runs/H2O")
    assert got == "/work/results/0003_H2O"


def test_resolve_detached_job_dir_returns_none_when_absent():
    """No control dir, or the file not yet written, resolves to None (caller falls back)."""
    remote = {"host": "wkstn", "remote_root": "/work"}

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        assert resolve_detached_job_dir(remote, "/work/.vasp_auto/runs/H2O") is None
    assert resolve_detached_job_dir(remote, "") is None


def test_poll_detached_job_states():
    remote = {"host": "wkstn", "remote_root": "/work"}

    def make(out):
        def f(cmd, **k):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=out, stderr="")
        return f

    with patch("subprocess.run", side_effect=make("DONE\n0\n")):
        s = poll_detached_job(remote, "/work/.vasp_auto/runs/H2O", "123")
        assert s["state"] == "completed" and s["return_code"] == "0"
    with patch("subprocess.run", side_effect=make("RUNNING\n")):
        s = poll_detached_job(remote, "/work/.vasp_auto/runs/H2O", "123")
        assert s["state"] == "running"
    with patch("subprocess.run", side_effect=make("UNKNOWN\n")):
        s = poll_detached_job(remote, "/work/.vasp_auto/runs/H2O", None)
        assert s["state"] == "unknown"


def test_remote_vasp_exe_handles_directory():
    # A directory path falls back to the standard binary; a real binary is kept.
    assert _remote_vasp_exe({"vasp_executable": "/opt/vasp/bin"}) == "/opt/vasp/bin/vasp_std"
    assert _remote_vasp_exe({"vasp_executable": "/opt/vasp/bin/"}) == "/opt/vasp/bin/vasp_std"
    assert _remote_vasp_exe({"vasp_executable": "/opt/vasp/bin/vasp_std"}) == "/opt/vasp/bin/vasp_std"
    assert _remote_vasp_exe({"vasp_executable": "/opt/vasp/bin/vasp_gam"}) == "/opt/vasp/bin/vasp_gam"
    with pytest.raises(ValueError, match="vasp_executable"):
        _remote_vasp_exe({})


def test_run_vasp_remote_ships_runs_and_marks(tmp_path):
    """Direct-SSH run: rsync up, mpirun over SSH, env sourced, then marker written."""
    job = _make_job_dir(tmp_path)
    remote = {
        "host": "wkstn", "user": "me", "name": "wkstn", "remote_root": "/work",
        "vasp_executable": "/opt/vasp/bin",  # directory -> /vasp_std
        "run_mode": "ssh", "env_setup": "source /opt/intel/oneapi/setvars.sh",
    }
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("shutil.which", return_value="/usr/bin/rsync"), \
         patch("subprocess.run", side_effect=fake_run), \
         patch("vasp_auto.runner.fetch_remote_results", return_value={}) as fetched:
        rc = run_vasp_remote(str(job), remote, cpus=8)

    assert rc == 0
    joined = [" ".join(c) for c in calls]
    # inputs shipped to remote_root/results/<job name>
    assert any(c.startswith("rsync") and "wkstn:/work/results/Au13/" in c for c in joined)
    # mpirun launched over SSH with the resolved binary, env sourced, ranks honoured
    run_cmd = next(c for c in joined if "mpirun" in c)
    assert "mpirun -np 8" in run_cmd
    assert "/opt/vasp/bin/vasp_std" in run_cmd
    assert "setvars.sh" in run_cmd
    # results pulled back so local parsers/viewers work
    fetched.assert_called_once()
    # marker tags the machine + remote dir, status comes out non-scheduler
    marker = json.loads((job / ".remote.json").read_text())
    assert marker["machine"] == "wkstn"
    assert marker["remote_dir"] == "/work/results/Au13"
    assert marker["mode"] == "ssh"


def test_run_vasp_remote_honours_remote_subdir(tmp_path):
    """remote_subdir keeps multi-case trials from colliding under remote_root."""
    job = _make_job_dir(tmp_path)
    remote = {"host": "wkstn", "remote_root": "/work",
              "vasp_executable": "/opt/vasp/bin/vasp_std", "run_mode": "ssh"}
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(" ".join(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("shutil.which", return_value="/usr/bin/rsync"), \
         patch("subprocess.run", side_effect=fake_run), \
         patch("vasp_auto.runner.fetch_remote_results", return_value={}):
        run_vasp_remote(str(job), remote, cpus=4, remote_subdir="H2O/scf_convergence/encut_400")

    assert any("/work/results/H2O/scf_convergence/encut_400" in c for c in calls)
    marker = json.loads((job / ".remote.json").read_text())
    assert marker["remote_dir"] == "/work/results/H2O/scf_convergence/encut_400"


# ---------------------------------------------------------------- remote connection / status / fetch

def test_test_remote_connection_all_ok():
    remote = {"host": "c.edu", "user": "me", "remote_root": "/scratch",
              "vasp_executable": "/opt/vasp_std", "scheduler": "slurm"}

    def fake_run(cmd, **kwargs):
        joined = " ".join(cmd)
        if "echo vasp_auto_ok" in joined:
            return subprocess.CompletedProcess(cmd, 0, "vasp_auto_ok\n", "")
        return subprocess.CompletedProcess(cmd, 0, "yes\n", "")  # all test -x/-d/command -v pass

    with patch("subprocess.run", side_effect=fake_run):
        res = check_remote_connection(remote)
    assert res["ok"] is True
    names = {c["name"]: c["ok"] for c in res["checks"]}
    assert names["ssh"] and names["vasp_executable"] and names["slurm"]


def test_test_remote_connection_ssh_fails():
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 255, "", "Permission denied (publickey)")

    with patch("subprocess.run", side_effect=fake_run):
        res = check_remote_connection({"host": "c.edu"})
    assert res["ok"] is False
    assert "denied" in res["message"].lower()


def test_test_remote_connection_missing_vasp():
    def fake_run(cmd, **kwargs):
        joined = " ".join(cmd)
        if "echo vasp_auto_ok" in joined:
            return subprocess.CompletedProcess(cmd, 0, "vasp_auto_ok\n", "")
        if "vasp_std" in joined:  # the test -x for the executable
            return subprocess.CompletedProcess(cmd, 0, "no\n", "")
        return subprocess.CompletedProcess(cmd, 0, "yes\n", "")

    remote = {"host": "c.edu", "vasp_executable": "/opt/vasp_std", "scheduler": "slurm"}
    with patch("subprocess.run", side_effect=fake_run):
        res = check_remote_connection(remote)
    assert res["ok"] is False
    names = {c["name"]: c["ok"] for c in res["checks"]}
    assert names["ssh"] is True and names["vasp_executable"] is False


def test_poll_remote_job_running():
    remote = {"host": "c.edu", "scheduler": "slurm"}

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, "999 debug vasp me R 1:00 1 n01\n", "")

    with patch("subprocess.run", side_effect=fake_run):
        res = poll_remote_job(remote, "999")
    assert res["state"] == "running"
    assert res["job_id"] == "999"


def test_fetch_remote_results_rsync(tmp_path):
    remote = {"host": "c.edu", "user": "me", "port": 2022}
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch("shutil.which", return_value="/usr/bin/rsync"), \
         patch("subprocess.run", side_effect=fake_run):
        res = fetch_remote_results(remote, "/scratch/Au13", str(tmp_path / "local"))

    assert res["transferred"] is True
    joined = " ".join(calls[0])
    assert joined.startswith("rsync")
    assert "--exclude WAVECAR" in joined  # heavy files skipped by default
    assert "me@c.edu:/scratch/Au13/" in joined


# ---------------------------------------------------------- remote browse + download

def test_list_remote_jobs_parses_and_sorts():
    """list_remote_jobs parses the TSV the remote script prints, newest first."""
    remote = {"host": "c.edu", "remote_root": "/work"}
    # Flags are o(utcar) v(asprun) z(oszicar) g(finished) k(illed) p(id present)
    # a(ctually alive); fields: mtime, flags, type+conv, energy, pid, path.
    out = (
        "1700000000\t1011000\ts1\t-27.8644\t-\t/work/Au13_relax\n"   # done + converged, oldest
        "1700009999\t0010000\ts0\t-\t-\t/work/Fe2O3_scf\n"           # OSZICAR, no live proc -> stalled
        "1700005000\t0000000\tn0\t-\t-\t/work/proj/NiO_prepared\n"   # prepared NEB (nested)
        "1700007000\t1010100\ts0\t-19.1\t-\t/work/CoO_killed\n"      # killed -> terminated
        "1700008000\t1000011\ts0\t-5.5\t12345\t/work/Cu_live\n"      # PID alive -> running
        "1700008200\t0010001\ts0\t-7.7\t-\t/work/Ni_cwd\n"           # live proc in dir -> running
        "1700008500\t1000010\ts0\t-6.6\t999999\t/work/Zn_stalled\n"  # PID dead -> stalled
        "garbage line without tabs\n"
    )
    with patch("vasp_auto.runner.remote_command",
               return_value=subprocess.CompletedProcess([], 0, out, "")):
        rows = list_remote_jobs(remote, "/work")

    st = {r["name"]: r["status"] for r in rows}
    assert st == {"Fe2O3_scf": "stalled", "Zn_stalled": "stalled", "Cu_live": "running",
                  "Ni_cwd": "running", "CoO_killed": "terminated",
                  "NiO_prepared": "prepared", "Au13_relax": "done"}
    assert rows[0]["modified_ts"] == 1700009999  # newest first
    done = next(r for r in rows if r["name"] == "Au13_relax")
    assert done["has_outcar"] is True and done["converged"] is True
    assert done["energy_eV"] == -27.8644
    live = next(r for r in rows if r["name"] == "Cu_live")
    assert live["pid"] == 12345
    nested = next(r for r in rows if r["name"] == "NiO_prepared")
    assert nested["rel"] == "proj/NiO_prepared" and nested["calculation_type"] == "tss"


def test_list_running_jobs_local_and_remote(tmp_path):
    """Aggregates local running job dirs + remotes; a down machine → errors."""
    import os
    (tmp_path / "0001_run" / "").mkdir(parents=True)
    (tmp_path / "0001_run" / "INCAR").write_text("X")
    (tmp_path / "0001_run" / "OSZICAR").write_text("   1 F= -.1E+02 E0= -.2345E+02  d E=0\n")
    (tmp_path / "0001_run" / ".pid").write_text(str(os.getpid()))  # alive -> running
    (tmp_path / "0002_done").mkdir()
    (tmp_path / "0002_done" / "OUTCAR").write_text(
        "reached required accuracy\n General timing and accounting informations for this run:\n")
    (tmp_path / "0003_stale").mkdir()   # output but nothing alive -> stalled, still on board
    (tmp_path / "0003_stale" / "OSZICAR").write_text("x")
    config = {
        "jobs_root": str(tmp_path),
        "remotes": {
            "gpu": {"host": "c.edu", "remote_root": "/work"},
            "noroot": {"host": "d.edu"},  # no remote_root → an error entry
        },
    }
    remote_out = "1700009999\t0010011\ts0\t-\t4242\t/work/Fe2O3_scf\n"  # alive on the remote
    with patch("vasp_auto.runner.remote_command",
               return_value=subprocess.CompletedProcess([], 0, remote_out, "")):
        res = list_running_jobs(config)

    names = {(j["machine"], j["name"]) for j in res["jobs"]}
    assert ("local", "0001_run") in names        # local running job included
    assert ("gpu", "Fe2O3_scf") in names         # remote running job included
    assert all(j["name"] != "0002_done" for j in res["jobs"])  # finished filtered out
    local = next(j for j in res["jobs"] if j["name"] == "0001_run")
    assert local["status"] == "running" and local["energy_eV"] == -23.45
    assert local["pid"] == os.getpid()
    stale = next(j for j in res["jobs"] if j["name"] == "0003_stale")
    assert stale["status"] == "stalled"          # files alone don't mean running
    assert {"machine": "noroot", "error": "no remote_root set"} in res["errors"]

    # running_only=False also surfaces the finished local job.
    with patch("vasp_auto.runner.remote_command",
               return_value=subprocess.CompletedProcess([], 0, remote_out, "")):
        every = list_running_jobs(config, running_only=False)
    assert any(j["name"] == "0002_done" and j["status"] == "done" for j in every["jobs"])

    # machine= narrows the scan: local-only never SSHes, one remote skips local.
    local_only = list_running_jobs(config, machine="local")
    assert all(j["machine"] == "local" for j in local_only["jobs"])
    with patch("vasp_auto.runner.remote_command",
               return_value=subprocess.CompletedProcess([], 0, remote_out, "")):
        gpu_only = list_running_jobs(config, machine="gpu")
    assert {j["machine"] for j in gpu_only["jobs"]} == {"gpu"}


def test_list_running_jobs_survives_unreachable_remote(tmp_path):
    """An SSH failure on one machine is captured, not raised."""
    config = {"jobs_root": str(tmp_path),
              "remotes": {"gpu": {"host": "c.edu", "remote_root": "/work"}}}
    with patch("vasp_auto.runner.remote_command",
               side_effect=OSError("ssh: connection refused")):
        res = list_running_jobs(config)
    assert res["jobs"] == []
    assert res["errors"] and res["errors"][0]["machine"] == "gpu"


def test_list_remote_jobs_raises_on_failure():
    remote = {"host": "c.edu"}
    with patch("vasp_auto.runner.remote_command",
               return_value=subprocess.CompletedProcess([], 255, "", "ssh: no route")):
        with pytest.raises(RuntimeError, match="no route"):
            list_remote_jobs(remote, "/work")


def test_list_remote_dir_parses_entries_and_parent():
    remote = {"host": "c.edu"}
    out = (
        "f\t2048\t1700000000\t/work/Au13/OUTCAR\n"
        "d\t0\t1700000500\t/work/Au13/trial_1\n"
        "f\t11\t1700000100\t/work/Au13/INCAR\n"
    )
    with patch("vasp_auto.runner.remote_command",
               return_value=subprocess.CompletedProcess([], 0, out, "")):
        data = list_remote_dir(remote, "/work/Au13")

    assert data["path"] == "/work/Au13"
    assert data["parent"] == "/work"
    # directories sort before files, then by name
    assert [e["name"] for e in data["entries"]] == ["trial_1", "INCAR", "OUTCAR"]
    outcar = next(e for e in data["entries"] if e["name"] == "OUTCAR")
    assert outcar["is_dir"] is False and outcar["size"] == 2048
    assert data["entries"][0]["is_dir"] is True


def test_fetch_remote_file_scp_command(tmp_path):
    remote = {"host": "c.edu", "user": "me", "port": 2022}
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        fetch_remote_file(remote, "/work/Au13/OUTCAR", tmp_path / "OUTCAR")

    cmd = calls[0]
    assert cmd[0] == "scp"
    assert "-P" in cmd and "2022" in cmd
    assert any("me@c.edu:" in c and "OUTCAR" in c for c in cmd)


def test_fetch_remote_file_raises_on_failure(tmp_path):
    remote = {"host": "c.edu"}
    with patch("subprocess.run",
               return_value=subprocess.CompletedProcess([], 1, "", "No such file")):
        with pytest.raises(RuntimeError, match="No such file"):
            fetch_remote_file(remote, "/work/nope", tmp_path / "nope")


def test_read_remote_text_splits_size_and_body():
    """read_remote_text parses the '<size>\\n<marker>\\n<text>' the script prints."""
    remote = {"host": "c.edu"}
    out = "42\n==VASP_AUTO_SPLIT==\nENCUT = 520\nISMEAR = 0\n"
    with patch("vasp_auto.runner.remote_command",
               return_value=subprocess.CompletedProcess([], 0, out, "")):
        data = read_remote_text(remote, "/work/Au13/INCAR", max_bytes=100)
    assert data["text"] == "ENCUT = 520\nISMEAR = 0\n"
    assert data["size"] == 42
    assert data["truncated"] is False


def test_read_remote_text_flags_truncation():
    remote = {"host": "c.edu"}
    out = "500000\n==VASP_AUTO_SPLIT==\nbig file head…\n"
    with patch("vasp_auto.runner.remote_command",
               return_value=subprocess.CompletedProcess([], 0, out, "")):
        data = read_remote_text(remote, "/work/Au13/OUTCAR", max_bytes=200_000)
    assert data["truncated"] is True
    assert data["size"] == 500000
