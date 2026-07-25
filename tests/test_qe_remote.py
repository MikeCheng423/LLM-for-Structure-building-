"""QE remote parity: the same three remote modes VASP has (ssh, scheduler,
detached offload) drive a Quantum ESPRESSO job. No real SSH anywhere."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from vasp_auto.cli import _forward_calc_flags, parse_args
from vasp_auto.runner import run_vasp_remote, submit_job_detached, submit_job_remote

POSCAR_SI = "Si\n1.0\n5.43 0 0\n0 5.43 0\n0 0 5.43\nSi\n1\nDirect\n0 0 0\n"


def _args(argv):
    old = sys.argv
    sys.argv = ["vasp-auto", *argv]
    try:
        return parse_args()
    finally:
        sys.argv = old


def test_forward_calc_flags_resolved_engine_not_local_paths():
    """The resolved engine is forwarded; host-local QE paths never are."""
    args = _args(["inputs/Si", "--qe-executable", "/usr/local/bin/pw.x",
                  "--pseudo-dir", "/home/me/pseudo"])
    flags = _forward_calc_flags(args, engine="qe")
    assert flags == ["--engine", "qe"]
    # engine from config (args.engine unset) is still forwarded explicitly
    assert _forward_calc_flags(_args(["inputs/Si"]), engine="qe") == ["--engine", "qe"]
    assert _forward_calc_flags(_args(["inputs/Si"])) == []


def test_run_vasp_remote_qe_command(tmp_path):
    """engine="qe" runs pw.x with -in pw.in and keeps pw.out + run.log."""
    job = tmp_path / "0001_Si"
    job.mkdir()
    (job / "pw.in").write_text("&CONTROL\n/\n")
    remote = {"host": "wkstn", "remote_root": "/work", "qe_executable": "/opt/qe/bin/pw.x"}
    scripts = []

    def fake_run(cmd, **kwargs):
        scripts.append(" ".join(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("shutil.which", return_value="/usr/bin/rsync"), \
         patch("subprocess.run", side_effect=fake_run):
        rc = run_vasp_remote(str(job), remote, cpus=4, engine="qe")

    assert rc == 0
    joined = "\n".join(scripts)
    assert "mpirun -np 4 /opt/qe/bin/pw.x -in pw.in > pw.out" in joined
    assert "cp -f pw.out run.log" in joined
    # marker written so the UI knows where the job ran
    assert (job / ".remote.json").exists()


def test_qe_remote_run_skips_vasp_env_setup(tmp_path):
    """QE never sources the machine's VASP env_setup (an Intel MPI environment
    silently runs an OpenMPI pw.x as N duplicate serial jobs); it sources its
    own qe_env_setup when the machine sets one."""
    job = tmp_path / "0001_Si"
    job.mkdir()
    remote = {"host": "wkstn", "remote_root": "/work",
              "env_setup": "source /opt/intel/oneapi/setvars.sh"}
    scripts = []

    def fake_run(cmd, **kwargs):
        scripts.append(" ".join(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("shutil.which", return_value="/usr/bin/rsync"), \
         patch("subprocess.run", side_effect=fake_run):
        run_vasp_remote(str(job), remote, cpus=2, engine="qe")
        assert "setvars.sh" not in "\n".join(scripts)

        scripts.clear()
        run_vasp_remote(str(job), {**remote, "qe_env_setup": "module load qe"},
                        cpus=2, engine="qe")
        joined = "\n".join(scripts)
        assert "module load qe" in joined and "setvars.sh" not in joined

        scripts.clear()
        run_vasp_remote(str(job), {**remote, "vasp_executable": "/opt/vasp/vasp_std"},
                        cpus=2)
        assert "setvars.sh" in "\n".join(scripts)


def test_submit_job_remote_qe_script(tmp_path):
    """Scheduler submission writes the QE template and its own environment."""
    job = tmp_path / "0001_Si"
    job.mkdir()
    (job / "pw.in").write_text("&CONTROL\n/\n")
    remote = {"host": "cluster.edu", "remote_root": "/scratch",
              "qe_executable": "/opt/qe/bin/pw.x", "scheduler": "slurm",
              "env_setup": "source /opt/intel/vasp.sh",
              "qe_env_setup": "module load qe"}

    def fake_run(cmd, **kwargs):
        out = "Submitted batch job 7" if "sbatch" in " ".join(cmd) else ""
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=out, stderr="")

    with patch("shutil.which", return_value=None), \
         patch("subprocess.run", side_effect=fake_run):
        result = submit_job_remote(str(job), remote, cpus=8, engine="qe")

    assert result["job_id"] == "7"
    script = (job / "submit.sh").read_text()
    assert '"/opt/qe/bin/pw.x" -in pw.in > pw.out' in script
    assert "module load qe" in script
    assert "source /opt/intel/vasp.sh" not in script


def test_submit_job_detached_qe_config(tmp_path):
    """A QE offload writes qe_executable (+ the machine's pseudo_dir) into the
    remote config.yaml instead of vasp_executable."""
    bundle = tmp_path / "Si"
    bundle.mkdir()
    (bundle / "POSCAR").write_text(POSCAR_SI)
    remote = {"host": "wkstn", "remote_root": "/work", "run_mode": "ssh_detached",
              "qe_executable": "/opt/qe/bin/pw.x", "pseudo_dir": "/opt/qe/pseudo"}

    import vasp_auto.runner as runner_mod
    shipped = {}
    real_ship = runner_mod._ship_file

    def cap_ship(local, target, remote_path, rem, ssh_opts):
        if remote_path.endswith("/config.yaml"):
            shipped["config"] = Path(local).read_text()
        real_ship(local, target, remote_path, rem, ssh_opts)

    def fake_run(cmd, **kwargs):
        out = "99" if "setsid bash" in " ".join(cmd) else ""
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=out, stderr="")

    with patch("shutil.which", return_value="/usr/bin/rsync"), \
         patch("subprocess.run", side_effect=fake_run), \
         patch("vasp_auto.runner._ship_file", side_effect=cap_ship), \
         patch("vasp_auto.runner.remote_engine_installed", return_value=True):
        submit_job_detached(case_dir=str(bundle), remote=remote, case_name="Si",
                            cpus=4, calc_flags=["--engine", "qe"], engine="qe")

    cfg = shipped["config"]
    assert "qe_executable: /opt/qe/bin/pw.x" in cfg
    assert "pseudo_dir: /opt/qe/pseudo" in cfg
    assert "vasp_executable" not in cfg


def test_offload_bundles_qe_pseudos(tmp_path, monkeypatch):
    """Without a remote pseudo library the offload ships the local UPFs and
    points --pseudo-dir at the shipped copy; no POTCAR is built."""
    import vasp_auto.potcar_finder as potcar_finder
    import vasp_auto.runner as runner
    from vasp_auto.cli import _run_detached_offload
    from vasp_auto.job_manager import make_case_info

    case = tmp_path / "Si"
    case.mkdir()
    (case / "POSCAR").write_text(POSCAR_SI)
    pseudo = tmp_path / "pseudo"
    pseudo.mkdir()
    (pseudo / "Si.pbe-n-kjpaw_psl.1.0.0.UPF").write_text("<UPF/>\n")

    captured = {}

    def fake_submit(case_dir, remote, case_name, cpus, calc_flags, local_job_dir=None,
                    on_progress=None, engine="vasp"):
        root = Path(case_dir)
        captured["tree"] = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
        captured["flags"] = list(calc_flags)
        captured["engine"] = engine
        return {"machine": remote.get("name"), "remote_dir": "/r/" + case_name,
                "inputs_dir": "/r/in", "control_dir": "/r/ctl", "pid": "5", "log": "/r/log"}

    def no_potcar(**kw):
        raise AssertionError("build_potcar must not run for a QE offload")

    monkeypatch.setattr(runner, "submit_job_detached", fake_submit)
    monkeypatch.setattr(potcar_finder, "build_potcar", no_potcar)

    args = _args(["inputs/Si", "--engine", "qe", "--calc-type", "relax"])
    args.cpus = 4
    config = {"pseudo_dir": str(pseudo)}
    case_info = make_case_info(case, tmp_path / "jobs", single_mode=True)
    remote = {"name": "apl2", "host": "h", "remote_root": "/r", "run_mode": "ssh_detached"}

    _run_detached_offload(case, case_info, args, config, remote, None, None,
                          "single", "Si", engine="qe")

    assert captured["engine"] == "qe"
    assert "POSCAR" in captured["tree"]
    assert "pseudo/Si.pbe-n-kjpaw_psl.1.0.0.UPF" in captured["tree"]
    assert "POTCAR" not in captured["tree"]
    assert "--engine" in captured["flags"] and "qe" in captured["flags"]
    i = captured["flags"].index("--pseudo-dir")
    assert captured["flags"][i + 1] == "inputs/Si/pseudo"

    # With the machine's own library nothing is shipped and no flag is added.
    remote_lib = {**remote, "pseudo_dir": "/opt/qe/pseudo"}
    _run_detached_offload(case, case_info, args, config, remote_lib, None, None,
                          "single", "Si", engine="qe")
    assert "--pseudo-dir" not in captured["flags"]
    assert not any(p.startswith("pseudo") for p in captured["tree"])


def test_run_one_case_dispatches_qe_remote(tmp_path, monkeypatch):
    """run_one_case routes a QE job through the remote ssh path with engine="qe"."""
    import vasp_auto.workflow as wf

    job = tmp_path / "0001_Si"
    job.mkdir()
    (job / ".engine").write_text("qe\n")
    (job / "pw.in").write_text("&CONTROL\n/\n")
    seen = {}

    def fake_remote(job_dir, remote, cpus=None, on_progress=None, engine="vasp"):
        seen["engine"] = engine
        return 0

    monkeypatch.setattr(wf, "run_vasp_remote", fake_remote)
    case_info = {"case_name": "Si", "job_dir": str(job), "calculation_type": "scf"}
    remote = {"host": "wkstn", "remote_root": "/work", "run_mode": "ssh",
              "qe_executable": "pw.x"}

    row = wf.run_one_case("proj", "single", case_info, vasp_executable=None,
                          cpus=2, remote=remote, engine="qe")

    assert seen["engine"] == "qe"
    assert row["engine"] == "qe"
    assert row["return_code"] == 0


def test_run_one_case_still_rejects_remote_ase(tmp_path):
    import vasp_auto.workflow as wf

    case_info = {"case_name": "Si", "job_dir": str(tmp_path), "calculation_type": "scf"}
    with pytest.raises(ValueError, match="ase"):
        wf.run_one_case("proj", "single", case_info, vasp_executable=None,
                        remote={"host": "h", "remote_root": "/w"}, engine="ase")
