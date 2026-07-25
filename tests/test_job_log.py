"""Tests for the human-readable job.log summary."""

from vasp_auto.job_log import build_job_log, write_job_log


def test_job_log_finished(finished_job):
    (finished_job / "INCAR").write_text(
        "SYSTEM = Al test\nENCUT = 450\nIBRION = 2\nNSW = 50\nISMEAR = 1\nSIGMA = 0.2\n",
        encoding="utf-8",
    )
    text = build_job_log(finished_job, case_name="Al", calc_type="relax")
    assert "Status       : finished" in text
    assert "geometry optimization" in text
    assert "Cutoff ENCUT : 450 eV" in text
    assert "Methfessel-Paxton" in text
    assert "RESULTS" in text and "Final energy" in text


def test_job_log_failed(tmp_path):
    # No OUTCAR/energy + a fatal error signature → a 'failed' summary that still
    # surfaces the problem, so a failure leaves a readable result behind.
    (tmp_path / "INCAR").write_text("ENCUT = 400\nRWIGS = 1.34\n", encoding="utf-8")
    (tmp_path / "run.log").write_text(
        "Error reading item RWIGS from file INCAR.\n"
        "  ----> I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----\n",
        encoding="utf-8",
    )
    text = build_job_log(tmp_path, case_name="bad", return_code=1)
    assert "Status       : failed" in text
    assert "DETECTED PROBLEMS" in text and "SICK_JOB" in text
    assert "No final energy" in text


def test_write_job_log_writes_file(finished_job):
    path = write_job_log(finished_job, case_name="Al")
    assert path is not None and path.name == "job.log"
    assert path.read_text(encoding="utf-8").startswith("=")


def test_write_job_log_never_raises(tmp_path):
    # An empty directory must not raise — a summary failure can't break a run.
    assert write_job_log(tmp_path) is not None


def test_resume_rebuilds_job_log_from_edited_incar(finished_job, monkeypatch):
    # A resumed job must clear the stale job.log while it runs and rebuild it
    # afterwards from the INCAR actually used (which may have been edited).
    from pathlib import Path

    import vasp_auto.workflow as wf
    from conftest import POSCAR_TEXT

    for name, text in (("KPOINTS", "k\n"), ("POTCAR", "p\n"),
                       ("POSCAR", POSCAR_TEXT), ("INCAR", "ENCUT = 520\n"),
                       ("job.log", "stale ENCUT : 400")):
        (finished_job / name).write_text(text, encoding="utf-8")

    def fake_run_vasp(job_dir, exe, cpus=None, on_progress=None):
        assert not (Path(job_dir) / "job.log").exists()  # cleared during the run
        return 0

    monkeypatch.setattr(wf, "run_vasp", fake_run_vasp)
    wf.resume_job(finished_job, "vasp", force=True)
    assert "520" in (finished_job / "job.log").read_text(encoding="utf-8")
