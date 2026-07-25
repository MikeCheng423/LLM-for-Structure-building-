from vasp_auto.workflow import (
    _neb_image_energy,
    build_row,
    count_ionic_steps,
    is_converged,
    neb_energy_profile,
    parse_energy_from_outcar,
    parse_outcar_summary,
    scan_vasp_errors,
    should_retry_failed,
)


def test_parse_outcar_summary(finished_job):
    summary = parse_outcar_summary(finished_job / "OUTCAR")
    assert summary["energy_eV"] == -12.5
    assert summary["converged"] is True


def test_parse_outcar_summary_missing_file(tmp_path):
    summary = parse_outcar_summary(tmp_path / "OUTCAR")
    assert summary["energy_eV"] is None
    assert summary["converged"] is False
    assert summary["energy_without_entropy_eV"] is None
    assert summary["magmom_total"] is None


def test_legacy_helpers_match_summary(finished_job):
    assert parse_energy_from_outcar(finished_job / "OUTCAR") == -12.5
    assert is_converged(finished_job / "OUTCAR") is True


def test_count_ionic_steps(finished_job):
    assert count_ionic_steps(finished_job / "OSZICAR") == 2


def test_scan_vasp_errors_finds_signature(tmp_path):
    job_dir = tmp_path
    (job_dir / "run.log").write_text(
        "running\nZBRENT: fatal error in bracketing\n", encoding="utf-8"
    )
    findings = scan_vasp_errors(job_dir)
    assert [finding["code"] for finding in findings] == ["ZBRENT"]
    assert "POTIM" in findings[0]["hint"]


def test_scan_vasp_errors_clean_run(finished_job):
    assert scan_vasp_errors(finished_job) == []


def test_build_row_scf(finished_job):
    case_info = {
        "case_name": "job",
        "job_dir": finished_job,
        "calculation_type": "scf",
    }
    row = build_row("proj", "single", case_info)
    assert row["status"] == "done"
    assert row["energy_eV"] == -12.5
    assert row["converged"] is True
    assert row["ionic_steps"] == 2


def test_build_row_missing_outputs(tmp_path):
    case_info = {
        "case_name": "job",
        "job_dir": tmp_path / "nothing",
        "calculation_type": "scf",
    }
    row = build_row("proj", "single", case_info)
    assert row["status"] == "missing"
    assert row["converged"] is False


def test_should_retry_failed(tmp_path, finished_job):
    converged_info = {"case_name": "job", "job_dir": finished_job, "calculation_type": "scf"}
    missing_info = {"case_name": "x", "job_dir": tmp_path / "nope", "calculation_type": "scf"}
    assert should_retry_failed(converged_info) is False
    assert should_retry_failed(missing_info) is True


# ---------------------------------------------------------------- TASK 1: NEB retry + barriers

OUTCAR_CONVERGED = """ vasp mock
  free  energy   TOTEN  =       -10.00000000 eV
  aborting loop because EDIFF is reached
"""

OSZICAR_ENERGY = "   1 F= -.9000000E+01 E0= -.9000000E+01  d E =-.0001E+01\n"


def _make_neb_job(tmp_path, n_movable=3, converged_interior=True):
    """Create a fake NEB job directory with 00..0(N+1) image dirs.

    Images: 00 (initial, no OUTCAR), 01..0N (interior, with OUTCARs if
    converged_interior=True), 0(N+1) (final, no OUTCAR).
    """
    job_dir = tmp_path / "neb_job"
    job_dir.mkdir()
    n_images = n_movable + 2  # including endpoints
    for i in range(n_images):
        img_dir = job_dir / f"{i:02d}"
        img_dir.mkdir()
        is_endpoint = (i == 0 or i == n_images - 1)
        if not is_endpoint:
            if converged_interior:
                (img_dir / "OUTCAR").write_text(OUTCAR_CONVERGED, encoding="utf-8")
            # OSZICAR for interior images too
            (img_dir / "OSZICAR").write_text(OSZICAR_ENERGY, encoding="utf-8")
    return job_dir


def test_should_retry_failed_neb_endpoints(tmp_path):
    """NEB 00–04 with OUTCARs only in interior 01-03 → converged, don't retry."""
    job_dir = _make_neb_job(tmp_path, n_movable=3, converged_interior=True)
    case_info = {"case_name": "neb", "job_dir": job_dir, "calculation_type": "tss"}
    assert should_retry_failed(case_info) is False


def test_should_retry_failed_neb_missing_interior_outcar(tmp_path):
    """NEB 00–04 with missing OUTCAR in one interior image → should retry."""
    job_dir = _make_neb_job(tmp_path, n_movable=3, converged_interior=False)
    case_info = {"case_name": "neb", "job_dir": job_dir, "calculation_type": "tss"}
    assert should_retry_failed(case_info) is True


def test_should_retry_failed_neb_unconverged_interior(tmp_path):
    """NEB with unconverged interior OUTCAR → should retry."""
    job_dir = _make_neb_job(tmp_path, n_movable=3, converged_interior=True)
    # Overwrite one interior OUTCAR with unconverged content.
    (job_dir / "01" / "OUTCAR").write_text(
        " vasp mock\n  free  energy   TOTEN  =  -10.0 eV\n", encoding="utf-8"
    )
    case_info = {"case_name": "neb", "job_dir": job_dir, "calculation_type": "tss"}
    assert should_retry_failed(case_info) is True


def test_neb_image_energy_from_outcar(tmp_path):
    img = tmp_path / "01"
    img.mkdir()
    (img / "OUTCAR").write_text(OUTCAR_CONVERGED, encoding="utf-8")
    assert _neb_image_energy(img) == -10.0


def test_neb_image_energy_from_oszicar(tmp_path):
    img = tmp_path / "00"
    img.mkdir()
    (img / "OSZICAR").write_text(OSZICAR_ENERGY, encoding="utf-8")
    energy = _neb_image_energy(img)
    assert energy is not None
    assert abs(energy - (-9.0)) < 0.01


def test_neb_row_barriers_use_endpoints(tmp_path):
    """Forward barrier is max(energies) − E(initial endpoint) from OSZICAR."""
    job_dir = _make_neb_job(tmp_path, n_movable=3, converged_interior=True)
    # Give each interior image a different energy via their OUTCARs.
    energies = [-8.0, -7.0, -8.5]  # max is -7.0 at image 02
    for i, e in enumerate(energies, start=1):
        outcar_text = (
            f" vasp mock\n"
            f"  free  energy   TOTEN  =  {e:.8f} eV\n"
            f"  aborting loop because EDIFF is reached\n"
        )
        (job_dir / f"{i:02d}" / "OUTCAR").write_text(outcar_text, encoding="utf-8")
    # Add OSZICAR to endpoint 00 (E = -9.0 eV) — endpoints have no OUTCAR so
    # _neb_image_energy falls back to OSZICAR.
    (job_dir / "00" / "OSZICAR").write_text(
        "   1 F= -.9000000E+01 E0= -.9000000E+01  d E =0\n", encoding="utf-8"
    )
    # Add OSZICAR to endpoint 04 (E = -8.5 eV).
    (job_dir / "04" / "OSZICAR").write_text(
        "   1 F= -.8500000E+01 E0= -.8500000E+01  d E =0\n", encoding="utf-8"
    )

    case_info = {"case_name": "neb", "job_dir": job_dir, "calculation_type": "tss"}
    row = build_row("proj", "single", case_info)

    # E_initial from 00/OSZICAR = -9.0; max overall = -7.0 → forward = 2.0
    assert row["neb_forward_barrier_eV"] is not None
    assert abs(row["neb_forward_barrier_eV"] - 2.0) < 0.01

    # E_final from 04/OSZICAR = -8.5; max = -7.0 → backward = 1.5
    assert row["neb_backward_barrier_eV"] is not None
    assert abs(row["neb_backward_barrier_eV"] - 1.5) < 0.01


def _write_neb_image(img_dir, energy, x):
    img_dir.mkdir(parents=True)
    (img_dir / "OUTCAR").write_text(
        f" vasp mock\n  free  energy   TOTEN  =  {energy:.8f} eV\n"
        "  aborting loop because EDIFF is reached\n",
        encoding="utf-8",
    )
    (img_dir / "POSCAR").write_text(
        "img\n1.0\n10 0 0\n0 10 0\n0 0 10\nH\n1\nCartesian\n"
        f"{x:.4f} 0.0 0.0\n",
        encoding="utf-8",
    )


def test_neb_energy_profile(tmp_path):
    job_dir = tmp_path / "neb"
    energies = [-10.0, -9.4, -8.9, -9.5, -10.2]  # barrier (TS) at image 2
    for i, e in enumerate(energies):
        _write_neb_image(job_dir / f"{i:02d}", e, x=i * 0.5)

    profile = neb_energy_profile(job_dir)
    assert profile["images"] == [0, 1, 2, 3, 4]
    assert profile["energies_eV"] == energies
    assert profile["relative_eV"][0] == 0.0
    assert abs(profile["forward_barrier_eV"] - 1.1) < 1e-6   # -8.9 - (-10.0)
    assert abs(profile["backward_barrier_eV"] - 1.3) < 1e-6  # -8.9 - (-10.2)
    assert abs(profile["delta_e_eV"] - (-0.2)) < 1e-6        # -10.2 - (-10.0)
    assert profile["ts_image"] == 2
    # reaction coordinate is normalised 0 → 1 and monotonic
    assert profile["reaction_coord"][0] == 0.0
    assert profile["reaction_coord"][-1] == 1.0
    assert profile["reaction_coord"] == sorted(profile["reaction_coord"])


def test_neb_energy_profile_endpoint_energies(tmp_path):
    # Standard NEB: endpoints 00/04 have POSCARs but no energy files.
    job_dir = tmp_path / "neb"
    for i, e in enumerate([-10.0, -9.4, -8.9, -9.5, -10.2]):
        _write_neb_image(job_dir / f"{i:02d}", e, x=i * 0.5)
    for name in ("00", "04"):
        (job_dir / name / "OUTCAR").unlink()

    profile = neb_energy_profile(job_dir)
    assert profile["images"] == [1, 2, 3]  # endpoints dropped without energies

    profile = neb_energy_profile(job_dir, endpoint_energies=(-11.0, -11.5))
    assert profile["images"] == [0, 1, 2, 3, 4]
    assert profile["energies_eV"][0] == -11.0
    assert profile["energies_eV"][-1] == -11.5
    assert abs(profile["forward_barrier_eV"] - 2.1) < 1e-6  # -8.9 - (-11.0)

    # one-sided pair works, and an existing energy file wins over the value
    profile = neb_energy_profile(job_dir, endpoint_energies=(None, -11.5))
    assert profile["images"] == [1, 2, 3, 4]
    profile = neb_energy_profile(job_dir, endpoint_energies=(-11.0, None))
    assert profile["energies_eV"][0] == -11.0


def test_neb_energy_profile_needs_two_images(tmp_path):
    job_dir = tmp_path / "neb"
    _write_neb_image(job_dir / "00", -10.0, x=0.0)
    assert neb_energy_profile(job_dir) is None
    assert neb_energy_profile(tmp_path / "missing") is None


def _no(*_a, **_k):
    raise AssertionError("wrong runner was called for a direct-SSH remote")


def test_run_one_case_ssh_remote_runs_remotely_and_tags_machine(finished_job, monkeypatch):
    """run_one_case with an ssh run_mode remote uses run_vasp_remote (not the
    local runner or the scheduler submit) and tags the row with the machine."""
    import json
    from vasp_auto import workflow

    case_info = {"case_name": "job", "job_dir": finished_job, "calculation_type": "scf"}
    remote = {"host": "wkstn", "name": "wkstn", "remote_root": "/work",
              "vasp_executable": "/opt/vasp/bin/vasp_std", "run_mode": "ssh"}
    seen = {}

    def fake_remote(job_dir, rem, cpus=None, on_progress=None, engine="vasp"):
        seen["remote"] = rem
        seen["cpus"] = cpus
        (finished_job / ".remote.json").write_text(
            json.dumps({"machine": "wkstn", "remote_dir": "/work/job", "mode": "ssh"})
        )
        return 0

    monkeypatch.setattr(workflow, "run_vasp_remote", fake_remote)
    monkeypatch.setattr(workflow, "run_vasp", _no)
    monkeypatch.setattr(workflow, "submit_job_remote", _no)

    row = workflow.run_one_case(
        "proj", "single", case_info, "vasp_std", cpus=8, remote=remote,
    )

    assert seen["remote"] is remote
    assert seen["cpus"] == 8
    assert row["status"] == "done"
    assert row["energy_eV"] == -12.5
    assert row["machine"] == "wkstn"
    assert row["remote_dir"] == "/work/job"


def test_run_one_case_scheduler_remote_still_submits(finished_job, monkeypatch):
    """A non-ssh remote keeps the fire-and-forget scheduler submission path."""
    from vasp_auto import workflow

    case_info = {"case_name": "job", "job_dir": finished_job, "calculation_type": "scf"}
    remote = {"host": "cluster", "name": "cluster", "remote_root": "/scratch",
              "vasp_executable": "/opt/vasp_std", "scheduler": "slurm"}

    monkeypatch.setattr(workflow, "run_vasp_remote", _no)
    monkeypatch.setattr(
        workflow, "submit_job_remote",
        lambda *a, **k: {"job_id": "99", "scheduler": "slurm", "host": "cluster",
                         "remote_dir": "/scratch/job"},
    )

    row = workflow.run_one_case(
        "proj", "single", case_info, "vasp_std", cpus=8, remote=remote,
    )
    assert row["status"] == "submitted"
    assert row["job_id"] == "99"
