"""Regression tests for the 2026-06-12 bug-hunt (v0.5.0).

Bug 1: negative-scale POSCAR (target-volume convention) produced mirrored
       geometry in structure tools, trajectories, and the UI viewer.
Bug 2: `;`-compound INCAR lines (ISMEAR = 0 ; SIGMA = 0.2) got conflicting
       duplicate tags appended by set_incar_value and were invisible to
       get_incar_value / spin_incar_text.
Bug 3: --retry-failed wiped CONTCAR/WAVECAR/CHGCAR before re-running,
       discarding the half-relaxed geometry and the SCF seed.
"""
from pathlib import Path

from vasp_auto.cli import _restore_restart_files, _stash_restart_files
from vasp_auto.incar import get_incar_value, set_incar_value, spin_incar_text
from vasp_auto.structure import read_poscar
from vasp_auto.trajectory import job_trajectory


NEGATIVE_SCALE_POSCAR = """Al, scale = target volume
-66.430125
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
Al
4
Direct
0.0 0.0 0.0
0.5 0.5 0.0
0.5 0.0 0.5
0.0 0.5 0.5
"""


# ------------------------------------------------------- bug 1: negative scale

def test_read_poscar_negative_scale_resolves_to_volume(tmp_path):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(NEGATIVE_SCALE_POSCAR, encoding="utf-8")
    struct = read_poscar(poscar)
    # |scale| is the target volume: a = 66.430125^(1/3) = 4.05 Å
    assert abs(struct["scale"] - 4.05) < 1e-6


def test_trajectory_negative_scale_frames_positive(tmp_path):
    (tmp_path / "POSCAR").write_text(NEGATIVE_SCALE_POSCAR, encoding="utf-8")
    (tmp_path / "CONTCAR").write_text(NEGATIVE_SCALE_POSCAR, encoding="utf-8")
    traj = job_trajectory(tmp_path)
    assert traj is not None
    assert traj["lattice"][0][0] > 0  # was -66.43 before the fix
    # second atom at fractional (0.5, 0.5, 0) → cartesian (2.025, 2.025, 0)
    assert abs(traj["frames"][0][1][0] - 2.025) < 1e-6


# --------------------------------------------------- bug 2: compound `;` lines

def test_set_incar_value_replaces_inside_compound_line(tmp_path):
    incar = tmp_path / "INCAR"
    incar.write_text("ISMEAR = 0 ; SIGMA = 0.2\nENCUT = 400\n", encoding="utf-8")
    set_incar_value(incar, "SIGMA", 0.05)
    text = incar.read_text(encoding="utf-8")
    assert text.count("SIGMA") == 1  # replaced in place, no duplicate appended
    assert "SIGMA = 0.05" in text
    assert "ISMEAR = 0" in text  # the rest of the compound line survives
    assert "ENCUT = 400" in text


def test_get_incar_value_finds_tag_inside_compound_line():
    text = "ISMEAR = 0 ; SIGMA = 0.2\n"
    assert get_incar_value(text, "SIGMA") == "0.2"
    assert get_incar_value(text, "ISMEAR") == "0"


def test_spin_incar_text_detects_ispin_in_compound_line(tmp_path):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(NEGATIVE_SCALE_POSCAR, encoding="utf-8")
    out = spin_incar_text("NSW = 0 ; ISPIN = 1\n", poscar)
    assert out.count("ISPIN") == 1
    assert get_incar_value(out, "ISPIN") == "2"


def test_set_incar_value_ignores_bang_comment_lines(tmp_path):
    incar = tmp_path / "INCAR"
    incar.write_text("! ENCUT = 300 disabled\nENCUT = 400\n", encoding="utf-8")
    set_incar_value(incar, "ENCUT", 450)
    text = incar.read_text(encoding="utf-8")
    assert "! ENCUT = 300 disabled" in text  # comment untouched
    assert "ENCUT = 450" in text
    assert "ENCUT = 400" not in text


# --------------------------------------------- bug 3: --retry-failed restarts

def test_retry_stash_preserves_restart_files(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "CONTCAR").write_text("half-relaxed geometry\n", encoding="utf-8")
    (job_dir / "WAVECAR").write_bytes(b"wavefunction")
    (job_dir / "OUTCAR").write_text("old output\n", encoding="utf-8")

    stash = _stash_restart_files(job_dir)
    assert stash is not None
    assert not (job_dir / "CONTCAR").exists()

    # simulate create_job_from_case(clean_job=True): wipe + fresh inputs
    for item in job_dir.iterdir():
        item.unlink()
    (job_dir / "POSCAR").write_text("fresh input geometry\n", encoding="utf-8")

    _restore_restart_files(job_dir, stash)
    assert (job_dir / "POSCAR").read_text() == "half-relaxed geometry\n"
    assert (job_dir / "WAVECAR").read_bytes() == b"wavefunction"
    assert not stash.exists()  # stash cleaned up


def test_retry_stash_none_when_nothing_to_keep(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "CONTCAR").write_text("", encoding="utf-8")  # empty = crashed run
    assert _stash_restart_files(job_dir) is None
    _restore_restart_files(job_dir, None)  # no-op, must not raise
