"""Tests for the four error-handling defects fixed in workflow.py / parser.py.

Defect 1 — SICK_JOB must be a generic fallback, not an independent error.
Defect 2 — ZBRENT hint must change when geometry already converged.
Defect 3 — apply_error_fixes must copy CONTCAR->POSCAR for ZBRENT restarts.
Defect 4 — parse_vasprun / build_row must not raise on truncated vasprun.xml.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers — minimal VASP log fragments
# ---------------------------------------------------------------------------

LOG_ZBRENT_ONLY = """\
running
 curvature:   0.00 expect dE= 0.000E+00 dE for cont linesearch  0.000E+00
|     ZBRENT: fatal error in bracketing                                       |
|      please rerun with smaller EDIFF, or copy CONTCAR                       |
|      to POSCAR and continue                                                 |
"""

LOG_ZBRENT_AND_SICK = (
    LOG_ZBRENT_ONLY
    + "\n|       ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----       |\n"
)

LOG_SICK_ONLY = """\
running
|       ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----       |
"""

LOG_EDDDAV_AND_SICK = """\
running
Error EDDDAV: Call to ZHEGV failed. Return code = 4 ...
|       ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----       |
"""

LOG_CLEAN = "running\n aborting loop because EDIFF is reached\n"

OUTCAR_CONVERGED = """\
 vasp mock
  free  energy   TOTEN  =       -27.07313700 eV
  aborting loop because EDIFF is reached
 General timing and accounting informations for this run:
"""

OUTCAR_HIGH_FORCE = """\
 vasp mock
  free  energy   TOTEN  =       -20.00000000 eV
"""

INCAR_WITH_EDIFFG = """\
SYSTEM = test
IBRION = 2
NSW = 100
EDIFFG = -0.02
ENCUT = 400
"""

CONTCAR_DIFFERENT = """\
Au test
1.0
14.9 0.0 0.0
0.0 14.9 0.0
0.0 0.0 14.9
Au
1
Direct
0.5001 0.4999 0.5000
"""

POSCAR_ORIGINAL = """\
Au test
1.0
14.9 0.0 0.0
0.0 14.9 0.0
0.0 0.0 14.9
Au
1
Direct
0.5000 0.5000 0.5000
"""

VASPRUN_TRUNCATED = """\
<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
 <generator>
  <i name="program" type="string">vasp </i>
 </generator>
 <incar>
  <i name="EDIFF">   0.00001</i>
 </incar>
 <calculation>
  <energy>
   <i name="e_fr_energy">   -27.07313700 </i>
  </energy>
"""
# Deliberately NOT closed — simulates a run.log truncated mid-write.


# ---------------------------------------------------------------------------
# Import the modules under test
# ---------------------------------------------------------------------------

from vasp_auto.workflow import (
    VASP_ERROR_RESTART_FROM_CONTCAR,
    apply_error_fixes,
    geometry_converged,
    report_vasp_errors,
    scan_vasp_errors,
)
from vasp_auto.parser import parse_vasprun
from vasp_auto.workflow import build_row


# ===========================================================================
# DEFECT 1 — SICK_JOB is a generic/fallback, not an independent error
# ===========================================================================


class TestSickJobFallback:
    def test_zbrent_plus_sick_job_reports_only_zbrent(self, tmp_path):
        """A log with ZBRENT AND the SICK_JOB banner must yield only ZBRENT."""
        (tmp_path / "run.log").write_text(LOG_ZBRENT_AND_SICK, encoding="utf-8")
        findings = scan_vasp_errors(tmp_path)
        codes = [f["code"] for f in findings]
        assert codes == ["ZBRENT"], f"Expected only ZBRENT, got {codes}"

    def test_edddav_plus_sick_job_reports_only_edddav(self, tmp_path):
        """A log with EDDDAV AND the SICK_JOB banner must yield only EDDDAV."""
        (tmp_path / "run.log").write_text(LOG_EDDDAV_AND_SICK, encoding="utf-8")
        findings = scan_vasp_errors(tmp_path)
        codes = [f["code"] for f in findings]
        assert codes == ["EDDDAV"], f"Expected only EDDDAV, got {codes}"

    def test_sick_job_alone_is_reported(self, tmp_path):
        """When SICK_JOB is the ONLY signal, it must still be returned."""
        (tmp_path / "run.log").write_text(LOG_SICK_ONLY, encoding="utf-8")
        findings = scan_vasp_errors(tmp_path)
        codes = [f["code"] for f in findings]
        assert "SICK_JOB" in codes, f"Expected SICK_JOB, got {codes}"

    def test_clean_run_no_findings(self, tmp_path):
        """A clean convergence line must produce no findings."""
        (tmp_path / "run.log").write_text(LOG_CLEAN, encoding="utf-8")
        assert scan_vasp_errors(tmp_path) == []

    def test_sick_job_hint_references_incar_poscar(self, tmp_path):
        """SICK_JOB hint must mention INCAR/POSCAR."""
        (tmp_path / "run.log").write_text(LOG_SICK_ONLY, encoding="utf-8")
        findings = scan_vasp_errors(tmp_path)
        assert findings
        assert "INCAR" in findings[0]["hint"] or "POSCAR" in findings[0]["hint"]


# ===========================================================================
# DEFECT 2 — ZBRENT hint changes when geometry is already converged
# ===========================================================================


class TestZbrentConvergedHint:
    def _make_zbrent_job(self, tmp_path, *, converged: bool) -> Path:
        """Write a minimal job dir with ZBRENT in run.log."""
        (tmp_path / "run.log").write_text(LOG_ZBRENT_ONLY, encoding="utf-8")
        if converged:
            (tmp_path / "OUTCAR").write_text(OUTCAR_CONVERGED, encoding="utf-8")
        else:
            (tmp_path / "OUTCAR").write_text(OUTCAR_HIGH_FORCE, encoding="utf-8")
        (tmp_path / "INCAR").write_text(INCAR_WITH_EDIFFG, encoding="utf-8")
        return tmp_path

    def test_geometry_converged_via_outcar_marker(self, tmp_path):
        """geometry_converged returns True when OUTCAR has convergence marker."""
        job_dir = self._make_zbrent_job(tmp_path, converged=True)
        assert geometry_converged(job_dir) is True

    def test_geometry_not_converged_without_marker(self, tmp_path):
        """geometry_converged returns False when no convergence marker present."""
        job_dir = self._make_zbrent_job(tmp_path, converged=False)
        assert geometry_converged(job_dir) is False

    def test_geometry_converged_missing_files(self, tmp_path):
        """geometry_converged returns False when no output files exist."""
        assert geometry_converged(tmp_path) is False

    def test_report_zbrent_converged_has_accept_hint(self, tmp_path, capsys):
        """report_vasp_errors with ZBRENT + converged geometry → 'accept' hint."""
        job_dir = self._make_zbrent_job(tmp_path, converged=True)
        report_vasp_errors(job_dir)
        out = capsys.readouterr().out
        assert "accept" in out.lower() or "contcar" in out.lower(), (
            f"Expected hint about accepting result or CONTCAR, got: {out!r}"
        )

    def test_report_zbrent_not_converged_has_generic_hint(self, tmp_path, capsys):
        """report_vasp_errors with ZBRENT + not converged → standard hint."""
        job_dir = self._make_zbrent_job(tmp_path, converged=False)
        report_vasp_errors(job_dir)
        out = capsys.readouterr().out
        # The standard ZBRENT hint should still reference CONTCAR restart
        assert "ZBRENT" in out

    def test_geometry_converged_via_run_log_marker(self, tmp_path):
        """geometry_converged also detects markers in run.log."""
        # Only run.log, no OUTCAR
        (tmp_path / "run.log").write_text(
            "running\n aborting loop because EDIFF is reached\nZBRENT: fatal\n",
            encoding="utf-8",
        )
        assert geometry_converged(tmp_path) is True


# ===========================================================================
# DEFECT 3 — apply_error_fixes copies CONTCAR->POSCAR for ZBRENT
# ===========================================================================


class TestApplyErrorFixesContcar:
    def _zbrent_finding(self) -> dict:
        return {"code": "ZBRENT", "file": "run.log", "hint": "ionic step failed"}

    def _setup_job(self, job_dir: Path, *, contcar_content: str, poscar_content: str) -> None:
        (job_dir / "CONTCAR").write_text(contcar_content, encoding="utf-8")
        (job_dir / "POSCAR").write_text(poscar_content, encoding="utf-8")
        (job_dir / "INCAR").write_text(INCAR_WITH_EDIFFG, encoding="utf-8")

    def test_zbrent_copies_contcar_to_poscar(self, tmp_path):
        """When CONTCAR != POSCAR and non-empty, apply_error_fixes copies CONTCAR."""
        self._setup_job(tmp_path, contcar_content=CONTCAR_DIFFERENT, poscar_content=POSCAR_ORIGINAL)
        applied = apply_error_fixes(tmp_path, [self._zbrent_finding()])
        assert "restart from CONTCAR" in applied
        assert (tmp_path / "POSCAR").read_text() == CONTCAR_DIFFERENT

    def test_zbrent_restart_action_reported(self, tmp_path):
        """apply_error_fixes must include the restart action string in its return."""
        self._setup_job(tmp_path, contcar_content=CONTCAR_DIFFERENT, poscar_content=POSCAR_ORIGINAL)
        applied = apply_error_fixes(tmp_path, [self._zbrent_finding()])
        assert any("restart" in a for a in applied)

    def test_zbrent_incar_fix_also_applied(self, tmp_path):
        """IBRION INCAR fix is still applied alongside the CONTCAR restart."""
        self._setup_job(tmp_path, contcar_content=CONTCAR_DIFFERENT, poscar_content=POSCAR_ORIGINAL)
        applied = apply_error_fixes(tmp_path, [self._zbrent_finding()])
        # Should include both the restart action and the IBRION nudge
        assert any("IBRION" in a for a in applied)

    def test_zbrent_no_copy_when_contcar_empty(self, tmp_path):
        """apply_error_fixes must NOT copy an empty CONTCAR."""
        self._setup_job(tmp_path, contcar_content="", poscar_content=POSCAR_ORIGINAL)
        applied = apply_error_fixes(tmp_path, [self._zbrent_finding()])
        assert "restart from CONTCAR" not in applied
        # POSCAR should be unchanged
        assert (tmp_path / "POSCAR").read_text() == POSCAR_ORIGINAL

    def test_zbrent_no_copy_when_contcar_missing(self, tmp_path):
        """apply_error_fixes must NOT fail or copy when CONTCAR doesn't exist."""
        (tmp_path / "POSCAR").write_text(POSCAR_ORIGINAL, encoding="utf-8")
        (tmp_path / "INCAR").write_text(INCAR_WITH_EDIFFG, encoding="utf-8")
        # CONTCAR intentionally absent
        applied = apply_error_fixes(tmp_path, [self._zbrent_finding()])
        assert "restart from CONTCAR" not in applied

    def test_zbrent_no_copy_when_contcar_same_as_poscar(self, tmp_path):
        """apply_error_fixes must NOT copy CONTCAR when it is identical to POSCAR."""
        self._setup_job(
            tmp_path, contcar_content=POSCAR_ORIGINAL, poscar_content=POSCAR_ORIGINAL
        )
        applied = apply_error_fixes(tmp_path, [self._zbrent_finding()])
        assert "restart from CONTCAR" not in applied

    def test_non_restart_error_no_contcar_copy(self, tmp_path):
        """EDDDAV finding must not trigger a CONTCAR copy."""
        self._setup_job(tmp_path, contcar_content=CONTCAR_DIFFERENT, poscar_content=POSCAR_ORIGINAL)
        edddav_finding = {"code": "EDDDAV", "file": "run.log", "hint": "Davidson failed"}
        applied = apply_error_fixes(tmp_path, [edddav_finding])
        assert "restart from CONTCAR" not in applied
        # POSCAR must be untouched
        assert (tmp_path / "POSCAR").read_text() == POSCAR_ORIGINAL

    def test_vasp_error_restart_from_contcar_set_contains_zbrent(self):
        """VASP_ERROR_RESTART_FROM_CONTCAR must include ZBRENT."""
        assert "ZBRENT" in VASP_ERROR_RESTART_FROM_CONTCAR


# ===========================================================================
# DEFECT 4 — parse_vasprun / build_row must not blow up on truncated XML
# ===========================================================================


class TestTruncatedVasprun:
    def test_parse_vasprun_truncated_returns_none(self, tmp_path):
        """parse_vasprun must return None (not raise) on truncated/malformed XML."""
        vasprun = tmp_path / "vasprun.xml"
        vasprun.write_text(VASPRUN_TRUNCATED, encoding="utf-8")
        result = parse_vasprun(vasprun)
        assert result is None

    def test_parse_vasprun_missing_file_returns_none(self, tmp_path):
        """parse_vasprun must return None when file does not exist."""
        result = parse_vasprun(tmp_path / "vasprun.xml")
        assert result is None

    def test_build_row_tolerates_truncated_vasprun(self, tmp_path):
        """build_row must not raise when vasprun.xml is truncated; falls back to OUTCAR."""
        # Create a minimal finished job with a truncated vasprun.xml.
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "OUTCAR").write_text(OUTCAR_CONVERGED, encoding="utf-8")
        (job_dir / "vasprun.xml").write_text(VASPRUN_TRUNCATED, encoding="utf-8")
        case_info = {
            "case_name": "test",
            "job_dir": job_dir,
            "calculation_type": "relax",
        }
        row = build_row("proj", "single", case_info)
        # Energy must come from OUTCAR fallback.
        assert row["energy_eV"] == pytest.approx(-27.073137, abs=1e-3)
        assert row["converged"] is True

    def test_build_row_no_exception_on_totally_empty_vasprun(self, tmp_path):
        """build_row must not raise when vasprun.xml is completely empty."""
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "OUTCAR").write_text(OUTCAR_CONVERGED, encoding="utf-8")
        (job_dir / "vasprun.xml").write_text("", encoding="utf-8")
        case_info = {
            "case_name": "test",
            "job_dir": job_dir,
            "calculation_type": "relax",
        }
        row = build_row("proj", "single", case_info)
        assert row["status"] == "done"

    def test_build_row_no_exception_on_invalid_xml(self, tmp_path):
        """build_row must not raise on completely invalid XML."""
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "OUTCAR").write_text(OUTCAR_CONVERGED, encoding="utf-8")
        (job_dir / "vasprun.xml").write_text("<<< not xml at all >>>", encoding="utf-8")
        case_info = {
            "case_name": "test",
            "job_dir": job_dir,
            "calculation_type": "relax",
        }
        row = build_row("proj", "single", case_info)
        assert row["energy_eV"] is not None  # OUTCAR fallback
