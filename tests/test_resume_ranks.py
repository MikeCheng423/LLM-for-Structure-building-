"""Resume must clamp MPI ranks to a multiple of IMAGES for NEB jobs.

Regression for the tlclab 0015/0016 NEB resumes: -np 12 with IMAGES=5 makes
VASP abort in M_divide / MPI_Cart_sub before doing any work.
"""
import subprocess

from vasp_auto.runner import _NEB_RANKS_SH
from vasp_auto.workflow import _ranks_for_images

NEB_INCAR = (
    "# vasp-auto --calc-type neb appends IMAGES automatically from --neb-images\n"
    "NCORE = 4\n"
    "IMAGES = 5\n"
)


def test_ranks_for_images(tmp_path):
    (tmp_path / "INCAR").write_text(NEB_INCAR)
    assert _ranks_for_images(tmp_path, 12) == 10  # the tlclab crash
    assert _ranks_for_images(tmp_path, 40) == 40
    assert _ranks_for_images(tmp_path, 3) == 5  # below one rank per image
    assert _ranks_for_images(tmp_path, None) == 5


def test_ranks_without_images_tag(tmp_path):
    (tmp_path / "INCAR").write_text("NSW = 100\n")
    assert _ranks_for_images(tmp_path, 12) == 12
    assert _ranks_for_images(tmp_path, None) is None


def _sh_ranks(tmp_path, preset):
    out = subprocess.run(
        ["bash", "-c", f'{preset}; {_NEB_RANKS_SH}; echo "${{ranks:-1}}"'],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_neb_ranks_shell_snippet(tmp_path):
    (tmp_path / "INCAR").write_text(NEB_INCAR)
    assert _sh_ranks(tmp_path, "ranks=12") == "10"
    assert _sh_ranks(tmp_path, "ranks=40") == "40"
    assert _sh_ranks(tmp_path, "true") == "5"  # no previous OUTCAR -> one per image
