"""Resume reuses the previous run's MPI rank count, read from OUTCAR."""
from vasp_auto.workflow import _previous_run_ranks


def test_no_outcar_means_unknown(tmp_path):
    assert _previous_run_ranks(tmp_path) is None


def test_vasp6_header(tmp_path):
    (tmp_path / "OUTCAR").write_text(
        " vasp.6.4.2\n running   16 mpi-ranks, with    1 threads/rank\n"
    )
    assert _previous_run_ranks(tmp_path) == 16


def test_vasp5_header(tmp_path):
    (tmp_path / "OUTCAR").write_text(" running on    8 total cores\n")
    assert _previous_run_ranks(tmp_path) == 8


def test_neb_image_outcar(tmp_path):
    (tmp_path / "01").mkdir()
    (tmp_path / "01" / "OUTCAR").write_text(" running on   32 total cores\n")
    assert _previous_run_ranks(tmp_path) == 32
