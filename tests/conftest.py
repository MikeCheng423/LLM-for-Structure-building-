import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


POSCAR_TEXT = """Al O test cell
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
Al O
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

OUTCAR_TEXT = """ vasp.6.4.2 mock output
  free  energy   TOTEN  =       -10.12345678 eV
  some other line
  aborting loop because EDIFF is reached
  free  energy   TOTEN  =       -12.50000000 eV
 General timing and accounting informations for this run:
"""

OSZICAR_TEXT = """       N       E                     dE             d eps
DAV:   1    -0.10E+02   -0.1E+02
   1 F= -.12500000E+02 E0= -.12500000E+02  d E =-.125000E+02
   2 F= -.12500000E+02 E0= -.12500000E+02  d E =-.000001E+02
"""


@pytest.fixture
def scf_case(tmp_path):
    case_dir = tmp_path / "case_scf"
    case_dir.mkdir()
    (case_dir / "POSCAR").write_text(POSCAR_TEXT, encoding="utf-8")
    return case_dir


@pytest.fixture
def tss_case(tmp_path):
    case_dir = tmp_path / "case_tss"
    for endpoint in ("initial", "final"):
        (case_dir / endpoint).mkdir(parents=True)
        (case_dir / endpoint / "POSCAR").write_text(POSCAR_TEXT, encoding="utf-8")
    return case_dir


@pytest.fixture
def finished_job(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "OUTCAR").write_text(OUTCAR_TEXT, encoding="utf-8")
    (job_dir / "OSZICAR").write_text(OSZICAR_TEXT, encoding="utf-8")
    return job_dir


@pytest.fixture
def potcar_library(tmp_path):
    library = tmp_path / "POTCAR_LIB"
    for name in ("Al", "O", "Fe", "Fe_pv"):
        (library / name).mkdir(parents=True)
        (library / name / "POTCAR").write_bytes(f"POTCAR {name}\n".encode())
    return library
