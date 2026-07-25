"""Canonical calculation types shared by the CLI and future GUI."""
from __future__ import annotations

from enum import StrEnum


class CalcType(StrEnum):
    SCF = "scf"
    RELAX = "relax"
    VCRELAX = "vcrelax"
    DOS = "dos"
    BANDS = "bands"
    CHARGE = "charge"
    NEB = "neb"
    MD = "md"
    PHONON = "phonon"
    HSE06 = "hse06"
    FREQ = "freq"
    OPTICS = "optics"
    WORKFUNCTION = "workfunction"


# One-line plain-language description per type, shown by the GUI forms and
# `--help`-style listings. Keep these short and free of VASP tag names.
CALC_TYPE_INFO: dict[CalcType, str] = {
    CalcType.SCF: "Single-point energy: total energy of a fixed structure",
    CalcType.RELAX: "Geometry optimisation: relax atoms to their lowest-energy positions",
    CalcType.VCRELAX: "Variable-cell relaxation: relax atoms and cell together (QE vc-relax)",
    CalcType.DOS: "Density of states: electronic states per energy (run after SCF)",
    CalcType.BANDS: "Band structure: energies along a k-path (run after SCF)",
    CalcType.CHARGE: "Charge density: writes CHGCAR for later DOS/bands steps",
    CalcType.NEB: "Transition state (NEB): needs initial/POSCAR and final/POSCAR",
    CalcType.MD: "Molecular dynamics: finite-temperature atomic motion",
    CalcType.PHONON: "Phonons / force constants via DFPT (IBRION=8); relax tightly first.",
    CalcType.HSE06: "Hybrid functional (HSE06): accurate band gaps, much slower",
    CalcType.FREQ: "Vibrational frequencies: zero-point energy and thermal corrections",
    CalcType.OPTICS: "Optical absorption: frequency-dependent dielectric function",
    CalcType.WORKFUNCTION: "Work function: planar-averaged potential of a slab (run on a slab)",
}


# Files each calculation type pulls from the previous step in a chained
# workflow: {output filename in previous step: input filename in this step}.
CHAIN_INPUTS: dict[CalcType, dict[str, str]] = {
    CalcType.SCF: {"CONTCAR": "POSCAR"},
    CalcType.RELAX: {"CONTCAR": "POSCAR"},
    CalcType.VCRELAX: {"CONTCAR": "POSCAR"},
    CalcType.DOS: {"CONTCAR": "POSCAR", "CHGCAR": "CHGCAR"},
    CalcType.BANDS: {"CONTCAR": "POSCAR", "CHGCAR": "CHGCAR"},
    CalcType.CHARGE: {"CONTCAR": "POSCAR"},
    CalcType.NEB: {},
    CalcType.MD: {"CONTCAR": "POSCAR"},
    CalcType.PHONON: {"CONTCAR": "POSCAR"},
    CalcType.HSE06: {"CONTCAR": "POSCAR", "CHGCAR": "CHGCAR", "WAVECAR": "WAVECAR"},
    CalcType.FREQ: {"CONTCAR": "POSCAR"},
    CalcType.OPTICS: {"CONTCAR": "POSCAR", "WAVECAR": "WAVECAR"},
    CalcType.WORKFUNCTION: {"CONTCAR": "POSCAR"},
}


def parse_calc_type(value: str | CalcType) -> CalcType:
    try:
        return CalcType(str(value).strip().lower())
    except ValueError:
        raise ValueError(
            f"Unknown calculation type: {value!r}. Supported: "
            + ", ".join(t.value for t in CalcType)
        ) from None
