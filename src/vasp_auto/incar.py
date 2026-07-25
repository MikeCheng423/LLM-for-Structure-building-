"""INCAR editing helpers shared by job preparation, chains, and convergence.

Pure text manipulation — no VASP knowledge beyond tag names lives elsewhere.
"""
from __future__ import annotations

from pathlib import Path


# Typical starting moments (μB) for spin-polarised runs. Values are initial
# guesses VASP refines during the SCF loop, not final answers.
DEFAULT_MAGMOMS: dict[str, float] = {
    "Fe": 5.0,
    "Co": 3.0,
    "Ni": 2.0,
    "Mn": 5.0,
    "Cr": 5.0,
    "V": 3.0,
    "Ti": 1.0,
    "Cu": 1.0,
    "O": 0.6,
}
FALLBACK_MAGMOM = 0.6


def _is_comment_line(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith(("#", "!"))


def _set_in_text(incar_text: str, key: str, value: str | int | float) -> str:
    """Replace `KEY = ...` in INCAR text (appending when absent).

    VASP allows several assignments on one line separated by `;`
    (`ISMEAR = 0 ; SIGMA = 0.2`); the matching assignment is replaced in
    place so no conflicting duplicate tag is ever appended.
    """
    key_upper = key.upper()
    updated = []
    found = False

    for line in incar_text.splitlines():
        if _is_comment_line(line) or "=" not in line:
            updated.append(line)
            continue

        segments = line.split(";")
        names = [seg.split("=", 1)[0].strip().upper() if "=" in seg else None for seg in segments]
        if key_upper not in names:
            updated.append(line)
            continue

        found = True
        rebuilt = [
            f"{key_upper} = {value}" if name == key_upper else seg.strip()
            for seg, name in zip(segments, names)
        ]
        updated.append(" ; ".join(rebuilt))

    if not found:
        updated.append(f"{key_upper} = {value}")

    return "\n".join(updated) + "\n"


def set_incar_value(incar_path: Path, key: str, value: str | int | float):
    text = incar_path.read_text(encoding="utf-8") if incar_path.exists() else ""
    incar_path.write_text(_set_in_text(text, key, value), encoding="utf-8")


def apply_incar_choices(incar_text: str, choices: dict[str, str | int | float]) -> str:
    """Apply {TAG: value} picks from the human-language picker onto INCAR text.

    Each tag is replaced in place or appended (via the same logic as
    :func:`set_incar_value`). Blank/None values are skipped — an unset number
    field means "leave it to VASP or the POTCAR default", not "write nothing".
    This is the write path for the `incar_catalog` form in the UI.
    """
    text = incar_text
    for key, value in choices.items():
        if value is None or str(value).strip() == "":
            continue
        text = _set_in_text(text, key, value)
    return text


def get_incar_value(incar_text: str, key: str) -> str | None:
    key_upper = key.upper()
    for line in incar_text.splitlines():
        if _is_comment_line(line):
            continue
        for segment in line.split(";"):
            if "=" not in segment:
                continue
            name, _, rest = segment.partition("=")
            if name.strip().upper() == key_upper:
                return rest.split("#")[0].split("!")[0].strip()
    return None


def parse_magmom_map(text: str | None) -> dict[str, float]:
    """Parse 'Fe:5.0,O:0.6' into {element: moment}."""
    if not text:
        return {}
    result = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        element, _, moment = item.partition(":")
        if not moment:
            raise ValueError(f"Use El:moment pairs, e.g. Fe:5.0 — got {item!r}")
        result[element.strip()] = float(moment)
    return result


def magmom_line(elements: list[str], counts: list[int], magmom_map: dict | None = None) -> str:
    """Build a VASP MAGMOM value like '2*5.0 4*0.6' from POSCAR composition."""
    moments = []
    for element, count in zip(elements, counts):
        moment = (magmom_map or {}).get(element)
        if moment is None:
            moment = DEFAULT_MAGMOMS.get(element, FALLBACK_MAGMOM)
        moments.append(f"{count}*{float(moment)}")
    return " ".join(moments)


def _poscar_composition(poscar_path: Path) -> tuple[list[str], list[int]]:
    lines = Path(poscar_path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        raise ValueError(f"POSCAR is too short: {poscar_path}")
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    if not elements or all(part.isdigit() for part in elements):
        raise ValueError(f"POSCAR must list element symbols on line 6: {poscar_path}")
    return elements, counts


def spin_incar_text(incar_text: str, poscar_path: Path, magmom_map: dict | None = None) -> str:
    """Return incar_text with ISPIN=2 and a MAGMOM line derived from the POSCAR.

    A MAGMOM the user already set is kept untouched.
    """
    text = _set_in_text(incar_text.rstrip("\n"), "ISPIN", 2)
    if get_incar_value(text, "MAGMOM") is None:
        elements, counts = _poscar_composition(poscar_path)
        text = text.rstrip("\n") + f"\nMAGMOM = {magmom_line(elements, counts, magmom_map)}\n"
    return text


def apply_spin_to_incar(incar_path: Path, poscar_path: Path, magmom_map: dict | None = None):
    incar_path = Path(incar_path)
    text = incar_path.read_text(encoding="utf-8") if incar_path.exists() else ""
    incar_path.write_text(spin_incar_text(text, poscar_path, magmom_map), encoding="utf-8")
