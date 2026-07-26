"""Write a finished structure out as files the rest of the pipeline can eat.

The bridge from "the model built something" to "I can run DFT on it". Every
successful build writes **one directory that is already a valid ``vasp-auto``
SCF case**: a VASP ``POSCAR`` plus a JSON sidecar. That layout is deliberate --
``vasp_auto.target_utils.get_case_type`` calls any directory containing a
``POSCAR`` file an ``scf`` case, so::

    vasp-auto <out-dir>/<case> --prepare

works with no extra step, and the parent directory holding several such cases is
a valid ``vasp-auto`` *project* target.

The sidecar carries the deterministic recipe, so the exact structure can be
rebuilt without the model: replay ``recipe["steps"]`` through
``ASEWorkspace.execute_or_raise`` and compare ``atoms_hash``.

Nothing here executes model-supplied paths: the caller chooses the output
directory, the case name is derived from the structure's own formula and recipe
hash, and file names are fixed constants.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .validation import atoms_hash, structure_invariants

SIDECAR_NAME = "structure.json"
POSCAR_NAME = "POSCAR"
SIDECAR_SCHEMA_VERSION = 1

# Stems for the optional extra formats; the extension comes from ase.io.
EXTRA_STEM = "structure"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class ExportError(RuntimeError):
    """A structure could not be written in the requested form."""


@dataclass(frozen=True)
class ExportResult:
    case_dir: Path
    poscar: Path
    sidecar: Path
    extra: tuple[Path, ...] = ()

    @property
    def paths(self) -> tuple[Path, ...]:
        return (self.poscar, *self.extra, self.sidecar)


def case_name(atoms, recipe_hash: str) -> str:
    """A stable, path-safe case name: chemical formula + short recipe hash.

    Content-addressed on purpose -- re-running the same request lands in the same
    directory and simply rewrites identical files, instead of littering.
    """
    try:
        formula = atoms.get_chemical_formula(mode="metal")
    except Exception:  # pragma: no cover - ase always provides a formula
        formula = "structure"
    formula = _UNSAFE.sub("", formula) or "structure"
    return f"{formula}-{recipe_hash[:8]}"


def _extension_for(fmt: str) -> str:
    from ase.io.formats import ioformats

    entry = ioformats.get(fmt)
    extensions = list(getattr(entry, "extensions", ()) or ()) if entry else []
    return extensions[0] if extensions else fmt


def normalize_formats(values: Iterable[str] | None) -> tuple[str, ...]:
    """Accept repeated flags and/or comma-separated lists; drop poscar aliases."""
    seen: list[str] = []
    for value in values or ():
        for piece in str(value).split(","):
            fmt = piece.strip().lower()
            if not fmt:
                continue
            if fmt in {"poscar", "vasp"}:
                # Always written; naming it again is a no-op rather than an error.
                continue
            if fmt not in seen:
                seen.append(fmt)
    return tuple(seen)


def write_bundle(
    atoms,
    case_dir: Path,
    *,
    request: str,
    recipe: dict[str, Any],
    recipe_hash: str,
    formats: Iterable[str] = (),
    tool_sequence: Iterable[str] = (),
    model_info: dict[str, Any] | None = None,
    controller_state: str | None = None,
    advisory: dict[str, Any] | None = None,
    mismatches: list[dict[str, Any]] | None = None,
) -> ExportResult:
    """Write POSCAR + optional extra formats + the JSON sidecar into `case_dir`."""
    from ase.io import write as ase_write

    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    poscar = case_dir / POSCAR_NAME
    try:
        ase_write(str(poscar), atoms, format="vasp")
    except Exception as exc:  # pragma: no cover - only on a broken ase install
        raise ExportError(f"could not write {poscar}: {exc}") from exc

    extra: list[Path] = []
    for fmt in normalize_formats(formats):
        target = case_dir / f"{EXTRA_STEM}.{_extension_for(fmt)}"
        try:
            ase_write(str(target), atoms, format=fmt)
        except Exception as exc:
            raise ExportError(f"cannot write format {fmt!r}: {exc}") from exc
        extra.append(target)

    payload: dict[str, Any] = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "generated_by": "vasp-auto-build",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request": request,
        "controller_state": controller_state,
        "recipe": recipe,
        "recipe_hash": recipe_hash,
        "atoms_hash": atoms_hash(atoms),
        "invariants": structure_invariants(atoms),
        "tool_sequence": list(tool_sequence),
        "files": {
            "poscar": poscar.name,
            "extra": [path.name for path in extra],
        },
        "model": model_info or {},
        "preflight_advisory": advisory,
        "value_mismatches": mismatches or [],
    }
    sidecar = case_dir / SIDECAR_NAME
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ExportResult(case_dir=case_dir, poscar=poscar, sidecar=sidecar, extra=tuple(extra))


def owns_case_dir(case_dir: Path) -> bool:
    """True when the directory is free, or is one we previously wrote.

    Guards against silently overwriting a hand-made VASP case that happens to sit
    at the chosen path.
    """
    case_dir = Path(case_dir)
    if not case_dir.exists():
        return True
    if not (case_dir / POSCAR_NAME).exists():
        return True
    return (case_dir / SIDECAR_NAME).exists()
