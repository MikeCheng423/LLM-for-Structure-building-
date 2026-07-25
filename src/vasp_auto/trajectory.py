"""Trajectory extraction for animations: XDATCAR (relax/MD) and NEB images.

Pure Python — returns cartesian frames ready for a viewer or export.
"""
from __future__ import annotations

from pathlib import Path

from vasp_auto.structure import per_atom_symbols, read_poscar


def _frac_to_cart(coord, lattice):
    return [sum(coord[i] * lattice[i][k] for i in range(3)) for k in range(3)]


def read_xdatcar(xdatcar_path: Path) -> dict | None:
    """Parse XDATCAR into {comment, lattice, symbols, frames} (fractional).

    Handles the common fixed-cell layout (one header, repeated
    'Direct configuration=' blocks) and tolerates repeated headers.
    """
    xdatcar_path = Path(xdatcar_path)
    if not xdatcar_path.exists():
        return None
    lines = xdatcar_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 8:
        return None

    try:
        scale = float(lines[1].split()[0])
        lattice = [[float(x) * scale for x in lines[i].split()[:3]] for i in (2, 3, 4)]
        elements = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
    except (ValueError, IndexError):
        return None
    natoms = sum(counts)

    symbols = []
    for element, count in zip(elements, counts):
        symbols.extend([element] * count)

    frames = []
    i = 7
    while i < len(lines):
        if "configuration" in lines[i].lower():
            block = lines[i + 1:i + 1 + natoms]
            frame = []
            try:
                for line in block:
                    parts = line.split()
                    frame.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except (ValueError, IndexError):
                break
            if len(frame) == natoms:
                frames.append(frame)
            i += 1 + natoms
        else:
            i += 1

    if not frames:
        return None
    return {"comment": lines[0], "lattice": lattice, "symbols": symbols, "frames": frames}


def _neb_image_dirs(job_dir: Path) -> list[Path]:
    return [p for p in sorted(job_dir.iterdir()) if p.is_dir() and p.name.isdigit()]


def _poscar_frame(poscar_path: Path) -> dict:
    struct = read_poscar(poscar_path)
    scale = struct["scale"]
    lattice = [[x * scale for x in row] for row in struct["lattice"]]
    if struct["cartesian"]:
        coords = [[x * scale for x in c] for c in struct["coords"]]
    else:
        coords = struct["coords"]
    return {
        "lattice": lattice,
        "symbols": per_atom_symbols(struct),
        "frac": None if struct["cartesian"] else coords,
        "cart": coords if struct["cartesian"] else None,
    }


def oszicar_energies(job_dir: Path) -> list[float] | None:
    """Free energy F (eV) of each ionic step from OSZICAR, or None.

    Ionic-step lines look like ``   1 F= -.27864425E+02 E0= ...``; the
    electronic-step lines between them have no ``F=`` column.
    """
    path = Path(job_dir) / "OSZICAR"
    if not path.exists():
        return None
    energies = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) > 2 and parts[1] == "F=":
            try:
                energies.append(float(parts[2]))
            except ValueError:
                pass
    return energies or None


def job_trajectory(job_dir: Path) -> dict | None:
    """Animation frames for a job directory, in cartesian Å.

    NEB jobs: one frame per image (CONTCAR when present, else POSCAR).
    Relax/MD jobs: XDATCAR frames; falls back to POSCAR → CONTCAR (2 frames).
    Returns {kind, comment, lattice, symbols, frames} or None.
    """
    job_dir = Path(job_dir)
    if not job_dir.exists():
        return None

    image_dirs = _neb_image_dirs(job_dir)
    if image_dirs:
        frames = []
        lattice = None
        symbols = None
        for image_dir in image_dirs:
            poscar = image_dir / "CONTCAR"
            if not poscar.exists() or poscar.stat().st_size == 0:
                poscar = image_dir / "POSCAR"
            if not poscar.exists():
                continue
            data = _poscar_frame(poscar)
            lattice = lattice or data["lattice"]
            symbols = symbols or data["symbols"]
            cart = data["cart"] or [_frac_to_cart(c, data["lattice"]) for c in data["frac"]]
            frames.append(cart)
        if len(frames) >= 2:
            return {
                "kind": "neb",
                "comment": f"NEB path ({len(frames)} images)",
                "lattice": lattice,
                "symbols": symbols,
                "frames": frames,
            }
        return None

    xdatcar = read_xdatcar(job_dir / "XDATCAR")
    if xdatcar and len(xdatcar["frames"]) >= 2:
        return {
            "kind": "relax",
            "comment": xdatcar["comment"],
            "lattice": xdatcar["lattice"],
            "symbols": xdatcar["symbols"],
            "frames": [
                [_frac_to_cart(c, xdatcar["lattice"]) for c in frame]
                for frame in xdatcar["frames"]
            ],
        }

    poscar, contcar = job_dir / "POSCAR", job_dir / "CONTCAR"
    if poscar.exists() and contcar.exists() and contcar.stat().st_size > 0:
        first, last = _poscar_frame(poscar), _poscar_frame(contcar)
        to_cart = lambda d: d["cart"] or [_frac_to_cart(c, d["lattice"]) for c in d["frac"]]
        return {
            "kind": "endpoints",
            "comment": "initial → final",
            "lattice": first["lattice"],
            "symbols": first["symbols"],
            "frames": [to_cart(first), to_cart(last)],
        }

    return None
