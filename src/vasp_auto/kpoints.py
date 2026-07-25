"""KPOINTS generation: uniform meshes, density-based meshes, line-mode paths."""
from __future__ import annotations

import math
from pathlib import Path


# High-symmetry k-paths in reciprocal (fractional) coordinates for common
# lattices. Used when --kpath names a preset instead of listing points.
KPATH_PRESETS: dict[str, list[tuple[str, tuple[float, float, float]]]] = {
    "cubic": [
        ("G", (0.0, 0.0, 0.0)),
        ("X", (0.0, 0.5, 0.0)),
        ("M", (0.5, 0.5, 0.0)),
        ("G", (0.0, 0.0, 0.0)),
        ("R", (0.5, 0.5, 0.5)),
        ("X", (0.0, 0.5, 0.0)),
    ],
    "fcc": [
        ("G", (0.0, 0.0, 0.0)),
        ("X", (0.5, 0.0, 0.5)),
        ("W", (0.5, 0.25, 0.75)),
        ("K", (0.375, 0.375, 0.75)),
        ("G", (0.0, 0.0, 0.0)),
        ("L", (0.5, 0.5, 0.5)),
    ],
    "bcc": [
        ("G", (0.0, 0.0, 0.0)),
        ("H", (0.5, -0.5, 0.5)),
        ("N", (0.0, 0.0, 0.5)),
        ("G", (0.0, 0.0, 0.0)),
        ("P", (0.25, 0.25, 0.25)),
    ],
    "hex": [
        ("G", (0.0, 0.0, 0.0)),
        ("M", (0.5, 0.0, 0.0)),
        ("K", (1 / 3, 1 / 3, 0.0)),
        ("G", (0.0, 0.0, 0.0)),
        ("A", (0.0, 0.0, 0.5)),
    ],
}


def parse_mesh(text: str | tuple) -> tuple[int, int, int]:
    """Parse '4', '4x4x1', or '4 4 1' into a 3-tuple of subdivisions."""
    if isinstance(text, (tuple, list)):
        parts = [int(x) for x in text]
    else:
        parts = [int(p) for p in str(text).lower().replace("x", " ").replace(",", " ").split()]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    raise ValueError(f"Invalid k-mesh: {text!r} (use N or NxNxN)")


def _cross(u, v):
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def _norm(u):
    return math.sqrt(_dot(u, u))


def read_lattice_from_poscar(poscar_path: Path) -> list[list[float]]:
    """Return the scaled lattice vectors (Å) from a POSCAR."""
    lines = Path(poscar_path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        raise ValueError(f"POSCAR is too short to contain a lattice: {poscar_path}")

    scale = float(lines[1].split()[0])
    lattice = [[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]

    if scale < 0:
        # Negative scale means target cell volume.
        volume = abs(_dot(lattice[0], _cross(lattice[1], lattice[2])))
        scale = (abs(scale) / volume) ** (1.0 / 3.0)

    return [[x * scale for x in row] for row in lattice]


def reciprocal_lengths(lattice: list[list[float]]) -> list[float]:
    """|b_i| in 2π/Å for the three reciprocal lattice vectors."""
    a1, a2, a3 = lattice
    volume = abs(_dot(a1, _cross(a2, a3)))
    two_pi = 2.0 * math.pi
    return [
        two_pi * _norm(_cross(a2, a3)) / volume,
        two_pi * _norm(_cross(a3, a1)) / volume,
        two_pi * _norm(_cross(a1, a2)) / volume,
    ]


def mesh_from_spacing(poscar_path: Path, kspacing: float) -> tuple[int, int, int]:
    """Mesh from a maximum k-point spacing in 1/Å (VASP KSPACING convention)."""
    if kspacing <= 0:
        raise ValueError("kspacing must be positive")
    lengths = reciprocal_lengths(read_lattice_from_poscar(poscar_path))
    return tuple(max(1, math.ceil(length / kspacing)) for length in lengths)


def mesh_kpoints_text(mesh: tuple[int, int, int], mode: str = "gamma", shift=(0, 0, 0)) -> str:
    centering = "Gamma" if mode.lower().startswith("g") else "Monkhorst-Pack"
    return "\n".join(
        [
            "Automatic mesh",
            "0",
            centering,
            f"{mesh[0]} {mesh[1]} {mesh[2]}",
            f"{shift[0]} {shift[1]} {shift[2]}",
            "",
        ]
    )


def guess_lattice_type(lattice: list[list[float]], tol: float = 0.01) -> str:
    """Heuristically guess the lattice type from cell vectors (pure Python, no spglib).

    Returns one of: "cubic", "fcc", "bcc", "hex", or "generic".

    Approach:
    - Compute |a|, |b|, |c| and the dot products a·b, a·c, b·c.
    - Derive the conventional cell angles α, β, γ from the dot products.
    - Classify by length ratios and angles (within tolerance).
    """
    a1, a2, a3 = lattice
    la = _norm(a1)
    lb = _norm(a2)
    lc = _norm(a3)

    if la < 1e-10 or lb < 1e-10 or lc < 1e-10:
        return "generic"

    cos_alpha = _dot(a2, a3) / (lb * lc)  # angle between b and c
    cos_beta = _dot(a1, a3) / (la * lc)   # angle between a and c
    cos_gamma = _dot(a1, a2) / (la * lb)  # angle between a and b

    def near(x, y):
        return abs(x - y) < tol

    # Clamp cosines to [-1, 1] before math.acos
    import math
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    cos_beta = max(-1.0, min(1.0, cos_beta))
    cos_gamma = max(-1.0, min(1.0, cos_gamma))

    alpha = math.degrees(math.acos(cos_alpha))
    beta = math.degrees(math.acos(cos_beta))
    gamma = math.degrees(math.acos(cos_gamma))

    all_equal_lengths = near(la, lb) and near(lb, lc)
    all_right_angles = near(alpha, 90.0) and near(beta, 90.0) and near(gamma, 90.0)
    ab_equal = near(la, lb)

    if all_equal_lengths and all_right_angles:
        # Cubic family — distinguish SC/FCC/BCC by the ratio la/lc is trivially 1.
        # A simple cubic primitive cell: a=b=c, 90/90/90.
        return "cubic"

    # FCC conventional primitive: a=b=c, α=β=γ=60°
    if all_equal_lengths and near(alpha, 60.0) and near(beta, 60.0) and near(gamma, 60.0):
        return "fcc"

    # BCC conventional primitive: a=b=c, α=β=γ=109.47°
    bcc_angle = math.degrees(math.acos(-1.0 / 3.0))  # ≈109.47°
    if all_equal_lengths and near(alpha, bcc_angle) and near(beta, bcc_angle) and near(gamma, bcc_angle):
        return "bcc"

    # Hexagonal: a=b≠c, α=β=90°, γ=120°
    if ab_equal and not near(la, lc) and near(alpha, 90.0) and near(beta, 90.0) and near(gamma, 120.0):
        return "hex"

    return "generic"


def auto_kpath(poscar_path: Path) -> list[tuple[str, tuple[float, float, float]]]:
    """Read the lattice from a POSCAR, guess the lattice type, and return the k-path preset.

    Raises ValueError for generic lattices (user must specify --kpath explicitly).
    """
    lattice = read_lattice_from_poscar(poscar_path)
    ltype = guess_lattice_type(lattice)
    if ltype == "generic":
        raise ValueError(
            f"Cannot auto-detect k-path for {poscar_path}: lattice type is 'generic'. "
            "Specify --kpath explicitly (cubic, fcc, bcc, hex, or 'G 0 0 0; X 0.5 0 0')."
        )
    return list(KPATH_PRESETS[ltype])


def parse_kpath(spec: str) -> list[tuple[str, tuple[float, float, float]]]:
    """Parse a preset name ('fcc') or 'G 0 0 0; X 0.5 0 0.5; ...' into points."""
    name = spec.strip().lower()
    if name in KPATH_PRESETS:
        return list(KPATH_PRESETS[name])

    points = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.replace(",", " ").split()
        if len(parts) != 4:
            raise ValueError(
                f"Invalid k-path segment: {item!r} (expected 'LABEL kx ky kz') "
                f"or a preset name: {', '.join(sorted(KPATH_PRESETS))}"
            )
        points.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))

    if len(points) < 2:
        raise ValueError("A k-path needs at least two points")
    return points


def line_mode_text(points, divisions: int = 20) -> str:
    lines = [
        "k-path for band structure",
        str(divisions),
        "Line-mode",
        "Reciprocal",
    ]
    for (label_a, point_a), (label_b, point_b) in zip(points, points[1:]):
        lines.append(f"{point_a[0]:.8f} {point_a[1]:.8f} {point_a[2]:.8f} ! {label_a}")
        lines.append(f"{point_b[0]:.8f} {point_b[1]:.8f} {point_b[2]:.8f} ! {label_b}")
        lines.append("")
    return "\n".join(lines)


def kpoints_text_from_spec(spec: dict, poscar_path: Path | None = None) -> str:
    """Build KPOINTS text from a spec dict.

    spec keys: mode (gamma|mp|line|spacing), mesh, spacing, kpath, divisions.
    The special kpath value "auto" auto-detects the lattice type from the POSCAR.
    """
    mode = (spec.get("mode") or "gamma").lower()

    if mode == "line":
        if not spec.get("kpath"):
            raise ValueError("line-mode KPOINTS needs a kpath (preset name or point list)")
        kpath = spec["kpath"]
        if isinstance(kpath, str) and kpath.strip().lower() == "auto":
            if poscar_path is None:
                raise ValueError("--kpath auto needs a POSCAR path to detect the lattice type")
            points = auto_kpath(Path(poscar_path))
        elif isinstance(kpath, list):
            points = kpath
        else:
            points = parse_kpath(kpath)
        return line_mode_text(points, int(spec.get("divisions") or 20))

    if spec.get("mesh"):
        mesh = parse_mesh(spec["mesh"])
    elif spec.get("spacing"):
        if poscar_path is None:
            raise ValueError("density-based KPOINTS needs a POSCAR to read the lattice")
        mesh = mesh_from_spacing(poscar_path, float(spec["spacing"]))
    else:
        raise ValueError("KPOINTS spec needs a mesh or a spacing")

    return mesh_kpoints_text(mesh, mode="mp" if mode == "mp" else "gamma")


def write_kpoints(path: Path, text: str):
    Path(path).write_text(text, encoding="utf-8")
