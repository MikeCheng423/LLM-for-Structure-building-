"""Pure-Python POSCAR structure tools: supercell, vacancy, substitution.

Keeps the engine free of ASE/numpy; ASE-backed builders live in ase_tools.py.
Velocities / predictor-corrector blocks after the coordinates are dropped.
"""
from __future__ import annotations

import math
from pathlib import Path


def _cell_volume(lattice: list[list[float]]) -> float:
    a, b, c = lattice
    return abs(
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def read_poscar(poscar_path: Path) -> dict:
    lines = Path(poscar_path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise ValueError(f"POSCAR is too short: {poscar_path}")

    comment = lines[0]
    scale = float(lines[1].split()[0])
    lattice = [[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]
    if scale < 0:
        # Negative scale is the VASP "target cell volume" convention.
        scale = (abs(scale) / _cell_volume(lattice)) ** (1.0 / 3.0)

    elements = lines[5].split()
    if not elements or all(part.isdigit() for part in elements):
        raise ValueError(
            f"POSCAR must use VASP 5 style element symbols on line 6: {poscar_path}"
        )
    counts = [int(x) for x in lines[6].split()]
    if len(elements) != len(counts):
        raise ValueError(f"Element symbols and counts do not match: {poscar_path}")

    idx = 7
    selective = lines[idx].strip().lower().startswith("s")
    if selective:
        idx += 1
    cartesian = lines[idx].strip().lower().startswith(("c", "k"))
    idx += 1

    natoms = sum(counts)
    coord_lines = lines[idx:idx + natoms]
    if len(coord_lines) != natoms:
        raise ValueError(f"POSCAR atom count does not match coordinates: {poscar_path}")

    coords = []
    flags = []
    for line in coord_lines:
        parts = line.split()
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        flags.append(parts[3:6] if selective else [])

    return {
        "comment": comment,
        "scale": scale,
        "lattice": lattice,
        "elements": elements,
        "counts": counts,
        "selective": selective,
        "cartesian": cartesian,
        "coords": coords,
        "flags": flags,
    }


def write_poscar(struct: dict, poscar_path: Path):
    lines = [struct["comment"], f"{struct['scale']:.14f}"]
    for row in struct["lattice"]:
        lines.append(f"  {row[0]: .16f} {row[1]: .16f} {row[2]: .16f}")
    lines.append("  " + "  ".join(struct["elements"]))
    lines.append("  " + "  ".join(str(c) for c in struct["counts"]))
    if struct["selective"]:
        lines.append("Selective dynamics")
    lines.append("Cartesian" if struct["cartesian"] else "Direct")
    for coord, flag in zip(struct["coords"], struct["flags"]):
        line = f"  {coord[0]: .16f} {coord[1]: .16f} {coord[2]: .16f}"
        if flag:
            line += "  " + " ".join(flag)
        lines.append(line)

    poscar_path = Path(poscar_path)
    poscar_path.parent.mkdir(parents=True, exist_ok=True)
    poscar_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def per_atom_symbols(struct: dict) -> list[str]:
    symbols = []
    for element, count in zip(struct["elements"], struct["counts"]):
        symbols.extend([element] * count)
    return symbols


def _regroup(symbols: list[str]) -> tuple[list[str], list[int]]:
    elements = []
    counts = []
    for symbol in symbols:
        if elements and elements[-1] == symbol:
            counts[-1] += 1
        else:
            elements.append(symbol)
            counts.append(1)
    return elements, counts


def make_supercell(struct: dict, repeat: tuple[int, int, int]) -> dict:
    na, nb, nc = (int(n) for n in repeat)
    if min(na, nb, nc) < 1:
        raise ValueError(f"Supercell repeats must be >= 1, got {repeat}")

    factors = (na, nb, nc)
    new_lattice = [[x * n for x in row] for row, n in zip(struct["lattice"], factors)]

    new_coords = []
    new_flags = []
    for coord, flag in zip(struct["coords"], struct["flags"]):
        for ia in range(na):
            for ib in range(nb):
                for ic in range(nc):
                    if struct["cartesian"]:
                        shift = [
                            ia * struct["lattice"][0][k]
                            + ib * struct["lattice"][1][k]
                            + ic * struct["lattice"][2][k]
                            for k in range(3)
                        ]
                        new_coords.append([coord[k] + shift[k] for k in range(3)])
                    else:
                        offsets = (ia, ib, ic)
                        new_coords.append(
                            [(coord[k] + offsets[k]) / factors[k] for k in range(3)]
                        )
                    new_flags.append(list(flag))

    images = na * nb * nc
    return {
        **struct,
        "lattice": new_lattice,
        "counts": [count * images for count in struct["counts"]],
        "coords": new_coords,
        "flags": new_flags,
    }


def _check_index(struct: dict, index: int):
    natoms = sum(struct["counts"])
    if not 1 <= index <= natoms:
        raise ValueError(f"Atom index {index} out of range 1..{natoms}")


def make_vacancy(struct: dict, index: int) -> dict:
    """Remove atom `index` (1-based, POSCAR order)."""
    _check_index(struct, index)
    symbols = per_atom_symbols(struct)
    position = index - 1

    symbols.pop(position)
    coords = [c for i, c in enumerate(struct["coords"]) if i != position]
    flags = [f for i, f in enumerate(struct["flags"]) if i != position]
    elements, counts = _regroup(symbols)

    return {**struct, "elements": elements, "counts": counts, "coords": coords, "flags": flags}


def substitute(struct: dict, index: int, new_element: str) -> dict:
    """Replace the element of atom `index` (1-based, POSCAR order)."""
    _check_index(struct, index)
    symbols = per_atom_symbols(struct)
    symbols[index - 1] = new_element
    elements, counts = _regroup(symbols)
    return {**struct, "elements": elements, "counts": counts}


def add_interstitial(struct: dict, element: str, position: tuple[float, float, float]) -> dict:
    """Insert an `element` atom at `position` (in the structure's own
    coordinate mode: fractional for Direct, Å for Cartesian)."""
    symbols = per_atom_symbols(struct)
    symbols.append(element)
    coords = [list(c) for c in struct["coords"]]
    coords.append([float(x) for x in position])
    flags = [list(f) for f in struct["flags"]]
    flags.append(["T", "T", "T"] if struct["selective"] else [])
    elements, counts = _regroup(symbols)
    return {**struct, "elements": elements, "counts": counts, "coords": coords, "flags": flags}


# --------------------------------------------------------------- coordinates

def _matvec(matrix, vector):
    return [sum(vector[i] * matrix[i][k] for i in range(3)) for k in range(3)]


def _invert3(m):
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    if abs(det) < 1e-12:
        raise ValueError("Lattice matrix is singular")
    cof = [
        [
            (m[(i + 1) % 3][(k + 1) % 3] * m[(i + 2) % 3][(k + 2) % 3]
             - m[(i + 1) % 3][(k + 2) % 3] * m[(i + 2) % 3][(k + 1) % 3]) / det
            for i in range(3)
        ]
        for k in range(3)
    ]
    return cof


def _frac_coords(struct: dict) -> list[list[float]]:
    """All coordinates as fractional, whatever the structure's mode."""
    if not struct["cartesian"]:
        return [list(c) for c in struct["coords"]]
    inverse = _invert3(struct["lattice"])
    return [_matvec(inverse, c) for c in struct["coords"]]


def cell_lengths(struct: dict) -> list[float]:
    """|a|, |b|, |c| in Å (the universal scale factor included)."""
    return [
        struct["scale"] * sum(x * x for x in row) ** 0.5
        for row in struct["lattice"]
    ]


def scale_cell(struct: dict, factors: tuple[float, float, float]) -> dict:
    """Scale the three lattice vectors; relative atom positions are kept.

    Fractional coordinates stay unchanged (the cell stretches under the
    atoms); Cartesian coordinates are remapped through the fractional ones.
    """
    factors = [float(f) for f in factors]
    if min(factors) <= 0:
        raise ValueError(f"Cell scale factors must be positive, got {factors}")
    new_lattice = [[x * f for x in row] for row, f in zip(struct["lattice"], factors)]
    if struct["cartesian"]:
        fracs = _frac_coords(struct)
        coords = [_matvec(new_lattice, f) for f in fracs]
    else:
        coords = [list(c) for c in struct["coords"]]
    return {**struct, "lattice": new_lattice, "coords": coords}


def move_atom(struct: dict, index: int, vector: tuple[float, float, float],
              absolute: bool = False) -> dict:
    """Move atom `index` (1-based): translate by `vector`, or place it at
    `vector` when absolute=True. Vector is in the structure's coordinate
    mode (fractional for Direct, Å for Cartesian)."""
    _check_index(struct, index)
    coords = [list(c) for c in struct["coords"]]
    position = index - 1
    if absolute:
        coords[position] = [float(x) for x in vector]
    else:
        coords[position] = [c + float(x) for c, x in zip(coords[position], vector)]
    return {**struct, "coords": coords}


def rotate_atoms(struct: dict, indices: list[int], axis, angle: float,
                 center: tuple[float, float, float] | None = None) -> dict:
    """Rotate the given atoms (1-based) by `angle` degrees about `axis`.

    Axis is "x"/"y"/"z" or a Cartesian 3-vector; `center` is a Cartesian
    point in Å (default: centroid of the selected atoms).
    """
    for index in indices:
        _check_index(struct, index)
    if isinstance(axis, str):
        try:
            axis = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis.lower()]
        except KeyError:
            raise ValueError(f'Rotation axis must be "x", "y", "z" or a 3-vector, got {axis!r}')
    axis = [float(x) for x in axis]
    norm = math.sqrt(sum(x * x for x in axis))
    if norm < 1e-12:
        raise ValueError("Rotation axis must be a non-zero vector")
    ux, uy, uz = (x / norm for x in axis)

    theta = math.radians(float(angle))
    cos, sin = math.cos(theta), math.sin(theta)
    # Rodrigues rotation matrix (rows act on column vectors).
    rot = [
        [cos + ux * ux * (1 - cos), ux * uy * (1 - cos) - uz * sin, ux * uz * (1 - cos) + uy * sin],
        [uy * ux * (1 - cos) + uz * sin, cos + uy * uy * (1 - cos), uy * uz * (1 - cos) - ux * sin],
        [uz * ux * (1 - cos) - uy * sin, uz * uy * (1 - cos) + ux * sin, cos + uz * uz * (1 - cos)],
    ]

    carts = cart_coords(struct)
    picked = [i - 1 for i in indices]
    if center is None:
        center = [sum(carts[i][k] for i in picked) / len(picked) for k in range(3)]
    center = [float(x) for x in center]

    coords = [list(c) for c in struct["coords"]]
    scale = struct["scale"]
    inverse = None if struct["cartesian"] else _invert3(scaled_lattice(struct))
    for i in picked:
        d = [carts[i][k] - center[k] for k in range(3)]
        rotated = [center[k] + sum(rot[k][j] * d[j] for j in range(3)) for k in range(3)]
        if struct["cartesian"]:
            coords[i] = [x / scale for x in rotated]
        else:
            coords[i] = _matvec(inverse, rotated)
    return {**struct, "coords": coords}


def parse_atom_selection(struct: dict, spec: str) -> list[int]:
    """Resolve an atom selection into sorted 1-based indices.

    Forms: "1,2,5-8" (indices/ranges), "z<0.25" or "z>0.7" (fractional
    height — the natural way to pick the bottom layers of a slab).
    """
    spec = str(spec).strip().replace(" ", "")
    natoms = sum(struct["counts"])

    if spec.lower().startswith("z<") or spec.lower().startswith("z>"):
        threshold = float(spec[2:])
        below = spec[1] == "<"
        fracs = _frac_coords(struct)
        indices = [
            i + 1 for i, frac in enumerate(fracs)
            if (frac[2] < threshold) == below
        ]
        if not indices:
            raise ValueError(f"No atoms match the selection {spec!r}")
        return indices

    indices = set()
    for part in spec.split(","):
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    for index in indices:
        if not 1 <= index <= natoms:
            raise ValueError(f"Atom index {index} out of range 1..{natoms}")
    if not indices:
        raise ValueError(f"Empty atom selection: {spec!r}")
    return sorted(indices)


def freeze_atoms(struct: dict, indices: list[int], axes: str = "XYZ") -> dict:
    """Freeze the given atoms (1-based) along the given axes.

    Turns on Selective dynamics; frozen axes get flag F (VASP: F = fixed,
    T = free). Atoms outside the selection keep their flags (default T T T).
    """
    axes = axes.upper()
    if not axes or any(a not in "XYZ" for a in axes):
        raise ValueError(f"Axes must be a combination of X, Y, Z — got {axes!r}")
    for index in indices:
        _check_index(struct, index)

    natoms = sum(struct["counts"])
    flags = [
        list(f) if f else ["T", "T", "T"]
        for f in (struct["flags"] if struct["selective"] else [[]] * natoms)
    ]
    frozen = set(indices)
    for i in range(natoms):
        if i + 1 in frozen:
            for axis_index, axis in enumerate("XYZ"):
                if axis in axes:
                    flags[i][axis_index] = "F"
    return {**struct, "selective": True, "flags": flags}


# --------------------------------------------------------- builder ops
# (see src/periodic_structure_builder_reimplementation.md — open-source
# reimplementation of the periodic-builder operations: cell <-> parameters,
# set-cell with frac/cart preservation, wrapping, coordination analysis and
# combining two structures with different unit cells)

# Covalent radii in Å (Cordero et al., Dalton Trans. 2008). Used for bond
# detection / coordination numbers; unknown elements fall back to 1.4 Å.
COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
    "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92,
    "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75,
    "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36,
    "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40,
    "At": 1.50, "Rn": 1.50, "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06,
    "Pa": 2.00, "U": 1.96, "Np": 1.90, "Pu": 1.87,
}


def scaled_lattice(struct: dict) -> list[list[float]]:
    """The lattice in Å with the universal scale factor applied."""
    scale = struct["scale"]
    return [[x * scale for x in row] for row in struct["lattice"]]


def cart_coords(struct: dict) -> list[list[float]]:
    """All coordinates as Cartesian Å, whatever the structure's mode."""
    lattice = scaled_lattice(struct)
    if struct["cartesian"]:
        scale = struct["scale"]
        return [[x * scale for x in c] for c in struct["coords"]]
    return [_matvec(lattice, c) for c in struct["coords"]]


def frac_coords(struct: dict) -> list[list[float]]:
    """All coordinates as fractional, whatever the structure's mode."""
    return _frac_coords(struct)


def cell_from_parameters(a: float, b: float, c: float,
                         alpha: float, beta: float, gamma: float) -> list[list[float]]:
    """Lattice matrix (rows = a, b, c vectors, Å) from cell parameters.

    Standard crystallographic convention: a along x, b in the xy plane.
    Angles in degrees: alpha = angle(b,c), beta = angle(a,c), gamma = angle(a,b).
    """
    if min(a, b, c) <= 0:
        raise ValueError(f"Cell lengths must be positive, got a={a}, b={b}, c={c}")
    al, be, ga = (math.radians(x) for x in (alpha, beta, gamma))
    if min(math.sin(ga), 1.0) <= 1e-9:
        raise ValueError(f"gamma = {gamma}° is degenerate")
    v1 = [a, 0.0, 0.0]
    v2 = [b * math.cos(ga), b * math.sin(ga), 0.0]
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq <= 0:
        raise ValueError(
            f"Impossible cell: a={a}, b={b}, c={c}, alpha={alpha}, beta={beta}, gamma={gamma}"
        )
    v3 = [cx, cy, math.sqrt(cz_sq)]
    return [v1, v2, v3]


def cell_parameters(lattice: list[list[float]]) -> dict:
    """a, b, c (Å) and alpha, beta, gamma (degrees) of a lattice matrix."""
    def norm(v):
        return math.sqrt(sum(x * x for x in v))

    def angle(u, v):
        dot = sum(ux * vx for ux, vx in zip(u, v))
        cosang = max(-1.0, min(1.0, dot / (norm(u) * norm(v))))
        return math.degrees(math.acos(cosang))

    a_v, b_v, c_v = lattice
    return {
        "a": norm(a_v), "b": norm(b_v), "c": norm(c_v),
        "alpha": angle(b_v, c_v), "beta": angle(a_v, c_v), "gamma": angle(a_v, b_v),
        "volume": _cell_volume(lattice),
    }


def set_cell(struct: dict, new_lattice: list[list[float]],
             preserve: str = "fractional") -> dict:
    """Replace the cell with `new_lattice` (Å, scale folded in afterwards).

    preserve="fractional": atoms keep their fractional coordinates — the
    structure deforms with the cell ("scale atoms").
    preserve="cartesian": atoms keep their absolute Å positions.
    """
    if preserve not in ("fractional", "cartesian"):
        raise ValueError(f"preserve must be 'fractional' or 'cartesian', got {preserve!r}")
    if _cell_volume(new_lattice) < 1e-9:
        raise ValueError("New cell is singular (zero volume)")

    if preserve == "fractional":
        fracs = _frac_coords(struct)
    else:
        carts = cart_coords(struct)
        inverse = _invert3(new_lattice)
        fracs = [_matvec(inverse, c) for c in carts]

    return {
        **struct,
        "scale": 1.0,
        "lattice": [list(row) for row in new_lattice],
        "cartesian": False,
        "coords": fracs,
    }


def wrap_to_cell(struct: dict) -> dict:
    """Wrap all atoms back into the [0,1) cell (fractional output)."""
    fracs = _frac_coords(struct)
    wrapped = [[x - math.floor(x) for x in f] for f in fracs]
    return {
        **struct,
        "scale": 1.0,
        "lattice": scaled_lattice(struct),
        "cartesian": False,
        "coords": wrapped,
    }


def delete_atoms(struct: dict, indices: list[int]) -> dict:
    """Remove several atoms at once (1-based POSCAR indices)."""
    unique = sorted(set(int(i) for i in indices), reverse=True)
    if not unique:
        raise ValueError("No atoms given to delete")
    result = struct
    for index in unique:
        result = make_vacancy(result, index)
    return result


def build_struct(comment: str, lattice: list[list[float]], symbols: list[str],
                 coords: list[list[float]], cartesian: bool = False,
                 flags: list[list[str]] | None = None) -> dict:
    """Assemble a structure dict from per-atom data (lattice in Å, scale 1)."""
    if len(symbols) != len(coords):
        raise ValueError(f"{len(symbols)} symbols but {len(coords)} coordinates")
    flags = [list(f) for f in flags] if flags else [[] for _ in symbols]
    if len(flags) != len(symbols):
        raise ValueError(f"{len(symbols)} symbols but {len(flags)} flag sets")
    selective = any(flags)
    if selective:
        flags = [f if f else ["T", "T", "T"] for f in flags]
    elements, counts = _regroup(list(symbols))
    return {
        "comment": comment or "structure",
        "scale": 1.0,
        "lattice": [list(row) for row in lattice],
        "elements": elements,
        "counts": counts,
        "selective": selective,
        "cartesian": bool(cartesian),
        "coords": [list(c) for c in coords],
        "flags": flags,
    }


def coordination(struct: dict, slack: float = 0.45) -> list[dict]:
    """Bond/coordination analysis with periodic images.

    Atoms i,j bond when d(i,j) < r_cov_i + r_cov_j + slack (Å) — the Jmol
    convention, robust for both short covalent bonds (H2) and ionic
    coordination (NaCl octahedra). Returns one entry per atom: {"index",
    "element", "coordination", "neighbors": [{"index", "element",
    "distance"}…]} (indices 1-based, distances Å).
    """
    symbols = per_atom_symbols(struct)
    carts = cart_coords(struct)
    lattice = scaled_lattice(struct)
    natoms = len(symbols)
    radii = [COVALENT_RADII.get(s, 1.4) for s in symbols]

    shifts = [
        [ia * lattice[0][k] + ib * lattice[1][k] + ic * lattice[2][k] for k in range(3)]
        for ia in (-1, 0, 1) for ib in (-1, 0, 1) for ic in (-1, 0, 1)
    ]
    neighbors: list[list[dict]] = [[] for _ in range(natoms)]
    for i in range(natoms):
        for j in range(natoms):
            cutoff = radii[i] + radii[j] + slack
            # Every bonded periodic image counts (an atom in a small cell can
            # coordinate the same neighbour through several images).
            for shift in shifts:
                d = math.sqrt(sum(
                    (carts[i][k] - carts[j][k] - shift[k]) ** 2 for k in range(3)
                ))
                if 1e-3 < d < cutoff:
                    neighbors[i].append(
                        {"index": j + 1, "element": symbols[j], "distance": round(d, 4)}
                    )
    return [
        {
            "index": i + 1,
            "element": symbols[i],
            "coordination": len(neighbors[i]),
            "neighbors": sorted(neighbors[i], key=lambda n: n["distance"]),
        }
        for i in range(natoms)
    ]


# ------------------------------------------------ prototype crystal library
# Pure-Python prototype structures for compounds ASE's bulk() cannot build
# (binary oxides, 2D sheets). Lattice constants are experimental values;
# relax before production use. Coordinates are fractional.

_SQRT3 = math.sqrt(3.0)

PROTOTYPES: dict[str, dict] = {
    "graphene": {
        "description": "single graphene sheet (hexagonal, vacuum along c)",
        "a": 2.468, "c": 15.0, "sheet": True,
        "atoms": [("C", (0.0, 0.0, 0.5)), ("C", (1 / 3, 2 / 3, 0.5))],
        "hexagonal": True,
    },
    "graphite": {
        "description": "AB (Bernal) graphite, P6_3/mmc",
        "a": 2.464, "c": 6.711, "sheet": False,
        "atoms": [
            ("C", (0.0, 0.0, 0.25)), ("C", (0.0, 0.0, 0.75)),
            ("C", (1 / 3, 2 / 3, 0.25)), ("C", (2 / 3, 1 / 3, 0.75)),
        ],
        "hexagonal": True,
    },
    "rutile-TiO2": {
        "description": "rutile TiO2, P4_2/mnm (u = 0.3048)",
        "a": 4.593, "c": 2.959, "sheet": False,
        "atoms": [
            ("Ti", (0.0, 0.0, 0.0)), ("Ti", (0.5, 0.5, 0.5)),
            ("O", (0.3048, 0.3048, 0.0)), ("O", (0.6952, 0.6952, 0.0)),
            ("O", (0.8048, 0.1952, 0.5)), ("O", (0.1952, 0.8048, 0.5)),
        ],
        "hexagonal": False,
    },
    "anatase-TiO2": {
        "description": "anatase TiO2, I4_1/amd conventional cell (u = 0.2066)",
        "a": 3.785, "c": 9.514, "sheet": False,
        "atoms": [
            ("Ti", (0.0, 0.0, 0.0)), ("Ti", (0.5, 0.5, 0.5)),
            ("Ti", (0.0, 0.5, 0.25)), ("Ti", (0.5, 0.0, 0.75)),
            ("O", (0.0, 0.0, 0.2066)), ("O", (0.0, 0.0, 0.7934)),
            ("O", (0.5, 0.5, 0.7066)), ("O", (0.5, 0.5, 0.2934)),
            ("O", (0.0, 0.5, 0.4566)), ("O", (0.0, 0.5, 0.0434)),
            ("O", (0.5, 0.0, 0.9566)), ("O", (0.5, 0.0, 0.5434)),
        ],
        "hexagonal": False,
    },
    "hBN": {
        "description": "single hexagonal boron nitride sheet (vacuum along c)",
        "a": 2.504, "c": 15.0, "sheet": True,
        "atoms": [("B", (0.0, 0.0, 0.5)), ("N", (1 / 3, 2 / 3, 0.5))],
        "hexagonal": True,
    },
}

# Forgiving lookup: "rutile", "TIO2-RUTILE", "anatase_tio2" all resolve.
_PROTOTYPE_ALIASES = {
    "rutile": "rutile-TiO2", "tio2": "rutile-TiO2", "tio2-rutile": "rutile-TiO2",
    "rutile-tio2": "rutile-TiO2", "anatase": "anatase-TiO2",
    "anatase-tio2": "anatase-TiO2", "tio2-anatase": "anatase-TiO2",
    "graphene": "graphene", "graphite": "graphite", "hbn": "hBN", "bn": "hBN",
}


def resolve_prototype(name: str) -> str:
    key = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    canonical = _PROTOTYPE_ALIASES.get(key)
    if canonical is None:
        for proto in PROTOTYPES:
            if proto.lower() == key:
                canonical = proto
                break
    if canonical is None:
        raise ValueError(
            f"Unknown prototype {name!r}. Available: " + ", ".join(sorted(PROTOTYPES))
        )
    return canonical


def make_prototype(name: str, a: float | None = None, c: float | None = None,
                   vacuum: float | None = None) -> dict:
    """Build a prototype crystal (graphene, graphite, rutile-TiO2,
    anatase-TiO2, hBN) as a structure dict. `a`/`c` override the tabulated
    lattice constants (Å); for 2D sheets `vacuum` sets the c box height."""
    proto = PROTOTYPES[resolve_prototype(name)]
    a = float(a) if a else proto["a"]
    c = float(c) if c else proto["c"]
    if proto["sheet"] and vacuum:
        c = float(vacuum)
    if min(a, c) <= 0:
        raise ValueError(f"Lattice constants must be positive, got a={a}, c={c}")

    if proto["hexagonal"]:
        lattice = [[a, 0.0, 0.0], [-a / 2.0, a * _SQRT3 / 2.0, 0.0], [0.0, 0.0, c]]
    else:
        lattice = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, c]]

    symbols = [element for element, _ in proto["atoms"]]
    coords = [list(position) for _, position in proto["atoms"]]
    return build_struct(f"{resolve_prototype(name)} ({proto['description']})",
                        lattice, symbols, coords)


def _in_plane_params(struct: dict) -> tuple[float, float, float]:
    """|a|, |b| (Å) and gamma (degrees) of the in-plane lattice vectors."""
    lattice = scaled_lattice(struct)
    params = cell_parameters(lattice)
    return params["a"], params["b"], params["gamma"]


def substitute_species(struct: dict, mapping: dict[str, str]) -> dict:
    """Replace all atoms of one element with another throughout the structure.

    ``mapping`` is e.g. ``{"Ti": "Sn"}`` to turn every Ti site into Sn,
    or ``{"Ti": "Sn", "O": "S"}`` for a double substitution.  Returns a new
    structure dict; the original is not mutated.
    """
    if not mapping:
        return struct
    new_symbols: list[str] = []
    for sym, coord in zip(
        [s for el, cnt in zip(struct["elements"], struct["counts"]) for s in [el] * cnt],
        struct["coords"],
    ):
        new_symbols.append(mapping.get(sym, sym))

    sub_label = ",".join(f"{k}→{v}" for k, v in mapping.items())
    comment = f"{struct['comment']} [{sub_label}]"
    return build_struct(
        comment,
        struct["lattice"],
        new_symbols,
        struct["coords"],
        cartesian=struct.get("cartesian", False),
        flags=struct.get("flags") or None,
    )


def match_supercells(host: dict, guest: dict, max_repeat: int = 6,
                     max_strain: float = 0.1, gamma_tol: float = 8.0,
                     max_results: int = 8) -> list[dict]:
    """Suggest in-plane supercell pairs that bring two different unit cells
    into registry for stacking (e.g. a TiO2 slab on graphene).

    Tries every (i x j) host and (k x l) guest repetition up to `max_repeat`
    along a and b, and reports combinations where straining the guest onto
    the host supercell needs at most `max_strain` (fractional) and the cell
    angles agree within `gamma_tol` degrees. Only diagonal supercells are
    searched (no rotated coincidence lattices). Sorted by strain, then size;
    proportional duplicates (2x2/2x2 vs 1x1/1x1) are dropped.

    Each entry: {"host_repeat", "guest_repeat", "strain_a", "strain_b",
    "strain_pct", "gamma_mismatch_deg", "host_atoms", "guest_atoms",
    "host_a", "host_b", "guest_a", "guest_b"} (lengths are the supercell
    lengths in Å).
    """
    host_a, host_b, host_gamma = _in_plane_params(host)
    guest_a, guest_b, guest_gamma = _in_plane_params(guest)
    host_natoms = sum(host["counts"])
    guest_natoms = sum(guest["counts"])

    gamma_mismatch = abs(host_gamma - guest_gamma)
    # The in-plane axes of a hexagonal cell can be paired at 60 or 120
    # degrees depending on the b-vector convention; both describe the same
    # sheet, so compare against the supplementary angle too.
    gamma_mismatch = min(gamma_mismatch, abs(180.0 - host_gamma - guest_gamma))
    if gamma_mismatch > gamma_tol:
        return []

    candidates = []
    for i in range(1, max_repeat + 1):
        for k in range(1, max_repeat + 1):
            if math.gcd(i, k) > 1:
                reducible_a = True
            else:
                reducible_a = False
            strain_a = (guest_a * k - host_a * i) / (host_a * i)
            if abs(strain_a) > max_strain:
                continue
            for j in range(1, max_repeat + 1):
                for l in range(1, max_repeat + 1):
                    if reducible_a and math.gcd(j, l) > 1:
                        continue  # a smaller proportional pair already covers this
                    strain_b = (guest_b * l - host_b * j) / (host_b * j)
                    if abs(strain_b) > max_strain:
                        continue
                    candidates.append({
                        "host_repeat": (i, j),
                        "guest_repeat": (k, l),
                        "strain_a": strain_a,
                        "strain_b": strain_b,
                        "strain_pct": max(abs(strain_a), abs(strain_b)) * 100.0,
                        "gamma_mismatch_deg": gamma_mismatch,
                        "host_atoms": host_natoms * i * j,
                        "guest_atoms": guest_natoms * k * l,
                        "host_a": host_a * i, "host_b": host_b * j,
                        "guest_a": guest_a * k, "guest_b": guest_b * l,
                    })

    # Round the strain for sorting so equal-strain candidates (e.g. 1x1 vs a
    # floating-point-identical 5x5) tie-break by cell size, smallest first.
    candidates.sort(
        key=lambda m: (round(m["strain_pct"], 4), m["host_atoms"] + m["guest_atoms"])
    )
    return candidates[:max_results]


def combine_structures(host: dict, guest: dict, mode: str = "stack",
                       gap: float = 2.0, vacuum: float = 10.0,
                       shift: tuple[float, float] = (0.0, 0.0),
                       strain_guest: bool = False) -> dict:
    """Merge two structures with different unit cells into one host cell.

    mode="stack" (e.g. an Au crystal deposited on a graphite sheet): the host
    keeps its in-plane a/b vectors; the guest is placed `gap` Å above the
    highest host atom along the c direction, and the c axis is extended to
    leave `vacuum` Å above the guest. `shift` translates the guest laterally
    (fractions of the host a/b vectors). With strain_guest=True the guest's
    fractional in-plane coordinates are re-expressed in the host a/b vectors
    (the guest lattice is strained to match the host — right for epitaxy);
    otherwise the guest keeps its own absolute geometry and is centred over
    the host cell (right for clusters / incommensurate deposits).

    mode="insert": the host cell is kept unchanged and the guest atoms are
    translated by `shift` (+ gap along z, Å) and dropped in — for molecules
    in pores or pre-built composites. Atoms falling outside the host cell
    are NOT wrapped; wrap explicitly if needed.
    """
    if mode not in ("stack", "insert"):
        raise ValueError(f"mode must be 'stack' or 'insert', got {mode!r}")

    host_lat = scaled_lattice(host)
    host_cart = cart_coords(host)
    host_symbols = per_atom_symbols(host)
    host_flags = (
        [list(f) if f else ["T", "T", "T"] for f in host["flags"]]
        if host["selective"] else [[] for _ in host_symbols]
    )
    guest_symbols = per_atom_symbols(guest)
    guest_flags = (
        [list(f) if f else ["T", "T", "T"] for f in guest["flags"]]
        if guest["selective"] else [[] for _ in guest_symbols]
    )

    if not host_symbols or not guest_symbols:
        raise ValueError("Both structures need at least one atom")

    if mode == "insert":
        offset = [float(shift[0]), float(shift[1]), float(gap)]
        guest_cart = [
            [c[k] + offset[k] for k in range(3)] for c in cart_coords(guest)
        ]
        new_lat = host_lat
    else:
        if strain_guest:
            # Re-express guest fractional a/b in the host vectors: the guest
            # is strained onto the host lattice (epitaxial match).
            guest_frac = _frac_coords(guest)
            guest_lat = scaled_lattice(guest)
            guest_cart = []
            for f in guest_frac:
                fa, fb = f[0] - math.floor(f[0]), f[1] - math.floor(f[1])
                in_plane = [fa * host_lat[0][k] + fb * host_lat[1][k] for k in range(3)]
                z = f[2] * guest_lat[2][2]
                guest_cart.append([in_plane[0], in_plane[1], in_plane[2] + z])
        else:
            # Keep the guest's own geometry; centre it over the host cell.
            guest_cart = cart_coords(guest)
            host_center = [
                (host_lat[0][k] + host_lat[1][k]) / 2.0 for k in range(2)
            ]
            gc = [sum(c[k] for c in guest_cart) / len(guest_cart) for k in range(2)]
            guest_cart = [
                [c[0] + host_center[0] - gc[0], c[1] + host_center[1] - gc[1], c[2]]
                for c in guest_cart
            ]
        # Lateral shift in fractions of the host a/b vectors.
        lateral = [
            float(shift[0]) * host_lat[0][k] + float(shift[1]) * host_lat[1][k]
            for k in range(3)
        ]
        guest_cart = [[c[k] + lateral[k] for k in range(3)] for c in guest_cart]

        # Drop the guest `gap` Å above the highest host atom (along z).
        host_top = max(c[2] for c in host_cart)
        guest_bottom = min(c[2] for c in guest_cart)
        dz = host_top + float(gap) - guest_bottom
        guest_cart = [[c[0], c[1], c[2] + dz] for c in guest_cart]

        # Extend c so `vacuum` Å sits above the guest. c must have a +z part.
        c_vec = host_lat[2]
        if c_vec[2] <= 1e-6:
            raise ValueError("Stack mode needs a host c vector with a positive z component")
        top = max(c[2] for c in guest_cart)
        needed = top + float(vacuum)
        factor = max(needed / c_vec[2], 1.0)
        new_lat = [list(host_lat[0]), list(host_lat[1]),
                   [x * factor for x in c_vec]]

    symbols = host_symbols + guest_symbols
    carts = host_cart + guest_cart
    selective = host["selective"] or guest["selective"]
    flags = host_flags + guest_flags if selective else None

    inverse = _invert3(new_lat)
    fracs = [_matvec(inverse, c) for c in carts]
    combined = build_struct(
        f"{host['comment'].strip()} + {guest['comment'].strip()}",
        new_lat, symbols, fracs, cartesian=False,
        flags=flags if selective else None,
    )
    if selective:
        combined["selective"] = True
        combined["flags"] = [f if f else ["T", "T", "T"] for f in combined["flags"]]
    return combined


def add_adsorbate(struct: dict, element: str, anchor_index: int, height: float) -> dict:
    """Place an adsorbate directly above atom `anchor_index` at `height` Å
    (along Cartesian z — the surface normal of a standard slab). `element` is a
    single atom (e.g. "O") or a molecule name ASE knows (e.g. "CO2", "H2O"): a
    molecule is built with ASE and its lowest atom sits at `height`."""
    _check_index(struct, anchor_index)
    scale = struct["scale"]
    lattice = scaled_lattice(struct)
    anchor_frac = _frac_coords(struct)[anchor_index - 1]
    base = _matvec(lattice, anchor_frac)
    base[2] += float(height)

    if element in COVALENT_RADII:            # single atom — no ASE needed
        symbols, offsets = [element], [[0.0, 0.0, 0.0]]
    else:                                    # molecule (CO2, H2O, …) via ASE
        from ase_auto_build.ase_tools import molecule_positions
        symbols, offsets = molecule_positions(element)

    inv = None if struct["cartesian"] else _invert3(lattice)
    for sym, off in zip(symbols, offsets):
        cart = [base[i] + off[i] for i in range(3)]
        position = [x / scale for x in cart] if struct["cartesian"] else _matvec(inv, cart)
        struct = add_interstitial(struct, sym, tuple(position))
    return struct
