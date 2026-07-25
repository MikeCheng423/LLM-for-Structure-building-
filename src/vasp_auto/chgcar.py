"""Volumetric VASP files (CHGCAR/LOCPOT/AECCAR): read, write, combine.

Pure Python, no numpy. Grids are stored as flat lists in VASP's Fortran
order (x fastest): index = ix + iy*nx + iz*nx*ny. For spin-polarised files
only the first block (total charge / potential) is read; the magnetisation
block and the PAW augmentation occupancies are ignored, which is exactly
what charge-density differences and Bader analysis need.
"""
from __future__ import annotations

import shutil
import subprocess
import re
from pathlib import Path


def read_volumetric(path: Path, spin: bool = False) -> dict:
    """Read a CHGCAR/LOCPOT/AECCAR file into {"poscar_lines", "grid", "data"}.

    CHGCAR-family files store rho*V_cell on the grid; LOCPOT stores the
    potential in eV directly. This function does not rescale either.

    With ``spin=True`` a spin-polarised CHGCAR's second grid block
    (magnetisation ρ↑−ρ↓) is also returned under ``"spin_data"`` — None when
    the file holds no second block (non-spin run, LOCPOT).
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 10:
        raise ValueError(f"Volumetric file is too short: {path}")

    counts = [int(x) for x in lines[6].split()]
    natoms = sum(counts)

    idx = 7
    if lines[idx].strip().lower().startswith("s"):
        idx += 1
    idx += 1  # coordinate-mode line
    idx += natoms

    # A blank separator line precedes the grid dimensions.
    while idx < len(lines) and not lines[idx].split():
        idx += 1
    poscar_lines = lines[:idx]

    grid_parts = lines[idx].split()
    if len(grid_parts) < 3:
        raise ValueError(f"Missing NGX NGY NGZ grid line in {path}")
    nx, ny, nz = (int(p) for p in grid_parts[:3])
    idx += 1

    npoints = nx * ny * nz

    def read_block(start: int) -> tuple[list[float], int]:
        block: list[float] = []
        i = start
        while i < len(lines) and len(block) < npoints:
            block.extend(float(x) for x in lines[i].split())
            i += 1
        return block, i

    data, idx = read_block(idx)
    if len(data) < npoints:
        raise ValueError(f"Grid data is truncated in {path}: {len(data)} < {npoints}")

    result = {"poscar_lines": poscar_lines, "grid": (nx, ny, nz), "data": data[:npoints]}

    if spin:
        # After the total block come PAW augmentation lines, then the grid
        # dimensions repeat to start the magnetisation block. Scan for that.
        result["spin_data"] = None
        dims = [str(nx), str(ny), str(nz)]
        while idx < len(lines):
            if lines[idx].split() == dims:
                spin_data, _ = read_block(idx + 1)
                if len(spin_data) >= npoints:
                    result["spin_data"] = spin_data[:npoints]
                break
            idx += 1

    return result


def write_volumetric(volume: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = volume["grid"]
    lines = list(volume["poscar_lines"])
    if lines and lines[-1].split():
        lines.append("")
    lines.append(f"  {nx}  {ny}  {nz}")

    data = volume["data"]
    for start in range(0, len(data), 5):
        lines.append(" ".join(f"{value: .11E}" for value in data[start:start + 5]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_same_grid(volumes: list[dict], paths: list[Path]):
    grids = {volume["grid"] for volume in volumes}
    if len(grids) > 1:
        raise ValueError(
            "Volumetric grids differ ("
            + ", ".join(f"{p}: {v['grid']}" for p, v in zip(paths, volumes))
            + "); all parts must come from runs with identical cell and PREC/ENCUT."
        )


def charge_difference(total_path: Path, part_paths: list[Path], output_path: Path) -> dict:
    """Write Δρ = ρ(total) − Σ ρ(parts), e.g. slab+adsorbate − slab − molecule.

    All files must share the same cell and FFT grid (run the fragments in
    the combined cell with frozen positions). Returns the difference volume.
    """
    paths = [Path(total_path)] + [Path(p) for p in part_paths]
    volumes = [read_volumetric(p) for p in paths]
    _check_same_grid(volumes, paths)

    data = list(volumes[0]["data"])
    for part in volumes[1:]:
        part_data = part["data"]
        for i in range(len(data)):
            data[i] -= part_data[i]

    diff = {"poscar_lines": volumes[0]["poscar_lines"], "grid": volumes[0]["grid"], "data": data}
    write_volumetric(diff, output_path)
    return diff


def charge_sum(paths: list[Path], output_path: Path) -> dict:
    """Write the grid sum of several files, e.g. AECCAR0 + AECCAR2 for Bader."""
    paths = [Path(p) for p in paths]
    volumes = [read_volumetric(p) for p in paths]
    _check_same_grid(volumes, paths)

    data = list(volumes[0]["data"])
    for part in volumes[1:]:
        part_data = part["data"]
        for i in range(len(data)):
            data[i] += part_data[i]

    total = {"poscar_lines": volumes[0]["poscar_lines"], "grid": volumes[0]["grid"], "data": data}
    write_volumetric(total, output_path)
    return total


def planar_average(volume: dict, axis: int = 2) -> list[float]:
    """Average the grid over the two perpendicular directions, plane by plane.

    Returns one value per grid plane along `axis` (0=a, 1=b, 2=c). For a
    LOCPOT this is the planar-averaged potential V(z) used for work functions.
    """
    nx, ny, nz = volume["grid"]
    data = volume["data"]
    sizes = (nx, ny, nz)
    n_axis = sizes[axis]
    sums = [0.0] * n_axis

    for iz in range(nz):
        for iy in range(ny):
            base = iy * nx + iz * nx * ny
            for ix in range(nx):
                index = (ix, iy, iz)[axis]
                sums[index] += data[base + ix]

    plane_points = (nx * ny * nz) // n_axis
    return [s / plane_points for s in sums]


def lattice_of(volume: dict) -> list[list[float]]:
    """The scaled lattice (Å) from a volumetric file's POSCAR header."""
    lines = volume["poscar_lines"]
    scale = float(lines[1].split()[0])
    lattice = [[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]
    if scale < 0:
        # Negative scale is the VASP "target cell volume" convention.
        volume_cell = abs(
            lattice[0][0] * (lattice[1][1] * lattice[2][2] - lattice[1][2] * lattice[2][1])
            - lattice[0][1] * (lattice[1][0] * lattice[2][2] - lattice[1][2] * lattice[2][0])
            + lattice[0][2] * (lattice[1][0] * lattice[2][1] - lattice[1][1] * lattice[2][0])
        )
        scale = (abs(scale) / volume_cell) ** (1.0 / 3.0)
    return [[x * scale for x in row] for row in lattice]


def cell_volume_of(volume: dict) -> float:
    """Cell volume in Å³ — divides CHGCAR-family grids (rho·V) into e/Å³."""
    a, b, c = lattice_of(volume)
    return abs(
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def slice_volume(volume: dict, axis: int = 2, fraction: float = 0.5) -> dict:
    """One grid plane perpendicular to `axis` at fractional position `fraction`.

    Returns {"shape": (n1, n2), "data": rows, "axes": (axis1, axis2),
    "position": frac} where data[j][i] runs over the two remaining lattice
    directions in cyclic order (axis 2 → rows along b, columns along a).
    """
    nx, ny, nz = volume["grid"]
    sizes = (nx, ny, nz)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    fraction = float(fraction) % 1.0
    plane = int(round(fraction * sizes[axis])) % sizes[axis]

    axis1, axis2 = (axis + 1) % 3, (axis + 2) % 3
    n1, n2 = sizes[axis1], sizes[axis2]
    data = volume["data"]

    def grid_index(ix, iy, iz):
        return ix + iy * nx + iz * nx * ny

    rows = []
    for j in range(n2):
        row = []
        for i in range(n1):
            indices = [0, 0, 0]
            indices[axis] = plane
            indices[axis1] = i
            indices[axis2] = j
            row.append(data[grid_index(*indices)])
        rows.append(row)
    return {
        "shape": (n1, n2),
        "data": rows,
        "axes": (axis1, axis2),
        "position": plane / sizes[axis],
    }


# --- Bader charge analysis (Henkelman group `bader` binary) -----------------

def zval_from_potcar(potcar_path: Path) -> list[float]:
    """Valence electron count per POTCAR species, from the ZVAL tags."""
    zvals = []
    with open(potcar_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "ZVAL" in line:
                after = line.split("ZVAL")[1]
                value = after.split("=")[1].split()[0].rstrip(";")
                zvals.append(float(value))
    if not zvals:
        raise ValueError(f"No ZVAL tags found in {potcar_path}")
    return zvals


def parse_acf(acf_path: Path) -> list[dict]:
    """Parse the bader ACF.dat table into per-atom electron counts."""
    rows = []
    for line in Path(acf_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit():
            rows.append(
                {
                    "index": int(parts[0]),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3]),
                    "electrons": float(parts[4]),
                }
            )
    if not rows:
        raise ValueError(f"No atom rows found in {acf_path}")
    return rows


def run_bader(job_dir: Path, bader_executable: str = "bader") -> dict:
    """Run Bader analysis in a job directory and return per-atom net charges.

    Needs CHGCAR (from LCHARG) and, for all-electron accuracy, AECCAR0 +
    AECCAR2 (from LAECHG, the `charge` calc type writes both). The summed
    reference is written as CHGCAR_sum. Requires the Henkelman `bader`
    binary on PATH (https://theory.cm.utexas.edu/henkelman/code/bader/).
    """
    job_dir = Path(job_dir)
    if (job_dir / ".engine").exists() and (job_dir / ".engine").read_text().strip() == "qe":
        return _run_qe_bader(job_dir, bader_executable)
    chgcar = job_dir / "CHGCAR"
    if not chgcar.exists():
        raise FileNotFoundError(f"Bader analysis needs a CHGCAR in {job_dir}")
    if shutil.which(bader_executable) is None:
        raise FileNotFoundError(
            f"'{bader_executable}' not found on PATH. Install the Henkelman group "
            "bader binary: https://theory.cm.utexas.edu/henkelman/code/bader/"
        )

    command = [bader_executable, "CHGCAR"]
    aeccar0 = job_dir / "AECCAR0"
    aeccar2 = job_dir / "AECCAR2"
    used_reference = False
    if aeccar0.exists() and aeccar2.exists():
        charge_sum([aeccar0, aeccar2], job_dir / "CHGCAR_sum")
        command += ["-ref", "CHGCAR_sum"]
        used_reference = True

    completed = subprocess.run(
        command, cwd=job_dir, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"bader failed in {job_dir}:\n{completed.stdout}\n{completed.stderr}")

    return {
        "acf_path": job_dir / "ACF.dat",
        "used_aeccar_reference": used_reference,
        "reference": "AECCAR0+AECCAR2" if used_reference else "CHGCAR",
        "charges": bader_net_charges(job_dir),
    }


def _upf_zval(path: Path) -> float:
    """Read the valence-electron count from XML or legacy UPF headers."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    patterns = (
        r"\bz_valence\s*=\s*['\"]?([0-9.+\-Ee]+)",
        r"\bZ[ _-]*valence\s*[:=]?\s*([0-9.+\-Ee]+)",
        r"^\s*([0-9.+\-Ee]+)\s+Z\s+valence",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return float(match.group(1))
    raise ValueError(f"No z_valence found in QE pseudopotential {path}")


def _qe_species_pseudos(job_dir: Path) -> dict[str, Path]:
    """Return element-to-UPF paths from the pw.x ATOMIC_SPECIES card."""
    lines = (job_dir / "pw.in").read_text(encoding="utf-8", errors="ignore").splitlines()
    mapping = {}
    in_species = False
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("ATOMIC_SPECIES"):
            in_species = True
            continue
        if not in_species:
            continue
        if not stripped or stripped.startswith(("ATOMIC_POSITIONS", "K_POINTS", "CELL_PARAMETERS", "CONSTRAINTS")):
            break
        parts = stripped.split()
        if len(parts) >= 3:
            pseudo = Path(parts[2])
            candidates = [job_dir / "pseudo" / pseudo.name, job_dir / pseudo]
            found = next((path for path in candidates if path.exists()), None)
            if found is None:
                raise FileNotFoundError(f"QE pseudopotential for {parts[0]} not found in {job_dir}")
            mapping[parts[0]] = found
    if not mapping:
        raise ValueError(f"No ATOMIC_SPECIES card found in {job_dir / 'pw.in'}")
    return mapping


def _run_qe_bader(job_dir: Path, bader_executable: str) -> dict:
    """Run Bader on the QE pp.x valence-density cube."""
    cube = job_dir / "charge-density.cube"
    if not cube.exists():
        raise FileNotFoundError(f"QE Bader analysis needs charge-density.cube in {job_dir}")
    if shutil.which(bader_executable) is None:
        raise FileNotFoundError(
            f"'{bader_executable}' not found on PATH. Install the Henkelman group bader binary."
        )
    completed = subprocess.run(
        [bader_executable, cube.name], cwd=job_dir, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"bader failed in {job_dir}:\n{completed.stdout}\n{completed.stderr}")

    from vasp_auto.structure import per_atom_symbols, read_poscar

    symbols = per_atom_symbols(read_poscar(job_dir / "POSCAR"))
    rows = parse_acf(job_dir / "ACF.dat")
    if len(rows) != len(symbols):
        raise ValueError(f"ACF.dat has {len(rows)} atoms but POSCAR has {len(symbols)} in {job_dir}")
    pseudo_by_element = _qe_species_pseudos(job_dir)
    zvals = {element: _upf_zval(path) for element, path in pseudo_by_element.items()}
    charges = [
        {**row, "element": element, "net_charge": zvals[element] - row["electrons"]}
        for row, element in zip(rows, symbols)
    ]
    return {
        "acf_path": job_dir / "ACF.dat",
        "used_aeccar_reference": False,
        "reference": "QE valence charge-density.cube",
        "charges": charges,
    }


def bader_net_charges(job_dir: Path) -> list[dict]:
    """Net atomic charges (ZVAL − Bader electrons) from an existing ACF.dat."""
    job_dir = Path(job_dir)
    acf_rows = parse_acf(job_dir / "ACF.dat")

    poscar = job_dir / "POSCAR"
    potcar = job_dir / "POTCAR"
    from vasp_auto.structure import per_atom_symbols, read_poscar

    struct = read_poscar(poscar)
    symbols = per_atom_symbols(struct)
    zvals_per_species = zval_from_potcar(potcar)
    if len(zvals_per_species) != len(struct["elements"]):
        raise ValueError(
            f"POTCAR has {len(zvals_per_species)} species but POSCAR lists "
            f"{len(struct['elements'])} in {job_dir}"
        )
    zval_by_atom = []
    for zval, count in zip(zvals_per_species, struct["counts"]):
        zval_by_atom.extend([zval] * count)

    if len(acf_rows) != len(symbols):
        raise ValueError(
            f"ACF.dat has {len(acf_rows)} atoms but POSCAR has {len(symbols)} in {job_dir}"
        )

    charges = []
    for row, symbol, zval in zip(acf_rows, symbols, zval_by_atom):
        charges.append(
            {
                "index": row["index"],
                "element": symbol,
                "electrons": row["electrons"],
                "net_charge": zval - row["electrons"],
            }
        )
    return charges
