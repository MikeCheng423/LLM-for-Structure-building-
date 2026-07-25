"""Gaussian cube volumetric data produced by Quantum ESPRESSO pp.x."""
from __future__ import annotations

from pathlib import Path


def read_cube(path: Path) -> dict:
    """Read one scalar Gaussian cube while preserving its complete header."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        raise ValueError(f"Cube file is too short: {path}")
    try:
        natoms = abs(int(lines[2].split()[0]))
        grid = tuple(abs(int(lines[index].split()[0])) for index in (3, 4, 5))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid cube header in {path}") from exc
    header_end = 6 + natoms
    if len(lines) < header_end:
        raise ValueError(f"Cube atom table is truncated: {path}")
    npoints = grid[0] * grid[1] * grid[2]
    try:
        data = [float(value) for line in lines[header_end:] for value in line.split()]
    except ValueError as exc:
        raise ValueError(f"Invalid cube grid value in {path}") from exc
    if len(data) < npoints:
        raise ValueError(f"Cube grid is truncated in {path}: {len(data)} < {npoints}")
    return {"header_lines": lines[:header_end], "grid": grid, "data": data[:npoints]}


def write_cube(volume: dict, path: Path) -> None:
    """Write a cube volume using the source header and six values per row."""
    lines = list(volume["header_lines"])
    data = volume["data"]
    for start in range(0, len(data), 6):
        lines.append(" ".join(f"{value: .10E}" for value in data[start:start + 6]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cube_difference(total_path: Path, part_paths: list[Path], output_path: Path) -> dict:
    """Write total minus component QE cube densities on an identical grid."""
    paths = [Path(total_path), *(Path(path) for path in part_paths)]
    volumes = [read_cube(path) for path in paths]
    grids = {volume["grid"] for volume in volumes}
    if len(grids) != 1:
        detail = ", ".join(f"{path}: {volume['grid']}" for path, volume in zip(paths, volumes))
        raise ValueError(f"Cube grids differ ({detail}); use identical cells and FFT grids.")
    data = list(volumes[0]["data"])
    for part in volumes[1:]:
        for index, value in enumerate(part["data"]):
            data[index] -= value
    result = {**volumes[0], "data": data}
    write_cube(result, output_path)
    return result


def cube_as_volumetric(volume: dict, poscar_path: Path) -> dict:
    """Convert cube z-fastest ordering to the x-fastest layout used by the UI."""
    nx, ny, nz = volume["grid"]
    source = volume["data"]
    data = [0.0] * len(source)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                cube_index = iz + nz * (iy + ny * ix)
                volume_index = ix + nx * (iy + ny * iz)
                data[volume_index] = source[cube_index]
    return {
        "poscar_lines": Path(poscar_path).read_text(encoding="utf-8").splitlines(),
        "grid": volume["grid"],
        "data": data,
    }
