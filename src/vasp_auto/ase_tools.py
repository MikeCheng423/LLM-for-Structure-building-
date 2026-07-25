from __future__ import annotations

from pathlib import Path


def require_ase():
    try:
        import ase  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ASE is not available in this Python environment. "
            "Install it with: pip install ase"
        ) from exc


def import_structure_to_case(
    structure_path: str | Path,
    case_dir: str | Path,
    input_format: str | None = None,
    index: str | int | None = None,
) -> Path:
    require_ase()
    from ase.io import read, write

    structure_path = Path(structure_path).expanduser().resolve()
    case_dir = Path(case_dir).expanduser().resolve()

    if not structure_path.exists():
        raise FileNotFoundError(f"ASE input structure not found: {structure_path}")

    read_index = index if index is not None else -1
    if isinstance(read_index, str) and read_index.lstrip("-").isdigit():
        read_index = int(read_index)
    atoms = read(str(structure_path), index=read_index, format=input_format)
    if isinstance(atoms, list):
        if len(atoms) != 1:
            raise ValueError(
                "ASE import selected multiple frames. Use --ase-index to select one frame."
            )
        atoms = atoms[0]

    case_dir.mkdir(parents=True, exist_ok=True)
    poscar_path = case_dir / "POSCAR"
    write(str(poscar_path), atoms, format="vasp", vasp5=True, direct=True, sort=True)
    return poscar_path


def build_bulk_case(
    symbol: str,
    case_dir: str | Path,
    crystalstructure: str | None = None,
    a: float | None = None,
    c: float | None = None,
    cubic: bool = False,
) -> Path:
    require_ase()
    from ase.build import bulk
    from ase.io import write

    case_dir = Path(case_dir).expanduser().resolve()
    kwargs = {"cubic": cubic}
    if crystalstructure:
        kwargs["crystalstructure"] = crystalstructure
    if a is not None:
        kwargs["a"] = a
    if c is not None:
        kwargs["c"] = c

    atoms = bulk(symbol, **kwargs)
    case_dir.mkdir(parents=True, exist_ok=True)
    poscar_path = case_dir / "POSCAR"
    write(str(poscar_path), atoms, format="vasp", vasp5=True, direct=True, sort=True)
    return poscar_path


def build_slab_case(
    source: str,
    case_dir: str | Path,
    miller: tuple[int, int, int] = (1, 1, 1),
    layers: int = 4,
    vacuum: float = 12.0,
    crystalstructure: str | None = None,
    a: float | None = None,
    repeat: tuple[int, int] | None = None,
) -> Path:
    """Build a surface slab from an element symbol or a structure file."""
    require_ase()
    from ase.build import bulk, surface
    from ase.io import read, write

    source_path = Path(source).expanduser()
    if source_path.exists():
        base = read(str(source_path))
    else:
        kwargs = {}
        if crystalstructure:
            kwargs["crystalstructure"] = crystalstructure
        if a is not None:
            kwargs["a"] = a
        base = bulk(source, **kwargs)

    slab = surface(base, tuple(int(i) for i in miller), int(layers))
    if repeat:
        slab = slab.repeat((int(repeat[0]), int(repeat[1]), 1))
    slab.center(vacuum=float(vacuum), axis=2)

    case_dir = Path(case_dir).expanduser().resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    poscar_path = case_dir / "POSCAR"
    write(str(poscar_path), slab, format="vasp", vasp5=True, direct=True, sort=True)
    return poscar_path


def molecule_positions(name: str) -> tuple[list[str], list[list[float]]]:
    """Cartesian geometry (Å) of an adsorbate. ``name`` is an ASE molecule
    (CO2, H2O, NH3, CH4 …) when known, otherwise a single atom of that element.
    Positions are shifted so the lowest atom sits at z=0, so placing them above
    a surface atom keeps the whole molecule above the slab."""
    require_ase()
    from ase.build import molecule

    try:
        atoms = molecule(name)
    except (KeyError, NotImplementedError, ValueError):
        from ase import Atoms
        atoms = Atoms(name, positions=[[0.0, 0.0, 0.0]])
    pos = atoms.get_positions()
    zmin = min(p[2] for p in pos)
    return (
        list(atoms.get_chemical_symbols()),
        [[float(p[0]), float(p[1]), float(p[2] - zmin)] for p in pos],
    )


def build_molecule_case(name: str, case_dir: str | Path, box: float = 12.0) -> Path:
    """Build an isolated molecule centred in a cubic box."""
    require_ase()
    from ase.build import molecule
    from ase.io import write

    atoms = molecule(name)
    atoms.cell = [float(box)] * 3
    atoms.pbc = True
    atoms.center()

    case_dir = Path(case_dir).expanduser().resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    poscar_path = case_dir / "POSCAR"
    write(str(poscar_path), atoms, format="vasp", vasp5=True, direct=True, sort=True)
    return poscar_path


def build_crystal_case(
    symbols: list[str] | str,
    basis: list[tuple[float, float, float]],
    spacegroup: int,
    case_dir: str | Path,
    a: float,
    b: float | None = None,
    c: float | None = None,
    alpha: float = 90.0,
    beta: float = 90.0,
    gamma: float = 90.0,
) -> Path:
    """Build a crystal from a space group + Wyckoff basis (ASE).

    ``symbols`` is one element per basis site (e.g. ["Na", "Cl"]); ``basis`` is
    the matching list of fractional coordinates of the representative sites.
    ``spacegroup`` is the international number (1–230). Lattice parameters
    default to a cubic cell (b = c = a, all angles 90°) when omitted.
    """
    require_ase()
    from ase.spacegroup import crystal
    from ase.io import write

    if isinstance(symbols, str):
        symbols = symbols.replace(",", " ").split()
    basis = [tuple(float(x) for x in site) for site in basis]
    if len(symbols) != len(basis):
        raise ValueError(
            f"Give one basis coordinate per symbol: {len(symbols)} symbols "
            f"but {len(basis)} basis sites."
        )
    cellpar = [float(a), float(b) if b else float(a), float(c) if c else float(a),
               float(alpha), float(beta), float(gamma)]
    atoms = crystal(symbols=symbols, basis=basis, spacegroup=int(spacegroup), cellpar=cellpar)

    case_dir = Path(case_dir).expanduser().resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    poscar_path = case_dir / "POSCAR"
    write(str(poscar_path), atoms, format="vasp", vasp5=True, direct=True, sort=True)
    return poscar_path


def build_nanotube_case(
    symbol: str,
    n: int,
    m: int,
    case_dir: str | Path,
    length: int = 1,
    bond: float | None = None,
    vacuum: float = 10.0,
) -> Path:
    """Build an (n, m) single-wall nanotube periodic along c, vacuum in a/b."""
    require_ase()
    from ase.build import nanotube
    from ase.io import write

    kwargs = {"symbol": symbol}
    if bond is not None:
        kwargs["bond"] = float(bond)
    atoms = nanotube(int(n), int(m), length=int(length), **kwargs)
    atoms.center(vacuum=float(vacuum), axis=(0, 1))
    atoms.pbc = True

    case_dir = Path(case_dir).expanduser().resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    poscar_path = case_dir / "POSCAR"
    write(str(poscar_path), atoms, format="vasp", vasp5=True, direct=True, sort=True)
    return poscar_path


def interpolate_neb_with_ase(
    initial_poscar: str | Path,
    final_poscar: str | Path,
    job_dir: str | Path,
    neb_images: int,
    method: str = "idpp",
    mic: bool = True,
) -> None:
    require_ase()
    from ase.io import read, write

    try:
        from ase.mep import NEB
    except ImportError:
        from ase.neb import NEB

    initial_poscar = Path(initial_poscar).expanduser().resolve()
    final_poscar = Path(final_poscar).expanduser().resolve()
    job_dir = Path(job_dir).expanduser().resolve()

    initial = read(str(initial_poscar), format="vasp")
    final = read(str(final_poscar), format="vasp")

    images = [initial]
    images.extend(initial.copy() for _ in range(int(neb_images)))
    images.append(final)

    try:
        neb = NEB(images, method="improvedtangent")
    except TypeError:
        neb = NEB(images)
    neb.interpolate(method=method, mic=mic)

    for index, atoms in enumerate(images):
        image_dir = job_dir / f"{index:02d}"
        image_dir.mkdir(parents=True, exist_ok=True)
        write(str(image_dir / "POSCAR"), atoms, format="vasp", vasp5=True, direct=True, sort=True)
