"""Quantum ESPRESSO (pw.x) engine: POSCAR -> pw.in, pseudopotential lookup.

This is the open-source counterpart to the VASP input path (incar.py +
potcar_finder.py + job_manager.py). It reuses the same POSCAR-based structure
logic (structure.read_poscar) and k-point machinery (kpoints.py) so the
builders, job layout and UI behave identically — only the emitted input file
and the run/parse steps differ.

Pure Python, no numpy, no third-party imports beyond the engine's own modules.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from vasp_auto.kpoints import (
    mesh_from_spacing,
    parse_kpath,
    parse_mesh,
)
from vasp_auto.structure import per_atom_symbols, read_poscar

# 1 Rydberg in electron-volts (CODATA). VASP reports eV; QE reports Ry — every
# energy crossing the boundary goes through this constant.
RY_TO_EV = 13.605693122994

# Our calc_type -> QE &CONTROL `calculation` value.
QE_CALC_MAP: dict[str, str] = {
    "scf": "scf",
    "relax": "relax",
    "vcrelax": "vc-relax",
    "dos": "nscf",
    "bands": "bands",
    "charge": "scf",
    "md": "md",
    "phonon": "scf",
    "hse06": "scf",
    "freq": "scf",
    "optics": "nscf",
    "workfunction": "scf",
}

# Calculation types implemented by the pw.x and companion-program stage runner.
# NEB is assembled separately because its endpoint case layout is different.
QE_SUPPORTED_CALC_TYPES = tuple(QE_CALC_MAP)
QE_ALL_CALC_TYPES = (*QE_SUPPORTED_CALC_TYPES, "neb")
QE_COMPANION_PROGRAMS = (
    "average.x", "bands.x", "dos.x", "dynmat.x", "epsilon.x", "neb.x", "ph.x", "pp.x",
    "projwfc.x",
)

# Standard atomic weights for ATOMIC_SPECIES. QE only uses the mass for
# dynamics, but a correct value avoids surprises. Covers the common elements;
# unknown species fall back to 1.0 with a note in the input comment.
ATOMIC_MASSES: dict[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.91,
    "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
    "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Hf": 178.49, "Ta": 180.95,
    "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08,
    "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98,
}

# Default QE plane-wave / SCF parameters when neither a config override nor a
# user-supplied pw.in sets them. ecutwfc/ecutrho are in Ry; degauss in Ry.
QE_DEFAULTS: dict[str, float | str] = {
    "ecutwfc": 50.0,
    "ecutrho": 400.0,
    "conv_thr": 1.0e-6,
    "degauss": 0.01,
    "smearing": "gaussian",
    "occupations": "smearing",
}


# ------------------------------------------------------------- pseudopotentials

def find_pseudo(element: str, pseudo_dir: str | Path | None, pseudo_map: dict | None = None) -> str:
    """Return the UPF filename for an element.

    ``pseudo_map`` (config.yaml) picks an explicit file per element, e.g.
    ``{"Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"}``. Otherwise the first file in
    ``pseudo_dir`` whose name starts with ``<El>.`` or ``<El>_``
    (case-insensitive) is used. Raises FileNotFoundError listing what was
    searched, mirroring potcar_finder.find_potcar_root.
    """
    pseudo_map = pseudo_map or {}
    if element in pseudo_map:
        return str(pseudo_map[element])

    if not pseudo_dir:
        raise FileNotFoundError(
            f"No pseudopotential directory configured for '{element}'. "
            "Set pseudo_dir in config.yaml (or --pseudo-dir), or add the element "
            "to pseudo_map."
        )

    directory = Path(pseudo_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"pseudo_dir does not exist: {directory}")

    el = element.lower()
    candidates = sorted(
        p.name for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".upf"
        and (p.name.lower().startswith(el + ".") or p.name.lower().startswith(el + "_"))
    )
    if candidates:
        return candidates[0]

    available = ", ".join(sorted(p.name for p in directory.glob("*.UPF"))) or "none"
    raise FileNotFoundError(
        f"No UPF pseudopotential for '{element}' in {directory}.\n"
        f"Expected a file named like {element}.*.UPF, or map it in pseudo_map.\n"
        f"Available: {available}"
    )


def pseudo_files_for_struct(struct: dict, pseudo_dir, pseudo_map=None) -> dict[str, str]:
    """{element: UPF filename} for every distinct species in the structure."""
    return {el: find_pseudo(el, pseudo_dir, pseudo_map) for el in struct["elements"]}


# ----------------------------------------------------------------- k-points

def _kpoints_card(spec: dict | None, struct: dict, poscar_path: Path | None, calc_type: str) -> str:
    """Build the K_POINTS card from the same kpoints spec the VASP path uses."""
    qe_calc = QE_CALC_MAP.get(calc_type, "scf")

    # Band structures walk a high-symmetry path (crystal_b).
    if qe_calc == "bands":
        kpath = (spec or {}).get("kpath") if spec else None
        if not kpath:
            raise ValueError(
                "QE bands calculation needs a --kpath (preset like fcc/hex or a "
                "'G 0 0 0; X 0.5 0 0.5' list)."
            )
        points = kpath if isinstance(kpath, list) else parse_kpath(kpath)
        divisions = int((spec or {}).get("divisions") or 20)
        lines = ["K_POINTS crystal_b", str(len(points))]
        for _, (kx, ky, kz) in points:
            lines.append(f"  {kx:.8f} {ky:.8f} {kz:.8f} {divisions}")
        return "\n".join(lines) + "\n"

    mesh = None
    if spec:
        if spec.get("mesh"):
            mesh = parse_mesh(spec["mesh"])
        elif spec.get("spacing"):
            if poscar_path is None:
                raise ValueError("density-based k-points need a POSCAR to read the lattice")
            mesh = mesh_from_spacing(poscar_path, float(spec["spacing"]))

    if mesh is None:
        # Mirror the VASP default (a single Gamma point) when nothing is asked.
        return "K_POINTS gamma\n"

    return f"K_POINTS automatic\n  {mesh[0]} {mesh[1]} {mesh[2]} 0 0 0\n"


# ----------------------------------------------------------------- input file

def _fmt(value) -> str:
    """Format a Python value as a Fortran namelist literal."""
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, str):
        return f"'{value}'"
    return str(value)


def _scaled_lattice(struct: dict) -> list[list[float]]:
    scale = struct["scale"]
    return [[x * scale for x in row] for row in struct["lattice"]]


def build_pw_input(
    struct: dict,
    calc_type: str,
    *,
    pseudos: dict[str, str],
    kpoints_card: str,
    prefix: str = "vasp_auto",
    qe_params: dict | None = None,
    spin: bool = False,
    magmom_map: dict | None = None,
) -> str:
    """Assemble a pw.x input file (pw.in) from a POSCAR structure dict.

    ``pseudos`` maps each element to its UPF filename. ``kpoints_card`` is the
    full K_POINTS block. ``qe_params`` overrides QE_DEFAULTS (ecutwfc/ecutrho in
    Ry, conv_thr, degauss, smearing, occupations).
    """
    calc_type = calc_type if calc_type in QE_CALC_MAP else "scf"
    qe_calc = QE_CALC_MAP[calc_type]
    params = {**QE_DEFAULTS, **(qe_params or {})}

    elements = struct["elements"]
    nat = sum(struct["counts"])
    ntyp = len(elements)

    control = {
        "calculation": qe_calc,
        "prefix": prefix,
        "outdir": "./tmp",
        "pseudo_dir": "./pseudo",
        "tprnfor": True,
        "verbosity": "high",
    }
    if qe_calc in ("relax", "vc-relax", "md"):
        control["tstress"] = True

    system = {
        "ibrav": 0,
        "nat": nat,
        "ntyp": ntyp,
        "ecutwfc": float(params["ecutwfc"]),
        "ecutrho": float(params["ecutrho"]),
        "occupations": str(params["occupations"]),
        "smearing": str(params["smearing"]),
        "degauss": float(params["degauss"]),
    }

    electrons = {"conv_thr": float(params["conv_thr"]), "mixing_beta": 0.7}

    if calc_type == "md":
        control["nstep"] = int(params.get("md_steps", 100))
        control["dt"] = float(params.get("md_dt", 20.0))
    elif calc_type == "hse06":
        system.update({
            "input_dft": "HSE",
            "exx_fraction": float(params.get("exx_fraction", 0.25)),
            "screening_parameter": float(params.get("screening_parameter", 0.106)),
        })
    elif calc_type in ("dos", "optics"):
        # NSCF post-processing needs enough empty states. Users can set qe_nbnd
        # explicitly; leaving it absent lets QE choose its normal default.
        if params.get("nbnd") is not None:
            system["nbnd"] = int(params["nbnd"])

    # Spin polarisation: nspin=2 plus a starting moment per *species* (QE indexes
    # starting_magnetization by the ATOMIC_SPECIES order).
    species_magmom: list[float] = []
    if spin:
        system["nspin"] = 2
        magmom_map = magmom_map or {}
        for el in elements:
            species_magmom.append(float(magmom_map.get(el, 0.1)))

    sections = []

    def emit(name: str, mapping: dict):
        body = "\n".join(f"  {k} = {_fmt(v)}" for k, v in mapping.items())
        sections.append(f"&{name}\n{body}\n/")

    emit("CONTROL", control)

    # starting_magnetization(i) entries live inside &SYSTEM.
    system_lines = "\n".join(f"  {k} = {_fmt(v)}" for k, v in system.items())
    if species_magmom:
        mag_lines = "\n".join(
            f"  starting_magnetization({i + 1}) = {_fmt(m)}"
            for i, m in enumerate(species_magmom)
        )
        system_lines = system_lines + "\n" + mag_lines
    sections.append(f"&SYSTEM\n{system_lines}\n/")

    emit("ELECTRONS", electrons)
    if qe_calc in ("relax", "vc-relax", "md"):
        if qe_calc == "md":
            emit("IONS", {
                "ion_dynamics": str(params.get("md_dynamics", "verlet")),
                "ion_temperature": str(params.get("md_thermostat", "rescaling")),
                "tempw": float(params.get("md_temperature", 300.0)),
            })
        else:
            emit("IONS", {"ion_dynamics": "bfgs"})
    if qe_calc == "vc-relax":
        emit("CELL", {"cell_dynamics": "bfgs"})

    # ATOMIC_SPECIES
    species_lines = ["ATOMIC_SPECIES"]
    for el in elements:
        mass = ATOMIC_MASSES.get(el, 1.0)
        species_lines.append(f"  {el} {mass:.4f} {pseudos[el]}")

    # CELL_PARAMETERS (angstrom) — ibrav=0 reads the cell explicitly.
    lattice = _scaled_lattice(struct)
    cell_lines = ["CELL_PARAMETERS angstrom"]
    for row in lattice:
        cell_lines.append(f"  {row[0]: .10f} {row[1]: .10f} {row[2]: .10f}")

    # ATOMIC_POSITIONS — fractional (crystal) for Direct POSCARs, else angstrom.
    symbols = per_atom_symbols(struct)
    if struct["cartesian"]:
        pos_header = "ATOMIC_POSITIONS angstrom"
        scale = struct["scale"]
        coords = [[c * scale for c in row] for row in struct["coords"]]
    else:
        pos_header = "ATOMIC_POSITIONS crystal"
        coords = struct["coords"]
    pos_lines = [pos_header]
    for sym, (x, y, z), flag in zip(symbols, coords, struct["flags"]):
        line = f"  {sym} {x: .10f} {y: .10f} {z: .10f}"
        # Selective-dynamics flags -> per-atom 0/1 force multipliers in QE.
        if flag:
            mults = " ".join("1" if f.upper() == "T" else "0" for f in flag[:3])
            line += f"  {mults}"
        pos_lines.append(line)

    blocks = [
        "\n".join(sections),
        "\n".join(species_lines),
        "\n".join(cell_lines),
        "\n".join(pos_lines),
        kpoints_card.rstrip("\n"),
    ]
    return "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------- job assembly

def _qe_params_from_config(config: dict | None) -> dict:
    """Pull qe_* overrides out of a config dict into a QE params dict."""
    config = config or {}
    params: dict = {}
    for key in (
        "ecutwfc", "ecutrho", "conv_thr", "degauss", "smearing", "occupations",
        "nbnd", "md_steps", "md_dt", "md_dynamics", "md_thermostat",
        "md_temperature", "exx_fraction", "screening_parameter",
    ):
        value = config.get(f"qe_{key}")
        if value is not None:
            params[key] = value
    return params


def _auxiliary_inputs(calc_type: str, config: dict, struct: dict) -> tuple[dict[str, str], list[dict]]:
    """Return companion-program input files and stage records for a calc type."""
    files: dict[str, str] = {}
    stages: list[dict] = []
    prefix = str(config.get("qe_prefix", "vasp_auto"))
    outdir = "./tmp"

    if calc_type == "dos":
        files["dos.in"] = (
            "&DOS\n"
            f"  prefix = '{prefix}'\n  outdir = '{outdir}'\n"
            "  fildos = 'dos.dat'\n"
            f"  DeltaE = {float(config.get('qe_dos_deltae', 0.01)):.10g}\n/\n"
        )
        stages.append({"program": "dos.x", "input": "dos.in", "output": "dos.out", "mpi": True})
        files["projwfc.in"] = (
            "&PROJWFC\n"
            f"  prefix = '{prefix}'\n  outdir = '{outdir}'\n"
            "  filpdos = 'pdos'\n"
            f"  DeltaE = {float(config.get('qe_dos_deltae', 0.01)):.10g}\n/\n"
        )
        stages.append({
            "program": "projwfc.x", "input": "projwfc.in", "output": "projwfc.out", "mpi": True,
        })
    elif calc_type == "bands":
        files["bands.in"] = (
            "&BANDS\n"
            f"  prefix = '{prefix}'\n  outdir = '{outdir}'\n"
            "  filband = 'bands.dat'\n/\n"
        )
        stages.append({"program": "bands.x", "input": "bands.in", "output": "bands.out", "mpi": True})
    elif calc_type in ("phonon", "freq"):
        qpoint = str(config.get("qe_ph_qpoint", "0.0 0.0 0.0"))
        files["ph.in"] = (
            "&INPUTPH\n"
            f"  prefix = '{prefix}'\n  outdir = '{outdir}'\n"
            f"  tr2_ph = {float(config.get('qe_tr2_ph', 1e-14)):.10g}\n"
            "  fildyn = 'dynamical-matrix'\n/\n"
            f"{qpoint}\n"
        )
        stages.append({"program": "ph.x", "input": "ph.in", "output": "ph.out", "mpi": True})
        if calc_type == "freq":
            files["dynmat.in"] = "&INPUT\n  fildyn = 'dynamical-matrix'\n  asr = 'zero-dim'\n/\n"
            stages.append({
                "program": "dynmat.x", "input": "dynmat.in", "output": "dynmat.out", "mpi": False,
            })
    elif calc_type == "optics":
        files["epsilon.in"] = (
            "&INPUTPP\n"
            f"  prefix = '{prefix}'\n  outdir = '{outdir}'\n  calculation = 'eps'\n/\n"
            "&ENERGY_GRID\n"
            f"  wmin = {float(config.get('qe_optics_wmin', 0.0)):.10g}\n"
            f"  wmax = {float(config.get('qe_optics_wmax', 30.0)):.10g}\n"
            f"  nw = {int(config.get('qe_optics_nw', 600))}\n/\n"
        )
        stages.append({
            "program": "epsilon.x", "input": "epsilon.in", "output": "epsilon.out", "mpi": True,
        })
    elif calc_type in ("charge", "workfunction"):
        plot_num = 0 if calc_type == "charge" else 11
        filplot = "charge-density.dat" if calc_type == "charge" else "electrostatic-potential.dat"
        files["pp.in"] = (
            "&INPUTPP\n"
            f"  prefix = '{prefix}'\n  outdir = '{outdir}'\n"
            f"  plot_num = {plot_num}\n  filplot = '{filplot}'\n/\n"
            "&PLOT\n  iflag = 3\n  output_format = 6\n"
            f"  fileout = '{'charge-density.cube' if calc_type == 'charge' else 'electrostatic-potential.cube'}'\n/\n"
        )
        stages.append({"program": "pp.x", "input": "pp.in", "output": "pp.out", "mpi": True})
        if calc_type == "workfunction":
            c_length = sum(x * x for x in _scaled_lattice(struct)[2]) ** 0.5
            files["average.in"] = (
                "1\nelectrostatic-potential.dat\n1.0\n1440\n3\n"
                f"{c_length:.10f}\n"
            )
            stages.append({
                "program": "average.x", "input": "average.in", "output": "average.out",
                "mpi": False, "rename": {"avg.dat": "potential-average.dat"},
            })
    return files, stages


def qe_stage_plan(calc_type: str, config: dict, struct: dict) -> tuple[dict[str, str], list[dict]]:
    """Build input files and an ordered execution plan for a standalone QE job."""
    auxiliary, post = _auxiliary_inputs(calc_type, config, struct)
    stages: list[dict] = []
    if calc_type in ("dos", "bands", "optics"):
        stages.append({"program": "pw.x", "input": "scf.in", "output": "scf.out", "mpi": True})
    stages.append({"program": "pw.x", "input": "pw.in", "output": "pw.out", "mpi": True})
    stages.extend(post)
    return auxiliary, stages


def _positions_block(pw_text: str, nat: int) -> tuple[list[str], int, int]:
    lines = pw_text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().upper().startswith("ATOMIC_POSITIONS"))
    return lines[start:start + nat + 1], start, start + nat + 1


def build_neb_input(
    initial: dict,
    final: dict,
    *,
    pseudos: dict[str, str],
    kpoints_card: str,
    images: int,
    qe_params: dict | None = None,
    spin: bool = False,
    magmom_map: dict | None = None,
    prefix: str = "vasp_auto",
) -> str:
    """Build a neb.x input from two POSCAR structures."""
    if initial["elements"] != final["elements"] or initial["counts"] != final["counts"]:
        raise ValueError("QE NEB endpoints must have identical species and atom counts")
    nat = sum(initial["counts"])
    engine = build_pw_input(
        initial, "scf", pseudos=pseudos, kpoints_card=kpoints_card,
        prefix=prefix, qe_params=qe_params, spin=spin, magmom_map=magmom_map,
    )
    first, start, end = _positions_block(engine, nat)
    final_text = build_pw_input(
        final, "scf", pseudos=pseudos, kpoints_card=kpoints_card,
        prefix=prefix, qe_params=qe_params, spin=spin, magmom_map=magmom_map,
    )
    last, _, _ = _positions_block(final_text, nat)

    engine_lines = engine.splitlines()
    engine_lines = [line for line in engine_lines if "calculation =" not in line]
    # Removing the calculation line changes the position index by one.
    _, start, end = _positions_block("\n".join(engine_lines), nat)
    positions = [
        "BEGIN_POSITIONS", "FIRST_IMAGE", *first,
        "LAST_IMAGE", *last, "END_POSITIONS",
    ]
    engine_lines[start:end] = positions
    species_index = next(i for i, line in enumerate(engine_lines) if line.strip() == "ATOMIC_SPECIES")
    engine_lines[species_index:species_index] = ["&IONS", "/", ""]

    path = [
        "BEGIN", "BEGIN_PATH_INPUT", "&PATH",
        "  restart_mode = 'from_scratch'",
        "  string_method = 'neb'",
        f"  nstep_path = {int((qe_params or {}).get('neb_steps', 100))}",
        f"  num_of_images = {int(images) + 2}",
        "  opt_scheme = 'broyden'",
        "  CI_scheme = 'auto'",
        f"  path_thr = {float((qe_params or {}).get('neb_path_thr', 0.05)):.10g}",
        "/", "END_PATH_INPUT", "BEGIN_ENGINE_INPUT",
        *engine_lines, "END_ENGINE_INPUT", "END", "",
    ]
    return "\n".join(path)


def create_qe_neb_job(
    case_info: dict,
    config: dict,
    *,
    neb_images: int = 5,
    kpoints_spec: dict | None = None,
    spin: bool = False,
    magmom_map: dict | None = None,
) -> Path:
    """Prepare a QE neb.x job from initial/POSCAR and final/POSCAR."""
    case_dir = Path(case_info["case_dir"]).resolve()
    job_dir = Path(case_info["job_dir"]).resolve()
    initial_path = case_dir / "initial" / "POSCAR"
    final_path = case_dir / "final" / "POSCAR"
    if not initial_path.exists() or not final_path.exists():
        raise FileNotFoundError(f"QE NEB needs initial/POSCAR and final/POSCAR in {case_dir}")
    initial = read_poscar(initial_path)
    final = read_poscar(final_path)
    pseudos = pseudo_files_for_struct(initial, config.get("pseudo_dir"), config.get("pseudo_map"))
    job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(case_dir / "initial", job_dir / "initial", dirs_exist_ok=True)
    shutil.copytree(case_dir / "final", job_dir / "final", dirs_exist_ok=True)
    text = build_neb_input(
        initial, final, pseudos=pseudos,
        kpoints_card=_kpoints_card(kpoints_spec, initial, initial_path, "scf"),
        images=neb_images, qe_params={**_qe_params_from_config(config),
                                     "neb_steps": config.get("qe_neb_steps", 100),
                                     "neb_path_thr": config.get("qe_neb_path_thr", 0.05)},
        spin=spin, magmom_map=magmom_map, prefix=str(config.get("qe_prefix", "vasp_auto")),
    )
    (job_dir / "neb.in").write_text(text, encoding="utf-8")
    (job_dir / "qe_stages.json").write_text(json.dumps({
        "version": 1, "calc_type": "neb",
        "stages": [{"program": "neb.x", "input": "neb.in", "output": "neb.out", "mpi": True}],
    }, indent=2) + "\n", encoding="utf-8")
    pseudo_out = job_dir / "pseudo"
    pseudo_out.mkdir(exist_ok=True)
    _copy_pseudos(pseudos, config, pseudo_out)
    (job_dir / ".engine").write_text("qe\n", encoding="utf-8")
    return job_dir


def _struct_and_pseudos(case_dir: Path, config: dict):
    poscar = case_dir / "POSCAR"
    if not poscar.exists():
        raise FileNotFoundError(f"Missing POSCAR in {case_dir}")
    struct = read_poscar(poscar)
    pseudos = pseudo_files_for_struct(
        struct, config.get("pseudo_dir"), config.get("pseudo_map")
    )
    return poscar, struct, pseudos


def create_qe_job(
    case_info: dict,
    config: dict,
    *,
    calc_type: str | None = None,
    kpoints_spec: dict | None = None,
    spin: bool = False,
    magmom_map: dict | None = None,
) -> Path:
    """Prepare a Quantum ESPRESSO job directory (pw.in + pseudo/ + .engine).

    A user-supplied pw.in in the case directory takes precedence over the
    generated one (mirrors the VASP case-INCAR rule).
    """
    case_dir = Path(case_info["case_dir"]).resolve()
    job_dir = Path(case_info["job_dir"]).resolve()
    calc_type = (calc_type or case_info.get("calculation_type") or "scf")
    if calc_type not in QE_CALC_MAP:
        raise ValueError(
            f"Quantum ESPRESSO backend does not support calc-type '{calc_type}'. "
            f"Supported: {', '.join(QE_SUPPORTED_CALC_TYPES)}."
        )

    job_dir.mkdir(parents=True, exist_ok=True)
    poscar, struct, pseudos = _struct_and_pseudos(case_dir, config)
    shutil.copy2(poscar, job_dir / "POSCAR")  # keep POSCAR for viewers/builders

    user_input = case_dir / "pw.in"
    if user_input.exists():
        if calc_type != "scf":
            print(f"Note      : {case_dir.name} has its own pw.in; it overrides the "
                  f"'{calc_type}' template")
        pw_text = user_input.read_text(encoding="utf-8")
    else:
        kcard = _kpoints_card(kpoints_spec, struct, poscar, calc_type)
        pw_text = build_pw_input(
            struct,
            calc_type,
            pseudos=pseudos,
            kpoints_card=kcard,
            qe_params=_qe_params_from_config(config),
            spin=spin,
            magmom_map=magmom_map,
            prefix=str(config.get("qe_prefix", "vasp_auto")),
        )
    (job_dir / "pw.in").write_text(pw_text, encoding="utf-8")

    auxiliary, stages = qe_stage_plan(calc_type, config, struct)
    if calc_type in ("dos", "bands", "optics"):
        scf_text = build_pw_input(
            struct, "scf", pseudos=pseudos,
            kpoints_card=_kpoints_card(kpoints_spec, struct, poscar, "scf"),
            prefix=str(config.get("qe_prefix", "vasp_auto")),
            qe_params=_qe_params_from_config(config), spin=spin, magmom_map=magmom_map,
        )
        (job_dir / "scf.in").write_text(scf_text, encoding="utf-8")
    for name, text in auxiliary.items():
        (job_dir / name).write_text(text, encoding="utf-8")
    (job_dir / "qe_stages.json").write_text(
        json.dumps({"version": 1, "calc_type": calc_type, "stages": stages}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Stage the pseudopotentials next to the input (pseudo_dir = ./pseudo).
    pseudo_out = job_dir / "pseudo"
    pseudo_out.mkdir(exist_ok=True)
    _copy_pseudos(pseudos, config, pseudo_out)

    (job_dir / ".engine").write_text("qe\n", encoding="utf-8")
    return job_dir


def _copy_pseudos(pseudos: dict[str, str], config: dict, dest: Path):
    pseudo_dir = config.get("pseudo_dir")
    for filename in dict.fromkeys(pseudos.values()):
        src = Path(pseudo_dir).expanduser() / filename if pseudo_dir else Path(filename)
        if src.exists():
            shutil.copy2(src, dest / Path(filename).name)
        elif not (dest / Path(filename).name).exists():
            # Pseudo not found locally: leave a clear breadcrumb rather than fail
            # silently — the run will report the missing file.
            print(f"Warning   : pseudopotential not found: {src}")


def preview_qe_job(
    case_info: dict,
    config: dict,
    *,
    calc_type: str | None = None,
    kpoints_spec: dict | None = None,
    spin: bool = False,
    magmom_map: dict | None = None,
) -> dict:
    """Return the QE input set as a dict without writing anything (dry-run/GUI)."""
    case_dir = Path(case_info["case_dir"]).resolve()
    calc_type = (calc_type or case_info.get("calculation_type") or "scf")
    if calc_type not in QE_CALC_MAP:
        raise ValueError(
            f"Quantum ESPRESSO backend does not support calc-type '{calc_type}'. "
            f"Supported: {', '.join(QE_SUPPORTED_CALC_TYPES)}."
        )

    poscar, struct, pseudos = _struct_and_pseudos(case_dir, config)
    user_input = case_dir / "pw.in"
    if user_input.exists():
        pw_text = user_input.read_text(encoding="utf-8")
    else:
        kcard = _kpoints_card(kpoints_spec, struct, poscar, calc_type)
        pw_text = build_pw_input(
            struct,
            calc_type,
            pseudos=pseudos,
            kpoints_card=kcard,
            qe_params=_qe_params_from_config(config),
            spin=spin,
            magmom_map=magmom_map,
            prefix=str(config.get("qe_prefix", "vasp_auto")),
        )

    auxiliary, stages = qe_stage_plan(calc_type, config, struct)
    preview = {
        "case_name": case_info["case_name"],
        "calculation_type": case_info.get("calculation_type", "scf"),
        "calc_type": calc_type,
        "engine": "qe",
        "job_dir": str(case_info["job_dir"]),
        "POSCAR": str(poscar),
        "pw.in": pw_text,
        "pseudos": ", ".join(f"{el}: {f}" for el, f in pseudos.items()),
        "stages": stages,
    }
    preview.update(auxiliary)
    return preview
