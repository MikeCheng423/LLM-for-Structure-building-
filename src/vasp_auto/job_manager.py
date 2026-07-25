from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

# Job folders are prefixed with a zero-padded global sequence number so a re-run
# never overwrites an earlier one, e.g. ``0001_Fe``, ``0002_Si``, ``0003_Fe``.
_JOB_NUMBER_RE = re.compile(r"^(\d+)_(.*)$")
_JOB_NUMBER_WIDTH = 4

from vasp_auto.ase_tools import interpolate_neb_with_ase
from vasp_auto.incar import apply_spin_to_incar, spin_incar_text
from vasp_auto.kpoints import kpoints_text_from_spec
from vasp_auto.potcar_finder import build_potcar, get_elements_from_poscar, map_potcar_dirs
from vasp_auto.target_utils import get_case_type


OPTIONAL_INPUTS = ["POTCAR"]
DEFAULT_NEB_IMAGES = 5

INCAR_TEMPLATE_FILES = {
    "scf": "INCAR_scf",
    "relax": "INCAR_optimize_structure",
    "dos": "INCAR_dos",
    "bands": "INCAR_bands",
    "charge": "INCAR_charge_density",
    "neb": "INCAR_neb",
    "md": "INCAR_md",
    "phonon": "INCAR_phonon",
    "hse06": "INCAR_hse06",
    "freq": "INCAR_freq",
    "optics": "INCAR_optics",
    "workfunction": "INCAR_workfunction",
}

DEFAULT_SCF_INCAR = """SYSTEM = vasp_auto_scf
PREC = Normal
ENCUT = 520
EDIFF = 1E-4
IBRION = -1
NSW = 0
ISMEAR = 0
SIGMA = 0.05
LREAL = .FALSE.
LWAVE = .FALSE.
LCHARG = .FALSE.
"""

DEFAULT_NEB_INCAR = """SYSTEM = vasp_auto_neb
PREC = Accurate
ENCUT = 520
EDIFF = 1E-4
EDIFFG = -0.03
IBRION = 3
POTIM = 0
NSW = 200
SPRING = -5
LCLIMB = .TRUE.
ISYM = 0
LREAL = .FALSE.
ALGO = Normal
ISMEAR = 0
SIGMA = 0.05
LWAVE = .FALSE.
LCHARG = .FALSE.
"""

DEFAULT_KPOINTS = """Automatic mesh
0
Gamma
1 1 1
0 0 0
"""

BUILTIN_INCAR_DEFAULTS = {
    "scf": DEFAULT_SCF_INCAR,
    "neb": DEFAULT_NEB_INCAR,
}


def _template_search_dirs() -> list[Path]:
    dirs = []
    env_root = os.environ.get("VASP_AUTO_ROOT")
    if env_root:
        dirs.append(Path(env_root) / "example")
    dirs.append(Path(__file__).resolve().parents[2] / "example")
    return dirs


def load_incar_template(calc_type: str) -> str:
    """Return the INCAR text for a calculation type.

    Looks for example/INCAR_<type> (VASP_AUTO_ROOT first, then the repo root)
    and falls back to the built-in default string when no template file exists.
    """
    filename = INCAR_TEMPLATE_FILES.get(calc_type)
    if filename:
        for directory in _template_search_dirs():
            template_path = directory / filename
            if template_path.exists():
                return template_path.read_text(encoding="utf-8")

    try:
        return BUILTIN_INCAR_DEFAULTS[calc_type]
    except KeyError:
        raise ValueError(
            f"No INCAR template found for calculation type '{calc_type}'. "
            f"Add example/{filename or f'INCAR_{calc_type}'} or use a built-in type: "
            + ", ".join(sorted(BUILTIN_INCAR_DEFAULTS))
        ) from None


def _max_job_number(output_root: Path) -> int:
    """Highest ``NNNN_`` prefix already used under ``output_root`` (0 if none)."""
    if not output_root.is_dir():
        return 0
    numbers = [
        int(m.group(1))
        for child in output_root.iterdir()
        if child.is_dir() and (m := _JOB_NUMBER_RE.match(child.name))
    ]
    return max(numbers, default=0)


def _allocate_numbered_dir(output_root: Path, case_name: str, create: bool) -> Path:
    """Return the next free ``NNNN_<case>`` dir under ``output_root``.

    With ``create=True`` the directory is claimed atomically (exclusive mkdir) so
    concurrent runs (``--parallel``) never collide on a number; with
    ``create=False`` it just predicts the name (for dry-run previews).
    """
    output_root.mkdir(parents=True, exist_ok=True)
    number = _max_job_number(output_root) + 1
    while True:
        candidate = output_root / f"{number:0{_JOB_NUMBER_WIDTH}d}_{case_name}"
        if not create:
            return candidate
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            number += 1


def _latest_numbered_dir(output_root: Path, case_name: str) -> Path:
    """The most recent existing job dir for ``case_name`` (highest number, else a
    bare ``<case>``). Used to re-read / continue a previous run. Falls back to the
    bare path when nothing exists yet."""
    best_number, best = -1, None
    if output_root.is_dir():
        for child in output_root.iterdir():
            if not child.is_dir():
                continue
            if child.name == case_name and best_number < 0:
                best_number, best = 0, child
            elif (m := _JOB_NUMBER_RE.match(child.name)) and m.group(2) == case_name:
                if int(m.group(1)) > best_number:
                    best_number, best = int(m.group(1)), child
    return best if best is not None else output_root / case_name


def make_case_info(case_dir, output_root, single_mode=False, job_mode="bare"):
    """Build the case/job descriptor.

    ``job_mode`` controls how the job directory is named (ignored in single_mode,
    which always runs in ``output_root`` itself):
      - ``"bare"``    : ``output_root/<case>`` (legacy; the default).
      - ``"new"``     : allocate & claim the next ``NNNN_<case>`` (a fresh run).
      - ``"preview"`` : predict the next ``NNNN_<case>`` without creating it.
      - ``"latest"``  : the most recent existing job dir (re-read / retry).
    """
    case_dir = Path(case_dir).resolve()
    output_root = Path(output_root).resolve()
    calculation_type = get_case_type(case_dir)

    if calculation_type is None:
        raise ValueError(
            f"Cannot identify calculation type for {case_dir}. "
            "Use POSCAR for SCF, or initial/POSCAR and final/POSCAR for TSS."
        )

    # Numbered modes apply in single mode too: a single-case run lands in
    # output_root/<NNNN>_<case> (e.g. jobs/Cu/0001_Cu) so reruns are kept side by
    # side. Only the legacy "bare" mode honours single_mode's run-in-place layout.
    if job_mode == "new":
        job_dir = _allocate_numbered_dir(output_root, case_dir.name, create=True)
    elif job_mode == "preview":
        job_dir = _allocate_numbered_dir(output_root, case_dir.name, create=False)
    elif job_mode == "latest":
        job_dir = _latest_numbered_dir(output_root, case_dir.name)
    elif single_mode:
        job_dir = output_root
    else:
        job_dir = output_root / case_dir.name

    return {
        "case_name": case_dir.name,
        "case_dir": case_dir,
        "job_dir": job_dir,
        "single_mode": single_mode,
        "calculation_type": calculation_type,
    }


def _safe_remove_contents(folder: Path):
    if not folder.exists():
        return
    for item in folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _copy_optional_inputs(case_dir: Path, job_dir: Path):
    for name in OPTIONAL_INPUTS:
        src = case_dir / name
        if src.exists():
            shutil.copy2(src, job_dir / name)


def _copy_or_default(src: Path, dst: Path, default_text: str):
    """Copy the user's input file when present, otherwise write the default.

    Generated files go only into the job directory; the user's case directory
    is never modified.
    """
    if src.exists():
        shutil.copy2(src, dst)
    else:
        dst.write_text(default_text, encoding="utf-8")


def _write_kpoints(case_dir: Path, job_dir: Path, kpoints_spec, poscar_path: Path):
    if kpoints_spec:
        text = kpoints_text_from_spec(kpoints_spec, poscar_path=poscar_path)
        (job_dir / "KPOINTS").write_text(text, encoding="utf-8")
    else:
        _copy_or_default(case_dir / "KPOINTS", job_dir / "KPOINTS", DEFAULT_KPOINTS)


def _copy_or_build_potcar(case_dir: Path, job_dir: Path, poscar_path: Path, potcar_root, potcar_map=None):
    input_potcar = case_dir / "POTCAR"
    if input_potcar.exists():
        shutil.copy2(input_potcar, job_dir / "POTCAR")
    else:
        build_potcar(
            poscar_path=str(poscar_path),
            potcar_root=potcar_root,
            output_path=str(job_dir / "POTCAR"),
            potcar_map=potcar_map,
        )


def _write_neb_incar(case_dir: Path, job_dir: Path, neb_images: int):
    template_path = case_dir / "INCAR.neb"
    if not template_path.exists():
        template_path = case_dir / "INCAR"

    if template_path.exists():
        lines = template_path.read_text(encoding="utf-8").splitlines()
        wrote_images = False
        updated = []
        for line in lines:
            if line.strip().upper().startswith("IMAGES"):
                updated.append(f"IMAGES = {neb_images}")
                wrote_images = True
            else:
                updated.append(line)
        if not wrote_images:
            updated.append(f"IMAGES = {neb_images}")
        (job_dir / "INCAR").write_text("\n".join(updated) + "\n", encoding="utf-8")
    else:
        default_text = load_incar_template("neb") + f"IMAGES = {neb_images}\n"
        (job_dir / "INCAR").write_text(default_text, encoding="utf-8")


def _select_incar_text(case_dir: Path, calc_type: str | None) -> str:
    """Pick the INCAR for an SCF-like job: user's case INCAR wins over templates."""
    case_incar = case_dir / "INCAR"
    if case_incar.exists():
        if calc_type and calc_type != "scf":
            print(
                f"Note      : {case_dir.name} has its own INCAR; "
                f"it takes precedence over the '{calc_type}' template"
            )
        return case_incar.read_text(encoding="utf-8")
    return load_incar_template(calc_type or "scf")


def _copy_scf_inputs(case_dir: Path, job_dir: Path, calc_type: str | None = None, kpoints_spec=None):
    poscar = case_dir / "POSCAR"
    if not poscar.exists():
        raise FileNotFoundError(f"Missing POSCAR in {case_dir}")

    shutil.copy2(poscar, job_dir / "POSCAR")
    (job_dir / "INCAR").write_text(_select_incar_text(case_dir, calc_type), encoding="utf-8")
    _write_kpoints(case_dir, job_dir, kpoints_spec, job_dir / "POSCAR")
    _copy_optional_inputs(case_dir, job_dir)


def _read_poscar_for_interpolation(poscar_path: Path):
    lines = poscar_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise ValueError(f"POSCAR is too short: {poscar_path}")

    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    atom_count = sum(counts)

    coord_mode_line = 7
    selective = lines[coord_mode_line].strip().lower().startswith("s")
    if selective:
        coord_mode_line += 1

    coord_mode = lines[coord_mode_line].strip()
    coord_start = coord_mode_line + 1
    coord_lines = lines[coord_start:coord_start + atom_count]
    if len(coord_lines) != atom_count:
        raise ValueError(f"POSCAR atom count does not match coordinates: {poscar_path}")

    coords = []
    suffixes = []
    for line in coord_lines:
        parts = line.split()
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        suffixes.append(parts[3:])

    return {
        "header": lines[:coord_start],
        "elements": elements,
        "counts": counts,
        "coord_mode": coord_mode,
        "coords": coords,
        "suffixes": suffixes,
        "tail": lines[coord_start + atom_count:],
    }


def _validate_interpolation_inputs(initial, final, initial_path: Path, final_path: Path):
    if initial["elements"] != final["elements"] or initial["counts"] != final["counts"]:
        raise ValueError(
            "initial/POSCAR and final/POSCAR must have the same element order "
            f"and atom counts: {initial_path}, {final_path}"
        )

    if initial["coord_mode"].strip().lower()[0] != final["coord_mode"].strip().lower()[0]:
        raise ValueError(
            "initial/POSCAR and final/POSCAR must use the same coordinate mode "
            "(Direct or Cartesian)."
        )


def _interpolate_coords(initial, final, step: int, total_steps: int):
    fraction = step / total_steps
    direct_mode = initial["coord_mode"].strip().lower().startswith("d")
    coords = []

    for start, end in zip(initial["coords"], final["coords"]):
        row = []
        for a, b in zip(start, end):
            delta = b - a
            if direct_mode:
                if delta > 0.5:
                    delta -= 1.0
                elif delta < -0.5:
                    delta += 1.0
                value = (a + fraction * delta) % 1.0
            else:
                value = a + fraction * delta
            row.append(value)
        coords.append(row)

    return coords


def _write_poscar(template, coords, output_path: Path):
    lines = list(template["header"])
    for coord, suffix in zip(coords, template["suffixes"]):
        line = f"{coord[0]: .16f} {coord[1]: .16f} {coord[2]: .16f}"
        if suffix:
            line += " " + " ".join(suffix)
        lines.append(line)
    lines.extend(template["tail"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_neb_inputs(
    case_dir: Path,
    job_dir: Path,
    neb_images: int,
    use_ase_neb: bool = False,
    ase_neb_method: str = "idpp",
):
    initial_path = case_dir / "initial" / "POSCAR"
    final_path = case_dir / "final" / "POSCAR"

    existing_image_dirs = [p for p in sorted(case_dir.iterdir()) if p.is_dir() and p.name.isdigit()]
    if (not initial_path.exists() or not final_path.exists()) and existing_image_dirs:
        for image_dir in existing_image_dirs:
            target_dir = job_dir / image_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_dir / "POSCAR", target_dir / "POSCAR")
        _write_neb_incar(case_dir, job_dir, max(len(existing_image_dirs) - 2, 1))
        _copy_or_default(case_dir / "KPOINTS", job_dir / "KPOINTS", DEFAULT_KPOINTS)
        _copy_optional_inputs(case_dir, job_dir)
        return

    if not initial_path.exists() or not final_path.exists():
        raise FileNotFoundError(
            f"TSS case requires initial/POSCAR and final/POSCAR in {case_dir}"
        )

    if use_ase_neb:
        interpolate_neb_with_ase(
            initial_poscar=initial_path,
            final_poscar=final_path,
            job_dir=job_dir,
            neb_images=neb_images,
            method=ase_neb_method,
        )
    else:
        initial = _read_poscar_for_interpolation(initial_path)
        final = _read_poscar_for_interpolation(final_path)
        _validate_interpolation_inputs(initial, final, initial_path, final_path)

        total_steps = neb_images + 1
        for step in range(total_steps + 1):
            image_dir = job_dir / f"{step:02d}"
            image_dir.mkdir(parents=True, exist_ok=True)
            if step == 0:
                shutil.copy2(initial_path, image_dir / "POSCAR")
            elif step == total_steps:
                shutil.copy2(final_path, image_dir / "POSCAR")
            else:
                coords = _interpolate_coords(initial, final, step, total_steps)
                _write_poscar(initial, coords, image_dir / "POSCAR")

    _write_neb_incar(case_dir, job_dir, neb_images)
    _copy_or_default(case_dir / "KPOINTS", job_dir / "KPOINTS", DEFAULT_KPOINTS)
    _copy_optional_inputs(case_dir, job_dir)


DEFAULT_SOLVATION_EPS = 78.4  # water dielectric constant


def _apply_solvation(job_dir: Path, eps: float = DEFAULT_SOLVATION_EPS):
    """Inject LSOL and EB_K into the job INCAR for implicit solvation (VASPsol).

    NOTE: requires a VASPsol-patched VASP binary. Without the patch, VASP will
    abort with an unknown tag error. See docs/MANUAL.md for setup instructions.
    """
    from vasp_auto.incar import set_incar_value
    incar = job_dir / "INCAR"
    set_incar_value(incar, "LSOL", ".TRUE.")
    set_incar_value(incar, "EB_K", eps)


def create_job_from_case(
    case_info,
    potcar_root=None,
    clean_job=True,
    neb_images=DEFAULT_NEB_IMAGES,
    use_ase_neb=False,
    ase_neb_method="idpp",
    potcar_map=None,
    calc_type=None,
    kpoints_spec=None,
    spin=False,
    magmom_map=None,
    solvation=False,
    solvation_eps=DEFAULT_SOLVATION_EPS,
    engine="vasp",
    config=None,
):
    case_dir = Path(case_info["case_dir"]).resolve()
    job_dir = Path(case_info["job_dir"]).resolve()
    calculation_type = case_info.get("calculation_type") or get_case_type(case_dir)

    job_dir.mkdir(parents=True, exist_ok=True)

    if clean_job:
        _safe_remove_contents(job_dir)

    if engine == "qe":
        # Open-source path: write Quantum ESPRESSO native inputs instead of
        # INCAR/KPOINTS/POTCAR.
        from vasp_auto.qe_tools import create_qe_job, create_qe_neb_job
        if calculation_type == "tss":
            return create_qe_neb_job(
                case_info, config or {}, neb_images=int(neb_images),
                kpoints_spec=kpoints_spec, spin=spin, magmom_map=magmom_map,
            )
        return create_qe_job(
            case_info,
            config or {},
            calc_type=calc_type,
            kpoints_spec=kpoints_spec,
            spin=spin,
            magmom_map=magmom_map,
        )

    if engine == "ase":
        # Generic ASE-calculator path: write run_ase.py + ase_calc.json instead of
        # INCAR/KPOINTS/POTCAR. NEB/TSS is VASP-only for now.
        from vasp_auto.ase_engine import create_ase_job
        if calculation_type == "tss":
            raise ValueError("ASE engine does not support TSS/NEB yet; use the VASP engine.")
        return create_ase_job(
            case_info,
            config or {},
            calc_type=calc_type,
            kpoints_spec=kpoints_spec,
            spin=spin,
            magmom_map=magmom_map,
        )

    if calculation_type == "tss":
        _copy_neb_inputs(
            case_dir,
            job_dir,
            int(neb_images),
            use_ase_neb=use_ase_neb,
            ase_neb_method=ase_neb_method,
        )
        potcar_poscar = job_dir / "00" / "POSCAR"
    else:
        _copy_scf_inputs(case_dir, job_dir, calc_type=calc_type, kpoints_spec=kpoints_spec)
        potcar_poscar = job_dir / "POSCAR"

    if spin:
        apply_spin_to_incar(job_dir / "INCAR", potcar_poscar, magmom_map)

    if solvation:
        _apply_solvation(job_dir, eps=solvation_eps)

    _copy_or_build_potcar(case_dir, job_dir, potcar_poscar, potcar_root, potcar_map=potcar_map)

    return job_dir


def preview_job_from_case(
    case_info,
    potcar_root=None,
    potcar_map=None,
    calc_type=None,
    kpoints_spec=None,
    neb_images=DEFAULT_NEB_IMAGES,
    spin=False,
    magmom_map=None,
    engine="vasp",
    config=None,
):
    """Build the full input set as a dict without writing any job files.

    Used by --dry-run and intended for GUI previews.
    """
    case_dir = Path(case_info["case_dir"]).resolve()
    calculation_type = case_info.get("calculation_type") or get_case_type(case_dir)

    if engine == "qe":
        from vasp_auto.qe_tools import preview_qe_job, build_neb_input, pseudo_files_for_struct
        if calculation_type == "tss":
            from vasp_auto.structure import read_poscar
            initial_path = case_dir / "initial" / "POSCAR"
            final_path = case_dir / "final" / "POSCAR"
            initial = read_poscar(initial_path)
            final = read_poscar(final_path)
            cfg = config or {}
            pseudos = pseudo_files_for_struct(initial, cfg.get("pseudo_dir"), cfg.get("pseudo_map"))
            from vasp_auto.qe_tools import _kpoints_card, _qe_params_from_config
            text = build_neb_input(
                initial, final, pseudos=pseudos,
                kpoints_card=_kpoints_card(kpoints_spec, initial, initial_path, "scf"),
                images=int(neb_images), qe_params=_qe_params_from_config(cfg),
                spin=spin, magmom_map=magmom_map,
            )
            return {
                "case_name": case_info["case_name"], "calculation_type": "tss",
                "calc_type": "neb", "engine": "qe", "job_dir": str(case_info["job_dir"]),
                "POSCAR": str(initial_path), "neb.in": text,
                "pseudos": ", ".join(f"{el}: {name}" for el, name in pseudos.items()),
                "stages": [{"program": "neb.x", "input": "neb.in", "output": "neb.out", "mpi": True}],
            }
        return preview_qe_job(
            case_info,
            config or {},
            calc_type=calc_type,
            kpoints_spec=kpoints_spec,
            spin=spin,
            magmom_map=magmom_map,
        )

    if engine == "ase":
        from vasp_auto.ase_engine import preview_ase_job
        if calculation_type == "tss":
            raise ValueError("ASE engine does not support TSS/NEB yet; use the VASP engine.")
        return preview_ase_job(
            case_info,
            config or {},
            calc_type=calc_type,
            kpoints_spec=kpoints_spec,
            spin=spin,
            magmom_map=magmom_map,
        )

    if calculation_type == "tss":
        incar_text = load_incar_template(calc_type or "neb") + f"IMAGES = {neb_images}\n"
        case_incar = case_dir / "INCAR"
        if (case_dir / "INCAR.neb").exists():
            incar_text = (case_dir / "INCAR.neb").read_text(encoding="utf-8")
        elif case_incar.exists():
            incar_text = case_incar.read_text(encoding="utf-8")
        poscar_path = case_dir / "initial" / "POSCAR"
        images = neb_images
    else:
        incar_text = _select_incar_text(case_dir, calc_type)
        poscar_path = case_dir / "POSCAR"
        images = None

    if spin and poscar_path.exists():
        incar_text = spin_incar_text(incar_text, poscar_path, magmom_map)

    if kpoints_spec:
        kpoints_text = kpoints_text_from_spec(kpoints_spec, poscar_path=poscar_path)
    elif (case_dir / "KPOINTS").exists():
        kpoints_text = (case_dir / "KPOINTS").read_text(encoding="utf-8")
    else:
        kpoints_text = DEFAULT_KPOINTS

    if (case_dir / "POTCAR").exists():
        potcar_summary = f"user-supplied: {case_dir / 'POTCAR'}"
    else:
        elements = get_elements_from_poscar(poscar_path)
        potcar_summary = ", ".join(map_potcar_dirs(elements, potcar_map))

    return {
        "case_name": case_info["case_name"],
        "calculation_type": calculation_type,
        "calc_type": calc_type or ("neb" if calculation_type == "tss" else "scf"),
        "job_dir": str(case_info["job_dir"]),
        "POSCAR": str(poscar_path),
        "INCAR": incar_text,
        "KPOINTS": kpoints_text,
        "POTCAR": potcar_summary,
        "neb_images": images,
    }
