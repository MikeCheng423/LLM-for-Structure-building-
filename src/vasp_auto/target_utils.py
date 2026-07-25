from pathlib import Path


def get_case_type(case_dir):
    case_dir = Path(case_dir)

    # Files (or symlinks to files) inside a project folder are not cases.
    if not case_dir.is_dir():
        return None

    # A case is defined by a POSCAR *file*; a subdirectory that happens to be
    # named POSCAR (e.g. a case folder literally called "POSCAR" sitting inside a
    # project root) must not make the parent look like a case.
    if (case_dir / "initial" / "POSCAR").is_file() and (case_dir / "final" / "POSCAR").is_file():
        return "tss"

    image_dirs = [p for p in case_dir.iterdir() if p.is_dir() and p.name.isdigit()] if case_dir.exists() else []
    if (case_dir / "INCAR").is_file() and len(image_dirs) >= 2 and (case_dir / "00" / "POSCAR").is_file():
        return "tss"

    if (case_dir / "POSCAR").is_file():
        return "scf"

    return None


def inspect_target(target_path):
    target_path = Path(target_path).resolve()

    if not target_path.exists():
        raise FileNotFoundError(f"Target not found: {target_path}")

    if target_path.is_file():
        raise ValueError(f"Target must be a directory, got file: {target_path}")

    single_case_type = get_case_type(target_path)

    if single_case_type:
        return {
            "mode": "single",
            "project_name": target_path.name,
            "case_dirs": [target_path],
            "case_types": {target_path.name: single_case_type},
        }

    case_dirs = []
    case_types = {}
    for child in sorted(target_path.iterdir()):
        case_type = get_case_type(child)
        if child.is_dir() and case_type:
            case_dirs.append(child)
            case_types[child.name] = case_type

    if not case_dirs:
        raise ValueError(
            f"No valid VASP case directories found under: {target_path}\n"
            "A simple SCF case only needs POSCAR. A TSS/NEB case needs "
            "initial/POSCAR and final/POSCAR."
        )

    return {
        "mode": "project",
        "project_name": target_path.name,
        "case_dirs": case_dirs,
        "case_types": case_types,
    }


def filter_case_dirs(case_dirs, selected_cases=None):
    if not selected_cases:
        return list(case_dirs)

    selected = set(selected_cases)
    return [c for c in case_dirs if c.name in selected]
