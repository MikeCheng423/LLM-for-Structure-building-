import re
import shutil
from pathlib import Path

# A POSCAR species symbol is a chemical element (e.g. "Au") optionally followed
# by a pseudopotential-variant suffix (e.g. "Fe_pv", "O_s"). VASP 6.x can write
# a decorated token into CONTCAR — e.g. "Au/d0044ae04e2" — when the POTCAR
# carries a "SHA256 = <hash> Au/POTCAR" line; reusing that CONTCAR as a POSCAR
# (the relax-restart path) would otherwise send a bogus name to the POTCAR
# finder. Keep only the leading element + optional _variant.
_SPECIES_RE = re.compile(r"^[A-Z][a-z]?(?:_[A-Za-z0-9]+)?")


def clean_species_symbol(token: str) -> str:
    """Strip VASP/CONTCAR decoration from a species token, keeping El[_variant]."""
    head = token.split("/", 1)[0].strip()
    match = _SPECIES_RE.match(head)
    return match.group(0) if match else head


def map_potcar_dirs(elements, potcar_map: dict | None = None):
    """Map element symbols to POTCAR sub-directory names.

    potcar_map (from config.yaml) selects pseudopotential variants,
    e.g. {"Fe": "Fe_pv", "O": "O_s"}. Unmapped elements use the bare symbol.
    """
    potcar_map = potcar_map or {}
    return [str(potcar_map.get(element, element)) for element in elements]


def get_elements_from_poscar(poscar_path: Path):
    with open(poscar_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError(f"POSCAR is too short to contain element symbols: {poscar_path}")

    elements = lines[5].split()
    if not elements or all(part.isdigit() for part in elements):
        raise ValueError(
            f"POSCAR does not contain element symbols on line 6: {poscar_path}. "
            "Use VASP 5 style POSCAR with symbols such as: Al O"
        )
    return [clean_species_symbol(element) for element in elements]


def candidate_potcar_roots(potcar_root: str | None = None, poscar_path: Path | None = None):
    roots = []

    if potcar_root:
        configured = Path(potcar_root).expanduser()
        roots.append(configured)
        parent = configured.parent
        roots.append(parent / "POTCAR")
        roots.append(parent / "potcar")

    if poscar_path is not None:
        poscar_path = Path(poscar_path).resolve()
        for parent in [poscar_path.parent, *poscar_path.parents]:
            roots.append(parent / "POTCAR")
            roots.append(parent / "potcar")

    roots.append(Path.cwd() / "POTCAR")
    roots.append(Path.cwd() / "potcar")
    roots.append(Path(__file__).resolve().parents[2] / "POTCAR")
    roots.append(Path(__file__).resolve().parents[2] / "potcar")

    unique_roots = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            unique_roots.append(resolved)
            seen.add(resolved)

    return unique_roots


def find_potcar_root(elements, potcar_root: str | None = None, poscar_path: Path | None = None):
    checked = []
    for root in candidate_potcar_roots(potcar_root, poscar_path):
        checked.append(root)
        if all((root / element / "POTCAR").exists() for element in elements):
            return root

    missing_by_root = []
    for root in checked:
        missing = [element for element in elements if not (root / element / "POTCAR").exists()]
        missing_by_root.append(f"{root}: missing {', '.join(missing)}")

    raise FileNotFoundError(
        "Could not find a POTCAR library containing all required entries.\n"
        f"POTCAR folders needed: {', '.join(elements)}\n"
        "Checked:\n"
        + "\n".join(missing_by_root)
    )


def build_potcar(poscar_path: str, potcar_root: str | None, output_path: str, potcar_map: dict | None = None):
    poscar_path = Path(poscar_path)
    output_path = Path(output_path)

    elements = get_elements_from_poscar(poscar_path)
    potcar_dirs = map_potcar_dirs(elements, potcar_map)
    potcar_root = find_potcar_root(potcar_dirs, potcar_root, poscar_path)

    with open(output_path, "wb") as outfile:
        for potcar_dir in potcar_dirs:
            potcar_file = potcar_root / potcar_dir / "POTCAR"

            with open(potcar_file, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    print(f"POTCAR created: {output_path} from {potcar_root} ({', '.join(potcar_dirs)})")
