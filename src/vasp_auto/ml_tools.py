"""Machine-learning interatomic potentials: pre-relax and screen structures.

The flagship MLIP backend is Meta FAIR's OMat24 effort (arXiv:2410.12771):
universal MLIPs trained on ~110M inorganic-materials DFT calculations, served
through the ``fairchem-core`` package.  A pre-relaxation typically lands within
a few meV/atom of the PBE minimum, cutting VASP ionic steps by an order of
magnitude.

MLIP backends (all optional, imported lazily):

- ``uma-*`` / other fairchem 2.x names: ``pretrained_mlip.get_predict_unit``
  + ``FAIRChemCalculator`` (``omat`` task for inorganic materials).
  Install: ``pip install fairchem-core`` (UMA checkpoints are gated on
  Hugging Face — ``huggingface-cli login`` once).
- ``--ml-checkpoint`` file: fairchem 1.x ``OCPCalculator`` with a downloaded
  OMat24 checkpoint (eqV2 …).
- ``emt``: ASE's built-in effective-medium potential (Al/Cu/Ni/…); no install,
  used by the tests and good for demos only.

Structure databases (all optional, imported lazily):

- ``mp``  — Materials Project via the ``mp-api`` package (local copy at
  ``/home/vv/api`` or system-installed ``mp-api``). Needs ``MP_API_KEY``.
- ``umat`` — META UMAT database (pending access grant; stub only for now).
"""
from __future__ import annotations

import sys
from pathlib import Path

from vasp_auto.ase_tools import require_ase

DEFAULT_ML_MODEL = "uma-s-1p1"
DEFAULT_ML_TASK = "omat"

# Selectable MLIP backends, in UI-presentation order. ``gated`` marks models
# whose weights need fairchem-core + a Hugging Face login (facebook/UMA access);
# ``emt`` runs out of the box with no download (simple metals only).
ML_MODELS = [
    {"name": "emt", "label": "emt — demo, no install (simple metals)", "gated": False},
    {"name": "uma-s-1p1", "label": "uma-s-1p1 — UMA small (fairchem + HF login)", "gated": True},
    {"name": "uma-s-1p2", "label": "uma-s-1p2 — UMA small v1.2 (fairchem + HF login)", "gated": True},
    {"name": "uma-m-1p1", "label": "uma-m-1p1 — UMA medium (fairchem + HF login)", "gated": True},
]

# Available structure databases, in UI-presentation order.
# ``available`` is False for databases whose access is not yet granted.
DATABASES = [
    {
        "id": "mp",
        "label": "Materials Project (mp-api, set MP_API_KEY env var)",
        "available": True,
    },
    {
        "id": "umat",
        "label": "META UMAT (pending access — available once access is granted)",
        "available": False,
    },
]

_INSTALL_HINT = (
    "Machine-learning relaxation needs the fairchem package for OMat24/UMA "
    "models: pip install fairchem-core. The UMA weights live in the GATED "
    "Hugging Face repo facebook/UMA — request access there, then authenticate "
    "(`huggingface-cli login` or export HF_TOKEN). Current UMA model names are "
    "uma-s-1p1 / uma-s-1p2 / uma-m-1p1 (the old 'uma-s-1' was renamed). For a "
    "dependency-free demo use --ml-model emt (ASE effective-medium, simple metals)."
)

_MP_INSTALL_HINT = (
    "Materials Project database access needs the mp-api package: "
    "pip install mp-api  (or install the local copy: pip install -e /home/vv/api). "
    "Set your API key via the MP_API_KEY environment variable or pass --mp-api-key."
)

# Path to the local mp-api source tree so it can be used without a system install.
_LOCAL_MP_API = Path(__file__).parents[4] / "api"


def get_ml_calculator(
    model: str = DEFAULT_ML_MODEL,
    task: str = DEFAULT_ML_TASK,
    checkpoint: str | None = None,
    device: str | None = None,
):
    """Return an ASE calculator for the requested MLIP backend."""
    require_ase()

    if model.lower() == "emt":
        from ase.calculators.emt import EMT

        return EMT()

    if checkpoint:
        try:
            from fairchem.core import OCPCalculator
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        return OCPCalculator(checkpoint_path=str(checkpoint), cpu=(device != "cuda"))

    try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    predictor = pretrained_mlip.get_predict_unit(model, device=device or "cpu")
    return FAIRChemCalculator(predictor, task_name=task)


def _import_mprester():
    """Import MPRester from system install or local /home/vv/api fallback."""
    try:
        from mp_api.client import MPRester
        return MPRester
    except ImportError:
        pass
    local = str(_LOCAL_MP_API)
    if local not in sys.path:
        sys.path.insert(0, local)
    try:
        from mp_api.client import MPRester
        return MPRester
    except ImportError as exc:
        raise ImportError(_MP_INSTALL_HINT) from exc


def _fetch_from_mp(query: str, api_key: str | None = None) -> tuple[str, str]:
    """Fetch one structure from the Materials Project; return (poscar_text, label).

    ``query`` is either a material_id (``mp-XXXX`` / ``mvc-XXXX``) or a
    chemical formula / chemsys string (e.g. ``Fe2O3``, ``Fe-O``).  When a
    formula matches multiple entries the lowest-energy stable structure is
    preferred; if none are stable the first result is used.
    """
    MPRester = _import_mprester()
    try:
        from pymatgen.io.vasp.inputs import Poscar
    except ImportError as exc:
        raise ImportError(
            "pymatgen is required for Materials Project structure fetch: "
            "pip install pymatgen"
        ) from exc

    with MPRester(api_key=api_key) as mpr:
        if query.startswith(("mp-", "mvc-")):
            structure = mpr.get_structure_by_material_id(query)
            label = query
        else:
            docs = mpr.materials.summary.search(
                formula=query,
                all_fields=False,
                fields=["material_id", "structure", "energy_above_hull", "is_stable"],
            )
            if not docs:
                raise ValueError(
                    f"No structure found for query {query!r} in Materials Project."
                )
            stable = [d for d in docs if d.get("is_stable")]
            doc = min(stable or docs, key=lambda d: d.get("energy_above_hull") or 0)
            structure = doc["structure"]
            label = str(doc.get("material_id", query))

    poscar_text = Poscar(structure).get_str()
    return poscar_text, label


def _fetch_from_umat(query: str, api_key: str | None = None) -> tuple[str, str]:
    """Placeholder for META UMAT database (access pending)."""
    raise NotImplementedError(
        "META UMAT database integration is pending access grant. "
        "Once access is available, set --db-source umat with a valid API key. "
        "Currently available database: --db-source mp (Materials Project)."
    )


def fetch_structure_from_db(
    query: str,
    db_source: str = "mp",
    api_key: str | None = None,
) -> tuple[str, str]:
    """Fetch a structure from an external material database as POSCAR text.

    Args:
        query:     Material ID (``mp-1234``) or formula/chemsys (``Fe2O3``,
                   ``Fe-O``) supported by the chosen database.
        db_source: ``"mp"`` for Materials Project, ``"umat"`` for META UMAT
                   (pending access).
        api_key:   API key for the database.  For MP this can also be set via
                   the ``MP_API_KEY`` environment variable.

    Returns:
        ``(poscar_text, label)`` — POSCAR-format string and a human-readable
        identifier (material_id or query string).
    """
    if db_source == "mp":
        return _fetch_from_mp(query, api_key=api_key)
    elif db_source == "umat":
        return _fetch_from_umat(query, api_key=api_key)
    else:
        known = ", ".join(d["id"] for d in DATABASES)
        raise ValueError(
            f"Unknown database source {db_source!r}. Choose one of: {known}."
        )


def search_mp(
    query: str,
    api_key: str | None = None,
    max_results: int = 20,
) -> list[dict]:
    """Search Materials Project and return a ranked list of candidate materials.

    Powers the UI "Materials Project" search tab: the user types a formula,
    chemical system, or material_id, and gets back the matching entries sorted
    by stability (lowest energy above hull first) so the most relevant prototype
    is at the top.

    Args:
        query:       Material ID (``mp-1234``), formula (``SnO2``, ``Fe2O3``),
                     or chemical system (``Fe-O``, ``Ti-O``).
        api_key:     MP API key (or set ``MP_API_KEY`` env var).
        max_results: Cap on the number of rows returned (default 20).

    Returns:
        A list of dicts, each with ``material_id``, ``formula``,
        ``energy_above_hull`` (eV/atom), ``is_stable`` (bool),
        ``spacegroup`` (symbol, may be ``""``), and ``nsites`` (int).
        Sorted by ``energy_above_hull`` ascending.
    """
    MPRester = _import_mprester()

    query = query.strip()
    fields = [
        "material_id",
        "formula_pretty",
        "energy_above_hull",
        "is_stable",
        "symmetry",
        "nsites",
    ]
    if query.startswith(("mp-", "mvc-")):
        params: dict = {"material_ids": [query]}
    elif "-" in query and not any(ch.isdigit() for ch in query):
        # Chemical system like "Fe-O" (no stoichiometry digits).
        params = {"chemsys": query}
    else:
        params = {"formula": query}

    with MPRester(api_key=api_key) as mpr:
        docs = mpr.materials.summary.search(
            **params, all_fields=False, fields=fields
        )

    rows: list[dict] = []
    for doc in docs:
        data = doc.model_dump() if hasattr(doc, "model_dump") else dict(doc)
        symmetry = data.get("symmetry")
        if isinstance(symmetry, dict):
            spacegroup = symmetry.get("symbol") or ""
        else:
            spacegroup = getattr(symmetry, "symbol", "") or ""
        rows.append(
            {
                "material_id": str(data.get("material_id", "")),
                "formula": data.get("formula_pretty") or "",
                "energy_above_hull": data.get("energy_above_hull"),
                "is_stable": bool(data.get("is_stable")),
                "spacegroup": spacegroup,
                "nsites": data.get("nsites"),
            }
        )

    rows.sort(key=lambda r: (r["energy_above_hull"] is None, r["energy_above_hull"] or 0.0))
    return rows[:max_results]


def prototype_from_mp(
    query: str,
    substitutions: dict[str, str] | None = None,
    api_key: str | None = None,
) -> dict:
    """Fetch a crystal structure from Materials Project and return a structure dict.

    This extends the built-in prototype library (graphene, rutile, …) to any
    of the ~150K materials in MP.  The returned structure dict is compatible
    with ``structure.write_poscar`` and the vasp_auto editor.

    Args:
        query:         Material ID (``mp-1234``) or formula/chemsys (``SnO2``,
                       ``Fe-O``).  For formulas the lowest-energy stable entry
                       is chosen automatically.
        substitutions: Optional element replacement map applied after fetching,
                       e.g. ``{"Ti": "Sn"}`` turns every Ti site into Sn —
                       producing the isostructural SnO2 from a TiO2 prototype.
                       Multiple substitutions are supported: ``{"Ti":"Ge","O":"S"}``.
        api_key:       MP API key (or set ``MP_API_KEY`` env var).

    Returns:
        A vasp_auto structure dict with ``comment``, ``lattice``, ``elements``,
        ``counts``, ``coords`` etc., ready to pass to ``write_poscar``.

    Examples::

        # Rutile SnO2 built from the MP rutile TiO2 prototype
        struct = prototype_from_mp("mp-2657", substitutions={"Ti": "Sn"})

        # Most stable Fe2O3 phase straight from MP
        struct = prototype_from_mp("Fe2O3")
    """
    MPRester = _import_mprester()
    try:
        from pymatgen.io.vasp.inputs import Poscar as _Poscar
    except ImportError as exc:
        raise ImportError("pymatgen is required: pip install pymatgen") from exc

    with MPRester(api_key=api_key) as mpr:
        if query.startswith(("mp-", "mvc-")):
            pymat_struct = mpr.get_structure_by_material_id(query)
            label = query
        else:
            docs = mpr.materials.summary.search(
                formula=query,
                all_fields=False,
                fields=["material_id", "structure", "energy_above_hull", "is_stable"],
            )
            if not docs:
                raise ValueError(
                    f"No structure found for query {query!r} in Materials Project."
                )
            stable = [d for d in docs if d.get("is_stable")]
            doc = min(stable or docs, key=lambda d: d.get("energy_above_hull") or 0)
            pymat_struct = doc["structure"]
            label = str(doc.get("material_id", query))

    from vasp_auto.structure import build_struct, substitute_species

    lattice = [[float(x) for x in row] for row in pymat_struct.lattice.matrix.tolist()]
    symbols = [str(s.element) for s in pymat_struct.species]
    coords = [[float(x) for x in row] for row in pymat_struct.frac_coords.tolist()]
    comment = f"MP:{label} {pymat_struct.formula}"
    struct = build_struct(comment, lattice, symbols, coords)

    if substitutions:
        struct = substitute_species(struct, substitutions)

    return struct


def ml_energy_from_db(
    query: str,
    db_source: str = "mp",
    api_key: str | None = None,
    model: str = DEFAULT_ML_MODEL,
    task: str = DEFAULT_ML_TASK,
    checkpoint: str | None = None,
    calculator=None,
) -> dict:
    """Fetch a structure from a database and compute a single-point MLIP energy.

    Combines :func:`fetch_structure_from_db` and :func:`ml_energy` in one
    call.  Returns the same dict as ``ml_energy`` plus ``db_source`` and
    ``db_query``.
    """
    import tempfile

    poscar_text, label = fetch_structure_from_db(query, db_source=db_source, api_key=api_key)
    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir) / label.replace("/", "_")
        staging.mkdir()
        (staging / "POSCAR").write_text(poscar_text, encoding="utf-8")
        result = ml_energy(staging, model=model, task=task, checkpoint=checkpoint, calculator=calculator)
    result.update({"db_source": db_source, "db_query": query, "db_label": label})
    return result


def ml_relax_from_db(
    query: str,
    output_dir: Path | None = None,
    db_source: str = "mp",
    api_key: str | None = None,
    model: str = DEFAULT_ML_MODEL,
    task: str = DEFAULT_ML_TASK,
    checkpoint: str | None = None,
    fmax: float = 0.05,
    steps: int = 200,
    relax_cell: bool = False,
    calculator=None,
) -> dict:
    """Fetch a structure from a database and ML-relax it.

    Combines :func:`fetch_structure_from_db` and :func:`ml_relax_case` in one
    call.  Returns the same dict as ``ml_relax_case`` plus ``db_source``,
    ``db_query``, and ``db_label``.
    """
    import tempfile

    poscar_text, label = fetch_structure_from_db(query, db_source=db_source, api_key=api_key)
    safe_label = label.replace("/", "_").replace(" ", "_")
    resolved_output = Path(output_dir) if output_dir else Path.cwd() / f"{safe_label}_ml"

    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir) / safe_label
        staging.mkdir()
        (staging / "POSCAR").write_text(poscar_text, encoding="utf-8")
        result = ml_relax_case(
            staging,
            output_dir=resolved_output,
            model=model,
            task=task,
            checkpoint=checkpoint,
            fmax=fmax,
            steps=steps,
            relax_cell=relax_cell,
            calculator=calculator,
        )
    result.update({"db_source": db_source, "db_query": query, "db_label": label})
    return result


def _write_xdatcar(path: Path, atoms, frames: list) -> None:
    """Write optimisation frames (scaled positions) as XDATCAR for the viewer."""
    symbols = atoms.get_chemical_symbols()
    elements: list[str] = []
    counts: list[int] = []
    for symbol in symbols:
        if elements and elements[-1] == symbol:
            counts[-1] += 1
        else:
            elements.append(symbol)
            counts.append(1)

    lines = ["ML relaxation", "1.0"]
    for row in atoms.cell[:]:
        lines.append(f"  {row[0]: .10f} {row[1]: .10f} {row[2]: .10f}")
    lines.append(" ".join(elements))
    lines.append(" ".join(str(c) for c in counts))
    for index, frame in enumerate(frames, start=1):
        lines.append(f"Direct configuration= {index:5d}")
        for coord in frame:
            lines.append(f"  {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ml_relax_case(
    case_dir: Path,
    output_dir: Path | None = None,
    model: str = DEFAULT_ML_MODEL,
    task: str = DEFAULT_ML_TASK,
    checkpoint: str | None = None,
    fmax: float = 0.05,
    steps: int = 200,
    relax_cell: bool = False,
    calculator=None,
) -> dict:
    """Relax case_dir/POSCAR with an MLIP; write a derived case directory.

    Follows the structure-edit convention: the input case is never mutated.
    The derived case gets the relaxed POSCAR plus an XDATCAR of the
    optimisation path (so the UI's animation button works on it).
    Returns {case_dir, energy_eV, max_force_eV_A, steps, converged, model}.
    """
    require_ase()
    from ase.io import read, write
    from ase.optimize import FIRE

    case_dir = Path(case_dir)
    poscar = case_dir / "POSCAR"
    if not poscar.exists():
        raise FileNotFoundError(f"ML relaxation needs a POSCAR in {case_dir}")

    atoms = read(poscar, format="vasp")
    atoms.calc = calculator or get_ml_calculator(model, task=task, checkpoint=checkpoint)

    target = atoms
    if relax_cell:
        from ase.filters import FrechetCellFilter

        target = FrechetCellFilter(atoms)

    frames = [atoms.get_scaled_positions().tolist()]
    optimizer = FIRE(target, logfile=None)
    optimizer.attach(lambda: frames.append(atoms.get_scaled_positions().tolist()))
    converged = optimizer.run(fmax=fmax, steps=steps)

    output_dir = Path(output_dir) if output_dir else case_dir.parent / (case_dir.name + "_ml")
    output_dir.mkdir(parents=True, exist_ok=True)
    write(output_dir / "POSCAR", atoms, format="vasp", direct=True)
    _write_xdatcar(output_dir / "XDATCAR", atoms, frames)

    forces = atoms.get_forces()
    max_force = max((sum(f * f for f in row)) ** 0.5 for row in forces)
    return {
        "case_dir": str(output_dir),
        "energy_eV": float(atoms.get_potential_energy()),
        "max_force_eV_A": float(max_force),
        "steps": len(frames) - 1,
        "converged": bool(converged),
        "model": "checkpoint" if checkpoint else model,
    }


def ml_energy(
    poscar_path: Path,
    model: str = DEFAULT_ML_MODEL,
    task: str = DEFAULT_ML_TASK,
    checkpoint: str | None = None,
    calculator=None,
) -> dict:
    """Single-point MLIP energy/forces — cheap screening before any VASP run."""
    require_ase()
    from ase.io import read

    poscar_path = Path(poscar_path)
    if poscar_path.is_dir():
        poscar_path = poscar_path / "POSCAR"
    atoms = read(poscar_path, format="vasp")
    atoms.calc = calculator or get_ml_calculator(model, task=task, checkpoint=checkpoint)

    forces = atoms.get_forces()
    max_force = max((sum(f * f for f in row)) ** 0.5 for row in forces)
    return {
        "energy_eV": float(atoms.get_potential_energy()),
        "max_force_eV_A": float(max_force),
        "model": "checkpoint" if checkpoint else model,
    }
