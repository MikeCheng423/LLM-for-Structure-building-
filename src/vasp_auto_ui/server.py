"""vasp_auto_ui — local web UI for the vasp_auto engine.

A workflow-builder front end: structure builders, calculation forms, workflow
builder, run console, and results table. Standard library only (http.server);
it imports vasp_auto for structure/preview/parse operations and launches the
vasp-auto CLI as a subprocess for runs, so UI and CLI behave identically.

Binds to 127.0.0.1 — this is a single-user local tool, not a public server.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vasp_auto.calc_types import CALC_TYPE_INFO, CalcType
from vasp_auto.config_loader import load_config, merge_local_config
from vasp_auto.job_manager import load_incar_template, make_case_info, preview_job_from_case
from vasp_auto.kpoints import KPATH_PRESETS
from vasp_auto.structure import (
    add_adsorbate,
    add_interstitial,
    build_struct,
    cart_coords,
    cell_parameters,
    combine_structures,
    frac_coords,
    freeze_atoms,
    make_prototype,
    make_supercell,
    make_vacancy,
    match_supercells,
    move_atom,
    parse_atom_selection,
    per_atom_symbols,
    read_poscar,
    resolve_prototype,
    rotate_atoms,
    scale_cell,
    scaled_lattice,
    substitute,
    write_poscar,
)
from vasp_auto import py4vasp_tools
from vasp_auto.parser import (
    aggregate_pdos, aggregate_qe_pdos, parse_bands, parse_dos, parse_pdos,
    parse_qe_bands, parse_qe_dos,
)
from vasp_auto.report import build_job_report, write_job_report
from vasp_auto.runner import (
    fetch_remote_file,
    fetch_remote_results,
    list_remote_cases,
    list_remote_dir,
    list_remote_jobs,
    list_running_jobs,
    kill_detached_job,
    kill_job_by_dir,
    delete_remote_dir,
    clear_remote_terminated,
    poll_detached_job,
    poll_remote_job,
    read_remote_text,
    remote_command,
    resolve_detached_job_dir,
    resolve_remote_run_mode,
    setup_remote_engine,
    check_remote_connection,
)
from vasp_auto.target_utils import get_case_type, inspect_target
from vasp_auto.trajectory import job_trajectory, oszicar_energies
from vasp_auto.workflow import (
    _read_pid as read_pid,
    _neb_image_energy,
    build_row,
    job_engine,
    neb_energy_profile,
    parse_magmoms,
    read_remote_marker,
    scan_vasp_errors,
)

STATIC_DIR = Path(__file__).parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[2]
UI_LOG_DIR = REPO_ROOT / "ui_logs"
# UI-managed remote machines (writable). config.yaml remote:/remotes: are also
# surfaced (read-only) so CLI and UI share the same machines.
REMOTES_FILE = REPO_ROOT / "remotes.json"

# Fields a remote-machine config may carry (everything else is ignored).
# "cpus" is a per-machine default core count the Calculate-tab CPU field
# pre-fills from; the run still passes it through the usual -n argument.
REMOTE_FIELDS = ("host", "user", "port", "ssh_key", "remote_root",
                 "vasp_executable", "scheduler", "run_mode", "env_setup", "cpus",
                 "max_jobs", "potcar_root", "qe_executable", "pseudo_dir", "qe_env_setup")

EDITABLE_FILES = {"INCAR", "KPOINTS", "POSCAR", "workflow.yaml", "config.yaml"}


def _editable(name: str) -> bool:
    """Standard input files, plus the per-step INCAR_<type>/KPOINTS_<type> the
    workflow editor writes. `isalnum` on the suffix blocks path traversal."""
    if name in EDITABLE_FILES:
        return True
    base, sep, suffix = name.partition("_")
    return bool(sep) and base in ("INCAR", "KPOINTS") and suffix.isalnum()

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _config() -> dict:
    explicit = REPO_ROOT / "config.yaml"
    return load_config(str(explicit)) if explicit.exists() else load_config()


# ---------------------------------------------------------------- API helpers

def api_meta(_query, _body):
    from vasp_auto.ai_providers import DEFAULT_PROVIDER, provider_catalog
    from vasp_auto.ase_engine import ASE_CALCULATORS, ASE_SUPPORTED_CALC_TYPES
    from vasp_auto.ml_tools import ML_MODELS
    from vasp_auto.qe_tools import QE_ALL_CALC_TYPES, QE_COMPANION_PROGRAMS

    config = _config()
    return {
        "calc_types": [t.value for t in CalcType],
        "calc_type_info": {t.value: CALC_TYPE_INFO.get(t, "") for t in CalcType},
        "kpath_presets": sorted(KPATH_PRESETS),
        "ml_models": ML_MODELS,
        "ai_providers": provider_catalog(),
        "ai_provider_default": DEFAULT_PROVIDER,
        "ase_calculators": list(ASE_CALCULATORS),
        "ase_calc_types": list(ASE_SUPPORTED_CALC_TYPES),
        "qe_calc_types": list(QE_ALL_CALC_TYPES),
        "qe_companion_programs": list(QE_COMPANION_PROGRAMS),
        # UI default: the user's configured model if set, else 'emt' (always runs,
        # no download) — the gated UMA models would otherwise fail on first click.
        "ml_model_default": config.get("ml_model") or "emt",
        "repo_root": str(REPO_ROOT),
        "inputs_root": str(REPO_ROOT / "inputs"),
        "config": {
            "engine": config.get("engine", "vasp"),
            "vasp_executable": config.get("vasp_executable"),
            "qe_executable": config.get("qe_executable", "pw.x"),
            "pseudo_dir": config.get("pseudo_dir"),
            "jobs_root": config.get("jobs_root"),
            "potcar_root": config.get("potcar_root"),
            "scheduler": config.get("scheduler", "local"),
            "potcar_map": config.get("potcar_map") or {},
            "ase_calculator": config.get("ase_calculator") or "emt",
            "ase_command": (config.get("ase_calc_params") or {}).get("command"),
        },
    }


def api_cases(query, _body):
    machine = (query.get("machine", ["local"])[0] or "local").strip()
    if _is_remote_machine(machine):
        # Browse cases that physically live on the selected machine over SSH.
        remote = _resolve_remote(machine)
        root = query.get("path", [None])[0] or _default_remote_cases_dir(remote)
        listing = list_remote_cases(remote, root)
        cases = [{"name": c["name"], "path": c["path"], "type": c["type"],
                  "machine": machine, "remote": True} for c in listing["cases"]]
        return {"path": listing["path"], "machine": machine, "cases": cases}

    root = Path(query.get("path", [str(REPO_ROOT / "inputs")])[0]).expanduser().resolve()
    if not root.exists():
        return {"path": str(root), "cases": []}

    def entry(case_dir: Path) -> dict | None:
        marker = _remote_case_marker(case_dir)
        if marker:
            return {"name": case_dir.name, "path": str(case_dir),
                    "type": marker.get("type", "single"),
                    "machine": marker.get("machine"), "remote": True}
        case_type = get_case_type(case_dir)
        if case_type:
            return {"name": case_dir.name, "path": str(case_dir), "type": case_type}
        return None

    cases = []
    own = entry(root)
    if own:
        cases.append(own)
    else:
        for child in sorted(root.iterdir()):
            if child.is_dir():
                info = entry(child)
                if info:
                    cases.append(info)
    return {"path": str(root), "cases": cases}


def _struct_payload(struct: dict, poscar: Path | None = None) -> dict:
    """Full editor model of a structure: lattice (Å), per-atom symbols,
    Cartesian + fractional coordinates, selective-dynamics flags, cell params."""
    lattice = scaled_lattice(struct)
    symbols = per_atom_symbols(struct)
    return {
        "comment": struct["comment"],
        "poscar": str(poscar) if poscar else None,
        "lattice": lattice,
        "symbols": symbols,
        "cartesian": cart_coords(struct),
        "frac": frac_coords(struct),
        "selective": struct["selective"],
        "flags": [list(f) if f else ["T", "T", "T"] for f in struct["flags"]]
                 if struct["selective"] else [[] for _ in symbols],
        "cell": cell_parameters(lattice),
        "counts": dict(zip(struct["elements"], struct["counts"])),
        "natoms": len(symbols),
    }


def _struct_from_payload(data: dict) -> dict:
    """Inverse of _struct_payload for editor → engine round trips (fractional)."""
    flags = data.get("flags") if data.get("selective") else None
    return build_struct(
        data.get("comment") or "structure",
        data["lattice"],
        data["symbols"],
        data["frac"],
        cartesian=False,
        flags=flags,
    )


def _find_poscar(case_dir: Path) -> Path:
    poscar = case_dir if case_dir.is_file() else case_dir / "POSCAR"
    if not poscar.exists():
        poscar = case_dir / "initial" / "POSCAR"
    if not poscar.exists():
        raise FileNotFoundError(f"No POSCAR in {case_dir}")
    return poscar


# ------------------------------------------------------ working-machine cases
#
# The Working-case selector picks a machine. When it is a remote, every case
# operation (list/load/edit/build/preview/run) targets a path *on that machine*
# over SSH — its files and results live there, nothing is kept on this computer.
# `_remote_loc` resolves an operation to that machine, and also still honours an
# older local .remote_case.json pointer (the previous build-on-remote flow) so
# any case created that way keeps working.

REMOTE_CASE_MARKER = ".remote_case.json"


def _remote_case_marker(case_dir: Path) -> dict | None:
    """The remote-case pointer for a case dir, or None for an ordinary local case."""
    f = Path(case_dir) / REMOTE_CASE_MARKER
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _default_remote_cases_dir(remote: dict) -> str:
    """Default working-cases directory on a machine (its <remote_root>/inputs)."""
    root = (remote.get("remote_root") or "").rstrip("/")
    return f"{root}/inputs" if root else "/"


def _is_remote_machine(machine) -> bool:
    machine = (machine or "").strip()
    return bool(machine) and machine != "local"


def _remote_loc(path_str: str, machine, case_type=None) -> dict | None:
    """Resolve a working-case operation to a remote location, or None for local.

    With a remote working `machine` (the Working-case selector), `path_str` is a
    path *on that machine*, used directly. Otherwise a local case dir that holds a
    .remote_case.json pointer still resolves to its remote (the build-on-remote
    flow). Returns a marker-shaped {machine, remote_dir, case_name, type} or None.
    """
    if _is_remote_machine(machine):
        remote_dir = str(path_str).rstrip("/")
        return {"machine": machine.strip(), "remote_dir": remote_dir,
                "case_name": remote_dir.rsplit("/", 1)[-1] or remote_dir,
                "type": case_type or "single"}
    return _remote_case_marker(Path(path_str).expanduser().resolve())


def _ship_dir_to_remote(local_dir: Path, machine: str, remote_dir: str) -> str:
    """Push a local directory's contents to an explicit remote directory."""
    import shlex
    from vasp_auto.runner import _run_checked, _ssh_options, _ssh_target, _transfer_dir
    remote = _resolve_remote(machine)
    remote_dir = remote_dir.rstrip("/")
    target = _ssh_target(remote)
    ssh_opts = _ssh_options(remote)
    _run_checked(["ssh", "-x", *ssh_opts, target, f"mkdir -p {shlex.quote(remote_dir)}"],
                 "remote mkdir case")
    _transfer_dir(local_dir, target, remote_dir, remote)
    return remote_dir


def _finalize_built_case(result: dict, body: dict, ctype: str = "single") -> dict:
    """Common tail for the direct-write builders (NEB/TSS, prototype, Materials
    Project, …): when the working machine is a remote, ship the freshly built case
    straight onto it under the working cases folder and leave nothing on this
    computer; otherwise return the local result unchanged."""
    machine = (body.get("machine") or "").strip()
    if not _is_remote_machine(machine):
        # Local case: still report its type so the UI tracks NEB/TSS vs single.
        return {**result, "type": ctype, "machine": "local"}
    remote = _resolve_remote(machine)
    case_dir = Path(result["case"])
    base = (body.get("root") or _default_remote_cases_dir(remote)).rstrip("/")
    remote_case = f"{base}/{case_dir.name}"
    _ship_dir_to_remote(case_dir, machine, remote_case)
    rel_poscar = Path(result["poscar"]).relative_to(case_dir).as_posix()
    shutil.rmtree(case_dir, ignore_errors=True)  # nothing left on this computer
    return {**result, "case": remote_case, "machine": machine, "remote": True,
            "remote_dir": remote_case, "type": ctype,
            "poscar": f"{remote_case}/{rel_poscar}"}


def _fetch_remote_case(marker: dict, dest: Path, names=None) -> Path:
    """Pull a remote case's input files into dest; return dest. A single case
    needs POSCAR; a NEB/TSS case needs initial/POSCAR and final/POSCAR."""
    remote = _resolve_remote(marker["machine"])
    remote_dir = str(marker["remote_dir"]).rstrip("/")
    dest.mkdir(parents=True, exist_ok=True)
    if marker.get("type") == "tss":
        wanted = ["initial/POSCAR", "final/POSCAR", "INCAR", "KPOINTS"]
        required = {"initial/POSCAR", "final/POSCAR"}
    else:
        wanted = list(names or ["POSCAR", "INCAR", "KPOINTS", "workflow.yaml"])
        required = {"POSCAR"}
    got = set()
    for rel in wanted:
        local = dest / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            fetch_remote_file(remote, f"{remote_dir}/{rel}", local)
            got.add(rel)
        except Exception:
            if rel in required:
                raise
    missing = required - got
    if missing:
        raise FileNotFoundError(f"Remote case {remote_dir} is missing {sorted(missing)}")
    return dest


def _fetch_remote_target(remote_name: str, target: str, dest_root: Path) -> Path:
    """Pull a case, or a whole project of cases, that already lives on a remote
    machine into a local staging dir under dest_root — mirroring how
    inspect_target tells single vs. project apart locally, but over SSH via
    list_remote_cases. Returns the local path to hand to the CLI (one case dir,
    or a project dir holding several fetched case subdirs)."""
    remote = _resolve_remote(remote_name)
    target = target.rstrip("/")
    listing = list_remote_cases(remote, target)
    cases = listing["cases"]
    if not cases:
        raise FileNotFoundError(
            f"No VASP case found under {target} on {remote_name} "
            "(needs POSCAR, or subfolders that do)."
        )
    if len(cases) == 1 and cases[0]["path"].rstrip("/") == target:
        case = cases[0]
        dest = dest_root / (case["name"] or "case")
        dest.mkdir(parents=True)
        marker = {"machine": remote_name, "remote_dir": case["path"], "type": case["type"]}
        _fetch_remote_case(marker, dest)
        return dest

    project = dest_root / (Path(target).name or "project")
    project.mkdir(parents=True)
    for case in cases:
        marker = {"machine": remote_name, "remote_dir": case["path"], "type": case["type"]}
        _fetch_remote_case(marker, project / case["name"])
    return project


@contextmanager
def _local_case(target, machine=None):
    """Yield a local case dir for `target`. For a case that lives on a remote
    machine (either a remote working machine, or a local .remote_case.json
    pointer) this fetches its inputs into a temp dir (cleaned up on exit); a plain
    local case is yielded unchanged. Returns (case_dir, loc-or-None)."""
    loc = _remote_loc(str(target), machine)
    if not loc:
        yield Path(target).expanduser().resolve(), None
        return
    tmp = tempfile.mkdtemp(prefix="vasp_auto_rc_")
    try:
        staging = Path(tmp) / (loc.get("case_name") or "case")
        staging.mkdir(parents=True)
        _fetch_remote_case(loc, staging)
        yield staging, loc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def api_structure(query, _body):
    path_str = query["path"][0]
    machine = query.get("machine", ["local"])[0]
    loc = _remote_loc(path_str, machine, (query.get("type", [None])[0]))
    if loc:
        with tempfile.TemporaryDirectory(prefix="vasp_auto_rc_") as tmp:
            dest = _fetch_remote_case(loc, Path(tmp))
            return _struct_payload(read_poscar(_find_poscar(dest)),
                                   Path(loc["remote_dir"]) / "POSCAR")
    poscar = _find_poscar(Path(path_str).expanduser().resolve())
    return _struct_payload(read_poscar(poscar), poscar)


def _unique_case_dir(parent: Path, name: str) -> Path:
    """First free of parent/name, name_2, name_3 … — a name-based save never
    clobbers an existing case (overwriting needs an explicit absolute dir)."""
    case_dir, n = parent / name, 1
    while case_dir.exists():
        n += 1
        case_dir = parent / f"{name}_{n}"
    return case_dir


def _remote_unique_case(remote: dict, base: str, name: str) -> str:
    """Remote twin of _unique_case_dir: one SSH round trip finds the first
    free folder name. On SSH trouble fall back to base/name unchanged (the
    later rsync will surface the real error)."""
    script = (f'd="{base}/{name}"; i=1; while [ -e "$d" ]; do i=$((i+1)); '
              f'd="{base}/{name}_$i"; done; echo "$d"')
    try:
        res = remote_command(remote, script, timeout=30)
        lines = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
        if res.returncode == 0 and lines:
            return lines[-1]
    except Exception:
        pass
    return f"{base}/{name}"


def api_structure_save(_query, body):
    """Save the editor's working structure as a case POSCAR (the explicit
    Save button, and the auto-save a Run performs on an unsaved structure —
    building/editing alone never writes case folders)."""
    struct = _struct_from_payload(body["structure"])
    machine = (body.get("machine") or "").strip()
    if _is_remote_machine(machine):
        # Working machine is a remote: write the case straight onto it (under the
        # working cases folder) so the structure and every later calculation live
        # on that machine — nothing is kept on this computer.
        remote = _resolve_remote(machine)
        if body.get("dir") and str(body["dir"]).startswith("/"):
            remote_case = str(body["dir"]).rstrip("/")
        else:
            name = body.get("name")
            if not name:
                raise ValueError("Give a case name to save")
            base = (body.get("root") or _default_remote_cases_dir(remote)).rstrip("/")
            remote_case = _remote_unique_case(remote, base, Path(str(name)).name)
        with tempfile.TemporaryDirectory(prefix="vasp_auto_rc_") as tmp:
            write_poscar(struct, Path(tmp) / "POSCAR")
            _ship_dir_to_remote(Path(tmp), machine, remote_case)
        return {"case": remote_case, "poscar": f"{remote_case}/POSCAR",
                "machine": machine, "remote_dir": remote_case, "remote": True}

    # An absolute dir is used as-is; otherwise a name lands under the chosen
    # location (`root`, picked with the 📁 browser) or the repo's inputs/ folder.
    if body.get("dir") and Path(str(body["dir"])).expanduser().is_absolute():
        case_dir = Path(str(body["dir"])).expanduser()
    else:
        name = body.get("name") or body.get("dir")
        if not name:
            raise ValueError("Give a case name or directory to save into")
        name = Path(str(name)).name  # a bare folder name, even if a path was typed
        root = Path(str(body["root"])).expanduser() if body.get("root") else REPO_ROOT / "inputs"
        case_dir = _unique_case_dir(root, name)
    write_poscar(struct, case_dir / "POSCAR")
    return {"case": str(case_dir), "poscar": str(case_dir / "POSCAR")}


def api_nl_build(_query, body):
    """AI builder chatbox: free text -> AI API -> JSON command -> ASE-built
    structure for the editor. Nothing is written. The endpoint, key and model
    come from the request (`provider`/`base_url`/`api_key`/`model`, set in the
    UI) or the provider's env var — any OpenAI-compatible API works."""
    from vasp_auto.nl_builder import build_from_text, describe_command

    cmd, struct = build_from_text(
        str(body.get("text", "")),
        api_key=(body.get("api_key") or None),
        provider=(body.get("provider") or None),
        base_url=(body.get("base_url") or None),
        model=(body.get("model") or None),
    )
    return {
        "command": cmd,
        "summary": describe_command(cmd),
        "structure": _struct_payload(struct),
    }


def api_nl_agent(_query, body):
    """AI builder, agentic mode: free text -> tool-calling worker that composes
    the structure primitives step by step. Returns the finished structure plus
    the transcript of tool calls. Nothing is written. `provider`/`base_url`/
    `model` pick the AI API and model (any OpenAI-compatible function-calling
    endpoint; needs a tool-capable model)."""
    from vasp_auto.nl_agent import agent_build_from_text

    struct, transcript = agent_build_from_text(
        str(body.get("text", "")),
        api_key=(body.get("api_key") or None),
        model=(body.get("model") or None),
        provider=(body.get("provider") or None),
        base_url=(body.get("base_url") or None),
    )
    return {"transcript": transcript, "structure": _struct_payload(struct)}


def _load_struct_arg(body: dict, key: str) -> dict:
    """A structure argument for combine/match: an inline editor payload
    (`<key>_struct`), a local case path/POSCAR, or — with a remote working
    `machine` in the body — a path on that machine, fetched over SSH."""
    inline = body.get(f"{key}_struct")
    if inline:
        return _struct_from_payload(inline)
    path = body.get(key)
    if not path:
        raise ValueError(f"Missing {key} structure")
    loc = _remote_loc(str(path), body.get("machine"))
    if loc:
        remote = _resolve_remote(loc["machine"])
        base = str(loc["remote_dir"]).rstrip("/")
        with tempfile.TemporaryDirectory(prefix="vasp_auto_rc_") as tmp:
            local = Path(tmp) / "POSCAR"
            try:
                fetch_remote_file(remote, f"{base}/POSCAR", local)
            except Exception:
                # The path may be the structure file itself, not a case folder.
                fetch_remote_file(remote, base, local)
            return read_poscar(local)
    return read_poscar(_find_poscar(Path(str(path)).expanduser().resolve()))


def api_combine(_query, body):
    """Combine two structures with different unit cells (stack/insert).

    Host and guest are case paths (on the working machine when it is a
    remote); either can instead be sent inline as `host_struct` /
    `guest_struct` (the editor's unsaved working structure).
    Returns the combined structure for the editor — nothing is written.
    """
    host, guest = _load_struct_arg(body, "host"), _load_struct_arg(body, "guest")
    # Optional in-plane supercells (from the cell-match suggestions) applied
    # before stacking, e.g. 9x9 graphene under a 4x4 TiO2 slab.
    if body.get("host_repeat"):
        i, j = (int(n) for n in body["host_repeat"][:2])
        host = make_supercell(host, (i, j, 1))
    if body.get("guest_repeat"):
        k, l = (int(n) for n in body["guest_repeat"][:2])
        guest = make_supercell(guest, (k, l, 1))

    shift = body.get("shift") or [0.0, 0.0]
    combined = combine_structures(
        host,
        guest,
        mode=body.get("mode") or "stack",
        gap=float(body.get("gap") if body.get("gap") is not None else 2.0),
        vacuum=float(body.get("vacuum") if body.get("vacuum") is not None else 10.0),
        shift=(float(shift[0]), float(shift[1])),
        strain_guest=bool(body.get("strain")),
    )
    return {"structure": _struct_payload(combined)}


def api_molecule(_query, body):
    """Adsorbate geometry for the Build-tab quick build: an ASE molecule
    (CO2, H2O, NH3, …) or a single atom, with the lowest atom at z=0. The editor
    places these atoms above the chosen surface atom."""
    from vasp_auto.ase_tools import molecule_positions
    symbols, offsets = molecule_positions(str(body["name"]).strip())
    return {"symbols": symbols, "offsets": offsets}


def _output_dir(body, default: Path) -> Path:
    output = body.get("output")
    if output:
        path = Path(output).expanduser()
        return path if path.is_absolute() else REPO_ROOT / "inputs" / path
    return default


def api_build(_query, body):
    action = body["action"]
    inputs = REPO_ROOT / "inputs"

    if body.get("to_editor") and action not in ("tss", "prototype", "mp", "import"):
        # Build into a throw-away directory and hand the structure to the
        # editor instead of creating a case (cases are made by Save only). Force
        # a local build here — the eventual Save is what ships it to a remote.
        # "import" handles to_editor itself: its `machine` names where the source
        # file lives (possibly a remote to fetch from), not a build destination.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="vasp_auto_build_") as tmp:
            result = api_build(_query, {**body, "to_editor": False, "machine": "local",
                                        "output": str(Path(tmp) / "editor_build")})
            return {"structure": _struct_payload(read_poscar(Path(result["poscar"])))}

    if action == "mp":
        # Materials Project prototype: fetch a structure by material_id or
        # formula, optionally substitute elements, and hand to the editor.
        from vasp_auto.ml_tools import prototype_from_mp

        subs = body.get("substitutions") or {}
        struct = prototype_from_mp(
            body["query"],
            substitutions=subs or None,
            api_key=body.get("api_key"),
        )
        if body.get("to_editor"):
            return {"structure": _struct_payload(struct)}
        safe = str(body["query"]).replace("/", "_")
        if subs:
            safe += "_" + "".join(f"{k}{v}" for k, v in subs.items())
        case_dir = _output_dir(body, inputs / safe)
        write_poscar(struct, case_dir / "POSCAR")
        return _finalize_built_case(
            {"case": str(case_dir), "poscar": str(case_dir / "POSCAR")}, body)

    if action == "prototype":
        # Pure-Python prototype crystals (graphene, graphite, rutile/anatase
        # TiO2, hBN) — no ASE needed.
        name = resolve_prototype(body["name"])
        struct = make_prototype(
            name,
            a=float(body["a"]) if body.get("a") else None,
            c=float(body["c"]) if body.get("c") else None,
            vacuum=float(body["vacuum"]) if body.get("vacuum") else None,
        )
        if body.get("to_editor"):
            return {"structure": _struct_payload(struct)}
        case_dir = _output_dir(body, inputs / name)
        write_poscar(struct, case_dir / "POSCAR")
        return _finalize_built_case(
            {"case": str(case_dir), "poscar": str(case_dir / "POSCAR")}, body)

    if action == "bulk":
        from vasp_auto.ase_tools import build_bulk_case
        symbol = body["symbol"]
        poscar = build_bulk_case(
            symbol=symbol,
            case_dir=_output_dir(body, inputs / symbol),
            crystalstructure=body.get("crystalstructure") or None,
            a=body.get("a") or None,
            c=body.get("c") or None,
            cubic=bool(body.get("cubic")),
        )
    elif action == "slab":
        from vasp_auto.ase_tools import build_slab_case
        miller = tuple(int(i) for i in body.get("miller", [1, 1, 1]))
        name = f"{body['source']}_slab" + "".join(str(abs(i)) for i in miller)
        poscar = build_slab_case(
            source=body["source"],
            case_dir=_output_dir(body, inputs / name),
            miller=miller,
            layers=int(body.get("layers") or 4),
            vacuum=float(body.get("vacuum") or 12.0),
            crystalstructure=body.get("crystalstructure") or None,
            a=body.get("a") or None,
            repeat=tuple(body["repeat"]) if body.get("repeat") else None,
        )
    elif action == "molecule":
        from vasp_auto.ase_tools import build_molecule_case
        poscar = build_molecule_case(
            body["name"],
            _output_dir(body, inputs / body["name"]),
            box=float(body.get("box") or 12.0),
        )
    elif action == "crystal":
        from vasp_auto.ase_tools import build_crystal_case
        symbols = body["symbols"]
        if isinstance(symbols, str):
            symbols = symbols.replace(",", " ").split()
        name = "".join(symbols) + f"_sg{int(body['spacegroup'])}"
        poscar = build_crystal_case(
            symbols=symbols,
            basis=[tuple(float(x) for x in site) for site in body["basis"]],
            spacegroup=int(body["spacegroup"]),
            case_dir=_output_dir(body, inputs / name),
            a=float(body["a"]),
            b=float(body["b"]) if body.get("b") else None,
            c=float(body["c"]) if body.get("c") else None,
            alpha=float(body.get("alpha") or 90.0),
            beta=float(body.get("beta") or 90.0),
            gamma=float(body.get("gamma") or 90.0),
        )
    elif action == "nanotube":
        from vasp_auto.ase_tools import build_nanotube_case
        symbol = body.get("symbol") or "C"
        n, m = int(body["n"]), int(body["m"])
        poscar = build_nanotube_case(
            symbol=symbol,
            n=n,
            m=m,
            case_dir=_output_dir(body, inputs / f"{symbol}_nt{n}{m}"),
            length=int(body.get("length") or 1),
            bond=float(body["bond"]) if body.get("bond") else None,
            vacuum=float(body.get("vacuum") or 10.0),
        )
    elif action == "import":
        from vasp_auto.ase_tools import import_structure_to_case
        import shutil
        import tempfile
        machine = (body.get("machine") or "").strip()
        fmt = body.get("format") or None
        rtmp = None
        try:
            if machine and machine != "local":
                # Pull the structure file off a remote machine over SSH into a
                # local temp file. Editing always happens in the local engine;
                # the edited structure re-ships to the working machine on Save.
                from vasp_auto.runner import fetch_remote_file
                remote = _resolve_remote(machine)
                remote_path = str(body["source"])
                rtmp = tempfile.mkdtemp(prefix="vasp_auto_import_")
                source = Path(fetch_remote_file(
                    remote, remote_path, Path(rtmp) / Path(remote_path).name))
            else:
                source = Path(body["source"]).expanduser()
            if body.get("to_editor"):
                # Convert into a throw-away POSCAR and hand it to the editor; no
                # case is created (Save does that, shipping to the working machine).
                with tempfile.TemporaryDirectory(prefix="vasp_auto_build_") as tmp:
                    poscar = import_structure_to_case(
                        structure_path=source, case_dir=Path(tmp) / "editor_build",
                        input_format=fmt)
                    return {"structure": _struct_payload(read_poscar(poscar))}
            poscar = import_structure_to_case(
                structure_path=source,
                case_dir=_output_dir(body, inputs / source.stem),
                input_format=fmt,
            )
        finally:
            if rtmp:
                shutil.rmtree(rtmp, ignore_errors=True)
    elif action == "tss":
        case_dir = _output_dir(body, inputs / (body.get("output") or "neb_case"))
        for endpoint, source_text in (("initial", body["initial"]), ("final", body["final"])):
            source = Path(source_text).expanduser().resolve()
            poscar_src = source if source.is_file() else source / "POSCAR"
            if not poscar_src.exists():
                raise FileNotFoundError(f"No POSCAR for the {endpoint} state: {source}")
            target = case_dir / endpoint / "POSCAR"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(poscar_src.read_bytes())
        case_dir = case_dir.resolve()
        return _finalize_built_case(
            {"case": str(case_dir), "poscar": str(case_dir / "initial" / "POSCAR")},
            body, ctype="tss")
    elif action == "edit":
        source = Path(body["source"]).expanduser().resolve()
        struct = read_poscar(source / "POSCAR")
        suffix = ""
        if body.get("supercell"):
            repeat = tuple(int(n) for n in body["supercell"])
            struct = make_supercell(struct, repeat)
            suffix += "_sc" + "x".join(str(n) for n in repeat)
        if body.get("vacancy"):
            struct = make_vacancy(struct, int(body["vacancy"]))
            suffix += f"_vac{body['vacancy']}"
        if body.get("substitute"):
            index, element = body["substitute"]
            struct = substitute(struct, int(index), str(element))
            suffix += f"_sub{index}{element}"
        if body.get("interstitial"):
            element, position = body["interstitial"]
            struct = add_interstitial(struct, str(element), tuple(float(x) for x in position))
            suffix += f"_int{element}"
        if body.get("adsorbate"):
            element, anchor, height = body["adsorbate"]
            struct = add_adsorbate(struct, str(element), int(anchor), float(height))
            suffix += f"_ads{element}{anchor}"
        if body.get("move_atom"):
            index, vector, absolute = body["move_atom"]
            struct = move_atom(struct, int(index), tuple(float(x) for x in vector),
                               absolute=bool(absolute))
            suffix += f"_mv{index}"
        if body.get("scale_cell"):
            from vasp_auto.cli import _parse_cell_spec
            struct = scale_cell(struct, _parse_cell_spec(struct, str(body["scale_cell"])))
            suffix += "_cell"
        if body.get("rotate"):
            selection, axis, angle = body["rotate"]
            indices = parse_atom_selection(struct, str(selection))
            struct = rotate_atoms(struct, indices, axis, float(angle))
            suffix += "_rot"
        if body.get("freeze"):
            selection, axes = body["freeze"]
            indices = parse_atom_selection(struct, str(selection))
            struct = freeze_atoms(struct, indices, axes=str(axes or "XYZ"))
            suffix += "_frz"
        if not suffix:
            raise ValueError("No structure edit requested")
        case_dir = _output_dir(body, source.parent / (source.name + suffix))
        write_poscar(struct, case_dir / "POSCAR")
        poscar = case_dir / "POSCAR"
    else:
        raise ValueError(f"Unknown build action: {action}")

    return _finalize_built_case({"case": str(poscar.parent), "poscar": str(poscar)}, body)


def _case_info_for(target: Path, config: dict):
    info = inspect_target(target)
    # Jobs live directly under the jobs root (jobs/NNNN_<case>), no project
    # sub-folder; the numbered run is resolved by make_case_info's "latest" mode.
    output_root = Path(config["jobs_root"])
    case_infos = [
        make_case_info(case_dir, output_root, single_mode=(info["mode"] == "single"),
                       job_mode="latest")
        for case_dir in info["case_dirs"]
    ]
    return info, case_infos


def _job_has_output(job_dir: Path) -> bool:
    """True if a job directory holds run output worth pointing the viewers at."""
    if not job_dir.is_dir():
        return False
    for name in ("OUTCAR", "vasprun.xml", "OSZICAR", "run.log", "CONTCAR"):
        if (job_dir / name).exists():
            return True
    # NEB/TSS keep output in numbered image dirs; convergence scans in a subdir.
    try:
        return any(
            p.is_dir() and (p.name.isdigit() or p.name == "scf_convergence")
            for p in job_dir.iterdir()
        )
    except OSError:
        return False


def _is_neb_job(job_dir: Path) -> bool:
    if (job_dir / "initial" / "POSCAR").exists() and (job_dir / "final" / "POSCAR").exists():
        return True
    return sum(1 for p in job_dir.iterdir() if p.is_dir() and p.name.isdigit()) >= 2


def _is_convergence_job(job_dir: Path) -> bool:
    """A convergence scan leaves its trials under a scf_convergence/ subdirectory."""
    return (job_dir / "scf_convergence").is_dir()


def _result_calc_type(job_dir: Path) -> str:
    if _is_neb_job(job_dir):
        return "tss"
    if _is_convergence_job(job_dir):
        return "convergence"
    return "scf"


def _job_dir_case_info(job_dir: Path) -> dict:
    """A minimal case_info built straight from a finished job directory, so the
    Results table and every per-row button operate on the real VASP output dir."""
    job_dir = Path(job_dir)
    return {
        "case_name": job_dir.name,
        "case_dir": job_dir,
        "job_dir": job_dir,
        "calculation_type": _result_calc_type(job_dir),
        "single_mode": True,
    }


def _scan_result_jobs(root: Path) -> list[Path]:
    """Find the actual VASP job directories under a results folder.

    Handles both layouts: flat ``<jobs_root>/<case>`` and nested
    ``<jobs_root>/<project>/<case>`` (descends one level into folders that are
    not jobs themselves). Returns the directories that hold real output.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    if _job_has_output(root):
        return [root]
    jobs: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if _job_has_output(child):
            jobs.append(child)
        else:
            # A project folder that itself contains job directories.
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and _job_has_output(sub):
                    jobs.append(sub)
    return jobs


def _result_case_infos(info: dict, config: dict):
    """Resolve each case to the job directory it was actually run in.

    Every job lives directly under the jobs root as ``<jobs_root>/<NNNN>_<case>``
    (one global number list per machine; no project sub-folder). ``"latest"``
    picks the highest-numbered run, falling back to a legacy bare ``<case>`` dir.
    """
    jobs_root = Path(config["jobs_root"])
    case_infos = [
        make_case_info(case_dir, jobs_root, single_mode=False, job_mode="latest")
        for case_dir in info["case_dirs"]
    ]
    return info, case_infos


def _overlay_ase_config(config: dict, body: dict) -> dict:
    """Merge the UI's ASE-engine choices (calculator, command path, extra params)
    onto a config dict — the same overlay the CLI's resolve_engine does, so a
    dry-run preview matches the eventual run."""
    overlay = dict(config)
    if body.get("ase_calculator"):
        overlay["ase_calculator"] = str(body["ase_calculator"])
    if body.get("ase_fmax"):
        overlay["ase_fmax"] = float(body["ase_fmax"])
    if body.get("ase_steps"):
        overlay["ase_steps"] = int(body["ase_steps"])
    params = dict(config.get("ase_calc_params") or {})
    extra = body.get("ase_params")
    if isinstance(extra, str) and extra.strip():
        extra = json.loads(extra)
    if isinstance(extra, dict):
        params.update(extra)
    if body.get("ase_command"):
        params["command"] = str(body["ase_command"])
    if params:
        overlay["ase_calc_params"] = params
    return overlay


def api_preview(_query, body):
    with _local_case(body["target"], body.get("machine")) as (target, _loc):
        return _preview_for_target(target, body)


def _preview_for_target(target: Path, body):
    config = merge_local_config(_config(), target)
    _info, case_infos = _case_info_for(target, config)

    engine = body.get("engine") or config.get("engine", "vasp")
    if body.get("pseudo_dir"):
        config = {**config, "pseudo_dir": body["pseudo_dir"]}
    if engine == "ase":
        config = _overlay_ase_config(config, body)
    kpoints = body.get("kpoints") or None
    previews = [
        preview_job_from_case(
            case_info,
            potcar_root=config.get("potcar_root"),
            potcar_map=config.get("potcar_map"),
            calc_type=body.get("calc_type") or None,
            kpoints_spec=kpoints,
            spin=bool(body.get("spin")),
            magmom_map=config.get("magmom_map"),
            engine=engine,
            config=config,
        )
        for case_info in case_infos
    ]
    return {"previews": previews}


def _job_mtime(job_dir: Path) -> float | None:
    """Most recent modification time of a job's output (for sorting by date)."""
    job_dir = Path(job_dir)
    times: list[float] = []
    for name in ("OUTCAR", "vasprun.xml", "OSZICAR", "run.log", "CONTCAR"):
        f = job_dir / name
        try:
            if f.exists():
                times.append(f.stat().st_mtime)
        except OSError:
            pass
    if not times:  # convergence/NEB keep output in subdirs
        try:
            times = [p.stat().st_mtime for p in job_dir.iterdir() if p.is_dir()]
        except OSError:
            times = []
    if not times:
        try:
            times = [job_dir.stat().st_mtime]
        except OSError:
            return None
    return max(times) if times else None


def _result_row(project: str, mode: str, case_info: dict) -> dict:
    job_dir = Path(case_info["job_dir"])
    row = build_row(project, mode, case_info)
    findings = scan_vasp_errors(job_dir)
    if findings:
        row["errors"] = "; ".join(f"{f['code']}: {f['hint']}" for f in findings)
    ts = _job_mtime(job_dir)
    if ts is not None:
        row["modified_ts"] = ts
        row["modified"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    return row


def _first_existing_excel(*paths) -> str | None:
    for path in paths:
        if path and Path(path).exists():
            return str(path)
    return None


def api_results(query, _body):
    target = Path(query["target"][0]).expanduser().resolve()
    config = merge_local_config(_config(), target)
    jobs_root = Path(config["jobs_root"]).resolve()

    # --- Results-folder mode --------------------------------------------------
    # When the target is the jobs/results folder (or any folder holding finished
    # job dirs rather than input cases), build rows straight from the real output
    # directories. This guarantees every row — and every per-row button — points
    # at the folder VASP actually wrote to, including jobs run under a name that
    # no longer matches an inputs case.
    is_jobs_location = target == jobs_root or target.is_relative_to(jobs_root)
    if not is_jobs_location:
        try:
            inspect_target(target)  # a valid inputs project/case → use inputs mode
        except (FileNotFoundError, ValueError):
            is_jobs_location = bool(_scan_result_jobs(target))

    if is_jobs_location:
        job_dirs = _scan_result_jobs(target)
        rows = []
        for job_dir in job_dirs:
            project = job_dir.parent.name if job_dir.parent != jobs_root else jobs_root.name
            rows.append(_result_row(project, "project", _job_dir_case_info(job_dir)))
        excel = None
        if target.is_dir():
            xlsx = sorted(target.glob("*.xlsx"))
            excel = str(xlsx[0]) if xlsx else None
        return {"project": target.name, "rows": rows, "excel": excel}

    # --- Inputs mode (linked view) -------------------------------------------
    info, case_infos = _result_case_infos(inspect_target(target), config)
    rows = [_result_row(info["project_name"], info["mode"], ci) for ci in case_infos]

    first = case_infos[0] if case_infos else None
    excel = _first_existing_excel(
        Path(first["job_dir"]) / f"{first['case_name']}.xlsx" if first else None,
        jobs_root / f"{info['project_name']}.xlsx",
    )
    return {"project": info["project_name"], "rows": rows, "excel": excel}


def api_template(query, _body):
    return {"text": load_incar_template(query["type"][0])}


def api_incar_options(query, _body):
    """The human-language INCAR catalog for the guided form.

    Filtered to the requested calculation type (``?type=dos``): a single-point
    run drops the ionic-movement settings, a DOS run gains its own fields.
    """
    from vasp_auto.incar_catalog import fields_by_group
    # Default to the conservative single-point set, not the whole catalog, so a
    # caller that forgets ?type= never shows ionic-movement fields in an SCF run.
    calc_type = (query.get("type") or ["scf"])[0]
    groups = {
        group: [
            {
                "tag": f.tag, "label": f.label, "kind": f.kind,
                "default": f.default, "unit": f.unit, "doc": f.doc,
                "options": [
                    {"label": o.label, "value": o.value, "hint": o.hint}
                    for o in f.options
                ],
            }
            for f in fields
        ]
        for group, fields in fields_by_group(calc_type).items()
    }
    return {"groups": groups}


def api_incar_apply(_query, body):
    """Apply the form's picks onto the current INCAR text and return the result."""
    from vasp_auto.incar import apply_incar_choices
    text = body.get("text") or ""
    choices = body.get("choices") or {}
    return {"text": apply_incar_choices(text, choices)}


# Local cache of remote job output, so the analysis viewers (DOS, PDOS,
# bands, volume, animation, report) work on remote results exactly like on
# local ones. One folder per remote job, keyed by machine + remote path.
REMOTE_RESULT_CACHE = UI_LOG_DIR / "remote_results"


def _analysis_dir(path: str, machine: str | None = None,
                  heavy: tuple[str, ...] = ()) -> Path:
    """Resolve an analysis request's job dir to a local directory.

    Local machine: the path itself. Remote machine: rsync the job's output
    (minus HEAVY_OUTPUTS) into the cache copy and analyze that. Every call
    re-syncs, so a still-running job shows fresh data; rsync makes the
    refresh incremental. ``heavy`` names files excluded from that sync
    (CHGCAR, LOCPOT, AECCAR…) which this request needs — each is pulled once
    and reused; one missing on the remote is skipped so the caller raises
    its own clear error.
    """
    if not _is_remote_machine(machine):
        return Path(path).expanduser().resolve()
    remote = _resolve_remote(machine)
    key = hashlib.sha1(f"{machine}:{path}".encode()).hexdigest()[:10]
    cache = REMOTE_RESULT_CACHE / machine / f"{Path(path.rstrip('/')).name}-{key}"
    fetch_remote_results(remote, path, str(cache))
    for name in heavy:
        if not (cache / name).exists():
            try:
                fetch_remote_file(remote, f"{path.rstrip('/')}/{name}", cache / name)
            except Exception:
                pass
    return cache


def _analysis_query_dir(query, heavy: tuple[str, ...] = ()) -> Path:
    """_analysis_dir for GET endpoints taking ?path=…[&machine=…]."""
    return _analysis_dir(query["path"][0], query.get("machine", ["local"])[0], heavy)


def api_trajectory(query, _body):
    job_dir = _analysis_query_dir(query)
    traj = job_trajectory(job_dir)
    if traj is None:
        raise FileNotFoundError(
            "No trajectory found — a relaxation needs XDATCAR (or POSCAR+CONTCAR), "
            "an NEB job needs image directories 00…NN."
        )
    if traj["kind"] == "relax":
        # Per-frame energies for the animation label: vaspout.h5 via py4vasp
        # when present, else OSZICAR. One XDATCAR frame per ionic step, except
        # that IBRION=2's final line-search re-evaluation adds one trailing
        # energy without a frame — allow that surplus, first-aligned. A larger
        # mismatch (e.g. a resumed job's partial OSZICAR) attaches nothing.
        energies = py4vasp_tools.step_energies(job_dir) or oszicar_energies(job_dir)
        n = len(traj["frames"])
        if energies and 0 <= len(energies) - n <= 1:
            traj["energies"] = energies[:n]
    return traj


def api_neb(query, _body):
    """Energy profile (reaction-coordinate plot) for an NEB/TSS job.

    Optional ``einit``/``efinal`` (with ``einit_m``/``efinal_m`` machines) name
    the endpoint relaxation runs; their energies fill in images 00/NN, which a
    standard NEB run never computes."""
    job_dir = _analysis_query_dir(query)
    endpoints = []
    for key in ("einit", "efinal"):
        path = (query.get(key, [""])[0] or "").strip()
        if not path:
            endpoints.append(None)
            continue
        run_dir = _analysis_dir(path, query.get(key + "_m", ["local"])[0])
        if run_dir.is_file():
            run_dir = run_dir.parent
        energy = _neb_image_energy(run_dir)
        if energy is None:
            raise FileNotFoundError(
                f"No energy (OUTCAR/OSZICAR/vasprun.xml) in the endpoint run: {path}")
        endpoints.append(energy)
    profile = neb_energy_profile(
        job_dir, endpoint_energies=tuple(endpoints) if any(e is not None for e in endpoints) else None)
    if profile is None:
        raise FileNotFoundError(
            "No NEB energy profile — a TSS/NEB job needs at least two image "
            "directories (00, 01, … NN) with energies (OUTCAR/OSZICAR/vasprun.xml)."
        )
    return profile


# Remote QE jobs exclude tmp/ from the rsync; the eigenvalue XML is pulled on
# demand as a "heavy" file (harmless miss for VASP/local jobs).
QE_SAVE_XML = "tmp/vasp_auto.save/data-file-schema.xml"


def api_dos(query, _body):
    job_dir = _analysis_query_dir(query, heavy=(QE_SAVE_XML,))
    if job_engine(job_dir) == "qe":
        dos = parse_qe_dos(job_dir)
        if dos is None:
            raise FileNotFoundError(
                "No QE eigenvalues in this job — run a 'dos'/'nscf' calculation "
                "(tmp/<prefix>.save/data-file-schema.xml must exist)."
            )
        return dos
    dos = py4vasp_tools.dos(job_dir) or parse_dos(job_dir / "vasprun.xml")
    if dos is None:
        raise FileNotFoundError(
            "No DOS in this job — run a 'dos' calculation (vasprun.xml must contain a DOS block)."
        )
    return dos


def api_pdos(query, _body):
    """Projected DOS aggregated to per-element s/p/d curves.

    Optional `atoms` query restricts to a selection ("1-4", "z>0.5", ...)
    resolved against the job POSCAR.
    """
    job_dir = _analysis_query_dir(query)
    struct = read_poscar(job_dir / "POSCAR")
    symbols = per_atom_symbols(struct)
    atoms = None
    selection = (query.get("atoms") or [""])[0].strip()
    if selection:
        atoms = parse_atom_selection(struct, selection)
    if job_engine(job_dir) == "qe":
        result = aggregate_qe_pdos(job_dir, symbols, atoms=atoms)
        if result is None:
            raise FileNotFoundError("No QE projected DOS in this job; run a dos calculation.")
        result["selection"] = selection or None
        result["natoms"] = len(symbols)
        return result
    pdos = parse_pdos(job_dir / "vasprun.xml")
    if pdos is None:
        raise FileNotFoundError(
            "No projected DOS in this job — run a 'dos' calculation (LORBIT=11 "
            "is set by the dos template; vasprun.xml must contain a partial DOS)."
        )
    result = aggregate_pdos(pdos, symbols, atoms=atoms)
    result["selection"] = selection or None
    result["natoms"] = len(symbols)
    return result


def api_bands(query, _body):
    job_dir = _analysis_query_dir(query, heavy=(QE_SAVE_XML,))
    if job_engine(job_dir) == "qe":
        bands = parse_qe_bands(job_dir)
        if bands is None:
            raise FileNotFoundError(
                "No QE band eigenvalues in this job — run a 'bands' calculation "
                "(a k-path) there first."
            )
        return bands
    bands = parse_bands(job_dir / "vasprun.xml", job_dir / "KPOINTS")
    if bands is None:
        raise FileNotFoundError(
            "No eigenvalues in this job — run a 'bands' calculation "
            "(line-mode KPOINTS) there first."
        )
    return bands


# Volumetric files the /api/volume endpoint may open, and how to label them.
VOLUME_FILES = {
    "CHGCAR": "charge density", "CHGCAR_diff": "charge-density difference",
    "CHGCAR_sum": "all-electron density", "LOCPOT": "local potential",
    "AECCAR0": "core density", "AECCAR2": "valence density", "PARCHG": "partial charge",
}
MAX_SLICE_POINTS = 160  # per direction, keeps the JSON payload bounded


def _volume_payload(volume: dict, file_name: str, axis: int, fraction: float) -> dict:
    from vasp_auto.chgcar import cell_volume_of, lattice_of, planar_average, slice_volume

    lattice = lattice_of(volume)
    lengths = [sum(x * x for x in row) ** 0.5 for row in lattice]
    # CHGCAR-family grids store rho*V_cell; LOCPOT stores eV directly.
    is_potential = file_name.upper().startswith("LOCPOT")
    factor = 1.0 if is_potential else 1.0 / cell_volume_of(volume)

    profile = [value * factor for value in planar_average(volume, axis=axis)]
    coords = [i * lengths[axis] / len(profile) for i in range(len(profile))]

    plane = slice_volume(volume, axis=axis, fraction=fraction)
    n1, n2 = plane["shape"]
    stride1 = max(1, -(-n1 // MAX_SLICE_POINTS))
    stride2 = max(1, -(-n2 // MAX_SLICE_POINTS))
    rows = [
        [plane["data"][j][i] * factor for i in range(0, n1, stride1)]
        for j in range(0, n2, stride2)
    ]
    axis1, axis2 = plane["axes"]
    return {
        "file": file_name,
        "kind": VOLUME_FILES.get(file_name, file_name),
        "unit": "eV" if is_potential else "e/Å³",
        "grid": list(volume["grid"]),
        "axis": axis,
        "profile": profile,
        "profile_coords_A": coords,
        "slice": {
            "data": rows,
            "position": plane["position"],
            "extent_A": [lengths[axis1], lengths[axis2]],
        },
    }


def api_volume(query, _body):
    """Planar average + one slice of a volumetric file (CHGCAR/LOCPOT/...)."""
    file_name = (query.get("file") or ["CHGCAR"])[0]
    if file_name not in VOLUME_FILES:
        raise ValueError(f"Not a known volumetric file: {file_name}")
    job_dir = _analysis_query_dir(query, heavy=(file_name,))
    axis = "abc".index((query.get("axis") or ["c"])[0].lower())
    fraction = float((query.get("frac") or ["0.5"])[0])

    path = job_dir / file_name
    if not path.exists():
        available = [name for name in VOLUME_FILES if (job_dir / name).exists()]
        raise FileNotFoundError(
            f"No {file_name} in this job"
            + (f" — available: {', '.join(available)}" if available else
               " — run with LCHARG (charge type) or LVHAR (workfunction type) first.")
        )
    from vasp_auto.chgcar import read_volumetric
    component = (query.get("component") or ["total"])[0].lower()
    volume = read_volumetric(path, spin=(component == "spin"))
    if component == "spin":
        if volume.get("spin_data") is None:
            raise FileNotFoundError(
                f"No spin density in {file_name} — needs a spin-polarised run "
                "(ISPIN=2) whose CHGCAR carries a second (magnetisation) block."
            )
        volume = {**volume, "data": volume["spin_data"]}
        payload = _volume_payload(volume, file_name, axis, fraction)
        payload["kind"] = "spin density ρ↑−ρ↓"
        return payload
    return _volume_payload(volume, file_name, axis, fraction)


def _slot_dir(value, heavy: tuple[str, ...] = ()) -> Path:
    """Resolve one analysis job slot to a local dir.

    Accepts a bare path string (= local) or a {"path", "machine"} object, so a
    multi-job analysis can mix machines. Remote slots route through
    _analysis_dir (rsync into the cache + on-demand heavy files).
    """
    if isinstance(value, dict):
        path, machine = value.get("path"), value.get("machine")
    else:
        path, machine = value, None
    if not path or not str(path).strip():
        raise ValueError("Missing job directory")
    return _analysis_dir(str(path), machine, heavy=heavy)


def api_chgdiff(_query, body):
    """Δρ = ρ(total) − Σρ(parts) from job dirs/CHGCAR paths; writes CHGCAR_diff.

    Each of total/parts may be a bare path (local) or {path, machine} — remote
    jobs are captured (with their CHGCAR) into the cache and analyzed locally.
    """
    from vasp_auto.chgcar import charge_difference
    from vasp_auto.qe_volumetric import cube_as_volumetric, cube_difference

    def chgcar_path(value):
        resolved = _slot_dir(value, heavy=("CHGCAR", "charge-density.cube"))
        if resolved.is_file():
            return resolved
        return (resolved / "charge-density.cube" if job_engine(resolved) == "qe"
                else resolved / "CHGCAR")

    total = chgcar_path(body["total"])
    parts = [chgcar_path(p) for p in body.get("parts") or []
             if (p if isinstance(p, dict) else str(p).strip())]
    if not parts:
        raise ValueError("Charge difference needs at least one part to subtract")
    if total.suffix.lower() == ".cube":
        output = total.parent / "charge-density-diff.cube"
        diff = cube_as_volumetric(cube_difference(total, parts, output), total.parent / "POSCAR")
    else:
        output = total.parent / "CHGCAR_diff"
        diff = charge_difference(total, parts, output)
    payload = _volume_payload(diff, output.name, 2, 0.5)
    payload["path"] = str(output)
    return payload


def api_adsorption(_query, body):
    """E_ads = E(slab+adsorbate) − E(slab) − scale·E(molecule) from job dirs.

    Each of total/slab/molecule may be a bare path or {path, machine}; VASP or
    QE (read_job_energy dispatches on the engine)."""
    from vasp_auto.analysis import adsorption_energy

    def job_dir(key):
        if not body.get(key):
            raise ValueError(f"Missing {key} job directory")
        return _slot_dir(body[key])

    return adsorption_energy(
        job_dir("total"), job_dir("slab"), job_dir("molecule"),
        molecule_scale=float(body.get("scale") or 1.0),
    )


def api_surface(_query, body):
    """Surface energy γ from a slab job and a bulk job (VASP or QE)."""
    from vasp_auto.analysis import surface_energy

    slab = _slot_dir(body.get("slab"))
    bulk = _slot_dir(body.get("bulk"))
    axis = "abc".index(str(body.get("axis") or "c").lower())
    return surface_energy(slab, bulk, axis=axis)


def api_gdiagram(_query, body):
    """Free-energy diagram + overpotential from ordered reaction steps.

    Each step: {label, delta_e | ads:{total,slab,molecule,scale}, freq_job?,
    n_electrons?}; job dirs inside ads/freq_job may be {path, machine}."""
    from vasp_auto.analysis import DEFAULT_TEMPERATURE_K, free_energy_diagram

    def resolve(step):
        out = dict(step)
        if step.get("ads"):
            out["ads"] = {k: (str(_slot_dir(v)) if k in ("total", "slab", "molecule") else v)
                          for k, v in step["ads"].items()}
        if step.get("freq_job"):
            out["freq_job"] = str(_slot_dir(step["freq_job"]))
        return out

    steps = [resolve(s) for s in body.get("steps") or []]
    if not steps:
        raise ValueError("A free-energy diagram needs at least one step")
    return free_energy_diagram(
        steps,
        temperature=float(body.get("T") or DEFAULT_TEMPERATURE_K),
        potential=float(body.get("U") or 0.0),
        u_equilibrium=float(body.get("U_eq") if body.get("U_eq") is not None else 1.23),
    )


def api_magmoms(query, _body):
    """Per-atom magnetic moments from the last OUTCAR magnetization block."""
    job_dir = _analysis_query_dir(query)
    moments = parse_magmoms(job_dir / "OUTCAR")
    if not moments:
        raise FileNotFoundError(
            "No per-atom magnetic moments in this job — needs a spin-polarised "
            "run (ISPIN=2) whose OUTCAR has a 'magnetization (x)' table."
        )
    struct = read_poscar(job_dir / "POSCAR")
    symbols = per_atom_symbols(struct)
    rows = [{"index": i + 1, "element": symbols[i] if i < len(symbols) else "?",
             "moment": m} for i, m in enumerate(moments)]
    return {"moments": rows, "total_moment": sum(moments), "natoms": len(moments)}


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def api_cache_size(_query, _body):
    """Size of the fetched-remote-results cache (volumetric pulls are big)."""
    size = _dir_size(REMOTE_RESULT_CACHE) if REMOTE_RESULT_CACHE.exists() else 0
    return {"bytes": size, "path": str(REMOTE_RESULT_CACHE)}


def api_cache_clear(_query, _body):
    """Delete the whole fetched-remote-results cache."""
    import shutil as _shutil
    if REMOTE_RESULT_CACHE.exists():
        _shutil.rmtree(REMOTE_RESULT_CACHE)
    return {"cleared": True, "path": str(REMOTE_RESULT_CACHE)}


def api_thermo(query, _body):
    """ZPE / U_vib / T·S / ΔG correction from a finished freq job."""
    from vasp_auto.analysis import DEFAULT_TEMPERATURE_K, thermo_from_job

    job_dir = _analysis_query_dir(query)
    temperature = float((query.get("T") or [DEFAULT_TEMPERATURE_K])[0])
    return thermo_from_job(job_dir, temperature=temperature)


def api_dband(query, _body):
    """d-band center/width of selected atoms from a finished dos job."""
    from vasp_auto.analysis import d_band_center, qe_d_band_center

    job_dir = _analysis_query_dir(query)
    selection = (query.get("atoms") or [""])[0].strip()
    if not selection:
        raise ValueError('Give an atom selection, e.g. "1-4" or "z>0.5"')
    struct = read_poscar(job_dir / "POSCAR")
    atoms = parse_atom_selection(struct, selection)
    emax_text = (query.get("emax") or [""])[0].strip()
    emax = float(emax_text) if emax_text else None
    result = (qe_d_band_center(job_dir, atoms, emax_eV=emax)
              if job_engine(job_dir) == "qe"
              else d_band_center(job_dir / "vasprun.xml", atoms, emax_eV=emax))
    result["selection"] = selection
    return result


def api_workfunction(query, _body):
    """Work function W = V_vacuum − E_Fermi from a LOCPOT slab run."""
    from vasp_auto.analysis import work_function

    job_dir = _analysis_query_dir(query, heavy=("LOCPOT",))
    axis = "abc".index((query.get("axis") or ["c"])[0].lower())
    return work_function(job_dir, axis=axis)


def api_optics(query, _body):
    """Absorption coefficient α(E) from a finished LOPTICS run."""
    from vasp_auto.analysis import absorption_spectrum

    job_dir = _analysis_query_dir(query)
    return absorption_spectrum(job_dir if job_engine(job_dir) == "qe"
                               else job_dir / "vasprun.xml")


def api_xrd(query, _body):
    """Simulated powder XRD pattern from the job's CONTCAR (else POSCAR)."""
    from vasp_auto.analysis import xrd_pattern

    job_dir = _analysis_query_dir(query)
    wavelength = (query.get("wavelength") or ["CuKa"])[0].strip() or "CuKa"
    try:
        wavelength = float(wavelength)
    except ValueError:
        pass
    return xrd_pattern(job_dir, wavelength=wavelength)


def api_bader(_query, body):
    """Run the Henkelman bader binary on a job's CHGCAR; like the CLI it
    writes bader_charges.csv next to it (downloadable)."""
    import csv

    from vasp_auto.chgcar import run_bader

    job_dir = _analysis_dir(str(body["path"]), body.get("machine"),
                            heavy=("CHGCAR", "AECCAR0", "AECCAR2", "charge-density.cube"))
    config = _config()
    result = run_bader(job_dir, config.get("bader_executable", "bader"))
    csv_path = job_dir / "bader_charges.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "element", "electrons", "net_charge_e"])
        for charge in result["charges"]:
            writer.writerow([charge["index"], charge["element"],
                             charge["electrons"], charge["net_charge"]])
    result["csv"] = str(csv_path)
    return result


def api_match(_query, body):
    """Supercell suggestions for combining two structures with different cells."""
    matches = match_supercells(
        _load_struct_arg(body, "host"), _load_struct_arg(body, "guest"),
        max_repeat=int(body.get("max_repeat") or 6),
        max_strain=float(body.get("max_strain") or 0.1),
        gamma_tol=float(body.get("gamma_tol") or 8.0),
    )
    return {"matches": matches}


def api_browse(query, _body):
    """Directory listing for the folder picker (single-user local tool)."""
    raw = (query.get("path") or [str(REPO_ROOT / "inputs")])[0]
    base = Path(raw).expanduser().resolve()
    if not base.is_dir():
        base = base.parent if base.parent.is_dir() else REPO_ROOT
    want_files = (query.get("files") or ["0"])[0] not in ("0", "", "false")

    directories = []
    files = []
    try:
        children = sorted(base.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        children = []
    for child in children:
        if child.name.startswith("."):
            continue
        if child.is_dir():
            directories.append({
                "name": child.name,
                "path": str(child),
                "type": get_case_type(child),
                "has_poscar": (child / "POSCAR").exists(),
            })
        elif want_files and child.is_file():
            files.append({"name": child.name, "path": str(child)})

    config = _config()
    return {
        "path": str(base),
        "parent": str(base.parent) if base.parent != base else None,
        "dirs": directories[:500],
        "files": files[:500],
        "roots": [
            {"name": "inputs", "path": str(REPO_ROOT / "inputs")},
            {"name": "jobs", "path": str(Path(config["jobs_root"]).expanduser().resolve())},
            {"name": "repo", "path": str(REPO_ROOT)},
            {"name": "home", "path": str(Path.home())},
        ],
    }


def api_mlenergy(_query, body):
    """Single-point MLIP energy/forces — cheap read-only screen (no files written)."""
    from vasp_auto.ml_tools import DEFAULT_ML_MODEL, DEFAULT_ML_TASK, ml_energy

    poscar_path = Path(body["case"]).expanduser().resolve()
    config = _config()
    return ml_energy(
        poscar_path,
        model=body.get("model") or config.get("ml_model") or DEFAULT_ML_MODEL,
        task=body.get("task") or config.get("ml_task") or DEFAULT_ML_TASK,
        checkpoint=body.get("checkpoint") or config.get("ml_checkpoint"),
    )


def api_mlrelax(_query, body):
    """Pre-relax a case POSCAR with an MLIP (Meta OMat24/UMA, or 'emt' demo)."""
    from vasp_auto.ml_tools import DEFAULT_ML_MODEL, DEFAULT_ML_TASK, ml_relax_case

    case_dir = Path(body["case"]).expanduser().resolve()
    config = _config()
    result = ml_relax_case(
        case_dir,
        model=body.get("model") or config.get("ml_model") or DEFAULT_ML_MODEL,
        task=body.get("task") or config.get("ml_task") or DEFAULT_ML_TASK,
        checkpoint=body.get("checkpoint") or config.get("ml_checkpoint"),
        fmax=float(body.get("fmax") or 0.05),
        steps=int(body.get("steps") or 200),
        relax_cell=bool(body.get("relax_cell")),
    )
    return result


def api_databases(_query, _body):
    """Return the list of available external structure databases."""
    from vasp_auto.ml_tools import DATABASES

    return {"databases": DATABASES}


def api_db_fetch(_query, body):
    """Fetch a structure from an external material database and return POSCAR text.

    Body fields:
        query     — material_id (mp-1234) or formula/chemsys (Fe2O3, Fe-O)
        db_source — "mp" or "umat" (default: "mp")
        api_key   — optional API key (MP: overrides MP_API_KEY env var)
        save_dir  — optional path; if set, POSCAR is written there and
                    the path is included in the response.

    Returns {poscar, label, db_source, saved_path?}.
    """
    from vasp_auto.ml_tools import fetch_structure_from_db

    query = body["query"]
    db_source = body.get("db_source") or "mp"
    api_key = body.get("api_key")
    poscar_text, label = fetch_structure_from_db(query, db_source=db_source, api_key=api_key)

    result: dict = {"poscar": poscar_text, "label": label, "db_source": db_source}
    if save_dir := body.get("save_dir"):
        dest = Path(save_dir).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "POSCAR").write_text(poscar_text, encoding="utf-8")
        result["saved_path"] = str(dest / "POSCAR")

    return result


def api_db_mlenergy(_query, body):
    """Fetch a structure from a database and compute a single-point MLIP energy.

    Body fields: query, db_source, api_key (optional), model, task, checkpoint.
    Returns the same dict as /api/mlenergy plus db_source, db_query, db_label.
    """
    from vasp_auto.ml_tools import DEFAULT_ML_MODEL, DEFAULT_ML_TASK, ml_energy_from_db

    config = _config()
    return ml_energy_from_db(
        body["query"],
        db_source=body.get("db_source") or "mp",
        api_key=body.get("api_key"),
        model=body.get("model") or config.get("ml_model") or DEFAULT_ML_MODEL,
        task=body.get("task") or config.get("ml_task") or DEFAULT_ML_TASK,
        checkpoint=body.get("checkpoint") or config.get("ml_checkpoint"),
    )


def api_db_search(_query, body):
    """Search a material database and return a ranked list of candidate materials.

    Body fields:
        query     — material_id (mp-1234), formula (SnO2), or chemsys (Fe-O)
        db_source — "mp" (default) or "umat" (pending access)
        api_key   — optional API key
        max       — optional result cap (default 20)

    Returns {results: [{material_id, formula, energy_above_hull, is_stable,
    spacegroup, nsites}, ...], db_source}.
    """
    db_source = body.get("db_source") or "mp"
    query = (body.get("query") or "").strip()
    if not query:
        raise ValueError("Enter a formula, chemical system, or material ID to search.")

    if db_source == "mp":
        from vasp_auto.ml_tools import search_mp

        results = search_mp(
            query,
            api_key=body.get("api_key"),
            max_results=int(body.get("max") or 20),
        )
    elif db_source == "umat":
        raise NotImplementedError(
            "META UMAT search is pending access grant. Use db_source 'mp' for now."
        )
    else:
        raise ValueError(f"Unknown database source {db_source!r}.")

    return {"results": results, "db_source": db_source}


_PMGRC_PATH = Path.home() / ".config" / ".pmgrc.yaml"


def _read_pmgrc() -> dict:
    import yaml

    try:
        loaded = yaml.safe_load(_PMGRC_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def api_mp_key_status(_query, _body):
    """Report whether a Materials Project API key is available to the server."""
    configured = bool(_read_pmgrc().get("PMG_MAPI_KEY") or os.environ.get("MP_API_KEY"))
    return {"configured": configured}


def api_mp_key_save(_query, body):
    """Save (or clear) the MP API key in ~/.config/.pmgrc.yaml.

    MPRester re-reads that file on every request, so the key takes effect
    immediately — no restart — and persists for both the UI and the CLI.
    An empty api_key clears the stored key.
    """
    import yaml

    settings = _read_pmgrc()
    if key := (body.get("api_key") or "").strip():
        settings["PMG_MAPI_KEY"] = key
    else:
        settings.pop("PMG_MAPI_KEY", None)
    _PMGRC_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PMGRC_PATH.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return {"configured": "PMG_MAPI_KEY" in settings}


def api_db_prototype(_query, body):
    """Fetch an MP structure as a prototype, with optional element substitution.

    Body fields:
        query          — material_id (mp-1234) or formula (SnO2, Fe2O3)
        substitutions  — optional dict {"Ti": "Sn"} for isostructural replacement
        api_key        — optional MP API key
        save_dir       — optional path; if set, POSCAR is written there

    Returns {poscar, comment, db_label, saved_path?}.
    """
    from vasp_auto.ml_tools import prototype_from_mp
    from vasp_auto.structure import write_poscar

    query = body["query"]
    substitutions = body.get("substitutions") or {}
    api_key = body.get("api_key")
    struct = prototype_from_mp(query, substitutions=substitutions or None, api_key=api_key)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".poscar", delete=False) as f:
        tmp = Path(f.name)
    write_poscar(struct, tmp)
    poscar_text = tmp.read_text(encoding="utf-8")
    tmp.unlink(missing_ok=True)

    result: dict = {"poscar": poscar_text, "comment": struct["comment"], "db_label": query}
    if save_dir := body.get("save_dir"):
        dest = Path(save_dir).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)
        write_poscar(struct, dest / "POSCAR")
        result["saved_path"] = str(dest / "POSCAR")
    return result


def api_db_mlrelax(_query, body):
    """Fetch a structure from a database and ML-relax it.

    Body fields: query, db_source, api_key (optional), model, task, checkpoint,
    fmax, steps, relax_cell, output_dir (optional).
    Returns the same dict as /api/mlrelax plus db_source, db_query, db_label.
    """
    from vasp_auto.ml_tools import DEFAULT_ML_MODEL, DEFAULT_ML_TASK, ml_relax_from_db

    config = _config()
    output_dir = body.get("output_dir")
    return ml_relax_from_db(
        body["query"],
        output_dir=Path(output_dir).expanduser().resolve() if output_dir else None,
        db_source=body.get("db_source") or "mp",
        api_key=body.get("api_key"),
        model=body.get("model") or config.get("ml_model") or DEFAULT_ML_MODEL,
        task=body.get("task") or config.get("ml_task") or DEFAULT_ML_TASK,
        checkpoint=body.get("checkpoint") or config.get("ml_checkpoint"),
        fmax=float(body.get("fmax") or 0.05),
        steps=int(body.get("steps") or 200),
        relax_cell=bool(body.get("relax_cell")),
    )


def api_report(_query, body):
    machine = body.get("machine")
    job_dir = _analysis_dir(str(body["job_dir"]), machine)
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Job directory not found: {job_dir}")
    case_name = body.get("case") or job_dir.name
    path = write_job_report(job_dir, case_name=case_name)
    if _is_remote_machine(machine):
        # For a remote job the report lands in the local cache copy — ship it
        # back so report.md shows up in the job's own output folder too.
        from vasp_auto.runner import _ship_file, _ssh_options, _ssh_target
        remote = _resolve_remote(machine)
        remote_path = f"{str(body['job_dir']).rstrip('/')}/report.md"
        try:
            _ship_file(path, _ssh_target(remote), remote_path, remote, _ssh_options(remote))
            path = remote_path
        except Exception:
            pass  # viewing still works; only the remote copy is missing
    return {"path": str(path), "text": build_job_report(job_dir, case_name=case_name)}


def api_file_get(query, _body):
    name = query["name"][0]
    if not _editable(name):
        raise ValueError(f"Not an editable file: {name}")
    dir_str = query["dir"][0]
    loc = _remote_loc(dir_str, query.get("machine", ["local"])[0])
    if loc:
        remote = _resolve_remote(loc["machine"])
        remote_path = f"{str(loc['remote_dir']).rstrip('/')}/{name}"
        try:
            with tempfile.TemporaryDirectory(prefix="vasp_auto_rc_") as tmp:
                local = fetch_remote_file(remote, remote_path, Path(tmp) / name)
                return {"exists": True, "text": local.read_text(encoding="utf-8")}
        except Exception:
            return {"exists": False, "text": ""}
    path = Path(dir_str).expanduser().resolve() / name
    return {"exists": path.exists(), "text": path.read_text(encoding="utf-8") if path.exists() else ""}


def api_file_save(_query, body):
    name = body["name"]
    if not _editable(name):
        raise ValueError(f"Not an editable file: {name}")
    loc = _remote_loc(body["dir"], body.get("machine"))
    if loc:
        from vasp_auto.runner import _ship_file, _ssh_options, _ssh_target
        remote = _resolve_remote(loc["machine"])
        remote_dir = str(loc["remote_dir"]).rstrip("/")
        with tempfile.TemporaryDirectory(prefix="vasp_auto_rc_") as tmp:
            src = Path(tmp) / name
            src.write_text(body["text"], encoding="utf-8")
            _ship_file(src, _ssh_target(remote), f"{remote_dir}/{name}", remote, _ssh_options(remote))
        return {"saved": f"{loc['machine']}:{remote_dir}/{name}"}
    directory = Path(body["dir"]).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    (directory / name).write_text(body["text"], encoding="utf-8")
    return {"saved": str(directory / name)}


def build_cli_args(body: dict) -> list[str]:
    args = [body["target"]]
    mode = body.get("mode", "run")
    if mode == "prepare":
        args.append("--prepare")
    elif mode == "dry":
        args.append("--dry-run")
    elif mode == "parse":
        args.append("--parse-only")

    if body.get("calc_type"):
        args += ["--calc-type", str(body["calc_type"])]
    if body.get("engine") and body["engine"] != "vasp":
        args += ["--engine", str(body["engine"])]
        if body.get("qe_executable"):
            args += ["--qe-executable", str(body["qe_executable"])]
        if body.get("pseudo_dir"):
            args += ["--pseudo-dir", str(body["pseudo_dir"])]
        if body.get("ase_calculator"):
            args += ["--ase-calculator", str(body["ase_calculator"])]
        if body.get("ase_command"):
            args += ["--ase-command", str(body["ase_command"])]
        if body.get("ase_params"):
            params = body["ase_params"]
            args += ["--ase-params", params if isinstance(params, str) else json.dumps(params)]
        if body.get("ase_fmax"):
            args += ["--ase-fmax", str(body["ase_fmax"])]
        if body.get("ase_steps"):
            args += ["--ase-steps", str(body["ase_steps"])]
    kpoints = body.get("kpoints") or {}
    if kpoints.get("mode"):
        args += ["--kpoints-mode", kpoints["mode"]]
    if kpoints.get("mesh"):
        args += ["--kmesh", str(kpoints["mesh"])]
    if kpoints.get("spacing"):
        args += ["--kspacing", str(kpoints["spacing"])]
    if kpoints.get("kpath"):
        args += ["--kpath", str(kpoints["kpath"])]
    if kpoints.get("divisions"):
        args += ["--kpath-divisions", str(kpoints["divisions"])]
    if body.get("cpus"):
        args += ["-n", str(body["cpus"])]
    if body.get("parallel"):
        args += ["--parallel", str(body["parallel"])]
    if body.get("workflow"):
        args += ["--workflow", str(body["workflow"])]
    if body.get("scheduler") and body["scheduler"] != "local":
        args += ["--scheduler", body["scheduler"]]
    if body.get("converge_encut"):
        args += ["--converge-encut", str(body["converge_encut"])]
    if body.get("converge_sigma"):
        args += ["--converge-sigma", str(body["converge_sigma"])]
    if body.get("converge_scf"):
        args.append("--converge-scf")
    if body.get("nelm_values"):
        args += ["--nelm-values", str(body["nelm_values"])]
    if body.get("kpoints_values"):
        args += ["--kpoints-values", str(body["kpoints_values"])]
    if body.get("energy_tol"):
        args += ["--energy-tol", str(body["energy_tol"])]
    if body.get("sigma_tol"):
        args += ["--sigma-tol", str(body["sigma_tol"])]
    if body.get("reuse_wavecar"):
        args.append("--reuse-wavecar")
    if body.get("spin"):
        args.append("--spin")
    if body.get("magmom"):
        args += ["--magmom", str(body["magmom"])]
    if body.get("auto_retry"):
        args += ["--auto-retry", str(body["auto_retry"])]
    if body.get("retry_failed"):
        args.append("--retry-failed")
    if body.get("resume"):
        args.append("--resume")
    if body.get("neb_images"):
        args += ["--neb-images", str(body["neb_images"])]
    return args


def api_run(_query, body):
    # A case that lives on a remote machine (the working machine, or a local
    # pointer) has no local POSCAR — fetch its inputs into a staging dir to run
    # from, and force its machine as the run target so the calculation happens and
    # is stored on that machine. (The staging dir is read by the async CLI
    # subprocess, so it is left for the OS temp cleaner rather than removed here.)
    existing_folder = body.get("source") == "existing_folder"
    if existing_folder and not body.get("machine"):
        raise ValueError("Run an existing folder requires selecting its machine first.")
    if existing_folder and body.get("machine") == "local":
        loc = None
    else:
        loc = _remote_loc(body["target"], body.get("machine"), body.get("case_type"))
    if loc:
        staging_root = Path(tempfile.mkdtemp(prefix="vasp_auto_rc_"))
        if _is_remote_machine(body.get("machine")):
            # "Run an existing folder" pointed at a machine: the folder may be one
            # case, or a project holding several — scan it over SSH the same way
            # inspect_target does locally, and fetch every case found.
            target = _fetch_remote_target(loc["machine"], loc["remote_dir"], staging_root)
        else:
            target = staging_root / (loc.get("case_name") or "case")
            target.mkdir(parents=True)
            _fetch_remote_case(loc, target)
        body = {**body, "target": str(target), "remote": loc["machine"]}

    # A structured workflow (e.g. one with a convergence step that carries its
    # own scan settings) is written to the case's workflow.yaml, which the CLI
    # then loads automatically — so no --workflow string is added.
    if body.get("workflow_yaml"):
        target = Path(body["target"]).expanduser().resolve()
        case_dir = target if target.is_dir() else target.parent
        (case_dir / "workflow.yaml").write_text(body["workflow_yaml"], encoding="utf-8")
        body = {k: v for k, v in body.items() if k != "workflow"}

    args = build_cli_args(body)

    # Run on a remote machine: write the chosen machine's config to a temp file
    # and hand it to the CLI, which ships the inputs over SSH and submits there.
    remote_name = body.get("remote")
    if remote_name and remote_name != "local":
        args += ["--remote-config", _write_remote_config(remote_name)]

    return _spawn_cli_job(args, body["target"])


def _write_remote_config(remote_name: str) -> str:
    """Write a chosen machine's config to a temp JSON file for --remote-config."""
    remotes = _all_remotes()
    if remote_name not in remotes:
        raise ValueError(f"Unknown remote machine: {remote_name}")
    remote_cfg = {k: v for k, v in remotes[remote_name].items() if k != "source"}
    UI_LOG_DIR.mkdir(parents=True, exist_ok=True)
    remote_file = UI_LOG_DIR / f"remote_{uuid.uuid4().hex[:12]}.json"
    remote_file.write_text(json.dumps(remote_cfg), encoding="utf-8")
    return str(remote_file)


def _spawn_cli_job(args: list[str], target_label: str) -> dict:
    """Launch ``python -m vasp_auto.cli <args>`` as a logged background job and
    register it in JOBS so the live log and ✕ stop button work. Returns {"token"}."""
    UI_LOG_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    log_path = UI_LOG_DIR / f"{token}.log"
    command = [sys.executable, "-u", "-m", "vasp_auto.cli", *args]
    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write("$ vasp-auto " + " ".join(args) + "\n\n")
    log_handle.flush()
    # Own session so the ✕ stop button can killpg the whole tree (mpirun/vasp),
    # not just the python wrapper.
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    with JOBS_LOCK:
        JOBS[token] = {
            "token": token,
            "args": args,
            "target": target_label,
            "started": datetime.now().strftime("%H:%M:%S"),
            "started_at": datetime.now(),
            "finished_at": None,
            "stopped": False,
            "process": process,
            "log_path": log_path,
        }
    return {"token": token}


def _latest_remote_job_dir(remote: dict, case_name: str) -> str | None:
    """The newest job directory on a remote machine whose name matches case_name
    (with or without a NNNN_ prefix), searched under <remote_root>/results and
    <remote_root>. Returns the full remote path, or None when none is found."""
    rr = (remote.get("remote_root") or "").rstrip("/")
    roots = [r for r in ((f"{rr}/results" if rr else ""), rr) if r]
    best = None
    for root in roots:
        try:
            jobs = list_remote_jobs(remote, root)
        except Exception:
            continue
        for j in jobs:
            base = re.sub(r"^\d+_", "", j["name"])
            if base == case_name or j["name"] == case_name:
                if best is None or j.get("modified_ts", 0) > best.get("modified_ts", 0):
                    best = j
    return best["path"] if best else None


def api_resume(_query, body):
    """Resume the latest unfinished job for the working case, in place.

    Local working machine: resume the latest numbered local job directory from its
    newest CONTCAR, reusing its INCAR/KPOINTS/POTCAR (no new job number). A remote
    working machine: resume the latest job directory on that machine in place
    (nothing is re-shipped). Either way the restart streams to the live log like a
    normal run, via ``vasp-auto --resume-job-dir`` under the hood.
    """
    machine = (body.get("machine") or "local").strip()
    cpus = body.get("cpus")

    if _is_remote_machine(machine):
        remote = _resolve_remote(machine)
        case_name = str(body.get("target") or "").rstrip("/").rsplit("/", 1)[-1] or "case"
        job_dir = _latest_remote_job_dir(remote, case_name)
        if not job_dir:
            raise ValueError(
                f"No job directory for '{case_name}' found on {machine} to resume "
                "(run it there first)."
            )
        if body.get("preview"):  # resolve only — the UI shows the INCAR editor first
            return {"job_dir": job_dir, "machine": machine}
        args = ["--resume-job-dir", job_dir]
        if cpus:
            args += ["-n", str(cpus)]
        # Offload machines resume detached (power-off-safe); record a local
        # tracking dir (a .remote.json marker) so the Results tab's 🛰 status / ⬇
        # fetch buttons can follow the run, exactly as for a fresh offload.
        if resolve_remote_run_mode(remote) == "ssh_detached":
            jobs_root = Path(load_config()["jobs_root"]).resolve()
            mirror = jobs_root / Path(job_dir).name
            mirror.mkdir(parents=True, exist_ok=True)
            args += ["--resume-local-mirror", str(mirror)]
        args += ["--remote-config", _write_remote_config(machine)]
        return _spawn_cli_job(args, f"{machine}:{job_dir}")

    # Local: resolve the latest numbered job dir for the working case.
    target = Path(body["target"]).expanduser().resolve()
    config = merge_local_config(load_config(), target)
    jobs_root = Path(config["jobs_root"]).resolve()
    info = inspect_target(target)
    case_info = make_case_info(
        target, jobs_root, single_mode=(info["mode"] == "single"), job_mode="latest"
    )
    job_dir = str(case_info["job_dir"])
    if body.get("preview"):
        return {"job_dir": job_dir, "machine": "local"}
    args = ["--resume-job-dir", job_dir]
    if cpus:
        args += ["-n", str(cpus)]
    return _spawn_cli_job(args, job_dir)


def _job_state(job: dict) -> dict:
    returncode = job["process"].poll()
    if returncode is not None and job["finished_at"] is None:
        job["finished_at"] = datetime.now()
    end = job["finished_at"] or datetime.now()
    return {
        "token": job["token"],
        "target": job["target"],
        "args": job["args"],
        "started": job["started"],
        "elapsed_s": int((end - job["started_at"]).total_seconds()),
        "running": returncode is None,
        "stopped": job["stopped"],
        "returncode": returncode,
        "pid": job["process"].pid,
    }


def api_stop(_query, body):
    token = body["token"]
    with JOBS_LOCK:
        job = JOBS.get(token)
    if job is None:
        raise KeyError(f"Unknown job: {token}")
    proc = job["process"]
    if proc.poll() is None:
        job["stopped"] = True
        try:  # kill the whole process group (mpirun/vasp children), see start_new_session
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        _mark_terminated_local(job.get("target"))
    return {"token": token, "stopped": True}


def _mark_terminated_local(target) -> None:
    """Best-effort: drop a .terminated marker in the newest unfinished job dir for
    ``target``'s case, so the Results/Running tabs read "terminated" not "running".

    ponytail: heuristic — the UI doesn't capture the CLI's chosen job number, so
    match the case name and pick the newest dir without a job.log. A miss just
    leaves the row reading "running"; nothing outside jobs_root is touched.
    """
    if not target:
        return
    try:
        t = Path(target).expanduser().resolve()
        jobs_root = Path(merge_local_config(load_config(), t)["jobs_root"]).resolve()
        cands = [d for d in [*jobs_root.glob(f"*_{t.name}"), jobs_root / t.name]
                 if d.is_dir() and not (d / "job.log").exists()]
        if cands:
            (max(cands, key=lambda d: d.stat().st_mtime) / ".terminated").write_text(
                "terminated", encoding="utf-8")
    except Exception:
        pass


def _assert_under_remote_root(machine: str, remote: dict, remote_dir: str) -> str:
    """Validate a client-supplied remote path sits strictly inside the machine's
    remote_root before any destructive op. The boundary is the machine's config,
    never the client. Returns the normalized path."""
    root = (remote.get("remote_root") or "").rstrip("/")
    rd = remote_dir.rstrip("/")
    if not root:
        raise ValueError(f"{machine} has no remote_root configured — refusing to touch {rd}.")
    if ".." in rd.split("/") or rd == root or not rd.startswith(root + "/"):
        raise ValueError(f"Refusing: {rd} is not inside {machine}'s work folder ({root}).")
    return rd


def api_job_delete(_query, body):
    """Delete a job/case folder. Local: strictly inside the configured jobs_root.
    Remote (``machine`` given): ``rm -rf`` over SSH, strictly inside that machine's
    remote_root. The delete boundary always comes from server config, never the
    client, so nothing outside can be removed."""
    machine = body.get("machine")
    if machine and _is_remote_machine(machine):
        remote = _resolve_remote(machine)
        rd = _assert_under_remote_root(machine, remote, str(body["job_dir"]))
        delete_remote_dir(remote, rd)
        return {"deleted": rd, "machine": machine}
    job_dir = Path(body["job_dir"]).expanduser().resolve()
    jobs_root = Path(_config()["jobs_root"]).resolve()
    if job_dir == jobs_root or not job_dir.is_relative_to(jobs_root):
        raise ValueError("Refusing to delete: only folders inside the jobs/results folder can be removed.")
    if not job_dir.is_dir():
        raise ValueError(f"Not a folder: {job_dir}")
    shutil.rmtree(job_dir)
    return {"deleted": str(job_dir)}


def api_kill_remote(_query, body):
    """Stop a running detached (offload) job on its remote machine, via its
    local mirror's ``.remote.json`` (machine + control dir + PID)."""
    job_dir = Path(body["job_dir"]).expanduser().resolve()
    marker = read_remote_marker(job_dir)
    if not marker:
        raise ValueError("This case was not submitted to a remote machine.")
    if marker.get("mode") != "ssh_detached":
        raise ValueError("Only detached (offload) jobs can be stopped from here — "
                         "scheduler jobs stop with the queue's scancel/qdel.")
    remote = _remote_for_marker(marker)
    result = kill_detached_job(remote, marker.get("control_dir", ""), marker.get("pid"))
    # Mark the local mirror so the Results tab reads "terminated" right away.
    try:
        (job_dir / ".terminated").write_text("terminated", encoding="utf-8")
    except OSError:
        pass
    result["machine"] = marker.get("machine") or marker.get("host")
    result["pid"] = marker.get("pid")
    return result


def _kill_local_job(job_dir: Path) -> dict:
    """Stop a local run: the process group of the PID recorded in ``.pid``, plus
    any live process whose cwd is inside the job dir — the same two signals the
    Running board uses, so anything it shows as "running" can be stopped."""
    from vasp_auto.workflow import pids_in_dir
    killed: list[int] = []
    pid = read_pid(job_dir)
    if pid is not None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            killed.append(pid)
        except (ProcessLookupError, PermissionError):
            pass
    for p in pids_in_dir(job_dir):
        try:
            os.kill(p, signal.SIGTERM)
            killed.append(p)
        except (ProcessLookupError, PermissionError):
            pass
    if killed:
        return {"killed": True, "pid": killed[0]}
    return {"killed": False, "pid": pid,
            "raw": "no live process found for this job — already finished?"}


def api_terminate(_query, body):
    """Stop a job by its job directory — one entry point for the Results tab.
    A job browsed on a remote machine (``machine`` given) is killed there by the
    PID in its ``<dir>/.pid``; a detached-offload job dies on its remote via its
    marker; a local run dies by its recorded PID. Either way a ``.terminated``
    marker is dropped so the row reads "terminated"."""
    machine = body.get("machine")
    if machine and _is_remote_machine(machine):
        remote = _resolve_remote(machine)
        result = kill_job_by_dir(remote, str(body["job_dir"]))
        result["machine"] = machine
        return result
    job_dir = Path(body["job_dir"]).expanduser().resolve()
    marker = read_remote_marker(job_dir)
    if marker and marker.get("mode") == "ssh_detached":
        remote = _remote_for_marker(marker)
        result = kill_detached_job(remote, marker.get("control_dir", ""), marker.get("pid"))
        result["machine"] = marker.get("machine") or marker.get("host")
        result.setdefault("pid", marker.get("pid"))
    else:
        result = _kill_local_job(job_dir)
    try:
        (job_dir / ".terminated").write_text("terminated", encoding="utf-8")
    except OSError:
        pass
    return result


def api_resume_job(_query, body):
    """Resume one specific job directory in place (local, detached-offload, or a
    job browsed on a remote machine), streaming to the live log like a fresh run.
    Clears any stale ``.terminated`` marker first so the restarted job isn't still
    shown as terminated."""
    cpus = body.get("cpus")
    machine = body.get("machine")
    # A job browsed directly on a remote machine: resume it in place on that
    # machine (offload machines resume detached; others run over SSH).
    if machine and _is_remote_machine(machine):
        remote = _resolve_remote(machine)
        remote_dir = str(body["job_dir"]).rstrip("/")
        clear_remote_terminated(remote, remote_dir)
        args = ["--resume-job-dir", remote_dir]
        if cpus:
            args += ["-n", str(cpus)]
        if resolve_remote_run_mode(remote) == "ssh_detached":
            mirror = Path(load_config()["jobs_root"]).resolve() / Path(remote_dir).name
            mirror.mkdir(parents=True, exist_ok=True)
            args += ["--resume-local-mirror", str(mirror)]
        args += ["--remote-config", _write_remote_config(machine)]
        return _spawn_cli_job(args, f"{machine}:{remote_dir}")

    job_dir = Path(body["job_dir"]).expanduser().resolve()
    marker = read_remote_marker(job_dir)
    try:
        (job_dir / ".terminated").unlink()
    except OSError:
        pass
    if marker and marker.get("mode") == "ssh_detached":
        machine = marker.get("machine") or marker.get("host")
        remote = _resolve_remote(machine)
        remote_dir = _resolve_remote_job_dir(job_dir, marker, remote) or marker.get("remote_dir")
        if not remote_dir:
            raise ValueError("No remote directory recorded for this offload job.")
        args = ["--resume-job-dir", remote_dir]
        if cpus:
            args += ["-n", str(cpus)]
        args += ["--resume-local-mirror", str(job_dir),
                 "--remote-config", _write_remote_config(machine)]
        return _spawn_cli_job(args, f"{machine}:{remote_dir}")
    args = ["--resume-job-dir", str(job_dir)]
    if cpus:
        args += ["-n", str(cpus)]
    return _spawn_cli_job(args, str(job_dir))


def api_jobs(_query, _body):
    with JOBS_LOCK:
        jobs = [_job_state(job) for job in JOBS.values()]
    jobs.sort(key=lambda j: j["started"], reverse=True)
    return {"jobs": jobs}


def api_running(query, _body):
    """Jobs running now across the local machine + every configured remote.

    ?all=1 lists finished/prepared jobs too; ?machine=local|<name> narrows the
    scan to one machine (default: everything). Unreachable machines are returned
    under "errors" so one down machine doesn't blank the whole board.
    """
    all_jobs = (query.get("all", ["0"])[0] or "0").lower() in ("1", "true", "yes")
    machine = (query.get("machine", [""])[0] or "").strip() or None
    result = list_running_jobs(_config(), running_only=not all_jobs, machine=machine)
    for j in result["jobs"]:
        j["modified"] = _fmt_ts(j.get("modified_ts"))
    return result


def api_job(query, _body):
    token = query["token"][0]
    with JOBS_LOCK:
        job = JOBS.get(token)
    if job is None:
        raise KeyError(f"Unknown job: {token}")
    state = _job_state(job)
    log_path: Path = job["log_path"]
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    state["log"] = text[-20000:]
    return state


# ---------------------------------------------------------------- remote machines

def _load_remotes_store() -> dict:
    if REMOTES_FILE.exists():
        try:
            return json.loads(REMOTES_FILE.read_text(encoding="utf-8"))
        except ValueError:
            return {}
    return {}


def _save_remotes_store(data: dict) -> None:
    REMOTES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _all_remotes() -> dict:
    """All known remote machines: config.yaml (read-only) + UI store (writable)."""
    config = _config()
    remotes: dict[str, dict] = {}
    for name, entry in (config.get("remotes") or {}).items():
        remotes[name] = {**entry, "name": name, "source": "config"}
    if config.get("remote"):
        remotes.setdefault("default", {**config["remote"], "name": "default", "source": "config"})
    for name, entry in _load_remotes_store().items():
        remotes[name] = {**entry, "name": name, "source": "ui"}
    return remotes


def _remote_from_body(body: dict) -> dict:
    """A remote config from explicit form fields, or by name from the store."""
    if body.get("host"):
        remote = {k: body[k] for k in REMOTE_FIELDS if body.get(k) not in (None, "")}
        for key in ("ssh_options", "scheduler_options"):
            if body.get(key):
                remote[key] = body[key]
        if body.get("name"):
            remote["name"] = body["name"]
        return remote
    name = body.get("name")
    remotes = _all_remotes()
    if name and name in remotes:
        return remotes[name]
    raise ValueError("Provide machine details (at least host) or a saved machine name.")


def _remote_for_marker(marker: dict) -> dict:
    """Find the saved machine (with credentials) a job was submitted to."""
    remotes = _all_remotes()
    name = marker.get("machine") or marker.get("host")
    if name in remotes:
        return remotes[name]
    for entry in remotes.values():
        if entry.get("host") == marker.get("host"):
            return entry
    # Last resort: host only (works if SSH config supplies user/key).
    return {"host": marker.get("host"), "scheduler": marker.get("scheduler", "slurm")}


def api_remotes(_query, _body):
    """List configured remote machines (newest editable ones from the UI store)."""
    return {"remotes": sorted(_all_remotes().values(), key=lambda r: r["name"])}


def api_remote_save(_query, body):
    name = (body.get("name") or "").strip()
    if not name:
        raise ValueError("Give the machine a name.")
    if not body.get("host"):
        raise ValueError("host is required.")
    if not body.get("remote_root"):
        raise ValueError("remote_root is required (a base directory on the remote).")
    if not body.get("vasp_executable") and not body.get("qe_executable"):
        raise ValueError("vasp_executable is required (the VASP path on the remote), "
                         "or qe_executable for a QE-only machine.")
    entry = {k: body[k] for k in REMOTE_FIELDS if body.get(k) not in (None, "")}
    if "cpus" in entry:
        try:
            entry["cpus"] = max(1, int(entry["cpus"]))
        except (TypeError, ValueError):
            del entry["cpus"]
    if "max_jobs" in entry:
        try:
            entry["max_jobs"] = max(1, int(entry["max_jobs"]))
        except (TypeError, ValueError):
            del entry["max_jobs"]
    if body.get("scheduler_options"):
        entry["scheduler_options"] = body["scheduler_options"]
    store = _load_remotes_store()
    store[name] = entry
    _save_remotes_store(store)
    return {"saved": name, "remote": {**entry, "name": name, "source": "ui"}}


def api_remote_delete(_query, body):
    name = body.get("name")
    store = _load_remotes_store()
    if name in store:
        del store[name]
        _save_remotes_store(store)
        return {"deleted": name}
    raise ValueError(f"No UI-managed machine named '{name}' to delete "
                     "(machines defined in config.yaml are read-only).")


def api_remote_test(_query, body):
    """Verify SSH + remote_root/VASP/scheduler — the Test connection button."""
    return check_remote_connection(_remote_from_body(body))


def api_remote_setup(_query, body):
    """Install the vasp_auto engine venv on a machine (the offload setup button)."""
    remote = _remote_from_body(body)
    result = setup_remote_engine(remote)
    result["machine"] = remote.get("name") or remote.get("host")
    return result


def _resolve_remote_job_dir(local_job_dir: Path, marker: dict, remote: dict) -> str | None:
    """The real job root on the remote, resolved from a detached offload's control dir.

    The remote engine numbers each run (``<remote_root>/results/<NNNN>_<case>``), so the
    placeholder recorded at submit time is not where the job actually lives. Read the
    engine-written path back, persist it into the local ``.remote.json`` so later calls
    are cheap, and return it. Falls back to the recorded ``remote_dir`` for non-offload
    jobs, or while the engine has not yet allocated its directory."""
    recorded = marker.get("remote_dir")
    if marker.get("mode") != "ssh_detached" or not marker.get("control_dir"):
        return recorded
    resolved = resolve_detached_job_dir(remote, marker["control_dir"])
    if resolved and resolved != recorded:
        marker["remote_dir"] = resolved
        try:
            (local_job_dir / ".remote.json").write_text(
                json.dumps(marker, indent=2), encoding="utf-8")
        except OSError:
            pass
    return resolved or recorded


def api_remote_status(_query, body):
    """Poll the remote for a submitted job's state (scheduler or detached offload)."""
    job_dir = Path(body["job_dir"]).expanduser().resolve()
    marker = read_remote_marker(job_dir)
    if not marker:
        raise ValueError("This case was not submitted to a remote machine.")
    remote = _remote_for_marker(marker)
    # Detached offload jobs are tracked by a PID + control dir, not a queue id.
    if marker.get("mode") == "ssh_detached":
        if not marker.get("control_dir"):
            raise ValueError("No control directory recorded for this offload job.")
        result = poll_detached_job(remote, marker["control_dir"], marker.get("pid"))
    else:
        if not marker.get("job_id"):
            raise ValueError("No remote job id recorded for this case.")
        result = poll_remote_job(remote, marker["job_id"], marker.get("state_file"))
    result["machine"] = marker.get("machine") or marker.get("host")
    result["remote_dir"] = _resolve_remote_job_dir(job_dir, marker, remote)
    return result


def api_remote_fetch(_query, body):
    """Pull result files back from the remote job dir so local viewers work."""
    job_dir = Path(body["job_dir"]).expanduser().resolve()
    marker = read_remote_marker(job_dir)
    if not marker:
        raise ValueError("This case was not submitted to a remote machine.")
    remote = _remote_for_marker(marker)
    remote_dir = _resolve_remote_job_dir(job_dir, marker, remote)
    if not remote_dir:
        raise ValueError("No remote directory recorded for this case.")
    result = fetch_remote_results(
        remote, remote_dir, job_dir,
        include_heavy=bool(body.get("heavy")),
    )
    # Build a fresh job.log from the pulled files so a readable summary exists even
    # when the remote engine predates job.log (or only wrote partial output).
    if (job_dir / "OUTCAR").exists():
        from vasp_auto.job_log import write_job_log
        write_job_log(job_dir, job_dir.name)
    result["machine"] = marker.get("machine") or marker.get("host")
    result["has_outcar"] = (job_dir / "OUTCAR").exists()
    result["has_vasprun"] = (job_dir / "vasprun.xml").exists()
    return result


# ----------------------------------------------------- browse + download files

def _fmt_ts(ts) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""


def _resolve_remote(name: str) -> dict:
    """Look up a saved/config remote machine (with credentials) by name."""
    remotes = _all_remotes()
    if not name or name not in remotes:
        raise ValueError(f"Unknown remote machine: {name!r}")
    return remotes[name]


def api_remote_jobs(_query, body):
    """List the job directories that live on a remote machine (newest first)."""
    remote = _resolve_remote(body.get("machine"))
    root = (body.get("dir") or "").strip() or remote.get("remote_root")
    if not root:
        raise ValueError("This machine has no remote_root set — type a jobs directory.")
    rows = list_remote_jobs(remote, root)
    machine = remote.get("name") or remote.get("host")
    for r in rows:
        r["modified"] = _fmt_ts(r.get("modified_ts"))
        r["machine"] = machine
    return {"machine": machine, "dir": root,
            "remote_root": remote.get("remote_root"), "rows": rows}


def api_remote_files(_query, body):
    """List the files/subdirs of one directory on a remote machine."""
    remote = _resolve_remote(body.get("machine"))
    path = (body.get("dir") or body.get("job_dir") or "").strip()
    if not path:
        raise ValueError("No remote directory given.")
    data = list_remote_dir(remote, path)
    for e in data["entries"]:
        e["modified"] = _fmt_ts(e.get("modified_ts"))
    data["machine"] = remote.get("name") or remote.get("host")
    return data


def _local_dir_entries(directory: Path) -> list[dict]:
    entries: list[dict] = []
    try:
        children = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError:
        return entries
    for child in children:
        try:
            st = child.stat()
        except OSError:
            continue
        is_dir = child.is_dir()
        entries.append({
            "name": child.name,
            "path": str(child),
            "is_dir": is_dir,
            "size": 0 if is_dir else st.st_size,
            "modified_ts": int(st.st_mtime),
            "modified": _fmt_ts(int(st.st_mtime)),
        })
    return entries


def api_job_files(_query, body):
    """List the files/subdirs of one local job directory (for download)."""
    directory = Path(body["dir"]).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    parent = str(directory.parent) if directory.parent != directory else None
    return {"path": str(directory), "parent": parent,
            "entries": _local_dir_entries(directory)}


# Maximum bytes shown in the in-browser file viewer (download for the full file).
TEXT_PREVIEW_MAX = 200_000
# Suffixes / names we never preview inline: pure binary, or proprietary (POTCAR).
_BINARY_SUFFIXES = {".xlsx", ".png", ".jpg", ".jpeg", ".pdf", ".gz", ".zip",
                    ".tar", ".bz2", ".xz", ".npy", ".h5", ".hdf5", ".pkl", ".bin", ".so"}
_NO_PREVIEW_NAMES = {"POTCAR", "WAVECAR", "CHGCAR", "CHG", "AECCAR0", "AECCAR1",
                     "AECCAR2", "WAVEDER", "TMPCAR", "PROOUT"}


def _previewable(name: str) -> bool:
    """Whether a file can be shown in the text viewer (vs download-only).

    POTCAR and the bulky volumetric/wavefunction binaries are never previewed —
    POTCAR content is proprietary and must not be printed.
    """
    if name in _NO_PREVIEW_NAMES:
        return False
    return Path(name).suffix.lower() not in _BINARY_SUFFIXES


def api_filetext(_query, body):
    """Return the text of one file (local or remote) for the in-browser viewer.

    Body: ``{"path": ..., "name"?: ..., "machine"?: ...}``. A ``machine`` other
    than "local" reads it over SSH (kept inside the machine's remote_root).
    Binary/proprietary files return ``{"previewable": False}`` (download only).
    """
    path_str = body.get("path") or ""
    name = body.get("name") or Path(path_str).name
    if not _previewable(name):
        return {"previewable": False, "name": name,
                "reason": "Binary or proprietary file — use the download button."}
    machine = body.get("machine")
    if machine and machine != "local":
        remote = _resolve_remote(machine)
        root = (remote.get("remote_root") or "").rstrip("/")
        if root and not (path_str == root or path_str.startswith(root + "/")):
            raise ValueError("Path is outside the machine's remote_root")
        data = read_remote_text(remote, path_str, TEXT_PREVIEW_MAX)
        return {"previewable": True, "name": name, **data}
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")
    size = path.stat().st_size
    text = path.read_bytes()[:TEXT_PREVIEW_MAX].decode("utf-8", "replace")
    return {"previewable": True, "name": name, "text": text,
            "size": size, "truncated": size > TEXT_PREVIEW_MAX}


GET_ROUTES = {
    "/api/meta": api_meta,
    "/api/remotes": api_remotes,
    "/api/cases": api_cases,
    "/api/structure": api_structure,
    "/api/results": api_results,
    "/api/template": api_template,
    "/api/incar_options": api_incar_options,
    "/api/file": api_file_get,
    "/api/jobs": api_jobs,
    "/api/running": api_running,
    "/api/job": api_job,
    "/api/trajectory": api_trajectory,
    "/api/neb": api_neb,
    "/api/dos": api_dos,
    "/api/pdos": api_pdos,
    "/api/bands": api_bands,
    "/api/volume": api_volume,
    "/api/browse": api_browse,
    "/api/thermo": api_thermo,
    "/api/dband": api_dband,
    "/api/workfunction": api_workfunction,
    "/api/optics": api_optics,
    "/api/xrd": api_xrd,
    "/api/magmoms": api_magmoms,
    "/api/cache_size": api_cache_size,
    "/api/mp_key": api_mp_key_status,
}

POST_ROUTES = {
    "/api/build": api_build,
    "/api/structure": api_structure_save,
    "/api/combine": api_combine,
    "/api/molecule": api_molecule,
    "/api/nl_build": api_nl_build,
    "/api/nl_agent": api_nl_agent,
    "/api/match": api_match,
    "/api/chgdiff": api_chgdiff,
    "/api/adsorption": api_adsorption,
    "/api/surface": api_surface,
    "/api/gdiagram": api_gdiagram,
    "/api/cache_clear": api_cache_clear,
    "/api/bader": api_bader,
    "/api/preview": api_preview,
    "/api/incar_apply": api_incar_apply,
    "/api/run": api_run,
    "/api/resume": api_resume,
    "/api/stop": api_stop,
    "/api/kill_remote": api_kill_remote,
    "/api/terminate": api_terminate,
    "/api/resume_job": api_resume_job,
    "/api/job_delete": api_job_delete,
    "/api/report": api_report,
    "/api/mlrelax": api_mlrelax,
    "/api/mlenergy": api_mlenergy,
    "/api/databases": api_databases,
    "/api/db_search": api_db_search,
    "/api/mp_key": api_mp_key_save,
    "/api/db_fetch": api_db_fetch,
    "/api/db_prototype": api_db_prototype,
    "/api/db_mlenergy": api_db_mlenergy,
    "/api/db_mlrelax": api_db_mlrelax,
    "/api/file": api_file_save,
    "/api/remote/save": api_remote_save,
    "/api/remote/delete": api_remote_delete,
    "/api/remote/test": api_remote_test,
    "/api/remote/setup": api_remote_setup,
    "/api/remote/status": api_remote_status,
    "/api/remote/fetch": api_remote_fetch,
    "/api/remote/jobs": api_remote_jobs,
    "/api/remote/files": api_remote_files,
    "/api/job/files": api_job_files,
    "/api/filetext": api_filetext,
}

# File types the /download endpoint will serve (summaries and reports only).
DOWNLOADABLE_SUFFIXES = {".xlsx", ".csv", ".md"}


def _downloadable(path: Path) -> bool:
    if path.suffix.lower() not in DOWNLOADABLE_SUFFIXES:
        return False
    config = _config()
    allowed_roots = [REPO_ROOT, Path(config["jobs_root"])]
    return any(path.is_relative_to(root) for root in allowed_roots)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass  # keep the terminal quiet; the UI has its own logs

    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self, routes, body):
        parsed = urlparse(self.path)
        handler = routes.get(parsed.path)
        if handler is None:
            self._send_json({"error": f"Unknown endpoint: {parsed.path}"}, status=404)
            return
        try:
            result = handler(parse_qs(parsed.query), body)
            self._send_json(result)
        except Exception as exc:  # surfaced to the UI as a banner
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            page = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            # Never cache: a restarted server must serve the fresh UI, or the
            # browser keeps calling old endpoints (e.g. /api/incar_options with
            # no ?type=) and shows stale controls.
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if parsed.path == "/download":
            self._send_download(parse_qs(parsed.query))
            return
        if parsed.path == "/download_local":
            self._send_download_local(parse_qs(parsed.query))
            return
        if parsed.path == "/download_remote":
            self._send_download_remote(parse_qs(parsed.query))
            return
        self._dispatch(GET_ROUTES, None)

    def _send_download(self, query):
        try:
            path = Path(query["path"][0]).expanduser().resolve()
        except (KeyError, IndexError):
            self._send_json({"error": "Missing path parameter"}, status=400)
            return
        if not path.exists() or not _downloadable(path):
            self._send_json({"error": f"Not downloadable: {path}"}, status=404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, filename: str):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_download_local(self, query):
        """Serve any individual file from a local job directory."""
        try:
            path = Path(query["path"][0]).expanduser().resolve()
        except (KeyError, IndexError):
            self._send_json({"error": "Missing path parameter"}, status=400)
            return
        if not path.is_file():
            self._send_json({"error": f"Not a file: {path}"}, status=404)
            return
        self._send_bytes(path.read_bytes(), path.name)

    def _send_download_remote(self, query):
        """Pull one file off a remote machine (scp) and stream it to the browser."""
        try:
            machine = query["machine"][0]
            rpath = query["path"][0]
        except (KeyError, IndexError):
            self._send_json({"error": "Missing machine/path parameter"}, status=400)
            return
        try:
            remote = _resolve_remote(machine)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        # Keep downloads inside the machine's work dir when one is configured.
        root = (remote.get("remote_root") or "").rstrip("/")
        if root and not (rpath == root or rpath.startswith(root + "/")):
            self._send_json({"error": "Path is outside the machine's remote_root"}, status=403)
            return
        import tempfile
        try:
            with tempfile.TemporaryDirectory() as td:
                local = fetch_remote_file(remote, rpath, Path(td) / Path(rpath).name)
                data = local.read_bytes()
        except Exception as exc:  # surfaced to the browser
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)
            return
        self._send_bytes(data, Path(rpath).name)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return
        self._dispatch(POST_ROUTES, body)


def create_server(host: str = "127.0.0.1", port: int = 8800) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Local web UI for vasp_auto.")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    httpd = create_server(args.host, args.port)
    url = f"http://{args.host}:{httpd.server_address[1]}/"
    print(f"vasp_auto UI running at {url}  (Ctrl-C to stop)")
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
