from __future__ import annotations

import json
import os
from pathlib import Path
import yaml


GLOBAL_CONFIG = Path("/opt/vasp_auto/config.yaml")


def load_yaml_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    return data


def find_config_file(explicit_path: str | None = None) -> Path | None:
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    candidates.extend([
        Path.cwd() / "config.yaml",
        Path(os.environ["VASP_AUTO_ROOT"]) / "config.yaml" if "VASP_AUTO_ROOT" in os.environ else None,
        Path(__file__).resolve().parents[2] / "config.yaml",
        Path.home() / ".vasp_auto" / "config.yaml",
        GLOBAL_CONFIG,
    ])

    for path in candidates:
        if path is None:
            continue
        resolved = path.resolve()
        if resolved.exists():
            return resolved

    if explicit_path:
        searched = "\n".join(str(p.resolve()) for p in candidates)
        raise FileNotFoundError(f"No config.yaml found. Searched:\n{searched}")

    return None


def default_config(base_dir: Path | None = None) -> dict:
    base_dir = base_dir or Path.cwd()
    return {
        "engine": os.environ.get("VASP_AUTO_ENGINE", "vasp"),
        "vasp_executable": os.environ.get("VASP_EXECUTABLE", "vasp_std"),
        "qe_executable": os.environ.get("QE_EXECUTABLE", "pw.x"),
        "pseudo_dir": os.environ.get("QE_PSEUDO_DIR"),
        "jobs_root": str(base_dir / "jobs"),
        "potcar_root": str(base_dir / "POTCAR"),
        "neb_images": 5,
    }


def _normalize_path(value, base_dir: Path):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((base_dir / path).resolve())


def load_ui_remotes(config_dir: Path) -> dict:
    """Remote machines saved by the web UI's Remote tab (``remotes.json``).

    The UI writes its machines to ``remotes.json`` next to ``config.yaml``; the CLI
    historically read only ``config.yaml``'s ``remotes:``, so a machine added in the
    UI was invisible to ``--remote NAME``. Reading the same file here gives one
    shared machine list across the UI and CLI (single source of truth, no drift).
    Returns the {name: config} mapping, or {} when there is no/invalid store.
    """
    repo_root = Path(__file__).resolve().parents[2]
    for path in (config_dir / "remotes.json", repo_root / "remotes.json"):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return {}
            return data if isinstance(data, dict) else {}
    return {}


def load_config(explicit_path: str | None = None) -> dict:
    config_path = find_config_file(explicit_path)
    base_dir = config_path.parent if config_path else Path.cwd()
    config = default_config(base_dir)

    if config_path:
        config.update(load_yaml_file(config_path))
        config["_config_path"] = str(config_path)
        config["_config_dir"] = str(config_path.parent)
    else:
        config["_config_path"] = ""
        config["_config_dir"] = str(base_dir)

    # Fold in machines saved by the web UI (remotes.json) so `--remote NAME` sees
    # anything added/edited there. The UI store wins on name clashes, matching the
    # UI's own view, so the two never disagree about a machine.
    ui_remotes = load_ui_remotes(Path(config["_config_dir"]))
    if ui_remotes:
        config["remotes"] = {**(config.get("remotes") or {}), **ui_remotes}

    if "job_root" in config and "jobs_root" not in config:
        config["jobs_root"] = config["job_root"]

    # jobs_root is user-facing output, so a relative value follows the
    # directory where the command is run. Library/config paths stay relative to
    # the config file location.
    if "jobs_root" in config and config["jobs_root"] is not None:
        config["jobs_root"] = _normalize_path(config["jobs_root"], Path.cwd())

    for key in [
        "potcar_root",
        "pseudo_dir",
        "cases_root",
        "templates_root",
    ]:
        if key in config and config[key] is not None:
            config[key] = _normalize_path(config[key], base_dir)

    if "vasp_executable" in config and config["vasp_executable"] is not None:
        exe = Path(str(config["vasp_executable"])).expanduser()
        config["vasp_executable"] = str(exe.resolve()) if exe.parent != Path(".") else str(exe)

    return config


def merge_local_config(config: dict, directory: Path) -> dict:
    """Overlay a per-project or per-case config.yaml on top of the base config.

    Relative paths in the local file resolve against its own directory.
    Returns the base config unchanged when the directory has no config.yaml
    (or it is the file already loaded globally).
    """
    local_path = Path(directory) / "config.yaml"
    if not local_path.exists():
        return config
    if str(local_path.resolve()) == config.get("_config_path"):
        return config

    merged = dict(config)
    merged.update(load_yaml_file(local_path))
    merged["_local_config"] = str(local_path.resolve())

    base_dir = local_path.parent
    if merged.get("jobs_root") != config.get("jobs_root") and merged.get("jobs_root"):
        merged["jobs_root"] = _normalize_path(merged["jobs_root"], Path.cwd())
    for key in ["potcar_root", "pseudo_dir", "cases_root", "templates_root"]:
        if merged.get(key) != config.get(key) and merged.get(key):
            merged[key] = _normalize_path(merged[key], base_dir)
    if merged.get("vasp_executable") != config.get("vasp_executable") and merged.get("vasp_executable"):
        exe = Path(str(merged["vasp_executable"])).expanduser()
        merged["vasp_executable"] = str(exe.resolve()) if exe.parent != Path(".") else str(exe)

    return merged
