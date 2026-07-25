"""Interactive (and scriptable) setup for VASP-auto remote-submission machines.

This is the ``vasp-auto-setup-remote`` console program. It does, on a Linux
host, everything needed to make ``vasp-auto <case> --remote NAME`` work against
a cluster or workstation:

  1. Make sure a local SSH key pair exists (generates an ed25519 key if not).
  2. Install your public key on the remote so logins are passwordless
     (``ssh-copy-id`` if present, otherwise a manual append over SSH).
  3. Verify the connection and that the remote has the VASP binary and a
     scheduler (reuses :func:`runner.check_remote_connection`).
  4. Save the machine so both the CLI and the web UI can use it:
       * ``config.yaml`` under ``remotes: <name>:``  (read by ``--remote NAME``)
       * ``remotes.json`` next to it                  (read by the UI Remote tab)

It installs *nothing* on the remote — VASP, MPI and the scheduler must already
be there (see ``docs/PORTABILITY.md``). Run with ``--help`` for the flags; with
no flags it asks interactively.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from . import runner

# Repo root: src/vasp_auto/remote_setup.py -> parents[2]. Matches the locations
# config_loader.find_config_file and the UI server use for config.yaml /
# remotes.json, so a machine saved here is visible to both.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
REMOTES_FILE = REPO_ROOT / "remotes.json"

# The fields persisted for a machine (mirrors the UI's REMOTE_FIELDS plus lists).
SCALAR_FIELDS = ("host", "user", "port", "ssh_key", "remote_root",
                 "vasp_executable", "potcar_root", "qe_executable", "pseudo_dir",
                 "qe_env_setup", "scheduler", "run_mode", "cpus")
LIST_FIELDS = ("ssh_options", "scheduler_options")


# ----------------------------------------------------------------- SSH key setup

def ensure_local_key(key_path: str | Path | None = None) -> Path:
    """Return a usable private-key path, generating an ed25519 key if needed.

    If ``key_path`` is given it is used (and created if absent). Otherwise an
    existing ``~/.ssh/id_ed25519`` or ``~/.ssh/id_rsa`` is reused; if neither
    exists a new passwordless ed25519 key is generated.
    """
    ssh_dir = Path.home() / ".ssh"
    if key_path:
        key = Path(key_path).expanduser()
    else:
        for candidate in ("id_ed25519", "id_rsa"):
            existing = ssh_dir / candidate
            if existing.exists():
                return existing
        key = ssh_dir / "id_ed25519"

    if key.exists():
        return key

    if shutil.which("ssh-keygen") is None:
        raise RuntimeError("ssh-keygen not found; install openssh-client first.")
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    print(f"Generating a new SSH key at {key} ...")
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(key), "-N", "", "-q"],
        check=True,
    )
    return key


def _public_key_path(key: Path) -> Path:
    return key.with_name(key.name + ".pub")


def install_public_key(remote: dict, key: Path) -> dict:
    """Install the public key on the remote so SSH logins need no password.

    Tries ``ssh-copy-id`` first, then falls back to appending the key to the
    remote ``~/.ssh/authorized_keys`` over SSH. Either way this may prompt once
    for the remote password. Returns {"ok", "method", "detail"}.
    """
    pub = _public_key_path(key)
    if not pub.exists():
        return {"ok": False, "method": None, "detail": f"public key {pub} missing"}

    target = runner._ssh_target(remote)
    ssh_opts = runner._ssh_options(remote)

    if shutil.which("ssh-copy-id"):
        cmd = ["ssh-copy-id", "-i", str(pub), *ssh_opts, target]
        print(f"Installing key with: {' '.join(shlex.quote(c) for c in cmd)}")
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return {"ok": True, "method": "ssh-copy-id", "detail": f"key installed on {target}"}
        # fall through to manual method on failure

    # Manual fallback: pipe the public key into authorized_keys.
    pub_text = pub.read_text(encoding="utf-8")
    remote_cmd = (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        "cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    )
    cmd = ["ssh", *ssh_opts, target, remote_cmd]
    print(f"Installing key with: {' '.join(shlex.quote(c) for c in cmd)}")
    result = subprocess.run(cmd, input=pub_text, text=True)
    if result.returncode == 0:
        return {"ok": True, "method": "authorized_keys", "detail": f"key appended on {target}"}
    return {"ok": False, "method": "authorized_keys",
            "detail": f"could not install key on {target} (exit {result.returncode})"}


# --------------------------------------------------------------------- verify

def verify_remote(remote: dict) -> dict:
    """Check SSH + remote_root + VASP binary + scheduler. See runner."""
    return runner.check_remote_connection(remote)


def print_verification(report: dict) -> None:
    status = "OK" if report.get("ok") else "PROBLEMS"
    print(f"\nConnection check for {report.get('host')}: {status}")
    for check in report.get("checks", []):
        mark = "✓" if check["ok"] else "✗"
        print(f"  {mark} {check['name']}: {check['detail']}")
    if not report.get("checks"):
        print(f"  ✗ {report.get('message')}")


# ----------------------------------------------------------------- persistence

def _clean_entry(remote: dict) -> dict:
    """Keep only the persistable, non-empty fields of a remote config."""
    entry: dict = {}
    for field in SCALAR_FIELDS:
        value = remote.get(field)
        if value not in (None, ""):
            entry[field] = value
    for field in LIST_FIELDS:
        value = remote.get(field)
        if value:
            entry[field] = list(value)
    return entry


def save_to_config(remote: dict, name: str, config_path: str | Path | None = None) -> Path:
    """Add the machine under ``remotes: <name>:`` in config.yaml.

    If the file has no ``remotes:`` mapping yet (the common case — the template
    ships it commented out), a fresh block is *appended* so all existing
    comments are preserved. If a real ``remotes:`` mapping already exists, the
    whole file is re-dumped through PyYAML (comments in that region are lost).
    """
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG
    entry = _clean_entry(remote)

    data = {}
    text = ""
    if path.exists():
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}

    existing_remotes = data.get("remotes")
    if isinstance(existing_remotes, dict):
        existing_remotes[name] = entry
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    # No remotes mapping in the live data: append a block, preserving comments.
    block = yaml.safe_dump({"remotes": {name: entry}}, sort_keys=False)
    sep = "" if text.endswith("\n") or not text else "\n"
    addition = f"{sep}\n# Added by vasp-auto-setup-remote\n{block}"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(addition)
    return path


def save_to_ui_store(remote: dict, name: str, store_path: str | Path | None = None) -> Path:
    """Merge the machine into remotes.json (read by the web UI Remote tab)."""
    path = Path(store_path).expanduser() if store_path else REMOTES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        store: dict = {}
        if path.exists():
            try:
                store = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                store = {}
        previous = store.get(name) if isinstance(store.get(name), dict) else {}
        store[name] = {**previous, **_clean_entry(remote), "name": name}
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    return path


# ----------------------------------------------------------------- interactive

def _prompt(label: str, default: str | None = None, required: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"{label}{suffix}: ").strip()
        if not answer and default is not None:
            return default
        if answer or not required:
            return answer
        print("  (required)")


def _collect_interactive() -> tuple[str, dict, str | None]:
    print("== VASP-auto remote machine setup ==\n")
    name = _prompt("Machine name (for --remote NAME)", "cluster", required=True)
    remote: dict = {}
    remote["host"] = _prompt("Hostname or IP", required=True)
    remote["user"] = _prompt("Username (blank = use ~/.ssh/config)")
    remote["port"] = _prompt("SSH port", "22")
    if remote["port"] == "22":
        remote.pop("port")
    remote["remote_root"] = _prompt("Remote base dir for jobs", "/scratch/$USER/vasp_jobs",
                                    required=True)
    remote["vasp_executable"] = _prompt("VASP binary path ON the remote (blank for QE-only)")
    remote["qe_executable"] = _prompt("QE pw.x path ON the remote (blank for VASP-only)")
    if remote.get("qe_executable"):
        remote["pseudo_dir"] = _prompt("QE UPF directory ON the remote")
    remote["scheduler"] = _prompt("Scheduler (slurm/pbs)", "slurm")
    key_path = _prompt("SSH private key to use (blank = auto)") or None
    extra = _prompt("Extra submit-script lines (semicolon-separated, e.g. "
                    "'module load vasp')")
    if extra:
        remote["scheduler_options"] = [s.strip() for s in extra.split(";") if s.strip()]
    return name, {k: v for k, v in remote.items() if v not in (None, "")}, key_path


# ----------------------------------------------------------------------- main

def run_setup(name: str, remote: dict, *, key_path: str | None = None,
              install_key: bool = True, verify: bool = True,
              save_config: bool = True, save_ui: bool = True,
              config_path: str | None = None) -> dict:
    """Programmatic entry point. Returns a summary dict of what was done."""
    summary: dict = {"name": name, "steps": []}

    key = ensure_local_key(key_path)
    summary["key"] = str(key)
    summary["steps"].append(f"using SSH key {key}")
    # Record the key in the saved config only if the user picked a non-default one.
    if key_path:
        remote.setdefault("ssh_key", str(key))

    if install_key:
        result = install_public_key(remote, key)
        summary["key_install"] = result
        summary["steps"].append(result["detail"])
        print(("✓ " if result["ok"] else "✗ ") + result["detail"])

    if verify:
        report = verify_remote(remote)
        summary["verify"] = report
        print_verification(report)

    if save_config:
        path = save_to_config(remote, name, config_path)
        summary["config_path"] = str(path)
        summary["steps"].append(f"saved to {path} (remotes: {name})")
        print(f"✓ Saved machine '{name}' to {path}")
    if save_ui:
        path = save_to_ui_store(remote, name)
        summary["ui_store"] = str(path)
        summary["steps"].append(f"saved to {path}")
        print(f"✓ Saved machine '{name}' to {path} (web UI)")

    print(f"\nDone. Submit with:  vasp-auto <case> --remote {name}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vasp-auto-setup-remote",
        description="Set up SSH and register a remote machine for vasp-auto "
                    "--remote. With no machine flags it runs interactively.",
    )
    p.add_argument("--name", help="machine name used by --remote NAME")
    p.add_argument("--host", help="remote hostname or IP")
    p.add_argument("--user", help="remote username (default: ~/.ssh/config)")
    p.add_argument("--port", help="SSH port")
    p.add_argument("--remote-root", help="base directory for jobs on the remote")
    p.add_argument("--vasp-executable", help="VASP binary path on the remote")
    p.add_argument("--qe-executable", help="Quantum ESPRESSO pw.x path on the remote")
    p.add_argument("--pseudo-dir", help="UPF pseudopotential directory on the remote")
    p.add_argument("--qe-env-setup", help="shell setup used only for QE runs")
    p.add_argument("--scheduler", default=None, choices=["slurm", "pbs"],
                   help="remote scheduler (default slurm)")
    p.add_argument("--ssh-key", help="private key to use (default: auto/generate)")
    p.add_argument("--scheduler-option", action="append", default=[], dest="scheduler_options",
                   help="extra submit-script line (repeatable), e.g. 'module load vasp'")
    p.add_argument("--no-keygen", action="store_true",
                   help="do not generate a key if none exists (error instead)")
    p.add_argument("--no-install-key", action="store_true",
                   help="skip copying the public key to the remote")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the SSH/VASP/scheduler connection check")
    p.add_argument("--no-save-config", action="store_true",
                   help="do not write config.yaml")
    p.add_argument("--no-save-ui", action="store_true",
                   help="do not write remotes.json (web UI store)")
    p.add_argument("--config", help="config.yaml path to write (default: repo config.yaml)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.host:  # non-interactive
        name = args.name or args.host
        remote = {"host": args.host}
        for flag in ("user", "port", "remote_root", "vasp_executable", "qe_executable",
                     "pseudo_dir", "qe_env_setup", "scheduler", "ssh_key"):
            value = getattr(args, flag)
            if value:
                remote[flag] = value
        if args.scheduler_options:
            remote["scheduler_options"] = args.scheduler_options
        key_path = args.ssh_key
    else:
        if not sys.stdin.isatty():
            print("No --host given and not running interactively. "
                  "Pass --host (see --help) or run in a terminal.", file=sys.stderr)
            return 2
        name, remote, key_path = _collect_interactive()

    if not remote.get("remote_root") or not (
        remote.get("vasp_executable") or remote.get("qe_executable")
    ):
        print("A remote needs remote_root and at least one DFT executable.", file=sys.stderr)
        return 2

    if args.no_keygen and not key_path:
        # Only reuse an existing key; ensure_local_key would otherwise generate one.
        ssh_dir = Path.home() / ".ssh"
        if not any((ssh_dir / k).exists() for k in ("id_ed25519", "id_rsa")):
            print("No existing SSH key and --no-keygen set.", file=sys.stderr)
            return 2

    try:
        run_setup(
            name, remote,
            key_path=key_path,
            install_key=not args.no_install_key,
            verify=not args.no_verify,
            save_config=not args.no_save_config,
            save_ui=not args.no_save_ui,
            config_path=args.config,
        )
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
