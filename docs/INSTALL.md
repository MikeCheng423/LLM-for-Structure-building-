# Installation: local control machine vs remote compute node

vasp-auto is one Python package (`vasp-auto`) whose **core install is the lean
engine** that runs VASP. Everything heavier lives in pip *extras*, so each machine
installs only what it needs.

## The two-machine model

| Machine | Role | Install |
|---|---|---|
| **Control machine** (laptop/workstation) | Web UI, structure builders, results/Excel, SSH orchestration of remote jobs | `pip install -e ".[local]"` |
| **Remote compute node** (HPC login/compute) | Lean engine — just runs VASP for offloaded jobs | `pip install -e .` (or `".[remote]"`) |

## Extras matrix

| Extra | Pulls in | Use it when |
|---|---|---|
| *(core, no extra)* | `PyYAML` | Remote compute node — engine only, runs VASP. |
| `remote` | *(nothing; core is enough)* | Same as core; named so `pip install vasp-auto[remote]` reads clearly. |
| `results` | `pandas`, `openpyxl` | You want Excel summaries of finished jobs. |
| `builder` | `ase` | Structure builders (CLI build commands, the web Builder, the AI builder) **and the `engine: ase` calculator backend** (scf/relax via any ASE calculator). |
| `local` | `pandas`, `openpyxl`, `ase` | **Full control machine** — UI + builders + results. |
| `ml` | `ase`, `fairchem-core` | ML pre-relax (OMat24 / UMA) before VASP. |
| `dev` | `pytest` + the heavy deps | Running the test suite. |
| `ase` | `ase` | Back-compat alias for the old `[ase]` extra. |

The web UI server itself is pure standard library (`http.server`), so it adds no
runtime dependency of its own — `[local]` is heavy only because of the builders
(`ase`) and results parsing (`pandas`/`openpyxl`).

## Why the engine is lean

The remote-offload path (`vasp-auto-setup-remote`, or the UI Remote tab) builds an
engine **wheel** on the control machine, ships it over SSH, and runs
`pip install <wheel>` inside a fresh venv on the remote. That install resolves only
the **core** dependency (`PyYAML`) — so a compute node never pulls `pandas`,
`openpyxl`, or `ase`. The CLI imports `pandas` lazily (only when writing an Excel
summary), so the engine imports cleanly without it.

## Setting up a remote machine

Easiest — from the control machine, let the tool do the SSH key, connection check,
and (optionally) engine install for you:

```bash
vasp-auto-setup-remote            # interactive; saves to config.yaml + remotes.json
```

Or add the machine in the **UI → Remote tab**. Either way you can then run:

```bash
vasp-auto inputs/Fe --remote NAME            # submit over SSH
```

For fully detached ("offload") runs where your laptop can power off, the remote
needs the engine venv — installed once via the setup tool's remote-engine step (or
`pip install -e .` by hand on the remote).

## Configuration files

```bash
cp config.yaml.example config.yaml
cp remotes.json.example remotes.json     # optional; the UI can manage machines instead
```

Both real files are git-ignored. `config.yaml` holds your `vasp_executable`,
`jobs_root`, and `potcar_root`; `remotes.json` holds per-machine SSH/scheduler
details. **POTCAR pseudopotentials are not shipped** — point `potcar_root` at your
own licensed VASP library.
