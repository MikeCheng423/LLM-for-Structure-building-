# vasp_auto on Linux, macOS, and Windows

Status and plan for running the tool on all three platforms (plan.txt item 10).

## What is already portable

- **Pure Python 3.12+** engine: `pathlib.Path` everywhere, no `os.path.join`,
  no platform-specific syscalls. Dependencies (PyYAML, pandas, openpyxl,
  optional ASE) all ship wheels for the three platforms.
- **Web UI**: stdlib `http.server`, binds 127.0.0.1, opens the default
  browser via `webbrowser` — works identically everywhere.
- **MPI launcher**: `mpirun` by default; the `VASP_AUTO_MPI` environment
  variable overrides it (e.g. `mpiexec` for MS-MPI or Intel MPI) without
  touching the code.
- **More MPI ranks than cores**: Open MPI counts *physical cores* as slots, so
  asking for more ranks than cores aborts with "not enough slots". The runner
  detects this and keeps the job running: it adds `--use-hwthread-cpus` when the
  requested ranks still fit the machine's hardware threads (e.g. 16 ranks on an
  8-core/16-thread CPU — no genuine oversubscription), or `--oversubscribe` when
  they exceed even the threads. A warning naming the physical-core count is
  written to `run.log` (and streamed to the UI log). Override with
  `VASP_AUTO_OVERSUBSCRIBE=1` (always force `--oversubscribe`) or `=0` (never add
  a flag — let Open MPI enforce its slot limit). For best performance, set CPU
  cores to the physical-core count. These flags are OpenMPI-specific and skipped
  for other launchers (Intel/MS-MPI have no slot concept).
- **No symlinks, no `chmod`-dependent behaviour** in the engine path
  (`submit.sh` gets exec bits, which Windows ignores harmlessly — scheduler
  submission is a cluster/Linux feature anyway).

## Per-platform guidance

### Linux (primary platform)
Works as documented in MANUAL.md. CI runs the full pytest + selfcheck here.

### macOS
1. `python3.12 -m venv venv && venv/bin/pip install -e .[ase]`
2. `brew install open-mpi` (provides `mpirun`); compile VASP or point
   `vasp_executable` at an existing build.
3. Everything else (UI, Excel, reports, animations) is platform-independent.

### Windows
**Recommended: WSL2.** Inside WSL the tool behaves exactly like Linux,
including `mpirun` and the bash selfcheck; the UI is reachable from the
Windows browser at `http://127.0.0.1:8800/` (WSL forwards localhost).

Native Windows (without WSL):
1. Install Python 3.12 from python.org, then `pip install -e .[ase]`.
2. Install MS-MPI and set `VASP_AUTO_MPI=mpiexec`; point `vasp_executable`
   at a Windows VASP build (or a remote scheduler).
3. Known gaps on native Windows:
   - `selfcheck/run_selfcheck.sh` needs bash (use Git Bash or WSL).
   - SLURM/PBS submission assumes a POSIX cluster (normally remote anyway).
   - The legacy `vasp_auto` shell launcher is POSIX-only; use the
     `vasp-auto` console script, which pip generates as a native .exe shim.

## Remaining plan (in priority order)

1. **CI matrix** — extend `.github/workflows/ci.yml` with
   `os: [ubuntu-latest, macos-latest, windows-latest]` for the pytest job
   (selfcheck stays Linux-only until item 3).
2. **Config-level launcher** — promote `VASP_AUTO_MPI` to a first-class
   `mpi_command:` key in config.yaml once a real Windows/macOS user needs it.
3. **Python selfcheck runner** — port `run_selfcheck.sh` to a small Python
   script so the end-to-end check runs natively on all three platforms.
4. **Packaged distribution** — `pipx`-installable release (or PyInstaller
   bundle for the UI) so non-Python users can install with one command.
