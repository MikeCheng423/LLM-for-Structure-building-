# Tutorial 02 — Geometry Optimisation (Relaxation)

This tutorial relaxes the atomic positions and unit cell of **bulk silicon**
(the same POSCAR from Tutorial 01) until all forces fall below 0.02 eV/Å.
You will learn how to rerun a failed or unconverged calculation with
`--retry-failed`, how to pull the Markdown calculation report, and how to
inspect the ionic-step trajectory. The same workflow applies to any material
that needs atomic or cell optimisation before an SCF, DOS, or surface
calculation.

---

## Prerequisites

- Tutorial 01 completed (you have `inputs/Si_bulk/POSCAR`).
- vasp_auto installed; `config.yaml` set up.

---

## Step 1 — Run the relaxation

```bash
vasp-auto inputs/Si_bulk \
  --calc-type relax \
  --kpoints-mode gamma --kmesh 6 \
  --cpus 8
```

The `relax` calc type loads `example/INCAR_optimize_structure`:

```
IBRION = 2      # conjugate-gradient ionic optimiser
NSW    = 100    # up to 100 ionic steps
ISIF   = 3      # relax ions, cell shape, and cell volume
EDIFFG = -0.02  # force criterion 0.02 eV/Å
```

vasp_auto prints one line per ionic step (from `run.log`) and writes
`jobs/Si_bulk/Si_bulk/OUTCAR`, `CONTCAR`, `XDATCAR`, and `vasprun.xml`.

The Excel row in `jobs/Si_bulk/Si_bulk.xlsx` gains columns:

| Column | Meaning |
|--------|---------|
| `ionic_steps` | Number of steps taken |
| `max_force_eVA` | Final maximum force (should be < 0.02) |
| `converged` | TRUE if EDIFFG criterion was satisfied |
| `energy_eV` | Total energy of the fully relaxed cell |

---

## Step 2 — Write a Markdown report

```bash
vasp-auto inputs/Si_bulk --calc-type relax --report --parse-only
```

`--report` writes `jobs/Si_bulk/Si_bulk/report.md`. The file contains:

- Case name, calc type, POSCAR element list.
- INCAR highlights (ENCUT, EDIFFG, NSW, ISIF).
- Result summary: energy, max force, ionic steps, converged flag.

It is a plain Markdown file you can include in a lab notebook or commit to
version control as a record of the calculation.

---

## Step 3 — Rerun a failed or unconverged case

If VASP hits NSW=100 without reaching the force criterion, the `converged`
column will be FALSE. `--retry-failed` reruns only those cases:

```bash
vasp-auto inputs/Si_bulk \
  --calc-type relax \
  --kpoints-mode gamma --kmesh 6 \
  --cpus 8 \
  --retry-failed
```

What vasp_auto does internally:

1. Checks whether the existing job converged (`should_retry_failed`).
2. Stashes `CONTCAR`, `WAVECAR`, and `CHGCAR` from the old job directory
   in a temporary holding folder.
3. Cleans the job directory and writes fresh input files.
4. Restores the stashed files: **CONTCAR becomes the new POSCAR** (continuing
   from the last ionic step), WAVECAR seeds the SCF, CHGCAR is kept.
5. Relaunches VASP.

This is safe to repeat multiple times. Each pass continues from where the
previous one stopped.

> **Web UI equivalent.** In `vasp-auto-ui` the **Resume** button (Calculate
> tab, or the bulk-action checkboxes in the Results tab) restarts the latest
> job in place from its newest CONTCAR. It first asks for the CPU count
> (leave blank to reuse the previous run's), then pops up the job's INCAR as
> an editable tag/value spreadsheet — handy for raising `NSW` or loosening
> `EDIFFG` before the restart. *Save & resume* applies the edits and
> relaunches; cancel aborts.

To also apply a known VASP error fix automatically (e.g. ZBRENT → POTIM
reduction):

```bash
vasp-auto inputs/Si_bulk --calc-type relax --kmesh 6 --retry-failed --auto-retry 3
```

`--auto-retry 3` applies up to three auto-fix + re-run cycles per case
(local scheduler only).

---

## Step 4 — Inspect the trajectory (CONTCAR and XDATCAR)

After convergence, `CONTCAR` contains the fully relaxed structure. Copy it
back as the next calculation's POSCAR:

```bash
cp jobs/Si_bulk/Si_bulk/CONTCAR inputs/Si_relaxed/POSCAR
```

The full ionic-step trajectory lives in `XDATCAR`. vasp_auto's trajectory
module can export it as Cartesian frame data for the web UI's 3D viewer:

```python
from vasp_auto.trajectory import parse_xdatcar
frames = parse_xdatcar("jobs/Si_bulk/Si_bulk/XDATCAR")
# frames[i] = {"lattice": ..., "positions": [...], "elements": [...]}
```

The web UI (`vasp-auto-ui`) renders this animation automatically in the
Results tab — click a job row to open the trajectory viewer.

---

## Step 5 — Project mode: relax many structures at once

If `inputs/` contains multiple case subdirectories (each with its own
`POSCAR`), point vasp_auto at the project root:

```bash
vasp-auto inputs/ \
  --calc-type relax \
  --kmesh 6 \
  --cpus 8 \
  --parallel 4
```

`--parallel 4` runs up to four cases concurrently (each in its own `mpirun`
subprocess). A single `jobs/inputs.xlsx` is written with one row per case.

---

## Quick reference

```bash
# Geometry optimisation, 8 cores, 6x6x6 Gamma mesh:
vasp-auto inputs/Si_bulk --calc-type relax --kpoints-mode gamma --kmesh 6 --cpus 8

# Write Markdown report to jobs/Si_bulk/Si_bulk/report.md:
vasp-auto inputs/Si_bulk --parse-only --report

# Rerun only unconverged/failed cases, continuing from CONTCAR:
vasp-auto inputs/Si_bulk --calc-type relax --kmesh 6 --retry-failed --cpus 8

# Auto-fix VASP errors and retry up to 3 times:
vasp-auto inputs/Si_bulk --calc-type relax --kmesh 6 --auto-retry 3 --cpus 8
```

---

**Next**: Tutorial 03 shows how to pick a converged ENCUT, SIGMA, and k-mesh
from scratch using the automated convergence scan.
