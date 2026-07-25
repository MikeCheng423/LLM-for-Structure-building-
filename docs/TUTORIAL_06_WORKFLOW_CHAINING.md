# Tutorial 06 — Workflow Chaining

vasp_auto can run any ordered sequence of calculation types in a single
command, automatically forwarding CONTCAR, CHGCAR, and WAVECAR between steps.
This tutorial builds a full **converge → relax → scf → dos** pipeline for
bulk silicon — the standard multi-step order — and
shows all three ways to specify a workflow: CLI flag, per-case `workflow.yaml`,
and project-wide `config.yaml`.

---

## Prerequisites

- `inputs/Si_bulk/POSCAR` (Tutorial 01).
- `config.yaml` with `vasp_executable` and `potcar_root`.

---

## Step 1 — One-liner workflow

```bash
vasp-auto inputs/Si_bulk \
  --workflow "converge,relax,scf,dos" \
  --converge-encut 400,450,500,520 \
  --converge-sigma 0.2,0.1,0.05 \
  --reuse-wavecar \
  --kpoints-values "4,6,8,10" \
  --cpus 8
```

vasp_auto executes five stages in the job directory
`jobs/Si_bulk/Si_bulk/`:

```
scf_convergence/   (converge step: ENCUT + SIGMA + NELM + KPOINTS scans)
01_relax/          INCAR_optimize_structure; gets ENCUT/SIGMA/KPOINTS from scan
02_scf/            INCAR_scf; CONTCAR from 01_relax
03_dos/            INCAR_dos; CHGCAR from 02_scf, CONTCAR from 02_scf
```

The `converge` step carries the selected ENCUT, SIGMA, NELM, and KPOINTS into
every later step automatically. A per-step `incar:` override in `workflow.yaml`
still wins over the carried values.

---

## Step 2 — Per-case workflow.yaml

For more control, place a `workflow.yaml` inside the case directory:

```yaml
# inputs/Si_bulk/workflow.yaml
steps:
  - calc_type: converge
    encut: "400,450,500,520"
    sigma: "0.2,0.1,0.05"
    kpoints: "4,6,8,10"
    reuse_wavecar: true
    energy_tol: 1e-4

  - calc_type: relax
    kpoints: "8"
    incar:
      ISIF: "3"
      NSW: "200"

  - calc_type: scf
    kpoints: "8"
    incar:
      LCHARG: ".TRUE."
      LWAVE: ".TRUE."

  - calc_type: dos
    kpoints: "12"
    incar:
      NEDOS: "4000"
      ISMEAR: "-5"
```

Then run without any `--workflow` flag:

```bash
vasp-auto inputs/Si_bulk --cpus 8
```

vasp_auto detects `workflow.yaml` automatically. Per-step `kpoints:` overrides
the carried k-mesh; per-step `incar:` keys are merged into the template.

---

## Step 3 — Project-wide workflow in config.yaml

To apply the same chain to every calculation in a project, add a `workflow:`
block to `config.yaml` (or to a project-level `inputs/config.yaml`):

```yaml
workflow: "relax,scf,dos"
```

or, for the full pipeline with defaults:

```yaml
workflow: "converge,relax,scf,dos"
```

Every case in `inputs/` will run through the chain unless it has its own
`workflow.yaml` (case file wins over config file).

---

## Step 4 — Dry-run the workflow

Before committing CPU time you can preview what each step would write:

```bash
vasp-auto inputs/Si_bulk \
  --workflow "relax,scf,dos" \
  --dry-run
```

Output:

```
Workflow  : relax, scf, dos (dry run, nothing written)
```

For full per-step previews, run each step individually with `--dry-run` and the
appropriate `--calc-type`.

---

## Step 5 — Precedence rules

Priority (highest → lowest) for INCAR/KPOINTS settings within a workflow step:

1. A `INCAR` file in the case directory (fully manual — vasp_auto uses it as-is).
2. Per-step `incar:` keys in `workflow.yaml`.
3. Carried values from the `converge` step (ENCUT, SIGMA, NELM, KPOINTS).
4. CLI flags (`--kmesh`, `--kpoints-mode`, …).
5. The `example/INCAR_<type>` template.

---

## Step 6 — Chaining across multiple input cases

Point vasp_auto at a project directory to chain every case:

```bash
vasp-auto inputs/ --workflow "relax,scf,dos" --kmesh 8 --cpus 8 --parallel 2
```

`--parallel 2` runs two cases at a time. The summary Excel lands at
`jobs/inputs.xlsx` with one row per workflow step per case.

---

## Quick reference

```bash
# CLI workflow:
vasp-auto inputs/Si_bulk --workflow "converge,relax,scf,dos" \
  --converge-encut 400,450,500,520 --converge-sigma 0.2,0.1,0.05 \
  --reuse-wavecar --kpoints-values "4,6,8,10" --cpus 8

# Simple relax → scf → dos (no convergence step):
vasp-auto inputs/Si_bulk --workflow "relax,scf,dos" --kmesh 8 --cpus 8

# Project-wide: run every case through the same chain:
vasp-auto inputs/ --workflow "relax,scf,dos" --kmesh 8 --parallel 4 --cpus 8

# Preview without running:
vasp-auto inputs/Si_bulk --workflow "relax,scf,dos" --dry-run
```

---

**Next**: Tutorial 07 walks through a NEB transition-state search for an atomic
diffusion barrier.
