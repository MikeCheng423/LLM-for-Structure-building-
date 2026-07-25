# Tutorial 03 — Numerical Convergence: ENCUT, SIGMA, NELM, and k-mesh

Before any production DFT study you must verify that results are not
sensitive to the numerical parameters — plane-wave cutoff (ENCUT), smearing
width (SIGMA), self-consistency cycle limit (NELM), and k-point density.
This tutorial demonstrates vasp_auto's automated convergence scanner on
**bulk silicon**, starting from an unrelaxed POSCAR. The scanner runs every
stage in sequence (ENCUT → SIGMA → NELM → KPOINTS), selects the cheapest
parameter set whose energy is within your tolerance, and writes a human-
readable Markdown report plus a CSV you can plot.

---

## Prerequisites

- `inputs/Si_bulk/POSCAR` (see Tutorial 01 for the POSCAR or the builder
  command).
- `config.yaml` configured with `vasp_executable`, `potcar_root`, etc.

---

## Step 1 — Full convergence scan

```bash
vasp-auto inputs/Si_bulk \
  --converge-encut 400,450,500,520,550 \
  --converge-sigma 0.2,0.1,0.05,0.02 \
  --converge-scf \
  --kpoints-values "4,6,8,10,12" \
  --reuse-wavecar \
  --cpus 8
```

Flag-by-flag explanation:

| Flag | What it does |
|------|-------------|
| `--converge-encut 400,450,500,520,550` | Runs five single-point SCF calculations with increasing ENCUT. Selects the first ENCUT for which `|E_N - E_{N-1}| ≤ energy_tol` (default 1e-4 eV). |
| `--converge-sigma 0.2,0.1,0.05,0.02` | Scans smearing widths. Selects the **largest** SIGMA whose entropy term T\*S per atom is below `--sigma-tol` (default 1e-3 eV/atom). |
| `--converge-scf` | Also scans NELM (self-consistency loop limit) and KPOINTS at the end. |
| `--kpoints-values "4,6,8,10,12"` | K-mesh values for the KPOINTS scan (uniform Gamma meshes). |
| `--reuse-wavecar` | Seeds each trial with the WAVECAR from the previous trial, cutting wall time by 30–50 %. |
| `--energy-tol` (optional) | Energy-difference stopping criterion in eV. Default: 1e-4. |
| `--sigma-tol` (optional) | Entropy tolerance in eV/atom for the SIGMA stage. Default: 1e-3. |

The scan stages are independent and sequential:

1. **ENCUT** — selected value is held fixed for all later stages.
2. **SIGMA** — largest smearing with T\*S/atom below `--sigma-tol`.
3. **NELM** — minimum SCF iterations that converge the energy.
4. **KPOINTS** — coarsest mesh within energy tolerance.

Trial jobs land under `jobs/Si_bulk/Si_bulk/scf_convergence/` in separate
subdirectories (`encut_400/`, `encut_450/`, …, `kpoints_4/`, `kpoints_6/`, …).

---

## Step 2 — Inspect the convergence report

After the scan, vasp_auto prints a summary to the terminal:

```
Selected  : ENCUT=520, SIGMA=0.05, NELM=60, KPOINTS=8
Report    : jobs/Si_bulk/Si_bulk/scf_convergence/convergence_report.md
```

Read the Markdown report:

```bash
cat jobs/Si_bulk/Si_bulk/scf_convergence/convergence_report.md
```

The report lists every trial with energy and T\*S, highlights the selected
value, and gives the energy difference to the next trial. A companion CSV
(`convergence_results.csv`) has columns `stage, value, energy_eV, ts_eV` for
easy plotting.

Also check `jobs/Si_bulk/Si_bulk.xlsx` — the `selected_encut`, `selected_sigma`,
`selected_nelm`, and `selected_kpoints` columns record the scan outcome.

---

## Step 3 — Use the converged settings for production runs

Once you know the converged parameters, use them explicitly:

```bash
# SCF with the settings found above:
vasp-auto inputs/Si_bulk \
  --calc-type scf \
  --kpoints-mode gamma --kmesh 8 \
  --cpus 8
```

Or pipe the whole thing as a chained workflow (see Tutorial 06):

```bash
vasp-auto inputs/Si_bulk \
  --workflow "converge,relax,scf,dos" \
  --converge-encut 400,450,500,520,550 \
  --converge-sigma 0.2,0.1,0.05 \
  --reuse-wavecar \
  --cpus 8
```

The `converge` step carries the selected ENCUT/SIGMA/NELM/KPOINTS into all
later steps automatically.

---

## Step 4 — Quick ENCUT-only scan (minimum viable test)

If you only want to check the ENCUT convergence without touching SIGMA or
k-points:

```bash
vasp-auto inputs/Si_bulk \
  --converge-encut 400,450,500,520 \
  --kmesh 6 \
  --cpus 8
```

Without `--converge-scf`, only the ENCUT stage runs; SIGMA and KPOINTS remain
at the template defaults.

---

## Step 5 — SIGMA scan on a metal

Metals use larger smearing widths. For fcc Cu (or any metal), widen the scan
range and tighten the entropy tolerance:

```bash
vasp-auto inputs/Cu_bulk \
  --converge-encut 400,450 \
  --converge-sigma 0.4,0.3,0.2,0.1 \
  --sigma-tol 2e-3 \
  --reuse-wavecar --cpus 8
```

The largest SIGMA for which T\*S/atom ≤ 2 meV is selected. Typical metals
converge around SIGMA = 0.2 eV.

---

## Quick reference

```bash
# Full four-stage scan (ENCUT → SIGMA → NELM → KPOINTS):
vasp-auto inputs/Si_bulk \
  --converge-encut 400,450,500,520 \
  --converge-sigma 0.2,0.1,0.05 \
  --converge-scf --kpoints-values "4,6,8,10" \
  --reuse-wavecar --cpus 8

# ENCUT only (fast check):
vasp-auto inputs/Si_bulk --converge-encut 400,450,500 --kmesh 6 --cpus 8

# Read the outcome report:
cat jobs/Si_bulk/Si_bulk/scf_convergence/convergence_report.md

# Tighter tolerances for publication-quality work:
vasp-auto inputs/Si_bulk \
  --converge-encut 400,450,500,520,550 \
  --converge-sigma 0.2,0.1,0.05,0.02 \
  --energy-tol 1e-5 --sigma-tol 5e-4 \
  --converge-scf --reuse-wavecar --cpus 8
```

---

**Next**: Tutorial 04 demonstrates the full relax → SCF → DOS → bands chain
and shows how to export the density of states and band structure to CSV.
