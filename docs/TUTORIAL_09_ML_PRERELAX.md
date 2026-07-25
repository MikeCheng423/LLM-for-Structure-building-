# Tutorial 09 — Machine-Learning Pre-Relaxation and Energy Screening

Running VASP for 100+ ionic steps on a rough initial geometry is expensive.
Machine-learning interatomic potentials (MLIPs) can pre-relax a structure in
seconds on CPU, producing a geometry so close to the DFT minimum that VASP
needs only a handful of ionic steps to finish. This tutorial demonstrates
vasp_auto's MLIP integration for **bulk Si and a Pt(111) slab**:

1. A fast single-point MLIP energy screen (`--ml-energy`) to triage a set
   of candidate structures before spending VASP time.
2. Full MLIP pre-relaxation (`--ml-relax`) followed by a VASP geometry
   optimisation.

vasp_auto supports the **Meta OMat24 / UMA model** via `fairchem-core` (the
recommended production MLIP) and the **ASE EMT** demo potential (no download
required, only for testing the pipeline).

---

## Prerequisites

```bash
# Core vasp_auto (no ML):
pip install -e .

# Full Meta UMA / OMat24 models (GPU recommended):
pip install -e .[ml]     # installs fairchem-core
```

For a quick pipeline test without downloading any model, use `--ml-model emt`.
This invokes the ASE Effective Medium Theory potential — it gives qualitatively
wrong energies but exercises every code path.

---

## Step 1 — Single-point MLIP energy screen

`--ml-energy` reads a POSCAR (or a case directory containing one) and prints
the MLIP energy and maximum atomic force without writing any files:

```bash
vasp-auto --ml-energy inputs/Si_bulk --ml-model emt
```

Output:

```
ML energy : -12.345678 eV
max|F|    :  0.0123 eV/Å
model     : emt
```

Screen a whole batch in a loop:

```bash
for d in inputs/*/; do
    echo "=== $d ==="
    vasp-auto --ml-energy "$d" --ml-model emt
done
```

To use the production OMat24 model (requires fairchem-core and a GPU):

```bash
vasp-auto --ml-energy inputs/Si_bulk --ml-model uma-s-1p1 --ml-task omat
```

`--ml-task` selects the UMA task head:

| Task | Use case |
|------|----------|
| `omat` | Inorganic bulk materials (default) |
| `oc20` | Adsorbates on metal surfaces (catalysis) |
| `omol` | Molecules |
| `odac` | Metal–organic frameworks |

---

## Step 2 — MLIP pre-relaxation + VASP refinement

```bash
# Step 2a: pre-relax with the EMT demo potential (or uma-s-1p1 for real work):
vasp-auto inputs/Si_bulk \
  --ml-relax \
  --ml-model emt \
  --ml-fmax 0.05 \
  --ml-steps 200

# Step 2b: VASP geometry optimisation starting from the ML-relaxed geometry:
vasp-auto inputs/Si_bulk_mlrelax \
  --calc-type relax \
  --kpoints-mode gamma --kmesh 6 \
  --cpus 8
```

What `--ml-relax` does:

1. Reads `inputs/Si_bulk/POSCAR`.
2. Runs an ASE BFGS relaxation driven by the MLIP until `max|F| ≤ --ml-fmax`
   or `--ml-steps` is reached.
3. Writes the relaxed geometry to `inputs/Si_bulk_mlrelax/POSCAR` (a derived
   case directory with `_mlrelax` suffix).
4. Prints the final energy, max force, and step count.

Flag details:

| Flag | Meaning |
|------|---------|
| `--ml-relax` | Run MLIP pre-relaxation before any VASP calculation. |
| `--ml-model emt` | MLIP model name. Use `uma-s-1p1` for production work. |
| `--ml-fmax 0.05` | Force convergence in eV/Å. Default: 0.05. |
| `--ml-steps 200` | Maximum MLIP ionic steps. Default: 200. |
| `--ml-relax-cell` | Also relax the unit cell (shape + volume) during ML pre-relax. |

---

## Step 3 — Pre-relax + VASP in one command

```bash
vasp-auto inputs/Si_bulk \
  --ml-relax \
  --ml-model emt \
  --calc-type relax \
  --kmesh 6 \
  --cpus 8
```

Without `--ml-only`, vasp_auto continues from the ML-relaxed geometry into
VASP automatically. The VASP relaxation starts from a geometry already close
to the minimum, so it converges in far fewer steps.

---

## Step 4 — MLIP-only: stop after pre-relaxation

```bash
vasp-auto inputs/Si_bulk \
  --ml-relax \
  --ml-only \
  --ml-model emt \
  --ml-relax-cell
```

`--ml-only` stops after writing the relaxed POSCAR — no VASP is run. Useful
for generating an ensemble of initial structures for a parameter sweep.

---

## Step 5 — Use a custom model checkpoint

If you have downloaded a specific fairchem checkpoint (e.g. the eqV2 OMat24
model from the Meta AI model hub):

```bash
vasp-auto inputs/Si_bulk \
  --ml-relax \
  --ml-checkpoint /path/to/eqV2_OMat24.pt \
  --ml-fmax 0.01 \
  --cpus 8
```

`--ml-checkpoint` overrides `--ml-model`.

---

## Quick reference

```bash
# Single-point MLIP energy screen (no VASP, no files written):
vasp-auto --ml-energy inputs/Si_bulk --ml-model emt

# MLIP pre-relax only (write relaxed POSCAR, skip VASP):
vasp-auto inputs/Si_bulk --ml-relax --ml-only --ml-model emt

# MLIP pre-relax + VASP relax in sequence:
vasp-auto inputs/Si_bulk --ml-relax --ml-model emt \
          --calc-type relax --kmesh 6 --cpus 8

# Relax cell shape/volume with MLIP too:
vasp-auto inputs/Si_bulk --ml-relax --ml-relax-cell --ml-model uma-s-1p1

# Use a local checkpoint:
vasp-auto inputs/Si_bulk --ml-relax --ml-checkpoint /data/models/eqV2_OMat24.pt
```

---

**Next**: Tutorial 10 covers optical absorption spectra with `--calc-type
optics` and the `--optics-parse` post-processor.
