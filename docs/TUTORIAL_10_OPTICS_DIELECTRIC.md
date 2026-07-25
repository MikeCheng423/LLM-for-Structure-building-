# Tutorial 10 — Optical Absorption and the Dielectric Function

The frequency-dependent dielectric function ε(ω) gives access to the optical
absorption spectrum, refractive index, and static dielectric constant of a
material. This tutorial computes the optical absorption coefficient α(E) of
**bulk silicon** in the independent-particle approximation using VASP's
`LOPTICS = .TRUE.` tag, then extracts the absorption spectrum and static
ε(0) with vasp_auto's `--optics-parse` post-processor.

---

## Prerequisites

- A converged SCF calculation on the system of interest (the WAVECAR is
  reused to seed the optics run).
- Tutorial 01 or Tutorial 03 completed for bulk Si.
- `config.yaml` pointing to the VASP binary and POTCAR library.

---

## Step 1 — Run a tight SCF first (or reuse an existing one)

The optics calculation is a static non-SCF run on the fixed charge density,
so you need a well-converged WAVECAR from a prior SCF:

```bash
vasp-auto inputs/Si_bulk \
  --calc-type scf \
  --kpoints-mode gamma --kmesh 8 \
  --cpus 8
```

The SCF template writes `LWAVE = .TRUE.`, so a `WAVECAR` will exist in
`jobs/Si_bulk/Si_bulk/`.

---

## Step 2 — Set up the optics case

Copy the SCF results into an optics input directory:

```bash
mkdir -p inputs/Si_optics
cp inputs/Si_bulk/POSCAR inputs/Si_optics/POSCAR
cp jobs/Si_bulk/Si_bulk/WAVECAR inputs/Si_optics/WAVECAR
```

---

## Step 3 — Run the optics calculation

```bash
vasp-auto inputs/Si_optics \
  --calc-type optics \
  --kpoints-mode gamma --kmesh 8 \
  --cpus 8
```

The `optics` template (`example/INCAR_optics`) sets:

```
LOPTICS = .TRUE.   # frequency-dependent dielectric function
NBANDS  = 64       # many empty states needed for optical matrix elements
NEDOS   = 2000     # dense frequency grid
CSHIFT  = 0.1      # Lorentzian broadening (eV)
```

Keep `--kmesh` the same as (or denser than) the SCF step. The run re-reads
the WAVECAR and computes the optical matrix elements; typical wall time is
2–4× the SCF run.

Results are stored in `vasprun.xml` (the full dielectric tensor as a function
of energy).

---

## Step 4 — Extract the absorption spectrum

```bash
vasp-auto --optics-parse jobs/Si_optics/Si_optics
```

Output to the terminal:

```
epsilon(0) : 12.3456 (direction-averaged static dielectric constant)
Wrote     : jobs/Si_optics/Si_optics/absorption.csv
```

`absorption.csv` has four columns:

| Column | Meaning |
|--------|---------|
| `energy_eV` | Photon energy in eV |
| `alpha_cm1` | Absorption coefficient α in cm⁻¹ |
| `epsilon_real` | Re[ε(ω)], real part of the dielectric function |
| `epsilon_imag` | Im[ε(ω)], imaginary part |

The absorption coefficient is derived from the imaginary part of the
direction-averaged dielectric function:

```
α(E) = (2ω/c) · Im[n(ω)]   where n = sqrt(ε)
```

---

## Step 5 — Read the static dielectric constant

The value `epsilon(0)` printed above is the real part of ε at zero frequency,
direction-averaged. For silicon, the GGA value is approximately 12.4 (slightly
overestimated; the experimental value is 11.7). For comparison:

| Material | ε(0) GGA | Experimental |
|----------|----------|-------------|
| Si | ~12.4 | 11.7 |
| Rutile TiO₂ (‖ to c) | ~7.5 | ~8.3 |
| GaAs | ~14 | 12.9 |

---

## Step 6 — Workflow: SCF → optics in one command

```bash
vasp-auto inputs/Si_bulk \
  --workflow "scf,optics" \
  --kmesh 8 \
  --cpus 8
```

The `optics` step automatically gets the WAVECAR from the SCF step via the
chain inputs (`CHAIN_INPUTS[CalcType.OPTICS] = {"CONTCAR": "POSCAR",
"WAVECAR": "WAVECAR"}`).

After the workflow:

```bash
vasp-auto --optics-parse jobs/Si_bulk/Si_bulk/02_optics
```

---

## Notes on accuracy

The independent-particle approximation (IPA) used by VASP's `LOPTICS`
underestimates the optical gap and overestimates ε(0) for most semiconductors.
For more accurate spectra:

- Use HSE06 (`--calc-type hse06`) for the self-consistent ground state.
- The GW method or the Bethe–Salpeter equation (BSE) are beyond the scope of
  vasp_auto's workflow engine; post-process with your preferred code.
- Increasing `NBANDS` and `NEDOS` improves convergence.

---

## Quick reference

```bash
# Run optics on an existing WAVECAR:
vasp-auto inputs/Si_optics --calc-type optics --kmesh 8 --cpus 8

# Chained SCF → optics:
vasp-auto inputs/Si_bulk --workflow "scf,optics" --kmesh 8 --cpus 8

# Extract absorption.csv and print epsilon(0):
vasp-auto --optics-parse jobs/Si_optics/Si_optics

# Parse a workflow step:
vasp-auto --optics-parse jobs/Si_bulk/Si_bulk/02_optics
```

---

**Next**: Tutorial 11 shows how to build supercells and introduce point defects
(vacancy, substitution, interstitial) for defect-physics calculations.
