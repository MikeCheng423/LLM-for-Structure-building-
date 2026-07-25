# Tutorial 11 — Point Defects: Supercells, Vacancies, Substitutions, and Interstitials

Point defects (vacancies, substitutional impurities, interstitials) dominate
carrier lifetimes in semiconductors, dopability in oxides, and catalytic
activity at defect sites. DFT defect calculations require large supercells so
the defect does not interact with its periodic image. This tutorial builds a
**2×2×2 silicon supercell** from the two-atom primitive cell and introduces
three different types of point defect using vasp_auto's pure-Python structure
tools — no ASE required.

---

## Prerequisites

- `inputs/Si_bulk/POSCAR` (Tutorial 01).
- `config.yaml` with `vasp_executable` and `potcar_root`.

---

## Step 1 — Build a 2×2×2 supercell

```bash
vasp-auto inputs/Si_bulk \
  --supercell 2x2x2 \
  --build-only
```

vasp_auto reads `inputs/Si_bulk/POSCAR` (8 atoms), tiles it into a 2×2×2
repeat (64 atoms), and writes the result to `inputs/Si_bulk_sc2x2x2/POSCAR`.
`--build-only` stops before preparing or running VASP.

Check the output:

```bash
head -5 inputs/Si_bulk_sc2x2x2/POSCAR
# Should read: Si bulk ... with 64 Si atoms
```

---

## Step 2 — Create a silicon vacancy (V_Si)

Remove atom 1 (the first Si in POSCAR order, 1-based index):

```bash
vasp-auto inputs/Si_bulk_sc2x2x2 \
  --vacancy 1 \
  --build-only
```

Output: `inputs/Si_bulk_sc2x2x2_vac1/POSCAR` with 63 atoms.

Relax the defect supercell:

```bash
vasp-auto inputs/Si_bulk_sc2x2x2_vac1 \
  --calc-type relax \
  --kpoints-mode gamma --kmesh 2x2x2 \
  --cpus 8
```

Use a coarser k-mesh than the primitive cell (the supercell is larger, so
fewer k-points are needed for equivalent sampling density).

---

## Step 3 — Substitute a silicon atom with carbon (Si:C)

Replace atom 32 (near the centre of the supercell) with carbon:

```bash
vasp-auto inputs/Si_bulk_sc2x2x2 \
  --substitute 32=C \
  --build-only
```

Output: `inputs/Si_bulk_sc2x2x2_sub32C/POSCAR` with one C at position 32.

The argument format is `INDEX=ELEMENT` (1-based, POSCAR order). The POTCAR is
built automatically from the updated element list (Si 63-atom + C 1-atom).

---

## Step 4 — Add a hydrogen interstitial (H_i)

Place a hydrogen atom at the tetrahedral interstitial site (0.125, 0.125,
0.125 in fractional coordinates of the 2×2×2 supercell):

```bash
vasp-auto inputs/Si_bulk_sc2x2x2 \
  --interstitial "H@0.125,0.125,0.125" \
  --build-only
```

Equivalently, use the `--add-atom` alias:

```bash
vasp-auto inputs/Si_bulk_sc2x2x2 \
  --add-atom "H@0.125,0.125,0.125" \
  --build-only
```

Output: `inputs/Si_bulk_sc2x2x2_intH/POSCAR` with 65 atoms (64 Si + 1 H).
The coordinate is in fractional (POSCAR Direct) mode.

---

## Step 5 — Chain structure building with relaxation

You can omit `--build-only` to proceed directly to VASP after building:

```bash
vasp-auto inputs/Si_bulk_sc2x2x2 \
  --vacancy 1 \
  --calc-type relax \
  --kpoints-mode gamma --kmesh 2x2x2 \
  --cpus 8
```

vasp_auto builds the defect structure in a derived directory and then runs the
relaxation on it without any additional steps.

---

## Step 6 — Compute the defect formation energy

The vacancy formation energy is:

```
E_f(V_Si) = E(Si_63) − E(Si_64) + μ_Si
```

where μ_Si is the chemical potential of Si (the energy per atom in bulk Si).
Extract the energies from the Excel summaries:

```bash
vasp-auto inputs/Si_bulk_sc2x2x2     --calc-type scf --kmesh 2x2x2 --cpus 8
vasp-auto inputs/Si_bulk_sc2x2x2_vac1 --calc-type scf --kmesh 2x2x2 --cpus 8
```

Then read `energy_eV` from the relevant rows of `jobs/…/<case>.xlsx` and
apply the formula above. For charged defects add a correction term (Freysoldt
or Lany–Zunger) — this is handled by external codes such as `pydefect`.

---

## Step 7 — Building more complex defect configurations

Combine structure operations (order matters — each writes a new derived dir):

```bash
# 3x3x1 supercell → vacancy at index 5 → relax:
vasp-auto inputs/Si_bulk \
  --supercell 3x3x1 \
  --vacancy 5 \
  --calc-type relax \
  --kmesh 2x2x4 \
  --cpus 8
```

All pure-Python operators (`--supercell`, `--vacancy`, `--substitute`,
`--interstitial`, `--adsorbate`, `--freeze`, `--scale-cell`) compose in the
order listed on the command line.

---

## Quick reference

```bash
# 2x2x2 supercell:
vasp-auto inputs/Si_bulk --supercell 2x2x2 --build-only

# Si vacancy at atom 1:
vasp-auto inputs/Si_bulk_sc2x2x2 --vacancy 1 --build-only

# Substitution (Si→C at atom 32):
vasp-auto inputs/Si_bulk_sc2x2x2 --substitute 32=C --build-only

# H interstitial at fractional position:
vasp-auto inputs/Si_bulk_sc2x2x2 --interstitial "H@0.125,0.125,0.125" --build-only

# Build + relax in one command:
vasp-auto inputs/Si_bulk_sc2x2x2 --vacancy 1 --calc-type relax --kmesh 2x2x2 --cpus 8
```

---

**Next**: Tutorial 12 demonstrates an ab-initio molecular dynamics (AIMD) run
and shows how to analyse the XDATCAR trajectory.
