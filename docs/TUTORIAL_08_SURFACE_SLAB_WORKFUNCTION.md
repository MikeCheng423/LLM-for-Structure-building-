# Tutorial 08 — Metal Surface Slab and Work Function

The work function W = V_vacuum − E_Fermi is a fundamental quantity for
understanding band alignment, catalytic activity, and electron emission at
metal surfaces. This tutorial builds a **six-layer Pt(111) slab**, freezes the
bottom three layers (bulk-like region) while relaxing the top three (surface
region), and computes the work function from the planar-averaged electrostatic
potential (LOCPOT). The same recipe applies to any metal surface.

---

## Prerequisites

- ASE installed (`pip install ase`).
- `POTCAR/Pt/POTCAR` (or `Pt_pv`; set `potcar_map: {Pt: Pt}` in `config.yaml`).
- `config.yaml` with `vasp_executable` and `potcar_root`.

---

## Step 1 — Build the Pt(111) slab

```bash
vasp-auto --ase-build-slab Pt \
          --ase-crystalstructure fcc --ase-a 3.924 \
          --ase-miller "1,1,1" \
          --ase-layers 6 \
          --ase-vacuum 15 \
          --ase-repeat 2x2 \
          --ase-output inputs/Pt111_6L --build-only
```

Flag summary:

| Flag | Meaning |
|------|---------|
| `--ase-build-slab Pt` | Element to use for the slab crystal. |
| `--ase-crystalstructure fcc --ase-a 3.924` | Lattice type and constant (Å). |
| `--ase-miller "1,1,1"` | Surface orientation. |
| `--ase-layers 6` | Number of atomic layers. |
| `--ase-vacuum 15` | Vacuum thickness in Å above the slab. |
| `--ase-repeat 2x2` | Lateral supercell repeat (2×2 in-plane). |

This creates `inputs/Pt111_6L/POSCAR` — a 24-atom slab (6 layers × 4 atoms
per layer) with 15 Å of vacuum.

---

## Step 2 — Freeze the bottom half of the slab

Freeze all atoms with fractional z below 0.4 (the bottom three layers):

```bash
vasp-auto inputs/Pt111_6L \
          --freeze "z<0.4" \
          --build-only
```

This writes `inputs/Pt111_6L_frz/POSCAR` with Selective dynamics enabled:
bottom atoms get `F F F` flags; top atoms remain `T T T`. The `_frz` suffix is
appended automatically.

To freeze only the in-plane directions (let the bottom layers breathe
vertically but not slide laterally):

```bash
vasp-auto inputs/Pt111_6L --freeze "z<0.4:XY" --build-only
```

---

## Step 3 — Relax the slab (top layers free)

```bash
vasp-auto inputs/Pt111_6L_frz \
  --calc-type relax \
  --kpoints-mode gamma --kmesh 4x4x1 \
  --cpus 16
```

`--kmesh 4x4x1` is a 4×4×1 Gamma-centred mesh — appropriate for a 2×2
surface cell. For a 3×3 cell use `3x3x1`; for convergence tests see Tutorial 03.

The relaxation runs with `ISIF = 2` (atomic positions only, cell fixed) from
`INCAR_optimize_structure`. Check the Excel row for `converged = TRUE` and
`max_force_eVA < 0.02`.

---

## Step 4 — Compute the work function

```bash
vasp-auto inputs/Pt111_6L_frz \
  --calc-type workfunction \
  --kpoints-mode gamma --kmesh 4x4x1 \
  --cpus 16
```

The `workfunction` template sets:

```
LVHAR  = .TRUE.   # LOCPOT contains ionic + Hartree potential only (no XC)
IDIPOL = 3        # dipole correction along z (required for asymmetric slabs)
LDIPOL = .TRUE.   # self-consistent dipole correction
```

vasp_auto runs VASP and writes a `LOCPOT` file in the job directory.

---

## Step 5 — Extract W = V_vacuum − E_Fermi

```bash
vasp-auto --work-function jobs/Pt111_6L_frz/Pt111_6L_frz
```

Terminal output:

```
Vacuum     : 5.4321 eV
E_Fermi    : -1.9876 eV
Work func. : 5.43 eV      (W = V_vac - E_F)
Wrote     : jobs/Pt111_6L_frz/Pt111_6L_frz/potential_profile.csv
```

`potential_profile.csv` has two columns (`frac_position`, `potential_eV`);
plot it to see the planar-averaged electrostatic potential as a function of
the z coordinate — the vacuum plateau is clearly visible.

Experimental Pt(111) work function: **5.65 eV** (GGA typically underestimates
by ~0.1–0.3 eV).

---

## Step 6 — Workflow shortcut: relax → work function in one command

```bash
vasp-auto inputs/Pt111_6L_frz \
  --workflow "relax,workfunction" \
  --kmesh 4x4x1 --cpus 16
```

Then parse:

```bash
vasp-auto --work-function jobs/Pt111_6L_frz/Pt111_6L_frz/02_workfunction
```

---

## Notes on slab thickness and vacuum

- Use at least 4–6 layers; test convergence of the surface energy vs. layers.
- The vacuum must be large enough (≥ 15 Å) so the potential is flat in the
  middle (true vacuum level). If the curve does not flatten, increase
  `--ase-vacuum`.
- For asymmetric slabs (different top and bottom terminations) `LDIPOL=.TRUE.`
  and `IDIPOL=3` are mandatory to cancel the artificial electric field from
  periodic boundary conditions.

---

## Quick reference

```bash
# Build 6-layer 2x2 Pt(111) slab with 15 Å vacuum:
vasp-auto --ase-build-slab Pt --ase-crystalstructure fcc --ase-a 3.924 \
          --ase-miller "1,1,1" --ase-layers 6 --ase-vacuum 15 --ase-repeat 2x2 \
          --ase-output inputs/Pt111_6L --build-only

# Freeze bottom half:
vasp-auto inputs/Pt111_6L --freeze "z<0.4" --build-only

# Relax (top layers free):
vasp-auto inputs/Pt111_6L_frz --calc-type relax --kmesh 4x4x1 --cpus 16

# Work-function calculation:
vasp-auto inputs/Pt111_6L_frz --calc-type workfunction --kmesh 4x4x1 --cpus 16

# Extract W and write potential_profile.csv:
vasp-auto --work-function jobs/Pt111_6L_frz/Pt111_6L_frz
```

---

**Next**: Tutorial 09 shows how to pre-relax structures with a machine-learning
interatomic potential before the VASP calculation.
