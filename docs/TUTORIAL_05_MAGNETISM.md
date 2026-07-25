# Tutorial 05 — Spin-Polarised Calculation: Magnetism in bcc Fe

Iron is a textbook ferromagnet; its bcc ground state carries a magnetic moment
of ~ 2.2 μ_B per atom. This tutorial computes the total energy, band gap, and
per-atom magnetic moments of **bcc iron** using vasp_auto's spin-polarised
mode. The same flags (`--spin`, `--magmom`, `magmom_map:`) apply to any
magnetic material — antiferromagnets, layered magnets, or adsorbates with
partial moments.

---

## Prerequisites

- `inputs/Fe_bulk/POSCAR` with the bcc Fe structure (one or two atoms per
  cell, see below).
- `POTCAR/Fe/POTCAR` (or `POTCAR/Fe_pv/POTCAR` — see Step 1).
- `config.yaml` configured with `vasp_executable` and `potcar_root`.

---

## Step 1 — Create the case directory

Paste the following two-atom bcc Fe POSCAR into `inputs/Fe_bulk/`:

```
bcc Fe (conventional 2-atom cell, a = 2.831 A)
   1.0
     2.831000000   0.000000000   0.000000000
     0.000000000   2.831000000   0.000000000
     0.000000000   0.000000000   2.831000000
   Fe
     2
Direct
  0.000000000  0.000000000  0.000000000
  0.500000000  0.500000000  0.500000000
```

The Fe_pv pseudopotential (semi-core 3p states in the valence) is strongly
recommended for magnetic iron. Add to `config.yaml`:

```yaml
potcar_map:
  Fe: Fe_pv
```

---

## Step 2 — Spin-polarised SCF with a fixed initial moment

```bash
vasp-auto inputs/Fe_bulk \
  --calc-type scf \
  --spin \
  --magmom "Fe:2.5" \
  --kpoints-mode gamma --kmesh 12 \
  --cpus 8
```

Flag details:

| Flag | Effect |
|------|--------|
| `--spin` | Sets `ISPIN = 2` in the INCAR. Without `--magmom`, vasp_auto assigns default starting moments (1.0 μ_B for 3d elements). |
| `--magmom "Fe:2.5"` | Sets the initial MAGMOM line to `2*2.50` for the two Fe atoms. Overrides any `magmom_map:` from `config.yaml`. |
| `--kmesh 12` | 12×12×12 Gamma mesh; dense mesh needed for iron's narrow d bands. |

The generated INCAR will contain:

```
ISPIN  = 2
MAGMOM = 2*2.50
```

---

## Step 3 — Read the magnetic moments from the output

After the run open `jobs/Fe_bulk/Fe_bulk.xlsx`. New columns appear for spin
calculations:

| Column | Meaning |
|--------|---------|
| `magmom_total` | Sum of all site moments in μ_B |
| `magmom_per_atom` | JSON list of per-atom moments (from OUTCAR) |
| `energy_eV` | Total energy (spin-paired and spin component) |

Expected result: `magmom_total ≈ 4.4 μ_B` for two Fe atoms (2.2 μ_B each).
The per-atom moments are extracted from the `magnetization` block in `OUTCAR`.

You can also grep the OUTCAR directly:

```bash
grep "magnetization (x)" jobs/Fe_bulk/Fe_bulk/OUTCAR | tail -5
```

---

## Step 4 — Relax the magnetic structure

For publication-quality results always relax the geometry spin-polarised:

```bash
vasp-auto inputs/Fe_bulk \
  --calc-type relax \
  --spin \
  --magmom "Fe:2.5" \
  --kpoints-mode gamma --kmesh 12 \
  --cpus 8
```

The `relax` template uses `ISIF = 3` (relax cell volume and shape as well as
atoms). The equilibrium lattice constant of bcc Fe with PBE is ~2.83 Å.

---

## Step 5 — Per-site moments from a `magmom_map` in config

If you always want the same initial moments for all Fe calculations, add them
to `config.yaml` once and omit `--magmom` from every command:

```yaml
magmom_map:
  Fe: 2.5
  O: 0.3
  C: 0.0
```

```bash
vasp-auto inputs/Fe_bulk --calc-type scf --spin --kmesh 12 --cpus 8
```

vasp_auto reads `magmom_map:` from the config, maps each atom in the POSCAR to
its element symbol, and constructs the MAGMOM line automatically.

---

## Step 6 — Antiferromagnetic ordering (two-sublattice)

To set up alternating moments (e.g. Néel antiferromagnetism on a 4-atom
supercell), provide explicit per-atom moments separated by commas:

```bash
# 4-atom supercell: atoms 1,2 spin-up, 3,4 spin-down
vasp-auto inputs/Fe_AFM --calc-type scf --spin \
  --magmom "Fe:2.5" --kmesh 10 --cpus 8
```

For a manual per-atom MAGMOM line, place an `INCAR` in the case directory
with the exact line you want and vasp_auto will use it verbatim:

```
# inputs/Fe_AFM/INCAR
MAGMOM = 2.5 2.5 -2.5 -2.5
```

---

## Quick reference

```bash
# Spin-polarised SCF with Fe initial moment 2.5 μ_B, 12x12x12 mesh:
vasp-auto inputs/Fe_bulk --calc-type scf --spin --magmom "Fe:2.5" --kmesh 12 --cpus 8

# Spin-polarised relaxation:
vasp-auto inputs/Fe_bulk --calc-type relax --spin --magmom "Fe:2.5" --kmesh 12 --cpus 8

# Use config.yaml magmom_map (no --magmom flag needed):
vasp-auto inputs/Fe_bulk --calc-type scf --spin --kmesh 12 --cpus 8

# Parse moments from a finished run:
vasp-auto inputs/Fe_bulk --parse-only
# Check magmom_total and magmom_per_atom columns in jobs/Fe_bulk/Fe_bulk.xlsx
```

---

**Next**: Tutorial 06 shows how to build a multi-step workflow
(`converge → relax → scf → dos`) in a single command.
