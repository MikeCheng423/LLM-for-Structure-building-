# Tutorial — Heterostructures: a TiO₂(111) slab on a graphene sheet

This manual builds a composite structure out of two materials with
**different unit cells**: a (111)-facet rutile TiO₂ slab deposited on a
graphene sheet. The same recipe applies to any slab-on-sheet or
deposit-on-substrate system (Au on graphite, MoS₂ on hBN, a cluster in a
pore…). Everything below runs without VASP; you are only producing a POSCAR.

Two tools do the heavy lifting:

- `--match-cells` — searches in-plane supercell pairs that bring the two
  lattices into registry, and tells you the residual strain.
- `--combine` — stacks one structure on top of another, extends the c axis,
  and leaves vacuum above the deposit.

All of this is also available interactively in the web UI
(`vasp-auto-ui` → Build tab → *Combine two structures*), where the
cell-match suggestions appear as clickable rows; this manual uses the CLI so
every step is scriptable.

---

## 1. Build the two ingredients

### 1.1 Graphene (the substrate)

Graphene ships in the pure-Python prototype library — no ASE needed:

```bash
vasp-auto --build-prototype "graphene:vacuum=20" --ase-output inputs/graphene --build-only
```

`vacuum=20` sets the height of the empty box along c; make it generous, the
TiO₂ slab will live in it later. (`a=…` would override the 2.468 Å lattice
constant.) Other prototypes: `graphite`, `hBN`, `rutile-TiO2`, `anatase-TiO2`.

### 1.2 The rutile TiO₂ bulk crystal

```bash
vasp-auto --build-prototype "rutile-TiO2" --ase-output inputs/rutile --build-only
```

### 1.3 Cut the (111) facet

Slab cutting goes through ASE (`pip install ase`), feeding it the POSCAR we
just made:

```bash
vasp-auto --ase-build-slab inputs/rutile/POSCAR \
          --ase-miller "1,1,1" --ase-layers 3 --ase-vacuum 10 \
          --ase-output inputs/tio2_111 --build-only
```

- `--ase-layers` counts repetitions of the (111) stacking unit — 3 gives a
  slab a TiO₂ study can relax; converge this for production work.
- The slab POSCAR is **unrelaxed and stoichiometric-as-cut**. Rutile (111)
  cuts produce polar, undercoordinated terminations; for publication-grade
  surfaces inspect the termination in the UI viewer (its coordination panel
  flags undercoordinated atoms) and delete/add atoms as needed (`--delete`,
  `--add-atom`).

> **Why (111)?** This manual follows the requested facet. The same commands
> with `--ase-miller "1,1,0"` give the more stable rutile (110), and
> `anatase-TiO2` + `"1,0,1"` the common anatase (101).

---

## 2. The problem: two different unit cells

Look at the in-plane cells (the UI Cell panel shows the same numbers):

| structure | a (Å) | b (Å) | γ (°) |
|---|---|---|---|
| graphene | 2.468 | 2.468 | 120.0 |
| TiO₂(111) slab | 5.464 | 5.464 | 72.9 |

Neither the lengths nor the angles agree. You cannot just paste one POSCAR
into the other: periodic boundary conditions force both materials into **one
shared cell**, so first you must decide how to reconcile them. There are two
strategies, and `vasp_auto` supports both.

### Strategy A — commensurate supercells (epitaxy, strained guest)

Find integer repetitions `(i×j)` of the substrate and `(k×l)` of the deposit
whose total lengths nearly agree, then strain the deposit onto the substrate
lattice. Good when a small-strain match exists (graphene on hBN: 1.4 %).

### Strategy B — incommensurate deposit (centred, unstrained)

Keep the deposit's own geometry, centre it over a substrate supercell large
enough to keep its periodic images apart, and accept that the composite cell
is only periodic for the substrate. Good for clusters and for pairs (like
this one) where no cheap commensurate match exists.

---

## 3. Strategy A: `--match-cells`

Ask for supercell suggestions — host (target argument) = substrate,
guest (flag) = deposit:

```bash
vasp-auto inputs/graphene --match-cells inputs/tio2_111 --build-only
```

With the default limits (≤6×6 repeats, ≤10 % strain, ≤8° angle mismatch)
this prints **No match** — the 120° hexagonal cell and the 72.9° oblique
(111) cell are too far apart in angle. Loosen the limits to see the best
available compromise:

```bash
vasp-auto inputs/graphene --match-cells inputs/tio2_111 \
          --match-max 8 --match-strain 0.06 --match-gamma-tol 15 --build-only
```

```
Host      : inputs/graphene/POSCAR (a/b = 2.468, 2.468 Å)
Guest     : inputs/tio2_111/POSCAR (a/b = 5.464, 5.464 Å)
Angle     : 12.94 deg in-plane mismatch (straining shears the guest by this much)
    host    guest  strain a  strain b   atoms
    7x7      3x3     -5.12%    -5.12%     206

Next step : apply the supercells, then stack with the guest strained on:
  vasp-auto inputs/graphene --supercell 7x7x1 --build-only
  vasp-auto inputs/tio2_111 --supercell 3x3x1 --build-only
  vasp-auto <host_supercell_case> --combine <guest_supercell_case> --combine-strain --build-only
```

Read the table before accepting: **5 % strain plus a 13° shear is a heavily
deformed slab**, acceptable for a quick screening of interface charge
transfer, not for quantitative adsorption or band alignment. When the
printed strain/angle is this large, prefer Strategy B. (For a well-matched
pair the table is the whole decision: pick the row with the smallest strain
you can afford atoms for.)

Following the printed commands:

```bash
vasp-auto inputs/graphene --supercell 7x7x1 --build-only      # → inputs/graphene_sc7x7x1
vasp-auto inputs/tio2_111 --supercell 3x3x1 --build-only      # → inputs/tio2_111_sc3x3x1
vasp-auto inputs/graphene_sc7x7x1 --combine inputs/tio2_111_sc3x3x1 \
          --combine-strain --combine-gap 3.0 --combine-vacuum 12 --build-only
```

`--combine-strain` re-expresses the guest's fractional in-plane coordinates
in the host vectors — that is what "straining the guest onto the host" means.

## 4. Strategy B: unstrained, centred deposit

Skip `--combine-strain`. The TiO₂ slab keeps its exact geometry and is
centred over the graphene cell; pick a substrate supercell big enough that
the slab's periodic images don't touch (slab is ~10.9 Å wide; 7×7 graphene
≈ 17.3 Å leaves a safe margin):

```bash
vasp-auto inputs/graphene --supercell 7x7x1 --build-only
vasp-auto inputs/graphene_sc7x7x1 --combine inputs/tio2_111 \
          --combine-gap 3.0 --combine-vacuum 12 --build-only
# → inputs/graphene_sc7x7x1_plus_tio2_111/POSCAR
```

What the flags mean (defaults in parentheses):

- `--combine-gap` (2.0) — Å between the highest substrate atom and the
  lowest deposit atom. ~3 Å is a sensible start for van-der-Waals contact;
  the relaxation will find the real distance.
- `--combine-vacuum` (10.0) — Å of empty space left above the deposit; the
  c axis is extended automatically.
- `--combine-shift "x,y"` — slides the deposit laterally, in fractions of
  the substrate a/b vectors, to test different adsorption registries.
- `--combine-mode insert` — alternative mode that keeps the host cell
  unchanged and just drops the guest atoms in (molecules in pores).

## 5. Sanity-check, then relax

Open the result in the UI (Build tab loads any case into the editor) and
check: the gap is right, no atoms overlap, vacuum remains above the slab.
Or from the terminal:

```bash
vasp-auto inputs/graphene_sc7x7x1_plus_tio2_111 --dry-run     # preview INCAR/KPOINTS/POTCAR
```

Then relax — fix the bottom of the system if you want the graphene to stay
flat, and pre-relax with the ML potential to save VASP steps:

```bash
vasp-auto inputs/graphene_sc7x7x1_plus_tio2_111 --freeze "z<0.2" --build-only
vasp-auto inputs/graphene_sc7x7x1_plus_tio2_111_frz --ml-relax --calc-type relax --kpoints-mode gamma --kmesh 2x2x1
```

For large heterostructure cells a Γ-centred 2×2×1 or even 1×1×1 mesh is the
usual starting point; converge it with `--converge-scf` if energies matter.

---

## 6. Quick reference

| step | command |
|---|---|
| substrate sheet | `--build-prototype "graphene:vacuum=20"` |
| bulk oxide | `--build-prototype "rutile-TiO2"` (or `anatase-TiO2`, `hBN`, `graphite`) |
| cut a facet | `--ase-build-slab CASE/POSCAR --ase-miller "1,1,1" --ase-layers N` |
| find matching supercells | `HOST --match-cells GUEST [--match-max N --match-strain F --match-gamma-tol D]` |
| make the supercells | `CASE --supercell IxJx1 --build-only` |
| stack, strained (epitaxy) | `HOST_SC --combine GUEST_SC --combine-strain --build-only` |
| stack, unstrained (centred) | `HOST_SC --combine GUEST --combine-gap G --combine-vacuum V --build-only` |
| registry scan | add `--combine-shift "0.33,0.33"` |
| freeze the substrate | `CASE --freeze "z<0.2" --build-only` |

In the web UI the whole chapter is the *Combine two structures* card on the
Build tab: load the host into the editor, pick the guest, press
**🔍 suggest cell match**, hit **use** on a row (strained stack) or
**Combine → editor** directly (unstrained) — nothing is written until you
press 💾 Save.
