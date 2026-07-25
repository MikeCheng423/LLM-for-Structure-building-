# Tutorial 07 — NEB Transition-State Search and Reaction Barrier

The Nudged Elastic Band (NEB) method finds the minimum-energy path between two
known stable structures and locates the transition state (the saddle point with
the highest energy). This tutorial computes the **H diffusion barrier between
two hollow sites on Pt(111)** using vasp_auto's TSS case format. The climbing-
image NEB variant (`LCLIMB = .TRUE.`) is used, which sharpens convergence on
the saddle point. The same recipe applies to vacancy hopping, defect migration,
surface reactions, or any other activated process.

---

## Prerequisites

- ASE installed (`pip install ase`).
- A relaxed Pt(111) slab with H adsorbed in two distinct hollow sites (the two
  endpoints). See Tutorial 08 for the slab-building and Tutorial 04 for
  relaxation.
- `config.yaml` with `vasp_executable`, `potcar_root`, and optionally
  `neb_images: 5`.

---

## Step 1 — TSS case directory structure

A TSS (transition-state search) case has two endpoints instead of one:

```
TSS/H_diffusion/
  initial/
    POSCAR    # H at the fcc hollow site (fully relaxed)
  final/
    POSCAR    # H at the hcp hollow site (fully relaxed)
```

vasp_auto detects the `initial/` + `final/` layout automatically and treats
the parent directory as an NEB case.

---

## Step 2 — Build the endpoint structures

If you are starting from scratch, build the Pt(111) slab and add H adsorbates
in two different positions. An approximate setup:

```bash
# Build the slab (requires ASE):
vasp-auto --ase-build-slab Pt --ase-miller "1,1,1" --ase-layers 4 \
          --ase-vacuum 12 --ase-repeat 3x3 \
          --ase-output inputs/Pt111 --build-only

# Freeze bottom two layers, relax the top two:
vasp-auto inputs/Pt111 --freeze "z<0.4" --build-only

vasp-auto inputs/Pt111_frz --calc-type relax --kmesh 4x4x1 --cpus 8

# Place H at fcc hollow (atom index chosen after inspecting the slab):
vasp-auto inputs/Pt111_frz --adsorbate "H@12+1.8" --build-only
cp -r inputs/Pt111_frz_adsH12 TSS/H_diffusion/initial

# Place H at hcp hollow (a nearby atom):
vasp-auto inputs/Pt111_frz --adsorbate "H@15+1.8" --build-only
cp -r inputs/Pt111_frz_adsH15 TSS/H_diffusion/final
```

Relax each endpoint independently before running NEB:

```bash
vasp-auto TSS/H_diffusion/initial --calc-type relax --kmesh 4x4x1 --cpus 8
vasp-auto TSS/H_diffusion/final   --calc-type relax --kmesh 4x4x1 --cpus 8
```

---

## Step 3 — Interpolate images with ASE IDPP

```bash
vasp-auto TSS/H_diffusion \
  --calc-type neb \
  --neb-images 5 \
  --ase-neb \
  --ase-neb-method idpp \
  --kpoints-mode gamma --kmesh 4x4x1 \
  --cpus 16
```

Flag details:

| Flag | Effect |
|------|--------|
| `--calc-type neb` | Selects `example/INCAR_neb` (LCLIMB=.TRUE., IBRION=3, SPRING=-5). |
| `--neb-images 5` | Creates 5 intermediate images (7 total including endpoints). |
| `--ase-neb` | Uses ASE NEB to interpolate coordinates between endpoints. |
| `--ase-neb-method idpp` | Image-Dependent Pair Potential interpolation (smoother than linear). |
| `--kmesh 4x4x1` | One k-mesh for all images; keep it consistent with endpoint calculations. |

vasp_auto writes image directories `00/`, `01/`, …, `06/` under the job
directory and appends `IMAGES = 5` to the INCAR automatically.

---

## Step 4 — Read the NEB results

After convergence (all forces below EDIFFG = -0.03 eV/Å), check the Excel
summary:

```bash
vasp-auto TSS/H_diffusion --parse-only
```

The row for the NEB case contains:

| Column | Meaning |
|--------|---------|
| `neb_barrier_eV` | Forward barrier = E(saddle) − E(initial) |
| `neb_forward_eV` | Same as `neb_barrier_eV` |
| `neb_backward_eV` | E(saddle) − E(final) |
| `neb_images` | Number of intermediate images |

Expected for H on Pt(111): forward barrier ≈ 0.05–0.15 eV depending on the
path.

The per-image energy profile can be found in `jobs/H_diffusion/H_diffusion/`
alongside the `run.log`. The web UI's Results tab renders the NEB profile
plot automatically.

---

## Step 5 — Animate the minimum-energy path

The NEB animation frame data comes from the `OUTCAR` files in each image
directory. In the web UI, open the Results tab and click the NEB row — the
pathway is animated in the 3D viewer. From Python:

```python
from vasp_auto.trajectory import parse_neb_trajectory
frames = parse_neb_trajectory("jobs/H_diffusion/H_diffusion")
# frames[i] = {"image": i, "lattice": ..., "positions": [...], "elements": [...], "energy_eV": ...}
```

---

## Step 6 — Tips for robust NEB convergence

1. **Always relax endpoints tightly** (EDIFFG = -0.01 eV/Å or better) before
   running NEB. Loose endpoint forces propagate to all images.
2. Use `--ase-neb-method idpp` rather than `linear`; it avoids atoms passing
   through each other during interpolation.
3. If NEB stalls, increase the spring constant in the case INCAR:
   `SPRING = -10` (stiffer bands reduce image drift).
4. For flat potential-energy surfaces increase `--neb-images` to 7 or more.
5. Freeze slab atoms (see Tutorial 08) so only the adsorbate images are
   displaced — this dramatically reduces the number of degrees of freedom and
   cost.

---

## Quick reference

```bash
# Full NEB with 5 ASE-IDPP images:
vasp-auto TSS/H_diffusion \
  --calc-type neb --neb-images 5 \
  --ase-neb --ase-neb-method idpp \
  --kmesh 4x4x1 --cpus 16

# Parse a finished NEB (extract barriers without re-running):
vasp-auto TSS/H_diffusion --parse-only

# Use more images for a smoother profile:
vasp-auto TSS/H_diffusion \
  --calc-type neb --neb-images 8 --ase-neb --kmesh 4x4x1 --cpus 20

# Linear interpolation instead of IDPP (faster but lower quality):
vasp-auto TSS/H_diffusion --calc-type neb --neb-images 5 \
  --ase-neb --ase-neb-method linear --kmesh 4x4x1 --cpus 16
```

---

**Next**: Tutorial 08 shows how to build a metal surface slab, freeze the
bottom layers, and compute the work function.
