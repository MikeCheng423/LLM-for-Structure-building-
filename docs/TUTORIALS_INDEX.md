# vasp_auto — Tutorial Learning Path

This page lists every tutorial in the recommended reading order. Each tutorial
is self-contained; prerequisites are listed at the top of the file. Start with
Tutorial 01 if you are new to vasp_auto; experienced VASP users can jump
directly to the topic they need.

---

## Numbered tutorials (01 → 12)

| # | File | One-line description |
|---|------|----------------------|
| 01 | [TUTORIAL_01_BASICS_SCF.md](TUTORIAL_01_BASICS_SCF.md) | Total energy of bulk Si from a POSCAR: `--calc-type scf`, `--kmesh`, `--dry-run`, Excel summary. |
| 02 | [TUTORIAL_02_RELAXATION.md](TUTORIAL_02_RELAXATION.md) | Geometry optimisation of Si: `--calc-type relax`, `--report`, `--retry-failed`, CONTCAR trajectory. |
| 03 | [TUTORIAL_03_CONVERGENCE.md](TUTORIAL_03_CONVERGENCE.md) | Pick ENCUT/SIGMA/NELM/k-mesh from scratch: `--converge-encut`, `--converge-sigma`, `--converge-scf`, `--reuse-wavecar`. |
| 04 | [TUTORIAL_04_DOS_BANDS.md](TUTORIAL_04_DOS_BANDS.md) | Semiconductor DOS + band structure: `--workflow "relax,scf,dos"`, `--kpath auto/fcc`, DOS/bands CSV export. |
| 05 | [TUTORIAL_05_MAGNETISM.md](TUTORIAL_05_MAGNETISM.md) | Spin-polarised bcc Fe: `--spin`, `--magmom`, `magmom_map:`, per-atom moment output. |
| 06 | [TUTORIAL_06_WORKFLOW_CHAINING.md](TUTORIAL_06_WORKFLOW_CHAINING.md) | Chained pipeline: `--workflow "converge,relax,scf,dos"`, `workflow.yaml`, per-step INCAR overrides. |
| 07 | [TUTORIAL_07_NEB_BARRIER.md](TUTORIAL_07_NEB_BARRIER.md) | NEB transition-state barrier: `--calc-type neb`, `--neb-images`, `--ase-neb`, forward/backward barrier parsing, animation. |
| 08 | [TUTORIAL_08_SURFACE_SLAB_WORKFUNCTION.md](TUTORIAL_08_SURFACE_SLAB_WORKFUNCTION.md) | Metal slab + work function: `--ase-build-slab`, `--ase-miller`, `--ase-layers`, `--freeze`, `--calc-type workfunction`, `--work-function`. |
| 09 | [TUTORIAL_09_ML_PRERELAX.md](TUTORIAL_09_ML_PRERELAX.md) | MLIP pre-relaxation and energy screening: `--ml-energy`, `--ml-relax`, `--ml-only`, `--ml-model emt/uma-s-1p1`. |
| 10 | [TUTORIAL_10_OPTICS_DIELECTRIC.md](TUTORIAL_10_OPTICS_DIELECTRIC.md) | Optical absorption (LOPTICS): `--calc-type optics`, `--optics-parse`, `absorption.csv`, ε(0). |
| 11 | [TUTORIAL_11_DEFECTS_SUPERCELLS.md](TUTORIAL_11_DEFECTS_SUPERCELLS.md) | Supercells and point defects: `--supercell`, `--vacancy`, `--substitute`, `--interstitial`, `--build-only`. |
| 12 | [TUTORIAL_12_AIMD.md](TUTORIAL_12_AIMD.md) | Finite-temperature AIMD: `--calc-type md`, XDATCAR trajectory, `--cpus`, `--parallel`, `--poll`, `--scheduler slurm`. |

---

## Existing topic tutorials

| File | One-line description |
|------|----------------------|
| [TUTORIAL_CATALYSIS.md](TUTORIAL_CATALYSIS.md) | Full catalysis study (HER on Pt): adsorption energies, CHE free-energy diagrams, DOS/d-band, charge-density difference, Bader, work function, optical absorption, NEB barriers — all chained. |
| [TUTORIAL_HETEROSTRUCTURE.md](TUTORIAL_HETEROSTRUCTURE.md) | Combining two materials with mismatched unit cells (TiO₂(111) on graphene): `--build-prototype`, `--match-cells`, `--combine`. |
| [TUTORIAL_CEO2_GRAPHENE_CO2.md](TUTORIAL_CEO2_GRAPHENE_CO2.md) | CO₂ adsorption on a CeO₂/graphene heterostructure — the same workflow executed interactively through the web UI (`vasp-auto-ui`). |

---

## Suggested learning paths

**New to vasp_auto**
01 → 02 → 03 → 04 → 06

**Surface catalysis study**
01 → 03 → 08 → TUTORIAL_CATALYSIS.md

**Defect physics**
01 → 02 → 03 → 11

**Magnetic materials**
01 → 02 → 05 → 04

**Reaction barriers (NEB)**
01 → 02 → 07

**Heterostructures**
TUTORIAL_HETEROSTRUCTURE.md → 08 → TUTORIAL_CEO2_GRAPHENE_CO2.md

**Fast structure screening with MLIPs**
01 → 09 → 03

**Finite-temperature dynamics**
01 → 12
