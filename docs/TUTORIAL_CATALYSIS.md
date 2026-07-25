# Tutorial — Photocatalysis & Electrocatalysis with vasp_auto

This tutorial walks through every calculation a catalysis study typically
needs — adsorption energies, free-energy (CHE) diagrams, DOS/PDOS and d-band
centers, band structures, charge densities and Bader charges, charge-density
differences, work functions, optical absorption, and NEB reaction barriers —
all driven from the `vasp-auto` command line. Every command shown runs real
VASP through your `config.yaml`; appending `--dry-run` previews the full
input set (INCAR/KPOINTS/POTCAR) without writing or running anything.

The running example is **hydrogen evolution (HER) on Pt(111)** plus a
photocatalyst-style analysis; substitute your own elements and surfaces
throughout.

---

## 0. Prerequisites

```bash
pip install -e .            # gives the vasp-auto command
pip install -e .[ml]        # optional: fairchem for ML pre-relaxation
```

`config.yaml` (repo root, overridable per project/case directory):

```yaml
vasp_executable: /opt/vasp/bin/vasp_std
jobs_root: jobs
potcar_root: POTCAR          # POTCAR/<El>/POTCAR library
potcar_map:                  # pseudopotential variants where recommended
  Pt: Pt
  Ti: Ti_pv
scheduler: local             # or slurm / pbs
# bader_executable: bader    # optional: Henkelman bader binary for --bader
```

Optional external tool: the Henkelman group `bader` binary
(<https://theory.cm.utexas.edu/henkelman/code/bader/>) for `--bader`.

**Conventions used below**

- Case directories (your inputs) live anywhere, e.g. `inputs/<name>/POSCAR`.
- Job directories (generated, never edit) land in `jobs/<name>/`.
- Chained workflow steps run in `jobs/<name>/01_relax/`, `02_scf/`, … and
  forward CONTCAR → POSCAR (and CHGCAR/WAVECAR where needed) automatically.
- Atom indices are 1-based in POSCAR order everywhere.

---

## 1. Converge the numerical settings first

Do this once per chemical system, on the bulk (or a small slab):

```bash
vasp-auto inputs/Pt_bulk \
  --converge-encut 400,450,500,550 \
  --converge-sigma 0.2,0.1,0.05 \
  --converge-scf --kpoints-values "6,8,10,12" \
  --reuse-wavecar
```

Stages run in order ENCUT → SIGMA → NELM → KPOINTS; each stage holds the
previously selected value fixed. The selected values are printed, written to
a step report + CSV in the job directory, and should then be written into
your INCARs / flags for everything that follows. `--reuse-wavecar` seeds
each trial with the previous WAVECAR/CHGCAR to cut wall time.

Rules of thumb for what follows: slabs use `--kmesh NxNx1`; energies that
will be compared (slab vs slab+adsorbate) must share ENCUT, k-density,
smearing, and pseudopotentials.

---

## 2. Build the surface

### 2.1 Bulk → slab

```bash
# Relaxed fcc Pt bulk (cell + ions):
vasp-auto --ase-build-bulk Pt --ase-crystalstructure fcc --ase-cubic \
  --ase-output inputs/Pt_bulk --build-only
vasp-auto inputs/Pt_bulk --calc-type relax --kmesh 12

# 4-layer 3x3 Pt(111) slab with 14 Å vacuum, from the relaxed bulk:
vasp-auto --ase-build-slab jobs/Pt_bulk/CONTCAR --ase-miller 1,1,1 \
  --ase-layers 4 --ase-vacuum 14 --ase-repeat 3x3 \
  --ase-output inputs/Pt111 --build-only
```

### 2.2 Freeze the bottom layers, relax the rest

Slab relaxations should keep the bulk-like bottom half fixed:

```bash
# Freeze everything below fractional height 0.45 (writes inputs/Pt111_frz):
vasp-auto inputs/Pt111 --freeze "z<0.45" --build-only
vasp-auto inputs/Pt111_frz --calc-type relax --kmesh 4x4x1
```

`--freeze` writes Selective dynamics flags; the selection can also be index
lists (`"1-18"`) and can be limited to axes (`"z<0.45:XY"`). For oxide or
magnetic surfaces add `--spin` (and `--magmom "Fe:5.0,O:0.6"`).

Other builders that compose with the same flags: `--supercell 2x2x1`,
`--vacancy N` (e.g. O vacancies in TiO₂), `--substitute N=El` (dopants),
`--adsorbate "El@N+h"`, `--combine` (deposit a cluster/sheet on a support).

---

## 3. Adsorption energies

The adsorption energy is assembled from **three** separately converged
total energies, all with identical settings:

> E_ads = E(slab+adsorbate) − E(slab) − scale · E(gas molecule)

### 3.1 The three jobs

```bash
# (a) gas-phase H2 in a 12 Å box (Gamma point only, spin if radical):
vasp-auto --ase-build-molecule H2 --ase-box 12 \
  --ase-output inputs/H2 --build-only
vasp-auto inputs/H2 --calc-type relax --kmesh 1

# (b) clean slab — already relaxed in step 2 (jobs/Pt111_frz)

# (c) slab + H on an fcc/top site: place H 1.0 Å above surface atom 27,
#     then relax (writes inputs/Pt111_frz_adsH27):
vasp-auto inputs/Pt111_frz --adsorbate "H@27+1.0" --build-only
vasp-auto inputs/Pt111_frz_adsH27 --calc-type relax --kmesh 4x4x1
```

### 3.2 Assemble E_ads

```bash
vasp-auto --adsorption-energy "jobs/Pt111_frz_adsH27,jobs/Pt111_frz,jobs/H2" \
  --molecule-scale 0.5
```

```
E(slab+ads): -245.123456 eV
E(slab)    : -241.654321 eV
E(molecule): -6.760000 eV x 0.5
E_ads      : -0.089135 eV
```

`--molecule-scale 0.5` references atomic H to ½ E(H₂) — the standard choice
for HER. Use `1.0` for molecular adsorption (CO, H₂O, O₂…), `0.5` with an
O₂ box (or water references, see §4) for atomic O. The command warns if any
of the three jobs did not converge. Negative = exothermic adsorption.

### 3.3 Screening many sites cheaply

Generate one case per site and let an MLIP pre-relax before VASP touches
anything (UMA's `oc20` head is trained on adsorbate/catalyst systems):

```bash
for site in 25 26 27; do
  vasp-auto inputs/Pt111_frz --adsorbate "H@${site}+1.0" \
    --ase-output inputs/screen/H_$site --build-only
done
vasp-auto inputs/screen --ml-relax --ml-task oc20 --parallel 3 --calc-type relax
```

The project's Excel summary (`jobs/screen/screen.xlsx`) tabulates all case
energies with a bar chart — the site ranking at a glance.

---

## 4. Free energies: ZPE, entropy, and the CHE

Electrocatalysis descriptors (ΔG_H* for HER; the four ΔG steps of OER/ORR)
need vibrational corrections on top of E_ads:

> ΔG = ΔE + ΔZPE + ΔU_vib − TΔS

### 4.1 Frequencies of the adsorbate

Run a `freq` job on the **relaxed** slab+adsorbate, with the slab atoms
frozen so only the adsorbate is displaced (Selective dynamics from §2.2 are
honoured — VASP builds the Hessian only for T T T atoms):

```bash
vasp-auto inputs/Pt111_frz_adsH27 --workflow "relax,freq" --kmesh 4x4x1
vasp-auto --thermo jobs/Pt111_frz_adsH27/02_freq --temperature 298.15
```

```
Modes      : 3 real, 0 imaginary
    1     1012.43 cm-1    125.534 meV
    ...
ZPE        : 0.172000 eV
U_vib      : 0.002100 eV
T*S        : 0.004900 eV
G correction (ZPE + U_vib - T*S): 0.169200 eV
G = E_DFT + correction: -244.954256 eV
```

Imaginary modes on a supposed minimum mean the geometry is not fully
relaxed — tighten EDIFFG and re-relax. Gas-phase molecules get the same
treatment (`--calc-type freq` on the H₂ box) plus standard tabulated
translational/rotational entropy; at 298.15 K the usual literature value is
G(H₂) = E(H₂) + 0.27 eV − TS with TS(H₂) = 0.40 eV.

### 4.2 ΔG_H* and the computational hydrogen electrode

With the CHE, μ(H⁺ + e⁻) = ½ μ(H₂) at U = 0 V vs RHE, so

> ΔG_H* = G(slab+H) − G(slab) − ½ G(H₂) ; shift by +eU for electrode potential.

A good HER catalyst has ΔG_H* ≈ 0 (Pt: ≈ −0.09 eV). For OER/ORR build the
four intermediates (*OH, *O, *OOH) with `--adsorbate`/`--interstitial`,
compute each G the same way, and reference O through
G(H₂O) − G(H₂) (the standard 2H₂O → O₂ + 4(H⁺+e⁻) construction with
ΔG = 4.92 eV); the overpotential is max(ΔG_i)/e − 1.23 V.

---

## 5. Electronic structure: DOS, PDOS, d-band center

### 5.1 DOS workflow

DOS runs are non-self-consistent on a converged charge density; the chained
workflow stages CONTCAR and CHGCAR automatically:

```bash
vasp-auto inputs/Pt111_frz --workflow "relax,scf,dos" --kmesh 4x4x1
```

The `dos` template sets `LORBIT = 11` (site/orbital projections),
`NEDOS = 2000` and tetrahedron smearing. Results land in
`jobs/Pt111_frz/03_dos/`; the UI's Results tab plots the spin-resolved,
Fermi-aligned DOS with PNG export, and `parser.parse_dos` /
`parser.parse_pdos` give the raw arrays for scripting.

### 5.2 d-band center

The classic activity descriptor — the first moment of the surface-atom
d-projected DOS relative to E_F:

```bash
# atoms 28-36 are the top-layer Pt atoms (or select by height: "z>0.55"):
vasp-auto --d-band "jobs/Pt111_frz/03_dos:z>0.55"
```

```
Atoms      : 9 selected (z>0.55)
d-band center: -2.4310 eV (vs E_F)
d-band width : 1.8120 eV
```

`--d-band-emax 0` integrates only the occupied d-states. Higher (less
negative) d-band centers bind adsorbates more strongly — compare across
strained/alloyed/doped surfaces at fixed settings.

---

## 6. Band structure and band gaps (photocatalysis)

A photocatalyst needs a gap that straddles the water redox levels. Band
structure along high-symmetry lines, fed by a converged SCF charge density:

```bash
vasp-auto inputs/TiO2 --workflow "relax,scf,bands" --kpath hex --kpath-divisions 30
```

(presets: `cubic`, `fcc`, `bcc`, `hex`; or explicit
`--kpath "G 0 0 0; X 0.5 0 0.5; ..."`). Every summary row already includes
`band_gap_eV`, `vbm_eV`, `cbm_eV`, and `fermi_eV` parsed from vasprun.xml.

PBE underestimates gaps; confirm with the hybrid functional template:

```bash
vasp-auto inputs/TiO2 --workflow "relax,scf,hse06"
```

(HSE06 reuses the PBE WAVECAR; expect ~an order of magnitude more CPU.)

---

## 7. Work function and band-edge alignment

The work function W = V_vacuum − E_F positions your band edges on an
absolute scale (vs vacuum; subtract 4.44 eV for NHE):

```bash
vasp-auto inputs/Pt111_frz --workflow "relax,workfunction" --kmesh 4x4x1
vasp-auto --work-function jobs/Pt111_frz/02_workfunction
```

```
Vacuum     : 5.9132 eV
E_Fermi    : 0.2110 eV
Work func. : 5.7022 eV
Wrote      : jobs/Pt111_frz/02_workfunction/potential_profile.csv
```

The `workfunction` template writes LOCPOT with `LVHAR = .TRUE.` (Hartree +
ionic potential only — the correct reference) and enables the dipole
correction (`LDIPOL`, `IDIPOL = 3`) needed for asymmetric slabs;
`potential_profile.csv` holds the planar-averaged V(z) for plotting the
vacuum plateau. For a semiconductor, absolute CBM/VBM = (E_edge − E_F) − W;
photocatalysis requires CBM above H⁺/H₂ (−4.44 eV) and VBM below O₂/H₂O
(−5.67 eV).

---

## 8. Charge density, density differences, Bader charges

### 8.1 High-quality charge density

```bash
vasp-auto inputs/Pt111_frz_adsH27 --workflow "relax,charge" --kmesh 4x4x1
```

The `charge` template writes CHGCAR **and** AECCAR0/AECCAR2
(`LAECHG = .TRUE.`), which Bader analysis needs.

### 8.2 Charge-density difference (bonding visualisation)

Δρ = ρ(slab+ads) − ρ(slab) − ρ(adsorbate) reveals charge transfer at the
bond. The three densities must be on **identical grids**: take the relaxed
combined CONTCAR, then delete the adsorbate (or the slab) *without moving
anything else* so cell and FFT grid match.

```bash
# fragments carved from the combined relaxed geometry (atom 37 is the H):
vasp-auto jobs/Pt111_frz_adsH27/01_relax --delete "37" \
  --ase-output inputs/frag_slab --build-only
vasp-auto jobs/Pt111_frz_adsH27/01_relax --delete "1-36" \
  --ase-output inputs/frag_H --build-only

# single-point charge densities, identical cell/grid in all three:
vasp-auto inputs/frag_slab --calc-type charge --kmesh 4x4x1
vasp-auto inputs/frag_H    --calc-type charge --kmesh 4x4x1

vasp-auto --chg-diff "jobs/Pt111_frz_adsH27/02_charge,jobs/frag_slab,jobs/frag_H"
```

`CHGCAR_diff` is written next to the total CHGCAR in normal CHGCAR format —
open it in VESTA and plot ±isosurfaces (yellow accumulation / cyan
depletion). The command refuses mismatched grids loudly.

### 8.3 Bader charges (oxidation states, charge transfer)

```bash
vasp-auto --bader jobs/Pt111_frz_adsH27/02_charge
```

```
Bader      : AECCAR0+AECCAR2 reference
   37 H  electrons   1.0850  net  -0.0850 e
...
Wrote      : .../bader_charges.csv
```

Net charge = ZVAL(POTCAR) − Bader electrons; negative = the atom gained
electrons. Requires the Henkelman `bader` binary on PATH (or set
`bader_executable:` in config.yaml).

---

## 9. Optical absorption (light harvesting)

The independent-particle dielectric function, run on top of a converged SCF:

```bash
vasp-auto inputs/TiO2 --workflow "relax,scf,optics"
vasp-auto --optics-parse jobs/TiO2/03_optics
```

```
epsilon(0) : 6.8312 (direction-averaged static dielectric constant)
Wrote      : jobs/TiO2/03_optics/absorption.csv
```

`absorption.csv` holds E (eV), α (cm⁻¹), and the real/imaginary dielectric
function — plot α vs photon energy and read the absorption onset against
the visible window (1.6–3.1 eV). Raise `NBANDS` in `example/INCAR_optics`
(empty states dominate the response); for quantitative spectra of small-gap
oxides consider an HSE06 + LOPTICS variant (copy the template and add the
hybrid tags).

---

## 10. Reaction barriers (NEB)

Activation energies for surface steps (e.g. Tafel H+H → H₂, water
dissociation, CO oxidation):

```bash
mkdir -p inputs/H_diffusion/{initial,final}
# initial/POSCAR: relaxed H on fcc site; final/POSCAR: relaxed H on hcp site
# (same cell, same atom order!)
vasp-auto inputs/H_diffusion --calc-type neb --neb-images 5 --ase-neb
```

`--ase-neb` uses IDPP interpolation (much better starting paths than
linear). The climbing-image template (`LCLIMB`) converges the saddle point;
the summary row reports `neb_forward_barrier_eV`, `neb_backward_barrier_eV`
and the full image-energy profile, and the UI animates the band (🎞).
`--ml-relax --ml-task oc20` on the endpoints first is a cheap way to
pre-converge them.

---

## 11. Solvation and electric fields (electrochemistry notes)

- **Implicit solvation**: requires a VASPsol(++)-patched VASP build; add
  `LSOL = .TRUE.` (+ `EB_K` for the dielectric) to any INCAR via the case
  INCAR or a `workflow.yaml` `incar:` override — no engine change needed.
- **Electric fields**: `EFIELD` + `LDIPOL`/`IDIPOL = 3` in a slab INCAR
  simulates field effects on adsorption.
- **Charged-slab / constant-potential methods** are not automated; treat the
  CHE (§4.2) as the standard potential dependence.

---

## 12. Putting it together: a screening project

```
inputs/HER_screen/
├── config.yaml              # per-project overrides (e.g. potcar_map)
├── Pt111_H/POSCAR
├── PtNi111_H/POSCAR
└── MoS2_edge_H/POSCAR
```

```bash
# everything, four cases at a time, retried on known VASP errors,
# with a Markdown report per job and one Excel summary:
vasp-auto inputs/HER_screen --workflow "relax,scf,dos" \
  --parallel 4 --auto-retry 2 --report

# harvest/refresh results later (e.g. after a SLURM batch):
vasp-auto inputs/HER_screen --parse-only --report
```

Each job directory gets `report.md` (setup, energies, problems); the
project Excel collects energy, convergence, band gap, magnetic moments,
NEB barriers, and error diagnostics per case.

---

## 13. Quick reference

| Quantity | Run | Analyse |
|---|---|---|
| Convergence (ENCUT/SIGMA/NELM/k) | `--converge-encut … --converge-sigma … --converge-scf` | printed + CSV report |
| Slab / molecule / adsorbate builds | `--ase-build-slab`, `--ase-build-molecule`, `--adsorbate`, `--freeze`, `--vacancy`, `--substitute`, `--combine` | — |
| Adsorption energy | 3 × `--calc-type relax` | `--adsorption-energy "T,S,M" --molecule-scale 0.5` |
| ZPE / ΔG corrections | `--calc-type freq` (or `relax,freq`) | `--thermo DIR --temperature 298.15` |
| DOS / PDOS | `--workflow "relax,scf,dos"` | UI DOS plot, `parse_pdos` |
| d-band center | (same dos job) | `--d-band "DIR:z>0.55"` |
| Band structure / gap | `--workflow "relax,scf,bands" --kpath hex` | row: `band_gap_eV`, `vbm/cbm` |
| Accurate gap | `--workflow "relax,scf,hse06"` | row: `band_gap_eV` |
| Work function | `--workflow "relax,workfunction"` | `--work-function DIR` |
| Charge density (+Bader files) | `--workflow "relax,charge"` | — |
| Charge-density difference | 3 × `--calc-type charge` (same grid) | `--chg-diff "AB,A,B"` |
| Bader charges | (charge job) | `--bader DIR` |
| Optical absorption | `--workflow "relax,scf,optics"` | `--optics-parse DIR` |
| Reaction barrier | TSS case + `--calc-type neb --ase-neb` | row: barriers; 🎞 animation |
| ML pre-screening | `--ml-relax --ml-task oc20` | — |

All analysis commands (`--adsorption-energy`, `--thermo`, `--work-function`,
`--d-band`, `--chg-diff`, `--bader`, `--optics-parse`) only read finished
job directories — they never launch VASP and are safe to re-run.
