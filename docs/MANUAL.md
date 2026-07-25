# vasp_auto — Operation Manual

`vasp_auto` automates VASP DFT calculations: given a POSCAR it builds INCAR,
KPOINTS, and POTCAR, launches VASP through `mpirun` (or SLURM/PBS), parses the
results (OUTCAR + vasprun.xml), and writes a styled Excel summary.
Version 0.8.0.

> **Tutorials** — a 12-file numbered learning path plus three topic tutorials
> is indexed in `docs/TUTORIALS_INDEX.md`. Highlights:
>
> | Tutorial | Topic |
> |----------|-------|
> | `TUTORIAL_01_BASICS_SCF.md` | First SCF calculation (`--dry-run`, Excel summary) |
> | `TUTORIAL_02_RELAXATION.md` | Geometry optimisation + `--retry-failed` |
> | `TUTORIAL_03_CONVERGENCE.md` | ENCUT/SIGMA/k-mesh convergence scan |
> | `TUTORIAL_04_DOS_BANDS.md` | DOS + bands workflow, CSV export |
> | `TUTORIAL_05_MAGNETISM.md` | Spin-polarised Fe, `--spin`/`--magmom` |
> | `TUTORIAL_06_WORKFLOW_CHAINING.md` | `converge → relax → scf → dos` chain |
> | `TUTORIAL_07_NEB_BARRIER.md` | NEB transition-state search |
> | `TUTORIAL_08_SURFACE_SLAB_WORKFUNCTION.md` | Slab builder + work function |
> | `TUTORIAL_09_ML_PRERELAX.md` | MLIP pre-relax + `--ml-energy` screen |
> | `TUTORIAL_10_OPTICS_DIELECTRIC.md` | Optical absorption, ε(0) |
> | `TUTORIAL_11_DEFECTS_SUPERCELLS.md` | Supercells, vacancies, substitutions |
> | `TUTORIAL_12_AIMD.md` | AIMD trajectory + `--poll` queue status |
> | `TUTORIAL_CATALYSIS.md` | Full catalysis study (HER on Pt) |
> | `TUTORIAL_HETEROSTRUCTURE.md` | TiO₂ on graphene, `--match-cells`/`--combine` |
> | `TUTORIAL_CEO2_GRAPHENE_CO2.md` | CeO₂/graphene + CO₂ adsorption via the web UI |

---

## 1. Installation and the global command

The package lives in `~/vasp_auto` and is editable-installed into its own
virtualenv. The command is symlinked into `~/.local/bin` (which is on PATH),
so it works **from any directory**:

```bash
vasp-auto --help          # primary command (console script)
vasp-auto-ui              # browser UI at http://127.0.0.1:8800/
vasp_auto --help          # legacy launcher script, same engine
```

To reinstall after moving the repository:

```bash
cd ~/vasp_auto
venv/bin/pip install -e . --no-build-isolation
ln -sf ~/vasp_auto/venv/bin/vasp-auto ~/.local/bin/vasp-auto
```

---

## 2. File structure

```
~/vasp_auto/
  config.yaml             global configuration (see §4)
  src/vasp_auto/          the Python package (engine)
  src/vasp_auto_ui/       the web UI package (vasp-auto-ui, see §11)
  example/                INCAR templates: INCAR_scf, INCAR_optimize_structure,
                          INCAR_dos, INCAR_bands, INCAR_charge_density,
                          INCAR_neb, INCAR_md, INCAR_phonon, INCAR_hse06,
                          INCAR_freq, INCAR_optics, INCAR_workfunction
  inputs/                 your calculation input directories
  jobs/                   generated job directories + Excel summaries (output)
  POTCAR/                 pseudopotential library: POTCAR/<El>/POTCAR
  TSS/                    NEB example inputs and jobs
  tests/                  pytest unit tests   (venv/bin/python -m pytest)
  selfcheck/              end-to-end feature check with a fake VASP (see §10)
  docs/                   this manual + tutorials
  venv/                   the virtualenv that runs vasp-auto
  .venv/ + ase-gui        separate ASE GUI toolchain (./ase-gui opens it)
```

A **case** is a directory containing a `POSCAR` (SCF-like) or
`initial/POSCAR` + `final/POSCAR` (NEB/TSS). A **project** is a directory
whose subdirectories are cases — vasp_auto auto-detects which one you gave it.

Anything you put in the case directory wins over generated defaults:
`INCAR`, `KPOINTS`, `POTCAR`, `INCAR_<type>` (per-step workflow template),
`workflow.yaml`, `config.yaml`.

Job layout produced under `jobs/`:

```
jobs/<project>/<case>/        INCAR KPOINTS POSCAR POTCAR OUTCAR run.log ...
jobs/<project>/<case>/01_relax/, 02_scf/, ...     (workflow steps)
jobs/<project>/<case>/scf_convergence/           (convergence trials + report)
jobs/<project>/<project>.xlsx                     (summary; single cases get
                                                   jobs/<case>/<case>.xlsx)
```

---

## 3. Quick start

```bash
# one SCF case (POSCAR in inputs/Al)
vasp-auto inputs/Al -n 8

# whole project, 2 cases at a time
vasp-auto inputs --parallel 2 -n 8

# preview the inputs without writing anything
vasp-auto inputs/Al --dry-run

# prepare job files but do not run
vasp-auto inputs/Al --prepare

# re-parse finished jobs and regenerate the Excel
vasp-auto inputs --parse-only

# rerun only the cases that did not converge (CONTCAR continues as POSCAR,
# WAVECAR/CHGCAR are kept as the SCF seed)
vasp-auto inputs --retry-failed

# run in the background (log under vasp_auto_background_logs/)
vasp-auto inputs --background
```

---

## 4. Configuration (`config.yaml`)

Search order: `./config.yaml` → `$VASP_AUTO_ROOT/config.yaml` → repo root →
`~/.vasp_auto/config.yaml` → `/opt/vasp_auto/config.yaml`. A `config.yaml`
inside the **project directory** and again inside a **case directory**
overlays the global one (global → project → case).

```yaml
vasp_executable: /home/vv/src/vasp.6.4.2/bin/vasp_std
jobs_root: /home/vv/vasp_auto/jobs
potcar_root: /home/vv/vasp_auto/POTCAR
neb_images: 5

potcar_map:            # pseudopotential variants per element
  Fe: Fe_pv
  O: O_s

magmom_map:            # initial moments (μB) for --spin runs
  Fe: 5.0
  O: 0.6

scheduler: local       # local | slurm | pbs
job_template: /path/to/submit_template.sh    # optional custom script
scheduler_options:     # extra lines injected into the submit script
  - "#SBATCH --partition=standard"
  - "module load vasp/6.4.2"

workflow:              # optional default workflow for every case
  - calc_type: relax
  - calc_type: scf
```

---

## 5. Calculation types

`--calc-type` selects a pre-validated INCAR template from `example/`:

| type    | template                  | purpose |
|---------|---------------------------|---------|
| scf     | INCAR_scf                 | static single point (default) |
| relax   | INCAR_optimize_structure  | geometry optimisation (IBRION=2) |
| dos     | INCAR_dos                 | non-SCF DOS (ICHARG=11, needs CHGCAR) |
| bands   | INCAR_bands               | band structure (ICHARG=11 + line-mode k-path) |
| charge  | INCAR_charge_density      | high-quality CHGCAR export |
| neb     | INCAR_neb                 | climbing-image NEB |
| md      | INCAR_md                  | ab-initio MD (NVT) |
| phonon  | INCAR_phonon              | DFPT phonons/dielectric |
| hse06   | INCAR_hse06               | hybrid functional (accurate gaps, slow) |

```bash
vasp-auto inputs/Al --calc-type relax
```

A case-supplied `INCAR` always takes precedence (a note is printed). Edit the
files in `example/` to change the defaults — no Python required.

### Open-source engine: Quantum ESPRESSO (no VASP licence needed)

VASP requires a paid licence. If you don't have one, run the **same** cases
through the open-source plane-wave code **Quantum ESPRESSO** (`pw.x`) with
`--engine qe`. The structure builders, job layout, Excel summary and the whole
web UI work identically — only the input file (`pw.in` instead of
`INCAR`/`KPOINTS`/`POTCAR`) and the run/parse steps differ.

```bash
# one-off:
vasp-auto inputs/Si --engine qe --pseudo-dir ~/qe/pseudo --calc-type relax --kmesh 6x6x6
# or make it the default in config.yaml:  engine: qe
```

Supported calc types: `scf`, `relax`, `vcrelax` (variable-cell relaxation),
`dos`, `bands`. NEB, phonons, convergence scans, workflow chaining and implicit
solvation remain VASP-only for now.

QE runs on remote machines through the same three modes as VASP — direct SSH
(`run_mode: ssh`), scheduler submission (`slurm`/`pbs`) and detached offload
(`ssh_detached`). Give the machine a `qe_executable` in remotes.json (default:
`pw.x` on its PATH) and, optionally, a `pseudo_dir` pointing at its own UPF
library; without one, offload ships the locally-resolved UPF files alongside
the POSCAR, so the remote needs no pseudopotential setup at all.

The machine's `env_setup` (its VASP toolchain, e.g. Intel oneAPI) is **not**
sourced for QE runs — an Intel `mpirun` first on PATH silently runs an
OpenMPI-built `pw.x` as N duplicate serial jobs. If the machine's QE build does
need environment (modules, a custom MPI), set `qe_env_setup` on the machine.

Pseudopotentials are **UPF** files (the QE analogue of POTCAR). Point
`pseudo_dir` at a directory of `.UPF` files; for each element the finder picks
the first file named `<El>.*.UPF`, or use `pseudo_map` to choose an exact file:

```yaml
engine: qe
qe_executable: pw.x
pseudo_dir: /home/vv/qe/pseudo
pseudo_map:
  Fe: Fe.pbe-spn-kjpaw_psl.1.0.0.UPF
qe_ecutwfc: 50      # plane-wave cutoff (Ry)
qe_ecutrho: 400     # density cutoff (Ry)
```

`--dry-run` prints the generated `pw.in`; a case that ships its own `pw.in`
overrides the generated one (same rule as a case-supplied INCAR). Each QE job
directory carries a `.engine` marker and writes `pw.out`, which the parser reads
for energy, convergence, forces and pressure into the summary.

### Generic ASE-calculator engine (any code, including DMol3)

`--engine ase` runs a case through **any calculator the [ASE](https://wiki.fysik.dtu.dk/ase/)
library can drive** — EMT, Lennard-Jones, Morse, Quantum ESPRESSO, GPAW, ABINIT,
CASTEP, SIESTA, NWChem, **DMol3** (BIOVIA Materials Studio), an ASE-driven VASP,
or an MLIP (MACE). Instead of INCAR/KPOINTS/POTCAR it writes a small,
self-contained `run_ase.py` driver plus `ase_calc.json`, so the job directory can
be re-run without vasp_auto on the path. `emt` is the default and needs no
external code, so the engine is testable anywhere.

```bash
# single-point with the built-in EMT potential (no external code):
vasp-auto inputs/Cu --engine ase --ase-calculator emt

# DMol3 geometry relaxation:
vasp-auto inputs/Au13 --engine ase --calc-type relax \
  --ase-calculator dmol3 \
  --ase-command "RunDMol3.sh PREFIX > PREFIX.out" \
  --ase-params '{"functional":"pbe","basis":"dnd","symmetry":"off"}'
```

or make it the default in `config.yaml`:

```yaml
engine: ase
ase_calculator: dmol3
ase_calc_params:
  command: RunDMol3.sh PREFIX > PREFIX.out   # the PREFIX token is required
  functional: pbe
  basis: dnd
  symmetry: "off"
ase_fmax: 0.05      # force tolerance for relax (eV/Å)
ase_steps: 200      # max BFGS optimiser steps
```

Supported calc types: `scf` (single-point energy) and `relax` (BFGS). The chosen
code and ASE must be installed where the job runs. Flags: `--ase-calculator NAME`,
`--ase-command PATH` (run command/binary → `ase_calc_params.command`; for file-IO
codes like DMol3 it **must contain the `PREFIX` placeholder**, which ASE replaces
with the job label — or set the code's env var, e.g. `DMOL_COMMAND`, and leave it
blank), `--ase-params JSON` (extra calculator keywords; use the special keys
`__module__`/`__class__` to reach a calculator not in the built-in menu),
`--ase-fmax`, `--ase-steps`. Each ASE job writes `CONTCAR` and `ase_results.json`
(energy, max force, ionic steps, converged), which the summary reads. NEB,
phonons, convergence scans and implicit solvation remain VASP-only for now.

### Spin-polarised runs

```bash
vasp-auto inputs/Fe2O3 --spin                       # ISPIN=2, guessed MAGMOM
vasp-auto inputs/Fe2O3 --spin --magmom "Fe:5.0,O:0.6"
```

`--spin` sets `ISPIN = 2` and derives a `MAGMOM` line from the POSCAR
composition (sensible per-element starting moments; override with `--magmom`
or a `magmom_map:` in config.yaml). A MAGMOM already present in the case
INCAR is kept. Works for single runs, previews, and workflow steps
(`spin: true` per step in workflow.yaml also works). Summary rows then pick
up the total magnetisation and per-atom moments from OUTCAR.

## 6. KPOINTS generation

```bash
vasp-auto inputs/Al --kmesh 6x6x6                      # Gamma-centred mesh
vasp-auto inputs/Al --kpoints-mode mp --kmesh 4x4x1    # Monkhorst-Pack
vasp-auto inputs/Al --kspacing 0.03                    # mesh from density (1/Å)
vasp-auto inputs/Al --calc-type bands --kpath fcc      # line-mode path preset
vasp-auto inputs/Al --kpath "G 0 0 0; X 0.5 0 0.5" --kpath-divisions 30
```

Presets: `cubic`, `fcc`, `bcc`, `hex`. Without any k-flag, the case `KPOINTS`
(or a Gamma 1×1×1 default) is used.

A special value `--kpath auto` auto-detects the lattice type from the POSCAR
lattice vectors (pure Python, no spglib needed) and selects the matching preset.
It raises an error for lattice types it cannot classify (generic/low-symmetry)
— in that case specify `--kpath` explicitly.

```bash
vasp-auto inputs/Al --calc-type bands --kpath auto     # auto-detects fcc
```

## 7. Workflows (chained steps)

Run relax → scf → dos in one command; outputs feed forward automatically
(CONTCAR → POSCAR; CHGCAR for dos/bands):

```bash
vasp-auto inputs/Al --workflow "relax,scf,dos"
```

or put a `workflow.yaml` in the case directory:

```yaml
steps:
  - calc_type: relax
  - calc_type: scf
    incar:            # per-step INCAR overrides
      ENCUT: 450
  - calc_type: dos
    kpoints: 8x8x8    # per-step mesh; or kpath: fcc for bands
```

Priority: `--workflow` flag > case `workflow.yaml` > config `workflow:` key.
Each step runs in `jobs/<case>/NN_<type>/`.

### Convergence as a workflow step

A workflow can start with a `converge` step — the classic
"convergence → optimisation → SCF → DOS" flow. It runs a short SCF scan (see
§8), then **carries the chosen ENCUT / SIGMA / NELM / k-mesh forward as
overrides for every later step**:

```bash
vasp-auto inputs/Al --workflow "converge,relax,scf,dos"
```

With no settings the converge step runs the default NELM + k-mesh scan; give it
explicit ranges in `workflow.yaml` to also scan ENCUT and SIGMA:

```yaml
steps:
  - calc_type: converge
    encut: "400,450,500"      # blank ENCUT/SIGMA → just NELM + k-mesh
    sigma: "0.2,0.1,0.05"
    kpoints: "3,4,5,6"
    energy_tol: 1e-4
  - calc_type: relax          # inherits the converged ENCUT/SIGMA/k-mesh
  - calc_type: scf
  - calc_type: dos
```

The scan output (CSV + Markdown report) lands in
`jobs/<case>/NN_converge/scf_convergence/`. A later step's own `incar:` /
`kpoints:` keys still override the carried settings. In the GUI, the Workflow
tab has a **Converge → Optimise → SCF → DOS** preset and a *Convergence step
settings* panel; the Calculate tab keeps the standalone convergence scan.

## 8. Convergence scans

```bash
vasp-auto inputs/Al --converge-encut 400,450,500,550        # ENCUT only
vasp-auto inputs/Al --converge-sigma 0.2,0.1,0.05           # smearing width
vasp-auto inputs/Al --converge-scf                          # NELM + KPOINTS
vasp-auto inputs/Al --converge-encut 400,450,500 --converge-scf \
    --nelm-values 40,60,80 --kpoints-values 3,4,5,6 --energy-tol 1e-4 \
    --reuse-wavecar
```

Stages run ENCUT → SIGMA → NELM → KPOINTS; each energy-based stage stops at
the first converged trial whose energy change is below `--energy-tol`, and
the selected value is held for the later stages. The SIGMA stage instead
selects the largest smearing whose entropy term T*S stays below
`--sigma-tol` (default 1 meV/atom) — the standard VASP guidance.
`--reuse-wavecar` seeds each trial with the previous trial's
WAVECAR/CHGCAR to cut wall time. Results: `jobs/<case>/scf_convergence/`
with a Markdown report and a CSV of every trial.

## 9. Structure building

Pure Python (no ASE needed):

```bash
vasp-auto inputs/Al --supercell 2x2x2 --build-only
vasp-auto inputs/Al --vacancy 5 --build-only             # remove atom 5
vasp-auto inputs/Al --substitute 12=Mg --build-only      # dope site 12
vasp-auto inputs/Al --add-atom "H@0.5,0.5,0.5" --build-only   # (= --interstitial)
vasp-auto inputs/Al --move-atom "5+0,0,0.05" --build-only     # shift atom 5
vasp-auto inputs/Al --move-atom "5@0.5,0.5,0.6" --build-only  # place atom 5
vasp-auto inputs/Al --scale-cell 1.02 --build-only            # 2% strain
vasp-auto inputs/Al --scale-cell "a=4.1,c=22.5" --build-only  # absolute Å
vasp-auto inputs/slab --freeze "z<0.45" --build-only          # fix bottom layers
vasp-auto inputs/slab --freeze "1-8:XY" --build-only          # fix x,y of atoms 1–8
```

`--freeze` turns on Selective dynamics (`F` = fixed); the `z<frac` form picks
every atom below that fractional height — the natural way to fix the bottom
layers of a slab. `--scale-cell` resizes the lattice with atoms following
fractionally (use it for strain studies and lattice-parameter tuning).

### Adsorption cases in one command

```bash
# O adsorbate 2.0 Å above surface atom 9, bottom layers frozen:
vasp-auto inputs/Pt111 --adsorbate "O@9+2.0" --freeze "z<0.45" --build-only
vasp-auto inputs/Pt111_adsO9_frz --calc-type relax --kmesh 4x4x1 -n 16
```

`--adsorbate "El@N+h"` places element El directly above atom N at height h Å
(along Cartesian z, the slab normal). E_ads = E(slab+adsorbate) − E(slab)
− E(adsorbate molecule); build the molecule reference with
`--ase-build-molecule O2 --ase-box 15`. The UI Build tab has a matching
"🧲 Adsorption quick build" card (adsorbate + freeze in one click), and the
edit card exposes move-atom/add-atom/cell-size/freeze with X/Y/Z axis
checkboxes. Note: ASE-built slabs are centred in the cell — check the frac-z
of your bottom layer (≈0.42 for a 3-layer slab) before choosing the
`z<…` threshold; an empty selection raises an error instead of silently
freezing nothing.

ASE-backed (install with `venv/bin/pip install ase`):

```bash
vasp-auto --ase-build-bulk Al --ase-crystalstructure fcc --ase-a 4.05
vasp-auto --ase-build-slab Al --ase-miller 1,1,1 --ase-layers 4 \
          --ase-vacuum 12 --ase-repeat 3x3
vasp-auto --ase-build-molecule H2O --ase-box 12
vasp-auto --ase-import structure.cif
```

Crystals can also be generated from a space group + Wyckoff basis (one element
and one fractional site per basis entry; symmetry fills in the rest), and
single-wall nanotubes from their chiral indices:

```bash
vasp-auto --ase-build-crystal "Na Cl" --ase-spacegroup 225 \
          --ase-basis "0,0,0;0.5,0.5,0.5" --ase-a 5.64 --build-only
vasp-auto --ase-build-crystal "Ti O" --ase-spacegroup 136 \
          --ase-basis "0,0,0;0.305,0.305,0" --ase-a 4.59 --ase-c 2.96 --build-only
vasp-auto --ase-build-nanotube C --ase-nt-n 5 --ase-nt-m 5 --ase-nt-length 3 --build-only
```

`--ase-build-crystal` accepts `--ase-b`/`--ase-c` and `--ase-alpha`/`--ase-beta`/
`--ase-gamma` (default cubic, 90°); `--ase-build-nanotube` accepts `--ase-nt-bond`
and reuses `--ase-vacuum` for the surrounding empty space.

All builders write a new case directory (named automatically or via
`--ase-output DIR`) and then continue into preparation/run unless
`--build-only` is given. Edits compose: build bulk → supercell → vacancy in
one command. TSS/NEB interpolation can use ASE IDPP via `--ase-neb`.

### Combining two structures (deposition / heterostructures)

Two structures with *different unit cells* can be merged into one cell —
e.g. an Au crystal deposited on a graphite sheet:

```bash
vasp-auto inputs/graphite --combine inputs/Au --combine-gap 2.5 \
          --combine-vacuum 12 --build-only
```

`--combine-mode stack` (default) keeps the host's in-plane lattice, places
the guest `--combine-gap` Å above the host's top atom and extends the c axis
to leave `--combine-vacuum` Å above the guest. The guest keeps its own
geometry centred over the cell; add `--combine-strain` to strain it onto the
host lattice instead (epitaxial match), and `--combine-shift "x,y"` to slide
it laterally (fractions of the host a/b). `--combine-mode insert` keeps the
host cell unchanged and just drops the guest atoms in. Selective-dynamics
flags survive from both sides. The same function is in the UI Build tab
("🧬 Combine two structures") where the result opens in the editor for
inspection before saving.

### Prototype crystals (no ASE needed)

A small built-in library covers compounds ASE's `bulk()` cannot build:

```bash
vasp-auto --build-prototype "graphene:vacuum=20" --build-only
vasp-auto --build-prototype "rutile-TiO2" --build-only
vasp-auto --build-prototype "anatase-TiO2:a=3.80,c=9.6" --build-only
```

Available: `graphene`, `graphite`, `rutile-TiO2`, `anatase-TiO2`, `hBN`
(aliases like `rutile`, `bn` work). `a=…`/`c=…` override the tabulated
lattice constants; for 2D sheets `vacuum=…` sets the box height. The UI
Build tab has a matching "Prototype crystal" card.

### Matching different unit cells (`--match-cells`)

When host and guest cells are incommensurate, ask for supercell pairs that
bring them into registry before combining:

```bash
vasp-auto inputs/graphene --match-cells inputs/tio2_slab --build-only
# tighter/looser search:
#   --match-max 8 (repeats), --match-strain 0.06, --match-gamma-tol 15 (deg)
```

It prints a table (host × guest repeats, strain per axis, total atoms) and
the exact follow-up `--supercell`/`--combine --combine-strain` commands.
Large printed strain or angle mismatch means you should combine *without*
`--combine-strain` instead (the guest sits unstrained, centred). The whole
slab-on-sheet workflow (a TiO₂(111) slab on graphene) is walked through in
**`docs/TUTORIAL_HETEROSTRUCTURE.md`**.

### Machine-learning pre-relaxation (Meta OMat24)

```bash
vasp-auto inputs/Fe2O3 --ml-relax -n 16        # ML relax, then VASP from there
vasp-auto inputs/Fe2O3 --ml-relax --ml-only    # ML relax only, no VASP
vasp-auto inputs/Al --ml-relax --ml-model emt --ml-only   # no-install demo
```

`--ml-relax` relaxes the case POSCAR with a machine-learned interatomic
potential **before** VASP and continues from the relaxed geometry (written to
a derived `<case>_ml` directory, original untouched). MLIP minima typically
sit within a few meV/atom of the PBE minimum, so the subsequent VASP
relaxation converges in a fraction of the ionic steps.

The default backend is Meta FAIR's **UMA / OMat24** family
(<https://huggingface.co/facebook/OMAT24>, arXiv:2410.12771), trained on
~110 M inorganic DFT calculations:

```bash
venv/bin/pip install fairchem-core     # one-time; UMA weights are gated:
huggingface-cli login                  # accept the licence on Hugging Face
```

Flags / config keys: `--ml-model` (`ml_model:` in config.yaml; default
`uma-s-1p1`; `emt` = ASE demo potential for simple metals, no install),
`--ml-task` (`omat` materials default, `oc20` catalysis, `omol` molecules,
`odac` MOFs), `--ml-checkpoint` (a downloaded fairchem 1.x OMat24 eqV2
checkpoint file), `--ml-fmax` (default 0.05 eV/Å), `--ml-steps` (default
200), `--ml-relax-cell` (also relax the cell), `--ml-only` (stop after ML).

The derived case contains an `XDATCAR` of the ML optimisation path, so the
UI's 🎞 animation works on it. The same feature is in the UI Build tab
("🤖 ML pre-relax"). Composes with the builders: `--ase-build-slab Pt
--ml-relax` builds, ML-relaxes, then runs VASP.

### MLIP single-point screen (--ml-energy)

```bash
vasp-auto --ml-energy inputs/Al --ml-model emt        # quick energy + max|F|
vasp-auto --ml-energy inputs/Fe2O3 --ml-model uma-s-1p1 # OMat24 screen (needs fairchem)
```

`--ml-energy TARGET` is a read-only screen: it reads the POSCAR in TARGET (a
case directory or a POSCAR file), calls `ml_tools.ml_energy`, prints the energy
and maximum force, and exits without writing any files. Useful for ranking
structures before committing to VASP runs. Honors `--ml-model`, `--ml-task`,
`--ml-checkpoint`. Also available as `POST /api/mlenergy` in the UI.

## 10. Running, schedulers, results

```bash
vasp-auto inputs/Al -n 16                      # mpirun -np 16, wait, parse
vasp-auto inputs/Al --scheduler slurm          # write submit.sh, sbatch it
vasp-auto inputs --parse-only                  # harvest results later
```

For slurm/pbs the row records the queue job id with status `submitted`;
collect results after the queue finishes with `--parse-only`.

### Polling a submitted job (--poll)

```bash
vasp-auto --scheduler slurm --poll 123456   # query squeue for job 123456
vasp-auto --scheduler pbs --poll 456.cluster
```

`--poll JOBID` queries the scheduler for the current state of a submitted job
and prints: `running`, `pending`, `completed`, or `unknown`. It exits
immediately without running any VASP or writing files. Gracefully returns
`unknown` when the binary (`squeue` or `qstat`) is not on PATH or the job ID
has been purged from the scheduler history (which typically means it completed).

### Running on a remote machine (`--remote`)

vasp_auto can prepare the inputs locally and run them on another machine over
SSH. Configure machines in the UI's **Remote** tab (saved to `remotes.json`) or
in `config.yaml` (a `remote:` block, or a `remotes:` map of named machines).
Each machine needs `host`, `remote_root` (a base work directory on the remote),
and `vasp_executable`; optional `user`, `port`, `ssh_key`, `ssh_options`.

```bash
vasp-auto inputs/Al --remote NAME      # run on the machine remotes[NAME]
vasp-auto inputs/Al --remote           # use the single config.yaml remote: block
```

Each machine has a **run mode** (`run_mode:`):

* **`ssh`** (direct `mpirun` over SSH) — for a workstation with no working
  scheduler. vasp_auto rsyncs the inputs to `<remote_root>/<case>`, runs
  `mpirun` there **synchronously**, then copies the results back so the local
  viewers/parsers and the Excel summary work unchanged. Convergence scans and
  chained workflows are supported in this mode (each trial/step runs on the
  remote). If the remote needs its toolchain set up in a non-interactive shell
  (Intel oneAPI, modules, …), put the command in **`env_setup:`**, e.g.
  `env_setup: "source /opt/intel/oneapi/setvars.sh"` — it is sourced before
  `mpirun` so MKL/MPI libraries are on the path.
* **`slurm` / `pbs`** (queue submission) — vasp_auto copies the inputs and
  submits `sbatch`/`qsub` over SSH, then exits, so the local host can be turned
  off. The row records the queue job id with status `submitted`; harvest later
  with the Results tab's 🛰 status / ⬇ fetch buttons (or `--parse-only`).

If `vasp_executable` points at a directory (e.g. `…/bin`), vasp_auto assumes the
standard `…/bin/vasp_std` binary. Either way the results row is tagged with the
machine name (shown in the Results-tab **machine** column) and the remote
directory the files live in.

#### Offload mode — run with the laptop off (`run_mode: ssh_detached`)

The `ssh` mode above keeps the local machine connected for the whole run. To
**offload** a calculation and power the laptop off, set the machine's
`run_mode: ssh_detached`. The full vasp_auto engine runs on the remote, so this
also covers the iterative paths (convergence scans, chained workflows) — not
just a single VASP launch.

One-time setup per machine (installs a private venv with vasp_auto + deps under
`<remote_root>/.vasp_auto`; needs Python 3.12 and internet on the remote):

```bash
vasp-auto --remote NAME --remote-setup          # or the UI Remote tab's
                                                #  "⚙ Set up offload engine" button
```

Then run as usual with the machine selected — vasp_auto ships the inputs
(POSCAR + a pre-built POTCAR, so the remote needs no POTCAR library), launches
`vasp-auto` on the remote under `setsid` (detached), records the PID, and
returns immediately:

```bash
vasp-auto inputs/Al --converge-encut 400,500,600 --remote NAME
# -> "Launched: NAME pid 12345 -> <remote_root>/results/Al"  then exits
```

The job keeps running after you disconnect. Later, check progress and pull the
results back from the Results tab (🛰 status polls the remote PID; ⬇ fetch copies
the finished files into the local job dir so the viewers/Excel work). If the
remote needs its toolchain in a non-interactive shell, set `env_setup` (e.g.
`source /opt/intel/oneapi/setvars.sh`) — it is sourced before the run.

The Excel summary contains: energy, convergence flag (colour-coded), Fermi
level, band gap, max force, pressure (from vasprun.xml), total magnetisation
and per-atom moments (spin runs), NEB barriers (total/forward/backward) and
per-image energies, ionic steps, detected errors, and the queue job id —
plus a bar chart of energy per case when there are two or more energies.

Known VASP failures (ZBRENT, EDDDAV, RHOSYG, Sub-Space-Matrix, ZPOTRF,
PRICEL, SGRCON, TOO FEW BANDS, "SICK JOB") are detected in `run.log`/`OUTCAR`
after every run; a one-line fix hint is printed and stored in the `errors`
column.

### Calculation reports

```bash
vasp-auto inputs/Al --report               # with a run
vasp-auto inputs --parse-only --report     # for finished jobs
```

`--report` writes a short Markdown `report.md` into every job directory:
calculation details (key INCAR tags, k-points), results (free energy, Fermi
level, band gap, forces, magnetisation, ionic steps), the NEB energy profile
with barriers, and any detected problems with fix hints. The UI offers the
same report per case from the Results tab (📄), with a download link.

### Implicit solvation (--solvation)

```bash
vasp-auto inputs/Al --solvation                       # water (EB_K=78.4)
vasp-auto inputs/Al --solvation --solvation-eps 36.6  # acetonitrile
```

`--solvation` injects `LSOL = .TRUE.` and `EB_K = <eps>` into the INCAR after
the template is loaded. This enables VASPsol's continuum solvation model, which
treats the solvent as a dielectric continuum around the solute/slab.

**Requirements**: a **VASPsol-patched VASP binary** (see
<https://github.com/henniggroup/VASPsol>). Without the patch VASP will abort
with an "unknown tag" error when it encounters `LSOL`. The `--solvation-eps`
flag (default 78.4 = water) sets the dielectric constant ε. Other common
values: 36.6 (acetonitrile), 24.9 (ethanol), 10.4 (dichloromethane).

The `solvation:` key in config.yaml enables it for all cases in a project;
`solvation_eps:` sets the dielectric constant.

### Automatic error recovery

```bash
vasp-auto inputs/Al --auto-retry 2
```

When a run fails with a known error that has a safe INCAR fix (e.g. EDDDAV →
`ALGO = All`, RHOSYG → `ISYM = 0`), the fix is applied to the job's INCAR and
the run repeated, up to N times. The summary records `auto_retries` and
`auto_fixes`. Errors without a safe generic fix (e.g. TOO FEW BANDS) are
never auto-fixed — the hint is printed instead. Local scheduler only.

## 11. Graphical UI

```bash
vasp-auto-ui              # opens http://127.0.0.1:8800/ (local only)
vasp-auto-ui --port 9000  # different port
```

A workflow-builder front end in the browser, four tabs sharing one case selector,
with inline help everywhere (toggle with the **? help** button), a light/dark
theme, and a first-run guide:

- **Build** — a full interactive structure editor (all open
  source). A **Build function** catalog at the top-left lists every builder
  grouped by purpose (Crystals, Surfaces & nano, Import, Interfaces, Defects &
  adsorption, Transition states, Refine) with a search box; pick one and only
  its form opens — you no longer scroll a stack of every builder at once.
  Builders (ASE bulk, prototype crystals, **space-group crystal from a Wyckoff
  basis**, slab with facet control, molecule-in-box, **single-wall nanotube**,
  CIF/XYZ import, two-structure combine, adsorption quick build, ML pre-relax,
  and an **AI builder** — describe a structure in plain words and it picks the
  recipe; works with **any OpenAI-compatible API** (Groq by default, or OpenAI,
  OpenRouter, DeepSeek, Anthropic, a local Ollama/LM Studio server, … — pick the
  provider, paste a key, optionally name a model; the AI only chooses the build
  command, coordinates are always built exactly by the code)
  load the result into an in-memory editor; **nothing is written until you press
  💾 Save** — only calculation outputs are saved automatically. In the 3D
  viewer you control single atoms with the cursor: click to select
  (shift-click extends), switch to ✥ move and drag atoms in the screen
  plane, nudge with the arrow keys (Shift = ×10, PgUp/PgDn = depth), Delete
  removes, Ctrl+Z/Ctrl+Y undo/redo; the ▭ select tool rubber-band-selects
  groups, clicking an element in the legend selects every atom of it, and
  selecting 2 or 3 atoms shows the **distance / angle** live. Toolbar
  toggles show bonds (covalent radii), atom number labels, the cell,
  an a/b/c axes gizmo and translucent periodic boundary images. The
  inspector panels edit the cell as a,b,c,α,β,γ (atoms following
  fractionally or staying in Å), wrap atoms, make supercells, edit a
  selected atom's element / fractional / Cartesian coordinates and
  per-axis freeze flags, and show its **coordination number with a
  neighbour-distance table**. A live POSCAR panel sits beside the viewer
  (hideable, copyable, and editable — apply text back to the editor).
  Relaxations/NEB paths can still be **animated** in the same viewer
  (play/pause + frame slider).
- **Calculate** — a **DFT engine** selector (VASP, Quantum ESPRESSO, or any
  **ASE calculator** — choose the calculator, e.g. DMol3, its run command/path,
  and JSON parameters, which pre-fill with sensible defaults); calculation types
  shown as cards with plain-language descriptions; spin/magnetism options;
  k-point fields that adapt to the chosen mode; convergence testing
  (ENCUT/SIGMA/NELM/k-mesh) in a collapsible panel; an auto-fix-and-retry switch;
  Preview/Prepare/Run/Parse buttons; an INCAR editor (load the case INCAR, the
  template for the selected type, or **Load from another work…** — pick any
  other case or finished job folder and its INCAR is copied into the editor,
  then *Save to case* to use it); and a live job console with
  elapsed time and a stop (✕) button per job (logs under `ui_logs/`).
  **Resume** restarts the case's latest job in place from its newest CONTCAR,
  reusing the job's own KPOINTS/POTCAR: a prompt first asks for the CPU count
  (blank = reuse the previous run's, read from its OUTCAR), then the job's
  INCAR opens as an editable tag/value spreadsheet — change values, add or
  remove tags (e.g. raise `NSW`), and *Save & resume*; cancel aborts the
  restart. Works for local and remote jobs alike (the edited INCAR is shipped
  back over SSH).
- **Workflow** — one-click presets (Optimise → SCF → DOS …, including
  **Converge → Optimise → SCF → DOS**), reorderable steps, optional spin, a
  *Convergence step settings* panel (ENCUT/SIGMA/NELM/k-mesh) whose values a
  leading `converge` step carries into the later steps, or save a
  `workflow.yaml` into the case for per-step overrides.
- **Results** — friendly summary table (status badges, ✓/✗ convergence,
  energies, band gap, magnetisation, NEB barrier, error hints on hover),
  optional auto-refresh, and a download link for the Excel summary. Per
  case: 📄 generate/view the Markdown report, 📈 plot the density of states
  (spin-resolved, Fermi-aligned), 🧬 projected DOS (per element/orbital,
  with an atom-selection filter), 🎼 band structure along the k-path,
  ⚡ volumetric data (CHGCAR/LOCPOT/AECCAR planar average + colour-map
  slice), 🎞 animate the relaxation or NEB path, and 🧪 the analysis card
  (ZPE/Gibbs thermochemistry, d-band center, work function with potential
  profile, Bader charges, absorption spectrum — the same maths as the CLI
  analysis commands). Standalone cards compute the charge-density
  difference Δρ and the three-job adsorption energy. Every chart exports
  as PNG and every dataset as CSV. Rows have checkboxes for bulk actions —
  delete, terminate, or **resume** the selected jobs in place from their
  newest CONTCAR; a bulk resume asks once for the CPU count, then shows each
  job's INCAR in the same editable spreadsheet before that job restarts
  (cancelling one skips just that job).

Cross-platform notes (Linux / macOS / Windows) live in
`docs/PORTABILITY.md`; on non-OpenMPI hosts set `VASP_AUTO_MPI=mpiexec`.

The UI is a separate package (`src/vasp_auto_ui/`, stdlib-only) that calls
the same engine functions as the CLI and launches runs through the CLI
itself, so behaviour is identical either way.

## 12. Self-check

`selfcheck/` contains example inputs for **every** feature and fake
`mpirun`/`vasp_std`/`sbatch` binaries, so the whole tool can be verified on
any machine in seconds, without VASP or a cluster:

```bash
~/vasp_auto/selfcheck/run_selfcheck.sh
```

41 checks cover: dry-run, templates, k-point modes, full runs, vasprun
parsing, error detection, parallel projects, workflows, structure tools,
prototype crystals, cell matching and stacking, convergence, scheduler
submission, NEB interpolation, per-case config, ASE builders, parse-only,
the adsorption-energy command, the web UI API, and the 236-test pytest
suite. Exit code 0 = all green.
Everything it writes stays inside `selfcheck/` (`jobs/`, `build/`, `logs/`).

Unit tests alone: `cd ~/vasp_auto && venv/bin/python -m pytest`.

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `VASP executable not found` | set `vasp_executable` in config.yaml or `export VASP_EXECUTABLE=...` |
| `Could not find a POTCAR library...` | check `potcar_root` and `potcar_map`; the error lists every directory checked and what was missing |
| `Cannot identify calculation type` | a case needs `POSCAR`, or `initial/POSCAR` + `final/POSCAR` for NEB |
| `VASP error: ZBRENT ...` printed | follow the printed hint (e.g. reduce POTIM); the case INCAR is yours to edit |
| command not found after moving the repo | re-create the symlink (§1) |
| GUI for structures | `~/vasp_auto/ase-gui` opens the ASE GUI from its own toolchain |
