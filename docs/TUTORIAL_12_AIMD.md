# Tutorial 12 — Ab-Initio Molecular Dynamics (AIMD)

Ab-initio molecular dynamics (AIMD) propagates ions on the Born–Oppenheimer
surface at a finite temperature, capturing anharmonic effects, diffusion,
phase transitions, and reaction pathways that static DFT misses. This tutorial
runs a **300 K NVT AIMD** simulation on a small **silicon melt seed** (8-atom
cell) for 500 steps using a Nosé–Hoover thermostat. The same recipe scales to
liquid metals, oxide melts, adsorbate dynamics, and transition-state sampling.

---

## Prerequisites

- `inputs/Si_md/POSCAR` — start from the relaxed Si bulk structure (Tutorial
  01/02) or any other starting geometry.
- `config.yaml` with `vasp_executable`, `potcar_root`, and POTCAR library.
- At least 8 CPU cores available; AIMD is communication-intensive, so MPI
  efficiency is high.

---

## Step 1 — Prepare the starting POSCAR

Use the relaxed bulk Si cell directly. To introduce some thermal disorder you
can slightly displace the atoms (optional):

```bash
# Start from the Tutorial 01 POSCAR:
mkdir -p inputs/Si_md
cp inputs/Si_bulk/POSCAR inputs/Si_md/POSCAR
```

If you want to start from the VASP-relaxed geometry:

```bash
cp jobs/Si_bulk/Si_bulk/CONTCAR inputs/Si_md/POSCAR
```

---

## Step 2 — Run the AIMD simulation

```bash
vasp-auto inputs/Si_md \
  --calc-type md \
  --kpoints-mode gamma --kmesh 2 \
  --cpus 16
```

The `md` calc type loads `example/INCAR_md`:

```
IBRION = 0      # molecular dynamics
NSW    = 1000   # steps (change in your INCAR if needed)
POTIM  = 1.0    # timestep in fs
TEBEG  = 300    # initial temperature (K)
TEEND  = 300    # final temperature (K)
SMASS  = 0      # Nosé–Hoover thermostat (NVT)
ALGO   = Fast   # mixed Davidson/RMM-DIIS for speed
ISMEAR = 0      # Gaussian smearing
SIGMA  = 0.1    # wider smearing is acceptable at finite T
```

For a production run you would override NSW and POTIM in a case INCAR:

```bash
mkdir -p inputs/Si_md
# POSCAR already in place
cat > inputs/Si_md/INCAR << 'EOF'
SYSTEM = Si AIMD 300K NVT

ENCUT  = 400
PREC   = Normal
EDIFF  = 1E-5
ALGO   = Fast

IBRION = 0
NSW    = 500
POTIM  = 2.0
TEBEG  = 300
TEEND  = 300
SMASS  = 0
ISYM   = 0
NELMIN = 4

ISMEAR = 0
SIGMA  = 0.1

LCHARG = .FALSE.
LWAVE  = .FALSE.
EOF
```

Then run:

```bash
vasp-auto inputs/Si_md \
  --kpoints-mode gamma --kmesh 2 \
  --cpus 16
```

vasp_auto detects the existing `INCAR` in the case directory and uses it
verbatim (highest-priority source). The k-mesh is still applied from the CLI
flag.

---

## Step 3 — Monitor the run

AIMD runs stream `run.log` to the terminal. Each ionic step prints the
electronic convergence iterations and the total energy. To follow the log in
another terminal:

```bash
tail -f jobs/Si_md/Si_md/run.log
```

Or launch the calculation in the background and monitor:

```bash
vasp-auto inputs/Si_md --kmesh 2 --cpus 16 --background
tail -f vasp_auto_background_logs/vasp_auto_*.log
```

---

## Step 4 — Run multiple AIMD simulations in parallel

To run a temperature-ramp series (300 K, 500 K, 700 K), create three case
directories each with their own INCAR (different TEBEG/TEEND) and use
`--parallel`:

```bash
# inputs/Si_md_300K/POSCAR + INCAR (TEBEG=300, TEEND=300)
# inputs/Si_md_500K/POSCAR + INCAR (TEBEG=500, TEEND=500)
# inputs/Si_md_700K/POSCAR + INCAR (TEBEG=700, TEEND=700)

vasp-auto inputs/ --cases Si_md_300K Si_md_500K Si_md_700K \
  --parallel 3 --cpus 8
```

Each case gets 8 CPUs and all three run concurrently. `--cpus 8` is passed to
every `mpirun` subprocess.

---

## Step 5 — Analyse the XDATCAR trajectory

VASP writes `XDATCAR` (ionic coordinates every step) and `OSZICAR` (energies
and temperatures every step). Parse the trajectory:

```python
from vasp_auto.trajectory import parse_xdatcar

frames = parse_xdatcar("jobs/Si_md/Si_md/XDATCAR")
print(f"{len(frames)} frames, {len(frames[0]['positions'])} atoms per frame")

# Frame zero positions (fractional):
for pos in frames[0]["positions"]:
    print(pos)
```

The web UI (`vasp-auto-ui`) renders the XDATCAR animation in the Results tab.
Click the Si_md job row and press Play.

---

## Step 6 — Submit long AIMD runs to a cluster (SLURM)

For a 10 000-step production run, use the SLURM scheduler:

```bash
vasp-auto inputs/Si_md \
  --calc-type md \
  --kmesh 2 \
  --cpus 64 \
  --scheduler slurm
```

vasp_auto writes `jobs/Si_md/Si_md/submit.sh` and calls `sbatch`. The Excel
row records the queue job ID and status `submitted`. When the job finishes:

```bash
vasp-auto inputs/Si_md --parse-only
```

To poll the queue for job status while it runs:

```bash
vasp-auto --poll <JOBID> --scheduler slurm
```

`--poll` queries `squeue -j <JOBID>` and prints the scheduler state
(`RUNNING`, `PENDING`, `COMPLETED`, etc.). It exits after printing; no
calculation is started.

---

## Quick reference

```bash
# 300 K NVT AIMD, 1000 steps, Gamma-point only:
vasp-auto inputs/Si_md --calc-type md --kmesh 1 --cpus 16

# 2x2x1 k-mesh, 16 cores:
vasp-auto inputs/Si_md --calc-type md --kpoints-mode gamma --kmesh 2 --cpus 16

# Run in background, follow log:
vasp-auto inputs/Si_md --kmesh 2 --cpus 16 --background
tail -f vasp_auto_background_logs/vasp_auto_*.log

# Parse XDATCAR from Python:
# from vasp_auto.trajectory import parse_xdatcar
# frames = parse_xdatcar("jobs/Si_md/Si_md/XDATCAR")

# Submit to SLURM:
vasp-auto inputs/Si_md --calc-type md --kmesh 2 --cpus 64 --scheduler slurm

# Poll queue status:
vasp-auto --poll 123456 --scheduler slurm

# Parse results after job completes:
vasp-auto inputs/Si_md --parse-only
```

---

This is the last numbered tutorial. For catalysis workflows (adsorption
energies, CHE diagrams, Bader charges, d-band centers) see
`docs/TUTORIAL_CATALYSIS.md`. For combining two materials with different unit
cells into a heterostructure see `docs/TUTORIAL_HETEROSTRUCTURE.md`. For
the interactive web UI walkthrough see `docs/TUTORIAL_CEO2_GRAPHENE_CO2.md`.

A full learning path is listed in `docs/TUTORIALS_INDEX.md`.
