# Tutorial 01 — Total Energy of a Bulk Crystal (SCF)

This tutorial computes the total electronic energy of **bulk silicon** (diamond
cubic, two atoms per cell) starting from nothing but a POSCAR. You will learn
how to dry-run the input set, choose a k-point mesh, launch a single-point
calculation, and read the results out of the Excel summary. The workflow takes
about five minutes of compute on a desktop workstation; the same procedure
applies word for word to any other material (substitute the POSCAR and element
name throughout).

---

## Prerequisites

- `vasp-auto` installed and on PATH (`pip install -e .` from the repo root).
- `config.yaml` at the repo root with at least:

  ```yaml
  vasp_executable: /path/to/vasp_std
  jobs_root: jobs
  potcar_root: POTCAR           # directory containing POTCAR/Si/POTCAR
  ```

- A Si POTCAR in `POTCAR/Si/POTCAR` (PAW-PBE recommended).
- `mpirun` on PATH (or adjust `vasp_executable` to an MPI-aware launcher).

---

## Step 1 — Create the case directory with a POSCAR

Create a directory `inputs/Si_bulk` and paste the following POSCAR into it.
This is the conventional cubic cell of silicon (a = 5.431 Å, space group
Fd-3m, Wyckoff 8a sites).

```
Si bulk (diamond cubic, a = 5.431 A)
   1.0
     5.431000000   0.000000000   0.000000000
     0.000000000   5.431000000   0.000000000
     0.000000000   0.000000000   5.431000000
   Si
     8
Direct
  0.000000000  0.000000000  0.000000000
  0.500000000  0.500000000  0.000000000
  0.500000000  0.000000000  0.500000000
  0.000000000  0.500000000  0.500000000
  0.250000000  0.250000000  0.250000000
  0.750000000  0.750000000  0.250000000
  0.750000000  0.250000000  0.750000000
  0.250000000  0.750000000  0.750000000
```

Alternatively you can build it with ASE (requires `pip install ase`):

```bash
vasp-auto --ase-build-bulk Si --ase-crystalstructure diamond --ase-cubic \
          --ase-output inputs/Si_bulk --build-only
```

---

## Step 2 — Preview the full input set (dry run)

Before running VASP it is good practice to inspect every generated file.
`--dry-run` prints the INCAR, KPOINTS, and POTCAR composition to the terminal
without writing or running anything:

```bash
vasp-auto inputs/Si_bulk \
  --calc-type scf \
  --kpoints-mode gamma --kmesh 6 \
  --dry-run
```

What vasp_auto does:

- Reads `example/INCAR_scf` (ENCUT=520, EDIFF=1e-6, ISMEAR=0, LCHARG=.TRUE.).
- Builds a 6×6×6 Gamma-centred k-point mesh.
- Concatenates `POTCAR/Si/POTCAR` into the POTCAR preview.
- Prints `--- dry run: Si_bulk (scf) ---` followed by each section.

Check that the k-mesh makes sense (silicon needs at least 6×6×6 for well-
converged energies) and that ENCUT is above the Si POTCAR ENMAX (~245 eV;
520 eV is more than sufficient).

---

## Step 3 — Run the SCF calculation

```bash
vasp-auto inputs/Si_bulk \
  --calc-type scf \
  --kpoints-mode gamma --kmesh 6 \
  --cpus 8
```

Key flags:

| Flag | Effect |
|------|--------|
| `--calc-type scf` | Selects `example/INCAR_scf` (NSW=0, IBRION=-1). |
| `--kpoints-mode gamma` | Gamma-centred Monkhorst–Pack mesh. |
| `--kmesh 6` | 6×6×6 mesh (scalar; use `6x6x1` for slabs). |
| `--cpus 8` | Passed as `-np 8` to `mpirun`. |

vasp_auto will:

1. Create `jobs/Si_bulk/Si_bulk/` and write INCAR, KPOINTS, POSCAR, POTCAR.
2. Run `mpirun -np 8 /path/to/vasp_std` inside that directory, streaming
   `run.log` to the terminal.
3. Parse `OUTCAR` + `vasprun.xml` for energy, Fermi level, band gap, and max
   force.
4. Write `jobs/Si_bulk/Si_bulk.xlsx` with one row per calculation.

Typical wall time: 1–2 minutes on 8 cores.

---

## Step 4 — Read the results

Open `jobs/Si_bulk/Si_bulk.xlsx`. The row for `Si_bulk` will contain:

| Column | Typical value |
|--------|--------------|
| `energy_eV` | −43.xx eV (8-atom cell) |
| `energy_per_atom_eV` | −5.4x eV/atom |
| `band_gap_eV` | ~0.6 eV (GGA underestimate; use HSE06 for the true gap) |
| `fermi_eV` | midgap reference |
| `converged` | TRUE (green) |
| `max_force_eVA` | near zero (no relaxation) |

The raw OUTCAR and `vasprun.xml` live in `jobs/Si_bulk/Si_bulk/`.

---

## Step 5 — Inspect the CHGCAR (optional)

Because `LCHARG = .TRUE.` is set in `INCAR_scf`, a `CHGCAR` file is written.
You can visualise the electron density in VESTA or pass it to a DOS run:

```bash
# Non-self-consistent DOS on a denser mesh (reuses the charge density):
vasp-auto inputs/Si_bulk \
  --calc-type dos \
  --kpoints-mode gamma --kmesh 10 \
  --cpus 8
```

See Tutorial 04 for the full SCF → DOS → bands workflow.

---

## Quick reference

```bash
# Dry run — inspect inputs without running VASP:
vasp-auto inputs/Si_bulk --calc-type scf --kmesh 6 --dry-run

# SCF on 8 cores, Gamma-centred 6x6x6 mesh:
vasp-auto inputs/Si_bulk --calc-type scf --kpoints-mode gamma --kmesh 6 --cpus 8

# Parse an already-finished job (regenerate Excel without re-running VASP):
vasp-auto inputs/Si_bulk --parse-only

# MP-style mesh from k-point spacing (0.25 1/Å → ~6x6x6 for Si):
vasp-auto inputs/Si_bulk --calc-type scf --kspacing 0.25 --cpus 8
```

---

**Next**: Tutorial 02 covers geometry optimisation (relaxing Si cell + ions)
and reading the animation trajectory.
