# Tutorial 04 — DOS, Band Structure, and Band Gap

This tutorial computes the electronic **density of states (DOS)** and **band
structure** of bulk silicon using a chained `relax → scf → dos` workflow,
then runs a separate bands calculation along the high-symmetry FCC Brillouin-
zone path. You will export the DOS and band structure to CSV files for
plotting, and read off the band gap from the Excel summary.

---

## Prerequisites

- `inputs/Si_bulk/POSCAR` (Tutorial 01 or the builder).
- `config.yaml` pointing to a VASP binary and POTCAR library.
- `pandas` and `openpyxl` installed (come with `pip install -e .`).

---

## Step 1 — Relax → SCF → DOS in one command

```bash
vasp-auto inputs/Si_bulk \
  --workflow "relax,scf,dos" \
  --kpoints-mode gamma --kmesh 8 \
  --cpus 8
```

vasp_auto creates three sequential subdirectories under `jobs/Si_bulk/Si_bulk/`:

```
jobs/Si_bulk/Si_bulk/
  01_relax/    INCAR (IBRION=2, NSW=100)  → optimises geometry
  02_scf/      INCAR (NSW=0, LCHARG=.TRUE.)  → writes CHGCAR
  03_dos/      INCAR (ICHARG=11, NEDOS=2000, LORBIT=11)  → DOS
```

File handoffs (managed automatically by `chain.py`):

- `01_relax/CONTCAR` → `02_scf/POSCAR`
- `02_scf/CONTCAR` → `03_dos/POSCAR`
- `02_scf/CHGCAR` → `03_dos/CHGCAR`  (non-SCF ICHARG=11 run)

The Excel row for each step is appended to `jobs/Si_bulk/Si_bulk.xlsx`.
After the DOS step the `band_gap_eV` column is populated (GGA value ~ 0.6 eV
for Si; the true experimental gap is 1.1 eV — use HSE06 for accuracy).

---

## Step 2 — Export the DOS to CSV

```bash
vasp-auto --dos-export jobs/Si_bulk/Si_bulk/03_dos
```

This writes two files:

- `jobs/Si_bulk/Si_bulk/03_dos/dos.csv` — columns: `energy_eV`, `total_up`
  (and `total_down` for spin-polarised runs).
- `jobs/Si_bulk/Si_bulk/03_dos/pdos.csv` — per-element s/p/d projections,
  one column per orbital-element combination, labelled `Si_s`, `Si_p`, `Si_d`,
  etc. Requires `LORBIT = 11` in the INCAR (set in the DOS template).

The Fermi level is printed to the terminal (`E_Fermi: X.XXXX eV`); subtract
it from the energy column if you want E-E_F on the x-axis.

---

## Step 3 — Band structure along the FCC path

Silicon's conventional Brillouin zone follows the FCC high-symmetry path.
Run a non-SCF bands calculation reusing the charge density from the SCF step:

```bash
# Copy CHGCAR from the SCF step into the bands case:
mkdir -p inputs/Si_bands
cp inputs/Si_bulk/POSCAR inputs/Si_bands/POSCAR
cp jobs/Si_bulk/Si_bulk/02_scf/CHGCAR inputs/Si_bands/CHGCAR

vasp-auto inputs/Si_bands \
  --calc-type bands \
  --kpath fcc \
  --kpath-divisions 30 \
  --cpus 8
```

`--kpath fcc` selects the preset FCC high-symmetry k-path
(Γ → X → U|K → Γ → L → W → X). Use `--kpath auto` if vasp_auto should
detect the lattice type from the cell vectors and pick the path automatically.
`--kpath-divisions 30` places 30 k-points between each pair of high-symmetry
points.

Alternatively, supply an explicit path:

```bash
vasp-auto inputs/Si_bands \
  --calc-type bands \
  --kpath "G 0 0 0; X 0.5 0 0.5; U 0.625 0.25 0.625; K 0.375 0.375 0.75; G 0 0 0; L 0.5 0.5 0.5" \
  --kpath-divisions 25 \
  --cpus 8
```

---

## Step 4 — Export the band structure to CSV

```bash
vasp-auto --bands-export jobs/Si_bands/Si_bands
```

Output: `jobs/Si_bands/Si_bands/bands.csv` with columns:

- `distance_invA` — cumulative k-path distance in 1/Å.
- `kx`, `ky`, `kz` — fractional coordinates of each k-point.
- `label` — high-symmetry label (e.g. `G`, `X`, `L`) where applicable.
- `band1`, `band2`, … — eigenvalue in eV relative to the VASP zero (subtract
  E_Fermi from the terminal output to get E-E_F).

---

## Step 5 — Alternative: full relax → scf → dos chain with DOS-mesh override

For an accurate DOS you often want a denser k-mesh in the DOS step than in
the SCF step. Use a per-step `workflow.yaml` in the case directory:

```yaml
# inputs/Si_bulk/workflow.yaml
steps:
  - calc_type: relax
    kpoints: "6"
  - calc_type: scf
    kpoints: "8"
    incar:
      LCHARG: ".TRUE."
  - calc_type: dos
    kpoints: "12"
    incar:
      NEDOS: "4000"
      ISMEAR: "-5"
```

Then run:

```bash
vasp-auto inputs/Si_bulk --cpus 8
```

vasp_auto reads the `workflow.yaml` automatically and applies the per-step
INCAR and KPOINTS overrides.

---

## Quick reference

```bash
# One-shot relax → scf → dos chain:
vasp-auto inputs/Si_bulk --workflow "relax,scf,dos" --kmesh 8 --cpus 8

# Export DOS (writes dos.csv + pdos.csv):
vasp-auto --dos-export jobs/Si_bulk/Si_bulk/03_dos

# Bands along FCC path (30 points per segment):
vasp-auto inputs/Si_bands --calc-type bands --kpath fcc --kpath-divisions 30 --cpus 8

# Auto-detect lattice type and pick k-path:
vasp-auto inputs/Si_bands --calc-type bands --kpath auto --cpus 8

# Export bands to CSV:
vasp-auto --bands-export jobs/Si_bands/Si_bands
```

---

**Next**: Tutorial 05 demonstrates a spin-polarised calculation on bcc Fe to
obtain the magnetic moment per atom.
