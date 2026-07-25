# VASP INCAR Examples

This directory contains example INCAR templates for common VASP workflows:

- `INCAR_optimize_structure`: relax atomic positions and, by default, the cell.
- `INCAR_scf`: converged static single-point calculation.
- `INCAR_charge_density`: static run that writes `CHGCAR` and optional Bader files.
- `INCAR_dos`: non-self-consistent DOS run using a prior `CHGCAR`.

Typical order:

1. Run `INCAR_optimize_structure`.
2. Copy `CONTCAR` to `POSCAR`.
3. Run `INCAR_scf` with a suitable KPOINTS mesh.
4. For charge analysis, run `INCAR_charge_density`.
5. For DOS, copy the SCF `CHGCAR`, use a denser KPOINTS mesh, and run `INCAR_dos`.

Adjust `ENCUT`, `ISMEAR`, `SIGMA`, `ISIF`, magnetic settings, DFT+U settings, and KPOINTS for the material being studied.
