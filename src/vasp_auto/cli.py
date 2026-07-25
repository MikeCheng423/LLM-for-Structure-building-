import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from vasp_auto.ase_tools import (
    build_bulk_case,
    build_crystal_case,
    build_molecule_case,
    build_nanotube_case,
    build_slab_case,
    import_structure_to_case,
)
from vasp_auto.calc_types import CalcType, parse_calc_type
from vasp_auto.chain import load_workflow_spec, run_workflow_case
from vasp_auto.config_loader import load_config, merge_local_config
from vasp_auto.convergence import (
    converge_scf_case,
    parse_encut_values,
    parse_kpoint_meshes,
    parse_nelm_values,
    parse_sigma_values,
)
from vasp_auto.incar import parse_magmom_map
from vasp_auto.job_manager import (
    create_job_from_case,
    make_case_info,
    preview_job_from_case,
)
from vasp_auto.kpoints import parse_mesh
from vasp_auto.report import write_job_report
from vasp_auto.structure import (
    add_adsorbate,
    add_interstitial,
    combine_structures,
    cell_lengths,
    delete_atoms,
    freeze_atoms,
    make_prototype,
    make_supercell,
    make_vacancy,
    match_supercells,
    move_atom,
    parse_atom_selection,
    read_poscar,
    resolve_prototype,
    scale_cell,
    substitute,
    write_poscar,
)
from vasp_auto.target_utils import filter_case_dirs, inspect_target
from vasp_auto.runner import resolve_remote_run_mode
from vasp_auto.workflow import (
    build_row,
    parse_existing_job,
    run_one_case,
    should_retry_failed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one VASP calculation or a project folder. SCF cases only need "
            "POSCAR; TSS/NEB cases need initial/POSCAR and final/POSCAR."
        )
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help='Target path, e.g. "." or "inputs/Al" or "TSS/cases/A"',
    )
    parser.add_argument(
        "--background",
        "--bg",
        action="store_true",
        help="Start vasp_auto in the background and write command output to vasp_auto_background_logs/.",
    )
    parser.add_argument(
        "-n",
        "--cpus",
        type=int,
        default=None,
        help="Number of CPUs/threads passed to VASP runner.",
    )
    parser.add_argument(
        "--calc-type",
        choices=[t.value for t in CalcType],
        default=None,
        help="Calculation type; selects the INCAR template from example/INCAR_<type>.",
    )
    parser.add_argument(
        "--engine",
        choices=["vasp", "qe", "ase"],
        default=None,
        help="DFT engine: vasp (default), qe (open-source Quantum ESPRESSO, pw.x), "
        "or ase (any ASE calculator, set via ase_calculator/ase_calc_params in "
        "config.yaml). Overrides 'engine:' in config.yaml.",
    )
    parser.add_argument(
        "--qe-executable",
        default=None,
        metavar="PATH",
        help="Path to the Quantum ESPRESSO pw.x binary (default: config qe_executable or 'pw.x').",
    )
    parser.add_argument(
        "--pseudo-dir",
        default=None,
        metavar="DIR",
        help="Directory of UPF pseudopotentials for the QE engine "
        "(overrides config pseudo_dir).",
    )
    parser.add_argument(
        "--ase-calculator",
        default=None,
        metavar="NAME",
        help="ASE calculator for the ase engine, e.g. emt, espresso, gpaw, mace "
        "(overrides config ase_calculator).",
    )
    parser.add_argument(
        "--ase-command",
        default=None,
        metavar="PATH",
        help="Run command / binary path for the ASE calculator (overrides "
        "ase_calc_params.command), e.g. 'pw.x' or 'mpirun -np 4 pw.x'.",
    )
    parser.add_argument(
        "--ase-params",
        default=None,
        metavar="JSON",
        help="Extra ASE calculator keyword args as a JSON object, merged into "
        'ase_calc_params, e.g. \'{"xc":"PBE","kpts":[4,4,4]}\'. Use the special '
        'keys "__module__"/"__class__" to reach a calculator not in the menu.',
    )
    parser.add_argument(
        "--ase-fmax",
        default=None,
        type=float,
        metavar="EV_PER_A",
        help="Force tolerance (eV/A) for the ase engine's relax (overrides ase_fmax).",
    )
    parser.add_argument(
        "--ase-steps",
        default=None,
        type=int,
        metavar="N",
        help="Max optimizer steps for the ase engine's relax (overrides ase_steps).",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create config.yaml from config.yaml.example (if missing) and exit. "
             "Run this once after installing, then edit the paths inside it.",
    )
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List the human-language INCAR settings and their choices, then exit.",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare jobs only, do not run VASP.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the full input set (INCAR/KPOINTS/POTCAR) without writing job files.",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Parse existing jobs only and regenerate Excel.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write a Markdown report.md (setup + results) into every job directory.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only rerun failed cases.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the latest unfinished job in place from its newest CONTCAR, "
             "reusing that job directory's own INCAR/KPOINTS/POTCAR (no new job number).",
    )
    parser.add_argument(
        "--resume-job-dir",
        default=None,
        metavar="DIR",
        help="Resume one explicit job directory in place from its newest CONTCAR "
             "(reusing its INCAR/KPOINTS/POTCAR). With --remote/--remote-config the "
             "directory is a path on that machine and the restart runs there.",
    )
    parser.add_argument(
        "--resume-local-mirror",
        default=None,
        metavar="DIR",
        help="With --resume-job-dir on a remote machine, also copy the results back "
             "into this local directory so local viewers work.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Only run selected case names, e.g. --cases A1 A3",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Run up to N cases concurrently (each case gets its own mpirun).",
    )
    parser.add_argument(
        "--scheduler",
        choices=["local", "slurm", "pbs"],
        default=None,
        help="Override the scheduler from config.yaml (local | slurm | pbs).",
    )
    parser.add_argument(
        "--remote",
        nargs="?",
        const="__default__",
        default=None,
        metavar="NAME",
        help="Send all input files to a remote machine over SSH and submit to its "
        "queue, then exit — so the local host can be turned off. Bare --remote uses "
        "the config.yaml 'remote:' block; --remote NAME uses remotes[NAME]. Each needs "
        "host, remote_root and vasp_executable.",
    )
    parser.add_argument(
        "--remote-config",
        default=None,
        metavar="FILE",
        help="Path to a JSON/YAML file holding one remote-machine config "
        "(host, remote_root, vasp_executable, …). Used by the UI to submit to a "
        "machine chosen in the Remote tab. Overrides --remote.",
    )
    parser.add_argument(
        "--remote-setup",
        action="store_true",
        help="Pre-install the vasp_auto engine into a venv on the remote machine "
        "(selected with --remote/--remote-config), then exit. Optional: an offload "
        "run (run_mode: ssh_detached) auto-installs the engine on first use; use this "
        "only to provision a machine ahead of time.",
    )
    parser.add_argument(
        "--workflow",
        default=None,
        help='Chained workflow, e.g. --workflow "relax,scf,dos". '
        "A workflow.yaml in the case directory or a workflow: key in config.yaml also works.",
    )
    parser.add_argument(
        "--neb-images",
        type=int,
        default=None,
        help="Number of intermediate images for TSS/NEB preparation. Default: config neb_images or 5.",
    )
    parser.add_argument(
        "--spin",
        action="store_true",
        help="Spin-polarised run: sets ISPIN=2 and derives a MAGMOM line from the POSCAR.",
    )
    parser.add_argument(
        "--magmom",
        default=None,
        metavar="El:moment,...",
        help='Initial magnetic moments for --spin, e.g. "Fe:5.0,O:0.6". '
        "Overrides the magmom_map from config.yaml.",
    )
    parser.add_argument(
        "--auto-retry",
        type=int,
        default=0,
        metavar="N",
        help="Retry a failed run up to N times, applying the known INCAR fix for "
        "each detected VASP error (local scheduler only). Default: 0 (off).",
    )
    # KPOINTS generation
    parser.add_argument(
        "--kpoints-mode",
        choices=["gamma", "mp", "line", "spacing"],
        default=None,
        help="KPOINTS generation mode: gamma/mp uniform mesh, line (band path), spacing (density).",
    )
    parser.add_argument(
        "--kmesh",
        default=None,
        help='Uniform k-mesh, e.g. "6" or "6x6x1" (with --kpoints-mode gamma or mp).',
    )
    parser.add_argument(
        "--kspacing",
        type=float,
        default=None,
        help="Maximum k-point spacing in 1/Angstrom; mesh is derived from the lattice.",
    )
    parser.add_argument(
        "--kpath",
        default=None,
        help='Line-mode k-path: a preset (cubic, fcc, bcc, hex) or "G 0 0 0; X 0.5 0 0.5; ...".',
    )
    parser.add_argument(
        "--kpath-divisions",
        type=int,
        default=20,
        help="Points per k-path segment for line-mode KPOINTS. Default: 20.",
    )
    # Convergence scans
    parser.add_argument(
        "--converge-scf",
        action="store_true",
        help="Run automatic SCF convergence: scan NELM then KPOINTS and write a step report.",
    )
    parser.add_argument(
        "--converge-encut",
        default=None,
        metavar="VALUES",
        help="Scan ENCUT first, e.g. --converge-encut 400,450,500,550. "
        "Combine with --converge-scf for a full ENCUT+NELM+KPOINTS scan.",
    )
    parser.add_argument(
        "--nelm-values",
        default=None,
        help="Comma-separated NELM values for --converge-scf, e.g. 40,60,80,100,120.",
    )
    parser.add_argument(
        "--kpoints-values",
        default=None,
        help='Comma-separated KPOINTS meshes for --converge-scf, e.g. "3,4,5,6" or "3x3x1,5x5x1".',
    )
    parser.add_argument(
        "--converge-sigma",
        default=None,
        metavar="VALUES",
        help="Scan smearing widths, e.g. --converge-sigma 0.2,0.1,0.05. Selects the "
        "largest SIGMA whose entropy term T*S stays below --sigma-tol per atom.",
    )
    parser.add_argument(
        "--energy-tol",
        type=float,
        default=1e-4,
        help="Energy change tolerance in eV for stopping convergence scans. Default: 1e-4.",
    )
    parser.add_argument(
        "--sigma-tol",
        type=float,
        default=1e-3,
        help="Entropy (T*S) tolerance in eV/atom for the SIGMA scan. Default: 1e-3.",
    )
    parser.add_argument(
        "--reuse-wavecar",
        action="store_true",
        help="Seed each convergence trial with WAVECAR/CHGCAR from the previous trial "
        "to cut scan wall time.",
    )
    # Structure builders (pure Python)
    parser.add_argument(
        "--supercell",
        default=None,
        metavar="NxNxN",
        help="Expand the target case POSCAR into a supercell, e.g. --supercell 2x2x2.",
    )
    parser.add_argument(
        "--vacancy",
        type=int,
        default=None,
        metavar="INDEX",
        help="Remove atom INDEX (1-based, POSCAR order) from the target case POSCAR.",
    )
    parser.add_argument(
        "--substitute",
        default=None,
        metavar="INDEX=El",
        help="Replace atom INDEX with element El in the target case POSCAR, e.g. 12=Mg.",
    )
    parser.add_argument(
        "--interstitial",
        "--add-atom",
        dest="interstitial",
        default=None,
        metavar="El@x,y,z",
        help="Add an atom at the given position (POSCAR coordinate mode), "
        'e.g. --add-atom "H@0.5,0.5,0.5". --interstitial is the same flag.',
    )
    parser.add_argument(
        "--delete",
        default=None,
        metavar="SEL",
        help='Delete a selection of atoms: "37", "1-36", "1,5,9", or "z>0.6" '
        "(fractional height) — e.g. to carve fragments for --chg-diff.",
    )
    parser.add_argument(
        "--adsorbate",
        default=None,
        metavar="El@N+h",
        help="Place an adsorbate El (an atom, or a molecule ASE knows: CO2, H2O, "
        'NH3 …) directly above atom N at height h Å, e.g. --adsorbate "CO2@12+2.0" '
        "(h defaults to 2.0).",
    )
    parser.add_argument(
        "--move-atom",
        default=None,
        metavar="N@x,y,z|N+dx,dy,dz",
        help='Move atom N: "5@0.5,0.5,0.6" places it, "5+0,0,0.05" translates it '
        "(POSCAR coordinate mode).",
    )
    parser.add_argument(
        "--scale-cell",
        default=None,
        metavar="SPEC",
        help='Resize the cell, atoms follow fractionally: "1.02" (uniform), '
        '"1.02,1.02,1.0" (per axis), or absolute lengths "a=4.1,c=22.5" in Å.',
    )
    parser.add_argument(
        "--freeze",
        default=None,
        metavar="SEL[:AXES]",
        help='Freeze atoms (Selective dynamics, F = fixed): "1-8", "1,2,5", or '
        '"z<0.3" (bottom slab layers by fractional height). Append :AXES to '
        'freeze only some directions, e.g. "z<0.3:XY".',
    )
    parser.add_argument(
        "--combine",
        default=None,
        metavar="PATH",
        help="Combine the target structure with another one (case dir or POSCAR "
        "file), e.g. deposit an Au crystal on a graphite sheet: point the "
        "target at the sheet and --combine at the Au case.",
    )
    parser.add_argument(
        "--combine-mode",
        choices=("stack", "insert"),
        default="stack",
        help="stack = place the guest above the target surface and extend c; "
        "insert = drop the guest atoms into the unchanged target cell. Default: stack.",
    )
    parser.add_argument(
        "--combine-gap",
        type=float,
        default=2.0,
        metavar="Å",
        help="Gap between the target's top atom and the guest's bottom atom (stack), "
        "or a z offset in Å (insert). Default: 2.0.",
    )
    parser.add_argument(
        "--combine-vacuum",
        type=float,
        default=10.0,
        metavar="Å",
        help="Vacuum left above the guest after stacking. Default: 10.0.",
    )
    parser.add_argument(
        "--combine-shift",
        default=None,
        metavar="x,y",
        help="Lateral shift of the guest in fractions of the target a/b vectors "
        '(stack) or Å (insert), e.g. "0.25,0.25".',
    )
    parser.add_argument(
        "--combine-strain",
        action="store_true",
        help="Strain the guest laterally onto the target lattice (epitaxial match) "
        "instead of keeping its own geometry centred over the cell.",
    )
    parser.add_argument(
        "--build-prototype",
        default=None,
        metavar="NAME[:opts]",
        help="Build a prototype crystal. "
        "Built-in library (no install): graphene, graphite, rutile-TiO2, anatase-TiO2, hBN "
        "with optional a/c/vacuum overrides — e.g. \"graphene:a=2.46,vacuum=18\". "
        "Materials Project library (mp-api): prefix with 'mp:' followed by a "
        "material_id or formula — e.g. \"mp:mp-2657\" or \"mp:SnO2\". "
        "Combine with --prototype-substitute to do isostructural element replacement "
        "after fetching — e.g. --prototype-substitute Ti=Sn to build rutile SnO2 "
        "from the rutile TiO2 prototype.",
    )
    parser.add_argument(
        "--prototype-substitute",
        default=None,
        metavar="El1=El2[,El3=El4]",
        help="Element substitution applied to the fetched prototype structure. "
        "Replaces every atom of element El1 with El2 (and so on for multiple pairs). "
        "Example: --prototype-substitute Ti=Sn  or  --prototype-substitute Ti=Ge,O=S. "
        "Works with both built-in and mp: prototypes.",
    )
    parser.add_argument(
        "--match-cells",
        default=None,
        metavar="PATH",
        help="Read-only: suggest in-plane supercell pairs that match the target "
        "structure (host) with another case/POSCAR (guest) for stacking — use "
        "before --combine when the two unit cells differ, e.g. TiO2 on graphene.",
    )
    parser.add_argument(
        "--match-max",
        type=int,
        default=6,
        metavar="N",
        help="Largest supercell repeat tried by --match-cells along a and b. Default: 6.",
    )
    parser.add_argument(
        "--match-strain",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="Largest acceptable lattice strain for --match-cells (fraction). Default: 0.1.",
    )
    parser.add_argument(
        "--match-gamma-tol",
        type=float,
        default=8.0,
        metavar="DEG",
        help="Largest acceptable in-plane angle mismatch for --match-cells in degrees "
        "(straining shears the guest by this much). Default: 8.",
    )
    # Structure builders (ASE-backed)
    parser.add_argument(
        "--ase-import",
        default=None,
        help="Read any ASE-supported structure file and write it as a VASP POSCAR case.",
    )
    parser.add_argument(
        "--ase-format",
        default=None,
        help="Optional ASE input format for --ase-import, e.g. cif, xyz, vasp.",
    )
    parser.add_argument(
        "--ase-index",
        default=None,
        help="Optional ASE frame index for --ase-import. Default: last frame.",
    )
    parser.add_argument(
        "--ase-output",
        default=None,
        help="Output case directory for structure builders.",
    )
    parser.add_argument(
        "--ase-build-bulk",
        default=None,
        metavar="ELEMENT",
        help="Build a bulk crystal case with ASE, e.g. --ase-build-bulk Al --ase-crystalstructure fcc.",
    )
    parser.add_argument(
        "--ase-crystalstructure",
        default=None,
        help="ASE bulk crystal structure, e.g. fcc, bcc, hcp, diamond, rocksalt.",
    )
    parser.add_argument(
        "--ase-a",
        type=float,
        default=None,
        help="ASE bulk lattice constant a in Angstrom.",
    )
    parser.add_argument(
        "--ase-c",
        type=float,
        default=None,
        help="ASE bulk lattice constant c in Angstrom.",
    )
    parser.add_argument(
        "--ase-cubic",
        action="store_true",
        help="Ask ASE to build a cubic conventional bulk cell where supported.",
    )
    parser.add_argument(
        "--ase-build-slab",
        default=None,
        metavar="ELEMENT_OR_FILE",
        help="Build a surface slab from an element symbol or structure file, e.g. --ase-build-slab Al.",
    )
    parser.add_argument(
        "--ase-miller",
        default="1,1,1",
        help='Miller indices for --ase-build-slab, e.g. "1,1,1" or "1 0 0".',
    )
    parser.add_argument(
        "--ase-layers",
        type=int,
        default=4,
        help="Number of atomic layers for --ase-build-slab. Default: 4.",
    )
    parser.add_argument(
        "--ase-vacuum",
        type=float,
        default=12.0,
        help="Vacuum thickness in Angstrom for --ase-build-slab. Default: 12.",
    )
    parser.add_argument(
        "--ase-repeat",
        default=None,
        metavar="NxM",
        help='In-plane repetition for --ase-build-slab, e.g. "3x3".',
    )
    parser.add_argument(
        "--ase-build-molecule",
        default=None,
        metavar="NAME",
        help="Build an isolated molecule in a box, e.g. --ase-build-molecule H2O.",
    )
    parser.add_argument(
        "--ase-box",
        type=float,
        default=12.0,
        help="Cubic box edge in Angstrom for --ase-build-molecule. Default: 12.",
    )
    parser.add_argument(
        "--ase-build-crystal",
        default=None,
        metavar="SYMBOLS",
        help='Build a crystal from a space group + Wyckoff basis, e.g. '
        '--ase-build-crystal "Na Cl" --ase-spacegroup 225 '
        '--ase-basis "0,0,0;0.5,0.5,0.5" --ase-a 5.64.',
    )
    parser.add_argument(
        "--ase-spacegroup",
        type=int,
        default=None,
        help="International space-group number (1-230) for --ase-build-crystal.",
    )
    parser.add_argument(
        "--ase-basis",
        default=None,
        help='Wyckoff basis for --ase-build-crystal: one "x,y,z" per symbol, '
        'separated by ";", e.g. "0,0,0;0.5,0.5,0.5".',
    )
    parser.add_argument(
        "--ase-b",
        type=float,
        default=None,
        help="Lattice constant b in Angstrom for --ase-build-crystal (default a).",
    )
    parser.add_argument(
        "--ase-alpha",
        type=float,
        default=90.0,
        help="Cell angle alpha in degrees for --ase-build-crystal. Default: 90.",
    )
    parser.add_argument(
        "--ase-beta",
        type=float,
        default=90.0,
        help="Cell angle beta in degrees for --ase-build-crystal. Default: 90.",
    )
    parser.add_argument(
        "--ase-gamma",
        type=float,
        default=90.0,
        help="Cell angle gamma in degrees for --ase-build-crystal. Default: 90.",
    )
    parser.add_argument(
        "--ase-build-nanotube",
        default=None,
        metavar="ELEMENT",
        help='Build an (n,m) single-wall nanotube, e.g. '
        '--ase-build-nanotube C --ase-nt-n 5 --ase-nt-m 5 --ase-nt-length 3.',
    )
    parser.add_argument(
        "--ase-nt-n",
        type=int,
        default=5,
        help="Chiral index n for --ase-build-nanotube. Default: 5.",
    )
    parser.add_argument(
        "--ase-nt-m",
        type=int,
        default=5,
        help="Chiral index m for --ase-build-nanotube. Default: 5.",
    )
    parser.add_argument(
        "--ase-nt-length",
        type=int,
        default=1,
        help="Number of unit cells along the nanotube axis. Default: 1.",
    )
    parser.add_argument(
        "--ase-nt-bond",
        type=float,
        default=None,
        help="Bond length in Angstrom for --ase-build-nanotube (default ASE value).",
    )
    parser.add_argument(
        "--build-only",
        "--ase-only",
        dest="build_only",
        action="store_true",
        help="Only create the generated structure case; do not prepare or run VASP.",
    )
    # Machine-learning pre-relaxation (Meta OMat24 / UMA via fairchem)
    parser.add_argument(
        "--ml-energy",
        default=None,
        metavar="TARGET",
        help="Single-point MLIP energy screen (read-only): compute energy + max|F| "
        "for the POSCAR in TARGET (dir or file) and print results without writing any files. "
        'Honors --ml-model/--ml-task/--ml-checkpoint. Example: --ml-energy inputs/Al --ml-model emt',
    )
    parser.add_argument(
        "--ml-relax",
        action="store_true",
        help="Pre-relax the case POSCAR with a machine-learning potential "
        "(Meta OMat24/UMA via fairchem) before VASP; writes a derived case.",
    )
    parser.add_argument(
        "--ml-only",
        action="store_true",
        help="Stop after the ML pre-relaxation; do not prepare or run VASP.",
    )
    parser.add_argument(
        "--ml-model",
        default=None,
        help='MLIP model name. Default: config ml_model or "uma-s-1p1" '
        '(OMat24-trained UMA). "emt" = ASE demo potential, no install needed.',
    )
    parser.add_argument(
        "--ml-task",
        default=None,
        help='UMA task head: omat (inorganic materials, default), oc20 '
        "(catalysis/adsorbates), omol (molecules), odac (MOFs), omc (crystals).",
    )
    parser.add_argument(
        "--ml-checkpoint",
        default=None,
        help="Path to a downloaded fairchem 1.x checkpoint (e.g. eqV2 OMat24) "
        "instead of a named model.",
    )
    parser.add_argument(
        "--ml-fmax",
        type=float,
        default=0.05,
        help="ML relaxation force convergence in eV/Å. Default: 0.05.",
    )
    parser.add_argument(
        "--ml-steps",
        type=int,
        default=200,
        help="Maximum ML relaxation steps. Default: 200.",
    )
    parser.add_argument(
        "--ml-relax-cell",
        action="store_true",
        help="Also relax the cell during ML pre-relaxation (FrechetCellFilter).",
    )
    # Material database integration (fetch structure before ML / VASP)
    parser.add_argument(
        "--db-source",
        default=None,
        metavar="DB",
        help="External material database to fetch a starting structure from. "
        "Choices: mp (Materials Project), umat (META UMAT — access pending). "
        "Use together with --db-query. Example: --db-source mp --db-query mp-1234",
    )
    parser.add_argument(
        "--db-query",
        default=None,
        metavar="QUERY",
        help="Material ID or formula/chemsys to fetch from the database set by "
        "--db-source. Examples: mp-1234, Fe2O3, Fe-O. "
        "Combine with --ml-relax or --ml-energy to run an ML calculation on the "
        "fetched structure, or use alone with --build-only to just download the POSCAR.",
    )
    parser.add_argument(
        "--db-prerelax",
        action="store_true",
        help="Use the database structure (--db-source + --db-query) as the "
        "pre-relaxed starting geometry and continue directly into the VASP workflow. "
        "The fetched POSCAR replaces the local case POSCAR; no MLIP is run. "
        "Ideal for bulk materials already in Materials Project: the MP structure "
        "is DFT-relaxed, so VASP can go straight to SCF/DOS/bands. "
        "Example: vasp-auto inputs/Al --db-source mp --db-query mp-134 "
        "--db-prerelax --calc-type dos",
    )
    parser.add_argument(
        "--mp-api-key",
        default=None,
        metavar="KEY",
        help="Materials Project API key (overrides the MP_API_KEY environment variable).",
    )
    # Catalysis analysis (post-processing of finished jobs; runs no VASP)
    parser.add_argument(
        "--adsorption-energy",
        default=None,
        metavar="TOTAL,SLAB,MOL",
        help="Adsorption energy from three finished job dirs: "
        "E(slab+ads) - E(slab) - scale*E(molecule). "
        'Example: --adsorption-energy "jobs/PtH,jobs/Pt,jobs/H2" --molecule-scale 0.5',
    )
    parser.add_argument(
        "--molecule-scale",
        type=float,
        default=1.0,
        help="Fraction of the reference molecule energy for --adsorption-energy, "
        "e.g. 0.5 to reference H from an H2 box. Default: 1.0.",
    )
    parser.add_argument(
        "--thermo",
        default=None,
        metavar="DIR",
        help="Parse a finished freq job (--calc-type freq): vibrational modes, "
        "ZPE, T*S, and the Gibbs correction ZPE + U_vib - T*S.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.15,
        help="Temperature in K for --thermo. Default: 298.15.",
    )
    parser.add_argument(
        "--work-function",
        default=None,
        metavar="DIR",
        help="Work function V_vacuum - E_Fermi from a finished workfunction job "
        "(LVHAR LOCPOT); also writes potential_profile.csv.",
    )
    parser.add_argument(
        "--d-band",
        default=None,
        metavar="DIR:SEL",
        help="d-band center of selected atoms from a finished dos job (LORBIT=11), "
        'e.g. "jobs/Pt_dos:13-16" or "jobs/Pt_dos:z>0.5" (selection like --freeze).',
    )
    parser.add_argument(
        "--d-band-emax",
        type=float,
        default=None,
        metavar="eV",
        help="Upper integration limit relative to E_F for --d-band (e.g. 0 = occupied "
        "states only). Default: the full energy grid.",
    )
    parser.add_argument(
        "--chg-diff",
        default=None,
        metavar="TOTAL,PART1,PART2",
        help="Charge-density difference rho(TOTAL) - sum(rho(PART)); arguments are job "
        "dirs or CHGCAR paths on identical grids. Writes CHGCAR_diff next to TOTAL.",
    )
    parser.add_argument(
        "--bader",
        default=None,
        metavar="DIR",
        help="Bader charge analysis of a finished charge job (CHGCAR + AECCAR0/2): "
        "runs the Henkelman bader binary and writes bader_charges.csv.",
    )
    parser.add_argument(
        "--optics-parse",
        default=None,
        metavar="DIR",
        help="Absorption coefficient alpha(E) from a finished optics job (LOPTICS); "
        "writes absorption.csv.",
    )
    parser.add_argument(
        "--xrd",
        default=None,
        metavar="DIR",
        help="Simulated powder XRD pattern (Cu Ka) of the relaxed structure "
        "(CONTCAR, else POSCAR); writes xrd.csv.",
    )
    parser.add_argument(
        "--dos-export",
        default=None,
        metavar="DIR",
        help="Export the density of states of a finished dos job to dos.csv "
        "(and per-element s/p/d projections to pdos.csv when LORBIT was set).",
    )
    parser.add_argument(
        "--bands-export",
        default=None,
        metavar="DIR",
        help="Export the band structure of a finished bands job to bands.csv "
        "(k-path distance + one column per band; high-symmetry labels included).",
    )
    parser.add_argument(
        "--poll",
        default=None,
        metavar="JOBID",
        help="Query the scheduler (--scheduler) for the status of a submitted job ID "
        "and print the result. Exits after printing. Does not affect --parse-only.",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="Show which jobs are running now across this machine and every "
        "configured remote, then exit. Add --all to include finished/prepared jobs.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="With --list-jobs, list every job (done/running/prepared), not just running.",
    )
    parser.add_argument(
        "--kill",
        default=None,
        metavar="JOB_DIR",
        help="Stop a running detached (offload) job by its local job directory: "
        "reads that dir's .remote.json for the machine + PID and kills the remote "
        "process group over SSH. Local runs stop with Ctrl-C; scheduler jobs with "
        "scancel/qdel. Exits after.",
    )
    # Implicit solvation (VASPsol-patched binary required)
    parser.add_argument(
        "--solvation",
        action="store_true",
        help="Enable implicit solvation (VASPsol): injects LSOL=.TRUE. and EB_K into "
        "the INCAR. Requires a VASPsol-patched VASP binary. "
        "Default solvent: water (EB_K=78.4). See docs/MANUAL.md.",
    )
    parser.add_argument(
        "--solvation-eps",
        type=float,
        default=78.4,
        metavar="EPS",
        help="Dielectric constant for implicit solvation (--solvation). "
        "Default: 78.4 (water). Examples: 36.6 (acetonitrile), 24.9 (ethanol).",
    )
    parser.add_argument(
        "--ase-neb",
        action="store_true",
        help="Use ASE NEB interpolation when preparing TSS/NEB cases.",
    )
    parser.add_argument(
        "--ase-neb-method",
        default="idpp",
        choices=["idpp", "linear"],
        help="ASE NEB interpolation method. Default: idpp.",
    )
    return parser.parse_args()


def launch_background():
    args = [arg for arg in sys.argv[1:] if arg not in {"--background", "--bg"}]
    log_dir = Path.cwd() / "vasp_auto_background_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"vasp_auto_{timestamp}.log"

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    env.setdefault("VASP_AUTO_ROOT", str(repo_root))
    src_path = str(repo_root / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src_path

    command = [sys.executable, "-u", "-m", "vasp_auto.cli", *args]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    print(f"vasp_auto started in background")
    print(f"PID       : {process.pid}")
    print(f"Log       : {log_path}")
    print(f"Follow log: tail -f {log_path}")


def _write_reports(rows: list[dict]):
    extra_keys = ("selected_encut", "selected_sigma", "selected_nelm", "selected_kpoints")
    written = 0
    for row in rows:
        job_dir = Path(row.get("job_dir", ""))
        if not job_dir.is_dir():
            continue
        extra = {key: row[key] for key in extra_keys if row.get(key) is not None}
        write_job_report(job_dir, case_name=row.get("case"), extra=extra)
        written += 1
    if written:
        print(f"Reports   : report.md written to {written} job director{'y' if written == 1 else 'ies'}")


def summary_excel_path(job_root: Path, output_root: Path, project_name: str, mode: str, case_infos: list[dict]):
    if mode == "single" and case_infos:
        job_dir = Path(case_infos[0]["job_dir"])
        return job_dir / f"{case_infos[0]['case_name']}.xlsx"

    return output_root / f"{project_name}.xlsx"


def _write_excel_summary(excel_path, all_results) -> bool:
    """Write the Excel results summary, skipping cleanly on a lean offload engine.

    The remote offload engine installs only the core dep (PyYAML); pandas/openpyxl
    are intentionally absent (docs/INSTALL.md, "Why the engine is lean"). When they
    are missing there is nothing to write here — the control machine re-parses the
    fetched results into Excel locally — so we skip rather than crash the run after
    VASP has already finished. Returns True iff an Excel file was written.
    """
    try:
        from vasp_auto.excel_writer import write_results_to_excel  # lazy: pandas optional (not on remote engine)
    except ImportError:
        print(f"Skipped Excel summary ({excel_path.name}): pandas/openpyxl not installed (lean engine).")
        return False
    write_results_to_excel(str(excel_path), all_results)
    print(f"Wrote Excel: {excel_path}")
    return True


def _parse_int_triplet(text: str) -> tuple[int, int, int]:
    parts = [int(p) for p in str(text).replace(",", " ").replace("x", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"Expected three integers, got: {text!r}")
    return (parts[0], parts[1], parts[2])


def _builder_output_dir(args, default_name: str) -> Path:
    if args.ase_output:
        return Path(args.ase_output)
    if args.target != ".":
        return Path(args.target)
    return Path(default_name)


def _parse_prototype_spec(text: str) -> tuple[str, dict]:
    """Parse a --build-prototype spec.

    Built-in:  'graphene:a=2.46,vacuum=18' → ("graphene", {"a": 2.46, "vacuum": 18.0})
    MP prefix: 'mp:mp-2657'  → ("mp:mp-2657", {})
               'mp:SnO2'     → ("mp:SnO2", {})
    """
    if text.startswith("mp:"):
        # Everything after "mp:" is the query (material_id or formula).
        # No key=value params for MP prototypes — use --prototype-substitute instead.
        return text, {}

    name, _, params_text = text.partition(":")
    overrides = {}
    for item in params_text.split(","):
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        key = key.strip().lower()
        if key not in ("a", "c", "vacuum") or not value.strip():
            raise ValueError(
                f'Use --build-prototype "NAME:a=X,c=Y,vacuum=Z" or "mp:QUERY", '
                f"got: {text!r}"
            )
        overrides[key] = float(value)
    return name.strip(), overrides


def _parse_substitution_spec(text: str) -> dict[str, str]:
    """'Ti=Sn,O=S' → {"Ti": "Sn", "O": "S"}."""
    result: dict[str, str] = {}
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        src, _, dst = pair.partition("=")
        if not src.strip() or not dst.strip():
            raise ValueError(
                f'--prototype-substitute expects "El1=El2,El3=El4", got: {text!r}'
            )
        result[src.strip()] = dst.strip()
    return result


def _run_builders(args) -> bool:
    """Run structure builders; returns True when only building was requested."""
    builders = [args.ase_import, args.ase_build_bulk, args.ase_build_slab,
                args.ase_build_molecule, args.build_prototype,
                args.ase_build_crystal, args.ase_build_nanotube]
    if sum(1 for b in builders if b) > 1:
        raise ValueError(
            "Use only one of --ase-import, --ase-build-bulk, --ase-build-slab, "
            "--ase-build-molecule, --build-prototype, --ase-build-crystal, "
            "--ase-build-nanotube."
        )

    built = False

    if args.build_prototype:
        name, overrides = _parse_prototype_spec(args.build_prototype)
        substitutions = _parse_substitution_spec(args.prototype_substitute) if args.prototype_substitute else {}

        if name.startswith("mp:"):
            from vasp_auto.ml_tools import prototype_from_mp
            query = name[3:]  # strip "mp:" prefix
            struct = prototype_from_mp(
                query,
                substitutions=substitutions or None,
                api_key=getattr(args, "mp_api_key", None),
            )
            safe_name = query.replace("/", "_")
            if substitutions:
                sub_tag = "_".join(f"{k}{v}" for k, v in substitutions.items())
                safe_name = f"{safe_name}_{sub_tag}"
            output_dir = _builder_output_dir(args, safe_name)
            poscar_path = output_dir / "POSCAR"
            write_poscar(struct, poscar_path)
            args.target = str(output_dir)
            print(f"MP proto  : {struct['comment']} -> {poscar_path}")
        else:
            if substitutions:
                struct = make_prototype(name, **overrides)
                from vasp_auto.structure import substitute_species
                struct = substitute_species(struct, substitutions)
            else:
                struct = make_prototype(name, **overrides)
            canonical = resolve_prototype(name)
            output_dir = _builder_output_dir(args, canonical)
            poscar_path = output_dir / "POSCAR"
            write_poscar(struct, poscar_path)
            args.target = str(output_dir)
            print(f"Prototype : {canonical} -> {poscar_path}")
        built = True

    if args.ase_import:
        output_dir = _builder_output_dir(args, Path(args.ase_import).stem)
        poscar_path = import_structure_to_case(
            structure_path=args.ase_import,
            case_dir=output_dir,
            input_format=args.ase_format,
            index=args.ase_index,
        )
        args.target = str(poscar_path.parent)
        print(f"ASE import: {args.ase_import} -> {poscar_path}")
        built = True

    if args.ase_build_bulk:
        output_dir = _builder_output_dir(args, args.ase_build_bulk)
        poscar_path = build_bulk_case(
            symbol=args.ase_build_bulk,
            case_dir=output_dir,
            crystalstructure=args.ase_crystalstructure,
            a=args.ase_a,
            c=args.ase_c,
            cubic=args.ase_cubic,
        )
        args.target = str(poscar_path.parent)
        print(f"ASE bulk  : {args.ase_build_bulk} -> {poscar_path}")
        built = True

    if args.ase_build_slab:
        miller = _parse_int_triplet(args.ase_miller)
        miller_name = "".join(str(abs(i)) for i in miller)
        output_dir = _builder_output_dir(args, f"{Path(args.ase_build_slab).stem}_slab{miller_name}")
        repeat = None
        if args.ase_repeat:
            parts = [int(p) for p in args.ase_repeat.lower().replace(",", " ").replace("x", " ").split()]
            repeat = (parts[0], parts[1]) if len(parts) >= 2 else (parts[0], parts[0])
        poscar_path = build_slab_case(
            source=args.ase_build_slab,
            case_dir=output_dir,
            miller=miller,
            layers=args.ase_layers,
            vacuum=args.ase_vacuum,
            crystalstructure=args.ase_crystalstructure,
            a=args.ase_a,
            repeat=repeat,
        )
        args.target = str(poscar_path.parent)
        print(f"ASE slab  : {args.ase_build_slab} {miller} -> {poscar_path}")
        built = True

    if args.ase_build_molecule:
        output_dir = _builder_output_dir(args, args.ase_build_molecule)
        poscar_path = build_molecule_case(args.ase_build_molecule, output_dir, box=args.ase_box)
        args.target = str(poscar_path.parent)
        print(f"ASE mol   : {args.ase_build_molecule} -> {poscar_path}")
        built = True

    if args.ase_build_crystal:
        if args.ase_spacegroup is None or not args.ase_basis or args.ase_a is None:
            raise ValueError(
                "--ase-build-crystal needs --ase-spacegroup, --ase-basis and --ase-a."
            )
        symbols = args.ase_build_crystal.replace(",", " ").split()
        basis = [
            tuple(float(x) for x in site.replace(",", " ").split())
            for site in args.ase_basis.split(";") if site.strip()
        ]
        name = "".join(symbols) + f"_sg{args.ase_spacegroup}"
        output_dir = _builder_output_dir(args, name)
        poscar_path = build_crystal_case(
            symbols=symbols, basis=basis, spacegroup=args.ase_spacegroup,
            case_dir=output_dir, a=args.ase_a, b=args.ase_b, c=args.ase_c,
            alpha=args.ase_alpha, beta=args.ase_beta, gamma=args.ase_gamma,
        )
        args.target = str(poscar_path.parent)
        print(f"ASE xtal  : {' '.join(symbols)} sg{args.ase_spacegroup} -> {poscar_path}")
        built = True

    if args.ase_build_nanotube:
        output_dir = _builder_output_dir(
            args, f"{args.ase_build_nanotube}_nt{args.ase_nt_n}{args.ase_nt_m}")
        poscar_path = build_nanotube_case(
            symbol=args.ase_build_nanotube, n=args.ase_nt_n, m=args.ase_nt_m,
            case_dir=output_dir, length=args.ase_nt_length,
            bond=args.ase_nt_bond, vacuum=args.ase_vacuum,
        )
        args.target = str(poscar_path.parent)
        print(f"ASE tube  : {args.ase_build_nanotube} ({args.ase_nt_n},{args.ase_nt_m}) -> {poscar_path}")
        built = True

    built = _apply_structure_edits(args) or built
    return built and args.build_only


def _parse_cell_spec(struct, text: str) -> tuple[float, float, float]:
    """Cell-resize spec: '1.02', '1.02,1.02,1.0', or absolute 'a=4.1,c=22.5' (Å)."""
    text = text.strip()
    if "=" in text:
        lengths = cell_lengths(struct)
        factors = [1.0, 1.0, 1.0]
        for item in text.split(","):
            axis, _, value = item.partition("=")
            axis = axis.strip().lower()
            if axis not in ("a", "b", "c") or not value:
                raise ValueError(f'Use --scale-cell "a=4.1,c=22.5", got: {text!r}')
            index = "abc".index(axis)
            factors[index] = float(value) / lengths[index]
        return tuple(factors)
    parts = [float(p) for p in text.replace("x", ",").split(",") if p.strip()]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    raise ValueError(f"Use one or three cell scale factors, got: {text!r}")


def _parse_move_spec(text: str) -> tuple[int, tuple[float, float, float], bool]:
    """'5@x,y,z' places atom 5 (absolute); '5+dx,dy,dz' translates it."""
    for separator, absolute in (("@", True), ("+", False)):
        index_text, found, vector_text = text.partition(separator)
        if found:
            parts = [float(p) for p in vector_text.replace(",", " ").split()]
            if len(parts) != 3:
                break
            return int(index_text), (parts[0], parts[1], parts[2]), absolute
    raise ValueError(f'Use --move-atom "N@x,y,z" or "N+dx,dy,dz", got: {text!r}')


def _parse_adsorbate_spec(text: str) -> tuple[str, int, float]:
    """'O@12+2.0' = element O above atom 12 at 2.0 Å (height defaults to 2.0)."""
    element, _, rest = text.partition("@")
    anchor_text, _, height_text = rest.partition("+")
    if not element.strip() or not anchor_text.strip():
        raise ValueError(f'Use --adsorbate "El@N+h", got: {text!r}')
    return element.strip(), int(anchor_text), float(height_text) if height_text else 2.0


def _parse_freeze_spec(text: str) -> tuple[str, str]:
    """'z<0.3' or '1-8' with optional ':AXES' suffix → (selection, axes)."""
    selection, _, axes = text.rpartition(":")
    if selection and axes and all(a in "XYZxyz" for a in axes):
        return selection, axes.upper()
    return text, "XYZ"


def _apply_structure_edits(args) -> bool:
    """Pure-Python POSCAR edits: supercell, vacancy, substitution, add-atom,
    adsorbate, move-atom, cell resize, freeze (Selective dynamics)."""
    if not (
        args.supercell or args.vacancy or args.substitute or args.interstitial
        or args.adsorbate or args.move_atom or args.scale_cell or args.freeze
        or args.combine or args.delete
    ):
        return False

    case_dir = Path(args.target).expanduser().resolve()
    poscar = case_dir / "POSCAR"
    if not poscar.exists():
        raise FileNotFoundError(f"Structure edits need a POSCAR in the target case: {case_dir}")

    struct = read_poscar(poscar)
    suffix = ""

    if args.combine:
        guest_path = Path(args.combine).expanduser().resolve()
        guest_poscar = guest_path if guest_path.is_file() else guest_path / "POSCAR"
        if not guest_poscar.exists():
            raise FileNotFoundError(f"--combine needs a POSCAR: {guest_path}")
        guest = read_poscar(guest_poscar)
        shift = (0.0, 0.0)
        if args.combine_shift:
            parts = [float(p) for p in args.combine_shift.replace(",", " ").split()]
            if len(parts) != 2:
                raise ValueError(f'Use --combine-shift "x,y", got: {args.combine_shift!r}')
            shift = (parts[0], parts[1])
        struct = combine_structures(
            struct, guest, mode=args.combine_mode, gap=args.combine_gap,
            vacuum=args.combine_vacuum, shift=shift, strain_guest=args.combine_strain,
        )
        guest_name = guest_poscar.parent.name if guest_poscar.name == "POSCAR" else guest_poscar.stem
        suffix += f"_plus_{guest_name}"

    if args.supercell:
        repeat = parse_mesh(args.supercell)
        struct = make_supercell(struct, repeat)
        suffix += "_sc" + "x".join(str(n) for n in repeat)

    if args.vacancy:
        struct = make_vacancy(struct, args.vacancy)
        suffix += f"_vac{args.vacancy}"

    if args.delete:
        indices = parse_atom_selection(struct, args.delete)
        struct = delete_atoms(struct, indices)
        suffix += "_del"
        print(f"Delete    : {len(indices)} atoms ({args.delete})")

    if args.substitute:
        index_text, _, element = args.substitute.partition("=")
        if not element:
            raise ValueError(f"Use --substitute INDEX=Element, got: {args.substitute!r}")
        struct = substitute(struct, int(index_text), element.strip())
        suffix += f"_sub{index_text}{element.strip()}"

    if args.interstitial:
        element, _, coords_text = args.interstitial.partition("@")
        parts = [float(p) for p in coords_text.replace(",", " ").split()]
        if not element.strip() or len(parts) != 3:
            raise ValueError(f'Use --add-atom "El@x,y,z", got: {args.interstitial!r}')
        struct = add_interstitial(struct, element.strip(), (parts[0], parts[1], parts[2]))
        suffix += f"_int{element.strip()}"

    if args.adsorbate:
        element, anchor, height = _parse_adsorbate_spec(args.adsorbate)
        struct = add_adsorbate(struct, element, anchor, height)
        suffix += f"_ads{element}{anchor}"

    if args.move_atom:
        index, vector, absolute = _parse_move_spec(args.move_atom)
        struct = move_atom(struct, index, vector, absolute=absolute)
        suffix += f"_mv{index}"

    if args.scale_cell:
        factors = _parse_cell_spec(struct, args.scale_cell)
        struct = scale_cell(struct, factors)
        suffix += "_cell"

    if args.freeze:
        selection, axes = _parse_freeze_spec(args.freeze)
        indices = parse_atom_selection(struct, selection)
        struct = freeze_atoms(struct, indices, axes=axes)
        suffix += "_frz"
        print(f"Freeze    : {len(indices)} atoms ({selection}) on {axes}")

    output_dir = Path(args.ase_output) if args.ase_output else case_dir.parent / (case_dir.name + suffix)
    write_poscar(struct, output_dir / "POSCAR")
    args.target = str(output_dir)
    print(f"Structure : {poscar} -> {output_dir / 'POSCAR'}")
    return True


def _run_match_cells(args) -> bool:
    """--match-cells: print supercell pairs matching target (host) and guest."""
    if not args.match_cells:
        return False

    host_dir = Path(args.target).expanduser().resolve()
    host_poscar = host_dir if host_dir.is_file() else host_dir / "POSCAR"
    if not host_poscar.exists():
        raise FileNotFoundError(f"--match-cells needs a POSCAR in the target: {host_dir}")
    guest_path = Path(args.match_cells).expanduser().resolve()
    guest_poscar = guest_path if guest_path.is_file() else guest_path / "POSCAR"
    if not guest_poscar.exists():
        raise FileNotFoundError(f"--match-cells needs a guest POSCAR: {guest_path}")

    host = read_poscar(host_poscar)
    guest = read_poscar(guest_poscar)
    matches = match_supercells(
        host, guest, max_repeat=args.match_max, max_strain=args.match_strain,
        gamma_tol=args.match_gamma_tol,
    )
    print(f"Host      : {host_poscar} (a/b = "
          f"{', '.join(f'{x:.3f}' for x in cell_lengths(host)[:2])} Å)")
    print(f"Guest     : {guest_poscar} (a/b = "
          f"{', '.join(f'{x:.3f}' for x in cell_lengths(guest)[:2])} Å)")
    if not matches:
        print(
            f"No match  : within {args.match_max}x{args.match_max} repeats, "
            f"{args.match_strain * 100:.0f}% strain and {args.match_gamma_tol:.0f} deg "
            "angle mismatch. Raise --match-max / --match-strain / --match-gamma-tol, "
            "or skip straining (--combine without --combine-strain centres the guest)."
        )
        return True

    print(f"Angle     : {matches[0]['gamma_mismatch_deg']:.2f} deg in-plane mismatch "
          "(straining shears the guest by this much)")
    print(f"{'host':>8} {'guest':>8} {'strain a':>9} {'strain b':>9} {'atoms':>7}")
    for m in matches:
        print(
            f"{m['host_repeat'][0]}x{m['host_repeat'][1]:<2}".rjust(8)
            + f" {m['guest_repeat'][0]}x{m['guest_repeat'][1]:<2}".rjust(9)
            + f" {m['strain_a'] * 100:>8.2f}%"
            + f" {m['strain_b'] * 100:>8.2f}%"
            + f" {m['host_atoms'] + m['guest_atoms']:>7}"
        )
    best = matches[0]
    print()
    print("Next step : apply the supercells, then stack with the guest strained on:")
    print(f"  vasp-auto {host_dir} --supercell "
          f"{best['host_repeat'][0]}x{best['host_repeat'][1]}x1 --build-only")
    print(f"  vasp-auto {guest_path} --supercell "
          f"{best['guest_repeat'][0]}x{best['guest_repeat'][1]}x1 --build-only")
    print(f"  vasp-auto <host_supercell_case> --combine <guest_supercell_case> "
          "--combine-strain --build-only")
    return True


def _write_csv(path: Path, header: list[str], rows: list[list]):
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote     : {path}")


def _run_analysis(args, config) -> bool:
    """Post-processing commands on finished jobs; True = nothing left to run."""
    requested = any(
        (args.adsorption_energy, args.thermo, args.work_function, args.d_band,
         args.chg_diff, args.bader, args.optics_parse, args.dos_export,
         args.bands_export, args.xrd)
    )
    if not requested:
        return False
    from vasp_auto import analysis
    from vasp_auto.chgcar import charge_difference, run_bader

    if args.adsorption_energy:
        dirs = [Path(p.strip()) for p in args.adsorption_energy.split(",") if p.strip()]
        if len(dirs) != 3:
            raise ValueError(
                f'Use --adsorption-energy "TOTAL,SLAB,MOL", got: {args.adsorption_energy!r}'
            )
        result = analysis.adsorption_energy(*dirs, molecule_scale=args.molecule_scale)
        print(f"E(slab+ads): {result['total_energy_eV']:.6f} eV")
        print(f"E(slab)    : {result['slab_energy_eV']:.6f} eV")
        print(
            f"E(molecule): {result['molecule_energy_eV']:.6f} eV"
            f" x {result['molecule_scale']}"
        )
        print(f"E_ads      : {result['adsorption_energy_eV']:.6f} eV")
        if not result["all_converged"]:
            print("WARNING    : at least one of the three jobs is not converged")

    if args.thermo:
        result = analysis.thermo_from_job(Path(args.thermo), temperature=args.temperature)
        print(f"Modes      : {result['n_modes']} real, {result['n_imaginary']} imaginary")
        for mode in result["modes"]:
            marker = " (imaginary)" if mode["imaginary"] else ""
            print(f"  {mode['index']:>3}  {mode['cm1']:>10.2f} cm-1  {mode['meV']:>9.3f} meV{marker}")
        if result["n_imaginary"]:
            print("WARNING    : imaginary modes present — structure may not be a minimum")
        print(f"T          : {result['temperature_K']:.2f} K")
        print(f"ZPE        : {result['zpe_eV']:.6f} eV")
        print(f"U_vib      : {result['u_vib_eV']:.6f} eV")
        print(f"T*S        : {result['ts_eV']:.6f} eV")
        print(f"G correction (ZPE + U_vib - T*S): {result['g_correction_eV']:.6f} eV")
        if result.get("g_total_eV") is not None:
            print(f"G = E_DFT + correction: {result['g_total_eV']:.6f} eV")

    if args.work_function:
        job_dir = Path(args.work_function)
        result = analysis.work_function(job_dir)
        print(f"Vacuum     : {result['vacuum_level_eV']:.4f} eV")
        print(f"E_Fermi    : {result['fermi_eV']:.4f} eV")
        print(f"Work func. : {result['work_function_eV']:.4f} eV")
        profile = result["profile_eV"]
        rows = [[i / len(profile), value] for i, value in enumerate(profile)]
        _write_csv(job_dir / "potential_profile.csv", ["frac_position", "potential_eV"], rows)

    if args.d_band:
        from vasp_auto.structure import parse_atom_selection, read_poscar
        dir_text, _, selection = args.d_band.rpartition(":")
        if not dir_text or not selection:
            raise ValueError(f'Use --d-band "DIR:SELECTION", got: {args.d_band!r}')
        job_dir = Path(dir_text)
        struct = read_poscar(job_dir / "POSCAR")
        atoms = parse_atom_selection(struct, selection)
        from vasp_auto.workflow import job_engine
        result = (
            analysis.qe_d_band_center(job_dir, atoms, emax_eV=args.d_band_emax)
            if job_engine(job_dir) == "qe"
            else analysis.d_band_center(job_dir / "vasprun.xml", atoms,
                                        emax_eV=args.d_band_emax)
        )
        print(f"Atoms      : {len(atoms)} selected ({selection})")
        print(f"d-band center: {result['d_band_center_eV']:.4f} eV (vs E_F)")
        print(f"d-band width : {result['d_band_width_eV']:.4f} eV")

    if args.chg_diff:
        paths = []
        for item in args.chg_diff.split(","):
            p = Path(item.strip()).expanduser()
            if p.is_file():
                paths.append(p)
            elif (p / ".engine").exists() and (p / ".engine").read_text().strip() == "qe":
                paths.append(p / "charge-density.cube")
            else:
                paths.append(p / "CHGCAR")
        if len(paths) < 2:
            raise ValueError(f'Use --chg-diff "TOTAL,PART1[,PART2...]", got: {args.chg_diff!r}')
        is_qe_cube = paths[0].suffix.lower() == ".cube"
        output = paths[0].parent / ("charge-density-diff.cube" if is_qe_cube else "CHGCAR_diff")
        if is_qe_cube:
            from vasp_auto.qe_volumetric import cube_difference
            cube_difference(paths[0], paths[1:], output)
        else:
            charge_difference(paths[0], paths[1:], output)
        print(f"Wrote     : {output} (rho_total - sum of {len(paths) - 1} parts)")

    if args.bader:
        job_dir = Path(args.bader)
        result = run_bader(job_dir, config.get("bader_executable", "bader"))
        reference = result.get("reference") or (
            "AECCAR0+AECCAR2 reference" if result["used_aeccar_reference"]
            else "CHGCAR only (no AECCARs found)"
        )
        print(f"Bader      : {reference}")
        rows = []
        for charge in result["charges"]:
            print(
                f"  {charge['index']:>3} {charge['element']:<2} "
                f"electrons {charge['electrons']:>8.4f}  net {charge['net_charge']:>+8.4f} e"
            )
            rows.append([charge["index"], charge["element"], charge["electrons"], charge["net_charge"]])
        _write_csv(job_dir / "bader_charges.csv", ["index", "element", "electrons", "net_charge_e"], rows)

    if args.dos_export:
        from vasp_auto.parser import (
            aggregate_pdos, aggregate_qe_pdos, parse_dos, parse_pdos, parse_qe_dos,
        )
        from vasp_auto.workflow import job_engine
        job_dir = Path(args.dos_export)
        dos = parse_qe_dos(job_dir) if job_engine(job_dir) == "qe" else parse_dos(job_dir / "vasprun.xml")
        if dos is None:
            raise FileNotFoundError(
                f"No DOS in {job_dir} — run a 'dos' calculation there first."
            )
        header = ["energy_eV", "total_up"] + (["total_down"] if len(dos["total"]) > 1 else [])
        rows = [
            [energy] + [channel[i] for channel in dos["total"]]
            for i, energy in enumerate(dos["energies"])
        ]
        print(f"E_Fermi    : {dos['efermi']:.4f} eV" if dos["efermi"] is not None else "E_Fermi    : ?")
        _write_csv(job_dir / "dos.csv", header, rows)

        struct = read_poscar(job_dir / "POSCAR")
        symbols = []
        for element, count in zip(struct["elements"], struct["counts"]):
            symbols.extend([element] * count)
        if job_engine(job_dir) == "qe":
            aggregated = aggregate_qe_pdos(job_dir, symbols)
        else:
            pdos = parse_pdos(job_dir / "vasprun.xml")
            aggregated = aggregate_pdos(pdos, symbols) if pdos is not None else None
        if aggregated is not None:
            labels = [
                f"{c['label'].replace(' ', '_')}" + (f"_spin{c['spin'] + 1}" if len(dos["total"]) > 1 else "")
                for c in aggregated["curves"]
            ]
            rows = [
                [energy] + [c["values"][i] for c in aggregated["curves"]]
                for i, energy in enumerate(aggregated["energies"])
            ]
            _write_csv(job_dir / "pdos.csv", ["energy_eV"] + labels, rows)

    if args.bands_export:
        from vasp_auto.parser import parse_bands, parse_qe_bands
        from vasp_auto.workflow import job_engine
        job_dir = Path(args.bands_export)
        bands = (parse_qe_bands(job_dir) if job_engine(job_dir) == "qe"
                 else parse_bands(job_dir / "vasprun.xml", job_dir / "KPOINTS"))
        if bands is None:
            raise FileNotFoundError(
                f"No eigenvalues in {job_dir} — run a 'bands' calculation there first."
            )
        nspins = len(bands["bands"])
        header = ["distance_invA", "kx", "ky", "kz", "label"]
        for spin in range(nspins):
            suffix = f"_spin{spin + 1}" if nspins > 1 else ""
            header += [f"band{b + 1}{suffix}" for b in range(len(bands["bands"][spin]))]
        label_at = {entry["index"]: entry["label"] for entry in bands["labels"]}
        rows = []
        for k, distance in enumerate(bands["distances"]):
            row = [distance] + list(bands["kpoints"][k]) + [label_at.get(k, "")]
            for spin_bands in bands["bands"]:
                row += [band[k] for band in spin_bands]
            rows.append(row)
        if bands["efermi"] is not None:
            print(f"E_Fermi    : {bands['efermi']:.4f} eV")
        print(f"Bands      : {len(bands['bands'][0])} bands x {len(bands['distances'])} k-points"
              + (f", {nspins} spins" if nspins > 1 else ""))
        _write_csv(job_dir / "bands.csv", header, rows)

    if args.xrd:
        job_dir = Path(args.xrd)
        result = analysis.xrd_pattern(job_dir)
        print(f"Structure  : {result['structure_file']}  "
              f"(λ = {result['wavelength_A']:.4f} Å, Cu Ka)")
        print(f"Peaks      : {len(result['two_theta_deg'])} in 10-90° 2θ")
        rows = [
            [t, i, d, f'"{hkl}"']
            for t, i, d, hkl in zip(result["two_theta_deg"], result["intensity"],
                                    result["d_spacing_A"], result["hkl"])
        ]
        _write_csv(job_dir / "xrd.csv",
                   ["two_theta_deg", "intensity", "d_spacing_A", "hkl"], rows)

    if args.optics_parse:
        job_dir = Path(args.optics_parse)
        result = analysis.absorption_spectrum(job_dir)
        rows = [
            [energy, alpha, e1, e2]
            for energy, alpha, e1, e2 in zip(
                result["energies_eV"], result["alpha_cm1"], result["real"], result["imag"]
            )
        ]
        _write_csv(
            job_dir / "absorption.csv",
            ["energy_eV", "alpha_cm1", "epsilon_real", "epsilon_imag"],
            rows,
        )
        static = result["real"][0] if result["real"] else None
        if static is not None:
            print(f"epsilon(0) : {static:.4f} (direction-averaged static dielectric constant)")

    return True


def _poll_job(args, config) -> bool:
    """--poll JOBID: query the scheduler for job status and print, then exit."""
    if not args.poll:
        return False
    from vasp_auto.runner import poll_job_status

    scheduler = args.scheduler or config.get("scheduler", "local")
    if scheduler == "local":
        print("Poll      : local scheduler — no queue to poll")
        return True

    result = poll_job_status(args.poll, scheduler=scheduler)
    print(f"Job ID    : {result['job_id']}")
    print(f"Scheduler : {result['scheduler']}")
    print(f"State     : {result['state']}")
    if result["raw"]:
        print(f"Raw       : {result['raw']}")
    return True


def _list_jobs(args, config) -> bool:
    """--list-jobs: print running jobs across local + every remote, then exit."""
    if not getattr(args, "list_jobs", False):
        return False
    from vasp_auto.runner import list_running_jobs

    result = list_running_jobs(config, running_only=not args.all)
    jobs = result["jobs"]
    if not jobs:
        print("No jobs running" if not args.all else "No jobs found")
    else:
        print(f"{'MACHINE':<12} {'STATUS':<9} {'TYPE':<11} {'ENERGY (eV)':>13}  JOB")
        for j in jobs:
            energy = f"{j['energy_eV']:.4f}" if "energy_eV" in j else "-"
            print(f"{j['machine']:<12} {j['status']:<9} "
                  f"{j.get('calculation_type', 'scf'):<11} {energy:>13}  {j['name']}")
    for err in result["errors"]:
        print(f"  ! {err['machine']}: {err['error']}")
    return True


def _kill_job(args, config) -> bool:
    """--kill JOB_DIR: stop a detached offload job on its remote machine, then exit."""
    if not args.kill:
        return False
    from vasp_auto.runner import kill_detached_job
    from vasp_auto.workflow import read_remote_marker

    job_dir = Path(args.kill).expanduser().resolve()
    marker = read_remote_marker(job_dir)
    if not marker or marker.get("mode") != "ssh_detached":
        raise SystemExit(
            f"No detached-offload job in {job_dir} (no ssh_detached .remote.json). "
            "Local runs stop with Ctrl-C; scheduler jobs with scancel/qdel."
        )
    name = marker.get("machine") or marker.get("host")
    remote = (config.get("remotes") or {}).get(name) or {"host": marker.get("host")}
    result = kill_detached_job(remote, marker.get("control_dir", ""), marker.get("pid"))
    try:  # local mirror marker so the UI Results tab reads "terminated"
        (job_dir / ".terminated").write_text("terminated", encoding="utf-8")
    except OSError:
        pass
    status = "stopped" if result["killed"] else "sent (job may already have exited)"
    print(f"Kill      : {name} pid {marker.get('pid') or '?'} — {status}")
    if result["raw"]:
        print(f"  {result['raw']}")
    return True


def _run_resume(args, config) -> bool:
    """--resume-job-dir DIR: resume one job in place, locally or on a remote.

    Local: reuse the directory's own INCAR/KPOINTS/POTCAR and restart from its
    newest CONTCAR (workflow.resume_job). Remote (--remote/--remote-config): do
    the same in place on the machine the job lives on — synchronously over SSH
    (runner.resume_job_remote) or, for offload machines (run_mode ssh_detached),
    detached under setsid so the local host can power off
    (runner.resume_job_detached). --resume-local-mirror records a local job dir so
    the UI's status/fetch buttons can track the run.
    """
    if not args.resume_job_dir:
        return False
    remote = resolve_remote(args, config)
    job_dir = args.resume_job_dir
    if remote:
        machine = remote.get("name") or remote.get("host")
        mirror = args.resume_local_mirror or None
        # Offload machines resume detached (setsid): the restart keeps running
        # after SSH disconnects, so the local host can power off. Other remotes
        # run mpirun synchronously over SSH and pull the results back.
        if resolve_remote_run_mode(remote) == "ssh_detached":
            from vasp_auto.runner import resume_job_detached
            print(f"Resume    : {job_dir} on {machine} (detached; local host can power off)")
            result = resume_job_detached(
                remote, job_dir, cpus=args.cpus,
                local_job_dir=mirror, on_progress=lambda m: print(m),
            )
            print(f"Launched  : {machine} pid {result['pid'] or '?'} -> {result['remote_dir']}")
            print("Fetch later from the Results tab (⬇) or with --parse-only after copying back.")
        else:
            from vasp_auto.runner import resume_job_remote
            print(f"Resume    : {job_dir} on {machine} (in place, from newest CONTCAR)")
            return_code = resume_job_remote(
                remote, job_dir, cpus=args.cpus,
                on_progress=lambda m: print(m), local_job_dir=mirror,
            )
            print(f"Finished  : remote resume of {job_dir} (rc={return_code})")
    else:
        from vasp_auto.workflow import resume_job
        resume_job(
            job_dir,
            vasp_executable=config.get("vasp_executable"),
            cpus=args.cpus,
        )
        print(f"Finished  : {job_dir}")
    return True


def _apply_remote_setup(args, config) -> bool:
    """--remote-setup: install the engine venv on the chosen remote, then exit."""
    if not args.remote_setup:
        return False
    from vasp_auto.runner import setup_remote_engine

    remote = resolve_remote(args, config)
    if not remote:
        raise SystemExit("--remote-setup needs a machine: use --remote NAME or --remote-config FILE.")
    machine = remote.get("name") or remote.get("host", "?")
    print(f"Remote setup : installing vasp_auto on {machine} ({remote.get('host', '?')}) …")
    result = setup_remote_engine(remote, on_progress=lambda m: print(f"  {m}"))
    if result["ok"]:
        print(f"Installed    : {result['vasp_auto']}")
        print("Ready        : offload runs (run_mode: ssh_detached) can now run here.")
    else:
        print("FAILED       : remote engine install did not complete.")
        print(result["detail"])
        raise SystemExit(1)
    return True


def _apply_ml_energy(args, config) -> bool:
    """--ml-energy: single-point MLIP screen; read-only, exits after printing.

    When --db-source and --db-query are also set, the structure is fetched from
    the external database instead of read from disk.
    """
    if not args.ml_energy and not (getattr(args, "db_source", None) and getattr(args, "db_query", None)):
        return False
    # DB-fetch path: --db-source + --db-query (--ml-energy TARGET is optional/ignored)
    db_source = getattr(args, "db_source", None)
    db_query = getattr(args, "db_query", None)
    if db_source and db_query and not args.ml_energy:
        # Pure DB fetch + energy mode (no local file needed)
        from vasp_auto.ml_tools import DEFAULT_ML_MODEL, DEFAULT_ML_TASK, ml_energy_from_db

        result = ml_energy_from_db(
            db_query,
            db_source=db_source,
            api_key=getattr(args, "mp_api_key", None),
            model=args.ml_model or config.get("ml_model") or DEFAULT_ML_MODEL,
            task=args.ml_task or config.get("ml_task") or DEFAULT_ML_TASK,
            checkpoint=args.ml_checkpoint or config.get("ml_checkpoint"),
        )
        print(f"DB source : {result['db_source']} ({result['db_label']})")
        print(f"ML energy : {result['energy_eV']:.6f} eV")
        print(f"max|F|    : {result['max_force_eV_A']:.4f} eV/Å")
        print(f"model     : {result['model']}")
        return True

    if not args.ml_energy:
        return False

    from vasp_auto.ml_tools import DEFAULT_ML_MODEL, DEFAULT_ML_TASK, ml_energy

    poscar_path = Path(args.ml_energy).expanduser().resolve()
    result = ml_energy(
        poscar_path,
        model=args.ml_model or config.get("ml_model") or DEFAULT_ML_MODEL,
        task=args.ml_task or config.get("ml_task") or DEFAULT_ML_TASK,
        checkpoint=args.ml_checkpoint or config.get("ml_checkpoint"),
    )
    print(f"ML energy : {result['energy_eV']:.6f} eV")
    print(f"max|F|    : {result['max_force_eV_A']:.4f} eV/Å")
    print(f"model     : {result['model']}")
    return True


def _apply_db_prerelax(args, config) -> bool:
    """--db-prerelax: replace the case POSCAR with a DB-fetched structure and continue.

    Fetches the structure from the database, writes it as POSCAR into the case
    directory (overwriting any existing POSCAR), then returns False so the normal
    VASP pipeline continues with the new geometry.  The case directory itself is
    not changed — only its POSCAR is replaced.

    Returns True only when --build-only is also set (stop after writing POSCAR).
    """
    if not getattr(args, "db_prerelax", False):
        return False
    db_source = getattr(args, "db_source", None)
    db_query = getattr(args, "db_query", None)
    if not (db_source and db_query):
        raise ValueError("--db-prerelax requires both --db-source and --db-query.")

    from vasp_auto.ml_tools import fetch_structure_from_db

    poscar_text, label = fetch_structure_from_db(
        db_query,
        db_source=db_source,
        api_key=getattr(args, "mp_api_key", None),
    )
    case_dir = Path(args.target).expanduser().resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "POSCAR").write_text(poscar_text, encoding="utf-8")
    print(f"DB prerelax: {db_source} / {label} -> {case_dir / 'POSCAR'}")
    return bool(getattr(args, "build_only", False))


def _apply_db_fetch(args, config) -> bool:
    """--db-source + --db-query + --build-only: download a structure as POSCAR.

    Returns True (stop pipeline) only when --build-only is set and no ML flag
    is active, so the fetched POSCAR can be saved to a case directory.
    """
    db_source = getattr(args, "db_source", None)
    db_query = getattr(args, "db_query", None)
    if not (db_source and db_query):
        return False
    # Handled by _apply_db_prerelax, _apply_ml_relax, or _apply_ml_energy.
    if getattr(args, "db_prerelax", False):
        return False
    if getattr(args, "ml_relax", False) or getattr(args, "ml_only", False):
        return False
    if args.ml_energy:
        return False

    from vasp_auto.ml_tools import fetch_structure_from_db

    poscar_text, label = fetch_structure_from_db(
        db_query,
        db_source=db_source,
        api_key=getattr(args, "mp_api_key", None),
    )
    target = Path(args.target).expanduser().resolve() if args.target else Path.cwd() / label.replace("/", "_")
    target.mkdir(parents=True, exist_ok=True)
    (target / "POSCAR").write_text(poscar_text, encoding="utf-8")
    print(f"DB fetch  : {db_source} / {label} -> {target / 'POSCAR'}")
    args.target = str(target)
    return bool(getattr(args, "build_only", False))


def _apply_ml_relax(args, config) -> bool:
    """--ml-relax: MLIP pre-relaxation into a derived case; True = stop after.

    When --db-source and --db-query are also set, the structure is fetched from
    the external database rather than read from a local POSCAR.
    """
    if not (args.ml_relax or args.ml_only):
        return False

    db_source = getattr(args, "db_source", None)
    db_query = getattr(args, "db_query", None)

    if db_source and db_query:
        from vasp_auto.ml_tools import DEFAULT_ML_MODEL, DEFAULT_ML_TASK, ml_relax_from_db

        result = ml_relax_from_db(
            db_query,
            db_source=db_source,
            api_key=getattr(args, "mp_api_key", None),
            model=args.ml_model or config.get("ml_model") or DEFAULT_ML_MODEL,
            task=args.ml_task or config.get("ml_task") or DEFAULT_ML_TASK,
            checkpoint=args.ml_checkpoint or config.get("ml_checkpoint"),
            fmax=args.ml_fmax,
            steps=args.ml_steps,
            relax_cell=args.ml_relax_cell,
        )
        args.target = result["case_dir"]
        print(
            f"DB source : {result['db_source']} ({result['db_label']})\n"
            f"ML relax  : {result['db_query']} -> {result['case_dir']} "
            f"(E = {result['energy_eV']:.6f} eV, max|F| = {result['max_force_eV_A']:.3f} eV/Å, "
            f"{result['steps']} steps, model {result['model']})"
        )
        return args.ml_only

    from vasp_auto.ml_tools import DEFAULT_ML_MODEL, DEFAULT_ML_TASK, ml_relax_case

    case_dir = Path(args.target).expanduser().resolve()
    result = ml_relax_case(
        case_dir,
        model=args.ml_model or config.get("ml_model") or DEFAULT_ML_MODEL,
        task=args.ml_task or config.get("ml_task") or DEFAULT_ML_TASK,
        checkpoint=args.ml_checkpoint or config.get("ml_checkpoint"),
        fmax=args.ml_fmax,
        steps=args.ml_steps,
        relax_cell=args.ml_relax_cell,
    )
    args.target = result["case_dir"]
    print(
        f"ML relax  : {case_dir.name} -> {result['case_dir']} "
        f"(E = {result['energy_eV']:.6f} eV, max|F| = {result['max_force_eV_A']:.3f} eV/Å, "
        f"{result['steps']} steps, model {result['model']})"
    )
    return args.ml_only


def _kpoints_spec_from_args(args) -> dict | None:
    if not (args.kpoints_mode or args.kmesh or args.kspacing or args.kpath):
        return None

    mode = args.kpoints_mode
    kpath = args.kpath
    if mode is None:
        # "auto" implies line mode with automatic preset detection.
        if kpath and kpath.strip().lower() == "auto":
            mode = "line"
        elif kpath:
            mode = "line"
        elif args.kspacing:
            mode = "spacing"
        else:
            mode = "gamma"

    return {
        "mode": mode,
        "mesh": args.kmesh,
        "spacing": args.kspacing,
        "kpath": kpath,
        "divisions": args.kpath_divisions,
    }


def _print_preview(preview: dict):
    print(f"--- dry run: {preview['case_name']} ({preview['calc_type']}) ---")
    print(f"job_dir   : {preview['job_dir']}")
    print(f"POSCAR    : {preview['POSCAR']}")
    if preview.get("engine") == "qe":
        print(f"pseudos   : {preview.get('pseudos', '')}")
        print("--- pw.in ---")
        print(preview["pw.in"].rstrip())
        print("-------------")
        return
    if preview.get("engine") == "ase":
        print(f"calculator: {preview.get('calculator')}")
        print("--- ase_calc.json ---")
        print(preview["ase_calc.json"].rstrip())
        print("-------------")
        return
    print(f"POTCAR    : {preview['POTCAR']}")
    if preview.get("neb_images"):
        print(f"NEB images: {preview['neb_images']}")
    print("--- INCAR ---")
    print(preview["INCAR"].rstrip())
    print("--- KPOINTS ---")
    print(preview["KPOINTS"].rstrip())
    print("-------------")


# Outputs worth carrying across a --retry-failed re-run: the half-relaxed
# geometry continues the optimisation, the wavefunction/density seed the SCF.
RESTART_FILES = ("CONTCAR", "WAVECAR", "CHGCAR")


def _stash_restart_files(job_dir: Path) -> Path | None:
    """Move restart outputs aside before the job directory is cleaned."""
    job_dir = Path(job_dir)
    if not job_dir.exists():
        return None
    movable = [
        name for name in RESTART_FILES
        if (job_dir / name).exists() and (job_dir / name).stat().st_size > 0
    ]
    if not movable:
        return None
    stash = Path(tempfile.mkdtemp(prefix=".restart_", dir=job_dir.parent))
    for name in movable:
        shutil.move(str(job_dir / name), str(stash / name))
    return stash


def _restore_restart_files(job_dir: Path, stash: Path | None):
    """Seed the cleaned job: CONTCAR becomes POSCAR; WAVECAR/CHGCAR return as-is."""
    if stash is None:
        return
    job_dir = Path(job_dir)
    contcar = stash / "CONTCAR"
    if contcar.exists():
        shutil.move(str(contcar), str(job_dir / "POSCAR"))
        print("Restart   : POSCAR seeded from previous CONTCAR")
    for name in ("WAVECAR", "CHGCAR"):
        seed = stash / name
        if seed.exists():
            shutil.move(str(seed), str(job_dir / name))
    shutil.rmtree(stash, ignore_errors=True)


def resolve_remote(args, config):
    """Resolve the remote-machine config from --remote-config / --remote NAME.

    Precedence: --remote-config FILE (JSON/YAML) > --remote NAME (config remotes[NAME])
    > bare --remote (config remote:). Returns the remote dict, or None for a local run.
    """
    if args.remote_config:
        path = Path(args.remote_config).expanduser()
        text = path.read_text(encoding="utf-8")
        import yaml
        remote = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)
        if not isinstance(remote, dict):
            raise SystemExit(f"--remote-config must contain a mapping: {path}")
        return remote

    if args.remote is None:
        return None

    if args.remote == "__default__":
        remote = config.get("remote")
        if not remote:
            raise SystemExit(
                "--remote requires a 'remote:' section in config.yaml "
                "(host, remote_root, vasp_executable)."
            )
        return dict(remote)

    remotes = config.get("remotes") or {}
    if args.remote not in remotes:
        available = ", ".join(sorted(remotes)) or "none configured"
        raise SystemExit(
            f"No remote machine named '{args.remote}' in config.yaml remotes: "
            f"(available: {available})."
        )
    remote = dict(remotes[args.remote])
    remote.setdefault("name", args.remote)
    return remote


def resolve_engine(args, config) -> tuple[str, dict]:
    """Resolve the DFT engine and overlay CLI engine overrides onto the config.

    Precedence: --engine flag > config 'engine:' (global/project/case) > 'vasp'.
    --qe-executable / --pseudo-dir override the matching config keys. Returns
    (engine, config) with the (possibly updated) config dict.
    """
    engine = args.engine or config.get("engine", "vasp")
    overrides = {}
    if args.qe_executable:
        overrides["qe_executable"] = args.qe_executable
    if args.pseudo_dir:
        overrides["pseudo_dir"] = str(Path(args.pseudo_dir).expanduser().resolve())
    if getattr(args, "ase_calculator", None):
        overrides["ase_calculator"] = args.ase_calculator
    if getattr(args, "ase_fmax", None) is not None:
        overrides["ase_fmax"] = args.ase_fmax
    if getattr(args, "ase_steps", None) is not None:
        overrides["ase_steps"] = args.ase_steps
    # ase_calc_params is a dict: start from config, merge --ase-params JSON, then
    # set the command path. So --ase-command and explicit params compose cleanly.
    ase_params = dict(config.get("ase_calc_params") or {})
    if getattr(args, "ase_params", None):
        try:
            extra = json.loads(args.ase_params)
        except ValueError as exc:
            raise SystemExit(f"--ase-params is not valid JSON: {exc}") from None
        if not isinstance(extra, dict):
            raise SystemExit("--ase-params must be a JSON object (e.g. '{\"xc\":\"PBE\"}').")
        ase_params.update(extra)
    if getattr(args, "ase_command", None):
        ase_params["command"] = args.ase_command
    if ase_params != (config.get("ase_calc_params") or {}):
        overrides["ase_calc_params"] = ase_params
    if overrides:
        config = {**config, **overrides}
    return engine, config


def _forward_calc_flags(args, engine="vasp") -> list[str]:
    """Reconstruct the calc-shaping CLI flags to forward to a remote offload run.

    The remote engine runs the case as a normal local job to itself, so this
    excludes target/remote/scheduler/parallel/background flags. Defaulted flags
    (energy_tol, sigma_tol, kpath_divisions, solvation_eps) are only forwarded
    when the feature that uses them is active. ``engine`` is the *resolved*
    engine (flag or config), forwarded explicitly; --qe-executable/--pseudo-dir
    are host-local paths and never forwarded — the machine's own values go into
    the remote config.yaml (see submit_job_detached) or the shipped bundle.
    """
    flags: list[str] = []

    def add(flag, value=None):
        flags.append(flag)
        if value is not None:
            flags.append(str(value))

    if args.calc_type:
        add("--calc-type", args.calc_type)
    if engine != "vasp":
        add("--engine", engine)
    if getattr(args, "ase_calculator", None):
        add("--ase-calculator", args.ase_calculator)
    if getattr(args, "ase_command", None):
        add("--ase-command", args.ase_command)
    if getattr(args, "ase_params", None):
        add("--ase-params", args.ase_params)
    if getattr(args, "ase_fmax", None) is not None:
        add("--ase-fmax", args.ase_fmax)
    if getattr(args, "ase_steps", None) is not None:
        add("--ase-steps", args.ase_steps)

    converge = bool(args.converge_scf or args.converge_encut or args.converge_sigma)
    if args.converge_scf:
        add("--converge-scf")
    if args.converge_encut:
        add("--converge-encut", args.converge_encut)
    if args.converge_sigma:
        add("--converge-sigma", args.converge_sigma)
    if args.nelm_values:
        add("--nelm-values", args.nelm_values)
    if args.kpoints_values:
        add("--kpoints-values", args.kpoints_values)
    if converge:
        add("--energy-tol", args.energy_tol)
        add("--sigma-tol", args.sigma_tol)
    if args.reuse_wavecar:
        add("--reuse-wavecar")

    if getattr(args, "kpoints_mode", None):
        add("--kpoints-mode", args.kpoints_mode)
    if getattr(args, "kmesh", None):
        add("--kmesh", args.kmesh)
    if getattr(args, "kspacing", None):
        add("--kspacing", args.kspacing)
    if getattr(args, "kpath", None):
        add("--kpath", args.kpath)
        add("--kpath-divisions", args.kpath_divisions)

    if args.spin:
        add("--spin")
    if args.magmom:
        add("--magmom", args.magmom)
    if args.workflow:
        add("--workflow", args.workflow)
    if args.neb_images:
        add("--neb-images", args.neb_images)
    if args.auto_retry:
        add("--auto-retry", args.auto_retry)
    if getattr(args, "solvation", False):
        add("--solvation")
        add("--solvation-eps", args.solvation_eps)
    return flags


def _bundle_tss_inputs(case_dir: Path, bundle: Path, case_name: str) -> Path:
    """Copy a TSS/NEB case's inputs into the offload bundle and return the POSCAR
    to build the (shared) POTCAR from.

    Handles both the endpoint layout (initial/POSCAR + final/POSCAR, which the
    remote engine interpolates) and an already-expanded image layout (00/, 01/,
    … POSCARs). INCAR/KPOINTS are forwarded if present; the NEB INCAR is otherwise
    built on the remote."""
    initial = case_dir / "initial" / "POSCAR"
    final = case_dir / "final" / "POSCAR"
    image_dirs = [p for p in sorted(case_dir.iterdir()) if p.is_dir() and p.name.isdigit()]
    if initial.exists() and final.exists():
        (bundle / "initial").mkdir()
        (bundle / "final").mkdir()
        shutil.copy2(initial, bundle / "initial" / "POSCAR")
        shutil.copy2(final, bundle / "final" / "POSCAR")
        potcar_src = bundle / "initial" / "POSCAR"
    elif image_dirs:
        for image_dir in image_dirs:
            (bundle / image_dir.name).mkdir()
            shutil.copy2(image_dir / "POSCAR", bundle / image_dir.name / "POSCAR")
        potcar_src = bundle / image_dirs[0].name / "POSCAR"
    else:
        raise SystemExit(
            f"{case_name}: TSS/NEB case needs initial/POSCAR and final/POSCAR "
            "(or expanded 00/, 01/, … image folders)."
        )
    for optional in ("INCAR", "KPOINTS"):
        if (case_dir / optional).exists():
            shutil.copy2(case_dir / optional, bundle / optional)
    return potcar_src


def _bundle_qe_pseudos(bundle: Path, case_name: str, config: dict, machine) -> list[str]:
    """Copy the case's UPF pseudopotentials into the offload bundle.

    Returns the --pseudo-dir flag pointing at the shipped copy — a path relative
    to the remote engine's working directory (remote_root), where run.sh cds."""
    from vasp_auto.qe_tools import pseudo_files_for_struct
    from vasp_auto.structure import read_poscar

    poscar = bundle / "POSCAR"
    if not poscar.exists():
        poscar = bundle / "initial" / "POSCAR"
    struct = read_poscar(poscar)
    pseudos = pseudo_files_for_struct(struct, config.get("pseudo_dir"), config.get("pseudo_map"))
    pseudo_dir = config.get("pseudo_dir")
    (bundle / "pseudo").mkdir()
    for fname in dict.fromkeys(pseudos.values()):
        src = Path(pseudo_dir).expanduser() / fname if pseudo_dir else Path(fname)
        if not src.exists():
            raise SystemExit(
                f"{case_name}: pseudopotential not found locally: {src}. "
                f"Set pseudo_dir in config.yaml, or give {machine} its own "
                "'pseudo_dir' in remotes.json."
            )
        shutil.copy2(src, bundle / "pseudo" / fname)
    return ["--pseudo-dir", f"inputs/{case_name}/pseudo"]


def _run_detached_offload(case_dir, case_info, args, config, remote, calc_type, kpoints_spec,
                          mode, project_name, engine="vasp"):
    """Offload a full calculation to the remote engine (run_mode: ssh_detached).

    Prepares an inputs bundle (a single-case POSCAR, or a TSS/NEB case's
    initial/final endpoints, plus a pre-built POTCAR and any user INCAR/KPOINTS —
    or, for QE, the UPF pseudopotentials and any user pw.in) locally, ships it,
    and launches the remote vasp_auto detached so the local host can be powered
    off. Returns (case_info, [row]).

    Fire-and-forget by design: submission takes seconds no matter how long VASP
    runs, and we return as soon as every case is launched — so a whole project
    folder gets submitted even if the local host then sleeps or the UI restarts.
    ponytail: --parallel does NOT block the local driver to cap remote
    concurrency (that abandoned the rest of the folder whenever the driver died
    mid-run); cap on the remote's own scheduler if you need it."""
    import tempfile
    from vasp_auto.potcar_finder import build_potcar
    from vasp_auto.runner import submit_job_detached

    case_dir = Path(case_dir)
    case_name = case_info["case_name"]
    machine = remote.get("name") or remote.get("host")

    is_tss = case_info["calculation_type"] == "tss"
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / case_name
        bundle.mkdir(parents=True)
        if is_tss:
            # NEB/TSS: ship the endpoints (initial/final POSCAR) or, for an
            # already-expanded case, every image POSCAR. The remote engine
            # interpolates the images, builds the NEB INCAR, and runs them all.
            potcar_src = _bundle_tss_inputs(case_dir, bundle, case_name)
        else:
            # POSCAR is required; INCAR/KPOINTS/pw.in are forwarded if the user
            # supplied them.
            poscar = case_dir / "POSCAR"
            if not poscar.exists():
                raise SystemExit(f"{case_name}: offload (ssh_detached) needs a POSCAR.")
            shutil.copy2(poscar, bundle / "POSCAR")
            # workflow.yaml is forwarded so a chained/converge workflow offloads too
            # (the remote engine re-reads it via load_workflow_spec).
            optionals = ("pw.in",) if engine == "qe" else ("INCAR", "KPOINTS", "workflow.yaml")
            for optional in optionals:
                if (case_dir / optional).exists():
                    shutil.copy2(case_dir / optional, bundle / optional)
            potcar_src = bundle / "POSCAR"

        flags = _forward_calc_flags(args, engine=engine)
        if engine == "qe":
            # Pseudopotentials: the machine's own UPF library (remote 'pseudo_dir',
            # written into its config.yaml by submit_job_detached), or ship the
            # local UPFs inside the bundle.
            if (remote.get("pseudo_dir") or "").strip():
                print(f"Pseudos   : from {machine}'s own library (not shipped)")
            else:
                flags += _bundle_qe_pseudos(bundle, case_name, config, machine)
        # POTCAR: if the machine has its own library (remote 'potcar_root'), let the
        # remote engine build it there so the local host needs no POTCAR library.
        # Otherwise pre-build it here and ship it (the remote then just copies it).
        elif (remote.get("potcar_root") or "").strip():
            print(f"POTCAR    : built on {machine} from its library (not shipped)")
        else:
            build_potcar(
                poscar_path=str(potcar_src),
                potcar_root=config.get("potcar_root"),
                output_path=str(bundle / "POTCAR"),
                potcar_map=config.get("potcar_map"),
            )

        print(f"Offload   : {case_name} -> {machine} (detached; local host can power off)")
        submission = submit_job_detached(
            case_dir=str(bundle),
            remote=remote,
            case_name=case_name,
            cpus=args.cpus,
            calc_flags=flags,
            local_job_dir=case_info["job_dir"],
            engine=engine,
        )

    print(f"Launched  : {machine} pid {submission['pid'] or '?'} -> {submission['remote_dir']}")
    print("Fetch later from the Results tab (⬇) or with --parse-only after copying back.")
    row = build_row(project_name, mode, case_info, status_override="remote")
    row["machine"] = machine
    row["remote_dir"] = submission["remote_dir"]
    if submission.get("pid"):
        row["job_id"] = submission["pid"]
    return case_info, [row]


def _process_case(case_dir, args, base_config, mode, project_name, output_root, neb_images, calc_type, kpoints_spec):
    """Handle one case end to end; returns (case_info, rows)."""
    config = merge_local_config(base_config, case_dir)
    engine, config = resolve_engine(args, config)
    remote = resolve_remote(args, config)
    # Numbered job folders (0001_Fe, 0002_Si …) so a re-run never overwrites an
    # earlier one. --retry-failed continues the latest existing job; --dry-run
    # only predicts the name; a normal run/prepare claims the next number.
    if args.retry_failed or args.resume:
        job_mode = "latest"
    elif args.dry_run:
        job_mode = "preview"
    else:
        job_mode = "new"
    case_info = make_case_info(case_dir, output_root, single_mode=(mode == "single"),
                               job_mode=job_mode)
    case_info["project"] = project_name
    # When the detached-offload engine runs us, it points VASP_AUTO_JOBDIR_FILE at a
    # file in its control dir; record the real (numbered) job root there so the
    # submitting host can resolve where the job actually lives on this machine.
    jobdir_file = os.environ.get("VASP_AUTO_JOBDIR_FILE")
    if jobdir_file:
        try:
            Path(jobdir_file).write_text(f"{case_info['job_dir']}\n", encoding="utf-8")
        except OSError:
            pass
    print(f"Case      : {case_dir.name}")
    print(f"Type      : {case_info['calculation_type']}")

    is_tss = case_info["calculation_type"] == "tss"
    converge_requested = args.converge_scf or args.converge_encut or args.converge_sigma

    # Resume runs the latest existing job directory in place from its newest
    # CONTCAR, reusing that directory's own INCAR/KPOINTS/POTCAR — it never
    # rebuilds inputs from the case dir or allocates a new job number.
    if args.resume:
        if remote:
            raise SystemExit(
                "--resume runs in place on a local job directory; remote resume is not supported."
            )
        from vasp_auto.workflow import resume_job
        row = resume_job(
            case_info["job_dir"],
            vasp_executable=config.get("vasp_executable"),
            cpus=args.cpus,
            project_name=project_name,
            case_name=case_info["case_name"],
            calculation_type=case_info["calculation_type"],
        )
        return case_info, [row]

    if engine == "ase":
        # Generic ASE calculators intentionally retain their narrower scope.
        if is_tss:
            raise ValueError(
                f"{case_dir.name}: TSS/NEB is not supported by the {engine} engine yet; use --engine vasp."
            )
        if converge_requested:
            raise ValueError(f"Convergence scans are VASP-only; not yet available for --engine {engine}.")
        if load_workflow_spec(case_dir, config, args.workflow):
            raise ValueError(f"Chained workflows are VASP-only; not yet available for --engine {engine}.")
        if args.solvation:
            raise ValueError(f"--solvation (VASPsol) is not available for --engine {engine}.")
    if engine == "qe" and args.solvation:
        raise ValueError(
            "--solvation selects VASPsol and is not portable to QE. Configure a QE Environ "
            "input explicitly in pw.in instead."
        )

    magmom_map = parse_magmom_map(args.magmom) if args.magmom else config.get("magmom_map")
    if args.spin:
        config = {**config, "spin": True, "magmom_map": magmom_map}

    if calc_type == CalcType.NEB and not is_tss:
        raise ValueError(
            f"{case_dir.name}: --calc-type neb needs a TSS case (initial/POSCAR and final/POSCAR)"
        )

    # Detached offload: the whole calculation (incl. convergence/workflow) runs on
    # the remote engine and we return immediately, so the local host can power off.
    if remote and not args.dry_run and not args.prepare and resolve_remote_run_mode(remote) == "ssh_detached":
        if engine == "ase":
            raise ValueError("Offload (ssh_detached) supports VASP and QE, not --engine ase.")
        return _run_detached_offload(
            case_dir, case_info, args, config, remote, calc_type, kpoints_spec, mode, project_name,
            engine=engine,
        )

    workflow_steps = None
    if not is_tss:
        workflow_steps = load_workflow_spec(case_dir, config, args.workflow)

    if workflow_steps:
        if converge_requested:
            raise ValueError("Use either a workflow or a convergence scan, not both.")
        if args.dry_run:
            names = ", ".join(str(step["calc_type"]) for step in workflow_steps)
            print(f"Workflow  : {names} (dry run, nothing written)")
            row = build_row(project_name, mode, case_info, status_override="dry-run")
            return case_info, [row]
        print(f"Workflow  : {', '.join(str(step['calc_type']) for step in workflow_steps)}")
        if engine == "qe":
            from vasp_auto.qe_chain import run_qe_workflow_case
            rows = run_qe_workflow_case(
                case_info, workflow_steps, config, cpus=args.cpus,
                prepare_only=args.prepare, remote=remote,
            )
        else:
            rows = run_workflow_case(
                case_info,
                workflow_steps,
                config,
                cpus=args.cpus,
                prepare_only=args.prepare,
                remote=remote,
            )
        for row in rows:
            row["project"] = project_name
        return case_info, rows

    if converge_requested and is_tss:
        print(f"Skip      : {case_dir.name} is TSS/NEB; convergence scans support SCF cases only")
        row = build_row(project_name, mode, case_info, status_override="skipped")
        return case_info, [row]

    if args.retry_failed and not should_retry_failed(case_info):
        print(f"Skip      : {case_dir.name} already converged")
        return case_info, [parse_existing_job(project_name, mode, case_info)]

    if args.dry_run:
        preview = preview_job_from_case(
            case_info,
            potcar_root=config.get("potcar_root"),
            potcar_map=config.get("potcar_map"),
            calc_type=str(calc_type) if calc_type else None,
            kpoints_spec=kpoints_spec,
            neb_images=neb_images,
            spin=args.spin,
            magmom_map=magmom_map,
            engine=engine,
            config=config,
        )
        _print_preview(preview)
        row = build_row(project_name, mode, case_info, status_override="dry-run")
        return case_info, [row]

    restart_stash = None
    if args.retry_failed and not is_tss:
        restart_stash = _stash_restart_files(case_info["job_dir"])

    solvation = args.solvation or bool(config.get("solvation"))
    create_job_from_case(
        case_info=case_info,
        potcar_root=config.get("potcar_root"),
        clean_job=True,
        neb_images=neb_images,
        use_ase_neb=args.ase_neb,
        ase_neb_method=args.ase_neb_method,
        potcar_map=config.get("potcar_map"),
        calc_type=str(calc_type) if calc_type else None,
        kpoints_spec=kpoints_spec,
        spin=args.spin,
        magmom_map=magmom_map,
        solvation=solvation,
        solvation_eps=args.solvation_eps,
        engine=engine,
        config=config,
    )
    _restore_restart_files(Path(case_info["job_dir"]), restart_stash)

    if args.prepare:
        row = build_row(project_name, mode, case_info, status_override="prepared")
        print(f"Prepared  : {case_dir.name}")
        return case_info, [row]

    if converge_requested:
        encut_values = parse_encut_values(args.converge_encut) if args.converge_encut else None
        sigma_values = parse_sigma_values(args.converge_sigma) if args.converge_sigma else None
        if engine == "qe":
            from vasp_auto.qe_convergence import converge_qe_case
            result = converge_qe_case(
                case_name=case_info["case_name"], base_job_dir=case_info["job_dir"],
                qe_executable=config.get("qe_executable", "pw.x"), config=config,
                cpus=args.cpus, kpoint_meshes=parse_kpoint_meshes(args.kpoints_values),
                encut_values=encut_values, sigma_values=sigma_values,
                energy_tolerance=args.energy_tol, sigma_tolerance=args.sigma_tol,
                scan_kpoints=args.converge_scf, remote=remote,
            )
        else:
            result = converge_scf_case(
                case_name=case_info["case_name"],
                base_job_dir=case_info["job_dir"],
                vasp_executable=config["vasp_executable"],
                cpus=args.cpus,
                nelm_values=parse_nelm_values(args.nelm_values),
                kpoint_meshes=parse_kpoint_meshes(args.kpoints_values),
                encut_values=encut_values,
                sigma_values=sigma_values,
                energy_tolerance=args.energy_tol,
                sigma_tolerance=args.sigma_tol,
                scan_nelm=args.converge_scf,
                scan_kpoints=args.converge_scf,
                reuse_wavecar=args.reuse_wavecar,
                remote=remote,
            )
        row = build_row(project_name, mode, case_info, status_override="converged-scan")
        row.update(
            {
                "selected_encut": result["selected_encut"],
                "selected_sigma": result["selected_sigma"],
                "selected_nelm": result["selected_nelm"],
                "selected_kpoints": result["selected_kpoints"],
                "energy_eV": result["selected_energy_eV"],
                "convergence_report": result["report_path"],
                "convergence_csv": result["csv_path"],
            }
        )
        if remote:
            row["machine"] = remote.get("name") or remote.get("host")
        print(
            f"Selected  : {'ecutwfc (Ry)' if engine == 'qe' else 'ENCUT'}={result['selected_encut']}, "
            f"{'degauss (Ry)' if engine == 'qe' else 'SIGMA'}={result['selected_sigma']}, "
            f"NELM={result['selected_nelm']}, KPOINTS={result['selected_kpoints']}"
        )
        print(f"Report    : {result['report_path']}")
        return case_info, [row]

    scheduler = args.scheduler or config.get("scheduler", "local")
    row = run_one_case(
        project_name=project_name,
        mode=mode,
        case_info=case_info,
        vasp_executable=config.get("vasp_executable"),
        cpus=args.cpus,
        scheduler=scheduler,
        job_template=config.get("job_template"),
        scheduler_options=config.get("scheduler_options"),
        auto_retry=args.auto_retry,
        remote=remote,
        engine=engine,
        qe_executable=config.get("qe_executable", "pw.x"),
        ase_python=config.get("ase_python"),
    )
    if not remote:
        print(f"Finished  : {case_dir.name}")
    return case_info, [row]


def _run_init(args) -> bool:
    """--init: create config.yaml from config.yaml.example, then exit.

    Gives a first-time user a working starting point instead of relying on the
    bare defaults (which point at ``vasp_std`` on PATH and ``./POTCAR`` and usually
    fail with a cryptic error). Writes ``config.yaml`` next to the example at the
    repository root — where :func:`config_loader.find_config_file` already looks —
    and never overwrites an existing one.
    """
    if not getattr(args, "init", False):
        return False
    repo_root = Path(__file__).resolve().parents[2]
    example = repo_root / "config.yaml.example"
    target = repo_root / "config.yaml"
    if target.exists():
        print(f"config.yaml already exists: {target}")
        print("Edit it directly, or delete it first to regenerate from the example.")
        return True
    if not example.exists():
        raise SystemExit(f"config.yaml.example not found next to the package ({example}).")
    shutil.copy2(example, target)
    print(f"Created   : {target}")
    print("Next steps:")
    print("  1. Edit config.yaml — set vasp_executable, potcar_root and jobs_root.")
    print("  2. (optional) cp remotes.json.example remotes.json, or add machines in the UI.")
    print("  3. Try it:  vasp-auto example/Si --prepare")
    return True


def _list_options(args) -> bool:
    """--list-options: print the INCAR catalog (labels, choices, defaults), then exit."""
    if not getattr(args, "list_options", False):
        return False
    from .incar_catalog import fields_by_group
    for group, fields in fields_by_group().items():
        print(f"\n{group}")
        for f in fields:
            unit = f" ({f.unit})" if f.unit else ""
            default = "" if f.default is None else f"  [default: {f.default}]"
            only = f"  (only: {', '.join(sorted(f.applies_to))})" if f.applies_to else ""
            print(f"  {f.label}{unit} -> {f.tag}{default}{only}")
            for o in f.options:
                hint = f"  # {o.hint}" if o.hint else ""
                print(f"      {o.label} = {o.value}{hint}")
    return True


def _warn_if_unconfigured(config, engine):
    """Nudge a first-time user who has no config.yaml toward ``vasp-auto --init``.

    With no config file the VASP engine falls back to bare defaults (``vasp_std``
    on PATH, ``./POTCAR``), which usually fail with a confusing error. Print a
    one-line note to stderr that points at the fix without blocking the run.
    """
    if config.get("_config_path") or config.get("_local_config") or engine != "vasp":
        return
    print(
        "Note      : no config.yaml found — using defaults "
        f"(vasp_executable={config.get('vasp_executable')!r}, "
        f"potcar_root={config.get('potcar_root')!r}).\n"
        "            Run `vasp-auto --init` to create one, then set your VASP "
        "binary and POTCAR library.",
        file=sys.stderr,
    )


def main():
    args = parse_args()
    if args.background:
        launch_background()
        return

    if _run_init(args):
        return

    if _list_options(args):
        return

    config = load_config()

    if _poll_job(args, config):
        return

    if _list_jobs(args, config):
        return

    if _kill_job(args, config):
        return

    if _run_resume(args, config):
        return

    if _apply_remote_setup(args, config):
        return

    if _run_analysis(args, config):
        return

    if _run_match_cells(args):
        return

    if _run_builders(args):
        return

    if _apply_ml_energy(args, config):
        return

    if _apply_db_prerelax(args, config):
        return

    if _apply_db_fetch(args, config):
        return

    if _apply_ml_relax(args, config):
        return

    target_path = Path(args.target).expanduser().resolve()
    config = merge_local_config(config, target_path)
    job_root = Path(config["jobs_root"]).resolve()
    job_root.mkdir(parents=True, exist_ok=True)

    info = inspect_target(target_path)
    mode = info["mode"]
    project_name = info["project_name"]
    case_dirs = filter_case_dirs(info["case_dirs"], args.cases)
    neb_images = args.neb_images or int(config.get("neb_images", 5))
    calc_type = parse_calc_type(args.calc_type) if args.calc_type else None
    kpoints_spec = _kpoints_spec_from_args(args)
    scheduler = args.scheduler or config.get("scheduler", "local")

    print(f"Mode      : {mode}")
    print(f"Project   : {project_name}")
    print(f"Target    : {target_path}")
    print(f"Jobs root : {job_root}")
    engine = args.engine or config.get("engine", "vasp")
    if engine == "qe":
        print(f"Engine    : Quantum ESPRESSO ({config.get('qe_executable', 'pw.x')})")
    elif engine == "ase":
        print(f"Engine    : ASE calculator ({config.get('ase_calculator', 'emt')})")
    if not args.parse_only:
        _warn_if_unconfigured(config, engine)
    print(f"CPU       : {args.cpus if args.cpus else 'default'}")
    if "tss" in info["case_types"].values():
        print(f"NEB images: {neb_images}")
    if calc_type:
        print(f"Calc type : {calc_type}")
    if kpoints_spec:
        print(f"KPOINTS   : {kpoints_spec['mode']}")
    remote_cfg = resolve_remote(args, config)
    if remote_cfg:
        print(f"Remote    : {remote_cfg.get('name') or remote_cfg.get('host', '?')} "
              f"({remote_cfg.get('host', '?')}, {remote_cfg.get('scheduler', 'slurm')})")
    elif scheduler != "local":
        print(f"Scheduler : {scheduler}")
    if args.parallel > 1:
        print(f"Parallel  : {args.parallel} cases")
    if args.ase_neb:
        print(f"ASE NEB   : {args.ase_neb_method}")
    if args.spin:
        print(f"Spin      : ISPIN=2 (initial moments: {args.magmom or 'defaults'})")
    if args.auto_retry:
        print(f"Auto-retry: up to {args.auto_retry} per case")
    if args.converge_scf or args.converge_encut or args.converge_sigma:
        print(f"Conv. scan: enabled (tol {args.energy_tol} eV)")
    print()

    # Every job lands directly under the jobs root as a numbered folder
    # (jobs/0001_Fe, jobs/0002_Si …) — no per-project sub-folder. One machine
    # keeps one global number list, regardless of project or calculation engine.
    output_root = job_root

    if args.parse_only:
        case_infos = []
        all_results = []
        for case_dir in case_dirs:
            case_info = make_case_info(case_dir, output_root, single_mode=(mode == "single"),
                                       job_mode="latest")
            case_infos.append(case_info)
            all_results.append(parse_existing_job(project_name, mode, case_info))

        if args.report:
            _write_reports(all_results)
        excel_path = summary_excel_path(job_root, output_root, project_name, mode, case_infos)
        _write_excel_summary(excel_path, all_results)
        return

    def process(case_dir):
        return _process_case(
            case_dir,
            args=args,
            base_config=config,
            mode=mode,
            project_name=project_name,
            output_root=output_root,
            neb_images=neb_images,
            calc_type=calc_type,
            kpoints_spec=kpoints_spec,
        )

    if args.parallel > 1 and len(case_dirs) > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results = list(pool.map(process, case_dirs))
    else:
        results = []
        for index, case_dir in enumerate(case_dirs):
            if index:
                print()
            results.append(process(case_dir))

    case_infos = [case_info for case_info, _ in results]
    all_results = [row for _, rows in results for row in rows]

    if args.dry_run:
        print()
        print("Dry run   : no job files or Excel summary written")
        return

    if args.report:
        _write_reports(all_results)

    excel_path = summary_excel_path(job_root, output_root, project_name, mode, case_infos)
    print()
    _write_excel_summary(excel_path, all_results)


if __name__ == "__main__":
    main()
