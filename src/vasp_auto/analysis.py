"""Catalysis post-processing: adsorption energies, vibrational thermochemistry,
d-band centers, work functions, optical absorption, simulated XRD patterns.

Operates on finished job directories; runs no VASP itself. Pure Python.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from vasp_auto.chgcar import planar_average, read_volumetric
from vasp_auto.parser import parse_dielectric, parse_pdos, parse_vasprun

BOLTZMANN_EV_K = 8.617333262e-5  # eV/K
HBARC_EV_CM = 1.973269804e-5     # ħc in eV·cm
DEFAULT_TEMPERATURE_K = 298.15


# --- energies from finished jobs --------------------------------------------

def read_job_energy(job_dir: Path) -> dict:
    """Final energy and convergence flag of a finished job directory.

    QE jobs (``.engine`` = qe, or a pw.out present) read pw.out; VASP jobs
    prefer OUTCAR (free energy TOTEN) and fall back to vasprun.xml. Raises
    when the directory holds no readable energy, naming the directory.
    """
    from vasp_auto.workflow import job_engine, parse_outcar_summary

    job_dir = Path(job_dir)

    if job_engine(job_dir) == "qe":
        from vasp_auto.parser import parse_pw_output

        summary = parse_pw_output(job_dir / "pw.out")
        if summary and summary.get("energy_eV") is not None:
            return {"energy_eV": summary["energy_eV"], "converged": summary["converged"]}
        raise FileNotFoundError(f"No final energy found (pw.out) in {job_dir}")

    outcar = job_dir / "OUTCAR"
    if outcar.exists():
        summary = parse_outcar_summary(outcar)
        if summary["energy_eV"] is not None:
            return {"energy_eV": summary["energy_eV"], "converged": summary["converged"]}

    vasprun = parse_vasprun(job_dir / "vasprun.xml")
    if vasprun and vasprun.get("energy_eV") is not None:
        return {"energy_eV": vasprun["energy_eV"], "converged": True}

    raise FileNotFoundError(f"No final energy found (OUTCAR/vasprun.xml) in {job_dir}")


def adsorption_energy(
    total_dir: Path,
    slab_dir: Path,
    molecule_dir: Path,
    molecule_scale: float = 1.0,
) -> dict:
    """E_ads = E(slab+adsorbate) − E(slab) − scale·E(reference molecule).

    `molecule_scale` references a fraction of the gas-phase molecule, e.g.
    0.5 with an H2 box for atomic H adsorption. Negative E_ads = exothermic.
    """
    total = read_job_energy(total_dir)
    slab = read_job_energy(slab_dir)
    molecule = read_job_energy(molecule_dir)

    e_ads = total["energy_eV"] - slab["energy_eV"] - molecule_scale * molecule["energy_eV"]
    return {
        "adsorption_energy_eV": e_ads,
        "total_energy_eV": total["energy_eV"],
        "slab_energy_eV": slab["energy_eV"],
        "molecule_energy_eV": molecule["energy_eV"],
        "molecule_scale": molecule_scale,
        "all_converged": total["converged"] and slab["converged"] and molecule["converged"],
    }


EV_A2_TO_J_M2 = 16.021766208  # 1 eV/Å² = 16.0218 J/m²


def _cross_area(lattice: list[list[float]], axis: int) -> float:
    """Area (Å²) of the cell face perpendicular to `axis` = |a⃗ × b⃗| of the
    two other lattice vectors."""
    u = lattice[(axis + 1) % 3]
    v = lattice[(axis + 2) % 3]
    cx = u[1] * v[2] - u[2] * v[1]
    cy = u[2] * v[0] - u[0] * v[2]
    cz = u[0] * v[1] - u[1] * v[0]
    return math.sqrt(cx * cx + cy * cy + cz * cz)


def surface_energy(slab_dir: Path, bulk_dir: Path, axis: int = 2) -> dict:
    """Surface energy γ = (E_slab − (N_slab/N_bulk)·E_bulk) / (2A).

    E from `read_job_energy` (VASP or QE). A is the slab cell face ⟂ `axis`
    (0=a,1=b,2=c; default c ⇒ area of the a,b plane). N are atom counts from
    each POSCAR. Two surfaces ⇒ the factor 2. Returns eV/Å² and J/m².
    """
    from vasp_auto.structure import read_poscar

    slab_dir, bulk_dir = Path(slab_dir), Path(bulk_dir)
    e_slab = read_job_energy(slab_dir)
    e_bulk = read_job_energy(bulk_dir)

    slab_struct = read_poscar(slab_dir / "POSCAR")
    bulk_struct = read_poscar(bulk_dir / "POSCAR")
    n_slab = sum(slab_struct["counts"])
    n_bulk = sum(bulk_struct["counts"])
    if n_bulk == 0:
        raise ValueError(f"Bulk POSCAR in {bulk_dir} has no atoms")

    scale = slab_struct["scale"]
    if scale <= 0:
        scale = 1.0  # negative-scale (target-volume) POSCARs are unusual for slabs
    lattice = [[x * scale for x in row] for row in slab_struct["lattice"]]
    area = _cross_area(lattice, axis)

    gamma = (e_slab["energy_eV"] - (n_slab / n_bulk) * e_bulk["energy_eV"]) / (2.0 * area)
    return {
        "surface_energy_eV_A2": gamma,
        "surface_energy_J_m2": gamma * EV_A2_TO_J_M2,
        "slab_energy_eV": e_slab["energy_eV"],
        "bulk_energy_eV": e_bulk["energy_eV"],
        "n_slab": n_slab,
        "n_bulk": n_bulk,
        "area_A2": area,
        "axis": axis,
        "all_converged": e_slab["converged"] and e_bulk["converged"],
    }


def free_energy_diagram(
    steps: list[dict],
    temperature: float = DEFAULT_TEMPERATURE_K,
    potential: float = 0.0,
    u_equilibrium: float = 1.23,
) -> dict:
    """Cumulative free-energy diagram + overpotential (computational H electrode).

    `steps` is an ordered list; each step is a *reaction step* whose ΔG is:

        ΔG = ΔE  + Δ(g_correction)  − n·e·U

    where per step:
      - ΔE is either an explicit ``delta_e`` (eV) or, when ``ads`` =
        ``{total, slab, molecule, scale?}`` job dirs is given, the adsorption
        energy of that step via `adsorption_energy`.
      - Δ(g_correction) comes from `thermo_from_job(freq_job, T)` when a
        ``freq_job`` dir is given (ZPE + U_vib − T·S), else 0.
      - ``n_electrons`` (default 0) couples the step to the potential U
        (proton-coupled electron transfer): each transferred electron lowers
        ΔG by e·U.

    Returns the ΔG per step and the cumulative staircase at U=0 and at U, the
    potential-determining step, the limiting potential U_L = −max(ΔG_i at U=0)/e
    and the overpotential η = |U_equilibrium − U_L| (sign per the reaction the
    caller supplies; 1.23 V is the OER/ORR default).
    """
    per_step = []
    for i, step in enumerate(steps):
        if "delta_e" in step and step["delta_e"] is not None:
            delta_e = float(step["delta_e"])
        elif step.get("ads"):
            ads = step["ads"]
            delta_e = adsorption_energy(
                Path(ads["total"]), Path(ads["slab"]), Path(ads["molecule"]),
                molecule_scale=float(ads.get("scale") or 1.0),
            )["adsorption_energy_eV"]
        else:
            raise ValueError(f"Step {i} ({step.get('label')}) needs delta_e or ads job dirs")

        g_corr = 0.0
        if step.get("freq_job"):
            g_corr = thermo_from_job(Path(step["freq_job"]), temperature)["g_correction_eV"]

        n_e = float(step.get("n_electrons") or 0.0)
        dg0 = delta_e + g_corr            # ΔG at U = 0
        dg = dg0 - n_e * potential        # ΔG at the applied potential
        per_step.append({
            "label": step.get("label", f"step {i + 1}"),
            "delta_e_eV": delta_e,
            "g_correction_eV": g_corr,
            "n_electrons": n_e,
            "delta_g0_eV": dg0,
            "delta_g_eV": dg,
        })

    def _cumulative(key):
        cum, total = [0.0], 0.0
        for s in per_step:
            total += s[key]
            cum.append(total)
        return cum

    dg0s = [s["delta_g0_eV"] for s in per_step]
    pds = max(range(len(per_step)), key=lambda i: dg0s[i]) if per_step else None
    max_dg0 = max(dg0s) if per_step else 0.0
    u_limiting = -max_dg0  # per electron; positive ΔG uphill ⇒ negative U_L
    return {
        "temperature_K": temperature,
        "potential_V": potential,
        "u_equilibrium_V": u_equilibrium,
        "steps": per_step,
        "cumulative_g0_eV": _cumulative("delta_g0_eV"),
        "cumulative_g_eV": _cumulative("delta_g_eV"),
        "potential_determining_step": per_step[pds]["label"] if pds is not None else None,
        "max_delta_g0_eV": max_dg0,
        "limiting_potential_V": u_limiting,
        "overpotential_V": abs(u_equilibrium - u_limiting),
    }


# --- vibrational frequencies and thermochemistry ----------------------------

def parse_frequencies(outcar_path: Path) -> list[dict]:
    """Vibrational modes from an IBRION=5/6/7/8 OUTCAR.

    Returns [{"index", "meV", "cm1", "THz", "imaginary"}], imaginary modes
    flagged (VASP prints them as 'f/i='). Empty when no frequency block exists.
    """
    outcar_path = Path(outcar_path)
    modes = []
    if not outcar_path.exists():
        return modes

    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # e.g. "   4 f  =   91.546624 THz   575.204660 2PiTHz 3053.668884 cm-1   378.617346 meV"
            #      "  12 f/i=    0.022552 THz     0.141698 2PiTHz    0.752259 cm-1     0.093268 meV"
            if "freq (" in line and "[THz]" in line and "[cm-1]" in line:
                # dynmat.x: freq ( 1) = -2.46 [THz] = -82.10 [cm-1]
                try:
                    index = int(line.split("(", 1)[1].split(")", 1)[0])
                    thz = float(line.split("=", 1)[1].split("[THz]", 1)[0])
                    cm1 = float(line.split("=", 2)[2].split("[cm-1]", 1)[0])
                except (IndexError, ValueError):
                    continue
                modes.append({"index": index, "THz": thz, "cm1": cm1,
                              "meV": abs(cm1) * 0.1239841984, "imaginary": cm1 < 0.0})
                continue
            if "2PiTHz" not in line or "meV" not in line:
                continue
            imaginary = "f/i" in line
            tokens = line.replace("f/i=", " ").replace("f", " ").replace("=", " ").split()
            try:
                index = int(tokens[0])
                thz = float(tokens[tokens.index("THz") - 1])
                cm1 = float(tokens[tokens.index("cm-1") - 1])
                mev = float(tokens[tokens.index("meV") - 1])
            except (IndexError, ValueError):
                continue
            modes.append(
                {"index": index, "THz": thz, "cm1": cm1, "meV": mev, "imaginary": imaginary}
            )
    return modes


def harmonic_thermochemistry(
    modes: list[dict],
    temperature: float = DEFAULT_TEMPERATURE_K,
) -> dict:
    """ZPE and harmonic-oscillator thermal corrections from vibrational modes.

    Imaginary modes are excluded (their count is reported — more than zero on
    a supposed minimum means the geometry is not fully relaxed). Returns all
    terms in eV: zpe, u_vib (thermal vibrational energy), ts (T·S_vib), and
    g_correction = ZPE + U_vib − T·S_vib, the term added to the electronic
    energy in computational-hydrogen-electrode free-energy diagrams.
    """
    real_mev = [mode["meV"] for mode in modes if not mode["imaginary"]]
    kT = BOLTZMANN_EV_K * temperature

    zpe = sum(0.001 * mev / 2.0 for mev in real_mev)
    u_vib = 0.0
    entropy = 0.0  # S in eV/K
    for mev in real_mev:
        energy = 0.001 * mev
        x = energy / kT
        occupation = 1.0 / (math.expm1(x))
        u_vib += energy * occupation
        entropy += BOLTZMANN_EV_K * (x * occupation - math.log(-math.expm1(-x)))

    ts = temperature * entropy
    return {
        "temperature_K": temperature,
        "n_modes": len(real_mev),
        "n_imaginary": sum(1 for mode in modes if mode["imaginary"]),
        "zpe_eV": zpe,
        "u_vib_eV": u_vib,
        "ts_eV": ts,
        "g_correction_eV": zpe + u_vib - ts,
    }


def thermo_from_job(job_dir: Path, temperature: float = DEFAULT_TEMPERATURE_K) -> dict:
    """Frequencies + thermochemistry of a finished freq job directory."""
    job_dir = Path(job_dir)
    source = job_dir / "dynmat.out" if (job_dir / "dynmat.out").exists() else job_dir / "OUTCAR"
    modes = parse_frequencies(source)
    if not modes:
        raise ValueError(
            f"No vibrational modes found in {source} — run with --calc-type freq first."
        )
    result = harmonic_thermochemistry(modes, temperature)
    result["modes"] = modes
    try:
        result["energy_eV"] = read_job_energy(job_dir)["energy_eV"]
        result["g_total_eV"] = result["energy_eV"] + result["g_correction_eV"]
    except FileNotFoundError:
        pass
    return result


# --- d-band center -----------------------------------------------------------

def _d_field_indices(fields: list[str]) -> list[int]:
    # LORBIT=11 labels: dxy, dyz, dz2, dxz, x2-y2 (older VASP: dx2); LORBIT=10: d.
    return [
        i for i, name in enumerate(fields)
        if name.lower().startswith("d") or "x2" in name.lower()
    ]


def d_band_center(
    vasprun_path: Path,
    atom_indices: list[int],
    emax_eV: float | None = None,
) -> dict:
    """d-band center (and width) of selected atoms, relative to the Fermi level.

    First moment of the d-projected DOS summed over `atom_indices` (1-based,
    POSCAR order) and both spins. `emax_eV` (relative to E_F) truncates the
    integral, e.g. 0.0 for the occupied d-band only; default integrates the
    whole grid. Needs a DOS run with LORBIT=11.
    """
    pdos = parse_pdos(vasprun_path)
    if pdos is None:
        raise ValueError(f"No projected DOS in {vasprun_path} — run a dos job with LORBIT=11.")

    d_indices = _d_field_indices(pdos["fields"])
    if not d_indices:
        raise ValueError(f"No d orbitals in projected DOS fields: {pdos['fields']}")

    efermi = pdos["efermi"] or 0.0
    energies = [e - efermi for e in pdos["energies"]]

    density = [0.0] * len(energies)
    for atom in atom_indices:
        spins = pdos["pdos"].get(atom)
        if spins is None:
            raise ValueError(f"Atom index {atom} not present in projected DOS (1-based)")
        for channels in spins:
            for field_index in d_indices:
                channel = channels[field_index]
                for i in range(len(density)):
                    density[i] += channel[i]

    # Trapezoidal moments over the (optionally truncated) energy grid.
    norm = 0.0
    first = 0.0
    second = 0.0
    for i in range(1, len(energies)):
        if emax_eV is not None and energies[i] > emax_eV:
            break
        de = energies[i] - energies[i - 1]
        rho = 0.5 * (density[i] + density[i - 1])
        e_mid = 0.5 * (energies[i] + energies[i - 1])
        norm += rho * de
        first += e_mid * rho * de
        second += e_mid * e_mid * rho * de

    if norm <= 0.0:
        raise ValueError("d-projected DOS integrates to zero over the selected window")

    center = first / norm
    width = math.sqrt(max(second / norm - center * center, 0.0))
    return {
        "d_band_center_eV": center,
        "d_band_width_eV": width,
        "atoms": list(atom_indices),
        "efermi_eV": efermi,
        "n_electrons_d": norm,
    }


def qe_d_band_center(
    job_dir: Path,
    atom_indices: list[int],
    emax_eV: float | None = None,
) -> dict:
    """d-band moments from the atomic d projections written by projwfc.x."""
    job_dir = Path(job_dir)
    selected = set(atom_indices)
    files = []
    for path in job_dir.glob("*.pdos_atm#*_wfc#*(d)"):
        match = re.search(r"pdos_atm#(\d+)\(", path.name)
        if match and int(match.group(1)) in selected:
            files.append(path)
    if not files:
        raise ValueError(
            f"No selected d-projected QE DOS files in {job_dir}; run a dos job with projwfc.x."
        )

    energy_grid = None
    density = None
    for path in sorted(files):
        rows = []
        header = ""
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lstrip().startswith("#"):
                header += " " + line.lower()
                continue
            try:
                values = [float(value) for value in line.split()]
            except ValueError:
                continue
            if len(values) >= 2:
                rows.append(values)
        if not rows:
            continue
        energies = [row[0] for row in rows]
        if energy_grid is None:
            energy_grid = energies
            density = [0.0] * len(energies)
        elif len(energies) != len(energy_grid) or any(
            abs(a - b) > 1e-7 for a, b in zip(energies, energy_grid)
        ):
            raise ValueError(f"QE projected-DOS energy grids differ in {job_dir}")

        # projwfc.x writes E, LDOS for non-spin and E, LDOSup, LDOSdw for
        # spin-polarized calculations. The remaining columns resolve m channels.
        spin_columns = 2 if "ldosdw" in header or "ldosdown" in header else 1
        for index, row in enumerate(rows):
            density[index] += sum(row[1:1 + spin_columns])

    if not energy_grid or density is None:
        raise ValueError(f"No readable d-projected QE DOS data in {job_dir}")
    efermi = _fermi_from_pw(job_dir / "pw.out") or 0.0
    energies = [energy - efermi for energy in energy_grid]
    norm = first = second = 0.0
    for index in range(1, len(energies)):
        if emax_eV is not None and energies[index] > emax_eV:
            break
        de = energies[index] - energies[index - 1]
        rho = 0.5 * (density[index] + density[index - 1])
        e_mid = 0.5 * (energies[index] + energies[index - 1])
        norm += rho * de
        first += e_mid * rho * de
        second += e_mid * e_mid * rho * de
    if norm <= 0.0:
        raise ValueError("QE d-projected DOS integrates to zero over the selected window")
    center = first / norm
    return {
        "d_band_center_eV": center,
        "d_band_width_eV": math.sqrt(max(second / norm - center * center, 0.0)),
        "atoms": list(atom_indices),
        "efermi_eV": efermi,
        "n_electrons_d": norm,
    }


# --- work function ------------------------------------------------------------

def _fermi_from_outcar(outcar_path: Path) -> float | None:
    if not Path(outcar_path).exists():
        return None
    fermi = None
    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "E-fermi" in line:
                try:
                    fermi = float(line.split(":")[1].split()[0])
                except (IndexError, ValueError):
                    pass
    return fermi


def _fermi_from_pw(pw_out: Path) -> float | None:
    fermi = None
    if not Path(pw_out).exists():
        return None
    for line in Path(pw_out).read_text(encoding="utf-8", errors="ignore").splitlines():
        if "fermi energy is" in line.lower():
            try:
                fermi = float(line.lower().split("fermi energy is", 1)[1].split()[0])
            except (IndexError, ValueError):
                pass
    return fermi


def work_function(job_dir: Path, axis: int = 2) -> dict:
    """Work function W = V_vacuum − E_Fermi from a LOCPOT slab run.

    The vacuum level is the maximum of the planar-averaged potential along
    `axis` (0=a, 1=b, 2=c). With LDIPOL the two vacuum sides of an asymmetric
    slab differ; the higher plateau (the reported maximum) is the work
    function of the surface facing it. Needs LVHAR = .TRUE. (the
    'workfunction' calc type).
    """
    job_dir = Path(job_dir)
    qe_profile = job_dir / "potential-average.dat"
    if qe_profile.exists():
        profile = []
        positions = []
        for line in qe_profile.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                positions.append(float(parts[0]) * 0.529177210903)
                profile.append(float(parts[1]) * 13.605693122994)
            except ValueError:
                continue
        fermi = _fermi_from_pw(job_dir / "pw.out")
        if not profile or fermi is None:
            raise ValueError(f"Incomplete QE work-function output in {job_dir}")
        vacuum = max(profile)
        return {"work_function_eV": vacuum - fermi, "vacuum_level_eV": vacuum,
                "fermi_eV": fermi, "axis": axis, "profile_eV": profile,
                "positions_A": positions}
    locpot = job_dir / "LOCPOT"
    if not locpot.exists():
        raise FileNotFoundError(
            f"No LOCPOT in {job_dir} — run with --calc-type workfunction (LVHAR = .TRUE.)"
        )

    fermi = None
    vasprun = parse_vasprun(job_dir / "vasprun.xml")
    if vasprun:
        fermi = vasprun.get("fermi_eV")
    if fermi is None:
        fermi = _fermi_from_outcar(job_dir / "OUTCAR")
    if fermi is None:
        raise ValueError(f"No Fermi level found (vasprun.xml/OUTCAR) in {job_dir}")

    volume = read_volumetric(locpot)
    profile = planar_average(volume, axis=axis)
    vacuum = max(profile)
    return {
        "work_function_eV": vacuum - fermi,
        "vacuum_level_eV": vacuum,
        "fermi_eV": fermi,
        "axis": axis,
        "profile_eV": profile,
    }


# --- optical absorption --------------------------------------------------------

def absorption_spectrum(vasprun_path: Path) -> dict:
    """Optical absorption coefficient α(E) in cm⁻¹ from a LOPTICS run.

    α = 2 E k / ħc with the extinction coefficient k built from the
    direction-averaged dielectric function. Returns {"energies_eV",
    "alpha_cm1", "real", "imag"}.
    """
    vasprun_path = Path(vasprun_path)
    job_dir = vasprun_path if vasprun_path.is_dir() else vasprun_path.parent
    epsr_path, epsi_path = job_dir / "epsr.dat", job_dir / "epsi.dat"
    if epsr_path.exists() and epsi_path.exists():
        def read_eps(path):
            rows = []
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                try:
                    values = [float(value) for value in line.split()]
                except ValueError:
                    continue
                if len(values) >= 4:
                    rows.append((values[0], sum(values[1:4]) / 3.0))
            return rows
        real_rows, imag_rows = read_eps(epsr_path), read_eps(epsi_path)
        count = min(len(real_rows), len(imag_rows))
        dielectric = {"energies": [real_rows[i][0] for i in range(count)],
                      "real": [real_rows[i][1] for i in range(count)],
                      "imag": [imag_rows[i][1] for i in range(count)]} if count else None
    else:
        dielectric = parse_dielectric(vasprun_path)
    if dielectric is None:
        raise ValueError(
            f"No dielectric function in {vasprun_path} — run with --calc-type optics "
            "(LOPTICS = .TRUE.) first."
        )

    alpha = []
    for energy, e1, e2 in zip(dielectric["energies"], dielectric["real"], dielectric["imag"]):
        modulus = math.sqrt(e1 * e1 + e2 * e2)
        k = math.sqrt(max(modulus - e1, 0.0) / 2.0)
        alpha.append(2.0 * energy * k / HBARC_EV_CM)

    return {
        "energies_eV": dielectric["energies"],
        "alpha_cm1": alpha,
        "real": dielectric["real"],
        "imag": dielectric["imag"],
    }


# --- X-ray diffraction ---------------------------------------------------------

def xrd_pattern(
    path: Path,
    wavelength: str | float = "CuKa",
    two_theta_range: tuple[float, float] = (10.0, 90.0),
) -> dict:
    """Simulated powder XRD pattern of a structure (Bragg-Brentano, pymatgen).

    ``path`` is a structure file (POSCAR/CONTCAR/CIF/…) or a job directory —
    a directory uses its CONTCAR when present, else POSCAR. ``wavelength`` is
    a radiation name ("CuKa", "MoKa", …) or a wavelength in Å. Returns
    {"two_theta_deg", "intensity", "hkl", "d_spacing_A", "wavelength_A",
    "structure_file"} with intensities normalised to 100.
    """
    try:
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ImportError("pymatgen is required for XRD: pip install pymatgen") from exc

    path = Path(path)
    if path.is_dir():
        contcar = path / "CONTCAR"
        path = contcar if contcar.is_file() and contcar.stat().st_size else path / "POSCAR"

    calculator = XRDCalculator(wavelength=wavelength)
    pattern = calculator.get_pattern(
        Structure.from_file(path), two_theta_range=two_theta_range
    )
    return {
        "two_theta_deg": [float(x) for x in pattern.x],
        "intensity": [float(y) for y in pattern.y],
        "hkl": [
            ", ".join("(" + " ".join(str(i) for i in h["hkl"]) + ")" for h in hkls)
            for hkls in pattern.hkls
        ],
        "d_spacing_A": [float(d) for d in pattern.d_hkls],
        "wavelength_A": float(calculator.wavelength),
        "structure_file": path.name,
    }
