"""Tests for analysis.py: adsorption energy, thermochemistry, d-band, work
function, optical absorption."""
import math
import sys

import pytest

from vasp_auto import analysis, cli

OUTCAR_CONVERGED = """ mock
  aborting loop because EDIFF is reached
  free  energy   TOTEN  =      {energy:.8f} eV
"""

FREQ_OUTCAR = """ vasp.6 mock
 Eigenvectors and eigenvalues of the dynamical matrix
   1 f  =   91.546624 THz   575.204660 2PiTHz 3053.668884 cm-1   378.617346 meV
   2 f  =   30.000000 THz   188.495559 2PiTHz 1000.685310 cm-1   124.069204 meV
   3 f/i=    0.022552 THz     0.141698 2PiTHz    0.752259 cm-1     0.093268 meV
  aborting loop because EDIFF is reached
  free  energy   TOTEN  =       -25.00000000 eV
"""

VASPRUN_PDOS = """<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
 <calculation>
  <energy><i name="e_fr_energy"> -10.0 </i></energy>
  <dos>
   <i name="efermi"> 2.0 </i>
   <total><array><set>
    <set comment="spin 1">
     <r> 0.0 1.0 1.0 </r>
     <r> 2.0 1.0 2.0 </r>
    </set>
   </set></array></total>
   <partial><array>
    <field>energy</field>
    <field>s</field>
    <field>dxy</field>
    <field>x2-y2</field>
    <set>
     <set comment="ion 1">
      <set comment="spin 1">
       <r> 0.0 0.1 1.0 1.0 </r>
       <r> 2.0 0.1 1.0 1.0 </r>
       <r> 4.0 0.1 0.0 0.0 </r>
      </set>
     </set>
     <set comment="ion 2">
      <set comment="spin 1">
       <r> 0.0 0.2 0.0 0.0 </r>
       <r> 2.0 0.2 0.0 0.0 </r>
       <r> 4.0 0.2 0.0 0.0 </r>
      </set>
     </set>
    </set>
   </array></partial>
  </dos>
 </calculation>
</modeling>
"""

VASPRUN_DIELECTRIC = """<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
 <calculation>
  <dielectricfunction>
   <imag><array><set>
    <r> 0.0 0.0 0.0 0.0 0.0 0.0 0.0 </r>
    <r> 2.0 3.0 3.0 3.0 0.0 0.0 0.0 </r>
   </set></array></imag>
   <real><array><set>
    <r> 0.0 4.0 4.0 4.0 0.0 0.0 0.0 </r>
    <r> 2.0 1.0 1.0 1.0 0.0 0.0 0.0 </r>
   </set></array></real>
  </dielectricfunction>
 </calculation>
</modeling>
"""


def _make_job(tmp_path, name, energy):
    job = tmp_path / name
    job.mkdir()
    (job / "OUTCAR").write_text(OUTCAR_CONVERGED.format(energy=energy), encoding="utf-8")
    return job


def test_adsorption_energy(tmp_path):
    total = _make_job(tmp_path, "slab_ads", -110.0)
    slab = _make_job(tmp_path, "slab", -100.0)
    mol = _make_job(tmp_path, "h2", -6.0)

    result = analysis.adsorption_energy(total, slab, mol, molecule_scale=0.5)
    assert result["adsorption_energy_eV"] == pytest.approx(-110.0 + 100.0 + 3.0)
    assert result["all_converged"]


def test_adsorption_energy_missing_job(tmp_path):
    total = _make_job(tmp_path, "slab_ads", -110.0)
    slab = _make_job(tmp_path, "slab", -100.0)
    with pytest.raises(FileNotFoundError):
        analysis.adsorption_energy(total, slab, tmp_path / "missing")


def test_parse_frequencies(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(FREQ_OUTCAR, encoding="utf-8")
    modes = analysis.parse_frequencies(outcar)

    assert len(modes) == 3
    assert modes[0]["meV"] == pytest.approx(378.617346)
    assert modes[0]["cm1"] == pytest.approx(3053.668884)
    assert not modes[0]["imaginary"]
    assert modes[2]["imaginary"]


def test_harmonic_thermochemistry_zpe_and_entropy():
    modes = [
        {"index": 1, "meV": 200.0, "cm1": 0.0, "THz": 0.0, "imaginary": False},
        {"index": 2, "meV": 10.0, "cm1": 0.0, "THz": 0.0, "imaginary": False},
        {"index": 3, "meV": 5.0, "cm1": 0.0, "THz": 0.0, "imaginary": True},
    ]
    result = analysis.harmonic_thermochemistry(modes, temperature=300.0)

    assert result["n_modes"] == 2
    assert result["n_imaginary"] == 1
    assert result["zpe_eV"] == pytest.approx(0.105)  # (0.2 + 0.01) / 2
    # A stiff 200 meV mode is frozen at 300 K; the soft 10 meV mode dominates TS.
    assert result["ts_eV"] > 0
    assert result["g_correction_eV"] == pytest.approx(
        result["zpe_eV"] + result["u_vib_eV"] - result["ts_eV"]
    )


def test_thermo_from_job(tmp_path):
    job = tmp_path / "freq"
    job.mkdir()
    (job / "OUTCAR").write_text(FREQ_OUTCAR, encoding="utf-8")
    result = analysis.thermo_from_job(job, temperature=298.15)

    assert result["n_modes"] == 2
    assert result["energy_eV"] == pytest.approx(-25.0)
    assert result["g_total_eV"] == pytest.approx(-25.0 + result["g_correction_eV"])


def test_thermo_from_job_without_modes(tmp_path):
    job = tmp_path / "scf"
    job.mkdir()
    (job / "OUTCAR").write_text(OUTCAR_CONVERGED.format(energy=-1.0), encoding="utf-8")
    with pytest.raises(ValueError, match="freq"):
        analysis.thermo_from_job(job)


def test_d_band_center(tmp_path):
    vasprun = tmp_path / "vasprun.xml"
    vasprun.write_text(VASPRUN_PDOS, encoding="utf-8")

    result = analysis.d_band_center(vasprun, [1])
    # d-DOS: 2.0 at E={0,2}, 0.0 at E=4; relative to efermi=2 -> {-2, 0, 2}.
    # Trapezoids: [-2,0] area 4 centred -1; [0,2] area 2 centred +1.
    assert result["d_band_center_eV"] == pytest.approx((4 * -1 + 2 * 1) / 6)
    assert result["n_electrons_d"] == pytest.approx(6.0)


def test_d_band_center_atom_without_d_weight(tmp_path):
    vasprun = tmp_path / "vasprun.xml"
    vasprun.write_text(VASPRUN_PDOS, encoding="utf-8")
    with pytest.raises(ValueError, match="zero"):
        analysis.d_band_center(vasprun, [2])


def test_d_band_center_occupied_only(tmp_path):
    vasprun = tmp_path / "vasprun.xml"
    vasprun.write_text(VASPRUN_PDOS, encoding="utf-8")
    result = analysis.d_band_center(vasprun, [1], emax_eV=0.0)
    assert result["d_band_center_eV"] == pytest.approx(-1.0)


def test_work_function(tmp_path, monkeypatch):
    job = tmp_path / "wf"
    job.mkdir()
    locpot_header = """slab
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 20.0
Al
1
Direct
0.0 0.0 0.1
"""
    # 1x1x4 grid along c: potential rises to a 5.0 eV vacuum plateau.
    (job / "LOCPOT").write_text(
        locpot_header + "\n  1  1  4\n -8.0 0.0 5.0 5.0\n", encoding="utf-8"
    )
    (job / "OUTCAR").write_text(" E-fermi :  -1.0     XC(G=0): ...\n", encoding="utf-8")

    result = analysis.work_function(job)
    assert result["vacuum_level_eV"] == pytest.approx(5.0)
    assert result["fermi_eV"] == pytest.approx(-1.0)
    assert result["work_function_eV"] == pytest.approx(6.0)
    assert len(result["profile_eV"]) == 4


def test_absorption_spectrum(tmp_path):
    vasprun = tmp_path / "vasprun.xml"
    vasprun.write_text(VASPRUN_DIELECTRIC, encoding="utf-8")

    result = analysis.absorption_spectrum(vasprun)
    assert result["alpha_cm1"][0] == pytest.approx(0.0)
    # E=2 eV, eps = 1 + 3i: |eps| = sqrt(10), k = sqrt((sqrt(10)-1)/2)
    k = math.sqrt((math.sqrt(10.0) - 1.0) / 2.0)
    assert result["alpha_cm1"][1] == pytest.approx(2.0 * 2.0 * k / analysis.HBARC_EV_CM)


def test_absorption_spectrum_missing_block(tmp_path):
    vasprun = tmp_path / "vasprun.xml"
    vasprun.write_text(VASPRUN_PDOS, encoding="utf-8")
    with pytest.raises(ValueError, match="optics"):
        analysis.absorption_spectrum(vasprun)


SLAB_POSCAR_HEADER = """slab
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 20.0
Al
{n}
Direct
"""


def _make_job_with_poscar(tmp_path, name, energy, natoms):
    job = _make_job(tmp_path, name, energy)
    coords = "\n".join(f"0.0 0.0 {i / (natoms + 1):.3f}" for i in range(natoms))
    (job / "POSCAR").write_text(SLAB_POSCAR_HEADER.format(n=natoms) + coords + "\n",
                                encoding="utf-8")
    return job


def test_surface_energy(tmp_path):
    slab = _make_job_with_poscar(tmp_path, "slab", -40.0, 4)
    bulk = _make_job_with_poscar(tmp_path, "bulk", -18.0, 2)
    # area = |a×b| = 16 Å²; γ = (-40 − (4/2)(−18)) / (2·16) = (-40+36)/32.
    result = analysis.surface_energy(slab, bulk)
    assert result["area_A2"] == pytest.approx(16.0)
    assert result["surface_energy_eV_A2"] == pytest.approx(-4.0 / 32.0)
    assert result["surface_energy_J_m2"] == pytest.approx((-4.0 / 32.0) * analysis.EV_A2_TO_J_M2)
    assert result["all_converged"]


def test_free_energy_diagram_staircase_and_overpotential():
    steps = [
        {"label": "*OH", "delta_e": 1.0, "n_electrons": 1},
        {"label": "*O", "delta_e": 0.5, "n_electrons": 1},
        {"label": "*OOH", "delta_e": -0.2, "n_electrons": 1},
    ]
    d = analysis.free_energy_diagram(steps, potential=0.5, u_equilibrium=1.23)
    assert d["cumulative_g0_eV"] == pytest.approx([0.0, 1.0, 1.5, 1.3])
    assert d["cumulative_g_eV"] == pytest.approx([0.0, 0.5, 0.5, -0.2])  # each −1·0.5
    assert d["potential_determining_step"] == "*OH"
    assert d["max_delta_g0_eV"] == pytest.approx(1.0)
    assert d["limiting_potential_V"] == pytest.approx(-1.0)
    assert d["overpotential_V"] == pytest.approx(abs(1.23 - (-1.0)))


def test_free_energy_diagram_from_freq_and_ads(tmp_path):
    total = _make_job(tmp_path, "slab_ads", -110.0)
    slab = _make_job(tmp_path, "slab", -100.0)
    mol = _make_job(tmp_path, "h2", -6.0)
    freq = tmp_path / "freq"
    freq.mkdir()
    (freq / "OUTCAR").write_text(FREQ_OUTCAR, encoding="utf-8")

    steps = [{"label": "ads", "ads": {"total": str(total), "slab": str(slab),
                                      "molecule": str(mol), "scale": 0.5},
              "freq_job": str(freq)}]
    d = analysis.free_energy_diagram(steps)
    g_corr = analysis.thermo_from_job(freq)["g_correction_eV"]
    assert d["steps"][0]["delta_e_eV"] == pytest.approx(-110.0 + 100.0 + 3.0)
    assert d["steps"][0]["g_correction_eV"] == pytest.approx(g_corr)


def test_read_job_energy_qe(tmp_path):
    from vasp_auto.parser import RY_TO_EV

    job = tmp_path / "qe"
    job.mkdir()
    (job / ".engine").write_text("qe", encoding="utf-8")
    (job / "pw.out").write_text(
        "!    total energy              =     -15.83792000 Ry\n"
        "     convergence has been achieved in  11 iterations\n",
        encoding="utf-8",
    )
    result = analysis.read_job_energy(job)
    assert result["energy_eV"] == pytest.approx(-15.83792000 * RY_TO_EV)
    assert result["converged"]


def test_read_volumetric_spin_block(tmp_path):
    from vasp_auto.chgcar import read_volumetric

    chgcar = tmp_path / "CHGCAR"
    chgcar.write_text(
        "c\n1.0\n4 0 0\n0 4 0\n0 0 4\nAl\n1\nDirect\n0 0 0\n"
        "\n 2 1 1\n 1.0 2.0\n"
        "augmentation occupancies 1 4\n 0.0 0.0 0.0 0.0\n"
        " 2 1 1\n 0.5 -0.5\n",
        encoding="utf-8",
    )
    volume = read_volumetric(chgcar, spin=True)
    assert volume["data"] == pytest.approx([1.0, 2.0])
    assert volume["spin_data"] == pytest.approx([0.5, -0.5])
    # Without spin=True the second block is not read.
    assert "spin_data" not in read_volumetric(chgcar)


def test_cli_adsorption_energy(tmp_path, monkeypatch, capsys):
    total = _make_job(tmp_path, "slab_ads", -110.0)
    slab = _make_job(tmp_path, "slab", -100.0)
    mol = _make_job(tmp_path, "h2", -6.0)

    monkeypatch.setattr(
        sys, "argv",
        ["vasp-auto", "--adsorption-energy", f"{total},{slab},{mol}", "--molecule-scale", "0.5"],
    )
    cli.main()
    out = capsys.readouterr().out
    assert "E_ads      : -7.000000 eV" in out


def test_cli_thermo(tmp_path, monkeypatch, capsys):
    job = tmp_path / "freq"
    job.mkdir()
    (job / "OUTCAR").write_text(FREQ_OUTCAR, encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["vasp-auto", "--thermo", str(job)])
    cli.main()
    out = capsys.readouterr().out
    assert "ZPE" in out
    assert "imaginary modes present" in out


def test_cli_delete_selection(tmp_path, monkeypatch, capsys):
    case = tmp_path / "case"
    case.mkdir()
    (case / "POSCAR").write_text(
        "c\n1.0\n4 0 0\n0 4 0\n0 0 4\nAl H\n2 1\nDirect\n0 0 0\n0.5 0.5 0\n0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "frag"
    monkeypatch.setattr(
        sys, "argv",
        ["vasp-auto", str(case), "--delete", "1-2", "--ase-output", str(out_dir), "--build-only"],
    )
    cli.main()
    poscar = (out_dir / "POSCAR").read_text(encoding="utf-8")
    assert "Al" not in poscar.splitlines()[5]
    assert "H" in poscar.splitlines()[5]


def test_cli_chg_diff(tmp_path, monkeypatch, capsys):
    header = "c\n1.0\n4 0 0\n0 4 0\n0 0 4\nAl\n1\nDirect\n0 0 0\n"
    for name, values in (("AB", "4.0 4.0"), ("A", "1.0 1.0"), ("B", "2.0 1.0")):
        d = tmp_path / name
        d.mkdir()
        (d / "CHGCAR").write_text(header + "\n 2 1 1\n" + values + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        ["vasp-auto", "--chg-diff", f"{tmp_path/'AB'},{tmp_path/'A'},{tmp_path/'B'}"],
    )
    cli.main()
    assert "CHGCAR_diff" in capsys.readouterr().out
    from vasp_auto.chgcar import read_volumetric
    assert read_volumetric(tmp_path / "AB" / "CHGCAR_diff")["data"] == pytest.approx([1.0, 2.0])
