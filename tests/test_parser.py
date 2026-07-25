from vasp_auto.parser import parse_vasprun

VASPRUN_TEXT = """<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
 <calculation>
  <energy>
   <i name="e_fr_energy">  -12.50000000 </i>
  </energy>
  <varray name="forces">
   <v>  0.00000000 0.00000000 0.10000000 </v>
   <v>  0.00000000 0.00000000 -0.05000000 </v>
  </varray>
  <varray name="stress">
   <v>  12.0 0.0 0.0 </v>
   <v>  0.0 12.0 0.0 </v>
   <v>  0.0 0.0 12.0 </v>
  </varray>
  <dos>
   <i name="efermi">  2.00000000 </i>
  </dos>
  <eigenvalues>
   <array>
    <set>
     <set comment="spin 1">
      <set comment="kpoint 1">
       <r>  1.0000 1.0000 </r>
       <r>  3.5000 0.0000 </r>
      </set>
     </set>
    </set>
   </array>
  </eigenvalues>
 </calculation>
</modeling>
"""


def test_parse_vasprun(tmp_path):
    vasprun = tmp_path / "vasprun.xml"
    vasprun.write_text(VASPRUN_TEXT, encoding="utf-8")

    result = parse_vasprun(vasprun)

    assert result["energy_eV"] == -12.5
    assert result["fermi_eV"] == 2.0
    assert result["max_force_eV_A"] == 0.1
    assert result["pressure_kB"] == 12.0
    assert result["vbm_eV"] == 1.0
    assert result["cbm_eV"] == 3.5
    assert result["band_gap_eV"] == 2.5
    assert result["ionic_steps"] == 1


def test_parse_vasprun_missing(tmp_path):
    assert parse_vasprun(tmp_path / "vasprun.xml") is None


def test_parse_vasprun_truncated(tmp_path):
    vasprun = tmp_path / "vasprun.xml"
    vasprun.write_text(VASPRUN_TEXT[: len(VASPRUN_TEXT) // 2], encoding="utf-8")
    assert parse_vasprun(vasprun) is None
