import json
import shutil
import threading
import urllib.request
from pathlib import Path

import pytest

from vasp_auto_ui import server as ui_server


@pytest.fixture(scope="module")
def ui_base():
    httpd = ui_server.create_server(port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _get(base, path):
    with urllib.request.urlopen(base + path) as response:
        return json.loads(response.read())


def _post(base, path, payload):
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(), method="POST"
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def test_meta_lists_calc_types(ui_base):
    meta = _get(ui_base, "/api/meta")
    assert "scf" in meta["calc_types"]
    assert "bands" in meta["calc_types"]
    assert meta["config"]["jobs_root"]


def test_index_served(ui_base):
    with urllib.request.urlopen(ui_base + "/") as response:
        page = response.read().decode()
    assert "vasp_auto" in page and "Structure editor" in page


def test_structure_endpoint(ui_base, scf_case):
    data = _get(ui_base, f"/api/structure?path={scf_case}")
    assert data["symbols"] == ["Al", "O"]
    assert data["natoms"] == 2
    # Direct (0.5, 0.5, 0.5) in a 4 Å cube -> cartesian (2, 2, 2).
    assert data["cartesian"][1] == pytest.approx([2.0, 2.0, 2.0])


def test_build_edit_supercell(ui_base, scf_case, tmp_path):
    out = tmp_path / "edited"
    data = _post(
        ui_base,
        "/api/build",
        {"action": "edit", "source": str(scf_case), "supercell": [2, 1, 1], "output": str(out)},
    )
    assert data["case"] == str(out)
    assert (out / "POSCAR").exists()
    structure = _get(ui_base, f"/api/structure?path={out}")
    assert structure["natoms"] == 4


def test_cases_listing(ui_base, scf_case):
    data = _get(ui_base, f"/api/cases?path={scf_case.parent}")
    names = [case["name"] for case in data["cases"]]
    assert scf_case.name in names


def test_template_endpoint(ui_base):
    data = _get(ui_base, "/api/template?type=relax")
    assert "IBRION" in data["text"]


def test_file_roundtrip(ui_base, scf_case):
    _post(ui_base, "/api/file", {"dir": str(scf_case), "name": "INCAR", "text": "ENCUT = 333\n"})
    data = _get(ui_base, f"/api/file?dir={scf_case}&name=INCAR")
    assert data["exists"] and "333" in data["text"]


def test_file_rejects_arbitrary_names(ui_base, scf_case):
    with pytest.raises(urllib.error.HTTPError):
        _post(ui_base, "/api/file", {"dir": str(scf_case), "name": "evil.sh", "text": "x"})


def test_build_cli_args():
    args = ui_server.build_cli_args(
        {
            "target": "/x/case",
            "mode": "run",
            "calc_type": "relax",
            "kpoints": {"mode": "mp", "mesh": "4x4x4"},
            "cpus": 8,
            "workflow": "relax,scf",
        }
    )
    assert args[0] == "/x/case"
    assert "--calc-type" in args and "relax" in args
    assert "--kmesh" in args and "--workflow" in args


def test_result_calc_type_detects_convergence(tmp_path):
    # A convergence scan leaves a scf_convergence/ subdir -> type "convergence".
    conv = tmp_path / "H2O"
    (conv / "scf_convergence" / "encut_400").mkdir(parents=True)
    (conv / "scf_convergence" / "encut_400" / "OUTCAR").write_text("x")
    assert ui_server._is_convergence_job(conv) is True
    assert ui_server._result_calc_type(conv) == "convergence"
    assert ui_server._job_dir_case_info(conv)["calculation_type"] == "convergence"

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "OUTCAR").write_text("x")
    assert ui_server._result_calc_type(plain) == "scf"


def test_result_row_has_modified_date(tmp_path):
    job = tmp_path / "H2O"
    job.mkdir()
    (job / "OUTCAR").write_text("free  energy   TOTEN  =  -1.0 eV\n")
    row = ui_server._result_row("p", "project", ui_server._job_dir_case_info(job))
    assert isinstance(row["modified_ts"], float)
    assert row["modified"]  # formatted "YYYY-MM-DD HH:MM"
    assert len(row["modified"]) == 16


def test_api_run_writes_workflow_yaml(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "POSCAR").write_text("dummy\n")

    captured = {}

    class FakeProc:
        def poll(self):
            return 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProc()

    monkeypatch.setattr(ui_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ui_server, "UI_LOG_DIR", tmp_path / "logs")

    yaml_text = 'steps:\n  - calc_type: converge\n    encut: "400,450"\n  - calc_type: scf\n'
    result = ui_server.api_run({}, {
        "target": str(case_dir), "mode": "run", "workflow_yaml": yaml_text,
    })
    assert "token" in result
    # The structured workflow lands in the case, and no --workflow string is passed.
    assert (case_dir / "workflow.yaml").read_text() == yaml_text
    assert "--workflow" not in captured["command"]


def test_api_run_fetches_remote_project_folder(tmp_path, monkeypatch):
    """'Run an existing folder' against a machine, pointed at a project folder
    (several case subdirs on the remote, not a POSCAR directly under it) should
    fetch every case found — not just look for a POSCAR at the given path."""
    remote_root = tmp_path / "remote_project"
    for name in ("caseA", "caseB"):
        case_dir = remote_root / name
        case_dir.mkdir(parents=True)
        (case_dir / "POSCAR").write_text(f"{name}\n")

    remote = {"name": "fake", "host": "h"}
    monkeypatch.setattr(ui_server, "_resolve_remote", lambda name: remote)

    def fake_list_cases(rmt, path):
        cases = [{"name": d.name, "path": str(d), "type": "scf"}
                  for d in sorted(Path(path).glob("*")) if (d / "POSCAR").is_file()]
        return {"path": path, "cases": cases}

    def fake_fetch(rmt, remote_path, local_path):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(remote_path, local_path)
        return Path(local_path)

    monkeypatch.setattr(ui_server, "list_remote_cases", fake_list_cases)
    monkeypatch.setattr(ui_server, "fetch_remote_file", fake_fetch)

    captured = {}

    class FakeProc:
        def poll(self):
            return 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProc()

    monkeypatch.setattr(ui_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ui_server, "_write_remote_config", lambda name: "irrelevant.json")
    monkeypatch.setattr(ui_server, "UI_LOG_DIR", tmp_path / "logs")

    ui_server.api_run({}, {
        "target": str(remote_root), "mode": "run", "source": "existing_folder",
        "machine": "fake",
    })

    # command = [python, -u, -m, vasp_auto.cli, <target>, ...] — target is positional.
    target = Path(captured["command"][4])
    assert (target / "caseA" / "POSCAR").is_file()
    assert (target / "caseB" / "POSCAR").is_file()


def test_existing_folder_requires_machine_and_honors_explicit_local(tmp_path, monkeypatch):
    folder = tmp_path / "case"
    folder.mkdir()
    (folder / "POSCAR").write_text("dummy\n")

    with pytest.raises(ValueError, match="selecting its machine"):
        ui_server.api_run({}, {
            "target": str(folder), "mode": "run", "source": "existing_folder",
        })

    monkeypatch.setattr(
        ui_server, "_remote_case_marker",
        lambda _path: (_ for _ in ()).throw(AssertionError("local choice was reinterpreted")),
    )
    captured = {}

    def fake_spawn(args, target):
        captured.update(args=args, target=target)
        return {"token": "test"}

    monkeypatch.setattr(ui_server, "_spawn_cli_job", fake_spawn)
    result = ui_server.api_run({}, {
        "target": str(folder), "mode": "run", "source": "existing_folder",
        "machine": "local",
    })
    assert result["token"] == "test"
    assert captured["target"] == str(folder)


def test_existing_folder_ui_enforces_machine_then_folder():
    html = (ui_server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert html.index('id="proj-remote"') < html.index('id="proj-dir"')
    assert 'id="proj-dir" placeholder="Select a machine first"' in html
    assert 'source: "existing_folder"' in html


def test_working_machine_direct_roundtrip(tmp_path, monkeypatch, scf_case):
    """With a remote working machine, saving writes the case straight onto that
    machine (nothing kept locally), api_cases lists the machine's cases, and the
    editor fetches a case back from it. The SSH layer is faked with a local dir."""
    import shutil as _sh
    from pathlib import Path as _P

    from vasp_auto import runner as _runner

    fake_root = tmp_path / "remote"
    inputs_dir = fake_root / "inputs"
    remote = {"name": "fake", "machine": "fake", "host": "h", "remote_root": str(fake_root)}
    monkeypatch.setattr(ui_server, "_resolve_remote", lambda name: remote)
    monkeypatch.setattr(_runner, "_run_checked", lambda *a, **k: "")

    def fake_transfer(local_dir, target, remote_dir, rmt):
        _sh.rmtree(remote_dir, ignore_errors=True)
        _sh.copytree(local_dir, remote_dir)

    def fake_fetch(rmt, remote_path, local_path):
        _P(local_path).parent.mkdir(parents=True, exist_ok=True)
        _sh.copy(remote_path, local_path)
        return _P(local_path)

    def fake_ship_file(local, target, remote_path, rmt, ssh_opts):
        _P(remote_path).parent.mkdir(parents=True, exist_ok=True)
        _sh.copy(local, remote_path)

    def fake_list_cases(rmt, path):
        cases = [{"name": d.name, "path": str(d), "type": "scf"}
                 for d in sorted(_P(path).glob("*")) if (d / "POSCAR").is_file()]
        return {"path": path, "cases": cases}

    monkeypatch.setattr(_runner, "_transfer_dir", fake_transfer)
    monkeypatch.setattr(_runner, "_ship_file", fake_ship_file)
    monkeypatch.setattr(ui_server, "fetch_remote_file", fake_fetch)
    monkeypatch.setattr(ui_server, "list_remote_cases", fake_list_cases)

    payload = ui_server.api_structure({"path": [str(scf_case)]}, None)
    res = ui_server.api_structure_save(
        {}, {"structure": payload, "name": "remcase", "machine": "fake"})

    # The case lives on the machine; nothing is created on this computer.
    assert res["remote"] is True and res["machine"] == "fake"
    assert res["case"] == str(inputs_dir / "remcase")
    assert (inputs_dir / "remcase" / "POSCAR").is_file()

    # api_cases lists the machine's cases (default <remote_root>/inputs).
    listed = ui_server.api_cases({"machine": ["fake"]}, None)
    assert listed["machine"] == "fake"
    assert any(c["name"] == "remcase" and c["remote"] and c["machine"] == "fake"
               for c in listed["cases"])

    # The editor fetches a case back from the machine by its remote path.
    fetched = ui_server.api_structure(
        {"path": [str(inputs_dir / "remcase")], "machine": ["fake"]}, None)
    assert fetched["natoms"] == payload["natoms"]

    # INCAR edits round-trip to the machine.
    ui_server.api_file_save({}, {"dir": str(inputs_dir / "remcase"), "name": "INCAR",
                                 "machine": "fake", "text": "ENCUT = 333\n"})
    got = ui_server.api_file_get(
        {"dir": [str(inputs_dir / "remcase")], "name": ["INCAR"], "machine": ["fake"]}, None)
    assert got["exists"] and "ENCUT = 333" in got["text"]


# ------------------------------------------------ interactive builder API

def test_structure_payload_has_editor_model(ui_base, scf_case):
    data = _get(ui_base, f"/api/structure?path={scf_case}")
    assert data["natoms"] == 2
    assert data["symbols"] == ["Al", "O"]
    assert data["frac"][1] == pytest.approx([0.5, 0.5, 0.5])
    assert data["cell"]["a"] == pytest.approx(4.0)
    assert data["cell"]["alpha"] == pytest.approx(90.0)
    assert data["selective"] is False
    assert len(data["flags"]) == 2


def test_structure_save_roundtrip(ui_base, scf_case, tmp_path):
    data = _get(ui_base, f"/api/structure?path={scf_case}")
    data["symbols"][1] = "Mg"
    data["frac"][1] = [0.25, 0.25, 0.25]
    out = tmp_path / "edited_case"
    saved = _post(ui_base, "/api/structure", {"structure": data, "dir": str(out)})
    again = _get(ui_base, f"/api/structure?path={saved['case']}")
    assert again["symbols"] == ["Al", "Mg"]
    assert again["frac"][1] == pytest.approx([0.25, 0.25, 0.25])


def test_structure_save_keeps_selective_flags(ui_base, scf_case, tmp_path):
    data = _get(ui_base, f"/api/structure?path={scf_case}")
    data["selective"] = True
    data["flags"] = [["F", "F", "F"], ["T", "T", "T"]]
    saved = _post(ui_base, "/api/structure", {"structure": data, "dir": str(tmp_path / "frozen")})
    again = _get(ui_base, f"/api/structure?path={saved['case']}")
    assert again["selective"] is True
    assert again["flags"][0] == ["F", "F", "F"]


def test_combine_endpoint_stack(ui_base, scf_case, tmp_path):
    guest_dir = tmp_path / "guest"
    guest_dir.mkdir()
    (guest_dir / "POSCAR").write_text(
        "H atom\n1.0\n2.0 0 0\n0 2.0 0\n0 0 2.0\nH\n1\nDirect\n0.0 0.0 0.0\n"
    )
    data = _post(ui_base, "/api/combine", {
        "host": str(scf_case), "guest": str(guest_dir),
        "mode": "stack", "gap": 2.0, "vacuum": 5.0,
    })
    s = data["structure"]
    assert s["symbols"] == ["Al", "O", "H"]
    assert s["poscar"] is None  # nothing was written
    # H sits 2 Å above the highest host atom (O at z = 2 Å).
    assert s["cartesian"][2][2] == pytest.approx(4.0)


def test_combine_accepts_inline_host(ui_base, scf_case):
    host = _get(ui_base, f"/api/structure?path={scf_case}")
    data = _post(ui_base, "/api/combine", {
        "host_struct": host, "guest": str(scf_case), "mode": "insert", "gap": 1.0,
    })
    assert data["structure"]["natoms"] == 4


def test_combine_fetches_remote_guest(tmp_path, monkeypatch, scf_case):
    """With a remote working machine, combine reads case paths from that
    machine over SSH — given as a case folder or as the POSCAR file itself."""
    import shutil as _sh
    from pathlib import Path as _P

    fake_root = tmp_path / "remote"
    guest_dir = fake_root / "inputs" / "guest"
    guest_dir.mkdir(parents=True)
    (guest_dir / "POSCAR").write_text(
        "H atom\n1.0\n2.0 0 0\n0 2.0 0\n0 0 2.0\nH\n1\nDirect\n0.0 0.0 0.0\n")

    monkeypatch.setattr(ui_server, "_resolve_remote",
                        lambda name: {"name": "fake", "host": "h",
                                      "remote_root": str(fake_root)})

    def fake_fetch(rmt, remote_path, local_path):
        _P(local_path).parent.mkdir(parents=True, exist_ok=True)
        _sh.copy(remote_path, local_path)  # missing path raises, like real SSH
        return _P(local_path)

    monkeypatch.setattr(ui_server, "fetch_remote_file", fake_fetch)

    host = ui_server.api_structure({"path": [str(scf_case)]}, None)
    for guest in (str(guest_dir), str(guest_dir / "POSCAR")):
        data = ui_server.api_combine({}, {
            "host_struct": host, "guest": guest, "machine": "fake",
            "mode": "stack", "gap": 2.0, "vacuum": 5.0})
        assert data["structure"]["symbols"] == ["Al", "O", "H"]


# ------------------------------------------------ pass-7 results endpoints

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
    <set>
     <set comment="ion 1">
      <set comment="spin 1">
       <r> 0.0 0.1 1.0 </r>
       <r> 2.0 0.1 1.0 </r>
      </set>
     </set>
     <set comment="ion 2">
      <set comment="spin 1">
       <r> 0.0 0.2 0.0 </r>
       <r> 2.0 0.2 0.0 </r>
      </set>
     </set>
    </set>
   </array></partial>
  </dos>
 </calculation>
</modeling>
"""

VASPRUN_BANDS = """<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
 <kpoints>
  <varray name="kpointlist">
   <v> 0.00 0.0 0.0 </v>
   <v> 0.25 0.0 0.0 </v>
   <v> 0.50 0.0 0.0 </v>
  </varray>
 </kpoints>
 <calculation>
  <eigenvalues><array><set>
   <set comment="spin 1">
    <set comment="kpoint 1"><r> 1.0 1.0 </r><r> 3.0 0.0 </r></set>
    <set comment="kpoint 2"><r> 1.5 1.0 </r><r> 3.5 0.0 </r></set>
    <set comment="kpoint 3"><r> 2.0 1.0 </r><r> 4.0 0.0 </r></set>
   </set>
  </set></array></eigenvalues>
  <dos><i name="efermi"> 2.0 </i></dos>
 </calculation>
</modeling>
"""

KPOINTS_ONE_SEGMENT = """k-path
3
Line-mode
Reciprocal
0.0 0.0 0.0 ! G
0.5 0.0 0.0 ! X
"""

# Matches the conftest Al-O POSCAR cell (4x4x4 -> volume 64 Å³).
CHGCAR_HEADER = """Al O test cell
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
Al O
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
CHGCAR_TEXT = CHGCAR_HEADER + "\n  2  1  2\n 64.0 128.0 192.0 256.0\n"


def test_pdos_endpoint(ui_base, scf_case):
    (scf_case / "vasprun.xml").write_text(VASPRUN_PDOS, encoding="utf-8")
    data = _get(ui_base, f"/api/pdos?path={scf_case}")
    labels = [curve["label"] for curve in data["curves"]]
    assert "Al s" in labels and "Al d" in labels and "O s" in labels
    assert data["natoms"] == 2

    restricted = _get(ui_base, f"/api/pdos?path={scf_case}&atoms=2")
    labels = [curve["label"] for curve in restricted["curves"]]
    assert "Al s" not in labels and "O s" in labels
    assert restricted["selection"] == "2"


def test_pdos_endpoint_without_partial_dos(ui_base, scf_case):
    with pytest.raises(urllib.error.HTTPError):
        _get(ui_base, f"/api/pdos?path={scf_case}")


def test_bands_endpoint(ui_base, scf_case):
    (scf_case / "vasprun.xml").write_text(VASPRUN_BANDS, encoding="utf-8")
    (scf_case / "KPOINTS").write_text(KPOINTS_ONE_SEGMENT, encoding="utf-8")
    data = _get(ui_base, f"/api/bands?path={scf_case}")
    assert data["bands"][0][0] == pytest.approx([1.0, 1.5, 2.0])
    assert data["labels"] == [{"index": 0, "label": "G"}, {"index": 2, "label": "X"}]


def test_volume_endpoint_chgcar(ui_base, scf_case):
    (scf_case / "CHGCAR").write_text(CHGCAR_TEXT, encoding="utf-8")
    data = _get(ui_base, f"/api/volume?path={scf_case}&file=CHGCAR&axis=c")
    assert data["unit"] == "e/Å³"
    assert data["grid"] == [2, 1, 2]
    # CHGCAR stores rho*V; planes (64,128) and (192,256) divided by V=64.
    assert data["profile"] == pytest.approx([1.5, 3.5])
    assert data["slice"]["data"]


def test_volume_endpoint_rejects_unknown_file(ui_base, scf_case):
    with pytest.raises(urllib.error.HTTPError):
        _get(ui_base, f"/api/volume?path={scf_case}&file=WAVECAR")


def test_chgdiff_endpoint(ui_base, tmp_path):
    for name, scale in (("AB", 4.0), ("A", 1.0), ("B", 2.0)):
        d = tmp_path / name
        d.mkdir()
        values = " ".join(str(64.0 * scale) for _ in range(4))
        (d / "CHGCAR").write_text(
            CHGCAR_HEADER + "\n  2  1  2\n " + values + "\n", encoding="utf-8",
        )
    data = _post(ui_base, "/api/chgdiff", {
        "total": str(tmp_path / "AB"),
        "parts": [str(tmp_path / "A"), str(tmp_path / "B")],
    })
    assert data["file"] == "CHGCAR_diff"
    assert data["profile"] == pytest.approx([1.0, 1.0])  # (4-1-2)*64/64 per plane
    assert (tmp_path / "AB" / "CHGCAR_diff").exists()


def test_match_endpoint(ui_base, tmp_path):
    def sheet_case(name, a):
        d = tmp_path / name
        d.mkdir()
        (d / "POSCAR").write_text(
            f"sheet\n1.0\n{a} 0 0\n0 {a} 0\n0 0 15.0\nC\n1\nDirect\n0 0 0.5\n",
            encoding="utf-8",
        )
        return d

    host = sheet_case("host", 2.0)
    guest = sheet_case("guest", 3.0)
    data = _post(ui_base, "/api/match", {"host": str(host), "guest": str(guest)})
    best = data["matches"][0]
    assert best["host_repeat"] == [3, 3]
    assert best["guest_repeat"] == [2, 2]
    assert best["strain_pct"] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------ pass-8 analysis endpoints

OUTCAR_CONVERGED = """ mock
  aborting loop because EDIFF is reached
  free  energy   TOTEN  =      {energy:.8f} eV
"""

FREQ_OUTCAR = """ vasp.6 mock
   1 f  =   91.546624 THz   575.204660 2PiTHz 3053.668884 cm-1   378.617346 meV
   2 f/i=    0.022552 THz     0.141698 2PiTHz    0.752259 cm-1     0.093268 meV
  aborting loop because EDIFF is reached
  free  energy   TOTEN  =       -25.00000000 eV
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


def _energy_job(tmp_path, name, energy):
    job = tmp_path / name
    job.mkdir()
    (job / "OUTCAR").write_text(OUTCAR_CONVERGED.format(energy=energy), encoding="utf-8")
    return job


def test_adsorption_endpoint(ui_base, tmp_path):
    total = _energy_job(tmp_path, "slab_ads", -110.0)
    slab = _energy_job(tmp_path, "slab", -100.0)
    mol = _energy_job(tmp_path, "h2", -6.0)
    data = _post(ui_base, "/api/adsorption", {
        "total": str(total), "slab": str(slab), "molecule": str(mol), "scale": 0.5,
    })
    assert data["adsorption_energy_eV"] == pytest.approx(-7.0)
    assert data["all_converged"] is True


def test_thermo_endpoint(ui_base, tmp_path):
    job = tmp_path / "freq"
    job.mkdir()
    (job / "OUTCAR").write_text(FREQ_OUTCAR, encoding="utf-8")
    data = _get(ui_base, f"/api/thermo?path={job}&T=300")
    assert data["temperature_K"] == pytest.approx(300.0)
    assert data["n_modes"] == 1
    assert data["n_imaginary"] == 1
    assert data["zpe_eV"] == pytest.approx(0.378617346 / 2.0)
    assert data["g_total_eV"] == pytest.approx(-25.0 + data["g_correction_eV"])


def test_dband_endpoint(ui_base, scf_case):
    (scf_case / "vasprun.xml").write_text(VASPRUN_PDOS, encoding="utf-8")
    data = _get(ui_base, f"/api/dband?path={scf_case}&atoms=1")
    assert data["atoms"] == [1]
    assert data["selection"] == "1"
    assert "d_band_center_eV" in data and "d_band_width_eV" in data


def test_dband_endpoint_needs_selection(ui_base, scf_case):
    (scf_case / "vasprun.xml").write_text(VASPRUN_PDOS, encoding="utf-8")
    with pytest.raises(urllib.error.HTTPError):
        _get(ui_base, f"/api/dband?path={scf_case}")


def test_workfunction_endpoint(ui_base, tmp_path):
    job = tmp_path / "wf"
    job.mkdir()
    (job / "LOCPOT").write_text(
        "slab\n1.0\n4.0 0 0\n0 4.0 0\n0 0 20.0\nAl\n1\nDirect\n0 0 0.1\n"
        "\n  1  1  4\n -8.0 0.0 5.0 5.0\n",
        encoding="utf-8",
    )
    (job / "OUTCAR").write_text(" E-fermi :  -1.0     XC(G=0): ...\n", encoding="utf-8")
    data = _get(ui_base, f"/api/workfunction?path={job}")
    assert data["work_function_eV"] == pytest.approx(6.0)
    assert len(data["profile_eV"]) == 4


def test_optics_endpoint(ui_base, tmp_path):
    job = tmp_path / "optics"
    job.mkdir()
    (job / "vasprun.xml").write_text(VASPRUN_DIELECTRIC, encoding="utf-8")
    data = _get(ui_base, f"/api/optics?path={job}")
    assert data["energies_eV"] == pytest.approx([0.0, 2.0])
    assert data["alpha_cm1"][0] == pytest.approx(0.0)
    assert data["alpha_cm1"][1] > 0


def test_bader_endpoint_without_binary(ui_base, tmp_path, monkeypatch):
    import shutil

    job = tmp_path / "job"
    job.mkdir()
    (job / "CHGCAR").write_text("x", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(urllib.error.HTTPError):
        _post(ui_base, "/api/bader", {"path": str(job)})


def test_browse_endpoint(ui_base, tmp_path):
    (tmp_path / "alpha").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    data = _get(ui_base, f"/api/browse?path={tmp_path}")
    names = [d["name"] for d in data["dirs"]]
    assert "alpha" in names and ".hidden" not in names
    assert data["files"] == []  # files only on request
    assert any(root["name"] == "inputs" for root in data["roots"])

    with_files = _get(ui_base, f"/api/browse?path={tmp_path}&files=1")
    assert [f["name"] for f in with_files["files"]] == ["notes.txt"]


# ---------------------------------------------------------------- remote machines

def test_remote_save_list_delete(ui_base, tmp_path, monkeypatch):
    """The Remote tab can add, list and delete a machine (UI-managed store)."""
    monkeypatch.setattr(ui_server, "REMOTES_FILE", tmp_path / "remotes.json")
    saved = _post(ui_base, "/api/remote/save", {
        "name": "cluster1", "host": "c.edu", "user": "me",
        "remote_root": "/scratch/me", "vasp_executable": "/opt/vasp_std",
        "scheduler": "slurm", "cpus": "32",
    })
    assert saved["saved"] == "cluster1"
    # a per-machine default core count is stored, coerced to a positive int
    assert saved["remote"]["cpus"] == 32

    listing = _get(ui_base, "/api/remotes")
    names = [r["name"] for r in listing["remotes"]]
    assert "cluster1" in names
    entry = next(r for r in listing["remotes"] if r["name"] == "cluster1")
    assert entry["host"] == "c.edu" and entry["source"] == "ui"
    assert entry["cpus"] == 32

    deleted = _post(ui_base, "/api/remote/delete", {"name": "cluster1"})
    assert deleted["deleted"] == "cluster1"
    assert "cluster1" not in [r["name"] for r in _get(ui_base, "/api/remotes")["remotes"]]


def test_remote_save_requires_fields(ui_base, tmp_path, monkeypatch):
    monkeypatch.setattr(ui_server, "REMOTES_FILE", tmp_path / "remotes.json")
    with pytest.raises(urllib.error.HTTPError):
        _post(ui_base, "/api/remote/save", {"name": "x", "host": "h"})  # no remote_root/exe


def test_remote_save_drops_bad_cpus(ui_base, tmp_path, monkeypatch):
    """A non-numeric default core count is ignored, not stored as junk."""
    monkeypatch.setattr(ui_server, "REMOTES_FILE", tmp_path / "remotes.json")
    saved = _post(ui_base, "/api/remote/save", {
        "name": "c2", "host": "c.edu", "remote_root": "/s",
        "vasp_executable": "/opt/vasp_std", "cpus": "lots",
    })
    assert "cpus" not in saved["remote"]


def test_remote_status_needs_marker(ui_base, tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    with pytest.raises(urllib.error.HTTPError):
        _post(ui_base, "/api/remote/status", {"job_dir": str(job)})


def test_results_job_dir_is_single_layout(ui_base, tmp_path, monkeypatch):
    """Regression: a case under a cases folder resolves to <jobs_root>/<case>,
    not <jobs_root>/<folder>/<case> (which broke every per-row button)."""
    cases = tmp_path / "inputs"
    (cases / "Si").mkdir(parents=True)
    (cases / "Si" / "POSCAR").write_text("Si\n1.0\n1 0 0\n0 1 0\n0 0 1\nSi\n1\nDirect\n0 0 0\n")
    jobs = tmp_path / "jobs"
    (jobs / "Si").mkdir(parents=True)

    base_cfg = ui_server._config()
    monkeypatch.setattr(ui_server, "_config",
                        lambda: {**base_cfg, "jobs_root": str(jobs)})
    data = _get(ui_base, "/api/results?target=" + str(cases))
    job_dirs = [r["job_dir"] for r in data["rows"]]
    assert str(jobs / "Si") in job_dirs
    assert str(jobs / "inputs" / "Si") not in job_dirs


def test_results_reads_jobs_folder_directly(ui_base, tmp_path, monkeypatch):
    """Pointing Results at the jobs folder lists the real job dirs (incl. ones
    whose name no longer matches an inputs case), each with its true job_dir."""
    jobs = tmp_path / "jobs"
    # a finished scf job, a convergence-only job, and a NEB job
    (jobs / "Fe").mkdir(parents=True)
    (jobs / "Fe" / "OUTCAR").write_text("E-fermi : 1.0\n")
    (jobs / "renamed_opt").mkdir(parents=True)
    (jobs / "renamed_opt" / "OUTCAR").write_text("E-fermi : 1.0\n")
    for img in ("00", "01", "02"):
        (jobs / "neb" / img).mkdir(parents=True)
        (jobs / "neb" / img / "POSCAR").write_text("x\n")
        (jobs / "neb" / img / "OUTCAR").write_text("E-fermi : 1.0\n")

    base_cfg = ui_server._config()
    monkeypatch.setattr(ui_server, "_config", lambda: {**base_cfg, "jobs_root": str(jobs)})

    data = _get(ui_base, "/api/results?target=" + str(jobs))
    by_case = {r["case"]: r for r in data["rows"]}
    assert set(by_case) == {"Fe", "renamed_opt", "neb"}
    # every row points at the real directory under jobs/
    for name, row in by_case.items():
        assert row["job_dir"] == str(jobs / name)
    assert by_case["neb"]["calculation_type"] == "tss"


def test_results_single_job_dir(ui_base, tmp_path, monkeypatch):
    """Pointing Results at one job directory returns just that job."""
    jobs = tmp_path / "jobs"
    (jobs / "Si").mkdir(parents=True)
    (jobs / "Si" / "OUTCAR").write_text("E-fermi : 1.0\n")
    base_cfg = ui_server._config()
    monkeypatch.setattr(ui_server, "_config", lambda: {**base_cfg, "jobs_root": str(jobs)})

    data = _get(ui_base, "/api/results?target=" + str(jobs / "Si"))
    assert [r["case"] for r in data["rows"]] == ["Si"]
    assert data["rows"][0]["job_dir"] == str(jobs / "Si")


def test_neb_endpoint(ui_base, tmp_path):
    """/api/neb returns the reaction-coordinate energy profile for a TSS job."""
    job = tmp_path / "neb"
    energies = [-10.0, -9.3, -8.8, -9.6, -10.1]
    for i, e in enumerate(energies):
        img = job / f"{i:02d}"
        img.mkdir(parents=True)
        (img / "OUTCAR").write_text(
            f" vasp\n  free  energy   TOTEN  =  {e:.8f} eV\n"
            "  aborting loop because EDIFF is reached\n", encoding="utf-8")
        (img / "POSCAR").write_text(
            "x\n1.0\n9 0 0\n0 9 0\n0 0 9\nH\n1\nCartesian\n"
            f"{i*0.4:.3f} 0 0\n", encoding="utf-8")

    data = _get(ui_base, f"/api/neb?path={job}")
    assert data["images"] == [0, 1, 2, 3, 4]
    assert data["ts_image"] == 2
    assert data["forward_barrier_eV"] == pytest.approx(1.2)   # -8.8 - (-10.0)
    assert data["relative_eV"][0] == 0.0
    assert data["reaction_coord"][-1] == pytest.approx(1.0)


def test_neb_endpoint_missing(ui_base, tmp_path):
    job = tmp_path / "scf"
    job.mkdir()
    (job / "OUTCAR").write_text("E-fermi : 1.0\n", encoding="utf-8")
    with pytest.raises(urllib.error.HTTPError):
        _get(ui_base, f"/api/neb?path={job}")


# ------------------------------------------------- remote/local browse + download

def test_remote_jobs_endpoint(ui_base, monkeypatch):
    """/api/remote/jobs lists jobs on a machine (date formatted), default dir = remote_root."""
    monkeypatch.setattr(ui_server, "_all_remotes",
        lambda: {"cl": {"name": "cl", "host": "c.edu", "remote_root": "/work"}})
    monkeypatch.setattr(ui_server, "list_remote_jobs",
        lambda remote, root: [{"path": root + "/Au13", "name": "Au13", "rel": "Au13",
                               "modified_ts": 1700000000, "status": "done",
                               "has_outcar": True, "has_vasprun": True}])
    data = _post(ui_base, "/api/remote/jobs", {"machine": "cl"})
    assert data["machine"] == "cl" and data["dir"] == "/work"
    row = data["rows"][0]
    assert row["name"] == "Au13" and row["machine"] == "cl" and row["modified"]


def test_remote_jobs_unknown_machine(ui_base, monkeypatch):
    monkeypatch.setattr(ui_server, "_all_remotes", lambda: {})
    with pytest.raises(urllib.error.HTTPError):
        _post(ui_base, "/api/remote/jobs", {"machine": "nope"})


def test_remote_files_endpoint(ui_base, monkeypatch):
    monkeypatch.setattr(ui_server, "_all_remotes",
        lambda: {"cl": {"name": "cl", "host": "c.edu", "remote_root": "/work"}})
    monkeypatch.setattr(ui_server, "list_remote_dir",
        lambda remote, path: {"path": path, "parent": "/work",
            "entries": [{"name": "OUTCAR", "path": path + "/OUTCAR",
                         "is_dir": False, "size": 10, "modified_ts": 1700000000}]})
    data = _post(ui_base, "/api/remote/files", {"machine": "cl", "dir": "/work/Au13"})
    assert data["machine"] == "cl"
    assert data["entries"][0]["name"] == "OUTCAR" and data["entries"][0]["modified"]


def test_job_files_endpoint(ui_base, tmp_path):
    """/api/job/files lists a local job dir's files and subdirs with sizes."""
    job = tmp_path / "Si"
    job.mkdir()
    (job / "OUTCAR").write_text("E-fermi : 1.0\n")
    (job / "trial_1").mkdir()
    data = _post(ui_base, "/api/job/files", {"dir": str(job)})
    by = {e["name"]: e for e in data["entries"]}
    assert by["OUTCAR"]["is_dir"] is False and by["OUTCAR"]["size"] > 0
    assert by["trial_1"]["is_dir"] is True
    assert data["parent"] == str(tmp_path)


def test_filetext_local_previewable(ui_base, tmp_path):
    """/api/filetext returns the text of a local file for the in-browser viewer."""
    job = tmp_path / "Si"
    job.mkdir()
    (job / "INCAR").write_text("ENCUT = 520\nISMEAR = 0\n")
    data = _post(ui_base, "/api/filetext", {"path": str(job / "INCAR")})
    assert data["previewable"] is True
    assert "ENCUT = 520" in data["text"]
    assert data["truncated"] is False


def test_filetext_blocks_potcar(ui_base, tmp_path):
    """POTCAR content is proprietary and must not be shown in the viewer."""
    job = tmp_path / "Si"
    job.mkdir()
    (job / "POTCAR").write_text("PAW_PBE Si ... secret ...\n")
    data = _post(ui_base, "/api/filetext", {"path": str(job / "POTCAR")})
    assert data["previewable"] is False
    assert "text" not in data


def test_filetext_remote(ui_base, monkeypatch):
    """A remote filetext request reads over SSH, constrained to remote_root."""
    monkeypatch.setattr(ui_server, "_all_remotes",
        lambda: {"cl": {"name": "cl", "host": "c.edu", "remote_root": "/work"}})
    monkeypatch.setattr(ui_server, "read_remote_text",
        lambda remote, path, max_bytes: {"text": "NSW = 50\n", "size": 8, "truncated": False})
    data = _post(ui_base, "/api/filetext",
                 {"machine": "cl", "path": "/work/Au13/INCAR", "name": "INCAR"})
    assert data["previewable"] is True and data["text"] == "NSW = 50\n"
    # a path outside remote_root is refused
    with pytest.raises(urllib.error.HTTPError):
        _post(ui_base, "/api/filetext",
              {"machine": "cl", "path": "/etc/passwd", "name": "passwd"})


def test_download_local_serves_file(ui_base, tmp_path):
    from urllib.parse import quote
    job = tmp_path / "Si"
    job.mkdir()
    (job / "OUTCAR").write_bytes(b"hello vasp")
    url = ui_base + "/download_local?path=" + quote(str(job / "OUTCAR"))
    with urllib.request.urlopen(url) as resp:
        body = resp.read()
        disp = resp.headers.get("Content-Disposition")
    assert body == b"hello vasp" and "OUTCAR" in disp


def test_download_local_missing_file(ui_base, tmp_path):
    from urllib.parse import quote
    url = ui_base + "/download_local?path=" + quote(str(tmp_path / "nope"))
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(url)


def test_download_remote_serves_file(ui_base, monkeypatch):
    from pathlib import Path
    from urllib.parse import quote
    monkeypatch.setattr(ui_server, "_all_remotes",
        lambda: {"cl": {"name": "cl", "host": "c.edu", "remote_root": "/work"}})
    monkeypatch.setattr(ui_server, "fetch_remote_file",
        lambda remote, rpath, local: (Path(local).write_bytes(b"remote bytes"), Path(local))[1])
    url = ui_base + "/download_remote?machine=cl&path=" + quote("/work/Au13/OUTCAR")
    with urllib.request.urlopen(url) as resp:
        assert resp.read() == b"remote bytes"


def test_download_remote_rejects_path_outside_root(ui_base, monkeypatch):
    from urllib.parse import quote
    monkeypatch.setattr(ui_server, "_all_remotes",
        lambda: {"cl": {"name": "cl", "host": "c.edu", "remote_root": "/work"}})
    url = ui_base + "/download_remote?machine=cl&path=" + quote("/etc/passwd")
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(url)
