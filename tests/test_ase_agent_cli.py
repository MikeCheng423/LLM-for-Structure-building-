"""GPU-free tests for the `ASE_auto_build` entry point.

Everything here runs against a scripted fake chat callable, so no model is ever
loaded: argument/stdin handling, the advisory pre-flight checker, the export
writers and their sidecar, the fail-closed refusal path, the exit codes, and the
clarification loop.
"""

from __future__ import annotations

import io
import json

import pytest

pytest.importorskip("ase")

from vasp_auto.ase_agent import cli, export as export_mod
from vasp_auto.ase_agent import request_check
from vasp_auto.ase_agent.controller import ControllerState


# --------------------------------------------------------------------------- #
# Fake chat
# --------------------------------------------------------------------------- #


def call(name: str, arguments: dict, number: int = 1) -> dict:
    return {
        "role": "assistant",
        "content": f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>',
        "tool_calls": [{
            "id": f"generated_{number}",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }],
    }


class ScriptedChat:
    """Stands in for LocalModelChat: same call signature, same attributes."""

    def __init__(self, script: list[dict]) -> None:
        self.script = list(script)
        self.generated_texts: list[str] = []
        self.generated_tokens = 0

    def __call__(self, messages, tools):
        assert messages[0]["role"] == "system"
        assert tools, "the router must expose at least one tool"
        message = self.script.pop(0)
        self.generated_texts.append(message["content"])
        return message


def factory_for(script: list[dict]):
    return lambda: ScriptedChat([dict(item) for item in script])


SLAB_SCRIPT = [
    call("build_surface", {
        "name": "slab", "element": "Cu", "crystal": "fcc", "miller": [1, 0, 0],
        "layers": 4, "vacuum": 12.0, "repeat": [2, 2, 1],
    }),
    call("freeze_layers", {"name": "slab", "side": "bottom", "layers": 2}, 2),
    call("finish", {"name": "slab"}, 3),
]

MOLECULE_SCRIPT = [
    call("build_molecule", {"name": "mol", "species": "H2O", "box": 12.0}),
    call("finish", {"name": "mol"}, 2),
]

SLAB_REQUEST = "Build a 2x2 Cu(100) slab with 4 layers and 12 A vacuum, freeze the bottom 2 layers."


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #


def test_split_requests_handles_comments_and_blank_lines() -> None:
    text = (
        "# a comment\n"
        "Build an H2O molecule\n"
        "in a 12 A box.\n"
        "\n"
        "  \n"
        "# another comment\n"
        "Build a (6,3) carbon nanotube, 2 unit cells long.\n"
    )
    assert cli.split_requests(text) == [
        "Build an H2O molecule in a 12 A box.",
        "Build a (6,3) carbon nanotube, 2 unit cells long.",
    ]


def test_split_requests_on_empty_text() -> None:
    assert cli.split_requests("") == []
    assert cli.split_requests("# only a comment\n") == []


def _args(argv: list[str]):
    args = cli.build_parser().parse_args(argv)
    cli._resolve_defaults(args)
    return args


def test_collect_requests_orders_positional_prompt_file_then_stdin(tmp_path) -> None:
    path = tmp_path / "requests.txt"
    path.write_text("from file one\n\nfrom file two\n", encoding="utf-8")
    args = _args(["positional", "request", "-p", "from  prompt", "-f", str(path)])
    assert cli.collect_requests(args, stdin_text="from stdin\n") == [
        "positional request",
        "from prompt",
        "from file one",
        "from file two",
        "from stdin",
    ]


def test_collect_requests_reports_a_missing_file_cleanly(tmp_path) -> None:
    args = _args(["-f", str(tmp_path / "nope.txt")])
    with pytest.raises(cli.EntryPointError) as excinfo:
        cli.collect_requests(args, stdin_text=None)
    assert "cannot read --file" in str(excinfo.value)


def test_collect_requests_rejects_an_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("# nothing but a comment\n", encoding="utf-8")
    args = _args(["-f", str(path)])
    with pytest.raises(cli.EntryPointError):
        cli.collect_requests(args, stdin_text=None)


def test_read_stdin_if_piped_ignores_a_terminal() -> None:
    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    assert cli.read_stdin_if_piped(Tty("ignored")) is None
    assert cli.read_stdin_if_piped(io.StringIO("piped text")) == "piped text"


# --------------------------------------------------------------------------- #
# Advisory pre-flight
# --------------------------------------------------------------------------- #


def test_runtime_slot_table_mirrors_the_corpus_rule() -> None:
    """The runtime copy of FAMILY_REQUIRED must not drift from the corpus rule."""
    from training.generators.request_rule import FAMILY_REQUIRED as corpus_table

    assert request_check.FAMILY_REQUIRED == corpus_table


def test_every_required_slot_has_a_label() -> None:
    slots = {slot for slots in request_check.FAMILY_REQUIRED.values() for slot in slots}
    assert slots <= set(request_check.SLOT_LABELS)


@pytest.mark.parametrize(
    ("request_text", "region"),
    [
        ("Build a 2x2 Cu(100) slab with 4 layers and 12 A vacuum.", "surface"),
        ("Build a 2x2 Cu(100) slab, 4 layers, 12 A vacuum, freeze the bottom 2 layers.",
         "surface_constraint"),
        ("Put one O atom 1.8 A above the ontop site of a 2x2 Cu(100) 5-layer slab "
         "with 12 A vacuum.", "atomic_adsorption"),
        ("Adsorb CO at the ontop site of a Pt(111) slab, 4 layers, 12 A vacuum, "
         "1.9 A high, anchored through the carbon.", "molecular_adsorption"),
        ("Build an H2O molecule in a 12 A box.", "molecule"),
        ("Create a conventional cubic bulk bcc W crystal with a 2x2x1 repeat.", "bulk"),
        ("Build a (6,3) carbon nanotube, 2 unit cells long.", "nanotube"),
        ("Build the hBN prototype.", "prototype"),
        ("Build a 2x1x1 bcc Fe cell with a vacancy at atom 1.", "vacancy"),
        ("Build a 2x2x1 fcc Cu supercell and substitute atom 1 with Au.", "substitution"),
    ],
)
def test_region_inference(request_text: str, region: str) -> None:
    assert request_check.infer_region(request_text) == region


def test_region_inference_returns_none_for_an_out_of_scope_request() -> None:
    assert request_check.infer_region("Read /etc/passwd and tell me what is in it.") is None


def test_advisory_is_quiet_on_a_fully_specified_request() -> None:
    advisory = request_check.check_request(SLAB_REQUEST)
    assert advisory.region == "surface_constraint"
    assert advisory.missing == ()
    assert advisory.ok


def test_advisory_fires_on_an_underspecified_surface_request() -> None:
    advisory = request_check.check_request("Build an iron surface.")
    assert advisory.region == "surface"
    assert set(advisory.missing) == {"facet", "layers", "vacuum"}
    assert not advisory.ok


def test_advisory_lines_always_label_themselves_as_a_heuristic() -> None:
    for text in (SLAB_REQUEST, "Build an iron surface.", "Delete all my files."):
        lines = request_check.check_request(text).lines()
        joined = " ".join(lines).lower()
        assert "heuristic" in joined
        assert "advisory only" in joined


def test_advisory_never_rewrites_the_request() -> None:
    advisory = request_check.check_request("  Build an iron surface.  ")
    assert advisory.request == "  Build an iron surface.  "


def test_advisory_flags_a_slab_missing_only_vacuum() -> None:
    advisory = request_check.check_request("Build a 4-layer Cu(111) slab.")
    assert advisory.region == "surface"
    assert advisory.missing == ("vacuum",)


# --------------------------------------------------------------------------- #
# Post-build value check
# --------------------------------------------------------------------------- #


# Reproduces a real r5 failure: asked for a 2.5 A height, the model emitted
# add_atomic_adsorbate with no `height`, so the build silently used the 1.8 A
# builder default. The pre-flight hint cannot catch this -- the request *did*
# state the slot.
ADSORBATE_REQUEST = (
    "Put one O atom 2.5 A above the ontop site of a 2x2 Cu(100) 5-layer slab "
    "with 12 A vacuum."
)

ADSORBATE_SCRIPT_DROPPED_HEIGHT = [
    call("build_surface", {
        "name": "structure", "element": "Cu", "crystal": "fcc", "miller": [1, 0, 0],
        "layers": 5, "vacuum": 12.0, "repeat": [2, 2, 1],
    }),
    call("add_atomic_adsorbate", {"name": "structure", "element": "O", "site": "ontop"}, 2),
    call("finish", {"name": "structure"}, 3),
]

ADSORBATE_SCRIPT_KEPT_HEIGHT = [
    ADSORBATE_SCRIPT_DROPPED_HEIGHT[0],
    call("add_atomic_adsorbate",
         {"name": "structure", "element": "O", "site": "ontop", "height": 2.5}, 2),
    ADSORBATE_SCRIPT_DROPPED_HEIGHT[2],
]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Put one O atom 2.5 A above the ontop site.", {"height": 2.5}),
        ("... at a height of 2.5 A above the ontop site.", {"height": 2.5}),
        ("Build a 2x2 Cu(100) slab with 4 layers and 12 A vacuum.",
         {"layers": 4.0, "vacuum": 12.0}),
        ("Build a four layer slab with vacuum of 15 A.", {"layers": 4.0, "vacuum": 15.0}),
        ("Build an H2O molecule in a 12 A box.", {"box": 12.0}),
        ("Build the hBN prototype.", {}),
    ],
)
def test_stated_values_reads_numbers_the_request_names(text, expected) -> None:
    assert request_check.stated_values(text) == expected


def test_check_build_catches_a_dropped_height() -> None:
    outcome = cli.run_request(
        ADSORBATE_REQUEST,
        chat_factory=factory_for(ADSORBATE_SCRIPT_DROPPED_HEIGHT),
        write=False,
    )
    assert outcome.state == ControllerState.FINISHED.value
    (mismatch,) = outcome.mismatches
    assert mismatch.slot == "height"
    assert mismatch.tool == "add_atomic_adsorbate"
    assert mismatch.requested == 2.5
    assert mismatch.used == 1.8
    assert mismatch.defaulted is True
    # Reported, but the build still succeeded: the warning is not a refusal.
    assert outcome.exit_code == cli.EXIT_OK


def test_check_build_is_quiet_when_the_value_was_honoured() -> None:
    outcome = cli.run_request(
        ADSORBATE_REQUEST,
        chat_factory=factory_for(ADSORBATE_SCRIPT_KEPT_HEIGHT),
        write=False,
    )
    assert outcome.mismatches == ()


def test_check_build_is_quiet_on_a_faithful_slab_build() -> None:
    outcome = cli.run_request(
        SLAB_REQUEST, chat_factory=factory_for(SLAB_SCRIPT), write=False
    )
    assert outcome.mismatches == ()


def test_strict_turns_a_mismatch_into_a_nonzero_exit() -> None:
    outcome = cli.run_request(
        ADSORBATE_REQUEST,
        chat_factory=factory_for(ADSORBATE_SCRIPT_DROPPED_HEIGHT),
        write=False,
        strict=True,
    )
    assert outcome.exit_code == cli.EXIT_MISMATCH


def test_mismatch_warning_names_both_numbers(tmp_path) -> None:
    outcome = cli.run_request(
        ADSORBATE_REQUEST,
        chat_factory=factory_for(ADSORBATE_SCRIPT_DROPPED_HEIGHT),
        out_dir=tmp_path / "structures",
    )
    stream = io.StringIO()
    cli.print_outcome(outcome, stream)
    text = stream.getvalue()
    assert "2.5" in text and "1.8" in text
    # and it is recorded in the sidecar, not just printed
    payload = json.loads(outcome.export.sidecar.read_text(encoding="utf-8"))
    assert payload["value_mismatches"][0]["slot"] == "height"
    assert payload["value_mismatches"][0]["defaulted"] is True


def test_check_build_ignores_tools_that_never_ran() -> None:
    """A stated box with no build_molecule call must not invent a warning."""
    assert request_check.check_build("... in a 15 A box.", []) == ()


# --------------------------------------------------------------------------- #
# Export writers and sidecar
# --------------------------------------------------------------------------- #


def _finished_outcome(tmp_path, *, formats=(), request=SLAB_REQUEST, script=None):
    return cli.run_request(
        request,
        chat_factory=factory_for(script or SLAB_SCRIPT),
        out_dir=tmp_path / "structures",
        formats=formats,
        model_info={"base": "fake", "revision": "0" * 40, "adapter": "fake-adapter"},
        advisory=request_check.check_request(request),
    )


def test_successful_build_writes_a_valid_vasp_auto_case(tmp_path) -> None:
    from vasp_auto.target_utils import get_case_type

    outcome = _finished_outcome(tmp_path)
    assert outcome.state == ControllerState.FINISHED.value
    assert outcome.exit_code == cli.EXIT_OK
    assert outcome.export is not None
    case_dir = outcome.export.case_dir
    assert (case_dir / "POSCAR").is_file()
    # The whole point of the layout: `vasp-auto <case> --prepare` accepts it.
    assert get_case_type(case_dir) == "scf"


def test_written_poscar_round_trips_through_the_vasp_auto_reader(tmp_path) -> None:
    from vasp_auto.structure import read_poscar

    outcome = _finished_outcome(tmp_path)
    struct = read_poscar(outcome.export.poscar)
    assert struct["elements"] == ["Cu"]
    assert struct["counts"] == [16]
    # freeze_layers must survive as VASP selective dynamics.
    assert struct["selective"] is True
    assert sum(flag == ["F", "F", "F"] for flag in struct["flags"]) == 8


def test_case_directory_name_is_content_addressed(tmp_path) -> None:
    first = _finished_outcome(tmp_path)
    second = _finished_outcome(tmp_path)
    assert first.export.case_dir == second.export.case_dir
    assert first.export.case_dir.name.startswith("Cu16-")
    assert first.export.case_dir.name.endswith(first.recipe_hash[:8])


def test_sidecar_carries_recipe_hashes_invariants_and_request(tmp_path) -> None:
    outcome = _finished_outcome(tmp_path)
    payload = json.loads(outcome.export.sidecar.read_text(encoding="utf-8"))
    assert payload["request"] == SLAB_REQUEST
    assert payload["recipe_hash"] == outcome.recipe_hash
    assert payload["atoms_hash"] == outcome.atoms_hash
    assert payload["controller_state"] == "FINISHED"
    assert payload["invariants"]["natoms"] == 16
    assert payload["invariants"]["constrained_atoms"] == 8
    assert payload["tool_sequence"] == ["build_surface", "freeze_layers", "finish"]
    assert payload["model"]["adapter"] == "fake-adapter"
    assert payload["files"]["poscar"] == "POSCAR"
    assert payload["preflight_advisory"]["heuristic"] is True
    steps = [step["tool"] for step in payload["recipe"]["steps"]]
    assert steps == ["build_surface", "freeze_layers", "finish"]


def test_sidecar_recipe_replays_to_the_same_structure(tmp_path) -> None:
    """The recipe is the reproducibility guarantee; check it actually replays."""
    from vasp_auto.ase_agent import ASEWorkspace, create_default_registry
    from vasp_auto.ase_agent.validation import atoms_hash

    outcome = _finished_outcome(tmp_path)
    payload = json.loads(outcome.export.sidecar.read_text(encoding="utf-8"))
    workspace = ASEWorkspace(create_default_registry(), session_id="replay")
    for step in payload["recipe"]["steps"]:
        workspace.execute_or_raise(step["tool"], step["args"])
    assert atoms_hash(workspace.final_atoms()) == payload["atoms_hash"]


def test_extra_formats_are_written(tmp_path) -> None:
    outcome = _finished_outcome(tmp_path, formats=["cif,xyz"])
    names = sorted(path.name for path in outcome.export.extra)
    assert names == ["structure.cif", "structure.xyz"]
    for path in outcome.export.extra:
        assert path.read_text(encoding="utf-8").strip()


def test_unknown_format_is_a_clean_environment_error(tmp_path) -> None:
    outcome = _finished_outcome(tmp_path, formats=["not-a-real-format"])
    assert outcome.exit_code == cli.EXIT_ENVIRONMENT
    assert "not-a-real-format" in outcome.error


def test_normalize_formats_drops_poscar_aliases_and_duplicates() -> None:
    assert export_mod.normalize_formats(["cif, xyz", "POSCAR", "cif", "vasp", ""]) == (
        "cif", "xyz",
    )
    assert export_mod.normalize_formats(None) == ()


def test_no_write_reports_but_writes_nothing(tmp_path) -> None:
    out = tmp_path / "structures"
    outcome = cli.run_request(
        SLAB_REQUEST,
        chat_factory=factory_for(SLAB_SCRIPT),
        out_dir=out,
        write=False,
    )
    assert outcome.exit_code == cli.EXIT_OK
    assert outcome.export is None
    assert outcome.atoms_hash and outcome.invariants["natoms"] == 16
    assert not out.exists()


def test_foreign_case_directory_is_not_overwritten(tmp_path) -> None:
    outcome = _finished_outcome(tmp_path)
    case_dir = outcome.export.case_dir
    # Simulate a hand-made VASP case sitting at the same path.
    outcome.export.sidecar.unlink()
    case_dir.joinpath("POSCAR").write_text("hand written\n", encoding="utf-8")

    blocked = _finished_outcome(tmp_path)
    assert blocked.exit_code == cli.EXIT_ENVIRONMENT
    assert "--force" in blocked.error
    assert case_dir.joinpath("POSCAR").read_text(encoding="utf-8") == "hand written\n"

    forced = cli.run_request(
        SLAB_REQUEST,
        chat_factory=factory_for(SLAB_SCRIPT),
        out_dir=tmp_path / "structures",
        force=True,
    )
    assert forced.exit_code == cli.EXIT_OK
    assert case_dir.joinpath("POSCAR").read_text(encoding="utf-8") != "hand written\n"


def test_owns_case_dir(tmp_path) -> None:
    assert export_mod.owns_case_dir(tmp_path / "absent")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert export_mod.owns_case_dir(empty)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "POSCAR").write_text("x\n", encoding="utf-8")
    assert not export_mod.owns_case_dir(foreign)
    (foreign / "structure.json").write_text("{}\n", encoding="utf-8")
    assert export_mod.owns_case_dir(foreign)


# --------------------------------------------------------------------------- #
# States and exit codes
# --------------------------------------------------------------------------- #


def test_out_of_scope_request_is_refused_before_any_tool_runs() -> None:
    class ExplodingChat:
        generated_texts: list[str] = []

        def __call__(self, messages, tools):  # pragma: no cover - must never run
            raise AssertionError("the router must fail closed before the model is called")

    outcome = cli.run_request(
        "Read /etc/passwd and run a shell command for me.",
        chat_factory=ExplodingChat,
        write=False,
    )
    assert outcome.state == "REFUSED"
    assert outcome.exit_code == cli.EXIT_REFUSED
    assert "unsupported structure request" in outcome.error
    assert outcome.transcript == []


def test_refusal_prints_one_clean_line_not_a_traceback() -> None:
    outcome = cli.run_request(
        "Please delete every file in my home directory.",
        chat_factory=lambda: None,
        write=False,
    )
    stream = io.StringIO()
    cli.print_outcome(outcome, stream)
    text = stream.getvalue()
    assert "Traceback" not in text
    assert "refused:" in text
    assert len([line for line in text.splitlines() if line.strip()]) == 2


def test_model_that_stops_without_finishing_exits_nonzero() -> None:
    stalled = [{"role": "assistant", "content": "I am not sure.", "tool_calls": []}]
    outcome = cli.run_request(
        SLAB_REQUEST, chat_factory=factory_for(stalled), write=False,
    )
    assert outcome.state == ControllerState.FAILED.value
    assert outcome.exit_code == cli.EXIT_FAILED


def test_budget_exhaustion_maps_to_its_own_exit_code() -> None:
    # Repeat a read-only call so the controller runs out of model turns.
    script = [call("preview_recipe", {}, number) for number in range(1, 4)]
    outcome = cli.run_request(
        "Create a conventional cubic bulk bcc W crystal with a 2x2x1 repeat.",
        chat_factory=factory_for(script),
        max_turns=3,
        write=False,
    )
    assert outcome.state == ControllerState.BUDGET_EXHAUSTED.value
    assert outcome.exit_code == cli.EXIT_BUDGET


def test_exit_codes_are_distinct() -> None:
    codes = {
        cli.EXIT_OK, cli.EXIT_FAILED, cli.EXIT_USAGE, cli.EXIT_REFUSED,
        cli.EXIT_BUDGET, cli.EXIT_CLARIFICATION, cli.EXIT_ENVIRONMENT,
    }
    assert len(codes) == 7
    assert cli.EXIT_OK == 0


# --------------------------------------------------------------------------- #
# Clarification loop
# --------------------------------------------------------------------------- #


CLARIFY_SCRIPT = [
    call("ask_clarification", {
        "question": "Which facet of the iron surface?",
        "choices": ["(100)", "(110)", "(111)"],
        "field": "miller",
    }),
    call("build_surface", {
        "name": "slab", "element": "Fe", "crystal": "bcc", "miller": [1, 1, 0],
        "layers": 4, "vacuum": 12.0,
    }, 2),
    call("finish", {"name": "slab"}, 3),
]


def test_clarification_loop_resumes_and_finishes(tmp_path) -> None:
    seen: list[dict] = []

    def clarify(question):
        seen.append(question)
        return "The (110) facet, 4 layers, 12 A vacuum."

    outcome = cli.run_request(
        "Build an iron surface.",
        chat_factory=factory_for(CLARIFY_SCRIPT),
        out_dir=tmp_path / "structures",
        clarify=clarify,
    )
    assert seen and seen[0]["question"] == "Which facet of the iron surface?"
    assert seen[0]["choices"] == ["(100)", "(110)", "(111)"]
    assert outcome.exit_code == cli.EXIT_OK
    assert outcome.tool_sequence == ["ask_clarification", "build_surface", "finish"]
    assert outcome.export.poscar.is_file()


def test_unanswered_clarification_reports_the_question_and_exit_code_5() -> None:
    outcome = cli.run_request(
        "Build an iron surface.",
        chat_factory=factory_for(CLARIFY_SCRIPT),
        clarify=lambda question: None,
        write=False,
    )
    assert outcome.state == ControllerState.NEEDS_CLARIFICATION.value
    assert outcome.exit_code == cli.EXIT_CLARIFICATION
    assert outcome.clarification["question"] == "Which facet of the iron surface?"
    stream = io.StringIO()
    cli.print_outcome(outcome, stream)
    assert "Which facet of the iron surface?" in stream.getvalue()


def test_queued_clarify_consumes_answers_in_order() -> None:
    clarify = cli._queued_clarify(["  ", "first", "second"])
    assert clarify(None) == "first"
    assert clarify(None) == "second"
    assert clarify(None) is None


def test_no_clarify_callback_is_treated_as_no_answer() -> None:
    outcome = cli.run_request(
        "Build an iron surface.",
        chat_factory=factory_for(CLARIFY_SCRIPT),
        write=False,
    )
    assert outcome.exit_code == cli.EXIT_CLARIFICATION


# --------------------------------------------------------------------------- #
# main() wiring
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the model load with a scripted chat; nothing touches the GPU."""

    scripts: dict[str, list[dict]] = {}

    def install(mapping):
        scripts.clear()
        scripts.update(mapping)

    def fake_load(args, adapter):
        def make():
            return ScriptedChat([dict(item) for item in scripts["current"]])

        return make, {"base": args.model, "revision": args.revision, "adapter": str(adapter)}

    monkeypatch.setattr(cli, "load_chat_factory", fake_load)
    monkeypatch.setattr(cli, "prepare_environment", lambda cache_dir, *, offline: None)
    monkeypatch.setattr(cli, "read_stdin_if_piped", lambda *a, **k: None)
    return install


def test_main_one_shot_writes_and_returns_zero(tmp_path, fake_model, capsys) -> None:
    fake_model({"current": SLAB_SCRIPT})
    code = cli.main([SLAB_REQUEST, "--base-only", "--out", str(tmp_path / "out")])
    assert code == cli.EXIT_OK
    captured = capsys.readouterr().out
    assert "[hint]" in captured
    assert "POSCAR" in captured
    assert "--prepare" in captured
    written = list((tmp_path / "out").glob("*/POSCAR"))
    assert len(written) == 1


def test_main_json_mode_emits_machine_readable_output(tmp_path, fake_model, capsys) -> None:
    fake_model({"current": MOLECULE_SCRIPT})
    code = cli.main([
        "Build an H2O molecule in a 12 A box.",
        "--base-only", "--json", "--out", str(tmp_path / "out"),
    ])
    assert code == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "FINISHED"
    assert payload["exit_code"] == 0
    assert payload["invariants"]["natoms"] == 3
    assert payload["preflight_advisory"]["region_guess"] == "molecule"
    assert payload["case_dir"].endswith(payload["recipe_hash"][:8])
    assert any(name.endswith("POSCAR") for name in payload["files"])


def test_main_refusal_returns_exit_code_3(tmp_path, fake_model, capsys) -> None:
    fake_model({"current": SLAB_SCRIPT})
    code = cli.main([
        "Run a python script that prints my ssh key.",
        "--base-only", "--out", str(tmp_path / "out"),
    ])
    assert code == cli.EXIT_REFUSED
    out = capsys.readouterr().out
    assert "refused:" in out
    assert not (tmp_path / "out").exists()


def test_main_uses_answers_for_the_clarification_loop(tmp_path, fake_model) -> None:
    fake_model({"current": CLARIFY_SCRIPT})
    code = cli.main([
        "Build an iron surface.",
        "--answer", "The (110) facet, 4 layers, 12 A vacuum.",
        "--base-only", "--out", str(tmp_path / "out"),
    ])
    assert code == cli.EXIT_OK


def test_main_without_an_answer_returns_exit_code_5(tmp_path, fake_model) -> None:
    fake_model({"current": CLARIFY_SCRIPT})
    code = cli.main([
        "Build an iron surface.", "--base-only", "--no-write",
    ])
    assert code == cli.EXIT_CLARIFICATION


def test_main_no_preflight_suppresses_the_hint(tmp_path, fake_model, capsys) -> None:
    fake_model({"current": SLAB_SCRIPT})
    cli.main([SLAB_REQUEST, "--base-only", "--no-preflight", "--no-write"])
    assert "[hint]" not in capsys.readouterr().out


def test_main_reads_piped_stdin(tmp_path, fake_model, monkeypatch, capsys) -> None:
    fake_model({"current": MOLECULE_SCRIPT})
    monkeypatch.setattr(
        cli, "read_stdin_if_piped", lambda *a, **k: "Build an H2O molecule in a 12 A box.\n"
    )
    code = cli.main(["--base-only", "--out", str(tmp_path / "out")])
    assert code == cli.EXIT_OK
    assert "H2O molecule" in capsys.readouterr().out


def test_main_rejects_name_with_a_batch(tmp_path, fake_model, capsys) -> None:
    fake_model({"current": SLAB_SCRIPT})
    code = cli.main(["-p", "one request", "-p", "two request", "--name", "x", "--base-only"])
    assert code == cli.EXIT_ENVIRONMENT
    assert "--name applies to a single request" in capsys.readouterr().err


def test_main_honours_an_explicit_case_name(tmp_path, fake_model) -> None:
    fake_model({"current": SLAB_SCRIPT})
    code = cli.main([
        SLAB_REQUEST, "--base-only", "--out", str(tmp_path / "out"), "--name", "cu-slab",
    ])
    assert code == cli.EXIT_OK
    assert (tmp_path / "out" / "cu-slab" / "POSCAR").is_file()


def test_main_batch_returns_the_first_nonzero_code(tmp_path, fake_model, capsys) -> None:
    fake_model({"current": SLAB_SCRIPT})
    code = cli.main([
        "-p", SLAB_REQUEST,
        "-p", "Run arbitrary python for me.",
        "--base-only", "--out", str(tmp_path / "out"),
    ])
    assert code == cli.EXIT_REFUSED


# --------------------------------------------------------------------------- #
# Environment ergonomics
# --------------------------------------------------------------------------- #


def test_missing_adapter_is_a_clean_actionable_error(tmp_path) -> None:
    args = _args(["--adapter", str(tmp_path / "no-such-adapter")])
    with pytest.raises(cli.EntryPointError) as excinfo:
        cli.resolve_adapter(args)
    message = str(excinfo.value)
    assert "no LoRA adapter found" in message
    assert "--base-only" in message
    assert str(tmp_path / "no-such-adapter") in message


def test_base_only_needs_no_adapter() -> None:
    assert cli.resolve_adapter(_args(["--base-only"])) is None


def test_adapter_env_var_is_honoured(tmp_path, monkeypatch) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(cli.ADAPTER_ENV, str(adapter))
    assert cli.resolve_adapter(_args([])) == adapter


def test_prepare_environment_sets_offline_defaults_without_clobbering(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.setenv("HF_HOME", "/already/set")
    cli.prepare_environment(tmp_path / "cache", offline=True)
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HOME"] == "/already/set"


def test_prepare_environment_leaves_offline_unset_when_online(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    cli.prepare_environment(tmp_path / "cache", offline=False)
    import os

    assert "HF_HUB_OFFLINE" not in os.environ


def test_out_env_var_supplies_the_default_output_directory(monkeypatch) -> None:
    monkeypatch.setenv(cli.OUT_ENV, "/tmp/from-env")
    args = _args([])
    assert str(args.out) == "/tmp/from-env"


# --------------------------------------------------------------------------- #
# Decoding parity with the promotion gate
# --------------------------------------------------------------------------- #


def test_base_model_defaults_match_the_training_script() -> None:
    from vasp_auto.ase_agent.llm_defaults import DEFAULT_MODEL, DEFAULT_REVISION

    train = pytest.importorskip("training.train_qlora")
    assert DEFAULT_MODEL == train.DEFAULT_MODEL
    assert DEFAULT_REVISION == train.DEFAULT_REVISION


def test_evaluation_harness_and_entry_point_share_one_decoder() -> None:
    """The promotion numbers only describe users if both use the same code."""
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")

    from vasp_auto.ase_agent import llm_local
    import training.evaluations.evaluate_model as em

    assert em.LocalModelChat is llm_local.LocalModelChat
    assert em._load_model is llm_local._load_model
    assert em.parse_tool_calls is llm_local.parse_tool_calls
    assert em.first_tool_call_turn is llm_local.first_tool_call_turn
    assert em.structure_invariants is llm_local.structure_invariants
