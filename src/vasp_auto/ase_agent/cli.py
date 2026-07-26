"""``ASE_auto_build`` -- describe a structure in plain language, get files back.

The user-facing front door to the fine-tuned ASE agent. You type (or pipe) a
natural-language structure request; the model plans a sequence of deterministic
ASE tool calls; a bounded workspace executes and validates every one of them;
and the finished structure is written to disk as a ready-to-run VASP case.

    ASE_auto_build "Build a 2x2 Cu(100) slab with 4 layers and 12 A vacuum"
    -> structures/Cu16-1a2b3c4d/POSCAR
       structures/Cu16-1a2b3c4d/structure.json
    -> vasp-auto structures/Cu16-1a2b3c4d --prepare

Why the files matter: the output directory *is* a ``vasp-auto`` SCF case (a
directory containing a ``POSCAR``), so the built structure feeds straight into
the DFT pipeline. The JSON sidecar carries the deterministic recipe, so the same
structure can be rebuilt later without the model.

Safety: the model never receives code, shell, filesystem, or network tools. It
selects from a fixed registry; the deterministic router narrows that registry per
request and *fails closed* on anything it cannot route, before a single tool runs.
Output paths are chosen by the user and this module, never by the model.

Decoding parity: the model is loaded and its tool calls parsed by
``vasp_auto.ase_agent.llm_local``, the same module the promotion gate
(``training/evaluations/evaluate_model.py``) uses -- so the measured exact-match
rate describes what this command does.

Exit codes (scriptable):

===  =========================================================
0    FINISHED -- a structure was built (and written)
1    FAILED -- the model gave up or the controller stopped it
2    command-line usage error (argparse)
3    REFUSED -- out of scope; the router fail-closed before any tool ran
4    BUDGET_EXHAUSTED -- turn/error budget hit
5    NEEDS_CLARIFICATION with no answer available (non-interactive)
6    environment problem (no CUDA, missing adapter, bad --format, ...)
7    --strict only: built, but a number you stated was not used
===  =========================================================

With several requests in one run, the first non-zero code wins.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import export as export_mod
from .controller import AgentController, ControllerResult, ControllerState
from .policy import AgentPolicy
from .request_check import RequestAdvisory, ValueMismatch, check_build, check_request
from .tools import create_default_registry
from .validation import atoms_hash, structure_invariants
from .workspace import ASEWorkspace

PROGRAM = "ASE_auto_build"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3
EXIT_BUDGET = 4
EXIT_CLARIFICATION = 5
EXIT_ENVIRONMENT = 6
EXIT_MISMATCH = 7

DEFAULT_OUT = Path("structures")
DEFAULT_ADAPTER_RELATIVE = Path("training/runs/pilot-qwen3-4b-r5/adapter")
DEFAULT_CACHE_RELATIVE = Path("training/cache/huggingface")

ADAPTER_ENV = "ASE_AUTO_BUILD_ADAPTER"
LEGACY_ADAPTER_ENV = "VASP_AUTO_ASE_ADAPTER"
OUT_ENV = "ASE_AUTO_BUILD_OUT"

_STATE_EXIT = {
    ControllerState.FINISHED: EXIT_OK,
    ControllerState.FAILED: EXIT_FAILED,
    ControllerState.BUDGET_EXHAUSTED: EXIT_BUDGET,
    ControllerState.NEEDS_CLARIFICATION: EXIT_CLARIFICATION,
    ControllerState.NEEDS_CONFIRMATION: EXIT_CLARIFICATION,
}


class EntryPointError(RuntimeError):
    """An environment/usage problem to report as one clean line, never a traceback."""


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #


def split_requests(text: str) -> list[str]:
    """Split a description file or stdin blob into requests.

    Rule (documented in ``--help``): ``#`` comment lines are dropped, blank lines
    separate requests, and each request's internal newlines collapse to spaces.
    A one-line file is therefore simply one request.
    """
    blocks: list[str] = []
    current: list[str] = []
    for line in (text or "").splitlines():
        if line.lstrip().startswith("#"):
            continue
        if not line.strip():
            if current:
                blocks.append(" ".join(current))
                current = []
            continue
        current.append(line.strip())
    if current:
        blocks.append(" ".join(current))
    return [block for block in (" ".join(b.split()) for b in blocks) if block]


def collect_requests(args, *, stdin_text: str | None) -> list[str]:
    """Assemble the ordered request list from positional/--prompt/--file/stdin."""
    requests: list[str] = []
    positional = " ".join(args.description).strip()
    if positional:
        requests.append(" ".join(positional.split()))
    for prompt in args.prompt or ():
        cleaned = " ".join(str(prompt).split())
        if cleaned:
            requests.append(cleaned)
    for path in args.file or ():
        target = Path(path)
        try:
            content = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise EntryPointError(f"cannot read --file {target}: {exc.strerror or exc}") from exc
        found = split_requests(content)
        if not found:
            raise EntryPointError(f"--file {target} contains no request text")
        requests.extend(found)
    if stdin_text:
        requests.extend(split_requests(stdin_text))
    return requests


def read_stdin_if_piped(stdin=None) -> str | None:
    """Return piped stdin text, or None when stdin is a terminal/unavailable."""
    stream = sys.stdin if stdin is None else stdin
    if stream is None:
        return None
    try:
        if stream.isatty():
            return None
    except Exception:
        return None
    try:
        return stream.read()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# One request
# --------------------------------------------------------------------------- #


@dataclass
class BuildOutcome:
    request: str
    state: str
    exit_code: int
    error: str | None = None
    advisory: RequestAdvisory | None = None
    mismatches: tuple[ValueMismatch, ...] = ()
    export: export_mod.ExportResult | None = None
    invariants: dict[str, Any] | None = None
    atoms_hash: str | None = None
    recipe: dict[str, Any] | None = None
    recipe_hash: str | None = None
    tool_sequence: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    model_turns: list[str] = field(default_factory=list)
    clarification: dict[str, Any] | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "state": self.state,
            "exit_code": self.exit_code,
            "error": self.error,
            "atoms_hash": self.atoms_hash,
            "recipe_hash": self.recipe_hash,
            "recipe": self.recipe,
            "invariants": self.invariants,
            "tool_sequence": self.tool_sequence,
            "clarification": self.clarification,
            "preflight_advisory": _advisory_json(self.advisory),
            "value_mismatches": _mismatch_json(self.mismatches),
            "files": (
                [str(path) for path in self.export.paths] if self.export else []
            ),
            "case_dir": str(self.export.case_dir) if self.export else None,
        }


def _advisory_json(advisory: RequestAdvisory | None) -> dict[str, Any] | None:
    if advisory is None:
        return None
    return {
        "heuristic": True,
        "region_guess": advisory.region,
        "required_slots": list(advisory.required),
        "slots_that_look_unstated": list(advisory.missing),
    }


def _mismatch_json(mismatches: Iterable[ValueMismatch]) -> list[dict[str, Any]]:
    return [
        {
            "slot": item.slot,
            "tool": item.tool,
            "argument": item.argument,
            "requested": item.requested,
            "used": item.used,
            "defaulted": item.defaulted,
        }
        for item in mismatches
    ]


def _tool_sequence(transcript: Iterable[dict[str, Any]]) -> list[str]:
    return [item["tool"] for item in transcript if item["success"]]


def run_request(
    request: str,
    *,
    chat_factory: Callable[[], Any],
    out_dir: Path | None = None,
    case_name: str | None = None,
    formats: Sequence[str] = (),
    max_turns: int = 6,
    max_new_tokens: int = 256,
    clarify: Callable[[dict[str, Any] | None], str | None] | None = None,
    write: bool = True,
    force: bool = False,
    strict: bool = False,
    model_info: dict[str, Any] | None = None,
    advisory: RequestAdvisory | None = None,
) -> BuildOutcome:
    """Drive one request through a fresh workspace and (optionally) write files.

    ``chat_factory`` returns a *fresh* chat callable per request; tests pass a
    scripted fake, ``main`` passes a ``LocalModelChat`` bound to the loaded model.
    """
    registry = create_default_registry()
    policy = AgentPolicy(max_model_turns=max_turns, max_tool_calls=max_turns)
    workspace = ASEWorkspace(registry, policy=policy, session_id="ASE_auto_build")
    chat = chat_factory()
    controller = AgentController(workspace, chat)

    try:
        result: ControllerResult = controller.run(request)
        while result.state is ControllerState.NEEDS_CLARIFICATION:
            question = workspace.pending_clarification
            answer = clarify(question) if clarify is not None else None
            if not answer:
                return BuildOutcome(
                    request=request,
                    state=ControllerState.NEEDS_CLARIFICATION.value,
                    exit_code=EXIT_CLARIFICATION,
                    error="the model asked a clarifying question and no answer was available",
                    advisory=advisory,
                    clarification=question,
                    transcript=list(result.transcript),
                    model_turns=list(getattr(chat, "generated_texts", []) or []),
                    tool_sequence=_tool_sequence(result.transcript),
                )
            result = controller.resume(answer)
    except Exception as exc:  # fail-closed refusal is an expected outcome, not a crash
        return BuildOutcome(
            request=request,
            state="REFUSED",
            exit_code=EXIT_REFUSED,
            error=f"{type(exc).__name__}: {exc}",
            advisory=advisory,
            model_turns=list(getattr(chat, "generated_texts", []) or []),
        )

    outcome = BuildOutcome(
        request=request,
        state=result.state.value,
        exit_code=_STATE_EXIT.get(result.state, EXIT_FAILED),
        error=result.error,
        advisory=advisory,
        transcript=list(result.transcript),
        model_turns=list(getattr(chat, "generated_texts", []) or []),
        tool_sequence=_tool_sequence(result.transcript),
    )
    if result.state is not ControllerState.FINISHED:
        return outcome

    atoms = workspace.final_atoms()
    recipe = workspace.recipe()
    digest = workspace.recipe_hash()
    outcome.recipe = recipe
    outcome.recipe_hash = digest
    outcome.atoms_hash = atoms_hash(atoms)
    outcome.invariants = structure_invariants(atoms)
    # Deterministic: did the build use the numbers the request actually stated?
    outcome.mismatches = check_build(request, outcome.transcript)
    if outcome.mismatches and strict:
        outcome.exit_code = EXIT_MISMATCH
    if not write:
        return outcome

    base = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    target = base / (case_name or export_mod.case_name(atoms, digest))
    if not force and not export_mod.owns_case_dir(target):
        outcome.exit_code = EXIT_ENVIRONMENT
        outcome.error = (
            f"{target} already holds a POSCAR that {PROGRAM} did not write; "
            "pass --force to overwrite or choose another --name/--out"
        )
        return outcome
    try:
        outcome.export = export_mod.write_bundle(
            atoms,
            target,
            request=request,
            recipe=recipe,
            recipe_hash=digest,
            formats=formats,
            tool_sequence=outcome.tool_sequence,
            model_info=model_info,
            controller_state=outcome.state,
            advisory=_advisory_json(advisory),
            mismatches=_mismatch_json(outcome.mismatches),
        )
    except export_mod.ExportError as exc:
        outcome.exit_code = EXIT_ENVIRONMENT
        outcome.error = str(exc)
    return outcome


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #


def _formula(invariants: dict[str, Any]) -> str:
    return "".join(f"{el}{n}" for el, n in invariants["formula"].items())


def _fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


def print_advisory(advisory: RequestAdvisory, stream=None) -> None:
    stream = sys.stdout if stream is None else stream
    for line in advisory.lines():
        print(line, file=stream)


def print_outcome(outcome: BuildOutcome, stream=None, *, verbose: bool = False) -> None:
    # Resolve sys.stdout at call time: a default argument would bind the stream
    # captured at import, which breaks redirection.
    stream = sys.stdout if stream is None else stream
    print(file=stream)
    if outcome.state == "REFUSED":
        print(f"  refused: {outcome.error}", file=stream)
        print("  (the deterministic router fail-closed; no tool was executed)", file=stream)
        print(file=stream)
        return

    print(f"  state            : {outcome.state}", file=stream)
    if outcome.error:
        print(f"  note             : {outcome.error}", file=stream)
    if outcome.clarification:
        question = outcome.clarification.get("question") or "(no question text)"
        print(f"  question         : {question}", file=stream)
        choices = outcome.clarification.get("choices") or []
        if choices:
            print(f"  choices          : {', '.join(str(c) for c in choices)}", file=stream)

    if verbose:
        for number, text in enumerate(outcome.model_turns, start=1):
            snippet = " ".join(text.split())
            if len(snippet) > 220:
                snippet = snippet[:220] + " ..."
            print(f"  model turn {number:>2}    : {snippet or '(empty)'}", file=stream)

    if outcome.transcript:
        print("  executed calls   :", file=stream)
        for number, item in enumerate(outcome.transcript, start=1):
            status = "ok " if item["success"] else "ERR"
            print(f"    {number}. [{status}] {item['tool']}({_fmt_args(item['args'])})", file=stream)
            if not item["success"]:
                print(f"          -> {item['result']}", file=stream)
    else:
        print("  executed calls   : (none)", file=stream)

    for mismatch in outcome.mismatches:
        print(f"  {mismatch.line()}", file=stream)
    if outcome.mismatches:
        print(
            "  [warn] restate it as e.g. 'at a height of 2.5 A' and rebuild, or edit "
            "the POSCAR; nothing was silently corrected.",
            file=stream,
        )

    if outcome.invariants:
        inv = outcome.invariants
        print("  final structure  :", file=stream)
        print(f"    formula        : {_formula(inv)}  ({inv['natoms']} atoms)", file=stream)
        print(f"    cell lengths A : {inv['cell_lengths']}   angles: {inv['cell_angles']}", file=stream)
        print(f"    pbc            : {inv['pbc']}   constrained atoms: {inv['constrained_atoms']}", file=stream)
        print(f"    atoms_hash     : {outcome.atoms_hash}", file=stream)
        print(f"    recipe_hash    : {outcome.recipe_hash}", file=stream)

    if outcome.export is not None:
        print("  wrote            :", file=stream)
        for path in outcome.export.paths:
            print(f"    {path}", file=stream)
        print(f"  next             : vasp-auto {outcome.export.case_dir} --prepare", file=stream)
    print(file=stream)


# --------------------------------------------------------------------------- #
# Environment / model
# --------------------------------------------------------------------------- #


def repo_root() -> Path:
    """The checkout this package was imported from (src/vasp_auto/ase_agent -> root)."""
    return Path(__file__).resolve().parents[3]


def default_cache_dir() -> Path | None:
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home)
    candidate = repo_root() / DEFAULT_CACHE_RELATIVE
    return candidate if candidate.is_dir() else None


def resolve_adapter(args) -> Path | None:
    """Locate the promoted adapter, or explain exactly where we looked."""
    if args.base_only:
        return None
    candidates: list[Path] = []
    if args.adapter is not None:
        candidates.append(Path(args.adapter))
    elif os.environ.get(ADAPTER_ENV) or os.environ.get(LEGACY_ADAPTER_ENV):
        # LEGACY_ADAPTER_ENV is the pre-rename name; still honoured so an
        # existing shell export keeps working.
        candidates.append(Path(
            os.environ.get(ADAPTER_ENV) or os.environ[LEGACY_ADAPTER_ENV]))
    else:
        candidates.append(repo_root() / DEFAULT_ADAPTER_RELATIVE)
        candidates.append(Path.cwd() / DEFAULT_ADAPTER_RELATIVE)
    seen: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if (candidate / "adapter_config.json").is_file():
            return candidate
        if resolved not in [path.resolve() for path in seen]:
            seen.append(candidate)
    looked = "\n".join(f"    {candidate}" for candidate in seen)
    raise EntryPointError(
        "no LoRA adapter found (looked for adapter_config.json in):\n"
        f"{looked}\n"
        f"  pass --adapter <dir>, set {ADAPTER_ENV}, or use --base-only to talk to "
        "the frozen base model."
    )


def prepare_environment(cache_dir: Path | None, *, offline: bool) -> None:
    """Set the HF/WSL environment before transformers is imported.

    Uses ``setdefault`` throughout, so an explicit environment always wins.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if cache_dir is not None:
        os.environ.setdefault("HF_HOME", str(cache_dir))
    # WSL keeps the CUDA driver stubs here; without it torch reports no device.
    wsl_lib = "/usr/lib/wsl/lib"
    if Path(wsl_lib).is_dir():
        path = os.environ.get("PATH", "")
        if wsl_lib not in path.split(os.pathsep):
            os.environ["PATH"] = os.pathsep.join(filter(None, (wsl_lib, path)))


def load_chat_factory(args, adapter: Path | None) -> tuple[Callable[[], Any], dict[str, Any]]:
    """Load the model once and return a factory for per-request chat wrappers."""
    try:
        from . import llm_local
    except ImportError as exc:
        raise EntryPointError(
            "the local model stack is not installed in this interpreter "
            f"({exc}). Install torch, transformers, peft, and bitsandbytes, or run "
            "with the project's training virtualenv."
        ) from exc

    import torch

    if not torch.cuda.is_available():
        raise EntryPointError(
            "no CUDA device is visible, and the 4-bit model needs one.\n"
            "  checks: `nvidia-smi` works; on WSL /usr/lib/wsl/lib is on PATH; "
            "the interpreter has a CUDA-enabled torch build."
        )

    label = "base (no adapter)" if adapter is None else str(adapter)
    print(f"Loading {args.model} @ {args.revision[:8]} ... adapter: {label}", flush=True)
    try:
        model, tokenizer = llm_local.load_model(
            model=args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
            adapter=adapter,
        )
    except Exception as exc:
        raise EntryPointError(
            f"could not load {args.model}: {type(exc).__name__}: {exc}\n"
            f"  cache dir: {args.cache_dir}\n"
            "  if the weights are not cached locally, re-run with --online."
        ) from exc
    print("Model ready.", flush=True)

    def factory():
        return llm_local.LocalModelChat(
            model, tokenizer, tool_override=None, max_new_tokens=args.max_new_tokens
        )

    return factory, {
        "base": args.model,
        "revision": args.revision,
        "adapter": None if adapter is None else str(adapter),
    }


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


EPILOG = """\
input modes (combined in this order):
  ASE_auto_build "Build an H2O molecule in a 12 A box."     one-shot, positional
  ASE_auto_build --prompt "..." --prompt "..."              one-shot, repeatable
  ASE_auto_build --file requests.txt                        a description file
  echo "..." | ASE_auto_build                               piped stdin
  ASE_auto_build                                            interactive REPL

  In a --file or on stdin, '#' lines are comments and a blank line starts a new
  request; a single-line file is simply one request.

output:
  Each successful build writes one directory that is already a valid vasp-auto
  SCF case -- POSCAR plus a structure.json sidecar (recipe, recipe_hash,
  atoms_hash, invariants, original request):

      structures/Cu16-1a2b3c4d/POSCAR
      structures/Cu16-1a2b3c4d/structure.json
      vasp-auto structures/Cu16-1a2b3c4d --prepare

checks:
  Before the model runs, a keyword hint reports required slots that look
  unstated -- advisory only. After it runs, a deterministic check compares the
  numbers you stated (layers, vacuum, box, height) against the calls actually
  executed, and warns when the structure did not use one. --strict turns that
  warning into exit 7.

exit codes:
  0 finished   1 failed   2 usage   3 refused (out of scope)
  4 budget exhausted   5 unanswered clarification   6 environment problem
  7 --strict: a stated number was not used
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Describe a structure in plain language; the fine-tuned ASE agent builds "
            "it with deterministic tools and writes a ready-to-run VASP case."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "description",
        nargs="*",
        default=[],
        help="The structure request, as plain words. Quote it if it contains shell "
             "characters such as parentheses.",
    )
    parser.add_argument(
        "-p", "--prompt", action="append", metavar="TEXT",
        help="A structure request; repeat for several requests in one model load.",
    )
    parser.add_argument(
        "-f", "--file", action="append", type=Path, metavar="PATH",
        help="Read request text from a file ('-' is not special; pipe stdin instead).",
    )
    parser.add_argument(
        "--answer", action="append", metavar="TEXT",
        help="Pre-supplied answer to a clarifying question, consumed in order. Lets "
             "the clarification loop run non-interactively.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "-o", "--out", type=Path, default=None, metavar="DIR",
        help=f"Directory that collects the case directories (default: ./{DEFAULT_OUT}, "
             f"or ${OUT_ENV}).",
    )
    output.add_argument(
        "--name", metavar="NAME",
        help="Case-directory name instead of the derived formula+hash name. Only "
             "valid with a single request.",
    )
    output.add_argument(
        "--format", action="append", metavar="FMT",
        help="Extra ase.io format(s) to write next to POSCAR, e.g. --format cif "
             "--format xyz, or --format cif,xyz. POSCAR is always written.",
    )
    output.add_argument(
        "--force", action="store_true",
        help="Overwrite a case directory that holds a POSCAR this tool did not write.",
    )
    output.add_argument(
        "--no-write", dest="write", action="store_false",
        help="Build and report, but write no files.",
    )
    output.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Print one JSON object per request on stdout instead of the human report.",
    )
    output.add_argument(
        "-v", "--verbose", action="store_true",
        help="Also print the model's raw generated turns.",
    )

    check = parser.add_argument_group("advisory pre-flight")
    check.add_argument(
        "--no-preflight", dest="preflight", action="store_false",
        help="Skip the keyword slot-coverage hint printed before the model runs.",
    )
    check.add_argument(
        "--strict", action="store_true",
        help="Exit 7 when the built structure does not use a number the request "
             "stated (e.g. an adsorption height the model dropped). The warning is "
             "printed either way; this only makes it fail a script.",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--adapter", type=Path, default=None, metavar="DIR",
        help=f"LoRA adapter directory (default: ${ADAPTER_ENV}, else "
             f"<repo>/{DEFAULT_ADAPTER_RELATIVE}).",
    )
    model.add_argument(
        "--base-only", action="store_true",
        help="Load the frozen base model without the adapter (A/B comparison).",
    )
    model.add_argument("--model", default=None, metavar="ID",
                       help="Base model id (default: the promoted Qwen3-4B-Instruct-2507).")
    model.add_argument("--revision", default=None, metavar="SHA",
                       help="Base model revision pinned during training.")
    model.add_argument(
        "--cache-dir", type=Path, default=None, metavar="DIR",
        help="Hugging Face cache (default: $HF_HOME, else <repo>/"
             f"{DEFAULT_CACHE_RELATIVE}).",
    )
    model.add_argument(
        "--online", action="store_true",
        help="Allow Hugging Face Hub network access; offline is the default.",
    )
    model.add_argument("--max-turns", type=int, default=6, metavar="N",
                       help="Model-turn and tool-call ceiling per request (default: 6).")
    model.add_argument("--max-new-tokens", type=int, default=256, metavar="N",
                       help="Generation ceiling per model turn (default: 256).")
    return parser


def _resolve_defaults(args) -> None:
    """Fill in defaults that need the environment (kept out of argparse)."""
    from .llm_defaults import DEFAULT_MODEL, DEFAULT_REVISION

    if args.model is None:
        args.model = DEFAULT_MODEL
    if args.revision is None:
        args.revision = DEFAULT_REVISION
    if args.out is None:
        args.out = Path(os.environ.get(OUT_ENV) or DEFAULT_OUT)


# --------------------------------------------------------------------------- #
# REPL
# --------------------------------------------------------------------------- #


def _interactive_clarify(question: dict[str, Any] | None, stream=None) -> str | None:
    stream = sys.stdout if stream is None else stream
    text = (question or {}).get("question") or "the model needs one more detail"
    print(f"  [needs clarification] {text}", file=stream)
    choices = (question or {}).get("choices") or []
    if choices:
        print(f"  choices: {', '.join(str(choice) for choice in choices)}", file=stream)
    try:
        answer = input("  clarify> ").strip()
    except (EOFError, KeyboardInterrupt):
        print(file=stream)
        return None
    return answer or None


def _queued_clarify(answers: list[str]):
    def clarify(question: dict[str, Any] | None) -> str | None:
        while answers:
            answer = answers.pop(0).strip()
            if answer:
                return answer
        return None

    return clarify


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def _emit(outcome: BuildOutcome, args) -> None:
    if args.as_json:
        print(json.dumps(outcome.as_json(), indent=2, sort_keys=True))
    else:
        print_outcome(outcome, verbose=args.verbose)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _resolve_defaults(args)

    try:
        formats = export_mod.normalize_formats(args.format)
        stdin_text = read_stdin_if_piped()
        requests = collect_requests(args, stdin_text=stdin_text)
        if args.name and len(requests) > 1:
            raise EntryPointError("--name applies to a single request; drop it for a batch")
        adapter = resolve_adapter(args)
        if args.cache_dir is None:
            args.cache_dir = default_cache_dir()
        prepare_environment(args.cache_dir, offline=not args.online)
        chat_factory, model_info = load_chat_factory(args, adapter)
    except EntryPointError as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return EXIT_ENVIRONMENT

    pending_answers = list(args.answer or [])

    def run(request: str, clarify) -> BuildOutcome:
        advisory = check_request(request) if args.preflight else None
        if advisory is not None and not args.as_json:
            print_advisory(advisory)
        return run_request(
            request,
            chat_factory=chat_factory,
            out_dir=args.out,
            case_name=args.name,
            formats=formats,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            clarify=clarify,
            write=args.write,
            force=args.force,
            strict=args.strict,
            model_info=model_info,
            advisory=advisory,
        )

    if requests:
        worst = EXIT_OK
        clarify = _queued_clarify(pending_answers)
        for request in requests:
            if not args.as_json:
                print(f"\nrequest> {request}")
            outcome = run(request, clarify)
            _emit(outcome, args)
            if worst == EXIT_OK:
                worst = outcome.exit_code
        return worst

    return _repl(run, args)


def _repl(run, args) -> int:
    print(
        f"{PROGRAM}: interactive mode. Describe a structure; 'quit' or Ctrl-D exits.\n"
        f"Finished structures are written under {args.out}/."
    )
    last = EXIT_OK
    while True:
        try:
            request = input("\nrequest> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n(interrupted; type 'quit' to exit)")
            continue
        if not request:
            continue
        if request.lower() in {"quit", "exit", ":q"}:
            break
        try:
            outcome = run(request, _interactive_clarify)
        except KeyboardInterrupt:
            print("\n(request interrupted)")
            continue
        _emit(outcome, args)
        last = outcome.exit_code
    return last


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
