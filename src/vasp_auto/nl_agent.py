"""Agentic natural-language structure builder ("the worker").

Where :mod:`nl_builder` turns one sentence into ONE JSON command, this module
runs a tool-calling loop: the model is given a toolbox of this project's own
structure primitives (build a bulk/slab/molecule/nanotube, make a supercell,
add an adsorbate, combine two structures) and calls them in sequence until the
requested structure is built. It composes steps, so open-ended requests like
"3x3 Pt(111) slab with CO on top and an O nearby" work — not just the six fixed
actions of the single-shot path.

The model still never emits atomic coordinates: every tool is a thin wrapper
over deterministic ASE / ``structure.py`` code, so geometry stays exact. The
call is the OpenAI-compatible *function calling* API over plain ``urllib`` (no
SDK), routed through :mod:`ai_providers` so any provider works (Groq by default;
OpenAI, OpenRouter, DeepSeek, a local server, ... by ``provider``/``base_url``).
Pick a capable model — tool chaining needs one that follows the function-calling
protocol well (Groq's ``llama-3.3-70b-versatile`` is the default; the tiny 8B
single-shot model is not reliable enough).
"""
from __future__ import annotations

import json

from . import ai_providers
from .nl_builder import (
    _atoms_to_struct,
    _build_base,
    _supercell,
    _topmost_index,
)
from .structure import (
    add_adsorbate,
    cell_lengths,
    combine_structures,
    make_supercell,
    read_poscar,
)

# Default agent model when none is selected (Groq's tool-capable 70B).
DEFAULT_AGENT_MODEL = ai_providers.PROVIDERS["groq"]["agent_model"]

SYSTEM_PROMPT = """You are a structure-building worker for VASP/ASE calculations.
The user describes a material in words; you BUILD it by calling the provided
tools, step by step. You never write atomic coordinates — the tools do all the
geometry exactly.

How to work:
- Start by building a base (bulk, surface, molecule, or nanotube). Give each
  structure a short name (e.g. "slab", "co").
- Modify or extend it with further tool calls (supercell, adsorbate, combine).
  Reuse a structure's name to refer to it again.
- A single adsorbed ATOM: build the slab, then add_adsorbate. A whole molecule
  on a slab: build the slab and the molecule separately, then combine them
  (mode="stack" to sit it above the surface, "insert" to drop it into the cell).
- Miller indices are always three integers, e.g. [1,0,0] for the (100) plane.
  A supercell is up to three integers, e.g. [3,3,1] for a 3x3 in-plane repeat.
- Pick sensible defaults when the user is vague (slab: 4 layers, 12 A vacuum;
  fcc/bcc/hcp as appropriate for the element).
- When the final structure is complete, call finish(name) exactly once with the
  name of the structure that is the answer. Do not call finish before the
  structure is fully built.
"""


# --- tool schemas (OpenAI-compatible function-calling format) ----------------
def _tool(name: str, description: str, properties: dict, required: list[str]):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_NAME = {"type": "string", "description": "Short name to store/refer to this structure."}
_MILLER = {
    "type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3,
    "description": "Miller indices, three integers e.g. [1,1,1].",
}
_SUPERCELL = {
    "type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3,
    "description": "Repeats per axis, e.g. [3,3,1].",
}

TOOLS = [
    _tool("build_bulk", "Build a bulk crystal of one element.", {
        "name": _NAME,
        "element": {"type": "string", "description": "Chemical symbol, e.g. Pt."},
        "crystal": {"type": "string", "enum": ["fcc", "bcc", "hcp", "diamond", "sc"],
                    "description": "Crystal structure (omit to let ASE choose)."},
        "a": {"type": "number", "description": "Lattice constant a in Angstrom (optional)."},
        "c": {"type": "number", "description": "Lattice constant c in Angstrom (optional)."},
        "supercell": _SUPERCELL,
    }, ["name", "element"]),
    _tool("build_surface", "Build a slab (surface) of one element.", {
        "name": _NAME,
        "element": {"type": "string"},
        "miller": _MILLER,
        "layers": {"type": "integer", "description": "Number of atomic layers (default 4)."},
        "vacuum": {"type": "number", "description": "Vacuum in Angstrom (default 12)."},
        "crystal": {"type": "string", "enum": ["fcc", "bcc", "hcp", "diamond", "sc"]},
        "a": {"type": "number"},
        "supercell": _SUPERCELL,
    }, ["name", "element", "miller"]),
    _tool("build_molecule", "Build a molecule by its ASE g2 name (CO, H2O, CO2, NH3, O2 ...).", {
        "name": _NAME,
        "species": {"type": "string", "description": "ASE g2 molecule name, e.g. CO."},
    }, ["name", "species"]),
    _tool("build_nanotube", "Build an (n,m) carbon nanotube.", {
        "name": _NAME,
        "n": {"type": "integer"}, "m": {"type": "integer"},
    }, ["name", "n", "m"]),
    _tool("make_supercell", "Repeat an existing structure into a supercell (modifies it in place).", {
        "name": {"type": "string", "description": "Name of the structure to repeat."},
        "repeat": _SUPERCELL,
    }, ["name", "repeat"]),
    _tool("add_adsorbate", "Add a single atom above a slab (default: above the topmost atom).", {
        "name": {"type": "string", "description": "Name of the slab to modify."},
        "element": {"type": "string", "description": "Adsorbate element, e.g. O."},
        "height": {"type": "number", "description": "Height above the anchor atom in Angstrom (default 1.8)."},
        "anchor_index": {"type": "integer", "description": "1-based atom to sit above (default: highest atom)."},
    }, ["name", "element"]),
    _tool("combine", "Combine two existing structures into one (host keeps its cell).", {
        "name": _NAME,
        "host": {"type": "string", "description": "Name of the host structure (keeps its cell)."},
        "guest": {"type": "string", "description": "Name of the guest structure placed onto/into the host."},
        "mode": {"type": "string", "enum": ["stack", "insert"], "description": "stack = above the surface; insert = into the cell."},
        "gap": {"type": "number", "description": "Gap above the host in Angstrom (default 2.0)."},
        "vacuum": {"type": "number", "description": "Vacuum above the guest in Angstrom (default 10.0)."},
        "shift": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                  "description": "In-plane shift of the guest as fractions of host a/b."},
        "strain": {"type": "boolean", "description": "Strain the guest to the host lattice (epitaxy)."},
    }, ["name", "host", "guest"]),
    _tool("import_poscar", "Load an existing structure from a POSCAR/CONTCAR file path.", {
        "name": _NAME,
        "path": {"type": "string", "description": "Path to a POSCAR/CONTCAR file or a folder containing one."},
    }, ["name", "path"]),
    _tool("finish", "Declare the final structure. Call once when the build is complete.", {
        "name": {"type": "string", "description": "Name of the finished structure."},
    }, ["name"]),
]


class _Workspace:
    """Holds the named structures the model builds and dispatches tool calls.

    Tool results are returned to the model as short text summaries (formula,
    atom count, cell) — never raw coordinates — to keep the loop cheap and to
    preserve the "the model never sees geometry" property.
    """

    def __init__(self):
        self.structures: dict[str, dict] = {}
        self.last_name: str | None = None
        self.result_name: str | None = None
        self.finished = False

    def _get(self, name: str) -> dict:
        if name not in self.structures:
            raise KeyError(f"no structure named {name!r}; built so far: "
                           f"{', '.join(self.structures) or '(none)'}")
        return self.structures[name]

    def _store(self, name: str, struct: dict) -> str:
        self.structures[name] = struct
        self.last_name = name
        return self._summary(name, struct)

    @staticmethod
    def _summary(name: str, struct: dict) -> str:
        formula = "".join(f"{e}{c}" for e, c in zip(struct["elements"], struct["counts"]))
        natoms = sum(struct["counts"])
        a, b, c = cell_lengths(struct)
        return f"ok: '{name}' = {formula} ({natoms} atoms), cell {a:.2f}x{b:.2f}x{c:.2f} A"

    def call(self, tool: str, args: dict) -> str:
        try:
            return self._call(tool, args)
        except Exception as exc:  # surfaced to the model so it can recover
            return f"error: {type(exc).__name__}: {exc}"

    def _call(self, tool: str, args: dict) -> str:
        if tool in ("build_bulk", "build_surface", "build_molecule", "build_nanotube"):
            act = {"build_bulk": "bulk", "build_surface": "surface",
                   "build_molecule": "molecule", "build_nanotube": "nanotube"}[tool]
            cmd = {"action": act, **args}
            if tool == "build_molecule":
                cmd["name"] = args["species"]  # _build_base reads cmd["name"] for molecules
            atoms = _build_base(cmd)
            struct = _atoms_to_struct(atoms)
            if args.get("supercell"):
                struct = make_supercell(struct, _supercell(args["supercell"]))
            return self._store(args["name"], struct)

        if tool == "make_supercell":
            struct = make_supercell(self._get(args["name"]), _supercell(args["repeat"]))
            return self._store(args["name"], struct)

        if tool == "add_adsorbate":
            slab = self._get(args["name"])
            anchor = int(args["anchor_index"]) if args.get("anchor_index") else _topmost_index(slab)
            struct = add_adsorbate(slab, str(args["element"]), anchor,
                                   float(args.get("height", 1.8)))
            return self._store(args["name"], struct)

        if tool == "combine":
            host, guest = self._get(args["host"]), self._get(args["guest"])
            shift = args.get("shift") or [0.0, 0.0]
            struct = combine_structures(
                host, guest,
                mode=args.get("mode") or "stack",
                gap=float(args.get("gap", 2.0)),
                vacuum=float(args.get("vacuum", 10.0)),
                shift=(float(shift[0]), float(shift[1])),
                strain_guest=bool(args.get("strain", False)),
            )
            return self._store(args["name"], struct)

        if tool == "import_poscar":
            from pathlib import Path
            path = Path(str(args["path"])).expanduser()
            poscar = path if path.is_file() else path / "POSCAR"
            return self._store(args["name"], read_poscar(poscar))

        if tool == "finish":
            self._get(args["name"])  # validate it exists
            self.result_name = args["name"]
            self.finished = True
            return f"ok: final structure is '{args['name']}'"

        raise ValueError(f"unknown tool {tool!r}")


def _ai_chat(messages: list, tools: list, api_key: str, model: str,
             url: str = ai_providers.GROQ_URL) -> dict:
    """One OpenAI-compatible chat-completions call (with tools) to ``url``."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
    }
    return ai_providers.chat(url, payload, api_key, timeout=60)


def agent_build_from_text(text: str, api_key: str | None = None,
                          model: str | None = None, max_steps: int = 16,
                          provider: str | None = None, base_url: str | None = None,
                          _chat=None) -> tuple[dict, list[dict]]:
    """Run the tool-calling worker on free text.

    Returns ``(struct, transcript)`` where ``struct`` is the finished structure
    (POSCAR dict, for the editor — nothing is written to disk) and ``transcript``
    is the list of ``{"tool", "args", "result"}`` steps the model took, so the UI
    can show exactly what the worker did.

    ``provider`` / ``base_url`` / ``model`` pick the AI API (see
    :mod:`ai_providers`); all default to Groq's tool-capable model.

    ``_chat`` is an injection point for testing — a callable with the same
    signature as :func:`_ai_chat`. In normal use it defaults to the real call.
    """
    if not text or not text.strip():
        raise ValueError("Describe the structure to build.")
    url, key, model, _ = ai_providers.resolve(
        provider=provider, base_url=base_url, model=model, api_key=api_key,
        agent=True,
    )
    chat = _chat or _ai_chat

    ws = _Workspace()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text.strip()},
    ]
    transcript: list[dict] = []

    for _ in range(max_steps):
        resp = chat(messages, TOOLS, key, model, url)
        msg = resp["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        # Echo the assistant turn (with its tool_calls) before the tool results.
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            **({"tool_calls": tool_calls} if tool_calls else {}),
        })
        if not tool_calls:
            break  # model produced a final text answer with no further actions
        for tc in tool_calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = ws.call(fn, args)
            transcript.append({"tool": fn, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", fn),
                "content": result,
            })
        if ws.finished:
            break

    name = ws.result_name or ws.last_name
    if name is None:
        raise ValueError(
            "The worker did not build anything. Try rephrasing, or use a "
            "stronger model (openai/gpt-oss-120b)."
        )
    return ws.structures[name], transcript
