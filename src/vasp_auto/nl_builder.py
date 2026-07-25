"""Natural-language structure builder.

A chatbox sends a sentence ("Pt(111) slab, 4 layers, with O adsorbed on top");
any OpenAI-compatible AI API turns it into ONE JSON build command, and ASE + this
project's own structure primitives execute it.  The model only ever chooses an
action and fills parameters — it never emits atomic coordinates, so geometry is
always exact and a misread can't corrupt a cell.

No third-party SDK: the call is a plain ``urllib`` POST through
:mod:`ai_providers`, which picks the endpoint (Groq by default; OpenAI,
OpenRouter, DeepSeek, a local Ollama/LM Studio server, ... by ``provider`` or
``base_url``).  The API key comes from the provider's environment variable or,
for the UI, the per-request ``api_key`` argument — never hardcoded or written to
disk here.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from . import ai_providers
from .structure import (
    add_adsorbate,
    cart_coords,
    combine_structures,
    read_poscar,
)

# Back-compat alias; the canonical default now lives in ai_providers.PROVIDERS.
GROQ_MODEL = ai_providers.PROVIDERS["groq"]["model"]

SYSTEM_PROMPT = """You convert a materials-science request into ONE JSON build
command. Respond with JSON only — no prose. Available actions and their fields:

bulk:      {"action":"bulk","element":"Pt","crystal":"fcc"|"bcc"|"hcp"|"diamond"|null,
            "a":3.92|null,"c":null,"supercell":[1,1,1]|null}
surface:   {"action":"surface","element":"Pt","crystal":null,"a":null,
            "miller":[1,1,1],"layers":4,"vacuum":12.0,"supercell":[3,3,1]|null}
molecule:  {"action":"molecule","name":"CO"}            // ASE g2 names (CO, H2O, CO2, NH3, O2...)
nanotube:  {"action":"nanotube","n":5,"m":5,"supercell":null}
adsorbate: {"action":"adsorbate","slab":{<a surface command>},
            "adsorbate_element":"O","height":1.8}        // ONE atom above the top surface atom
combine:   {"action":"combine","host":{<command>},"guest":{<command>},
            "mode":"stack"|"insert","gap":2.0,"vacuum":10.0,"shift":[0.0,0.0],"strain":false}

Rules:
- adsorbate adds a SINGLE atom (O, H, N, C...). For a molecular adsorbate, use
  combine with host=the slab (a surface command) and guest=the molecule, mode "insert".
- combine.host and combine.guest are themselves full commands (recurse).
- Use null for unknown numbers; pick sensible defaults (slab vacuum 12, layers 4).
- "miller" is ALWAYS a list of three integers: the (100) plane is [1,0,0], (111)
  is [1,1,1]. Never write it as a single number like 100.
- Lattice constants are in Angstrom. Never output coordinates or lattice vectors.
"""


def _ask_ai(text: str, api_key: str | None = None, provider: str | None = None,
            base_url: str | None = None, model: str | None = None) -> dict:
    """POST the request to the chosen AI API and return the parsed JSON command."""
    url, key, chosen, supports_json = ai_providers.resolve(
        provider=provider, base_url=base_url, model=model, api_key=api_key,
    )
    payload = {
        "model": chosen,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
    }
    if supports_json:
        payload["response_format"] = {"type": "json_object"}  # forces valid JSON back
    body = ai_providers.chat(url, payload, key, timeout=30)
    content = body["choices"][0]["message"]["content"]
    return json.loads(_json_text(content))


def _json_text(content: str) -> str:
    """Pull the JSON object out of a model reply.

    With ``response_format=json_object`` the reply is already pure JSON, but
    providers/models without that mode may wrap it in a ```json fence or add a
    line of prose. Strip a fenced block, else take the outermost ``{...}``.
    """
    text = (content or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def _atoms_to_struct(atoms) -> dict:
    """Convert an ASE Atoms object to this project's POSCAR struct dict by
    round-tripping through a temporary POSCAR — reuses read_poscar exactly."""
    from ase.io import write

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "POSCAR"
        write(path, atoms, format="vasp")
        return read_poscar(path)


def _topmost_index(struct: dict) -> int:
    """1-based index of the highest atom along z — the default adsorption site."""
    zs = [c[2] for c in cart_coords(struct)]
    return max(range(len(zs)), key=zs.__getitem__) + 1


def _miller(value) -> tuple[int, int, int]:
    """Coerce model-supplied Miller indices to exactly three ints.

    ASE rejects anything that isn't three non-zero-overall integers with a bare
    "invalid surface type". The model may hand back a list, a string like
    "111"/"1 1 1"/"1,1,1", or floats — normalise all of them and give a clear
    error when it really can't be read.
    """
    if value is None:
        raise ValueError("This surface is missing its Miller indices (e.g. [1,1,1]).")
    if isinstance(value, str):
        s = value.strip().strip("()[]{}")
        parts = re.split(r"[,\s]+", s) if re.search(r"[,\s]", s) else list(s)
        value = [p for p in parts if p not in ("", "(", ")")]
    if not isinstance(value, (list, tuple)):
        value = [value]
    try:
        idx = [int(round(float(i))) for i in value]
    except (TypeError, ValueError):
        raise ValueError(f"Could not read Miller indices from {value!r}.") from None
    # The model often collapses a low-index plane into one number: 100 -> (1,0,0),
    # 111 -> (1,1,1). Expand a single 2-3 digit value into its digits.
    if len(idx) == 1 and 10 <= idx[0] <= 999:
        idx = [int(d) for d in str(idx[0])]
    if len(idx) != 3:
        raise ValueError(f"Miller indices need exactly three numbers, got {idx}.")
    if not any(idx):
        raise ValueError("Miller indices can't be all zero.")
    return tuple(idx)


def _supercell(value):
    """Coerce a model-supplied supercell to a 3-tuple of repeats, or None.

    ASE's ``repeat`` needs exactly three values; the model often gives two for an
    in-plane slab ("3x3" -> [3,3]) or one for a uniform box. Pad [a,b] with a
    third axis of 1, expand a single value to [n,n,n], and accept "3x3x1" strings.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = int(value)
        return (n, n, n)
    if isinstance(value, str):
        value = [p for p in re.split(r"[x,\s*]+", value.strip().lower()) if p]
    try:
        rep = [int(round(float(n))) for n in value]
    except (TypeError, ValueError):
        raise ValueError(f"Could not read supercell from {value!r}.") from None
    if not rep:
        return None
    if len(rep) == 1:
        rep = rep * 3
    elif len(rep) == 2:
        rep = [rep[0], rep[1], 1]
    elif len(rep) != 3:
        raise ValueError(f"Supercell needs up to three numbers, got {rep}.")
    if any(n < 1 for n in rep):
        raise ValueError(f"Supercell repeats must be at least 1, got {rep}.")
    return tuple(rep)


def _build_base(cmd: dict):
    """Build one of the four ASE-backed primitives, return an ASE Atoms."""
    from ase.build import bulk, molecule, nanotube, surface

    act = cmd.get("action")
    if act == "bulk":
        kw = {}
        if cmd.get("crystal"):
            kw["crystalstructure"] = cmd["crystal"]
        if cmd.get("a"):
            kw["a"] = float(cmd["a"])
        if cmd.get("c"):
            kw["c"] = float(cmd["c"])
        return bulk(cmd["element"], **kw)
    if act == "surface":
        base_kw = {}
        if cmd.get("crystal"):
            base_kw["crystalstructure"] = cmd["crystal"]
        if cmd.get("a"):
            base_kw["a"] = float(cmd["a"])
        # surface() indexes planes off the conventional (cubic) cell; fall back
        # to the default cell for structures with no cubic form (e.g. hcp).
        try:
            base = bulk(cmd["element"], cubic=True, **base_kw)
        except Exception:
            base = bulk(cmd["element"], **base_kw)
        return surface(
            base,
            _miller(cmd.get("miller")),
            int(cmd.get("layers") or 4),
            vacuum=float(cmd.get("vacuum") or 12.0),
        )
    if act == "molecule":
        atoms = molecule(cmd["name"])
        atoms.center(vacuum=6.0)
        return atoms
    if act == "nanotube":
        atoms = nanotube(int(cmd["n"]), int(cmd["m"]))
        atoms.center(vacuum=6.0)
        return atoms
    raise ValueError(f"Unsupported base action: {act!r}")


_KNOWN_ACTIONS = ("bulk", "surface", "molecule", "nanotube", "adsorbate", "combine")
_ENVELOPE_KEYS = ("command", "build", "structure", "result", "spec", "cmd")


def _normalize_command(cmd):
    """Coax a slightly-off model response into a build command with an ``action``.

    Small models sometimes wrap the command (``{"command": {...}}``), return a
    one-item list, or name the field ``type`` instead of ``action``. Unwrap those
    common shapes so a near-miss still builds; otherwise return as-is and let the
    caller raise a clear error.
    """
    if isinstance(cmd, list):
        cmd = cmd[0] if cmd else {}
    if not isinstance(cmd, dict):
        raise ValueError(f"Model did not return a JSON object (got {type(cmd).__name__}).")
    if "action" not in cmd and "type" in cmd and cmd["type"] in _KNOWN_ACTIONS:
        cmd = {**cmd, "action": cmd["type"]}
    if "action" in cmd:
        return cmd
    # A wrapped command: {"command": {<real command>}} and friends, or any
    # single-key envelope whose value is itself a command.
    for key in (*_ENVELOPE_KEYS, *([next(iter(cmd))] if len(cmd) == 1 else [])):
        inner = cmd.get(key)
        if isinstance(inner, (dict, list)):
            unwrapped = _normalize_command(inner)
            if "action" in unwrapped:
                return unwrapped
    return cmd


def _build_one(cmd: dict) -> dict:
    """Recursively execute a build command, always returning a struct dict."""
    cmd = _normalize_command(cmd)
    act = cmd.get("action")
    if act in ("bulk", "surface", "molecule", "nanotube"):
        atoms = _build_base(cmd)
        rep = _supercell(cmd.get("supercell"))
        if rep and rep != (1, 1, 1):
            atoms = atoms.repeat(rep)
        return _atoms_to_struct(atoms)
    if act == "adsorbate":
        slab = _build_one(cmd["slab"])
        anchor = _topmost_index(slab)
        return add_adsorbate(
            slab,
            str(cmd["adsorbate_element"]),
            anchor,
            float(cmd.get("height", 1.8)),
        )
    if act == "combine":
        host = _build_one(cmd["host"])
        guest = _build_one(cmd["guest"])
        shift = list(cmd.get("shift") or [])
        sx = float(shift[0]) if len(shift) > 0 else 0.0
        sy = float(shift[1]) if len(shift) > 1 else 0.0
        return combine_structures(
            host,
            guest,
            mode=cmd.get("mode") or "stack",
            gap=float(cmd.get("gap", 2.0)),
            vacuum=float(cmd.get("vacuum", 10.0)),
            shift=(sx, sy),
            strain_guest=bool(cmd.get("strain", False)),
        )
    raise ValueError(
        f"The AI returned a command this builder can't run (action={act!r}). "
        f"Try rephrasing. Model returned: {json.dumps(cmd)[:200]}"
    )


def describe_command(cmd: dict) -> str:
    """Human-readable, one-line summary of a build command tree — mirrors the
    actions in ``_build_one`` so the UI can show what the AI actually built."""
    act = cmd.get("action")
    if act == "bulk":
        crystal = f"{cmd['crystal']} " if cmd.get("crystal") else ""
        base = f"{crystal}{cmd.get('element', '?')} bulk"
    elif act == "surface":
        try:
            miller = "".join(str(i) for i in _miller(cmd.get("miller")))
        except ValueError:
            miller = "?"
        base = (
            f"{cmd.get('element', '?')}({miller}) slab, "
            f"{int(cmd.get('layers') or 4)} layers, "
            f"{float(cmd.get('vacuum') or 12.0):g} Å vacuum"
        )
    elif act == "molecule":
        base = f"{cmd.get('name', '?')} molecule"
    elif act == "nanotube":
        base = f"({cmd.get('n', '?')},{cmd.get('m', '?')}) nanotube"
    elif act == "adsorbate":
        return (
            f"{describe_command(cmd['slab'])} + {cmd.get('adsorbate_element', '?')} "
            f"adsorbate @{float(cmd.get('height', 1.8)):g} Å"
        )
    elif act == "combine":
        mode = cmd.get("mode") or "stack"
        gap = float(cmd.get("gap", 2.0))
        return (
            f"{describe_command(cmd['host'])} + {describe_command(cmd['guest'])} "
            f"({mode}, gap {gap:g} Å)"
        )
    else:
        return f"{act} command"
    try:
        rep = _supercell(cmd.get("supercell"))
    except ValueError:
        rep = None
    if rep and rep != (1, 1, 1):
        base += " ×" + "×".join(str(n) for n in rep)
    return base


def build_from_text(text: str, api_key: str | None = None,
                    provider: str | None = None, base_url: str | None = None,
                    model: str | None = None) -> tuple[dict, dict]:
    """Turn free text into (command, struct_dict). Nothing is written to disk;
    the struct dict is meant to load into the editor for preview-before-save.

    ``provider`` / ``base_url`` / ``model`` pick the AI API (see
    :mod:`ai_providers`); all default to Groq when omitted."""
    if not text or not text.strip():
        raise ValueError("Describe the structure to build.")
    cmd = _normalize_command(_ask_ai(
        text.strip(), api_key=api_key, provider=provider,
        base_url=base_url, model=model,
    ))
    struct = _build_one(cmd)
    return cmd, struct
