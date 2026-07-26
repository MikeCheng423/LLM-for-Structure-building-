"""Fill agent-authored prompt templates per case and keep only rule-conforming ones.

Agents produce diverse ``str.format`` templates per family using the placeholder
vocabulary below (e.g. ``"Build a {suffix} {element}({facet}) slab, {layers}
layers, {vacuum} A vacuum."``). For each concrete case the placeholders are
substituted with the canonical slot values, and every filled prompt is checked
against ``request_rule`` before it may enter the corpus. Templates that reference
a placeholder a case does not provide are skipped for that case; filled prompts
that drop a required slot are rejected. No template can change a slot value —
they only rephrase — so the executed recipe stays the ground truth.
"""

from __future__ import annotations

from typing import Any

from . import request_rule as rr


# Placeholders a template may use, grouped by family, for the agent brief.
FAMILY_PLACEHOLDERS: dict[str, tuple[str, ...]] = {
    "bulk": ("element", "crystalline", "repeat3"),
    "vacancy": ("element", "crystalline", "repeat3", "defect_ordinal"),
    "substitution": ("element", "crystalline", "repeat3", "defect_ordinal", "dopant"),
    "surface": ("element", "crystalline", "facet", "layers", "vacuum", "suffix"),
    "surface_constraint": ("element", "crystalline", "facet", "layers", "vacuum", "suffix", "freeze_side", "freeze_count"),
    "atomic_adsorption": ("element", "facet", "layers", "vacuum", "suffix", "adsorbate", "site", "height"),
    "molecular_adsorption": ("element", "facet", "layers", "vacuum", "suffix", "adsorbate", "site", "height", "anchor"),
    "molecule": ("species", "box"),
    "nanotube": ("chirality", "length"),
    "prototype": ("prototype",),
}


def case_slots(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Rich structured slots for one case, derived from its recipe steps."""
    slots: dict[str, Any] = {}
    for step in steps:
        tool, args = step["tool"], step.get("args", {})
        if tool == "build_bulk":
            slots["element"] = args["element"]
            slots["crystalline"] = args.get("crystal")
        elif tool == "repeat":
            slots["repeat"] = tuple(args["repeat"])
        elif tool == "build_surface":
            slots["element"] = args["element"]
            slots["crystalline"] = args.get("crystal")
            slots["facet"] = tuple(args["miller"])
            slots["layers"] = args["layers"]
            slots["vacuum"] = args["vacuum"]
            if args.get("repeat"):
                slots["repeat"] = tuple(args["repeat"])
        elif tool == "freeze_layers":
            slots["freeze"] = (args["side"], args["layers"])
        elif tool == "build_molecule":
            slots["species"] = args["species"]
            slots["box"] = args["box"]
        elif tool == "build_nanotube":
            slots["chirality"] = (args["n"], args["m"])
            slots["length"] = args["length"]
        elif tool == "build_prototype":
            slots["prototype"] = args["prototype"]
        elif tool == "make_vacancy":
            slots["defect_indices"] = list(args["selector"].get("indices", []))
        elif tool == "substitute":
            slots["defect_indices"] = list(args["selector"].get("indices", []))
            slots["dopant"] = args["element"]
        elif tool == "add_atomic_adsorbate":
            slots["adsorbate"] = args["element"]
            slots["site"] = args["site"]
            slots["height"] = args["height"]
        elif tool == "add_molecular_adsorbate":
            slots["adsorbate"] = args["species"]
            slots["site"] = args["site"]
            slots["height"] = args["height"]
            slots["anchor"] = args["anchor"]
    return slots


def _num(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


# Spoken forms for the anchor atom. Anything unlisted falls back to its symbol.
_ELEMENT_WORDS = {"H": "hydrogen", "C": "carbon", "N": "nitrogen", "O": "oxygen", "S": "sulfur"}


def _anchor_word(species: str, anchor: int) -> str:
    """Name the anchor atom from ASE's real atom ordering for `species`.

    ``anchor`` is 1-based over ``ase.build.molecule(species)``, whose ordering is
    per-species and not alphabetical -- CO is stored as (O, C), so atom 1 is the
    oxygen. Resolving the word here keeps the prompt honest about which atom the
    recipe actually anchors instead of assuming a fixed element.
    """
    from ase.build import molecule

    symbols = molecule(species).get_chemical_symbols()
    index = int(anchor) - 1
    if not 0 <= index < len(symbols):
        raise ValueError(f"anchor {anchor} out of range for {species} ({len(symbols)} atoms)")
    symbol = symbols[index]
    return _ELEMENT_WORDS.get(symbol, symbol)


def format_dict(slots: dict[str, Any]) -> dict[str, str]:
    """Clean placeholder -> surface-form strings for str.format substitution."""
    d: dict[str, str] = {}
    if slots.get("element"):
        d["element"] = slots["element"]
    if slots.get("crystalline"):
        d["crystalline"] = slots["crystalline"]
    if slots.get("facet"):
        face = "".join(str(v) for v in slots["facet"])
        d["facet"] = face
        d["face"] = face
    if slots.get("layers") is not None:
        d["layers"] = str(slots["layers"])
    if slots.get("vacuum") is not None:
        d["vacuum"] = _num(slots["vacuum"])
    if slots.get("repeat"):
        a, b, c = slots["repeat"]
        d["suffix"] = f"{a}x{b}"
        d["repeat3"] = f"{a}x{b}x{c}"
    else:
        d["suffix"] = "1x1"
        d["repeat3"] = "1x1x1"
    if slots.get("adsorbate"):
        d["adsorbate"] = slots["adsorbate"]
    if slots.get("site"):
        d["site"] = slots["site"]
    if slots.get("height") is not None:
        d["height"] = _num(slots["height"])
    if slots.get("species"):
        d["species"] = slots["species"]
    if slots.get("box") is not None:
        d["box"] = _num(slots["box"])
    if slots.get("chirality"):
        n, m = slots["chirality"]
        d["chirality"] = f"({n},{m})"
        d["n"], d["m"] = str(n), str(m)
    if slots.get("length") is not None:
        d["length"] = str(slots["length"])
    if slots.get("dopant"):
        d["dopant"] = slots["dopant"]
    if slots.get("prototype"):
        d["prototype"] = slots["prototype"]
    if slots.get("defect_indices"):
        first = slots["defect_indices"][0]
        d["defect_ordinal"] = "first" if first == 1 else f"index {first}"
        d["defect_index"] = str(first)
    if slots.get("freeze"):
        side, count = slots["freeze"]
        d["freeze_side"] = side
        d["freeze_count"] = str(count)
    if slots.get("anchor") is not None:
        d["anchor"] = _anchor_word(slots["adsorbate"], slots["anchor"])
    return d


def fill_templates(
    family: str,
    templates: list[str],
    steps: list[dict[str, Any]],
) -> list[str]:
    """Return conforming filled prompts for one case; skip templates that don't fit."""
    slots = case_slots(steps)
    fmt = format_dict(slots)
    rule_values = rr.slot_values(steps)
    prompts: list[str] = []
    seen: set[str] = set()
    for template in templates:
        try:
            prompt = template.format(**fmt).strip()
        except (KeyError, IndexError):
            continue
        if prompt in seen:
            continue
        if rr.missing_slots(family, rule_values, prompt):
            continue
        seen.add(prompt)
        prompts.append(prompt)
    return prompts
