"""Structured request rule: the slots a prompt MUST name for each region.

The corpus is designed within this rule so that a conforming request maps to
exactly one canonical structure — ambiguity is reduced at the source rather
than being resolved by the model. Required slots are the *non-inferable
structural determinants* of a region; a slot that another stated slot fixes
(e.g. crystal phase implied by the element in our material set) is not required.

The rule is spec-driven: for each case the required slot *values* are extracted
from its canonical recipe steps, and a paraphrase conforms only if every
required value is present in its text. `ask_clarification` cases are the sole
exception — they deliberately omit exactly one required slot.
"""

from __future__ import annotations

import re
from typing import Any


# Region -> ordered required slot keys. Optional slots (repeat, crystalline on a
# surface, nanotube vacuum) are intentionally absent.
FAMILY_REQUIRED: dict[str, tuple[str, ...]] = {
    "bulk": ("element", "crystalline"),
    "vacancy": ("element", "crystalline", "defect_site"),
    "substitution": ("element", "crystalline", "defect_site", "dopant"),
    "surface": ("element", "facet", "layers", "vacuum"),
    "surface_constraint": ("element", "facet", "layers", "vacuum", "freeze"),
    "atomic_adsorption": ("element", "facet", "layers", "vacuum", "adsorbate", "site", "height"),
    "molecular_adsorption": ("element", "facet", "layers", "vacuum", "adsorbate", "site", "height", "anchor"),
    "molecule": ("species", "box"),
    "nanotube": ("chirality", "length"),
    "prototype": ("prototype",),
    "clarification": (),  # handled by clarification_ok(); one slot is withheld on purpose
}


def slot_values(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the canonical slot values a conforming prompt must state."""
    values: dict[str, Any] = {}
    for step in steps:
        tool, args = step["tool"], step.get("args", {})
        if tool == "build_bulk":
            values["element"] = args["element"]
            values["crystalline"] = args.get("crystal")
        elif tool == "repeat":
            values["repeat"] = tuple(args["repeat"])
        elif tool == "build_surface":
            values["element"] = args["element"]
            values["crystalline"] = args.get("crystal")
            values["facet"] = "".join(str(v) for v in args["miller"])
            values["layers"] = args["layers"]
            values["vacuum"] = args["vacuum"]
            if args.get("repeat"):
                values["repeat"] = tuple(args["repeat"])
        elif tool == "freeze_layers":
            values["freeze"] = (args["side"], args["layers"])
        elif tool == "build_molecule":
            values["species"] = args["species"]
            values["box"] = args["box"]
        elif tool == "build_nanotube":
            values["chirality"] = (args["n"], args["m"])
            values["length"] = args["length"]
        elif tool == "build_prototype":
            values["prototype"] = args["prototype"]
        elif tool == "make_vacancy":
            values["defect_site"] = args["selector"]
        elif tool == "substitute":
            values["defect_site"] = args["selector"]
            values["dopant"] = args["element"]
        elif tool == "add_atomic_adsorbate":
            values["adsorbate"] = args["element"]
            values["site"] = args["site"]
            values["height"] = args["height"]
        elif tool == "add_molecular_adsorbate":
            values["adsorbate"] = args["species"]
            values["site"] = args["site"]
            values["height"] = args["height"]
            values["anchor"] = args["anchor"]
    return values


def _num_forms(value: Any) -> set[str]:
    number = float(value)
    forms = {f"{number:g}"}
    if number.is_integer():
        forms.add(str(int(number)))
    return forms


def _has_number_near(text: str, value: Any, keywords: tuple[str, ...]) -> bool:
    for form in _num_forms(value):
        for keyword in keywords:
            if re.search(rf"{re.escape(form)}\b[^.]{{0,18}}{keyword}", text) or re.search(
                rf"{keyword}[^.]{{0,18}}\b{re.escape(form)}\b", text
            ):
                return True
    return False


def _slot_present(slot: str, value: Any, text: str) -> bool:
    """Lenient presence check for one slot value in already-casefolded text."""
    if slot in ("element", "dopant"):
        return re.search(rf"\b{re.escape(str(value).casefold())}\b", text) is not None
    if slot == "crystalline":
        return value is None or str(value).casefold() in text
    if slot == "facet":
        compact = re.sub(r"[ ,()\-]", "", text)
        return str(value) in compact
    if slot == "layers":
        return _has_number_near(text, value, ("layer",))
    if slot == "vacuum":
        return _has_number_near(text, value, ("vacuum", "a\\b", "å", "angstrom", "ang\\b"))
    if slot == "box":
        return _has_number_near(text, value, ("box", "cell", "å", "angstrom", "a\\b"))
    if slot == "species":
        return re.search(rf"\b{re.escape(str(value).casefold())}\b", text) is not None
    if slot == "adsorbate":
        return re.search(rf"\b{re.escape(str(value).casefold())}\b", text) is not None
    if slot == "site":
        return str(value).casefold() in text
    if slot == "height":
        return _has_number_near(text, value, ("high", "above", "height", "å", "angstrom", "a\\b"))
    if slot == "anchor":
        return ("anchor" in text) or ("through" in text) or ("carbon" in text)
    if slot == "repeat":
        a, b, c = value
        compact = re.sub(r"\s", "", text)
        return f"{a}x{b}" in compact or f"{a}x{b}x{c}" in compact
    if slot == "chirality":
        n, m = value
        compact = re.sub(r"[ ()]", "", text)
        return f"{n},{m}" in compact or f"({n},{m})" in text
    if slot == "length":
        return _has_number_near(text, value, ("unit cell", "cell", "long", "length"))
    if slot == "freeze":
        side, _ = value
        verbs = r"freeze|frozen|fix|constrain|immobil|lock|pin|clamp|hold|anchor"
        return bool(re.search(verbs, text)) and str(side).casefold() in text
    if slot == "defect_site":
        return bool(re.search(r"\bfirst\b|\batom 1\b|\bsite 1\b|\bindex 1\b|\bposition 1\b", text))
    if slot == "prototype":
        head = re.split(r"[-_]", str(value).casefold())[-1]
        return str(value).casefold() in text or head in text
    return True


def missing_slots(family: str, values: dict[str, Any], prompt: str) -> list[str]:
    """Return the required slots whose value is absent from the prompt.

    A non-trivial supercell `repeat` is required in every region even though it
    is not listed per-family: a filled recipe that repeats 2x2 but a prompt that
    omits it would be genuinely ambiguous.
    """
    text = f" {prompt.casefold()} "
    missing = [
        slot
        for slot in FAMILY_REQUIRED.get(family, ())
        if slot in values and not _slot_present(slot, values[slot], text)
    ]
    repeat = values.get("repeat")
    if repeat and tuple(repeat) != (1, 1, 1) and not _slot_present("repeat", tuple(repeat), text):
        missing.append("repeat")
    return missing


def conforms(family: str, values: dict[str, Any], prompt: str) -> bool:
    return not missing_slots(family, values, prompt)


def clarification_ok(withheld_field: str, prompt: str, followup: str) -> bool:
    """A clarification prompt must withhold its field and the follow-up supply it."""
    text = f" {prompt.casefold()} "
    follow = f" {followup.casefold()} " if followup else ""
    field = withheld_field.casefold()
    # The withheld structural field (e.g. 'miller'/facet, 'structure_type') must
    # not be pinned in the first prompt but must be resolved by the follow-up.
    return field not in text and bool(follow)
