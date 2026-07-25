"""Deterministically limit model tool exposure to supported request families.

Trigger vocabularies are kept in sync with the training corpus's structured
request rule (``training/generators/request_rule.py``): every phrasing the
corpus may emit must route here to a superset of its recipe's tools. Truly
unsupported or adversarial requests still match nothing and fail closed.
"""

from __future__ import annotations

import re

from .registry import ToolRegistry


# Requests that are ambiguous by design (a clarification family): expose the
# clarification tool plus the plausible builders so the model can ask first.
_CLARIFY = re.compile(
    r"\bclarif\w*|\bwhether\b|isolated pair|structural form|ask me\b|ask whether\b"
    r"|ask which\b|but ask\b|before (?:choosing|building|deciding)"
)
_CLARIFY_TOOLS = (
    "ask_clarification", "build_bulk", "build_surface", "build_molecule",
    "build_prototype", "build_crystal", "build_nanotube", "repeat", "finish",
)


_ROUTES = (
    (r"\b(?:freeze|frozen|fix|constrain|immobil\w*|lock|pin|clamp|hold|anchor)\w*\b",
     ("build_surface", "freeze_layers", "finish")),
    (r"\bvacanc\w*\b"
     r"|\b(?:remov\w*|delet\w*|knock\s*out|eject\w*|strip\w*|eliminat\w*|extract\w*|pull\w*|take\s+(?:out|away))\b[^.]{0,20}\b(?:atom|site)\b"
     r"|\b(?:atom|site)\b[^.]{0,20}\b(?:remov\w*|delet\w*|eliminat\w*|gone|absent)\b",
     ("build_bulk", "repeat", "make_vacancy", "finish")),
    (r"\bsubstitut\w*\b|\bdope\w*\b|\bin place of\b|\b(?:replace|swap)\b[^.]{0,20}\b(?:atom|site)\b",
     ("build_bulk", "repeat", "substitute", "finish")),
    (r"\bnanotube\b|\bcnt\b", ("build_nanotube", "finish")),
    (r"\bprototype\b|\b(?:hbn|nacl|zincblende|rocksalt|graphene|graphite|rutile|anatase)\b",
     ("build_prototype", "finish")),
    (r"\b(?:molecule|molecular|isolated calculation)\b", ("build_molecule", "finish")),
    (r"\b(?:surface|slab)\b|\([0-9]{3}\)", ("build_surface", "finish")),
    (r"\b(?:bulk|crystal|cell)\b", ("build_bulk", "repeat", "finish")),
    (r"\b(?:fcc|bcc|hcp|diamond|aluminium|aluminum)\b", ("build_bulk", "repeat", "finish")),
    (r"\bfinish\b.*\bstructure\b", ("finish",)),
)


def route_tools(request: str, registry: ToolRegistry) -> list[dict]:
    """Return schemas for one supported request region, or fail closed."""
    text = request.casefold()
    adsorption = re.search(r"\b(?:adsorb\w*|ontop|above an? .*site)\b", text)
    if adsorption or (" molecule " in f" {text} " and re.search(r"\b(?:surface|slab)\b|\([0-9]{3}\)", text)):
        adsorbate = (
            "add_molecular_adsorbate"
            if re.search(r"\b(?:co|h2o|nh3|ch4|molecule)\b", text)
            else "add_atomic_adsorbate"
        )
        names = ("build_surface", adsorbate, "finish")
        return _schemas(registry, names)
    if _CLARIFY.search(text):
        return _schemas(registry, _CLARIFY_TOOLS)
    for pattern, names in _ROUTES:
        if re.search(pattern, text):
            if names == ("build_surface", "finish") and not re.search(r"\d", text):
                names = ("build_surface", "ask_clarification", "finish")
            return _schemas(registry, names)
    raise ValueError(
        "unsupported structure request; ask for a bulk crystal, surface/slab, "
        "molecule, nanotube, prototype, adsorption, vacancy, substitution, or constraint"
    )


def _schemas(registry: ToolRegistry, names: tuple[str, ...]) -> list[dict]:
    selected = set(names)
    return [
        schema
        for schema in registry.function_schemas()
        if schema["function"]["name"] in selected
    ]
