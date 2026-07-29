"""Deterministically limit model tool exposure to supported request families.

Trigger vocabularies are kept in sync with the training corpus's structured
request rule (``training/generators/request_rule.py``): every phrasing the
corpus may emit must route here to a superset of its recipe's tools. Truly
unsupported or adversarial requests still match nothing and fail closed.
"""

from __future__ import annotations

import re

from ..structure import COMPOUND_FAMILIES, COMPOUND_LATTICE
from .registry import ToolRegistry

# Compound prototypes are named by family ('rocksalt') or by composition
# ('MgO'), so both must route to build_prototype. Derived from the tables in
# structure.py rather than restated, so a composition added there routes here
# without a second edit. `route_tools` casefolds the request first.
_COMPOUND_FAMILY_WORDS = "|".join(sorted(COMPOUND_FAMILIES))
_COMPOUND_FORMULAS = "|".join(
    sorted(
        {formula.casefold() for table in COMPOUND_LATTICE.values() for formula in table},
        key=len,
        reverse=True,  # longest first so 'ceo2' wins over a hypothetical 'ceo'
    )
)
_COMPOUND_TRIGGER = (
    rf"\b(?:{_COMPOUND_FAMILY_WORDS})\b|\b(?:{_COMPOUND_FORMULAS})\b"
)
_COMPOUND_RE = re.compile(_COMPOUND_TRIGGER)


# Requests that are ambiguous by design (a clarification family): expose the
# clarification tool plus the plausible builders so the model can ask first.
_CLARIFY = re.compile(
    r"\bclarif\w*|\bwhether\b|isolated pair|structural form|ask me\b|ask whether\b"
    r"|ask which\b|but ask\b|before (?:choosing|building|deciding)"
)
_CLARIFY_TOOLS = (
    "ask_clarification", "build_bulk", "build_surface", "build_molecule",
    "build_prototype", "build_crystal", "build_nanotube", "build_nanoparticle",
    "combine", "repeat", "finish",
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
    (r"\bprototype\b|\b(?:hbn|graphene|graphite|rutile|anatase)\b",
     ("build_prototype", "finish")),
    (r"\b(?:molecule|molecular|isolated calculation)\b", ("build_molecule", "finish")),
    (r"\b(?:surface|slab)\b|\([0-9]{3}\)", ("build_surface", "finish")),
    (r"\b(?:bulk|crystal|cell)\b", ("build_bulk", "repeat", "finish")),
    (r"\b(?:fcc|bcc|hcp|diamond|aluminium|aluminum)\b", ("build_bulk", "repeat", "finish")),
    # Compound fallback, deliberately last: a request that names a composition but
    # no structure word ("Build rocksalt MgO."). Anything that *does* say slab,
    # vacancy or crystal must reach its own route first and be rewritten by
    # _for_compound, so a compound slab still gets build_surface.
    (_COMPOUND_TRIGGER, ("build_prototype", "repeat", "finish")),
    (r"\bfinish\b.*\bstructure\b", ("finish",)),
)


def _for_compound(names: tuple[str, ...]) -> tuple[str, ...]:
    """Swap the elemental bulk builder for the prototype builder.

    `build_bulk` wraps ase.build.bulk and takes one element, so it can never
    build a compound. Swapping rather than adding keeps the exposed set minimal:
    the model is not offered a tool that is guaranteed to fail on this request.
    `build_surface` needs no swap -- it takes a `prototype` argument and cuts the
    facet from the compound cell itself. Edits (repeat, make_vacancy,
    substitute, freeze_layers) act on whatever the active structure is.
    """
    return tuple(
        "build_prototype" if name == "build_bulk" else name for name in names
    )


def route_tools(request: str, registry: ToolRegistry) -> list[dict]:
    """Return schemas for one supported request region, or fail closed."""
    text = request.casefold()
    surface_request = bool(re.search(r"\b(?:surface|slab|facet|support)\w*\b|\([0-9]{3,4}\)", text))
    particle_request = bool(re.search(
        r"\b(?:nanoparticle|nanocluster|cluster|icosahedr\w*|octahedr\w*|cuboctahedr\w*)\b",
        text,
    ))
    if particle_request:
        names = (
            ("build_surface", "build_nanoparticle", "combine", "finish")
            if surface_request
            else ("build_nanoparticle", "finish")
        )
        if not re.search(r"\b(?:icosahedr\w*|octahedr\w*|cuboctahedr\w*)\b", text) or (
            surface_request and not re.search(r"\([0-9]{3,4}\)", text)
        ):
            names = (*names[:-1], "ask_clarification", names[-1])
        return _schemas(registry, names)
    adsorption = re.search(r"\b(?:adsorb\w*|ontop|above an? .*site)\b", text)
    if adsorption or (" molecule " in f" {text} " and re.search(r"\b(?:surface|slab)\b|\([0-9]{3}\)", text)):
        adsorbate = (
            "add_molecular_adsorbate"
            if re.search(
                r"\b(?:co2|co|h2o|h2|o2|n2|nh3|ch4|oh|no|c2h4|c6h6|ch3oh|hcooh|molecule|molecular)\b",
                text,
            )
            else "add_atomic_adsorbate"
        )
        names = ("build_surface", adsorbate, "finish")
        return _schemas(registry, names)
    if _CLARIFY.search(text):
        return _schemas(registry, _CLARIFY_TOOLS)
    compound = bool(_COMPOUND_RE.search(text))
    for pattern, names in _ROUTES:
        if re.search(pattern, text):
            if surface_request and "make_vacancy" in names:
                names = ("build_surface", "make_vacancy", "finish")
            elif surface_request and "substitute" in names:
                names = ("build_surface", "substitute", "finish")
            if surface_request and not re.search(r"\([0-9]{3,4}\)|\b[0-9]{3,4}\b", text):
                names = (*names[:-1], "ask_clarification", names[-1])
            if names == ("build_surface", "finish") and not re.search(r"\d", text):
                names = ("build_surface", "ask_clarification", "finish")
            if compound:
                names = _for_compound(names)
            return _schemas(registry, names)
    raise ValueError(
        "unsupported structure request; ask for a bulk crystal, surface/slab, "
        "molecule, nanotube, nanoparticle, supported cluster, prototype, adsorption, "
        "vacancy, substitution, or constraint"
    )


def _schemas(registry: ToolRegistry, names: tuple[str, ...]) -> list[dict]:
    selected = set(names)
    return [
        schema
        for schema in registry.function_schemas()
        if schema["function"]["name"] in selected
    ]
