"""Advisory pre-flight check: which required slots look unstated in a request?

The training corpus obeys a *structured request rule* (`training/CORPUS_RULE.md`,
machine form in `training/generators/request_rule.py`): a request maps to exactly
one canonical structure only if it names every non-inferable determinant of its
region. Give the model a request that drops one and it will still build *a* valid
structure — just maybe not the one you meant.

This module gives that rule back to the user *before* the model loads, so the
feedback is instant and costs no GPU time.

It is deliberately weaker than the corpus-side checker. `request_rule` compares a
prompt against the *known canonical slot values* of a case; at run time there is
no canonical case yet, so all we can do is look for the *shape* of each slot
(a facet-looking token, a number next to "layer", a freeze verb...). Consequences,
stated plainly because the CLI prints them as a hint:

* the region is a keyword guess and can be wrong;
* a slot can be reported unstated when it was phrased unusually (false alarm);
* a slot can look stated when the value is not the one you meant (missed).

It is therefore **advisory only**: it never blocks, never rewrites, never
reorders. The deterministic tool router remains the only gate that refuses
anything.

The module also provides the *post-build* counterpart, :func:`check_build`.
Where the pre-flight hint asks "did you state this slot?", the post-build check
asks "did the structure actually use the number you stated?" — comparing the
request against the executed transcript. That catches the failure mode a hint
cannot: a fully-specified request whose value the model dropped, silently
falling back to a builder default (a request for a 2.5 A adsorption height
building at the default 1.8 A, say). It is deterministic, not a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Mirrors ``training.generators.request_rule.FAMILY_REQUIRED``. Duplicated rather
# than imported because ``training/`` is not part of the installed runtime;
# ``tests/test_ase_agent_cli.py`` asserts the two tables stay identical.
FAMILY_REQUIRED: dict[str, tuple[str, ...]] = {
    "bulk": ("element", "crystalline"),
    "vacancy": ("element", "crystalline", "defect_site"),
    "substitution": ("element", "crystalline", "defect_site", "dopant"),
    "surface": ("element", "facet", "layers", "vacuum"),
    "surface_constraint": ("element", "facet", "layers", "vacuum", "freeze"),
    "atomic_adsorption": ("element", "facet", "layers", "vacuum", "adsorbate", "site", "height"),
    "molecular_adsorption": (
        "element", "facet", "layers", "vacuum", "adsorbate", "site", "height", "anchor",
    ),
    "molecule": ("species", "box"),
    "nanotube": ("chirality", "length"),
    "prototype": ("prototype",),
    "clarification": (),
}


# Human-readable slot names for the hint line.
SLOT_LABELS: dict[str, str] = {
    "element": "element (host chemical symbol)",
    "crystalline": "crystal phase (fcc/bcc/hcp/diamond)",
    "facet": "facet (Miller indices, e.g. (111))",
    "layers": "layers (slab thickness)",
    "vacuum": "vacuum (gap in A)",
    "freeze": "which layers to freeze (side + count)",
    "adsorbate": "adsorbate species",
    "site": "adsorption site (ontop/bridge/hollow)",
    "height": "adsorption height in A",
    "anchor": "anchor atom of the adsorbed molecule",
    "species": "molecular species (formula)",
    "box": "box edge in A",
    "chirality": "chirality (n,m)",
    "length": "nanotube length",
    "prototype": "prototype name",
    "defect_site": "which atom the defect targets",
    "dopant": "dopant element",
}


_ELEMENT_WORDS = (
    "aluminium", "aluminum", "carbon", "cobalt", "copper", "gold", "hydrogen",
    "iron", "magnesium", "molybdenum", "nickel", "nitrogen", "oxygen",
    "palladium", "platinum", "rhodium", "ruthenium", "silicon", "silver",
    "sodium", "titanium", "tungsten", "vanadium", "zinc",
)

# Symbols are matched case-sensitively against the raw text, which is how people
# write them. "I" is excluded: the English pronoun would fire on every sentence.
_SYMBOL_RE = re.compile(
    r"(?<![A-Za-z])("
    r"H|He|Li|Be|B|C|N|O|F|Ne|Na|Mg|Al|Si|P|S|Cl|Ar|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|"
    r"Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|Xe|"
    r"Cs|Ba|La|Ce|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi"
    r")(?![a-z])"
)
_MOLECULE_RE = re.compile(
    r"(?<![A-Za-z0-9])(CO2|CO|H2O|H2|O2|N2|NH3|CH4|OH|NO|C2H4|C6H6|CH3OH|HCOOH)"
    r"(?![A-Za-z0-9])"
)
_MOLECULE_WORDS = ("water", "ammonia", "methane", "benzene", "ethylene", "methanol")

_NUMBER = r"\d+(?:\.\d+)?"
_WORD_NUMBERS = r"one|two|three|four|five|six|seven|eight|nine|ten"

# Compound prototype families and compositions, from the same tables the builder
# and the router use, so a composition added there is recognised here too.
from ..structure import (  # noqa: E402
    COMPOUND_FAMILIES,
    COMPOUND_LATTICE,
    parse_binary_formula,
)

_COMPOUND_RE = re.compile(
    r"\b(?:" + "|".join(sorted(COMPOUND_FAMILIES)) + r")\b|\b(?:"
    + "|".join(sorted(
        {f.casefold() for table in COMPOUND_LATTICE.values() for f in table},
        key=len, reverse=True,
    )) + r")\b"
)


def _near(text: str, keywords: str, *, number: str = _NUMBER, window: int = 20) -> bool:
    """True when a number and a keyword sit within `window` characters, either order."""
    return bool(
        re.search(rf"(?:{number})\b[^.]{{0,{window}}}?(?:{keywords})", text)
        or re.search(rf"(?:{keywords})[^.]{{0,{window}}}?\b(?:{number})", text)
    )


def _chemical_tokens(raw: str) -> list[str]:
    """Distinct chemical tokens (symbols and common formulas) in the raw text."""
    found = _MOLECULE_RE.findall(raw) + _SYMBOL_RE.findall(raw)
    seen: list[str] = []
    for token in found:
        if token not in seen:
            seen.append(token)
    lowered = f" {raw.casefold()} "
    for word in _ELEMENT_WORDS + _MOLECULE_WORDS:
        if re.search(rf"\b{word}\b", lowered) and word not in seen:
            seen.append(word)
    return seen


def _has_element(raw: str, text: str) -> bool:
    return bool(_chemical_tokens(raw))


def _slot_stated(slot: str, raw: str, text: str) -> bool:
    """Heuristic: does the request look like it *states* this slot at all?"""
    if slot == "element":
        return _has_element(raw, text)
    if slot == "crystalline":
        # The compound prototype families (rocksalt, zincblende, wurtzite,
        # fluorite) name a crystal structure just as fcc/bcc do, so a compound
        # substrate states this slot: 'fluorite-CeO2' needs no separate phase.
        return bool(re.search(r"\b(?:fcc|bcc|hcp|diamond|sc|simple cubic|rocksalt|"
                              r"zincblende|wurtzite|fluorite|face[- ]centred|"
                              r"face[- ]centered|body[- ]centred|body[- ]centered|"
                              r"hexagonal close)\b", text))
    if slot == "facet":
        compact = re.sub(r"[ ,\-]", "", text)
        return bool(
            re.search(r"\(\d{3,4}\)", compact)
            or re.search(r"\b(?:100|110|111|211|0001)\b", compact)
            or re.search(r"\bmiller\b", text)
        )
    if slot == "layers":
        return _near(text, r"layer|monolayer", number=rf"{_NUMBER}|{_WORD_NUMBERS}")
    if slot == "vacuum":
        return bool(re.search(r"\bvacuum\b", text)) and _near(text, r"vacuum")
    if slot == "freeze":
        verbs = r"freeze|frozen|fix|fixed|constrain|immobil|lock|pin|clamp|hold"
        return bool(re.search(verbs, text)) and bool(re.search(r"\b(?:top|bottom)\b", text))
    if slot == "adsorbate":
        # An adsorption request names both a host and an adsorbate.
        return len(_chemical_tokens(raw)) >= 2
    if slot == "dopant":
        return len(_chemical_tokens(raw)) >= 2
    if slot == "site":
        return bool(re.search(r"\bon-?top\b|\bon top\b|\bbridge\b|\bhollow\b|"
                              r"\bfcc site\b|\bhcp site\b|\bsite\b", text))
    if slot == "height":
        return _near(text, r"above|high\b|height|angstrom|å|\ba\b")
    if slot == "anchor":
        return bool(re.search(r"\banchor\w*\b|\bthrough (?:the )?(?:c\b|carbon|o\b|oxygen|"
                              r"n\b|nitrogen)|\bbonded through\b|\bvia (?:the )?carbon\b", text))
    if slot == "species":
        return bool(_MOLECULE_RE.search(raw)) or bool(
            re.search(rf"\b(?:{'|'.join(_MOLECULE_WORDS)})\b", text)
        ) or bool(_SYMBOL_RE.search(raw))
    if slot == "box":
        return _near(text, r"box|cell|angstrom|å\b")
    if slot == "chirality":
        return bool(re.search(r"\(\s*\d+\s*,\s*\d+\s*\)", text))
    if slot == "length":
        return _near(text, r"unit cell|cells|long|length|nm\b|angstrom|å")
    if slot == "prototype":
        return bool(re.search(r"\b(?:graphene|graphite|hbn|h-bn|rutile|anatase|nacl|"
                              r"rocksalt|zincblende|wurtzite|perovskite)\b", text))
    if slot == "defect_site":
        return bool(re.search(r"\batom\s+\d+\b|\bsite\s+\d+\b|\bindex\s+\d+\b|"
                              r"\bposition\s+\d+\b|\bfirst\b|\bsecond\b|\bthird\b|"
                              r"\bcorner\b|\bcentre\b|\bcenter\b", text))
    return True


# Region inference. Ordered most-specific first; the keys are FAMILY_REQUIRED keys.
def infer_region(request: str) -> str | None:
    """Best-effort keyword guess at which structure region a request belongs to."""
    text = f" {request.casefold()} "
    molecular = bool(_MOLECULE_RE.search(request)) or bool(
        re.search(rf"\b(?:molecule|molecular|{'|'.join(_MOLECULE_WORDS)})\b", text)
    )
    if re.search(r"\badsorb\w*\b|\bon-?top site\b|\badsorbate\b|\badatom\b", text) or (
        re.search(r"\b(?:above|onto|on)\b[^.]{0,30}\b(?:site|slab|surface)\b", text)
        and re.search(r"\b(?:surface|slab)\b|\(\d{3}\)", text)
    ):
        return "molecular_adsorption" if molecular else "atomic_adsorption"
    if re.search(r"\bnanotube\b|\bcnt\b|\bswcnt\b", text):
        return "nanotube"
    # Fixed prototype names only. The *compound* families (rocksalt, wurtzite, ...)
    # are checked last instead: they combine with every other region -- a
    # 'rocksalt-MgO(100) slab' is a surface request, not a prototype request --
    # so matching them here would mask the more specific region.
    if re.search(r"\b(?:hbn|h-bn|graphene|graphite|rutile|anatase|prototype)\b", text):
        return "prototype"
    if re.search(r"\bvacanc\w*\b|\b(?:remov\w*|delet\w*|knock\s*out)\b[^.]{0,20}\b(?:atom|site)\b",
                 text):
        return "vacancy"
    if re.search(r"\bsubstitut\w*\b|\bdope\w*\b|\bdoping\b|\bin place of\b|"
                 r"\b(?:replace|swap)\b[^.]{0,25}\b(?:atom|site|with)\b", text):
        return "substitution"
    surface = bool(re.search(r"\b(?:surface|slab|facet)\b|\(\d{3}\)", text))
    if surface and re.search(r"\bfreeze|\bfrozen|\bfix|\bconstrain|\bimmobil|\block|"
                             r"\bpin\b|\bclamp", text):
        return "surface_constraint"
    if surface:
        return "surface"
    if molecular:
        return "molecule"
    if re.search(r"\bbulk\b|\bcrystal\b|\bunit cell\b|\bsupercell\b|\blattice\b|"
                 r"\b(?:fcc|bcc|hcp|diamond)\b", text):
        return "bulk"
    if _COMPOUND_RE.search(text):  # a bare composition, no structure word
        return "prototype"
    return None


# --------------------------------------------------------------------------- #
# Post-build check: did the structure actually use the numbers you stated?
# --------------------------------------------------------------------------- #

# Numeric determinants we can both (a) read out of a request unambiguously and
# (b) compare against a builder argument. Each entry maps a slot to the tool it
# lands on, the argument name, and the builder's default when the model omits it
# (see tools_build.py / tools_surface.py -- kept in sync by the tests).
NUMERIC_SLOTS: dict[str, tuple[tuple[str, ...], str, float]] = {
    "layers": (("build_surface",), "layers", 4.0),
    "vacuum": (("build_surface",), "vacuum", 12.0),
    "box": (("build_molecule",), "box", 12.0),
    "height": (("add_atomic_adsorbate", "add_molecular_adsorbate"), "height", 1.8),
}

_WORD_VALUE = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_UNIT = r"(?:\s*(?:å|a|ang|angstroms?)\b)?"

# Ordered patterns per slot; the first match wins. Each must capture the number
# in group 1. Written narrowly: a miss (no warning) is far better than a false
# alarm that trains users to ignore the check.
_VALUE_PATTERNS: dict[str, tuple[str, ...]] = {
    "layers": (
        rf"({_NUMBER}|{_WORD_NUMBERS})[\s-]*(?:atomic[\s-]*)?layers?\b",
        rf"layers?\s*(?:=|:|of)?\s*({_NUMBER})\b",
    ),
    "vacuum": (
        rf"({_NUMBER}){_UNIT}\s*(?:of\s+)?vacuum\b",
        rf"vacuum\s*(?:gap|spacing|region|layer)?\s*(?:=|:|of)?\s*({_NUMBER})",
    ),
    "box": (
        rf"({_NUMBER}){_UNIT}\s*(?:cubic\s*)?box\b",
        rf"box\s*(?:edge|size|length)?\s*(?:=|:|of)?\s*({_NUMBER})",
    ),
    "height": (
        rf"(?:height|distance)\s*(?:=|:|of)?\s*({_NUMBER})",
        rf"({_NUMBER}){_UNIT}\s*(?:above|over|high|from the (?:surface|slab|top))",
    ),
}


def stated_values(request: str) -> dict[str, float]:
    """Numeric slot values the request states outright, e.g. {'vacuum': 12.0}."""
    text = f" {(request or '').casefold()} "
    found: dict[str, float] = {}
    for slot, patterns in _VALUE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            token = match.group(1)
            value = float(_WORD_VALUE[token]) if token in _WORD_VALUE else float(token)
            found[slot] = value
            break
    return found


@dataclass(frozen=True)
class ValueMismatch:
    """A number the request stated that the built structure did not use."""

    slot: str
    tool: str
    argument: str
    requested: float
    used: float
    defaulted: bool

    def line(self) -> str:
        how = (
            f"the model omitted '{self.argument}', so {self.tool} used its default"
            if self.defaulted
            else f"{self.tool} was called with {self.argument}={self.used:g}"
        )
        return (
            f"[warn] you asked for {self.slot} {self.requested:g}, but the structure "
            f"uses {self.used:g} -- {how}."
        )


def check_build(request: str, transcript) -> tuple[ValueMismatch, ...]:
    """Compare numbers stated in the request against the calls actually executed.

    Deterministic and post-hoc: it reads the executed transcript, so unlike the
    pre-flight hint it catches the dangerous case -- a request that *did* state a
    determinant which the model then dropped, silently falling back to a builder
    default. Only slots present in both the request and the transcript are
    checked; anything unclear is skipped rather than guessed.
    """
    stated = stated_values(request)
    if not stated:
        return ()
    executed = [
        item for item in (transcript or [])
        if item.get("success") and isinstance(item.get("args"), dict)
    ]
    mismatches: list[ValueMismatch] = []
    for slot, requested in stated.items():
        tools, argument, default = NUMERIC_SLOTS[slot]
        for item in executed:
            if item["tool"] not in tools:
                continue
            raw = item["args"].get(argument)
            defaulted = raw is None
            try:
                used = float(default if defaulted else raw)
            except (TypeError, ValueError):
                continue
            if abs(used - requested) > 1e-6:
                mismatches.append(ValueMismatch(
                    slot=slot, tool=item["tool"], argument=argument,
                    requested=requested, used=used, defaulted=defaulted,
                ))
            break
    return tuple(mismatches)


@dataclass(frozen=True)
class CompositionMismatch:
    """A compound the request named whose elements are not in the structure.

    Shares the `.line()` / `--strict` channel with :class:`ValueMismatch`, because
    it is the same class of failure: the build silently did something other than
    what the request said.
    """

    requested: str
    missing: tuple[str, ...]
    built: str

    slot: str = "composition"
    tool: str = "build_prototype"
    argument: str = "prototype"
    defaulted: bool = False

    def line(self) -> str:
        absent = ", ".join(self.missing)
        return (
            f"[warn] you asked for {self.requested}, but the structure is {self.built} "
            f"-- no {absent}. A different compound was built."
        )


def check_composition(request: str, invariants: dict | None) -> tuple[CompositionMismatch, ...]:
    """Verify a named compound's elements are actually present in the result.

    Deliberately narrow: only compound prototype names ('rocksalt-NiO', 'CeO2')
    are checked, not every element symbol in the request. Substitution and
    vacancy legitimately remove elements a request mentions, so a broader check
    would cry wolf -- and a missed warning is better than one users learn to
    ignore. Catches the observed failure where a request for rocksalt-NiO built
    rocksalt-MgO and still reported FINISHED.
    """
    from ..structure import split_compound

    if not invariants:
        return ()
    formula = invariants.get("formula") or {}
    present = {str(symbol) for symbol in formula}
    if not present:
        return ()

    built = "".join(f"{el}{n}" for el, n in formula.items())
    seen: set[str] = set()
    found: list[CompositionMismatch] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z][A-Za-z0-9]*)?", request or ""):
        try:
            compound = split_compound(token)
        except ValueError:
            continue
        if compound is None:
            continue
        family, chemical = compound
        name = f"{family}-{chemical}"
        if name in seen:
            continue
        seen.add(name)
        parsed = parse_binary_formula(chemical)
        if parsed is None:
            continue
        wanted = {parsed[0], parsed[2]}
        absent = tuple(sorted(wanted - present))
        if absent:
            found.append(CompositionMismatch(
                requested=name, missing=absent, built=built,
            ))
    return tuple(found)


@dataclass(frozen=True)
class RequestAdvisory:
    """Result of the heuristic pre-flight check. Never authoritative."""

    request: str
    region: str | None
    required: tuple[str, ...] = ()
    missing: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return self.region is not None and not self.missing

    def lines(self) -> list[str]:
        """Terminal-ready hint lines, always self-labelled as a heuristic."""
        if self.region is None:
            return [
                "[hint] keyword heuristic: could not guess a structure region for this "
                "request.",
                "[hint] the deterministic router may refuse it; supported regions are bulk, "
                "surface/slab, molecule, nanotube, prototype, adsorption, vacancy, "
                "substitution, constraint.",
                "[hint] advisory only - the request is sent to the model unchanged.",
            ]
        if not self.missing:
            named = ", ".join(self.required) if self.required else "(none)"
            return [
                f"[hint] keyword heuristic: looks like a '{self.region}' request; "
                f"required slots that appear stated: {named}.",
                "[hint] advisory only - a keyword match is not a guarantee the values are "
                "the ones you meant.",
            ]
        labels = "; ".join(SLOT_LABELS.get(slot, slot) for slot in self.missing)
        return [
            f"[hint] keyword heuristic: looks like a '{self.region}' request; these "
            f"required slots appear UNSTATED: {labels}.",
            "[hint] an under-specified request still builds a valid structure - just "
            "maybe not the one you meant (see training/CORPUS_RULE.md).",
            "[hint] advisory only - nothing is blocked or rewritten; the request is sent "
            "to the model unchanged.",
        ]


def check_request(request: str) -> RequestAdvisory:
    """Heuristically report required slots that look unstated. Never blocks."""
    raw = request or ""
    text = f" {raw.casefold()} "
    region = infer_region(raw)
    if region is None:
        return RequestAdvisory(request=raw, region=None)
    required = FAMILY_REQUIRED.get(region, ())
    missing = tuple(slot for slot in required if not _slot_stated(slot, raw, text))
    return RequestAdvisory(request=raw, region=region, required=required, missing=missing)
