"""Guided request builder: compose a slot-complete request before the model runs.

`request_check` tells you *after the fact* that a slot looks unstated. This module
removes the failure mode instead: pick a region, answer one question per required
slot, and the composed request is slot-complete by construction.

Three properties are deliberate, and `tests/test_guided.py` pins all three.

**The question list is derived, not written.** :data:`REGION_SLOTS` is checked
against ``request_check.FAMILY_REQUIRED`` at import time by
:func:`_assert_covers_rule`, so a slot added to the corpus rule breaks the wizard
loudly rather than silently letting an under-specified prompt through.

**The phrasings are corpus phrasings.** Each entry in :data:`TEMPLATES` is taken
from ``training/generators/paraphrase_templates_r6/<region>.json`` -- the same
templates the r5/r6 corpora were generated from -- so a composed request lands in
the distribution the adapter was fine-tuned on rather than in whatever wording
this module's author would have invented. They are copied rather than imported
because ``training/`` is not part of the installed runtime, the same reasoning
that duplicates ``FAMILY_REQUIRED``.

**The adsorption phrasings were chosen by measurement, not by reading the manual.**
r5 drops a stated adsorption height for most phrasings, and USER_MANUAL section 7's
advice (``at a height of 2.5 A``) is *not* sufficient on its own -- with the unit
written ``Å`` that form is still dropped. The worst case is Cu(100)/O/ontop, the
combination the corpus over-represents; of eleven phrasings measured against the
promoted adapter only one survived it, and that is the template used here. It
holds for 2.5 and 4.0 A on Cu(100)/O, and for Pt(111)/H, Fe(110)/N and Ni(111)/C.
Retention is still a property of the prompt rather than a guarantee, so the
post-build check in ``request_check.check_build`` -- and ``--strict`` -- remain the
thing that actually protects a batch run. See ``docs/GUIDED_INPUT.md``.

Values are normalised to the surface forms the generator emitted
(``training/generators/template_fill.py``): a facet is ``100`` not ``(1,1,1)``, a
repeat is ``2x2``, an anchor is ``carbon``, atom 1 is ``the first atom``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .request_check import FAMILY_REQUIRED, check_request


class SlotError(ValueError):
    """A slot answer that could not be parsed. The message is shown to the user."""


# --------------------------------------------------------------------------- #
# Value parsers -- each returns the surface form the corpus generator produced
# --------------------------------------------------------------------------- #


def _num(value: float) -> str:
    """Render a number the way ``template_fill._num`` did: 12.0 -> '12'."""
    return str(int(value)) if float(value).is_integer() else f"{float(value):g}"


def _positive_int(raw: str, what: str) -> int:
    try:
        value = int(raw.strip())
    except ValueError:
        raise SlotError(f"{what} must be a whole number, got {raw!r}") from None
    if value < 1:
        raise SlotError(f"{what} must be at least 1, got {value}")
    return value


def _positive_float(raw: str, what: str) -> float:
    try:
        value = float(raw.strip())
    except ValueError:
        raise SlotError(f"{what} must be a number, got {raw!r}") from None
    if value <= 0:
        raise SlotError(f"{what} must be greater than 0, got {_num(value)}")
    return value


def parse_element(raw: str) -> str:
    """A chemical symbol, normalised to ASE's capitalisation ('cu' -> 'Cu').

    One element, because that is what ``build_bulk``/``build_surface`` accept --
    they wrap ``ase.build.bulk``, which takes a single element. A compound typed
    here gets redirected to the prototype region rather than a bare rejection.
    """
    from ase.data import chemical_symbols

    token = raw.strip()
    if not token:
        raise SlotError("give a chemical symbol, e.g. Cu")
    symbol = token[0].upper() + token[1:].lower()
    if symbol in chemical_symbols:
        return symbol
    raise SlotError(_element_rejection(token))


def _element_rejection(token: str) -> str:
    """Explain a rejected element, redirecting compounds to the prototype region."""
    from ..structure import parse_binary_formula, split_compound

    if parse_binary_formula(token) is None:
        return f"{token!r} is not a chemical symbol (try Cu, Fe, Pt, Au)"
    # It looks like a binary compound. Say whether we can build it, and where.
    try:
        compound = split_compound(token)
    except ValueError:
        compound = None
    if compound is not None:
        return (
            f"{token} is a compound -- this slot takes one element, because bulk "
            f"and slab builders wrap ase.build.bulk. Build it in the 'prototype' "
            f"region as '{compound[0]}-{compound[1]}'."
        )
    return (
        f"{token} is a compound -- this slot takes one element. Compounds are built "
        f"in the 'prototype' region, but {token} has no tabulated prototype; see "
        f"structure.COMPOUND_LATTICE for the compositions that do."
    )


def parse_substrate(raw: str) -> str:
    """One element, or a compound prototype name.

    Returns either a symbol ('Cu') or a canonical '<family>-<formula>'
    ('rocksalt-MgO'). The hyphenated form is what r5 handles first-try; naming the
    bare formula makes it guess the family and retry (docs/GUIDED_INPUT.md).
    A compound also states the crystal phase, so the phase question is then
    skipped -- see :func:`_substrate_is_compound`.
    """
    from ase.data import chemical_symbols
    from ..structure import split_compound

    token = raw.strip()
    if not token:
        raise SlotError("give an element (Cu) or a compound (MgO)")
    symbol = token[0].upper() + token[1:].lower()
    if symbol in chemical_symbols and len(token) <= 2:
        return symbol
    compound = split_compound(token)  # raises with a helpful message when close
    if compound is not None:
        return f"{compound[0]}-{compound[1]}"
    if symbol in chemical_symbols:
        return symbol
    raise SlotError(
        f"{token!r} is neither a chemical symbol nor a tabulated compound "
        f"(try Cu, Fe, or MgO, ZnO, CeO2, rocksalt-NiO)"
    )


def _substrate_is_compound(values: dict[str, Any]) -> bool:
    """True when the substrate answer already named a crystal family."""
    from ..structure import COMPOUND_FAMILIES

    substrate = str(values.get("element") or "")
    return substrate.split("-")[0].lower() in COMPOUND_FAMILIES


def parse_phase(raw: str) -> str:
    """A crystal phase from the ``build_bulk`` / ``build_surface`` enum."""
    phase = raw.strip().lower()
    if phase not in PHASES:
        raise SlotError(f"phase must be one of {', '.join(PHASES)}, got {raw!r}")
    return phase


def parse_facet(raw: str) -> str:
    """Miller indices as the templates write them: '(1,1,1)' -> '111'."""
    digits = re.sub(r"[\s(),\[\]-]", "", raw.strip())
    if not digits.isdigit() or len(digits) not in (3, 4):
        raise SlotError(
            f"a facet is 3 or 4 Miller indices, e.g. 111, (100), 0001 -- got {raw!r}"
        )
    return digits


def parse_repeat(raw: str) -> tuple[int, int, int]:
    """A supercell repeat: '2', '2x2', '2 2 1', '2,2,1' -> (2, 2, 1)."""
    parts = [part for part in re.split(r"[x,\s*]+", raw.strip().lower()) if part]
    if not 1 <= len(parts) <= 3:
        raise SlotError(f"a repeat is 1 to 3 numbers, e.g. 2x2x1 -- got {raw!r}")
    values = [_positive_int(part, "each repeat factor") for part in parts]
    while len(values) < 3:
        values.append(1)
    return values[0], values[1], values[2]


def parse_site(raw: str) -> str:
    """An adsorption site from the ``add_*_adsorbate`` enum."""
    site = raw.strip().lower().replace("on-top", "ontop").replace("on top", "ontop")
    site = re.sub(r"\s*(site|hollow site)$", "", site).strip() or site
    if site.endswith(" hollow"):  # 'fcc hollow' / 'hcp hollow' both build a hollow
        site = "hollow"
    if site not in SITES:
        raise SlotError(f"site must be one of {', '.join(SITES)}, got {raw!r}")
    return site


def parse_species(raw: str) -> str:
    """A molecule ASE can build, e.g. H2O, CO, NH3."""
    from ase.collections import g2

    token = raw.strip()
    if not token:
        raise SlotError("give a molecular formula, e.g. H2O")
    names = set(g2.names)
    if token in names:
        return token
    for name in names:  # forgive case: 'h2o' -> 'H2O'
        if name.lower() == token.lower():
            return name
    raise SlotError(f"ASE has no molecule {token!r} (try H2O, CO, CO2, NH3, CH4)")


def parse_anchor(raw: str) -> str:
    """The adsorbed molecule's bonding atom, as a word ('C' -> 'carbon')."""
    token = raw.strip().lower()
    if not token:
        raise SlotError("give the anchor atom, e.g. carbon")
    word = _ANCHOR_WORDS.get(token, token)
    if not re.fullmatch(r"[a-z]+", word):
        raise SlotError(f"give the anchor atom as a word, e.g. carbon -- got {raw!r}")
    return word


def parse_chirality(raw: str) -> str:
    """Nanotube chirality: '6,3' or '(6,3)' -> '(6,3)'."""
    parts = [p for p in re.split(r"[(),\s]+", raw.strip()) if p]
    if len(parts) != 2:
        raise SlotError(f"chirality is two numbers, e.g. (6,3) -- got {raw!r}")
    n, m = (_positive_int(parts[0], "n"), int(parts[1]) if parts[1].isdigit() else -1)
    if m < 0:
        raise SlotError(f"chirality is two numbers, e.g. (6,3) -- got {raw!r}")
    return f"({n},{m})"


def parse_prototype(raw: str) -> str:
    """A named prototype, resolved through ASE_auto_build's alias table."""
    from ..structure import resolve_prototype

    try:
        return resolve_prototype(raw.strip())
    except Exception as exc:  # resolve_prototype raises with the available names
        raise SlotError(str(exc)) from None


def parse_defect_site(raw: str) -> str:
    """Which atom the defect targets, phrased as the templates do."""
    token = raw.strip().lower()
    if token in {"first", "1st"}:
        return "first"
    index = _positive_int(token, "the atom index")
    return "first" if index == 1 else f"index {index}"


def parse_side(raw: str) -> str:
    side = raw.strip().lower()
    if side not in SIDES:
        raise SlotError(f"side must be 'top' or 'bottom', got {raw!r}")
    return side


PHASES = ("fcc", "bcc", "hcp", "diamond", "sc")
SITES = ("ontop", "bridge", "hollow")
SIDES = ("top", "bottom")
CONVENTIONS = ("primitive", "conventional cubic")

_ANCHOR_WORDS = {
    "c": "carbon", "o": "oxygen", "n": "nitrogen", "h": "hydrogen", "s": "sulfur",
}


# --------------------------------------------------------------------------- #
# Slot schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Slot:
    """One question the wizard asks.

    `rule_slot` names the ``FAMILY_REQUIRED`` entry this question satisfies, which
    is what lets :func:`_assert_covers_rule` prove the wizard asks for everything
    the corpus rule demands. Two slots may share one `rule_slot` (a freeze is a
    side plus a count); a purely optional question has ``rule_slot=None``.
    """

    key: str
    label: str
    parse: Callable[[str], Any]
    rule_slot: str | None = None
    required: bool = True
    hint: str = ""
    default: Any = None
    # Asked only when this predicate over the answers so far is False. Used where
    # one answer already supplies another's slot -- a compound substrate names the
    # crystal family, so 'crystal phase' becomes redundant rather than optional.
    # `required` stays True: the rule slot is still covered, just by another answer.
    skip_if: Callable[[dict[str, Any]], bool] | None = None

    def ask(self, raw: str) -> Any:
        """Parse one answer, applying the default when the answer is blank."""
        if not raw.strip():
            if self.required:
                raise SlotError(f"{self.label} is required -- it decides the structure")
            return self.default
        return self.parse(raw)


def _slot(key, label, parse, rule_slot=None, *, required=True, hint="", default=None,
          skip_if=None):
    return Slot(key, label, parse, rule_slot, required, hint, default, skip_if)


_ELEMENT = _slot("element", "element", parse_element, "element", hint="Cu, Fe, Pt")
# Regions whose builders can take a compound: build_surface accepts a `prototype`,
# and repeat/make_vacancy/substitute act on whatever the active structure is.
_SUBSTRATE = _slot("element", "element or compound", parse_substrate, "element",
                   hint="Cu, Fe, or MgO / ZnO / rocksalt-NiO")
_FACET = _slot("facet", "facet (Miller)", parse_facet, "facet", hint="111, 100, (110)")
_LAYERS = _slot("layers", "layers", lambda r: _positive_int(r, "layers"), "layers",
                hint="slab thickness, e.g. 4")
_VACUUM = _slot("vacuum", "vacuum (A)", lambda r: _positive_float(r, "vacuum"), "vacuum",
                hint="gap above the slab, e.g. 12")
_HEIGHT = _slot("height", "height (A)", lambda r: _positive_float(r, "height"), "height",
                hint="adsorbate height above the site, e.g. 1.8")
_SITE = _slot("site", "site", parse_site, "site", hint="/".join(SITES))
_PHASE_REQ = _slot("phase", "crystal phase", parse_phase, "crystalline",
                   hint="/".join(PHASES), skip_if=_substrate_is_compound)
_PHASE_OPT = _slot("phase", "crystal phase", parse_phase, required=False,
                   hint=f"optional; {'/'.join(PHASES)}",
                   skip_if=_substrate_is_compound)
_REPEAT_PLANE = _slot("repeat", "in-plane repeat", parse_repeat, required=False,
                      hint="optional; e.g. 2x2 (default 1x1)", default=(1, 1, 1))
_REPEAT_BULK = _slot("repeat", "supercell repeat", parse_repeat, required=False,
                     hint="optional; e.g. 2x2x1 (default 1x1x1)", default=(1, 1, 1))
_CONVENTION = _slot("convention", "cell convention",
                    lambda r: _choice(r, CONVENTIONS, "cell convention"), required=False,
                    hint="optional; primitive / conventional cubic")
_DEFECT = _slot("defect_site", "which atom", parse_defect_site, "defect_site",
                hint="atom index, e.g. 1")


def _choice(raw: str, allowed: tuple[str, ...], what: str) -> str:
    token = " ".join(raw.strip().lower().split())
    for option in allowed:
        if token == option or (len(token) > 2 and option.startswith(token)):
            return option
    raise SlotError(f"{what} must be one of {', '.join(allowed)}, got {raw!r}")


_SURFACE_BASE = (_SUBSTRATE, _FACET, _LAYERS, _VACUUM, _PHASE_OPT, _REPEAT_PLANE)

REGION_SLOTS: dict[str, tuple[Slot, ...]] = {
    "bulk": (_SUBSTRATE, _PHASE_REQ, _REPEAT_BULK, _CONVENTION),
    "surface": _SURFACE_BASE,
    "surface_constraint": _SURFACE_BASE + (
        _slot("freeze_side", "freeze which side", parse_side, "freeze",
              hint="/".join(SIDES)),
        _slot("freeze_count", "how many layers to freeze",
              lambda r: _positive_int(r, "the frozen-layer count"), "freeze",
              hint="e.g. 2"),
    ),
    "atomic_adsorption": _SURFACE_BASE + (
        _slot("adsorbate", "adsorbate atom", parse_element, "adsorbate", hint="O, H, C"),
        _SITE, _HEIGHT,
    ),
    "molecular_adsorption": _SURFACE_BASE + (
        _slot("adsorbate", "adsorbate molecule", parse_species, "adsorbate", hint="CO"),
        _SITE, _HEIGHT,
        _slot("anchor", "anchor atom", parse_anchor, "anchor",
              hint="the bonding atom, e.g. carbon"),
    ),
    "molecule": (
        _slot("species", "species", parse_species, "species", hint="H2O, CO2, NH3"),
        _slot("box", "box edge (A)", lambda r: _positive_float(r, "the box edge"), "box",
              hint="cubic box, e.g. 12"),
    ),
    "nanotube": (
        _slot("chirality", "chirality (n,m)", parse_chirality, "chirality", hint="(6,3)"),
        _slot("length", "length (unit cells)", lambda r: _positive_int(r, "length"),
              "length", hint="e.g. 2"),
    ),
    "prototype": (
        _slot("prototype", "prototype", parse_prototype, "prototype",
              hint="graphene, hBN, rutile-TiO2, or <family> <formula> e.g. "
                   "rocksalt MgO"),
    ),
    "vacancy": (_SUBSTRATE, _PHASE_REQ, _DEFECT, _REPEAT_BULK, _CONVENTION),
    "substitution": (
        _SUBSTRATE, _PHASE_REQ, _DEFECT,
        _slot("dopant", "dopant", parse_element, "dopant", hint="the replacing element"),
        _REPEAT_BULK, _CONVENTION,
    ),
}

# Menu order: bulk-like first, then surfaces, then the standalone builders.
REGION_ORDER = (
    "bulk", "surface", "surface_constraint", "atomic_adsorption",
    "molecular_adsorption", "molecule", "nanotube", "prototype",
    "vacancy", "substitution",
)

REGION_BLURBS: dict[str, str] = {
    "bulk": "elemental bulk crystal",
    "surface": "surface slab",
    "surface_constraint": "surface slab with frozen layers",
    "atomic_adsorption": "single atom adsorbed on a slab",
    "molecular_adsorption": "molecule adsorbed on a slab",
    "molecule": "isolated molecule in a box",
    "nanotube": "single-walled carbon nanotube",
    "prototype": "prototype: oxides/sulfides (MgO, ZnO, CeO2), graphene, hBN",
    "vacancy": "bulk supercell with an atom removed",
    "substitution": "bulk supercell with an atom replaced",
}


def _assert_covers_rule() -> None:
    """Fail at import if the wizard stopped matching the corpus rule."""
    rule_regions = {name for name in FAMILY_REQUIRED if name != "clarification"}
    if set(REGION_SLOTS) != rule_regions:
        raise RuntimeError(
            "guided.REGION_SLOTS is out of sync with request_check.FAMILY_REQUIRED: "
            f"{set(REGION_SLOTS) ^ rule_regions}"
        )
    if set(REGION_ORDER) != rule_regions:
        raise RuntimeError("guided.REGION_ORDER does not list every region")
    for region, slots in REGION_SLOTS.items():
        asked = {slot.rule_slot for slot in slots if slot.required and slot.rule_slot}
        required = set(FAMILY_REQUIRED[region])
        if asked != required:
            raise RuntimeError(
                f"guided.REGION_SLOTS[{region!r}] asks for {sorted(asked)} but the "
                f"corpus rule requires {sorted(required)}"
            )


_assert_covers_rule()


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #

# Verbatim from training/generators/paraphrase_templates_r6/<region>.json, except
# the two adsorption regions -- see ADAPTED_TEMPLATES below. Chosen so that the
# filled prompt satisfies check_request with zero missing slots.
TEMPLATES: dict[str, str] = {
    "bulk":
        "Create a bulk {crystalline} {element} crystal using a {repeat3} repeat.",
    "surface":
        "Build a {suffix} {element}({facet}) slab with {layers} layers and "
        "{vacuum} Å vacuum.",
    "surface_constraint":
        "Build a {suffix} {element}({facet}) slab ({layers} layers, {vacuum} Å "
        "vacuum) and freeze the {freeze_count} {freeze_side} layers.",
    "atomic_adsorption":
        "Put one {adsorbate} atom at a height of {height} angstrom on the {site} "
        "site of a {suffix} {element}({facet}) {layers}-layer slab with {vacuum} "
        "Å vacuum.",
    "molecular_adsorption":
        "I need a {suffix} {element}({facet}) slab, {layers} layers thick with "
        "{vacuum} angstrom of vacuum, carrying {adsorbate} bound at the {site} "
        "site, anchored through its {anchor}, at a height of {height} angstrom.",
    "molecule":
        "Build an isolated {species} molecule in a {box} Å cubic box.",
    "nanotube":
        "Build a {chirality} carbon nanotube {length} unit cells long.",
    "prototype":
        "Build the {prototype} prototype.",
    "vacancy":
        "Build a {repeat3} {crystalline} {element} supercell and remove the "
        "{defect_ordinal} atom.",
    "substitution":
        "Build a {repeat3} {crystalline} {element} supercell and replace the "
        "{defect_ordinal} atom with {dopant}.",
}

# The two adsorption templates are corpus templates with one clause rewritten;
# every other region's template is byte-identical to its source, which
# tests/test_guided.py enforces. Each entry is (source template index, why).
ADAPTED_TEMPLATES: dict[str, tuple[int, str]] = {
    "atomic_adsorption": (
        4,
        "template 4's skeleton with '{height} Å above the {site} site' rewritten to "
        "'at a height of {height} angstrom on the {site} site'. Template 4 as written "
        "loses the height on Cu(100)/O/ontop; this form keeps it. The unit matters: "
        "the same sentence with 'Å' instead of 'angstrom' is dropped again.",
    ),
    "molecular_adsorption": (
        8,
        "template 8 with 'anchored through its {anchor},' spliced in from template 4 "
        "of the same file. Template 8 alone says 'bound ... through its carbon', "
        "which request_check's anchor heuristic does not recognise, so the wizard "
        "would have flagged its own prompt as missing the anchor.",
    ),
}

# Every adsorption record in pilot_r5/r6 used exactly one height per tool, so any
# other value is extrapolation for the adapter. cross_check says so out loud.
CORPUS_HEIGHTS: dict[str, float] = {
    "atomic_adsorption": 1.8,
    "molecular_adsorption": 1.9,
}


def format_dict(region: str, values: dict[str, Any]) -> dict[str, str]:
    """Map parsed slot values onto the template's placeholder vocabulary.

    Mirrors ``training/generators/template_fill.format_dict`` for the placeholders
    the wizard uses, including its defaults for an unstated repeat.
    """
    out: dict[str, str] = {}
    phase = values.get("phase")
    element = values.get("element")
    compound = _substrate_is_compound(values)

    if element:
        # A surface template has no {crystalline} placeholder, so an optional phase
        # rides along on the element: 'fcc Cu(100)', as in surface_constraint #5.
        # A compound needs neither -- 'rocksalt-MgO' already names the family, and
        # that hyphenated form is what the model maps straight onto a prototype.
        out["element"] = (
            element if compound or not phase or region in _BULK_LIKE
            else f"{phase} {element}"
        )
    if region in _BULK_LIKE:
        convention = values.get("convention")
        if compound:
            # {crystalline} is a required placeholder in these templates; the
            # family is the phase, so let the name carry it and blank the slot.
            out["crystalline"] = ""
            out["element"] = element
        elif phase:
            out["crystalline"] = f"{convention} {phase}" if convention else phase

    a, b, c = values.get("repeat") or (1, 1, 1)
    out["suffix"] = f"{a}x{b}"
    out["repeat3"] = f"{a}x{b}x{c}"

    for key in ("facet", "site", "anchor", "prototype", "chirality", "species",
                "dopant", "freeze_side"):
        if values.get(key) is not None:
            out[key] = str(values[key])
    for key in ("layers", "length", "freeze_count"):
        if values.get(key) is not None:
            out[key] = str(values[key])
    for key in ("vacuum", "height", "box"):
        if values.get(key) is not None:
            out[key] = _num(values[key])
    if values.get("adsorbate"):
        out["adsorbate"] = str(values["adsorbate"])
    if values.get("defect_site"):
        out["defect_ordinal"] = str(values["defect_site"])
    return out


_BULK_LIKE = ("bulk", "vacancy", "substitution")


def compose(region: str, values: dict[str, Any]) -> str:
    """Render a slot-complete request for `region` from parsed slot values."""
    if region not in TEMPLATES:
        raise KeyError(f"unknown region {region!r}")
    filled = TEMPLATES[region].format(**format_dict(region, values))
    # A blanked placeholder (a compound needs no separate phase) leaves a double
    # space; collapse rather than ship odd whitespace to the model.
    return re.sub(r"\s+", " ", filled).strip()


def cross_check(region: str, values: dict[str, Any]) -> tuple[str, ...]:
    """Warnings a single slot cannot catch, e.g. freezing more layers than exist."""
    warnings: list[str] = []
    layers, frozen = values.get("layers"), values.get("freeze_count")
    if layers and frozen and frozen > layers:
        warnings.append(
            f"you are freezing {frozen} layers of a {layers}-layer slab -- the whole "
            "slab would be fixed"
        )
    repeat, defect = values.get("repeat"), values.get("defect_site")
    if defect == "first" and repeat == (1, 1, 1) and region == "vacancy":
        warnings.append(
            "removing the only atom of a 1x1x1 cell leaves nothing; give a repeat"
        )
    trained = CORPUS_HEIGHTS.get(region)
    height = values.get("height")
    if trained is not None and height is not None and abs(height - trained) > 1e-9:
        warnings.append(
            f"every adsorption record the adapter trained on used height {_num(trained)} "
            f"A, so {_num(height)} is extrapolation -- this wording keeps it in the "
            "cases measured, but check the executed call, or run with --strict"
        )
    return tuple(warnings)


# --------------------------------------------------------------------------- #
# The wizard
# --------------------------------------------------------------------------- #


@dataclass
class _Console:
    """Thin I/O seam so the wizard is testable without a terminal."""

    read: Callable[[str], str] = input
    write: Callable[[str], None] = print
    lines: list[str] = field(default_factory=list)

    def say(self, text: str = "") -> None:
        self.lines.append(text)
        self.write(text)


class Cancelled(Exception):
    """The user typed 'q' at a prompt."""


def _prompt(console: _Console, label: str) -> str:
    try:
        raw = console.read(label)
    except EOFError:
        raise Cancelled() from None
    if raw.strip().lower() in {"q", "quit", ":q"}:
        raise Cancelled()
    return raw


def choose_region(console: _Console) -> str:
    console.say("\nWhat kind of structure? ('q' cancels)")
    width = max(len(name) for name in REGION_ORDER)
    for number, name in enumerate(REGION_ORDER, start=1):
        console.say(f"  {number:2d}) {name:<{width}}  {REGION_BLURBS[name]}")
    while True:
        raw = _prompt(console, "region> ").strip().lower()
        if raw.isdigit() and 1 <= int(raw) <= len(REGION_ORDER):
            return REGION_ORDER[int(raw) - 1]
        if raw in REGION_SLOTS:
            # Exact name wins: 'surface' is also a prefix of 'surface_constraint'.
            return raw
        matches = [name for name in REGION_ORDER if name.startswith(raw)] if raw else []
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            console.say(f"  ambiguous -- did you mean {' or '.join(matches)}?")
        else:
            console.say(f"  pick 1-{len(REGION_ORDER)} or type a region name")


def fill_slots(region: str, console: _Console) -> dict[str, Any]:
    """Ask every slot for `region`. 'b' steps back one question."""
    slots = REGION_SLOTS[region]
    required = [slot for slot in slots if slot.required]
    # Questions can outnumber rule slots -- a freeze is a side plus a count -- so
    # this counts questions, and _report counts the rule slots they satisfy.
    console.say(
        f"\n{region}: {len(required)} required question(s)"
        f"{', then optional ones' if len(required) < len(slots) else ''}."
        "  ('b' goes back, 'q' cancels)"
    )
    width = max(len(slot.label) for slot in slots)
    values: dict[str, Any] = {}
    index = 0
    while index < len(slots):
        slot = slots[index]
        if slot.skip_if is not None and slot.skip_if(values):
            values.pop(slot.key, None)  # a step back may have made it redundant
            index += 1
            continue
        mark = " " if slot.required else "?"
        hint = f"  [{slot.hint}]" if slot.hint else ""
        raw = _prompt(console, f" {mark}{slot.label:<{width}}{hint} > ")
        if raw.strip().lower() == "b":
            index = max(0, index - 1)
            continue
        try:
            values[slot.key] = slot.ask(raw)
        except SlotError as exc:
            console.say(f"   -> {exc}")
            continue
        index += 1
    return values


def _report(console: _Console, region: str, values: dict[str, Any], request: str) -> None:
    console.say("\ncomposed request:")
    console.say(f"  {request}")
    advisory = check_request(request)
    if advisory.region == region and not advisory.missing:
        console.say(f"  [ok] pre-flight: all {len(FAMILY_REQUIRED[region])} required "
                    f"slots for '{region}' are stated.")
    else:  # pragma: no cover - _assert_covers_rule and the tests keep this unreachable
        console.say(f"  [warn] pre-flight still reports: region={advisory.region}, "
                    f"missing={advisory.missing or '()'}")
    for warning in cross_check(region, values):
        console.say(f"  [warn] {warning}")


def run_wizard(console: _Console | None = None, *, region: str | None = None) -> str | None:
    """Drive the guided form. Returns the composed request, or None if cancelled."""
    console = _Console() if console is None else console
    pinned = region
    try:
        while True:
            chosen = pinned or choose_region(console)
            values = fill_slots(chosen, console)
            request = compose(chosen, values)
            _report(console, chosen, values, request)
            console.say("")
            answer = _prompt(
                console, "[enter] build   [e] edit   [r] restart   [q] cancel > "
            ).strip().lower()
            if answer in {"", "y", "build"}:
                return request
            # 'r' returns to the menu (unless a region was pinned by --region);
            # anything else re-asks this region's slots without the menu.
            pinned = region if answer == "r" else chosen
    except Cancelled:
        console.say("(cancelled)")
        return None
