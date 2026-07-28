# Structured request rule for the training corpus

A training request is valid only if it **names every required slot** for its
region. The corpus is designed *within* this rule so a conforming prompt maps to
exactly one canonical structure — ambiguity is removed at the source rather than
resolved by the model at run time. This generalizes the r4 fix (which made one
family explicit by hand) into a rule the whole corpus obeys.

The machine-checkable form lives in `training/generators/request_rule.py`; this
document is the human contract.

## Slot vocabulary (the "necessary information")

| slot | meaning | example surface form |
| --- | --- | --- |
| `element` | chemical symbol of the host | `Pt`, `Fe` |
| `crystalline` | crystal phase | `fcc`, `bcc`, `diamond` |
| `facet` | Miller indices of a surface | `(111)`, `(100)` |
| `layers` | slab thickness in atomic layers | `4 layers` |
| `vacuum` | vacuum gap in Å | `12 Å vacuum` |
| `species` | molecule/compound formula | `CO`, `H2O`, `TiO2` |
| `box` | cubic cell edge for a molecule (Å) | `12 Å box` |
| `adsorbate` | adsorbed atom or molecule | `O`, `CO` |
| `site` | adsorption site | `ontop`, `bridge`, `fcc hollow` |
| `height` | adsorption height (Å) | `1.9 Å above` |
| `anchor` | anchor atom for a molecular adsorbate | `through carbon` |
| `chirality` | nanotube `(n,m)` | `(6,3)` |
| `length` | nanotube length in unit cells | `2 unit cells` |
| `defect_site` | atom targeted by a defect | `atom 1`, `the first site` |
| `dopant` | substituent element | `Au` |
| `freeze` | constraint side + count | `fix the bottom two layers` |

## Required slots per region

Required = the **non-inferable structural determinants**. A slot fixed by
another stated slot is *not* required (in our material set the element implies
the crystal phase for a **surface**, so `crystalline` is optional there; for a
**bulk** the phase is stated because several hosts admit more than one phase).

| region | required slots | optional |
| --- | --- | --- |
| `bulk` | element, crystalline | repeat |
| `vacancy` | element, crystalline, defect_site | repeat |
| `substitution` | element, crystalline, defect_site, dopant | repeat |
| `surface` | element, facet, layers, vacuum | crystalline, repeat |
| `surface_constraint` | element, facet, layers, vacuum, freeze | crystalline, repeat |
| `atomic_adsorption` | element, facet, layers, vacuum, adsorbate, site, height | crystalline, repeat |
| `molecular_adsorption` | element, facet, layers, vacuum, adsorbate, site, height, anchor | crystalline, repeat |
| `molecule` | species, box | charge, multiplicity |
| `nanotube` | chirality, length | vacuum |
| `prototype` | prototype name | — |
| `clarification` | (region slots) **minus exactly one**, deliberately withheld | — |

`clarification` is the one region that intentionally violates the rule: the
first prompt omits exactly one required slot (e.g. the facet of "an iron
surface"), the model must call `ask_clarification`, and the follow-up user turn
supplies the missing slot.

## Paraphrase contract

Paraphrases vary surface form freely — word order, synonyms, scientific vs.
decimal notation, shorthand — **but must state every required slot value for
their case**. A paraphrase that drops a required value (e.g. omits vacuum on a
slab) is rejected by `request_rule.missing_slots` before it can enter the
corpus, in addition to the existing execute + replay + invariant + policy gates.
No paraphrase may introduce a slot value that differs from the canonical recipe.

## Planned region: `crystal` — NOT ACTIVE

Reserved for `build_crystal` (space group + fractional basis), the one builder
with no corpus coverage. It is **not** in `request_rule.FAMILY_REQUIRED` and must
not be added there until the whole change lands together — the guided input form
asserts rule/UI agreement at import and will break every entry point otherwise.
Planned required slots: `formula`, `spacegroup`, `lattice`.

Full plan, including why this family is higher-risk than the others and the extra
promotion bars it needs: **HANDBOOK §12**.

## Current conformance (pilot_r4, 1,314 prompts)

Auditing the existing templated prompts against this rule: **43.3% conform.**
`atomic_adsorption` and `surface_constraint` are 0% (they drop `vacuum`);
`surface` and `substitution` are 67%. Bringing the corpus within the rule means
rewriting those canonical templates and generating rule-conforming paraphrases
to replace the ambiguous ones.
