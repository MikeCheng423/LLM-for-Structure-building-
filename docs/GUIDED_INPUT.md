# Guided input — `ASE_auto_build --guided`

A request maps to one correct structure only if it names every structural
determinant (`training/CORPUS_RULE.md`). The pre-flight hint tells you *after*
you have typed a request that a slot looks unstated. Guided input removes the
failure mode instead: pick a region, answer one question per required slot, and
the composed request is slot-complete by construction.

```bash
ASE_auto_build --guided                            # region menu, then the form
ASE_auto_build --guided --region surface_constraint # skip the menu
```

Inside the REPL, type `guided` at any `request>` prompt to do the same for one
request. `b` goes back a question, `q` cancels. The form runs **before the model
loads**, so a mistyped slot costs no GPU time.

```
$ ASE_auto_build --guided --region atomic_adsorption

atomic_adsorption: 7 required slot(s), then optional ones.  ('b' goes back, 'q' cancels)
  element              [Cu, Fe, Pt]                    > Cu
  facet (Miller)       [111, 100, (110)]               > 100
  layers               [slab thickness, e.g. 4]        > 5
  vacuum (A)           [gap above the slab, e.g. 12]   > 12
 ?crystal phase        [optional; fcc/bcc/hcp/...]     >
 ?in-plane repeat      [optional; e.g. 2x2]            > 2x2
  adsorbate atom       [O, H, C]                       > O
  site                 [ontop/bridge/hollow]           > ontop
  height (A)           [e.g. 1.8]                      > 2.5

composed request:
  Put one O atom at a height of 2.5 angstrom on the ontop site of a 2x2 Cu(100) 5-layer slab with 12 Å vacuum.
  [ok] pre-flight: all 7 required slots for 'atomic_adsorption' are stated.
  [warn] every adsorption record the adapter trained on used height 1.8 A, so 2.5 is
         extrapolation -- this wording keeps it in the cases measured, but check the
         executed call, or run with --strict

[enter] build   [e] edit   [r] restart   [q] cancel >
```

## Why the wording is not free-form

Two constraints shape `TEMPLATES` in `src/ase_auto_build/ase_agent/guided.py`.

**In-distribution phrasing.** Each template is a phrasing template the r5/r6
corpora were generated from (`training/generators/paraphrase_templates_r6/`), so
a composed request lands in the distribution the adapter was fine-tuned on. They
are copied into the runtime rather than imported because `training/` is not part
of the installed package — the same reason `FAMILY_REQUIRED` is duplicated in
`request_check.py`.

**Slot visibility to the pre-flight check.** The corpus-side rule
(`generators/request_rule.py`) is stronger than the runtime heuristic
(`request_check.py`): it compares a prompt against known canonical values, which
do not exist yet at run time. Some corpus templates therefore pass the corpus
rule but read as missing a slot to the runtime checker. Templates were screened
so that the composed request satisfies `check_request` with **zero** missing
slots — otherwise the wizard would flag its own output.

## The adsorption height: what was actually measured

`USER_MANUAL.md` §7 says r5 drops the height in `2.5 Å above the ontop site` but
honours `at a height of 2.5 Å`. **That advice is not sufficient.** The unit token
matters as much as the clause, and the worst case is Cu(100)/O/ontop — the
combination the corpus over-represents.

Measured against the promoted r5 adapter, greedy decoding
(`llm_local.py` sets `do_sample=False`, so these are reproducible), on
`2x2 Cu(100)`, 5 layers, 12 Å vacuum, O at ontop, requested height 2.5 Å:

| Phrasing | Height |
| --- | --- |
| `...drop an O atom onto the ontop site at a height of 2.5 Å.` (corpus tpl 9) | dropped |
| `...adsorb O at the ontop site with a height of 2.5 A.` (tpl 6) | dropped |
| `Put one O atom 2.5 Å above the ontop site of...` (tpl 4) | dropped |
| `...carrying a single O adsorbate positioned 2.5 angstrom above the ontop site.` (tpl 8) | dropped |
| `...then position O 2.5 Å above the ontop adsorption site.` (tpl 11) | dropped |
| `Build 2x2 Cu(100), 5 layers, 12 angstrom vacuum, and place O at the ontop site 2.5 angstrom high.` (tpl 1) | dropped |
| `...drop an O atom onto the ontop site at a height of 2.5 angstrom.` (tpl 9, unit spelled) | dropped |
| `Put one O atom at a height of 2.5 **Å** on the ontop site of...` | dropped |
| **`Put one O atom at a height of 2.5 angstrom on the ontop site of...`** | **kept** |

The last row is the template the wizard uses. The `Å` / `angstrom` rows differ by
one token and nothing else. Repeating an identical prompt reproduced its result,
confirming the split is deterministic rather than sampling noise.

That template also held for:

| Case | Requested | Result |
| --- | --- | --- |
| Cu(100)/O/ontop | 4.0 | kept |
| Pt(111)/H/bridge | 1.3 | kept |
| Fe(110)/N/hollow | 3.2 | kept |
| Ni(111)/C/hollow | 2.2 | kept |
| Cu(100)/O/ontop | 1.8 | omitted → builder default 1.8, correct |

The molecular template (`add_molecular_adsorbate`) kept 2.4 Å on Fe(110)/CO
directly, and reached 2.6 Å on Cu(100)/CO after one failed call and a self-correcting
retry.

### Why this is fragile at all

Every adsorption record in `pilot_r5` and `pilot_r6` uses exactly **one** height
per tool:

```
pilot_r5/train: add_atomic_adsorbate 1.8 ×738,  add_molecular_adsorbate 1.9 ×174
pilot_r6/train: add_atomic_adsorbate 1.8 ×1230, add_molecular_adsorbate 1.9 ×290
```

The height slot is constant across the entire corpus, so the adapter had no
signal that it is a variable at all. Any other value is extrapolation, which is
why `cross_check` says so in the form and why the wizard's phrasing is a measured
result rather than a principle. **The durable fix is corpus-side** — vary the
adsorption height in a future revision — not a better prompt. Until then:

> `--strict` is the guarantee, not the phrasing. It turns the post-build
> mismatch warning into exit code 7 so a batch fails instead of quietly writing
> wrong DFT inputs.

## Metal oxides, sulfides and other compounds

`build_bulk` and `build_surface` wrap `ase.build.bulk`, which takes a **single
element** — so the form's element slot takes one too, and a compound typed there
is redirected rather than rejected:

```
  element  [Cu, Fe, Pt] > MgO
   -> MgO is a compound -- this slot takes one element, because bulk and slab
      builders wrap ase.build.bulk. Build it in the 'prototype' region as
      'rocksalt-MgO'.
```

Compounds are built in the **prototype** region, from composition-parameterised
families in `structure.py` (`COMPOUND_FAMILIES` / `COMPOUND_LATTICE`):
`rocksalt`, `zincblende`, `wurtzite`, `fluorite` — 45 tabulated compositions.
Names are forgiving (`rocksalt MgO`, `MgO rocksalt`, `rocksalt-MgO`, or bare
`MgO` when only one family lists it), and an untabulated composition is refused
rather than given an invented lattice constant.

### What r5 does with prototypes it never trained on

The adapter trained on five prototype names. Asked for the new ones, it initially
failed: it emitted `prototype='rocksalt'` with the composition dropped and a
spurious `c=0.0`, then retried the identical call to the turn budget.

The fix was **the tool description**, not the model. `build_prototype`'s
description now states the `<family>-<formula>` convention with four examples,
and the schema accepts an optional `formula` argument for the case where the
model splits them anyway. After that change, all seven measured requests
succeeded on the first call:

| Request | Emitted | Result |
| --- | --- | --- |
| `Build the rocksalt-MgO prototype.` | `prototype='rocksalt-MgO'` | Mg4O4 |
| `Build the rocksalt-NiO prototype.` | `prototype='rocksalt-NiO'` | Ni4O4 |
| `Build the rocksalt-PbS prototype.` | `prototype='rocksalt-PbS'` | Pb4S4 |
| `Build the fluorite-UO2 prototype.` | `prototype='fluorite-UO2'` | O8U4 |
| `Build a bulk MgO rocksalt crystal.` | `prototype='rocksalt-MgO'` | Mg4O4 |
| `Build a wurtzite ZnO crystal.` | `prototype='wurtzite-ZnO'` | O2Zn2 |
| `Build a zincblende ZnS crystal.` | `prototype='zincblende-ZnS'` | S4Zn4 |

> **Comparability caveat.** Changing `build_prototype`'s description changes the
> prompt for prototype requests, so prototype-family numbers in existing eval
> reports are not strictly comparable across this change. Nothing else moved.

### Compound surfaces, vacancies and substitutions

These work too. `ase.build.surface` cuts a slab from *any* Atoms, so a compound
slab only needs the right substrate — and the substrate keeps travelling in the
existing `element` argument, which now carries either a symbol or a compound:

```
build_surface(name='slab', element='rocksalt-MgO', miller=[1,0,0], layers=4, vacuum=12.0)
```

`repeat`, `make_vacancy`, `substitute` and `freeze_layers` needed no change at
all — they act on whatever the active structure is. The router swaps `build_bulk`
for `build_prototype` when a request names a compound, so the model is never
offered a builder that cannot succeed.

> **One substrate argument, learned the hard way.** The first attempt made
> `element` optional, added a parallel `prototype` argument, and let `crystal`
> name a compound family. r5 generalised the wrong lesson — that `crystal` names
> the material — and started emitting `build_surface(crystal='fcc', ...)` with no
> element at all, breaking ordinary **elemental** slabs, and silently dropping the
> lateral `repeat` on others. A long tool description made it worse. The working
> shape is: one required substrate in `element`, a terse tool description, and the
> compound guidance on the *argument* rather than the tool. A redundant
> `crystal='rocksalt'` alongside `element='rocksalt-MgO'` is now simply ignored,
> because r5 emits it every time and rejecting it only bought a wasted turn.

In the form, the element question becomes **element or compound**, and answering
a compound *skips the crystal-phase question* — `rocksalt-MgO` already names the
family:

```
  element or compound  [Cu, Fe, or MgO / ZnO / rocksalt-NiO] > MgO
  facet (Miller)       [111, 100, (110)]                     > 100
  ...
composed request:
  Build a 2x2 rocksalt-MgO(100) slab with 4 layers and 12 Å vacuum.
```

The hyphenated name is deliberate: measured on r5, `rocksalt-MgO(110)` builds on
the **first** call, while a bare `MgO(100)` makes the model guess the family,
fail validation, and retry. Final measurements — every one first-call, no retries:

| Request | Result |
| --- | --- |
| `2x2 rocksalt-MgO(100) slab, 4 layers, 12 Å` | Mg64O64 |
| `2x2 rocksalt-NiO(110) slab, 5 layers, 12 Å` | Ni80O80 |
| `2x2 fluorite-CeO2(111) slab, 3 layers, 15 Å` | Ce48O96 |
| `2x2 wurtzite-ZnO(001) slab, 4 layers, 15 Å` | O32Zn32 |
| `2x2x1 rocksalt-MgO supercell, remove the first atom` | Mg15O16 |
| `2x2x1 rocksalt-MgO supercell, replace the first atom with Ca` | Ca1Mg15O16 |

Elemental slabs are unchanged and were re-measured alongside them: `2x2 Cu(100)`
4 layers + freeze bottom 2 → Cu16 with 8 constrained; `2x2 Cu(111)` → Cu16; the
2.5 Å O-on-Cu(100) adsorption → Cu20O1 with the height kept.
| `2x2 fluorite-CeO2(111) slab, 3 layers, 15 Å` | 2 (one retry) | Ce48O96 |

> **A wrong compound is now caught.** During this work a request for
> `rocksalt-NiO` silently built `rocksalt-MgO` and reported `FINISHED`: the
> post-build check only compared *numbers*. `request_check.check_composition` now
> verifies that a named compound's elements are actually present, and shares the
> `--strict` channel, so that case exits 7. It is deliberately narrow — only
> compound names are checked, not every symbol, because substitution and vacancy
> legitimately remove elements a request mentions.

### Still unsupported

- **Non-binary compounds** — perovskites (SrTiO3), corundum (Fe2O3), spinels.
  `build_crystal` (space group + basis) can express them but is never routed and
  has zero corpus records. Plan: **HANDBOOK §12**.
- **Untabulated compositions** are refused rather than guessed; pass an explicit
  `a` (and `c`) to force one.
- **Polar and reconstructed surfaces** are cut geometrically, not repaired. A
  cut like ZnO(0001) is a valid termination of the bulk, not a physically
  stabilised surface — inspect before running DFT.

## Keeping the form honest

`guided.py` derives its question list from `request_check.FAMILY_REQUIRED` and
verifies the mapping at import (`_assert_covers_rule`), so adding a required slot
to the corpus rule without updating the wizard raises at import rather than
silently emitting under-specified prompts. `tests/test_guided.py` pins the same
property per region, checks every composed request against `check_request`,
checks the numbers survive `stated_values` round-trip, and asserts the
non-adsorption templates are byte-identical to their corpus source.
