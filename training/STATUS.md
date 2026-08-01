# Training status — 2026-08-02 Asia/Taipei

## Phase 2 corpus: half the training text is now prose (2026-08-02)

Phase 2 of `JOURNAL_ROLE_R3_SCHEDULE.md` is complete. `build_journal_corpus.py`
renders a deterministic ~50% share of cases as prose requiring interpretation
instead of `Use "fcc" for the crystal structure.`-style literal frames, so the
corpus teaches the *mapping* prose → canonical schema value. The other half stays
literal: the model needs both skills.

- Corpus: 5,000 records, 1,187 prose / 1,188 literal cases (ratio 0.4998).
  Splits 4,040 / 450 / 510.
- **Targets are provably unchanged.** Regenerating with the pre-Phase-2 builder
  and diffing per id: 5,000/5,000 records identical `payload_hash`,
  `split_group`, and split assignment. The only delta is `sources[].text`
  (2,374 records). Prose rendering did not move a single graded target.
- Leakage is guarded by `training/generators/prose_leak_guard.py`, a shared leaf
  module both generators import so they cannot drift. Generation fails closed if
  a rendered sentence contains a JSON literal or bare enum/boolean spelling of
  the value it grounds.
- **`prose_holdout` stays held out.** The training pool is exported as
  `TRAINING_PROSE_TEMPLATES`; `build_prose_holdout.py` imports it and fails
  closed on any shared surface form. Measured at the rendered-text level: 88
  distinct holdout sentences vs 578 training sentences, **0 shared**.

Two guard defects found and fixed before commit:

1. `_bare_leak` excluded `.` from the word boundary on both sides so an integer
   token `1` would not match `1.2`. That also made every **sentence-final**
   literal invisible — `"The centering flag is true."` passed the guard. The
   `.` exclusion now applies to numeric tokens only. Audited both datasets under
   the corrected guard: **0 actual leaks**, so this was latent, not
   contamination.
2. `build_prose_holdout.CRYSTAL_NAMES["bcc"]` contained `"the bcc phase"`, which
   spells the target enum verbatim. No production case reached it (the rebuilt
   holdout is byte-identical, `ef810397…`), but any bcc defect case would have
   leaked. Replaced with `"the body-centered cubic form"`.

Gates: `evaluate_journal_corpus.py` `passed: true` (execution 1.0, forbidden 0.0,
zero split-group overlap), `run_vertical_slice_gate.py` `passed: true` exit 0,
full suite 875 passed / 2 skipped.

**Caveat carried into r3:** the r2 numbers in the next section were measured on a
*superseded* build of `prose_holdout`. Phase 2 moved wording between the pools to
enforce disjointness — Miller indices went `(111)` → `111-oriented` / `{111}`,
and `"face-centred cubic"` moved into the training pool. The datasets are
git-ignored, so nothing in git records the change. r2 must be re-measured on the
current file (`ef810397…`, ~1.2 h GPU) before any r3-vs-r2 comparison is valid.

## prose_holdout: the extractor cannot read prose (2026-07-31)

> **Superseded input set.** The numbers below were measured 2026-07-31 13:18
> against the *pre-Phase-2* build of `journal_holdout_prose/test.jsonl`. The file
> at that path is now `ef810397…` and its wording differs (see the Phase 2 entry
> above). The failure *modes* below still stand — they are what Phase 2 was built
> to fix — but the per-field rates do not describe the current file.

Phase 1 of `JOURNAL_ROLE_R3_SCHEDULE.md` is measured. The journal-role **r2**
adapter (`training/runs/pilot-qwen3-4b-journal-role-r2/adapter`) was evaluated on
`training/datasets/journal_holdout_prose/test.jsonl` — 52 cases / 104 records
whose source text contains **no JSON literal of any target value**.

Report: `training/evaluations/journal-role-r2-prose-holdout.json`, `complete:
true`, 80 records sampled (40 Evidence Extractor / 40 Spec Planner —
`--sample-size 80` takes `count//2` per role), 1.23 h wall clock,
`evaluation_ids_sha256` `7f1e43e698b56b6b…`. Sample: adsorbate 26, surface 22,
defect 16, bulk 16.

| metric | value |
| --- | --- |
| schema-valid rate | 0.800 (64/80) |
| exact-payload rate | **0.000 (0/80)** |
| executable-or-safe rate | 0.500 (40/80) |
| provenance recall | 0.779 (415/533) |
| forbidden-action rate | 0.000 |
| negative-safety rate | `null` (0 negative records) |

The summary's `field_recall` (0.787) understates the extractor damage and must
not be quoted for that role: `_flatten` treats the whole `claims` list as one
field, so every Evidence Extractor record scores a fixed 2/3 (`contradictions`
and `unresolved_fields` are empty and always match). The per-field numbers below
come from parsing `generated_text` and matching claims by field path.

### Per-field breakdown — Evidence Extractor (40 records, 422 target claims)

"present" = a claim with that exact field path appears at all; "exact" = it
appears *and* its value equals the target.

| field | n | present | present rate | exact | exact rate | what it emitted instead |
| --- | --- | --- | --- | --- | --- | --- |
| `material.formula` | 40 | 39 | 0.975 | 39 | **0.975** | 1× `material.element: ["Al"]` |
| `material.crystal_structure` | 40 | 40 | 1.000 | 24 | 0.600 | `"cubic close-packed"`×13, `"face-centred cubic"`×2, `"cubic close packed"`×1 |
| `model.layers` | 33 | 33 | 1.000 | 33 | **1.000** | — |
| `model.miller_indices` | 33 | 0 | 0.000 | 0 | **0.000** | `model.facet:"111"`, `model.surface_orientation:"(100)"`, `model.exposed_plane:"(111)"`, `model.surface_plane`, `model.surface` |
| `model.supercell` | 40 | 33 | 0.825 | 27 | 0.675 | `"2×2"`×3, `"2x2"`×2, `[2,2]`×1 (plus 7× as `model.surface_cell:"2x2"`) |
| `model.vacuum_angstrom` | 33 | 23 | 0.697 | 23 | 0.697 | 10× renamed `model.vacuum_angstroms` — the *number* was right in 33/33 |
| `model.center` | 40 | 40 | 1.000 | 26 | 0.650 | `"cell"`×4, `"surface"`×4, `"simulation_cell"`×3, `"center"`×2, `"box"`×1 |
| `model.atom_ordering` | 40 | 40 | 1.000 | 5 | **0.125** | `"ase"`×21, `"ase-default"`×11, `"ASE-default"`×3 |
| `model.periodic_boundary_conditions` | 40 | 40 | 1.000 | 0 | **0.000** | `["x","y"]`×17, `["surface"]`×16, `["x","y","z"]`×6, `["all"]`×1 |
| `requested_outputs` | 40 | 40 | 1.000 | 16 | 0.400 | `["cif","poscar"]`×11, `["cif","vasp_poscar"]`×10, `["CIF","POSCAR"]`×3 |
| `modifications[0].site` | 13 | 0 | 0.000 | 0 | **0.000** | `adatom.site:"atop"`, `adsorbate.binding_site`, `modifications.adsorbate.site:"on-top"` |
| `modifications[0].height_angstrom` | 13 | 0 | 0.000 | 0 | **0.000** | `adatom.distance_angstrom` etc.; value right under *some* name in 9/13 |
| `modifications[0].element` | 10 | 0 | 0.000 | 0 | **0.000** | `adsorbate.formula`, `adatom.atom`; value right under some name in 3/10 |
| `modifications[0].species` | 7 | 0 | 0.000 | 0 | **0.000** | value right under some name in 5/7 |

Aggregate claim level: **193/422 = 0.457** exact. **0 of 40** records had every
claim right. 30 distinct field paths were invented that the schema does not
have; `model.kind` was volunteered on all 40 (the extractor is not asked for it).

`schema_valid` is 1.000 for this role and means nothing here — the
`evidence_ledger` schema constrains neither claim field names nor value types, so
a ledger full of `["x","y"]` and `"ase"` validates cleanly and the damage only
surfaces at `catalyst_spec` validation one stage later. That is exactly the live
smoke failure, reproduced offline.

### Verdict: broad interpretation failure, not a narrow enum problem

Only **2 of 14** target fields survive prose: `model.layers` (33/33 — "a
four-layer slab" → `4` works) and `material.formula` (39/40 — "aluminum" → `Al`
works). Everything else fails, and the two fields the live smoke exposed
(`center` 0.650, `atom_ordering` 0.125) are *not* the worst cells in the table:
`periodic_boundary_conditions` is 0/40, `miller_indices` is 0/33, and all four
`modifications[...]` fields are 0.

Three distinct failure modes, all present:

1. **Value not canonicalised** — `"cubic close-packed"` for `fcc`, `"ase"` for
   `ase_default`, `"poscar"` for `vasp`, `"cell"` for `true`. Enum-shaped.
2. **Type not converted** — `"(111)"` instead of `[1,1,1]`, `"2×2"` instead of
   `[2,2,1]`, `["x","y"]` instead of `[true,true,false]`. The model reproduces
   the prose token rather than the schema's type. Only 1/33 records produced
   `[1,1,1]`-shaped Miller indices under any field name.
3. **Field path invented** — `model.facet`, `model.vacuum_angstroms`,
   `adatom.site`, `adsorbate.formula`. Here the *reading* is usually right
   (vacuum: 33/33 correct numbers, height: 9/13) and only the destination path
   is wrong. This is the cheapest subset to fix and the largest single block of
   zeros in the table.

So Phase 2 cannot be scoped to `center` and `atom_ordering`.

### Spec Planner: unaffected where it can copy, 0.000 where it must derive

The planner never sees prose — it is prompted with the EvidenceLedger JSON — and
on this set it copies **perfectly**: `material.formula`, `crystal_structure`,
`layers`, `miller_indices`, `supercell`, `vacuum_angstrom`, `center`,
`atom_ordering`, `periodic_boundary_conditions`, `requested_outputs`,
`schema_version`, `modifications[0].{element,height_angstrom,site,species}` —
every one 40/40 or n/n exact.

Every field the ledger does *not* state is 0/n:

| derived field | n | exact |
| --- | --- | --- |
| `model.kind` | 40 | 0.000 |
| `modifications[0].operation` | 20 | 0.000 |
| `model.fixed_layers_from_bottom` | 31 | 0.000 |
| `modifications[0].site_index` | 13 | 0.000 |
| `modifications[0].anchor` | 7 | 0.000 |
| `modifications[0].selector.{layer.count,layer.side,ordinal}` | 7 each | 0.000 |

Consequences: schema-valid 0.600 (all 16 failures are
`catalyst_spec.modifications[0] missing required fields ['operation']`, i.e.
every adsorbate/defect case), and **executable-or-safe 0.000** — `policy_gate`
refuses all 40 with `unsupported model kind None`, so no proposal ever reaches
`dispatch_spec`. Exact-payload is 0/40 for this role too.

**Caveat — the planner is not a clean control on this set.** In
`journal_roles_v1` the ledger carries `model.kind` in 1918/2020 planner records
and `modifications[0].operation` in 809; in `prose_holdout` those claims are
absent by construction (prose cannot state them as literals) and the reference
targets mark them `evidence_type: "derived"`. So the planner half measures a
*different* held-out axis — derivation from an incomplete ledger — and it shows
the same pathology as the extractor: r2 copies what is written down and produces
nothing that is not. The prose axis itself leaves the planner untouched.

### Context against r2's and r1's other measured sets

| set | adapter | schema | exact | notes |
| --- | --- | --- | --- | --- |
| `journal_roles_v1` main test | **r2** | 1.000 | 0.967 | in-distribution |
| `template_holdout` | r1 | 1.000 | 0.980 | held-out sentence frames, literal kept |
| `family_holdout` | r1 | 1.000 | 0.960 | held-out family |
| `prose_holdout` | **r2** | 0.800 | **0.000** | held-out value *representation* |

The holdout rows are **r1** numbers; r2 has no `template_holdout` /
`family_holdout` results (its cycle aborted at promotion) and the schedule
recommends not producing them. The only same-adapter comparison is r2's own
0.967 in-distribution exact rate against 0.000 here. The corpus's own held-out
sets vary phrasing while keeping the JSON literal in the sentence, which is why
they read 0.96–0.98: they never tested this axis.

### What Phase 2's corpus renderer must cover

Every field, not a shortlist. Concretely, the renderer needs prose forms whose
target is reached by interpretation for:

- **Enum canonicalisation** — `crystal_structure` (`face-centred cubic`, `cubic
  close-packed`, `ccp` → `fcc`), `atom_ordering` (`ASE-default`, `ASE default
  order`, `as written by ASE` → `ase_default`), `requested_outputs` (`CIF and
  POSCAR`, `CIF plus VASP inputs` → `["cif","vasp"]`), `site` (`on-top`, `atop`
  → `ontop`).
- **Prose → typed value** — `center` (`centered/centred in the cell` → `true`),
  `periodic_boundary_conditions` (`periodic in x and y, vacuum along z` →
  `[true,true,false]`; `periodic in all directions` → `[true,true,true]`),
  `miller_indices` (`(111)`, `the (100) facet` → `[1,1,1]`/`[1,0,0]`),
  `supercell` (`2×2`, `2 x 2`, `a 2 by 2 surface cell` → `[2,2,1]`).
- **Field-path discipline** — the schema path must be produced even when the
  paper's wording suggests another name: `vacuum_angstrom` (not
  `vacuum_angstroms`), `miller_indices` (not `facet`/`exposed_plane`),
  `modifications[0].{element,species,site,height_angstrom}` (not
  `adatom.*`/`adsorbate.*`).
- **Already fine, keep as-is** — `layers` and `formula` need no new coverage;
  they are the control showing 4B can do this when the mapping is learned.

Separately, and independently of the prose axis, the planner needs training
where the ledger *omits* the derived fields, so `model.kind`,
`modifications[0].operation`, `model.fixed_layers_from_bottom`,
`modifications[0].site_index`, `anchor` and `selector.*` are inferred rather
than copied. Today that gap alone would fail the live smoke on every adsorbate
and defect request even with a perfect extractor.

Regression target for r3: `prose_holdout` exact-payload rate ≫ 0.000 with
schema-valid 1.000 and executable-or-safe 1.000, measured with the same
`--sample-size 80`.

## r1 held-out results: strong offline generalization on both axes (2026-07-30)

Both section 10.3 held-out evaluations of the journal-role r1 adapter
(`training/runs/pilot-qwen3-4b-journal-role-r1/adapter`, Qwen3-4B QLoRA trained
on `journal_roles_v1`) finished: `complete: true`, 100 records sampled per set,
50 Evidence Extractor / 50 Spec Planner in each.

| metric | `template_holdout` | `family_holdout` |
| --- | --- | --- |
| schema-valid rate | 1.000 | 1.000 |
| exact-payload rate | **0.980** (98/100) | **0.960** (96/100) |
| executable-or-safe rate | 1.000 | 1.000 |
| provenance recall | 1.000 (582/582) | 1.000 (637/637) |
| forbidden-action rate | 0.000 | 0.000 |
| field recall / precision | 0.9979 (968/970) | 0.9960 (990/994) |

Reports: `training/evaluations/journal-role-r1-{template,family}-holdout.json`
(1.62 h and 1.84 h wall clock; `evaluation_ids_sha256` `4dfa5bc53566…` and
`9250b4993ba5…`). Sample composition:

| set | families | roles |
| --- | --- | --- |
| `template_holdout` | surface 43, bulk 39, adsorbate 18 | 50 / 50 |
| `family_holdout` | surface 38, bulk 28, adsorbate 20, supported_cluster 14 | 50 / 50 |

### Every non-exact record is one dropped claim, by the Evidence Extractor

All six misses across both sets are the same failure shape: an EvidenceLedger
that omits exactly one claim. None had a wrong value, none invented a claim,
none was malformed — all six stayed schema-valid and executable-or-safe, and the
Spec Planner was exact on all 100 of its records in both sets.

| set | record id | family | claim dropped | claims gen/exp |
| --- | --- | --- | --- | --- |
| template | `tmpl-surface-016-evidence_extractor` | surface | `model.miller_indices = [1,0,0]` | 11/12 |
| template | `tmpl-surface-036-evidence_extractor` | surface | `model.miller_indices = [1,1,0]` | 11/12 |
| family | `fam-supported-005-evidence_extractor` | supported_cluster | `modifications[0].operation = add_supported_cluster` | 17/18 |
| family | `fam-supported-009-evidence_extractor` | supported_cluster | `modifications[0].operation = add_supported_cluster` | 17/18 |
| family | `fam-supported-013-evidence_extractor` | supported_cluster | `modifications[0].operation = add_supported_cluster` | 17/18 |
| family | `fam-supported-014-evidence_extractor` | supported_cluster | `model.termination = ase_default` | 17/18 |

Two patterns worth carrying into r2. In `template_holdout`, both dropped claims
were rendered by the held-out **Chinese** phrase template
(`該研究以 {value} 作為{label}。`) — weak evidence, since all 50 Evidence
Extractor records contain at least one Chinese line and 48 of them were exact,
but the failures land there and nowhere else. In `family_holdout`, all four
misses are `supported_cluster` — the longest ledgers in the set at 18 claims —
so that cell is 5/9 exact while every other family is perfect; three of the four
drop the same field, `modifications[0].operation`.

### The 0.0 negative-safety rate is a divide-guard artifact, not a failure

Both summaries print `negative_safety_rate: 0.000`. Neither set contains a
single record of family `negative` (template: surface/bulk/adsorbate only;
family: those three plus supported_cluster — zero negatives in either), so the
old expression `sum(...) / max(1, 0)` evaluated to `0/1 = 0.0`. It measures
nothing here. `evaluate_journal_model.py` now emits `null` plus an explicit
`negative_record_count` when the denominator is empty, so a set without
negatives can no longer look like a safety failure; these two reports predate
that change and still carry the vacuous `0.0`.

### Standing caveats

- **`template_holdout` chiefly stresses the Evidence Extractor.** The Spec
  Planner is prompted with the EvidenceLedger JSON, not the prose, so wording
  novelty reaches it only through the one-sentence request. Its 50/50 exact
  score on this set is a much weaker generalization claim than the Extractor's.
- **The CO records conflate two variables.** r1 was trained before the CO anchor
  fix, so CO targets in `template_holdout` were never seen in training; those
  records mix wording novelty with the geometry correction. (Only one CO record
  fell into this 100-record sample, `tmpl-adsorbate-007-spec_planner`, and it
  was exact — so the conflation did not move these numbers, but it still makes
  CO an unusable signal for phrasing generalization.)
- **Offline strength does not clear the live-smoke failure.** r1 remains
  **unpromoted**: it failed its live CLI smoke, and none of these metrics
  address that. The corpus with the fixes is still untrained. That is exactly
  why the next step is the r2 cycle, not a promotion decision on r1.

## Held-out sets built; the GPU was here all along (2026-07-29)

**Correction: this machine has a GPU.** `nvidia-smi` is not on the default PATH
— it lives at `/usr/lib/wsl/lib/nvidia-smi`, which is exactly why every
`training/run_*.sh` script exports `PATH="/usr/lib/wsl/lib:$PATH"`. The device is
an NVIDIA RTX 2000 Ada Generation, 16 GB, and `torch.cuda.is_available()` is
`True` on torch 2.6.0+cu124. Earlier notes in this file that called the work
"GPU-bound" because no GPU was present were wrong; the work was PATH-bound.

Two of section 10.3's three held-out sets are generated by
`training/generators/build_journal_holdouts.py`, 200 records each,
evaluation-only (one `test.jsonl`, no train split):

| set | varies | held out |
| --- | --- | --- |
| `template_holdout` | wording only | five phrase templates and both request sentences, disjoint from the training pool; chemistry stays familiar |
| `family_holdout` | chemistry only | elements Rh, Ir, Pb, Mo, Ta; adsorbates N, C, NO, O2, OH; CeO2 and CaO supports; wording stays in-distribution |
| `journal_holdout` | — | **not built**; needs complete unseen papers and no licensed journal text exists here |

The "held out" claim is structural, not a copied constant:
`build_journal_corpus.py` now exposes `PHRASE_TEMPLATES`, `REQUEST_TEMPLATES` and
`PHASES`, and the holdout generator imports them, so changing the training pool
cannot silently turn a held-out set into an in-distribution one. Generation also
fails closed if a rendered source text appears in the training corpus or if
held-out chemistry turns out to be present. The refactor is output-preserving:
`journal_roles_v1`'s split hashes are unchanged.

Both sets pass the replay audit — 200 records, 100 ready builds, execution
success 1.0, zero forbidden actions, no errors.
`evaluate_journal_corpus.py` now accepts a dataset that declares only a test
split, instead of requiring all three.

**Caveat on interpreting `template_holdout`:** the Spec Planner is prompted with
the EvidenceLedger JSON, not the prose, so wording novelty reaches the planner
only through the one-sentence request. This set chiefly stresses the Evidence
Extractor role.

**Caveat on the adsorbate records:** r1 was trained before the CO anchor fix, so
the eight CO records in `template_holdout` carry a target it never saw. Those
records conflate wording novelty with the geometry correction.

Run with `bash training/run_journal_holdout_evaluation.sh` (defaults to a
100-record sample per set, roughly 150 s per record on this GPU). Reports land at
`training/evaluations/journal-role-r1-{template,family}-holdout.json` and carry
`complete: false` until the run finishes, so a partial report cannot satisfy the
promotion gate.


## Section 14 vertical slice completed; CO geometry fixed at runtime (2026-07-29)

All six suggested golden examples in `CATALYST_STRUCTURE_LLM.md` section 14 now
build, validate and export. They are carried in `tests/golden/` as a separate
`section: "14"` group, so section 7.2's 100 fixtures and their family counts are
unchanged. The gate reports 106 cases, 95 buildable, all six criteria at
threshold.

Three of the six needed work:

- **CO/Pt(111) did not build at all.** `ase.build.molecule("CO")` is stored as
  (O, C) with the carbon *below* the oxygen, so anchoring atom 1 at 1.85-1.9 A
  left the carbon ~0.70-0.75 A from the top metal layer. The r7 branch fixed this
  in the *corpus* by anchoring atom 2; the runtime still placed whatever it was
  given. `tools_surface._anchor_down` now mirrors the molecule about the anchor
  plane whenever another atom would sit below it, so the declared binding anchor
  is always the atom nearest the surface. Bond lengths are unchanged; this is a
  placement rule, not the user-facing orientation control the policy gate still
  refuses. A new `adsorbate_N_anchor_is_lowest` validation rule keeps it honest.
- **TiO2(110) and graphene were unreachable.** Fixed prototypes such as
  `rutile-TiO2`, `graphene`, `graphite` and `hBN` are neither `<family>-<formula>`
  compounds nor `ase.build.bulk` phases, so `_surface_substrate` fell through to a
  reference-data lookup and failed. It now consults the prototype table, the
  dispatcher stops passing a redundant `crystal` argument that tripped the phase
  enum, and `catalyst_spec.schema.json` accepts `rutile`, `anatase`, `graphene`,
  `graphite` and `hBN` as `material.crystal_structure`.
- `slab_layer_count` is host-aware: a compound layer resolves into sublayers
  (rutile TiO2(110) gives three z levels per requested layer), so compounds are
  checked for divisibility and elemental hosts for equality.

**`journal_roles_v1` carried the same CO defect.** Its 100 CO adsorption cases
used anchor 1 at 1.9 A, i.e. a 0.7497 A C-Pt contact recorded as a successful
build -- the same defect the r7 commit found in `pilot_r6`. The generator now
anchors atom 2. **The corpus must be regenerated before the journal track is
retrained**; the copy in the primary checkout is still the one that trained the
un-promoted r1 adapter and has been left alone as the record of that run. A
regenerated corpus was audited here: 5,000 records, 2,375 ready builds, replay
success 1.0, zero forbidden actions, no split-group overlap.

Also closed from the product document: the section 4 `paper` block
(`title`/`doi`/`year`) is accepted by `ASE_catalyst_build` and threaded into
`supplied_evidence`, every source-backed fact and the reproduction record, so
section 11's "value + DOI/page/table/figure" is satisfied; the review packet's
structure summary now describes the surface, adsorbates, defects and supported
clusters rather than only the invariants; and `run_candidate_set` implements
section 7's "generate separately named candidates rather than silently selecting
one" for an ambiguous bulk parent, labelling each selection `derived`.

Section 10.3's `template_holdout` and `family_holdout` are now built (see the
entry above). `journal_holdout` remains impossible here: it needs licensed
journal text this repository does not contain.

## Vertical-slice gate built; journal-role r1 promotion reverted (2026-07-29)

**The deterministic slice now has the section 9 gate it was supposed to pass
before an LLM was ever connected.** `tests/golden/` holds 100 immutable
fixtures (30 MP-bulk / 30 slab / 15 defect / 15 adsorbate / 10 refusal, hashes
pinned in `tests/golden/manifest.json`), and
`training/evaluations/run_vertical_slice_gate.py` measures all six criteria:

```text
schema_validation_rate            1.0   (100/100)
build_export_round_trip_rate      1.0   (89/89 buildable)
writes_outside_request_dir        0
arbitrary_code_paths              0
fields_without_provenance         0
ambiguity_returned_clarification  1.0   (11/11)
```

`tests/test_golden_corpus.py` keeps it a standing barrier. Domain review of the
fixtures (design section 7.1) has **not** happened; every case records
`domain_review: pending`, and `physical_reference_golden` is empty on purpose
because no licensed literature data or declared converged calculation exists
here.

**`journal_role_ready` is back to `false` for
`training/runs/pilot-qwen3-4b-journal-role-r1`.** The adapter scored 1.000
schema-valid / 0.970 exact-payload / 1.000 provenance on the 300-record
teacher-forced harness and was flagged ready at 08:32 on 2026-07-29 — then its
own live `ASE_catalyst_build` smoke failed at 08:34 on a fully specified fcc Pt
bulk:

```text
stage: spec_planning
SchemaValidationError: catalyst_spec missing required fields ['modifications']
```

Nothing in the promotion path read that result, and `forbidden_action_rate` was
a hardcoded `0.0`, so `zero_forbidden_actions` could not fail. Both are closed:
`evaluate_journal_model.py` now measures forbidden actions (a build the gate
should have refused, or a tool name outside the registry) and reports
`forbidden_action_measured`; `promote_journal_adapter.py` gained
`live_smoke_passed` and `forbidden_actions_measured` checks and takes
`--smoke-dir`; `run_journal_role_evaluation.sh` runs the smoke **before**
promotion via the new `--allow-unpromoted` bypass, which every package it
produces records in its reproduction block.

**Why the live run failed where the harness did not:** `journal_roles_v1` is
5,000 fully synthetic records whose wording comes from four English templates
plus one Traditional Chinese template selected by `ordinal % 5`, shared across
train/validation/test by construction. Of the five test sets design section 8
requires, only `negative` exists, and it is a family inside the same splits
rather than a held-out set:

| set | status |
| --- | --- |
| `iid_construction` | the single random 10% hash bucket, unnamed |
| `linguistic_ood` | **missing** — templates are shared across splits |
| `compositional_ood` | **missing** — no element x facet x adsorbate is withheld |
| `journal_holdout` | **missing** — the corpus contains no journal text at all |
| `negative` | present as a family, not held out (30 of 484 test records) |

Building those three sets and re-running the promotion suite on a GPU is the
next required step for the journal track. Until then `ASE_catalyst_build`
correctly refuses every adapter and is not production-ready.

Deterministic defects fixed alongside the gate: the exported `structure.json`
claimed `controller_state: VALIDATED` before validation ran; the overlap check
compared only the globally closest pair against its element-pair threshold, so a
short contact was invisible whenever a larger pair sat nearer; the export
round-trip compared only atom count and formula; and there were no slab
thickness, atom count, host stoichiometry, or requested-format rules. The
review packet's reproduction block now also carries the model, adapter sha256,
ASE version and decoding parameters.


## Catalyst supplement r3/r4 rejected; r5 remains production (2026-07-28)

r3 trained for 200 steps on 2,149 replay-verified records and restored
checkpoint 200. Its validation loss was `0.00190678`. All 252 LoRA-B tensors
contained nonzero values. Adapter SHA-256:
`ee8f8691da51fd477977cdadacaab97a66474b860e7d59f652a1ceb2ea218f42`.

The strict full-tool comparison used the same ordered 37 records and adversarial
set for r5 and r3. The promotion report rejected r3.

| Metric | r5 | r3 |
| --- | ---: | ---: |
| Exact structure | 81.1% | 81.1% |
| Invariants satisfied | 89.2% | 81.1% |
| Schema execution | 93.5% | 83.6% |
| Finish | 94.6% | 89.2% |
| Adversarial safety | 100% | 100% |

r3 fixed the held-out nanoparticle case but lost both nanotubes and both
molecular-adsorption cases. It also missed both supported clusters and one
adsorption-site structure. The run started from the frozen base, so it spent
adapter capacity relearning the r5 replay pool and regressed families that r5
handled.

r4 changed one training variable: it loaded the promoted r5 adapter as trainable
weights. It reused the frozen r3 corpus, lowered the learning rate to `1e-4`, and
ran 100 steps. Checkpoint 100 won with validation loss `0.00247306`. The exported
adapter matches checkpoint 100 and contains 252 LoRA-B tensors with 17,694,720
nonzero values out of 17,694,720. Adapter SHA-256:
`c0b684ef04af8d6a7c0ec806b3ea07d03f272935165fc51f2b4e0b4332360671`.

The 37-record screen rejected r4 before the 120-record gate. r4 reached 100%
schema execution, finish, and adversarial safety, but exact structure and
invariants were 78.4%, below r5's 81.1% and 89.2%. Eight outputs had the wrong
structure. r4 repaired the r3 nanotube and molecular-adsorption failures, then
omitted lateral repeats on elemental, oxide, and high-index slabs and missed both
supported clusters. r5 remains the production adapter.

## Journal-agent live smoke — prompt-only baseline rejected (2026-07-28)

Three immutable live smokes exercised the evidence and SpecProposal tools with
r5. The first omitted a required empty `contradictions` array. A stricter prompt
fixed that omission. The next run produced informal evidence fields and no
complete proposal tool call under 1,024 tokens. A 2,048-token run produced exact
field paths and a complete proposal, but encoded `model.center` as the string
`"geometry"`; schema validation rejected it before ASE execution.

The deterministic journal pipeline, schemas, policy gate, resolver, dispatcher,
validation, export, and failure records pass scripted tests. The r5 model is not
qualified for the Evidence Extractor/Spec Planner roles. A dedicated journal-role
corpus, training run, and held-out promotion remain required before
`ASE_catalyst_build` can be called production-ready. Smoke packages:
`training/evaluations/journal-agent-smoke/journal-smoke-{001,002,003}/`.

Reports:
`training/evaluations/catalyst-supplement-r3-full-pf2-{base,r5,adapter,promotion}.json`
and `training/evaluations/catalyst-supplement-r4-full-pf2-{adapter,promotion}.json`.
r4 reuses the matched r3 base/r5 reports.

## Catalyst supplement r2 evaluated — rejected, r5 remains production (2026-07-28)

The r2 full-auto run completed 200 QLoRA steps on 2,099 training records and
restored checkpoint 175, which had the best validation loss (`0.0017678794`).
The adapter integrity check found 252 LoRA-B tensors and 17,694,720 nonzero
values out of 17,694,720. Adapter SHA-256:
`b8079cbd9a037be43d4f7d6d669aaff6954b91df2b44fb1ae962ab77b888974a`.

The paired full-tool evaluation used the same ordered 32-record catalyst sample
for the frozen base, r5, and r2. The strict promotion report returned
`promoted: false`.

| Metric | Frozen base | r5 | r2 |
| --- | ---: | ---: | ---: |
| Exact structure | 21.9% | 75.0% | 71.9% |
| Invariants satisfied | 21.9% | 87.5% | 71.9% |
| Schema execution | 70.7% | 93.5% | 100% |
| Finish | 53.1% | 93.8% | 100% |
| Forbidden actions | 0 | 0 | 0 |

r2 removed all failed and unknown tool calls, but nine outputs had the wrong
structure. Five used primitive fcc/bcc cells because the model omitted
`cubic: true`; two selected an entire top layer instead of its first atom; one
added unrequested nanoparticle lattice/vacuum values; and one repeated a
clarified conventional Si cell. The 32-case catalyst slice also leaves several
new families with one held-out structure, so it cannot support a family-level
promotion claim.

r5 stays production. The next corpus revision must add reviewed structural
cases for cell convention, single-site layer selectors, supported-cluster
defaults, and clarification follow-ups. Keep 200 steps and the r5 replay pool;
do not increase paraphrase count or relax promotion thresholds. Reports:
`training/evaluations/catalyst-supplement-r2-full-pf2-{base,r5,adapter,promotion}.json`.

## r6 evaluated — regressed, NOT promoted (2026-07-26)

r6 tested the one open question against r5: does the promoted adapter generalize to
**genuinely novel phrasing**? A held-out phrasing pool
(`training/generators/paraphrase_templates_r6_heldout/`, 8 new templates/family,
provably disjoint from training) generated `training/datasets/pilot_r6_novel`, from
which `heldout_eval.jsonl` (3,218 records) keeps only prompts absent from *all*
training — novel phrasing over structures the model did train on. Base, the promoted
r5 adapter, and the new r6 adapter were scored on an identical 120-record sample
(`evaluation_ids_sha256` matched across all three).

| exact_structure_rate | frozen base | r5 adapter | r6 adapter |
| --- | ---: | ---: | ---: |
| SEEN phrasing (`pilot_r6/test`) | 13.3% | — | 50.0% |
| **NOVEL phrasing (held-out)** | 12.5% | **99.2%** | 55.8% |

- **Limitation #1 is resolved, and was never a real deficiency: r5 scores 99.2%
  exact on phrasing it never saw** (finish 100%, invariants 99.2%, schema 97.1%).
  Shared train/test phrasing was not inflating r5. **r5 stays the production
  adapter.**
- **r6 regressed and is NOT promotable.** Despite a lower best `eval_loss`
  (0.00261 vs r5 0.00549), `promote_adapter.py` returned `promoted: false` (6 checks
  failed). Root cause (from `generated_calls`): r6 hallucinates a spurious `"a"`
  lattice arg on `build_surface` and a wrong `"bond"` on `build_nanotube`, the call
  fails validation, and r6 retries to the 6-turn budget → `BUDGET_EXHAUSTED`. The
  break is confined to surface / nanotube / molecular_adsorption; molecule /
  prototype / substitution / vacancy stay 83–100%. Adversarial safety 5/5, zero
  forbidden executions.
- Recipe caution for the next run: r6 used `--max-prompts-per-case 10` and 300
  steps; the regression is in families the geometry work does not touch, so the
  cap/step recipe is the suspect lever. Reports: `training/evaluations/*-r6-*-pf6-s120.json`;
  tables via `training/evaluations/tabulate_r6.py`.

## r5 promoted — `production_ready` flipped (2026-07-25)

The r5 adapter passed the fail-closed full-registry promotion gate and was
promoted. `production_ready` is now `true` in the run manifest
(`training/runs/pilot-qwen3-4b-r5/manifest.json`) — the reviewed integration
step that `README.md` requires ("retain `production_ready: false` … until a
separate reviewed integration task consumes a passing promotion report").

- Corpus: `training/datasets/pilot_r5`, 2,607 records; replay 100% execution and
  zero forbidden actions (`corpus_audit.passed: true`).
- Training: 200-step QLoRA on `Qwen/Qwen3-4B-Instruct-2507` @ rev `cdbee75f…`,
  best validation loss `0.005493` (checkpoint-200), all 252 LoRA-B tensors
  nonzero, adapter SHA-256 `4959a891…`.
- Structural fix (no retrain): `ASEWorkspace.execute` now defaults an omitted
  `name` to the active structure, closing the `build_surface` "missing required
  fields ['name']" gap on surface/adsorption prompts. The fix is
  fingerprint-neutral, so the r5 corpus stays valid.
- Promotion gate — paired base vs. adapter, full tool mode, 120 shared records
  (`training/evaluations/promotion-r5fix120-full-pf2-s120.json`,
  `promoted: true`, all 23 checks green):

  | Metric | Frozen base | r5 adapter |
  | --- | ---: | ---: |
  | Exact structure | 10.8% | 95.0% |
  | Invariants satisfied | 10.8% | 95.0% |
  | Schema execution | 58.2% | 96.4% |
  | Finish | 35.0% | 99.2% |
  | Adversarial safety | — | 100% |

  `atomic_adsorption` reached 100% exact (n=26). The manifest carries a
  `promotion` provenance block naming the report and its 23 passing checks.
- Honest caveat: train and test share the same 10 templates per family (grouped
  by output hash — no structure/recipe leakage, but shared phrasing form).
  Novel-phrasing generalization is still untested and remains the next lever.

## Current stage

- The bounded request/tool router is implemented. Unsupported random requests
  fail closed; supported regions include bulk, surfaces, molecules, nanotubes,
  prototypes, adsorption, vacancies, substitutions, and constraints.
- The r3 corpus was regenerated with the runtime system prompt as its single
  source of truth: 1,314 records (1,038 train, 129 validation, 147 test).
- Corpus replay passed with 100% execution and zero forbidden actions.
- Maximum rendered length is 1,547 tokens; training used a 1,600-token limit.
- The 100-step r3 QLoRA run completed in 2,696.8 seconds. Final validation loss
  was 0.017504. All 252 LoRA-B tensors are nonzero.
- Adapter SHA-256:
  `c5f937baec6059a4e65220a2adf4b6765b2815ba10c08a420a08c3a3241e2cd5`.
- Adapter path: `training/runs/pilot-qwen3-4b-r3/adapter`.

## r3 evaluation result

The one-example-per-family routed r3 adapter gate completed all 12 families.
Schema execution and finish rates were 100%, with zero forbidden, failed, or
unknown calls. Exact structure and exact tool sequence rates were each 11/12.
The two discrepancies were:

- `al-fcc-1x1x1-p1`: exact final structure, but an unnecessary identity
  `repeat([1,1,1])` caused the tool-sequence miss.
- `fe-110-5-2x2-co-p1`: the model used three slab layers instead of the
  expected five, causing the molecular-adsorption structure miss.

The molecular-adsorption family therefore fails the 25% per-family promotion
threshold. A larger r3 evaluation cannot promote this adapter.

## r4 corrective run (historical — superseded by r5)

The r4 corpus changes only 108 molecular-adsorption prompts so every reference
parameter is explicit. Corpus replay passes for all 1,314 records with zero
forbidden actions. Rendered length remains safe at a maximum of 1,547 tokens
for the 1,600-token training limit.

The 100-step r4 QLoRA run was launched detached at
`2026-07-25T00:02:42+08:00` using:

```text
training/run_pilot_r4_remote.sh
```

Run path: `training/runs/pilot-qwen3-4b-r4/`

Log: `training/runs/pilot-qwen3-4b-r4.log`

## Promotion status — complete for r5

The r4 corrective path above was superseded by the r5 corpus rebuild. The r5
adapter completed the full promotion pipeline and passed every gate:

1. Corpus replay + audit passed (2,607 records, zero forbidden actions).
2. One-per-family and stratified full-mode adapter gates passed.
3. Paired base/adapter full-mode evaluation on 120 shared records.
4. `promote_adapter.py` → `promoted: true`, all 23 checks green.
5. `production_ready` flipped to `true` in the r5 manifest (recorded above).

Open follow-ups (not promotion blockers): novel-phrasing generalization is now
**tested and passing for r5 (99.2%, see the r6 section above)**; the r6 "more
steps + richer phrasing" retrain regressed and was rejected. Remaining: the
adsorbate-overlap geometry fix (tracked separately on the r7 branch).
