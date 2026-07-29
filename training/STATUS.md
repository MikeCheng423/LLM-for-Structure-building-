# Training status — 2026-07-29 Asia/Taipei

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
