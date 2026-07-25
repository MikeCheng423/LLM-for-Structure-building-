# Training status — 2026-07-25 19:13 Asia/Taipei

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

Open follow-ups (not promotion blockers): test novel-phrasing generalization
(train/test still share phrasing templates); retrain r6 with more steps (r5
validation loss was still dropping at step 200); fix the one 0.65 Å
adsorbate-overlap geometry case in the generator.
