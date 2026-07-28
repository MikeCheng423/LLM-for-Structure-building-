# ASE-agent LLM — Training Handbook

This handbook is the human-readable contract for how the Structure_building
fine-tune is built, measured, and promoted. It complements the operational
notes in `training/README.md`, the current run state in `training/STATUS.md`,
and the corpus rule in `training/CORPUS_RULE.md`. The end-user guide is
`training/USER_MANUAL.md`.

---

## 1. What we are building

A small instruction model is fine-tuned to act as a **structure-building
agent**: it reads a natural-language request ("a 4-layer 2×2 Cu(100) slab with
12 Å vacuum, bottom two layers fixed") and emits a sequence of **validated ASE
tool calls** that a deterministic workspace executes to produce one canonical
`ase.Atoms` structure.

The model never runs arbitrary code. It only emits `<tool_call>` JSON against a
fixed registry; a bounded controller executes each call, and unsupported or
forbidden requests **fail closed** before any tool runs. Correctness is judged
by the *structure that comes out*, not by the text.

---

## 2. Model and method

| Choice | Value | Why |
| --- | --- | --- |
| Base model | `Qwen/Qwen3-4B-Instruct-2507` | Small, strong tool-use instruct model; Apache-2.0. |
| Pinned revision | `cdbee75f17c01a7cc42f958dc650907174af0554` | Frozen so an upstream update cannot silently change a run. |
| Quantization | 4-bit NF4, double-quant, bf16 compute | Fits training on a 16 GiB RTX 2000 Ada. |
| Adapter | LoRA rank 16, all attention + MLP projections (252 LoRA-B tensors) | Selection/format learning, not capacity, is the bottleneck. |
| Max sequence length | 1600 tokens | Longest rendered trajectory measured ≈ 1,547 tokens. |
| Optimizer schedule | cosine LR, seed 42 | Deterministic, reproducible. |

QLoRA training lives in `training/train_qlora.py`. It always writes
`production_ready: false` into the run manifest — a successful optimizer run is
never, by itself, a shippable model (see §7).

---

## 3. Data pipeline and the structured-request rule

The single most important design decision: **the deterministic generator is the
authority for every tool call and observation. Language models only author free
text (prompt phrasing).**

- The generator builds each case by executing the *real* ASE workspace, so tool
  calls, arguments, and observations are ground truth — never model-authored.
- Phrasing diversity comes from rule-gated templates (`str.format` templates
  authored per family). An agent may write templates; it may **never** write a
  tool call or a tool observation. `evaluate_corpus.py` observation/recipe-hash
  checks reject any drift.

The **structured-request rule** (`training/CORPUS_RULE.md`, machine form in
`training/generators/request_rule.py`) removes ambiguity *at the source*: a
prompt is valid only if it names **every required slot** for its region, so a
conforming prompt maps to exactly one canonical structure. Any generated or
paraphrased prompt that drops a required slot is rejected by
`request_rule.missing_slots` before it can enter the corpus.

Why it matters: the r1–r4 bottleneck was tool *selection* plus templated-phrasing
memorization, not model capacity. Ambiguous prompts (only ~43% named all
required slots) let the model guess. The rule closes that gap.

Every accepted record passes, in order: schema validation → deterministic recipe
execution → replay equivalence → structural invariant checks → forbidden-action
policy → slot-conformance. Splits are grouped by **final-structure hash**, so
equivalent structures or recipes can never straddle train/validation/test.

---

## 4. The corpus (r5)

- Location: `training/datasets/pilot_r5/` (`train.jsonl`, `validation.jsonl`,
  `test.jsonl`, `manifest.json`).
- Size: **2,607 records** — 2,067 train / 252 validation / 288 test.
- Replay: 100% execution success, zero forbidden actions (`corpus_audit.passed`).
- Regions covered: `bulk`, `surface`, `surface_constraint`, `atomic_adsorption`,
  `molecular_adsorption`, `molecule`, `nanotube`, `prototype`, `vacancy`,
  `substitution`, plus multi-turn `clarification` and transactional
  `error_recovery`.

---

## 5. Training

```bash
# See training/run_pilot_r5_remote.sh for the pinned r5 invocation.
PYTHONPATH=src:. PATH=/usr/lib/wsl/lib:$PATH HF_HUB_DISABLE_XET=1 \
  .venv/bin/python training/train_qlora.py \
  --dataset training/datasets/pilot_r5/train.jsonl \
  --eval-dataset training/datasets/pilot_r5/validation.jsonl \
  --output-dir training/runs/pilot-qwen3-4b-r5 \
  --max-length 1600 --max-steps 200
```

The r5 run: 200 steps, effective batch ≈ 8, best validation loss **0.005493**
(checkpoint-200, restored into the final adapter), all 252 LoRA-B tensors
nonzero, adapter SHA-256 `4959a891…`. **Training loss is not a promotion
criterion** — it only tells you the optimizer moved.

---

## 6. Evaluation

`training/evaluations/evaluate_model.py` runs the adapter (and the frozen base,
for comparison) end-to-end through the *same* controller used at deployment, then
replays and scores each produced structure.

Metrics per record and per family:

| Metric | Meaning |
| --- | --- |
| `exact_structure_rate` | Produced structure's `atoms_hash` equals the reference. The strict bar. |
| `invariants_satisfied_rate` | Formula, atom count, pbc, cell lengths/angles, constrained-atom count match within tolerance — order-independent, tolerant of harmless float drift. |
| `exact_tool_sequence_rate` | Executed tool names equal the reference recipe's. |
| `schema_execution_success_rate` | Fraction of emitted calls that validated and executed. |
| `finish_rate` | Reached a `finish`-selected structure. |
| `forbidden_action_rate` | Calls outside the registry / policy. Must be 0. |

**Tool modes.** `--tool-mode record` exposes only the tools the reference
trajectory used — an oracle-assisted *diagnostic* that **cannot promote**.
`--tool-mode full` exposes the entire runtime registry and is the **deployment
gate**: the model must *select* the right tools, not just fill arguments.

**Sampling.** A deterministic stratified sampler (`--per-family` /
`--min-per-family` coverage floor, then `--sample-size` filled by id-hash) picks
an identical, identically ordered set for the base and adapter runs, verified
downstream by `evaluation_ids_sha256`. Use `--limit N` only for a quick smoke.

**Adversarial.** `--negative training/datasets/adversarial/negative.jsonl`
scores forbidden requests for refusal and zero forbidden execution.

---

## 7. Promotion gate (fail-closed)

`training/evaluations/promote_adapter.py` turns three reports (corpus replay,
frozen-base eval, adapter eval) into a single `promoted: true|false`. Promotion
requires **all** of its checks; there are 23. Defaults:

| Threshold | Default |
| --- | --- |
| `minimum_exact_structure_rate` | 0.50 |
| `minimum_schema_execution_rate` | 0.95 |
| `minimum_family_exact_structure_rate` | 0.25 |
| `minimum_invariant_rate` | 0.85 |
| `minimum_family_invariant_rate` | 0.50 |
| `minimum_adversarial_safety_rate` | 1.0 |
| `required_tool_mode` | `full` |

Plus structural guards: same evaluation dataset / ordered records / tool mode /
record count / recipe families between base and adapter; no regression vs base on
exact / invariant / finish / schema; per-family floors met; zero forbidden
actions; adversarial evaluated with zero forbidden executions.

**`production_ready` is separate from `promoted`.** `train_qlora.py`
hard-codes `production_ready: false`. A passing promotion report is *necessary
but not sufficient*: a human **reviewed integration step** consumes the report
and flips the flag in that run's manifest, recording a `promotion` provenance
block (which report, its checks, the metrics). This is the only path to
production-ready; it is per-run and deliberately manual.

---

## 8. Iteration history

| Rev | Change | Outcome |
| --- | --- | --- |
| r1 | First full corpus, base QLoRA | 30% full-registry exact — tool **selection** was the bottleneck, not capacity. Not promoted. |
| r3 | Bounded request/tool router (`tool_router.py`); runtime system prompt as single source of truth | Selection improved; molecular-adsorption still failed the per-family floor. |
| r4 | Made molecular-adsorption prompts fully explicit | Corrective, still template-bound. |
| r5 | Rebuilt the whole corpus under the structured-request rule; rule-gated agent-authored phrasing; `ASEWorkspace.execute` defaults an omitted `name` to the active structure | **Promoted 2026-07-25.** |

The r5 name-default fix is fingerprint-neutral (it changes nothing about the
canonical structures), so the r5 corpus stayed valid without a retrain; it only
closed a "missing required fields ['name']" gap on surface/adsorption prompts.

---

## 9. Current status (r5, promoted)

Full-registry promotion gate, paired base vs. adapter, 120 stratified records
(`training/evaluations/promotion-r5fix120-full-pf2-s120.json`, all 23 checks):

| Metric | Frozen base | r5 adapter |
| --- | ---: | ---: |
| Exact structure | 10.8% | **95.0%** |
| Invariants satisfied | 10.8% | 95.0% |
| Schema execution | 58.2% | 96.4% |
| Finish | 35.0% | 99.2% |
| Adversarial safety | — | 100% |

`atomic_adsorption` reached 100% exact (n=26). `production_ready` is flipped to
`true` in `training/runs/pilot-qwen3-4b-r5/manifest.json` with its provenance
block.

---

## 10. Known limitations (be honest about these)

1. **Shared phrasing form.** Train and test share the same ~10 templates per
   family (grouped by output hash — no structure/recipe leakage, but the same
   phrasing *style*). Genuinely novel-phrasing generalization is **untested**;
   it is the single most important open question.
2. **Bulk cell-convention ambiguity.** On a bulk request that does not state the
   cell convention, the model may return the **primitive** cell where the
   reference uses the **conventional cubic** cell (or vice-versa) — a legitimate
   but non-matching structure. This is the leading residual bulk miss. The user
   manual teaches the phrasing that removes it.
3. **eval loss still dropping at step 200** — more steps may help (r6).
4. **The adsorption height is a corpus constant, not a variable.** Every
   adsorption record in r5 and r6 uses exactly one height per tool
   (`add_atomic_adsorbate` 1.8 Å, `add_molecular_adsorbate` 1.9 Å), so the model
   never learned that the slot varies and drops a stated height for most
   phrasings. Measured on r5: 10 of 11 phrasings lost it on Cu(100)/O/ontop, and
   the surviving one depends on the *unit token* (`angstrom`, not `Å`) — see
   `docs/GUIDED_INPUT.md`. `--guided` composes the wording that works and
   `--strict` fails the build otherwise, but **the fix is corpus-side**: vary the
   height in the next revision. This is the top open defect.
5. **Compound coverage is table-bound, and untrained.** Metal oxides, sulfides
   and III–Vs build through the composition-parameterised prototype families
   (45 tabulated compositions) and compound slabs/vacancies/substitutions work,
   but no corpus record teaches any of it — the capability rests on the tool
   *description* steering r5, which measured 7/7 but is not a trained behaviour.
   Non-binary compounds need §12.

---

## 11. How to iterate (r6 recipe)

1. Extend phrasing templates (more paraphrase forms per family) and/or add
   families; keep the deterministic generator as the authority and re-run
   `evaluate_corpus.py` — it must stay 100% executable, zero forbidden, 100%
   routable, and slot-conforming.
2. Regenerate `training/datasets/pilot_r6/` with grouped splits.
3. Re-check token lengths with `analyze_tokens.py`; keep `--max-length` above the
   measured maximum.
4. Train with more steps; keep seed 42 for comparability.
5. Run paired base/adapter `--tool-mode full` evals on an identical stratified
   sample; run the adversarial pass.
6. `promote_adapter.py`; only on a clean pass does a reviewed step flip
   `production_ready` for that run.
7. To probe limitation #1 deliberately, evaluate on a held-out set whose phrasing
   templates never appeared in training.

---

## 12. Planned family: `build_crystal` (registered, unrouted, untrained)

`build_crystal` is the only builder in the registry with **zero corpus coverage**.
It exists, it works, and no record teaches it — so it is dead surface today. This
section is the plan to change that; nothing here is active until a revision
implements it.

### 12.1 Why it is needed

`build_bulk` is elemental. `build_prototype` covers binaries through the
composition-parameterised families (`rocksalt`, `zincblende`, `wurtzite`,
`fluorite` — see `docs/GUIDED_INPUT.md`). Neither can express:

- ternaries and perovskites — SrTiO3, BaTiO3, LaAlO3;
- non-binary-ratio oxides — corundum Fe2O3 / Al2O3, spinels;
- any structure whose basis is not one of the four tabulated Wyckoff patterns.

`build_crystal(symbols, basis, spacegroup, a, [b, c, alpha, beta, gamma])` is
general enough for all of them. It is the natural next capability, and the only
one that needs *training* rather than a table entry.

### 12.2 The honest risk, from r6 and r7

This family is **higher-risk than any existing one**, and the reason is on record:
both failed revisions failed the same way. r6 hallucinated a spurious `a` on
`build_surface` and a wrong `bond` on `build_nanotube`; r7 invented an `xy`
argument the corpus never teaches. Every call then failed validation and the
model retried to the turn budget (`BUDGET_EXHAUSTED`).

`build_crystal` has by far the largest argument surface in the registry — a list
of symbols, a **nested array of fractional coordinates**, an integer space group,
and up to six cell parameters. It is exactly the shape that invites that failure
mode, and a wrong basis coordinate produces a structure that *builds successfully
and is silently wrong* — the composition check in `request_check.check_composition`
would not catch a displaced site.

Mitigations, in order of expected value:

1. **Train it as a closed vocabulary, not free construction.** Restrict the corpus
   to a curated table of named crystals (SrTiO3, BaTiO3, LaAlO3, Fe2O3, Al2O3,
   MgAl2O4, ...) with fixed reference bases. The model then learns to *recall* a
   basis for a named material, not to invent coordinates. This is the single most
   important decision in the plan.
2. **Prefer a name-driven path.** Strongly consider extending `build_prototype`'s
   table instead for any material that fits, and reserve `build_crystal` for
   structures that genuinely need a space-group basis. A table entry needs no
   retrain; this family does.
3. **Add a deterministic post-build check** for basis plausibility — minimum
   interatomic distance and stoichiometry against the requested formula — so a
   displaced site fails loudly like a dropped height does.

### 12.3 Corpus and rule changes

Add a `crystal` region to the structured-request rule with required slots:

| Slot | Meaning |
| --- | --- |
| `formula` | the material, e.g. SrTiO3 — the key into the reference table |
| `spacegroup` | stated by name or number (Pm-3m / 221) |
| `lattice` | the lattice constant(s) the reference uses |

`repeat` stays optional. The generator builds each case by executing the real
workspace, as every other family does — **no agent may author a basis**, exactly
as no agent may author a tool call (§3).

### 12.4 Coupled edits — all four, or the build breaks

Adding the region touches four places, and one of them fails at *import* if
missed:

1. `training/generators/request_rule.py` — `FAMILY_REQUIRED["crystal"]`, plus
   `CORPUS_RULE.md` for the human form.
2. `src/ase_auto_build/ase_agent/request_check.py` — the mirrored
   `FAMILY_REQUIRED`, a `_slot_stated` branch per new slot, and an `infer_region`
   keyword rule. `tests/test_ase_agent_cli.py` asserts the two tables stay
   identical.
3. `src/ase_auto_build/ase_agent/tool_router.py` — a route exposing
   `build_crystal`; today it appears only in `_CLARIFY_TOOLS`, so no ordinary
   request can reach it.
4. `src/ase_auto_build/ase_agent/guided.py` — a `crystal` entry in
   `REGION_SLOTS`, `REGION_ORDER`, `TEMPLATES` and `REGION_BLURBS`.
   **`_assert_covers_rule()` raises at import time** if the rule gains a region
   or a required slot the wizard does not ask for, so an incomplete change breaks
   every entry point immediately rather than silently emitting under-specified
   prompts. That is deliberate — do not weaken the assertion to land the change.

Screen any new phrasing template against `check_request` before committing it:
the corpus rule is stronger than the runtime heuristic, so a template can be
rule-conforming yet read as missing a slot at run time.

### 12.5 Gate

Standard promotion rules (§7) apply unchanged, plus two family-specific bars,
because "it executed" is not evidence of a correct crystal:

- **Per-family exact-structure ≥ the 25% floor** in `--tool-mode full`, measured
  paired against the frozen base on an identical sample.
- **Zero spurious-argument loops** on the `crystal` family — the r6/r7 failure
  mode, checked in `generated_calls` rather than inferred from the exact-match
  rate.

Adding a family changes the family mix for *every* other family, so the whole
paired evaluation must be re-run; do not compare a `crystal`-bearing revision
against an r5-era report per-family without saying so.

### 12.6 Sequencing

Land this **after** the current adsorption-height corpus defect (§10.4), not
alongside it. r6 and r7 each changed more than one thing and neither could be
promoted; the regression cause took a separate investigation both times. One
change per revision.

---

## 13. Safety model

- The model emits only registry tool calls; the workspace executes them. No
  arbitrary Python / shell / filesystem / network / calculator construction.
- The bounded router fail-closes on unsupported or forbidden requests before any
  tool executes; the adversarial gate proves zero forbidden execution.
- Never place API keys, remote credentials, proprietary POTCAR files, private
  structures, or unredacted transcripts in training data (`AGENTS.md`).

---

## 14. File map

| Path | Role |
| --- | --- |
| `training/train_qlora.py` | QLoRA trainer; writes the run manifest. |
| `training/CORPUS_RULE.md` / `generators/request_rule.py` | Structured-request rule (human / machine). |
| `training/generators/` | Deterministic corpus generator, template fill, paraphrase templates. |
| `training/evaluations/evaluate_corpus.py` | Corpus replay / drift gate. |
| `training/evaluations/evaluate_model.py` | End-to-end held-out model evaluation. |
| `training/evaluations/promote_adapter.py` | Fail-closed promotion decision. |
| `training/chat_agent.py` | Interactive entry point to the deployed model. |
| `training/runs/<run>/manifest.json` | Per-run provenance + `production_ready`. |
| `training/USER_MANUAL.md` | End-user guide with worked examples. |
