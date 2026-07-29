# Catalyst structure LLM supplemental training schedule

Status: r3 and r4 rejected; r5 remains production
Baseline to preserve: `training/runs/pilot-qwen3-4b-r5/adapter`  
Proposed supplement revision: `catalyst-supplement-r1`

Implementation update (2026-07-28): the registry now includes deterministic
icosahedral, octahedral, and cuboctahedral nanoparticles. Supported clusters
reuse the bounded `build_surface` + `build_nanoparticle` + `combine` recipe, and
surface vacancies/substitutions compose `build_surface` with typed top-layer
selectors. The resulting r2 corpus contains 2,646 replay-verified trajectories
on registry `phase1-f6f5dcb756b74fb7`; its audited maximum is 1,800 tokens.
Termination control, repeated coverage/coadsorption, and lattice matching remain
future runtime work.

Training update (2026-07-28): r2 completed its 200-step full-auto run and
passed adapter integrity, but the strict 32-record paired gate rejected it.
r2 reached 100% schema execution and finish with zero forbidden calls, while
exact structure and invariants reached 71.9%, below r5's 75.0% and 87.5% on
the same IDs. `training/STATUS.md` records the failure classes and evidence.

Corrective update (2026-07-28): r3 restored structural defaults but regressed
nanotube and molecular-adsorption families because it trained a new adapter from
the frozen base. r4 continued from the promoted r5 adapter and repaired those
families, but regressed slab repeats and supported clusters. The strict screen
rejected r4 at 78.4% exact structure. The run changed no corpus target or
promotion threshold and records both source and output adapter hashes.

This schedule extends the bounded r5 Qwen3-4B tool-calling model toward the
journal-reconstruction scope in `CATALYST_STRUCTURE_LLM.md`. It does not replace
r5 until the expanded runtime, corpus, chemistry holdouts, and promotion gates
all pass. More paraphrases alone are not a useful supplement: r5 already reached
99.2% exact structure on valid novel phrasing, while r6/r7 regressed after the
larger 300-step recipe.

## Target and non-goals

The supplement targets structures that are missing or materially under-covered:
alloy and compound surfaces, oxides, surface defects, adsorption-site and
coverage variation, coadsorption, and supported clusters/interfaces. It retains
the current r5 recipe groups as replay protection.

This round does not train activity prediction, global structure optimization,
amorphous reconstruction, inference from microscopy alone, or unrestricted
Python/shell execution. Ambiguous and unsupported requests must clarify or fail
closed.

## Dataset target

Create **1,150 new reviewed structure cases**, then render three to five prompt
forms per case (3,450-5,750 new conversations; never more than six per case).
Cases, not paraphrases, are the unit used to meet the allocation below.

| Family | New cases | Required variation |
| --- | ---: | --- |
| Elemental surfaces and adsorption | 150 | Retained low-index coverage plus stepped/high-index facets and bridge/fcc/hcp/ontop sites |
| Ordered/substituted alloy surfaces | 180 | Composition, layer/site substitution, ordered and dilute surface alloys |
| Compound and oxide surfaces | 200 | At least multiple compositions, facets, and explicit terminations; include NiO, TiO2, and CeO2-style cases |
| Surface defects | 150 | Surface vacancies and substitutions with deterministic site selectors |
| Coverage and coadsorption | 180 | Repeated adsorbates, variable coverage, multiple adsorbates, orientation, and collision cases |
| Supported atoms, clusters, and interfaces | 180 | Single atoms, small clusters, support placement, lattice matching, and interface gaps |
| Ambiguous, contradictory, or unsupported | 110 | Clarification, candidate branching, provenance conflicts, and fail-closed refusals |
| **Total** | **1,150** | |

Across the families, include English and Traditional Chinese prompts, unit
conversions, corrections, conflicting evidence, and multi-turn clarification.
Record source/license metadata and locators, but exclude proprietary structures,
POTCAR data, credentials, private user material, and unnecessary copyrighted
text.

Retain every valid r5 recipe group at least once in the training replay pool and
retain the complete r5 evaluation set as a regression suite. Cap duplicate prompt
forms and balance the rendered training records so the supplement does not repeat
the r7 skew toward atomic adsorption. Prefer another reviewed structure over an
additional paraphrase.

## Schedule and gates

### Phase 0 — freeze the training contract (2-3 days)

1. Finish and test the in-progress compound and molecular-adsorbate geometry
   changes.
2. Freeze a schema version, runtime registry version/fingerprint, system prompt,
   and normalized-structure hashing rule.
3. Mark reports generated before the geometry change as incompatible unless their
   registry fingerprint and output hashes match. In particular, do not use the
   stale non-`.r6set` r5 novel report as promotion evidence.
4. Write the family taxonomy, parameter bounds, split keys, and provenance fields
   into the dataset manifest before generation starts.

Exit gate: the runtime and dataset contract are versioned, and a registry change
would force corpus regeneration rather than silently reusing old targets.

### Phase 1 — make the runtime expressive (2-3 weeks)

Implement deterministic, allow-listed operations before writing examples for:

- ordered/substituted alloy slabs and surface vacancy/substitution;
- explicit terminations, stepped facets, and distinct adsorption sites;
- repeated adsorbates, coverage, and coadsorption;
- supported atoms/clusters, lattice matching, and interfaces.

Each operation must have deterministic selectors and replay, atom-budget limits,
composition/site/coverage/interface invariants, overlap checks, and fail-closed
routing. Add runtime probes for the seven dataset families and adversarial probes
that demonstrate arbitrary execution is still unavailable.

Exit gate: every positive benchmark family can be expressed through the frozen
registry and passes deterministic construction and invariants. If a family cannot
be represented, keep it as an unsupported negative; do not teach a fictitious
tool call.

### Phase 2 — build and review the benchmark (2-3 weeks)

1. Generate parameterized ground-truth recipes within documented chemical bounds.
2. Execute each case, validate atom count/composition/PBC/cell, site and coverage,
   constraints, distances/overlaps, and requested-format round trips.
3. Review all ambiguous/unsupported cases and a stratified sample of positive
   cases with a domain expert; correct the recipe, not only its wording.
4. Group by normalized final-structure hash before splitting. Keep all prompt forms
   and correction turns for one structure in one partition.
5. Produce train/validation/test plus named challenges for unseen composition,
   adsorbate, facet/termination, interface, and complete publication or
   material-adsorbate holdouts.

Exit gate: 1,150 new cases meet the allocation; each split contains its intended
families; no normalized structure, recipe, paper, or near-duplicate excerpt leaks
across its declared holdout boundary.

### Phase 3 — render and audit the corpus (3-5 days)

Generate a new directory such as `training/datasets/catalyst_supplement_r1/`;
never append it to an older revision. Include the r5 replay pool, dataset manifest,
licenses/provenance, registry fingerprint, and deterministic seed.

Run the existing corpus and token audits:

```bash
PYTHONPATH=src:. .venv/bin/python training/evaluations/evaluate_corpus.py \
  training/datasets/catalyst_supplement_r1 \
  --report training/evaluations/catalyst_supplement_r1_corpus_report.json

PYTHONPATH=src:. HF_HUB_DISABLE_XET=1 .venv/bin/python \
  training/evaluations/analyze_tokens.py \
  training/datasets/catalyst_supplement_r1 \
  --report training/evaluations/catalyst_supplement_r1_token_report.json
```

Exit gate:

- 100% schema validation, construction, deterministic replay, and structural
  invariant success for accepted trajectories;
- zero forbidden actions;
- zero split leakage under the declared hash/group keys;
- every record fingerprint matches the frozen registry;
- maximum rendered length is known and the chosen training length covers it with
  headroom rather than truncation.

### Phase 4 — 4B QLoRA pilot (one GPU day after gates pass)

Start from the proven r5 recipe, changing the dataset while holding the other
training variables fixed:

- base: `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`;
- QLoRA rank 16, learning rate `2e-4`, seed 42;
- per-device batch 1, gradient accumulation 8 (effective batch 8);
- 200 optimizer steps, validation every 25 steps, restore the best checkpoint;
- official model chat template and assistant-output loss masking;
- maximum length selected from the Phase 3 token report (the prior 1,600/1,792
  values are not assumed adequate for the expanded schema).

Launch into a new, non-existent run directory:

```bash
PYTHONPATH=src:. PATH=/usr/lib/wsl/lib:$PATH \
HF_HOME=training/cache/huggingface HF_HUB_DISABLE_XET=1 \
.venv/bin/python training/train_qlora.py \
  --dataset training/datasets/catalyst_supplement_r1/train.jsonl \
  --eval-dataset training/datasets/catalyst_supplement_r1/validation.jsonl \
  --output-dir training/runs/catalyst-supplement-r1 \
  --max-length <AUDITED_MAX_LENGTH> \
  --max-steps 200 \
  --gradient-accumulation 8 \
  --eval-steps 25 \
  --learning-rate 2e-4 \
  --lora-rank 16 \
  --seed 42 \
  --minimum-records 1000
```

GPU launch conditions are all mandatory:

1. The Phase 0-3 exit gates are recorded and passing.
2. `.venv` sees CUDA and the expected 16 GiB RTX 2000 Ada; `pip check` passes.
3. NVIDIA telemetry, not the unreliable WSL PyTorch peak fields, shows no other
   training/inference process and at least 12 GiB free immediately before launch.
4. A one-step run using the longest rendered record succeeds without truncation or
   OOM.
5. The base revision is cached/resolvable and the destination directory is absent
   or empty.

If any condition fails, queue the run and continue CPU-side review/auditing. Do
not compete for a partially occupied GPU.

### Phase 5 — evaluate and promote (2-4 days)

First run one deterministic case per family in full-tool mode. Stop there if any
family cannot execute, finish, or satisfy its invariants. Otherwise evaluate the
frozen base, r5, and the supplement adapter on identical ordered IDs for the full
standard and chemistry-challenge sets, plus the adversarial set. Use normalized
structure as the primary score; report exact tool sequence only as a diagnostic.

Promotion requires all of the following:

- at least 95% overall exact structure and invariant satisfaction;
- at least 90% exact structure and invariant satisfaction in every catalyst
  family;
- at least 98% schema execution and at least 98% finish;
- 100% adversarial safety and zero forbidden executions;
- no geometry, safety, or family regression on the retained r5 suite;
- correct composition, facet/termination, site, height/orientation, coverage,
  constraints, overlap status, and interface gap where applicable;
- a passing review of an untouched, stratified expert sample;
- matching dataset hashes, ordered evaluation IDs, tool mode, registry
  fingerprint, and complete reports for every compared model.

`promote_adapter.py` can enforce the stricter exact/execution/invariant thresholds
with explicit flags, but its report must be supplemented with checks for the 98%
finish threshold, registry fingerprint, chemistry challenges, r5 regression
suite, and expert review. Keep `production_ready: false` until a separate reviewed
promotion consumes all of that evidence.

## Failure and fallback policy

- **Runtime or corpus gate fails:** fix the deterministic operation or source
  recipe, regenerate the affected revision, and rerun the full audit. Do not train
  around invalid targets.
- **Longest-record smoke OOMs:** confirm the GPU is idle; then use a GPU with enough
  memory or reduce sequence size only by simplifying legitimately redundant
  prompt text. Never truncate required schema/provenance fields.
- **Pilot loses an old family:** keep r5 deployed, inspect generated calls, and add
  a small number of reviewed counterexamples for that failure. Change one variable
  in the next run; do not repeat the r6/r7 combination of more prompts and 300
  steps.
- **New family misses its gate:** distinguish router/runtime errors from model
  selection errors. Fix routing/runtime first; for model errors, add structure
  diversity rather than paraphrase volume and rerun the 200-step 4B pilot.
- **4B still fails after runtime, balance, leakage, and target validity pass:** run
  one paired 7B/8B comparison using the same records, seed, tool exposure, and
  evaluation IDs. Scale only if it improves the failed chemistry/long-horizon
  families enough to justify memory and latency.

## Evidence basis

- `CATALYST_STRUCTURE_LLM.md`: target scope, deterministic schema/tool boundary,
  validation requirements, 5,000-20,000-example direction, and milestone gates.
- `training/STATUS.md`: r5 promotion evidence, r6 regression, and the warning that
  lower validation loss did not predict deployable behavior.
- `training/evaluations/CATALYST_4B_CAPABILITY.md`: current chemistry gaps, journal
  probes, 1,000-1,500-case benchmark recommendation, challenge splits, r5-derived
  200-step recipe, promotion thresholds, and paired larger-model fallback.
- `training/README.md` and the r5 manifest: pinned model/environment, replay-before-
  training rule, 16 GiB GPU evidence, full-tool promotion workflow, and reviewed
  `production_ready` boundary.
