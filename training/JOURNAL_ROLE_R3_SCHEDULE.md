# Journal-role r3 schedule — teaching the extractor to read prose

Status: **proposed**, 2026-07-31. Supersedes nothing; r2 stays unpromoted and
r5 remains the shipping adapter (`production_ready: true`, direct ASE tool
calling — this document does not touch that track).

## 1. Why

Two defects blocked every live `ASE_catalyst_build` run since 2026-07-28. Both
were invisible to the offline harness, and the second is the reason r3 is
needed at all.

**Defect 1 — tool-schema key order (fixed, commit `60a3f74`).**
`build_journal_corpus.py` writes the corpus JSONL with `sort_keys=True`, so the
`tools` stored in every record load back alphabetically ordered.
`apply_chat_template` renders the tool schema into the prompt verbatim, so the
model only ever trained on sorted tool properties. At inference
`evidence_call`/`proposal_call` built the tool in Python insertion order; the
model mirrored that unfamiliar order in its output and dropped
`catalyst_spec.modifications`. Canonicalising the tool makes the rendered prompt
byte-identical to training (`sha 37b4fe2b754bd167` both ways).

The offline harness reads tools back from the sorted JSONL, so it *cannot*
observe this defect. That is how a 0.967 exact-payload rate coexisted with a
100% live failure rate.

**Defect 2 — the extractor copies literals, it does not read prose (open).**
With defect 1 fixed the live smoke advances one validation stage and fails on:

```text
catalyst_spec.model.atom_ordering must be one of ['ase_default']
```

The extractor emitted `"ase-default"` and `model.center = "geometry"`. The cause
is the corpus renderer: every source sentence is a template wrapped around a
JSON literal of the target value.

| | source text |
| --- | --- |
| corpus | `The reported atom ordering is "ase_default".` |
| live smoke | `...centered geometry, ASE-default atom ordering, and CIF plus VASP outputs.` |

The corpus never requires the model to *map* a prose form onto a canonical
schema value — only to copy a literal that is already present. Real paper
excerpts, which are the entire premise of this system (`CATALYST_STRUCTURE_LLM.md`
§1), never look like the corpus.

**This also limits `template_holdout`.** It varies the sentence *frame* while
keeping the embedded JSON literal, so its 0.980 exact rate does not measure
prose interpretation at all. The set is still valid for what it tests; it simply
does not test this axis.

## 2. Principle

Measure the gap before spending 16 GPU-hours closing it. A cheap prose-style
evaluation set tells us whether this is a narrow enum-mapping problem (a handful
of fields whose schema admits one canonical spelling) or a broad interpretation
failure, and it becomes the regression test for the corpus fix either way.

**Explicitly rejected:** deterministically coercing `"ase-default"` →
`ase_default` at the pipeline boundary. It would turn the smoke green today, but
`CATALYST_STRUCTURE_LLM_AGENT_DESIGN.md` §3 forbids silently normalising model
output into reported values, and it would conceal precisely the capability the
product requires.

## 3. Phases

### Phase 1 — `prose_holdout`, and measure the gap (cheap)

Build an evaluation-only set whose source text is genuine prose requiring
interpretation, with identical canonical targets to the existing families.
Cost: ~1 h CPU to build, ~1–2 h GPU to measure.

Requirements:

- 40–60 cases across `bulk`, `surface`, `adsorbate`, `defect`.
- Source text contains **no JSON literal of any target value**. Generation must
  fail closed if `json.dumps(value)` appears verbatim in the rendered text —
  this is the property that distinguishes the set from `template_holdout`.
- Cover the prose forms the smoke exposed: `centered`/`centred slab` → `true`,
  `ASE-default atom ordering` → `"ase_default"`, `2 x 2` and `2×2` supercells,
  `four-layer` → `4`, `(111)` → `[1,1,1]`, `15 Å of vacuum` → `15.0`,
  `CIF plus POSCAR` → `["cif","vasp"]`.
- Every target still passes `policy_gate` and `dispatch_spec`.

Exit gate: a measured extractor field-recall and exact-payload rate for r2 on
`prose_holdout`, per field, so we know which fields fail and how often.

### Phase 2 — corpus renderer (conditional on Phase 1)

Change `build_journal_corpus.py` so a share of records render each value as
prose rather than a JSON literal, keeping the deterministic generator as the
authority for every target (`CLAUDE.md`: an agent may write phrasing, never a
tool call or observation). Keep the existing literal forms for part of the
corpus so the model retains both skills.

Exit gate: regenerate, `evaluate_journal_corpus.py` passes (execution 1.0, zero
forbidden actions, no split-group overlap), and `prose_holdout` remains disjoint
from training wording.

### Phase 3 — r3 and the full cycle

`run_journal_role_r3.sh` with hyperparameters identical to r1/r2 so the corpus
change is the only variable, then the existing hardened driver: vertical-slice
gate → offline eval vs r5 baseline → **live smoke before promotion** → holdout
eval, now including `prose_holdout`. ~16 h unattended.

Promotion stays fail-closed. `journal_role_ready` flips only if every check
including the live smoke passes; `production_ready` remains human-only.

## 4. Verification

- Full suite green at every commit (856 tests at time of writing).
- `run_vertical_slice_gate.py` exits 0 — the deterministic slice is unaffected
  by any of this and must stay that way.
- Phase 1's generator fails closed on literal leakage, tested.
- The live smoke is the acceptance test for the whole schedule. It has been the
  only check telling the truth about this track.

## 5. Loose end

r2 has no held-out numbers: stage 3 of its cycle aborted correctly when
promotion failed. ~4 h to run. Recommended to **skip** — the corpus is about to
change and those numbers will not carry forward.
