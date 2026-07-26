# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ASE_auto_build` — a QLoRA fine-tune of `Qwen/Qwen3-4B-Instruct-2507` that turns a
plain-language structure request into a sequence of **validated ASE tool calls**,
executed by a deterministic bounded workspace to produce one canonical
`ase.Atoms` structure plus a replayable recipe sidecar.

The model never runs code. It emits `<tool_call>` JSON against a fixed registry;
everything else is deterministic Python.

Read `training/HANDBOOK.md` first — it is the authoritative contract for how the
fine-tune is built, measured, and promoted. `training/STATUS.md` holds live run
state; `training/CORPUS_RULE.md` holds the request rule.

## Commands

The venv is `.venv/`. Scripts under `training/` are run as modules from the repo
root and need both `src/` and the root importable:

```bash
export PYTHONPATH="$PWD/src:$PWD"
export HF_HOME="$PWD/training/cache/huggingface"
export HF_HUB_DISABLE_XET=1
```

```bash
# Tests — conftest.py puts src/ on sys.path, so bare pytest works.
.venv/bin/python -m pytest -q                              # whole suite
.venv/bin/python -m pytest tests/test_ase_agent_cli.py -q  # one file
.venv/bin/python -m pytest -q -k "adsorption and not molecular"
.venv/bin/python -m pytest "tests/test_training_generation.py::test_schema_drift_is_rejected" -q

# Run the agent (needs a CUDA GPU + the [agent] extra)
.venv/bin/ASE_auto_build "Build a 2x2 Cu(100) slab, 4 layers, 12 A vacuum."
.venv/bin/ASE_auto_build --base-only --json "..."   # frozen base, no adapter

# Corpus: generate, then gate it
.venv/bin/python training/generators/generate_corpus.py \
    --templates-dir training/generators/paraphrase_templates_r6 \
    --output-dir training/datasets/pilot_r6
.venv/bin/python training/evaluations/evaluate_corpus.py training/datasets/pilot_r6
.venv/bin/python training/evaluations/analyze_tokens.py training/datasets/pilot_r6

# Train (see training/run_pilot_r*_remote.sh for the pinned invocations)
.venv/bin/python training/train_qlora.py \
    --dataset training/datasets/pilot_r6/train.jsonl \
    --eval-dataset training/datasets/pilot_r6/validation.jsonl \
    --output-dir training/runs/pilot-qwen3-4b-r6 --max-length 1600 --max-steps 200

# Evaluate + promote (whole pipeline is scripted per revision)
bash training/run_r6_evaluation.sh
```

Installing: `pip install -e ".[agent]"` (install a CUDA-matched `torch` first —
it is deliberately unpinned). `[dev]` is just pytest; the suite is GPU-free
because model interaction goes through scripted fakes.

## Architecture

**Runtime path** (`src/ase_auto_build/ase_agent/`) — a request flows:

`cli.py` → `request_check.py` (advisory slot check) → `tool_router.py` →
`controller.py` → `workspace.py` → `export.py`

- `tool_router.route_tools()` narrows the full registry to the tools a request
  could legitimately need, and **fails closed** — an unroutable request is
  refused *before any tool runs*. This is the safety boundary, not a hint.
- `controller.py` owns `SYSTEM_PROMPT`, the single source of truth for the
  prompt at both training and deployment. Changing it invalidates comparisons
  against existing eval reports.
- `workspace.ASEWorkspace` executes each validated call against real ASE and is
  the only thing that touches geometry. Policy bounds (`policy.py`) cap model
  turns and tool calls.
- `llm_local.py` loads the model and parses tool calls. `cli.py` and
  `evaluations/evaluate_model.py` both import it, so the measured exact-match
  rate describes what the shipped command actually does. Keep it that way.
- `export.py` writes `POSCAR` + `structure.json` (recipe, `recipe_hash`,
  `atoms_hash`, invariants, tool sequence) — the sidecar is what makes a build
  reproducible without the model.

`structure.py`, `target_utils.py`, and `ase_tools.py` sit at the package root
(not under `ase_agent/`) and are the agent's only dependencies outside its own
subpackage — `tools_build.py` imports `make_prototype` from `structure`.

**Training path** (`training/`) — the governing rule:

> The deterministic generator is the authority for every tool call and
> observation. Language models only author free text (prompt phrasing).

An agent may write `str.format` paraphrase templates; it may **never** write a
tool call or a tool observation. Each generated record executes the *real*
workspace, so calls and observations are ground truth, and
`evaluate_corpus.py`'s observation/recipe-hash checks reject drift.

Every accepted record passes, in order: schema validation → recipe execution →
replay equivalence → structural invariants → forbidden-action policy → slot
conformance (`generators/request_rule.py`, the machine form of
`CORPUS_RULE.md`). Splits are grouped by **final-structure hash**, so equivalent
structures can never straddle train/test.

## Non-obvious constraints

- **`promoted` ≠ `production_ready`.** `train_qlora.py` hard-codes
  `production_ready: false`. `promote_adapter.py` yields `promoted: true` only
  if all 23 checks pass — and that is *necessary but not sufficient*. A human
  reviewed step flips `production_ready` in that run's manifest with a
  provenance block. Never flip it automatically.
- **`--tool-mode record` cannot promote.** It exposes only the tools the
  reference trajectory used — an oracle-assisted diagnostic. `--tool-mode full`
  is the deployment gate, because tool *selection* was the r1–r4 bottleneck.
- **Base and adapter must see an identical, identically ordered sample.**
  Verified downstream by `evaluation_ids_sha256`; the promotion gate compares
  dataset, ordering, tool mode, record count, and families between the two runs.
- **Training loss is not a promotion criterion.** It only says the optimizer
  moved.
- Known open limitation: train and test share phrasing templates, so novel-phrasing
  generalization is measured separately against a held-out phrasing set
  (`pilot_r6_novel/`). See HANDBOOK §10.

## Repository boundary — important

This checkout contains more than what is published. GitHub
(`MikeCheng423/LLM-for-Structure-building-`) carries **only** ASE_auto_build:
the agent, `training/`, its tests, and packaging. The VASP/QE calculation code,
the `vasp_auto_ui` web UI, `example/`, and their tutorials and tests are on disk
but **git-ignored on purpose** — they are not deleted and must not be.

Consequences when working here:

- `.gitignore` allowlists `src/ase_auto_build/`'s three shared modules by name,
  so a **new** top-level module in that directory is ignored by default. Add it
  to the allowlist deliberately if it belongs to the agent.
- `python -m pytest` locally runs ~616 tests (including the VASP ones); the
  published subset is ~156. Both should pass.
- The package was renamed `vasp_auto` → `ase_auto_build`. `vasp_auto_ui` and
  `vasp_auto_background_logs` kept their names — they belong to the calculation
  side. `/home/vv/vasp_auto` in `AGENTS.md` / `SNAPSHOT.md` is a **different,
  external, read-only** reference checkout; do not rename or touch it.
- `VASP_AUTO_ASE_ADAPTER` is still honoured as a legacy alias for
  `ASE_AUTO_BUILD_ADAPTER`.

## Safety model (from AGENTS.md)

- Never put API keys, remote credentials, proprietary POTCAR files, private
  structures, or unredacted transcripts into training data.
- Never give the model arbitrary Python, shell, filesystem, network, or
  calculator-constructor execution. The registry is the whole surface.
- Preserve the deterministic recipe and validation for every accepted trajectory.
