# Training environment

Training is isolated from the original `vasp_auto` checkout. On the `tlclab`
WSL host the workspace lives at `/home/tlclab/Structure_building` and its Python
environment lives at `/home/tlclab/Structure_building/.venv`.

The host currently exposes an NVIDIA RTX 2000 Ada GPU with 16 GiB VRAM through
WSL, driver 551.61, and CUDA compatibility 12.4. Use the CUDA 12.4 PyTorch wheel;
do not install a Linux NVIDIA driver inside WSL.

```bash
cd /home/tlclab/Structure_building
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r training/requirements-train.txt
```

Verify that the environment is using the WSL GPU:

```bash
PATH=/usr/lib/wsl/lib:$PATH .venv/bin/python - <<'PY'
import torch

assert torch.cuda.is_available(), "PyTorch cannot see the WSL GPU"
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
```

## Readiness gate

Do not start a production fine-tune until the versioned ASE tool registry and
trajectory exporter exist. Every training record must have passed schema
validation, deterministic recipe execution, structural invariant checks, and
the forbidden-action policy. A generic GPU smoke test is not a trained
Structure_building model and must not be published as one.

The pinned pilot base is `Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554` (Apache-2.0). The revision is fixed
so an upstream update cannot silently change a run.

For an environment-only one-step test:

```bash
cd /home/tlclab/Structure_building
PYTHONPATH=. .venv/bin/python training/generators/make_smoke_dataset.py \
  training/runs/smoke/input.jsonl
PYTHONPATH=. PATH=/usr/lib/wsl/lib:$PATH HF_HUB_DISABLE_XET=1 \
  .venv/bin/python training/train_qlora.py \
  --dataset training/runs/smoke/input.jsonl \
  --output-dir training/runs/smoke/qwen3-4b-smoke \
  --max-length 512 \
  --max-steps 1 \
  --gradient-accumulation 1 \
  --allow-smoke-data
```

The resulting manifest is deliberately marked `production_ready: false`.

## Verified remote smoke run

On 2026-07-24, the environment and one-step QLoRA path were verified on the
`tlclab` WSL host. CUDA BF16 matrix multiplication and bitsandbytes NF4
quantization passed. The Qwen3 4B smoke run completed one optimizer step with a
nonzero `2e-4` learning rate, loss `2.2283`, and runtime `3.81 s`. All saved
LoRA-B tensors contained updated values. These numbers establish environment
functionality only and are not a model-quality result.

The base-model cache occupies about 7.6 GiB. A smoke adapter and tokenizer
occupy about 142 MiB. Both are ignored by Git under `training/cache/` and
`training/runs/`.

## Production corpus

Generate trajectories from the authoritative runtime registry, execute every
step, replay every recipe, and create grouped splits:

```bash
cd /home/tlclab/Structure_building
PYTHONPATH=src:. .venv/bin/python training/generators/generate_corpus.py \
  --output-dir training/datasets/pilot
PYTHONPATH=src:. .venv/bin/python training/evaluations/evaluate_corpus.py \
  training/datasets/pilot \
  --report training/evaluations/remote_pilot_corpus_report.json
```

The initial corpus contains 1,293 trajectories: 1,068 train, 108 validation,
and 117 test. Its three paraphrases for each underlying generated case remain in
one partition. Local and remote replay both report 100% execution/equivalence,
zero exact recipe/final-structure overlap across splits, and zero forbidden
actions.

The next corpus revision is generated separately under
`training/datasets/pilot_r2`. It contains 1,314 trajectories (1,038 train, 129
validation, 147 test) and adds 12 transactional error-recovery examples plus 9
multi-turn clarification examples.
The controller resumes these clarification sessions with the complete model and
tool context; it does not start a second stateless request. Keep registry
versions separate—never append r2 records to an r1 training file.

R2 assigns splits at the final-structure-hash group level, so equivalent
outputs and recipes cannot cross partitions even when reached through different
recovery paths. The stratifier guarantees every one of its 12 recipe families
appears in train, validation, and test.

Analyze the exact Qwen-rendered token lengths:

```bash
PYTHONPATH=src:. HF_HUB_DISABLE_XET=1 .venv/bin/python \
  training/evaluations/analyze_tokens.py training/datasets/pilot \
  --report training/evaluations/pilot_token_report.json
```

The longest trajectory is 1,502 tokens, so the pilot uses a 1,536-token limit.
A longest-record memory smoke test reserved 8.15 GiB through PyTorch and
completed one optimizer update on the 16 GiB RTX 2000 Ada.

## Remote pilot

`training/run_pilot_remote.sh` launches the pinned 100-step production pilot
with effective batch size 8 and validation every 25 steps. Run it in WSL tmux:

```bash
tmux new-session -d -s ase-agent-pilot \
  "bash /home/tlclab/Structure_building/training/run_pilot_remote.sh"
tmux capture-pane -p -t ase-agent-pilot -S -120
```

The run directory and console log are:

```text
training/runs/pilot-qwen3-4b-r1/
training/runs/pilot-qwen3-4b-r1.log
```

### Completed r1 pilot and evaluation

The isolated r1 pilot completed on the `tlclab` WSL host on 2026-07-24. It ran
100 optimizer steps in 2,612.375 seconds (43 minutes 32 seconds), with effective
batch size 8. Validation loss was measured every 25 steps; checkpoint 75 was
best at `0.0023228973` and was restored for the final adapter. The final adapter
file SHA-256 is
`322668136b0d54e76d36051fd09d8020f9f30ad5c498583fad61d5ed90b22bfe`.
All 252 LoRA-B tensors contained nonzero trained values. The environment passed
`pip check`, and NVIDIA telemetry showed the run fit within the 16 GiB GPU
without an out-of-memory failure.

The PyTorch peak-allocation fields in the saved manifest are not usable as
physical VRAM measurements under this WSL configuration: its virtual allocator
reported values larger than the GPU. Use NVIDIA telemetry for capacity checks.

The corrected evaluator allows exactly one complete tool call per model turn,
matching the sequential controller contract. On one deterministic held-out
record from each of the 10 r1 families, results were:

| Model | Tool exposure | Exact structure | Schema execution | Finish | Forbidden |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen base | reference-record tools | 40% | 66.7% | 70% | 0% |
| r1 adapter | reference-record tools | 100% | 100% | 100% | 0% |
| Frozen base | full runtime registry | 10% | 76.0% | 30% | 0% |
| r1 adapter | full runtime registry | 30% | 88.1% | 90% | 0% |

The adapter therefore learned the target trajectories, but selecting among the
complete runtime registry remains the limiting problem. The fail-closed full
registry gate did **not** promote this adapter: it missed the required 50% exact
structure rate, 95% schema-execution rate, and 25% minimum per-family exact
rate. The manifest remains `production_ready: false`. The authoritative
artifacts are:

```text
training/evaluations/base-r1-record-pf1.json
training/evaluations/adapter-r1-record-pf1.json
training/evaluations/base-r1-full-pf1.json
training/evaluations/adapter-r1-full-pf1.json
training/evaluations/promotion-r1-full-pf1.json
training/evaluations/pilot_r1_training_manifest.json
training/evaluations/pilot_r1_trainer_state.json
```

Do not spend a larger promotion sample on r1: the one-per-family gate already
fails deterministic thresholds. The strongest next iteration is a versioned,
deterministic tool router in front of the model, evaluated in full-registry
mode; alternatively, retrain with the complete registry exposed in every
training prompt. Use the r2 corpus for that experiment and a 1,600-token limit
(its measured maximum is 1,531 tokens). Promotion must still be decided only by
the full-registry evaluator.

Training loss is not a promotion criterion. After training, use
`training/evaluations/evaluate_model.py` against the held-out test split for
both the frozen base model and adapter. Promotion requires executable tool
calls, zero forbidden actions, and no regression in exact final-structure rate.

Start with one deterministic example from each recipe family to validate the
generation/parser path. `--tool-mode record` is useful as an oracle-assisted
diagnostic because it exposes only tools used by the reference trajectory. It
cannot promote an adapter. The deployment gate uses `--tool-mode full` and
therefore tests selection from the complete runtime registry. Run the same
three-per-family sample for base and adapter:

```bash
COMMON="PYTHONPATH=src:. PATH=/usr/lib/wsl/lib:$PATH \
HF_HOME=training/cache/huggingface HF_HUB_DISABLE_XET=1"

env $COMMON .venv/bin/python training/evaluations/evaluate_model.py \
  training/datasets/pilot/test.jsonl --per-family 3 --tool-mode full \
  --output training/evaluations/base_r1_full_stratified.json

env $COMMON .venv/bin/python training/evaluations/evaluate_model.py \
  training/datasets/pilot/test.jsonl --per-family 3 --tool-mode full \
  --adapter training/runs/pilot-qwen3-4b-r1/adapter \
  --output training/evaluations/adapter_r1_full_stratified.json
```

Make the fail-closed decision from those reports:

```bash
PYTHONPATH=src:. .venv/bin/python training/evaluations/promote_adapter.py \
  --corpus-report training/evaluations/remote_pilot_corpus_report.json \
  --base-report training/evaluations/base_r1_full_stratified.json \
  --adapter-report training/evaluations/adapter_r1_full_stratified.json \
  --output training/evaluations/pilot_r1_promotion.json
```

On the configured WSL host, `training/run_pilot_evaluation_remote.sh 1 full`
runs the matching base/adapter one-per-family gate end to end. Increase the
first argument to `3` for the promotion-sized sample after the one-example path
has completed successfully.

The command exits nonzero when any gate fails. A saved adapter and low
validation loss do not make a run production-ready; retain
`production_ready: false` in the training manifest until a separate reviewed
integration task consumes a passing promotion report.

Production mode in `train_qlora.py` also replays all three corpus splits before
loading the base model. It requires the canonical `train.jsonl` and
`validation.jsonl` pair from one generated corpus directory and records the
replay evidence in the run manifest. Embedded per-record validation flags alone
are not trusted.
