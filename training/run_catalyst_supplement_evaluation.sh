#!/usr/bin/env bash
set -euo pipefail

project_root=/home/tlclab/Structure_building
dataset=training/datasets/catalyst_supplement_r1/test.jsonl
adapter=training/runs/pilot-qwen3-4b-catalyst-supplement-r1/adapter
prefix=training/evaluations/catalyst-supplement-r1-full-pf2

cd "$project_root"
[[ -f "$adapter/adapter_model.safetensors" ]] || {
    echo "Training adapter is not ready: $adapter" >&2
    exit 2
}

export PYTHONPATH="$project_root/src:$project_root"
export PATH="/usr/lib/wsl/lib:$PATH"
export HF_HOME="$project_root/training/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

.venv/bin/python training/evaluations/evaluate_model.py \
    "$dataset" --per-family 2 --tool-mode full \
    --output "${prefix}-base.json"
.venv/bin/python training/evaluations/evaluate_model.py \
    "$dataset" --per-family 2 --tool-mode full \
    --adapter training/runs/pilot-qwen3-4b-r5/adapter \
    --output "${prefix}-r5.json"
.venv/bin/python training/evaluations/evaluate_model.py \
    "$dataset" --per-family 2 --tool-mode full \
    --adapter "$adapter" \
    --output "${prefix}-adapter.json"

# Compare to promoted r5, not only the weak frozen base. A rejected promotion is
# a successful fail-closed evaluation outcome and intentionally exits nonzero.
.venv/bin/python training/evaluations/promote_adapter.py \
    --corpus-report training/evaluations/catalyst_supplement_r1_corpus_report.json \
    --base-report "${prefix}-r5.json" \
    --adapter-report "${prefix}-adapter.json" \
    --minimum-exact-rate 0.95 \
    --minimum-execution-rate 0.98 \
    --minimum-family-exact-rate 0.90 \
    --minimum-invariant-rate 0.95 \
    --minimum-family-invariant-rate 0.90 \
    --output "${prefix}-promotion.json"
