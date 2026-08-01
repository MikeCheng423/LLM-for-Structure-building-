#!/usr/bin/env bash
set -euo pipefail

project_root=/home/tlclab/Structure_building
revision=${1:-r2}
dataset_revision=${2:-$revision}
[[ "$revision" =~ ^r[0-9]+$ ]] || {
    echo "Revision must look like r2, got: $revision" >&2
    exit 2
}
[[ "$dataset_revision" =~ ^r[0-9]+$ ]] || {
    echo "Dataset revision must look like r2, got: $dataset_revision" >&2
    exit 2
}
dataset="training/datasets/catalyst_supplement_${dataset_revision}/test.jsonl"
adapter="training/runs/pilot-qwen3-4b-catalyst-supplement-${revision}/adapter"
prefix="training/evaluations/catalyst-supplement-${revision}-full-pf2"
baseline_prefix="training/evaluations/catalyst-supplement-${dataset_revision}-full-pf2"
corpus_report="training/evaluations/catalyst_supplement_${dataset_revision}_corpus_report.json"
negative="training/datasets/adversarial/negative.jsonl"
base_report="${prefix}-base.json"
r5_report="${prefix}-r5.json"
if [[ "$dataset_revision" != "$revision" ]]; then
    base_report="${baseline_prefix}-base.json"
    r5_report="${baseline_prefix}-r5.json"
fi

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

if [[ "$dataset_revision" == "$revision" ]]; then
    .venv/bin/python training/evaluations/evaluate_model.py \
        "$dataset" --per-family 2 --tool-mode full \
        --negative "$negative" \
        --output "$base_report"
    .venv/bin/python training/evaluations/evaluate_model.py \
        "$dataset" --per-family 2 --tool-mode full \
        --negative "$negative" \
        --adapter training/runs/pilot-qwen3-4b-r5/adapter \
        --output "$r5_report"
else
    [[ -f "$base_report" && -f "$r5_report" ]] || {
        echo "Reusable baseline reports are missing for $dataset_revision" >&2
        exit 2
    }
fi
.venv/bin/python training/evaluations/evaluate_model.py \
    "$dataset" --per-family 2 --tool-mode full \
    --negative "$negative" \
    --adapter "$adapter" \
    --output "${prefix}-adapter.json"

# Compare to promoted r5, not only the weak frozen base. A rejected promotion is
# a successful fail-closed evaluation outcome and intentionally exits nonzero.
.venv/bin/python training/evaluations/promote_adapter.py \
    --corpus-report "$corpus_report" \
    --base-report "$r5_report" \
    --adapter-report "${prefix}-adapter.json" \
    --minimum-exact-rate 0.95 \
    --minimum-execution-rate 0.98 \
    --minimum-finish-rate 0.98 \
    --minimum-family-exact-rate 0.90 \
    --minimum-invariant-rate 0.95 \
    --minimum-family-invariant-rate 0.90 \
    --output "${prefix}-promotion.json"

# A passing two-per-family screen earns a larger matched regression gate. The
# shell exits above on rejection, so weak adapters do not consume this GPU time.
wide_prefix="training/evaluations/catalyst-supplement-${revision}-full-mf2-s120"
.venv/bin/python training/evaluations/evaluate_model.py \
    "$dataset" --min-per-family 2 --sample-size 120 --tool-mode full \
    --negative "$negative" \
    --adapter training/runs/pilot-qwen3-4b-r5/adapter \
    --output "${wide_prefix}-r5.json"
.venv/bin/python training/evaluations/evaluate_model.py \
    "$dataset" --min-per-family 2 --sample-size 120 --tool-mode full \
    --negative "$negative" \
    --adapter "$adapter" \
    --output "${wide_prefix}-adapter.json"
.venv/bin/python training/evaluations/promote_adapter.py \
    --corpus-report "$corpus_report" \
    --base-report "${wide_prefix}-r5.json" \
    --adapter-report "${wide_prefix}-adapter.json" \
    --minimum-exact-rate 0.95 \
    --minimum-execution-rate 0.98 \
    --minimum-finish-rate 0.98 \
    --minimum-family-exact-rate 0.90 \
    --minimum-invariant-rate 0.95 \
    --minimum-family-invariant-rate 0.90 \
    --output "${wide_prefix}-promotion.json"
