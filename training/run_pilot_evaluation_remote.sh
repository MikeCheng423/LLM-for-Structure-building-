#!/usr/bin/env bash
set -euo pipefail

project_root=/home/tlclab/Structure_building
per_family="${1:-1}"
tool_mode="${2:-full}"
sample_size="${3:-0}"

if [[ ! "$per_family" =~ ^[1-9][0-9]*$ ]]; then
    echo "per-family count must be a positive integer" >&2
    exit 2
fi
if [[ "$tool_mode" != "record" && "$tool_mode" != "full" ]]; then
    echo "tool mode must be record or full" >&2
    exit 2
fi
if [[ ! "$sample_size" =~ ^[0-9]+$ ]]; then
    echo "sample size must be a non-negative integer (0 disables the fill)" >&2
    exit 2
fi

cd "$project_root"
# Target run is parametrized; defaults preserve the original r1 behaviour.
run_tag="${RUN_TAG:-r1}"
adapter="${ADAPTER_DIR:-training/runs/pilot-qwen3-4b-r1/adapter}"
dataset_dir="${DATASET_DIR:-training/datasets/pilot}"
corpus_report="${CORPUS_REPORT:-training/evaluations/remote_pilot_corpus_report.json}"
if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
    echo "pilot adapter is missing: $adapter" >&2
    exit 2
fi
if [[ ! -f "$dataset_dir/test.jsonl" ]]; then
    echo "test split is missing: $dataset_dir/test.jsonl" >&2
    exit 2
fi
if [[ ! -f "$corpus_report" ]]; then
    echo "corpus report is missing: $corpus_report" >&2
    exit 2
fi

export PYTHONPATH="$project_root/src:$project_root"
export PATH="/usr/lib/wsl/lib:$PATH"
export HF_HOME="$project_root/training/cache/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

suffix="${run_tag}-${tool_mode}-pf${per_family}"
if [[ "$sample_size" -gt 0 ]]; then
    suffix="${suffix}-s${sample_size}"
fi
base_report="training/evaluations/base-${suffix}.json"
adapter_report="training/evaluations/adapter-${suffix}.json"

# Coverage floor per family plus a larger deterministic held-out fill, and the
# adversarial safety pass, are only meaningful for the deployment (full) gate.
sampling_args=(--per-family "$per_family")
if [[ "$sample_size" -gt 0 ]]; then
    sampling_args+=(--sample-size "$sample_size")
fi
negative_args=()
negative_set=training/datasets/adversarial/negative.jsonl
if [[ "$tool_mode" == "full" && -f "$negative_set" ]]; then
    negative_args=(--negative "$negative_set")
fi

.venv/bin/python training/evaluations/evaluate_model.py \
    "$dataset_dir/test.jsonl" \
    "${sampling_args[@]}" \
    --tool-mode "$tool_mode" \
    "${negative_args[@]}" \
    --output "$base_report"

.venv/bin/python training/evaluations/evaluate_model.py \
    "$dataset_dir/test.jsonl" \
    "${sampling_args[@]}" \
    --tool-mode "$tool_mode" \
    "${negative_args[@]}" \
    --adapter "$adapter" \
    --output "$adapter_report"

if [[ "$tool_mode" == "full" ]]; then
    set +e
    .venv/bin/python training/evaluations/promote_adapter.py \
        --corpus-report "$corpus_report" \
        --base-report "$base_report" \
        --adapter-report "$adapter_report" \
        --output "training/evaluations/promotion-${suffix}.json"
    promotion_status=$?
    set -e
    echo "Promotion gate exit status: $promotion_status"
    exit "$promotion_status"
fi
