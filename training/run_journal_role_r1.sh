#!/usr/bin/env bash
set -euo pipefail

project_root=/home/tlclab/Structure_building
run_name=pilot-qwen3-4b-journal-role-r1
output_dir="$project_root/training/runs/$run_name"
log_file="$project_root/training/runs/$run_name.log"

cd "$project_root"
mkdir -p training/runs
if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty run: $output_dir" >&2
    exit 2
fi

export PYTHONPATH="$project_root/src:$project_root"
export PATH="/usr/lib/wsl/lib:$PATH"
export HF_HOME="$project_root/training/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

exec > >(tee -a "$log_file") 2>&1
echo "Starting $run_name at $(date --iso-8601=seconds)"
.venv/bin/python training/train_qlora.py \
    --dataset training/datasets/journal_roles_v1/train.jsonl \
    --eval-dataset training/datasets/journal_roles_v1/validation.jsonl \
    --output-dir "training/runs/$run_name" \
    --initial-adapter training/runs/pilot-qwen3-4b-r5/adapter \
    --corpus-contract journal \
    --max-length 2048 \
    --max-steps 200 \
    --gradient-accumulation 8 \
    --eval-steps 25 \
    --learning-rate 1e-4 \
    --lora-rank 16 \
    --seed 42 \
    --minimum-records 3000
echo "Finished $run_name at $(date --iso-8601=seconds)"
