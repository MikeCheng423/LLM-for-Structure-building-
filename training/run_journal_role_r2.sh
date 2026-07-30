#!/usr/bin/env bash
set -euo pipefail

worktree_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
primary_root=/home/tlclab/Structure_building
run_name=pilot-qwen3-4b-journal-role-r2
output_dir="$primary_root/training/runs/$run_name"
log_file="$primary_root/training/runs/$run_name.log"

cd "$worktree_root"
mkdir -p "$primary_root/training/runs"
if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty run: $output_dir" >&2
    exit 2
fi

export PYTHONPATH="$worktree_root/src:$worktree_root"
export PATH="/usr/lib/wsl/lib:$PATH"
export HF_HOME="$primary_root/training/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

exec > >(tee -a "$log_file") 2>&1
echo "Starting $run_name at $(date --iso-8601=seconds)"
"$primary_root"/.venv/bin/python training/train_qlora.py \
    --dataset training/datasets/journal_roles_v1/train.jsonl \
    --eval-dataset training/datasets/journal_roles_v1/validation.jsonl \
    --output-dir "$output_dir" \
    --initial-adapter "$primary_root/training/runs/pilot-qwen3-4b-r5/adapter" \
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
