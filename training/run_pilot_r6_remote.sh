#!/usr/bin/env bash
set -euo pipefail

project_root=/home/tlclab/Structure_building
run_name=pilot-qwen3-4b-r6
output_dir="$project_root/training/runs/$run_name"
log_file="$project_root/training/runs/$run_name.log"

# Overridable knobs. r6: richer phrasing corpus (pilot_r6), more steps than r5
# (r5 val loss was still easing at step 200). max_length keeps headroom over the
# measured 1,547-token maximum.
max_length="${MAXLEN:-1792}"
max_steps="${MAXSTEPS:-300}"

cd "$project_root"
mkdir -p training/runs
if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty run: $output_dir" >&2
    exit 2
fi

export PYTHONPATH="$project_root/src:$project_root"
export PATH="/usr/lib/wsl/lib:$PATH"
export HF_HOME="$project_root/training/cache/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

exec > >(tee -a "$log_file") 2>&1

echo "Starting $run_name at $(date --iso-8601=seconds) (max_length=$max_length max_steps=$max_steps)"
.venv/bin/python training/train_qlora.py \
    --dataset training/datasets/pilot_r6/train.jsonl \
    --eval-dataset training/datasets/pilot_r6/validation.jsonl \
    --output-dir "training/runs/$run_name" \
    --max-length "$max_length" \
    --max-steps "$max_steps" \
    --gradient-accumulation 8 \
    --eval-steps 25 \
    --minimum-records 1000
echo "Finished $run_name at $(date --iso-8601=seconds)"
