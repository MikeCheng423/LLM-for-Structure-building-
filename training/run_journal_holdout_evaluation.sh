#!/usr/bin/env bash
# Evaluate the journal-role adapter on the section 10.3 held-out sets.
#
# `journal_holdout` is not included: it needs complete unseen papers, and this
# repository contains no licensed journal text.
set -euo pipefail

worktree_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
primary_root=/home/tlclab/Structure_building
cd "$worktree_root"
export PYTHONPATH="$worktree_root/src:$worktree_root"
export PATH="/usr/lib/wsl/lib:$PATH"          # WSL keeps nvidia-smi here, not on the default PATH
export HF_HOME="$primary_root/training/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

adapter="${ADAPTER:-$primary_root/training/runs/pilot-qwen3-4b-journal-role-r1/adapter}"
sample="${SAMPLE_SIZE:-100}"
report_prefix="${REPORT_PREFIX:-journal-role-r1}"

for set_name in template family prose; do
    # prose_holdout is the Phase 2 regression test (JOURNAL_ROLE_R3_SCHEDULE.md
    # section 3). It holds only 104 records, and Phase 1 measured r2 on 80 of
    # them (40 per role); the r3 comparison is only valid against an identical
    # sample, so this set keeps its own size rather than the shared default.
    set_sample="$sample"
    if [[ "$set_name" == prose ]]; then
        set_sample="${PROSE_SAMPLE_SIZE:-80}"
    fi
    "$primary_root"/.venv/bin/python training/evaluations/evaluate_journal_model.py \
        "training/datasets/journal_holdout_${set_name}/test.jsonl" \
        --sample-size "$set_sample" \
        --max-new-tokens 900 \
        --cache-dir "$primary_root/training/cache/huggingface" \
        --adapter "$adapter" \
        --output "training/evaluations/${report_prefix}-${set_name}-holdout.json"
done
