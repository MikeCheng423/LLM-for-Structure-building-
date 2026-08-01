#!/usr/bin/env bash
# Sequential driver for the journal-role r2 cycle: train, evaluate + promote
# against the main test set, then evaluate against the section 10.3 held-out
# sets. Each stage is a separate script so a failure at any point stops the
# whole cycle (set -e) rather than silently continuing on stale results.
set -euo pipefail

worktree_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
primary_root=/home/tlclab/Structure_building
run_name=pilot-qwen3-4b-journal-role-r2
candidate_adapter="$primary_root/training/runs/$run_name/adapter"

echo "==================================================================="
echo "[1/3] Training: run_journal_role_r2.sh"
echo "==================================================================="
bash "$worktree_root/training/run_journal_role_r2.sh"

echo "==================================================================="
echo "[2/3] Main-set evaluation + promotion: run_journal_role_evaluation.sh"
echo "==================================================================="
CANDIDATE_ADAPTER="$candidate_adapter" \
REPORT_PREFIX=journal-role-r2 \
    bash "$worktree_root/training/run_journal_role_evaluation.sh"

echo "==================================================================="
echo "[3/3] Held-out evaluation: run_journal_holdout_evaluation.sh"
echo "==================================================================="
ADAPTER="$candidate_adapter" \
REPORT_PREFIX=journal-role-r2 \
    bash "$worktree_root/training/run_journal_holdout_evaluation.sh"

echo "==================================================================="
echo "journal-role r2 full cycle complete"
echo "==================================================================="
