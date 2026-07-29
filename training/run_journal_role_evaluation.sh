#!/usr/bin/env bash
set -euo pipefail

project_root=/home/tlclab/Structure_building
cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root"
export PATH="/usr/lib/wsl/lib:$PATH"
export HF_HOME="$project_root/training/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

common=(
    training/evaluations/evaluate_journal_model.py
    training/datasets/journal_roles_v1/test.jsonl
    --sample-size 300
    --max-new-tokens 900
)
.venv/bin/python "${common[@]}" \
    --adapter training/runs/pilot-qwen3-4b-r5/adapter \
    --output training/evaluations/journal-role-r1-baseline.json
.venv/bin/python "${common[@]}" \
    --adapter training/runs/pilot-qwen3-4b-journal-role-r1/adapter \
    --output training/evaluations/journal-role-r1-adapter.json
.venv/bin/python training/evaluations/promote_journal_adapter.py \
    --baseline training/evaluations/journal-role-r1-baseline.json \
    --adapter training/evaluations/journal-role-r1-adapter.json \
    --output training/evaluations/journal-role-r1-promotion.json \
    --manifest training/runs/pilot-qwen3-4b-journal-role-r1/manifest.json

.venv/bin/python -m ase_auto_build.ase_agent.catalyst_cli \
    --input training/evaluations/journal-agent-r1-smoke-input.json \
    --out training/evaluations/journal-agent-r1-smoke-output \
    --adapter training/runs/pilot-qwen3-4b-journal-role-r1/adapter \
    --max-new-tokens 1200
