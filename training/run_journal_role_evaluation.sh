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

# The deterministic slice must still pass before any model result is believed.
.venv/bin/python training/evaluations/run_vertical_slice_gate.py \
    --report training/evaluations/vertical_slice_gate.json

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

# The live smoke runs BEFORE promotion and is an input to it. r1 scored 100%
# schema-valid offline and still failed its first free-running request, so an
# offline report alone must not be able to flip journal_role_ready.
smoke_root="training/evaluations/journal-agent-r1-smoke-output/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$smoke_root"
set +e
.venv/bin/python -m ase_auto_build.ase_agent.catalyst_cli \
    --input training/evaluations/journal-agent-r1-smoke-input.json \
    --out "$smoke_root" \
    --adapter training/runs/pilot-qwen3-4b-journal-role-r1/adapter \
    --allow-unpromoted \
    --max-new-tokens 1200
smoke_status=$?
set -e
echo "live smoke exit status: ${smoke_status} (package: ${smoke_root})"

# Nonzero here means the gate refused; that is the intended fail-closed outcome.
.venv/bin/python training/evaluations/promote_journal_adapter.py \
    --baseline training/evaluations/journal-role-r1-baseline.json \
    --adapter training/evaluations/journal-role-r1-adapter.json \
    --output training/evaluations/journal-role-r1-promotion.json \
    --manifest training/runs/pilot-qwen3-4b-journal-role-r1/manifest.json \
    --smoke-dir "$smoke_root"
