#!/usr/bin/env bash
set -euo pipefail

# r6 evaluation: (1) seen-phrasing deployment gate + promotion on pilot_r6/test,
# (2) novel-phrasing generalization on the held-out phrasing set, comparing the
# frozen base, the promoted r5 adapter, and the new r6 adapter on identical
# deterministically-sampled records (limitation #1 probe).

project_root=/home/tlclab/Structure_building
cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root"
export PATH="/usr/lib/wsl/lib:$PATH"
export HF_HOME="$project_root/training/cache/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

py=.venv/bin/python
EV=training/evaluations
R6=training/runs/pilot-qwen3-4b-r6/adapter
R5=training/runs/pilot-qwen3-4b-r5/adapter
SEEN=training/datasets/pilot_r6/test.jsonl
NOVEL=training/datasets/pilot_r6_novel/heldout_eval.jsonl
NEG=training/datasets/adversarial/negative.jsonl
CORPUS=$EV/pilot_r6_corpus_report.json
PF="${PF:-6}"
SAMPLE="${SAMPLE:-120}"
tag="pf${PF}-s${SAMPLE}"
SAMP=(--per-family "$PF" --sample-size "$SAMPLE" --tool-mode full)

for p in "$R6/adapter_model.safetensors" "$R5/adapter_model.safetensors" "$SEEN" "$NOVEL" "$NEG" "$CORPUS"; do
    [[ -e "$p" ]] || { echo "required input missing: $p" >&2; exit 2; }
done

log=$EV/eval-r6-$tag.log
exec > >(tee -a "$log") 2>&1
echo "=== r6 evaluation start $(date -Is)  tag=$tag ==="

echo "[1/6] frozen base on SEEN (pilot_r6/test) + adversarial"
$py $EV/evaluate_model.py "$SEEN" "${SAMP[@]}" --negative "$NEG" \
    --output "$EV/base-r6-seen-$tag.json"

echo "[2/6] r6 adapter on SEEN + adversarial"
$py $EV/evaluate_model.py "$SEEN" "${SAMP[@]}" --negative "$NEG" --adapter "$R6" \
    --output "$EV/adapter-r6-seen-$tag.json"

echo "[3/6] promotion gate (seen phrasing, deployment)"
set +e
$py $EV/promote_adapter.py \
    --corpus-report "$CORPUS" \
    --base-report "$EV/base-r6-seen-$tag.json" \
    --adapter-report "$EV/adapter-r6-seen-$tag.json" \
    --output "$EV/promotion-r6-seen-$tag.json"
echo "promotion gate exit status: $?"
set -e

echo "[4/6] frozen base on NOVEL phrasing (held-out)"
$py $EV/evaluate_model.py "$NOVEL" "${SAMP[@]}" \
    --output "$EV/base-r6-novel-$tag.json"

echo "[5/6] promoted r5 adapter on NOVEL phrasing"
$py $EV/evaluate_model.py "$NOVEL" "${SAMP[@]}" --adapter "$R5" \
    --output "$EV/adapter-r5-novel-$tag.json"

echo "[6/6] r6 adapter on NOVEL phrasing"
$py $EV/evaluate_model.py "$NOVEL" "${SAMP[@]}" --adapter "$R6" \
    --output "$EV/adapter-r6-novel-$tag.json"

echo "=== tabulating $(date -Is) ==="
$py $EV/tabulate_r6.py "$tag"
echo "=== r6 evaluation done $(date -Is) ==="
