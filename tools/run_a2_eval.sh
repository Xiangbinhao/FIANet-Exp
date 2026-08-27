#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage:"
    echo "  bash tools/run_a2_eval.sh <TAG> <CHECKPOINT> [EXTRA_ARGS...]"
    exit 1
fi

TAG="$1"
CHECKPOINT="$2"
shift 2

EXTRA_ARGS=("$@")

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found:"
    echo "  $CHECKPOINT"
    exit 2
fi

OUTPUT_DIR="experiments/A2/${TAG}"
LOG_PATH="logs/a2_${TAG}_full.log"

mkdir -p logs
rm -rf "$OUTPUT_DIR"
rm -f "$LOG_PATH"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true

CUDA_VISIBLE_DEVICES=0 nohup python -u \
tools/s0_size_eval_a2.py \
  --dataset rrsisd \
  --model lavt_one \
  --split test \
  --img_size 480 \
  --num_tmem 3 \
  --workers 2 \
  --swin_type base \
  --window12 \
  --resume "$CHECKPOINT" \
  --bert_tokenizer ./bert-base-uncased \
  --ck_bert ./bert-base-uncased \
  --refer_data_root /home/ubuntu/data/RRSIS-D \
  --s0-tag "${TAG}_A2" \
  --s0-output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}" \
  > "$LOG_PATH" 2>&1 &

PID=$!

echo "Started A2 evaluation"
echo "Tag:        $TAG"
echo "Checkpoint: $CHECKPOINT"
echo "PID:        $PID"
echo "Log:        $LOG_PATH"
