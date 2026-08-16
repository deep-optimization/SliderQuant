#!/usr/bin/env bash
source "$(dirname "$0")/preflight.sh"

RUN_DIR="$SCALEQ_ROOT/runs/smoke-$SCALEQ_RUN_ID"
if complete_for_current_source "$RUN_DIR"; then
    exit 0
fi
require_complete "$AYOT_DIR"
mkdir -p "$RUN_DIR"
resume_args=()
if test -f "$RUN_DIR/training_state.pt"; then
    resume_args=(--train_resume "$RUN_DIR/training_state.pt")
fi

cd "$REPO_DIR"
"$TORCHRUN_BIN" --standalone --nproc_per_node="$SCALEQ_GPU_COUNT" main.py \
    --config configs/scaleq-qwen3-1p7b/config.yaml \
    --model "$MODEL_DIR" \
    --calib_manifest "$AYOT_DIR/manifest.json" \
    --calib_subset "$AYOT_DIR/subset-32.json" \
    --nsamples 32 \
    --epochs 2 \
    --output_dir "$RUN_DIR" \
    --cache_dir "$CACHE_DIR" \
    "${resume_args[@]}" \
    --use_ddp

mark_complete "$RUN_DIR"
