#!/usr/bin/env bash
# Build the immutable 2,048-row calibration artifact.
source "$(dirname "$0")/preflight.sh"

RUN_DIR="$AYOT_DIR"
if complete_for_current_source "$RUN_DIR"; then
    exit 0
fi
mkdir -p "$RUN_DIR"

cd "$REPO_DIR"
"$TORCHRUN_BIN" --standalone --nproc_per_node="$SCALEQ_GPU_COUNT" scripts/build_ayot_calibration.py \
    --model "$MODEL_DIR" \
    --model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --out-root "$SCALEQ_ROOT/ayot" \
    --version "$SCALEQ_RUN_ID" \
    --seed 2

mark_complete "$RUN_DIR"
