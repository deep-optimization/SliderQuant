#!/usr/bin/env bash
source "$(dirname "$0")/preflight.sh"

RUN_DIR="$SCALEQ_ROOT/evaluation/baseline-$SCALEQ_RUN_ID"
if complete_for_current_source "$RUN_DIR"; then
    exit 0
fi
mkdir -p "$RUN_DIR"

cd "$REPO_DIR"
"$TORCHRUN_BIN" --standalone --nproc_per_node="$SCALEQ_GPU_COUNT" scripts/eval/scaleq_eval.py \
    --model "$MODEL_DIR" \
    --model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --output-dir "$RUN_DIR" \
    --batch-size "$SCALEQ_EVAL_BATCH_SIZE"

if ! test -e "$RUN_DIR/EVALPLUS_COMMANDS.txt"; then
cat >"$RUN_DIR/EVALPLUS_COMMANDS.txt" <<EOF
evalplus.evaluate --dataset humaneval --samples $RUN_DIR/humaneval/evalplus-samples.jsonl
evalplus.evaluate --dataset mbpp --samples $RUN_DIR/mbpp/evalplus-samples.jsonl
EOF
fi
mark_complete "$RUN_DIR"
