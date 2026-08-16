#!/usr/bin/env bash
# Stage eval: merge the trained factors, export the fake-quantized model, then
# generate all five ScaleQ benchmark outputs.
source "$(dirname "$0")/preflight.sh"

FULL_DIR="$SCALEQ_ROOT/runs/full-$SCALEQ_RUN_ID"
RUN_DIR="$SCALEQ_ROOT/evaluation/scaleq-$SCALEQ_RUN_ID"
MERGED_ROOT="$SCALEQ_ROOT/merged-model/$SCALEQ_RUN_ID"
MERGED_MODEL="$MERGED_ROOT/qwen3-SliderQuant"
MERGE_PROVENANCE="$MERGED_MODEL/SCALEQ_MERGE_PROVENANCE"
if complete_for_current_source "$RUN_DIR"; then
    exit 0
fi
require_complete "$FULL_DIR"
mkdir -p "$RUN_DIR"

cd "$REPO_DIR"
checkpoint_sha="$(sha256sum "$FULL_DIR/slider_parameters.pth" | cut -d' ' -f1)"
if test -e "$MERGED_MODEL"; then
    test -f "$MERGED_MODEL/config.json"
    test -f "$MERGED_MODEL/model.safetensors"
    read -r merged_source merged_code merged_checkpoint <"$MERGE_PROVENANCE"
    test "$merged_source" = "$SCALEQ_SHA"
    test "$merged_code" = "$SCALEQ_CODE_SHA"
    test "$merged_checkpoint" = "$checkpoint_sha"
else
    if ! test -e "$FULL_DIR/parameters.pth"; then
        "$PYTHON_BIN" scripts/export_bittern_checkpoint.py \
            --input "$FULL_DIR/slider_parameters.pth" \
            --output "$FULL_DIR/parameters.pth" \
            --require-lora
    fi

    "$PYTHON_BIN" main.py \
        --config configs/scaleq-qwen3-1p7b/config.yaml \
        --model "$MODEL_DIR" \
        --calib_manifest "$AYOT_DIR/manifest.json" \
        --resume "$FULL_DIR/slider_parameters.pth" \
        --nsamples 1 \
        --output_dir "$RUN_DIR" \
        --cache_dir "$CACHE_DIR" \
        --export_model_path "$MERGED_ROOT" \
        --test_mode --weight_merge --no-use_ddp
    test ! -e "$MERGE_PROVENANCE"
    printf '%s %s %s\n' \
        "$SCALEQ_SHA" "$SCALEQ_CODE_SHA" "$checkpoint_sha" \
        >"$MERGE_PROVENANCE"
fi

"$TORCHRUN_BIN" --standalone --nproc_per_node="$SCALEQ_GPU_COUNT" scripts/eval/scaleq_eval.py \
    --model "$MERGED_MODEL" \
    --output-dir "$RUN_DIR" \
    --batch-size "$SCALEQ_EVAL_BATCH_SIZE"

if ! test -e "$RUN_DIR/EVALPLUS_COMMANDS.txt"; then
cat >"$RUN_DIR/EVALPLUS_COMMANDS.txt" <<EOF
evalplus.evaluate --dataset humaneval --samples $RUN_DIR/humaneval/evalplus-samples.jsonl
evalplus.evaluate --dataset mbpp --samples $RUN_DIR/mbpp/evalplus-samples.jsonl
EOF
fi
mark_complete "$RUN_DIR"
