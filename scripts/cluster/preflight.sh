#!/usr/bin/env bash
# Shared setup for every ScaleQ stage. Source it from a stage launcher for
# validated paths and completion-marker helpers.
set -euo pipefail

: "${SCALEQ_SHA:?set by cluster/render_jobs.py}"
: "${SCALEQ_CODE_SHA:?set by cluster/render_jobs.py}"
: "${SCALEQ_ROOT:?set by cluster/render_jobs.py}"
: "${SCALEQ_STAGE:?set by cluster/render_jobs.py}"
: "${SCALEQ_RUN_ID:?set by cluster/render_jobs.py}"
: "${SCALEQ_GPU_COUNT:?set by cluster/render_jobs.py}"
: "${SCALEQ_EVAL_BATCH_SIZE:?set by cluster/render_jobs.py}"
: "${SCALEQ_MODEL_DIR:?set by cluster/render_jobs.py}"
: "${SCALEQ_AYOT_DIR:?set by cluster/render_jobs.py}"

# The renderer clones the pinned commit and runs these launchers from its
# detached worktree. Node-local caches stay next to that worktree.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_DIR="$SCALEQ_ROOT/cache/scaleq"
PYTHON_BIN="$(command -v python)"
TORCHRUN_BIN="$(command -v torchrun)"
MODEL_DIR="$SCALEQ_MODEL_DIR"
AYOT_DIR="$SCALEQ_AYOT_DIR"
export HF_HOME="$SCALEQ_ROOT/cache/huggingface"

test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$SCALEQ_CODE_SHA"
test -x "$PYTHON_BIN"
mkdir -p "$CACHE_DIR" "$HF_HOME" "$SCALEQ_ROOT/models" "$SCALEQ_ROOT/environment/$SCALEQ_CODE_SHA"
"$PYTHON_BIN" "$REPO_DIR/scripts/cluster/prepare_model.py" "$MODEL_DIR"
test -f "$MODEL_DIR/config.json"
"$PYTHON_BIN" -c 'import os, torch, transformers
print(torch.__version__, transformers.__version__, torch.cuda.device_count())
assert torch.cuda.device_count() == int(os.environ["SCALEQ_GPU_COUNT"])'
PIP_FREEZE="$SCALEQ_ROOT/environment/$SCALEQ_CODE_SHA/$SCALEQ_STAGE-pip-freeze.txt"
NVIDIA_SMI="$SCALEQ_ROOT/environment/$SCALEQ_CODE_SHA/$SCALEQ_STAGE-nvidia-smi-q.txt"
if ! test -e "$PIP_FREEZE"; then
    "$PYTHON_BIN" -m pip freeze >"$PIP_FREEZE"
fi
if ! test -e "$NVIDIA_SMI"; then
    nvidia-smi -q >"$NVIDIA_SMI"
fi

complete_for_current_source() {
    local marker="$1/COMPLETE"
    local completed_sha
    test -f "$marker" || return 1
    read -r completed_sha _ <"$marker"
    test "$completed_sha" = "$SCALEQ_SHA"
}

require_complete() {
    complete_for_current_source "$1" || {
        printf 'missing COMPLETE marker for source %s in %s\n' "$SCALEQ_SHA" "$1" >&2
        return 1
    }
}

mark_complete() {
    local marker="$1/COMPLETE"
    if test -e "$marker"; then
        complete_for_current_source "$1"
        return
    fi
    printf '%s %s\n' "$SCALEQ_SHA" "$(date -u +%FT%TZ)" >"$marker"
}
