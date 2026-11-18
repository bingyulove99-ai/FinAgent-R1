#!/usr/bin/env bash
# Merge a verl FSDP actor checkpoint into a Hugging Face model directory.
#
# verl saves sharded FSDP checkpoints under:
#   outputs/<exp>/global_step_N/actor/
# The merger below writes a standard HF directory that vLLM can load.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
if ! "$PYTHON_BIN" -c "import sys" >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON:-python}"
fi

CKPT="${1:?Usage: merge_checkpoint.sh <global_step_N/actor> [target_dir]}"
TARGET="${2:-$(dirname "$CKPT")/huggingface}"

"$PYTHON_BIN" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$CKPT" \
    --target_dir "$TARGET"

echo "Merged checkpoint: $TARGET"
echo "Evaluate with: MODEL=$TARGET bash $ROOT_DIR/projects/finagent_r1/scripts/evaluate.sh"
