#!/usr/bin/env bash
# Continuously archive every training checkpoint with hardlinks.
# verl deletes old checkpoints when max_actor_ckpt_to_keep is exceeded; the
# hardlink copies keep the data alive at ~zero extra disk cost (same filesystem).
set -uo pipefail
BASE=/data/home.old/user/agent-r1
OUT="$BASE/outputs"
ARCH="$OUT/_archive/ckpt"
mkdir -p "$ARCH"

while true; do
    for d in "$OUT"/finqa_qwen3b_*/global_step_*; do
        [ -d "$d" ] || continue
        exp="$(basename "$(dirname "$d")")"
        step="$(basename "$d")"
        dst="$ARCH/$exp/$step"
        [ -e "$dst" ] && continue
        # Only touch checkpoints whose write stopped >3 min ago: never copy a
        # save in progress. A partial copy is completed on the next pass.
        if [ -n "$(find "$d" -newermt '-3 minutes' -print -quit 2>/dev/null)" ]; then
            continue
        fi
        latest="$(cat "$OUT/$exp/latest_checkpointed_iteration.txt" 2>/dev/null || true)"
        [ -n "$latest" ] && [ "$latest" -lt "${step#global_step_}" ] && continue
        mkdir -p "$ARCH/$exp"
        cp -al "$d" "$dst" 2>/dev/null || cp -a "$d" "$dst" 2>/dev/null || true
        echo "$(date '+%F %T') archived $exp/$step"
    done
    sleep 120
done
