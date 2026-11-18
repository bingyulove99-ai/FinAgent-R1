#!/usr/bin/env bash
# Emit results/<tag>_summary.md as soon as each algorithm's test evaluation is done.
set -uo pipefail
BASE=/data/home.old/user/agent-r1
PY="$BASE/envs/agentr1/bin/python"
while true; do
    for a in grpo rloo reinforce ppo token_gae; do
        tag="$a"
        [ "$a" = "ppo" ] && tag="ppo_gae"
        if [ -f "$BASE/logs/eval_done_${a}" ] && [ ! -f "$BASE/results/${tag}_summary.md" ]; then
            "$PY" "$BASE/summarize_algo.py" "$tag" > "$BASE/logs/summary_${tag}.log" 2>&1 \
                && echo "$(date '+%F %T') summarized $tag"
        fi
    done
    sleep 120
done
