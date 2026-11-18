#!/usr/bin/env bash
# Compact pipeline status.
B=/data/home.old/user/agent-r1
for a in grpo rloo reinforce ppo token_gae; do
    f="$B/logs/train_$a.log"
    [ -f "$f" ] || continue
    prog=$(grep -a "Training Progress" "$f" 2>/dev/null | tail -1 | sed 's/\x1b\[[0-9;]*m//g' | grep -oE "[0-9]+/195|[0-9]+%")
    dev=$(grep -a "val-core" "$f" 2>/dev/null | tail -1 | sed 's/\x1b\[[0-9;]*m//g' | grep -oE "val-core/finqa/acc/mean@1:[-0-9.]+" | tail -1 | cut -d: -f2)
    echo "$a: ${prog:-n/a} dev=${dev:-n/a}"
done
if [ -f "$B/results/ppo_gae_all.metrics.json" ] 2>/dev/null; then :; fi
for t in grpo rloo reinforce ppo_gae token_gae; do
    m="$B/results/${t}_all.metrics.json"
    [ -f "$m" ] && echo "test $t: acc=$(python3 -c "import json;print(json.load(open('$m'))['summary']['accuracy'])" 2>/dev/null)"
done
tail -1 "$B/logs/runner_v2.log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g'
[ -f "$B/logs/ALL_FINALIZED" ] && echo "ALL_FINALIZED"
