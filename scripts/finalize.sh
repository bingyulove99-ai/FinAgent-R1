#!/usr/bin/env bash
# Wait for run_all_trainings_v2.sh to finish all evaluations, then run the
# baseline evaluations, aggregate every result, and emit per-algorithm summaries.
# Safe to start at any time: it never touches the GPUs while training or a
# per-algorithm eval is still using them.
set -uo pipefail
source /data/home.old/user/agent-r1/env.sh
BASE=/data/home.old/user/agent-r1
cd /home/user/Agent-R1-main
LOGDIR="$BASE/logs"
STAMP() { date '+%F %T'; }
PY="$BASE/envs/agentr1/bin/python"

ALGOS="grpo rloo reinforce ppo_gae token_gae"
MARKERS="grpo rloo reinforce ppo token_gae"   # v2 marker names

echo "[$(STAMP)] finalizer started"
while true; do
    n=0
    for a in $MARKERS; do [ -f "$LOGDIR/eval_done_${a}" ] && n=$((n + 1)); done
    if [ "$n" -eq 5 ]; then
        echo "[$(STAMP)] all five evaluations done"
        break
    fi
    if ! pgrep -f "run_all_trainings_v2" >/dev/null 2>&1; then
        # Runner gone: allow a grace period for a final in-flight eval, then stop.
        if ! pgrep -f "main_agent_ppo" >/dev/null 2>&1 && ! pgrep -f "infer_vllm" >/dev/null 2>&1; then
            echo "[$(STAMP)] runner not found and GPUs idle; finalizing with $n/5 evals done"
            break
        fi
    fi
    sleep 300
done

sleep 60
echo "[$(STAMP)] === baseline evaluations ==="
bash "$BASE/eval_model.sh" base_direct "$BASE/models/Qwen2.5-3B-Instruct" direct \
    > "$LOGDIR/eval_base_direct.log" 2>&1
echo "[$(STAMP)] base_direct exit=$?"
bash "$BASE/eval_model.sh" base_react "$BASE/models/Qwen2.5-3B-Instruct" agent \
    > "$LOGDIR/eval_base_react.log" 2>&1
echo "[$(STAMP)] base_react exit=$?"

echo "[$(STAMP)] === aggregation ==="
"$PY" "$BASE/aggregate_results.py" > "$LOGDIR/aggregate.log" 2>&1
echo "[$(STAMP)] aggregate exit=$?"

for a in $ALGOS; do
    "$PY" "$BASE/summarize_algo.py" "$a" > "$LOGDIR/summary_${a}.log" 2>&1
done
echo "[$(STAMP)] ALL FINALIZED"
touch "$LOGDIR/ALL_FINALIZED"
