#!/usr/bin/env bash
# Evaluate one model on the full test split (1,147 questions), sharded across
# GPU 0 and GPU 3, then score all / seen_docs / unseen_docs from the same
# predictions (no re-inference needed for subsets).
#
# Robustness: clears leftover Ray/vLLM processes from a just-finished training
# run before starting, and retries any shard that failed or did not finish
# (concurrent vLLM engines on one GPU can fail KV-cache allocation) -- one
# shard at a time.
#
# Usage: bash eval_model.sh <tag> <model_path> [agent|direct]
set -uo pipefail
source /data/home.old/user/agent-r1/env.sh
cd /home/user/Agent-R1-main

TAG="${1:?usage: eval_model.sh <tag> <model_path> [agent|direct]}"
MODEL="${2:?model path required}"
MODE="${3:-agent}"
RES="$BASE/results"
LOGDIR="$BASE/logs"
mkdir -p "$RES"

export PYTHONPATH="$PWD/projects/finagent_r1:$PWD/upstream/Agent-R1"
PY="$BASE/envs/agentr1/bin/python"
TEST_FILE="$BASE/finqa/test.parquet"
CORPUS="$BASE/finqa/corpus.jsonl"

shard_done() {  # $1 = shard index
    local log="$LOGDIR/eval_${TAG}_shard$1.log"
    [ -f "$RES/${TAG}_shard$1.jsonl" ] && grep -q '"avg_retrieved_chunks"' "$log" 2>/dev/null
}

run_shard() {  # $1 = shard index, $2 = gpu, $3 = util
    CUDA_VISIBLE_DEVICES="$2" "$PY" -m finagent_r1.evaluation.infer_vllm \
        --model "$MODEL" \
        --test_file "$TEST_FILE" \
        --corpus "$CORPUS" \
        --output "$RES/${TAG}_shard$1.jsonl" \
        --mode "$MODE" \
        --num_shards 4 --shard_index "$1" \
        --gpu_memory_utilization "$3" \
        --no_turns \
        > "$LOGDIR/eval_${TAG}_shard$1.log" 2>&1
}

# Clear residue from the preceding training run so vLLM can allocate KV cache.
"$BASE/envs/agentr1/bin/ray" stop --force >/dev/null 2>&1 || true
pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null || true
sleep 20

if [ -f "$RES/${TAG}_test.jsonl" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "predictions exist: $RES/${TAG}_test.jsonl (FORCE=1 to redo)"
else
    rm -f "$RES/${TAG}_shard"*.jsonl
    # 4 shards: 2 vLLM processes per GPU, started with a small stagger.
    for i in 0 1 2 3; do
        gpu=$([ "$i" -lt 2 ] && echo 0 || echo 3)
        run_shard "$i" "$gpu" 0.40 &
        sleep 25
    done
    wait
    pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null
    sleep 3

    # Retry failed / unfinished shards one at a time.
    for i in 0 1 2 3; do
        if shard_done "$i"; then continue; fi
        gpu=$([ "$i" -lt 2 ] && echo 0 || echo 3)
        echo "retrying shard $i on GPU $gpu"
        pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null
        sleep 5
        rm -f "$RES/${TAG}_shard${i}.jsonl"
        run_shard "$i" "$gpu" 0.45
        pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null
        sleep 3
    done

    for i in 0 1 2 3; do
        shard_done "$i" || { echo "ERROR: shard $i still incomplete"; exit 1; }
    done

    cat "$RES/${TAG}_shard0.jsonl" "$RES/${TAG}_shard1.jsonl" \
        "$RES/${TAG}_shard2.jsonl" "$RES/${TAG}_shard3.jsonl" > "$RES/${TAG}_test.jsonl"
fi

for subset in all seen_docs unseen_docs; do
    "$PY" -m finagent_r1.evaluation.evaluate_predictions \
        --predictions "$RES/${TAG}_test.jsonl" \
        --subset "$subset" \
        --by_ticker \
        --output_md "$RES/${TAG}_${subset}.md" \
        --output_json "$RES/${TAG}_${subset}.metrics.json" \
        > "$LOGDIR/score_${TAG}_${subset}.log" 2>&1
done
echo "=== $TAG ($MODE) ==="
RES="$RES" TAG="$TAG" "$PY" - <<'PY'
import json
import os
res, tag = os.environ["RES"], os.environ["TAG"]
for s in ["all", "seen_docs", "unseen_docs"]:
    d = json.load(open(f"{res}/{tag}_{s}.metrics.json"))["summary"]
    if not d.get("count") or "accuracy" not in d:
        print(f"{s:12s} n={d.get('count', 0):5d} (empty split)")
        continue
    print(f"{s:12s} n={d['count']:5d} acc={d['accuracy']:.4f} score={d['mean_score']:.4f} "
          f"fmt={d['format_rate']:.4f} steps={d['avg_steps']:.2f} "
          f"invalid={d['invalid_call_rate']:.4f} toolerr={d['tool_error_rate']:.4f}")
PY
