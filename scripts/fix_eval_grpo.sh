#!/usr/bin/env bash
# Repair the GRPO test evaluation: rerun missing shards 1-3, assemble, score,
# mark eval_done, and emit the per-algorithm summary.
set -uo pipefail
source /data/home.old/user/agent-r1/env.sh
BASE=/data/home.old/user/agent-r1
cd /home/user/Agent-R1-main
RES="$BASE/results"
LOGDIR="$BASE/logs"
PY="$BASE/envs/agentr1/bin/python"
MODEL="$BASE/outputs/finqa_qwen3b_grpo/global_step_195/huggingface"
TEST="$BASE/finqa/test.parquet"
CORPUS="$BASE/finqa/corpus.jsonl"
export PYTHONPATH="$PWD/projects/finagent_r1:$PWD/upstream/Agent-R1"

for i in 1 2 3; do
    gpu=$([ "$i" -lt 2 ] && echo 0 || echo 3)
    echo "$(date '+%T') starting shard $i on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" -m finagent_r1.evaluation.infer_vllm \
        --model "$MODEL" --test_file "$TEST" --corpus "$CORPUS" \
        --output "$RES/grpo_shard${i}.jsonl" --mode agent \
        --num_shards 4 --shard_index "$i" \
        --gpu_memory_utilization 0.40 --no_turns \
        > "$LOGDIR/eval_grpo_shard${i}.log" 2>&1 &
done
wait
echo "$(date '+%T') all shards finished"
pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null
sleep 3

for i in 0 1 2 3; do
    [ -f "$RES/grpo_shard${i}.jsonl" ] || { echo "MISSING shard $i"; exit 1; }
done
cat "$RES"/grpo_shard0.jsonl "$RES"/grpo_shard1.jsonl \
    "$RES"/grpo_shard2.jsonl "$RES"/grpo_shard3.jsonl > "$RES/grpo_test.jsonl"
echo "$(date '+%T') assembled $(wc -l < "$RES/grpo_test.jsonl") rows"

for subset in all seen_docs unseen_docs; do
    "$PY" -m finagent_r1.evaluation.evaluate_predictions \
        --predictions "$RES/grpo_test.jsonl" --subset "$subset" --by_ticker \
        --output_md "$RES/grpo_${subset}.md" \
        --output_json "$RES/grpo_${subset}.metrics.json" \
        > "$LOGDIR/score_grpo_${subset}.log" 2>&1 || echo "score $subset failed"
done
"$PY" "$BASE/summarize_algo.py" grpo > "$LOGDIR/summary_grpo.log" 2>&1
touch "$LOGDIR/eval_done_grpo"
echo "$(date '+%T') GRPO eval repaired"
