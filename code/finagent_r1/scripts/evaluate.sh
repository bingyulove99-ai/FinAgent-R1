#!/usr/bin/env bash
# Run inference on the test split and score the predictions.
#
# Usage:
#   MODEL=/path/to/checkpoint TEST_FILE=$PWD/data/finqa/test.parquet \
#     bash projects/finagent_r1/scripts/evaluate.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_DIR:$ROOT_DIR/upstream/Agent-R1:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON:-python3}"
if ! "$PYTHON_BIN" -c "import sys" >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON:-python}"
fi

MODEL="${MODEL:?Set MODEL to a Hugging Face model id or merged checkpoint path}"
TEST_FILE="${TEST_FILE:-$ROOT_DIR/data/finqa/test.parquet}"
CORPUS="${FINQA_CORPUS_PATH:-$ROOT_DIR/data/finqa/corpus.jsonl}"
MODEL_TAG="$(basename "$(dirname "$MODEL")")_$(basename "$MODEL")"
OUTPUT="${OUTPUT:-$ROOT_DIR/results/${MODEL_TAG}_${SUBSET:-all}.jsonl}"
SUBSET="${SUBSET:-all}"
export FINQA_CORPUS_PATH="$CORPUS"

mkdir -p "$(dirname "$OUTPUT")"

"$PYTHON_BIN" -m finagent_r1.evaluation.infer_vllm \
    --model "$MODEL" \
    --test_file "$TEST_FILE" \
    --corpus "$CORPUS" \
    --output "$OUTPUT" \
    --subset "$SUBSET" \
    "$@"

"$PYTHON_BIN" -m finagent_r1.evaluation.evaluate_predictions \
    --predictions "$OUTPUT" \
    --subset "$SUBSET" \
    --by_ticker \
    --output_md "${OUTPUT%.jsonl}.md" \
    --output_json "${OUTPUT%.jsonl}.metrics.json"
