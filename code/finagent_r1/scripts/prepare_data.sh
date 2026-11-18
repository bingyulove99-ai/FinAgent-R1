#!/usr/bin/env bash
# Download/prepare FinQA and write train/dev/test parquet + corpus.jsonl.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON:-python3}"
if ! "$PYTHON_BIN" -c "import sys" >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON:-python}"
fi
DATA_DIR="${FINQA_DATA_DIR:-$ROOT_DIR/data/finqa}"
RAW_DIR="${FINQA_RAW_DIR:-$DATA_DIR/raw}"

mkdir -p "$DATA_DIR"

if [[ "${1:-}" == "--hf" ]]; then
    shift
    "$PYTHON_BIN" -m finagent_r1.data.prepare_finqa \
        --hf_dataset "${HF_DATASET:-dreamerdeo/finqa}" \
        --output_dir "$DATA_DIR" "$@"
else
    "$PYTHON_BIN" -m finagent_r1.data.prepare_finqa \
        --download \
        --raw_dir "$RAW_DIR" \
        --output_dir "$DATA_DIR" "$@"
fi

"$PYTHON_BIN" -m finagent_r1.data.dataset_stats --data_dir "$DATA_DIR"

echo
if [[ ! -f "$DATA_DIR/train.parquet" ]]; then
    echo "WARNING: $DATA_DIR/train.parquet is missing." >&2
    echo "Install pyarrow (pip install pyarrow) and rerun this script." >&2
    echo "Agent-R1 training expects parquet files, not JSONL." >&2
fi

echo "Data ready in $DATA_DIR"
echo "  export FINQA_CORPUS_PATH=$DATA_DIR/corpus.jsonl"
