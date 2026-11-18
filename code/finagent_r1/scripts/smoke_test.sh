#!/usr/bin/env bash
# CPU-only smoke test: unit tests + tiny data build + reward/eval plumbing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON:-python3}"
if ! "$PYTHON_BIN" -c "import sys" >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON:-python}"
fi

echo "== 1. Unit tests (framework-independent core) =="
cd "$PROJECT_DIR"
"$PYTHON_BIN" -m unittest discover -s tests -v

echo
echo "== 2. Build a tiny dataset from the fixtures =="
SMOKE_DIR="$ROOT_DIR/data/finqa_smoke"
rm -rf "$SMOKE_DIR"
"$PYTHON_BIN" -m finagent_r1.data.prepare_finqa \
    --raw_dir "$PROJECT_DIR/tests/fixtures/finqa_raw" \
    --output_dir "$SMOKE_DIR" \
    --no_parquet

echo
echo "== 3. Dataset audit =="
"$PYTHON_BIN" -m finagent_r1.data.dataset_stats --data_dir "$SMOKE_DIR" --no_markdown

echo
echo "== 4. Score the gold responses with the training reward =="
SMOKE_DIR_PY="$SMOKE_DIR"
if command -v cygpath >/dev/null 2>&1; then
    SMOKE_DIR_PY="$(cygpath -w "$SMOKE_DIR")"
fi
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
from finagent_r1.rewards import score_answer

smoke = Path(r"$SMOKE_DIR_PY")
rows = [json.loads(line) for line in (smoke / "test.jsonl").read_text(encoding="utf-8").splitlines() if line]
score = 0.0
for row in rows:
    extra = row["extra_info"]
    response = f"<analysis>fixture</analysis><answer>{extra['answer']}</answer>"
    result = score_answer(response, extra["answer"], extra)
    score += result["score"]
    print(f"  {extra['question_id']}: correct={result['correct']} score={result['score']}")
print(f"mean score={score / max(1, len(rows)):.4f}")
PY

echo
echo "Smoke test passed."
