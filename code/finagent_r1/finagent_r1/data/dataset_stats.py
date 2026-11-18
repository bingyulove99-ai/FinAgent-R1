#!/usr/bin/env python3
"""Audit the generated FinQA dataset and write an honest dataset card.

Usage::

    python -m finagent_r1.data.dataset_stats --data_dir data/finqa

The script reads ``train/dev/test`` and ``corpus.jsonl``, prints a profile and
writes ``DATASET_CARD.md`` + ``dataset_card.json`` next to the data.  It never
prints benchmark accuracy numbers: only dataset statistics.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from finagent_r1.data.prepare_finqa import _json_safe, summarize


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _percentile(values: Sequence[int | float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def profile(data_dir: Path) -> dict[str, Any]:
    rows_by_split = {
        split: _load_rows(data_dir / f"{split}.jsonl")
        or _load_rows(data_dir / f"{split}.parquet")
        for split in ("train", "dev", "test")
    }
    corpus_rows = _load_rows(data_dir / "corpus.jsonl")
    if not corpus_rows:
        corpus_rows = _load_rows(data_dir / "corpus.parquet")

    corpus = [
        {
            "chunk_id": row.get("chunk_id"),
            "doc_id": row.get("doc_id"),
            "ticker": row.get("ticker"),
            "year": row.get("year"),
            "kind": row.get("kind"),
            "text": row.get("text") or row.get("content") or "",
        }
        for row in corpus_rows
    ]

    summary = summarize(corpus, rows_by_split)

    chunks_per_doc = Counter(chunk["doc_id"] for chunk in corpus)
    counts = list(chunks_per_doc.values())
    summary["corpus"]["chunks_per_doc"] = {
        "min": min(counts) if counts else 0,
        "median": statistics.median(counts) if counts else 0,
        "max": max(counts) if counts else 0,
    }

    for split, rows in rows_by_split.items():
        question_lengths = [len(row["extra_info"]["question"]) for row in rows]
        answer_lengths = [len(str(row["extra_info"].get("answer", ""))) for row in rows]
        years = [row["extra_info"].get("year") for row in rows if row["extra_info"].get("year")]
        summary["splits"][split].update(
            {
                "question_chars_p50": _percentile(question_lengths, 0.50),
                "question_chars_p95": _percentile(question_lengths, 0.95),
                "answer_chars_p50": _percentile(answer_lengths, 0.50),
                "year_range": [min(years), max(years)] if years else None,
                "top_tickers": Counter(
                    row["extra_info"].get("ticker")
                    for row in rows
                    if row["extra_info"].get("ticker")
                ).most_common(10),
            }
        )

    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# FinQA Agent Dataset Card",
        "",
        "> This card is generated from the files in `data/finqa`. It reports dataset "
        "statistics only; it contains no model results.",
        "",
        "## Source",
        "",
        "- Dataset: FinQA (Chen et al., EMNLP 2021), 8,281 QA pairs over 2,789 S&P 500 filings.",
        "- Original split: 6,251 train / 883 dev / 1,147 test.",
        "- Official repository: <https://github.com/cszyssrs/FinQA>.",
        "- The corpus is derived from `pre_text`, `table` and `post_text` of each filing.",
        "",
        "## Splits",
        "",
        "| Split | Rows | Unique filings | Tickers | Numeric rows | Doc seen in train | Q chars p50 | Q chars p95 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "dev", "test"):
        item = summary["splits"].get(split, {})
        lines.append(
            "| {split} | {rows} | {docs} | {tickers} | {numeric} | {seen} | {p50} | {p95} |".format(
                split=split,
                rows=item.get("rows", 0),
                docs=item.get("unique_docs", 0),
                tickers=item.get("unique_tickers", 0),
                numeric=item.get("numeric_rows", 0),
                seen=item.get("doc_seen_in_train_rows", 0),
                p50=item.get("question_chars_p50", 0),
                p95=item.get("question_chars_p95", 0),
            )
        )

    lines += [
        "",
        "`Doc seen in train` marks rows whose filing also appears in the training "
        "split. The official FinQA split is question-level, so the same report can "
        "appear in train and test. Report metrics on both the full test split and "
        "the document-disjoint subset (`doc_seen_in_train == False`).",
        "",
        "## Program length (multi-step proxy)",
        "",
        "| Split | 0 steps | 1 step | 2 steps | 3 steps | 4+ steps |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "dev", "test"):
        hist = summary["splits"].get(split, {}).get("program_length_histogram", {})
        lines.append(
            f"| {split} | {hist.get('0', 0)} | {hist.get('1', 0)} | {hist.get('2', 0)} | "
            f"{hist.get('3', 0)} | {hist.get('4+', 0)} |"
        )

    corpus = summary["corpus"]
    lines += [
        "",
        "## Retrieval corpus",
        "",
        f"- Chunks: **{corpus.get('chunks', 0)}**",
        f"- Unique filings: **{corpus.get('unique_docs', 0)}**",
        f"- Chunks with ticker/year scope: **{corpus.get('chunks_with_scope', 0)}**",
        f"- Chunks per filing: min={corpus.get('chunks_per_doc', {}).get('min', 0)}, "
        f"median={corpus.get('chunks_per_doc', {}).get('median', 0)}, "
        f"max={corpus.get('chunks_per_doc', {}).get('max', 0)}",
        f"- Chunk kinds: `{json.dumps(corpus.get('kinds', {}), ensure_ascii=False)}`",
        "",
        "## Evaluation protocol",
        "",
        "1. Validate on `dev` every N updates; never train on it.",
        "2. Report final numbers on `test` only once.",
        "3. Report the full test split and the `doc_seen_in_train == False` subset separately.",
        "4. Metrics: numeric execution match (tolerance 1e-3 relative / 1e-4 absolute), "
        "format rate, average steps, tool-call validity, tool-error rate.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/finqa")
    parser.add_argument("--no_markdown", action="store_true")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser()
    summary = profile(data_dir)

    safe_summary = _json_safe(summary)
    (data_dir / "dataset_card.json").write_text(
        json.dumps(safe_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.no_markdown:
        (data_dir / "DATASET_CARD.md").write_text(render_markdown(summary), encoding="utf-8")

    print(json.dumps(safe_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
