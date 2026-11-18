#!/usr/bin/env python3
"""Score an inference JSONL file with the same reward logic used in training.

Expected input (one JSON object per line), produced by ``infer_vllm.py``::

    {
      "id": "...",
      "prediction": "...",          # final answer text or full response
      "answer": "...",              # FinQA answer string
      "extra_info": {"exe_ans": 0.75, "is_numeric": true, ...},
      "response": "<analysis>...</analysis><answer>...</answer>",
      "steps": 3,
      "tool_call_count": 3,
      "invalid_tool_calls": 0,
      "tool_errors": 0
    }

This script is intentionally honest: it only aggregates the file you give it.
It contains no reference or fabricated benchmark table.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from finagent_r1.rewards import extract_final_answer, score_answer


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extra(record: dict[str, Any]) -> dict[str, Any]:
    extra = record.get("extra_info")
    return dict(extra) if isinstance(extra, dict) else {}


def _prediction_text(record: dict[str, Any]) -> str:
    response = record.get("response") or record.get("solution") or ""
    answer = extract_final_answer(response)
    if answer:
        return response
    prediction = str(record.get("prediction") or "").strip()
    if prediction:
        return f"<answer>{prediction}</answer>"
    return ""


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    extra = _extra(record)
    extra.setdefault("step_count", record.get("steps", 1))
    extra.setdefault("invalid_tool_calls", record.get("invalid_tool_calls", 0))
    extra.setdefault("tool_errors", record.get("tool_errors", 0))

    ground_truth = record.get("answer", extra.get("answer", ""))
    result = score_answer(_prediction_text(record), ground_truth, extra)
    result.update(
        {
            "id": record.get("id"),
            "ticker": extra.get("ticker", record.get("ticker")),
            "year": extra.get("year", record.get("year")),
            "doc_seen_in_train": extra.get("doc_seen_in_train"),
            "steps": record.get("steps", extra.get("step_count", 1)),
            "tool_call_count": record.get("tool_call_count", 0),
            "invalid_tool_calls": record.get("invalid_tool_calls", 0),
            "tool_errors": record.get("tool_errors", 0),
            "retrieved_chunks": record.get("retrieved_chunks", 0),
            "retrieval_hits": record.get("retrieval_hits", 0),
            "retrieval_total": record.get("retrieval_total", 0),
        }
    )
    return result


def aggregate(scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored)
    if total == 0:
        return {"count": 0}

    def mean(key: str) -> float:
        return round(sum(float(item.get(key, 0) or 0) for item in scored) / total, 4)

    tool_calls = sum(int(item.get("tool_call_count", 0) or 0) for item in scored)
    invalid = sum(int(item.get("invalid_tool_calls", 0) or 0) for item in scored)
    errors = sum(int(item.get("tool_errors", 0) or 0) for item in scored)
    recall_rows = [
        float(item["retrieval_hits"]) / float(item["retrieval_total"])
        for item in scored
        if item.get("retrieval_total")
    ]

    return {
        "count": total,
        "accuracy": mean("correct"),
        "mean_score": mean("score"),
        "mean_f1": mean("f1"),
        "answer_rate": mean("answer_found"),
        "format_rate": round(
            sum(1 for item in scored if item.get("format_reward", 0) >= 0.1) / total, 4
        ),
        "avg_steps": mean("steps"),
        "avg_tool_calls": round(tool_calls / total, 4),
        "invalid_call_rate": round(invalid / max(1, tool_calls), 4),
        "tool_error_rate": round(errors / max(1, tool_calls), 4),
        "avg_retrieved_chunks": mean("retrieved_chunks"),
        "retrieval_recall": round(sum(recall_rows) / len(recall_rows), 4) if recall_rows else None,
    }


def filter_subset(rows: Sequence[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    if subset == "all":
        return list(rows)
    if subset == "unseen_docs":
        return [
            row
            for row in rows
            if _extra(row).get("doc_seen_in_train") is False
        ]
    if subset == "seen_docs":
        return [
            row
            for row in rows
            if _extra(row).get("doc_seen_in_train") is True
        ]
    raise ValueError(f"Unknown subset: {subset}")


def group_metrics(scored: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        groups[str(item.get(key) or "UNKNOWN")].append(item)
    return {name: aggregate(items) for name, items in sorted(groups.items())}


def render_markdown(summary: dict[str, Any], by_ticker: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Evaluated rows | {summary.get('count', 0)} |",
        f"| Inference errors (excluded) | {summary.get('inference_errors', 0)} |",
        f"| Numeric / text accuracy | {summary.get('accuracy', 0):.4f} |",
        f"| Mean composite reward | {summary.get('mean_score', 0):.4f} |",
        f"| Mean token F1 | {summary.get('mean_f1', 0):.4f} |",
        f"| Answer rate | {summary.get('answer_rate', 0):.4f} |",
        f"| Format rate | {summary.get('format_rate', 0):.4f} |",
        f"| Average steps | {summary.get('avg_steps', 0):.2f} |",
        f"| Average tool calls | {summary.get('avg_tool_calls', 0):.2f} |",
        f"| Invalid-call rate | {summary.get('invalid_call_rate', 0):.4f} |",
        f"| Tool-error rate | {summary.get('tool_error_rate', 0):.4f} |",
        f"| Avg retrieved chunks | {summary.get('avg_retrieved_chunks', 0):.2f} |",
        f"| Retrieval recall (gold evidence) | "
        f"{summary['retrieval_recall'] if summary.get('retrieval_recall') is not None else 'n/a'} |",
    ]
    if by_ticker:
        lines += [
            "",
            "### Per-ticker accuracy",
            "",
            "| Ticker | Rows | Accuracy | Avg steps |",
            "| :--- | ---: | ---: | ---: |",
        ]
        for ticker, item in sorted(by_ticker.items(), key=lambda pair: -pair[1].get("count", 0)):
            lines.append(
                f"| {ticker} | {item.get('count', 0)} | {item.get('accuracy', 0):.4f} | "
                f"{item.get('avg_steps', 0):.2f} |"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--subset", default="all", choices=["all", "seen_docs", "unseen_docs"])
    parser.add_argument("--output_md", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--by_ticker", action="store_true")
    args = parser.parse_args(argv)

    all_rows = load_predictions(Path(args.predictions))
    error_count = sum(1 for row in all_rows if "error" in row)
    rows = filter_subset([row for row in all_rows if "error" not in row], args.subset)
    if error_count:
        print(f"Skipped {error_count} inference-error record(s).")
    scored = [score_record(row) for row in rows]
    summary = aggregate(scored)
    summary["inference_errors"] = error_count
    by_ticker = group_metrics(scored, "ticker") if args.by_ticker else {}

    payload = {
        "subset": args.subset,
        "summary": summary,
        "by_ticker": by_ticker,
    }
    markdown = render_markdown(summary, by_ticker)
    print(markdown)

    if args.output_md:
        Path(args.output_md).write_text(markdown + "\n", encoding="utf-8")
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
