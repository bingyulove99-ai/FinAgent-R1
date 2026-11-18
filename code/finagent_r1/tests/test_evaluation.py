"""Tests for offline prediction aggregation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finagent_r1.evaluation.evaluate_predictions import (  # noqa: E402
    aggregate,
    filter_subset,
    score_record,
)


def _record(seen: bool, response: str, answer: str = "25%", exe_ans: float = 25.0):
    return {
        "id": "q1",
        "response": response,
        "answer": answer,
        "extra_info": {
            "exe_ans": exe_ans,
            "is_numeric": True,
            "doc_seen_in_train": seen,
            "ticker": "ACME",
        },
        "steps": 2,
        "tool_call_count": 2,
        "invalid_tool_calls": 0,
        "tool_errors": 0,
    }


class EvaluationTest(unittest.TestCase):
    def test_score_record_uses_response(self):
        scored = score_record(_record(True, "<analysis>x</analysis><answer>25%</answer>"))
        self.assertTrue(scored["correct"])
        self.assertEqual(scored["steps"], 2)

    def test_aggregate(self):
        first = _record(True, "<analysis>x</analysis><answer>25%</answer>")
        first["retrieval_hits"] = 1
        first["retrieval_total"] = 2
        records = [
            score_record(first),
            score_record(_record(False, "<analysis>x</analysis><answer>20%</answer>")),
        ]
        summary = aggregate(records)
        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["avg_tool_calls"], 2.0)
        self.assertAlmostEqual(summary["retrieval_recall"], 0.5)

    def test_filter_subset(self):
        rows = [_record(True, "<answer>25%</answer>"), _record(False, "<answer>25%</answer>")]
        self.assertEqual(len(filter_subset(rows, "all")), 2)
        self.assertEqual(len(filter_subset(rows, "seen_docs")), 1)
        self.assertEqual(len(filter_subset(rows, "unseen_docs")), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
