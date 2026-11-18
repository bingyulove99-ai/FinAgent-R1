"""Unit tests for the composite reward."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finagent_r1.rewards import (  # noqa: E402
    compute_score,
    numeric_match,
    score_answer,
)


class NumericMatchTest(unittest.TestCase):
    def test_percent_and_decimal_forms(self):
        self.assertTrue(numeric_match("25%", 25.0, target_is_percent=True))
        self.assertTrue(numeric_match("0.25", 25.0, target_is_percent=True))
        self.assertTrue(numeric_match("25.0", 0.25, target_is_percent=True))

    def test_plain_target_is_not_scaled(self):
        # Without a percent flag, 1 must not match 100.
        self.assertFalse(numeric_match("1", 100.0, target_is_percent=False))

    def test_percent_target_is_not_scaled_the_wrong_way(self):
        # 41.38% should not match a prediction of 4138.
        self.assertFalse(numeric_match("4138", 41.38, target_is_percent=True))
        self.assertTrue(numeric_match("0.4138", 41.38, target_is_percent=True))

    def test_tolerance(self):
        self.assertTrue(numeric_match("25.02%", 25.0, target_is_percent=True))
        self.assertFalse(numeric_match("26%", 25.0, target_is_percent=True))

    def test_first_numbers_only(self):
        # Reward hacking guard: only the first eight numbers are inspected.
        text = "1 2 3 4 5 6 7 8 25"
        self.assertFalse(numeric_match(text, 25.0, target_is_percent=True))


class ScoreTest(unittest.TestCase):
    def test_correct_numeric_answer(self):
        result = score_answer(
            "<analysis>done</analysis><answer>25%</answer>",
            "25%",
            {"exe_ans": 25.0, "is_numeric": True, "step_count": 1},
        )
        self.assertTrue(result["correct"])
        self.assertAlmostEqual(result["score"], 1.1, places=4)

    def test_no_answer_tag_scores_zero_outcome(self):
        result = score_answer(
            "<analysis>25%</analysis>",
            "25%",
            {"exe_ans": 25.0, "is_numeric": True, "step_count": 1},
        )
        self.assertFalse(result["correct"])
        self.assertEqual(result["outcome"], 0.0)
        self.assertAlmostEqual(result["score"], 0.05, places=4)

    def test_efficiency_discount(self):
        result = score_answer(
            "<analysis>x</analysis><answer>25%</answer>",
            "25%",
            {"exe_ans": 25.0, "is_numeric": True, "step_count": 6},
        )
        self.assertAlmostEqual(result["efficiency_multiplier"], max(0.85, 0.96**5), places=4)

    def test_tool_penalty_is_capped(self):
        result = score_answer(
            "<analysis>x</analysis><answer>25%</answer>",
            "25%",
            {"exe_ans": 25.0, "is_numeric": True, "tool_errors": 100},
        )
        self.assertAlmostEqual(result["tool_penalty"], 0.15, places=4)
        self.assertAlmostEqual(result["score"], 0.95, places=4)

    def test_text_answer_f1(self):
        result = score_answer(
            "<analysis>x</analysis><answer>net sales increased</answer>",
            "net sales increased",
            {"is_numeric": False, "step_count": 1},
        )
        self.assertTrue(result["correct"])
        self.assertAlmostEqual(result["score"], 1.1, places=4)

    def test_compute_score_signature(self):
        score = compute_score(
            "finqa",
            "<analysis>x</analysis><answer>10%</answer>",
            "10%",
            [{"exe_ans": 10.0, "is_numeric": True}],
        )
        self.assertAlmostEqual(score, 1.1, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
