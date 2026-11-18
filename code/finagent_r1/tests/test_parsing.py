"""Unit tests for response parsing and prompt construction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finagent_r1.parsing import (  # noqa: E402
    TRUNCATION_MARKER,
    build_user_content,
    extract_query,
    format_observations,
    merge_scope,
    merge_tools_kwargs,
    parse_action,
)


class ParseActionTest(unittest.TestCase):
    def test_parses_tool_call_and_answer(self):
        text = (
            '<analysis>need revenue</analysis>'
            '<tool_call>{"name": "financial_retriever", "arguments": {"query": "revenue 2021"}}</tool_call>'
        )
        action = parse_action(text)
        self.assertEqual(action.malformed_tool_calls, 0)
        self.assertEqual(len(action.tool_calls), 1)
        self.assertEqual(action.tool_calls[0].name, "financial_retriever")
        self.assertEqual(action.tool_calls[0].arguments["query"], "revenue 2021")
        self.assertFalse(action.has_answer)

    def test_accepts_parameters_and_tool_keys(self):
        text = (
            '<tool_call>{"name": "python_calc", "parameters": {"code": "1+1"}}</tool_call>'
            '<tool_call>{"tool": "financial_retriever", "args": {"query": "cash"}}</tool_call>'
        )
        action = parse_action(text)
        self.assertEqual(len(action.tool_calls), 2)
        self.assertEqual(action.tool_calls[0].arguments["code"], "1+1")
        self.assertEqual(action.tool_calls[1].name, "financial_retriever")

    def test_accepts_list_of_calls(self):
        action = parse_action(
            '<tool_call>[{"name": "financial_retriever", "arguments": {"query": "a"}}, '
            '{"name": "python_calc", "arguments": {"code": "1+1"}}]</tool_call>'
        )
        self.assertEqual(action.malformed_tool_calls, 0)
        self.assertEqual([call.name for call in action.tool_calls], ["financial_retriever", "python_calc"])

    def test_counts_unclosed_and_invalid_calls(self):
        action = parse_action('<tool_call>{"name": "x"')
        self.assertEqual(action.malformed_tool_calls, 1)
        self.assertEqual(action.tool_calls, ())

        action = parse_action("<tool_call>not json</tool_call>")
        self.assertEqual(action.malformed_tool_calls, 1)

    def test_last_answer_wins(self):
        action = parse_action(
            "<answer>first</answer><analysis>x</analysis><answer>second</answer>"
        )
        self.assertEqual(action.answer, "second")

    def test_code_fence_is_tolerated(self):
        action = parse_action(
            '<tool_call>```json\n{"name": "python_calc", "arguments": {"code": "2*3"}}\n```</tool_call>'
        )
        self.assertEqual(action.malformed_tool_calls, 0)
        self.assertEqual(action.tool_calls[0].arguments["code"], "2*3")


class PromptTest(unittest.TestCase):
    def test_extract_query_from_messages(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "What is revenue?"},
        ]
        self.assertEqual(extract_query(messages), "What is revenue?")

    def test_merge_scope_reads_extra_info(self):
        scope = merge_scope(
            {
                "extra_info": {"ticker": "acme", "year": "2021"},
                "reward_model": {"ground_truth": "25%"},
            }
        )
        self.assertEqual(scope["ticker"], "ACME")
        self.assertEqual(scope["year"], 2021)
        self.assertEqual(scope["ground_truth"], "25%")

    def test_merge_tools_kwargs_reads_corpus_path(self):
        merged = merge_tools_kwargs(
            {
                "extra_info": {
                    "ticker": "ACME",
                    "year": 2021,
                    "corpus_path": "/data/finqa/corpus.jsonl",
                },
                "reward_model": {"ground_truth": "25%"},
            }
        )
        self.assertEqual(merged["ticker"], "ACME")
        self.assertEqual(merged["year"], 2021)
        self.assertEqual(merged["corpus_path"], "/data/finqa/corpus.jsonl")

    def test_observation_truncation_keeps_newest(self):
        old = "OLD " * 100
        new = "NEW " * 100
        text = format_observations([old, new], window=2, max_chars=220)
        self.assertTrue(text.startswith(TRUNCATION_MARKER))
        self.assertTrue(text.rstrip().endswith("NEW"))
        self.assertNotIn("OLD OLD OLD", text)

    def test_build_user_content_contains_scope_and_step(self):
        content = build_user_content(
            query="What is revenue?",
            ticker="ACME",
            fiscal_year=2021,
            scratchpad=["calc: 100-80 = 20"],
            history_actions=["financial_retriever(...)"],
            observations=["[table] Revenue 100"],
            tool_feedback="",
            step_index=2,
            max_steps=6,
        )
        self.assertIn("ticker=ACME", content)
        self.assertIn("fiscal_year=2021", content)
        self.assertIn("Step 2/6", content)
        self.assertIn("Revenue 100", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
