"""End-to-end tests of the framework-independent episode loop."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finagent_r1.core import EpisodeConfig, Generation, run_episode  # noqa: E402
from finagent_r1.data.prepare_finqa import _read_json_payload, build_corpus  # noqa: E402
from finagent_r1.tools import FinancialRetrieverTool, PythonFinancialCalcTool  # noqa: E402

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "finqa_raw"


def _corpus_path() -> Path:
    examples = []
    for split in ("train", "dev", "test"):
        examples.extend(_read_json_payload(_FIXTURES / f"{split}.json"))
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    with handle:
        for chunk in build_corpus(examples):
            handle.write(json.dumps(chunk) + "\n")
    return Path(handle.name)


class ScriptedGenerator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def __call__(self, prompt_ids, params):
        text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return Generation(token_ids=[7] * 12, text=text)


async def _render(messages, tools):
    text = "".join(str(message.get("content")) for message in messages)
    return [1] * max(1, len(text) // 4)


class EpisodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus_path = _corpus_path()

    @classmethod
    def tearDownClass(cls):
        cls.corpus_path.unlink(missing_ok=True)

    def _tools(self):
        return [
            FinancialRetrieverTool(corpus_path=str(self.corpus_path)),
            PythonFinancialCalcTool(),
        ]

    def _run(self, responses, config=None, ticker="ACME", year=2021):
        generator = ScriptedGenerator(responses)
        result = asyncio.run(
            run_episode(
                question="What was revenue growth?",
                ticker=ticker,
                year=year,
                generate_fn=generator,
                render_fn=_render,
                decode_fn=lambda ids: "decoded",
                tools=self._tools(),
                config=config or EpisodeConfig(prompt_length=2048, response_length=256),
                tools_kwargs={"ticker": ticker, "year": year},
            )
        )
        return result, generator

    def test_retrieve_calculate_answer(self):
        result, generator = self._run(
            [
                '<analysis>need revenue</analysis><tool_call>{"name": "financial_retriever", '
                '"arguments": {"query": "revenue"}}</tool_call>',
                '<analysis>calculate</analysis><tool_call>{"name": "python_calc", '
                '"arguments": {"code": "(100 - 80) / 80 * 100"}}</tool_call>',
                "<analysis>done</analysis><answer>25%</answer>",
            ]
        )
        self.assertEqual(len(result.turns), 3)
        self.assertEqual([turn.stop for turn in result.turns], [False, False, True])
        self.assertEqual(result.final_answer, "25%")
        self.assertEqual(result.stats["retriever_calls"], 1)
        self.assertEqual(result.stats["calc_calls"], 1)
        self.assertEqual(result.stats["tool_call_count"], 2)
        self.assertEqual(result.stats["tool_errors"], 0)
        self.assertGreater(result.stats["retrieved_chunks"], 0)
        self.assertTrue(any("Revenue" in item for item in result.scratchpad))
        self.assertTrue(any("calc:" in item for item in result.scratchpad))
        self.assertEqual(generator.calls, 3)

    def test_malformed_call_is_recoverable(self):
        result, _ = self._run(
            [
                '<tool_call>{"name": "financial_retriever"</tool_call>',
                "<analysis>done</analysis><answer>10%</answer>",
            ]
        )
        self.assertEqual(len(result.turns), 2)
        self.assertEqual(result.stats["invalid_tool_calls"], 1)
        self.assertEqual(result.final_answer, "10%")

    def test_prose_without_answer_stops(self):
        result, _ = self._run(["I am not sure."])
        self.assertEqual(len(result.turns), 1)
        self.assertTrue(result.turns[0].stop)
        self.assertEqual(result.final_answer, "")

    def test_max_steps_is_respected(self):
        config = EpisodeConfig(max_steps=3, prompt_length=2048, response_length=256)
        tool_call = (
            '<tool_call>{"name": "financial_retriever", "arguments": {"query": "revenue"}}</tool_call>'
        )
        result, generator = self._run([tool_call], config=config)
        self.assertEqual(len(result.turns), 3)
        self.assertEqual(generator.calls, 3)
        self.assertTrue(result.turns[-1].stop)
        self.assertEqual(result.final_answer, "")

    def test_prompt_budget_is_enforced(self):
        config = EpisodeConfig(max_steps=2, prompt_length=300, response_length=128)
        result, _ = self._run(
            [
                '<tool_call>{"name": "financial_retriever", "arguments": {"query": "revenue"}}</tool_call>',
                "<analysis>done</analysis><answer>25%</answer>",
            ],
            config=config,
        )
        self.assertTrue(all(len(turn.prompt_ids) <= 300 for turn in result.turns))
        self.assertEqual(result.stats["context_truncated"], 1)

    def test_unknown_tool_is_a_tool_error(self):
        result, _ = self._run(
            [
                '<tool_call>{"name": "web_search", "arguments": {"query": "revenue"}}</tool_call>',
                "<analysis>done</analysis><answer>25%</answer>",
            ]
        )
        self.assertEqual(result.stats["tool_errors"], 1)
        self.assertEqual(result.stats["tool_call_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
