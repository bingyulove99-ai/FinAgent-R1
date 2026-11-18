"""Integration test for the vLLM evaluator without requiring vLLM.

``VLLMGenerator`` imports vLLM lazily, so the rest of ``infer_vllm`` can be
tested with a scripted generator and a fake tokenizer.  This covers record
packaging, answer scoring, retrieval recall, and direct mode.
"""

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

from finagent_r1.core import EpisodeConfig, Generation  # noqa: E402
from finagent_r1.data.prepare_finqa import _read_json_payload, build_corpus  # noqa: E402
from finagent_r1.evaluation.infer_vllm import _run_one  # noqa: E402
from finagent_r1.tools import FinancialRetrieverTool, PythonFinancialCalcTool  # noqa: E402

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "finqa_raw"


class ScriptedGenerator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0

    def generate(self, prompt_ids, _params=None):
        index = self.index
        self.index += 1
        return Generation(token_ids=[1] * 8, text=self.responses[min(index, len(self.responses) - 1)])


class FakeTokenizer:
    def apply_chat_template(self, messages, tools=None, **kwargs):
        text = "".join(str(message.get("content")) for message in messages)
        return [1] * (len(text) // 4 + 5)

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


def _build_corpus_path() -> Path:
    examples = []
    for split in ("train", "dev", "test"):
        examples.extend(_read_json_payload(_FIXTURES / f"{split}.json"))
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    with handle:
        for chunk in build_corpus(examples):
            handle.write(json.dumps(chunk) + "\n")
    return Path(handle.name)


class InferAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus_path = _build_corpus_path()

    @classmethod
    def tearDownClass(cls):
        cls.corpus_path.unlink(missing_ok=True)

    def _row(self):
        return {
            "id": "fixture-q1",
            "prompt": [{"role": "user", "content": "What was revenue growth?"}],
            "extra_info": {
                "question_id": "fixture-q1",
                "question": "What was revenue growth?",
                "answer": "25%",
                "exe_ans": 25.0,
                "is_numeric": True,
                "ticker": "ACME",
                "year": 2021,
                "gold_evidence": ["Revenue | 100.0 | 80.0"],
                "doc_seen_in_train": True,
            },
        }

    def _tools(self):
        return [
            FinancialRetrieverTool(corpus_path=str(self.corpus_path)),
            PythonFinancialCalcTool(),
        ]

    def test_agent_mode_record(self):
        responses = [
            '<analysis>need revenue</analysis><tool_call>{"name": "financial_retriever", '
            '"arguments": {"query": "revenue"}}</tool_call>',
            '<analysis>calc</analysis><tool_call>{"name": "python_calc", '
            '"arguments": {"code": "(100 - 80) / 80 * 100"}}</tool_call>',
            "<analysis>done</analysis><answer>25%</answer>",
        ]
        record = asyncio.run(
            _run_one(
                row=self._row(),
                generator=ScriptedGenerator(responses),
                tokenizer=FakeTokenizer(),
                tools=self._tools(),
                config=EpisodeConfig(prompt_length=2048, response_length=256),
                mode="agent",
                include_turns=True,
            )
        )
        self.assertTrue(record["correct"])
        self.assertEqual(record["steps"], 3)
        self.assertEqual(record["tool_call_count"], 2)
        self.assertEqual(record["retrieval_total"], 1)
        self.assertEqual(record["retrieval_hits"], 1)
        self.assertEqual(record["retrieval_recall"], 1.0)
        self.assertEqual(len(record["turns"]), 3)

    def test_direct_mode_record(self):
        record = asyncio.run(
            _run_one(
                row=self._row(),
                generator=ScriptedGenerator(["<answer>25%</answer>"]),
                tokenizer=FakeTokenizer(),
                tools=self._tools(),
                config=EpisodeConfig(prompt_length=2048, response_length=256),
                mode="direct",
                include_turns=False,
            )
        )
        self.assertTrue(record["correct"])
        self.assertEqual(record["steps"], 1)
        self.assertEqual(record["tool_call_count"], 0)
        self.assertEqual(record["retrieval_total"], 0)
        self.assertIsNone(record["retrieval_recall"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
