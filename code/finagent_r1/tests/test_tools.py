"""Unit tests for retrieval scoping and the arithmetic sandbox."""

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

from finagent_r1.data.prepare_finqa import _read_json_payload, build_corpus  # noqa: E402
from finagent_r1.tools import (  # noqa: E402
    FinancialRetrieverTool,
    PythonFinancialCalcTool,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "finqa_raw"


def _build_temp_corpus() -> Path:
    examples = []
    for split in ("train", "dev", "test"):
        examples.extend(_read_json_payload(_FIXTURES / f"{split}.json"))
    corpus = build_corpus(examples)
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    with handle:
        for chunk in corpus:
            handle.write(json.dumps(chunk) + "\n")
    return Path(handle.name)


class RetrieverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus_path = _build_temp_corpus()

    @classmethod
    def tearDownClass(cls):
        cls.corpus_path.unlink(missing_ok=True)

    def _run(self, args, **kwargs):
        tool = FinancialRetrieverTool(corpus_path=str(self.corpus_path))
        return asyncio.run(tool.execute(args, **kwargs))

    def test_scoped_retrieval(self):
        response, reward, meta = self._run(
            {"query": "revenue", "top_k": 3},
            tools_kwargs={"ticker": "ACME", "year": 2021},
        )
        self.assertIsNone(meta.get("error"))
        self.assertGreater(meta["retrieved_count"], 0)
        self.assertIn("ACME", response.text)
        self.assertNotIn("GAMMA", response.text)
        self.assertEqual(reward, 0.0)

    def test_scope_selects_correct_company(self):
        response, _, meta = self._run(
            {"query": "revenue"},
            tools_kwargs={"ticker": "GAMMA", "year": 2023},
        )
        self.assertIsNone(meta.get("error"))
        self.assertIn("90.0", response.text)

    def test_no_match_returns_error(self):
        _, reward, meta = self._run(
            {"query": "quantum entanglement"},
            tools_kwargs={"ticker": "ACME", "year": 2021},
        )
        self.assertEqual(meta.get("error"), "no_match")
        self.assertEqual(reward, -0.05)

    def test_empty_query_is_rejected(self):
        _, reward, meta = self._run({"query": "  "})
        self.assertEqual(meta.get("error"), "empty_query")
        self.assertEqual(reward, -0.05)

    def test_per_sample_corpus_path_override(self):
        override = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        with override:
            override.write(
                json.dumps(
                    {
                        "chunk_id": "zeta-1",
                        "doc_id": "ZETA/2024/page_1.pdf",
                        "ticker": "ZETA",
                        "year": 2024,
                        "kind": "table",
                        "text": "|  | 2024 |\n| Revenue | 42.0 |",
                    }
                )
                + "\n"
            )
        override_path = Path(override.name)
        try:
            tool = FinancialRetrieverTool(corpus_path=str(self.corpus_path))
            response, _, meta = asyncio.run(
                tool.execute(
                    {"query": "revenue"},
                    tools_kwargs={
                        "ticker": "ZETA",
                        "year": 2024,
                        "corpus_path": str(override_path),
                    },
                )
            )
            self.assertIn("42.0", response.text)
            self.assertEqual(meta["sources"], ["zeta-1"])
        finally:
            override_path.unlink(missing_ok=True)


class CalcTest(unittest.TestCase):
    def _run(self, code):
        tool = PythonFinancialCalcTool()
        return asyncio.run(tool.execute({"code": code}))

    def test_arithmetic(self):
        response, reward, meta = self._run("(100 - 80) / 80 * 100")
        self.assertEqual(reward, 0.0)
        self.assertIn("25", response.text)
        self.assertEqual(meta["result"], 25.0)

    def test_assignment_snippet(self):
        response, _, meta = self._run("growth = (120 - 100) / 100 * 100")
        self.assertIn("20", response.text)
        self.assertEqual(meta["result"], 20.0)

    def test_percent_symbol_is_rejected_with_hint(self):
        response, reward, meta = self._run("25%")
        self.assertEqual(reward, -0.05)
        self.assertEqual(meta["error"], "percent_symbol")
        self.assertIn("multiply by 100", response.text)

    def test_code_execution_is_blocked(self):
        _, reward, meta = self._run("__import__('os').system('echo hacked')")
        self.assertEqual(reward, -0.05)
        self.assertTrue(meta["error"].startswith("CalculationError"))

    def test_attribute_access_is_blocked(self):
        _, reward, meta = self._run("(1).__class__")
        self.assertEqual(reward, -0.05)
        self.assertTrue(meta["error"].startswith("CalculationError"))

    def test_huge_exponent_is_blocked(self):
        _, reward, meta = self._run("2 ** 1000")
        self.assertEqual(reward, -0.05)
        self.assertIn("Exponent", meta["error"])

    def test_unary_minus(self):
        _, _, meta = self._run("-5 + 10")
        self.assertEqual(meta["result"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
