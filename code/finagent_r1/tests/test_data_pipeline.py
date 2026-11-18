"""Tests for the FinQA -> Agent-R1 data conversion."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finagent_r1.data.prepare_finqa import (  # noqa: E402
    _as_table,
    _read_json_payload,
    build_corpus,
    build_dataset,
    chunk_text,
    expand_examples,
    parse_scope,
    qa_row,
    table_chunks,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "finqa_raw"


def _raw_splits():
    return {
        split: _read_json_payload(_FIXTURES / f"{split}.json")
        for split in ("train", "dev", "test")
    }


class ScopeTest(unittest.TestCase):
    def test_parse_scope_from_filename(self):
        ticker, year, doc_id = parse_scope(
            {"filename": "ACME/2021/page_10.pdf", "id": "Single_ACME/2021/page_10.pdf-3"}
        )
        self.assertEqual(ticker, "ACME")
        self.assertEqual(year, 2021)
        self.assertEqual(doc_id, "ACME/2021/page_10.pdf")

    def test_parse_scope_from_id(self):
        ticker, year, doc_id = parse_scope({"id": "Double_GAMMA/2023/page_5.pdf-1"})
        self.assertEqual(ticker, "GAMMA")
        self.assertEqual(year, 2023)
        self.assertTrue(doc_id)


class ChunkTest(unittest.TestCase):
    def test_chunk_text_respects_budget(self):
        text = "Sentence one. " * 200
        chunks = chunk_text(text, max_chars=120)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))

    def test_table_chunks_repeat_header(self):
        table = [["", "2021", "2020"]] + [["Revenue", str(i), str(i - 1)] for i in range(25)]
        chunks = table_chunks(table, rows_per_chunk=10)
        self.assertGreaterEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertTrue(chunk.startswith("|  | 2021 | 2020 |"))

    def test_table_dict_and_json_string(self):
        table = {"header": ["", "2021"], "rows": [["Revenue", "100"]]}
        self.assertEqual(_as_table(table), [["", "2021"], ["Revenue", "100"]])
        self.assertEqual(
            _as_table('[["", "2021"], ["Revenue", "100"]]'),
            [["", "2021"], ["Revenue", "100"]],
        )

    def test_expand_qa_list(self):
        examples = [
            {
                "filename": "ACME/2021/page_1.pdf",
                "pre_text": ["x"],
                "qa": [
                    {"id": "q1", "question": "one?", "answer": "1"},
                    {"id": "q2", "question": "two?", "answer": "2"},
                ],
            }
        ]
        expanded = expand_examples(examples)
        self.assertEqual([row["id"] for row in expanded], ["q1", "q2"])
        self.assertEqual(expanded[0]["filename"], "ACME/2021/page_1.pdf")


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus, cls.rows = build_dataset(_raw_splits())

    def test_split_sizes(self):
        self.assertEqual(len(self.rows["train"]), 4)
        self.assertEqual(len(self.rows["dev"]), 1)
        self.assertEqual(len(self.rows["test"]), 2)

    def test_corpus_scopes(self):
        docs = {chunk["doc_id"] for chunk in self.corpus}
        self.assertEqual(len(docs), 4)
        self.assertTrue(all(chunk["ticker"] for chunk in self.corpus))
        self.assertTrue(all(chunk["year"] for chunk in self.corpus))

    def test_agent_r1_schema(self):
        row = self.rows["train"][0]
        self.assertEqual(row["data_source"], "finqa")
        self.assertEqual(row["agent_name"], "finqa_agent")
        self.assertIn("ground_truth", row["reward_model"])
        self.assertEqual(row["prompt"][0]["role"], "user")
        self.assertEqual(row["prompt"][0]["content"], row["extra_info"]["question"])
        self.assertEqual(row["extra_info"]["ticker"], "ACME")
        env_kwargs = json.loads(row["env_kwargs"])
        self.assertIn("ticker", env_kwargs["tools_kwargs"])

    def test_document_overlap_flags(self):
        test_rows = self.rows["test"]
        by_ticker = {row["extra_info"]["ticker"]: row for row in test_rows}
        self.assertTrue(by_ticker["ACME"]["extra_info"]["doc_seen_in_train"])
        self.assertFalse(by_ticker["GAMMA"]["extra_info"]["doc_seen_in_train"])

    def test_numeric_metadata(self):
        row = self.rows["train"][0]
        self.assertTrue(row["extra_info"]["is_numeric"])
        self.assertIsInstance(row["extra_info"]["exe_ans"], float)
        self.assertTrue(row["extra_info"]["program"])

    def test_qa_row_falls_back_to_exe_ans(self):
        row = qa_row(
            {
                "id": "Single_TEST/2020/page_1.pdf-1",
                "filename": "TEST/2020/page_1.pdf",
                "question": "What is 1 + 1?",
                "exe_ans": 2.0,
            },
            "test",
        )
        self.assertEqual(row["extra_info"]["answer"], "2.0")
        self.assertEqual(row["extra_info"]["exe_ans"], 2.0)
        self.assertTrue(row["extra_info"]["is_numeric"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
