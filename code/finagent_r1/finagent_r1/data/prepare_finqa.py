#!/usr/bin/env python3
"""Build the FinQA agent dataset: train/dev/test Parquet + a retrieval corpus.

Data source
-----------
FinQA (Chen et al., EMNLP 2021): 8,281 question-answer pairs over 2,789
financial reports from S&P 500 companies.  Official split: 6,251 train /
883 dev / 1,147 test.

This script accepts the official JSON files (``train.json``, ``dev.json``,
``test.json``) or a Hugging Face dataset.  It does **not** redistribute the
data; run it once on the Linux training machine:

    python -m finagent_r1.data.prepare_finqa --download --output_dir data/finqa

It produces::

    data/finqa/
      train.parquet / train.jsonl
      dev.parquet   / dev.jsonl
      test.parquet  / test.jsonl
      corpus.jsonl
      corpus_manifest.json
      dataset_summary.json

Each QA row follows the Agent-R1/verl schema (``prompt``, ``data_source``,
``reward_model``, ``extra_info``, ``agent_name``, ``env_kwargs``).  Each corpus
row is a chunk with ``chunk_id``, ``doc_id``, ``ticker``, ``year``, ``kind``
and ``text``.  Retrieval is scoped by ``ticker`` + ``year`` at query time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_URL_BASES = [
    "https://raw.githubusercontent.com/cszyssrs/FinQA/main/dataset",
    "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset",
    "https://raw.githubusercontent.com/czyssrs/FinQA/master/dataset",
]

# Common Hugging Face mirrors of FinQA.  The schema differs slightly between
# mirrors; ``flatten_example`` handles the optional ``qa`` nesting.
HF_FALLBACKS = ["dreamerdeo/finqa", "ChanceFocus/finqa"]

SPLIT_ALIASES = {
    "train": ["train"],
    "dev": ["dev", "validation", "valid"],
    "test": ["test"],
}

_SCOPE_RE = re.compile(r"([A-Za-z0-9.\-]{1,12})/(\d{4})/")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_json_payload(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "examples", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
    raise ValueError(f"Unrecognised FinQA JSON payload: {path}")


def _download_split(split: str, raw_dir: Path, url_bases: Sequence[str]) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{split}.json"
    if target.exists() and target.stat().st_size > 0:
        return target

    last_error: Exception | None = None
    for base in url_bases:
        url = f"{base.rstrip('/')}/{split}.json"
        try:
            print(f"Downloading {url}")
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "finagent-r1-data-prep/0.1"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) < 1000:
                raise ValueError(f"downloaded file is unexpectedly small ({len(data)} bytes)")
            target.write_bytes(data)
            print(f"Wrote {target} ({len(data) / 1e6:.1f} MB)")
            return target
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            print(f"  failed: {exc}", file=sys.stderr)
            time.sleep(1)

    raise RuntimeError(
        f"Could not download {split}.json. Download the FinQA repository manually "
        f"and pass --raw_dir pointing at its dataset/ directory. Last error: {last_error}"
    )


def _load_hf_split(
    dataset_name: str,
    config_name: str | None,
    split: str,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise RuntimeError("Install `datasets` to use Hugging Face: pip install datasets") from exc

    last_error: Exception | None = None
    for alias in SPLIT_ALIASES.get(split, [split]):
        try:
            dataset = load_dataset(dataset_name, config_name, split=alias)
            print(f"  using split '{alias}' for '{split}'")
            return expand_examples([dict(row) for row in dataset])
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load {dataset_name} split {split}: {last_error}")


def _load_from_hf(
    dataset_names: Sequence[str],
    config_name: str | None,
) -> dict[str, list[dict[str, Any]]] | None:
    for name in dataset_names:
        try:
            splits = {
                split: _load_hf_split(name, config_name, split)
                for split in ("train", "dev", "test")
            }
            print(f"Loaded Hugging Face dataset {name}")
            return splits
        except Exception as exc:
            print(f"Hugging Face dataset {name} failed: {exc}", file=sys.stderr)
    return None


def flatten_example(example: dict[str, Any]) -> dict[str, Any]:
    """Flatten the optional ``qa`` nesting used by some HF mirrors."""

    flat = dict(example)
    qa = example.get("qa")
    if isinstance(qa, dict):
        flat.update(qa)
    return flat


def expand_examples(examples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand mirrors that store a *list* of QA dicts under ``qa``.

    The official FinQA JSON is one QA per row.  Some Hugging Face mirrors pack
    several questions for one filing into ``qa: [...]``; flattening them keeps
    the rest of the pipeline unchanged.
    """

    expanded: list[dict[str, Any]] = []
    for example in examples:
        qa = example.get("qa")
        if isinstance(qa, list) and qa:
            for item in qa:
                if not isinstance(item, dict):
                    continue
                merged = dict(example)
                merged.pop("qa", None)
                merged.update(item)
                expanded.append(merged)
        else:
            expanded.append(example)
    return expanded


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return _as_text_list(parsed)
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in value.split("\n") if part.strip()] or [value.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _as_table(value: Any) -> list[list[str]]:
    if value is None:
        return []
    if isinstance(value, dict):
        header = value.get("header")
        rows = value.get("rows")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            table = []
            if isinstance(header, Sequence) and not isinstance(header, (str, bytes)):
                table.append([str(cell).strip() for cell in header])
            for row in rows:
                if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                    table.append([str(cell).strip() for cell in row])
                else:
                    table.append([str(row).strip()])
            if table:
                return table
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return _as_table(parsed)
            except json.JSONDecodeError:
                pass
        rows = []
        for line in value.splitlines():
            line = line.strip()
            if not line:
                continue
            cells = [cell.strip() for cell in re.split(r"\s*\|\s*|\t", line)]
            if cells:
                rows.append(cells)
        return rows
    rows: list[list[str]] = []
    for row in value:
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
            rows.append([str(cell).strip() for cell in row])
        else:
            rows.append([str(row).strip()])
    return rows


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("$", "")
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number / 100.0 if percent else number


def parse_scope(example: dict[str, Any]) -> tuple[str, int | None, str]:
    """Return ``(ticker, year, doc_id)`` parsed from filename/id."""

    filename = str(example.get("filename") or "").strip()
    qid = str(example.get("id") or example.get("question_id") or "").strip()

    source = filename or qid
    match = _SCOPE_RE.search(source)
    ticker = match.group(1).upper() if match else ""
    year = int(match.group(2)) if match else None

    if filename:
        doc_id = filename
    else:
        # "Single_JPM/2018/page_224.pdf-2" -> "Single_JPM/2018/page_224.pdf"
        doc_id = qid.rsplit("-", 1)[0] if "-" in qid else qid
    if not doc_id:
        digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:10]
        doc_id = f"doc_{digest}"
    return ticker, year, doc_id


def _gold_evidence(example: dict[str, Any]) -> list[str]:
    gold = example.get("gold_inds")
    if isinstance(gold, str):
        stripped = gold.strip()
        if stripped[:1] in {"{", "["}:
            try:
                gold = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    if isinstance(gold, dict):
        return [str(value).strip() for value in gold.values() if str(value).strip()]
    if isinstance(gold, Sequence) and not isinstance(gold, (str, bytes)):
        return [str(value).strip() for value in gold if str(value).strip()]
    if isinstance(gold, str) and gold.strip():
        return [gold.strip()]
    return []


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def chunk_text(text: str, max_chars: int = 700) -> list[str]:
    """Split a paragraph into chunks of at most ``max_chars`` characters."""

    text = _WS_RE.sub(" ", text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())

    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
            continue
        for start in range(0, len(chunk), max_chars):
            result.append(chunk[start : start + max_chars].strip())
    return [chunk for chunk in result if chunk]


def render_table(table: Sequence[Sequence[str]], header: Sequence[str] | None = None) -> str:
    """Render a table as pipe-separated rows with a repeated header."""

    lines = []
    if header:
        lines.append("| " + " | ".join(header) + " |")
    for row in table:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def table_chunks(table: Sequence[Sequence[str]], rows_per_chunk: int = 10, max_chars: int = 2000) -> list[str]:
    if not table:
        return []
    header = table[0]
    body = table[1:] if len(table) > 1 else table
    chunks: list[str] = []
    for start in range(0, len(body), max(1, rows_per_chunk)):
        block = body[start : start + rows_per_chunk]
        rendered = render_table(block, header=header)
        if len(rendered) > max_chars:
            rendered = rendered[:max_chars]
        chunks.append(rendered)
    return chunks


def build_corpus(
    examples: Iterable[dict[str, Any]],
    max_chars: int = 700,
    rows_per_chunk: int = 10,
) -> list[dict[str, Any]]:
    """Build a deduplicated chunk corpus from every unique filing document."""

    docs: dict[str, dict[str, Any]] = {}
    for raw in examples:
        example = flatten_example(raw)
        ticker, year, doc_id = parse_scope(example)
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "ticker": ticker,
                "year": year,
                "pre_text": _as_text_list(example.get("pre_text")),
                "post_text": _as_text_list(example.get("post_text")),
                "table": _as_table(example.get("table")),
            }

    chunks: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(doc_id: str, ticker: str, year: int | None, kind: str, text: str) -> None:
        text = _WS_RE.sub(" ", text).strip()
        if not text:
            return
        key = (doc_id, text)
        if key in seen:
            return
        seen.add(key)
        chunks.append(
            {
                "chunk_id": f"{doc_id}::{kind}::{len(chunks):06d}",
                "doc_id": doc_id,
                "ticker": ticker,
                "year": year,
                "kind": kind,
                "text": text,
            }
        )

    for doc in docs.values():
        for paragraph in doc["pre_text"]:
            for chunk in chunk_text(paragraph, max_chars=max_chars):
                add(doc["doc_id"], doc["ticker"], doc["year"], "pre_text", chunk)

        for chunk in table_chunks(doc["table"], rows_per_chunk=rows_per_chunk):
            add(doc["doc_id"], doc["ticker"], doc["year"], "table", chunk)

        for paragraph in doc["post_text"]:
            for chunk in chunk_text(paragraph, max_chars=max_chars):
                add(doc["doc_id"], doc["ticker"], doc["year"], "post_text", chunk)

    return chunks


# ---------------------------------------------------------------------------
# QA rows
# ---------------------------------------------------------------------------


def qa_row(example: dict[str, Any], split: str) -> dict[str, Any]:
    """Convert one FinQA example to an Agent-R1 dataset row."""

    example = flatten_example(example)
    ticker, year, doc_id = parse_scope(example)

    question = str(example.get("question") or example.get("query") or "").strip()
    answer = example.get("answer", example.get("exe_ans", ""))
    answer_str = str(answer).strip() if answer is not None else ""
    exe_ans_raw = example.get("exe_ans")
    exe_ans = _as_float(exe_ans_raw if exe_ans_raw is not None else answer)

    program = example.get("program", [])
    if isinstance(program, str):
        # Official FinQA stores programs as one string, e.g.
        # "subtract(5829, 5735), divide(#0, 5735)".  Split on the top-level
        # "), " separator so the step histogram reflects real program length.
        parts = [part.strip() for part in re.split(r"\),\s*", program.strip()) if part.strip()]
        program = [part if part.endswith(")") else part + ")" for part in parts]
    elif not isinstance(program, Sequence):
        program = []

    qid = str(example.get("id") or example.get("question_id") or f"{split}_{doc_id}")

    # The prompt stays as the clean user question.  The filing scope lives in
    # extra_info and is injected once by the agent flow
    # (core.build_user_content -> "Filing scope: TICKER, fiscal_year YYYY").
    content = question

    extra_info = {
        "index": None,  # filled by the writer
        "question_id": qid,
        "question": question,
        "answer": answer_str,
        "exe_ans": exe_ans,
        "is_numeric": exe_ans is not None,
        "program": [str(step) for step in program],
        "gold_evidence": _gold_evidence(example),
        "ticker": ticker,
        "year": year,
        "doc_id": doc_id,
        "split": split,
        "doc_seen_in_train": False,
    }

    return {
        "data_source": "finqa",
        "prompt": [{"role": "user", "content": content}],
        "agent_name": "finqa_agent",
        "reward_model": {"ground_truth": answer_str, "style": "rule"},
        "extra_info": extra_info,
        "env_kwargs": json.dumps(
            {"tools_kwargs": {"ticker": ticker, "year": year}},
            ensure_ascii=False,
        ),
    }


def build_qa_rows(examples: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows = [qa_row(example, split) for example in examples]
    rows = [row for row in rows if row["extra_info"]["question"]]
    seen: Counter = Counter()
    for index, row in enumerate(rows):
        row["extra_info"]["index"] = index
        qid = str(row["extra_info"]["question_id"])
        seen[qid] += 1
        if seen[qid] > 1:
            # HF mirrors that pack several QAs under one filing can reuse the
            # parent id; make every row id unique for resumable evaluation.
            row["extra_info"]["question_id"] = f"{qid}#{seen[qid]}"
    return rows


def mark_document_overlap(train_rows: list[dict[str, Any]], other_rows: list[dict[str, Any]]) -> None:
    """Mark rows whose filing document also appears in the training split."""

    train_docs = {row["extra_info"]["doc_id"] for row in train_rows}
    for row in other_rows:
        row["extra_info"]["doc_seen_in_train"] = row["extra_info"]["doc_id"] in train_docs


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def write_split(rows: Sequence[dict[str, Any]], output_dir: Path, split: str, parquet: bool = True) -> None:
    write_jsonl(output_dir / f"{split}.jsonl", rows)
    if not parquet:
        return
    try:
        import pandas as pd

        pd.DataFrame(list(rows)).to_parquet(output_dir / f"{split}.parquet", index=False)
        print(f"Wrote {output_dir / f'{split}.parquet'}")
    except Exception as exc:
        print(
            f"WARNING: could not write {split}.parquet ({exc}). "
            "Install pyarrow (pip install pyarrow) and rerun; Agent-R1 training "
            "expects parquet files.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_raw_splits(
    raw_dir: Path,
    download: bool,
    url_bases: Sequence[str],
    hf_dataset: str | None = None,
    hf_config: str | None = None,
    source: str = "auto",
) -> dict[str, list[dict[str, Any]]]:
    """Load the three official splits from local JSON, Hugging Face, or GitHub.

    ``source`` is one of ``auto`` (HF first, then GitHub), ``hf`` or
    ``github``.  Local JSON files always win when they exist.
    """

    local = {split: raw_dir / f"{split}.json" for split in ("train", "dev", "test")}
    if all(path.exists() for path in local.values()):
        print(f"Using local FinQA JSON files in {raw_dir}")
        return {
            split: expand_examples(_read_json_payload(path))
            for split, path in local.items()
        }

    if hf_dataset or (download and source in {"auto", "hf"}):
        names = [hf_dataset] if hf_dataset else HF_FALLBACKS
        splits = _load_from_hf(names, hf_config)
        if splits is not None:
            return splits
        if source == "hf":
            raise RuntimeError(
                "Hugging Face download failed. Try setting HF_ENDPOINT=https://hf-mirror.com, "
                "or download the FinQA JSON files manually and pass --raw_dir."
            )

    if not download and source != "github":
        raise FileNotFoundError(
            f"Missing local FinQA JSON files in {raw_dir}. Pass --download, "
            "--raw_dir pointing at FinQA's dataset/ directory, or --hf_dataset."
        )

    splits = {}
    for split in ("train", "dev", "test"):
        path = local[split]
        if not path.exists():
            path = _download_split(split, raw_dir, url_bases)
        splits[split] = expand_examples(_read_json_payload(path))
    return splits


def build_dataset(
    raw_splits: dict[str, list[dict[str, Any]]],
    *,
    max_train: int = 0,
    max_dev: int = 0,
    max_test: int = 0,
    seed: int = 42,
    corpus_max_chars: int = 700,
    table_rows_per_chunk: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Pure function used by both the CLI and the unit tests."""

    rng = random.Random(seed)

    def sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if limit <= 0 or limit >= len(rows):
            return list(rows)
        indices = sorted(rng.sample(range(len(rows)), limit))
        return [rows[index] for index in indices]

    train_raw = sample(raw_splits.get("train", []), max_train)
    dev_raw = sample(raw_splits.get("dev", []), max_dev)
    test_raw = sample(raw_splits.get("test", []), max_test)

    # The corpus is built from *all* raw examples, even when the QA rows are
    # subsampled for a smoke run.  Otherwise a sampled test question could
    # reference a filing that is absent from the retrieval corpus.
    all_raw = (
        list(raw_splits.get("train", []))
        + list(raw_splits.get("dev", []))
        + list(raw_splits.get("test", []))
    )
    corpus = build_corpus(
        all_raw,
        max_chars=corpus_max_chars,
        rows_per_chunk=table_rows_per_chunk,
    )

    rows = {
        "train": build_qa_rows(train_raw, "train"),
        "dev": build_qa_rows(dev_raw, "dev"),
        "test": build_qa_rows(test_raw, "test"),
    }
    mark_document_overlap(rows["train"], rows["dev"])
    mark_document_overlap(rows["train"], rows["test"])
    return corpus, rows


def summarize(
    corpus: Sequence[dict[str, Any]],
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset": "FinQA (Chen et al., EMNLP 2021)",
        "source_note": (
            "Counts below describe the files produced by this run. They are not "
            "experimental results."
        ),
        "splits": {},
        "corpus": {
            "chunks": len(corpus),
            "unique_docs": len({chunk["doc_id"] for chunk in corpus}),
            "chunks_with_scope": sum(1 for chunk in corpus if chunk.get("ticker") and chunk.get("year")),
            "kinds": dict(Counter(chunk["kind"] for chunk in corpus)),
        },
    }

    for split, rows in rows_by_split.items():
        program_lengths = Counter(
            min(4, len(row["extra_info"].get("program") or []))
            for row in rows
        )
        summary["splits"][split] = {
            "rows": len(rows),
            "unique_docs": len({row["extra_info"]["doc_id"] for row in rows}),
            "unique_tickers": len({row["extra_info"]["ticker"] for row in rows if row["extra_info"]["ticker"]}),
            "numeric_rows": sum(1 for row in rows if row["extra_info"]["is_numeric"]),
            "doc_seen_in_train_rows": sum(
                1 for row in rows if row["extra_info"].get("doc_seen_in_train")
            ),
            "program_length_histogram": {
                ("4+" if key == 4 else str(key)): value
                for key, value in sorted(program_lengths.items())
            },
            "mean_question_chars": round(
                sum(len(row["extra_info"]["question"]) for row in rows) / max(1, len(rows)), 1
            ),
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/finqa")
    parser.add_argument("--raw_dir", default=None, help="Directory containing train.json/dev.json/test.json")
    parser.add_argument("--download", action="store_true", help="Download FinQA if local files are absent")
    parser.add_argument("--source", choices=["auto", "hf", "github"], default="auto",
                        help="Download source: Hugging Face first, GitHub first, or both")
    parser.add_argument("--raw_url_base", default=None, help="Override the GitHub raw URL base")
    parser.add_argument("--hf_dataset", default=None, help="Optional Hugging Face dataset name")
    parser.add_argument("--hf_config", default=None)
    parser.add_argument("--max_train", type=int, default=0, help="0 = all")
    parser.add_argument("--max_dev", type=int, default=0)
    parser.add_argument("--max_test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus_max_chars", type=int, default=700)
    parser.add_argument("--table_rows_per_chunk", type=int, default=10)
    parser.add_argument("--no_parquet", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir else output_dir / "raw"
    url_bases = [args.raw_url_base] if args.raw_url_base else DEFAULT_URL_BASES

    raw_splits = load_raw_splits(
        raw_dir,
        download=args.download,
        url_bases=url_bases,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        source=args.source,
    )

    corpus, rows_by_split = build_dataset(
        raw_splits,
        max_train=args.max_train,
        max_dev=args.max_dev,
        max_test=args.max_test,
        seed=args.seed,
        corpus_max_chars=args.corpus_max_chars,
        table_rows_per_chunk=args.table_rows_per_chunk,
    )

    corpus_path = output_dir / "corpus.jsonl"
    write_jsonl(corpus_path, corpus)
    corpus_abs = str(corpus_path.resolve())
    for rows in rows_by_split.values():
        for row in rows:
            # Carry the absolute corpus path in the sample itself.  This keeps
            # retrieval working even if a Ray worker does not inherit
            # FINQA_CORPUS_PATH from the driver environment.
            row["extra_info"]["corpus_path"] = corpus_abs

    for split, rows in rows_by_split.items():
        write_split(rows, output_dir, split, parquet=not args.no_parquet)

    summary = summarize(corpus, rows_by_split)
    safe_summary = _json_safe(summary)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(safe_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "corpus_manifest.json").write_text(
        json.dumps(safe_summary["corpus"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    test_stats = summary["splits"].get("test", {})
    if test_stats.get("rows") and not test_stats.get("numeric_rows"):
        print(
            "WARNING: the test split has no numeric labels (exe_ans missing). "
            "Use dev as the held-out evaluation set, or verify the source files.",
            file=sys.stderr,
        )
    print(f"\nOutputs written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
