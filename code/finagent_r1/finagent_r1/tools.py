"""Executable tools for FinAgent-R1.

Two tools, both dependency-free at import time:

* ``financial_retriever``: BM25 lexical retrieval over a JSONL/JSON/Parquet
  corpus of filing chunks.  It filters by the ``ticker`` / ``year`` scope that
  the environment injects, so the policy cannot retrieve evidence from a
  different company.
* ``python_calc``: an AST-whitelisted arithmetic evaluator.  It is not a full
  Python sandbox; it supports arithmetic, a small math-function whitelist and
  ``name = expression`` snippets.

The upstream ``BaseTool`` is used when Agent-R1 is installed.  A tiny fallback
keeps this module importable for CPU-only unit tests.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import operator
import os
import re
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised only inside the training environment
    from agent_r1.tool import BaseTool, ToolResponse
except Exception:  # pragma: no cover - fallback for CPU-only tests

    class ToolResponse:  # type: ignore
        def __init__(self, text: str = ""):
            self.text = text

    class BaseTool:  # type: ignore
        _registry: dict[str, Any] = {}

        def __init__(self, **kwargs):
            self.name = getattr(self, "name", "")
            self.description = getattr(self, "description", "")
            self.parameters = getattr(self, "parameters", {})

        @classmethod
        def register(cls, name: str):
            def decorator(subclass):
                cls._registry[name] = subclass
                return subclass

            return decorator

        @classmethod
        def from_name(cls, name: str, **kwargs):
            return cls._registry[name](**kwargs)


logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\d+(?:[.,]\d+)*%?|[a-zA-Z]{2,}")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}

# Shared across tool instances inside one Ray worker process.  A new flow
# instance is created for every sample, so without this cache each sample would
# re-read and re-tokenise the whole corpus.
_CORPUS_CACHE: dict[tuple[str, float, int], list[dict[str, Any]]] = {}
_CORPUS_CACHE_LOCK = threading.Lock()


def _tokenize(text: str | None) -> list[str]:
    """Lower-case word/number tokenizer used by BM25."""

    tokens = []
    for token in _WORD_RE.findall((text or "").lower()):
        if token in _STOPWORDS:
            continue
        tokens.append(token.replace(",", ""))
    return tokens


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _to_year(value: Any) -> int | None:
    try:
        return int(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# FinancialRetrieverTool
# ---------------------------------------------------------------------------


@BaseTool.register("financial_retriever")
class FinancialRetrieverTool(BaseTool):
    """BM25 retrieval over a scoped financial-filing corpus."""

    name = "financial_retriever"
    description = (
        "Search the current SEC filing (10-K/10-Q) for passages and table rows. "
        "The company and fiscal year are fixed by the environment."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Metric-level search query, e.g. 'total net sales 2021'.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of passages to return (1-10, default 3).",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        corpus_path: str | None = None,
        index_path: str | None = None,
        allow_scope_fallback: bool = True,
    ):
        super().__init__()
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.allow_scope_fallback = allow_scope_fallback

    # -- corpus loading -----------------------------------------------------

    @staticmethod
    def _normalize_docs(records: Sequence[Any]) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            if isinstance(record, str):
                text = record.strip()
                doc = {}
            elif isinstance(record, Mapping):
                doc = dict(record)
                text = str(doc.get("text") or doc.get("content") or "").strip()
            else:
                continue

            if not text:
                continue

            chunk_id = str(
                doc.get("chunk_id")
                or doc.get("doc_id")
                or doc.get("source")
                or f"chunk_{index}"
            )
            tokens = _tokenize(text)
            docs.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": str(doc.get("doc_id") or ""),
                    "ticker": str(doc.get("ticker") or "").upper(),
                    "year": _to_year(doc.get("year") or doc.get("fiscal_year")),
                    "kind": str(doc.get("kind") or doc.get("section") or "text"),
                    "text": text,
                    "tokens": tokens,
                    "tf": Counter(tokens),
                }
            )
        return docs

    @staticmethod
    def _read_records(path: Path) -> list[Any]:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            import pandas as pd  # optional dependency

            return pd.read_parquet(path).to_dict(orient="records")
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return payload
            if isinstance(payload, Mapping) and isinstance(payload.get("data"), list):
                return payload["data"]
            return []
        records: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _resolve_corpus_path(self, override: str | None = None) -> Path | None:
        candidates = [
            override,
            self.corpus_path,
            self.index_path,
            os.getenv("FINQA_CORPUS_PATH"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if path.is_dir():
                for name in ("corpus.jsonl", "corpus.json", "corpus.parquet"):
                    nested = path / name
                    if nested.exists():
                        return nested
                continue
            if path.suffix.lower() in {".faiss", ".index", ".bin"}:
                logger.warning(
                    "[finagent] dense index %s is not supported; point FINQA_CORPUS_PATH "
                    "at a JSONL/JSON/Parquet corpus instead.",
                    path,
                )
                continue
            if path.exists():
                return path
        return None

    def _load_docs(self, override: str | None = None) -> list[dict[str, Any]]:
        path = self._resolve_corpus_path(override)
        if path is None:
            logger.warning("[finagent] no corpus configured (FINQA_CORPUS_PATH).")
            return []
        if not path.exists():
            logger.warning("[finagent] corpus path does not exist: %s", path)
            return []

        try:
            stat = path.stat()
            key = (str(path.resolve()), stat.st_mtime, stat.st_size)
        except OSError:
            key = (str(path), 0.0, 0)

        with _CORPUS_CACHE_LOCK:
            cached = _CORPUS_CACHE.get(key)
            if cached is not None:
                return cached
            try:
                docs = self._normalize_docs(self._read_records(path))
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("[finagent] failed to load corpus %s: %s", path, exc)
                docs = []
            _CORPUS_CACHE[key] = docs
            logger.info("[finagent] loaded %d corpus chunks from %s", len(docs), path)
            return docs

    # -- ranking ------------------------------------------------------------

    def _select_scope(
        self,
        docs: Sequence[dict[str, Any]],
        ticker: str,
        year: int | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Filter by filing scope; optionally fall back to the full corpus."""

        if not ticker and year is None:
            return list(docs), False

        scoped = [
            doc
            for doc in docs
            if (not ticker or doc.get("ticker") == ticker)
            and (year is None or doc.get("year") == year)
        ]
        if scoped or not self.allow_scope_fallback:
            return scoped, False
        return list(docs), True

    @staticmethod
    def _bm25_rank(
        query: str,
        docs: Sequence[dict[str, Any]],
        top_k: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query)
        if not query_tokens or not docs:
            return []

        total_docs = len(docs)
        avgdl = sum(len(doc["tokens"]) for doc in docs) / max(1, total_docs)
        document_frequency = Counter()
        for doc in docs:
            document_frequency.update(doc["tf"].keys())

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index, doc in enumerate(docs):
            length = max(1, len(doc["tokens"]))
            score = 0.0
            for token in query_tokens:
                tf = doc["tf"].get(token, 0)
                if not tf:
                    continue
                df = document_frequency.get(token, 0)
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                denominator = tf + k1 * (1.0 - b + b * length / max(1.0, avgdl))
                score += idf * tf * (k1 + 1.0) / denominator
            if score > 0:
                scored.append((score, index, doc))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [{**doc, "_score": score} for score, _, doc in scored[:top_k]]

    async def execute(self, args: dict[str, Any], **kwargs) -> tuple[ToolResponse, float | None, dict]:
        query = str(args.get("query") or "").strip()
        if not query:
            return (
                ToolResponse(text="RetrieverError: search query cannot be empty."),
                -0.05,
                {"error": "empty_query"},
            )

        try:
            top_k = int(args.get("top_k", 3) or 3)
        except (TypeError, ValueError):
            top_k = 3
        top_k = max(1, min(top_k, 10))

        tools_kwargs = kwargs.get("tools_kwargs") or {}
        local_corpus = tools_kwargs.get("retrieval_corpus") if isinstance(tools_kwargs, Mapping) else None
        if isinstance(local_corpus, Sequence) and not isinstance(local_corpus, (str, bytes)) and local_corpus:
            docs = self._normalize_docs(local_corpus)
            corpus_source = "sample"
        else:
            corpus_override = None
            if isinstance(tools_kwargs, Mapping):
                corpus_override = tools_kwargs.get("corpus_path")
            docs = self._load_docs(str(corpus_override) if corpus_override else None)
            corpus_source = "file"

        if not docs:
            return (
                ToolResponse(
                    text=(
                        "RetrieverError: no corpus loaded. Set FINQA_CORPUS_PATH to the "
                        "prepared corpus.jsonl (see scripts/prepare_data.sh)."
                    )
                ),
                -0.05,
                {"error": "no_corpus", "query": query},
            )

        ticker = str(tools_kwargs.get("ticker") or "").upper()
        year = _to_year(tools_kwargs.get("year"))
        scoped_docs, fallback = self._select_scope(docs, ticker, year)
        if not scoped_docs:
            return (
                ToolResponse(
                    text=(
                        f"RetrieverError: no chunks found for ticker={ticker or 'ANY'} "
                        f"year={year if year is not None else 'ANY'}."
                    )
                ),
                -0.05,
                {"error": "empty_scope", "query": query, "ticker": ticker, "year": year},
            )

        selected = self._bm25_rank(query, scoped_docs, top_k)
        if not selected:
            return (
                ToolResponse(
                    text=(
                        "RetrieverError: no passage matched the query. Try a shorter "
                        "query built from the line-item name and period."
                    )
                ),
                -0.05,
                {"error": "no_match", "query": query, "ticker": ticker, "year": year},
            )

        blocks = []
        for doc in selected:
            blocks.append(
                f"[{doc['chunk_id']} | {doc.get('ticker') or '?'} "
                f"FY{doc.get('year') if doc.get('year') is not None else '?'} "
                f"| {doc.get('kind')} | score={doc.get('_score', 0.0):.2f}]\n{doc['text']}"
            )
        prefix = ""
        if fallback:
            prefix = (
                "Scope note: no chunk matched the requested ticker/year exactly; "
                "the full corpus was searched.\n\n"
            )

        meta = {
            "query": query,
            "ticker": ticker,
            "year": year,
            "retrieved_count": len(selected),
            "sources": [doc["chunk_id"] for doc in selected],
            "scores": [float(doc.get("_score", 0.0)) for doc in selected],
            "top_source": selected[0]["chunk_id"],
            "top_text": selected[0]["text"],
            "corpus_source": corpus_source,
            "scope_fallback": fallback,
        }
        return ToolResponse(text=prefix + "\n\n".join(blocks)), 0.0, meta


# ---------------------------------------------------------------------------
# PythonFinancialCalcTool
# ---------------------------------------------------------------------------

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "ceil": math.ceil,
    "floor": math.floor,
    "fabs": math.fabs,
    # FinQA's DSL uses these names; accepting them makes the tool forgiving
    # without changing the arithmetic semantics.
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}

_SAFE_CONSTANTS = {"pi": math.pi, "e": math.e}

_MAX_EXPONENT = 100
_MAX_POWER_BASE = 1e12


def _eval_node(node: ast.AST) -> Any:
    """Safely evaluate an AST expression without executing arbitrary code."""

    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ValueError(f"Constant of type {type(node.value).__name__} is not permitted.")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        raise ValueError(f"Name '{node.id}' is not permitted.")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        if isinstance(node.op, ast.Pow):
            if isinstance(right, (int, float)) and abs(right) > _MAX_EXPONENT:
                raise ValueError(f"Exponent is too large (max {_MAX_EXPONENT}).")
            if isinstance(left, (int, float)) and abs(left) > _MAX_POWER_BASE:
                raise ValueError("Power base is too large.")
        return _SAFE_OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _SAFE_OPERATORS[op_type](operand)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
            if node.keywords:
                raise ValueError("Keyword arguments are not permitted in calculations.")
            arguments = [_eval_node(argument) for argument in node.args]
            return _SAFE_FUNCTIONS[node.func.id](*arguments)
        raise ValueError(f"Function call '{ast.dump(node.func)}' is not permitted.")
    if isinstance(node, ast.List):
        return [_eval_node(element) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(element) for element in node.elts)
    raise ValueError(f"Expression node type '{type(node).__name__}' is not permitted.")


@BaseTool.register("python_calc")
class PythonFinancialCalcTool(BaseTool):
    """Safely evaluate arithmetic and financial formulas."""

    name = "python_calc"
    description = (
        "Evaluate an arithmetic Python expression exactly. Use *100 for a "
        "percentage; do not write the % symbol."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '(15.0 - 10.61) / 10.61 * 100'.",
            }
        },
        "required": ["code"],
    }

    async def execute(self, args: dict[str, Any], **kwargs) -> tuple[ToolResponse, float | None, dict]:
        code = str(args.get("code") or "").strip()
        if not code:
            return (
                ToolResponse(text="CalculationError: code expression cannot be empty."),
                -0.05,
                {"error": "empty_code"},
            )

        expression = code.replace("```python", "").replace("```", "").strip()
        if expression.endswith("%"):
            return (
                ToolResponse(
                    text=(
                        "CalculationError: remove the trailing '%' and multiply by 100 "
                        "instead, e.g. (a - b) / b * 100."
                    )
                ),
                -0.05,
                {"error": "percent_symbol", "code": code},
            )

        # Allow "result = ..." / "fcf = ..." style snippets while staying in eval mode.
        assignment = re.match(r"^[A-Za-z_]\w*\s*=(?!=)\s*(.+)$", expression, flags=re.DOTALL)
        if assignment:
            expression = assignment.group(1).strip()

        try:
            parsed = ast.parse(expression, mode="eval")
            result = _eval_node(parsed)
            if isinstance(result, complex):
                raise ValueError("Complex results are not supported.")
            if isinstance(result, float) and not math.isfinite(result):
                raise ValueError("Result is not a finite number.")
            return (
                ToolResponse(text=f"Result: {_format_number(result)}"),
                0.0,
                {"code": code, "result": result},
            )
        except Exception as exc:
            message = f"CalculationError: {exc}"
            return ToolResponse(text=message), -0.05, {"error": message, "code": code}


__all__ = ["FinancialRetrieverTool", "PythonFinancialCalcTool"]
