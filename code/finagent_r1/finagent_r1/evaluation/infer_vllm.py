#!/usr/bin/env python3
"""Run the FinQA agent (or a direct-answer baseline) with vLLM.

Examples
--------
Base-model zero-shot agent on the first 100 test rows::

    python -m finagent_r1.evaluation.infer_vllm \
        --model Qwen/Qwen2.5-3B-Instruct \
        --test_file data/finqa/test.parquet \
        --corpus data/finqa/corpus.jsonl \
        --limit 100 \
        --output results/qwen3b_base_test.jsonl

Trained checkpoint::

    python -m finagent_r1.evaluation.infer_vllm \
        --model outputs/finqa_grpo/huggingface \
        --test_file data/finqa/test.parquet \
        --corpus data/finqa/corpus.jsonl \
        --output results/qwen3b_grpo_test.jsonl

Direct-answer baseline (no tools)::

    python -m finagent_r1.evaluation.infer_vllm --mode direct ...

The evaluator reuses :func:`finagent_r1.core.run_episode`, so prompts, tool
scope, parsing and stop rules are identical to training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

from finagent_r1.core import EpisodeConfig, Generation, run_episode
from finagent_r1.evaluation.evaluate_predictions import aggregate, group_metrics
from finagent_r1.parsing import coerce_mapping
from finagent_r1.prompts import DIRECT_ANSWER_SYSTEM_PROMPT, FINQA_TOOL_SCHEMAS
from finagent_r1.rewards import score_answer
from finagent_r1.tools import FinancialRetrieverTool, PythonFinancialCalcTool


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-serializable values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prompt_content(row: dict[str, Any]) -> str:
    extra = coerce_mapping(row.get("extra_info"))
    question = str(extra.get("question") or "").strip()
    if question:
        return question
    prompt = row.get("prompt")
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, Sequence):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "").strip()
    return ""


def _row_extra(row: dict[str, Any]) -> dict[str, Any]:
    extra = coerce_mapping(row.get("extra_info"))
    reward_model = coerce_mapping(row.get("reward_model"))
    if "answer" not in extra and "ground_truth" in reward_model:
        extra["answer"] = reward_model["ground_truth"]
    return extra


def _apply_chat_template(tokenizer, messages: list[dict], tools: list[dict] | None) -> list[int]:
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = tokenizer.encode(text)
    if isinstance(ids, dict):
        ids = ids.get("input_ids", [])
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


class VLLMGenerator:
    """Synchronous vLLM wrapper exposed as an async callable."""

    def __init__(self, args):
        from vllm import LLM, SamplingParams

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        self.llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            dtype=args.dtype,
        )

    def generate(self, prompt_ids: list[int], _params: dict[str, Any] | None = None) -> Generation:
        try:
            outputs = self.llm.generate(
                prompts=[{"prompt_token_ids": prompt_ids}],
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
        except TypeError:
            outputs = self.llm.generate([prompt_ids], self.sampling_params)
        output = outputs[0].outputs[0]
        token_ids = list(getattr(output, "token_ids", []) or [])
        text = getattr(output, "text", None)
        return Generation(token_ids=token_ids, text=text)


async def _run_one(
    *,
    row: dict[str, Any],
    generator: VLLMGenerator,
    tokenizer,
    tools: list[Any],
    config: EpisodeConfig,
    mode: str,
    include_turns: bool,
) -> dict[str, Any]:
    extra = _row_extra(row)
    question = _prompt_content(row)
    ticker = str(extra.get("ticker") or "").upper()
    year = extra.get("year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None

    started = time.perf_counter()

    if mode == "direct":
        messages = [
            {"role": "system", "content": DIRECT_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt_ids = _apply_chat_template(tokenizer, messages, tools=None)
        generation = generator.generate(prompt_ids)
        response_text = generation.text or tokenizer.decode(
            generation.token_ids, skip_special_tokens=True
        )
        stats = {"step_count": 1, "tool_call_count": 0, "invalid_tool_calls": 0,
                 "tool_errors": 0, "retrieved_chunks": 0, "answer_found": 0}
        retrieval_total = 0
        retrieval_hits = 0
        turns_payload: list[dict[str, Any]] = []
    else:
        async def generate_fn(prompt_ids: list[int], params: dict[str, Any]) -> Generation:
            return generator.generate(prompt_ids, params)

        async def render_fn(messages: list[dict], schemas: list[dict] | None) -> list[int]:
            return _apply_chat_template(tokenizer, messages, schemas)

        result = await run_episode(
            question=question,
            ticker=ticker,
            year=year,
            generate_fn=generate_fn,
            render_fn=render_fn,
            decode_fn=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
            tools=tools,
            config=config,
            sampling_params={},
            tools_kwargs={"ticker": ticker, "year": year},
        )
        response_text = result.turns[-1].response_text
        stats = dict(result.stats)

        # Cheap retrieval recall against FinQA's gold evidence.  Strip table
        # pipes/punctuation, normalise whitespace, and compare the first 80
        # characters; this is a diagnostic, not a training reward.
        def _norm_evidence(text: Any) -> str:
            text = str(text).lower()
            text = re.sub(r"[^a-z0-9.%\- ]", " ", text)
            return re.sub(r"\s+", " ", text).strip()

        gold = extra.get("gold_evidence")
        gold = list(gold) if gold is not None else []
        gold_norm = [_norm_evidence(item) for item in gold if str(item).strip()]
        retrieval_text = " ".join(
            _norm_evidence(item.observation)
            for turn in result.turns
            for item in turn.tool_results
            if item.name == "financial_retriever"
        )
        retrieval_total = len(gold_norm)
        retrieval_hits = sum(1 for item in gold_norm if item[:80] in retrieval_text)

        turns_payload = (
            [
                {
                    "step": turn.step_index,
                    "response": turn.response_text,
                    "stop": turn.stop,
                    "tools": [
                        {
                            "name": item.name,
                            "arguments": item.arguments,
                            "ok": item.ok,
                            "observation": item.observation[:1200],
                        }
                        for item in turn.tool_results
                    ],
                }
                for turn in result.turns
            ]
            if include_turns
            else []
        )

    extra_for_score = dict(extra)
    extra_for_score.update(
        {
            "step_count": stats.get("step_count", 1),
            "invalid_tool_calls": stats.get("invalid_tool_calls", 0),
            "tool_errors": stats.get("tool_errors", 0),
        }
    )
    scored = score_answer(response_text, extra.get("answer", ""), extra_for_score)

    record = {
        "id": extra.get("question_id") or extra.get("id") or row.get("id"),
        "ticker": ticker,
        "year": year,
        "question": question,
        "prediction": scored["prediction"],
        "answer": extra.get("answer", ""),
        "exe_ans": extra.get("exe_ans"),
        "response": response_text,
        "solution": response_text,
        "extra_info": extra,
        "steps": stats.get("step_count", 1),
        "tool_call_count": stats.get("tool_call_count", 0),
        "invalid_tool_calls": stats.get("invalid_tool_calls", 0),
        "tool_errors": stats.get("tool_errors", 0),
        "retrieved_chunks": stats.get("retrieved_chunks", 0),
        "retrieval_hits": retrieval_hits,
        "retrieval_total": retrieval_total,
        "retrieval_recall": (
            round(retrieval_hits / retrieval_total, 4) if retrieval_total else None
        ),
        "answer_found": scored["answer_found"],
        "correct": scored["correct"],
        "score": scored["score"],
        "f1": scored["f1"],
        "format_reward": scored["format_reward"],
        "latency_s": round(time.perf_counter() - started, 3),
        "turns": turns_payload,
    }
    return record


def _load_done_ids(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("id") is not None:
                done.add(str(record["id"]))
    return done


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument(
        "--corpus",
        default=os.getenv("FINQA_CORPUS_PATH", "data/finqa/corpus.jsonl"),
    )
    parser.add_argument("--output", default="results/inference.jsonl")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--mode", choices=["agent", "direct"], default="agent")
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--subset", choices=["all", "seen_docs", "unseen_docs"], default="all")
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_turns", action="store_true")
    parser.add_argument("--progress_every", type=int, default=20)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    rows = _load_rows(Path(args.test_file))
    if args.subset != "all":
        wanted = args.subset == "unseen_docs"
        rows = [
            row
            for row in rows
            if bool(_row_extra(row).get("doc_seen_in_train")) is not wanted
        ]
    rows = rows[args.start_index :]
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    generator = VLLMGenerator(args)
    tools = [
        FinancialRetrieverTool(corpus_path=args.corpus),
        PythonFinancialCalcTool(),
    ]
    config = EpisodeConfig(
        max_steps=args.max_steps,
        prompt_length=args.max_prompt_length,
        response_length=args.max_tokens,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _load_done_ids(output_path) if args.resume else set()

    records: list[dict[str, Any]] = []
    started = time.time()
    with output_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            row_id = str(_row_extra(row).get("question_id") or _row_extra(row).get("id") or index)
            if row_id in done_ids:
                continue
            try:
                record = asyncio.run(
                    _run_one(
                        row=row,
                        generator=generator,
                        tokenizer=tokenizer,
                        tools=tools,
                        config=config,
                        mode=args.mode,
                        include_turns=not args.no_turns,
                    )
                )
            except Exception as exc:  # keep the run alive and record the failure
                record = {
                    "id": row_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "question": _prompt_content(row),
                    "correct": False,
                    "score": 0.0,
                    "f1": 0.0,
                    "answer_found": False,
                    "steps": 0,
                }
            record = _json_safe(record)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
                elapsed = time.time() - started
                rate = (index + 1) / max(1e-6, elapsed)
                print(
                    f"[{index + 1}/{len(rows)}] {rate:.2f} rows/s, "
                    f"elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )

    scored = [record for record in records if "error" not in record]
    summary = {
        "model": args.model,
        "mode": args.mode,
        "subset": args.subset,
        "test_file": args.test_file,
        "inference_errors": len(records) - len(scored),
        "summary": aggregate(scored),
        "by_ticker": group_metrics(scored, "ticker"),
    }
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # vLLM V1 engine teardown can hang after all output is flushed; exit hard
    # so evaluation shards release their GPU memory promptly.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
