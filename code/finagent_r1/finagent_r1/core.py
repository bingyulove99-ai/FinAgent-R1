"""Framework-agnostic multi-turn episode loop.

``run_episode`` owns the control flow that is shared by training and offline
evaluation:

1. build the state the policy is allowed to see (scratchpad + sliding
   observation window + feedback);
2. call the model through an injected async ``generate_fn``;
3. parse tool calls / the final answer;
4. execute tools and update the scratchpad;
5. stop on an answer, on a protocol dead-end, or at ``max_steps``.

The function never imports ``verl`` or ``agent_r1``.  The training adapter
(``flow.py``) injects an async server manager; the evaluator injects vLLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

from finagent_r1.parsing import (
    ParsedAction,
    build_user_content,
    malformed_tool_call_feedback,
    parse_action,
)
from finagent_r1.prompts import FINQA_SYSTEM_PROMPT, FINQA_TOOL_SCHEMAS

logger = logging.getLogger(__name__)


class EpisodeError(RuntimeError):
    """Raised when a trajectory cannot produce a single usable turn."""


@dataclass
class EpisodeConfig:
    """Rollout controls shared by training and evaluation."""

    max_steps: int = 6
    max_parallel_calls: int = 2
    observation_window: int = 2
    prompt_length: int = 4096
    response_length: int = 768
    max_obs_chars: int = 6000
    max_observation_chars: int = 3500
    max_scratchpad_items: int = 8
    max_history_actions: int = 10
    context_mode: str = "scratchpad_window"  # or "full" for the ablation


@dataclass
class Generation:
    """A single model generation."""

    token_ids: list[int]
    logprobs: Optional[list[float]] = None
    text: Optional[str] = None
    routed_experts: Any = None


@dataclass
class ToolExecution:
    """The environment's view of one tool call."""

    name: str
    arguments: dict[str, Any]
    observation: str
    meta: dict[str, Any]
    ok: bool
    label: str
    latency: float = 0.0


@dataclass
class TurnRecord:
    """Everything needed to build one ``AgentFlowStep``."""

    step_index: int
    prompt_ids: list[int]
    response_ids: list[int]
    response_text: str
    response_logprobs: Optional[list[float]]
    routed_experts: Any
    action: ParsedAction
    tool_results: list[ToolExecution]
    stop: bool
    stats: dict[str, int]
    latency_generate: float = 0.0
    latency_tools: float = 0.0


@dataclass
class EpisodeResult:
    """A complete multi-step trajectory."""

    turns: list[TurnRecord]
    stats: dict[str, int]
    final_answer: str
    scratchpad: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


async def _render_prompt(
    *,
    question: str,
    ticker: str,
    year: int | None,
    scratchpad: Sequence[str],
    history_actions: Sequence[str],
    observations: Sequence[str],
    tool_feedback: str,
    step_index: int,
    config: EpisodeConfig,
    render_fn: Callable[[list[dict], list[dict] | None], Awaitable[list[int]]],
    stats: dict[str, int],
) -> list[int]:
    """Render the next prompt, shrinking context before truncating tokens."""

    max_obs_chars = config.max_obs_chars
    scratch_limit = config.max_scratchpad_items
    history_limit = config.max_history_actions
    window = 1000 if config.context_mode == "full" else config.observation_window

    prompt_ids: list[int] = []
    while True:
        user_content = build_user_content(
            query=question,
            ticker=ticker,
            fiscal_year=year,
            scratchpad=list(scratchpad)[-scratch_limit:],
            history_actions=list(history_actions)[-history_limit:],
            observations=observations,
            tool_feedback=tool_feedback,
            step_index=step_index,
            max_steps=config.max_steps,
            observation_window=window,
            max_obs_chars=max_obs_chars,
        )
        messages = [
            {"role": "system", "content": FINQA_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt_ids = await render_fn(messages, FINQA_TOOL_SCHEMAS)
        if len(prompt_ids) <= config.prompt_length:
            return prompt_ids

        if max_obs_chars > 400:
            max_obs_chars = max(400, int(max_obs_chars * 0.7))
            continue
        if scratch_limit > 4:
            scratch_limit -= 2
            continue
        if history_limit > 4:
            history_limit -= 2
            continue
        break

    # Last resort: keep the system/tool preamble and the generation suffix,
    # dropping the middle of the conversation.  This is rare and logged.
    keep_head = min(len(prompt_ids), max(128, config.prompt_length // 4))
    keep_tail = max(0, config.prompt_length - keep_head)
    stats["context_truncated"] = 1
    logger.warning(
        "[finagent][step=%d] prompt=%d > budget=%d; middle truncated.",
        step_index,
        len(prompt_ids),
        config.prompt_length,
    )
    tail = prompt_ids[-keep_tail:] if keep_tail > 0 else []
    return prompt_ids[:keep_head] + tail


def _tool_label(name: str, arguments: dict[str, Any]) -> str:
    if name == "financial_retriever":
        return f"Retriever Result for '{arguments.get('query', '')}'"
    if name == "python_calc":
        return f"Calculation Result for '{arguments.get('code', '')}'"
    return "Tool Error"


async def _execute_tool(
    name: str,
    arguments: dict[str, Any],
    tools_by_name: dict[str, Any],
    tools_kwargs: dict[str, Any],
) -> ToolExecution:
    tool = tools_by_name.get(name)
    if tool is None:
        return ToolExecution(
            name=name,
            arguments=arguments,
            observation=(
                f"Unknown tool: '{name}'. Available tools: "
                "financial_retriever, python_calc."
            ),
            meta={"error": "unknown_tool"},
            ok=False,
            label="Tool Error",
        )

    start = time.perf_counter()
    try:
        response, _reward, meta = await tool.execute(arguments, tools_kwargs=tools_kwargs)
        text = (getattr(response, "text", None) or "").strip()
        meta = meta if isinstance(meta, dict) else {}
        ok = not bool(meta.get("error")) and bool(text)
        if not text:
            text = "Tool returned an empty response."
    except Exception as exc:  # keep the rollout alive on a broken tool
        text = f"ToolError: {type(exc).__name__}: {exc}"
        meta = {"error": "exception"}
        ok = False

    return ToolExecution(
        name=name,
        arguments=arguments,
        observation=text,
        meta=meta,
        ok=ok,
        label=_tool_label(name, arguments),
        latency=time.perf_counter() - start,
    )


def _update_scratchpad(scratchpad: list[str], result: ToolExecution, limit: int) -> None:
    if not result.ok:
        return

    if result.name == "python_calc" and "result" in result.meta:
        entry = f"calc: {result.arguments.get('code')} = {result.meta['result']}"
    elif result.name == "financial_retriever":
        source = result.meta.get("top_source") or "evidence"
        excerpt = " ".join(str(result.meta.get("top_text") or "").split())[:280]
        if not excerpt:
            return
        entry = f"[{source}] {excerpt}"
    else:
        return

    if not scratchpad or scratchpad[-1] != entry:
        scratchpad.append(entry)
    del scratchpad[:-max(1, limit)]


async def run_episode(
    *,
    question: str,
    ticker: str,
    year: int | None,
    generate_fn: Callable[[list[int], dict[str, Any]], Awaitable[Generation]],
    render_fn: Callable[[list[dict], list[dict] | None], Awaitable[list[int]]],
    decode_fn: Callable[[list[int]], str],
    tools: Sequence[Any],
    config: EpisodeConfig,
    sampling_params: Optional[dict[str, Any]] = None,
    tools_kwargs: Optional[dict[str, Any]] = None,
) -> EpisodeResult:
    """Run one multi-turn financial-QA trajectory.

    Args:
        question: The user question.
        ticker: Company ticker used to scope retrieval.
        year: Fiscal year used to scope retrieval.
        generate_fn: ``async (prompt_ids, sampling_params) -> Generation``.
        render_fn: ``async (messages, tool_schemas) -> prompt_ids``.
        decode_fn: ``token_ids -> text``.
        tools: Tool objects exposing ``.name`` and ``.execute(args, **kwargs)``.
        config: Rollout controls.
        sampling_params: Forwarded to ``generate_fn`` (temperature, top_p, ...).
        tools_kwargs: Per-sample environment payload (scope, corpus override).

    Returns:
        ``EpisodeResult`` with one ``TurnRecord`` per LLM call.

    Raises:
        EpisodeError: If the model server returns no token for the first step.
    """

    if config.max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    tools_by_name = {getattr(tool, "name", ""): tool for tool in tools}
    tools_kwargs = dict(tools_kwargs or {})
    sampling_params = dict(sampling_params or {})

    scratchpad: list[str] = []
    history_actions: list[str] = []
    observations: list[str] = []
    tool_feedback = ""

    stats: dict[str, int] = {
        "step_count": 0,
        "invalid_tool_calls": 0,
        "tool_errors": 0,
        "tool_call_count": 0,
        "retriever_calls": 0,
        "calc_calls": 0,
        "retrieved_chunks": 0,
        "answer_found": 0,
        "empty_generation": 0,
        "context_truncated": 0,
    }

    turns: list[TurnRecord] = []
    total_generate = 0.0
    total_tools = 0.0

    for step_idx in range(config.max_steps):
        is_last_step = step_idx == config.max_steps - 1

        prompt_ids = await _render_prompt(
            question=question,
            ticker=ticker,
            year=year,
            scratchpad=scratchpad,
            history_actions=history_actions,
            observations=observations,
            tool_feedback=tool_feedback,
            step_index=step_idx + 1,
            config=config,
            render_fn=render_fn,
            stats=stats,
        )

        start = time.perf_counter()
        generation = await generate_fn(prompt_ids, sampling_params)
        latency_generate = time.perf_counter() - start
        total_generate += latency_generate

        response_ids = list(generation.token_ids or [])[: config.response_length]
        if not response_ids:
            stats["empty_generation"] = 1
            logger.warning("[finagent][step=%d] model returned zero tokens.", step_idx + 1)
            # Make the previous turn terminal so the framework still computes a
            # reward instead of returning an all-zero, signal-free trajectory.
            if turns:
                turns[-1].stop = True
            break

        response_text = generation.text
        if response_text is None:
            response_text = decode_fn(response_ids)
        response_text = response_text[: config.response_length * 8]

        logprobs = generation.logprobs
        response_logprobs = (
            [float(value) for value in logprobs[: config.response_length]]
            if logprobs is not None
            else None
        )

        action = parse_action(response_text)
        stats["step_count"] = step_idx + 1
        stats["invalid_tool_calls"] += action.malformed_tool_calls

        has_answer = action.has_answer
        has_calls = bool(action.tool_calls)
        recoverable_malformed = (
            action.malformed_tool_calls > 0 and not has_answer and not is_last_step
        )
        should_stop = has_answer or is_last_step or (not has_calls and not recoverable_malformed)

        tool_results: list[ToolExecution] = []
        latency_tools = 0.0

        if not should_stop and has_calls:
            calls = action.tool_calls[: max(1, config.max_parallel_calls)]
            tool_start = time.perf_counter()
            tool_results = await asyncio.gather(
                *(
                    _execute_tool(call.name, call.arguments, tools_by_name, tools_kwargs)
                    for call in calls
                )
            )
            latency_tools = time.perf_counter() - tool_start
            total_tools += latency_tools

            feedback_lines: list[str] = []
            for result in tool_results:
                stats["tool_call_count"] += 1
                history_actions.append(
                    f"{result.name}({json.dumps(result.arguments, ensure_ascii=False)})"[:300]
                )

                if result.ok:
                    observations.append(
                        f"[{result.label}]\n{result.observation[: config.max_observation_chars]}"
                    )
                    _update_scratchpad(scratchpad, result, config.max_scratchpad_items)
                    if result.name == "financial_retriever":
                        stats["retriever_calls"] += 1
                        stats["retrieved_chunks"] += int(result.meta.get("retrieved_count", 0) or 0)
                    elif result.name == "python_calc":
                        stats["calc_calls"] += 1
                else:
                    stats["tool_errors"] += 1
                    feedback_lines.append(f"{result.label}: {result.observation}")
                    observations.append(
                        f"[{result.label}]\n{result.observation[: config.max_observation_chars]}"
                    )

            malformed_feedback = malformed_tool_call_feedback(action.malformed_tool_calls)
            if malformed_feedback:
                feedback_lines.append(malformed_feedback)
            tool_feedback = "\n".join(feedback_lines[-3:])

            # Bound the in-memory context.  "full" mode keeps more history for
            # the context-management ablation; the prompt renderer still
            # enforces the token budget.
            observation_cap = 64 if config.context_mode == "full" else 8
            del observations[:-observation_cap]
            del history_actions[:-40]
        elif not should_stop and recoverable_malformed:
            # No parseable call was produced.  Spend this step teaching the
            # policy the JSON protocol instead of silently looping.
            tool_feedback = malformed_tool_call_feedback(action.malformed_tool_calls)

        turn = TurnRecord(
            step_index=step_idx + 1,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_text=response_text,
            response_logprobs=response_logprobs,
            routed_experts=generation.routed_experts,
            action=action,
            tool_results=list(tool_results),
            stop=should_stop,
            stats=dict(stats),
            latency_generate=latency_generate,
            latency_tools=latency_tools,
        )
        turns.append(turn)

        if should_stop:
            break

    if not turns:
        raise EpisodeError(
            "The model server returned no tokens; check the vLLM/rollout worker."
        )

    final_answer = turns[-1].action.answer or ""
    stats["answer_found"] = 1 if final_answer else 0

    return EpisodeResult(
        turns=turns,
        stats=stats,
        final_answer=final_answer,
        scratchpad=scratchpad,
        metrics={
            "generate_sequences": total_generate,
            "tool_calls": total_tools,
        },
    )


__all__ = [
    "EpisodeConfig",
    "EpisodeError",
    "EpisodeResult",
    "Generation",
    "ToolExecution",
    "TurnRecord",
    "run_episode",
]
