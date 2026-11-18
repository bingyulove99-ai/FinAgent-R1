"""Framework-independent parsing and prompt-construction helpers.

Nothing in this module imports ``verl`` or ``agent_r1``.  That is deliberate:
the exact same functions are used by the training flow, the vLLM evaluator and
the unit tests, so a parsing bug cannot hide behind a framework boundary.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from finagent_r1.prompts import FINQA_USER_PROMPT

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

TRUNCATION_MARKER = "\n...[observation truncated to fit the prompt budget]"

_MALFORMED_FEEDBACK = (
    "One or more <tool_call> blocks could not be parsed. Use strict JSON with "
    'double quotes: {"name": "financial_retriever", "arguments": {"query": "..."}}.'
)


@dataclass(frozen=True)
class ToolCall:
    """A parsed tool invocation."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ParsedAction:
    """Structured view of one LLM response."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    malformed_tool_calls: int = 0
    answer: str | None = None

    @property
    def has_answer(self) -> bool:
        return bool(self.answer)


def _safe_parse_json(text: str) -> Any:
    """Parse a JSON object, tolerating code fences and Python literals."""

    text = (text or "").strip()
    if not text:
        return None
    text = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def parse_action(text: str | None) -> ParsedAction:
    """Parse ``<tool_call>`` and ``<answer>`` blocks from a model response.

    A tool call is accepted in any of these shapes::

        {"name": "financial_retriever", "arguments": {"query": "..."}}
        {"name": "financial_retriever", "parameters": {"query": "..."}}
        {"tool": "financial_retriever", "args": {"query": "..."}}

    Unclosed ``<tool_call>`` tags are counted as malformed.  They are
    recoverable: the environment feeds a correction hint back and lets the
    policy retry on the next step.
    """

    text = text or ""
    blocks = _TOOL_CALL_RE.findall(text)
    malformed = max(0, text.count("<tool_call>") - len(blocks))

    calls: list[ToolCall] = []
    for raw in blocks:
        parsed = _safe_parse_json(raw)
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, Mapping):
            items = [parsed]
        else:
            malformed += 1
            continue

        for item in items:
            if not isinstance(item, Mapping):
                malformed += 1
                continue

            name = str(item.get("name") or item.get("tool") or "").strip()
            arguments = item.get("arguments", item.get("parameters", item.get("args", {})))
            if isinstance(arguments, str):
                arguments = _safe_parse_json(arguments)

            if not name:
                malformed += 1
                continue
            if not isinstance(arguments, Mapping):
                arguments = {}
            calls.append(ToolCall(name=name, arguments=dict(arguments)))

    answers = [match.strip() for match in _ANSWER_RE.findall(text) if match.strip()]
    return ParsedAction(
        text=text,
        tool_calls=tuple(calls),
        malformed_tool_calls=malformed,
        answer=answers[-1] if answers else None,
    )


def extract_query(raw_prompt: Any) -> str:
    """Extract the user question from ``raw_prompt``.

    ``RLHFDataset`` exposes ``raw_prompt`` as chat messages when
    ``data.return_raw_chat=True``; older code paths may pass a string or a list
    of token ids.  All three are handled.
    """

    if raw_prompt is None:
        return ""
    if isinstance(raw_prompt, str):
        return raw_prompt.strip()
    if isinstance(raw_prompt, Mapping):
        return str(raw_prompt.get("content") or "").strip()
    if isinstance(raw_prompt, Sequence) and not isinstance(raw_prompt, (bytes, bytearray)):
        messages = list(raw_prompt)
        for message in reversed(messages):
            if isinstance(message, Mapping) and message.get("role") == "user":
                content = str(message.get("content") or "").strip()
                if content:
                    return content
        for message in messages:
            if isinstance(message, Mapping):
                content = str(message.get("content") or "").strip()
                if content:
                    return content
    return str(raw_prompt).strip()


def coerce_mapping(value: Any) -> dict[str, Any]:
    """Best-effort conversion of a dict / JSON string / 1-element list to a dict."""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return coerce_mapping(tolist())
        except Exception:
            return {}
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return coerce_mapping(value[0])
    return {}


def merge_scope(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Collect the filing scope and reward metadata from a dataset row.

    Agent-R1 forwards the dataset columns as ``**kwargs``.  FinQA rows carry
    ``ticker`` and ``year`` inside ``extra_info``; ``env_kwargs`` is also
    accepted for compatibility with the framework's environment recipes.
    """

    extra_info = coerce_mapping(kwargs.get("extra_info"))
    env_kwargs = coerce_mapping(kwargs.get("env_kwargs"))
    env_tools = coerce_mapping(env_kwargs.get("tools_kwargs"))

    ticker = (
        extra_info.get("ticker")
        or env_tools.get("ticker")
        or kwargs.get("ticker")
        or ""
    )
    year = extra_info.get("year")
    if year is None:
        year = env_tools.get("year")
    if year is None:
        year = kwargs.get("year")

    try:
        year = int(year) if year is not None and str(year).strip() else None
    except (TypeError, ValueError):
        year = None

    scope: dict[str, Any] = {
        "ticker": str(ticker).strip().upper(),
        "year": year,
    }

    reward_model = coerce_mapping(kwargs.get("reward_model"))
    if "ground_truth" in reward_model:
        scope["ground_truth"] = reward_model["ground_truth"]
    elif "ground_truth" in extra_info:
        scope["ground_truth"] = extra_info["ground_truth"]

    return scope


def merge_tools_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Build the per-sample keyword payload passed to every tool.

    The retriever uses ``ticker`` / ``year`` to restrict the corpus; the
    ``corpus_path`` override is useful when a dataset carries its own corpus.
    """

    merged: dict[str, Any] = dict(merge_scope(kwargs))

    env_kwargs = coerce_mapping(kwargs.get("env_kwargs"))
    merged.update(coerce_mapping(env_kwargs.get("tools_kwargs")))
    merged.update(coerce_mapping(kwargs.get("tools_kwargs")))

    extra_info = coerce_mapping(kwargs.get("extra_info"))
    if "retrieval_corpus" not in merged and isinstance(extra_info.get("retrieval_corpus"), list):
        merged["retrieval_corpus"] = extra_info["retrieval_corpus"]
    if not merged.get("corpus_path") and extra_info.get("corpus_path"):
        merged["corpus_path"] = extra_info["corpus_path"]

    return merged


def format_observations(observations: Sequence[str], window: int, max_chars: int) -> str:
    """Render the most recent observations, truncating the block if needed."""

    recent = list(observations)[-max(1, int(window)) :]
    text = "\n\n".join(recent)
    if not text:
        return "No external observations yet."
    if max_chars > 0 and len(text) > max_chars:
        # Keep the *tail*: the most recent observation is the one the policy
        # needs for the next decision.  Truncating the head drops older
        # evidence instead of the fresh evidence.
        budget = max(0, max_chars - len(TRUNCATION_MARKER))
        text = TRUNCATION_MARKER + text[-budget:].lstrip()
    return text


def format_history(actions: Sequence[str], limit: int = 10) -> str:
    """Render previous tool calls, keeping the most recent ``limit``."""

    recent = list(actions)[-max(1, int(limit)) :]
    if not recent:
        return "None"
    return "\n".join(f"[{index + 1}] {action}" for index, action in enumerate(recent))


def build_user_content(
    *,
    query: str,
    ticker: str,
    fiscal_year: int | str | None,
    scratchpad: Sequence[str],
    history_actions: Sequence[str],
    observations: Sequence[str],
    tool_feedback: str,
    step_index: int,
    max_steps: int,
    observation_window: int = 2,
    max_obs_chars: int = 6000,
) -> str:
    """Build the user message for one step.

    ``FINQA_USER_PROMPT`` contains no literal JSON, so ``str.format`` is safe.
    """

    scratchpad_str = (
        "\n".join(f"- {item}" for item in scratchpad)
        if scratchpad
        else "No verified facts recorded yet."
    )
    observations_str = format_observations(observations, observation_window, max_obs_chars)
    feedback_str = (tool_feedback or "").strip() or "None"

    return FINQA_USER_PROMPT.format(
        user_query=query,
        ticker=ticker or "UNKNOWN",
        fiscal_year=fiscal_year if fiscal_year is not None else "UNKNOWN",
        step_index=step_index,
        max_steps=max_steps,
        scratchpad=scratchpad_str,
        history_actions=format_history(history_actions),
        latest_observations=observations_str,
        tool_feedback=feedback_str,
    )


def malformed_tool_call_feedback(count: int) -> str:
    """Human-readable correction hint for malformed ``<tool_call>`` blocks."""

    if count <= 0:
        return ""
    return f"{count} malformed call(s). {_MALFORMED_FEEDBACK}"


__all__ = [
    "ParsedAction",
    "TRUNCATION_MARKER",
    "ToolCall",
    "build_user_content",
    "coerce_mapping",
    "extract_query",
    "format_history",
    "format_observations",
    "malformed_tool_call_feedback",
    "merge_scope",
    "merge_tools_kwargs",
    "parse_action",
]
