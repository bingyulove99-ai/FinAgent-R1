"""Composite rule reward for FinQA-style financial QA.

Reward = outcome + format - tool penalty, where:

* outcome is numeric match against ``exe_ans`` (preferred) or token F1 for
  free-text answers;
* format rewards the ``<analysis>`` / ``<answer>`` protocol;
* a mild step-efficiency multiplier discourages redundant tool calls;
* malformed tool calls and tool errors subtract up to 0.15.

Only the terminal step is scored.  Intermediate steps in the Agent-R1 flow
carry ``reward_score=0.0``, so this function is called exactly once per
trajectory by the framework's reward loop.
"""

from __future__ import annotations

import math
import re
import string
from typing import Any

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?\s*%?")

_MAX_TOOL_PENALTY = 0.15
_TOOL_ERROR_PENALTY = 0.03
_MAX_PRED_NUMBERS = 8


def _normalize_text(text: str) -> str:
    """Lower-case, strip punctuation/articles and collapse whitespace."""

    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_final_answer(solution_str: str | None) -> str:
    """Return the content of the last ``<answer>`` block, or ``""``."""

    if not solution_str:
        return ""
    matches = [match.strip() for match in _ANSWER_RE.findall(solution_str) if match.strip()]
    return matches[-1] if matches else ""


def _parse_numbers(text: str) -> list[tuple[float, bool]]:
    """Return ``(value, is_percent)`` pairs found in ``text``."""

    results: list[tuple[float, bool]] = []
    cleaned = (text or "").replace(",", "").replace("$", "")
    for match in _NUMBER_RE.findall(cleaned):
        token = match.strip()
        is_percent = token.endswith("%")
        if is_percent:
            token = token[:-1].strip()
        try:
            results.append((float(token), is_percent))
        except ValueError:
            continue
    return results


def _target_value(ground_truth: Any, extra_info: dict[str, Any]) -> tuple[float | None, bool, str]:
    """Resolve the numeric target and whether the answer is a percentage.

    ``exe_ans`` is preferred because it is the result of executing FinQA's
    annotated program.  The percent flag still comes from the answer string so
    that ``41.38%`` matches both ``41.38`` and ``0.4138``.
    """

    gt_str = str(ground_truth).strip() if ground_truth is not None else ""
    parsed_gt = _parse_numbers(gt_str)
    target_is_percent = bool(parsed_gt and parsed_gt[0][1])

    exe_ans = extra_info.get("exe_ans")
    if exe_ans is not None and not isinstance(exe_ans, bool):
        try:
            value = float(exe_ans)
            if math.isfinite(value):
                return value, target_is_percent, gt_str
        except (TypeError, ValueError):
            pass

    if parsed_gt:
        return parsed_gt[0][0], target_is_percent, gt_str
    return None, False, gt_str


def numeric_match(
    prediction: str,
    target: float,
    *,
    target_is_percent: bool = False,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-4,
) -> bool:
    """Check any number in ``prediction`` against ``target``.

    Percent/decimal scaling is only enabled when the answer string is a
    percentage.  Without that guard, a plain target of ``100`` would falsely
    match a prediction of ``1``.
    """

    target_candidates = [target]
    if target_is_percent:
        # FinQA's exe_ans may be either percentage points (41.38) or a decimal
        # (0.4138).  Scale in the plausible direction only; scaling both ways
        # would let a prediction of 4138 match a target of 41.38.
        if abs(target) >= 1:
            target_candidates.append(target / 100.0)
        else:
            target_candidates.append(target * 100.0)

    for value, is_percent in _parse_numbers(prediction)[:_MAX_PRED_NUMBERS]:
        candidates = [value]
        if is_percent:
            candidates.append(value / 100.0)
        for candidate in candidates:
            for target_candidate in target_candidates:
                if math.isclose(candidate, target_candidate, rel_tol=rel_tol, abs_tol=abs_tol):
                    return True
    return False


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 for free-text answers."""

    pred_tokens = _normalize_text(prediction).split()
    gt_tokens = _normalize_text(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = sum(min(pred_tokens.count(word), gt_tokens.count(word)) for word in common) / len(pred_tokens)
    recall = sum(min(pred_tokens.count(word), gt_tokens.count(word)) for word in common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_extra_dict(extra_info: Any) -> dict[str, Any]:
    """Normalize the reward loop's ``extra_info`` payload to a plain dict."""

    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, (list, tuple)) and len(extra_info) == 1:
        return _as_extra_dict(extra_info[0])
    tolist = getattr(extra_info, "tolist", None)
    if callable(tolist):
        try:
            return _as_extra_dict(tolist())
        except Exception:
            return {}
    return {}


def score_answer(
    solution_str: str | None,
    ground_truth: Any,
    extra_info: Any = None,
) -> dict[str, Any]:
    """Return the full score breakdown for one terminal response."""

    extra = _as_extra_dict(extra_info)
    prediction = extract_final_answer(solution_str)
    target, target_is_percent, gt_str = _target_value(ground_truth, extra)
    is_numeric = bool(extra.get("is_numeric", target is not None)) and target is not None

    if prediction:
        if is_numeric:
            correct = numeric_match(prediction, target, target_is_percent=target_is_percent)
            outcome = 1.0 if correct else 0.0
            f1 = 0.0
        else:
            norm_pred = _normalize_text(prediction)
            norm_gt = _normalize_text(gt_str)
            correct = bool(norm_pred) and (norm_pred == norm_gt or norm_gt in norm_pred)
            f1 = token_f1(prediction, gt_str)
            outcome = 1.0 if correct else f1
    else:
        correct = False
        f1 = 0.0
        outcome = 0.0

    solution = solution_str or ""
    format_reward = 0.0
    if "<analysis>" in solution and "</analysis>" in solution:
        format_reward += 0.05
    # An empty <answer></answer> should not earn the answer-format reward.
    if prediction and "<answer>" in solution and "</answer>" in solution:
        format_reward += 0.05

    step_count = max(1, _safe_int(extra.get("step_count"), 1))
    efficiency_multiplier = max(0.85, 0.96 ** max(0, step_count - 1))

    invalid_tool_calls = max(0, _safe_int(extra.get("invalid_tool_calls"), 0))
    tool_errors = max(0, _safe_int(extra.get("tool_errors"), 0))
    tool_penalty = min(
        _MAX_TOOL_PENALTY,
        _TOOL_ERROR_PENALTY * (invalid_tool_calls + tool_errors),
    )

    total = outcome * efficiency_multiplier + format_reward - tool_penalty
    total = float(max(-0.2, min(1.2, total)))

    return {
        "score": round(total, 4),
        "outcome": outcome,
        "correct": bool(correct),
        "f1": round(f1, 4),
        "format_reward": round(format_reward, 4),
        "efficiency_multiplier": round(efficiency_multiplier, 4),
        "tool_penalty": round(tool_penalty, 4),
        "answer_found": bool(prediction),
        "prediction": prediction,
        "target": target,
        "target_is_percent": target_is_percent,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
    **kwargs: Any,
) -> float:
    """Entry point expected by ``custom_reward_function`` in verl/Agent-R1."""

    if solution_str is None:
        return 0.0
    return float(score_answer(solution_str, ground_truth, extra_info)["score"])


__all__ = [
    "compute_score",
    "extract_final_answer",
    "numeric_match",
    "score_answer",
    "token_f1",
]
