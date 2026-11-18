#!/usr/bin/env python3
"""Aggregate all evaluation metrics into the final report.

Reads results/<tag>_{all,seen_docs,unseen_docs}.metrics.json and the
per-question prediction files, builds the docs/00 section 8.4 result table,
and computes paired-bootstrap 95% CIs plus McNemar tests (docs/00 section 8.5).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

RES = Path("/data/home.old/user/agent-r1/results")

CONFIGS = [
    ("base_direct", "3B direct (no tools)"),
    ("base_react", "3B zero-shot ReAct"),
    ("grpo", "3B + GRPO"),
    ("rloo", "3B + RLOO"),
    ("reinforce", "3B + REINFORCE"),
    ("ppo_gae", "3B + PPO/step-GAE"),
    ("token_gae", "3B + Token-GAE"),
]


def load_metrics(tag: str, subset: str):
    p = RES / f"{tag}_{subset}.metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["summary"]


def load_correct(tag: str):
    p = RES / f"{tag}_test.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        out[str(r.get("id"))] = bool(r.get("correct"))
    return out


def bootstrap_ci(correct: dict, n_boot: int = 1000, seed: int = 42):
    vals = list(correct.values())
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    acc = sum(vals) / n
    boots = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n))
        boots.append(s / n)
    boots.sort()
    return acc, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def mcnemar(a: dict, b: dict):
    """Exact McNemar on shared question ids (two-sided binomial)."""
    ids = sorted(set(a) & set(b))
    n01 = sum(1 for i in ids if not a[i] and b[i])
    n10 = sum(1 for i in ids if a[i] and not b[i])
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return n01, n10, min(1.0, p)


def main():
    lines = ["# FinAgent-R1 Evaluation Summary", ""]
    lines += [
        "All numbers come from `results/*.metrics.json` (real runs, no estimates).",
        "Test split: 1,147 FinQA questions. The official FinQA release splits at",
        "document level, so every test document is unseen in train (`seen_docs` is",
        "empty); `unseen_docs` therefore equals `all`.",
        "",
        "## Main result table (numeric accuracy unless noted)",
        "",
        "| Config | Acc (all) | 95% CI | Acc (unseen) | Format | Ans rate | Avg steps | Invalid call | Tool error | Retrieval recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    correct = {}
    for tag, label in CONFIGS:
        m_all = load_metrics(tag, "all")
        m_uns = load_metrics(tag, "unseen_docs")
        if m_all is None:
            lines.append(f"| {label} | — | — | — | — | — | — | — | — | — |")
            continue
        correct[tag] = load_correct(tag)
        acc, lo, hi = bootstrap_ci(correct[tag])
        rr = m_all.get("retrieval_recall")
        rr_s = f"{rr:.4f}" if rr is not None else "n/a"
        uns_acc = f"{m_uns['accuracy']:.4f}" if m_uns else "n/a"
        lines.append(
            f"| {label} | {m_all['accuracy']:.4f} | [{lo:.4f}, {hi:.4f}] | {uns_acc} | "
            f"{m_all['format_rate']:.4f} | {m_all['answer_rate']:.4f} | {m_all['avg_steps']:.2f} | "
            f"{m_all['invalid_call_rate']:.4f} | {m_all['tool_error_rate']:.4f} | {rr_s} |"
        )
    lines += ["", "## Composite reward / F1", "",
              "| Config | Mean score (all) | Mean F1 (all) | Mean score (unseen) |",
              "|---|---:|---:|---:|"]
    for tag, label in CONFIGS:
        m_all = load_metrics(tag, "all")
        m_uns = load_metrics(tag, "unseen_docs")
        if m_all is None:
            continue
        uns = f"{m_uns['mean_score']:.4f}" if m_uns else "n/a"
        lines.append(f"| {label} | {m_all['mean_score']:.4f} | {m_all['mean_f1']:.4f} | {uns} |")

    if "base_react" in correct:
        lines += ["", "## McNemar tests vs zero-shot ReAct (paired, two-sided)", "",
                  "| Config | ReAct-only correct | Model-only correct | p-value |",
                  "|---|---:|---:|---:|"]
        for tag, label in CONFIGS:
            if tag in ("base_direct", "base_react") or tag not in correct:
                continue
            n01, n10, p = mcnemar(correct["base_react"], correct[tag])
            lines.append(f"| {label} | {n01} | {n10} | {p:.4g} |")
    if "grpo" in correct:
        lines += ["", "## McNemar tests vs GRPO", "",
                  "| Config | GRPO-only correct | Model-only correct | p-value |",
                  "|---|---:|---:|---:|"]
        for tag, label in CONFIGS:
            if tag in ("base_direct", "base_react", "grpo") or tag not in correct:
                continue
            n01, n10, p = mcnemar(correct["grpo"], correct[tag])
            lines.append(f"| {label} | {n01} | {n10} | {p:.4g} |")

    lines += ["", "Single seed (42) per algorithm; CIs are question-level paired bootstrap (1,000 resamples).", ""]
    out = RES / "SUMMARY.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()