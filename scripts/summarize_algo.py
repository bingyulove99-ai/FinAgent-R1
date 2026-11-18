#!/usr/bin/env python3
"""Build a compact per-algorithm summary from real run artifacts.

Usage: summarize_algo.py <algo-tag>    e.g. grpo | rloo | reinforce | ppo_gae | token_gae

Reads:
  results/<tag>_{all,seen_docs,unseen_docs}.metrics.json
  logs/devacc_<algo>.tsv           (step -> dev score, greedy)
  logs/train_<algo>.log            (KL / entropy / reward trajectory)
  outputs/<exp>/selection.txt      (checkpoint-selection rule)
Writes results/<tag>_summary.md and prints it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BASE = Path("/data/home.old/user/agent-r1")
RES = BASE / "results"
LOGS = BASE / "logs"

EXP = {
    "grpo": "finqa_qwen3b_grpo",
    "rloo": "finqa_qwen3b_rloo",
    "reinforce": "finqa_qwen3b_reinforce",
    "ppo_gae": "finqa_qwen3b_ppo_gae",
    "token_gae": "finqa_qwen3b_token_gae",
}
TRAIN_LOG = {"ppo_gae": "ppo", "token_gae": "token_gae"}
ALGO_NAME = {"ppo_gae": "ppo", "token_gae": "token_gae",
             "grpo": "grpo", "rloo": "rloo", "reinforce": "reinforce"}
STEP_PAT = re.compile(
    r"step:(\d+) - .*?actor/entropy:([\d.eE+-]+).*?actor/kl_loss:([\d.eE+-]+)"
    r".*?critic/score/mean:([\d.eE+-]+)"
)


def metrics(tag: str, subset: str):
    p = RES / f"{tag}_{subset}.metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["summary"]


def dev_accs(algo: str) -> dict[int, float]:
    p = LOGS / f"devacc_{algo}.tsv"
    out: dict[int, float] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                s, a = line.split("\t")
                out[int(s)] = float(a)
    return out


def train_traj(algo: str):
    name = TRAIN_LOG.get(algo, algo)
    p = LOGS / f"train_{name}.log"
    rows = []
    if p.exists():
        for line in p.read_text(errors="ignore").splitlines():
            m = STEP_PAT.search(line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return rows


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in EXP:
        print(f"usage: summarize_algo.py {{{'|'.join(EXP)}}}", file=sys.stderr)
        return 2
    algo = sys.argv[1]
    tag = algo
    exp = EXP[algo]
    run_name = ALGO_NAME[algo]
    sel = BASE / "outputs" / exp / "selection.txt"
    sel_txt = sel.read_text().strip().replace("\n", "; ") if sel.exists() else "n/a"

    accs = dev_accs(run_name)
    traj = train_traj(algo)
    m_all, m_uns, m_seen = metrics(tag, "all"), metrics(tag, "unseen_docs"), metrics(tag, "seen_docs")

    lines = [f"# {algo} result summary", "", f"- experiment: `{exp}`", f"- selection: {sel_txt}"]
    if accs:
        best_step = max(accs, key=accs.get)
        lines += [
            f"- dev score (greedy, 883 q): base={accs.get(0, float('nan')):.4f} "
            f"best={accs[best_step]:.4f} @ step {best_step} final={accs[max(accs)]:.4f}",
        ]
    if traj:
        kl = [r[2] for r in traj]
        ent = [r[1] for r in traj]
        rew = [r[3] for r in traj]
        lines += [
            f"- train steps logged: {len(traj)}; KL last/max: {kl[-1]:.3f}/{max(kl):.3f}; "
            f"entropy min/median: {min(ent):.4f}/{sorted(ent)[len(ent)//2]:.4f}; "
            f"train reward first/median/last: {rew[0]:.4f}/{sorted(rew)[len(rew)//2]:.4f}/{rew[-1]:.4f}",
        ]
    if m_all:
        lines += [
            "",
            "| subset | n | acc | score | F1 | format | answer | steps | invalid | toolerr | recall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, m in (("all", m_all), ("seen_docs", m_seen), ("unseen_docs", m_uns)):
            if not m or not m.get("count") or "accuracy" not in m:
                lines.append(f"| {name} | 0 | n/a (empty split) | — | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {name} | {m['count']} | {m['accuracy']:.4f} | {m['mean_score']:.4f} | "
                f"{m.get('mean_f1', 0):.4f} | {m['format_rate']:.4f} | {m['answer_rate']:.4f} | "
                f"{m['avg_steps']:.2f} | {m['invalid_call_rate']:.4f} | {m['tool_error_rate']:.4f} | "
                f"{m.get('retrieval_recall', float('nan')):.4f} |"
            )
    else:
        lines += ["", "test metrics: NOT YET AVAILABLE"]

    text = "\n".join(lines) + "\n"
    out = RES / f"{tag}_summary.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
