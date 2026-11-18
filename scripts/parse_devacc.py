#!/usr/bin/env python3
"""Parse per-step dev accuracy from a verl Agent-R1 training log.

Writes a TSV (step, acc) and prints the best step, e.g. ``50 0.2412``.
Validation at step N runs on the policy after the step-N update and before
checkpoint N is written, so step N maps 1:1 to ``global_step_N``.
"""
from __future__ import annotations

import re
import sys

PAT = re.compile(r"step:(\d+) - .*?val-core/finqa/acc/(?:mean@1|mean):([0-9.]+)")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: parse_devacc.py <train.log> [out.tsv]", file=sys.stderr)
        return 2
    log = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None

    accs: dict[int, float] = {}
    with open(log, "r", errors="ignore") as fh:
        for line in fh:
            m = PAT.search(line)
            if m:
                accs[int(m.group(1))] = float(m.group(2))

    if out:
        with open(out, "w") as fh:
            for step in sorted(accs):
                fh.write(f"{step}\t{accs[step]:.6f}\n")

    if not accs:
        print("no dev acc found", file=sys.stderr)
        return 1
    best_step = max(accs, key=accs.get)
    print(f"dev_acc: {accs}")
    print(f"best: {best_step} {accs[best_step]:.6f}")
    print(f"BEST_STEP={best_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
