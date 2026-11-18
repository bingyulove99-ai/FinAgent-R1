# FinAgent-R1 Evaluation Summary

All numbers come from `results/*.metrics.json` (real runs, no estimates).
Test split: 1,147 FinQA questions. The official FinQA release splits at
document level, so every test document is unseen in train (`seen_docs` is
empty); `unseen_docs` therefore equals `all`.

## Main result table (numeric accuracy unless noted)

| Config | Acc (all) | 95% CI | Acc (unseen) | Format | Ans rate | Avg steps | Invalid call | Tool error | Retrieval recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B direct (no tools) | 0.0052 | [0.0017, 0.0096] | 0.0052 | 0.0000 | 1.0000 | 1.00 | 0.0000 | 0.0000 | n/a |
| 3B zero-shot ReAct | 0.0724 | [0.0584, 0.0872] | 0.0724 | 0.0017 | 0.3575 | 4.70 | 0.4214 | 0.0050 | 0.2213 |
| 3B + GRPO | 0.3078 | [0.2807, 0.3339] | 0.3078 | 0.0000 | 0.7742 | 3.74 | 0.1089 | 0.0046 | 0.2372 |
| 3B + RLOO | 0.2677 | [0.2415, 0.2929] | 0.2677 | 0.0000 | 0.8649 | 3.16 | 0.0944 | 0.0124 | 0.2336 |
| 3B + REINFORCE | 0.2110 | [0.1866, 0.2363] | 0.2110 | 0.0009 | 0.8893 | 3.66 | 0.1284 | 0.0161 | 0.2080 |
| 3B + PPO/step-GAE | 0.1997 | [0.1761, 0.2223] | 0.1997 | 0.0122 | 0.6600 | 4.19 | 0.3690 | 0.0031 | 0.2409 |
| 3B + Token-GAE | 0.1055 | [0.0889, 0.1247] | 0.1055 | 0.0035 | 0.7140 | 3.08 | 2.2479 | 0.0175 | 0.1597 |

## Composite reward / F1

| Config | Mean score (all) | Mean F1 (all) | Mean score (unseen) |
|---|---:|---:|---:|
| 3B direct (no tools) | 0.0552 | 0.0009 | 0.0552 |
| 3B zero-shot ReAct | 0.0435 | 0.0002 | 0.0435 |
| 3B + GRPO | 0.3118 | 0.0003 | 0.3118 |
| 3B + RLOO | 0.2832 | 0.0000 | 0.2832 |
| 3B + REINFORCE | 0.2264 | 0.0001 | 0.2264 |
| 3B + PPO/step-GAE | 0.1832 | 0.0002 | 0.1832 |
| 3B + Token-GAE | 0.0344 | 0.0001 | 0.0344 |

## McNemar tests vs zero-shot ReAct (paired, two-sided)

| Config | ReAct-only correct | Model-only correct | p-value |
|---|---:|---:|---:|
| 3B + GRPO | 292 | 22 | 2.323e-61 |
| 3B + RLOO | 250 | 26 | 3.927e-47 |
| 3B + REINFORCE | 193 | 34 | 3.586e-28 |
| 3B + PPO/step-GAE | 184 | 38 | 3.583e-24 |
| 3B + Token-GAE | 96 | 58 | 0.002749 |

## McNemar tests vs GRPO

| Config | GRPO-only correct | Model-only correct | p-value |
|---|---:|---:|---:|
| 3B + RLOO | 57 | 103 | 0.0003439 |
| 3B + REINFORCE | 49 | 160 | 6.11e-15 |
| 3B + PPO/step-GAE | 83 | 207 | 2.191e-13 |
| 3B + Token-GAE | 42 | 274 | 7.039e-43 |

Single seed (42) per algorithm; CIs are question-level paired bootstrap (1,000 resamples).
