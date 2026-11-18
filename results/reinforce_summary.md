# reinforce result summary

- experiment: `finqa_qwen3b_reinforce`
- selection: selection_rule: best dev accuracy; best_step: 25; dev_acc: 0.182747
- dev score (greedy, 883 q): base=0.0557 best=0.1827 @ step 25 final=0.0228
- train steps logged: 195; KL last/max: 8.256/11.599; entropy min/median: 0.0016/0.0061; train reward first/median/last: 0.0086/0.0066/0.0236

| subset | n | acc | score | F1 | format | answer | steps | invalid | toolerr | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1147 | 0.2110 | 0.2264 | 0.0001 | 0.0009 | 0.8893 | 3.66 | 0.1284 | 0.0161 | 0.2080 |
| seen_docs | 0 | n/a (empty split) | — | — | — | — | — | — | — | — |
| unseen_docs | 1147 | 0.2110 | 0.2264 | 0.0001 | 0.0009 | 0.8893 | 3.66 | 0.1284 | 0.0161 | 0.2080 |
