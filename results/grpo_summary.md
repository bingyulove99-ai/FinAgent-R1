# grpo result summary

- experiment: `finqa_qwen3b_grpo`
- selection: selection_rule: best dev accuracy; best_step: 195; dev_acc: 0.286425
- dev score (greedy, 883 q): base=0.0652 best=0.2864 @ step 195 final=0.2864
- train steps logged: 195; KL last/max: 0.273/0.501; entropy min/median: 0.0105/0.0181; train reward first/median/last: 0.0056/0.0463/0.0657

| subset | n | acc | score | F1 | format | answer | steps | invalid | toolerr | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1147 | 0.3078 | 0.3118 | 0.0003 | 0.0000 | 0.7742 | 3.74 | 0.1089 | 0.0046 | 0.2372 |
| seen_docs | 0 | n/a (empty split) | — | — | — | — | — | — | — | — |
| unseen_docs | 1147 | 0.3078 | 0.3118 | 0.0003 | 0.0000 | 0.7742 | 3.74 | 0.1089 | 0.0046 | 0.2372 |
