# ppo_gae result summary

- experiment: `finqa_qwen3b_ppo_gae`
- selection: selection_rule: best dev accuracy; best_step: 150; dev_acc: 0.100943
- dev score (greedy, 883 q): base=nan best=0.1009 @ step 150 final=0.1009
- train steps logged: 186; KL last/max: 0.727/1.902; entropy min/median: 0.0344/0.0890; train reward first/median/last: -0.0160/-0.0058/0.0031

| subset | n | acc | score | F1 | format | answer | steps | invalid | toolerr | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1147 | 0.1997 | 0.1832 | 0.0002 | 0.0122 | 0.6600 | 4.19 | 0.3690 | 0.0031 | 0.2409 |
| seen_docs | 0 | n/a (empty split) | — | — | — | — | — | — | — | — |
| unseen_docs | 1147 | 0.1997 | 0.1832 | 0.0002 | 0.0122 | 0.6600 | 4.19 | 0.3690 | 0.0031 | 0.2409 |
