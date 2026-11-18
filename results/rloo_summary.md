# rloo result summary

- experiment: `finqa_qwen3b_rloo`
- selection: selection_rule: best dev accuracy; best_step: 100; dev_acc: 0.251497
- dev score (greedy, 883 q): base=0.0568 best=0.2515 @ step 100 final=0.0749
- train steps logged: 195; KL last/max: 6.338/9.112; entropy min/median: 0.0007/0.0140; train reward first/median/last: 0.0090/0.0252/-0.0019

| subset | n | acc | score | F1 | format | answer | steps | invalid | toolerr | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1147 | 0.2677 | 0.2832 | 0.0000 | 0.0000 | 0.8649 | 3.16 | 0.0944 | 0.0124 | 0.2336 |
| seen_docs | 0 | n/a (empty split) | — | — | — | — | — | — | — | — |
| unseen_docs | 1147 | 0.2677 | 0.2832 | 0.0000 | 0.0000 | 0.8649 | 3.16 | 0.0944 | 0.0124 | 0.2336 |
