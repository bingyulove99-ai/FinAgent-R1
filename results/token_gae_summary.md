# token_gae result summary

- experiment: `finqa_qwen3b_token_gae`
- selection: selection_rule: best dev accuracy (early stop due to CUDA OOM + policy collapse); best_step: 50; dev_acc: -0.014243827859569652; note: 训练在 step 75 附近因 critic 更新 OOM 崩溃；策略已退化（响应长度 ~757/768、KL 5.4），v2 已自动重试并从 step 50 续训，但为避免重复崩溃与无效计算，手动停止并按 dev 最佳 checkpoint (step 50) 合并评测。checkpoint 25/50 均保留，可后续用 RESUME=1 续训。

| subset | n | acc | score | F1 | format | answer | steps | invalid | toolerr | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1147 | 0.1055 | 0.0344 | 0.0001 | 0.0035 | 0.7140 | 3.08 | 2.2479 | 0.0175 | 0.1597 |
| seen_docs | 0 | n/a (empty split) | — | — | — | — | — | — | — | — |
| unseen_docs | 1147 | 0.1055 | 0.0344 | 0.0001 | 0.0035 | 0.7140 | 3.08 | 2.2479 | 0.0175 | 0.1597 |
