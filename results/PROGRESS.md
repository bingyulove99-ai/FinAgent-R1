# FinAgent-R1 运行进度（最终版）

更新时间：2026-11-18 04:30（本地）

## 总状态：五算法训练+全量评测已完成并汇总

| 配置 | test 数值准确率 | 95% CI | 状态 |
|---|---:|---:|---|
| 3B 直答（无工具） | 0.0052 | [0.0017, 0.0096] | ✅ |
| 3B 零样本 ReAct | 0.0724 | [0.0584, 0.0872] | ✅ |
| 3B + GRPO | **0.3078** | [0.2807, 0.3339] | ✅ |
| 3B + RLOO | 0.2677 | [0.2415, 0.2929] | ✅ |
| 3B + REINFORCE | 0.2110 | [0.1866, 0.2363] | ✅ |
| 3B + PPO/step-GAE | 0.1997 | [0.1761, 0.2223] | ✅ |
| 3B + Token-GAE | 0.1055 | [0.0889, 0.1247] | ✅（按 dev 最佳 step50） |

统计检验：五种 RL 算法均显著优于零样本 ReAct（McNemar p<0.05，四种 p<1e-20）；
GRPO 显著优于其余算法。详见 `results/SUMMARY.md`。

## Token-GAE 续训（为严格完成 1 epoch）

- 原始运行在 step ~75 因策略退化（响应 757/768、KL 5.4）+ critic 更新 CUDA OOM 崩溃。
- 已按 dev 最佳 step 50 完成合并与全量评测（0.1055），结果与 checkpoint 均已保留。
- 曾以**减半 micro-batch/token 预算**（actor/critic 4096 tokens/GPU、micro-batch 1，
  其余超参与五算法统一配置一致）从 step 50 续训至 195 步；完成后若出现更优 dev checkpoint
  会重新合并评测，否则结果不变（仍为 step 50）。
- 日志：`logs/train_token_gae_resume.log`，脚本：`resume_token_gae.sh`（含 3 次重试与清理）。

## 产物

- 汇总：`results/SUMMARY.md`、`results/FINAL_REPORT.md`
- 每算法：`results/<algo>_summary.md` + `results/<algo>_test.jsonl` + metrics json
- 总结文档：`docs/05_experiment_summary_zh.md`
- 快照：`/home/user/FinAgent-R1-summary-2026-11-16/`（14MB，含 7 组同题 60-case 样例）
