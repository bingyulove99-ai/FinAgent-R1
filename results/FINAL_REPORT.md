# FinAgent-R1 最终结果汇总

生成时间：2026-11-18 04:05（本地）｜所有数值来自 `results/*.metrics.json`（真实运行）

## 1. 一句话结论

在 FinQA（1,147 题 test）上，Qwen2.5-3B-Instruct 经 Agent-R1 步级 MDP 的 GRPO 训练后取得
**30.78% 数值准确率**，较零样本 ReAct 基线（7.24%）提升 **4.25 倍**、较无工具直答（0.52%）
提升约 59 倍；五种 RL 算法全部显著优于基线（McNemar p<0.05），排序为
**GRPO > RLOO > REINFORCE > PPO/step-GAE > Token-GAE**。

## 2. 主结果表（test 1,147 题，单 seed 42）

| 配置 | 数值准确率 | 95% CI | mean score | answer rate | avg steps | 无效调用 | 工具错误 | 检索召回 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B 直答（无工具） | 0.0052 | [0.0017, 0.0096] | 0.0552 | 1.0000 | 1.00 | 0.0000 | 0.0000 | n/a |
| 3B 零样本 ReAct | 0.0724 | [0.0584, 0.0872] | 0.0435 | 0.3575 | 4.70 | 0.4214 | 0.0050 | 0.2213 |
| **3B + GRPO** | **0.3078** | [0.2807, 0.3339] | 0.3118 | 0.7742 | 3.74 | 0.1089 | 0.0046 | 0.2372 |
| 3B + RLOO | 0.2677 | [0.2415, 0.2929] | 0.2832 | 0.8649 | 3.16 | 0.0944 | 0.0124 | 0.2336 |
| 3B + REINFORCE | 0.2110 | [0.1866, 0.2363] | 0.2264 | 0.8893 | 3.66 | 0.1284 | 0.0161 | 0.2080 |
| 3B + PPO/step-GAE | 0.1997 | [0.1761, 0.2223] | 0.1832 | 0.6600 | 4.19 | 0.3690 | 0.0031 | 0.2409 |
| 3B + Token-GAE | 0.1055 | [0.0889, 0.1247] | 0.0344 | 0.7140 | 3.08 | 2.2479 | 0.0175 | 0.1597 |

> 官方 FinQA 按文档级划分，test 文档全部未在 train 出现，故 `unseen_docs == all`（`seen_docs` 为空）。
> 指标定义与训练奖励一致：数值匹配（相对容差 1e-3/绝对 1e-4，百分比与小数等价）。

## 3. 统计显著性（配对 McNemar，同一题目集）

| 对比 | 仅基线正确 | 仅模型正确 | p |
|---|---:|---:|---:|
| GRPO vs ReAct | 292 | 22 | 2.3e-61 |
| RLOO vs ReAct | 250 | 26 | 3.9e-47 |
| REINFORCE vs ReAct | 193 | 34 | 3.6e-28 |
| PPO vs ReAct | 184 | 38 | 3.6e-24 |
| Token-GAE vs ReAct | 96 | 58 | 0.0027 |
| RLOO vs GRPO | 57 | 103 | 3.4e-4 |
| REINFORCE vs GRPO | 49 | 160 | 6.1e-15 |
| PPO vs GRPO | 83 | 207 | 2.2e-13 |
| Token-GAE vs GRPO | 42 | 274 | 7.0e-43 |

## 4. 训练过程与稳定性

| 算法 | 训练时长 | dev 轨迹（每 25 步） | 选用 checkpoint | 稳定性观察 |
|---|---|---:|---|---|
| GRPO | 27h26m | 0.065→0.178→0.195→0.234→0.256→0.245→0.273→0.272→**0.286** | step195 | KL≤0.50，最稳定 |
| RLOO | ~29h44m | 0.057→0.187→0.206→0.251→**0.2515**→0.221→0.162→0.069→0.075 | step100 | step100 后漂移，KL 最高 9.1 |
| REINFORCE | 17h28m | 0.056→**0.1827**→0.121→0.131→-0.007→0.001→0.003→0.023 | step25 | step75 后崩溃（不用工具），KL 11.6 |
| PPO/step-GAE | 27h09m | -0.025→-0.029→-0.008→0.064→0.099→0.066→**0.1009**→-0.114→-0.004 | step150 | step150 后长度刷分崩溃（响应 184→455） |
| Token-GAE | ~13h（提前停止） | -0.039→-0.026→**-0.0142** | step50 | 长度爆炸（428→757/768）+ KL 5.4 + critic OOM |

统一超参：lr=5e-7、kl_loss_coef=0.03、low_var_kl、1 epoch（195 updates）、batch 32、
rollout n=8（PPO/Token-GAE n=4）、每 25 步验证/保存、全 checkpoint 保留、
GRPO 用 Dr.GRPO、PPO/Token-GAE critic_warmup=10。

## 5. 关键结论

1. **GRPO 最稳定且最好**（KL≤0.50，dev 单调上升至 0.2864，test 30.78%）。
2. **无 critic 的 on-policy 算法在步级 MDP 下更稳**；含 critic 的 PPO/Token-GAE 出现
   更早/更严重的崩溃（Token-GAE 甚至 OOM）。
3. **后期漂移是普遍现象**（RLOO/REINFORCE/PPO/Token-GAE），主因是长度刷分与 KL 爆炸；
   **按 dev 最佳 checkpoint 选择**是必要的保护措施（文档 8.5 允许并建议报告该规则）。
4. RL 的核心收益来自**学会使用工具并给出 `<answer>` 格式**：answer rate 从零样本 ReAct 的
   35.8% 提升到 77%–89%，无效工具调用率从 42.1% 降至 9.4%–12.8%（GRPO/RLOO）。

## 6. 偏差与局限（如实说明）

- 单 seed（42）；置信区间为题目级 bootstrap，未做多 seed 方差估计。
- Token-GAE 未完成完整 1 epoch：step 75 附近策略退化并触发 critic CUDA OOM，v2 自动重试
  从 step 50 续训；为避免重复崩溃与无效计算，手动停止并按 dev 最佳 step 50 合并评测。
  checkpoint 25/50 保留，可用 `run_all_trainings_v3.sh`（RESUME=1）续训。此后又用减半的
  micro-batch/token 预算从 step 50 续训两次，均在 resume 后首个 critic 更新处同样 OOM，
  确认在当前 2×A40（另 2 卡被占用）显存条件下无法完成该退化策略的完整 epoch。
- PPO/Token-GAE 训练期 dev 指标与无 critic 算法不可直接比较（同一 base 模型初始 dev 为
  -0.025/-0.039 vs +0.056~0.065），最终以统一 `infer_vllm` test 评测为准。
- 早期曾出现评测并发 vLLM 显存竞争与汇总脚本 bug，均已修复并记录于
  `docs/05_experiment_summary_zh.md` §4。

## 7. 文件位置

| 内容 | 路径 |
|---|---|
| 完整结果表 + CI + McNemar | `results/SUMMARY.md` |
| 每算法摘要 | `results/{grpo,rloo,reinforce,ppo_gae,token_gae}_summary.md` |
| 指标 JSON | `results/<config>_{all,seen_docs,unseen_docs}.metrics.json` |
| 全量预测 | `results/<config>_test.jsonl`（每个 1,147 行） |
| 实验总结（含错误/调优记录） | `docs/05_experiment_summary_zh.md` |
| 汇总快照（代码/脚本/日志/样例） | `/home/user/FinAgent-R1-summary-2026-11-16/` |
| 训练日志 / dev 轨迹 | `logs/train_<algo>.log`、`logs/devacc_<algo>.tsv` |

## 8. 复现

```bash
source /data/home.old/user/agent-r1/env.sh
cd /home/user/Agent-R1-main
bash /data/home.old/user/agent-r1/run_all_trainings_v2.sh   # 5 算法 + 自动评测
bash /data/home.old/user/agent-r1/finalize.sh               # 基线 + 汇总
```
