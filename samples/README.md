# 样例数据与预测（便于 case 审查）

本目录包含**少量**样例（非全量），用于快速查看具体 case。全量数据仍在原工作区。

## data/ —— 数据集样例

| 文件 | 内容 | 行数 |
|---|---|---:|
| `test_cases60.jsonl` | 测试集前 60 条（与 predictions 中所有 case 文件一一对应） | 60 |
| `train_sample40.jsonl` | 训练集前 40 条 | 40 |
| `dev_sample40.jsonl` | 验证集前 40 条 | 40 |
| `corpus_sample40.jsonl` | 检索语料前 40 个 chunk | 40 |

数据行字段（RLHF parquet 同构）：
`data_source, prompt, agent_name, reward_model, extra_info, env_kwargs`；
`extra_info` 内含 `question_id, question, answer, exe_ans, is_numeric, program,
gold_evidence, ticker, year, doc_id, split`。

## predictions/ —— 各算法预测样例（与 test_cases60 相同的 60 个 id）

| 文件 | 内容 |
|---|---|
| `<algo>_cases60.jsonl` | 60 条完整预测记录（原始字段，含多轮 response/turns） |
| （已有：grpo / rloo / reinforce / ppo_gae） | |
| `<algo>_cases60.md` | 同一批 case 的速览表（问题/标准答案/预测/正确/score/steps） |

预测记录关键字段：
`id, ticker, year, question, prediction, answer, exe_ans, response, extra_info,
steps, tool_call_count, invalid_tool_calls, tool_errors, retrieved_chunks,
retrieval_hits, retrieval_total, retrieval_recall, answer_found, correct, score,
f1, format_reward, latency_s, turns`。

## 如何按 case 细看

1. 在 `<algo>_cases60.md` 找到感兴趣的 id（如 `ETR/2016/page_23.pdf-2`）。
2. 在 `../data/test_cases60.jsonl` 中 `grep '<id>'` 查看原始问题、gold program/evidence。
3. 在 `<algo>_cases60.jsonl` 中 `grep '<id>'` 查看完整多轮轨迹（`response` 为最终回答，
   `turns` 含每轮 prompt/response、工具调用结果）。
4. 对照 `score` 与 `correct` 理解奖励：`score = 结果匹配 × 效率 + 格式 − 工具惩罚`。

> 说明：已包含 GRPO / RLOO / REINFORCE / PPO 四组；Token-GAE 将在其训练与评测完成后加入。


## trajectories/ —— 完整多轮轨迹采样（含 turns）

| 文件 | 内容 |
|---|---|
| `<tag>_traj12.jsonl` | 同一批 test 前 12 题的**完整多轮轨迹**（每条记录含 `turns` 列表：每轮的 step、模型 response、stop 标记、工具调用与返回） |
| `<tag>_traj12.log` | 该次采样推理日志 |

已覆盖：base_react、grpo、rloo、reinforce、ppo_gae、token_gae。
每轮 `tools` 字段包含工具名、参数、observation（检索到的 chunk / 计算结果）与是否成功，
可据此完整复盘 Agent 的每一步决策。示例：

```bash
grep 'ETR/2016/page_23.pdf-2' samples/trajectories/grpo_traj12.jsonl | python -m json.tool
```
