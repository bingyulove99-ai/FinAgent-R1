# FinAgent-R1 学习指南

> 目标：从一个"会用 verl 跑脚本"的人，变成"能解释每一步为什么这样设计、能改框架、能定位训练异常"的人。
>
> 建议周期：4 周，每天 2–3 小时。所有路径相对工作区根目录。

---

## 0. 先建立一张地图

```
用户问题
  │
  ▼
FinQA parquet ──► RLHFDataset ──► AgentFlowWorker
                                      │
                                      ▼
                              FinQAAgentFlow.run
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            core.run_episode                    AgentFlowBase._postprocess
       （prompt/工具/scratchpad/停止）        （padding/mask/reward worker）
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                              DataProto（多 step）
                                      │
                                      ▼
                              RayAgentTrainer.fit
                     ┌────────────────┼────────────────┐
                     ▼                ▼                ▼
               advantage          critic update     actor update
             (GRPO/RLOO/...)      (if any)          (PPO clip + KL)
```

你需要能闭卷画出这张图，并解释每个箭头的输入输出。

---

## 第 1 周：框架与 MDP

### Day 1：读 README 与核心概念

读：

- `upstream/Agent-R1/README.md`
- `upstream/Agent-R1/docs/core-concepts/step-level-mdp.md`
- `upstream/Agent-R1/docs/core-concepts/layered-abstractions.md`

回答：

1. 为什么 Agent-R1 不把工具观测拼进 response 的 loss？
2. `AgentFlowBase`、`AgentEnvLoop`、`ToolEnv` 分别适合什么任务？
3. step-level MDP 的 state 和传统 RL 的 state 有什么不同？

### Day 2：读 `agent_flow.py`

重点：

- `AgentFlowStep`（:64）
- `AgentFlowBase`（:129）
- `_postprocess`（:265）
- `register`（:454）
- `AgentFlowWorkerBase.generate_sequences`（:525）
- `_postprocess`（:640）

练习：

- 手写一个最小的 `AgentFlowBase` 子类，只做一轮：用户问题 → LLM → `<answer>`，不调用工具。
- 解释 `response_mask` 在 `_postprocess` 里是如何被 `response_mask_ids * response_attention_mask` 构造的。
- 找到 `reward_tensor[0, valid_length - 1] = reward_score`，解释为什么 reward 放在最后一个有效 token。

### Day 3：读 `AgentEnvLoop` 与 `BaseTool`

重点：

- `agent_r1/agent_flow/agent_env_loop.py:100`
- `agent_r1/env/base.py` 的 `Observation` / `Action` / `AgentEnv`
- `agent_r1/tool/base.py` 的 `BaseTool.run`

练习：

- 用 `AgentEnvLoop` + 自定义 `AgentEnv` 实现一个"猜数字"环境（3 次机会，返回"大了/小了"）。
- 对比：这个环境用 `AgentFlowBase` 写会不会更简单？为什么？

### Day 4：读 rollout 路径

重点：

- `AgentFlowManager`（:820）
- `_initialize_llm_servers`（:856）
- `_init_agent_flow_workers`（:895）
- `generate_sequences`（:912）

练习：

- 解释 `num_workers`、`tensor_model_parallel_size`、`data_parallel_size` 三者的关系。
- 如果 4 张卡、TP=1、DP=1，vLLM 有几个 replica？
- 为什么每个样本 new 一个 flow 实例？这对 `__init__` 里加载大文件有什么要求？

### Day 5：读 trainer 前半段

重点：

- `RayAgentTrainer.fit`（`ray_trainer.py:876`）到 advantage 之前
- `batch.repeat(n)` + `sample_level_repeat(num_steps)` + `union`
- `_pad_dataproto_to_world_size`
- `_balance_batch`

练习：

- 画出一个 batch 从 `[B]` 到 `[B*n*T]` 的维度变化。
- 解释 `trajectory_uids` 和 `step_indices` 为什么不能用行号代替。

### Day 6：读 advantage 与 core_algos

重点：

- `AgentAdvantageEstimator`（:31）
- `compute_gae_advantage_return`（:66）
- `compute_token_gae_advantage_return`（:148）
- `compute_grpo_outcome_advantage`（:260）
- `compute_reinforce_outcome_advantage`（:347）
- `compute_rloo_outcome_advantage`（:393）
- `compute_gigpo_outcome_advantage`（:549）

练习（手算）：

假设 GRPO，G=2，两条轨迹回报 `R=[1.0, 0.0]`：
- 均值、标准差、两条轨迹的 advantage 各是多少？
- 如果 `norm_adv_by_std_in_grpo=False`，advantage 是多少？
- 如果 G=1，代码会怎么处理？

### Day 7：复盘 + 写笔记

用自己的话写一页：

- step-level MDP 与 token-level 的 3 个区别；
- `response_mask` 的 3 个作用；
- GRPO/RLOO/REINFORCE/PPO 的 1 句话区别；
- 你还不能解释的 3 个点。

---

## 第 2 周：本项目代码

### Day 8：数据管线

读：

- `projects/finagent_r1/finagent_r1/data/prepare_finqa.py`
- `data/dataset_stats.py`

练习：

- 用 fixtures 跑 `prepare_finqa.py`，打开 `dataset_summary.json`，解释每个字段。
- 修改 `chunk_text(max_chars=300)`，观察 corpus 数量和检索效果。
- 解释为什么 corpus 要从**全部**原始文档构建，而不是只从采样后的 QA 行构建。

### Day 9：工具

读：

- `tools.py` 的 `FinancialRetrieverTool`
- `PythonFinancialCalcTool`

练习：

- 手算 BM25 分数：query=`revenue 2021`，两个 chunk 的 tf/idf 差异。
- 构造一个 `top_k=3` 的检索，观察不同 query 的排序。
- 尝试绕过 AST 沙箱（`().__class__`、`eval`、`open`、`__import__`），确认都被拒绝。
- 解释 `_CORPUS_CACHE` 为什么必需。

### Day 10：奖励

读：

- `rewards.py`

练习：

- 列出 reward 的 4 个组成部分和权重。
- 构造 5 个边界案例：
  - `<answer>25%</answer>` vs `25.0`
  - `<answer>0.25</answer>` vs `25%`
  - `<answer>25</answer>` vs `100`
  - 没有 `<answer>` 但 analysis 里有正确数字
  - 答案里堆了 10 个数字
- 解释 `target_is_percent` 的作用，以及为什么不能无条件 `value*100`。
- 思考：这个奖励可能被怎样 hack？如何改？

### Day 11：episode loop

读：

- `core.py` 的 `run_episode`
- `_render_prompt`
- `_update_scratchpad`

练习：

- 画出 3 步轨迹的 messages 变化。
- 把 `max_obs_chars` 从 6000 改成 1000，观察 prompt token 数和最终答案质量。
- 把 `context_mode` 改成 `full`，观察 prompt 长度和平均步数。
- 解释 `recoverable_malformed` 的处理逻辑：为什么 malformed call 要消耗一步而不是直接终止？

### Day 12：Agent-R1 适配层

读：

- `flow.py`

练习：

- 对照 `AgentEnvLoop.run`，列出 `FinQAAgentFlow.run` 多了什么、少了什么。
- 解释 `reward_score=None if turn.stop else 0.0`。
- 解释 `extra_info` 里的 `step_count/tool_errors/invalid_tool_calls` 如何到达 reward 函数。
- 如果 `routed_experts` 不是 numpy 而是 list，`_postprocess` 会怎样？如何修？

### Day 13：评测

读：

- `evaluation/infer_vllm.py`
- `evaluation/evaluate_predictions.py`

练习：

- 用 fixtures 造一个 `preds.jsonl`，跑 `evaluate_predictions`。
- 解释 `seen_docs` / `unseen_docs` 的过滤逻辑。
- 如果模型输出 `<answer>25</answer>` 而 `exe_ans=0.25`、`answer="25%"`，会发生什么？
- 为什么评测要复用 `core.run_episode`，而不是另写一套 loop？

### Day 14：复盘 + 改一个小功能

选一个做：

- 给 retriever 加 `ticker`/`year` 之外的 `doc_id` 过滤；
- 给 reward 加"检索命中 gold evidence"的部分奖励；
- 给 scratchpad 加去重/摘要；
- 给 flow 加一个 `context_mode=full` 的消融开关（已有，验证它）。

写 200 字说明：改了什么、为什么、怎么测。

---

## 第 3 周：算法与系统

### Day 15：GRPO 推导

- 从 REINFORCE 的 `∇J = E[∇logπ * R]` 出发，推导带 baseline 的形式；
- 解释为什么 GRPO 用组均值/组标准差；
- 写出 `A_i` 公式，并说明 `std` 为 0 时的处理；
- 思考：GRPO 的 bias/variance 与 G 的关系。

### Day 16：PPO / GAE

- 推导 TD error、GAE、returns；
- 解释 `gamma` / `lam` 的极端取值（0/1）分别意味着什么；
- 解释 PPO clip 目标；
- 解释 value loss 为什么只在一个 token 上（step-level）或所有 token 上（token-level）。

### Day 17：RLOO / REINFORCE / StepPO

- 写出 RLOO 的留一均值；
- 比较 RLOO 与 GRPO 的 baseline；
- 解释 REINFORCE 的 step-level 折扣回报；
- 读上游 `examples/gsm8k/run_steppo.sh`，确认 StepPO 用的是 `gae`，并解释为什么本项目不提供 `adv_estimator=steppo`。

### Day 18：KL 与熵

- 解释 `low_var_kl` 相比普通 KL 的好处；
- 解释 `use_kl_in_reward` vs `use_kl_loss` 的区别；
- 为什么 KL 系数太大会让模型学不动，太小会 reward hacking；
- 熵系数的作用与副作用。

### Day 19：显存与分布式

- 自己算 3B/7B 的 FSDP 显存账本（不看文档）；
- 解释 FSDP ZeRO-3 分片了什么；
- 解释 vLLM 的 `gpu_memory_utilization` 和 `free_cache_engine`；
- 解释 A40 无 NVLink 对 4 卡训练的影响。

### Day 20：多轮 batching

读 `trajectory_batching.py`：

- 为什么 mini-batch 要按轨迹而不是按行？
- `sample_mask` 解决什么问题？
- 如果一条轨迹有 6 步，mini_batch_size=16，会发生什么？

### Day 21：复盘 + 画系统图

画一张包含 Ray、FSDP、vLLM、Ref、Critic、AgentFlowWorker 的部署图，标注每张卡上跑什么、什么时候切换。

---

## 第 4 周：实验与面试

### Day 22：跑通 GRPO 小实验

```bash
EPOCHS=1 TRAIN_BATCH_SIZE=16 ROLLOUT_N=4 \
bash projects/finagent_r1/scripts/train_grpo.sh
```

记录：

- 50/100/150 update 的 reward、格式率、平均步数；
- 验证集 accuracy；
- 显存峰值（`nvidia-smi`）；
- 吞吐（tokens/s）。

### Day 23：评测与对比

跑 base / direct / GRPO 三个配置，填结果表。

### Day 24：消融

选两个：

- 去掉 scratchpad；
- 去掉 sliding window；
- 去掉 efficiency；
- 去掉 tool penalty；
- BM25 top_k=1 vs 3 vs 5。

### Day 25：失败分析

从 predictions 里找 20 个错误，分类：

- 检索错（找到错误行/年份）；
- 计算错（公式/算术）；
- 格式错（没 answer/非法 JSON）；
- 过早停止；
- 死循环；
- ground truth 本身有歧义。

每类写一个改进假设。

### Day 26：读面试手册

`docs/04_interview_handbook.md`，先自己回答，再看答案。

### Day 27：模拟面试

找一个人（或录音）问你 10 个问题，要求：

- 每个回答 60–90 秒；
- 先给结论，再给 2–3 个理由；
- 不回避"没做过/没测过"，说明会怎么测。

### Day 28：整理作品集

- GitHub README（含架构图、数据卡、运行命令）；
- 一页 results 表（真实数据）；
- 3 个 failure case；
- 1 个你改过的框架点。

---

## 附录 A：必读代码清单

| 优先级 | 文件 | 重点 |
|---|---|---|
| P0 | `agent_r1/agent_flow/agent_flow.py` | Step/Base/Worker/registry |
| P0 | `agent_r1/trainer/ppo/ray_trainer.py` | fit、advantage dispatch |
| P0 | `agent_r1/trainer/ppo/core_algos.py` | 5 种 estimator |
| P0 | `projects/finagent_r1/finagent_r1/core.py` | episode loop |
| P0 | `projects/finagent_r1/finagent_r1/flow.py` | 适配层 |
| P1 | `agent_r1/agent_flow/agent_env_loop.py` | 标准环境循环 |
| P1 | `agent_r1/trainer/ppo/trajectory_batching.py` | 轨迹 batching |
| P1 | `projects/finagent_r1/finagent_r1/tools.py` | BM25/AST |
| P1 | `projects/finagent_r1/finagent_r1/rewards.py` | 奖励 |
| P2 | `agent_r1/workers/fsdp_workers.py` | FSDP/vLLM |
| P2 | `agent_r1/reward_loop/reward_loop.py` | reward worker |
| P2 | `agent_r1/config/config.py` | AgentFlowConfig |

## 附录 B：自测题（闭卷）

1. `AgentFlowStep.response_mask` 的 1 和 0 分别代表什么？
2. GRPO 的 advantage 为什么要按 `trajectory_uids` 汇总？
3. RLOO 与 GRPO 的 baseline 有何不同？
4. Step GAE 和 Token GAE 的 critic 分别在什么粒度上估 value？
5. 为什么 `reward_tensor` 只写在终止步的最后一个有效 token？
6. 为什么每条样本 new 一个 flow 实例？retriever 如何避免重复加载语料？
7. `doc_seen_in_train` 为什么重要？
8. 你的 reward 可能被怎样 hack？举 3 个例子。
9. 4×A40 上 3B GRPO 的显存大头是什么？PPO 多在哪里？
10. 如果训练 reward 不升，你会按什么顺序排查？

## 附录 C：推荐论文/资料

- Agent-R1: <https://arxiv.org/abs/2511.14460>
- verl: <https://verl.readthedocs.io/>
- FinQA: Chen et al., EMNLP 2021, <https://arxiv.org/abs/2109.00122>
- DeepSeek-R1: <https://arxiv.org/abs/2501.12948>
- GRPO / DeepSeekMath: <https://arxiv.org/abs/2402.03300>
- PPO: <https://arxiv.org/abs/1707.06347>
- RLOO: <https://arxiv.org/abs/2402.14740>
- GiGPO: 见 Agent-R1 相关引用
- ReAct: <https://arxiv.org/abs/2210.03629>
