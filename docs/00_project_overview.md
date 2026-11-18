# FinAgent-R1 项目文档

> **项目定位**：在 Agent-R1 的 step-level MDP 框架上，做一个可复现、可扩展、面向金融 filings 的个人 agentic RL 项目。
>
> **当前状态**：代码、数据管线、奖励函数、工具、CPU 测试和训练/评测脚本已完成；**仓库不包含任何训练日志、checkpoint 或实测准确率**。本文中的显存、耗时、token 量均为工程估算，最终以 4×A40 实机为准。任何对外展示的指标必须来自 `scripts/evaluate.sh` 的真实运行。

---

## 1. 项目背景

### 1.1 任务

FinQA 是一个基于 SEC 财报（10-K / 10-Q）的数值推理问答数据集：给定一份财报的正文与表格，回答需要**检索证据 + 多步计算**的问题，例如：

> "What was the percentage growth in revenue from 2020 to 2021?"

要回答正确，模型必须：

1. 找到正确的报表行（revenue 的 2020/2021 两列）；
2. 选择正确的公式（(100−80)/80×100）；
3. 精确执行算术；
4. 用带单位的最终答案结束。

这正好对应真实金融投研中的三类失败模式：

- **检索失败**：找错年份、找错指标、被相似行干扰；
- **计算幻觉**：7B 以下模型心算多位数除法、百分比、CAGR 时错误率很高；
- **流程崩溃**：prompt-based ReAct 容易输出非法 JSON、重复同一查询、死循环。

### 1.2 为什么是 agentic RL，而不是 RAG / SFT

| 方案 | 能学到什么 | 局限 |
|---|---|---|
| 单轮 RAG | 一次检索 + 一次回答 | 检索错了没有第二次机会；不能根据中间结果改写查询 |
| Prompt ReAct | 按提示词多轮调用工具 | 行为是"借来的"，小模型格式稳定性差；无法通过试错优化策略 |
| SFT | 模仿专家轨迹 | 只能模仿，遇到第 1 步偏差不会自我纠偏；需要大量高质量轨迹 |
| 单轮 RL（DeepSeek-R1 式） | 端到端优化答案 | 把多轮交互压成一条 token 序列，工具观测也进入 loss，信用分配粗糙 |
| **Step-level agentic RL（本项目）** | 以"一步交互"为决策单元，优化整条轨迹 | 工程复杂；奖励稀疏；需要可靠的环境和工具 |

本项目的核心论点不是"RL 一定比 SFT 好"，而是：**当任务本身是"检索→验证→计算→回答"的多步决策过程时，把每一步当作 MDP transition 来优化，比把整条轨迹当作一个 token 序列更符合问题的因果结构。**

### 1.3 与旧版实验文档的区别

工作区中此前的 `fin_research` 版本使用 13 个模板合成数据（553/647 行），并在文档中给出了 84.7% 等无法复现的数字。该版本已删除。本项目的原则是：

- 数据来自公开的 FinQA，规模、划分、文档重叠都可审计；
- 文档只写"怎么跑、怎么评"，不写没有日志支撑的指标；
- 训练和评测共用同一套 prompt / tool / parser / reward 代码，避免"训练一套、评测另一套"。

---

## 2. 数据集

### 2.1 来源与规模

| 项目 | 数值 |
|---|---|
| 数据集 | **FinQA**（Chen et al., EMNLP 2021） |
| 原始 QA 对 | **8,281** |
| 原始财报文档 | **2,789** 份（S&P 500 公司 filings） |
| 任务类型 | 表格/正文证据检索 + 数值推理（加减乘除、百分比、比率） |
| 标注 | 每题有人工标注的 reasoning program 与 `exe_ans` |
| 官方划分 | train **6,251** / dev **883** / test **1,147** |
| 官方仓库 | <https://github.com/cszyssrs/FinQA> |
| 常用 HF 镜像 | `dreamerdeo/finqa`、`ChanceFocus/finqa` |

> 数字以官方 release 为准。`prepare_finqa.py` 会在 `dataset_summary.json` 中打印**本次实际生成**的行数；如果镜像版本有差异，以生成结果为准。

### 2.2 训练/验证/测试划分

采用官方 question-level split，**不重新洗牌**，以便和 FinQA 文献对比：

| Split | 行数 | 占比 | 用途 |
|---|---:|---:|---|
| train | 6,251 | 75.5% | 只用于训练 |
| dev | 883 | 10.7% | 每 25 个 update 验证一次；早停/选 checkpoint |
| test | 1,147 | 13.9% | 最终只评一次 |

**重要：官方 split 是"问题级"而不是"文档级"。** 同一份 10-K 可能同时出现在 train 和 test 中。直接在 test 上报告高分，可能高估泛化能力。因此数据管线额外写了一个字段：

```json
"extra_info": {"doc_seen_in_train": true/false}
```

评测时分别报告：

1. `test / all`：完整 1,147 题；
2. `test / unseen_docs`：文档从未在 train 出现过的子集（真正的文档级泛化）；
3. `test / seen_docs`：文档在 train 出现过的子集（用于诊断记忆/检索）。

这样既保留与文献的可比性，又不把"见过同一份财报"包装成泛化能力。

### 2.3 检索语料构建

`prepare_finqa.py` 把每份财报拆成三类 chunk：

| kind | 来源 | 处理 |
|---|---|---|
| `pre_text` | 表格前的正文段落 | 按句切分，单 chunk ≤ 700 字符 |
| `table` | 表格 | 第一行作为表头；每 10 行一个 chunk，**每个 chunk 重复表头** |
| `post_text` | 表格后的正文 | 同 `pre_text` |

每条 corpus 记录：

```json
{
  "chunk_id": "ACME/2021/page_10.pdf::table::000123",
  "doc_id": "ACME/2021/page_10.pdf",
  "ticker": "ACME",
  "year": 2021,
  "kind": "table",
  "text": "|  | 2021 | 2020 |\n| Revenue | 100.0 | 80.0 |"
}
```

**规模估算**：2,789 份文档，每份通常产生 10–30 个 chunk，预计 **3 万–7 万个 chunk**，JSONL 约 100–200 MB。精确数量见 `data/finqa/DATASET_CARD.md`。

**检索 scope**：环境从 `extra_info.ticker/year` 取出公司代码和财年，注入 `financial_retriever`。策略只能在该公司该财年的 chunk 中检索，不能跨公司"抄答案"。如果 scope 在语料中找不到，工具默认回退到全语料并在 observation 中注明 `scope_fallback`（可通过 `allow_scope_fallback: false` 关闭）。

**一个需要说明的泄漏边界**：corpus 包含 train/dev/test 的全部文档。训练题按 ticker/year 检索时，可能读到同公司、同年份、但只出现在 test split 的另一份 filing 的 chunk。这不是答案泄漏（corpus 不含 QA 标注），但属于文档级信息重叠；因此最终必须同时报告 `unseen_docs` 子集。

### 2.4 Agent-R1 数据契约

每条 QA 行同时满足 verl `RLHFDataset` 和本项目 flow 的约定：

| 列 | 内容 |
|---|---|
| `prompt` | `[{"role": "user", "content": "<question>"}]`（scope 由 flow 从 `extra_info` 注入，避免重复） |
| `data_source` | `"finqa"` |
| `agent_name` | `"finqa_agent"`（路由到我们的 AgentFlow） |
| `reward_model` | `{"ground_truth": "<answer>", "style": "rule"}` |
| `extra_info` | `question_id`, `question`, `answer`, `exe_ans`, `is_numeric`, `program`, `gold_evidence`, `ticker`, `year`, `doc_id`, `split`, `doc_seen_in_train`, `corpus_path`（绝对路径，避免 Ray worker 丢失环境变量） |
| `env_kwargs` | `{"tools_kwargs": {"ticker": ..., "year": ...}}` 的 JSON 字符串 |

`program` 保留 FinQA 的符号程序（如 `divide(137, 183)`），用于统计"多步程度"和后续可选的 step-level reward shaping；训练时**不把 program 泄漏给模型**。

### 2.5 数据质量控制

`dataset_stats.py` 输出 `DATASET_CARD.md`，包含：

- 每个 split 的行数、唯一文档数、唯一 ticker 数；
- 数值题比例；
- question 字符长度 p50/p95；
- program 长度直方图（0/1/2/3/4+ 步）；
- 文档重叠行数；
- corpus chunk 数、每文档 chunk 数 min/median/max；
- 年份范围与 top tickers。

**建议的验收标准**（不通过就修数据，不要直接训练）：

- train/dev/test 行数与官方一致（或明确记录了子采样）；
- `corpus.jsonl` 中 100% chunk 有 ticker 和 year；
- 至少 95% 的 test 行能在 corpus 中找到同 ticker/year 的 chunk；
- 随机抽 20 题人工检查：问题、答案、`exe_ans`、scope 一致。

---

## 3. 模型选择

### 3.1 主模型：Qwen2.5-3B-Instruct

| 维度 | 说明 |
|---|---|
| 参数量 | ~3.1B |
| 精度 | bf16（A40 是 Ampere，支持 bf16） |
| 上下文 | 32K（本项目截到 4K prompt + 768 response） |
| 许可 | Apache 2.0 |
| 指令/工具能力 | 原生支持 chat template 与 tool schema，适合多轮工具调用 |
| 为什么不用更大 | 4×48GB 要同时放 Actor、Reference、Critic（PPO）和 vLLM KV cache；3B 能在不 offload 的情况下稳定训练，迭代快，适合个人项目 |

### 3.2 备选：Qwen2.5-7B-Instruct

7B 不是不能跑，而是**默认配置下余量很小**：

- GRPO（无 critic）：需要 Actor + Ref + vLLM，约 43–48GB/卡，必须把 `gpu_memory_utilization` 降到 0.30–0.35，并开启激活检查点；
- PPO（有 critic）：Actor + Critic + Ref + vLLM，4×48GB 基本必须 `param_offload` / `optimizer_offload` 或 LoRA，吞吐会明显下降；
- 如果一定要 7B，建议先跑 GRPO/RLOO，PPO 用 LoRA critic 或换 3B。

### 3.3 单卡显存账本（bf16 + FSDP ZeRO-3，4 卡均摊）

| 组成 | 3B | 7B |
|---|---:|---:|
| 参数（bf16） | 6.2 GB | 15.2 GB |
| 梯度（bf16） | 6.2 GB | 15.2 GB |
| AdamW 一阶+二阶（fp32） | 24.8 GB | 60.8 GB |
| Reference 模型（bf16，仅 KL） | 6.2 GB | 15.2 GB |
| **静态合计 / 4 卡** | **43.4 / 4 ≈ 10.9 GB** | **106.4 / 4 ≈ 26.6 GB** |
| vLLM（colocate） | 0.45 × 48 ≈ 21.6 GB | 0.30–0.35 × 48 ≈ 14–17 GB |
| 激活（micro-batch 4，梯度检查点） | 约 5–8 GB | 约 8–12 GB |
| **估算峰值/卡** | **约 38–41 GB** | **约 49–56 GB（需 offload）** |

PPO 额外加一个同样大小的 Critic，3B 时每卡再 +10.9 GB 静态占用；因此 PPO 脚本把 vLLM 降到 0.35，并建议 `TRAIN_BATCH_SIZE=16` 起步。

---

## 4. 任务形式化：Step-level MDP

Agent-R1 把一次多轮交互建模为：

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma, H)
$$

| 元素 | 本项目定义 |
|---|---|
| State $s_t$ | 第 $t$ 步模型看到的完整 prompt：system prompt + 问题 + filing scope + scratchpad + 最近 $k$ 条 observation + 错误反馈 |
| Action $a_t$ | 模型在第 $t$ 步生成的完整文本：`<analysis>` + 1–2 个 `<tool_call>`，或 `<answer>` |
| Transition $\mathcal{T}$ | 环境解析 action → 执行工具（检索/计算）→ 更新 scratchpad/observation → 构造 $s_{t+1}$ |
| Reward $r_t$ | 中间步 0；终止步由 `rewards.compute_score` 计算（结果正确性 + 格式 − 工具错误） |
| Horizon $H$ | `max_steps=6` |
| Discount $\gamma$ | GRPO/RLOO 用轨迹回报；REINFORCE/GAE 用 0.99 |

**关键点：State 由环境控制，而不是简单 append。** 本项目的环境在每一步做三件事：

1. **Evidence scratchpad**：把已验证的计算结果和检索到的关键段落压缩成短事实，避免长上下文淹没；
2. **Sliding window**：只向模型展示最近 `observation_window=2` 条工具观测，每条最多 3500 字符；
3. **Prompt budget**：按字符逐步收缩 observation / scratchpad / history，最后才做 token 级截断，保证 4K 预算内不爆。

这套机制对应论文中的 "flexible context management"：环境有权决定模型下一步看到什么。

---

## 5. 强化学习算法

### 5.1 算法总览

| 算法 | `algorithm.adv_estimator` | 是否需要 Critic | 基线/优势来源 | 本项目脚本 |
|---|---|---|:---:|---|
| GRPO | `grpo` | 否 | 同一 prompt 的 $G$ 条轨迹的组均值/组标准差 | `train_grpo.sh` |
| RLOO | `rloo` | 否 | 留一法均值 | `train_rloo.sh` |
| REINFORCE | `reinforce` | 否 | 步级折扣回报 + batch 白化 | `train_reinforce.sh` |
| PPO / step-level GAE | `gae` | 是 | Critic 的 step-level value + GAE | `train_ppo.sh` |
| Token-GAE | `token_gae` | 是 | Critic 的 token-level value + GAE | `train_token_gae.sh` |
| StepPO-style | `gae` | 是 | 与 PPO 相同；Agent-R1 用 step-level `gae` 暴露 StepPO | `train_ppo.sh` |
| GiGPO | `gigpo` | 否 | 组内再按 anchor observation 分组 | 未启用（需要 `anchor_obs`） |
| REMAX | `remax` | 否 | 贪心/采样对照 | 上游 agent flow 未实现 |

> **关于 StepPO 的准确性说明**：Agent-R1 v0.1.0 并没有一个名为 `steppo` 的 estimator。上游 `examples/gsm8k/run_steppo.sh` 使用的是 `algorithm.adv_estimator=gae`，差别在于轨迹以 **step** 为单位、奖励按 step 组织。因此本项目把 "StepPO-style" 实现为 `train_ppo.sh` 的 step-level GAE，而不是伪造一个不存在的算法开关。

### 5.2 GRPO

对每个 prompt $q$，采样 $G$ 条轨迹，轨迹总回报 $R_i$：

$$
A_i = \frac{R_i - \operatorname{mean}(R_1,\dots,R_G)}
{\operatorname{std}(R_1,\dots,R_G) + \epsilon}
$$

同一轨迹内所有 step/token 共享 $A_i$。Agent-R1 的 `compute_grpo_outcome_advantage` 先按 `trajectory_uids` 把 step reward 求和成轨迹回报，再广播回每个 step 的每个 action token。

- 优点：不需要 critic，显存省一半；组内比较天然消除 prompt 难度差异；
- 缺点：$G$ 太小方差大；$G$ 太大 rollout 成本高；全对/全错的组没有梯度。

### 5.3 RLOO

$$
A_i = R_i - \frac{1}{G-1}\sum_{j \ne i} R_j
$$

不做 std 归一化。相比 GRPO，RLOO 的基线是无偏的留一均值，通常方差略低，但对 $G$ 的要求相同（$G \ge 2$，建议 8）。

### 5.4 REINFORCE

对 step-level 回报做折扣：

$$
G_t = \sum_{k=t}^{H} \gamma^{k-t} r_k
$$

Agent-R1 的 `compute_reinforce_outcome_advantage` 在 step 级计算 $G_t$，再对整个 batch 做白化：

$$
A_t = \frac{G_t - \mu_G}{\sigma_G + \epsilon}
$$

无 critic，实现简单；但白化是 batch 级的，和 prompt 难度耦合，通常不如 GRPO/RLOO 稳定。

### 5.5 PPO / GAE（step-level）

Critic 学 $V_\phi(s_t)$，step-level TD 误差：

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

$$
A_t = \sum_{l=0}^{H-t} (\gamma\lambda)^l \delta_{t+l}
$$

Agent-R1 在 `core_algos.compute_gae_advantage_return` 中按 `trajectory_uids`/`step_indices` 做 step 级 GAE；`critic_vf_loss_response_mask` 在非 token-GAE 模式下只对每个 step 的第一个 token 计算 value loss，避免 critic 被工具观测 token 干扰。

### 5.6 Token-GAE

与 PPO 相同的 GAE 公式，但把"时间步"换成 action 内部的 token。它给每个生成 token 一个 value，信用分配最细，但：
- critic 训练目标更稀疏、更 noisy；
- 显存和计算量更高；
- 在多轮工具场景下不一定优于 step-level。

本项目把它作为**消融**，而不是默认。

### 5.7 KL、熵与优势归一化

- `actor.use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl`：约束策略不要偏离 SFT 初始模型太远，防止 reward hacking；
- `entropy_coeff=0.001`：轻微熵正则，维持探索；
- `norm_adv_by_std_in_grpo=True`：GRPO 原始做法；若训练不稳定，可试 Dr.GRPO 的 `False`；
- `algorithm.use_kl_in_reward=False`：KL 放在 actor loss 里，而不是 reward 里，避免 reward 尺度随 KL 漂移。

---

## 6. 训练配置与轮次

### 6.1 默认超参数（4×A40，Qwen2.5-3B）

| 参数 | 值 | 说明 |
|---|---:|---|
| `trainer.n_gpus_per_node` | 4 | 单机 4 卡 |
| `data.train_batch_size` | 32 prompts | 每个 update 的 prompt 数 |
| `actor_rollout_ref.rollout.n` | 8（GRPO/RLOO/REINFORCE） | 每 prompt 的轨迹数 |
| | 4（PPO/Token-GAE） | 有 critic 时降低 rollout 成本 |
| `ppo_mini_batch_size` | 16 | PPO mini-batch |
| `ppo_micro_batch_size_per_gpu` | 4 | 单卡 micro-batch |
| `actor.optim.lr` | 1e-6 | AdamW |
| warmup | 5% | `lr_warmup_steps_ratio=0.05` |
| `kl_loss_coef` | 0.001 | low-var KL |
| `entropy_coeff` | 0.001 | |
| `max_prompt_length` | 4096 | |
| `max_response_length` | 768 | 单步响应 |
| `max_steps` | 6 | 每条轨迹最多 6 次 LLM 调用 |
| `trainer.total_epochs` | 2 | 默认；首次建议 `EPOCHS=1` |
| `save_freq` / `test_freq` | 50 / 25 | 每 50 个 update 存 checkpoint，每 25 个验证 |
| `rollout.gpu_memory_utilization` | 0.45（无 critic）/ 0.35（有 critic） | |

### 6.2 训练步数与 token 估算

- 每个 epoch 的 update 数：$6251 / 32 \approx 196$；
- 2 epochs：约 **392 个 update**；
- 每个 update 的轨迹数：$32 \times 8 = 256$（GRPO/RLOO）；
- 平均每条轨迹约 3–4 步，每步 prompt+response 约 1,500–2,500 token；
- 粗估每个 epoch **3–6 亿 token**（含 padding 和重复上下文）。

在 4×A40 上，1 epoch 大约 **12–36 小时**，取决于平均步数、vLLM 吞吐、是否 `remove_padding`、数据加载和 PCIe 开销。首次实验建议：

```bash
EPOCHS=1 TRAIN_BATCH_SIZE=16 bash projects/finagent_r1/scripts/train_grpo.sh
```

先确认 50–100 个 update 内 reward/格式率有上升，再放大 batch 和 epoch。

### 6.3 训练过程中应观察的指标

| 指标 | 期望趋势 | 异常信号 |
|---|---|---|
| `critic/score/mean` 或 reward | 上升 | 长期不动：奖励太稀疏/工具不可用 |
| `actor/entropy` | 缓慢下降 | 迅速归零：熵系数太小/学习率太大 |
| `response_length` | 稳定或略降 | 暴涨：模型在刷长度 |
| `num_steps`（`agent_flow/...`） | 下降 | 一直 6 步：模型学不会停止 |
| invalid tool call 比例 | 下降 | 不降：prompt/格式或 parser 有问题 |
| KL | 小幅上升后稳定 | 爆炸：KL 系数太小或 lr 太大 |
| `val` accuracy | 上升 | 训练升验证降：过拟合/文档泄漏 |

---

## 7. Agent-R1 框架如何使用

### 7.1 三层抽象

| 层 | 类 | 本项目用法 |
|---|---|---|
| Layer 1 | `AgentFlowBase` | `FinQAAgentFlow` 继承它，完全控制多轮循环、上下文和 step 打包 |
| Layer 2 | `AgentEnvLoop` + `AgentEnv` | 未使用；我们的环境逻辑直接写在 flow 里（检索/计算/scratchpad） |
| Layer 3 | `ToolEnv` + `BaseTool` | 工具类继承 `BaseTool` 并注册，但由 flow 直接调用，便于并行和 scope 注入 |

选择 Layer 1 的原因：FinQA 需要"检索结果写 scratchpad + 滑动窗口 + 动态 prompt 预算"，这些不是标准 `reset/step` 环境能自然表达的。

### 7.2 一次训练的完整数据流

```
Parquet (prompt/agent_name/reward_model/extra_info)
  → RLHFDataset
  → AgentFlowWorker.generate_sequences
  → _run_agent_flow(agent_name="finqa_agent")
  → FinQAAgentFlow.run
      → core.run_episode
          → render prompt (system + user + tools)
          → server_manager.generate (vLLM)
          → parse <tool_call>/<answer>
          → execute retriever / calculator
          → update scratchpad + observations
      → 每个 turn 包成 AgentFlowStep
      → 中间步 reward=0.0；终止步 reward=None
      → AgentFlowBase._postprocess:
          padding、response_mask、reward worker
  → AgentFlowWorker._postprocess:
      concat steps、trajectory_uids、step_indices、rm_scores
  → RayAgentTrainer.fit:
      repeat n、sample-level repeat num_steps、union rollout
      → token_level_scores = rm_scores
      → KL penalty (optional)
      → compute_advantage(grpo/rloo/reinforce/gae/token_gae)
      → critic update (if any)
      → actor update
```

### 7.3 关键 mask

- `response_mask`：LLM 生成的 token 为 1，工具 observation 和 padding 为 0。**环境观测绝不回传梯度**。
- `attention_mask`：padding 为 0。
- `sample_mask`：trajectory-aware mini-batch 的 padding 行。
- `rm_scores`：只有终止步的最后一个有效 token 有非零 reward；其余为 0。

### 7.4 本项目的扩展点

| 扩展点 | 文件 | 说明 |
|---|---|---|
| AgentFlow | `projects/finagent_r1/finagent_r1/flow.py` | `@register("finqa_agent")` |
| 工具 | `projects/finagent_r1/finagent_r1/tools.py` | `financial_retriever`、`python_calc` |
| 奖励 | `projects/finagent_r1/finagent_r1/rewards.py` | `compute_score` |
| 数据 | `projects/finagent_r1/finagent_r1/data/prepare_finqa.py` | FinQA → parquet/corpus |
| 训练脚本 | `projects/finagent_r1/scripts/train_*.sh` | 5 种算法 |
| 评测 | `projects/finagent_r1/finagent_r1/evaluation/` | vLLM 推理 + 离线打分 |

所有扩展都不修改 `upstream/Agent-R1`，只通过 `PYTHONPATH` 和 Hydra 配置注入。

---

## 8. 评估协议

### 8.1 指标

| 指标 | 定义 |
|---|---|
| **Numeric accuracy** | 从 `<answer>` 中抽取数字，与 `exe_ans` 比较；相对容差 `1e-3`、绝对容差 `1e-4`；百分比与小数等价 |
| **Token F1** | 非数值答案的 token-level F1 |
| **Composite reward** | 与训练奖励完全一致：outcome × efficiency + format − tool penalty |
| **Answer rate** | 有非空 `<answer>` 的比例 |
| **Format rate** | 同时出现 `<analysis>` 和 `<answer>` 的比例 |
| **Avg steps** | 每条轨迹的 LLM 调用次数 |
| **Avg tool calls** | 每条轨迹的工具调用次数 |
| **Invalid-call rate** | malformed tool call / 总 tool call |
| **Tool-error rate** | 工具报错 / 总 tool call |
| **Retrieval chunks** | 每条轨迹检索到的 chunk 数 |
| **Retrieval recall** | 检索到的 observation 是否覆盖 FinQA `gold_evidence`（前 80 字符归一化匹配；诊断指标，不参与训练奖励） |

### 8.2 评测子集

- `all`：完整 test；
- `seen_docs`：文档在 train 出现过；
- `unseen_docs`：文档从未在 train 出现（更严格的泛化）。

### 8.3 必须报告的对照

| 配置 | 目的 |
|---|---|
| Qwen2.5-3B zero-shot direct（无工具） | 下界，测模型自身金融知识 |
| Qwen2.5-3B zero-shot ReAct（同一 flow、同一工具） | 测 RL 相对 prompt 工程的增益 |
| + SFT（可选） | 测 RL 相对行为克隆的增益 |
| GRPO / RLOO / REINFORCE / PPO / Token-GAE | 算法横向对比 |
| 去掉 scratchpad / 去掉 sliding window | 上下文消融 |
| 去掉 tool penalty / 去掉 efficiency | 奖励消融 |

### 8.4 结果表模板（训练后填写）

> 下面所有单元格都必须来自 `results/*.metrics.json`，不要手填估计值。

| 配置 | Numeric acc (all) | Numeric acc (unseen) | Format | Avg steps | Tool validity | Tool error |
|---|---:|---:|---:|---:|---:|---:|
| 3B direct | 待填 | 待填 | 待填 | 1.0 | — | — |
| 3B zero-shot ReAct | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 3B + GRPO | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 3B + RLOO | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 3B + REINFORCE | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 3B + PPO/step-GAE | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 3B + Token-GAE | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

### 8.5 统计显著性

- 1,147 题上比较两个算法，建议用 paired bootstrap（1,000 次）或 McNemar 检验；
- 报告均值 ± 95% CI；
- 至少 2 个 seed（如果预算允许），否则明确声明单 seed；
- 不要只报最好 checkpoint，报告选择规则（如 dev accuracy 最高的 checkpoint）。

---

## 9. 代码与目录

```text
upstream/Agent-R1/                 # 原框架，不改
projects/finagent_r1/
├── finagent_r1/
│   ├── prompts.py
│   ├── parsing.py
│   ├── tools.py
│   ├── rewards.py
│   ├── core.py
│   ├── flow.py
│   ├── configs/flow.yaml
│   ├── data/prepare_finqa.py
│   ├── data/dataset_stats.py
│   └── evaluation/{infer_vllm,evaluate_predictions}.py
├── scripts/{setup_env,prepare_data,smoke_test,train_*,evaluate,merge_checkpoint}.sh
├── tests/                          # 60 个 CPU-only 单测
└── requirements.txt
docs/                               # 本文档 + 框架指南 + runbook + 学习指南 + 面试手册
data/finqa/                         # 生成数据（gitignored）
outputs/                            # 训练输出（gitignored）
```

---

## 10. 风险、局限与后续工作

### 10.1 已知局限

1. **检索是词法的**：BM25 对同义词、缩写（"net sales" vs "revenue"）不如稠密检索。后续可换 BGE/FAISS，但需要额外 GPU/内存，且要重新测延迟。
2. **scope 由数据注入**：模型不需要自己判断公司/年份，任务比"开放域金融检索"简单；这是为了在 4×A40 上可控。
3. **奖励是结果制**：中间步 reward=0，信用分配仍偏稀疏；step-level reward shaping（例如检索命中 gold evidence 给部分奖励）是后续方向，但要防止 reward hacking。
4. **文档泄漏**：必须同时报告 `unseen_docs`。
5. **没有实测结果**：所有性能结论需要你自己跑。
6. **7B/PPO 显存风险**：默认配置针对 3B；7B 需要 offload/LoRA。
7. **数据许可**：FinQA 来自公开财报，但使用前请确认上游许可和公司内部合规要求。

### 10.2 后续可做的实验

- 稠密检索（BGE + FAISS）vs BM25；
- scratchpad / sliding window / full-context 消融；
- step-level reward shaping（检索命中、计算正确性）；
- SFT 冷启动 + RL；
- GiGPO（需要为环境提供 anchor observation）；
- LoRA vs full fine-tuning；
- 7B 在 4×A40 上的 offload 配置。

---

## 11. 一页纸总结

- **数据**：FinQA 8,281 题 / 2,789 份财报；官方 6,251/883/1,147；额外标记文档级 unseen。
- **模型**：Qwen2.5-3B-Instruct，bf16，FSDP ZeRO-3，4×A40 48GB 可无 offload 跑 GRPO/RLOO/REINFORCE；PPO 需降 vLLM 显存。
- **任务**：step-level MDP；state 由环境重塑；action 是 analysis + tool calls / answer；reward 在终止步。
- **算法**：GRPO、RLOO、REINFORCE、PPO/step-GAE、Token-GAE；StepPO 通过 step-level GAE 复现。
- **训练**：32 prompts/update × 8 rollouts，2 epochs ≈ 392 updates；1 epoch 约 12–36 小时（估算）。
- **评测**：numeric accuracy + 格式/步数/工具指标；all 与 unseen_docs 分开报告。
- **原则**：不伪造结果；训练与评测共用同一套 core 代码；扩展代码与 upstream 完全分离。
