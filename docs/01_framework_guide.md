# Agent-R1 框架深度解读

> 本文讲的是 `upstream/Agent-R1/` 里的**原框架**，以及本项目如何在**不改 upstream** 的前提下接入。所有行号以当前工作区版本为准，升级 upstream 后请重新定位。

---

## 1. 框架定位

Agent-R1 不是"又一个 PPO 实现"，它解决的是 **LLM agent 训练中的轨迹表示问题**：

- 单轮 RL 把一次交互视为 `prompt -> response`，工具调用只是 response 里的文本；
- 多轮 agent 的真实交互是 `state_0 -> action_0 -> env -> state_1 -> action_1 -> ...`，其中 `state_t` 由环境决定，action 之间还夹着工具观测；
- 如果直接把所有东西拼成一条 token 序列，会有三个问题：
  1. 工具观测 token 被算进 loss；
  2. 无法在步骤之间裁剪/重写上下文；
  3. 信用分配只能在 token 级做，无法表达"第 2 步的检索动作导致了最终失败"。

Agent-R1 的答案是：**以 step 为轨迹的基本单元**，每个 step 记录 `prompt_ids / response_ids / response_mask / reward / num_turns`，再在 step 内部做 token-level policy loss。

---

## 2. 目录与入口

```
agent_r1/
├── agent_flow/
│   ├── agent_flow.py          # AgentFlowBase, AgentFlowStep, Worker, Manager, registry
│   ├── agent_env_loop.py      # AgentEnvLoop: 标准 reset/step 循环
│   └── single_step_agent_flow.py
├── env/
│   ├── base.py                # AgentEnv, Observation, Action
│   └── envs/tool.py           # ToolEnv
├── tool/
│   ├── base.py                # BaseTool registry
│   └── schema.py              # OpenAI function schema / ToolResponse
├── reward_loop/reward_loop.py # RewardLoopWorker, custom reward fn
├── trainer/
│   ├── main_agent_ppo.py      # Hydra entrypoint + Ray TaskRunner
│   └── ppo/
│       ├── ray_trainer.py     # RayAgentTrainer, fit(), advantage dispatch
│       ├── core_algos.py      # GAE / GRPO / RLOO / REINFORCE / GiGPO
│       ├── trajectory_batching.py
│       └── metric_utils.py
├── workers/
│   ├── fsdp_workers.py        # AsyncActorRolloutRefWorker, CriticWorker
│   └── engine_workers.py      # 新 engine 实现
└── config/
    ├── agent_ppo_trainer.yaml
    └── overrides/rollout.yaml
```

训练入口：

```bash
python3 -m agent_r1.trainer.main_agent_ppo <hydra overrides>
```

`main_agent_ppo.py` 初始化 Ray，创建 `TaskRunner`，在 `TaskRunner.run` 里：

1. 注册 `ActorRollout` / `Critic` / `RefPolicy` / `RewardModel` 的 Ray worker；
2. `validate_config` 检查是否需要 critic / ref；
3. 加载 tokenizer、dataset、reward manager；
4. 创建 `RayAgentTrainer`，`init_workers()`，`fit()`。

---

## 3. Step-level MDP

### 3.1 形式化

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{T},\mathcal{R},\gamma,H)
$$

- $s_t$：第 $t$ 步的 observation，**由环境构造**；
- $a_t$：第 $t$ 步 LLM 生成的完整文本（可以包含 CoT、tool call、answer）；
- $\mathcal{T}(s_t,a_t)$：解析 action、执行工具、更新环境、返回 $s_{t+1}$；
- $r_t$：即时奖励；本项目只在终止步给非零；
- $H$：`max_steps`。

### 3.2 与 token-level MDP 的差别

| 维度 | token-level | step-level |
|---|---|---|
| 时间单位 | 每个 token | 每次 LLM 调用 |
| 环境观测 | 在序列中间，和 action 混在一起 | 独立字段，`response_mask=0` |
| 上下文 | append-only | 可裁剪、摘要、重写 |
| 信用分配 | token GAE | step GAE / 轨迹组比较 |
| 停止 | 生成 EOS | 环境判断 `done` 或 `max_steps` |

### 3.3 轨迹表示

`AgentFlowStep`（`agent_r1/agent_flow/agent_flow.py:64`）：

```python
class AgentFlowStep(BaseModel):
    prompt_ids: list[int]
    response_ids: list[int]          # LLM tokens + tool observation tokens
    response_mask: list[int] | None  # 1=LLM token, 0=tool/pad
    response_logprobs: list[float] | None
    reward_score: float | None
    num_turns: int = 2
    extra_fields: dict[str, Any]
```

一个多轮轨迹在 `AgentFlowWorkerBase._postprocess` 里被拼成 `DataProto`：

- `prompts`: `[num_steps, prompt_length]`
- `responses`: `[num_steps, response_length]`
- `response_mask`: `[num_steps, response_length]`
- `rm_scores`: 只有终止步最后一个有效 response token 有值
- `non_tensor_batch["trajectory_uids"]`: 同一条轨迹的多个 step 共享一个 uid
- `non_tensor_batch["step_indices"]`: 0..T-1
- `non_tensor_batch["__num_turns__"]`: 每条轨迹的步数

这三者让 advantage estimator 能"把 step 还原成轨迹"。

---

## 4. 三层抽象

| 层 | 类 | 控制权 | 适用场景 |
|---|---|---|---|
| Layer 1 | `AgentFlowBase`（`agent_flow.py:129`） | 完全自定义 | 复杂 agent、动态上下文、多智能体 |
| Layer 2 | `AgentEnvLoop`（`agent_env_loop.py:21`） + `AgentEnv`（`env/base.py:41`） | 标准 reset/step | 有明确环境状态的任务（ALFWorld、WebShop） |
| Layer 3 | `ToolEnv` + `BaseTool`（`tool/base.py:7`） | 只定义工具 | 单/多工具调用（GSM8K、HotpotQA） |

### 4.1 AgentFlowBase 的钩子

- `__init__`：拿到 `trainer_config`、`server_manager`（vLLM 异步客户端）、`reward_loop_worker`、tokenizer、dataset；
- `apply_chat_template(messages, tools, ...)`：把 chat messages 转 token ids；
- `run(sampling_params, **kwargs)`（抽象）：子类实现多轮循环；
- `_postprocess(step, **kwargs)`：padding、position_ids、attention_mask、reward 计算；
- `_compute_score`：如果 `reward_score is None`，调用 `RewardLoopWorker.compute_score`。

**关键约定**：子类每生成一轮，就构造一个 `AgentFlowStep` 并 `await self._postprocess(step, **kwargs)`。中间步 `reward_score=0.0`，终止步 `reward_score=None`。

### 4.2 AgentEnvLoop

`AgentEnvLoop.run`（`agent_env_loop.py:100`）：

```python
obs = env.reset(**kwargs)
for step_idx in range(max_steps):
    prompt_ids = await self._obs_to_prompt(obs, tools)
    output = await self.server_manager.generate(...)
    action = Action(text=response_text, token_ids=response_ids)
    next_obs, reward, done, info = await env.step(action)
    step = AgentFlowStep(..., reward_score=reward)
    step = await self._postprocess(step, **kwargs)
    if done: break
    obs = next_obs
```

它要求环境自己维护状态、解析 action、返回 reward/done。

### 4.3 ToolEnv / BaseTool

- `BaseTool` 用 `@BaseTool.register("name")` 注册；
- `ToolEnv` 解析 `<tool_call>`，调用 `BaseTool.from_name(...)`，把 `ToolResponse` 写回 observation；
- 工具的 `execute(args, **kwargs) -> (ToolResponse, reward, meta)`。

本项目的工具**注册了但不由 `ToolEnv` 调用**，而是在 `FinQAAgentFlow` 里直接并行调用，原因是需要：
- 注入 ticker/year scope；
- 控制 `max_parallel_calls`；
- 把检索/计算结果写入 scratchpad；
- 统计工具错误和 retrieved chunks。

---

## 5. Rollout 路径

### 5.1 Manager / Worker

`AgentFlowManager`（`agent_flow.py:820`）：

1. 根据 `rollout.tensor_model_parallel_size × data_parallel_size × pipeline_model_parallel_size` 计算 vLLM replica 数；
2. 初始化 vLLM server（`init_hybrid` 或 `init_standalone`）；
3. 创建 `num_workers` 个 `AgentFlowWorker`（Ray actor，CPU 侧）；
4. `generate_sequences` 把 batch 切块分给 worker，`ray.get` 后 `DataProto.concat`。

`AgentFlowWorkerBase.generate_sequences`（`agent_flow.py:525`）：

1. 读 sampling params（temperature/top_p/logprobs）；
2. 如果 batch 没有 `agent_name`，填 `default_agent_flow`；
3. 为每条样本创建 `_run_agent_flow` task；
4. `asyncio.gather` 并发跑所有样本；
5. `_postprocess` 合并。

`_run_agent_flow`（`agent_flow.py:604`）：

```python
assert agent_name in _agent_flow_registry
agent_flow = hydra.utils.instantiate(config=..., **shared)
output = await agent_flow.run(sampling_params, **kwargs)
```

**注意**：每条样本都会 new 一个 flow 实例。所以 flow 的 `__init__` 里不要加载大文件；本项目的 retriever 用模块级 `_CORPUS_CACHE` 缓存语料。

### 5.2 注册机制

```python
_agent_flow_registry: dict[str, dict] = {}

def register(agent_name):
    def decorator(subclass):
        _agent_flow_registry[agent_name] = {"_target_": f"{module}.{qualname}"}
        return subclass
    return decorator
```

YAML 方式：

```yaml
- name: finqa_agent
  _target_: finagent_r1.flow.FinQAAgentFlow
  max_steps: 6
```

训练参数：

```
actor_rollout_ref.rollout.agent.agent_flow_config_path=.../flow.yaml
actor_rollout_ref.rollout.agent.default_agent_flow=finqa_agent
```

`AgentFlowWorkerBase.__init__` 会 `OmegaConf.load` 这个 YAML 并写入 registry。

---

## 6. Step 打包与 mask

`AgentFlowBase._postprocess`（`agent_flow.py:265`）做以下事：

1. `extra_fields["raw_prompt"] = kwargs["raw_prompt"]`；
2. 对 `prompt_ids` 做 left padding 到 `rollout.prompt_length`；
3. 对 `response_ids` 做 right padding 到 `rollout.response_length`；
4. `response_mask = response_mask_ids * response_attention_mask`；
5. `attention_mask = concat(prompt_mask, response_mask)`；
6. `input_ids = concat(prompt_ids, response_ids)`；
7. `position_ids = compute_position_id_with_mask(attention_mask)`；
8. 如果 `reward_score is None`，调用 `_compute_score`；
9. 返回 `_InternalAgentFlowStep`。

### 6.1 为什么 response_mask 很重要

多轮轨迹的 `responses` 里既有 LLM token，也有工具返回的文本：

```
responses:     |<- LLM action ->|<- tool observation ->|<- LLM action ->|<- pad ->|
response_mask: | 1 1 1 ... 1 1 | 0 0 0 ... 0 0 0 0 0 | 1 1 1 ... 1 1 | 0 ... 0 |
```

- policy loss 只在 `response_mask=1` 上计算；
- 工具观测不产生梯度；
- advantage 广播到 action token，不污染 observation。

**本项目的实现选择**：`FinQAAgentFlow` 的每个 `AgentFlowStep.response_ids` **只包含 LLM 生成的 token**，不把工具观测拼进 response；工具观测会进入下一步的 prompt。因此每个 step 的 `response_mask` 实际上全为 1（除 padding），reward 落在最后一个 token 上是正确的。这样做的代价是轨迹的 prompt 会重复携带历史，但换来了更清晰的 step 边界和更安全的 mask。

### 6.2 reward 放在哪

`AgentFlowWorkerBase._postprocess`（`agent_flow.py:640`）：

```python
reward_tensor = torch.zeros_like(step.response_mask)
valid_length = step.response_mask.sum().item()
reward_tensor[0, valid_length - 1] = float(step.reward_score)
```

即：**终止步的最后一个有效 action token** 拿到 reward，其余为 0。GRPO 再按 `trajectory_uids` 把 step reward 求和成轨迹回报。

---

## 7. Trainer 数据流

`RayAgentTrainer.fit`（`ray_trainer.py:876`）核心：

```python
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
batch = batch.repeat(repeat_times=rollout.n, interleave=True)
num_steps = gen_batch_output.meta_info.pop("num_steps")
batch = batch.sample_level_repeat(num_steps)
batch = batch.union(gen_batch_output)
...
batch.batch["token_level_scores"] = reward_tensor   # 来自 rm_scores
batch.batch["token_level_rewards"] = token_level_scores
batch = compute_advantage(batch, adv_estimator=..., gamma=..., lam=...)
if self.use_critic: self._update_critic(batch)
if self.global_steps >= critic_warmup: self._update_actor(batch)
```

关键点：

- `repeat(n)`：同一 prompt 的 $n$ 条 rollout 复制成 $n$ 行；
- `sample_level_repeat(num_steps)`：每条轨迹再按 step 数展开；
- `union(gen_batch_output)`：把 rollout 产生的 tensor 合进来；
- `rm_scores` → `token_level_scores`；
- KL 可以在 reward 里（`use_kl_in_reward=True`）或在 actor loss 里（`use_kl_loss=True`）；
- advantage 在 driver 进程算，不在 worker 里算。

### 7.1 trajectory-aware mini-batch

`trajectory_batching.py:17` 的 `prepare_trajectory_mini_batch`：

- 按 `trajectory_uids` 把整条轨迹放进同一个 mini-batch，避免同一条轨迹的 step 被拆到不同卡；
- 用 `sample_mask` 标记为对齐 DP world size 而补的 padding 行；
- padding 行的 loss mask 置 0。

这对多轮 RL 很关键：如果同一条轨迹的 step 被拆开，GAE 的跨 step 递归和 GRPO 的轨迹回报都会错。

### 7.2 critic 的 value mask

`critic_vf_loss_response_mask`（`ray_trainer.py:211`）：

- `token_gae`：`response_mask` 原样，每个 token 都算 value loss；
- 其他 estimator（step-level GAE）：只保留每个 step 的**第一个 action token**，即 `value_mask[:, 0] = response_mask[:, 0]`。

原因：step-level critic 只需要"这一步开始前的 state value"，不需要每个 token 的 value。

---

## 8. 优势估计器

`AgentAdvantageEstimator`（`core_algos.py:31`）：

```python
GAE = "gae"
TOKEN_GAE = "token_gae"
GRPO = "grpo"
REINFORCE = "reinforce"
REMAX = "remax"
RLOO = "rloo"
GIGPO = "gigpo"
```

`compute_advantage`（`ray_trainer.py:220`）根据 estimator 分派：

| estimator | 函数 | 关键输入 | 是否需要 critic |
|---|---|---|---|
| `gae` | `compute_gae_advantage_return` | values, trajectory_uids, step_indices | 是 |
| `token_gae` | `compute_token_gae_advantage_return` | values, response_mask | 是 |
| `grpo` | `compute_grpo_outcome_advantage` | index(uid), trajectory_uids | 否 |
| `reinforce` | `compute_reinforce_outcome_advantage` | trajectory_uids, step_indices | 否 |
| `rloo` | `compute_rloo_outcome_advantage` | index(uid), trajectory_uids | 否 |
| `gigpo` | `compute_gigpo_outcome_advantage` | anchor_obs, index, trajectory_uids | 否 |

`need_critic_agent_ppo`（`ray_trainer.py:200`）：只有 `gae` / `token_gae` 需要 critic（除非 config 显式设置 `critic.enable`）。

### 8.1 GRPO 的实现细节

`compute_grpo_outcome_advantage`（`core_algos.py:260`）：

1. `_trajectory_total_scores`：把每个 step 的 `token_level_rewards * response_mask` 求和，再按 `trajectory_uids` 汇总；
2. 按 `index`（prompt uid）分组，算组均值/标准差；
3. 每条轨迹一个 advantage；
4. 广播回该轨迹的每个 step、每个 token，乘 `response_mask`。

### 8.2 RLOO

`compute_rloo_outcome_advantage`（`core_algos.py:393`）：留一均值作为 baseline，不做 std 归一化。

### 8.3 REINFORCE

`compute_reinforce_outcome_advantage`（`core_algos.py:347`）：

- `compute_step_discounted_returns` 按 step 做折扣回报；
- 对整个 batch 做 whitening；
- 无 critic。

### 8.4 Step GAE vs Token GAE

- Step GAE：把每个 step 当成一个"时间步"，critic 只在 step 起点估 value；GAE 跨 step 递归。
- Token GAE：把 action 内部每个 token 当成时间步，跳过 `response_mask=0` 的观测 token，不推进递归；critic 每个 token 一个 value。

多轮工具场景通常 step GAE 更稳，因为工具观测的长度和噪声会污染 token-level critic。

### 8.5 GiGPO

`compute_gigpo_outcome_advantage`（`core_algos.py:549`）：

- 先算轨迹级相对 advantage（类似 GRPO）；
- 再按 `anchor_obs` 把不同轨迹中"相同/相似 observation"的 step 分到一组，算 step-level 相对 advantage；
- 最后用 `step_advantage_w` 加权。

它要求 flow 在每个 step 的 `extra_fields["anchor_obs"]` 里提供可哈希的 observation。本项目的 observation 包含检索文本，天然几乎不重复，收益有限，所以未启用。

---

## 9. Worker 与 Engine

### 9.1 FSDP worker

`agent_r1/workers/fsdp_workers.py`：

- `AsyncActorRolloutRefWorker`（:24）继承 verl 的实现，负责 Actor、vLLM rollout、Reference logprob；
- `CriticWorker`（:41）负责 value 前向/反向。

### 9.2 vLLM 与 FSDP 时分复用

`AgentFlowManager._initialize_llm_servers`：

- 如果传入 `worker_group`，调用 `server.init_hybrid(worker_group)`：vLLM 和 FSDP 共享同一组 GPU；
- rollout 阶段 vLLM 工作，训练阶段 vLLM sleep/free KV cache，FSDP 接管。

`gpu_memory_utilization` 控制 vLLM 预留的显存比例。太小 → KV cache 不够、rollout 慢；太大 → 训练 OOM。

### 9.3 Ray

- `RayAgentTrainer` 在 driver 上编排；
- 每个 role 是一个 Ray actor group；
- `ResourcePoolManager` 把 `global_pool` 映射到 4 张卡；
- worker 之间用 NCCL/PCIe 通信（A40 无 NVLink，PCIe 4.0 x16）。

---

## 10. 配置系统

`agent_r1/config/agent_ppo_trainer.yaml`：

```yaml
defaults:
  - ppo_trainer                       # verl 的完整 PPO 配置
  - /overrides/rollout@actor_rollout_ref.rollout
  - _self_
hydra:
  searchpath:
    - pkg://verl.trainer.config
```

这意味着：

- 所有 verl 的 PPO 配置键都可以用；
- `actor_rollout_ref.rollout` 被替换成 Agent-R1 的 `AgentFlowConfig`；
- `AgentFlowConfig` 字段（`config/config.py`）：`num_workers`, `default_agent_flow`, `agent_flow_config_path`, `max_steps`, `skip_special_tokens`。

常用 override：

```
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.rollout.n=8
actor_rollout_ref.rollout.agent.default_agent_flow=finqa_agent
actor_rollout_ref.rollout.agent.agent_flow_config_path=...
custom_reward_function.path=...
custom_reward_function.name=compute_score
data.return_raw_chat=True
trainer.total_epochs=2
```

---

## 11. 如何扩展（以本项目为例）

### 11.1 最小接入清单

1. **数据**：parquet 必须含 `prompt`、`data_source`、`reward_model`；可选 `agent_name`、`extra_info`、`env_kwargs`。
2. **Flow**：继承 `AgentFlowBase`，实现 `run`；每轮 `_postprocess`。
3. **注册**：`@register("finqa_agent")` + YAML `_target_`。
4. **奖励**：实现 `compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float`。
5. **工具**：继承 `BaseTool`，实现 `execute`。
6. **启动**：`actor_rollout_ref.rollout.agent.agent_flow_config_path` + `custom_reward_function.path`。

### 11.2 本项目的实现选择

- `core.py` 不 import verl/agent_r1，保证同一套逻辑可用于训练、vLLM 离线评测和单测；
- `flow.py` 只做适配：把 `Generation` 转 `AgentFlowStep`，把 stats 塞进 `extra_info`；
- 工具用模块级 corpus cache，避免每个样本重复加载语料；
- 训练脚本把 `upstream/Agent-R1` 和 `projects/finagent_r1` 都放进 `PYTHONPATH`，不修改 upstream。

---

## 12. 常见坑

| 现象 | 原因 | 修复 |
|---|---|---|
| `Agent flow xxx not registered` | YAML 没加载 / `_target_` 路径错 | 检查 `agent_flow_config_path` 和 `default_agent_flow` |
| reward 全是 0 | 终止步没触发 / reward fn 没被调用 | 检查 `reward_score=None`、`custom_reward_function.path` |
| loss 在工具观测上也下降 | `response_mask` 全 1 | 确保 `_postprocess` 里传了正确的 `response_mask` |
| 同一条轨迹被拆到不同 mini-batch | 没用 trajectory-aware batching | 检查 `trajectory_uids`/`step_indices`、`trainer.balance_batch` |
| prompt 超长 | 上下文无限 append | 在 flow 里做预算收缩；本项目 `core._render_prompt` |
| vLLM OOM | `gpu_memory_utilization` 太高 | 降到 0.35–0.45；PPO 更低 |
| FSDP OOM | micro-batch 太大 / 没开梯度检查点 | `ppo_micro_batch_size_per_gpu=2`、`enable_gradient_checkpointing=True` |
| 多卡 hang | NCCL/PCIe 拓扑、端口冲突 | `NCCL_DEBUG=INFO`、`NCCL_P2P_DISABLE=1` 排查 |
| `response_length` 截断 | 模型输出太长 | 降低 `max_response_length` 或加长度惩罚 |
| 训练 reward 升、val 降 | 过拟合/文档泄漏 | 报告 unseen_docs；加 KL/早停 |

---

## 13. 推荐阅读顺序

1. `README.md` → 理解 step-level MDP 的动机；
2. `agent_r1/agent_flow/agent_flow.py` → `AgentFlowStep`、`AgentFlowBase`、registry、worker；
3. `agent_r1/agent_flow/agent_env_loop.py` → 最小可运行循环；
4. `agent_r1/trainer/ppo/ray_trainer.py` → `fit()`、`compute_advantage`；
5. `agent_r1/trainer/ppo/core_algos.py` → 各 estimator 的数学实现；
6. `agent_r1/trainer/ppo/trajectory_batching.py` → 多轮 batching；
7. `agent_r1/workers/fsdp_workers.py` → FSDP + vLLM；
8. 本项目的 `core.py` → 一个不依赖框架的 episode loop；
9. 本项目的 `flow.py` → 如何把 loop 接进 Agent-R1。
