# FinAgent-R1 Runbook（4×A40 48GB）

> 目标机器：单节点 4× NVIDIA A40 48GB，Ubuntu 22.04/24.04，Python 3.10+。
> 本文假设工作区位于 `$WORKSPACE`（例如 `/data/Agent-R1-main`），所有命令都在 `$WORKSPACE` 下执行。

---

## 0. 目录约定

```bash
export WORKSPACE=/path/to/Agent-R1-main
cd "$WORKSPACE"
```

```
upstream/Agent-R1/        # 原框架，只读
projects/finagent_r1/     # 本项目
data/finqa/               # 生成数据
outputs/                  # 训练输出
results/                  # 评测输出
```

---

## 1. 环境安装

### 1.1 Python

```bash
conda create -n agentr1 python=3.10 -y
conda activate agentr1
python --version    # 必须 >= 3.10
```

### 1.2 PyTorch（A40 = Ampere sm_86）

按机器驱动选择 CUDA wheel。示例（CUDA 12.4）：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A40 **不支持 FP8**，请使用 bf16；vLLM 的 `dtype` 保持 `bfloat16`，不要传 `float8`。

### 1.3 verl 0.7.0

Agent-R1 当前要求 `verl==0.7.0`。请按官方指南安装（不同 verl 小版本的 vLLM/Ray 依赖不同）：

```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout v0.7.0        # tag 名以仓库为准；也可能是 0.7.0
pip install -e .
cd ..
```

如果官方指南要求额外安装 vLLM / Ray / FlashAttention，请以 verl 0.7.0 的 `requirements` 为准。安装完成后验证：

```bash
python - <<'PY'
import torch, verl, vllm, ray, transformers
print("torch", torch.__version__)
print("verl", getattr(verl, "__version__", "?"))
print("vllm", vllm.__version__)
print("ray", ray.__version__)
print("transformers", transformers.__version__)
PY
```

### 1.4 本项目依赖

```bash
pip install -r projects/finagent_r1/requirements.txt
```

`pyarrow` 是写 parquet 的硬依赖。没有它，数据脚本只能写 JSONL，Agent-R1 训练会找不到 `train.parquet`。

### 1.5 环境自检

```bash
bash projects/finagent_r1/scripts/setup_env.sh
```

它会打印 Python 版本、`nvidia-smi`、关键包版本，并给出缺失依赖的安装顺序。

### 1.6 预下载模型（可选，但推荐）

训练/评测时每个 Ray worker 都会加载 tokenizer，第一次运行会从 Hugging Face 下载模型。网络不稳时建议提前下载到本地：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir models/Qwen2.5-3B-Instruct
```

然后：

```bash
FIN_MODEL_PATH=$PWD/models/Qwen2.5-3B-Instruct \
bash projects/finagent_r1/scripts/train_grpo.sh
```

评测时 `MODEL` 也指向同一个本地目录。

---

## 2. 数据准备

### 2.1 自动下载

```bash
bash projects/finagent_r1/scripts/prepare_data.sh
```

行为：

1. 如果 `data/finqa/raw/{train,dev,test}.json` 已存在，直接用；
2. 否则尝试 Hugging Face：`dreamerdeo/finqa`、`ChanceFocus/finqa`；
3. `--source github` 时尝试官方 GitHub raw JSON；
4. 生成 `train/dev/test.parquet`、`corpus.jsonl`、`DATASET_CARD.md`。

国内网络建议：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash projects/finagent_r1/scripts/prepare_data.sh
```

### 2.2 手动下载（网络受限时）

1. 从 FinQA 官方仓库下载 `dataset/train.json`、`dataset/dev.json`、`dataset/test.json`；
2. 放到 `data/finqa/raw/`；
3. 运行：

```bash
export PYTHONPATH="$PWD/projects/finagent_r1"
python -m finagent_r1.data.prepare_finqa \
  --raw_dir data/finqa/raw \
  --output_dir data/finqa
python -m finagent_r1.data.dataset_stats --data_dir data/finqa
```

### 2.3 验收

打开 `data/finqa/DATASET_CARD.md`，确认：

- train/dev/test 行数分别为 6,251 / 883 / 1,147（或你显式子采样后的数量）；
- corpus 所有 chunk 都有 ticker/year；
- test 中既有 `doc_seen_in_train=True` 也有 `False`；
- 随机抽 10 题看 question/answer/scope 是否匹配。

### 2.4 子采样（快速试跑）

```bash
python -m finagent_r1.data.prepare_finqa \
  --raw_dir data/finqa/raw \
  --output_dir data/finqa_small \
  --max_train 400 --max_dev 100 --max_test 100
```

注意：corpus 仍然从**全部**原始文档构建，避免子采样后检索不到证据。

---

## 3. CPU 冒烟测试

无需 GPU：

```bash
bash projects/finagent_r1/scripts/smoke_test.sh
```

它会：

1. 跑 60 个单测（解析、奖励、BM25、AST、episode loop、数据转换）；
2. 用 `tests/fixtures/finqa_raw/` 生成一个 tiny dataset；
3. 跑数据审计；
4. 用 gold answer 跑一遍 reward，确认训练奖励链路。

预期最后输出 `Smoke test passed.`。

---

## 4. 训练

### 4.1 第一轮：GRPO，1 epoch，小 batch

```bash
cd "$WORKSPACE"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
FIN_MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
EPOCHS=1 TRAIN_BATCH_SIZE=16 \
bash projects/finagent_r1/scripts/train_grpo.sh
```

先观察 50–100 个 update：

- reward 是否上升；
- invalid tool call 是否下降；
- 平均步数是否从 5–6 降到 3–4；
- `actor/entropy` 是否缓慢下降而不是瞬间归零；
- 验证集 accuracy 是否上升。

### 4.2 五种算法

```bash
# GRPO
bash projects/finagent_r1/scripts/train_grpo.sh

# RLOO
bash projects/finagent_r1/scripts/train_rloo.sh

# REINFORCE
bash projects/finagent_r1/scripts/train_reinforce.sh

# PPO / step-level GAE（也是 StepPO-style）
# 注意：PPO 需要同时放 actor + critic + ref + vLLM，脚本默认 actor micro-batch=2
bash projects/finagent_r1/scripts/train_ppo.sh

# Token-GAE（消融）
bash projects/finagent_r1/scripts/train_token_gae.sh
```

常用环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `FIN_MODEL_PATH` | `Qwen/Qwen2.5-3B-Instruct` | 模型或本地路径 |
| `FINQA_TRAIN` / `FINQA_VAL` | `data/finqa/{train,dev}.parquet` | 数据 |
| `FINQA_CORPUS_PATH` | `data/finqa/corpus.jsonl` | 检索语料 |
| `EPOCHS` | 2 | 训练轮次 |
| `TRAIN_BATCH_SIZE` | 32 | prompts/update |
| `ROLLOUT_N` | 8（GRPO/RLOO/REINFORCE）/4（PPO） | 每条 prompt 的轨迹数 |
| `MINI_BATCH_SIZE` | 16 | PPO mini-batch |
| `MICRO_BATCH_SIZE` | 4 | 单卡 micro-batch |
| `GPU_MEMORY_UTILIZATION` | 0.45（无 critic）/0.35（有 critic） | vLLM 显存比例 |
| `SAVE_FREQ` / `TEST_FREQ` | 50 / 25 | checkpoint / 验证频率 |
| `OUTPUT_ROOT` | `outputs` | 输出根目录 |
| `LOGGER` | `["console"]` | 可设 `["console","wandb"]` |

### 4.3 显存调优顺序

如果 OOM，按以下顺序降：

1. `MICRO_BATCH_SIZE=2`（actor/critic micro-batch）；
2. `GPU_MEMORY_UTILIZATION=0.30`；
3. `TRAIN_BATCH_SIZE=16`；
4. `MINI_BATCH_SIZE=8`；
5. `ROLLOUT_N=4`；
6. PPO 再考虑 `critic.model.fsdp_config.param_offload=True`；
7. 最后才是 `actor.fsdp_config.optimizer_offload=True`（会明显变慢）。

### 4.4 7B 配置（可选，不推荐默认）

```bash
FIN_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
EPOCHS=1 TRAIN_BATCH_SIZE=8 MICRO_BATCH_SIZE=2 \
GPU_MEMORY_UTILIZATION=0.30 \
bash projects/finagent_r1/scripts/train_grpo.sh
```

如果仍 OOM，需要：
- `actor_rollout_ref.actor.fsdp_config.param_offload=True`
- 或改用 LoRA（需要改 verl 配置，本项目未提供脚本）。

PPO + 7B + 4×48GB 不建议；Actor+Critic+Ref+vLLM 基本放不下。

### 4.5 输出与 checkpoint

```
outputs/<experiment>/
├── global_step_50/
│   └── actor/            # FSDP shards + optimizer
├── global_step_100/
├── ...
└── latest_checkpointed_iteration.txt
```

每个 3B 全量 checkpoint（model+optimizer）约 **25–35 GB**。如果磁盘有限：

- 把 `SAVE_FREQ` 调大（如 100）；
- 训练结束后手动删除中间 checkpoint；
- 只保留 dev 指标最好的 1–2 个。

### 4.6 断点续训

verl 会在 `trainer.default_local_dir` 下记录 `latest_checkpointed_iteration.txt`。重新执行同一脚本时，verl 默认会从最新 checkpoint 恢复（具体行为以 verl 0.7.0 为准）。如果想从头开始，删除或改名 `outputs/<experiment>`。

---

## 5. 合并 checkpoint

verl 默认保存的是 FSDP 分片，不是 Hugging Face 目录。合并：

```bash
bash projects/finagent_r1/scripts/merge_checkpoint.sh \
  outputs/finqa_qwen3b_grpo/global_step_100/actor
```

默认输出到 `outputs/finqa_qwen3b_grpo/global_step_100/huggingface`。

如果提示 `No module named verl.model_merger`，检查：
- verl 是否安装在当前 conda 环境；
- verl 版本是否 0.7.0；
- 某些 verl 版本用 `python -m verl.model_merger merge --backend fsdp ...`，参数以 `python -m verl.model_merger --help` 为准。

---

## 6. 评测

### 6.1 训练后 agent

```bash
MODEL=outputs/finqa_qwen3b_grpo/global_step_100/huggingface \
TEST_FILE=data/finqa/test.parquet \
SUBSET=all \
bash projects/finagent_r1/scripts/evaluate.sh
```

输出：

- `results/<model>_test.jsonl`：逐题预测、步数、工具统计；
- `results/<model>_test.metrics.json`：汇总；
- `results/<model>_test.md`：Markdown 表。

### 6.2 Base model 对照

```bash
MODEL=Qwen/Qwen2.5-3B-Instruct SUBSET=all \
bash projects/finagent_r1/scripts/evaluate.sh
```

### 6.3 Direct answer 对照（无工具）

```bash
MODEL=Qwen/Qwen2.5-3B-Instruct \
bash projects/finagent_r1/scripts/evaluate.sh --mode direct
```

### 6.4 文档级泛化子集

```bash
MODEL=... SUBSET=unseen_docs bash projects/finagent_r1/scripts/evaluate.sh
MODEL=... SUBSET=seen_docs   bash projects/finagent_r1/scripts/evaluate.sh
```

### 6.5 多卡并行评测

vLLM 默认用 1 卡（`--tensor_parallel_size 1`）。可以用 4 个进程分片：

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i \
  OUTPUT=results/grpo_shard$i.jsonl \
  bash projects/finagent_r1/scripts/evaluate.sh \
    --num_shards 4 --shard_index $i &
done
wait
cat results/grpo_shard*.jsonl > results/grpo_test.jsonl
python -m finagent_r1.evaluation.evaluate_predictions \
  --predictions results/grpo_test.jsonl --by_ticker
```

注意：每个 shard 都会加载一份模型和 corpus，需要足够 CPU 内存。

### 6.6 评测耗时

1,147 题 × 平均 3–5 步 ≈ 4,000–6,000 次 vLLM 调用。单卡 3B 大约 **1–3 小时**，取决于 `max_tokens`、平均步数和 prompt 长度。先用 `--limit 50` 估算。

---

## 7. 常见问题

### 7.1 数据与路径

**Q: `train.parquet` 不存在？**
- 没装 `pyarrow`；`pip install pyarrow` 后重跑 `prepare_data.sh`。
- 或用了 `--no_parquet`，只生成了 JSONL；训练需要 parquet。

**Q: `Agent flow finqa_agent not registered`？**
- `PYTHONPATH` 没包含 `projects/finagent_r1`；
- `agent_flow_config_path` 路径错；
- YAML 里 `_target_` 写错。

**Q: retriever 一直返回 `no_corpus`？**
- `FINQA_CORPUS_PATH` 没设或文件不存在；
- 每个 Ray worker 进程都要能读到该路径（共享文件系统，不要用相对路径）。

**Q: retriever 返回 `empty_scope`？**
- 数据里的 `ticker/year` 和 corpus 不一致；
- 检查 `parse_scope` 和 `DATASET_CARD.md`；
- 临时把 `allow_scope_fallback: true`（默认）打开，会 fallback 到全语料。

### 7.2 训练

**Q: vLLM OOM**
```bash
GPU_MEMORY_UTILIZATION=0.30 bash ...
```
PPO 建议 0.30–0.35；GRPO 可 0.40–0.45。

**Q: FSDP OOM**
- `MICRO_BATCH_SIZE=2`；
- 确认 `actor_rollout_ref.model.enable_gradient_checkpointing=True`（脚本默认）；
- PPO 的 critic 也要开 `enable_gradient_checkpointing=True`。

**Q: NCCL timeout / hang**
```bash
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1   # PCIe 机器排查用
export NCCL_IB_DISABLE=1    # 无 InfiniBand 时
```
A40 无 NVLink，4 卡走 PCIe；确认 `nvidia-smi topo -m` 正常。

**Q: reward 一直是 0**
- 模型从不输出 `<answer>`：先跑 base model zero-shot，看格式率；
- 工具全报错：检查 corpus；
- `custom_reward_function.path` 没传或函数名不是 `compute_score`；
- 终止步没触发：`max_steps` 到了但最后一步只有 tool call，reward 看到的是 tool call 文本，没有 answer。

**Q: 训练很慢**
- 平均步数接近 6：模型没学会停；先加格式奖励/减少 `max_steps`；
- prompt 太长：检查 `max_obs_chars` 和 scratchpad 是否无限增长；
- `gpu_memory_utilization` 太低：KV cache 不足，rollout 排队；
- Ray worker 数太多：`AGENT_WORKERS=4`（每个 worker 都会加载 corpus 索引）。

**Q: 训练 loss NaN**
- 降 lr 到 5e-7；
- 检查 reward 是否出现 NaN（`exe_ans` 可能为 NaN）；
- `entropy_coeff=0` 试试；
- 确认 bf16 而不是 fp16（A40 bf16 更稳）。

### 7.3 评测

**Q: vLLM 加载 merged checkpoint 失败**
- 确认目录里有 `config.json`、`tokenizer*`、`*.safetensors`；
- 合并命令的 `--backend fsdp` 是否正确；
- 某些 verl 版本需要 `--target_dir` 已存在或为空。

**Q: 评测结果和训练日志里的 reward 不一致**
- 训练日志的 reward 是 rollout 时的 old policy；评测是最终 checkpoint；
- 评测的 step/efficiency 可能与训练不同（temperature=0 vs 训练采样）；
- 用 `evaluate_predictions.py` 的 composite reward 对齐训练口径。

**Q: 结果虚高**
- 检查是否在 `test` 上训练过；
- 分别报告 `seen_docs` 和 `unseen_docs`；
- 确认 `doc_seen_in_train` 字段正确。

### 7.4 版本

**Q: Python 3.8 能跑吗？**
- 本项目的 CPU 单测可以；但 Agent-R1/verl 0.7 需要 Python 3.10+（上游代码使用了 `X | Y` 类型语法）。

**Q: 一定要 `verl==0.7.0` 吗？**
- 当前 upstream README 明确要求 0.7.0；其他版本可能 API 不兼容（尤其是 `AgentFlowConfig`、`AsyncLLMServerManager`、`DataProto`）。

**Q: A40 能用 FlashAttention 吗？**
- 可以（Ampere），但安装 wheel 要和 CUDA/torch 匹配；装不上时 vLLM 会回退到其他 attention backend，不影响正确性。

---

## 8. 资源规划

| 项目 | 估算 |
|---|---|
| 模型权重（3B bf16） | ~6.2 GB |
| 每个全量 checkpoint | ~25–35 GB |
| 数据集（parquet + corpus） | ~0.2–0.5 GB |
| 1 epoch 训练 | 约 12–36 h（4×A40，取决于步数/吞吐） |
| 2 epochs | 约 1–3 天 |
| 单卡评测 1,147 题 | 约 1–3 h |
| 5 种算法 × 2 epochs | 约 1–2 周（串行）；建议先各跑 1 epoch |

**建议**：先 GRPO 1 epoch → 评测 → 再决定是否跑其他算法。不要一上来并行跑 5 个实验，4 张卡会被互相抢显存。

---

## 9. 最小可复现实验清单

1. `setup_env.sh` 通过；
2. `prepare_data.sh` 生成 parquet + corpus；
3. `smoke_test.sh` 通过；
4. GRPO `EPOCHS=1 TRAIN_BATCH_SIZE=16` 跑 50 update，reward/格式率有变化；
5. 合并 checkpoint；
6. 评测 base / GRPO / direct 三个配置；
7. 把 `results/*.metrics.json` 填进 `docs/00_project_overview.md` 的结果表；
8. 再决定是否跑 RLOO/REINFORCE/PPO/Token-GAE 和消融。
