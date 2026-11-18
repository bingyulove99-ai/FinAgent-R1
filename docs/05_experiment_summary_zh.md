# FinAgent-R1 实验总结（2026-11-12 ~ 持续更新）

> 本文档由本次运行会话生成，记录仓库结构、环境、训练/评测设置、遇到的错误与修复、
> 调优决策、已完成的实验结果与当前进展。所有数值均来自 `results/*.metrics.json`
> 与 `logs/*.log` 的真实运行，无估计值。

---

## 1. 代码仓库结构（哪些是有效运行代码）

```
/home/user/Agent-R1-main/
├── upstream/Agent-R1/              # 上游 Agent-R1 框架（保持原样，不修改）
│   └── agent_r1/
│       ├── trainer/main_agent_ppo.py       # 训练入口（Hydra + Ray）
│       ├── trainer/ppo/ray_trainer.py      # 训练/验证循环（step-level PPO 扩展）
│       ├── trainer/ppo/core_algos.py       # GAE/GRPO/RLOO/REINFORCE/token-GAE 优势估计
│       ├── workers/{fsdp,engine}_workers.py# FSDP actor/critic + vLLM rollout worker
│       ├── agent_flow/                     # AgentFlow 抽象与环境循环
│       ├── reward_loop/                    # 奖励计算循环
│       ├── tool/、env/                     # 工具与环境的基类
│       └── recipes/、examples/             # 上游示例（本项目未使用）
│
├── projects/finagent_r1/           # ★ 本项目扩展层（FinQA agentic RL）
│   ├── finagent_r1/
│   │   ├── core.py                 # 多轮 episode 循环（max_steps/scratchpad/停止规则）
│   │   ├── flow.py                 # 注册到 Agent-R1 的 finqa_agent flow
│   │   ├── tools.py                # BM25 检索器（ticker/year 作用域）+ AST 白名单计算器
│   │   ├── rewards.py              # 复合奖励：结果匹配 × 效率 + 格式 - 工具惩罚
│   │   ├── parsing.py              # 工具调用/答案解析
│   │   ├── prompts.py              # 系统提示与工具 schema
│   │   ├── configs/flow.yaml       # flow 注册配置
│   │   ├── data/prepare_finqa.py   # FinQA → parquet/jsonl/corpus 数据管线
│   │   ├── data/dataset_stats.py   # 数据集审计
│   │   └── evaluation/
│   │       ├── infer_vllm.py       # vLLM 推理（agent/direct，分片）
│   │       └── evaluate_predictions.py  # 指标计算（acc/F1/格式/工具统计）
│   ├── scripts/
│   │   ├── common.sh               # 公共 Hydra 参数（lr/batch/kl/agent workers…）
│   │   ├── train_{grpo,rloo,reinforce,ppo,token_gae}.sh  # 五种算法训练入口
│   │   ├── merge_checkpoint.sh     # FSDP → HuggingFace 合并
│   │   ├── prepare_data.sh / setup_env.sh / smoke_test.sh
│   │   └── evaluate.sh             # 单模型评测入口（调用 evaluation/）
│   ├── tests/                      # 60 个 CPU 单元测试（全部通过）
│   └── requirements.txt
│
├── docs/                           # 项目文档（00 总览 / 01 框架 / 02 runbook / 03 学习 / 04 面试）
├── data/                           # data/finqa（train/dev/test parquet+jsonl、corpus.jsonl 64,810 chunks）
├── scripts/                        # 仅 README（工作区脚本实际在 /data/home.old/...，见下）
├── README.md                       # 工作区说明
└── Makefile

/home/user/Agent-R1-main/models -> /data/home.old/user/agent-r1/models   (Qwen2.5-3B-Instruct, 符号链接)
/home/user/Agent-R1-main/outputs -> /data/home.old/user/agent-r1/outputs (训练输出，符号链接)
```

### 运行用的工作区脚本（在 `/data/home.old/user/agent-r1/`，不在 git 仓库内）

| 文件 | 作用 |
|---|---|
| `env.sh` | 运行时环境：conda env、CUDA_VISIBLE_DEVICES=0,3、N_GPUS=2、HF 离线、路径 |
| `run_all_trainings_v2.sh` | ★ 主流水线：依次训练 5 算法，每算法独立目录、崩溃重试续训、合并 dev 最佳 checkpoint、立即全量 test 评测 |
| `run_all_trainings_v3.sh` | 同 v2，增加了 `RESUME=1` 显式续训开关（手动重启时用） |
| `eval_model.sh` | 加固版评测：清理残留 → 4 分片错峰启动 → 缺失/未完成分片串行补跑 → 三子集评分 |
| `finalize.sh` | 等待全部算法完成后：跑 base_direct/base_react 基线 → `aggregate_results.py` → 每算法 summary |
| `summarize_algo.py` | 生成 `results/<algo>_summary.md`（dev 轨迹、KL/熵、test 指标） |
| `parse_devacc.py` | 从训练日志解析每步 dev 分数并选最佳 step |
| `archive_checkpoints.sh` | 每 2 分钟把新 checkpoint 硬链接到 `outputs/_archive/ckpt/`，防止被轮转删除 |
| `watch_summaries.sh` | 每个算法评测完成即自动生成 summary |
| `fix_eval_grpo*.sh` | GRPO 评测并发失败时的一次性补跑脚本（历史） |
| `aggregate_results.py` | 汇总 7 个配置的结果表、bootstrap 95% CI、McNemar 检验 → `results/SUMMARY.md` |

### 产物目录

| 路径 | 内容 |
|---|---|
| `outputs/finqa_qwen3b_<algo>/global_step_N/` | FSDP checkpoint（actor[+critic]），每 25 步，全量保留 |
| `outputs/finqa_qwen3b_<algo>/global_step_N/huggingface/` | 合并后的 HF 模型（仅 dev 最佳 step） |
| `outputs/finqa_qwen3b_<algo>/selection.txt` | checkpoint 选择规则与 dev 分数 |
| `outputs/_archive/ckpt/` | 硬链接归档（防止 checkpoint 轮转丢失） |
| `results/<algo>_test.jsonl` | 全量 test（1,147 题）预测 |
| `results/<algo>_{all,seen_docs,unseen_docs}.metrics.json` | 指标 |
| `results/<algo>_summary.md`、`results/PROGRESS.md` | 每算法/总体进度摘要 |
| `logs/train_<algo>.log`、`logs/devacc_<algo>.tsv` | 训练日志与 dev 轨迹 |

---

## 2. 机器与环境

- GPU：4×A40 48GB，其中 **GPU 1/2 被其他用户占用**，本项目使用 **GPU 0 + GPU 3**（2 卡）
- CPU 144 核，内存 251GB，`/data` 剩余约 5.1TB
- Python 3.10 conda env：`/data/home.old/user/agent-r1/envs/agentr1`
  （PyTorch 2.8、verl 0.7.0、vLLM V1、Ray、flash-attn 2.8.3）
- 模型：`Qwen2.5-3B-Instruct`（本地离线）
- 数据：FinQA train 6,251 / dev 883 / test 1,147 题；检索语料 64,810 chunks
- 任务：Agent-R1 step-level MDP，最多 6 步；策略可调用 `financial_retriever`（ticker/year 作用域 BM25）
  与 `python_calc`（AST 白名单）；复合奖励 = 数值匹配 × 效率 + 格式 - 工具惩罚

---

## 3. 训练与评测设置（五算法统一）

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-3B-Instruct |
| epoch / updates | 1 epoch = 195 updates（batch 32） |
| rollout n | 8（GRPO/RLOO/REINFORCE）；4（PPO/Token-GAE，文档约定） |
| lr | **5e-7**（文档默认 1e-6；见 §4/§5） |
| warmup | 5%（约 10 步） |
| KL | `use_kl_loss=True`, `low_var_kl`, **coef=0.03**（默认 0.001） |
| entropy_coeff | 0.001 |
| GRPO 优势 | **Dr.GRPO**（`norm_adv_by_std_in_grpo=False`） |
| PPO/Token-GAE | **critic_warmup=10** + critic/actor optimizer offload + dynamic bsz |
| 保存/验证 | 每 25 步保存与 dev 验证（883 题，贪心） |
| checkpoint 保留 | `max_*_ckpt_to_keep=10`（全部 8 个保存点）+ 硬链接归档 |
| 最终模型选择 | **dev 分数最高的 checkpoint**（文档 8.5 要求报告选择规则） |
| 评测 | 全量 test 1,147 题；`infer_vllm` 贪心、同一 flow/工具；指标 all/seen/unseen |

---

## 4. 遇到的错误、困难与修复（按时间顺序）

1. **旧会话运行静默续训了漂移 checkpoint**
   - 现象：旧 `run_all_trainings.sh` 使用固定 EXP_NAME，verl `resume_mode=auto` 自动从
     `global_step_25`（已漂移）继续，lr 修正被优化器状态覆盖，KL 3.2→6.4 持续恶化。
   - 修复：v2 每个算法使用独立输出目录；启动前归档陈旧目录；重试仅在同一算法目录内续训；
     v3 增加 `RESUME=1` 显式续训开关；checkpoint 全量保留 + 硬链接归档。

2. **KL 爆炸 / reward hacking（旧配置）**
   - 现象：run1/run2 在 ~step 15 后 KL 从 0.001→4.8→7.3，train reward 0.24→0.05，
     策略退化为不调用工具直接作答。
   - 修复：lr 1e-6→5e-7、kl_coef 0.001→0.03、GRPO 用 Dr.GRPO。GRPO 新运行 KL 最高 0.50，
     dev 从 0.0652 单调升至 0.2864。

3. **PPO 随机初始化 critic 破坏首个 actor 更新**
   - 现象：quicktest 中 critic vpred≈3、grad_norm≈4734、vf_loss≈50，1 步后 dev 从 0.070 跌至 -0.024。
   - 修复：`trainer.critic_warmup=10`（前 10 步只训 critic）。正式 PPO 中 vpred 收敛到 0.07。

4. **评测并发 vLLM 显存竞争**
   - 现象：训练刚结束时 4 个评测分片中 3 个报
     `No available memory for the cache blocks`（同卡两个 vLLM 进程竞争 KV cache）。
   - 修复：`eval_model.sh` 加固——评测前 `ray stop`+清理 EngineCore+等待 20s；分片错峰启动；
     对失败/未完成分片逐个串行补跑（0.45 利用率）。

5. **`eval_model.sh` 末尾汇总 `s: unbound variable`**
   - 现象：4 个分片与评分全部成功，但结尾打印汇总时 heredoc 中 `${s}` 被 shell 展开，
     `set -u` 报错退出 → 流水线误报“评测失败”。
   - 修复：heredoc 改引号形式，变量经环境变量传入；并空子集（seen_docs=0）跳过。

6. **`summarize_algo.py` 空子集 KeyError**
   - 现象：`seen_docs` 为空（数据按文档级划分，test 文档全部未见），summary 只有 count=0，
     无 `accuracy` 键 → KeyError。
   - 修复：空子集输出 "n/a (empty split)"。

7. **RLOO/REINFORCE 后期漂移（算法特性）**
   - RLOO：KL 最高 9.1、熵最低 0.0007、dev 0.2515(100)→0.0685(175)。
   - REINFORCE：KL 最高 11.6、熵中位 0.006、dev 0.1827(25)→0.0006(125)，响应长度暴涨至触顶。
   - 处置：全 checkpoint 保留 + dev 最佳选择（RLOO→step100，REINFORCE→step25）；
     在最终报告中如实报告漂移现象。

8. **PPO 训练期 dev 指标异常**
   - 现象：同一 base 模型，GRPO/RLOO/REINFORCE 初始 dev 为 +0.056~0.065，PPO 为 **-0.0254**；
     PPO 训练侧熵 1.0（其余 ~0.013）。疑似 PPO（含 critic）验证/rollout 路径的系统性差异。
   - 处置：保留原配置完成实验；checkpoint 选择仍用训练期 dev 排序；最终以统一 `infer_vllm`
     test 评测为权威结果，并在报告中注明该差异。

9. **checkpoint 轮转风险**
   - 旧配置 `max_actor_ckpt_to_keep=4` 会删除早期（可能最优）checkpoint。
   - 修复：v2 设为 10（保留全部 8 个保存点）+ `archive_checkpoints.sh` 硬链接归档。

10. **会话多次长时间挂起 / 无人在场**
    - 处置：所有训练/评测/汇总进程 `setsid nohup` 独立运行；v2/finalize/watch_summaries
      全程自动化；done/eval_done 标记保证重跑不重复训练；实测跨 9 小时挂起训练不受影响。

---

## 5. 调优决策记录

| 参数/机制 | 项目默认 | 采用值 | 原因 |
|---|---|---|---|
| Actor lr | 1e-6 | **5e-7** | 抑制 KL 爆炸（旧运行 25 步 KL 2.7） |
| kl_loss_coef | 0.001 | **0.03** | 更强偏离锚定 |
| GRPO 优势归一化 | std 归一 | **Dr.GRPO(False)** | 文档 5.7 稳定性回退 |
| critic_warmup | 0 | **10** | 避免随机 critic 破坏 actor |
| max ckpt 保留 | 3/4 | **10 + 硬链接归档** | 支持 best-dev 回滚 |
| save/test freq | 50/25 | **25/25** | 更细粒度验证/回滚 |
| 最终 checkpoint | 最新 | **dev 最佳**（报告规则） | 漂移下保证结果可用 |
| 评测并行 | 2 进程/卡 | **错峰 + 失败串行补跑** | 规避 KV cache 显存竞争 |
| 算法隔离 | 无 | **独立目录 + 预清理 + 进程清理** | 避免算法间交叉污染 |

---

## 6. 实验结果（截至 2026-11-16 10:20）

> dev = 883 题开发集复合奖励（贪心）；test = 1,147 题数值准确率（同一推理脚本）；
> 单 seed（42）；按 dev 最高 checkpoint 评测并报告选择规则（文档 8.5）。

| 算法 | 状态 | 训练时长 | 最佳 dev | test Acc | test score | answer rate | avg steps | invalid | tool err | recall |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO | ✅ 完成 | 27h26m | **0.2864** (step195) | **0.3078** | 0.3118 | 0.7742 | 3.74 | 0.1089 | 0.0046 | 0.2372 |
| RLOO | ✅ 完成 | ~29h44m | **0.2515** (step100) | **0.2677** | 0.2832 | 0.8649 | 3.16 | 0.0944 | 0.0124 | 0.2336 |
| REINFORCE | ✅ 完成 | 17h28m | **0.1827** (step25) | **0.2110** | 0.2264 | 0.8893 | 3.66 | 0.1284 | 0.0161 | 0.2080 |
| PPO | ✅ 完成 | 27h09m | **0.1009** (step150) | **0.1997** | 0.1832 | 0.6600 | 4.19 | 0.3690 | 0.0031 | 0.2409 |
| Token-GAE | ⚠️ 提前停止 | ~13h | **-0.0142** (step50) | **0.1055** | 0.0344 | 0.7140 | 3.08 | 2.2479 | 0.0175 | 0.1597 |
| base_direct（无工具直答） | ✅ | — | — | **0.0052** | 0.0552 | 1.0000 | 1.00 | 0.0000 | 0.0000 | n/a |
| base_react（零样本 ReAct） | ✅ | — | — | **0.0724** | 0.0435 | 0.3575 | 4.70 | 0.4214 | 0.0050 | 0.2213 |

**dev 轨迹（每 25 步）**

- GRPO：0.0652 → 0.1780 → 0.1952 → 0.2337 → 0.2557 → 0.2448 → 0.2730 → 0.2717 → **0.2864**
- RLOO：0.0568 → 0.1871 → 0.2058 → 0.2507 → **0.2515** → 0.2211 → 0.1623 → 0.0685 → 0.0749
- REINFORCE：0.0557 → **0.1827** → 0.1206 → 0.1312 → -0.0071 → 0.0006 → 0.0028 → 0.0228
- PPO：-0.0254 → -0.0290 → -0.0080 → +0.0640 → +0.0991 → +0.0664 → **+0.1009(150)** → -0.1136(175，长度崩溃) → -0.0040(195)

**主要观察**

1. GRPO 在本任务上最稳定且效果最好（KL≤0.50，dev 持续上升至 0.2864）。
2. RLOO/REINFORCE 在 ~75–100 步后出现明显漂移（KL 9–12，熵趋 0，策略退化为少用/不用工具），
   说明 3B 模型 + 步级 MDP 下 on-policy 方差较大；best-dev checkpoint 选择是必要保护。
3. 所有算法均使用相同超参；结果可比性以统一 test 推理为准。
4. 效率指标：训练后 answer rate 提升至 77%–89%，工具错误率 ≤1.6%。

---

## 7. 当前进行情况与后续计划

- **PPO 已完成**（195/195，27h09m）：按 dev 最佳选 step 150 评测，test acc **19.97%**；
  后期出现长度刷分崩溃（响应长度 184→455），best-dev 选择规避。
- **Token-GAE**：2026-11-17 07:34 启动。训练到 step 75 附近时出现 (a) 策略退化（响应长度
  428→757/768、KL 0.5→5.4）与 (b) critic 更新 CUDA OOM（GPU0 44.4GB 容量，vLLM 预留 13.8GB
  + 长响应激活）。v2 已自动重试并从 step 50 续训；为避免重复 OOM 与无效计算，手动停止并按
  dev 最佳 checkpoint（step 50，dev=-0.0142）合并评测，test acc **10.55%**，无效调用率 2.25。
  checkpoint 25/50 均保留，可用 `RESUME=1` 续训；此偏差在最终报告中如实说明。
  - 为严格完成 1 epoch，随后以**减半 micro-batch/token 预算**（actor/critic 4096 tokens/GPU、
    micro-batch 1、vLLM 0.30）从 step 50 续训两次；两次均在 resume 后首个
    `critic_update_critic` 处同样 OOM（错误与首次相同），确认在当前 2×A40 显存分配
    （另 2 卡被其他用户占用）下无法完成该退化策略的完整 epoch。已停止重试，保留 step 50 的
    评测结果与全部 checkpoint。
- **评测与汇总全部完成**：base_direct 0.52%，base_react 7.24%；五算法 GRPO 30.78% > RLOO 26.77%
  > REINFORCE 21.10% > PPO 19.97% > Token-GAE 10.55%。结果见 `results/SUMMARY.md`。
- **收尾（finalize.sh 自动执行）**：base_direct / base_react 基线评测 → `aggregate_results.py`
  生成 `results/SUMMARY.md`（含 bootstrap 95% CI 与 McNemar 检验）→ 每算法 summary。
- **最终报告**：`results/SUMMARY.md` + 本文档更新版（含 5 算法完整结果与结论）。

---

## 8. 复现命令（核心）

```bash
# 环境
source /data/home.old/user/agent-r1/env.sh
cd /home/user/Agent-R1-main

# 数据（已生成，可重复）
bash projects/finagent_r1/scripts/prepare_data.sh

# CPU 单测
cd projects/finagent_r1 && python -m unittest discover -s tests -q   # 0 失败

# 单算法训练（示例：GRPO）
EXP_NAME=finqa_qwen3b_grpo EPOCHS=1 TRAIN_BATCH_SIZE=32 \
  ACTOR_LR=5e-7 bash projects/finagent_r1/scripts/train_grpo.sh \
  actor_rollout_ref.actor.kl_loss_coef=0.03 \
  algorithm.norm_adv_by_std_in_grpo=False

# 全流水线（5 算法 + 自动评测）
bash /data/home.old/user/agent-r1/run_all_trainings_v2.sh

# 单独评测与汇总
bash /data/home.old/user/agent-r1/eval_model.sh grpo \
  outputs/finqa_qwen3b_grpo/global_step_195/huggingface agent
python /data/home.old/user/agent-r1/aggregate_results.py
```

---

## 9. 最终统计结论（2026-11-18 04:05）

配对 McNemar 检验（同一 1,147 题，单 seed）：

| 对比 | 仅基线正确 | 仅模型正确 | p 值 |
|---|---:|---:|---:|
| GRPO vs 零样本 ReAct | 292 | 22 | 2.3e-61 |
| RLOO vs 零样本 ReAct | 250 | 26 | 3.9e-47 |
| REINFORCE vs 零样本 ReAct | 193 | 34 | 3.6e-28 |
| PPO/step-GAE vs 零样本 ReAct | 184 | 38 | 3.6e-24 |
| Token-GAE vs 零样本 ReAct | 96 | 58 | 0.0027 |
| RLOO vs GRPO | 57 | 103 | 0.00034 |
| REINFORCE vs GRPO | 49 | 160 | 6.1e-15 |
| PPO/step-GAE vs GRPO | 83 | 207 | 2.2e-13 |
| Token-GAE vs GRPO | 42 | 274 | 7.0e-43 |

**结论**：五种 RL 算法均显著优于零样本 ReAct 基线（p<0.05，其中四种 p<1e-20）；
算法间排序为 GRPO > RLOO > REINFORCE > PPO > Token-GAE（两两 McNemar 均显著）。
GRPO（Dr.GRPO + lr 5e-7 + kl 0.03）在本任务上训练最稳定（KL≤0.50）且效果最好；
RLOO/REINFORCE/PPO/Token-GAE 均出现不同程度的中后期漂移（KL 爆炸/长度刷分），
按 dev 最佳 checkpoint 选择后仍显著优于基线。详细表格与置信区间见 `results/SUMMARY.md`。
