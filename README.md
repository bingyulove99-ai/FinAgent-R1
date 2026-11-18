# FinAgent-R1

**Step-level agentic reinforcement learning for financial question answering.**

FinAgent-R1 trains [Qwen2.5-3B-Instruct](https://github.com/QwenLM/Qwen2.5) to answer multi-step numerical questions over SEC filings from [FinQA](https://github.com/cszyssrs/FinQA), using the step-level MDP framework of [Agent-R1](https://github.com/AgentR1/Agent-R1). The agent calls two tools — a ticker/year-scoped BM25 retriever and an AST-whitelisted calculator — and is optimized end-to-end with **GRPO**, **RLOO**, **REINFORCE**, **PPO/step-GAE**, or **Token-GAE**.

Unlike a single-turn RAG or prompt-based ReAct setup, every tool interaction is a decision step: the environment reshapes the context (evidence scratchpad, sliding observation window, prompt budget), and the policy is rewarded only on the final answer.

## Results

Test split: **1,147 FinQA questions**, single seed 42. For each algorithm the checkpoint is selected by dev accuracy.

| Config | Numeric acc | 95% CI | Answer rate | Avg steps | Invalid calls / traj |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-3B direct (no tools) | 0.52% | [0.17, 0.96] | 100.00% | 1.00 | — |
| Qwen2.5-3B zero-shot ReAct | 7.24% | [5.84, 8.72] | 35.75% | 4.70 | 0.42 |
| **Qwen2.5-3B + GRPO** | **30.78%** | [28.07, 33.39] | 77.42% | 3.74 | 0.11 |
| Qwen2.5-3B + RLOO | 26.77% | [24.15, 29.29] | 86.49% | 3.16 | 0.09 |
| Qwen2.5-3B + REINFORCE | 21.10% | [18.66, 23.63] | 88.93% | 3.66 | 0.13 |
| Qwen2.5-3B + PPO/step-GAE | 19.97% | [17.61, 22.23] | 66.00% | 4.19 | 0.37 |
| Qwen2.5-3B + Token-GAE | 10.55% | [8.89, 12.47] | 71.40% | 3.08 | 2.25 |

- GRPO improves numeric accuracy **4.25×** over zero-shot ReAct and **~59×** over the no-tool baseline.
- All five RL algorithms beat zero-shot ReAct (paired McNemar, p < 0.05; four with p < 1e-20).
- GRPO significantly outperforms RLOO, REINFORCE, PPO, and Token-GAE (p ≤ 3.4e-4).

Full tables (composite reward, F1, invalid-call and tool-error rates, retrieval recall), bootstrap CIs, and all McNemar tests are in [`results/SUMMARY.md`](results/SUMMARY.md) and [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md). Training dynamics and failure analysis are in [`docs/05_experiment_summary_zh.md`](docs/05_experiment_summary_zh.md) and [`docs/ISSUES_AND_FIXES.md`](docs/ISSUES_AND_FIXES.md).

## Method

**Step-level MDP** (horizon `H = 6`):

| Element | Definition |
|---|---|
| State | System prompt + question + filing scope (`ticker/year`) + evidence scratchpad + sliding window of the last 2 tool observations |
| Action | `<analysis>` plus 1–2 `<tool_call>`s, or a final `<answer>` |
| Transition | Parse action → execute tool → update scratchpad and observations → rebuild next prompt |
| Reward | Intermediate steps 0; terminal step = outcome (numeric match / token F1) + format bonus − tool penalties, with a step-efficiency discount, clipped to `[-0.2, 1.2]` |
| Advantage | GRPO / RLOO / REINFORCE / step-level GAE / token-level GAE (selectable per run) |

**Tools**

- `financial_retriever` — BM25 over filing chunks, scoped to the question's ticker and fiscal year (no cross-company leakage); falls back to the full corpus with an explicit note when scope is missing.
- `python_calc` — arithmetic evaluator restricted to an AST operator whitelist.

**Training/eval parity** — the same episode loop (`core.py`), prompts, parser, tools, and reward function are used for training and evaluation. `core.py` has no `verl`/`agent_r1` imports; backends (vLLM for eval, Agent-R1 worker for training) are injected.

Default configuration: 4× NVIDIA A40 48GB, bf16, FSDP ZeRO-3, train batch 32 prompts × 8 rollouts, `lr=1e-6`, `max_prompt_length=4096`, `max_response_length=768`. See [`code/finagent_r1/scripts/common.sh`](code/finagent_r1/scripts/common.sh) and [`docs/00_project_overview.md`](docs/00_project_overview.md) for the full hyperparameter and memory budget.

## Repository layout

```
.
├── code/
│   ├── finagent_r1/       # project code: flow, tools, rewards, core loop,
│   │                      # data pipeline, vLLM evaluation, training scripts, 60 CPU tests
│   └── agent_r1/          # snapshot of the Agent-R1 framework (upstream, unmodified)
├── docs/                  # overview, framework guide, runbook, learning guide,
│   │                      # interview handbook, experiment summary, issues & fixes
├── results/               # metrics JSON (7 configs), summaries, bootstrap CIs, McNemar tests
├── samples/               # 60-case predictions for all 7 configs, 12 full trajectories,
│   │                      # small splits of FinQA data and retrieval corpus
└── scripts/               # orchestration helpers used for the multi-algorithm sweep
```

## Quick start

Requires Linux, Python ≥ 3.10, and (for training) 4× A40 48GB with `torch`, `vllm`, `ray`, and `verl==0.7.0`. Detailed installation: [`docs/02_runbook.md`](docs/02_runbook.md).

```bash
git clone https://github.com/bingyulove99-ai/FinAgent-R1.git
cd FinAgent-R1

# The training scripts expect the framework at upstream/Agent-R1; this repo
# stores the snapshot in code/, so link it into the expected location.
mkdir -p upstream && ln -s ../code upstream/Agent-R1

# 1. Verify the environment (does not install packages)
bash code/finagent_r1/scripts/setup_env.sh

# 2. Download FinQA and build parquet + retrieval corpus (-> data/finqa/)
bash code/finagent_r1/scripts/prepare_data.sh

# 3. CPU-only tests (no GPU needed)
cd code/finagent_r1 && python -m unittest discover -s tests -v && cd ../..

# 4. Train (choose one)
bash code/finagent_r1/scripts/train_grpo.sh        # ~27h on 4×A40
# train_rloo.sh | train_reinforce.sh | train_ppo.sh | train_token_gae.sh

# 5. Evaluate a checkpoint on the test split
MODEL=outputs/finqa_qwen3b_grpo/global_step_195/huggingface \
  bash code/finagent_r1/scripts/evaluate.sh
```

## Case inspection

`samples/predictions/<config>_cases60.md` shows side-by-side predictions for all 7 configurations on the same 60 test questions; `samples/trajectories/<tag>_traj12.jsonl` contains full multi-turn trajectories (per-step responses, tool calls, stop reasons) for the first 12 questions. See [`samples/README.md`](samples/README.md).

## Documentation

| Document | Contents |
|---|---|
| [`docs/00_project_overview.md`](docs/00_project_overview.md) | Background, dataset, model choice, MDP formulation, algorithms, training config, evaluation protocol |
| [`docs/01_framework_guide.md`](docs/01_framework_guide.md) | How Agent-R1 works internally, file by file |
| [`docs/02_runbook.md`](docs/02_runbook.md) | Install, data prep, training, evaluation, troubleshooting |
| [`docs/03_learning_guide.md`](docs/03_learning_guide.md) | Study plan and code-reading exercises |
| [`docs/04_interview_handbook.md`](docs/04_interview_handbook.md) | Interview questions with detailed answers |
| [`docs/05_experiment_summary_zh.md`](docs/05_experiment_summary_zh.md) | Experiment log (Chinese): setup, errors, tuning, results |
| [`docs/ISSUES_AND_FIXES.md`](docs/ISSUES_AND_FIXES.md) | Every issue encountered, root cause, and fix |
| [`docs/workspace_readme_original.md`](docs/workspace_readme_original.md) | Historical README of the original workspace |

## Reproducibility notes

- Single seed (42) per algorithm; all checkpoints for a run share the same data and hyperparameters except the algorithm switch.
- Checkpoints are selected on dev accuracy, not on test; the test split is evaluated once per selected checkpoint.
- Confidence intervals are question-level paired bootstrap (1,000 resamples); significance is paired McNemar.
- The Token-GAE run was early-stopped and resumed with reduced micro-batch/token budget after a critic OOM; details in [`results/PROGRESS.md`](results/PROGRESS.md).
- **Not included in this repository:** model weights/checkpoints, full prediction dumps (1,147 rows per config), training logs, and the full FinQA dataset/corpus. Small, non-leaking samples are under `samples/`.

## Data and privacy

- Data comes from the public FinQA benchmark (SEC filings of S&P 500 companies); no private or proprietary data is used.
- Absolute paths in metadata and documents have been replaced with generic placeholders; no usernames, emails, hostnames, IP addresses, or API keys are present in this repository.
- Please check the upstream licenses before redistributing datasets, checkpoints, or derived corpora.

## Acknowledgements

- [Agent-R1](https://github.com/AgentR1/Agent-R1) (USTC) — Apache 2.0 framework; a snapshot lives in `code/agent_r1/`.
- [FinQA](https://github.com/cszyssrs/FinQA) (Chen et al., EMNLP 2021) — dataset.
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) — base policy (Apache 2.0).
- [verl](https://github.com/volcengine/verl) — distributed RL infrastructure.