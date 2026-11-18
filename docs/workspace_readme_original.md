# FinAgent-R1 Workspace

This workspace contains two clearly separated layers:

```
.
├── upstream/Agent-R1/          # pristine Agent-R1 framework (do not edit)
├── projects/finagent_r1/       # our extension: FinQA agentic-RL project
├── docs/                       # project plan, framework guide, runbook,
│                               # learning guide, interview handbook
├── data/finqa/                 # generated data (gitignored)
└── scripts/                    # workspace-level helpers (optional)
```

## Why the split?

`upstream/Agent-R1` is the original open-source framework.  We never modify it.
All task-specific code lives in `projects/finagent_r1`, which imports the
framework through `PYTHONPATH`.  This prevents the classic problem where a
personal recipe slowly edits the framework until upstream updates are
impossible to merge.

## Project in one paragraph

**FinAgent-R1** trains `Qwen2.5-3B-Instruct` to answer multi-step questions over
SEC filings from **FinQA** (8,281 QA pairs over 2,789 reports).  The agent has
two tools: a ticker/year-scoped BM25 retriever over filing text and tables, and
an AST-whitelisted arithmetic evaluator.  Each turn is a **step-level MDP
transition** in Agent-R1; the policy is trained with GRPO, RLOO, REINFORCE,
PPO/step-level GAE, or token-GAE.  The default hardware target is **4× NVIDIA
A40 48GB**.

## Quick start (Linux training node)

```bash
# 0. install verl==0.7.0 + PyTorch + vLLM + Ray (see docs/02_runbook.md)
bash projects/finagent_r1/scripts/setup_env.sh

# 1. prepare FinQA
bash projects/finagent_r1/scripts/prepare_data.sh

# 2. CPU-only sanity check (no GPU needed)
bash projects/finagent_r1/scripts/smoke_test.sh

# 3. train
bash projects/finagent_r1/scripts/train_grpo.sh

# 4. evaluate
MODEL=outputs/finqa_qwen3b_grpo/global_step_100/huggingface \
  bash projects/finagent_r1/scripts/evaluate.sh
```

## Documentation map

| Document | What it answers |
|---|---|
| [`docs/00_project_overview.md`](docs/00_project_overview.md) | Background, dataset, model choice, algorithms, epochs, framework, evaluation protocol |
| [`docs/01_framework_guide.md`](docs/01_framework_guide.md) | How Agent-R1 works internally, file by file |
| [`docs/02_runbook.md`](docs/02_runbook.md) | Install, data prep, training, evaluation, troubleshooting |
| [`docs/03_learning_guide.md`](docs/03_learning_guide.md) | 4-week study plan and code-reading exercises |
| [`docs/04_interview_handbook.md`](docs/04_interview_handbook.md) | 60+ interview questions with detailed answers |

## Status

The code, data pipeline, reward, tools and CPU tests are complete.  **No
training run is bundled.**  Any accuracy number must come from a run on your
Linux node; the documentation intentionally contains no fabricated results.

## Cleanup note

The previous prototype was removed:

- `data/fin_research/` — 13-template synthetic data (553 train / 647 test), not a real benchmark.
- `recipes/fin_research/`, `examples/fin_research/`, `tests/test_fin_research_flow.py` — old prototype code mixed into the upstream recipe tree.
- `docs/zh/{experiments_fin_research,fin_research_data_engineering,fin_research_interview_bible,fin_research_technical_report}.md` — documents containing unreproducible benchmark numbers.
- `.smoke-deps/`, `.smoke-tmp/`, all `__pycache__/`.

The upstream framework is now isolated under `upstream/Agent-R1/` and is not modified.

## Acknowledgements

- [Agent-R1](https://github.com/AgentR1/Agent-R1) (USTC) — Apache 2.0 framework.
- [FinQA](https://github.com/cszyssrs/FinQA) (Chen et al., EMNLP 2021) — dataset.
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) — base policy.
- [verl](https://github.com/volcengine/verl) — distributed RL infrastructure.

Please check each upstream license before redistributing data, checkpoints or
derived corpora.
