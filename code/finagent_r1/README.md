# FinAgent-R1

A personal agentic-RL project built on top of [Agent-R1](../../upstream/Agent-R1),
using **FinQA** and **Qwen2.5-3B-Instruct**, targeting **4× NVIDIA A40 48GB**.

## What is in this package

```
finagent_r1/
├── prompts.py                       # system/user prompts + tool schemas
├── parsing.py                       # <tool_call>/<answer> parsing, prompt builder
├── tools.py                         # scoped BM25 retriever + AST calculator
├── rewards.py                       # numeric-tolerant composite reward
├── core.py                          # framework-agnostic multi-turn episode loop
├── flow.py                          # Agent-R1 AgentFlowBase adapter
├── configs/flow.yaml                # agent flow registry entry
├── data/prepare_finqa.py            # FinQA -> parquet + corpus
├── data/dataset_stats.py            # dataset card / audit
├── evaluation/infer_vllm.py         # vLLM agent / direct-answer inference
└── evaluation/evaluate_predictions.py
```

The design rule is simple: **`core.py` never imports verl/agent_r1**.  Training
and evaluation inject different model backends into the same episode loop, so
prompts, tools, parsing, stop rules and rewards cannot drift apart.

## Reward contract

* Intermediate tool-use turns: `reward_score = 0.0`
* Terminal turn: `reward_score = None` → Agent-R1 calls `rewards.compute_score`
* Reward = outcome (numeric match / token F1) + format (0.1) − tool penalty,
  multiplied by a mild step-efficiency discount, clipped to `[-0.2, 1.2]`.

## Supported algorithms

| Script | `adv_estimator` | Critic | Note |
|---|---|---|---|
| `train_grpo.sh` | `grpo` | no | group-relative baseline |
| `train_rloo.sh` | `rloo` | no | leave-one-out baseline |
| `train_reinforce.sh` | `reinforce` | no | step-level discounted returns |
| `train_ppo.sh` | `gae` | yes (3B) | step-level GAE; also Agent-R1's StepPO-style path |
| `train_token_gae.sh` | `token_gae` | yes (3B) | token-level credit assignment ablation |

`gigpo` requires `anchor_obs` and is not enabled for this environment.
`remax` is not implemented for agent flows in the upstream trainer.

## Tests

```bash
cd projects/finagent_r1
python -m unittest discover -s tests -v
```

60 CPU-only tests cover parsing, reward math, BM25 scoping, the AST sandbox,
the full episode loop, prompt-budget shrinking, and the FinQA conversion.

## Status / honesty note

No training logs or checkpoints are included.  Reported metrics must be produced
by `scripts/evaluate.sh` on your own run.  The default configuration is sized
for 4×A40; 7B requires offload or LoRA and is not the default.
