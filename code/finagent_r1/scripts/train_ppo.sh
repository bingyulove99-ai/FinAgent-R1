#!/usr/bin/env bash
# PPO with step-level GAE and a 3B critic.
#
# This is also the reproduction of Agent-R1's "StepPO-style" training: the
# upstream project exposes StepPO through the step-level `gae` estimator
# (see upstream examples/gsm8k/run_steppo.sh).  The actor and critic are both
# Qwen2.5-3B, so 4x48GB is sufficient; if you switch to a 7B model, enable
# critic/ref offload or reduce the batch size.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

run_train "${EXP_NAME:-finqa_qwen3b_ppo_gae}" \
    algorithm.adv_estimator=gae \
    algorithm.gamma="${GAMMA:-0.99}" \
    algorithm.lam="${LAM:-1.0}" \
    critic.enable=True \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-4}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_ACTOR_MICRO_BATCH:-2}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${PPO_GPU_MEMORY_UTILIZATION:-0.35}" \
    critic.model.path="$MODEL_PATH" \
    critic.optim.lr="${CRITIC_LR:-1e-5}" \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu="${CRITIC_MICRO_BATCH_SIZE:-2}" \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    trainer.critic_warmup=10 \
    "$@"
