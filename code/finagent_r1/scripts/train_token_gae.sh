#!/usr/bin/env bash
# Token-level GAE: a critic is trained on every generated token, not once per
# step.  It is a useful ablation of Agent-R1's step-level credit assignment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

run_train "${EXP_NAME:-finqa_qwen3b_token_gae}" \
    algorithm.adv_estimator=token_gae \
    algorithm.gamma="${GAMMA:-0.99}" \
    algorithm.lam="${LAM:-0.95}" \
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
