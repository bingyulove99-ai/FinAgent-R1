#!/usr/bin/env bash
# Finish Token-GAE epoch 1 from step 50 with a memory-safe micro-batch budget.
# Root cause of the earlier crash: critic_update_critic CUDA OOM once responses
# grew to ~757 tokens (33.3GB torch + 13.8GB vLLM > 44.4GB GPU). Here we halve
# the per-GPU token budget and micro-batches, and lower vLLM reservation.
# Everything else is identical to run_all_trainings_v2.sh. Checkpoint selection
# remains "best dev"; the already-evaluated step-50 model stays the fallback.
set -uo pipefail
source /data/home.old/user/agent-r1/env.sh
cd /home/user/Agent-R1-main

export EPOCHS=1
export TRAIN_BATCH_SIZE=32
export VAL_BATCH_SIZE=883
export SAVE_FREQ=25
export TEST_FREQ=25
export MAX_ACTOR_CKPT_TO_KEEP=10
export MAX_CRITIC_CKPT_TO_KEEP=10
export AGENT_WORKERS=4
export EXP_NAME=finqa_qwen3b_token_gae
export ACTOR_LR=5e-7
export PPO_GPU_MEMORY_UTILIZATION=0.30
export PPO_ACTOR_MICRO_BATCH=1
export CRITIC_MICRO_BATCH_SIZE=1

LOGDIR=/data/home.old/user/agent-r1/logs
rc=1
for attempt in 1 2 3; do
    echo "[$(date '+%F %T')] === RESUME token_gae attempt=$attempt (from step $(cat /data/home.old/user/agent-r1/outputs/finqa_qwen3b_token_gae/latest_checkpointed_iteration.txt 2>/dev/null)) ==="
    bash projects/finagent_r1/scripts/train_token_gae.sh \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
        actor_rollout_ref.actor.kl_loss_coef=0.03 \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        +ray_kwargs.ray_init.object_store_memory=20000000000 \
        data.dataloader_num_workers=2 \
        critic.model.fsdp_config.param_offload=True \
        critic.model.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        critic.use_dynamic_bsz=True \
        critic.ppo_max_token_len_per_gpu=4096 \
        > "$LOGDIR/train_token_gae_resume.log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] === RESUME attempt $attempt exit=$rc ==="
    [ "$rc" -eq 0 ] && break
    pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null || true; pkill -9 -f "ray::WorkerDict" 2>/dev/null || true
    sleep 30
done
echo "[$(date '+%F %T')] token_gae resume finished rc=$rc"
