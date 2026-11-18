#!/usr/bin/env bash
# Shared environment and Hydra overrides for FinAgent-R1.
# Source this file from every training script:
#   source "$(dirname "$0")/common.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_DIR:$ROOT_DIR/upstream/Agent-R1:${PYTHONPATH:-}"
export FINQA_CORPUS_PATH="${FINQA_CORPUS_PATH:-$ROOT_DIR/data/finqa/corpus.jsonl}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

PYTHON_BIN="${PYTHON:-python3}"
if ! "$PYTHON_BIN" -c "import sys" >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON:-python}"
fi
MODEL_PATH="${FIN_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_DATA="${FINQA_TRAIN:-$ROOT_DIR/data/finqa/train.parquet}"
VAL_DATA="${FINQA_VAL:-$ROOT_DIR/data/finqa/dev.parquet}"
FLOW_CONFIG="$PROJECT_DIR/finagent_r1/configs/flow.yaml"
REWARD_FN="$PROJECT_DIR/finagent_r1/rewards.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
LOGGER="${LOGGER:-[\"console\"]}"
EXP_NAME="${EXP_NAME:-finagent}"

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "ERROR: training data not found: $TRAIN_DATA" >&2
    echo "Run: bash $PROJECT_DIR/scripts/prepare_data.sh" >&2
    exit 1
fi

COMMON_ARGS=(
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_DATA"
    data.val_files="$VAL_DATA"
    data.train_batch_size="${TRAIN_BATCH_SIZE:-32}"
    data.val_batch_size="${VAL_BATCH_SIZE:-16}"
    data.max_prompt_length=4096
    data.max_response_length=768
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.return_raw_chat=True
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}"
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE:-16}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE:-4}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0.001}"
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.45}"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.prompt_length=4096
    actor_rollout_ref.rollout.response_length=768
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH:-8}"
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$FLOW_CONFIG"
    actor_rollout_ref.rollout.agent.default_agent_flow=finqa_agent
    actor_rollout_ref.rollout.agent.max_steps=6
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS:-4}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MICRO_BATCH:-8}"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    custom_reward_function.path="$REWARD_FN"
    custom_reward_function.name=compute_score
    reward_model.enable=False
    trainer.project_name=finagent_r1
    trainer.n_gpus_per_node="${N_GPUS:-4}"
    trainer.nnodes=1
    trainer.save_freq="${SAVE_FREQ:-50}"
    trainer.test_freq="${TEST_FREQ:-25}"
    trainer.total_epochs="${EPOCHS:-2}"
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-3}"
    trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP:-3}"
    trainer.default_local_dir="$OUTPUT_ROOT/$EXP_NAME"
    trainer.logger="$LOGGER"
)

run_train() {
    local exp_name="$1"
    shift
    EXP_NAME="$exp_name" \
    "$PYTHON_BIN" -m agent_r1.trainer.main_agent_ppo \
        "${COMMON_ARGS[@]}" \
        trainer.experiment_name="$exp_name" \
        trainer.default_local_dir="$OUTPUT_ROOT/$exp_name" \
        "$@"
}

echo "FinAgent-R1 experiment: ${EXP_NAME:-<unset>}"
echo "  model:  $MODEL_PATH"
echo "  train:  $TRAIN_DATA"
echo "  val:    $VAL_DATA"
echo "  corpus: $FINQA_CORPUS_PATH"
echo "  GPUs:   $CUDA_VISIBLE_DEVICES"
