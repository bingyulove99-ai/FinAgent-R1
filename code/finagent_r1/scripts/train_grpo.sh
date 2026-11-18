#!/usr/bin/env bash
# GRPO: group-relative policy optimization, no critic.
# Default: Qwen2.5-3B-Instruct, 4x A40 48GB, 8 rollouts per question.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

run_train "${EXP_NAME:-finqa_qwen3b_grpo}" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    critic.enable=False \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
    "$@"
