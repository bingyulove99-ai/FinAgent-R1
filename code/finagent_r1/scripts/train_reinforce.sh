#!/usr/bin/env bash
# REINFORCE: step-level discounted returns with batch whitening, no critic.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

run_train "${EXP_NAME:-finqa_qwen3b_reinforce}" \
    algorithm.adv_estimator=reinforce \
    algorithm.gamma="${GAMMA:-0.99}" \
    algorithm.lam=1.0 \
    critic.enable=False \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
    "$@"
