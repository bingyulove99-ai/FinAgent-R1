#!/usr/bin/env bash
# RLOO: leave-one-out baseline, no critic.  Requires n >= 2 rollouts.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

run_train "${EXP_NAME:-finqa_qwen3b_rloo}" \
    algorithm.adv_estimator=rloo \
    critic.enable=False \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
    "$@"
