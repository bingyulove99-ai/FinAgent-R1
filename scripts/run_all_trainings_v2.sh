#!/usr/bin/env bash
# Sequential runner v2 for all five FinAgent-R1 RL trainings on 2x A40 (GPU 0,3).
#
# Guarantees:
#  * every algorithm starts from the base policy in its own output directory
#    (a stale directory is archived, never silently resumed);
#  * a crashed run is retried, resuming only its own latest checkpoint;
#  * Ray / vLLM state is torn down between algorithms so runs cannot leak
#    processes or GPU memory into each other;
#  * all checkpoints are kept (8 saves/run) and the best-dev-accuracy
#    checkpoint is merged to HF; the selection rule is written next to it.
#
# Usage: bash run_all_trainings_v2.sh [algo ...]   (default: all five)
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
# Shared stability settings for every algorithm: half the documented lr and a
# strong low-variance KL anchor (docs/00 section 5.7 + runbook 4.x remedy).
export ACTOR_LR="${ACTOR_LR:-5e-7}"
export KL_COEF="${KL_COEF:-0.03}"

LOGDIR="$BASE/logs"
OUT="$BASE/outputs"
ARCH="$OUT/_archive"
mkdir -p "$LOGDIR" "$ARCH"
STAMP() { date '+%F %T'; }

DYN_ARGS=(
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}"
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    +ray_kwargs.ray_init.object_store_memory=20000000000
    data.dataloader_num_workers=2
)

cleanup_gpu() {
    "$BASE/envs/agentr1/bin/ray" stop --force >/dev/null 2>&1 || true
    pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "[m]ain_agent_ppo" 2>/dev/null || true
    sleep 15
}

ALGOS="${*:-grpo rloo reinforce ppo token_gae}"
for algo in $ALGOS; do
    exp="finqa_qwen3b_${algo}"
    [ "$algo" = "ppo" ] && exp="finqa_qwen3b_ppo_gae"

    if [ -f "$LOGDIR/done_${algo}" ]; then
        echo "[$(STAMP)] SKIP $algo (already completed: $LOGDIR/done_${algo})"
        continue
    fi

    # --- fresh, isolated output dir for this algorithm -------------------
    if [ -d "$OUT/$exp" ]; then
        dst="$ARCH/${exp}_$(date +%Y%m%d_%H%M%S)"
        echo "[$(STAMP)] archiving stale $OUT/$exp -> $dst"
        mv "$OUT/$exp" "$dst"
    fi
    export EXP_NAME="$exp"

    # --- memory layout ----------------------------------------------------
    EXTRA=("${DYN_ARGS[@]}")
    if [ "$algo" = "grpo" ]; then
        # Dr.GRPO: no std normalisation, the documented instability fallback.
        EXTRA+=(algorithm.norm_adv_by_std_in_grpo=False)
    fi
    if [ "$algo" = "ppo" ] || [ "$algo" = "token_gae" ]; then
        EXTRA+=(
            critic.model.fsdp_config.param_offload=True
            critic.model.fsdp_config.optimizer_offload=True
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
            critic.use_dynamic_bsz=True
            critic.ppo_max_token_len_per_gpu=8192
        )
        export PPO_GPU_MEMORY_UTILIZATION=0.30
    else
        export GPU_MEMORY_UTILIZATION=0.35
    fi

    # --- training with crash-only resume ---------------------------------
    rc=1
    for attempt in 1 2 3; do
        echo "[$(STAMP)] === START $algo (exp=$exp, lr=$ACTOR_LR, kl=$KL_COEF, attempt=$attempt) ==="
        bash "projects/finagent_r1/scripts/train_${algo}.sh" "${EXTRA[@]}" \
            > "$LOGDIR/train_${algo}.log" 2>&1
        rc=$?
        echo "[$(STAMP)] === END $algo exit=$rc ==="
        [ "$rc" -eq 0 ] && break
        cp "$LOGDIR/train_${algo}.log" "$LOGDIR/train_${algo}.attempt${attempt}.log"
        echo "[$(STAMP)] RETRY $algo (verl resumes from its own latest checkpoint)"
        cleanup_gpu
    done

    cp "$LOGDIR/train_${algo}.log" "$LOGDIR/train_${algo}.final.log"
    "$BASE/envs/agentr1/bin/python" "$BASE/parse_devacc.py" \
        "$LOGDIR/train_${algo}.log" "$LOGDIR/devacc_${algo}.tsv" \
        > "$LOGDIR/devacc_${algo}.log" 2>&1 || true

    if [ "$rc" -ne 0 ]; then
        echo "[$(STAMP)] TRAINING FAILED: $algo (see $LOGDIR/train_${algo}.log)"
        cleanup_gpu
        continue
    fi

    # --- merge the best dev-accuracy checkpoint --------------------------
    best=$(
        "$BASE/envs/agentr1/bin/python" - "$LOGDIR/train_${algo}.log" <<'PY'
import re, sys
pat = re.compile(r"step:(\d+) - .*?val-core/finqa/acc/(?:mean@1|mean):([0-9.]+)")
accs = {}
for line in open(sys.argv[1], errors="ignore"):
    m = pat.search(line)
    if m:
        accs[int(m.group(1))] = float(m.group(2))
print(max(accs, key=accs.get) if accs else "")
PY
    )
    if [ -z "$best" ]; then
        last=$(cat "$OUT/$exp/latest_checkpointed_iteration.txt" 2>/dev/null || true)
        best="$last"
        sel="last (no dev-acc lines parsed)"
    else
        sel="best dev accuracy"
    fi
    ckpt="$OUT/$exp/global_step_${best}/actor"
    if [ -n "$best" ] && [ -d "$ckpt" ]; then
        echo "[$(STAMP)] merging $exp global_step_${best} ($sel)"
        bash projects/finagent_r1/scripts/merge_checkpoint.sh "$ckpt" \
            > "$LOGDIR/merge_${algo}.log" 2>&1
        mrc=$?
        echo "[$(STAMP)] merge exit=$mrc -> $OUT/$exp/global_step_${best}/huggingface"
        if [ "$mrc" -eq 0 ]; then
            echo "$best" > "$OUT/$exp/best_checkpoint.txt"
            { echo "selection_rule: $sel"; echo "best_step: $best";
              echo "dev_acc: $(grep -E "^${best}\b" "$LOGDIR/devacc_${algo}.tsv" 2>/dev/null | cut -f2)";
            } > "$OUT/$exp/selection.txt"
            touch "$LOGDIR/done_${algo}"
            # Evaluate on the full test split immediately, while the GPU is idle.
            tag="${exp#finqa_qwen3b_}"
            hf="$OUT/$exp/global_step_${best}/huggingface"
            echo "[$(STAMP)] evaluating $algo on full test split"
            if bash "$BASE/eval_model.sh" "$tag" "$hf" agent \
                    > "$LOGDIR/eval_${algo}.log" 2>&1; then
                touch "$LOGDIR/eval_done_${algo}"
                echo "[$(STAMP)] eval $algo done"
            else
                echo "[$(STAMP)] WARNING: eval $algo failed (rerun run_all_evals.sh)"
            fi
        fi
    else
        echo "[$(STAMP)] WARNING: no checkpoint found for $exp (best=$best)"
    fi

    cleanup_gpu
done
echo "[$(STAMP)] ALL DONE"
