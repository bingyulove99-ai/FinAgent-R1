#!/usr/bin/env bash
# FinAgent-R1 runtime environment for this machine (2x A40: GPU 0 + GPU 3).
# Source before any train/eval script:  source /data/home.old/user/agent-r1/env.sh
export BASE=/data/home.old/user/agent-r1
export PATH="$BASE/envs/agentr1/bin:$PATH"
export PYTHON="$BASE/envs/agentr1/bin/python"
export CONDA_PREFIX="$BASE/envs/agentr1"
# Isolate from ~/.local/lib/python3.10 (pre-existing user site with old deps).
export PYTHONNOUSERSITE=1

export HF_HOME="$BASE/hf_cache"
export HF_ENDPOINT=https://hf-mirror.com
unset TRANSFORMERS_CACHE
export PIP_CACHE_DIR="$BASE/pip_cache"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

# Hardware adaptation: only GPU 0 and GPU 3 are free (1/2 serve other jobs).
export CUDA_VISIBLE_DEVICES=0,3
export N_GPUS=2
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# Local model + data (project symlinks point here).
export FIN_MODEL_PATH="$BASE/models/Qwen2.5-3B-Instruct"
export FINQA_CORPUS_PATH="$BASE/finqa/corpus.jsonl"
export OUTPUT_ROOT="$BASE/outputs"
export FINQA_RAW_DIR="$BASE/finqa_raw"

# Offline after downloads complete; flip to 0 if something must be fetched.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1