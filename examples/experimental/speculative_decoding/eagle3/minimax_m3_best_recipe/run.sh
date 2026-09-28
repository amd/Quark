#!/usr/bin/env bash
# Thin selector for the public MiniMax-M3 large-model runner skeleton.
set -euo pipefail
PROFILE_DIR="$(cd "$(dirname "$0")" && pwd)"
export RUNNER_PROFILE="minimax_m3_best_recipe"
export CONFIG_DIR="${CONFIG_DIR:-$PROFILE_DIR/configs}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-$CONFIG_DIR/large_model_eagle3.yaml}"
export DRAFT_CONFIG_SOURCE="${DRAFT_CONFIG_SOURCE:-}"
export BASE_MODEL="${BASE_MODEL:-amd/MiniMax-M3-MXFP4}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-minimax-m3}"
export TARGET_TP_SIZE="${TARGET_TP_SIZE:-${target_tp_size:-4}}"
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-1}"
# EX_DIR and CACHE_DIR default in _common.sh, outside the packaged asset tree.
exec bash "$PROFILE_DIR/../common/run_all.sh" "$@"
