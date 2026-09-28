#!/usr/bin/env bash
# Thin selector for the validated Qwen3-8B runner profile.
set -euo pipefail
PROFILE_DIR="$(cd "$(dirname "$0")" && pwd)"
export RUNNER_PROFILE="qwen3_8b_quick_start"
export CONFIG_DIR="${CONFIG_DIR:-$PROFILE_DIR/configs}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-$CONFIG_DIR/qwen3_8b_eagle3.yaml}"
export DRAFT_CONFIG_SOURCE="${DRAFT_CONFIG_SOURCE:-$CONFIG_DIR/qwen3_8b_eagle3_draft.json}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen}"
export TARGET_TP_SIZE="${TARGET_TP_SIZE:-${target_tp_size:-1}}"
# EX_DIR and CACHE_DIR default in _common.sh, outside the packaged asset tree.
exec bash "$PROFILE_DIR/../common/run_all.sh" "$@"
