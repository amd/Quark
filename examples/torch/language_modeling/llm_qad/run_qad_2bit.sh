#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Parameterized QAD 2-bit full-weight training (Block-AP warmup + end-to-end JSD KD).
# Works for Phi-4 (Phi3) and QwQ-32B (Qwen2); override via environment variables.
#
# Required:
#   STUDENT_DIR  2-bit PTQ student checkpoint (from TwoBitScalar PTQ)
#   TEACHER_DIR  BF16 teacher checkpoint
#   OUTPUT_DIR   where to write the QAD checkpoint
#
# Common overrides (defaults match the QAD v6 recipe):
#   GPUS=0,1,2,3   MAX_STEPS=60000   GROUP_SIZE=64   LR=5e-6
#   DATASET=taskmix   TASK_INCLUDE=all   SEQ_LEN=1024
#   BLOCK_AP=1 (warmup on)   ON_POLICY=1   LEARNED_SCALES=1
#
# Example (Phi-4):
#   STUDENT_DIR=/path/phi4_2bit TEACHER_DIR=/path/phi-4 OUTPUT_DIR=/path/phi4_qad_v6 \
#     bash run_qad_2bit.sh
set -euo pipefail

: "${STUDENT_DIR:?set STUDENT_DIR to the 2-bit PTQ student checkpoint}"
: "${TEACHER_DIR:?set TEACHER_DIR to the BF16 teacher checkpoint}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

GPUS="${GPUS:-0,1,2,3}"
MAX_STEPS="${MAX_STEPS:-60000}"
GROUP_SIZE="${GROUP_SIZE:-64}"
LR="${LR:-5e-6}"
DATASET="${DATASET:-taskmix}"
TASK_INCLUDE="${TASK_INCLUDE:-all}"
SEQ_LEN="${SEQ_LEN:-1024}"
KD_LOSS="${KD_LOSS:-jsd}"
BLOCK_AP="${BLOCK_AP:-1}"
ON_POLICY="${ON_POLICY:-1}"
LEARNED_SCALES="${LEARNED_SCALES:-1}"

EXTRA=()
[ "${BLOCK_AP}" = "1" ]       && EXTRA+=(--block_ap_warmup)
[ "${ON_POLICY}" = "1" ]      && EXTRA+=(--on_policy_kd)
[ "${LEARNED_SCALES}" = "1" ] && EXTRA+=(--learned_scales)
[ -n "${BLOCK_AP_RESUME:-}" ] && EXTRA+=(--block_ap_resume "${BLOCK_AP_RESUME}")

NGPU="$(echo "${GPUS}" | tr ',' '\n' | grep -c .)"

echo "QAD 2-bit: student=${STUDENT_DIR} teacher=${TEACHER_DIR} -> ${OUTPUT_DIR}"
echo "  GPUS=${GPUS} (${NGPU})  steps=${MAX_STEPS}  group=${GROUP_SIZE}  lr=${LR}  kd=${KD_LOSS}"
echo "  extra: ${EXTRA[*]:-(none)}"

CUDA_VISIBLE_DEVICES="${GPUS}" python train_qad_2bit_fullweight.py \
    --model_dir "${STUDENT_DIR}" \
    --teacher_model_dir "${TEACHER_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --group_size "${GROUP_SIZE}" \
    --dataset "${DATASET}" \
    --task_include "${TASK_INCLUDE}" \
    --seq_len "${SEQ_LEN}" \
    --max_steps "${MAX_STEPS}" \
    --lr "${LR}" \
    --kd_loss_type "${KD_LOSS}" \
    --bf16 \
    $( [ "${NGPU}" -gt 1 ] && echo --multi_gpu ) \
    "${EXTRA[@]}"
