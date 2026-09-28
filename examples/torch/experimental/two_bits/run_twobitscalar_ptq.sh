#!/bin/bash
# TwoBitScalar 2-bit weight-only PTQ (AWQ + SRHT + 4-level scalar).
#
# TwoBitScalar is an experimental algorithm (quark.experimental.torch.twobitscalar).
# It is driven through the standard PTQ entry point via `--quant_algo twobitscalar`
# plus a config file (`--quant_algo_config_file`), i.e. the same config-driven
# pattern used by every other algorithm - no per-algorithm CLI flags.
#
# Produces a fake-quantized HF model (~2.25 effective bits/weight) suitable as
# the initialization for the 2-bit QAD / LoRA+KD pipelines.
#
# Usage:
#   MODEL_DIR=/path/to/bf16 OUTPUT_DIR=/path/to/out bash run_twobitscalar_ptq.sh
#
# Optional env overrides:
#   CONFIG (twobitscalar_config.json), GPU (0), NUM_CALIB (4096), SEQ_LEN (1024)
#
# To dump the per-layer SRHT/AWQ sidecar (for factored / shared-rotation exports),
# add "dump_sidecar_dir": "/path/to/sidecar" to the config JSON.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to a BF16 HF model dir}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
GPU="${GPU:-0}"
NUM_CALIB="${NUM_CALIB:-4096}"
SEQ_LEN="${SEQ_LEN:-1024}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/twobitscalar_config.json}"
PTQ="$SCRIPT_DIR/../../language_modeling/llm_ptq/quantize_quark.py"

CUDA_VISIBLE_DEVICES="$GPU" python3 "$PTQ" \
  --model_dir "$MODEL_DIR" \
  --quant_scheme bfp16 --quant_algo twobitscalar \
  --quant_algo_config_file twobitscalar "$CONFIG" \
  --exclude_layers "*embed_tokens*" "*lm_head*" \
  --num_calib_data "$NUM_CALIB" --seq_len "$SEQ_LEN" --batch_size 1 \
  --evaluation_dataset wikitext --export_weight_format fake_quantized --model_export hf_format \
  --output_dir "$OUTPUT_DIR"
