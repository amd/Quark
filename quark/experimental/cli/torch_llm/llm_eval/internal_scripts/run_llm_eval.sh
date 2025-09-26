#!/bin/bash
set -e

#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# install pkgs from llm_eval
pip install -r requirements.txt

# verify cuda and quark
python -c 'import torch;print(torch.cuda.is_available())'
python -c 'import quark'

export PRETRAINED_MODEL_DIR="microsoft/Phi-3.5-mini-instruct"
export QUANTIZED_MODEL_DIR_TORCH="Phi-3.5-mini-instruct-awq-torch/"

cd "../llm_ptq/"
echo "Now in $(pwd)"

python quantize_quark.py --model_dir $PRETRAINED_MODEL_DIR \
        --quant_scheme w_int4_per_group_sym \
        --num_calib_data 128 \
        --quant_algo awq \
        --dataset pileval_for_awq_benchmark \
        --seq_len 512 \
        --model_export hf_format \
        --skip_evaluation \
        --output_dir $QUANTIZED_MODEL_DIR_TORCH

cd "../llm_eval/"
echo "Now in $(pwd)"

echo -n "--------------------------------------------\n"
echo -n "TESTING LM EVAL IN STANDARD MODE\n"
echo -n "--------------------------------------------\n"

# Evaluating Model Without quantization
echo -n "Torch Model Without quantization: PPL"
python llm_eval.py \
    --model_args pretrained=$PRETRAINED_MODEL_DIR \
    --ppl \
    --trust_remote_code \
    --batch_size 1 \
    --device cuda

echo -n "--------------------------------------------\n"
echo -n "Torch Model Without quantization: ROUGE/METEOR on 10 samples"
python llm_eval.py \
    --model_args pretrained=$PRETRAINED_MODEL_DIR \
    --rouge \
    --meteor\
    --dataset xsum\
    --trust_remote_code \
    --batch_size 1 \
    --num_eval_data 10 \
    --device cpu

echo -n "--------------------------------------------\n"
echo -n "Torch Model Without quantization: LM Harness Tasks"
python llm_eval.py \
    --model_args pretrained=$PRETRAINED_MODEL_DIR \
    --model hf \
    --tasks mmlu_management\
    --batch_size 1 \
    --device cuda

echo -n "--------------------------------------------\n"

#Evaluating Quantized Model
echo -n "Quantized Torch Model: PPL"
python llm_eval.py \
    --model_args pretrained=$PRETRAINED_MODEL_DIR \
    --model_reload \
    --import_file_format hf_format \
    --import_model_dir "../llm_ptq/$QUANTIZED_MODEL_DIR_TORCH" \
    --ppl \
    --trust_remote_code \
    --batch_size 1 \
    --device cuda

echo -n "--------------------------------------------\n"
echo -n "Quantized Torch Model: ROUGE/METEOR on 10 samples"
python llm_eval.py \
    --model_args pretrained=$PRETRAINED_MODEL_DIR \
    --model_reload \
    --import_file_format hf_format \
    --import_model_dir "../llm_ptq/$QUANTIZED_MODEL_DIR_TORCH" \
    --rouge \
    --meteor \
    --dataset xsum \
    --trust_remote_code \
    --batch_size 1 \
    --num_eval_data 10 \
    --device cpu

echo -n "--------------------------------------------\n"
echo -n "Quantized Torch Model: LM Harness Tasks"
python llm_eval.py \
    --model_args pretrained=$PRETRAINED_MODEL_DIR \
    --model_reload \
    --import_file_format hf_format \
    --import_model_dir "../llm_ptq/$QUANTIZED_MODEL_DIR_TORCH" \
    --model hf \
    --tasks mmlu_management \
    --batch_size 1 \
    --device cuda

echo -n "--------------------------------------------\n"
echo -n "TESTING LM EVAL IN OFFLINE MODE\n"
echo -n "--------------------------------------------\n"

echo -n "1. Retrieve Dataset for 20 samples GSM8k"
python llm_eval.py \
    --mode offline \
    --retrieve_dataset \
    --tasks gsm8k \
    --limit 20 \
    --num_fewshot 0

echo -n "--------------------------------------------\n"
echo -n "2. Run Evaluation Scores on OGA Predictions"
python llm_eval.py \
    --mode offline \
    --eval_mode \
    --outputs_path "./internal_scripts/onnx_format_gsm8k_predictions_example.txt" \
    --tasks gsm8k \
    --limit 20 \
    --num_fewshot 0 \
    --eor "<EOR>"
