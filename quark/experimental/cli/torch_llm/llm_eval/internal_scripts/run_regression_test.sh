#!/bin/bash
# set -e

#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

if [ -n "$WORKSPACE_ROOT_DIR" ]; then
    cd ${WORKSPACE_ROOT_DIR}/examples/torch/language_modeling/llm_eval
    log_dir=${WORKSPACE_ROOT_DIR}/logs_regression_test
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd ${SCRIPT_DIR}/../
    log_dir=${SCRIPT_DIR}/../../../../../logs_regression_test
fi
mkdir -p $log_dir
pip install -r requirements.txt

# echo modelzoo repoinfo and version for check if updates need merged here

if [ -n "$PRETRAINED_MODEL_DIR" ]; then
    declare -A MODEL_DATAPATH=(
    ['Phi-3.5-mini-instruct']="${PRETRAINED_MODEL_DIR}/microsoft/Phi-3.5-mini-instruct"
    )
else
    PRETRAINED_MODEL_DIR="/group/ossmodelzoo/quark_torch/huggingface_pretrained_models/"
    declare -A MODEL_DATAPATH=(
    ['Phi-3.5-mini-instruct']="${PRETRAINED_MODEL_DIR}/microsoft/Phi-3.5-mini-instruct"
    )
fi


export MODEL_DIR="${MODEL_DATAPATH['Phi-3.5-mini-instruct']}"
export QUANTIZED_MODEL_DIR_TORCH="Phi-3.5-mini-instruct-awq-torch/"


TOTAL_TIME=0
echo "pwd -> $(pwd)"

cd ../llm_ptq/
pwd
SECONDS=0
log_file=$log_dir/llm_eval_ptq_export_hf.log
python quantize_quark.py --model_dir $MODEL_DIR \
        --quant_scheme w_int4_per_group_sym \
        --num_calib_data 128 \
        --quant_algo awq \
        --dataset pileval_for_awq_benchmark \
        --seq_len 512 \
        --model_export hf_format \
        --skip_evaluation \
        --output_dir $QUANTIZED_MODEL_DIR_TORCH > $log_file 2>&1
ret_stat=$?
[ $ret_stat -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))
if [[ $ret_stat != 0 ]];then echo 'Not export OK, exit';exit 1;fi

echo "--------------------------------------------"
echo "TESTING LM EVAL IN STANDARD MODE"
echo "--------------------------------------------"
cd ../llm_eval
pwd

SECONDS=0
log_file=$log_dir/llm_eval_no_quantization_gpu.log
echo "Torch Model Without quantization: PPL"
python llm_eval.py \
    --model_args pretrained=$MODEL_DIR \
    --ppl \
    --trust_remote_code \
    --batch_size 1 \
    --device cuda > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

SECONDS=0
log_file=$log_dir/llm_eval_no_quantization_rouge_mentor_cpu.log
echo "Torch Model Without quantization: ROUGE/METEOR on 10 samples"
python llm_eval.py \
    --model_args pretrained=$MODEL_DIR \
    --rouge \
    --meteor\
    --dataset xsum\
    --trust_remote_code \
    --batch_size 1 \
    --num_eval_data 10 \
    --device cpu > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

SECONDS=0
log_file=$log_dir/llm_eval_no_quantization_lm_harness_gpu.log
echo "Torch Model Without quantization: LM Harness Tasks"
python llm_eval.py \
    --model_args pretrained=$MODEL_DIR \
    --model hf \
    --tasks mmlu_management\
    --batch_size 1 \
    --device cuda > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

SECONDS=0
log_file=$log_dir/llm_eval_quantization_gpu.log
echo "Quantized Torch Model: PPL"
python llm_eval.py \
    --model_args pretrained=$MODEL_DIR \
    --model_reload \
    --import_file_format hf_format \
    --import_model_dir "../llm_ptq/$QUANTIZED_MODEL_DIR_TORCH" \
    --ppl \
    --trust_remote_code \
    --batch_size 1 \
    --device cuda > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

SECONDS=0
log_file=$log_dir/llm_eval_quantization_rouge_mentor_gpu.log
echo "Quantized Torch Model: ROUGE/METEOR on 10 samples"
python llm_eval.py \
    --model_args pretrained=$MODEL_DIR \
    --model_reload \
    --import_file_format hf_format \
    --import_model_dir "../llm_ptq/$QUANTIZED_MODEL_DIR_TORCH" \
    --rouge \
    --meteor \
    --dataset xsum \
    --trust_remote_code \
    --batch_size 1 \
    --num_eval_data 10 \
    --device cuda > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

SECONDS=0
log_file=$log_dir/llm_eval_quantization_lm_harness_gpu.log
echo "Quantized Torch Model: LM Harness Tasks"
python llm_eval.py \
    --model_args pretrained=$MODEL_DIR \
    --model_reload \
    --import_file_format hf_format \
    --import_model_dir "../llm_ptq/$QUANTIZED_MODEL_DIR_TORCH" \
    --model hf \
    --tasks mmlu_management \
    --batch_size 1 \
    --device cuda > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

echo "--------------------------------------------"
echo "TESTING LM EVAL IN OFFLINE MODE"
echo "--------------------------------------------"

SECONDS=0
log_file=$log_dir/llm_eval_offline_retrieve_dataset.log
echo "1. Retrieve Dataset for 20 samples GSM8k"
python llm_eval.py \
    --mode offline \
    --retrieve_dataset \
    --tasks gsm8k \
    --limit 20 \
    --num_fewshot 0 > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

# change outputs_path to 'gsm8k_references_limit-20.txt'
SECONDS=0
log_file=$log_dir/llm_eval_offline_oga.log
echo "2. Run Evaluation Scores on OGA Predictions"
python llm_eval.py \
    --mode offline \
    --eval_mode \
    --outputs_path "./gsm8k_references_limit-20.txt" \
    --tasks gsm8k \
    --limit 20 \
    --num_fewshot 0 \
    --eor "<EOR>"  > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

# check OGA flow
SECONDS=0
log_file=$log_dir/llm_eval_builder_export_cpu_import_cpu.log
echo "TEST on builder export"
python3 -m onnxruntime_genai.models.builder \
	-i $MODEL_DIR \
	-o ./export_onnx \
	-p fp32 \
	-e cpu > $log_file 2>&1
python llm_eval.py \
    --model_args pretrained=$MODEL_DIR \
    --import_file_format onnx_format \
    --import_model_dir ./export_onnx \
    --model hf \
    --tasks mmlu_management \
    --batch_size 1 \
    --device cpu >> $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

echo "run lm-evaluation-harness metrics offline for ONNX models"
# 1
SECONDS=0
log_file=$log_dir/llm_eval_harness_offline_retrieve_dataset.log
echo "lm-evaluation-harness retrieve dataset"
python llm_eval.py \
    --mode offline \
    --retrieve_dataset \
    --tasks tinyGSM8k \
    --num_fewshot 0 > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

# 2
SECONDS=0
log_file=$log_dir/llm_eval_harness_offline_builder_export_cpu.log
echo "TEST on builder export"
python3 -m onnxruntime_genai.models.builder \
       -i $MODEL_DIR \
       -o ./export_onnx \
       -p fp32 \
       -e cpu > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

# 3
SECONDS=0
log_file=$log_dir/llm_eval_harness_offline_oga_references.log
echo "lm-evaluation-harness oga_references"
python llm_eval.py \
    --mode offline \
    --oga_references \
    --inputs example_files/tinyGSM8k_inputs_limit-None.json \
    --import_model_dir ./export_onnx/ \
    --import_file_format onnx_format \
    --tasks tinyGSM8k \
    --num_fewshot 0 \
    --case default \
    --model_name Phi3.5-mini-instruct-onnx \
    --eor "<EOR>" > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

# 4
SECONDS=0
log_file=$log_dir/llm_eval_harness_offline_oga_eval_pth.log
echo "lm-evaluation-harness eval on pth"
python llm_eval.py \
    --mode offline \
    --eval_mode \
    --outputs_path "./Phi3.5-mini-instruct-onnx_tinyGSM8k_limit-None_default.txt" \
    --tasks tinyGSM8k \
    --num_fewshot 0 \
    --eor "<EOR>" > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

# 5
SECONDS=0
log_file=$log_dir/llm_eval_harness_offline_oga_eval_quantized.log
echo "lm-evaluation-harness eval quantized"
python llm_eval.py \
    --mode offline \
    --eval_mode \
    --outputs_path example_files/tinyGSM8k_references_limit-None.txt \
    --tasks tinyGSM8k \
    --num_fewshot 0 \
    --eor "<EOR>" > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

# 6.1
SECONDS=0
log_file=$log_dir/llm_eval_harness_offline_limit-None_default.log
#echo "lm-evaluation-harness eval on pth"
echo "$(basename $log_file)"
python llm_eval.py \
    --mode offline \
    --eval_mode \
    --outputs_path example_files/deepseek-qwen1.5B_tinyGSM8k_limit-None_default.txt \
    --tasks tinyGSM8k \
    --num_fewshot 0 \
    --eor "<EOR>" > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

# 6.2
SECONDS=0
log_file=$log_dir/llm_eval_harness_offline_limit-None_psu_prompt_eos_stop.log
#echo "lm-evaluation-harness eval on pth"
echo "$(basename $log_file)"
python llm_eval.py \
    --mode offline \
    --eval_mode \
    --outputs_path example_files/deepseek-qwen1.5B_tinyGSM8k_limit-None_psu_prompt_eos_stop.txt \
    --tasks tinyGSM8k \
    --num_fewshot 0 \
    --eor "<EOR>" > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

# 6.3
SECONDS=0
log_file=$log_dir/llm_eval_harness_offline_limit-None_psu_prompt.log
#echo "lm-evaluation-harness eval on pth"
echo "$(basename $log_file)"
python llm_eval.py \
    --mode offline \
    --eval_mode \
    --outputs_path example_files/deepseek-qwen1.5B_tinyGSM8k_limit-None_psu_prompt.txt \
    --tasks tinyGSM8k \
    --num_fewshot 0 \
    --eor "<EOR>" > $log_file 2>&1
[ $? -eq 0 ] && echo "TestResult: PASSED" |tee -a $log_file || echo "TestResult: FAILED" |tee -a $log_file
date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
TOTAL_TIME=$((TOTAL_TIME+SECONDS))

date -ud @${TOTAL_TIME} "+Time elapsed: %H:%M:%S" |tee -a ${log_dir}/llm_eval_total_time.log
