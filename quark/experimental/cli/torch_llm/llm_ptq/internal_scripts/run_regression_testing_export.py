#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import itertools
import multiprocessing
import os
import subprocess
import time

# This script is only used for windows cpu test now.

wkspace_root_dir = os.environ.get("WORKSPACE_ROOT_DIR")
pretrained_model_dir = os.environ.get("PRETRAINED_MODEL_DIR")
script_dir = os.path.dirname(os.path.abspath(__file__))
if wkspace_root_dir:
    os.chdir(os.path.join(wkspace_root_dir, "examples", "torch", "language_modeling", "llm_ptq"))
    log_dir = os.path.join(wkspace_root_dir, "logs_regression_test")
else:
    os.chdir(os.path.join(script_dir, "../"))
    log_dir = os.path.join(script_dir, "..", "..", "..", "..", "..", "logs_regression_test")
if pretrained_model_dir:
    model_datapath = {
        "Facebook_Opt_125m": os.path.join(pretrained_model_dir, "facebook/opt-125m"),
        "Qwen_Qwen1.5-0.5B": os.path.join(pretrained_model_dir, "Qwen/Qwen1.5-0.5B"),
        "Meta_Llama_2_7b": os.path.join(pretrained_model_dir, "meta-llama/Llama-2-7b"),
        "Meta_Llama_3_8B": os.path.join(pretrained_model_dir, "meta-llama/Meta-Llama-3-8B"),
    }
else:
    pretrained_model_dir = "/group/ossmodelzoo/quark_torch/huggingface_pretrained_models/"
    model_datapath = {
        "Facebook_Opt_125m": os.path.join("facebook/opt-125m"),
        "Qwen_Qwen1.5-0.5B": os.path.join("Qwen/Qwen1.5-0.5B"),
        "Meta_Llama_2_7b": os.path.join(pretrained_model_dir, "meta-llama/Llama-2-7b"),
        "Meta_Llama_3_8B": os.path.join(pretrained_model_dir, "meta-llama/Meta-Llama-3-8B"),
    }


def verify_quark():
    print("the result of print(torch.cuda.is_available():")
    subprocess.run(["python", "-c", "import torch; print(torch.cuda.is_available())"])
    print("the result of import quark:")
    subprocess.run(["python", "-c", "import quark"])
    print(f"HUGGINGFACE_HUB_CACHE in run_regression_test_export.py is: {os.environ.get('HUGGINGFACE_HUB_CACHE')}")
    print(f"HF_DATASETS_CACHE in run_regression_test_export.py is: {os.environ.get('HF_DATASETS_CACHE')}")
    print(f"CUDA_VISIBLE_DEVICES in run_regression_test_export.py is: {os.environ.get('CUDA_VISIBLE_DEVICES')}")


def main():
    # test_model_list = [
    #     'Facebook_Opt_125m',
    #     'Qwen_Qwen1.5-0.5B',
    #     'Meta_Llama_2_7b',
    #     'Meta_Llama_3_8B'
    # ]
    test_model_list = ["Meta_Llama_2_7b", "Meta_Llama_3_8B"]

    test_cfg_list = [
        "Without_Quantization",
        "W_Int4_Per_Group_Sym",
        "W_FP8_A_FP8_O_FP8",
        "W_FP8_A_FP8_KV_Cache_FP8",
        "W_Int4_Per_Channel_Sym",
        "W_Int8_A_Int8_Per_Tensor_Sym_Dynamic",
        "W_Int8_A_Int8_Per_Tensor_Sym",
        "W_UInt4_A_Bfloat16_Per_Group_Asym",
        # "W_UInt4_Per_Group_Asym_with_SmoothQuant",
        # "W_UInt4_Per_Group_Asym_with_GPTQ",
        # "W_UInt4_Per_Group_Asym_with_AWQ"
    ]

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    start_time = time.time()
    cmd_list = list(itertools.product(test_model_list, test_cfg_list))
    print(cmd_list)
    with multiprocessing.Pool(processes=10) as pool:
        pool.starmap(run_subprocess, cmd_list)
    overall_timecost = time.time() - start_time
    with open(os.path.join(log_dir, "llm_total_time.log"), "a") as log:
        log.write(f"{overall_timecost}\n")


def get_options(cfg, model_dir, model_name):
    if model_name == "Facebook_Opt_125m" and cfg not in [
        "Without_Quantization",
        "W_Int8_A_Int8_Per_Tensor_Sym_Dynamic",
        "W_UInt4_Per_Group_Asym_with_GPTQ",
    ]:
        opt_ex = "--model_export onnx"
    else:
        opt_ex = ""
    if model_name == "Meta_Llama_2_7b" and cfg == "W_UInt4_Per_Group_Asym_with_AWQ":
        opt_ex = "--model_export gguf --group_size 32"
    if cfg == "Without_Quantization":
        opt_basic = f"--model_dir {model_dir} --skip_quantization"
    elif cfg == "W_Int8_A_Int8_Per_Tensor_Sym":
        opt_basic = f"--model_dir {model_dir} --output_dir output_dir --quant_scheme w_int8_a_int8_per_tensor_sym "
    elif cfg == "W_Int8_A_Int8_Per_Tensor_Sym_Dynamic":
        opt_basic = (
            f"--model_dir {model_dir} --output_dir output_dir --quant_scheme w_int8_a_int8_per_tensor_sym_dynamic "
        )
    elif cfg == "W_FP8_A_FP8_KV_Cache_FP8":
        if model_name == "Meta_Llama_2_7b":
            opt_ex += " --model_export vllm_adopted_safetensors"
        opt_basic = f"--model_dir {model_dir} --output_dir output_dir --quant_scheme w_fp8_a_fp8 --kv_cache_dtype fp8 "
    elif cfg == "W_FP8_A_FP8_O_FP8":
        if model_name == "Meta_Llama_2_7b":
            opt_ex += " --model_export vllm_adopted_safetensors"
        opt_basic = f"--model_dir {model_dir} --output_dir output_dir --quant_scheme w_fp8_a_fp8_o_fp8 "
    elif cfg == "W_Int4_Per_Channel_Sym":
        opt_basic = f"--model_dir {model_dir} --output_dir output_dir --quant_scheme w_int4_per_channel_sym "
    elif cfg == "W_Int4_Per_Group_Sym":
        if model_name == "Meta_Llama_2_7b":
            opt_ex += " --model_export vllm_adopted_safetensors"
        opt_basic = f"--model_dir {model_dir} --output_dir output_dir --quant_scheme w_int4_per_group_sym "
    elif cfg == "W_UInt4_A_Bfloat16_Per_Group_Asym":
        opt_basic = f"--model_dir {model_dir} --output_dir output_dir --quant_scheme w_uint4_a_bfloat16_per_group_asym "
    elif cfg == "W_UInt4_Per_Group_Asym_with_AWQ":
        opt_basic = f"--model_dir {model_dir} --output_dir output_dir --quant_scheme w_uint4_per_group_asym --quant_algo awq --dataset pileval_for_awq_benchmark --seq_len 512"
    elif cfg == "W_UInt4_Per_Group_Asym_with_GPTQ":
        opt_basic = f"--model_dir {model_dir} --output_dir output_dir --quant_scheme w_uint4_per_group_asym --quant_algo gptq --dataset wikitext_for_gptq_benchmark --seq_len 2048"
    elif cfg == "W_UInt4_Per_Group_Asym_with_SmoothQuant":
        opt_basic = f"--model_dir {model_dir} --output_dir output_dir --quant_scheme w_uint4_per_group_asym --quant_algo smoothquant --dataset pileval_for_awq_benchmark --seq_len 512"
    return opt_basic, opt_ex


def run_subprocess(model_name, cfg):
    model_dir = model_datapath[model_name]
    opt_basic, opt_ex = get_options(cfg, model_dir, model_name)
    log_file = os.path.join(log_dir, f"{model_name}_{cfg}.log")
    cmd = f"python quantize_quark.py {opt_basic} {opt_ex} --num_calib_data 4 --num_eval_data 8 --device cpu "
    print(cmd)
    start_time = time.time()
    with open(log_file, "w") as log:
        log.write(cmd + "\n")
        subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
        elapsed_time = time.time() - start_time
        log.write(f"Time elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}\n")


if __name__ == "__main__":
    main()
