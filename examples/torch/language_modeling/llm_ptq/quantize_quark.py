#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import torch
from transformers import AutoProcessor

from quark.torch import (
    LLMTemplate,
    ModelQuantizer,
    export_gguf,
    export_onnx,
    export_safetensors,
    import_model_from_safetensors,
    load_params,
    save_params,
)
from quark.torch.export.api import _move_quantizer_to_dict
from quark.torch.quantization.config.config import load_quant_algo_config_from_file
from quark.torch.utils import TPDeviceManager

# TODO: Using sys.path.append is bad practice.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quark.contrib.llm_eval import eval_model
from quark.torch.utils.llm import (
    check_compatibility_before_quantization,
    get_calib_dataloader,
    get_model,
    get_tokenizer,
    prepare_for_moe_quant,
    revert_model_patching,
)

# The code below demonstrates how to register custom model templates and
# quantization schemes. If you need to add support for a new model architecture
# or define custom quantization configurations, uncomment and modify this section.
#
# To use:
#   1. Uncomment the code below
#   2. Modify the templates and/or schemes to match your model's architecture and/or quantization scheme
#   3. Run quantize_quark.py with your custom --quant_scheme name if new quantization schemes are registered
#

# from quark.torch.quantization.config.config import (
#     Int8PerTensorSpec,
#     QLayerConfig,
# )

# # --- Custom Model Templates ---
# # Define templates for model architectures not in the built-in list.
# # Model: internlm/internlm2-chat-7b
# internlm2_template = LLMTemplate(
#     model_type="internlm2",
#     kv_layers_name=["*wqkv"],
#     q_layer_name="*wqkv",
#     exclude_layers_name=["lm_head"],
# )
# LLMTemplate.register_template(internlm2_template)
# print(f"[INFO]: Registered template '{internlm2_template.model_type}'")

# # --- Custom Quantization Schemes ---
# # Define custom quantization schemes using Quark's public QuantizationSpec classes.
# # These schemes can then be used via --quant_scheme <scheme_name>.
# # INT8 weight-only quantization
# int8_wo_scheme = QLayerConfig(weight=Int8PerTensorSpec().to_quantization_spec())
# LLMTemplate.register_scheme("int8_wo", config=int8_wo_scheme)
# print(f"[INFO]: Registered quantization scheme 'int8_wo'")


def _get_hf_model_config(model_dir: str) -> dict:
    """Read config.json from the model directory without loading the model."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        return json.load(f)


def _build_quant_config(args: argparse.Namespace, model_config_type: str):
    """Build quant_config from args and model_config_type (shared by normal and file-to-file paths)."""
    if model_config_type not in LLMTemplate.list_available():
        error_msg = (
            f"\n[ERROR]: Model type '{model_config_type}' is not supported.\n\n"
            f"Available templates: {LLMTemplate.list_available()}\n\n"
            f"To add support for this model, uncomment and modify the 'Custom Model Templates'\n"
            f"section at the top of this file to register a template for '{model_config_type}'.\n"
        )
        raise ValueError(error_msg)
    template = LLMTemplate.get(model_config_type)

    # Load algorithm configs from files if provided
    algo_configs = {}
    if args.quant_algo_config_file is not None:
        for algo_name, algo_config_file in args.quant_algo_config_file:
            algo_configs[algo_name] = load_quant_algo_config_from_file(algo_config_file)
            print(f"[INFO]: Loaded algorithm configuration for {algo_name} from {algo_config_file}.")

    # Build layer_config if --layer_quant_scheme is provided
    layer_config = {}
    if args.layer_quant_scheme is not None:
        for layer_info in args.layer_quant_scheme:
            layer_name = layer_info[0]
            layer_scheme = layer_info[1]
            layer_config[layer_name] = layer_scheme

    quant_config = template.get_config(
        scheme=args.quant_scheme,
        algorithm=args.quant_algo,
        kv_cache_scheme=args.kv_cache_dtype,
        min_kv_scale=args.min_kv_scale,
        layer_config=layer_config,
        attention_scheme=args.attention_dtype,
        exclude_layers=args.exclude_layers,
        algo_configs=algo_configs if algo_configs else None,
    )
    return quant_config


def main(args: argparse.Namespace) -> None:
    # File-to-file quantization mode: bypass model loading, calibration and quantization,
    # directly quantize safetensors files shard-by-shard and export.
    if args.file2file_quantization:
        print("\n[INFO]: File-to-file quantization mode enabled.")
        hf_model_config = _get_hf_model_config(args.model_dir)
        model_config_type = hf_model_config.get("model_type", hf_model_config.get("architectures", [None])[0])
        quant_config = _build_quant_config(args, model_config_type)

        print("\n[INFO]: Quantizing safetensors shards directly (file-to-file) ...")
        quantizer = ModelQuantizer(quant_config)
        quantizer.direct_quantize_checkpoint(
            pretrained_model_path=args.model_dir,
            save_path=args.output_dir,
        )
        print(f"[INFO]: File-to-file quantization output saved to {args.output_dir}")
        return

    # 1. Define original model
    print("\n[INFO]: Loading model ...")

    # We currently use CPU memory to load large models because GPU memory is typically smaller.
    # The model will be dispatched to different GPUs based on the total number of GPUs specified by torchrun --nproc-per-node.
    # TODO:
    # The current method results in high CPU memory consumption due to multiple copies of the same model.
    # We plan to address this in the future by implementing a more efficient way to dispatch the model to devices.
    if args.use_tp:
        device = "cpu"
    else:
        device = args.device

    model, model_dtype = get_model(
        args.model_dir,
        args.data_type,
        device,
        args.multi_gpu,
        args.multi_device,
        args.model_attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    prepare_for_moe_quant(model)

    # Check model compatibility with current Transformers version
    print("\n[INFO]: Checking model compatibility ...")
    check_compatibility_before_quantization(model, raise_on_error=False)

    model_type = model.config.model_type if hasattr(model.config, "model_type") else model.config.architectures[0]
    tokenizer = get_tokenizer(
        args.model_dir, max_seq_len=args.seq_len, model_type=model_type, trust_remote_code=args.trust_remote_code
    )

    multimodal = True if model_type in ["mllama", "llama4", "gemma3", "qwen3_vl_moe", "deepseek_vl_v2"] else False
    if multimodal:
        processor = AutoProcessor.from_pretrained(args.model_dir)
        if args.model_export is not None:
            export_dir = Path(args.output_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            processor.save_pretrained(args.output_dir)

    if args.use_tp:
        TPDeviceManager.tp_mesh_init()

    # 2. (Optional) Reload quantized model
    if args.params_load:
        print("\nRestore quantized model from json and safetensors file ...")
        model = load_params(model, json_path=args.json_path, safetensors_path=args.safetensors_path)
        args.skip_quantization = True
    elif args.model_reload:
        print("\nRestore quantized model from hf_format safetensors file ...")

        # TODO: This should be moved to quark namespace.
        # Revert model transformations that were useful only for quantization (Transformers-specific).
        revert_model_patching(model)

        model = import_model_from_safetensors(model, model_dir=args.import_model_dir, multi_device=args.multi_device)
        args.skip_quantization = True

    if args.use_tp:
        if TPDeviceManager._tp_mesh is not None:
            _move_quantizer_to_dict(model.model)

            device = TPDeviceManager._device
            tp_mesh = TPDeviceManager._tp_mesh

            model.tensor_parallel(tp_mesh)
            model.to(device)
        else:
            warnings.warn(
                "Quark tensor parallelism is not initialized properly. Please check the torchrun settings.", UserWarning
            )
            return

    # 3. Define calibration dataloader(still need this step for weight only and dynamic quantization in Quark for current version.)
    print("\n[INFO]: Loading dataset ...")
    # When the model is small, accelerate will place it on the last device
    main_device = model.device if args.multi_gpu or args.multi_device else args.device
    calib_dataloader = get_calib_dataloader(
        dataset_name=args.dataset,
        processor=processor if multimodal else None,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_calib_data=args.num_calib_data,
        seqlen=args.seq_len,
        device=main_device,
    )

    # 4. Quantization
    if not args.skip_quantization:
        model_config_type = (
            model.config.model_type if hasattr(model.config, "model_type") else model.config.architectures[0]
        )

        quant_config = _build_quant_config(args, model_config_type)

        if getattr(args, "kv_cache_post_rope", False):
            if hasattr(quant_config, "kv_cache_post_rope"):
                quant_config.kv_cache_post_rope = True
            else:
                warnings.warn(
                    "--kv_cache_post_rope specified but quant_config has no 'kv_cache_post_rope' field; flag ignored.",
                    RuntimeWarning,
                )

        # In-place replacement of model modules with quantized versions
        quantizer = ModelQuantizer(quant_config, args.multi_device)
        model = quantizer.quantize_model(model, calib_dataloader)
        args.exclude_layers = quantizer.config.exclude

        # After quantization, freeze models - moving from soft weights that are quantized on the fly
        # to e.g. `QuantLinear.weight` actually holding the fake quantized weights.
        model = quantizer.freeze(model)

        # TODO: This should be moved to quark namespace.
        # Optionally, revert model transformations that were useful only for quantization (Transformers-specific).
        revert_model_patching(model)

    if args.model_export is not None:
        if args.custom_mode != "quark" and args.export_weight_format == "fake_quantized":
            raise ValueError("Exporting with 'fake_quantized' only supports custom_mode=quark")

        # Export option 1: hugging-face safetensors format
        if "hf_format" in args.model_export:
            print("\n[INFO]: Exporting hugging face format safetensors...")
            with torch.no_grad():
                export_safetensors(
                    model=model,
                    output_dir=args.output_dir,
                    custom_mode=args.custom_mode,
                    weight_format=args.export_weight_format,
                    pack_method=args.pack_method,
                )
                if not multimodal:
                    tokenizer.save_pretrained(args.output_dir)
        # Export option 2: onnx
        if "onnx" in args.model_export:
            print("\n[INFO]: Exporting onnx graph...")
            with torch.inference_mode():
                batch_iter = iter(calib_dataloader)
                input_args = next(batch_iter)
                if "uint4" in args.quant_scheme or "int4" in args.quant_scheme:
                    uint4_int4_flag = True
                else:
                    uint4_int4_flag = False

                export_onnx(
                    model=model, output_dir=args.output_dir, input_args=input_args, uint4_int4_flag=uint4_int4_flag
                )
        # Export option 3: gguf
        if "gguf" in args.model_export:
            print("\n[INFO]: Exporting gguf model...")
            with torch.inference_mode():
                export_gguf(model, output_dir=args.output_dir, model_type=model_type, tokenizer_path=args.model_dir)

    if args.torch_compile:
        print("\n[INFO]: Calling PyTorch 2 torch.compile...")
        # Note: The model after torch.compile may not be able to export to other format
        model = torch.compile(model)

    if args.params_save:
        save_params(model, model_type=model_type, export_dir=args.save_dir)

    if not args.skip_evaluation:
        print("\n[INFO]: Evaluating ...")
        args.use_ppl_eval_model = True
        eval_model(
            args,
            model,
            main_device,
            save_metrics_to_csv=args.save_metrics_to_csv,
            output_dir=args.metrics_output_dir,
            multimodal=multimodal,
        )

    if args.use_tp:
        TPDeviceManager.tp_cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # Argument for model
    parser.add_argument(
        "--model_dir",
        help="Specify where the HuggingFace model is. This example support Llama, OPT models",
        required=True,
    )
    parser.add_argument("--device", help="Device for running the quantizer", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument(
        "--model_attn_implementation",
        help="The attention implementation to use in the model",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--multi_device",
        action="store_true",
        help="we allow you to use this mode to run a model quantization that exceeds the size of your gpu memory if you use args.multi_gpu and still run into OOM "
        "now it only supports thr common quantization without algorithms, please note that this can lead to very slow quantization.",
    )

    # Argument for calibration dataset
    parser.add_argument(
        "--dataset",
        help="Dataset for calibration",
        default="pileval",
        choices=[
            "pileval",
            "wikitext",
            "cnn_dailymail",
            "pileval_for_awq_benchmark",
            "wikitext_for_gptq_benchmark",
            "HuggingFaceH4/ultrachat_200k",
            "ScienceQA",
        ],
    )
    parser.add_argument(
        "--data_type", help="Datatype of the model", default="auto", choices=["auto", "float16", "bfloat16", "float32"]
    )
    parser.add_argument("--seq_len", type=int, help="Sequence length of data", default=512)
    parser.add_argument("--batch_size", help="Batch size for calibration.", type=int, default=1)
    parser.add_argument("--num_calib_data", help="Number of samples for calibration.", type=int, default=512)

    # Argument for quantization
    parser.add_argument("--skip_quantization", action="store_true")
    parser.add_argument(
        "--file2file_quantization",
        action="store_true",
        help="Enable file-to-file quantization mode. Quantizes safetensors shards directly without loading the full model into memory. "
        "Bypasses model loading, calibration, and standard quantization flow. Requires --model_export hf_format.",
    )

    parser.add_argument(
        "--quant_scheme",
        help="Quantization scheme to use. Supported schemes: all built-in schemes and custom schemes registered."
        "For the built-in schemes and their detailed configuration, see https://quark.docs.amd.com/latest/pytorch/user_guide_config_for_llm.html. "
        "To register custom schemes, please uncomment and modify the 'Custom Quantization Schemes' section at the top of this file.",
        choices=LLMTemplate.get_supported_schemes(),
        default=None,
        type=str,
    )

    parser.add_argument(
        "--layer_quant_scheme",
        action="append",
        nargs=2,
        metavar=("PATTERN", "QUANT_SCHEME"),
        help="Directly specify a quantization scheme for layers matching the given pattern. "
        "Can be repeated for multiple patterns. "
        "Example: --quant_scheme int4_wo_128 --layer_quant_scheme lm_head int8 "
        "(results in lm_head using int8 while other layers use int4_wo_128). "
        "Supports wildcards: --layer_quant_scheme '*down_proj' fp8",
    )

    parser.add_argument(
        "--kv_cache_dtype", "--kv_cache_quant_scheme", help="KV Cache dtype.", default=None, choices=["fp8", None]
    )

    parser.add_argument("--min_kv_scale", help="Minimum value of KV Cache scale.", type=float, default=0.0)
    parser.add_argument(
        "--kv_cache_post_rope",
        action="store_true",
        help="If set, quantize KV cache after RoPE (inside cache) instead of at k_proj/v_proj outputs.",
    )
    parser.add_argument(
        "--attention_dtype", help="The dtype of attention quantization.", type=str, default=None, choices=["fp8"]
    )
    parser.add_argument(
        "--quant_algo",
        default=None,
        type=lambda s: s.split(","),
        metavar="alg1,alg2",
        help="Comma-separated list of algorithms. Options include awq, gptq, smoothquant, rotation.",
    )
    parser.add_argument(
        "--quant_algo_config_file",
        action="append",
        nargs=2,
        metavar=("ALGO_NAME", "CONFIG_FILE"),
        help="Specify a configuration file for a specific quantization algorithm. "
        "Can be repeated for multiple algorithms. "
        "Example: --quant_algo_config_file awq ./awq_config.json --quant_algo_config_file gptq ./gptq_config.json "
        "(provides custom config files for AWQ and GPTQ algorithms).",
    )

    parser.add_argument(
        "--exclude_layers",
        type=str,
        nargs="*",  # Allows to pass a list of strings
        default=None,  # Default is None to allow model-specific layer exclusion
        help='List of layers to exclude from quantization. Default depends on model type. Usage: `--exclude_layers "*down_proj*" "*31.fc*" "*k_proj"`. To avoid excluding layers at all, simply use `--exclude_layers` without any argument.',
    )

    # Argument for reloading
    parser.add_argument("--model_reload", help="safetensors or pth model reload", action="store_true")
    parser.add_argument("--import_model_dir", help="directory of hf or quark model")
    parser.add_argument("--params_load", help="Model parameters load", action="store_true")
    parser.add_argument("--json_path", help="Specify the path of saved json file")
    parser.add_argument("--safetensors_path", help="Specify the path of saved safetensors file")

    # Argument for export
    parser.add_argument(
        "--model_export",
        help="Model export format",
        default=None,
        action="append",
        choices=[None, "onnx", "hf_format", "gguf"],
    )
    parser.add_argument(
        "--custom_mode",
        help="When selecting `--custom_mode awq` or `--custom_mode fp8`, this legacy argument allows to export FP8 and AWQ models in the custom format they were exported with with quark<1.0, with custom config saved in the config.json, and config checkpoint format (AWQ uses `qzeros`, `qweight`, transposed `scales`).",
        default="quark",
        type=str,
        choices=["quark", "awq", "fp8"],
    )
    parser.add_argument("--torch_compile", help="Model torch compile", action="store_true")
    parser.add_argument(
        "--pack_method", type=str, help="Pack method for awq_export", default="reorder", choices=["order", "reorder"]
    )
    parser.add_argument("--output_dir", default="exported_model")
    parser.add_argument(
        "--export_weight_format",
        type=str,
        help="Whether to export weights compressed or uncompressed",
        default="real_quantized",
        choices=["fake_quantized", "real_quantized"],
    )

    # Argument for saving
    parser.add_argument("--params_save", help="Model parameters save", action="store_true")
    parser.add_argument(
        "--save_dir",
        help="Directory to save model parameters as safetensors or pth, in the case when --params_save is used.",
        default="model_params",
    )

    # Argument for evaluation
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--use_ppl_eval_model", action="store_true")
    parser.add_argument("--save_metrics_to_csv", action="store_true")
    parser.add_argument("--metrics_output_dir", default="metrics_output_dir", help="Output path of csv with metrics.")
    parser.add_argument(
        "--tasks",
        default=None,
        type=str,
        metavar="task1,task2",
        help="Comma-separated list of task names or task groupings to evaluate on.",
    )
    parser.add_argument("--use_ppl_eval_for_kv_cache", action="store_true")
    parser.add_argument(
        "--ppl_eval_for_kv_cache_context_size",
        type=int,
        help="Context size used in PPL evaluation for KV cache.",
        default=1024,
    )
    parser.add_argument(
        "--ppl_eval_for_kv_cache_sample_size",
        type=int,
        help="Sample size used in PPL evaluation for KV cache.",
        default=512,
    )
    parser.add_argument(
        "--ppl_eval_for_kv_cache_patch_size",
        type=int,
        help="Patch size used in PPL evaluation for KV cache.",
        default=None,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Batch size used for evaluation. Acceptable values are 'auto', 'auto:N' or N, where N is a positive integer. Default is `1`.",
    )
    parser.add_argument(
        "--max_eval_batch_size",
        type=int,
        default=64,
        metavar="P",
        help="Maximal batch size to try with `--batch_size auto`.",
    )
    parser.add_argument(
        "--num_eval_data",
        help="Number of samples for evaluation. The default value is -1, which means the entire dataset is used for evaluation.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=None, metavar="N", help="Number of examples in few-shot context"
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Providing `--apply_chat_template` without an argument will apply the default chat template to the prompt.",
    )
    parser.add_argument("--use_mlperf_rouge", action="store_true")
    parser.add_argument("--eval_data_dir", help="Dataset for evaluation", type=str, default=None)
    parser.add_argument(
        "--use_tp", action="store_true", help="Enable tensor parallelism exclusively for model evaluation."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--trust_remote_code",
        action="store_true",
        dest="trust_remote_code",
        help="Enable execution of custom model code from the Hub (use only with repositories you fully trust).",
    )
    group.add_argument(
        "--no_trust_remote_code",
        action="store_false",
        dest="trust_remote_code",
        help="Disable execution of custom model code from the Hub (safer, recommended if unsure).",
    )
    parser.set_defaults(trust_remote_code=True)
    args = parser.parse_args()

    if args.layer_quant_scheme is not None:
        for layer_info in args.layer_quant_scheme:
            if len(layer_info) != 2:
                raise ValueError(
                    f"Invalid --layer_quant_scheme argument: {layer_info}. "
                    f"Expected exactly 2 values (PATTERN, QUANT_SCHEME), but got {len(layer_info)}."
                )

    if args.quant_algo_config_file is not None:
        for algo_config in args.quant_algo_config_file:
            if len(algo_config) != 2:
                raise ValueError(
                    f"Invalid --quant_algo_config_file argument: {algo_config}. "
                    f"Expected exactly 2 values (ALGO_NAME, CONFIG_FILE), but got {len(algo_config)}."
                )
            algo_name, config_file = algo_config
            if not os.path.isfile(config_file):
                raise ValueError(
                    f"Configuration file '{config_file}' for algorithm '{algo_name}' does not exist. "
                    f"Please provide a valid config file path."
                )

    main(args)
