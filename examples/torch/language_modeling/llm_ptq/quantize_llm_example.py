#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoProcessor

from quark.torch import LLMTemplate
from quark.torch.export.api import ModelExporter
from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
from quark.torch.quantization.api import ModelQuantizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_eval.evaluation import ppl_eval
from llm_utils.data_preparation import get_calib_dataloader
from llm_utils.model_preparation import get_model, get_model_type, get_tokenizer


def main(args: argparse.Namespace):
    print(f"Starting quantization of model: {args.model}")
    print(f"Quantization scheme: {args.scheme}")
    if args.algorithm:
        print(f"Algorithm: {args.algorithm}")
    if args.kv_cache_scheme:
        print(f"KV cache scheme: {args.kv_cache_scheme}")
    if args.attention_scheme:
        print(f"Attention scheme: {args.attention_scheme}")

    print("\n[INFO]: Loading model ...")
    model, _ = get_model(
        args.model,
        args.data_type,
        args.device,
        args.multi_gpu,
        args.multi_device,
        args.model_attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )

    model_type = get_model_type(model)

    print("\n[INFO]: Loading tokenizer ...")
    tokenizer = get_tokenizer(
        args.model, max_seq_len=args.seq_len, model_type=model_type, trust_remote_code=args.trust_remote_code
    )
    multimodal = True if model_type in ["mllama"] else False
    processor = None
    if multimodal:
        processor = AutoProcessor.from_pretrained(args.model)
        export_dir = Path(args.output_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(args.output_dir)

    # Get the template and create quantization config
    print("\n[INFO]: Creating quantization configuration ...")
    model_config_type = (
        model.config.model_type if hasattr(model.config, "model_type") else model.config.architectures[0]
    )
    template = LLMTemplate.get(model_config_type)

    quant_config = template.get_config(
        scheme=args.scheme,
        algorithm=args.algorithm,
        kv_cache_scheme=args.kv_cache_scheme,
        attention_scheme=args.attention_scheme,
    )

    print("\n[INFO]: Loading the calibration dataset...")
    calib_dataloader = None
    main_device = model.device if args.multi_gpu or args.multi_device else args.device
    if args.scheme == "fp8" or args.algorithm is not None:
        calib_dataloader = get_calib_dataloader(
            dataset_name=args.dataset,
            processor=processor if multimodal else None,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_calib_data=args.num_calib_data,
            seqlen=args.seq_len,
            device=main_device,
        )

    print("\n[INFO]: Quantizing model ...")
    quantizer = ModelQuantizer(quant_config, args.multi_device)
    quantized_model = quantizer.quantize_model(model, calib_dataloader)

    print("\n[INFO]: Exporting the quantized model...")
    NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized", pack_method="reorder")
    export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

    model_exporter = ModelExporter(config=export_config, export_dir=args.output_dir)
    model_exporter.export_safetensors_model(model=quantized_model, quant_config=quant_config)

    print("\n[INFO]: Evaluating the quantized model...")

    # Prepare test data for perplexity evaluation
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    ppl = ppl_eval(quantized_model, testenc, main_device)
    print(f"\n[INFO] Perplexity: {ppl.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize LLM models using Quark")
    # Argument for model
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    # Argument for device
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument(
        "--multi_device",
        action="store_true",
        help="we allow you to use this mode to run a model quantization that exceeds the size of your gpu memory if you use args.multi_gpu and still run into OOM "
        "now it only supports the common quantization without algorithms, please note that this can lead to very slow quantization.",
    )
    parser.add_argument("--model_attn_implementation", default=None, help="Model attention implementation")

    # Argument for calibration dataset
    parser.add_argument(
        "--dataset",
        help="Dataset for calibration",
        default="pileval",
        choices=[
            "pileval",
            "wikitext",
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
    parser.add_argument(
        "--scheme",
        default="fp8",
        choices=[
            "fp8",
            "int4_wo_32",
            "int4_wo_64",
            "int4_wo_128",
            "uint4_wo_32",
            "uint4_wo_64",
            "uint4_wo_128",
            "mxfp4",
            "mxfp6_e3m2",
            "mxfp6_e2m3",
        ],
        help="Quantization scheme",
    )
    parser.add_argument(
        "--algorithm", choices=["awq", "gptq", "smoothquant", "autosmoothquant"], help="Quantization algorithm"
    )
    parser.add_argument("--kv_cache_scheme", choices=["fp8"], help="KV cache quantization scheme")
    parser.add_argument("--attention_scheme", choices=["fp8"], help="Attention quantization scheme")
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

    main(args)
