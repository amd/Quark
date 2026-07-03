#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Gracefully handle imports, as user may not be aware of dependencies required.
try:
    import argparse
    import os
    from pathlib import Path

    import torch
    from datasets import load_dataset
    from transformers import AutoProcessor  # type: ignore[attr-defined]
    from transformers.processing_utils import ProcessorMixin

except ImportError as e:  # pragma: no cover
    print(
        f"AMD Quark CLI dependencies need to be installed with `pip install amd-quark[cli]` or `pip install -r requirements-cli.txt`: {e}."
    )
    exit(1)

# Gracefully handle imports, as user may not have Quark installed.
try:
    from quark.common.utils.log import ScreenLogger
    from quark.contrib.llm_eval import ppl_eval, ppl_eval_with_synthetic_dataset
    from quark.experimental.cli import base_cli
    from quark.torch import (
        LLMTemplate,
        ModelQuantizer,
        export_safetensors,
    )
    from quark.torch.utils.llm import (
        get_calib_dataloader,
        get_model,
        get_tokenizer,
        preprocess_for_quantization,
    )
except ImportError as e:
    print(f"AMD Quark needs to be installed with e.g. `pip3 install amd-quark`: {e}.")
    exit(1)

logger = ScreenLogger(__name__)


class TorchLLM_PTQ_CLI(base_cli.BaseQuarkCLICommand):
    """
    This class is the Quark CLI torch-llm-ptq subcommand.

    Design
    ------
    This class was created initially as a superset of all PyTorch LLM PTQ examples from the Quark Examples package.
    As such there is a very large number parser arguments and lot of package imports/requirements.
    These will be reduced to a minimal set of most-useful arguments, which will make the subcommand's help output easier to follow.
    """

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser) -> None:
        # Detect device in a consistent way to improve on example scripts defaulting to GPU.
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Argument for model
        parser.add_argument(
            "--model_dir",
            help="Specify where the HuggingFace model is. This example support Llama, OPT models",
            required=True,
        )
        parser.add_argument(
            "--device", help="Device for running the quantizer", default=device, choices=["cuda", "cpu"]
        )
        parser.add_argument(
            "--multi_device",
            action="store_true",
            help="we allow you to use this mode to run a model quantization that exceeds the size of your gpu memory if you use multi_gpu and still run into OOM "
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
        parser.add_argument("--seq_len", type=int, help="Sequence length of data", default=512)
        parser.add_argument("--batch_size", help="Batch size for calibration.", type=int, default=1)
        parser.add_argument("--num_calib_data", help="Number of samples for calibration.", type=int, default=512)

        # Argument for quantization
        parser.add_argument(
            "--quant_scheme",
            help="Supported quantization scheme name. Must be the built-in quantization scheme supported by LLMTemplate for the model type. "
            "For the built-in schemes and their detailed configuration information, see docs/source/pytorch/user_guide_config_for_llm.rst "
            "(Supported Quantization Schemes section). ",
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
            "Example: --quant_scheme int4_wo_128 --layer_quant_scheme lm_head int4_wo_32 "
            "(results in lm_head using int4_wo_32 while other layers use int4_wo_128). "
            "Supports wildcards: --layer_quant_scheme '*down_proj' fp8",
        )
        parser.add_argument(
            "--kv_cache_dtype",
            "--kv_cache_quant_scheme",
            help="KV Cache dtype.",
            default=None,
            choices=[
                "fp8",
                "fp8_dynamic",
                "int8_per_tensor_static",
                "int8_per_tensor_dynamic",
                "int8_per_token",
                "mxfp8",
                "fp6e2m3_per_group",
                "mxfp6_e2m3",
                "fp6e3m2_per_group",
                "mxfp6_e3m2",
                "fp4_per_group",
                "mxfp4",
                None,
            ],
        )
        parser.add_argument(
            "--quant_algo",
            default=None,
            type=lambda s: s.split(","),
            metavar="alg1,alg2",
            help="Comma-separated list of algorithms. Options include awq, gptq, smoothquant, autosmoothquant, rotation, quarot.",
        )
        parser.add_argument(
            "--exclude_layers",
            type=str,
            nargs="*",  # Allows to pass a list of strings
            default=None,  # Default is None to allow model-specific layer exclusion
            help='List of layers to exclude from quantization. Default depends on model type. Usage: `--exclude_layers "*down_proj*" "*31.fc*" "*k_proj"`. To avoid excluding layers at all, simply use `--exclude_layers` without any argument.',
        )

        parser.add_argument("--output_dir", default="exported_model")

        # Argument for evaluation
        parser.add_argument("--skip_evaluation", action="store_true")
        parser.add_argument(
            "--evaluation_dataset",
            help="Dataset for evaluation",
            default="wikitext",
            choices=["wikitext", "wikitext_gpt_oss_120b"],
        )

        parser.add_argument(
            "--no_trust_remote_code",
            action="store_true",
            default=False,
            help="Disable execution of custom model code from the Hub (safer, recommended if unsure).",
        )

    def run(self) -> None:
        """
        Execute the torch-llm-ptq subcommand.
        Relevant command-line arguments are used here.
        Subroutines and utilities invoked here are found under the `torch_llm/` subdirectory.
        """
        args = self.args

        # Set CWD - Current Working Directory (some of the files here depend on relative paths).
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        dname += "/torch_llm/llm_ptq/"
        os.makedirs(dname, exist_ok=True)
        os.chdir(dname)
        logger.info(f"Working directory is now={dname}")

        # 1. Define original model
        logger.info("Loading model ...")

        # We currently use CPU memory to load large models because GPU memory is typically smaller.
        # The model will be dispatched to different GPUs based on the total number of GPUs specified by torchrun --nproc-per-node.
        # TODO:
        # The current method results in high CPU memory consumption due to multiple copies of the same model.
        # We plan to address this in the future by implementing a more efficient way to dispatch the model to devices.
        device = args.device

        # Convert no_trust_remote_code to trust_remote_code
        trust_remote_code = not args.no_trust_remote_code

        model, model_dtype = get_model(
            args.model_dir, "auto", device, True, args.multi_device, trust_remote_code=trust_remote_code
        )
        preprocess_for_quantization(model)

        model_type = model.config.model_type if hasattr(model.config, "model_type") else model.config.architectures[0]

        tokenizer = get_tokenizer(args.model_dir, max_seq_len=args.seq_len, model_type=model_type)
        multimodal = True if model_type in ["mllama", "llama4"] else False
        if multimodal:
            processor: ProcessorMixin = AutoProcessor.from_pretrained(args.model_dir)  # type: ignore[no-untyped-call]
            export_dir = Path(args.output_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            processor.save_pretrained(args.output_dir)

        # 3. Define calibration dataloader(still need this step for weight only and dynamic quantization in Quark for current version.)
        logger.info("Loading dataset ...")
        # When the model is small, accelerate will place it on the last device
        main_device = model.device if hasattr(model, "device") else args.device
        calib_dataloader = get_calib_dataloader(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_calib_data=args.num_calib_data,
            seqlen=args.seq_len,
            device=main_device,
        )

        # 4. Quantization
        # 4-1. Get LLMTemplate used for quantization configuration

        template = LLMTemplate.get(model_type)

        # 4-2. Build layer_config if --layer_quant_scheme is provided
        layer_config = {}
        if args.layer_quant_scheme is not None:
            for layer_info in args.layer_quant_scheme:
                if len(layer_info) != 2:
                    raise ValueError(
                        f"Invalid --layer_quant_scheme argument: {layer_info}. "
                        f"Expected exactly 2 values (PATTERN, QUANT_SCHEME), but got {len(layer_info)}."
                    )
                layer_name = layer_info[0]
                layer_scheme = layer_info[1]
                layer_config[layer_name] = layer_scheme

        quant_config = template.get_config(
            scheme=args.quant_scheme,
            algorithm=args.quant_algo,
            kv_cache_scheme=args.kv_cache_dtype,
            layer_config=layer_config,
            exclude_layers=args.exclude_layers,
        )

        # 4-3. In-place replacement of model modules with quantized versions.
        quantizer = ModelQuantizer(quant_config, args.multi_device)
        model = quantizer.quantize_model(model, calib_dataloader)  # type: ignore[var-annotated]
        args.exclude_layers = quantizer.config.exclude

        # 5. (Optional) Model freeze
        # If user want to export the quantized model, please freeze the quantized model first
        model = quantizer.freeze(model)

        # 6. (Optional) Model exporting
        logger.info("Exporting hugging face format safetensors...")
        with torch.no_grad():
            export_safetensors(
                model=model,
                output_dir=args.output_dir,
                custom_mode="quark",
                weight_format="real_quantized",
                pack_method="reorder",
            )
            if not multimodal:
                tokenizer.save_pretrained(args.output_dir)  # type: ignore[attr-defined]

        if not args.skip_evaluation:
            logger.info("Evaluating ...")
            # Prepare test data for perplexity evaluation
            if args.evaluation_dataset == "wikitext":
                testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
                testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")  # type: ignore[operator]
                ppl = ppl_eval(model, testenc, main_device)
            elif (
                args.evaluation_dataset == "wikitext_gpt_oss_20b" or args.evaluation_dataset == "wikitext_gpt_oss_120b"
            ):
                ppl = ppl_eval_with_synthetic_dataset(tokenizer, model, main_device, args.evaluation_dataset)
            else:
                raise ValueError(f"Unsupported evaluation dataset: {args.evaluation_dataset}")
            logger.info(f"Perplexity: {ppl.item()}")
