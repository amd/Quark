#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from dataclasses import dataclass, field

from quark.torch import LLMTemplate


@dataclass
class QuarkQuantArguments:
    """
    QuarkQuantArguments: args to perfrom Quark PTQ.
    For any update, please refer to torch/language_modeling/llm_ptq/quantize_quark.py
    take quantize_quark.py as templete.
    """

    skip_qlora_train: bool = field(default=False, metadata={"help": "Whether to skip training."})
    cache_dir: str | None = field(default=None, metadata={"help": "The path to cache dataset."})
    device_map: str = field(
        default="auto", metadata={"help": "Device for running the quantizer", "choices": ["cuda", "cpu"]}
    )
    multi_device: bool = field(
        default=False,
        metadata={
            "help": (
                "we allow you to use this mode to run a model quantization that exceeds the size of your gpu memory if you use args.multi_gpu and still run into OOM "
                "now it only supports thr common quantization without algorithms, please note that this can lead to very slow quantization."
            )
        },
    )
    skip_quantization: bool = field(default=False, metadata={"help": ("Whether skip quantization.")})
    calib_dataset: str = field(
        default="pileval",
        metadata={
            "help": "Dataset for calibration",
            "choices": [
                "pileval",
                "wikitext",
                "cnn_dailymail",
                "pileval_for_awq_benchmark",
                "wikitext_for_gptq_benchmark",
                "HuggingFaceH4/ultrachat_200k",
                "ScienceQA",
            ],
        },
    )
    seq_len: int = field(default=512, metadata={"help": "Sequence length of data"})
    batch_size: int = field(default=1, metadata={"help": "Batch size for calibration."})
    num_calib_data: int = field(default=128, metadata={"help": "Number of samples for calibration."})
    quant_scheme: str = field(
        default="mxfp4",
        metadata={
            "help": "Supported quant_scheme in the script. If there is no suitable quantization strategy among the options, users can customize the quantization configuration according to their own needs.",
            "choices": LLMTemplate.get_supported_schemes(),
        },
    )
    kv_cache_dtype: str | None = field(default=None, metadata={"help": "KV Cache dtype.", "choices": ["fp8", None]})
    min_kv_scale: float = field(default=0.0, metadata={"help": "Minimum value of KV Cache scale."})
    attention_dtype: str | None = field(
        default=None, metadata={"help": "The dtype of attention quantization.", "choices": ["fp8", None]}
    )
    quant_algo: str | None = field(
        default=None,
        metadata={
            "help": "Algorithms used for quantization.",
            "choices": ["awq", "gptq", "smoothquant", "rotation", None],
        },
    )
    exclude_layers: str | None = field(
        default=None, metadata={"help": "List of layers to exclude from quantization. Default depends on model type."}
    )
    model_export: str | None = field(
        default=None,
        metadata={
            "help": "Model export format",
            "choices": [
                None,
                "hf_format",
            ],
        },
    )
    custom_mode: str = field(
        default="quark", metadata={"help": "Model export format", "choices": ["quark", "awq", "fp8"]}
    )
    export_weight_format: str = field(
        default="real_quantized",
        metadata={
            "help": "Whether to export weights compressed or uncompressed",
            "choices": ["fake_quantized", "real_quantized"],
        },
    )
    pack_method: str = field(
        default="reorder", metadata={"help": "Pack method for awq_export.", "choices": ["order", "reorder"]}
    )
    quant_out_dir: str = field(default="exported_model", metadata={"help": "Path for quantized model."})
    skip_evaluation: bool = field(default=False, metadata={"help": "Whether skip evaluation after quantization."})
    use_ppl_eval_model: bool = field(default=False)
    save_metrics_to_csv: bool = field(default=False)
    metrics_output_dir: str = field(default="metrics_output_dir")
    use_ppl_eval_for_kv_cache: bool = field(default=False)
    ppl_eval_for_kv_cache_context_size: int = field(
        default=1024, metadata={"help": "Context size used in PPL evaluation for KV cache."}
    )
    ppl_eval_for_kv_cache_sample_size: int = field(
        default=512, metadata={"help": "Sample size used in PPL evaluation for KV cache."}
    )
    ppl_eval_for_kv_cache_patch_size: int | None = field(
        default=None, metadata={"help": "Patch size used in PPL evaluation for KV cache."}
    )
    eval_batch_size: int = field(default=1, metadata={"help": "Batch size used for evaluation."})
    max_eval_batch_size: int = field(
        default=64, metadata={"help": "Maximal batch size to try with `--batch_size auto`."}
    )
    num_eval_data: int = field(
        default=-1,
        metadata={
            "help": "Number of samples for evaluation. The default value is -1, which means the entire dataset is used for evaluation."
        },
    )
    num_fewshot: int | None = field(default=None, metadata={"help": "Number of examples in few-shot context."})  # NOTE
    use_mlperf_rouge: bool = field(default=False)
    eval_data_dir: str | None = field(default=None, metadata={"help": "Dataset for evaluation."})
    evaluation_dataset: str = field(default="wikitext_gpt_oss_120b", metadata={"help": "Dataset for evaluation"})
    tasks: str | None = field(default=None)
    export_path: str | None = field(default=None)
    import_model_dir: str | None = field(default=None)
