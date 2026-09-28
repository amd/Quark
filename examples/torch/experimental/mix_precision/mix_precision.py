import argparse
import logging

from quark.experimental.torch.mix_precision import MixPrecisionConfig, MixPrecisionQuantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix-Precision auto-search with vLLM.")
    parser.add_argument("--model_dir", type=str, required=True, help="Hugging Face model ID or local directory.")
    parser.add_argument(
        "--granularity",
        type=str,
        default="module",
        help="Search granularity. Module is currently supported; per-layer support is planned.",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="mi355",
        choices=["mi300", "mi325", "mi355"],
        help="Target hardware.",
    )
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "ppl"],
        help="Evaluation metric: GSM8K accuracy (higher is better) or Wikitext-2 PPL (lower is better).",
    )
    parser.add_argument(
        "--gsm8k_num_samples",
        type=int,
        default=1319,
        help="[GSM8K only] Number of questions to evaluate (maximum 1319).",
    )
    parser.add_argument(
        "--gsm8k_max_new_tokens",
        type=int,
        default=256,
        help="[GSM8K only] Maximum generated tokens per question.",
    )
    parser.add_argument("--num_calib_samples", type=int, default=64, help="Number of calibration samples.")
    parser.add_argument("--calib_seq_len", type=int, default=2048, help="Calibration sequence length.")
    parser.add_argument(
        "--eval_threshold",
        type=float,
        default=1.02,
        help="Maximum accuracy degradation ratio (1.02 allows a 2%% relative degradation).",
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="Stop once the adjacent roofline valid/invalid accuracy frontier is found.",
    )
    parser.add_argument("--export_best_model", action="store_true", help="Export the best-config model.")
    parser.add_argument(
        "--file2file_quantization",
        action="store_true",
        help=(
            "Restrict search to calibration-free configs and export the best config with file-to-file quantization. "
            "Calibration-dependent configs such as mxfp4_fp8 (W4A8) are removed with a warning; the source "
            "checkpoint must provide safetensors weights."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Export directory (default: <model_name>-bestquantconfig).",
    )
    parser.add_argument("--min_kv_scale", type=float, default=0.0, help="Minimum KV-cache scale.")
    parser.add_argument(
        "--max_configs",
        type=int,
        default=None,
        help="Maximum candidate configs to evaluate after full-space Roofline scoring.",
    )
    parser.add_argument(
        "--search_configs",
        type=str,
        nargs="+",
        default=None,
        metavar="MODE",
        help=(
            "Quantization modes to search (default: all modes supported by --hardware). "
            "Example: --search_configs native ptpc_fp8 mxfp4"
        ),
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=None,
        help="Layer-name patterns to exclude. The API defaults are used when omitted.",
    )
    parser.add_argument(
        "--skip-baseline-eval",
        action="store_true",
        dest="skip_baseline_eval",
        help="Skip baseline evaluation before config search.",
    )
    parser.add_argument(
        "--kv-cache-quant",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="kv_cache_quant",
        help="Include KV-cache quantization in the search. Default: disabled.",
    )
    args, runtime_args = parser.parse_known_args()
    args.runtime_args = runtime_args
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    config = MixPrecisionConfig(
        granularity=args.granularity,
        hardware=args.hardware,
        eval_metrics=[args.eval_metric],
        eval_threshold=args.eval_threshold,
        eval_num_samples=args.gsm8k_num_samples,
        eval_max_new_tokens=args.gsm8k_max_new_tokens,
        skip_baseline_eval=args.skip_baseline_eval,
        num_calib_samples=args.num_calib_samples,
        calib_seq_len=args.calib_seq_len,
        min_kv_scale=args.min_kv_scale,
        exclude_patterns=args.exclude,
        max_configs=args.max_configs,
        early_stop=args.early_stop,
        search_modes=args.search_configs,
        kv_cache_quant=args.kv_cache_quant,
        file2file_quantization=args.file2file_quantization,
    )
    quantizer = MixPrecisionQuantizer(config)
    quantizer.search(model_path=args.model_dir, runtime_args=args.runtime_args)

    if args.export_best_model:
        quantizer.export_best(args.output_dir, file2file_quantization=args.file2file_quantization)


if __name__ == "__main__":
    main()
