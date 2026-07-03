import argparse
import logging
import os
from pathlib import Path

from transformers import AutoTokenizer

from quark.experimental.torch.llm.mix_precision import (
    ConfigEvalResult,
    ConfigSearcher,
    HardwareTarget,
    MixPrecisionConfig,
    ModuleSearchConfig,
    SearchGranularity,
    SearchResult,
    apply_quant_config,
    build_vllm_engine_kwargs,
    display_results,
    evaluate_gsm8k_offline,
    evaluate_ppl_offline,
    load_transformers_model,
)
from quark.experimental.torch.llm.mix_precision.utils import (
    DEFAULT_EXCLUDE_PATTERNS,
    create_qconfig_from_quant_config,
)
from quark.torch import export_safetensors
from quark.torch.quantization.api import ModelQuantizer
from quark.torch.utils.llm import preprocess_for_quantization


def _parse_enum(enum_cls, raw: str, arg_name: str):
    try:
        return enum_cls[raw.upper()]
    except KeyError as exc:
        valid = ", ".join([e.name.lower() for e in enum_cls])
        raise ValueError(f"invalid {arg_name}: {raw}, valid values: {valid}") from exc


def _build_module_search_config(args: argparse.Namespace) -> ModuleSearchConfig:
    layer_sensitivity = None
    if args.self_attn_only:
        layer_sensitivity = {"self_attn": 3}
    elif args.mlp_only:
        layer_sensitivity = {"mlp": 1}

    return ModuleSearchConfig(
        layer_sensitivity=layer_sensitivity,
        layer_modes=args.search_configs or None,
        kv_cache_modes=None if args.kv_cache_quant else ["native"],
    )


def run_search_vllm(args: argparse.Namespace, model_path: str, config: MixPrecisionConfig) -> SearchResult:
    """vLLM path: shared LLM() + QuarkFakeQuantWorker for fake-quant search."""
    logging.info("Loading meta model for QConfig generation: %s", model_path)
    model = load_transformers_model(model_path, torch_dtype="auto", device_map="meta")
    try:
        preprocess_for_quantization(model)
    except ValueError as exc:
        logging.warning("preprocess_for_quantization skipped: %s", exc)

    searcher = ConfigSearcher(
        search_config=config.get_search_config(),
        hardware=config.hardware,
    )
    all_configs = searcher.generate_sorted_configs()
    if args.max_configs is not None:
        all_configs = all_configs[: args.max_configs]
    total_configs = len(all_configs)
    logging.info("Generated %d candidate configs", total_configs)

    from vllm import LLM

    metric_name = config.eval_metrics[0] if config.eval_metrics else "gsm8k"

    def _run_eval(llm_obj):
        if metric_name == "ppl":
            r = evaluate_ppl_offline(
                llm_obj,
                seq_len=args.ppl_seq_len,
                max_chunks=args.ppl_max_chunks or None,
            )
            return float(r["ppl"])
        r = evaluate_gsm8k_offline(
            llm_obj,
            num_questions=min(args.gsm8k_num_samples, 1319),
            max_tokens=args.gsm8k_max_new_tokens,
        )
        return float(r["accuracy"])

    # For ppl: lower is better → valid when value <= baseline * threshold
    # For gsm8k: higher is better → valid when value >= baseline * (2 - threshold)
    def _is_valid(value, baseline):
        if baseline is None:
            return True
        if metric_name == "ppl":
            return value <= baseline * args.eval_threshold
        return value >= baseline * (2 - args.eval_threshold)

    quant_dataset = os.environ.get("QUANT_DATASET", "pileval")
    quant_calib_size = int(config.num_calib_samples)
    llm = None
    saved_quant_cfg = os.environ.pop("QUANT_CFG", None)

    try:
        llm = LLM(
            model=model_path,
            worker_cls="quark.experimental.plugin.fakequant_worker.QuarkFakeQuantWorker",
            **build_vllm_engine_kwargs(args.vllm_cli_args, LLM),
        )

        # Baseline evaluation
        baseline_metric_value = None
        baseline_metrics: dict = {}
        if not args.skip_baseline_eval:
            logging.info("Evaluating baseline (%s)...", metric_name)
            baseline_metric_value = _run_eval(llm)
            baseline_metrics = {metric_name: baseline_metric_value}
            logging.info("Baseline %s: %.4f", metric_name, baseline_metric_value)
        else:
            baseline_metrics = {metric_name: "SKIPPED"}
            logging.info("Skipping baseline eval")

        results_list = []
        best_result = None

        for rank, quant_config in enumerate(all_configs):
            logging.info("Config %d/%d: %s", rank + 1, total_configs, quant_config)
            requantized = False
            try:
                qconfig = create_qconfig_from_quant_config(
                    model=model,
                    config=quant_config,
                    exclude_patterns=config.exclude_patterns,
                    min_kv_scale=args.min_kv_scale,
                )
                qconfig_dict = qconfig.to_dict()

                llm.reset_prefix_cache()
                if hasattr(llm, "reset_mm_cache"):
                    llm.reset_mm_cache()

                requant_result = llm.collective_rpc(
                    "requantize_with_config",
                    args=(qconfig_dict, quant_dataset, quant_calib_size, int(config.calib_seq_len)),
                )
                requantized = True
                logging.info("  requantize result: %s", requant_result[0] if requant_result else None)

                metric_value = _run_eval(llm)
                metrics = {metric_name: metric_value}

                is_valid = _is_valid(metric_value, baseline_metric_value)
                relative_change = {
                    metric_name: (
                        (metric_value - baseline_metric_value) / baseline_metric_value if baseline_metric_value else 0
                    ),
                }
                res = ConfigEvalResult(
                    config=quant_config,
                    metrics=metrics,
                    relative_change=relative_change,
                    is_valid=is_valid,
                    rank=rank + 1,
                )
                results_list.append(res)
                logging.info("  %s: %.4f, valid: %s", metric_name, metric_value, is_valid)
                if is_valid and (best_result is None or res.rank > best_result.rank):
                    best_result = res
                if args.early_stop and not is_valid:
                    logging.info("Threshold exceeded, stopping search")
                    break
            except Exception as e:
                logging.error("Error evaluating config %s: %s", quant_config, e)
            finally:
                if requantized:
                    reset_result = llm.collective_rpc("reset_to_original")
                    if not all(reset_result):
                        raise RuntimeError(f"reset_to_original failed: {reset_result}")
                llm.reset_prefix_cache()
                if hasattr(llm, "reset_mm_cache"):
                    llm.reset_mm_cache()
    finally:
        if llm is not None:
            del llm
        if saved_quant_cfg is not None:
            os.environ["QUANT_CFG"] = saved_quant_cfg
        else:
            os.environ.pop("QUANT_CFG", None)

    result = SearchResult(
        best_config=best_result.config if best_result else None,
        all_results=results_list,
        baseline_metrics=baseline_metrics,
        total_configs_evaluated=len(results_list),
        total_configs_available=total_configs,
        search_time_seconds=0.0,
        granularity=config.granularity,
        hardware=config.hardware,
    )
    display_results(result, metric_name)

    if args.export_best_model and result.best_config is not None:
        output_dir = Path(args.output_dir or (model_path.split("/")[-1] + "-bestquantconfig"))
        logging.info("Exporting best config model to %s", output_dir)
        model_full = load_transformers_model(model_path, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        quantized_model = apply_quant_config(
            model=model_full,
            config=result.best_config,
            tokenizer=tokenizer,
            num_calib_samples=config.num_calib_samples,
            hardware=config.hardware,
            exclude_patterns=config.exclude_patterns,
            min_kv_scale=config.min_kv_scale,
        )
        quantized_model = ModelQuantizer.freeze(quantized_model)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_safetensors(quantized_model, str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logging.info("Model saved to %s", output_dir)

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Mix-Precision auto-search with vLLM + GSM8K.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to HuggingFace model.")
    parser.add_argument(
        "--granularity",
        type=str,
        default="module",
        help="Search granularity (module or block).",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="mi300",
        help="Target hardware (mi300, mi200, ...).",
    )
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "ppl"],
        help="Eval metric: gsm8k accuracy (higher better) or wikitext-2 ppl (lower better).",
    )
    parser.add_argument(
        "--ppl_seq_len",
        type=int,
        default=2048,
        help="[ppl only] PPL chunk length (wikitext-2). Only used when --eval_metric ppl.",
    )
    parser.add_argument(
        "--ppl_max_chunks",
        type=int,
        default=0,
        help="[ppl only] Cap on number of PPL chunks (0 = full wikitext-2 test split). "
        "Only used when --eval_metric ppl.",
    )
    parser.add_argument(
        "--gsm8k_num_samples",
        type=int,
        default=1319,
        help="[gsm8k only] Number of GSM8K questions to evaluate (max 1319). Only used when --eval_metric gsm8k.",
    )
    parser.add_argument(
        "--gsm8k_max_new_tokens",
        type=int,
        default=256,
        help="[gsm8k only] Max new tokens for GSM8K generation. Only used when --eval_metric gsm8k.",
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=64,
        help="Number of calibration samples.",
    )
    parser.add_argument(
        "--calib_seq_len",
        type=int,
        default=2048,
        help="Sequence length for calibration.",
    )
    parser.add_argument(
        "--eval_threshold",
        type=float,
        default=1.02,
        help="Max allowed accuracy drop ratio (e.g. 1.02 = allow 2%% drop from baseline).",
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="Stop search at first config exceeding threshold.",
    )
    parser.add_argument(
        "--export_best_model",
        action="store_true",
        help="Export the best-config quantized model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Export directory (default: <model_name>-bestquantconfig).",
    )
    parser.add_argument(
        "--min_kv_scale",
        type=float,
        default=0.0,
        help="Minimum kv-cache scale.",
    )
    parser.add_argument(
        "--max_configs",
        type=int,
        default=None,
        help="Maximum number of configs to evaluate.",
    )
    parser.add_argument(
        "--search_configs",
        type=str,
        nargs="+",
        default=None,
        metavar="MODE",
        help=(
            "Quantization modes to search for layer partitions "
            "(default: all modes supported by --hardware). "
            "Example: --search_configs native ptpc_fp8 mxfp4"
        ),
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=DEFAULT_EXCLUDE_PATTERNS.copy(),
        help="Layer-name patterns to exclude from quantization.",
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
        help="Include kv_cache quantization modes in the search. Default: False (kv_cache stays native). Use --no-kv-cache-quant to disable explicitly.",
    )
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--self-attn-only",
        action="store_true",
        dest="self_attn_only",
        help="Search only self_attn layer modes.",
    )
    layer_group.add_argument(
        "--mlp-only",
        action="store_true",
        dest="mlp_only",
        help="Search only mlp/MoE layer modes.",
    )
    args, vllm_cli_args = parser.parse_known_args()
    args.vllm_cli_args = vllm_cli_args
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    model_path = args.model_dir.rstrip("/")
    granularity = _parse_enum(SearchGranularity, args.granularity, "granularity")
    hardware = _parse_enum(HardwareTarget, args.hardware, "hardware")
    config = MixPrecisionConfig(
        granularity=granularity,
        hardware=hardware,
        eval_metrics=[args.eval_metric],
        eval_num_samples=args.gsm8k_num_samples,
        eval_max_new_tokens=args.gsm8k_max_new_tokens,
        num_calib_samples=args.num_calib_samples,
        early_stop=args.early_stop,
        min_kv_scale=args.min_kv_scale,
        exclude_patterns=list(args.exclude),
        max_configs=args.max_configs,
        calib_seq_len=args.calib_seq_len,
        eval_threshold=args.eval_threshold,
        module_search_config=_build_module_search_config(args),
    )
    run_search_vllm(args, model_path, config)


if __name__ == "__main__":
    main()
