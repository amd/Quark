#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""AutoRound driver script: quantize a real LLM (INT4 or MXFP4, weight-only or weight+activation)
and report wikitext2 PPL / task-eval scores.

1. Loads a HF causal LM + tokenizer.
2. Evaluates BF16 wikitext2 perplexity (baseline).
3. Unless ``--eval_only_bf16``, calibrates a quantized copy (``quantize_model``, in
   ``quant_schemes.py``, per ``--quant_scheme``), then tunes it with AutoRound
   (``blockwise_tuning_algo`` -> ``AutoRoundProcessor``) on a calibration dataloader.
4. Evaluates the tuned model's wikitext2 perplexity (and any ``--tasks``).
5. Prints and writes a JSON summary comparing BF16 vs AutoRound-tuned results.

``get_blockwise_tuning_dataloader`` (data_preparation.py) is imported, not copied, from
``examples/torch/language_modeling/llm_qat/efficientqat`` via a ``sys.path`` insert (a sibling
example, not an installable package).

Example:
    python main.py \\
        --model /group/amdneuralopt/huggingface/pretrained_models/meta-llama/Llama-2-7b-hf \\
        --group_size 128 --iters 200
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from quark.common.utils.log import ScreenLogger
from quark.contrib.llm_eval import ppl_eval
from quark.torch.algorithm.api import blockwise_tuning_algo
from quark.torch.quantization.config.algo_configs import get_algo_config

# pile-10k calibration (matches SignRound §4.1) is a first-class Quark utility.
from quark.torch.utils.llm.data_preparation import get_pile10k_dataloader
from quark.torch.utils.llm.model_preparation import get_model, preprocess_for_quantization

# The EfficientQAT example directory is not an installable package; import its small
# reusable helpers by inserting its path onto sys.path instead of copying the code.
# This driver lives at examples/torch/experimental/autoround/ (unvalidated algorithms go under
# examples/torch/experimental/, mirroring quark/experimental/torch/); efficientqat lives
# at examples/torch/language_modeling/llm_qat/efficientqat/, hence the "../../language_modeling/..".
_EFFICIENTQAT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "language_modeling", "llm_qat", "efficientqat")
)
sys.path.insert(0, _EFFICIENTQAT_DIR)

from data_preparation import get_blockwise_tuning_dataloader  # noqa: E402
from quant_schemes import (  # noqa: E402
    MX_GROUP_SIZE,
    MX_QUANT_SCHEMES,
    MX_WA_QUANT_SCHEMES,
    SUPPORTED_QUANT_SCHEMES,
    quantize_model,
)

# INT4 quant_scheme name from this driver's own quant_schemes.py, used as the default
# weight-only scheme for AutoRound calibration: symmetric INT4 per-group weight quantization.
DEFAULT_INT4_QUANT_SCHEME = "int4_wo_sym"

logger = ScreenLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Path or HF hub id of the model to quantize.")
    parser.add_argument("--group_size", type=int, default=128, help="Per-group weight quantization group size.")
    parser.add_argument("--iters", type=int, default=200, help="AutoRound optimization steps per block.")
    parser.add_argument(
        "--lr", type=float, default=None, help="AutoRound learning rate for rounding offset V. Default: 1/iters."
    )
    parser.add_argument(
        "--minmax_lr",
        type=float,
        default=None,
        help="AutoRound learning rate for the minmax-tuning clip params. Default: same as --lr "
        "(i.e. 1/iters unless --lr is set), so it scales with --iters instead of staying fixed "
        "at the shipped per-model default config's value.",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="c4",
        choices=["c4", "redpajama", "pile"],
        help="Calibration dataset. 'pile' = NeelNanda/pile-10k (SignRound paper §4.1).",
    )
    parser.add_argument("--calib_samples", type=int, default=128, help="Number of calibration samples.")
    parser.add_argument("--seqlen", type=int, default=2048, help="Calibration/eval sequence length.")
    parser.add_argument(
        "--quant_scheme",
        type=str,
        default=DEFAULT_INT4_QUANT_SCHEME,
        choices=SUPPORTED_QUANT_SCHEMES,
        help="Quant scheme used to build the calibrated model before AutoRound tuning ("
        "naming follows quark/torch/quantization/config/template.py's convention where a "
        "template scheme exists). INT2/INT4/UINT4 ('*_wo_*') are weight-only. MXFP4 has two "
        "schemes: 'mxfp4_weight_only' (weight-only) and 'mxfp4' (weight+activation, activation "
        "quantization has no learnable parameters -- only the weight side is tuned); both use "
        "Quark's 'even' scale_calculation_mode (matches Quark's HIP kernel).",
    )
    parser.add_argument(
        "--eval_only_bf16", action="store_true", help="Only evaluate the BF16 baseline; skip AutoRound quantization."
    )
    parser.add_argument(
        "--eval_only_rtn",
        action="store_true",
        help="Evaluate BF16 and plain RTN quantization (--quant_scheme, no AutoRound tuning); "
        "skip AutoRound. Uses the same eval path as the AutoRound result, for an apples-to-apples "
        "BF16 / RTN / AutoRound comparison.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional cap on wikitext2 samples, for a quick smoke run."
    )
    parser.add_argument("--model_trust_remote_code", action="store_true")
    parser.add_argument(
        "--tasks",
        default=None,
        type=str,
        metavar="task1,task2",
        help="Comma-separated lm-eval-harness task/group names to evaluate on BF16 + AutoRound "
        "(e.g. 'mmlu' or 'mmlu,gsm8k'). Matches quantize_quark.py's --tasks.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        metavar="N",
        help="Few-shot count for --tasks (paper uses 0 for MMLU). Matches quantize_quark.py's --num_fewshot.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=str,
        default="8",
        metavar="auto|auto:N|N",
        help="lm-eval batch size for --tasks, forwarded to task_eval. Matches quantize_quark.py's --eval_batch_size.",
    )
    parser.add_argument(
        "--max_eval_batch_size",
        type=int,
        default=64,
        help="Max batch size to try with --eval_batch_size auto. Matches quantize_quark.py's --max_eval_batch_size.",
    )
    parser.add_argument("--out", type=str, default="autoround_ppl_result.json", help="Path to write JSON summary.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for AutoRound's per-step minibatch sampling (random.sample in the tuning loop is "
        "otherwise unseeded, like the official auto-round repo's own sampler — set this to make "
        "runs at a fixed config reproducible / comparable).",
    )
    return parser.parse_args()


def eval_tasks(
    model,
    tokenizer,
    tasks: str,
    num_fewshot: int | None = None,
    batch_size: str = "8",
    max_batch_size: int = 64,
    output_dir: str = "task_eval_output",
) -> dict[str, float]:
    """Evaluate arbitrary lm-eval-harness tasks/groups, returning {task_name: score}.

    ``tasks`` is a comma-separated string (e.g. ``"mmlu"`` or ``"mmlu,wikitext"``), matching
    ``quantize_quark.py``'s ``--tasks``. Calls ``quark.contrib.llm_eval.task_eval`` — the same
    entry point ``quantize_quark.py``'s ``eval_model`` uses for ``--tasks`` — instead of driving
    lm-eval-harness directly, so this picks up Quark's HFLM wiring (``create_from_arg_obj``
    override, ``SUPPORTED_MODEL_ARGS`` validation) rather than duplicating it. ``task_eval`` writes
    results to disk (no return value); scores are read back from the results json it produces
    under ``output_dir``. Accuracy-style tasks (``acc``/``acc_norm``/``exact_match``) are returned
    as a 0-100 percentage; perplexity-style tasks (e.g. ``wikitext``'s ``word_perplexity``) are
    returned as-is (a raw perplexity value is not a percentage).
    """
    from quark.contrib.llm_eval import task_eval

    model.eval()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_eval(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        tasks=tasks,
        num_fewshot=num_fewshot,
        output_path=str(out_dir),
    )

    result_files = sorted(out_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not result_files:
        raise RuntimeError(f"task_eval did not write a results_*.json under {out_dir}")
    with open(result_files[-1]) as f:
        results = json.load(f)

    ACCURACY_METRIC_NAMES = ("acc", "acc_norm", "exact_match")
    PERPLEXITY_METRIC_NAMES = ("word_perplexity", "byte_perplexity", "bits_per_byte")

    scores: dict[str, float] = {}
    for task_name in (t.strip() for t in tasks.split(",")):
        task_results = results["results"].get(task_name, {})
        # The primary metric key varies by task (acc,none / acc_norm,none / exact_match,none /
        # word_perplexity,none / ...).
        accuracy_key = next((k for k in task_results if k.split(",")[0] in ACCURACY_METRIC_NAMES), None)
        perplexity_key = next((k for k in task_results if k.split(",")[0] in PERPLEXITY_METRIC_NAMES), None)
        if accuracy_key is not None:
            scores[task_name] = float(task_results[accuracy_key]) * 100.0
        elif perplexity_key is not None:
            scores[task_name] = float(task_results[perplexity_key])
    return scores


def load_model_and_tokenizer(model_path: str, trust_remote_code: bool) -> tuple[torch.nn.Module, AutoTokenizer]:
    # Use Quark's own `get_model()` rather than a bare `AutoModelForCausalLM.from_pretrained`:
    # some architectures (e.g. Qwen3.5's dense checkpoints, which ship as an image/video-text
    # wrapper with config.model_type=="qwen3_5") need a model-type-specific loading branch to
    # resolve to their text-only backbone; `get_model()` already knows about these.
    model, _ = get_model(
        model_path,
        data_type="auto",
        multi_gpu="auto",
        trust_remote_code=trust_remote_code,
    )
    tokenizer_kwargs = {"trust_remote_code": trust_remote_code, "use_fast": False}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, **tokenizer_kwargs)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    return model, tokenizer


def eval_wikitext2_ppl(model: torch.nn.Module, tokenizer: AutoTokenizer, limit: int | None = None) -> float:
    """Evaluate wikitext2 perplexity, reusing ``quark.contrib.llm_eval.ppl_eval``.

    Mirrors the eval block in ``examples/torch/language_modeling/llm_qat/efficientqat/main.py``.
    """
    model.eval()
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    test_text = testdata["text"][:limit] if limit is not None else testdata["text"]
    testenc = tokenizer("\n\n".join(test_text), return_tensors="pt")
    ppl = ppl_eval(model, testenc, next(model.parameters()).device)
    return float(ppl)


def main() -> None:
    args = parse_args()
    lr = args.lr if args.lr is not None else 1.0 / args.iters
    _t_start = time.perf_counter()
    timings: dict[str, float] = {}

    logger.info(f"Loading model and tokenizer from {args.model} ...")
    _t = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(args.model, args.model_trust_remote_code)
    timings["load_model_seconds"] = time.perf_counter() - _t

    logger.info("Evaluating BF16 wikitext2 PPL (baseline) ...")
    _t = time.perf_counter()
    bf16_ppl = eval_wikitext2_ppl(model, tokenizer, limit=args.limit)
    timings["bf16_ppl_seconds"] = time.perf_counter() - _t
    logger.info(f"BF16 wikitext2 PPL: {bf16_ppl}")

    bf16_task_results: dict[str, float] = {}
    autoround_task_results: dict[str, float] = {}
    if args.tasks:
        logger.info(f"Evaluating BF16 on tasks={args.tasks} ({args.num_fewshot}-shot) ...")
        _t = time.perf_counter()
        bf16_task_results = eval_tasks(
            model,
            tokenizer,
            args.tasks,
            args.num_fewshot,
            args.eval_batch_size,
            args.max_eval_batch_size,
            output_dir=os.path.join(os.path.dirname(args.out) or ".", "task_eval_output", "bf16"),
        )
        timings["bf16_tasks_seconds"] = time.perf_counter() - _t
        logger.info(f"BF16 task results: {bf16_task_results}")

    rtn_ppl: float | None = None
    rtn_task_results: dict[str, float] = {}
    autoround_ppl: float | None = None
    autoround_seconds: float | None = None
    peak_gpu_mem_gib: float | None = None
    if not args.eval_only_bf16:
        if args.quant_scheme in (*MX_QUANT_SCHEMES, *MX_WA_QUANT_SCHEMES) and args.group_size != MX_GROUP_SIZE:
            logger.info(
                f"--group_size={args.group_size} is not used by "
                f"--quant_scheme={args.quant_scheme} (OCP MX group size is fixed at "
                f"{MX_GROUP_SIZE}) — overriding to {MX_GROUP_SIZE}."
            )
            args.group_size = MX_GROUP_SIZE
        logger.info(f"Weight-only quantizing model with scheme={args.quant_scheme}, group_size={args.group_size} ...")
        _t = time.perf_counter()
        preprocess_for_quantization(model)
        model = quantize_model(model, args.quant_scheme, args.group_size)
        timings["quantize_calibrate_seconds"] = time.perf_counter() - _t

        # Report the plain-RTN result (before any AutoRound tuning) using the exact same eval
        # path as BF16 / AutoRound below, so all three numbers are directly comparable.
        logger.info("Evaluating RTN INT4 wikitext2 PPL ...")
        _t = time.perf_counter()
        rtn_ppl = eval_wikitext2_ppl(model, tokenizer, limit=args.limit)
        timings["rtn_ppl_seconds"] = time.perf_counter() - _t
        logger.info(f"RTN INT4 wikitext2 PPL: {rtn_ppl}")

        if args.tasks:
            logger.info(f"Evaluating RTN INT4 on tasks={args.tasks} ({args.num_fewshot}-shot) ...")
            _t = time.perf_counter()
            rtn_task_results = eval_tasks(
                model,
                tokenizer,
                args.tasks,
                args.num_fewshot,
                args.eval_batch_size,
                args.max_eval_batch_size,
                output_dir=os.path.join(os.path.dirname(args.out) or ".", "task_eval_output", "rtn"),
            )
            timings["rtn_tasks_seconds"] = time.perf_counter() - _t
            logger.info(f"RTN INT4 task results: {rtn_task_results}")

    if not args.eval_only_bf16 and not args.eval_only_rtn:
        logger.info("Loading a fresh full-precision reference model for AutoRound reconstruction target ...")
        ref_model, _ = load_model_and_tokenizer(args.model, args.model_trust_remote_code)
        ref_model.eval()

        logger.info(
            f"Building {args.calib_dataset} calibration dataloader "
            f"(train_size={args.calib_samples}, seqlen={args.seqlen}) ..."
        )
        # batch_size=1 on both paths: AutoRoundConfig.batch_size alone controls sequences-per-step
        # (see optimize_wrappers_signed_sgd), so the loader itself must yield one sequence per item
        # -- otherwise the two batching levels compound (e.g. a default loader batch_size=2 x
        # AutoRoundConfig.batch_size=8 => 16 sequences/step instead of 8).
        if args.calib_dataset == "pile":
            train_loader, _val_loader = get_pile10k_dataloader(
                tokenizer=tokenizer,
                train_size=args.calib_samples,
                seqlen=args.seqlen,
                batch_size=1,
            )
        else:
            train_loader, _val_loader = get_blockwise_tuning_dataloader(
                name=args.calib_dataset,
                tokenizer=tokenizer,
                train_size=args.calib_samples,
                seqlen=args.seqlen,
                batch_size=1,
            )

        # Use the per-model default AutoRound config (inside_layer_modules / model_decoder_layers)
        # from the central algo-config map, like GPTQ/AWQ; override run-specific hyperparameters.
        model_type = getattr(model.config, "model_type", None)
        base_cfg = get_algo_config("autoround", model_type) if model_type else None
        if base_cfg is None:
            raise ValueError(
                f"No default AutoRound config for model_type={model_type!r}. "
                f"Add an entry to AUTOROUND_MAP in quark/torch/quantization/config/algo_configs.py."
            )
        autoround_cfg = copy.deepcopy(base_cfg)
        autoround_cfg.iters = args.iters
        autoround_cfg.lr = lr
        # Follow --lr (and therefore --iters) by default, matching the class docstring's stated
        # 1/iters convention -- otherwise this would stay at whatever the shipped per-model
        # default config carries regardless of --iters.
        autoround_cfg.minmax_lr = args.minmax_lr if args.minmax_lr is not None else lr
        logger.info(f"AutoRound config: {asdict(autoround_cfg)}")

        is_accelerate = hasattr(model, "hf_device_map")
        logger.info(f"Running AutoRound blockwise tuning (seed={args.seed}) ...")
        # Seed right before tuning starts (not earlier): isolates the tuning loop's per-step
        # `random.sample` minibatch draws (autoround.py) from any randomness already consumed by
        # model loading / BF16 eval above, so runs at a fixed seed are directly comparable.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats()
        _t_ar = time.perf_counter()
        model = blockwise_tuning_algo(ref_model, model, autoround_cfg, is_accelerate, train_loader)
        autoround_seconds = time.perf_counter() - _t_ar
        # Peak GPU memory during AutoRound (holds the FP reference + quantized model + block
        # activations) — the run's high-water mark. Reported in GiB for the visible device(s).
        peak_gpu_mem_gib = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None

        timings["autoround_tuning_seconds"] = autoround_seconds

        logger.info("Evaluating AutoRound INT4 wikitext2 PPL ...")
        _t = time.perf_counter()
        autoround_ppl = eval_wikitext2_ppl(model, tokenizer, limit=args.limit)
        timings["autoround_ppl_seconds"] = time.perf_counter() - _t
        logger.info(f"AutoRound INT4 wikitext2 PPL: {autoround_ppl}")

        if args.tasks:
            logger.info(f"Evaluating AutoRound INT4 on tasks={args.tasks} ({args.num_fewshot}-shot) ...")
            _t = time.perf_counter()
            autoround_task_results = eval_tasks(
                model,
                tokenizer,
                args.tasks,
                args.num_fewshot,
                args.eval_batch_size,
                args.max_eval_batch_size,
                output_dir=os.path.join(os.path.dirname(args.out) or ".", "task_eval_output", "autoround"),
            )
            timings["autoround_tasks_seconds"] = time.perf_counter() - _t
            logger.info(f"AutoRound INT4 task results: {autoround_task_results}")

    delta = None if autoround_ppl is None else autoround_ppl - bf16_ppl
    rtn_delta = None if rtn_ppl is None else rtn_ppl - bf16_ppl
    total_seconds = time.perf_counter() - _t_start
    summary = {
        "model": args.model,
        "group_size": args.group_size,
        "iters": args.iters,
        "quant_scheme": args.quant_scheme,
        "calib_dataset": args.calib_dataset,
        "calib_samples": args.calib_samples,
        "seed": args.seed,
        "bf16_ppl": bf16_ppl,
        "rtn_int4_ppl": rtn_ppl,
        "rtn_delta": rtn_delta,
        "autoround_int4_ppl": autoround_ppl,
        "delta": delta,
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot if args.tasks else None,
        "bf16_task_results": bf16_task_results,
        "rtn_int4_task_results": rtn_task_results,
        "autoround_int4_task_results": autoround_task_results,
        "autoround_seconds": autoround_seconds,
        "total_seconds": total_seconds,
        "peak_gpu_mem_gib": peak_gpu_mem_gib,
        "timings": timings,
    }
    logger.info(
        f"SUMMARY | model={args.model} | group_size={args.group_size} | iters={args.iters} | "
        f"bf16_ppl={bf16_ppl} | rtn_int4_ppl={rtn_ppl} | autoround_int4_ppl={autoround_ppl} | delta={delta} | "
        f"autoround_seconds={autoround_seconds} | total_seconds={total_seconds} | peak_gpu_mem_gib={peak_gpu_mem_gib}"
    )
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote summary to {args.out}")


if __name__ == "__main__":
    main()
