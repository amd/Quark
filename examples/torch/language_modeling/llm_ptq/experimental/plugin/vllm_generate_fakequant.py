#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Use Quark fakequant models with vLLM LLM.generate method."""

import os
import sys

from common import find_repo_root
from vllm import LLM, SamplingParams

from quark.common.utils.log import ScreenLogger

# Ensure workers can import our custom worker module when using spawn
repo_root = find_repo_root()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + f"{repo_root}"


logger = ScreenLogger(__name__)


def main() -> None:
    """Main function to demonstrate LLM.generate usage with Quark quantization."""
    import argparse

    parser = argparse.ArgumentParser(description="vLLM LLM.generate with Quark quantization support")
    parser.add_argument("model", type=str, help="The path or name of the model to load")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, my name is",
        help="Prompt text for generation",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Multiple prompts for batch generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager execution (disable CUDA graph)",
    )

    args = parser.parse_args()

    # Set quantization environment variables if not already set
    quant_cfg = os.environ.get("QUANT_CFG")
    quant_dataset = os.environ.get("QUANT_DATASET", "cnn_dailymail")
    quant_calib_size = os.environ.get("QUANT_CALIB_SIZE", "512")

    if quant_cfg:
        logger.info("[QUARK] Quantization enabled:")
        logger.info("  QUANT_CFG: %s", quant_cfg)
        logger.info("  QUANT_DATASET: %s", quant_dataset)
        logger.info("  QUANT_CALIB_SIZE: %s", quant_calib_size)
        if not args.enforce_eager:
            logger.warning("[QUARK] Overriding --enforce-eager to True because QUANT_CFG is set")
        args.enforce_eager = True
    else:
        logger.info("[QUARK] Quantization disabled (QUANT_CFG not set)")

    # Initialize LLM with custom worker
    logger.info("[QUARK] Loading model: %s", args.model)
    llm = LLM(
        model=args.model,
        worker_cls="quark.experimental.plugin.fakequant_worker.QuarkFakeQuantWorker",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    # from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k, evaluate_gsm8k_offline
    # Prepare prompts
    if args.prompts:
        prompts = args.prompts
    else:
        prompts = [args.prompt]

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Generate
    logger.info("[QUARK] Generating with %d prompt(s)...", len(prompts))
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    logger.info("[QUARK] Generation results:")
    print("=" * 80)
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        logger.info("\nPrompt %d: %r", i + 1, prompt)
        logger.info("Generated: %r", generated_text)
        print("-" * 80)


if __name__ == "__main__":
    main()
