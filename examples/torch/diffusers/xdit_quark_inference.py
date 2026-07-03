#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2023 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
xDiT inference with AMD Quark quantization support.

This script provides a standalone interface for running xDiT distributed inference with
optional AMD Quark quantization (FP8 or MXFP4). It wraps xDiT's xFuserModelRunner with
integrated quantization and supports single-GPU or multi-GPU execution via torchrun.
"""

import csv
import json
import os
from pathlib import Path

import torch

from quark.common.utils.log import ScreenLogger

# AMD Quark imports
from quark.torch.quantization.api import ModelQuantizer
from quark.torch.quantization.config.config import (
    Config,
    FP8E4M3PerTensorSpec,
    OCP_MXFP4Spec,
    QuantizationConfig,
)

logger = ScreenLogger(__name__)

# Import xDiT components
try:
    from xfuser import xFuserArgs
    from xfuser.config import FlexibleArgumentParser
    from xfuser.runner import xFuserModelRunner
except ImportError as e:
    logger.error(f"Failed to import xDiT components: {e}")
    raise

# Allow single-GPU runs to work without torchrun
if not os.environ.get("RANK"):
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    logger.info("Running in single-GPU mode (no torchrun detected)")


def quantize_model_with_quark(
    module: torch.nn.Module,
    quant_mode: str = "fp8",
    device: torch.device | None = None,
    is_dynamic: bool = True,
    calib_dataloader: object | None = None,
) -> torch.nn.Module:
    """
    Quantize a model using AMD Quark with FP8 or MXFP4 quantization.

    :param torch.nn.Module module: PyTorch module to quantize.
    :param str quant_mode: Quantization mode, either ``"fp8"`` (FP8 E4M3) or ``"mxfp4"`` (MXFP4).
    :param torch.device | None device: Target device for the quantized model.
    :param bool is_dynamic: Whether to use dynamic quantization (``True``) or static (``False``).
    :param object | None calib_dataloader: Calibration dataloader for static quantization
        (required when ``is_dynamic=False``).

    :return: Quantized PyTorch module with weights replaced.
    :rtype: torch.nn.Module
    """
    logger.info(f"Quantizing model with AMD Quark ({quant_mode} mode, dynamic={is_dynamic})...")

    # Configure quantization spec based on mode
    if quant_mode.lower() == "fp8":
        quant_spec = FP8E4M3PerTensorSpec(
            observer_method="min_max", scale_type="float", is_dynamic=is_dynamic
        ).to_quantization_spec()
        logger.info("Using FP8 E4M3 quantization format")
    elif quant_mode.lower() == "mxfp4":
        quant_spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=is_dynamic).to_quantization_spec()
        logger.info("Using OCP MXFP4 quantization format")
    else:
        raise ValueError(f"Unsupported quantization mode: {quant_mode}. Choose 'fp8' or 'mxfp4'")

    # Create quantization config for both weights and activations
    quant_config = QuantizationConfig(weight=quant_spec, input_tensors=quant_spec)

    # Initialize ModelQuantizer
    quantizer = ModelQuantizer(Config(global_quant_config=quant_config))

    # Quantize the model
    # For dynamic quantization, calib_dataloader is optional
    # For static quantization, it's required
    if not is_dynamic and calib_dataloader is None:
        logger.info(
            "Warning: Static quantization requested but no calibration dataloader provided. Using dynamic quantization."
        )
        is_dynamic = True

    if is_dynamic or calib_dataloader is None:
        # Dynamic quantization - no calibration needed
        quantized_model = quantizer.quantize_model(module)
    else:
        # Static quantization with calibration
        logger.info("Running calibration for static quantization...")
        quantized_model = quantizer.quantize_model(module, calib_dataloader)

    # Move to device if specified
    if device is not None:
        quantized_model = quantized_model.to(device)

    logger.info(f"Model quantization with AMD Quark ({quant_mode}) completed successfully")
    return quantized_model


class xDiTQuarkInferenceRunner:
    """
    Wrapper around xDiT's xFuserModelRunner with integrated Quark quantization support.

    This class provides quantization capabilities without modifying the xDiT codebase.
    """

    def __init__(self, config: dict):
        """
        Initialize the inference runner.

        :param dict config: Dictionary containing model and inference configuration.
        """
        self.config = config
        self.quark_config = {
            "enabled": config.get("use_quark_quantize", False),  # Default: disabled (opt-in)
            "mode": config.get("quark_quantization_mode", "fp8"),
            "dynamic": config.get("quark_dynamic_quantization", True),
        }

        # Initialize xDiT model runner
        self.runner = xFuserModelRunner(config)
        self.is_quantized = False

    def apply_quark_quantization(self):
        """Apply AMD Quark quantization to the model after initialization but before inference."""
        if not self.quark_config["enabled"]:
            logger.info("Quark quantization is disabled")
            return

        if self.is_quantized:
            logger.warning("Model already quantized, skipping...")
            return

        logger.info("=" * 80)
        logger.info("Applying AMD Quark Quantization")
        logger.info(f"  Mode: {self.quark_config['mode']}")
        logger.info(f"  Dynamic: {self.quark_config['dynamic']}")
        logger.info("=" * 80)

        try:
            # Access the transformer/UNet from the pipeline
            if hasattr(self.runner.model.pipe, "transformer"):
                model_to_quantize = self.runner.model.pipe.transformer
                logger.info("Quantizing transformer model...")
            elif hasattr(self.runner.model.pipe, "unet"):
                model_to_quantize = self.runner.model.pipe.unet
                logger.info("Quantizing UNet model...")
            else:
                logger.warning("Could not find transformer or unet in pipeline. Skipping quantization.")
                return

            # Apply Quark quantization
            device = next(model_to_quantize.parameters()).device
            quantized_model = quantize_model_with_quark(
                module=model_to_quantize,
                quant_mode=self.quark_config["mode"],
                device=device,
                is_dynamic=self.quark_config["dynamic"],
                calib_dataloader=None,  # Dynamic quantization doesn't require calibration
            )

            # Replace the model in the pipeline
            if hasattr(self.runner.model.pipe, "transformer"):
                self.runner.model.pipe.transformer = quantized_model
            elif hasattr(self.runner.model.pipe, "unet"):
                self.runner.model.pipe.unet = quantized_model

            self.is_quantized = True
            logger.info("Quark quantization applied successfully!")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to apply Quark quantization: {e}")
            logger.error("Continuing without quantization...")
            raise

    def initialize(self, input_args: dict) -> None:
        """Initialize the model and apply quantization if enabled."""
        logger.info("Initializing xDiT model...")
        self.runner.initialize(input_args)

        # Apply quantization after model is initialized
        if self.quark_config["enabled"]:
            self.apply_quark_quantization()

    def run(self, input_args: dict):
        """Run inference with the model and return output and timings."""
        return self.runner.run(input_args)

    def profile(self, input_args: dict):
        """Profile the model execution and return output, timings, and profile data."""
        return self.runner.profile(input_args)

    def preprocess_args(self, input_args: dict) -> dict:
        """Preprocess and validate input arguments before inference."""
        return self.runner.preprocess_args(input_args)

    def print_args(self, args: dict) -> None:
        """Print configuration arguments including Quark quantization settings."""
        self.runner.print_args(args)
        logger.info("Quark Quantization Settings:")
        logger.info(f"  Enabled: {self.quark_config['enabled']}")
        if self.quark_config["enabled"]:
            logger.info(f"  Mode: {self.quark_config['mode']}")
            logger.info(f"  Dynamic: {self.quark_config['dynamic']}")

    def save(self, output=None, timings=None, profile=None, save_once=True) -> None:
        """Save outputs, timings, and profile data to disk."""
        self.runner.save(output, timings, profile, save_once)

    def cleanup(self) -> None:
        """Release distributed and model resources."""
        self.runner.cleanup()


def parse_arguments() -> dict:
    """
    Parse command line arguments for xDiT inference and Quark quantization.

    :return: Parsed arguments as a dictionary.
    :rtype: dict
    """
    parser = FlexibleArgumentParser(description="xDiT Inference with Quark Quantization Support")

    # Add xDiT runner arguments
    parser = xFuserArgs.add_runner_args(parser)

    # Add Quark quantization arguments
    parser.add_argument(
        "--use_quark_quantize", action="store_true", default=False, help="Enable AMD Quark quantization for the model"
    )
    parser.add_argument(
        "--quark_quantization_mode",
        type=str,
        default="fp8",
        choices=["fp8", "mxfp4"],
        help="Quantization mode: fp8 (FP8 E4M3) or mxfp4 (OCP MXFP4). Default: fp8",
    )
    parser.add_argument(
        "--quark_dynamic_quantization",
        action="store_true",
        default=True,
        help="Use dynamic quantization (default: True). Set to False for static quantization with calibration.",
    )
    parser.add_argument(
        "--csv_prompt_file",
        type=str,
        default=None,
        help="Path to a CSV file with columns: image_id, caption_id, prompt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save generated images. Filenames use {image_id}-{caption_id}.png format when using --csv_prompt_file.",
    )

    # Parse arguments
    args = parser.parse_args()
    return vars(args)


def _load_csv_prompts(csv_path: str) -> list[dict]:
    """
    Load prompts from a CSV file with columns: ``image_id``, ``caption_id``, ``prompt``.

    :param str csv_path: Path to the CSV file.

    :return: List of dictionaries with ``image_id``, ``caption_id``, and ``prompt`` keys.
    :rtype: list[dict]
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "image_id": row["image_id"].strip(),
                    "caption_id": row["caption_id"].strip(),
                    "prompt": row["prompt"].strip(),
                }
            )
    return rows


def main() -> None:
    """Main entry point: parse args, initialize runner, and process prompts."""
    # Parse arguments
    args = parse_arguments()

    # Display runtime information
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        logger.info("=" * 80)
        logger.info("xDiT Inference with AMD Quark Quantization")
        logger.info("=" * 80)
        logger.info(f"World Size: {world_size}")
        logger.info(f"Current Rank: {rank}")
        if world_size > 1:
            logger.info("Running in distributed mode (torchrun detected)")
        logger.info("=" * 80)

    # Determine output directory
    output_dir = args.get("output_dir")
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    # Load prompts: CSV > JSON > single prompt
    prompts_to_process = []
    use_csv_naming = False

    if args.get("csv_prompt_file"):
        csv_path = args["csv_prompt_file"]
        logger.info(f"Loading prompts from CSV: {csv_path}")
        csv_rows = _load_csv_prompts(csv_path)
        for row in csv_rows:
            prompts_to_process.append(
                {
                    "image_id": row["image_id"],
                    "caption_id": row["caption_id"],
                    "prompt": row["prompt"],
                }
            )
        use_csv_naming = True
        logger.info(f"Loaded {len(prompts_to_process)} prompt(s) from CSV")
    elif args.get("prompt_file"):
        logger.info(f"Loading prompts from {args['prompt_file']}")
        with open(args["prompt_file"]) as f:
            prompts_data = json.load(f)
        prompts_to_process = prompts_data if isinstance(prompts_data, list) else [prompts_data]
        logger.info(f"Loaded {len(prompts_to_process)} prompt(s) from file")
    else:
        prompt_text = args.get("prompt", "")
        if isinstance(prompt_text, list):
            prompt_text = " ".join(prompt_text)
        prompts_to_process = [{"id": 1, "prompt": prompt_text}]

    # Create runner with Quark support
    runner = xDiTQuarkInferenceRunner(args)
    runner.print_args(args)

    # Process each prompt
    for idx, prompt_entry in enumerate(prompts_to_process):
        prompt_text = prompt_entry.get("prompt", "")
        logger.info(f"Processing prompt {idx + 1}/{len(prompts_to_process)}: {prompt_text}")

        # Update args with current prompt
        current_args = args.copy()
        current_args["prompt"] = prompt_text

        input_args = runner.preprocess_args(current_args)

        # Initialize only once (quantization happens here)
        if idx == 0:
            runner.initialize(input_args)

        # Run inference
        if args.get("profile", False):
            output, timings, profile = runner.profile(input_args)
            runner.save(profile=profile)
        else:
            output, timings = runner.run(input_args)

            # Save with custom filename when using CSV input and output_dir
            if use_csv_naming and output_dir and rank == 0:
                image_id = prompt_entry["image_id"]
                caption_id = prompt_entry["caption_id"]
                filename = f"{image_id}-{caption_id}.png"
                save_path = output_dir / filename
                if hasattr(output, "images") and output.images:
                    output.images[0].save(str(save_path))
                    logger.info(f"Saved image to {save_path}")
                else:
                    logger.warning(f"No images in output for prompt {idx + 1}, skipping save")
            else:
                runner.save(output=output, timings=timings)

        logger.info(f"Completed prompt {idx + 1}/{len(prompts_to_process)}")

    # Cleanup
    runner.cleanup()

    if rank == 0:
        logger.info("=" * 80)
        logger.info("Inference completed successfully!")
        if use_csv_naming and output_dir:
            logger.info(f"Images saved to: {output_dir}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
