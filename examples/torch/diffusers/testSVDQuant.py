#!/usr/bin/env python3
"""
Test SVDQuant with multiple quantization modes for diffusion models.

Supported models:
- Stable Diffusion XL (SDXL) — quantizes the UNet
- FLUX.1-dev — quantizes the Transformer

This script validates that SVDQuant works correctly in Quark by running
quantization with different modes:
- w4a16 (INT4 weights + FP16 activations)
- w4a4 (INT4 weights + INT4 activations)
- mxfp4 (OCP Microscaling FP4 weights + activations)

For each mode, it:
1. Loads the pipeline (SDXL UNet or FLUX Transformer)
2. Collects calibration data
3. Applies SVDQuant + quantization
4. Generates a test image
5. Saves the quantized model

Usage examples:
    # SDXL (default)
    python testSVDQuant.py --modes w4a16 mxfp4

    # FLUX.1-dev
    python testSVDQuant.py \\
        --model_id black-forest-labs/FLUX.1-dev \\
        --modes w4a16 mxfp4
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm  # noqa: F401  # Kept for legacy progress bars in caller scripts.

# Add Quark root to path (go up 3 levels from examples/torch/diffusers/)
QUARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if os.path.isdir(QUARK_ROOT) and QUARK_ROOT not in sys.path:
    sys.path.insert(0, QUARK_ROOT)

from quark.common.utils.log import ScreenLogger  # noqa: E402
from quark.torch import ModelQuantizer  # noqa: E402
from quark.torch.algorithm.svdquant.svdquant import (  # noqa: E402
    QUANT_MODE_TO_SCHEME,
    ErrorCorrectedModule,
    SVDQuantProcessor,
    build_quant_layer_config,
)
from quark.torch.quantization.config.config import (  # noqa: E402
    QConfig,
    QLayerConfig,
    SVDQuantConfig,
)
from quark.torch.quantization.utils import RuntimeOptions, enable_native_inference  # noqa: E402
from quark.torch.utils.diffusers import get_calib_dataloader  # noqa: E402

logger = ScreenLogger(__name__)

DEFAULT_NEGATIVE_PROMPT = "normal quality, low quality, worst quality, low res, blurry, nsfw, nude."

FLUX_GENERATION_ARGS: dict[str, object] = {
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "max_sequence_length": 512,
}

DEFAULT_MODULE_FOR_MODEL: dict[str, str] = {
    "sdxl": "unet",
    "flux": "transformer",
    "sd3": "transformer",
}

SVDQUANT_EXCLUDE_PATTERNS: dict[str, list[str]] = {
    "sdxl": [
        "*time_embedding*",
        "*add_time_proj*",
        "*conv_in*",
        "*conv_out*",
        "*time_proj*",
        "*add_embedding*",
    ],
    "flux": [
        "*x_embedder*",
        "*context_embedder*",
        "*time_text_embed*",
        "*norm_out*",
        "*proj_out*",
        "*norm1.linear*",
        "*norm1_context.linear*",
    ],
    "sd3": [
        "*time_text_embed*",
        "*context_embedder*",
        "*pos_embed*",
        "*norm_out*",
        "*proj_out*",
    ],
}

QUANT_EXCLUDE_PATTERNS: dict[str, list[str]] = {
    "sdxl": [
        "*time_embedding*",
        "*time_emb_proj*",
        "*conv_in*",
        "*conv_out*",
        "*conv_shortcut*",
        "*add_embedding*",
        "*attn2.to_k*",
        "*attn2.to_v*",
        "*correction*",
    ],
    "flux": [
        "*x_embedder*",
        "*context_embedder*",
        "*time_text_embed*",
        "*norm_out*",
        "*proj_out*",
        "*correction*",
    ],
    "sd3": [
        "*time_text_embed*",
        "*context_embedder*",
        "*pos_embed*",
        "*norm_out*",
        "*proj_out*",
        "*correction*",
    ],
}

# Layers that receive W4A16 even when the global mode quantises activations.
QUANT_W4A16_OVERRIDE_PATTERNS: dict[str, list[str]] = {
    "flux": [
        "*norm1.linear*",
        "*norm1_context.linear*",
    ],
}

DEFAULT_COCO2014_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "inference", "text_to_image", "coco2014")
)


def load_coco2014_prompts(
    coco_dir: str = DEFAULT_COCO2014_DIR,
    max_prompts: int | None = None,
) -> list[str]:
    """Load calibration captions from COCO2014 ``captions/captions_source.tsv``."""
    tsv_path = os.path.join(coco_dir, "captions", "captions_source.tsv")
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(
            f"COCO2014 captions file not found at {tsv_path}.\nPlease follow COCO2014_SETUP.md to download the dataset."
        )

    prompts: list[str] = []
    with open(tsv_path, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header
            cols = line.split("\t")
            if len(cols) >= 3:
                prompts.append(cols[2].strip())

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    logger.info(f"Loaded {len(prompts)} COCO2014 calibration prompts from {tsv_path}")
    return prompts


def load_pipeline(model_id: str, device: str = "cuda"):
    from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline

    model_lower = model_id.lower()

    if "flux" in model_lower:
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
        return pipe, "flux"
    elif "stable-diffusion-3" in model_lower:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        pipe.to(device)
        return pipe, "sd3"
    else:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            device_map="balanced",
        )
        return pipe, "sdxl"


def get_quantize_target(pipe, model_type: str, module_name: str) -> nn.Module:
    """
    Retrieve the target module to quantize from the diffusion pipeline.

    Args:
        pipe: The diffusion pipeline object containing various modules.
        model_type: Type of the model (e.g., 'sdxl', 'flux', 'sd3').
        module_name: Name of the module to retrieve (e.g., 'unet', 'transformer', 'vae').

    Returns:
        nn.Module: The requested module from the pipeline.
    """
    if module_name == "unet":
        return pipe.unet
    elif module_name == "transformer":
        return pipe.transformer
    elif module_name == "vae":
        return pipe.vae.decoder
    elif module_name.startswith("text_encoder"):
        return getattr(pipe, module_name)
    else:
        return pipe.__dict__[module_name]


def set_quantize_target(pipe, model_type: str, module_name: str, model: nn.Module):
    if module_name == "unet":
        pipe.unet = model
    elif module_name == "transformer":
        pipe.transformer = model
    elif module_name == "vae":
        pipe.vae.decoder = model
    elif module_name.startswith("text_encoder"):
        setattr(pipe, module_name, model)
    else:
        pipe.__dict__[module_name] = model


@torch.no_grad()
def generate_image(
    pipe, prompt: str, model_type: str, n_steps: int = 30, seed: int | None = None, device: str = "cuda"
) -> Image.Image:
    gen_device = "cpu" if model_type in ("flux",) else device
    generator = None
    if seed is not None:
        generator = torch.Generator(device=gen_device).manual_seed(seed)

    if model_type == "flux":
        image = pipe(
            prompt=[prompt],
            num_inference_steps=n_steps,
            generator=generator,
            **FLUX_GENERATION_ARGS,
        ).images[0]
    else:
        image = pipe(
            prompt=[prompt],
            num_inference_steps=n_steps,
            negative_prompt=[DEFAULT_NEGATIVE_PROMPT],
            generator=generator,
            guidance_scale=8.0,
        ).images[0]

    return image


def collect_calibration_data(
    pipe,
    module_name: str,
    model_type: str,
    prompts: list[str],
    n_steps: int = 20,
    device: str = "cuda",
    max_captures: int = 200,
) -> DataLoader:
    """Collect calibration data via Quark's diffusers utility.

    Returns a DataLoader yielding ``dict[str, Any]`` batches keyed by the
    target submodule's forward parameter names, matching the contract
    consumed by ``ModelQuantizer.quantize_model`` via ``model(**data)``.
    """
    target = get_quantize_target(pipe, model_type, module_name)

    pipe_kwargs: dict[str, object] = {"negative_prompt": [DEFAULT_NEGATIVE_PROMPT], "guidance_scale": 8.0}
    if model_type == "flux":
        # Flux pipelines do not accept negative_prompt/guidance_scale the same way.
        pipe_kwargs = dict(FLUX_GENERATION_ARGS)

    logger.info(f"Calibration running for {len(prompts)} prompts")
    dataloader = get_calib_dataloader(
        pipe,
        target,
        prompts=prompts,
        n_steps=n_steps,
        seed=42,
        device=device,
        **pipe_kwargs,
    )

    # Cap the captured samples so very long calibration sets don't blow up memory.
    if max_captures is not None and len(dataloader.dataset.captured) > max_captures:
        dataloader.dataset.captured = dataloader.dataset.captured[:max_captures]
        logger.info(f"Capped to {max_captures} samples")
    logger.info(f"Captured {len(dataloader.dataset.captured)} samples")

    return dataloader


def quantize_model_with_svdquant(
    pipe,
    module_name: str,
    model_type: str,
    dataloader: DataLoader,
    quant_config: QConfig,
    svd_config: SVDQuantConfig,
    device: str = "cuda",
) -> nn.Module:
    """Apply SVDQuant and then regular quantization, in place on the pipeline."""

    target = get_quantize_target(pipe, model_type, module_name)

    logger.info(f"Applying SVDQuant to {module_name}...")
    processor = SVDQuantProcessor(
        model=target,
        quant_algo_config=svd_config,
        calib_data=dataloader,
    )
    processor.apply()

    logger.info(f"Quantizing {module_name}...")
    quantizer = ModelQuantizer(quant_config)
    quantized_model = quantizer.quantize_model(target, dataloader)

    return quantized_model


DEFAULT_TEST_PROMPT = "A serene mountain lake at sunset with snow-capped peaks reflecting in the water"

DEFAULT_CALIB_PROMPTS = [
    "A serene lake reflecting mountains at sunset",
    "A futuristic city with flying cars at night",
    "A close-up portrait of a person with dramatic lighting",
    "An abstract geometric pattern in bright colors",
    "A fantasy dragon perched on a crystal mountain",
]


def _save_quantized_model(quantized_module: nn.Module, module_name: str, output_dir: str) -> None:
    """Freeze and export the quantized model in the canonical (export) format.

    This is the standard SVDQuant checkpoint (``ErrorCorrectedModule`` with a
    ``QParamsLinear`` residual + the low-rank correction + smooth factors). It is
    intentionally saved *before* any native-inference conversion so the on-disk
    artifact is identical regardless of ``--native_inference`` -- native inference
    is a runtime mode re-enabled after loading, not a save format.
    """
    print("Saving quantized model...")
    try:
        from quark.torch import save_params

        frozen_model = ModelQuantizer.freeze(quantized_module)

        for _, param in frozen_model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        for _, buffer in frozen_model.named_buffers():
            if not buffer.is_contiguous():
                buffer.data = buffer.data.contiguous()

        save_params(frozen_model, model_type=module_name, export_dir=output_dir)

        smooth_factors = {}
        for name, module in frozen_model.named_modules():
            if isinstance(module, ErrorCorrectedModule) and module.smooth_factor is not None:
                smooth_factors[name] = module.smooth_factor.detach().cpu()
        if smooth_factors:
            sf_path = os.path.join(output_dir, "smooth_factors.pt")
            torch.save(smooth_factors, sf_path)
            print(f"✓ Saved {len(smooth_factors)} smooth_factors to {sf_path}")

        print(f"✓ Model saved to: {output_dir}")
    except Exception as e:
        print(f"✗ ERROR saving model: {e}")
        import traceback

        traceback.print_exc()


def test_quantization_mode(
    pipe,
    model_id: str,
    model_type: str,
    module_name: str,
    mode: str,
    calib_dataset,
    output_base_dir: str,
    test_prompt: str,
    device: str = "cuda",
    svd_rank: int = 32,
    smooth_alpha: float = 0.5,
    search_alpha: bool = True,
    alpha_candidates: list[float] | None = None,
    alpha_search_max_samples: int = 8,
    use_gptq: bool = False,
    gptq_n_bits: int = 4,
    gptq_symmetric: bool = True,
    gptq_group_size: int = -1,
    gptq_blocksize: int = 128,
    gptq_percdamp: float = 0.01,
    gptq_actorder: bool = False,
    n_gen_steps: int = 50,
    native_inference: bool = False,
    native_linear_mode: str = "mxfp4",
    svdquant_overlap_streams: bool = False,
) -> dict:
    """Test a single quantization mode. Returns a dict with test results."""
    logger.info("\n" + "=" * 80)
    logger.info(f"Testing quantization mode: {mode.upper()}")
    logger.info("=" * 80)

    start_time = time.time()

    output_dir = os.path.join(output_base_dir, mode)
    os.makedirs(output_dir, exist_ok=True)

    svd_exclude = SVDQUANT_EXCLUDE_PATTERNS.get(model_type, SVDQUANT_EXCLUDE_PATTERNS["sdxl"])
    svd_config = SVDQuantConfig(
        name="svdquant",
        svd_rank=svd_rank,
        smooth_alpha=smooth_alpha,
        search_alpha=search_alpha,
        alpha_candidates=alpha_candidates,
        alpha_search_max_samples=alpha_search_max_samples,
        exclude_patterns=svd_exclude,
        min_layer_size=256,
        use_gptq=use_gptq,
        gptq_n_bits=gptq_n_bits,
        gptq_symmetric=gptq_symmetric,
        gptq_group_size=gptq_group_size,
        gptq_blocksize=gptq_blocksize,
        gptq_percdamp=gptq_percdamp,
        gptq_actorder=gptq_actorder,
    )

    try:
        quant_layer_config = build_quant_layer_config(mode)
        logger.info(f"Quantization config for {mode}:")
        logger.info(f"  Weight spec: {quant_layer_config.weight}")
        if quant_layer_config.input_tensors:
            logger.info(f"  Activation spec: {quant_layer_config.input_tensors}")
        else:
            logger.info("  Activation spec: None (FP16)")
    except Exception as e:
        logger.error(f"Failed to build quantization config for {mode}: {e}")
        return {
            "mode": mode,
            "success": False,
            "error": str(e),
            "time": 0,
        }

    # Conv2d layers in UNet: strip activation quant when present, and exclude
    # from weight quantization when group_size is incompatible with kernel dims
    # (e.g. int4_wo_64 with ch_axis=-1 vs 3×3 kernels where dim[-1]=3).
    layer_type_overrides: dict[type[nn.Module], QLayerConfig] = {}
    if model_type == "sdxl":
        weight_spec = quant_layer_config.weight
        group_size = getattr(weight_spec, "group_size", None)
        if group_size is not None and group_size > 3:
            layer_type_overrides[nn.Conv2d] = QLayerConfig(weight=None)
            logger.info(f"  Conv2d override: excluded (group_size={group_size} incompatible with 3×3 kernels)")
        elif quant_layer_config.input_tensors is not None:
            conv_weight_only = QLayerConfig(weight=quant_layer_config.weight)
            layer_type_overrides[nn.Conv2d] = conv_weight_only
            logger.info("  Conv2d override: weight-only (no activation quantization)")

    layer_name_overrides: dict[str, QLayerConfig] = {}
    w4a16_patterns = QUANT_W4A16_OVERRIDE_PATTERNS.get(model_type, [])
    if quant_layer_config.input_tensors is not None and w4a16_patterns:
        weight_only_config = QLayerConfig(weight=quant_layer_config.weight)
        for pattern in w4a16_patterns:
            layer_name_overrides[pattern] = weight_only_config
        logger.info(f"  W4A16 overrides: {w4a16_patterns}")

    quant_exclude = QUANT_EXCLUDE_PATTERNS.get(model_type, QUANT_EXCLUDE_PATTERNS["sdxl"])
    quant_config = QConfig(
        global_quant_config=quant_layer_config,
        layer_type_quant_config=layer_type_overrides,
        layer_quant_config=layer_name_overrides,
        exclude=list(quant_exclude),
    )

    logger.info(f"Loading fresh pipeline for {mode}...")
    test_pipe = None
    test_pipe, _ = load_pipeline(model_id, device)

    logger.info(f"Applying SVDQuant + {mode} quantization...")
    try:
        quantized_module = quantize_model_with_svdquant(
            pipe=test_pipe,
            module_name=module_name,
            model_type=model_type,
            dataloader=calib_dataset,
            quant_config=quant_config,
            svd_config=svd_config,
            device=device,
        )

        set_quantize_target(test_pipe, model_type, module_name, quantized_module)

        logger.info("✓ Quantization successful")

    except Exception as e:
        logger.error(f"✗ ERROR during quantization: {e}")
        import traceback

        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "mode": mode,
            "success": False,
            "error": str(e),
            "time": time.time() - start_time,
        }

    if native_inference:
        # Save the canonical checkpoint BEFORE converting to native inference, so
        # the saved model matches the non-native run and loads via the standard
        # SVDQuant path. (disable_native_inference() before save would instead
        # drop the residual on export, since the reverted residual is a
        # QParamsLinear that export does not re-emit inside the ErrorCorrectedModule.)
        _save_quantized_model(quantized_module, module_name, output_dir)
        print(f"Enabling native inference (mode={native_linear_mode}, overlap_streams={svdquant_overlap_streams})...")
        try:
            n_native = enable_native_inference(
                quantized_module,
                runtime_options=RuntimeOptions(
                    native_linear_mode=native_linear_mode,
                    svdquant_overlap_streams=svdquant_overlap_streams,
                ),
            )
            print(f"✓ Native inference enabled for {n_native} layers")
        except Exception as e:
            print(f"✗ ERROR enabling native inference: {e}")
            import traceback

            traceback.print_exc()

    logger.info("Generating test image...")
    try:
        output_image = generate_image(
            pipe=test_pipe,
            prompt=test_prompt,
            model_type=model_type,
            n_steps=n_gen_steps,
            seed=42,
            device=device,
        )

        image_path = os.path.join(output_dir, "test_output.png")
        output_image.save(image_path)
        logger.info(f"✓ Image saved to: {image_path}")

        img_array = np.array(output_image)
        mean_intensity = img_array.mean()
        std_intensity = img_array.std()

        is_valid = mean_intensity > 10 and mean_intensity < 245 and std_intensity > 10
        if is_valid:
            logger.info(f"✓ Image appears valid (mean={mean_intensity:.1f}, std={std_intensity:.1f})")
        else:
            logger.warning(f"⚠ WARNING: Image may be invalid (mean={mean_intensity:.1f}, std={std_intensity:.1f})")

    except Exception as e:
        logger.error(f"✗ ERROR during image generation: {e}")
        import traceback

        traceback.print_exc()
        is_valid = False
        image_path = None

    if not native_inference:
        # Non-native run: save after generation. (Native runs save the canonical
        # checkpoint before enabling native inference, above.)
        _save_quantized_model(quantized_module, module_name, output_dir)

    with contextlib.suppress(NameError):
        del test_pipe
    with contextlib.suppress(NameError):
        del quantized_module
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - start_time

    result = {
        "mode": mode,
        "success": True,
        "time": elapsed,
        "image_path": image_path if "image_path" in locals() else None,
        "image_valid": is_valid if "is_valid" in locals() else False,
        "output_dir": output_dir,
    }

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Mode {mode.upper()} completed in {elapsed:.1f}s")
    logger.info(f"Status: {'✓ SUCCESS' if result['success'] else '✗ FAILED'}")
    logger.info(f"{'=' * 80}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test SVDQuant with multiple quantization modes for diffusion models (SDXL, FLUX.1-dev)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Model ID from HuggingFace (e.g. stabilityai/stable-diffusion-xl-base-1.0, black-forest-labs/FLUX.1-dev)",
    )
    parser.add_argument(
        "--module_name",
        type=str,
        default=None,
        choices=["unet", "transformer", "vae"],
        help="Which module to quantize (auto-detected from model_id when omitted: "
        "unet for SDXL, transformer for FLUX/SD3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["w4a16", "w4a4", "mxfp4"],
        choices=list(QUANT_MODE_TO_SCHEME.keys()),
        help="Quantization modes to test",
    )
    parser.add_argument(
        "--svd_rank",
        type=int,
        default=32,
        help="SVD rank for low-rank decomposition",
    )
    parser.add_argument(
        "--smooth_alpha",
        type=float,
        default=0.5,
        help="Alpha for activation smoothing",
    )
    parser.add_argument(
        "--search_alpha",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Search per-layer smoothing alpha to minimise post-SVD MSE (default: disabled)",
    )
    parser.add_argument(
        "--alpha_candidates",
        type=float,
        nargs="+",
        default=None,
        help="Custom alpha candidates for search (space-separated floats, default: linspace 0.05..0.95)",
    )
    parser.add_argument(
        "--alpha_search_max_samples",
        type=int,
        default=8,
        help="Number of cached input activations per layer for alpha search (default: 8)",
    )
    parser.add_argument(
        "--use_gptq",
        action="store_true",
        help="Use GPTQ (instead of RTN) to quantize residual weights",
    )
    parser.add_argument(
        "--gptq_n_bits",
        type=int,
        default=4,
        help="GPTQ target bit-width (default: 4)",
    )
    parser.add_argument(
        "--gptq_symmetric",
        type=bool,
        default=True,
        help="GPTQ symmetric quantization (default: True)",
    )
    parser.add_argument(
        "--gptq_group_size",
        type=int,
        default=-1,
        help="GPTQ group size (-1 = per-channel, >0 = per-group)",
    )
    parser.add_argument(
        "--gptq_blocksize",
        type=int,
        default=128,
        help="GPTQ block size (columns processed per block)",
    )
    parser.add_argument(
        "--gptq_percdamp",
        type=float,
        default=0.01,
        help="GPTQ Hessian damping percentage",
    )
    parser.add_argument(
        "--gptq_actorder",
        action="store_true",
        help="GPTQ activation ordering (reorder columns by importance)",
    )
    parser.add_argument(
        "--use_coco2014",
        action="store_true",
        help="Use COCO2014 captions as calibration prompts (see COCO2014_SETUP.md)",
    )
    parser.add_argument(
        "--coco2014_dir",
        type=str,
        default=DEFAULT_COCO2014_DIR,
        help="Path to coco2014/ directory containing captions/captions_source.tsv",
    )
    parser.add_argument(
        "--n_calib_prompts",
        type=int,
        default=3,
        help="Number of calibration prompts to use (fewer = faster)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=10,
        help="Number of inference steps for calibration (fewer = faster)",
    )
    parser.add_argument(
        "--n_gen_steps",
        type=int,
        default=None,
        help="Number of inference steps for post-quantization test image generation "
        "(default: 50 for FLUX, 30 for others)",
    )
    parser.add_argument(
        "--native_inference",
        action="store_true",
        help="After quantization, enable Aiter native inference (replaces ErrorCorrectedModule / "
        "QParamsLinear layers with fused native kernels) before generating the test image.",
    )
    parser.add_argument(
        "--native_linear_mode",
        type=str,
        default="mxfp4",
        choices=["auto", "fp8_per_tensor", "mxfp4"],
        help="Native inference linear mode (only used with --native_inference).",
    )
    parser.add_argument(
        "--svdquant_overlap_streams",
        action="store_true",
        help="Run the SVDQuant low-rank correction on a second CUDA stream to overlap the residual GEMM.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_svdquant_results",
        help="Base directory to save results",
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default=DEFAULT_TEST_PROMPT,
        help="Test prompt to generate image after quantization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info(f"\n[1/4] Loading pipeline: {args.model_id}")
    pipe, model_type = load_pipeline(args.model_id, args.device)
    logger.info(f"Pipeline loaded (type: {model_type})")

    if args.module_name is None:
        args.module_name = DEFAULT_MODULE_FOR_MODEL.get(model_type, "unet")
        logger.info(f"Auto-detected module to quantize: {args.module_name}")

    if args.n_gen_steps is None:
        args.n_gen_steps = 50 if model_type == "flux" else 30

    calib_source = "COCO2014" if args.use_coco2014 else "built-in"
    rounding_method = "GPTQ" if args.use_gptq else "RTN"
    model_label = model_type.upper()
    logger.info("=" * 80)
    logger.info(f"SVDQuant Multi-Mode Test for {model_label}")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Module: {args.module_name}")
    logger.info(f"Modes to test: {', '.join(args.modes)}")
    logger.info(f"SVD Rank: {args.svd_rank}")
    logger.info(f"Smooth Alpha: {args.smooth_alpha}")
    alpha_search_label = "per-layer search" if args.search_alpha else "global (fixed)"
    logger.info(f"Alpha mode: {alpha_search_label}")
    if args.search_alpha:
        logger.info(f"  Alpha search samples: {args.alpha_search_max_samples}")
        if args.alpha_candidates:
            logger.info(f"  Alpha candidates: {args.alpha_candidates}")
    logger.info(f"Residual rounding: {rounding_method}")
    if args.use_gptq:
        logger.info(
            f"  GPTQ: n_bits={args.gptq_n_bits}, symmetric={args.gptq_symmetric}, "
            f"group_size={args.gptq_group_size}, blocksize={args.gptq_blocksize}, "
            f"percdamp={args.gptq_percdamp}, actorder={args.gptq_actorder}"
        )
    logger.info(f"SVDQuant excludes: {SVDQUANT_EXCLUDE_PATTERNS.get(model_type, ['(default)'])}")
    logger.info(f"Quant excludes: {QUANT_EXCLUDE_PATTERNS.get(model_type, ['(default)'])}")
    logger.info(f"Calibration: {args.n_calib_prompts} prompts x {args.n_steps} steps ({calib_source})")
    logger.info(f"Generation steps: {args.n_gen_steps}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_coco2014:
        calib_prompts = load_coco2014_prompts(
            coco_dir=args.coco2014_dir,
            max_prompts=args.n_calib_prompts,
        )
    else:
        calib_prompts = DEFAULT_CALIB_PROMPTS[: args.n_calib_prompts]

    logger.info("\n[2/4] Collecting calibration data...")
    logger.info(f"Using {len(calib_prompts)} prompts ({calib_source}):")
    for i, prompt in enumerate(calib_prompts, 1):
        logger.info(f"  {i}. {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    calib_dataset = collect_calibration_data(
        pipe=pipe,
        module_name=args.module_name,
        model_type=model_type,
        prompts=calib_prompts,
        n_steps=args.n_steps,
        device=args.device,
        max_captures=200,
    )
    logger.info(f"✓ Collected {len(calib_dataset)} calibration samples")

    logger.info(f"\n[3/4] Testing {len(args.modes)} quantization modes...")
    results = []

    for i, mode in enumerate(args.modes, 1):
        logger.info(f"\n--- Mode {i}/{len(args.modes)}: {mode.upper()} ---")

        result = test_quantization_mode(
            pipe=pipe,
            model_id=args.model_id,
            model_type=model_type,
            module_name=args.module_name,
            mode=mode,
            calib_dataset=calib_dataset,
            output_base_dir=args.output_dir,
            test_prompt=args.test_prompt,
            device=args.device,
            svd_rank=args.svd_rank,
            smooth_alpha=args.smooth_alpha,
            search_alpha=args.search_alpha,
            alpha_candidates=args.alpha_candidates,
            alpha_search_max_samples=args.alpha_search_max_samples,
            use_gptq=args.use_gptq,
            gptq_n_bits=args.gptq_n_bits,
            gptq_symmetric=args.gptq_symmetric,
            gptq_group_size=args.gptq_group_size,
            gptq_blocksize=args.gptq_blocksize,
            gptq_percdamp=args.gptq_percdamp,
            gptq_actorder=args.gptq_actorder,
            n_gen_steps=args.n_gen_steps,
            native_inference=args.native_inference,
            native_linear_mode=args.native_linear_mode,
            svdquant_overlap_streams=args.svdquant_overlap_streams,
        )

        results.append(result)

    logger.info("\n" + "=" * 80)
    logger.info("[4/4] SUMMARY")
    logger.info("=" * 80)

    success_count = sum(1 for r in results if r["success"])

    logger.info(f"\nTested {len(results)} modes, {success_count} succeeded\n")

    for result in results:
        mode = result["mode"]
        status = "✓" if result["success"] else "✗"
        time_str = f"{result['time']:.1f}s" if result["success"] else "N/A"

        if result["success"]:
            img_status = "[Image OK]" if result.get("image_valid") else "[Image Warning]"
            logger.info(f"{status} {mode.upper():8s}  {time_str:>8s}  {img_status}  {result['output_dir']}")
        else:
            logger.info(f"{status} {mode.upper():8s}  {time_str:>8s}  Error: {result.get('error', 'Unknown')}")

    logger.info("\n" + "=" * 80)

    if success_count == len(results):
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("SVDQuant is working correctly in Quark for all tested modes.")
    elif success_count > 0:
        logger.warning(f"⚠ PARTIAL SUCCESS: {success_count}/{len(results)} modes passed")
        logger.warning("Some quantization modes may have issues.")
    else:
        logger.error("✗ ALL TESTS FAILED")
        logger.error("SVDQuant integration may have issues.")

    logger.info("=" * 80)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"SVDQuant Multi-Mode Test Summary ({model_label})\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.model_id}\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Module: {args.module_name}\n")
        f.write(f"Modes tested: {', '.join(args.modes)}\n")
        f.write(f"SVD Rank: {args.svd_rank}\n")
        f.write(f"Smooth Alpha: {args.smooth_alpha}\n")
        f.write(f"Residual rounding: {rounding_method}\n")
        if args.use_gptq:
            f.write(
                f"  GPTQ: n_bits={args.gptq_n_bits}, symmetric={args.gptq_symmetric}, "
                f"group_size={args.gptq_group_size}, blocksize={args.gptq_blocksize}, "
                f"percdamp={args.gptq_percdamp}, actorder={args.gptq_actorder}\n"
            )
        f.write(f"Calibration: {len(calib_prompts)} prompts x {args.n_steps} steps ({calib_source})\n")
        f.write(f"Generation steps: {args.n_gen_steps}\n\n")
        f.write("Results:\n")
        for result in results:
            f.write(
                f"  {result['mode']:8s}: ",
            )
            if result["success"]:
                f.write(f"SUCCESS ({result['time']:.1f}s)\n")
            else:
                f.write(f"FAILED - {result.get('error', 'Unknown')}\n")
        f.write(f"\nSuccess rate: {success_count}/{len(results)}\n")

    logger.info(f"\nSummary saved to: {summary_path}")

    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
