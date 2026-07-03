#
# Copyright (C) 2025 - 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quantize and evaluate diffusion models with AMD Quark.

This script uses ``quark.torch.utils.diffusers.get_calib_dataloader`` to
collect calibration data directly from a pipeline run (capturing the target
submodule's inputs as dicts keyed by its forward parameter names), then
quantizes the submodule with ``ModelQuantizer``.  It supports SDXL and
SD1.5 + ControlNet, the models exercised by the diffusers regression CI.

Quantization schemes (``--quant_scheme``):

* ``w_int8_per_tensor_sym`` -- INT8 per-tensor weight-only
* ``w_int8_a_int8``         -- INT8 per-tensor weight + activation (w8a8)
* ``w_fp8_a_fp8``           -- FP8 E4M3 per-tensor weight + activation
* ``svdquant_w4a16``        -- SVDQuant INT4 weight-only with low-rank correction

Evaluation reuses the MLPerf COCO2014 CLIP/FID harness (the ``tools.clip``
and ``tools.fid`` modules from the mlcommons inference repo must be on
``PYTHONPATH``), preserving the regression golden-data contract.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from quark.torch import ModelQuantizer, save_params
from quark.torch.quantization import (
    FP8E4M3PerTensorSpec,
    Int4PerChannelSpec,
    Int8PerTensorSpec,
)
from quark.torch.quantization.config.config import QConfig, QLayerConfig, SVDQuantConfig
from quark.torch.utils.diffusers import get_calib_dataloader

FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()
INT8_PER_TENSOR_SPEC = Int8PerTensorSpec(
    observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
).to_quantization_spec()
INT4_PER_CHANNEL_SPEC = Int4PerChannelSpec(
    symmetric=True, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
).to_quantization_spec()

DEFAULT_NEGATIVE_PROMPT = "normal quality, low quality, worst quality, low res, blurry, nsfw, nude."

# Per-model layers to exclude from quantization (sensitive embedding / conv
# layers).  Carried over from the previous per-model JSON configs -- this is the
# tuned knowledge that keeps quantized image quality acceptable.  Reformatted
# from models/*.json into an in-script table so there is no external-file
# indirection.
EXCLUDE_LAYERS = {
    "sdxl": ["*time_embedding*", "*time_emb_proj*", "*conv_in*", "*conv_out*", "*conv_shortcut*", "*add_embedding*"],
    "sd15": [],
}

# SVDQuant decomposition exclude patterns (the SVD branch is skipped for these),
# also from the previous JSON configs.
SVDQUANT_EXCLUDE = {
    "sdxl": ["time_embedding", "add_time_proj", "conv_in", "conv_out", "time_proj", "add_embedding"],
    "sd15": ["time_embedding", "conv_in", "conv_out", "time_proj"],
}


def build_qconfig(quant_scheme: str, model_family: str) -> QConfig:
    """Build a ``QConfig`` for the requested scheme, applying per-model excludes."""
    exclude = EXCLUDE_LAYERS.get(model_family, [])
    if quant_scheme == "w_int8_per_tensor_sym":
        return QConfig(global_quant_config=QLayerConfig(weight=INT8_PER_TENSOR_SPEC), exclude=list(exclude))
    if quant_scheme == "w_int8_a_int8":
        return QConfig(
            global_quant_config=QLayerConfig(weight=INT8_PER_TENSOR_SPEC, input_tensors=INT8_PER_TENSOR_SPEC),
            exclude=list(exclude),
        )
    if quant_scheme == "w_fp8_a_fp8":
        return QConfig(
            global_quant_config=QLayerConfig(weight=FP8_PER_TENSOR_SPEC, input_tensors=FP8_PER_TENSOR_SPEC),
            exclude=list(exclude),
        )
    if quant_scheme == "svdquant_w4a16":
        svd_exclude = SVDQUANT_EXCLUDE.get(model_family, [])
        return QConfig(
            global_quant_config=QLayerConfig(weight=INT4_PER_CHANNEL_SPEC),
            exclude=[*exclude, "*correction*"],
            algo_config=[
                SVDQuantConfig(svd_rank=32, search_alpha=False, min_layer_size=256, exclude_patterns=svd_exclude)
            ],
        )
    raise ValueError(f"Unsupported quant_scheme: {quant_scheme}")


def load_coco_prompts(tsv_path: str, limit: int | None = None) -> list[str]:
    """Read prompts from an MLPerf COCO2014 captions TSV (column 2, skip header)."""
    prompts: list[str] = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            cols = line.split("\t")
            if len(cols) >= 3:
                prompts.append(cols[2].strip())
    return prompts[:limit] if limit else prompts


def build_pipeline(args: argparse.Namespace) -> tuple[object, str, str]:
    """Load the pipeline and return ``(pipe, target_submodule_name, model_family)``."""
    if args.controlnet_id:
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

        controlnet = ControlNetModel.from_pretrained(args.controlnet_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            args.model_id, controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(args.device)
        return pipe, "unet", "sd15"

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(args.device)
    return pipe, "unet", "sdxl"


def make_canny(image_path: str, size: int = 512) -> Image.Image:
    """Canny-edge preprocess for ControlNet conditioning.

    The conditioning image is resized to ``size`` x ``size`` so SD1.5
    ControlNet generates at its native resolution; otherwise a large source
    image (e.g. 1024x1024) drives 1024px generation and blows up UNet
    activation memory.
    """
    import cv2
    from diffusers.utils import load_image

    image = load_image(image_path).convert("RGB").resize((size, size))
    edges = cv2.Canny(np.array(image), 100, 200)[:, :, None]
    return Image.fromarray(np.concatenate([edges, edges, edges], axis=2))


def make_generator(seed: int) -> torch.Generator:
    """Fixed CPU generator for reproducible, cross-GPU-stable generation.

    Sampling the initial noise on CPU with a fixed seed yields identical
    latents on every GPU architecture (unlike an on-device generator), which
    is what keeps the regression golden CLIP/FID stable across the
    heterogeneous CI runner pool.  It is also model-agnostic -- the pipeline
    samples the correct latent shape for each model.
    """
    return torch.Generator(device="cpu").manual_seed(seed)


def generate_image(
    pipe,
    prompt: str,
    args: argparse.Namespace,
    control_image: Image.Image | None = None,
    generator: torch.Generator | None = None,
):
    """Generate a single image, handling the ControlNet vs plain path."""
    if args.controlnet_id:
        return pipe(
            prompt=[prompt],
            negative_prompt=[DEFAULT_NEGATIVE_PROMPT],
            image=control_image,
            num_inference_steps=args.n_steps,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            generator=generator,
            guidance_scale=8.0,
        ).images[0]
    return pipe(
        prompt=[prompt],
        negative_prompt=[DEFAULT_NEGATIVE_PROMPT],
        num_inference_steps=args.n_steps,
        generator=generator,
        guidance_scale=8.0,
    ).images[0]


def quantize_submodule(pipe, target_name: str, args: argparse.Namespace, model_family: str) -> None:
    """Calibrate and quantize ``pipe.<target_name>`` in place."""
    target = getattr(pipe, target_name)
    prompts = load_coco_prompts(args.calib_prompts, args.calib_size)

    pipe_kwargs: dict[str, object] = {"guidance_scale": 8.0}
    if args.controlnet_id:
        pipe_kwargs["image"] = make_canny(args.input_image)
        pipe_kwargs["controlnet_conditioning_scale"] = args.controlnet_conditioning_scale

    dataloader = get_calib_dataloader(pipe, target, prompts, n_steps=args.n_steps, **pipe_kwargs)
    qconfig = build_qconfig(args.quant_scheme, model_family)
    setattr(pipe, target_name, ModelQuantizer(qconfig).quantize_model(target, dataloader))


@torch.no_grad()
def evaluate_coco(pipe, args: argparse.Namespace) -> None:
    """Generate images for COCO2014 test prompts and report CLIP + FID."""
    os.makedirs(args.save_images_dir, exist_ok=True)
    test_prompts = load_coco_prompts(args.test_prompts, args.test_size)

    control_image = make_canny(args.input_image) if args.controlnet_id else None
    images: list[np.ndarray] = []
    for idx, prompt in enumerate(tqdm(test_prompts, desc="Generating images")):
        # Per-image fixed seed: deterministic and cross-GPU stable, while
        # still giving each prompt distinct noise.
        generator = make_generator(args.seed + idx)
        image = generate_image(pipe, prompt, args, control_image, generator=generator)
        image.save(os.path.join(args.save_images_dir, f"{idx}.png"))
        images.append(np.array(image, dtype=np.uint8))

    from tools.clip.clip_encoder import CLIPEncoder
    from tools.fid.fid_score import compute_fid

    clip = CLIPEncoder(device=torch.device("cuda"))
    clip_scores = [
        100 * clip.get_clip_score(prompt, Image.fromarray(img)).item()
        for prompt, img in zip(test_prompts, images, strict=True)
    ]
    print("clip_score:", float(np.mean(clip_scores)))

    statistics_path = "./inference/text_to_image/tools/val2014.npz"
    print("fid:", compute_fid(images, statistics_path, torch.device("cuda")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize and evaluate diffusion models with Quark.")
    parser.add_argument("--model_id", required=True, help="HF model id, e.g. stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--controlnet_id", default=None, help="Optional ControlNet model id (SD1.5 canny).")
    parser.add_argument(
        "--input_image",
        default="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png",
        help="Conditioning image (path or URL) used for ControlNet calibration and generation.",
    )
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.5)
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--n_steps", type=int, default=20, help="Denoising steps per generation/calibration run.")

    parser.add_argument(
        "--quant_scheme",
        default="w_int8_per_tensor_sym",
        choices=["w_int8_per_tensor_sym", "w_int8_a_int8", "w_fp8_a_fp8", "svdquant_w4a16"],
    )
    parser.add_argument("--skip_quantization", action="store_true", help="Run the fp16 baseline (no quantization).")
    parser.add_argument("--calib_prompts", default="", help="COCO2014 calibration captions TSV.")
    parser.add_argument("--calib_size", type=int, default=50, help="Number of calibration prompts.")

    parser.add_argument("--export", default=None, choices=["safetensor", None])
    parser.add_argument("--export_path", default="./quantized_models")

    parser.add_argument("--test", action="store_true", help="Run the COCO2014 CLIP/FID evaluation.")
    parser.add_argument("--test_prompts", default="", help="COCO2014 test captions TSV.")
    parser.add_argument("--test_size", type=int, default=50, help="Number of test prompts.")
    parser.add_argument("--save_images_dir", default="test_coco2014_result")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    pipe, target_name, model_family = build_pipeline(args)

    if not args.skip_quantization:
        print(f"[INFO] Quantizing {target_name} ({args.quant_scheme}) ...")
        quantize_submodule(pipe, target_name, args, model_family)

        if args.export == "safetensor":
            print(f"[INFO] Exporting {target_name} to safetensors -> {args.export_path}")
            frozen = ModelQuantizer.freeze(getattr(pipe, target_name))
            save_params(frozen, model_type=target_name, export_dir=args.export_path)

    if args.test:
        print("[INFO] Evaluating on COCO2014 (CLIP + FID) ...")
        evaluate_coco(pipe, args)


if __name__ == "__main__":
    main()
