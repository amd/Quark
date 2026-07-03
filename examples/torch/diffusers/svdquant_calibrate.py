#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""SVDQuant calibration and grid-search driver for diffusion models.

This example helps you pick the SVDQuant settings that matter most for the
quality / speed trade-off, by sweeping them and scoring each configuration:

  * the per-layer smoothing *alpha* (searched per layer when ``--search_alpha``
    is set, otherwise a fixed ``--smooth_alpha``);
  * GPTQ on/off for the residual weights (use ``--gptq both`` to compare);
  * the number of calibration samples (sweepable via ``--n_calib_samples``).

For every ``(gptq, n_samples)`` cell in the grid it applies SVDQuant and then
quantizes on a fresh pipeline -- using the same ``SVDQuantProcessor.apply()``
followed by ``ModelQuantizer.quantize_model()`` flow as ``quantize_diffusers.py``
and ``testSVDQuant.py`` -- evaluates the result against a high-precision
reference, and writes a ranked table so you can pick the best configuration.

The per-layer ``alpha`` search uses only ``--alpha_search_max_samples`` activations
(the full calibration set still drives smoothing and the GPTQ Hessian), so it stays
cheap; this script just exposes ``--search_alpha`` and sweeps the other choices.

Examples
--------
    # FLUX.1-dev, w4a16: per-layer alpha search, compare GPTQ off vs on:
    python svdquant_calibrate.py \\
        --model_id black-forest-labs/FLUX.1-dev --mode w4a16 \\
        --gptq both --n_calib_samples 128 --eval_metric ref_image

    # SDXL, mxfp4, sweep the number of calibration samples, GPTQ off:
    python svdquant_calibrate.py \\
        --model_id stabilityai/stable-diffusion-xl-base-1.0 --mode mxfp4 \\
        --gptq off --n_calib_samples 64 128 256

    # Fast proxy (no image generation) - rank configs by submodule output MSE:
    python svdquant_calibrate.py --gptq both --eval_metric module_mse
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import re
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

# Make the sibling testSVDQuant.py importable and add the Quark root to path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
QUARK_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if os.path.isdir(QUARK_ROOT) and QUARK_ROOT not in sys.path:
    sys.path.insert(0, QUARK_ROOT)

from testSVDQuant import (  # noqa: E402
    DEFAULT_CALIB_PROMPTS,
    DEFAULT_COCO2014_DIR,
    DEFAULT_MODULE_FOR_MODEL,
    QUANT_EXCLUDE_PATTERNS,
    QUANT_W4A16_OVERRIDE_PATTERNS,
    SVDQUANT_EXCLUDE_PATTERNS,
    collect_calibration_data,
    generate_image,
    get_quantize_target,
    load_coco2014_prompts,
    load_pipeline,
    set_quantize_target,
)

from quark.common.utils.log import ScreenLogger  # noqa: E402
from quark.torch import ModelQuantizer  # noqa: E402
from quark.torch.algorithm.svdquant.svdquant import (  # noqa: E402
    QUANT_MODE_TO_SCHEME,
    SVDQuantProcessor,
    build_quant_layer_config,
)
from quark.torch.quantization.config.config import (  # noqa: E402
    QConfig,
    QLayerConfig,
    SVDQuantConfig,
)

logger = ScreenLogger(__name__)

DEFAULT_EVAL_PROMPTS = [
    "A serene mountain lake at sunset with snow-capped peaks reflecting in the water",
    "A close-up portrait of an elderly fisherman with weathered skin, dramatic lighting",
]


# ---------------------------------------------------------------------------
# Config builders (mirror quantize_diffusers.py / testSVDQuant.py)
# ---------------------------------------------------------------------------


def build_qconfig(model_type: str, mode: str) -> QConfig:
    """Build the post-SVDQuant ``QConfig`` for ``mode``, mirroring testSVDQuant.

    Carries the SDXL Conv2d and Flux W4A16 overrides so the low-bit pass does
    not trip over incompatible group sizes / activation specs.
    """
    quant_layer_config = build_quant_layer_config(mode)

    layer_type_overrides: dict[type[nn.Module], QLayerConfig] = {}
    if model_type == "sdxl":
        weight_spec = quant_layer_config.weight
        group_size = getattr(weight_spec, "group_size", None)
        if group_size is not None and group_size > 3:
            layer_type_overrides[nn.Conv2d] = QLayerConfig(weight=None)
        elif quant_layer_config.input_tensors is not None:
            layer_type_overrides[nn.Conv2d] = QLayerConfig(weight=quant_layer_config.weight)

    layer_name_overrides: dict[str, QLayerConfig] = {}
    w4a16_patterns = QUANT_W4A16_OVERRIDE_PATTERNS.get(model_type, [])
    if quant_layer_config.input_tensors is not None and w4a16_patterns:
        weight_only = QLayerConfig(weight=quant_layer_config.weight)
        for pattern in w4a16_patterns:
            layer_name_overrides[pattern] = weight_only

    quant_exclude = QUANT_EXCLUDE_PATTERNS.get(model_type, QUANT_EXCLUDE_PATTERNS["sdxl"])
    return QConfig(
        global_quant_config=quant_layer_config,
        layer_type_quant_config=layer_type_overrides,
        layer_quant_config=layer_name_overrides,
        exclude=list(quant_exclude),
    )


def make_svd_config(
    model_type: str,
    args: argparse.Namespace,
    *,
    search_alpha: bool,
    use_gptq: bool,
) -> SVDQuantConfig:
    svd_exclude = SVDQUANT_EXCLUDE_PATTERNS.get(model_type, SVDQUANT_EXCLUDE_PATTERNS["sdxl"])
    return SVDQuantConfig(
        name="svdquant",
        svd_rank=args.svd_rank,
        smooth_alpha=args.smooth_alpha,
        search_alpha=search_alpha,
        alpha_candidates=list(args.alpha_candidates) if args.alpha_candidates else None,
        alpha_search_max_samples=args.alpha_search_max_samples,
        exclude_patterns=list(svd_exclude),
        min_layer_size=args.min_layer_size,
        use_gptq=use_gptq,
        gptq_n_bits=args.gptq_n_bits,
        gptq_symmetric=args.gptq_symmetric,
        gptq_group_size=args.gptq_group_size,
        gptq_blocksize=args.gptq_blocksize,
        gptq_percdamp=args.gptq_percdamp,
        gptq_actorder=args.gptq_actorder,
    )


def slice_loader(master_loader: DataLoader, n: int, device: str) -> tuple[DataLoader, int]:
    """Return a DataLoader over the first ``n`` captures of the master pool."""
    captured = master_loader.dataset.captured
    n = max(1, min(n, len(captured)))
    dataset_cls = type(master_loader.dataset)
    dataset = dataset_cls(captured[:n], torch.device(device))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=master_loader.collate_fn)
    return loader, n


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _mse_psnr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return mse, float("inf")
    psnr = 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)
    return mse, psnr


class _LPIPS:
    """Optional LPIPS scorer; no-op if the ``lpips`` package is unavailable."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.model = None
        try:
            import lpips  # type: ignore

            self.model = lpips.LPIPS(net="alex").to(device).eval()
            logger.info("LPIPS enabled (net=alex)")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"LPIPS unavailable ({exc}); skipping LPIPS")

    def __call__(self, a_img: Image.Image, b_img: Image.Image) -> float | None:
        if self.model is None:
            return None

        def prep(img: Image.Image) -> torch.Tensor:
            arr = _to_rgb_array(img) / 127.5 - 1.0
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            return float(self.model(prep(a_img), prep(b_img)).item())


def _extract_tensor(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if isinstance(out, tuple | list):
        return _extract_tensor(out[0])
    if hasattr(out, "sample"):
        return out.sample
    raise TypeError(f"Cannot extract a tensor from submodule output of type {type(out)!r}")


@torch.no_grad()
def module_reference_outputs(target: nn.Module, probe_dicts: list[dict[str, Any]]) -> list[torch.Tensor]:
    outs: list[torch.Tensor] = []
    for data in probe_dicts:
        outs.append(_extract_tensor(target(**data)).detach().float().cpu())
    return outs


@torch.no_grad()
def module_output_mse(
    target: nn.Module,
    probe_dicts: list[dict[str, Any]],
    ref_outs: list[torch.Tensor],
) -> float:
    total = 0.0
    for data, ref in zip(probe_dicts, ref_outs, strict=False):
        cur = _extract_tensor(target(**data)).detach().float().cpu()
        total += float(((cur - ref) ** 2).mean().item())
    return total / max(1, len(probe_dicts))


def _slug(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return s[:maxlen] or "img"


# ---------------------------------------------------------------------------
# Grid cell: SVDQuant + quantize + score (canonical two-step flow)
# ---------------------------------------------------------------------------


def run_cell(
    *,
    args: argparse.Namespace,
    model_type: str,
    module_name: str,
    master_loader: DataLoader,
    reference: list[dict[str, Any]],
    probe_dicts: list[dict[str, Any]] | None,
    ref_outs: list[torch.Tensor] | None,
    lpips_fn: _LPIPS | None,
    use_gptq: bool,
    n_samples: int,
    n_gen_steps: int,
    out_dir: str,
) -> dict[str, Any]:
    tag = f"gptq{'On' if use_gptq else 'Off'}_n{n_samples}"
    cell_dir = os.path.join(out_dir, tag)
    os.makedirs(cell_dir, exist_ok=True)

    logger.info(f"CELL {tag}: gptq={use_gptq}, n_calib_samples={n_samples}, metric={args.eval_metric}")

    started = time.time()
    row: dict[str, Any] = {
        "tag": tag,
        "use_gptq": use_gptq,
        "n_calib_samples": n_samples,
        "mode": args.mode,
        "search_alpha": args.search_alpha,
        "success": False,
        "error": None,
        "score": None,
        "metrics": {},
    }

    pipe = None
    quantized = None
    try:
        pipe, _ = load_pipeline(args.model_id, args.device)
        target = get_quantize_target(pipe, model_type, module_name)
        loader, used = slice_loader(master_loader, n_samples, args.device)
        row["n_calib_samples"] = used

        # Apply SVDQuant, then quantize -- same flow as quantize_diffusers.py.
        svd_config = make_svd_config(model_type, args, search_alpha=args.search_alpha, use_gptq=use_gptq)
        SVDQuantProcessor(target, svd_config, loader).apply()

        qconfig = build_qconfig(model_type, args.mode)
        quantized = ModelQuantizer(qconfig).quantize_model(target, loader)
        set_quantize_target(pipe, model_type, module_name, quantized)

        if args.eval_metric == "ref_image":
            row["metrics"], row["score"] = _eval_ref_image(
                pipe, model_type, reference, lpips_fn, n_gen_steps, args, cell_dir
            )
        elif args.eval_metric == "module_mse":
            assert probe_dicts is not None and ref_outs is not None
            mm = module_output_mse(quantized, probe_dicts, ref_outs)
            row["metrics"] = {"module_mse": mm}
            row["score"] = mm
        else:  # "none"
            for ref in reference:
                img = generate_image(pipe, ref["prompt"], model_type, n_gen_steps, args.seed, args.device)
                img.save(os.path.join(cell_dir, _slug(ref["prompt"]) + ".png"))

        if args.save_models:
            _save_model(quantized, module_name, cell_dir)

        row["success"] = True
        logger.info(f"CELL {tag}: OK  score={row['score']}  metrics={row['metrics']}")

    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)
        logger.warning(f"CELL {tag} failed: {exc}")
        logger.debug(traceback.format_exc())
    finally:
        with contextlib.suppress(Exception):
            del quantized
        with contextlib.suppress(Exception):
            del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    row["seconds"] = round(time.time() - started, 1)
    return row


def _eval_ref_image(
    pipe: Any,
    model_type: str,
    reference: list[dict[str, Any]],
    lpips_fn: _LPIPS | None,
    n_gen_steps: int,
    args: argparse.Namespace,
    cell_dir: str,
) -> tuple[dict[str, float], float]:
    mses: list[float] = []
    psnrs: list[float] = []
    lpips_vals: list[float] = []
    for ref in reference:
        img = generate_image(pipe, ref["prompt"], model_type, n_gen_steps, args.seed, args.device)
        img.save(os.path.join(cell_dir, _slug(ref["prompt"]) + ".png"))
        mse, psnr = _mse_psnr(_to_rgb_array(img), ref["arr"])
        mses.append(mse)
        psnrs.append(psnr)
        if lpips_fn is not None:
            lp = lpips_fn(img, ref["pil"])
            if lp is not None:
                lpips_vals.append(lp)

    metrics: dict[str, float] = {
        "mean_mse": float(np.mean(mses)),
        "mean_psnr": float(np.mean([p for p in psnrs if math.isfinite(p)] or [float("inf")])),
    }
    if lpips_vals:
        metrics["mean_lpips"] = float(np.mean(lpips_vals))
    return metrics, metrics["mean_mse"]


def _save_model(quantized: nn.Module, module_name: str, cell_dir: str) -> None:
    from quark.torch import save_params

    frozen = ModelQuantizer.freeze(quantized)
    for _, param in frozen.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()
    for _, buffer in frozen.named_buffers():
        if not buffer.is_contiguous():
            buffer.data = buffer.data.contiguous()
    save_params(frozen, model_type=module_name, export_dir=cell_dir)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

GPTQ_CHOICES = {"off": [False], "on": [True], "both": [False, True]}
SCORE_DIRECTION = {"ref_image": "min", "module_mse": "min", "none": "none"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SVDQuant calibration + grid search (alpha / GPTQ / #samples) for diffusion models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model / quantization.
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev", help="HuggingFace model id")
    parser.add_argument("--module_name", default=None, choices=["unet", "transformer", "vae"], help="Auto if omitted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="w4a16", choices=list(QUANT_MODE_TO_SCHEME.keys()))
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--min_layer_size", type=int, default=256)

    # Calibration pool.
    parser.add_argument("--use_coco2014", action="store_true", help="Use COCO2014 captions for calibration prompts")
    parser.add_argument("--coco2014_dir", default=DEFAULT_COCO2014_DIR)
    parser.add_argument("--n_calib_prompts", type=int, default=8, help="Prompts used to build the master capture pool")
    parser.add_argument("--n_steps", type=int, default=20, help="Denoising steps per calibration prompt")

    # Smoothing alpha (per-layer search done inside SVDQuantProcessor).
    parser.add_argument("--smooth_alpha", type=float, default=0.5, help="Fixed alpha (used when --no-search_alpha)")
    parser.add_argument(
        "--search_alpha",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search per-layer alpha to minimize post-SVD MSE (uses --alpha_search_max_samples)",
    )
    parser.add_argument("--alpha_candidates", type=float, nargs="+", default=None, help="Alpha search grid")
    parser.add_argument(
        "--alpha_search_max_samples",
        type=int,
        default=8,
        help="Calibration samples used for the per-layer alpha search",
    )

    # Grid sweeps.
    parser.add_argument("--gptq", default="off", choices=list(GPTQ_CHOICES.keys()), help="Residual GPTQ on/off/both")
    parser.add_argument("--n_calib_samples", type=int, nargs="+", default=[128], help="Calibration-sample sweep")
    parser.add_argument("--gptq_n_bits", type=int, default=4)
    parser.add_argument("--gptq_symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gptq_group_size", type=int, default=-1)
    parser.add_argument("--gptq_blocksize", type=int, default=128)
    parser.add_argument("--gptq_percdamp", type=float, default=0.01)
    parser.add_argument("--gptq_actorder", action="store_true")

    # Evaluation.
    parser.add_argument("--eval_metric", default="ref_image", choices=list(SCORE_DIRECTION.keys()))
    parser.add_argument("--eval_prompts", nargs="+", default=None, help="Prompts for ref_image / none metrics")
    parser.add_argument("--n_eval_prompts", type=int, default=2)
    parser.add_argument("--module_mse_samples", type=int, default=4, help="Probe captures for module_mse")
    parser.add_argument("--n_gen_steps", type=int, default=None, help="Eval gen steps (default 50 flux / 30 other)")

    # Output.
    parser.add_argument("--output_dir", default="./svdquant_calib_results")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("SVDQuant calibration / grid search")
    logger.info(f"  model={args.model_id}  mode={args.mode}")
    logger.info(f"  gptq={args.gptq} -> {GPTQ_CHOICES[args.gptq]}  n_calib_samples={args.n_calib_samples}")
    logger.info(f"  search_alpha={args.search_alpha} (alpha_search_max_samples={args.alpha_search_max_samples})")
    logger.info(f"  eval_metric={args.eval_metric}")

    pipe, model_type = load_pipeline(args.model_id, args.device)
    module_name = args.module_name or DEFAULT_MODULE_FOR_MODEL.get(model_type, "unet")
    n_gen_steps = args.n_gen_steps if args.n_gen_steps is not None else (50 if model_type == "flux" else 30)
    target = get_quantize_target(pipe, model_type, module_name)
    logger.info(f"model_type={model_type}, module={module_name}, eval gen steps={n_gen_steps}")

    if args.use_coco2014:
        calib_prompts = load_coco2014_prompts(coco_dir=args.coco2014_dir, max_prompts=args.n_calib_prompts)
    else:
        base = DEFAULT_CALIB_PROMPTS
        calib_prompts = [base[i % len(base)] for i in range(args.n_calib_prompts)]

    # Reserve extra captures so module_mse can probe on a held-out (disjoint) tail.
    extra_probes = args.module_mse_samples if args.eval_metric == "module_mse" else 0
    max_needed = max(args.n_calib_samples) + extra_probes
    logger.info(f"[1/3] Collecting master calibration pool (up to {max_needed} captures) ...")
    master_loader = collect_calibration_data(
        pipe, module_name, model_type, calib_prompts, n_steps=args.n_steps, device=args.device, max_captures=max_needed
    )

    reference: list[dict[str, Any]] = []
    probe_dicts: list[dict[str, Any]] | None = None
    ref_outs: list[torch.Tensor] | None = None
    lpips_fn: _LPIPS | None = None
    eval_prompts = args.eval_prompts or DEFAULT_EVAL_PROMPTS[: args.n_eval_prompts]

    if args.eval_metric == "ref_image":
        logger.info("[2/3] Generating high-precision reference images ...")
        lpips_fn = _LPIPS(args.device)
        ref_dir = os.path.join(args.output_dir, "reference")
        os.makedirs(ref_dir, exist_ok=True)
        for prompt in eval_prompts:
            img = generate_image(pipe, prompt, model_type, n_gen_steps, args.seed, args.device)
            img.save(os.path.join(ref_dir, _slug(prompt) + ".png"))
            reference.append({"prompt": prompt, "pil": img, "arr": _to_rgb_array(img)})
    elif args.eval_metric == "module_mse":
        logger.info("[2/3] Capturing reference submodule outputs on a held-out tail ...")
        captured = master_loader.dataset.captured
        k = min(args.module_mse_samples, max(1, len(captured) - 1))
        probe_dicts = [master_loader.dataset[i] for i in range(len(captured) - k, len(captured))]
        ref_outs = module_reference_outputs(target, probe_dicts)
        # Restrict calibration to the disjoint head so cells never train on the probes.
        master_loader.dataset.captured = captured[: len(captured) - k]
    elif args.eval_metric == "none":
        reference = [{"prompt": p, "pil": None, "arr": None} for p in eval_prompts]

    calib_pool = len(master_loader.dataset.captured)
    sweep = sorted({min(n, calib_pool) for n in args.n_calib_samples})
    if sweep != sorted(set(args.n_calib_samples)):
        logger.warning(f"n_calib_samples clamped to calibration pool -> {sweep}")

    # Free the original pipeline before reloading fresh ones per grid cell.
    del pipe, target
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_cells = len(GPTQ_CHOICES[args.gptq]) * len(sweep)
    logger.info(f"[3/3] Grid search over gptq x n_samples ({n_cells} cells) ...")
    rows: list[dict[str, Any]] = []
    for use_gptq in GPTQ_CHOICES[args.gptq]:
        for n_samples in sweep:
            rows.append(
                run_cell(
                    args=args,
                    model_type=model_type,
                    module_name=module_name,
                    master_loader=master_loader,
                    reference=reference,
                    probe_dicts=probe_dicts,
                    ref_outs=ref_outs,
                    lpips_fn=lpips_fn,
                    use_gptq=use_gptq,
                    n_samples=n_samples,
                    n_gen_steps=n_gen_steps,
                    out_dir=args.output_dir,
                )
            )

    _write_results(args, model_type, module_name, calib_pool, rows)
    return 0 if all(r["success"] for r in rows) else 1


def _write_results(
    args: argparse.Namespace,
    model_type: str,
    module_name: str,
    pool: int,
    rows: list[dict[str, Any]],
) -> None:
    direction = SCORE_DIRECTION[args.eval_metric]
    scored = [r for r in rows if r["success"] and r["score"] is not None]
    if direction == "min":
        scored.sort(key=lambda r: r["score"])
    elif direction == "max":
        scored.sort(key=lambda r: -r["score"])
    best = scored[0] if scored else None

    results = {
        "args": vars(args),
        "model_type": model_type,
        "module_name": module_name,
        "calib_pool": pool,
        "score_metric": args.eval_metric,
        "score_direction": direction,
        "best": best,
        "rows": rows,
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    lines = ["SVDQuant calibration / grid search summary", "=" * 60, ""]
    lines.append(f"Model:    {args.model_id} ({model_type}/{module_name})")
    lines.append(f"Mode:     {args.mode}")
    lines.append(f"Metric:   {args.eval_metric} ({direction})")
    lines.append(f"Search alpha: {args.search_alpha}")
    lines.append("")
    header = f"{'cell':<16}{'gptq':<6}{'nsamp':<7}{'score':<14}{'time(s)':<9}status"
    lines.append(header)
    lines.append("-" * len(header))
    ordered = scored + [r for r in rows if r not in scored]
    for r in ordered:
        score = "n/a" if r["score"] is None else f"{r['score']:.5g}"
        status = "OK" if r["success"] else f"FAIL ({r['error']})"
        lines.append(
            f"{r['tag']:<16}{str(r['use_gptq']):<6}{r['n_calib_samples']:<7}{score:<14}{r['seconds']:<9}{status}"
        )
    lines.append("")
    if best is not None:
        lines.append(f"BEST: {best['tag']}  score={best['score']:.6g}  metrics={best['metrics']}")
    else:
        lines.append("BEST: n/a (no scored successful cells)")

    summary = "\n".join(lines)
    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    logger.info("\n" + summary)
    logger.info(f"Results written to {results_path}")


if __name__ == "__main__":
    sys.exit(main())
