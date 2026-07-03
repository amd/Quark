#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Stage 1: NVFP4 weight quantization for DeepSeek-V4-Pro.

This script re-quantizes the MoE expert weights to NVFP4 (FP4 per-group +
FP8-E4M3 per-tensor scale) using Quark's file-to-file API, producing ``weight`` /
``weight_scale`` / ``weight_scale_2`` for every quantized expert.

It is the first stage of the end-to-end pipeline; the per-expert activation
``input_scale`` is collected separately by Stage 2
(``stage2_calibrate_input_scale.py``) and attached by Stage 3
(``stage3_merge.py``).

DeepSeek-V4-Pro ships an already-quantized checkpoint:

* routed experts (``ffn.experts.N.w{1,2,3}``): packed FP4 nibbles in an I8 buffer
  with a sibling ``.scale`` (F8_E8M0, 1x32 block) -- the deepseek MXFP4 wire format.
* shared experts / attention: FP8 (E4M3) weight with a sibling ``.scale``
  (F8_E8M0, 128x128 block).

Quark's file-to-file path recognizes both conventions, dequantizes each weight to
BF16 on the fly, then re-quantizes the MoE expert weights to NVFP4. Everything
else (attention, the router gate, norms, embeddings, the output head, the MTP
block) is excluded and kept in its original checkpoint format via
``keep_excluded_layers_as_original_model_state=True``.

NVFP4 weight spec (two-stage scale-quant; the activation ``input_scale`` is
calibrated separately in Stage 2):

* first stage:  FP4 per-group (group_size=16, static, fp32 scale)
* second stage: FP8 E4M3 per-tensor (static, fp32 scale) applied to the per-group
  scale (``is_scale_quant``)

File-to-file mode quantizes weights only; the activation (input) scale is a
separate calibration step (Stage 2). The packed scale-quant export is handled
natively by Quark's file-to-file path.

Usage::

    python stage1_quantize_weight.py \\
        --input-model-path /path/to/DeepSeek-V4-Pro \\
        --output-path /path/to/output-nvfp4 \\
        --device cuda

Requirements:

* Triton (FP8 dequant kernel) + Quark's MX kernel (FP4 dequant), on a CUDA GPU.
"""

from __future__ import annotations

import argparse

from quark.common.utils.log import ScreenLogger
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig
from quark.torch.quantization.config.template import QuantizationSchemeCollection

logger = ScreenLogger(__name__)

# NVFP4 scheme from Quark's built-in scheme registry (no model-specific template needed).
_NVFP4_SCHEME = QuantizationSchemeCollection().get_scheme("nvfp4")

# fnmatch patterns (matched against the module name, i.e. tensor name minus
# ".weight") for everything that is NOT a MoE expert. We quantize ONLY the MoE
# expert weights:
#   layers.N.ffn.experts.M.{w1,w2,w3}       routed experts (I8-packed FP4 source)
#   layers.N.ffn.shared_experts.{w1,w2,w3}  shared experts (FP8 source)
# All other modules are excluded and, with
# ``keep_excluded_layers_as_original_model_state=True``, kept in their original
# checkpoint format (FP8 / BF16) rather than dequantized to BF16.
MOE_ONLY_EXCLUDE_PATTERNS = [
    "*attn*",  # attention: wq_a/b, wkv, wo_a/b, compressor.*, attn/kv/q norms
    "*ffn.gate",  # MoE router gate (exact suffix, does not match expert w1/w2/w3)
    "*ffn_norm",  # post-attention / FFN RMSNorm
    "embed",  # token embeddings
    "head",  # output projection
    "norm",  # final model norm
    "mtp*",  # multi-token-prediction block (its experts included)
]


def build_nvfp4_quant_config():
    """
    Build a QConfig that quantizes the MoE expert weights to NVFP4,
    using Quark's built-in ``nvfp4`` scheme.

    Only the routed/shared MoE expert weights are quantized; every other module is
    excluded (see ``MOE_ONLY_EXCLUDE_PATTERNS``) and preserved in its original
    checkpoint format by the caller. The activation ``input_tensors`` is set to
    ``None`` at this stage — it is calibrated separately in Stage 2 and attached
    by Stage 3.

    :return: Quark QConfig with the built-in NVFP4 weight spec applied to MoE experts only.
    :rtype: QConfig
    """
    # Use weight spec from the built-in nvfp4 scheme; set input_tensors=None
    # because Stage 1 quantizes weights only.
    global_quant_config = QLayerConfig(
        input_tensors=None,
        output_tensors=None,
        weight=_NVFP4_SCHEME.config.weight,
    )
    return QConfig(global_quant_config=global_quant_config, exclude=MOE_ONLY_EXCLUDE_PATTERNS)


def quantize_weights(args: argparse.Namespace) -> None:
    """
    Run Stage 1: NVFP4 quantization of the MoE expert weights.

    :param argparse.Namespace args: Parsed CLI arguments (input/output paths, device).

    :return: None
    """
    quant_config = build_nvfp4_quant_config()

    logger.info(
        "DeepSeek-V4-Pro NVFP4 file-to-file quantization (MoE experts only)\n"
        f"  source     : {args.input_model_path}\n"
        f"  output     : {args.output_path}\n"
        f"  device     : {args.device}\n"
        "  quantized  : layers.N.ffn.experts.* and layers.N.ffn.shared_experts.*\n"
        "  kept as-is : everything else (attn / gate / norm / embed / head / mtp)"
    )

    quantizer = ModelQuantizer(quant_config)
    quantizer.direct_quantize_checkpoint(
        pretrained_model_path=args.input_model_path,
        save_path=args.output_path,
        device=args.device,
        keep_excluded_layers_as_original_model_state=True,
    )

    logger.info(f"Weight quantization complete. Output written to: {args.output_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: Parsed arguments namespace.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end NVFP4 quantization for DeepSeek-V4-Pro. Stage 1 (this entry "
            "point) does file-to-file NVFP4 quantization of the MoE expert "
            "weights, dequantizing the source MXFP4/FP8 weights on the fly "
            "and re-quantizing to NVFP4 (FP4 per-group + FP8 E4M3 per-tensor scale)."
        )
    )
    parser.add_argument(
        "--input-model-path",
        dest="input_model_path",
        type=str,
        required=True,
        help="Path to the DeepSeek-V4-Pro checkpoint directory.",
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        type=str,
        required=True,
        help="Directory to write the NVFP4-quantized checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for tensor operations, e.g. 'cuda' or 'cuda:0' (default: %(default)s).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    quantize_weights(parse_args())
