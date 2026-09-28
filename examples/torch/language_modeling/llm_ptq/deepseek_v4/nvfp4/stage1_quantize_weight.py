#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Stage 1: NVFP4 weight quantization for DeepSeek-V4-Pro.

This script re-quantizes the routed MoE expert weights to NVFP4 (FP4 per-group +
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
BF16 on the fly, then re-quantizes the routed MoE expert weights to NVFP4.
Everything else (the shared experts, attention, the router gate, norms,
embeddings, the output head, the MTP block) is excluded and kept in its original
checkpoint format via ``keep_excluded_layers_as_original_model_state=True``.

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

# The modules that are NOT routed MoE experts fall into two distinct groups, each
# expressed by its own fnmatch pattern list (matched against the module name, i.e.
# the tensor name minus ".weight"). By default we quantize ONLY the routed MoE
# expert weights (layers.N.ffn.experts.M.{w1,w2,w3}, the I8-packed FP4 source).
#
# 1) ``EXCLUDE_LAYERS_DEFAULT`` -- truly NOT quantized. These modules are already
#    16-bit (BF16) in the source checkpoint and stay 16-bit in the output.
# 2) ``KEEP_ORIGINAL_FORMAT_LAYERS_DEFAULT`` -- kept in the model's ORIGINAL quantized
#    format (passthrough). These ship pre-quantized (FP8 for attention / shared
#    experts) and are copied through unchanged -- the output is still a quantized
#    representation, we simply do not touch them.
#
# The two lists are the defaults for the ``--exclude_layers`` and
# ``--keep_original_format_layers`` CLI flags. Both stages combine them (union) into a
# single Quark ``exclude`` set, so neither group is re-quantized to NVFP4. To also
# quantize the shared experts (full NVFP4 W4A4), drop ``*shared_experts*`` from
# ``--keep_original_format_layers``; pass the SAME two lists to Stage 2 so the per-expert
# ``input_scale`` is calibrated end-to-end for whatever is quantized.
EXCLUDE_LAYERS_DEFAULT = [
    "*ffn.gate",  # MoE router gate (exact suffix, does not match expert w1/w2/w3)
    "*ffn_norm",  # post-attention / FFN RMSNorm
    "embed",  # token embeddings
    "head",  # output projection
    "norm",  # final model norm
    "mtp*",  # multi-token-prediction block (its experts included)
]
KEEP_ORIGINAL_FORMAT_LAYERS_DEFAULT = [
    "*attn*",  # attention: wq_a/b, wkv, wo_a/b, compressor.* (FP8 source, kept FP8)
    "*shared_experts*",  # shared experts: kept in their original FP8 format
]


def build_nvfp4_quant_config(exclude_layers: list[str]):
    """
    Build a QConfig that quantizes the MoE expert weights to NVFP4, using Quark's
    built-in ``nvfp4`` scheme.

    Every module matching one of the ``exclude_layers`` fnmatch patterns is excluded
    from NVFP4 quantization. The caller passes the UNION of the two CLI groups
    (``--exclude_layers`` for truly-not-quantized 16-bit layers and
    ``--keep_original_format_layers`` for original-format passthrough layers), because
    from Quark's point of view both groups are simply "not re-quantized to NVFP4".
    With the defaults only the routed MoE expert weights are quantized and the shared
    experts stay FP8; drop ``*shared_experts*`` from ``--keep_original_format_layers``
    to quantize them too. The activation ``input_tensors`` is set to ``None`` at this
    stage — it is calibrated separately in Stage 2 and attached by Stage 3.

    :param list[str] exclude_layers: fnmatch patterns (matched against the module name)
        for every module to exclude from NVFP4 quantization (the union of the two groups).
    :return: Quark QConfig with the built-in NVFP4 weight spec applied to the
        non-excluded MoE experts.
    :rtype: QConfig
    """
    # Use weight spec from the built-in nvfp4 scheme; set input_tensors=None
    # because Stage 1 quantizes weights only.
    global_quant_config = QLayerConfig(
        input_tensors=None,
        output_tensors=None,
        weight=_NVFP4_SCHEME.config.weight,
    )
    return QConfig(global_quant_config=global_quant_config, exclude=exclude_layers)


def quantize_weights(args: argparse.Namespace) -> None:
    """
    Run Stage 1: NVFP4 quantization of the MoE expert weights.

    :param argparse.Namespace args: Parsed CLI arguments (input/output paths, device).

    :return: None
    """
    # Both groups are excluded from NVFP4 quantization; combine into one Quark
    # exclude set. keep_excluded_layers_as_original_model_state=True then keeps
    # already-quantized excluded layers (FP8 attention / shared experts) in their
    # source format and leaves the 16-bit ones (embed/head/norm/gate/mtp) as BF16.
    combined_exclude = args.exclude_layers + args.keep_original_format_layers
    quant_config = build_nvfp4_quant_config(combined_exclude)

    logger.info(
        "DeepSeek-V4-Pro NVFP4 file-to-file quantization\n"
        f"  source              : {args.input_model_path}\n"
        f"  output              : {args.output_path}\n"
        f"  device              : {args.device}\n"
        "  quantized           : MoE expert weights not matching either exclude list\n"
        f"  exclude (→16-bit)   : {args.exclude_layers}\n"
        f"  keep original format: {args.keep_original_format_layers}"
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
    parser.add_argument(
        "--exclude_layers",
        nargs="+",
        default=EXCLUDE_LAYERS_DEFAULT,
        metavar="PATTERN",
        help="fnmatch patterns (matched against the module name) for modules that are "
        "TRULY NOT quantized: they are kept as 16-bit (BF16) in the output. Use this for "
        "layers that are already 16-bit in the source checkpoint (embeddings, output head, "
        "norms, router gate, MTP block). (default: %(default)s).",
    )
    parser.add_argument(
        "--keep_original_format_layers",
        nargs="+",
        default=KEEP_ORIGINAL_FORMAT_LAYERS_DEFAULT,
        metavar="PATTERN",
        help="fnmatch patterns (matched against the module name) for modules to KEEP in the "
        "model's ORIGINAL quantized format (passthrough), e.g. FP8 attention and shared "
        "experts. These are not re-quantized to NVFP4 and not dequantized to BF16; the output "
        "stays in the source format. Drop '*shared_experts*' to instead quantize the shared "
        "experts to NVFP4 (full W4A4). Pass the SAME two lists to Stage 2 so input_scale is "
        "calibrated for whatever is quantized (default: %(default)s).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    quantize_weights(parse_args())
