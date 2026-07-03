#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import re
import subprocess

import torch

from quark.common.utils.import_utils import is_transformers_available, is_transformers_version_higher_or_equal
from quark.common.utils.log import ScreenLogger
from quark.torch.quantization.config.type import Dtype

logger = ScreenLogger(__name__)

if is_transformers_available() and is_transformers_version_higher_or_equal("4.99"):
    from transformers.core_model_loading import WeightRenaming
else:
    WeightRenaming = None  # type: ignore

AWQ_QUANT_DTYPES = [Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8]
AWQ_LOAD_MAP = {
    "scales": "weight_quantizer.scale",
    "qzeros": "weight_quantizer.zero_point",
    "qweight": "weight",
    "bias": "bias",
}
LOAD_MAP = {
    "weight_scale": "weight_quantizer.scale",
    "weight_zero_point": "weight_quantizer.zero_point",
    "bias_scale": "bias_quantizer.scale",
    "bias_zero_point": "bias_quantizer.zero_point",
    "input_scale": "input_quantizer.scale",
    "input_zero_point": "input_quantizer.zero_point",
    "output_scale": "output_quantizer.scale",
    "output_zero_point": "output_quantizer.zero_point",
}
LOAD_MAP_MULTI = {
    "weight_scale": "weight_quantizer.0.scale",
    "weight_zero_point": "weight_quantizer.0.zero_point",
    "bias_scale": "bias_quantizer.0.scale",
    "bias_zero_point": "bias_quantizer.0.zero_point",
    "input_scale": "input_quantizer.0.scale",
    "input_zero_point": "input_quantizer.0.zero_point",
    "output_scale": "output_quantizer.0.scale",
    "output_zero_point": "output_quantizer.0.zero_point",
    "weight_scale_2": "weight_quantizer.1.scale",
    "weight_zero_point_2": "weight_quantizer.1.zero_point",
    "bias_scale_2": "bias_quantizer.1.scale",
    "bias_zero_point_2": "bias_quantizer.1.zero_point",
    "input_scale_2": "input_quantizer.1.scale",
    "input_zero_point_2": "input_quantizer.1.zero_point",
    "output_scale_2": "output_quantizer.1.scale",
    "output_zero_point_2": "output_quantizer.1.zero_point",
}
REVERSE_AWQ_LOAD_MAP = {
    "weight": "qweight",
    "bias": "bias",
    "weight_quantizer.scale": "scales",
    "weight_quantizer.zero_point": "qzeros",
}
REVERSE_LOAD_MAP = {value: key for key, value in LOAD_MAP.items()}
FAKE_QUANTIZED_LOAD_MAP = {
    "weight": "weight",
    "weight_scale": "_weight_quantizer.scale",
    "weight_zero_point": "_weight_quantizer.zero_point",
    "bias": "bias",
    "bias_scale": "_bias_quantizer.scale",
    "bias_zero_point": "_bias_quantizer.zero_point",
    "input_scale": "_input_quantizer.scale",
    "input_zero_point": "_input_quantizer.zero_point",
    "output_scale": "_output_quantizer.scale",
    "output_zero_point": "_output_quantizer.zero_point",
}
SAVE_MAP = {
    "weight": "weight",
    "weight_scale": "weight_scale",
    "weight_zero_point": "weight_zero_point",
}

AWQ_SAVE_MAP = {
    "weight": "qweight",
    "weight_scale": "scales",
    "weight_zero_point": "qzeros",
}

MISMATCHING_PARAMETERS_NAMES = [
    r".*weight_quantizer\..*\.scale",
    r".*bias_quantizer\..*\.scale",
    r".*input_quantizer\..*\.scale",
    r".*output_quantizer\..*\.scale",
    r".*weight_quantizer\..*\.zero_point",
    r".*bias_quantizer\..*\.zero_point",
    r".*input_quantizer\..*\.zero_point",
    r".*output_quantizer\..*\.zero_point",
]

REVERSE_MISMATCHING_PARAMETERS_NAMES = [
    r".*weight_scale_.*",
    r".*bias_scale_.*",
    r".*input_scale_.*",
    r".*output_scale_.*",
    r".*weight_zero_point_.*",
    r".*bias_zero_point_.*",
    r".*input_zero_point_.*",
    r".*output_zero_point_.*",
]


def _check_scaled_mm_available_dev() -> str | None:
    """
    Determine if torch._scaled_mm is available, there are three return values, None, "hip", "cuda"
    """
    scaled_mm_available_dev = None

    if not torch.cuda.is_available():
        return scaled_mm_available_dev
    if torch.version.cuda is not None:
        device = torch.device("cuda")
        compute_capability = torch.cuda.get_device_capability(device)
        major, minor = compute_capability
        if (major, minor) >= (9, 0) or (major == 8 and minor >= 9):
            scaled_mm_available_dev = "cuda"

    elif torch.version.hip is not None:
        result = subprocess.run("rocminfo | grep -i 'gfx'", capture_output=True, text=True, shell=True)

        if result.returncode != 0:
            raise RuntimeError("The `rocminfo` command failed or was not found.")

        output = result.stdout.strip()
        matches = re.findall(r"gfx(\d+)", output.lower())

        scaled_mm_available_dev = "hip" if len(matches) > 0 else None
        for match in matches:
            version_number = int(match)
            if version_number < 940:
                # In general, all video card models should be the same,
                # All graphics cards must be eligible
                scaled_mm_available_dev = None
                break
        if scaled_mm_available_dev == "hip":
            logger.warning(
                "When the dtype of your model is float32 and custom_mode = 'fp8', a version of torch (rocm) lower than 2.4.0 will result in calculation errors of 'torch._scaled_mm'. "
                "If you find that the ppl value is large, try to increase the version of torch. Besides, you should ensure your torch version matches your rocm to prevent errors."
            )
    return scaled_mm_available_dev


SCALED_MM_AVAILABLE_DEV = _check_scaled_mm_available_dev()


if is_transformers_available() and is_transformers_version_higher_or_equal("4.99"):
    # Single-level quantization mappings (e.g., weight_scale -> weight_quantizer.scale), i.e. non-sequential.
    QUARK_WEIGHT_CONVERSIONS = [
        WeightRenaming(source_patterns=key, target_patterns=value) for key, value in LOAD_MAP.items()
    ]

    # Sequential quantization mappings (e.g. weight_scale -> weight_quantizer.0.scale, weight_scale_2 -> weight_quantizer.1.scale)
    QUARK_WEIGHT_CONVERSIONS.extend(
        [WeightRenaming(source_patterns=key, target_patterns=value) for key, value in LOAD_MAP_MULTI.items()]
    )

    QUARK_AWQ_WEIGHT_CONVERSIONS = [
        WeightRenaming(source_patterns=key, target_patterns=value) for key, value in AWQ_LOAD_MAP.items()
    ]
else:
    QUARK_WEIGHT_CONVERSIONS = []
    QUARK_AWQ_WEIGHT_CONVERSIONS = []
