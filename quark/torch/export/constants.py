#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import re
import subprocess
from typing import Optional

import torch

from quark.torch.quantization.config.type import Dtype

AWQ_QUANT_DTYPES = [Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8]
AWQ_LOAD_MAP = {
    "qweight": "weight",
    "bias": "bias",
    "scales": "weight_quantizer.scale",
    "qzeros": "weight_quantizer.zero_point",
}
LOAD_MAP = {
    "weight": "weight",
    "weight_scale": "weight_quantizer.scale",
    "weight_zero_point": "weight_quantizer.zero_point",
    "bias": "bias",
    "bias_scale": "bias_quantizer.scale",
    "bias_zero_point": "bias_quantizer.zero_point",
    "input_scale": "input_quantizer.scale",
    "input_zero_point": "input_quantizer.zero_point",
    "output_scale": "output_quantizer.scale",
    "output_zero_point": "output_quantizer.zero_point",
}
REVERSE_AWQ_LOAD_MAP = {
    "weight": "qweight",
    "bias": "bias",
    "weight_quantizer.scale": "scales",
    "weight_quantizer.zero_point": "qzeros",
}
REVERSE_LOAD_MAP = {
    "weight": "weight",
    "weight_quantizer.scale": "weight_scale",
    "weight_quantizer.zero_point": "weight_zero_point",
    "bias": "bias",
    "bias_quantizer.scale": "bias_scale",
    "bias_quantizer.zero_point": "bias_zero_point",
    "input_quantizer.scale": "input_scale",
    "input_quantizer.zero_point": "input_zero_point",
    "output_quantizer.scale": "output_scale",
    "output_quantizer.zero_point": "output_zero_point",
}
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
            print(
                "[Warning] When the dtype of your model is float32 and custom_mode = 'fp8', a version of torch (rocm) lower than 2.4.0 will result in calculation errors of 'torch._scaled_mm', \n"
                "If you find that the ppl value is large, try to increase the version of torch. Besides, you should ensure your torch version matches your rocm to prevent errors."
            )
    return scaled_mm_available_dev


SCALED_MM_AVAILABLE_DEV = _check_scaled_mm_available_dev()
