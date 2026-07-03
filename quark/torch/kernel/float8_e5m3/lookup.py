#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch


def _generate_e5m3_lookup_table() -> dict[torch.device, torch.Tensor]:
    """Generate lookup table for all 256 E5M3 bit patterns to float32 values.

    E5M3 format: 5 exponent bits, 3 mantissa bits, no sign bit (unsigned).
    Exponent bias: 15

    Returns:
        Tensor of shape (256,) mapping uint8 bit patterns to float32 values.
    """
    values = []
    EXPONENT_BIAS = 15

    # Iterate through all possible 8-bit combinations (0-255)
    for bits in range(256):
        exponent = (bits >> 3) & 0x1F  # 5 exponent bits
        mantissa = bits & 0x7  # 3 mantissa bits

        # Special case: NaN (exponent = 31 AND mantissa = 7)
        if exponent == 31 and mantissa == 7:
            values.append(float("nan"))
            continue

        # Special case: Zero
        if exponent == 0 and mantissa == 0:
            values.append(0.0)
            continue

        # Subnormal numbers (exponent = 0, mantissa != 0)
        if exponent == 0:
            value = (2 ** (-EXPONENT_BIAS + 1)) * (mantissa * 2 ** (-3))
        else:
            # Normal numbers: 2^(exponent - bias) * (1 + mantissa/8)
            value = (2 ** (exponent - EXPONENT_BIAS)) * (1 + mantissa * 2 ** (-3))

        values.append(value)

    e5m3_lookup_table_cpu = torch.tensor(values, dtype=torch.float32, device="cpu")

    e5m3_lookup_table = {torch.device("cpu"): e5m3_lookup_table_cpu}

    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        e5m3_lookup_table[device] = e5m3_lookup_table_cpu.to(device)

    return e5m3_lookup_table


# Pre-compute lookup table for E5M3 dequantization
E5M3_LOOKUP_TABLE = _generate_e5m3_lookup_table()
