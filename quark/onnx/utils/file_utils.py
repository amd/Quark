#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import csv
from typing import Any

CRYPTO_MODE: bool = False


def update_crypto_mode(crypto_mode: bool) -> None:
    global CRYPTO_MODE
    CRYPTO_MODE = crypto_mode


def save_quantized_info(input_rows: list[Any], write_mode: str = "a") -> None:
    quantized_info_path = "quantized_info.csv"

    if not CRYPTO_MODE:
        with open(quantized_info_path, write_mode, newline="") as file:
            writer = csv.writer(file)
            writer.writerows(input_rows)
