# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

from . import sd15_unet


def pre_optimize_passes() -> list[str]:
    return sd15_unet.pre_optimize_passes()


def finalize() -> list[str]:
    return sd15_unet.finalize()


def optimize(
    input_model_path: Path,
    output_model_path: Path,
    save_as_external: bool,
    size_threshold: int,
    external_data_extension: str,
) -> None:
    sd15_unet.optimize(
        input_model_path,
        output_model_path,
        save_as_external,
        size_threshold,
        external_data_extension,
    )
