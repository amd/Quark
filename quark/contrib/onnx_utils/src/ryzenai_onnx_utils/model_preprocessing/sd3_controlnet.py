# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

from . import mmdit


def pre_optimize_passes() -> list[str]:
    return mmdit.pre_optimize_passes()


def finalize() -> list[str]:
    return mmdit.finalize()


def optimize(
    input_model_path: Path,
    output_model_path: Path,
    save_as_external: bool,
    size_threshold: int,
    external_data_extension: str,
) -> None:
    mmdit.optimize(
        input_model_path,
        output_model_path,
        save_as_external,
        size_threshold,
        external_data_extension,
    )
