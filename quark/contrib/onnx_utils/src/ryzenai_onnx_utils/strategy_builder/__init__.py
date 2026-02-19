# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import abc
import enum
from pathlib import Path

from .pass_builder import PassBuilder


class ModelType(enum.StrEnum):
    LLM = "llm"


class Priority(enum.StrEnum):
    PERFORMANCE = "performance"
    MEMORY = "memory"
    DEFAULT = "default"


class MladfVersion(enum.StrEnum):
    AIE2_V1 = "v1"
    AIE2_V2 = "v2"
    AIE4_V1 = "aie4_v1"
    FLAT = "flat"


class StrategyBuilder(abc.ABC):
    @abc.abstractmethod
    def save(self, output_strategy_path: Path): ...


__all__ = ["PassBuilder", "StrategyBuilder", "ModelType", "Priority", "MladfVersion"]
