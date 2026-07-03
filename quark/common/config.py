#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

from quark import __version__
from quark.common.data_type import (
    BaseDtype,
    BaseObserverBase,
    BaseQSchemeType,
    BaseRoundType,
    BaseScaleType,
    BaseZeroPointType,
)

T = TypeVar("T", bound="BaseConfigImpl")


@dataclass(eq=True)
class BaseConfigImpl:
    name = ""

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        return cls(**data)

    def update_from_dict(self, data: dict[str, Any]) -> None:
        for field_name in data:
            setattr(self, field_name, data[field_name])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BaseAlgoConfig(BaseConfigImpl):
    pass


@dataclass
class BaseQATSpec(BaseConfigImpl):
    pass


@dataclass(eq=True)
class BaseQTensorConfig(ABC):
    dtype: BaseDtype

    observer_cls: type[BaseObserverBase] | None = None

    # Quantization Specification for BaseDtype in [Bfloat16, FP8, Int, MX]

    is_dynamic: bool | None = None

    qscheme: BaseQSchemeType | None = None

    ch_axis: int | None = None

    group_size: int | None = None

    block_size: tuple[int, int] | None = None

    ###################################################################################################
    # Quantization Specification for BaseDtype in [Int]

    symmetric: bool | None = None

    round_method: BaseRoundType | None = None

    scale_type: BaseScaleType | None = None

    scale_format: str | None = None

    scale_calculation_mode: str | None = None

    qat_spec: Any | None = None
    ###################################################################################################
    # Quantization Specification for BaseDtype in [MX]
    mx_element_dtype: BaseDtype | None = None
    ###################################################################################################
    # Quantization zero point Specification for BaseDtype
    zero_point_type: BaseZeroPointType | None = None

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize tensor configuration to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseQTensorConfig":
        """Deserialize tensor configuration from dictionary."""
        pass


@dataclass(eq=True)
class BaseQLayerConfig(ABC):
    input_tensors: BaseQTensorConfig | list[BaseQTensorConfig] | None = None

    output_tensors: BaseQTensorConfig | list[BaseQTensorConfig] | None = None

    weight: BaseQTensorConfig | list[BaseQTensorConfig] | None = None

    bias: BaseQTensorConfig | list[BaseQTensorConfig] | None = None

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize layer configuration to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseQLayerConfig":
        """Deserialize layer configuration from dictionary."""
        pass


@dataclass
class BaseQConfig(ABC):
    """
    Base class for top-level quantization configuration containing
    common attributes shared between PyTorch and ONNX
    """

    # Global quantization configuration applied to the entire model unless overridden at the layer level.
    global_quant_config: BaseQLayerConfig

    # A list of layer names to be excluded from quantization, enabling selective quantization of the model.
    exclude: list[str] = field(default_factory=list)

    # Optional configuration for the quantization algorithm, such as GPTQ, AWQ and Qronos
    # After this process, the datatype/fake_datatype of weights will be changed with quantization scales.
    algo_config: list[BaseAlgoConfig] | None = None

    # Log level for printing on screen
    log_severity_level: int | None = 1

    # Version of the quantization tool
    version: str | None = __version__

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseQConfig":
        pass
