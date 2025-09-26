#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Exporting and Importing API for PyTorch."""

from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import Config

from quark.shares.utils.import_utils import (
    is_accelerate_available,
    is_gguf_available_and_version_0_6_0,
    is_safetensors_available,
    is_transformers_available,
)
from quark.shares.utils.log import ScreenLogger
from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
from quark.torch.export.json_export.builder.native_model_info_builder import NativeModelInfoBuilder
from quark.torch.export.main_export.model_post_process import ModelPostProcessor
from quark.torch.export.main_export.quant_config_parser import QuantConfigParser, get_layer_quant_config
from quark.torch.export.main_import.pretrained_config import PretrainedConfig
from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.export.onnx import convert_model_to_uint4_int4, export_onnx_model_optimization
from quark.torch.export.safetensors import _load_weights_from_safetensors, export_hf_model, import_hf_model
from quark.torch.export.utils import (
    _build_quantized_model,
    _convert_quantized_model,
    _handle_multi_device_loading,
    _untie_parameters,
)
from quark.torch.quantization.config.type import QuantizationMode
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize
from quark.torch.utils import setattr_recursive

if is_gguf_available_and_version_0_6_0():
    from quark.torch.export.gguf_export.api import convert_exported_model_to_gguf, insert_quant_info_from_gguf
if is_transformers_available():
    from transformers import PreTrainedModel
if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
if is_safetensors_available():
    from safetensors.torch import save_file

__all__ = [
    "export_safetensors",
    "export_onnx",
    "export_gguf",
    "import_model_from_safetensors",
    "ModelExporter",
    "ModelImporter",
    "save_params",
]

logger = ScreenLogger(__name__)


class BaseExporter(ABC):
    """Base class for all model exporters."""

    def __init__(self, **kwargs: Any) -> None:
        self.output_dir: Path

    @abstractmethod
    def _validate(self) -> None:
        """Validate export parameters and model compatibility."""
        raise NotImplementedError("`_validate` method could not be called directly in BaseExporter")

    @abstractmethod
    def _export_impl(self) -> None:
        """Perform the actual export operation."""
        raise NotImplementedError("`_export_impl` method could not be called directly in BaseExporter")

    def _export(self, *args: Any, **kwargs: Any) -> None:
        """Process the model for validation and export. Sets attributes, validates, and exports."""

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._validate()
        self._export_impl()


class SafetensorsExporter(BaseExporter):
    """Base exporter for Safetensors format."""

    def __init__(
        self, model: torch.nn.Module, output_dir: Path, custom_mode: str, weight_format: str, pack_method: str
    ) -> None:
        super().__init__()

        self.model = model
        self.output_dir = output_dir
        self.custom_mode = custom_mode
        self.weight_format = weight_format
        self.pack_method = pack_method

    def _validate(self) -> None:
        """Validate Safetensors export parameters."""
        if self.weight_format not in ["real_quantized", "fake_quantized"]:
            raise ValueError(
                f"Weight_format must be one of `real_quantized`, `fake_quantized` when exporting to safetensors format, got {self.weight_format}."
            )
        if self.pack_method not in ["reorder", "order"]:
            raise ValueError(
                f"Pack_method must be one of `reorder`, `order` when exporting to safetensors format, got {self.pack_method}."
            )

        # Validate that model has quant_config
        if getattr(self.model, "quark_quantized", False) and getattr(self.model, "quant_config", None) is None:
            raise ValueError("Model must have a 'quant_config' attribute if it is quantized with quark.")

        # Validate model type
        if not is_transformers_available() or not isinstance(self.model, PreTrainedModel):
            raise NotImplementedError(
                "Exporting to safetensors format is currently only supported for Transformers models. Please open an issue."
            )

    def _prepare_quantization_config(
        self, quant_config: Config, temp_json_config: JsonExporterConfig, config_parser: QuantConfigParser
    ) -> dict[str, Any]:
        """Prepare quantization configuration. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _prepare_quantization_config")

    def _export_impl(self) -> None:
        """Export model to Safetensors format."""
        # Get quant_config from the model
        quant_config = getattr(self.model, "quant_config", None)

        # Create a copy of the model to avoid modifying the original
        original_config = self.model.config.__dict__.copy()

        # Create temporary config objects for processing
        temp_json_config = JsonExporterConfig(
            weight_format=self.weight_format,
            pack_method=self.pack_method,
            kv_cache_group=getattr(quant_config, "kv_cache_group", []),
            min_kv_scale=getattr(quant_config, "min_kv_scale", 0.0),
        )

        if quant_config is not None:
            # Parse quantization config
            config_parser = QuantConfigParser(quant_config, temp_json_config)

            # Prepare quantization configuration
            quantization_config_dict = self._prepare_quantization_config(quant_config, temp_json_config, config_parser)

            # Update model config with quantization info
            self.model.config.update({"quantization_config": quantization_config_dict})

        # Process model for export
        processor = ModelPostProcessor(
            self.model,
            temp_json_config,
            custom_mode=self.custom_mode,
            output_quant=quant_config is not None and quant_config.global_quant_config.output_tensors is not None,
        )
        processor.merge_scale()
        processed_model = processor.get_processed_model()

        # Export using HF format
        export_hf_model(model=processed_model, export_dir=str(self.output_dir))

        # Reset model config to original state
        self.model.config.__dict__.clear()
        self.model.config.__dict__.update(original_config)

        # Reset model to original state
        processor.reset_model()

        logger.info(f"Successfully exported model to Safetensors format in {self.custom_mode} mode: {self.output_dir}")


class QuarkSafetensorsExporter(SafetensorsExporter):
    """Exporter for Safetensors format in quark mode."""

    def _validate(self) -> None:
        """Validate quark mode export parameters."""
        super()._validate()
        if self.custom_mode != "quark":
            raise ValueError(f"QuarkSafetensorsExporter only supports custom_mode='quark', got {self.custom_mode}.")

    def _prepare_quantization_config(
        self, quant_config: Config, temp_json_config: JsonExporterConfig, config_parser: QuantConfigParser
    ) -> dict[str, Any]:
        """Prepare quantization configuration for quark mode."""
        quark_quant_config = quant_config.to_dict()
        quantization_config_dict = {}

        # Handle quark mode
        _, inferred_custom_mode = config_parser.get_custom_config()

        if inferred_custom_mode != "quark":
            logger.info(
                f"The quantized model is being exported with the default `custom_mode='quark'`, but the `Config` used also matches with the custom_mode `'{inferred_custom_mode}'`."
            )

        quark_quant_config["export"] = dataclasses.asdict(temp_json_config)
        quantization_config_dict.update(quark_quant_config)
        return quantization_config_dict


class CustomSafetensorsExporter(SafetensorsExporter):
    """Exporter for Safetensors format in custom modes (awq, fp8)."""

    def _validate(self) -> None:
        """Validate custom mode export parameters."""
        super()._validate()
        if self.custom_mode not in ["awq", "fp8"]:
            raise ValueError(
                f"CustomSafetensorsExporter only supports custom_mode in ['awq', 'fp8'], got {self.custom_mode}."
            )

    def _prepare_quantization_config(
        self, quant_config: Config, temp_json_config: JsonExporterConfig, config_parser: QuantConfigParser
    ) -> dict[str, Any]:
        """Prepare quantization configuration for custom modes (awq, fp8)."""
        quark_quant_config = quant_config.to_dict()
        quantization_config_dict = {}
        config_parser = QuantConfigParser(quant_config, temp_json_config)

        # Handle custom modes (awq, fp8)
        custom_config, inferred_custom_mode = config_parser.get_custom_config()
        if inferred_custom_mode != self.custom_mode:
            raise ValueError(
                f"Requested to export the model in the custom mode `{self.custom_mode}`, but the quantization config used does not appear to match with this `custom_mode`."
            )

        if len(custom_config) > 0:
            quantization_config_dict.update(custom_config)
        else:
            quantization_config_dict.update(quark_quant_config)

        # Add export info for HF format
        quantization_config_dict["export"] = dataclasses.asdict(temp_json_config)
        return quantization_config_dict


class OnnxExporter(BaseExporter):
    """Exporter for ONNX format."""

    def __init__(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        input_args: tuple[Any, ...],
        opset_version: int | None,
        input_names: list[str],
        output_names: list[str],
        verbose: bool,
        do_constant_folding: bool,
        operator_export_type: torch.onnx.OperatorExportTypes,
        uint4_int4_flag: bool,
    ) -> None:
        super().__init__()
        # Declare attributes that will be set by _export method

        self.model = model
        self.output_dir = output_dir
        self.input_args = input_args
        self.opset_version = opset_version
        self.input_names = input_names
        self.output_names = output_names
        self.verbose = verbose
        self.do_constant_folding = do_constant_folding
        self.operator_export_type = operator_export_type
        self.uint4_int4_flag = uint4_int4_flag

    def _validate(self) -> None:
        """Validate ONNX export parameters."""
        # Basic validation - ONNX export is generally more permissive
        if not isinstance(self.input_args, (torch.Tensor, tuple)):
            raise ValueError("input_args must be a torch.Tensor or tuple")

    def _export_impl(self) -> None:
        """Export model to ONNX format."""
        logger.info("Start exporting quantized onnx model ...")

        # When transformers version in upper than 4.55.0, the use_cache option will cause DynamicCache in ONNX export and failed to export.
        # So we need to disable the use_cache option to avoid DynamicCache in ONNX export.
        if hasattr(self.model, "config"):
            original_use_cache = getattr(self.model.config, "use_cache", None)
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

        # Enable fake quantization for ONNX export
        for module in self.model.modules():
            if isinstance(module, ScaledFakeQuantize):
                module.disable_observer()
                module.enable_fake_quant()

        # Define output path
        onnx_path = self.output_dir / "quark_model.onnx"

        # Export to ONNX
        torch.onnx.export(
            self.model.eval(),
            self.input_args,
            str(onnx_path),
            verbose=self.verbose,
            input_names=self.input_names,
            output_names=self.output_names,
            opset_version=self.opset_version,
            do_constant_folding=self.do_constant_folding,
            operator_export_type=self.operator_export_type,
        )

        # Handle uint4/int4 conversion if needed
        if self.uint4_int4_flag:
            convert_model_to_uint4_int4(str(onnx_path))
        else:
            logger.info(f"Quantized onnx model exported to {onnx_path} successfully.")

        # restore the use_cache option
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = original_use_cache

        logger.info(f"Successfully exported model to ONNX format: {onnx_path}")


class GgufExporter(BaseExporter):
    """Exporter for GGUF format."""

    def __init__(
        self, model: torch.nn.Module, output_dir: Path, model_type: str, tokenizer_path: Union[str, Path]
    ) -> None:
        super().__init__()

        self.model = model
        self.output_dir = output_dir
        self.model_type = model_type
        self.tokenizer_path = tokenizer_path

    def _validate(self) -> None:
        """Validate GGUF export parameters."""
        if not self.model_type:
            raise ValueError("model_type must be specified for GGUF export")

        # Check if tokenizer_path is a local path or HuggingFace model name
        if Path(self.tokenizer_path).exists():
            # It's a local path, validate it exists
            actual_tokenizer_path = self.tokenizer_path
        else:
            # Assume it's a HuggingFace model name - let the GGUF converter handle validation
            actual_tokenizer_path = self.tokenizer_path

        self.actual_tokenizer_path = actual_tokenizer_path

    def _export_impl(self) -> None:
        """Export model to GGUF format."""
        logger.info("Start exporting GGUF model ...")

        # First export to quark format (JSON + safetensors)
        temp_quark_dir = self.output_dir / "temp_quark_export"
        temp_quark_dir.mkdir(exist_ok=True)

        temp_config = JsonExporterConfig()
        params_dict: dict[str, torch.Tensor] = {}
        builder = NativeModelInfoBuilder(model=self.model, config=temp_config)
        info = builder.build_model_info(params_dict)

        # Save JSON info
        json_path = temp_quark_dir / f"{self.model_type}.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=4)

        # Handle tensor sharing for safetensors
        data_ptr_list: list[str] = []
        for key, value in params_dict.items():
            if str(value.data_ptr()) in data_ptr_list:
                params_dict[key] = value.clone()
            else:
                data_ptr_list.append(str(value.data_ptr()))

        # Save safetensors
        if not is_safetensors_available():
            raise ImportError(
                "The function `export_gguf` requires the package `safetensors` to be installed, but it was not found. Please install `safetensors`."
            )

        safetensors_path = temp_quark_dir / f"{self.model_type}.safetensors"
        save_file(params_dict, safetensors_path)

        # Convert to GGUF format
        gguf_output_path = self.output_dir / f"{self.model_type}.gguf"
        convert_exported_model_to_gguf(
            model_name=self.model_type,
            json_path=json_path,
            safetensor_path=safetensors_path,
            tokenizer_dir=self.actual_tokenizer_path,
            output_file_path=gguf_output_path,
        )

        # Clean up temporary files
        shutil.rmtree(temp_quark_dir)

        logger.info(f"Successfully exported model to GGUF format: {gguf_output_path}")


def export_safetensors(
    model: torch.nn.Module,
    output_dir: Union[str, Path],
    custom_mode: str = "quark",
    weight_format: str = "real_quantized",
    pack_method: str = "reorder",
) -> None:
    """
    Export the quantized PyTorch model to Safetensors format.

    The model's network architecture or configuration is stored in the json file, and parameters including weight, bias, scale, and zero_point are stored in the safetensors file.

    :param torch.nn.Module model: The quantized model to be exported.
    :param Union[str, Path] output_dir: Directory to save the exported files.
    :param str custom_mode: Export mode determining quantization handling. Defaults to ``"quark"``. Possible values are:

        * ``"quark"``: standard quark format. This is the default and recommended format that should be favored.
        * ``"awq"``: targets AutoAWQ library.
        * ``"fp8"``: targets vLLM-compatible fp8 models.
    :param str weight_format: How to handle quantized parameters. Defaults to ``"real_quantized"``. Possible values are:

        * ``"real_quantized"``: actual quantized parameters.
        * ``"fake_quantized"``: QDQ (Quantize-Dequantize) representation of quantized parameters.
    :param str pack_method: Real_quantized parameter packing strategy. Defaults to ``"reorder"``. Possible values are:

        * ``"reorder"``: reorder the real_quantized parameters layout for hardware.
        * ``"order"``: keep the original real_quantized parameters layout.

    :return: ``None``

    Example:

        .. code-block:: python

            from quark.torch import export_safetensors

            export_path = "./output_dir"
            export_safetensors(model, export_path, custom_mode="quark", weight_format="real_quantized", pack_method="reorder")
    """
    # Get quant_config from the model
    if getattr(model, "quark_quantized", False) and getattr(model, "quant_config", None) is None:
        raise ValueError("Model must have a 'quant_config' attribute if it is quantized with quark.")

    if custom_mode != "quark":
        logger.warning(
            f"The 'custom_mode' parameter is deprecated and will be removed in version 1.0. "
            f"Currently using custom_mode='{custom_mode}', but only 'quark' mode will be supported in the future. "
            f"Please migrate to using custom_mode='quark'."
        )

    # Choose the appropriate exporter based on custom_mode
    if custom_mode == "quark":
        exporter_cls = QuarkSafetensorsExporter
    elif custom_mode in ["awq", "fp8"]:
        exporter_cls = CustomSafetensorsExporter  # type: ignore
    else:
        raise ValueError(f"Custom_mode must be one of `quark`, `fp8`, `awq`, got {custom_mode}.")

    exporter = exporter_cls(
        model=model,
        output_dir=Path(output_dir),
        custom_mode=custom_mode,
        weight_format=weight_format,
        pack_method=pack_method,
    )
    exporter._export()


def export_onnx(
    model: torch.nn.Module,
    output_dir: Union[str, Path],
    input_args: tuple[Any, ...],
    opset_version: int | None = None,
    input_names: list[str] = [],
    output_names: list[str] = [],
    verbose: bool = False,
    do_constant_folding: bool = True,
    operator_export_type: torch.onnx.OperatorExportTypes = torch.onnx.OperatorExportTypes.ONNX,
    uint4_int4_flag: bool = False,
) -> None:
    """
    Export the onnx graph of the quantized PyTorch model.

    :param torch.nn.Module model: The quantized model to be exported.
    :param Union[str, Path] output_dir: Directory to save the ONNX file
    :param Union[torch.Tensor, Tuple[float]] input_args: Example inputs for ONNX tracing.
    :param Optional[int] opset_version: The version of the ONNX opset to target. If not set, it will be valued the latest version that is stable for the current version of PyTorch. Defaults to ``None``.
    :param List[str] input_names: Names to assign to the input nodes of the onnx graph, in order. Defaults to ``[]``.
    :param List[str] output_names: Names to assign to the output nodes of the onnx graph, in order. Defaults to ``[]``.
    :param bool verbose: Flag to control showing verbose log or no. Defaults to ``False``.
    :param bool do_constant_folding: Flag to apply constant folding optimization. Defaults to ``True``.
    :param torch.onnx.OperatorExportTypes operator_export_type: Export operator type in onnx graph. The choices include ``OperatorExportTypes.ONNX``, ``OperatorExportTypes.ONNX_FALLTHROUGH``, ``OperatorExportTypes.ONNX_ATEN`` and ``OperatorExportTypes.ONNX_ATEN_FALLBACK``. Defaults to ``OperatorExportTypes.ONNX``.
    :param bool uint4_int4_flag: Flag to indicate uint4/int4 quantized model or not. Defaults to ``False``.

    :return: None

    Example:

    .. code-block:: python

        from quark.torch import export_onnx

        export_onnx(model, output_dir, input_args)

    **Note**:
        Mix quantization of int4/uint4 and int8/uint8 is not supported currently.
        In other words, if the model contains both quantized nodes of uint4/int4 and uint8/int8, this function cannot be used to export the ONNX graph.
    """
    exporter = OnnxExporter(
        model=model,
        output_dir=Path(output_dir),
        input_args=input_args,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        verbose=verbose,
        do_constant_folding=do_constant_folding,
        operator_export_type=operator_export_type,
        uint4_int4_flag=uint4_int4_flag,
    )
    exporter._export()


def export_gguf(
    model: torch.nn.Module,
    output_dir: Union[str, Path],
    model_type: str,
    tokenizer_path: Union[str, Path],
) -> None:
    """
    Export the gguf file of the quantized PyTorch model.

    :param torch.nn.Module model: The quantized model to be exported.
    :param Union[str, Path] output_dir: Directory to save the GGUF file
    :param str model_type: The model type of the model, e.g. ``"gpt2"``, ``"gptj"``, or ``"llama"``.
    :param Union[str, Path] tokenizer_path: Tokenizer needs to be encoded into gguf model. This argument specifies the directory path of the tokenizer, which contains tokenizer.json, tokenizer_config.json and/or tokenizer.model.

    :return: None

    Example:

    .. code-block:: python

        from quark.torch import export_gguf
        export_gguf(model, output_dir, model_type, tokenizer_path)

    Note:
        Currently, only support asymetric int4 per_group weight-only quantization, and the group_size must be 32.
        Supported models include Llama2-7b, Llama2-13b, Llama2-70b, and Llama3-8b.
    """
    if not is_gguf_available_and_version_0_6_0():
        raise ImportError(
            "The function `export_gguf` requires the package `gguf==0.6.0` to be installed, but it was not found. Please install `gguf==0.6.0`."
        )

    exporter = GgufExporter(
        model=model, output_dir=Path(output_dir), model_type=model_type, tokenizer_path=tokenizer_path
    )
    exporter._export()


class BaseImporter(ABC):
    """Base class for all model importers."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def _validate(self) -> None:
        """Validate import parameters and model compatibility."""
        pass

    @abstractmethod
    def _import_impl(self) -> torch.nn.Module:
        """Perform the actual import operation."""
        pass

    def _import(self, *args: Any, **kwargs: Any) -> torch.nn.Module:
        """Process the model for validation and import. Sets attributes, validates, and imports."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._validate()
        return self._import_impl()


class SafetensorsImporter(BaseImporter):
    """Importer for Safetensors format."""

    def __init__(self) -> None:
        super().__init__()
        # Declare attributes that will be set by _import method
        self.model: torch.nn.Module
        self.model_dir: str
        self.multi_device: bool

    def _validate(self) -> None:
        """Validate Safetensors import parameters."""
        if not is_safetensors_available():
            raise ImportError(
                "The function `import_model_from_safetensors` requires the package `safetensors` to be installed, but it was not found. Please install `safetensors`."
            )

    def _import_impl(self) -> torch.nn.Module:
        """Import model from Safetensors format."""
        logger.info("Start importing safetensors quantized model ...")

        # Create temporary model config object
        model_config = PretrainedConfig(pretrained_dir=self.model_dir)

        # Load weights from file, on cpu device.
        checkpoint_weights = _load_weights_from_safetensors(self.model_dir)

        original_model_on_meta_device = False
        for name, param in chain(self.model.named_parameters(), self.model.named_buffers()):
            if param.device.type == "meta":
                original_model_on_meta_device = True
                break

        if original_model_on_meta_device:
            has_non_persistent_buffers = any(
                len(submodule._non_persistent_buffers_set) > 0 for submodule in self.model.modules()
            )

            if has_non_persistent_buffers:
                raise NotImplementedError(
                    "Reloading a safetensors model using the original non-quantized model placed on meta device while it contains non-persistent buffers is not supported, as the non-persistent buffers can not be reloaded from the serialized checkpoint. Please consider initializing the original non-quantized model on cpu or cuda device. Please open an issue for the feature to be supported."
                )

        # Build model with quantization support
        model = _build_quantized_model(self.model, model_config, checkpoint_weights)

        # Handle parameter untying
        if is_accelerate_available():
            _untie_parameters(model, checkpoint_weights)

        # Get current model state dict
        model_state_dict = model.state_dict()

        # In case we are loading the quantized weights into a model that is not on meta device,
        # we re-use the original device the weights were placed on, as `assign=True` is used later.
        # This is helpful e.g. in case the original model was dispatched to multiple
        # devices ahead of time with `accelerate`.
        for name, param in model_state_dict.items():
            if name not in checkpoint_weights:
                raise ValueError(f"The loaded checkpoint misses the key {name} present in the model weights.")
            else:
                if param.device.type != "meta":
                    checkpoint_weights[name] = checkpoint_weights[name].to(param.device)

        # Handle multi-device loading if enabled
        if self.multi_device and is_accelerate_available():
            _handle_multi_device_loading(model, checkpoint_weights)

        # Load weights into model with strict=False to handle missing quantization parameters
        model.load_state_dict(checkpoint_weights, assign=True, strict=False)

        # Convert model
        model = _convert_quantized_model(model, model_config)

        logger.info("safetensors quantized model imported successfully.")
        return model


def import_model_from_safetensors(
    model: torch.nn.Module, model_dir: str, multi_device: bool = False
) -> torch.nn.Module:
    """
    Imports a quantized model from the local directory ``model_dir`` into a non-quantized model ``model``.

    :param torch.nn.Module model: The non-quantized model, that will be transformed in place to a quantized model using the ``"quantization_config"`` in the ``config.json`` file retrieved in the local directory ``model_dir``, and in which quantized weights will be loaded into.
    :param str model_dir: Directory containing the model files (``config.json`` and ``model.safetensors``)
    :param bool multi_device: Whether to use multi-device loading using Accelerate library. Defaults to ``False``.

    :return: The model with loaded weights and proper quantization modules.
    """
    importer = SafetensorsImporter()
    return importer._import(model=model, model_dir=model_dir, multi_device=multi_device)


# TODO: remove in the next minor release (deprecated in quark==1.0.0).
class ModelExporter:
    """
    Provides an API for exporting quantized PyTorch deep learning models.
    This class converts the quantized model to json-pth, json-safetensors files or onnx graph, and saves to export_dir.

    .. deprecated:: 1.0.0
        ModelExporter is deprecated. Use the new dedicated export functions instead:
        - export_safetensors() for Safetensors format
        - export_onnx() for ONNX format
        - export_gguf() for GGUF format

    :param ExporterConfig config: Configuration object containing settings for exporting.
    :param Union[Path, str] export_dir: The target export directory.
    """

    def __init__(self, config: ExporterConfig, export_dir: Union[Path, str] = tempfile.gettempdir()) -> None:
        logger.warning(
            "ModelExporter is deprecated and will be removed in a future version. "
            "Please use the new dedicated export functions: export_safetensors(), "
            "export_onnx(), export_gguf()."
        )
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    def export_quark_model(self, model: nn.Module, quant_config: Config, custom_mode: str = "quark") -> None:
        """
        Exports the quantized PyTorch model to quark file format using json and pth files.

        The model's network architecture or configuration is stored in the json file, and parameters including weight, bias, scale, and zero_point are stored in the pth file.

        :param torch.nn.Module model: The quantized model to be exported.
        :param Config quant_config: Configuration object containing settings for quantization. Default is ``None``.
        :param str custom_mode: Whether to export the quantization config and model in a custom format expected by some downstream library. Possible options:

            * ``"quark"``: standard quark format. This is the default and recommended format that should be favored.
            * ``"awq"``: targets AutoAWQ library.
            * ``"fp8"``: targets vLLM-compatible fp8 models.

        :return: ``None``

        Example:

        .. code-block:: python


            from quark.torch import ModelExporter
            from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig

            export_path = "./output_dir"

            NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized",
                                                       pack_method="reorder")
            export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)
            exporter = ModelExporter(config=export_config, export_dir=export_path)
            quant_config = get_config(args, model_type)

            exporter.export_quark_model(model, quant_config=quant_config, custom_mode=args.custom_mode)

        Note:
            Currently, default exporting quark format (json + pth).
        """

        if custom_mode not in ["quark", "fp8", "awq"]:
            raise ValueError(
                f"The supported values for `custom_mode` are {['quark', 'fp8', 'awq', 'auto']} but custom_mode={custom_mode} was provided. Please check your code or open an issue in Quark repository."
            )

        if quant_config is None:
            raise ValueError("quant_config should not be None when exporting default format files")

        logger.info("Start exporting quark format quantized model ...")
        model = self.get_export_model(model=model, quant_config=quant_config, custom_mode=custom_mode)
        self.save_quark_export_model(model)
        self.reset_model(model)
        if self.config.json_export_config.weight_format == "real_quantized":
            logger.info(f"quark_format real_quantized model exported to {self.export_dir} successfully.")
        else:
            logger.info(f"quark_format fake_quantized model exported to {self.export_dir} successfully.")

    def get_export_model(
        self, model: nn.Module, quant_config: Config, custom_mode: str = "quark", add_export_info_for_hf: bool = True
    ) -> nn.Module:
        """
        Merges scales, replaces modules of the quantized model to prepare for export, and add export information in config.json.

        Scale merging selects the maximum scale value in specified `weight_group` as the scale for each module in the group.

        Build kv_scale selects the maximum kv_scale value in `kv_group` as the scale for the key projection output quantization and value projection output quantization.

        Module replacement converts the model's module (e.g. `QuantLinear`) according to the weight_format (to `QparamsLinear`).


        :param torch.nn.Module model: The quantized model to be exported.
        :param Config quant_config: Model quantization configuration.
        :param str custom_mode: Whether to export the quantization config and model in a custom format expected by some downstream library. Possible options:

            * ``"quark"``: standard quark format. This is the default and recommended format that should be favored.
            * ``"awq"``: targets AutoAWQ library.
            * ``"fp8"``: targets vLLM-compatible fp8 models.
        :param bool add_export_info_for_hf: Whether to add export info of quark to ``config.json`` when using hf_format_export. When loading the model, we recover the kv_cache in autofp8 format through the weight file, but we need the name of kv_layer, it is very cumbersome to get it from quark's map, it is more reasonable to get it from config. If we find ``kv_scale`` in weight_flie and there is no special kv_layer_name, we will use k_proj,v_proj to recover kv_cache by default.
        """

        quark_quant_config = quant_config.to_dict()
        quantization_config_dict = {}
        config_parser = QuantConfigParser(quant_config, self.config.json_export_config)
        if custom_mode != "quark":
            # Some quantization methods (fp8, awq) might be used in external libraries directly. Quark's `Config` is parsed
            # to detect whether we may add custom keys in the config.json `quantization_config` to make loading quark models
            # in external libraries easier.
            custom_config, inferred_custom_mode = config_parser.get_custom_config()
            if inferred_custom_mode != custom_mode:
                raise ValueError(
                    f"Requested to export the model in the custom mode `{custom_mode}`, but the quantization config used does not appear to match with this `custom_mode`. If using `custom_mode='awq'` or `custom_mode='fp8'`, please make sure the quantization config is well defined to match these custom modes. Alternatively, please use `custom_mode='quark'` or open an issue in Quark repository."
                )

            # This custom_config might be empty.
            if len(custom_config) > 0:
                quantization_config_dict.update(custom_config)
            else:
                quantization_config_dict.update(quark_quant_config)
            if add_export_info_for_hf:
                quantization_config_dict["export"] = dataclasses.asdict(self.config.json_export_config)
        else:
            _, inferred_custom_mode = config_parser.get_custom_config()

            if inferred_custom_mode != "quark":
                logger.info(
                    f"The quantized model is being exported in `ModelExporter.export_model_info` with the default `custom_mode='quark'`, which uses the standard format to export quark. However, the `Config` used also matches with the custom_mode `'{inferred_custom_mode}'`, which is not recommended but may temporarily facilitate usage in some downstream libraries. If you would like to use this custom export, please use `ModelExporter.export_model_info(..., custom_mode='{inferred_custom_mode}')`."
                )

            quark_quant_config["export"] = dataclasses.asdict(self.config.json_export_config)
            quantization_config_dict.update(quark_quant_config)

        model.config.update({"quantization_config": quantization_config_dict})

        # Map `QuantLinear` (fake quantization) to `QparamsLinear` ("real" quantization, where weights have low precision).
        self.processor = ModelPostProcessor(
            model,
            self.config.json_export_config,
            custom_mode=custom_mode,
            output_quant=quant_config.global_quant_config.output_tensors is not None,
        )
        self.processor.merge_scale()
        model = self.processor.get_processed_model()
        return model

    def save_quark_export_model(self, model: nn.Module) -> None:
        torch.save(model.state_dict(), self.export_dir.joinpath("model_state_dict.pth"))
        with open(self.export_dir.joinpath("config.json"), "w") as json_file:
            json.dump(model.config.to_dict(), json_file, indent=4)

    def reset_model(self, model: nn.Module) -> None:
        """
        Restore exported model to frozen Model for inferring, restore config content.
        """
        model.config.__dict__.pop("quantization_config")
        model = self.processor.reset_model()

    def export_onnx_model(
        self,
        model: nn.Module,
        input_args: tuple[Any, ...],
        input_names: list[str] = [],
        output_names: list[str] = [],
        verbose: bool = False,
        opset_version: int | None = None,
        do_constant_folding: bool = True,
        operator_export_type: torch.onnx.OperatorExportTypes = torch.onnx.OperatorExportTypes.ONNX,
        uint4_int4_flag: bool = False,
    ) -> None:
        """
        This function aims to export onnx graph of the quantized PyTorch model.

        :param torch.nn.Module model: The quantized model to be exported.
        :param Tuple[Any, ...] input_args: Example inputs for this quantized model.
        :param List[str] input_names: Names to assign to the input nodes of the onnx graph, in order. Defaults to ``[]``.
        :param List[str] output_names: Names to assign to the output nodes of the onnx graph, in order. Defaults to ``[]``.
        :param bool verbose: Flag to control showing verbose log or no. Default is ``False``.
        :param Optional[int] opset_version: The version of the default (ai.onnx) opset to target. If not set, it will be valued the latest version that is stable for the current version of PyTorch. Defaults to ``None``.
        :param torch.onnx.OperatorExportTypes operator_export_type: Export operator type in onnx graph. The choices include ``OperatorExportTypes.ONNX``, ``OperatorExportTypes.ONNX_FALLTHROUGH``, ``OperatorExportTypes.ONNX_ATEN`` and ``OperatorExportTypes.ONNX_ATEN_FALLBACK``. Default is ``OperatorExportTypes.ONNX``.
        :param bool uint4_int4_flag: Flag to indicate uint4/int4 quantized model or not. Default is ``False``.

        :return: None

        Example:

        .. code-block:: python

            from quark.torch import ModelExporter
            from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig

            export_config = ExporterConfig(json_export_config=JsonExporterConfig())
            exporter = ModelExporter(config=export_config, export_dir=export_path)
            exporter.export_onnx_model(model, input_args)

        **Note**:
            Mix quantization of int4/uint4 and int8/uint8 is not supported currently.
            In other words, if the model contains both quantized nodes of uint4/int4 and uint8/int8, this function cannot be used to export the ONNX graph.
        """
        logger.info("Start exporting quantized onnx model ...")

        # When transformers version in upper than 4.55.0, the use_cache option will cause DynamicCache in ONNX export and failed to export.
        # So we need to disable the use_cache option to avoid DynamicCache in ONNX export.
        if hasattr(model, "config"):
            original_use_cache = getattr(model.config, "use_cache", None)
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

        for module in model.modules():
            if isinstance(module, ScaledFakeQuantize):
                module.disable_observer()
                module.enable_fake_quant()
        onnx_path = str(self.export_dir / "quark_model.onnx")
        torch.onnx.export(
            model.eval(),
            input_args,
            onnx_path,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            operator_export_type=operator_export_type,
        )
        export_onnx_model_optimization(onnx_path)
        if uint4_int4_flag:
            convert_model_to_uint4_int4(onnx_path)
        else:
            logger.info(f"Quantized onnx model exported to {onnx_path} successfully.")

        # restore the use_cache option
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = original_use_cache

    def export_gguf_model(self, model: nn.Module, tokenizer_path: Union[str, Path], model_type: str) -> None:
        """
        This function aims to export gguf file of the quantized PyTorch model.

        :param torch.nn.Module model: The quantized model to be exported.
        :param Union[str, Path] tokenizer_path model_type: Tokenizer needs to be encoded into gguf model. This argument specifies the directory path of the tokenizer, which contains tokenizer.json, tokenizer_config.json and/or tokenizer.model.
        :param str model_type: The model type of the model, e.g. ``"gpt2"``, ``"gptj"``, or ``"llama"``.

        :return: None

        Example:

        .. code-block:: python

            from quark.torch import ModelExporter
            from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
            export_config = ExporterConfig(json_export_config=JsonExporterConfig())
            exporter = ModelExporter(config=export_config, export_dir=export_path)
            exporter.export_gguf_model(model, tokenizer_path, model_type)

        Note:
            Currently, only support asymetric int4 per_group weight-only quantization, and the group_size must be 32.
            Supported models include Llama2-7b, Llama2-13b, Llama2-70b, and Llama3-8b.
        """
        if not is_gguf_available_and_version_0_6_0():
            raise ImportError(
                "The function `export_gguf_model` requires the package `gguf==0.6.0` to be installed, but it was not found. Please install `gguf==0.6.0`."
            )

        logger.info("Start exporting gguf quantized model ...")

        save_params(model, model_type, export_dir=self.export_dir)

        json_path = self.export_dir / f"{model_type}.json"
        params_path = self.export_dir / f"{model_type}.safetensors"
        gguf_path = self.export_dir / f"{model_type}.gguf"

        convert_exported_model_to_gguf(model_type, json_path, params_path, tokenizer_path, gguf_path)

        if json_path.exists():
            json_path.unlink()
        if params_path.exists():
            params_path.unlink()

        logger.info(f"GGUF quantized model exported to {gguf_path} successfully.")

    def export_safetensors_model(
        self, model: nn.Module, quant_config: Config, custom_mode: str = "quark", **kwargs: Any
    ) -> None:
        """
        Exports the quantized PyTorch model to the safetensors format.

        :param torch.nn.Module model: The quantized model to be exported.
        :param Config quant_config: Configuration object containing settings for quantization. Default is ``None``.
        :param str custom_mode: Whether to export the quantization config and model in a custom format expected by some downstream library. Possible options:

            * ``"quark"``: standard quark format. This is the default and recommended format that should be favored.
            * ``"awq"``: targets AutoAWQ library.
            * ``"fp8"``: targets vLLM-compatible fp8 models.
        """
        if quant_config is None:
            raise ValueError("quant_config should not be None when exporting Hugging Face safetensors format files.")

        if not is_transformers_available() or not isinstance(model, PreTrainedModel):
            raise NotImplementedError(
                "Exporting to safetensors format is currently only supported for Transformers models. Please open an issue."
            )
        else:
            # add_export_info_for_hf=True means export info of quark will be added in config.json, see the description of the get_export_model function
            model = self.get_export_model(
                model, quant_config=quant_config, custom_mode=custom_mode, add_export_info_for_hf=True
            )
            export_hf_model(model=model, export_dir=self.export_dir, **kwargs)  # type: ignore[arg-type]

        # The export_func replaces some of the model's submodules and modifies the contents of the config, so restore them.
        self.reset_model(model=model)

    def export_model_info_from_gguf(self, model: nn.Module, gguf_path: str, model_type: str) -> None:
        if not is_safetensors_available():
            raise ImportError(
                "The function `export_model_info_from_gguf` requires the package `safetensors` to be installed, but it was not found. Please install `safetensors`."
            )

        if not is_gguf_available_and_version_0_6_0():
            raise ImportError(
                "The function `export_model_info_from_gguf` requires the package `gguf==0.6.0` to be installed, but it was not found. Please install `gguf==0.6.0`."
            )

        logger.info("Start exporting quantized model from gguf model ...")

        params_dict: dict[str, torch.Tensor] = {}
        builder = NativeModelInfoBuilder(model=model, config=self.config.json_export_config)
        info = builder.build_model_info(params_dict)

        info, params_dict = insert_quant_info_from_gguf(model_type, info, params_dict, gguf_path)

        json_path = self.export_dir / f"{model_type}_from_gguf.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=4)

        # handle tensors shared
        data_ptr_list: list[str] = []
        for key, value in params_dict.items():
            if str(value.data_ptr()) in data_ptr_list:
                params_dict[key] = value.clone()
            else:
                data_ptr_list.append(str(value.data_ptr()))

        params_path = self.export_dir / f"{model_type}_from_gguf.safetensors"
        save_file(params_dict, params_path)

        logger.info(f"Exported quantized model from gguf model to {self.export_dir} successfully.")


def save_params(
    model: nn.Module,
    model_type: str,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    export_dir: Union[Path, str] = tempfile.gettempdir(),
    quant_mode: QuantizationMode = QuantizationMode.eager_mode,
    compressed: bool = False,
    reorder: bool = True,
) -> None:
    """
    Save the network architecture or configurations and parameters of the quantized model.
    For eager mode quantization, the model's configurations are stored in json file, and parameters including weight, bias, scale, and zero_point are stored in safetensors file.
    For fx_graph mode quantization, the model's network architecture and parameters are stored in pth file.

    :param torch.nn.Module model: The quantized model to be saved.
    :param str model_type: The type of the model, e.g. gpt2, gptj, llama or gptnext.
    :param Optional[Tuple[Any, ...]] args: Example tuple inputs for this quantized model. Only available for fx_graph mode quantization. Default is ``None``.
    :param Optional[Dict[str, Any]] kwargs: Example dict inputs for this quantized model. Only available for fx_graph mode quantization. Default is ``None``.
    :param Union[Path, str] export_dir: The target export directory.
    :param QuantizationMode quant_mode: The quantization mode. The choice includes ``QuantizationMode.eager_mode`` and ``QuantizationMode.fx_graph_mode``. Default is ``QuantizationMode.eager_mode``.
    :param bool compressed: Export the compressed (real quantized) model or QDQ model, Default is ``False`` and it exports the QDQ model.
    :param bool reorder: pack method, uses pack the weight (eg. packs four ``torch.int8`` value into one ``torch.int32`` value). Default is ``True``.

    :return: None

    Examples:

    .. code-block:: python

        # eager mode:
        from quark.torch import save_params
        save_params(model, model_type=model_type, export_dir="./save_dir")

    .. code-block:: python

        # fx_graph mode:
        from quark.torch.export.api import save_params
        save_params(model,
                    model_type=model_type,
                    args=example_inputs,
                    export_dir="./save_dir",
                    quant_mode=QuantizationMode.fx_graph_mode)
    """
    logger.info("Start saving parameters of quantized model ...")
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    if quant_mode is QuantizationMode.eager_mode:
        if not is_safetensors_available():
            raise ImportError(
                "The function `save_params` with `quant_mode=QuantizationMode.eager_mode` requires the package `safetensors` to be installed, but it was not found. Please install `safetensors`."
            )
        params_dict: dict[str, torch.Tensor] = {}
        builder = NativeModelInfoBuilder(model=model, config=JsonExporterConfig())
        info = builder.build_model_info(params_dict, compressed=compressed, reorder=reorder)
        json_path = export_dir / f"{model_type}.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=4)

        # handle tensors shared
        data_ptr_list: list[str] = []
        for key, value in params_dict.items():
            if str(value.data_ptr()) in data_ptr_list:
                params_dict[key] = value.clone()
            else:
                data_ptr_list.append(str(value.data_ptr()))

        params_path = export_dir / f"{model_type}.safetensors"
        save_file(params_dict, params_path)
    elif quant_mode is QuantizationMode.fx_graph_mode:
        if args is None:
            raise ValueError("args should not be None when saving fx_graph_mode quantized model")
        model_file_path = export_dir / f"{model_type}_quantized.pth"
        exported_model = torch.export.export(model, args, kwargs=kwargs)
        torch.export.save(exported_model, model_file_path)

    logger.info(f"Parameters of quantized model saved to {export_dir} successfully.")


class ModelImporter:
    """
    Provides an API for importing quantized PyTorch deep learning models.
    This class load json-pth or json-safetensors files to model.

    .. deprecated:: 1.0.0
        ModelImporter is deprecated. Use the dedicated import functions instead:
        - import_model_from_safetensors() for Safetensors format

    :param str model_info_dir: The target import directory.
    :param str saved_format: Specifies the format to load from. This can be ``"quark_format"`` or ``"hf_format"`` (or ``"safetensors"``). Defaults to ``"quark_format"``.
    :param bool multi_device: Whether or not to use gpu + cpu mode to import models via "accelerate".
    """

    SUPPORTED_FORMATS = ["quark_format", "hf_format", "safetensors"]

    def __init__(self, model_info_dir: str, saved_format: str = "quark_format", multi_device: bool = False) -> None:
        logger.warning(
            "ModelImporter is deprecated and will be removed in a future version. "
            "Please use the new dedicated import functions: import_model_from_safetensors()."
        )
        self.model_info_dir = model_info_dir

        if saved_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Loading a model with `ModelImporter.import_model_info` using the format `format={format}` is not supported. Supported formats are 'quark_format', 'hf_format' and 'safetensors'."
            )

        self.saved_format = saved_format
        self.multi_device = multi_device
        self.model_config = self.get_model_config()

        if self.model_config.weight_format == "fake_quantized":
            self.is_real_quantized_mode = False
        else:
            self.is_real_quantized_mode = True

    def get_model_config(self) -> PretrainedConfig:
        model_config = PretrainedConfig(pretrained_dir=self.model_info_dir)
        return model_config

    def get_model_state_dict(self) -> dict[str, Any]:
        model_state_dict: dict[str, Any] = torch.load(Path(self.model_info_dir) / "model_state_dict.pth")
        return model_state_dict

    def import_model_info(self, model: nn.Module) -> nn.Module:
        """
        Reloads a serialized quantized model, based on the non-quantized module.

        This function aims to import quark(json-pth) files of the Hugging Face large language model.

        It could recover the weight, bias, scale, and zeropoint information of the model and execute the inference.

        :param torch.nn.Module model: The original Hugging Face large language model.

        :return: Model with quantized weights and modules.
        :rtype: torch.nn.Module

        Example:

        .. code-block:: python

            from quark.torch import ModelImporter

            model_importer = ModelImporter(model_info_dir="./import_model_dir")
            model = importer.import_model_info(model)

        """
        if self.saved_format == "quark_format":
            logger.info("Start importing quark_format(pth_json) quantized model ...")
            model_state_dict = self.get_model_state_dict()
            model = _build_quantized_model(model, self.model_config, model_state_dict)
            model.load_state_dict(model_state_dict)
            model = _convert_quantized_model(model, self.model_config)
            logger.info("quark_format(pth_json) quantized model imported successfully.")
        elif self.saved_format in ["safetensors", "hf_format"]:
            model = import_hf_model(model_importer=self, model=model, model_info_dir=self.model_info_dir)
        else:
            raise ValueError(
                f"Could not parse the format {self.saved_format} in ModelImporter.import_model_info. This is a bug, please open an issue."
            )

        return model


def _map_to_quark(model: nn.Module, quantization_config: Config, pack_method: str, custom_mode: str) -> None:
    """
    Maps a non-quantized model (possibly on meta device) to a model with QParamsLinear layers with weights not initialized. This function is useful to later load a checkpoint in the quark model using `model.load_state_dict(state_dict)`.

    Parameters:
        model (torch.nn.Module): An instance of the original not-quantized model. This model may be on `meta` device, or may have random weights.
        quantization_config (Config): The quantization configuration orginally used to quantize the model in Quark.
        pack_method (str): The packing method used when the model was serialized.
        custom_mode (str): The custom mode to use to initialize the `QParamsLinear` layers. The recommended mode is simply quark-native `"quark"`, but `"awq"` and `"fp8"` are also available.
    """
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for op_name, float_module in tqdm(named_modules.items()):
        op_type = type(float_module)
        layer_quantization_config = get_layer_quant_config(quantization_config, op_type, op_name)

        if layer_quantization_config is not None and isinstance(float_module, nn.Linear):
            qparams_linear = QParamsLinear.from_module(
                float_module,
                custom_mode,
                pack_method,
                quant_config=layer_quantization_config,
            )
            # for multi_device, hook can offer info.
            if hasattr(float_module, "_hf_hook"):
                hook = float_module._hf_hook
                quark_hook = AlignDevicesHook(
                    execution_device=hook.execution_device,
                    offload=hook.offload,
                    io_same_device=hook.io_same_device,
                    weights_map=hook.weights_map,
                    offload_buffers=hook.offload_buffers,
                    place_submodules=hook.place_submodules,
                    skip_keys=hook.skip_keys,
                    tied_params_map=hook.tied_params_map,
                )
                add_hook_to_module(qparams_linear, quark_hook)
            setattr_recursive(model, op_name, qparams_linear)
            float_module.to("meta")
            del float_module
            # You have to add this func to lower the peak memory.
            torch.cuda.empty_cache()


def _move_quantizer_to_dict(model: nn.Module) -> None:
    """
    Move the model's QParamsLinear quantizer to a dict which will work will tp

    Parameters:
        model (torch.nn.Module): An instance of the original not-quantized model. This model may be on `meta` device, or may have random weights.
    """
    dict_name = "_quant_dict"
    quantizer_names = ["weight_quantizer", "input_quantizer", "output_quantizer", "bias_quantizer"]
    named_modules = dict(model.named_modules(remove_duplicate=False))

    for module_name, float_module in tqdm(named_modules.items()):
        # If the current object have the quantizer specified as input names, update it to Nine and save to the dict.
        if isinstance(float_module, (torch.nn.Linear, torch.nn.Module)):
            if hasattr(float_module, dict_name):
                qdict = {}
                for quantizer_name in quantizer_names:
                    if hasattr(float_module, quantizer_name):
                        quantizer = getattr(float_module, quantizer_name, None)
                        if quantizer is not None:
                            qdict[quantizer_name] = quantizer
                            setattr(float_module, quantizer_name, None)

                if len(qdict) > 0:
                    setattr(float_module, dict_name, qdict)
