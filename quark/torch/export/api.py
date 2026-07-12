#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Exporting and Importing API for PyTorch."""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from tqdm import tqdm

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import QConfig

from quark.common.utils.import_utils import (
    is_accelerate_available,
    is_diffusers_available,
    is_gguf_available_and_minimum_version,
    is_safetensors_available,
    is_transformers_available,
    is_transformers_version_higher_or_equal,
)
from quark.common.utils.log import ScreenLogger
from quark.torch.algorithm.rotation.rotation import RotationProcessor
from quark.torch.export.config.config import JsonExporterConfig
from quark.torch.export.constants import QUARK_AWQ_WEIGHT_CONVERSIONS, QUARK_WEIGHT_CONVERSIONS
from quark.torch.export.json_export.builder.native_model_info_builder import NativeModelInfoBuilder
from quark.torch.export.main_export.model_post_process import ModelPostProcessor
from quark.torch.export.main_export.quant_config_parser import QuantConfigParser, get_layer_quant_config
from quark.torch.export.main_import.pretrained_config import PretrainedConfig
from quark.torch.export.nn.modules.qparamslinear import QParamsLinear, QParamsLinearWithRotation
from quark.torch.export.onnx import convert_model_to_uint4_int4, export_onnx_model_optimization
from quark.torch.export.safetensors import _load_weights_from_safetensors, export_hf_model
from quark.torch.export.utils import (
    _build_quantized_model,
    _convert_quantized_model,
    _fix_loaded_weights_key_mismatch,
    _fix_state_dict_key_on_save,
    _handle_multi_device_loading,
    _synthesize_missing_zero_points,
    _untie_parameters,
    restore_aux_files,
)
from quark.torch.quantization.config.type import QuantizationMode
from quark.torch.quantization.inverse_quantizer import is_prequantized_linear
from quark.torch.quantization.model_transformation import (
    export_cache_state_dict_from_model,
    import_model_with_cache_from_safetensors,
)
from quark.torch.quantization.tensor_quantize import NonScaledFakeQuantize, ScaledFakeQuantize
from quark.torch.utils import getattr_recursive, setattr_recursive

if is_gguf_available_and_minimum_version():
    from quark.torch.export.gguf_export.api import convert_exported_model_to_gguf
if is_transformers_available():
    from transformers import PreTrainedModel
    from transformers.modeling_utils import _get_tied_weight_keys
if is_accelerate_available():
    from accelerate import dispatch_model
    from accelerate.hooks import add_hook_to_module
    from accelerate.utils import infer_auto_device_map
if is_diffusers_available() or TYPE_CHECKING:
    from diffusers import ModelMixin
if is_safetensors_available():
    from safetensors.torch import save_file

from quark.torch.utils.accelerate_helper import clone_align_devices_hook
from quark.torch.utils.llm.device_mapping import create_auto_adjusted_device_map
from quark.torch.utils.llm.model_preparation import create_model_skeleton, get_device_max_memory

__all__ = [
    "export_safetensors",
    "export_onnx",
    "export_gguf",
    "export_llama_cpp_gguf",
    "list_llama_cpp_export_formats",
    "import_model_from_safetensors",
    "save_params",
]

logger = ScreenLogger(__name__)


def _get_submodule_or_none(model: nn.Module, module_path: str) -> nn.Module | None:
    """Resolve a dotted submodule path without raising exceptions.

    Works with nested modules and ModuleList entries (numeric names).
    Returns None if any segment is missing.
    """
    cur: nn.Module = model
    for part in module_path.split("."):
        modules = getattr(cur, "_modules", None)
        if not isinstance(modules, dict) or part not in modules:
            return None
        cur = modules[part]
    return cur


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
        self,
        model: torch.nn.Module,
        output_dir: Path,
        custom_mode: str,
        weight_format: str,
        pack_method: str,
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
        self, quant_config: QConfig, temp_json_config: JsonExporterConfig
    ) -> dict[str, Any]:
        """Prepare quantization configuration. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _prepare_quantization_config")

    def _export_impl(self) -> None:
        """Export model to Safetensors format."""
        # Get quant_config from the model
        quant_config = getattr(self.model, "quant_config", None)

        # Create a copy of the model to avoid modifying the original
        original_config = self.model.config.__dict__.copy()

        # Prepare cache for export if present
        cache_prepared = False
        if quant_config is not None:
            from quark.torch.quantization.model_transformation import prepare_model_for_cache_export

            cache_prepared = prepare_model_for_cache_export(self.model, quant_config)

        # Create temporary config objects for processing
        # If kv_cache is not quantized, there is no need to export kv_cache name.
        kv_cache_group = (
            getattr(quant_config, "kv_cache_group", []) if getattr(quant_config, "kv_cache_quant_config", {}) else []
        )
        temp_json_config = JsonExporterConfig(
            weight_format=self.weight_format,
            pack_method=self.pack_method,
            kv_cache_group=kv_cache_group,
            min_kv_scale=getattr(quant_config, "min_kv_scale", 0.0),
        )

        # Process model before building the config dict below, so preserve-prequantized
        # mutations to `layer_quant_config` / `exclude` land in the saved config.json.
        processor = ModelPostProcessor(
            self.model,
            temp_json_config,
            custom_mode=self.custom_mode,
            output_quant=quant_config is not None and quant_config.global_quant_config.output_tensors is not None,
            quantization_config=quant_config,
        )
        processor.merge_scale()
        processed_model = processor.get_processed_model()

        if quant_config is not None:
            quantization_config_dict = self._prepare_quantization_config(quant_config, temp_json_config)
            self.model.config.update({"quantization_config": quantization_config_dict})

        # Export cache state dict if present
        cache_state_dict = {}
        if cache_prepared and quant_config is not None:
            cache_state_dict = export_cache_state_dict_from_model(self.model, quant_config)
            if cache_state_dict:
                logger.info(f"Including cache quantization state dict with {len(cache_state_dict)} entries")

        # Export using HF format (single call), optionally attach alias-only output scales
        alias_only_cache: dict[str, torch.Tensor] = (
            {k: v for k, v in cache_state_dict.items() if k.endswith(".output_scale")} if cache_state_dict else {}
        )

        inserted_buffers: list[tuple[nn.Module, str]] = []
        if alias_only_cache:
            for full_key, tensor in alias_only_cache.items():
                module_path = full_key.rsplit(".", 1)[0]
                target_module = _get_submodule_or_none(processed_model, module_path)
                if target_module is None:
                    logger.debug(
                        "[CACHE EXPORT] Skipping alias %s because module path %s was not found",
                        full_key,
                        module_path,
                    )
                    continue

                existing_attr = getattr(target_module, "output_scale", None)
                if isinstance(existing_attr, torch.Tensor):
                    with torch.no_grad():
                        existing_attr.copy_(tensor)
                elif existing_attr is None:
                    target_module.register_buffer("output_scale", tensor.clone())
                    inserted_buffers.append((target_module, "output_scale"))
                else:
                    logger.debug(
                        "[CACHE EXPORT] Unable to attach output_scale to %s because attribute already exists",
                        module_path,
                    )
                    continue

        if self.weight_format == "real_quantized":
            # Useful only for Transformers models, see the comment below regarding the serialization keys.
            if is_transformers_version_higher_or_equal("4.99"):
                # like weight_quantizer.0.scale, weight_quantizer.1.scale

                if self.custom_mode == "awq":
                    processed_model._weight_conversions = QUARK_AWQ_WEIGHT_CONVERSIONS
                else:
                    processed_model._weight_conversions = QUARK_WEIGHT_CONVERSIONS
            else:
                processed_model._fix_state_dict_key_on_save = staticmethod(_fix_state_dict_key_on_save)

        # Export using HF format
        export_hf_model(model=processed_model, export_dir=str(self.output_dir))

        # Restore tokenizer and auxiliary files from the source checkpoint. Tokenizer files are
        # always overwritten to preserve the original format (tools such as maybe_save_preprocessors
        # may have written a re-serialized version with a different tokenizer class). Other
        # auxiliary files are only copied when missing. Best-effort: never fails a successful export.
        restore_aux_files(original_config.get("_name_or_path"), self.output_dir)

        # Clean up any temporary buffers we inserted
        for module, buffer_name in inserted_buffers:
            buffers = getattr(module, "_buffers", None)
            if isinstance(buffers, dict) and buffer_name in buffers:
                del buffers[buffer_name]
            if hasattr(module, buffer_name):
                delattr(module, buffer_name)

        # Reset model config to original state
        self.model.config.__dict__.clear()
        self.model.config.__dict__.update(original_config)

        logger.info(f"Successfully exported model to Safetensors format in {self.custom_mode} mode: {self.output_dir}")


class QuarkSafetensorsExporter(SafetensorsExporter):
    """Exporter for Safetensors format in quark mode."""

    def _validate(self) -> None:
        """Validate quark mode export parameters."""
        super()._validate()
        if self.custom_mode != "quark":
            raise ValueError(f"QuarkSafetensorsExporter only supports custom_mode='quark', got {self.custom_mode}.")

    def _prepare_quantization_config(
        self, quant_config: QConfig, temp_json_config: JsonExporterConfig
    ) -> dict[str, Any]:
        """Prepare quantization configuration for quark mode."""
        quark_quant_config = quant_config.to_dict()
        quantization_config_dict = {}

        # Handle quark mode
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
        self, quant_config: QConfig, temp_json_config: JsonExporterConfig
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


class DiffusersSafetensorsExporter(BaseExporter):
    """Exporter for diffusers ModelMixin models to safetensors format."""

    def __init__(self, model: ModelMixin, output_dir: Path) -> None:
        super().__init__()
        self.model = model
        self.output_dir = output_dir

    def _validate(self) -> None:
        if not is_diffusers_available() or not isinstance(self.model, ModelMixin):
            raise NotImplementedError("DiffusersSafetensorsExporter only supports diffusers ModelMixin models.")

    def _export_impl(self) -> None:
        self.model.save_pretrained(self.output_dir)  # type: ignore[operator]
        logger.info(f"Successfully exported diffusers model to {self.output_dir}")

        quant_config = getattr(self.model, "quant_config", None)
        if quant_config is not None:
            config_path = self.output_dir / "config.json"
            with open(config_path) as f:
                config = json.load(f)
            config["quantization_config"] = quant_config.to_dict()
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)


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
        dynamo: bool = False,
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
        self.dynamo = dynamo

    def _validate(self) -> None:
        """Validate ONNX export parameters."""
        # Basic validation - ONNX export is generally more permissive
        if not isinstance(self.input_args, torch.Tensor | tuple):
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
        try:
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
                dynamo=self.dynamo,
            )
        except Exception as e:
            if not self.dynamo:
                raise RuntimeError(
                    f"The ONNX export failed during `torch.onnx.export` call. This could be due to an issue in the model definition not compatible with torch.jit.trace-based ONNX export, or other issues. Consider trying to export using `dynamo=True`, refer to `quark.torch.export.api.export_onnx` API documentation. Error: {e}"
                ) from e
            else:
                raise RuntimeError(
                    f"The ONNX export failed during `torch.onnx.export` call. Consider trying to export using `dynamo=False`, refer to `quark.torch.export.api.export_onnx` API documentation. Error: {e}"
                ) from e

        export_onnx_model_optimization(onnx_path)

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

    def __init__(self, model: torch.nn.Module, output_dir: Path, model_type: str, tokenizer_path: str | Path) -> None:
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
    output_dir: str | Path,
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
    # Diffusers models use a dedicated exporter that calls ModelMixin.save_pretrained directly.
    if is_diffusers_available() and isinstance(model, ModelMixin):
        exporter: BaseExporter = DiffusersSafetensorsExporter(model=model, output_dir=Path(output_dir))
        exporter._export()
        return

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
    output_dir: str | Path,
    input_args: tuple[Any, ...],
    opset_version: int | None = None,
    input_names: list[str] = [],
    output_names: list[str] = [],
    verbose: bool = False,
    do_constant_folding: bool = True,
    operator_export_type: torch.onnx.OperatorExportTypes = torch.onnx.OperatorExportTypes.ONNX,
    uint4_int4_flag: bool = False,
    dynamo: bool = False,
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
    :param bool dynamo: Whether to export the model with ``torch.export`` ExportedProgram instead of TorchScript. Please refer to `PyTorch documentation <https://docs.pytorch.org/docs/stable/onnx_dynamo.html#torch.onnx.export>`__ for more details. Defaults to ``False``.

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
        dynamo=dynamo,
    )
    exporter._export()


def export_gguf(
    model: torch.nn.Module,
    output_dir: str | Path,
    model_type: str,
    tokenizer_path: str | Path,
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
    if not is_gguf_available_and_minimum_version():
        raise ImportError(
            "The function `export_gguf` requires the package `gguf>=0.10.0` to be installed, but it was not found. Please install `gguf>=0.10.0`."
        )

    exporter = GgufExporter(
        model=model, output_dir=Path(output_dir), model_type=model_type, tokenizer_path=tokenizer_path
    )
    exporter._export()


def export_llama_cpp_gguf(
    quark_model_dir: str | Path,
    output_dir: str | Path,
    export_format: str,
    *,
    name: str | None = None,
    llama_cpp_dir: str | Path = "/home/l/work/llama.cpp",
    libggml: str | Path | None = None,
    tokenizer_source: str | Path | None = None,
    group_size: int = 128,
    pack_method: str = "reorder",
    split_max_size: str = "8G",
    max_tensors: int | None = None,
    dry_run: bool = False,
    keep_staging: bool = False,
) -> Path:
    """Export a Quark AWQ checkpoint to llama.cpp-compatible GGUF.

    This is separate from ``export_gguf``, which only supports a narrow set of
    native Quark GGUF layouts. ``export_llama_cpp_gguf`` targets the public
    llama.cpp quant families such as ``q4_k_m``, ``q8_0``, and ``iq4_nl``.

    See ``quark.torch.export.llama_cpp_export`` for format details.
    """
    from quark.torch.export.llama_cpp_export.api import export_llama_cpp_gguf as _export

    return _export(
        quark_model_dir=quark_model_dir,
        output_dir=output_dir,
        export_format=export_format,
        name=name,
        llama_cpp_dir=llama_cpp_dir,
        libggml=libggml,
        tokenizer_source=tokenizer_source,
        group_size=group_size,
        pack_method=pack_method,
        split_max_size=split_max_size,
        max_tensors=max_tensors,
        dry_run=dry_run,
        keep_staging=keep_staging,
    )


def list_llama_cpp_export_formats() -> list[str]:
    """List llama.cpp export formats supported by ``export_llama_cpp_gguf``."""
    from quark.torch.export.llama_cpp_export.api import list_llama_cpp_export_formats as _list

    return _list()


class BaseImporter(ABC):
    """Base class for all model importers."""

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

        # For some transformer models, the huggingface checkpoint keys are different from the model.state_dict keys.
        # So we need to convert the checkpoint keys to model.state_dict keys using the _checkpoint_conversion_mapping attribute.
        if hasattr(self.model, "_checkpoint_conversion_mapping") and len(self.model._checkpoint_conversion_mapping) > 0:
            checkpoint_conversion_mapping = self.model._checkpoint_conversion_mapping
            converted_weights = {}
            for name, param in checkpoint_weights.items():
                new_name = name
                for pattern, replacement in checkpoint_conversion_mapping.items():
                    converted_name = re.sub(pattern, replacement, name)
                    if converted_name != name:
                        new_name = converted_name
                        break  # Apply first matching pattern only
                converted_weights[new_name] = param
            checkpoint_weights = converted_weights

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
                if not self.multi_device:
                    for name, module in self.model.named_modules():
                        for buffer_name in module._non_persistent_buffers_set:
                            # NOTE: How could a buffer be None here? Does this case really exist anywhere?
                            buffer = getattr(module, buffer_name, None)
                            if buffer is not None and buffer.device.type == "meta":
                                raise ValueError(
                                    "Importing a model containing non-persistent buffers on meta device is not supported. Consider initializing the model with `no_init_weights()` and `init_empty_weights()` context to initialize non-persistent buffers only."
                                )

                # Initialize non-persistent buffers from meta device (only when multi_device=True)
                for name, module in self.model.named_modules():
                    if len(module._non_persistent_buffers_set) > 0:
                        target_device = (
                            module._hf_hook.execution_device
                            if hasattr(module, "_hf_hook")
                            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        )
                        for buffer_name in module._non_persistent_buffers_set:
                            buffer = getattr(module, buffer_name, None)
                            if buffer is not None and buffer.device.type == "meta":
                                setattr(
                                    module,
                                    buffer_name,
                                    torch.zeros(
                                        buffer.shape,
                                        dtype=buffer.dtype,
                                        device=target_device,
                                    ),
                                )

        # Build model with quantization support
        model = _build_quantized_model(self.model, model_config, checkpoint_weights)

        # Save cache-related keys BEFORE any filtering or state_dict operations
        # (needed for real_quantized mode where these keys are not in model's state_dict)
        cache_state_dict = {
            k: v for k, v in checkpoint_weights.items() if ".output_scale" in k and ("k_proj" in k or "v_proj" in k)
        }

        # e.g. fix .qweight -> .weight_quantizer.weight
        # .weight_scale -> .weight_quantizer.scale
        checkpoint_weights = _fix_loaded_weights_key_mismatch(
            checkpoint_weights,
            weight_format=model_config.weight_format,
            custom_mode=model_config.quantization_config["quant_method"],
        )

        # Symmetric file-to-file exports may omit flat zero_point keys.
        model_state_dict = model.state_dict()
        checkpoint_weights = _synthesize_missing_zero_points(
            checkpoint_weights, model_state_dict
        )

        # Handle parameter untying
        if is_accelerate_available():
            _untie_parameters(model, checkpoint_weights)

        if cache_state_dict:
            logger.debug(
                f"Saved {len(cache_state_dict)} cache keys before filtering. Cache keys: {', '.join(list(cache_state_dict.keys()))}."
            )

        # In case we are loading the quantized weights into a model that is not on meta device,
        # we re-use the original device the weights were placed on, as `assign=True` is used later.
        # This is helpful e.g. in case the original model was dispatched to multiple
        # devices ahead of time with `accelerate`.
        missing_keys = []
        for name, param in model_state_dict.items():
            if name not in checkpoint_weights:
                missing_keys.append(name)
            else:
                if param.device.type != "meta":
                    checkpoint_weights[name] = checkpoint_weights[name].to(param.device)

        # NOTE: We may be loading a checkpoint with untied weights (e.g. embed_tokens.weight and lm_head.weight are not duplicated in the checkpoint). In this case, and only in this case, we authorize missing keys, handled later by `model.tie_weights()`.
        if hasattr(model, "tie_weights"):
            tied_weights_keys = _get_tied_weight_keys(model)
            if len(set(missing_keys) - set(tied_weights_keys)) > 0:
                raise ValueError(
                    f"The loaded checkpoint misses the keys {missing_keys} present in the model weights (which is larger than the tied weights {tied_weights_keys}). Some weights are not initialized. Please open an issue or double check your model."
                )
        elif len(missing_keys) > 0:
            raise ValueError(
                f"The loaded checkpoint misses the weight keys {missing_keys} present in the model weights. Some weights are not initialized. Please open an issue or double check your model."
            )

        # Handle multi-device loading if enabled
        if self.multi_device and is_accelerate_available():
            _handle_multi_device_loading(model, checkpoint_weights)

        # Load weights into model with strict=False to handle missing quantization parameters
        model.load_state_dict(checkpoint_weights, assign=True, strict=False)

        if hasattr(model, "tie_weights"):
            # Main goal here is to call `model.tie_weights()`, as tied weights may have not been loaded.
            # NOTE: Quark somewhat "supports" quantizing `lm_head` nn.Linear. However, there is no proper handling of tied weights in this case (embed_tokens still serialized in bf16/fp16 - thus in practice untied).
            # Essentially, we do not want `tie_weights()` to replace quantized weights (e.g. lm_head), that were previously loaded in `model.load_state_dict`, by their non-quantized counterpart (e.g. embed_tokens).
            tied_weights_keys = _get_tied_weight_keys(model)
            data_ptrs = {name: getattr_recursive(model, name).data_ptr() for name in tied_weights_keys}

            model.tie_weights()

            modified_parameters_names = [
                name for name in tied_weights_keys if getattr_recursive(model, name).data_ptr() != data_ptrs[name]
            ]

            if len(set(missing_keys) - set(modified_parameters_names)) > 0:
                raise ValueError(
                    f"The loaded checkpoint misses the weight keys {missing_keys} present in the model weights, and weight tying did not initialize them properly (`model.tie_weights()` tied only {modified_parameters_names}. Please open an issue."
                )

            checkpoint_weights = {
                key: checkpoint_weights[key] for key in modified_parameters_names if key in checkpoint_weights
            }

            for name in checkpoint_weights:
                # We do not use `load_state_dict` here as we may have a size mismatch after tying weights.
                setattr_recursive(model, name, torch.nn.Parameter(checkpoint_weights[name], requires_grad=False))

        # Convert model
        model = _convert_quantized_model(model, model_config)

        config_from_model = getattr(model, "quant_config", None)
        if config_from_model is not None:
            # Use the saved cache state dict for import
            import_model_with_cache_from_safetensors(model, cache_state_dict, config_from_model)

        logger.info("safetensors quantized model imported successfully.")
        return model


def import_model_from_safetensors(
    model: torch.nn.Module | None,
    model_dir: str,
    multi_device: bool = False,
    trust_remote_code: bool = False,
    attn_implementation: str = "eager",
    device: str = "cuda",
    multi_gpu: str | bool | None = False,
) -> torch.nn.Module:
    """
    Imports a quantized model from the local directory ``model_dir``.

    When ``model`` (a non-quantized ``nn.Module``) is provided, quantized weights are loaded into it and returned as-is.

    The option ``model=None`` is only supported for Transformers (https://github.com/huggingface/transformers) models, not arbitrary ``nn.Module``. In this case, a model skeleton is automatically created from the config
    in ``model_dir`` on meta device (zero memory), quantized weights are loaded, and the
    model is dispatched to the target device(s) before returning.

    :param model: The model to load weights into, or ``None`` to create a skeleton automatically.
    :param str model_dir: Directory containing the model files (``config.json`` and ``model.safetensors``)
    :param bool multi_device: Whether to use multi-device loading using Accelerate library. Defaults to ``False``.
    :param bool trust_remote_code: Whether to trust remote code for custom models. Only used when ``model`` is ``None``. Defaults to ``False``.
    :param str attn_implementation: Attention implementation to use. Only used when ``model`` is ``None``. Defaults to ``"eager"``.
    :param str device: Target device when not using multi-GPU. Only used when ``model`` is ``None``. Defaults to ``"cuda"``.
    :param multi_gpu: Multi-GPU strategy. Only used when ``model`` is ``None``. Defaults to ``False``.

    :return: The model with loaded weights and proper quantization modules.
    """
    importer = SafetensorsImporter()

    # model is not None → load weights into existing model, return as-is.
    if model is not None:
        return importer._import(model=model, model_dir=model_dir, multi_device=multi_device)

    # model is None → create skeleton, load weights, then dispatch to device below.
    model = create_model_skeleton(
        model_dir=model_dir,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    model = importer._import(model=model, model_dir=model_dir, multi_device=multi_device)

    if multi_gpu or multi_device:  # pragma: no cover
        max_memory = get_device_max_memory() if multi_device else None
        device_map = create_auto_adjusted_device_map(model.config, max_memory=max_memory)
        if isinstance(device_map, str):
            logger.warning(
                "create_auto_adjusted_device_map returned '%s', falling back to infer_auto_device_map", device_map
            )
            device_map = infer_auto_device_map(model, max_memory=max_memory)
        model = dispatch_model(model, device_map)
    else:
        model = model.to(device)

    model.eval()
    return model


def save_params(
    model: nn.Module,
    model_type: str,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    export_dir: Path | str = tempfile.gettempdir(),  # noqa: B008
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

    for name, submodule in model.named_modules():
        if isinstance(submodule, ScaledFakeQuantize | NonScaledFakeQuantize):
            if not submodule.frozen_params and not submodule.is_dynamic:
                raise ValueError(
                    f"`model = ModelQuantizer.freeze(model)` needs to be called prior to running `save_params`, but found soft parameters in the model (in {name}). Please double check your code or open an issue."
                )

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


def _map_to_quark(model: nn.Module, quantization_config: QConfig, pack_method: str, custom_mode: str) -> None:
    """
    Maps a non-quantized model (possibly on meta device) to a model with QParamsLinear layers with weights not initialized. This function is useful to later load a checkpoint in the quark model using `model.load_state_dict(state_dict)`.

    Parameters:
        model (torch.nn.Module): An instance of the original not-quantized model. This model may be on `meta` device, or may have random weights.
        quantization_config (QConfig): The quantization configuration orginally used to quantize the model in Quark.
        pack_method (str): The packing method used when the model was serialized.
        custom_mode (str): The custom mode to use to initialize the `QParamsLinear` layers. The recommended mode is simply quark-native `"quark"`, but `"awq"` and `"fp8"` are also available.
    """
    named_modules = dict(model.named_modules(remove_duplicate=False))

    layers_online_rotation = []
    rotation_config = quantization_config.get_rotation_config()
    if rotation_config is not None:
        layers_online_rotation = RotationProcessor.get_online_rotation_layers(rotation_config, model)

        if rotation_config.r3:
            raise NotImplementedError(
                "Reloading a model quantization using rotation algorithm with r3=True is not supported at the moment. Please open an issue."
            )

    for op_name, float_module in tqdm(named_modules.items()):
        # Check if this is a pre-quantized linear (e.g. FP8Linear, compressed linear layer from compressed-tensors library).
        is_prequant = is_prequantized_linear(float_module)

        # For pre-quantized linears, use nn.Linear type to get config; otherwise use actual type
        config_type = nn.Linear if is_prequant else type(float_module)
        layer_quantization_config = get_layer_quant_config(quantization_config, config_type, op_name)

        # Handle both nn.Linear and pre-quantized linears in quantization list
        if layer_quantization_config is not None and (isinstance(float_module, nn.Linear) or is_prequant):
            if op_name in layers_online_rotation:
                qparams_linear_cls = QParamsLinearWithRotation
            else:
                qparams_linear_cls = QParamsLinear

            qparams_linear = qparams_linear_cls.from_module(
                linear=float_module,
                custom_mode=custom_mode,
                pack_method=pack_method,
                algo_config=quantization_config.algo_config,
                quant_config=layer_quantization_config,
            )

            # for multi_device, hook can offer info.
            if hasattr(float_module, "_hf_hook"):
                hook = float_module._hf_hook
                quark_hook = clone_align_devices_hook(hook)
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
        if isinstance(float_module, torch.nn.Linear | torch.nn.Module):  # pragma: no cover
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
