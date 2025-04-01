#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Exporting and Importing API for PyTorch."""

from __future__ import annotations
import json
import tempfile
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional, Any, Callable, cast
import dataclasses

import torch
import torch.nn as nn
from safetensors.torch import save_file
from tqdm import tqdm
import subprocess
import re
from functools import partial

from quark.shares.utils.log import ScreenLogger, log_errors
from quark.torch.quantization.utils import set_op_by_name
from quark.torch.export.main_import.pretrained_config import PretrainedConfig
from quark.torch.export.utils import preprocess_import_info
from quark.torch.quantization.nn.modules import QuantLinear
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, ScaledFakeQuantize
from quark.torch.export.config.config import JsonExporterConfig, ExporterConfig
from quark.torch.quantization.config.config import Config, QuantizationSpec
from quark.torch.quantization.config.type import QuantizationMode
from quark.torch.quantization.model_transformation import prepare_for_attention_quant
from quark.torch.export.json_export.builder.native_model_info_builder import NativeModelInfoBuilder
from quark.torch.export.main_export.model_post_process import ModelPostProcessor
from quark.torch.export.main_export.quant_config_parser import QuantConfigParser, get_layer_quant_config
from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.export.nn.modules.realquantizer import RealQuantizerBase
import quark.torch.export.nn.modules.qparamslinear

__all__ = ["ModelExporter", "save_params", "ModelImporter"]

logger = ScreenLogger(__name__)


def check_scaled_mm_available_dev() -> Optional[str]:
    '''
    Determine if torch._scaled_mm is available, there are three return values, None, "hip", "cuda"
    '''
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


class ModelExporter:
    """
    Provides an API for exporting quantized Pytorch deep learning models.
    This class converts the quantized model to json-pth, json-safetensors files or onnx graph, and saves to export_dir.

    Args:
        config (ExporterConfig): Configuration object containing settings for exporting.
        export_dir (Union[Path, str]): The target export directory. This could be a string or a pathlib.Path(string) object.
    """

    def __init__(self, config: ExporterConfig, export_dir: Union[Path, str] = tempfile.gettempdir()) -> None:
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    def export_quark_model(self, model: nn.Module, quant_config: Config, custom_mode: str = "quark") -> None:
        """
        This function aims to export json and pth files of the quantized Pytorch model by quark file format.
        The model's network architecture or configuration is stored in the json file, and parameters including weight, bias, scale, and zero_point are stored in the pth file.

        Parameters:
            model (transformers.PreTrainedModel): The quantized model to be exported.
            quant_config (Config): Configuration object containing settings for quantization. Default is None.
            custom_mode (str): Whether to export the quantization config and model in a custom format expected by some downstream library. Possible options:
                - `"quark"`: standard quark format. This is the default and recommended format that should be favored.
                - `"awq"`: targets AutoAWQ library.
                - `"fp8"`: targets vLLM-compatible fp8 models.

        Returns:
            None
        **Examples**:

            .. code-block:: python

                # default exporting:
                export_path = "./output_dir"
                from quark.torch import ModelExporter
                from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig, OnnxExporterConfig
                NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized",
                                                           pack_method="reorder")
                export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG, onnx_export_config=OnnxExporterConfig())
                exporter = ModelExporter(config=export_config, export_dir=export_path)
                quant_config = get_config(args.quant_scheme, args.group_size, args.model_dir, args.kv_cache_dtype, args.fp8_attention_quant, args.exclude_layers, args.pre_quantization_optimization, args.pre_optimization_config_file_path, args.quant_algo, args.quant_algo_config_file_path, model_type)
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
            logger.info("quark_format real_quantized model exported to {} successfully.".format(self.export_dir))
        else:
            logger.info("quark_format fake_quantized model exported to {} successfully.".format(self.export_dir))

    def get_export_model(self,
                         model: nn.Module,
                         quant_config: Config,
                         custom_mode: str = "quark",
                         add_export_info_for_hf: bool = True) -> nn.Module:
        '''
        Merges scales, replaces modules of the quantized model to prepare for export, and add export information in config.json.

        Scale merging selects the maximum scale value in specified `weight_group` as the scale for each module in the group.

        Build kv_scale selects the maximum kv_scale value in `kv_group` as the scale for the key projection output quantization and value projection output quantization.

        Module replacement converts the model's module (e.g. `QuantLinear`) according to the weight_format (to `QparamsLinear`).

        Parameters:
            model (transformers.PreTrainedModel): The quantized model to be exported.
            quant_config (Config): Configuration object containing settings for quantization.
            custom_mode (str): Whether to export the quantization config and model in a custom format expected by some downstream library. Possible options:
                - `"quark"`: standard quark format. This is the default and recommended format that should be favored.
                - `"awq"`: targets AutoAWQ library.
                - `"fp8"`: targets vLLM-compatible fp8 models.
        add_export_info_for_hf (bool): Whether to add export info of quark to config.json when using hf_format_export. When loading the model, we recover the kv_cache in autofp8 format through the weight file, but we need the name of kv_layer, it is very cumbersome to get it from quark's map, it is more reasonable to get it from config. If we find kv_scale in weight_flie and there is no special kv_layer_name, we will use k_proj,v_proj to recover kv_cache by default.
        '''

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
        self.processor = ModelPostProcessor(model,
                                            self.config.json_export_config,
                                            custom_mode=custom_mode,
                                            output_quant=quant_config.global_quant_config.output_tensors is not None)
        self.processor.merge_scale()
        model = self.processor.get_processed_model()
        return model

    def save_quark_export_model(self, model: nn.Module) -> None:
        torch.save(model.state_dict(), self.export_dir.joinpath('model_state_dict.pth'))
        with open(self.export_dir.joinpath('config.json'), 'w') as json_file:
            json.dump(model.config.to_dict(), json_file, indent=4)

    def reset_model(self, model: nn.Module) -> None:
        '''
        Restore exported model to freezed Model for inferring, restore config content.
        '''
        model.config.__dict__.pop("quantization_config")
        model = self.processor.reset_model()

    def export_onnx_model(self,
                          model: nn.Module,
                          input_args: Union[torch.Tensor, Tuple[float]],
                          input_names: List[str] = [],
                          output_names: List[str] = [],
                          verbose: bool = False,
                          opset_version: Optional[int] = None,
                          do_constant_folding: bool = True,
                          operator_export_type: torch.onnx.OperatorExportTypes = torch.onnx.OperatorExportTypes.ONNX,
                          uint4_int4_flag: bool = False) -> None:
        """
        This function aims to export onnx graph of the quantized Pytorch model.

        Parameters:
            model (torch.nn.Module): The quantized model to be exported.
            input_args (Union[torch.Tensor, Tuple[float]]): Example inputs for this quantized model.
            input_names (List[str]): Names to assign to the input nodes of the onnx graph, in order. Default is empty list.
            output_names (List[str]): Names to assign to the output nodes of the onnx graph, in order. Default is empty list.
            verbose (bool): Flag to control showing verbose log or no. Default is False
            opset_version (Optional[int]): The version of the default (ai.onnx) opset to target. If not set, it will be valued the latest version that is stable for the current version of PyTorch.
            do_constant_folding (bool): Apply the constant-folding optimization. Default is False
            operator_export_type (torch.onnx.OperatorExportTypes): Export operator type in onnx graph. The choices include OperatorExportTypes.ONNX, OperatorExportTypes.ONNX_FALLTHROUGH, OperatorExportTypes.ONNX_ATEN and OperatorExportTypes.ONNX_ATEN_FALLBACK. Default is OperatorExportTypes.ONNX.
            uint4_int4_flag (bool): Flag to indicate uint4/int4 quantized model or not. Default is False.

        Returns:
            None

        **Examples**:

            .. code-block:: python

                from quark.torch import ModelExporter
                from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
                export_config = ExporterConfig(json_export_config=JsonExporterConfig())
                exporter = ModelExporter(config=export_config, export_dir=export_path)
                exporter.export_onnx_model(model, input_args)

        Note:
            Mix quantization of int4/uint4 and int8/uint8 is not supported currently.
            In other words, if the model contains both quantized nodes of uint4/int4 and uint8/int8, this function cannot be used to export the ONNX graph.
        """
        from quark.torch.export.onnx import convert_model_to_uint4_int4
        logger.info("Start exporting quantized onnx model ...")

        for module in model.modules():
            if isinstance(module, ScaledFakeQuantize):
                module.disable_observer()
                module.enable_fake_quant()
        onnx_path = str(self.export_dir / "quark_model.onnx")
        torch.onnx.export(model.eval(),
                          input_args,
                          onnx_path,
                          verbose=verbose,
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=opset_version,
                          do_constant_folding=do_constant_folding,
                          operator_export_type=operator_export_type)
        if uint4_int4_flag:
            convert_model_to_uint4_int4(onnx_path)
        else:
            logger.info("Quantized onnx model exported to {} successfully.".format(onnx_path))

    def export_gguf_model(self, model: nn.Module, tokenizer_path: Union[str, Path], model_type: str) -> None:
        """
        This function aims to export gguf file of the quantized Pytorch model.

        Parameters:
            model (torch.nn.Module): The quantized model to be exported.
            tokenizer_path (Union[str, Path]): Tokenizer needs to be encoded into gguf model. This argument specifies the directory path of tokenizer which contains tokenizer.json, tokenizer_config.json and/or tokenizer.model
            model_type (str): The type of the model, e.g. gpt2, gptj, llama or gptnext.

        Returns:
            None

        **Examples**:

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

        logger.info("Start exporting gguf quantized model ...")

        save_params(model, model_type, export_dir=self.export_dir)

        json_path = self.export_dir / f"{model_type}.json"
        params_path = self.export_dir / f"{model_type}.safetensors"
        gguf_path = self.export_dir / f"{model_type}.gguf"
        from quark.torch.export.gguf_export.api import convert_exported_model_to_gguf
        convert_exported_model_to_gguf(model_type, json_path, params_path, tokenizer_path, gguf_path)

        if json_path.exists():
            json_path.unlink()
        if params_path.exists():
            params_path.unlink()

        logger.info("GGUF quantized model exported to {} successfully.".format(gguf_path))

    def export_model_info_from_gguf(self, model: nn.Module, gguf_path: str, model_type: str) -> None:

        logger.info("Start exporting quantized model from gguf model ...")

        params_dict: Dict[str, torch.Tensor] = {}
        builder = NativeModelInfoBuilder(model=model, config=self.config.json_export_config)
        info = builder.build_model_info(params_dict)
        from quark.torch.export.gguf_export.api import insert_quant_info_from_gguf
        info, params_dict = insert_quant_info_from_gguf(model_type, info, params_dict, gguf_path)
        json_path = self.export_dir / f"{model_type}_from_gguf.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=4)

        # handle tensors shared
        data_ptr_list: List[str] = []
        for key, value in params_dict.items():
            if str(value.data_ptr()) in data_ptr_list:
                params_dict[key] = value.clone()
            else:
                data_ptr_list.append(str(value.data_ptr()))

        params_path = self.export_dir / f"{model_type}_from_gguf.safetensors"
        save_file(params_dict, params_path)

        logger.info("Exported quantized model from gguf model to {} successfully.".format(self.export_dir))


def save_params(model: nn.Module,
                model_type: str,
                args: Optional[Tuple[Any, ...]] = None,
                kwargs: Optional[Dict[str, Any]] = None,
                export_dir: Union[Path, str] = tempfile.gettempdir(),
                quant_mode: QuantizationMode = QuantizationMode.eager_mode,
                compressed: bool = False,
                reorder: bool = True) -> None:
    """
    Save the network architecture or configurations and parameters of the quantized model.
    For eager mode quantization, the model's configurations are stored in json file, and parameters including weight, bias, scale, and zero_point are stored in safetensors file.
    For fx_graph mode quantization, the model's network architecture and parameters are stored in pth file.

    Parameters:
        model (torch.nn.Module): The quantized model to be saved.
        model_type (str): The type of the model, e.g. gpt2, gptj, llama or gptnext.
        args (Optional[Tuple[Any, ...]]): Example tuple inputs for this quantized model. Only available for fx_graph mode quantization. Default is None.
        kwargs (Optional[Dict[str, Any]]): Example dict inputs for this quantized model. Only available for fx_graph mode quantization. Default is None.
        export_dir (Union[Path, str]): The target export directory. This could be a string or a pathlib.Path(string) object.
        quant_mode (QuantizationMode): The quantization mode. The choice includes "QuantizationMode.eager_mode" and "QuantizationMode.fx_graph_mode". Default is "QuantizationMode.eager_mode".
        compressed (bool): export the compressed (real quantized) model or QDQ model, Default is False and export the QDQ model
        reorder (bool): pack method, uses pack the weight(eg. packs four torch.int8 value into one torch.int32 value). Default is True

    Returns:
        None

    **Examples**:

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
        params_dict: Dict[str, torch.Tensor] = {}
        builder = NativeModelInfoBuilder(model=model, config=JsonExporterConfig())
        info = builder.build_model_info(params_dict, compressed=compressed, reorder=reorder)
        json_path = export_dir / f"{model_type}.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=4)

        # handle tensors shared
        data_ptr_list: List[str] = []
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

    logger.info("Parameters of quantized model saved to {} successfully.".format(export_dir))


@log_errors
class ModelImporter:
    """
    Provides an API for importing quantized Pytorch deep learning models.
    This class load json-pth or json-safetensors files to model.

    Args:
        model_info_dir (str): The target import directory.
    """

    def __init__(self, model_info_dir: str) -> None:
        self.model_info_dir = model_info_dir

    def get_model_config(self) -> PretrainedConfig:
        model_config = PretrainedConfig(pretrained_dir=self.model_info_dir)
        return model_config

    def get_model_state_dict(self) -> Dict[str, Any]:
        model_state_dict: Dict[str, Any] = torch.load(Path(self.model_info_dir) / "model_state_dict.pth")
        return model_state_dict

    def import_model_info(self, model: nn.Module) -> nn.Module:
        """
        This function aims to import quark(json-pth) files of the HuggingFace large language model.

        It could recover the weight, bias, scale, and zeropoint information of the model and execute the inference

        Parameters:
            model (transformers.PreTrainedModel): The original HuggingFace large language model.

        Returns:
            model: Models that have completed weight import
        **Examples**:

            .. code-block:: python

                # default exporting:
                import_model_dir = "./import_model_dir"
                from quark.torch import ModelImporter
                importer = ModelImporter(model_info_dir=args.import_model_dir)
                model = importer.import_model_info(model)

        """
        logger.info("Start importing quark_format(pth_json) quantized model ...")
        model_config = self.get_model_config()
        model_state_dict = self.get_model_state_dict()
        model = self.import_model(model, model_config, model_state_dict)
        model.load_state_dict(model_state_dict)
        logger.info("quark_format(pth_json) quantized model imported successfully.")
        return model

    def import_model(self, model: nn.Module, model_config: PretrainedConfig, model_state_dict: Dict[str,
                                                                                                    Any]) -> nn.Module:
        """
        This function uses the loaded state_dict and config to build the model
        """
        quark.torch.export.nn.modules.qparamslinear.SCALED_MM_AVAILABLE_DEV = check_scaled_mm_available_dev()
        if model_config.quantization_config is None:
            logger.info("This is a non-quantized model")
            return model
        custom_mode = model_config.quantization_config["quant_method"]
        assert custom_mode in ["fp8", "awq", "quark"]
        is_kv_cache = False
        model_state_dict, is_kv_cache, kv_layers_name = preprocess_import_info(
            model_state_dict=model_state_dict,
            is_kv_cache=is_kv_cache,
            kv_layers_name=model_config.kv_layers_name,
            custom_mode=custom_mode)
        if custom_mode != "quark":
            # For `"awq"` and `"fp8"` custom modes, there is no way to determine whether bias is quantized simply looking at the serialized `quantization_config`.
            is_bias_quantized = any("bias.scales" in key or "bias_scale" in key for key in model_state_dict.keys())
            quantization_config = QuantConfigParser.from_custom_config(model_config.quantization_config,
                                                                       is_bias_quantized=is_bias_quantized,
                                                                       is_kv_cache=is_kv_cache,
                                                                       kv_layers_name=kv_layers_name)
        else:
            quantization_config = Config.from_dict(model_config.quantization_config)

        is_real_quantized_mode = True if model_config.weight_format in [None, "real_quantized"] else False

        if quantization_config.softmax_quant_spec is not None:
            if is_real_quantized_mode:
                get_quantize = partial(RealQuantizerBase.from_fake_quantizer,
                                       quantizer=None,
                                       reorder=False,
                                       real_quantized=False,
                                       float_dtype=torch.float32)
                get_quantize = cast(Callable[[QuantizationSpec], Union[FakeQuantizeBase, RealQuantizerBase]],
                                    get_quantize)
            else:
                get_quantize = FakeQuantizeBase.get_fake_quantize
            prepare_for_attention_quant(model, quantization_config, get_quantize)

        if is_real_quantized_mode:
            logger.info("In-place OPs replacement start.")
            _map_to_quark(
                model,
                quantization_config,
                model_config.pack_method,  # type: ignore[arg-type]
                custom_mode)

        # The high precision (fake quantize) serialization is only used by quark format.
        else:
            logger.info("In-place OPs replacement start.")
            named_modules = dict(model.named_modules(remove_duplicate=False))
            for name, float_module in tqdm(named_modules.items()):
                layer_quantization_config = get_layer_quant_config(quantization_config, type(float_module), name)
                if layer_quantization_config is not None and isinstance(float_module, nn.Linear):
                    # TODO: add other types of modules, will del "original save_param and load_params in quantize_quark.py"
                    quant_module = QuantLinear.from_float(float_module, layer_quantization_config)
                    set_op_by_name(model, name, quant_module)
            named_modules = dict(model.named_modules(remove_duplicate=False))
            for name, module in named_modules.items():
                if isinstance(module, FakeQuantizeBase):
                    freezed_quantized_module = module.to_freezed_module()
                    set_op_by_name(model, name, freezed_quantized_module)
        logger.info("Converting quantized ops end")

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

            set_op_by_name(model, op_name, qparams_linear)
