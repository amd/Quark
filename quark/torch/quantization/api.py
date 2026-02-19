#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization API for PyTorch."""

import json
import time
from typing import Any, Iterable

import torch
import torch.fx
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import quark.torch.kernel
from quark.shares.utils.import_utils import is_safetensors_available, is_transformers_available
from quark.shares.utils.log import ScreenLogger, log_errors
from quark.torch.algorithm.api import add_algorithm_config_by_model, apply_advanced_quant_algo
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.config_verification import ConfigVerifier
from quark.torch.quantization.config.type import Dtype, QSchemeType, QuantizationMode
from quark.torch.quantization.file2file_quantization import quantize_model_per_safetensor
from quark.torch.quantization.graph.processor.pre_check_befor_quant import check_supported_model_and_config
from quark.torch.quantization.graph.processor.processor import (
    post_calib_optimize,
    post_quant_optimize,
    prepare_quant_model,
)
from quark.torch.quantization.model_transformation import process_model_transformation
from quark.torch.quantization.nn.modules import (
    QuantConv2d,
    QuantConvTranspose2d,
    QuantEmbedding,
    QuantEmbeddingBag,
    QuantLinear,
)
from quark.torch.quantization.nn.modules.mixin import QuantMixin
from quark.torch.quantization.tensor_quantize import (
    FakeQuantizeBase,
    NonScaledFakeQuantize,
    ScaledFakeQuantize,
    SequentialQuantize,
    enable_or_disable_quantizer,
)
from quark.torch.quantization.utils import count_calibration_tokens, deep_compare
from quark.torch.utils import (
    QUARK_COUNT_OBSERVED_SAMPLES,
    QUARK_TOKENS_DISTRIBUTION_PATH,
    TOKEN_DISTRIBUTION_THRESHOLD,
    create_pack_method,
    getattr_recursive,
    gpu_memory_profiled,
    setattr_recursive,
)

if is_transformers_available():
    from transformers.feature_extraction_utils import BatchFeature

import os
from collections import Counter
from pathlib import Path

from quark.torch.quantization.debug import check_scale_stats, collect_quantization_statistics, insert_stats_hooks

if is_safetensors_available():
    from safetensors.torch import load_file

__all__ = ["ModelQuantizer", "load_params"]

logger = ScreenLogger(__name__)

QUARK_QUANT_OPS: dict[
    str, type[QuantConv2d | QuantConvTranspose2d | QuantLinear | QuantEmbedding | QuantEmbeddingBag]
] = {
    "QuantConv2d": QuantConv2d,
    "QuantConvTranspose2d": QuantConvTranspose2d,
    "QuantLinear": QuantLinear,
    "QuantEmbedding": QuantEmbedding,
    "QuantEmbeddingBag": QuantEmbeddingBag,
}


class ModelQuantizer:
    """
    Provides an API for quantizing deep learning models using PyTorch.

    This class handles the configuration and processing of the model for quantization based on user-defined parameters. It is essential to ensure that the 'config' provided has all necessary quantization parameters defined. This class assumes that the model is compatible with the quantization settings specified in 'config'.

    :param QConfig config: The model quantization configuration.
    """

    def __init__(self, config: QConfig, multi_device: bool = False) -> None:
        self.config = config
        self._is_accelerate: bool | None = None
        self.multi_device: bool = multi_device
        self.config_verifier = ConfigVerifier(config)
        self.init_config()

    @gpu_memory_profiled(tag=" QuantizeModel")  # type: ignore[arg-type]
    def quantize_model(
        self,
        model: nn.Module,
        dataloader: DataLoader[torch.Tensor]
        | DataLoader[list[dict[str, torch.Tensor]]]
        | DataLoader[dict[str, torch.Tensor]]
        | DataLoader[list["BatchFeature"]]
        | None = None,
    ) -> nn.Module:
        """
        Quantizes the given PyTorch model to optimize its performance and reduce its size.

        The dataloader is used to provide data necessary for calibration during the quantization process. Depending on the type of data provided (either tensors directly or structured as lists or dictionaries of tensors), the function will adapt the quantization approach accordingly.

        It is important that the model and dataloader are compatible in terms of the data they expect and produce. Misalignment in data handling between the model and the dataloader can lead to errors during the quantization process.

        :param torch.nn.Module model: The PyTorch model to be quantized. This model should be already trained and ready for quantization.

        :param Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]], DataLoader[Dict[str, torch.Tensor]], DataLoader[List[BatchFeature]]]] dataloader: The ``torch.utils.data.DataLoader`` providing data that the quantization process will use for calibration. This can be a simple ``DataLoader`` returning tensors, or a more complex structure returning either a list of dictionaries or a dictionary of tensors.

        :return: The quantized version of the input model. This model is now optimized for inference with reduced size and potentially improved performance on targeted devices.
        :rtype: torch.nn.Module

        Example:

        .. code-block:: python

            # Model & Data preparation
            from torch.utils.data import DataLoader
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from quark.torch.quantization.config.config import QConfig
            from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
            from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver

            from quark.torch import ModelQuantizer

            model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype="auto")
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

            quant_spec = QTensorConfig(
                dtype=Dtype.uint4,
                observer_cls=PerGroupMinMaxObserver,
                symmetric=False,
                scale_type=ScaleType.float,
                round_method=RoundType.half_even,
                qscheme=QSchemeType.per_group,
                ch_axis=1,
                is_dynamic=False,
                group_size=128
            )
            quant_config = QConfig(global_quant_config=QLayerConfig(weight=quant_spec))

            text = "Hello, how are you?"
            tokenized_outputs = tokenizer(text, return_tensors="pt")
            calib_dataloader = DataLoader(tokenized_outputs['input_ids'])

            quantizer = ModelQuantizer(quant_config)
            quant_model = quantizer.quantize(model, calib_dataloader)
        """
        logger.info(f"Quantizing with the quantization configuration:\n{self.config}")

        # Step0: Pre quant device check
        self._check_model_device(model)

        # Step1: Prepare quantization model for graph mode and eager mode
        model = self._prepare_model(model)

        # Step2[optional]: Apply Advanced quant algo such as gptq, awq, qronos ...
        model = self._apply_advanced_quant_algo(model, dataloader)

        # Step3[optional]: Do calibration
        model = self._do_calibration(model, dataloader)

        # Step4[optional]: Post calib optimization
        model = self._do_post_calib_optimazation(model)

        # Optionally, collect statistics on the quantization errors over the network weights/activations.
        if os.environ.get("QUARK_DEBUG", None) is not None:
            log_dir = Path(os.environ["QUARK_DEBUG"])
            log_dir.mkdir(parents=True, exist_ok=True)

            stats: dict[str, Any] = {}
            dataloader = dataloader if not self.config_verifier.is_all_dynamic else None

            with insert_stats_hooks(model, stats, log_dir):
                collect_quantization_statistics(model, dataloader, stats, log_dir)

        # Check the scale of the quantized model.
        if os.getenv("QUARK_CHECK_SCALE") == "1":
            check_scale_stats(model, self.config)

        # Add quant_config to attribute of the quantized model, so that it can be used for export
        model.quant_config = self.config
        # Add a flag to indicate that the model is quantized
        model.quark_quantized = True

        return model

    @gpu_memory_profiled(tag=" DirectQuantizeCheckpoint")  # type: ignore[arg-type]
    def direct_quantize_checkpoint(
        self,
        pretrained_model_path: str,
        save_path: str,
        device: str | torch.device = "cuda",
    ) -> None:
        """
        Quantize model weights by processing each safetensors file independently (file-to-file mode).

        This method provides memory-efficient weight-only quantization by processing safetensors
        files one at a time, without loading the full model into GPU memory. This is particularly
        useful for quantizing very large models that exceed available GPU memory.

        The quantized shards and all configuration files (``config.json``,
        ``model.safetensors.index.json``, tokenizer files, etc.) are written to ``save_path``.

        :param str pretrained_model_path: Path to the pretrained model directory
            containing safetensors files.
        :param str save_path: Directory path to save the quantized safetensors files.
        :param str | torch.device device: Device for tensor operations (e.g., ``"cuda"``,
            ``"cuda:0"``, ``"cpu"``). Default is ``"cuda"``.

        Example:

        .. code-block:: python

            from quark.torch.quantization.config.config import QConfig, QLayerConfig, OCP_MXFP4Spec

            from quark.torch import ModelQuantizer

            weight_spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
            quant_config = QConfig(global_quant_config=QLayerConfig(weight=weight_spec))

            quantizer = ModelQuantizer(quant_config)
            quantizer.direct_quantize_checkpoint(
                pretrained_model_path="/path/to/model",
                save_path="/path/to/output",
            )
        """
        logger.info(f"File-to-file quantization with the configuration:\n{self.config}")
        quantize_model_per_safetensor(
            pretrained_model_path=pretrained_model_path,
            quant_config=self.config,
            save_path=save_path,
            device=device,
        )
        logger.info(f"File-to-file quantization completed. Output saved to {save_path}")

    def _check_model_device(self, model: nn.Module) -> None:
        # using accelerate cause, device can not be cpu or disk, temporarily
        if hasattr(model, "hf_device_map"):
            if not self.multi_device:
                for _, layer_device in model.hf_device_map.items():
                    if layer_device == "cpu" or layer_device == "disk":
                        # TODO: We should handle this for customers.
                        raise MemoryError(
                            "Out of memory. The available GPU memory is insufficient to load the entire model. You can try adding '--multi_device' "
                        )

            self._is_accelerate = True
        else:
            self._is_accelerate = False

    def _generate_complete_config_by_model(
        self,
        model: nn.Module,
        dataloader: DataLoader[torch.Tensor]
        | DataLoader[list[dict[str, torch.Tensor]]]
        | DataLoader[dict[str, torch.Tensor]]
        | DataLoader[list["BatchFeature"]]
        | None,
    ) -> None:
        """
        Generates a complete configuration based on the provided model and dataloader.
        """
        self.config = add_algorithm_config_by_model(model, dataloader, self.config)

    @staticmethod
    @torch.no_grad()
    def freeze(
        model: nn.Module | torch.fx.GraphModule, quantize: bool | None = None
    ) -> nn.Module | torch.fx.GraphModule:
        """
        Freezes the quantized model by replacing ``FakeQuantize`` modules with ``FrozenFakeQuantize`` modules.

        In order to be able to compile a quantized model through ``torch.compile``, this method needs to be applied.

        :param torch.nn.Module model: The neural network model containing quantized layers.
        :param Optional[bool] quantize: Whether to effectively quantize weights, moving away from soft weights that are quantized on the fly to e.g. `QuantLinear.weight` actually holding the fake quantized weights. This can be disabled e.g. if we would like simply to move to use ``FrozenFakeQuantize`` from a model using ``QuantLinear`` that is already holding the fake quantized weights in high-precision. Defaults to ``True`` for PyTorch eager models, and ``False`` for FX graph models.

        :return: The modified model with ``FakeQuantize`` modules replaced by ``FrozenFakeQuantize`` modules.
        :rtype: torch.nn.Module
        """
        logger.info("Freeze model start.")
        # ----replace FakeQuantize to FrozenFakeQuantize --------------
        named_modules = dict(model.named_modules(remove_duplicate=False))

        # NOTE: For GraphModules, we disable soft weight quantization as
        # ONNX export expects us to run `QuantMixin.get_quant_weight`, `QuantMixin.get_quant_bias` in order to do pattern matching and add `QuantizeLinear` nodes in the ONNX graph for the weights.
        # This may actually also be required for non-graph models that pass
        # through ONNX export, it should be clarified with Haoliang.
        if isinstance(model, torch.fx.GraphModule):
            if quantize:
                raise ValueError(
                    f"ModelQuantizer.freeze does not quantize the weights when using a model that is a torch.fx.GraphModule, but got quantize={quantize}. Please check your code or open an issue."
                )
            quantize = False
        elif quantize is None:
            quantize = True

        # Lists the FakeQuantizeBase names which are modified calling
        # `FakeQuantizeBase.to_frozen_module`.
        frozen_names = []

        quantizer_names = set()
        total_modules = len(named_modules)
        logger.info("Freezing quantizers for quantized modules...")
        for idx, (name, module) in enumerate(
            tqdm(named_modules.items(), desc="Freezing quantized modules", total=total_modules)
        ):
            if isinstance(module, QuantMixin):
                for subname, submodule in module.named_modules():
                    if isinstance(submodule, FakeQuantizeBase):
                        full_name = name + "." + subname
                        quantizer_names.add(full_name)

                        if not submodule.is_dynamic:
                            frozen_names.append(full_name)

                            if quantize:
                                if "weight_quantizer" in subname:
                                    module.weight.data = module.get_quant_weight(module.weight)
                                elif "bias_quantizer" in subname and module.bias is not None:
                                    # TODO: better understand why in some cases (e.g. in test/test_for_torch/test_fx_quant_align_hw_pow_of_2.py's `test_torch_fold_bn_after_concat_strategy`) we do have a `bias_quantizer` module attached, but the bias is None.
                                    module.bias.data = module.get_quant_bias(module.bias)

                            frozen_quantized_module = submodule.to_frozen_module(frozen_params=quantize)

                            setattr_recursive(model, full_name, frozen_quantized_module)

        # ----if model is quantized in fx.graph mode--------------
        if isinstance(model, torch.fx.GraphModule):
            # The graph may have `ScaledFakeQuantize` that are not leafs of
            # of `QuantMixin`, specifically the case for activation quantization.
            # See e.g. the test `test_pixel_shuffle_annotation`.
            for name, module in named_modules.items():
                if isinstance(module, FakeQuantizeBase) and name not in quantizer_names:
                    quantizer_names.add(name)

                    if not module.is_dynamic:
                        frozen_quantized_module = module.to_frozen_module(frozen_params=False)
                        setattr_recursive(model, name, frozen_quantized_module)
                        frozen_names.append(name)

            model = model.freeze_model()
            assert isinstance(model, torch.fx.GraphModule)
            model = post_quant_optimize(model=model, hw_constrain=True)  # TODO pass argument

        if len(frozen_names) > 0:
            frozen_names = "\n- ".join(frozen_names)  # type: ignore
            logger.debug(f"Converted to frozen quantizers: \n- {frozen_names}")

        logger.info("Freeze model end.")
        return model

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        if self.config.quant_mode is QuantizationMode.eager_mode:
            return process_model_transformation(model, self.config)
        elif self.config.quant_mode is QuantizationMode.fx_graph_mode:
            # Quantization with torch.fx does not support some quantization config and some FX graphs.
            # This raises an error if the config / model used are not supported.
            check_supported_model_and_config(model, self.config)  # type: ignore [arg-type]

            return prepare_quant_model(model, self.config).eval()  # type: ignore [arg-type]

    def _apply_advanced_quant_algo(
        self,
        model: nn.Module,
        dataloader: DataLoader[torch.Tensor]
        | DataLoader[list[dict[str, torch.Tensor]]]
        | DataLoader[dict[str, torch.Tensor]]
        | DataLoader[list["BatchFeature"]]
        | None = None,
    ) -> nn.Module:
        return apply_advanced_quant_algo(model, self.config, self._is_accelerate, dataloader)

    def _check_token_distribution(
        self,
        model: nn.Module,
        dataloader: DataLoader[torch.Tensor]
        | DataLoader[list[dict[str, torch.Tensor]]]
        | DataLoader[dict[str, torch.Tensor]]
        | DataLoader[list["BatchFeature"]],
    ) -> None:
        """
        A helper function that warns when a MoE module
        received 0 token throughout the calibration process.
        """
        assert 0.0 <= TOKEN_DISTRIBUTION_THRESHOLD <= 1.0, "threshold should be in [0.0, 1.0]"
        total_token_count = count_calibration_tokens(dataloader)
        if total_token_count == 0:
            logger.warning("No tokens found in calibration dataset. Skipping token distribution check.")
            return

        # Get the observer token count for each module
        token_counts: Counter[str] = Counter()
        for name, module in model.named_modules():
            if isinstance(module, ScaledFakeQuantize):
                if "_input_quantizer" in name:
                    if module.observer._num_observed_tokens is not None:
                        token_counts[name.replace("._input_quantizer", "")] = module.observer._num_observed_tokens

        # Track MoE experts with zero tokens for special warning
        moe_experts_zero_tokens: list[str] = []

        for module_name, token_count in token_counts.items():
            # Check if this is a MoE expert module
            is_moe_expert = "expert" in module_name.lower()

            if is_moe_expert and token_count == 0:
                moe_experts_zero_tokens.append(module_name)
            elif (token_count / float(total_token_count)) <= TOKEN_DISTRIBUTION_THRESHOLD:
                logger.warning(
                    f"The module: {module_name} "
                    f"received {token_count} tokens less than {TOKEN_DISTRIBUTION_THRESHOLD * 100:.1f}% "
                    f"of all {total_token_count} calibration tokens."
                )

        # Emit warning for MoE experts with incomplete coverage
        if moe_experts_zero_tokens:
            experts = moe_experts_zero_tokens[:10]
            more = f"  ... and {len(moe_experts_zero_tokens) - 10} more\n" if len(moe_experts_zero_tokens) > 10 else ""
            logger.warning(
                f"\nMoE EXPERT COVERAGE WARNING: {len(moe_experts_zero_tokens)} expert(s) received 0 tokens.\n"
                f"Affected: {', '.join(experts)}\n{more}"
                f"Impact: Unactivated experts may cause empty weights, incomplete artifacts, or vLLM load failures.\n"
                f"Actions: 1) Increase --calib_size  2) Use diverse calibration data"
            )

        # Output the tokens distribution if enabled.
        if QUARK_TOKENS_DISTRIBUTION_PATH:
            token_stats = {
                "total_token_count": total_token_count,
                "token_counts": dict(token_counts),
                "threshold": TOKEN_DISTRIBUTION_THRESHOLD,
            }

            timestamp = int(time.time())
            filename = f"tokens_distribution_{timestamp}.json"
            try:
                with open(os.path.join(QUARK_TOKENS_DISTRIBUTION_PATH, filename), "w", encoding="utf-8") as f:
                    json.dump(token_stats, f, indent=2, ensure_ascii=False)
                logger.info(f"Token distribution statistics saved to {filename}")
            except Exception as e:
                logger.error(f"Failed to save token distribution statistics: {e}")

    # when using multi_device, you must add it here or offload will fail.
    # The gpu memory used for gradients cannot be cleaned up by torch.cuda.empty_cache()
    @torch.no_grad()
    def _do_calibration(
        self,
        model: nn.Module,
        dataloader: DataLoader[torch.Tensor]
        | DataLoader[list[dict[str, torch.Tensor]]]
        | DataLoader[dict[str, torch.Tensor]]
        | DataLoader[list["BatchFeature"]]
        | None = None,
    ) -> nn.Module:
        # just calib, turn off quantize
        if self.config_verifier.is_all_dynamic:  # TODO: to be deperated
            logger.info("Dynamic quantization, no calibration.")
        elif self.config_verifier.is_weight_only or (
            self.config_verifier.is_act_dynamic and not self.config_verifier.is_act_contain_scale_per_tensor
        ):
            logger.info("Weight calibration start.")
            for module in model.modules():
                if isinstance(module, ScaledFakeQuantize):
                    module.enable_observer()
                    module.disable_fake_quant()

            # Simply run through the observers to set min_val, max_val, scale and zero_point buffers for the weight and bias.
            self._calibrate_all_params(model)
            logger.info("Weight calibration end.")
        else:
            logger.info("Calibration start.")
            for module in model.modules():
                if isinstance(module, ScaledFakeQuantize):
                    module.enable_observer()
                    module.disable_fake_quant()

            assert dataloader is not None
            # Calibrate all weights and biases by iterating through named_modules.
            # Disable observers for weight and bias quantizers after calibration
            if self.config.quant_mode is QuantizationMode.eager_mode:
                # as for fx graph mode: QuantizedConvBatchNorm2d, must perform calib during forward
                logger.info("Calibrating all weights and biases...")
                self._calibrate_all_params(model)
                for module in model.modules():
                    if isinstance(module, QuantMixin):
                        for quantizer in [module._weight_quantizer, module._bias_quantizer]:
                            if quantizer is None:
                                continue
                            quantizers = [quantizer] if isinstance(quantizer, ScaledFakeQuantize) else quantizer
                            for q in quantizers:
                                if isinstance(q, ScaledFakeQuantize):
                                    q.disable_observer()
                logger.info("Running forward pass calibration for activations...")
            elif self.config.quant_mode is QuantizationMode.fx_graph_mode:
                logger.info("Calibrating for fx graph model...")

            # Forward pass calibration (mainly for activations)
            with torch.no_grad():
                for data in tqdm(dataloader):
                    if isinstance(data, dict):  # pragma: no cover
                        model(**data)
                    elif is_transformers_available() and isinstance(data, BatchFeature):  # pragma: no cover
                        _ = model(**data)
                    else:
                        model(data)

            if QUARK_COUNT_OBSERVED_SAMPLES:
                self._check_token_distribution(model, dataloader)

            clear_memory()
            logger.info("Calibration end.")
        logger.info("Model quantization has been completed.")

        # step5[optional]: do evaluation, turn on quantize

        # Some algorithms handle the quantization of weights, bypassing the attached quantizers.
        override_weight_quantizers = False
        if self.config.algo_config is not None:
            for algo_config in self.config.algo_config:
                if algo_config.name.lower() in ["gptq", "gptaq", "qronos"] and not algo_config.static_groups:  # type: ignore
                    override_weight_quantizers = True
                    break

        if override_weight_quantizers:
            logger.warning(
                f"The algorithm configuration {self.config.algo_config} with dynamic groups setting (static_groups=false) do not support FakeQuantize for export. "
                f"Turn off FakeQuantize for weight while keeping FakeQuantize enabled for activation to run evaluations."
            )
            named_modules = dict(model.named_modules(remove_duplicate=False))
            for _, module in tqdm(named_modules.items()):
                if isinstance(module, QuantMixin):
                    if module._weight_quantizer is not None:
                        enable_or_disable_quantizer(module._weight_quantizer, enable=False)

                    if module._input_quantizer is not None:
                        enable_or_disable_quantizer(module._input_quantizer, enable=True)

                    if module._output_quantizer is not None:
                        enable_or_disable_quantizer(module._output_quantizer, enable=True)
        else:
            for name, module in model.named_modules():
                if isinstance(module, ScaledFakeQuantize):
                    if module.is_dynamic and not (
                        module.is_scale_quant and module.qscheme == QSchemeType.per_tensor
                    ):  # For dynamic quantization, observer should be enable and update qparam every time.
                        module.enable_observer()
                        module.enable_fake_quant()
                    else:
                        module.disable_observer()
                        module.enable_fake_quant()
                elif isinstance(module, NonScaledFakeQuantize):
                    module.enable_fake_quant()
        return model

    def _calibrate_all_params(self, model: nn.Module) -> None:
        named_modules = dict(model.named_modules(remove_duplicate=False))
        for name, module in tqdm(named_modules.items(), desc="Calibrating all params"):
            if isinstance(module, QuantMixin):
                # Calibrate weight
                if module._weight_quantizer is not None and isinstance(
                    module._weight_quantizer, (ScaledFakeQuantize, SequentialQuantize)
                ):
                    weight_quantizers: list[ScaledFakeQuantize] | SequentialQuantize = (
                        [module._weight_quantizer]
                        if isinstance(module._weight_quantizer, ScaledFakeQuantize)
                        else module._weight_quantizer
                    )

                    # Check if weight has already been calibrated (scale != 1 indicates calibrated)
                    is_not_calibrated = all(hasattr(quantizer, "scale") for quantizer in weight_quantizers) and all(
                        quantizer.scale.numel() == 1 and quantizer.scale.item() == 1 for quantizer in weight_quantizers
                    )

                    if is_not_calibrated and module.weight is not None:
                        if module.weight.device == torch.device("meta"):
                            weight = module._hf_hook.weights_map["weight"].data
                            _ = module.get_quant_weight(weight.to(module._hf_hook.execution_device))
                            del weight
                        else:
                            _ = module.get_quant_weight(module.weight)

                # Calibrate bias
                if module._bias_quantizer is not None and isinstance(
                    module._bias_quantizer, (ScaledFakeQuantize, SequentialQuantize)
                ):
                    bias_quantizers: list[ScaledFakeQuantize] | SequentialQuantize = (
                        [module._bias_quantizer]
                        if isinstance(module._bias_quantizer, ScaledFakeQuantize)
                        else module._bias_quantizer
                    )

                    is_not_calibrated = all(hasattr(quantizer, "scale") for quantizer in bias_quantizers) and all(
                        quantizer.scale.numel() == 1 and quantizer.scale.item() == 1 for quantizer in bias_quantizers
                    )

                    if is_not_calibrated and module.bias is not None:
                        if module.bias.device == torch.device("meta"):
                            bias = module._hf_hook.weights_map["bias"].data
                            _ = module.get_quant_bias(bias.to(module._hf_hook.execution_device))
                            del bias
                        else:
                            _ = module.get_quant_bias(module.bias)

                torch.cuda.empty_cache()

        clear_memory()

    def _do_post_calib_optimazation(self, model: nn.Module) -> nn.Module:
        """
        In some case:
            1. After calibration: get weight, activation and bias scale
            2. Some hw constrain need let: bias_scale = weight_scale * act_scale
        After calibration, we need to do some optimization, and then perform QAT/export.
        """
        if self.config.quant_mode is QuantizationMode.eager_mode:
            # remain this API TODO
            assert isinstance(model, nn.Module)
            return model
        elif self.config.quant_mode is QuantizationMode.fx_graph_mode:
            """
            In calibration: observer will record tensor's distribution. Scale and ZP will be calculated.
            In some hardware constrain case.
                e.g. b_scale = w_scale * a_scale  (we need to modify bias_scale after calibration)
            """
            assert isinstance(model, torch.fx.GraphModule)
            model = post_calib_optimize(model)
        return model  # type: ignore[no-any-return]

    def init_config(self) -> None:
        logger.info("Configuration checking start.")
        # TODO: Verify quant algo
        self.config_verifier.verify_config()
        if self.config_verifier.is_weight_only:
            config_parsing_result = "weight only quantization"
        else:
            if self.config_verifier.is_act_dynamic:
                config_parsing_result = "weight quantization and activation dynamic quantization"
            else:
                config_parsing_result = "weight quantization and activation static quantization"
        logger.info(f"Configuration checking end. The configuration is effective. This is {config_parsing_result}.")


def get_name_and_info(model_info: dict[str, Any], parent_key: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    for key, value in model_info.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            if value.get("type", None) is not None and value.get("weight", None) is not None:
                yield new_key, value
            else:
                yield from get_name_and_info(value, new_key)
        else:
            continue


# TODO: This function is only used in load_params, add support for SequentialQuantize later
def from_float_and_dict(
    float_module: nn.Module,
    quant_info: dict[str, Any],
    param_dict: dict[str, torch.Tensor],
    layer_name: str,
    compressed: bool = False,
    reorder: bool = True,
) -> nn.Module:
    input_tensors = None
    quant_params: dict[str, torch.Tensor | None] = {}
    if quant_info.get("input_quant") is not None:
        input_tensors = QTensorConfig.from_dict(quant_info["input_quant"])

        if input_tensors is not None and not input_tensors.is_dynamic:
            quant_params["input_scale"] = param_dict[layer_name + ".input_scale"]  # pragma: no cover
            quant_params["input_zero_point"] = param_dict[layer_name + ".input_zero_point"]  # pragma: no cover

    output_tensors = None
    if quant_info.get("output_quant") is not None:
        output_tensors = QTensorConfig.from_dict(quant_info["output_quant"])

        if output_tensors is not None and not output_tensors.is_dynamic:
            quant_params["output_scale"] = param_dict[layer_name + ".output_scale"]
            quant_params["output_zero_point"] = param_dict[layer_name + ".output_zero_point"]

    weight_qspec: QTensorConfig | None = None
    weight_key = quant_info.get("weight")
    if weight_key is None:
        raise KeyError("Missing 'weight' in quant_info")
    weight_tensor = param_dict[weight_key]
    if quant_info.get("weight_quant") is not None:
        weight_qspec = QTensorConfig.from_dict(quant_info["weight_quant"])
        weight_scale = param_dict[layer_name + ".weight_scale"]
        weight_zero_point = param_dict[layer_name + ".weight_zero_point"]
        if compressed:
            assert isinstance(weight_qspec, QTensorConfig), "weight_qspec must be QTensorConfig instance"
            assert isinstance(weight_qspec.qscheme, QSchemeType), "weight_qspec.qscheme must be QSchemeType instance"
            assert isinstance(weight_qspec.dtype, Dtype), "weight_qspec.dtype must be Dtype instance"
            pack_method = create_pack_method(qscheme=weight_qspec.qscheme.value, dtype=weight_qspec.dtype.value)
            weight_tensor = pack_method.unpack(
                weight_tensor,
                reorder,
                **({"origin_packed_axis_size": weight_scale.shape[-1]} if weight_scale.shape != torch.Size([]) else {}),
            )

            weight_tensor = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                weight_qspec.dtype.value,
                weight_tensor,
                weight_scale,
                weight_zero_point,
                weight_qspec.ch_axis,
                weight_qspec.group_size,
                weight_qspec.qscheme.value,
            )

        quant_params["weight_scale"] = weight_scale
        quant_params["weight_zero_point"] = weight_zero_point

    module_config = QLayerConfig(input_tensors=input_tensors, output_tensors=output_tensors, weight=weight_qspec)

    bias_tensor = None
    bias_key = quant_info.get("bias")
    bias_tensor = param_dict[bias_key] if bias_key is not None else None

    quant_module: nn.Module
    if quant_info["type"] in QUARK_QUANT_OPS:
        quant_module = QUARK_QUANT_OPS[quant_info["type"]].from_float(
            float_module,
            module_config,
            reload=True,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
        )
    else:
        raise ValueError(f"The type {quant_info['type']} dose not support in Quark now!")
    quant_module.load_quant_params(quant_params)
    return quant_module


# TODO: add support for SequentialQuantize later
# TODO: better `reorder` doc
# TODO: Consider deprecating this - is anybody really using it?
@log_errors
def load_params(
    model: nn.Module | None = None,
    json_path: str = "",
    safetensors_path: str = "",
    pth_path: str = "",
    quant_mode: QuantizationMode = QuantizationMode.eager_mode,
    compressed: bool = False,
    reorder: bool = True,
) -> nn.Module:
    """
    Instantiates a quantized model from saved model files, which is generated from the :py:func:`quark.torch.export.api.save_params` function.

    :param torch.nn.Module model: The original Pytorch model.
    :param str json_path: The path of the saved json file. Only available for eager mode quantization.
    :param str safetensors_path: The path of the saved safetensors file. Only available for eager mode quantization.
    :param str pth_path: The path of the saved ``.pth`` file. Only available for ``fx_graph`` mode quantization.
    :param QuantizationMode quant_mode: The quantization mode. The choice includes ``"QuantizationMode.eager_mode"`` and ``"QuantizationMode.fx_graph_mode"``. Default is ``"QuantizationMode.eager_mode"``.
    :param bool compressed: Whether the quantized model to load is stored using its compressed data type, or in a "fake quantized" format (QDQ).
    :param bool reorder: Reorder.

    :return: The reloaded quantized version of the input model.
    :rtype: torch.nn.Module

    Examples:

    .. code-block:: python

        # eager mode:
        from quark.torch import load_params
        model = load_params(model, json_path=json_path, safetensors_path=safetensors_path)

    .. code-block:: python

        # fx_graph mode:
        from quark.torch.quantization.api import load_params
        model = load_params(pth_path=model_file_path, quant_mode=QuantizationMode.fx_graph_mode)

    Note:
        This function does not support dynamic quantization for now.
    """

    if quant_mode is QuantizationMode.eager_mode:
        if not is_safetensors_available():
            raise ImportError(
                "The function `load_params` with `quant_mode=QuantizationMode.eager_mode` requires the package `safetensors` to be installed, but it was not found. Please install `safetensors`."
            )

        if model is None:
            raise ValueError("Model should not be none if loading eager_mode quantized model")
        if json_path == "" or safetensors_path == "":
            raise ValueError("Json_path and safetensors_path should not be empty if loading eager_mode quantized model")
        # load model structure and parameters
        with open(json_path) as file:
            model_dict = json.load(file)
        params_dict = load_file(safetensors_path)

        # verify exported model and float model have the same configuration
        model_config = model_dict["config"]
        if model_config:
            float_model_config: dict[str, Any] = {}
            if hasattr(model.config, "to_diff_dict"):
                float_model_config = model.config.to_diff_dict()
            elif hasattr(model.config, "items"):
                float_model_config = dict(model.config.items())

            if not deep_compare(model_config, float_model_config):
                raise RuntimeError("Exported model and float model are not the same model!")
        # assert ((json.dumps(model_config) == json.dumps(float_model_config)),
        #         "Exported model and float model are not the same model!")

        logger.info("In-place OPs replacement start.")
        for name, module_info in get_name_and_info(model_dict["structure"]):
            float_module = getattr_recursive(model, name)
            if module_info["type"] in QUARK_QUANT_OPS:
                module = from_float_and_dict(
                    float_module, module_info, params_dict, layer_name=name, compressed=compressed, reorder=reorder
                )
                setattr_recursive(model, name, module)
            else:
                device = float_module.weight.device
                weight_key = module_info.get("weight")
                if weight_key is not None:
                    float_module.weight.data = params_dict[weight_key].to(device)

                bias_key = module_info.get("bias")
                if bias_key is not None:
                    float_module.bias.data = params_dict[bias_key].to(device)

        # Convert to relevant quantizers to `FrozenScaledFakeQuantize`. This essentially removes the observers from weight quantizers. We disable running quantization, as loaded weights are already quantized.
        model = ModelQuantizer.freeze(model, quantize=False)

        logger.info("In-place OPs replacement end.")
    elif quant_mode is QuantizationMode.fx_graph_mode:
        if pth_path == "":
            raise ValueError("Pth_path should not be empty if loading eager_mode quantized model")
        loaded_quantized_ep = torch.export.load(pth_path)
        model = loaded_quantized_ep.module()

    return model
