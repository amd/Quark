from __future__ import annotations

import copy
import fnmatch
from types import TracebackType
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.blockwise_tuning.blockwise_utils import block_forward
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import get_device, get_named_quant_linears, move_to_device
from quark.torch.algorithm.utils.prepare import init_blockwise_algo, init_device_map, reset_model_kv_cache
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.quantization.config.type import QSchemeType
from quark.torch.quantization.observer.observer import PerChannelMinMaxObserver
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.utils import QUARK_DISABLE_CUDA_GRAPH

logger = ScreenLogger(__name__)

CPU = torch.device("cpu")
META = torch.device("meta")
CUDA = torch.device("cuda")


class BaseHessianAlgorithm:
    """Base class for Hessian-based quantization algorithms."""

    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.device = self.layer.weight.device
        self.columns = layer.weight.data.shape[1]
        if self.device == META:
            # should be execute_device, When cuda0 is very small, it could be any other value.
            self.device = self.layer._hf_hook.execution_device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        # Transformers might not be in the user environment, hence the class name check instead.
        if "transformers.pytorch_utils.Conv1D" in str(self.layer.__class__):
            W = W.t()

        self.H: torch.Tensor | None = torch.zeros((self.columns, self.columns), device=self.device, dtype=torch.float32)
        # Qronos
        # self.G: torch.Tensor | None = torch.zeros((self.columns, self.columns), device=self.device, dtype=torch.float32)
        self.nsamples = 0
        self.q_input: torch.Tensor | None = None

        self._setup_quantizer()

    def _setup_quantizer(self) -> None:
        """Setup quantizer based on quantization scheme."""
        self.original_qspec = self.layer._weight_quantizer.observer.qspec

        # for per group minmaxobserver: group_size > 1 and group_size == -1
        if self.original_qspec.qscheme == QSchemeType.per_group:
            from quark.torch.quantization.config.config import QTensorConfig

            self.adjusted_qspec = QTensorConfig(
                dtype=self.original_qspec.dtype,
                qscheme=QSchemeType.per_channel,
                observer_cls=PerChannelMinMaxObserver,
                symmetric=self.original_qspec.symmetric,
                scale_type=self.original_qspec.scale_type,
                round_method=self.original_qspec.round_method,  # useless for perchannel
                ch_axis=0,
                is_dynamic=self.original_qspec.is_dynamic,
                mx_element_dtype=self.original_qspec.mx_element_dtype,
                scale_format=self.original_qspec.scale_format,
                scale_calculation_mode=self.original_qspec.scale_calculation_mode,
            )
            # Due to the difference between cuda and cpu hardware architecture and calculation precision,
            # it will lead to the difference in the last few bits of the value obtained from the calculation,
            # this difference will be amplified by the calculation method of GPTQ, you should keep the consistency of the device.
            self.quantizer = FakeQuantizeBase.get_fake_quantize(self.adjusted_qspec, self.device)
        # pertensor & perchannel
        else:
            self.quantizer = self.layer._weight_quantizer

    def _preprocess_input(self, inp: torch.Tensor) -> torch.Tensor:
        """Preprocess input tensor for Linear layers."""
        inp = inp.float()
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if isinstance(self.layer, nn.Linear) and len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            assert not isinstance(self.layer.padding, str)
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        return inp

    def add_batch_quantized(self, quant_inp: torch.Tensor, out: torch.Tensor, name: str) -> None:
        raise NotImplementedError("Implement in subclass")

    def add_batch_nonquantized(self, inp: torch.Tensor, out: torch.Tensor, name: str) -> None:
        raise NotImplementedError("Implement in subclass")

    def _get_weight(self) -> torch.Tensor:
        """Get weight tensor from layer."""
        if get_device(self.layer) == META:
            return self.layer._hf_hook.weights_map["weight"].data.to(self.layer._hf_hook.execution_device)
        return self.layer.weight.data.clone()

    def _set_weight(self, Q: torch.Tensor, orig_dtype: torch.dtype) -> None:
        """Set quantized weight back to layer."""
        # Directly replace weight in dict with qweight
        if get_device(self.layer) == META:
            self.layer._hf_hook.weights_map["weight"].data = Q.reshape(self.layer.weight.shape).to(orig_dtype).to("cpu")
        else:
            self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)

    def _setup_quantizers(
        self, W: torch.Tensor, group_size: int, static_groups: bool
    ) -> tuple[list[FakeQuantizeBase], list[torch.Tensor], list[torch.Tensor]]:
        """Setup quantizers for different quantization schemes."""
        quantizers = []
        scale = []
        zero = []

        per_group = group_size is not None and group_size > 0

        if per_group:
            if static_groups:
                for i in range(0, self.columns, group_size):
                    quantizer = copy.deepcopy(self.quantizer)
                    quantizer.observe(W[:, i : (i + group_size)])
                    quantizers.append(quantizer)
                    scale.append(quantizer.scale)
                    zero.append(quantizer.zero_point)
            else:
                quantizers.append(self.quantizer)
        else:
            self.quantizer.observe(W)
            quantizers.append(self.quantizer)
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero_point)

        return quantizers, scale, zero

    def _apply_activation_order(
        self, W: torch.Tensor, H: torch.Tensor, actorder: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Apply activation ordering if enabled."""
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
            return W, H, perm, invperm
        return W, H, None, None

    def _update_layer_quantizer(self, scale: list[torch.Tensor], zero: list[torch.Tensor], group_size: int) -> None:
        """Update layer's quantizer with computed scale and zero point."""
        # scale and zero of perchannel, pertensor have been added to buffer
        # but per_group (any groupsize) need be added
        if group_size is not None:
            if group_size > 0:  # if not static, scale and zero_point need to be reordered when using quantization
                self.layer._weight_quantizer.scale = torch.cat([s.view(-1, 1) for s in scale], dim=1)
                self.layer._weight_quantizer.zero_point = torch.cat([z.view(-1, 1) for z in zero], dim=1)
            else:  # when group size == -1, static_group does not work
                self.layer._weight_quantizer.scale = self.quantizer.scale
                self.layer._weight_quantizer.zero_point = self.quantizer.zero_point

    def free(self) -> None:
        self.H = None
        clear_memory()


class BaseHessianProcessor(BaseAlgoProcessor):
    """Base processor for Hessian-based algorithms."""

    def __init__(self, model: nn.Module, quant_algo_config: Any, data_loader: DataLoader[torch.Tensor]) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.flags(enabled=True, allow_tf32=False)

        self.model = model
        self.block_size = quant_algo_config.block_size
        self.act_order = quant_algo_config.desc_act
        self.static_groups = quant_algo_config.static_groups
        self.inside_layer_modules = quant_algo_config.inside_layer_modules
        self.model_decoder_layers = quant_algo_config.model_decoder_layers
        self.data_loader = data_loader
        self.device_map = init_device_map(self.model)
        self.modules, self.module_kwargs, self.inps = init_blockwise_algo(
            self.model, self.model_decoder_layers, self.data_loader
        )

        # In case we use CUDA Graph, this dictionary holds all the recorded graphs and all the static inputs to the graphs. The graphs have no outputs and modify the input in place.
        self.graphs: dict[str, tuple[list[torch.cuda.CUDAGraph], dict[str, Any]]] = {}
        self._use_cuda = next(iter(model.parameters())).device.type == "cuda"
        self.use_cuda_graphs = not QUARK_DISABLE_CUDA_GRAPH and self.static_groups and self._use_cuda
        self._use_original_weight = False

    def _get_algorithm_instance(self, layer: nn.Module) -> Any:
        """Override this to return specific algorithm instance."""
        raise NotImplementedError

    def _quantize_layer(self, algo_instance: Any, layer: nn.Module, group_size: int) -> None:
        """Override this to call specific quantization method."""
        raise NotImplementedError

    def apply(self) -> None:
        cache_examples_on_gpu = True
        num_batches = len(self.inps)
        layer_inputs = [inp.clone() for inp in self.inps]
        orig_layer_inputs: list[torch.Tensor] | None = None
        if self._use_original_weight:
            orig_layer_inputs = [inp.clone() for inp in self.inps]
        layer_outputs: list[torch.Tensor] = []
        orig_layer_outputs: list[torch.Tensor] = []
        forward_pass_use_cache = reset_model_kv_cache(self.model, use_cache=False)

        for i in tqdm(range(len(self.modules)), desc=f"Applying {self.__class__.__name__}"):
            logger.info(f"Start quantizing layer {i + 1}/{len(self.modules)}")
            if self._use_original_weight:
                self.register_original_weights(self.modules[i])
            layer = self.modules[i]

            force_layer_back_to_cpu = get_device(layer) == CPU
            if force_layer_back_to_cpu:
                move_to_device(layer, self.device_map[f"{self.model_decoder_layers}.{i}"])

            current_layer_device = get_device(layer) if get_device(layer) != META else layer._hf_hook.execution_device

            all_inner_layer_modules = get_named_quant_linears(layer)
            assert self.inside_layer_modules is not None

            for layer_name in self.inside_layer_modules:
                matched_names = fnmatch.filter(all_inner_layer_modules.keys(), "*" + layer_name)
                grouped_inner_layers = {
                    inner_layer: all_inner_layer_modules[inner_layer]
                    for inner_layer in matched_names
                    if getattr(all_inner_layer_modules[inner_layer], "_weight_quantizer", None) is not None
                }

                algo_instances = {
                    name: self._get_algorithm_instance(grouped_inner_layers[name]) for name in grouped_inner_layers
                }

                # for qronos it's collecting both quantized and non-quantized
                # for gptq it's only collecting quantized
                self._collect_statistics(
                    layer,
                    grouped_inner_layers,
                    algo_instances,
                    layer_inputs,
                    orig_layer_inputs,
                    num_batches,
                    current_layer_device,
                )

                for name in grouped_inner_layers:
                    logger.info(f"Quantizing {name} in layer {i + 1}/{len(self.modules)}...")
                    self._quantize_layer(
                        algo_instances[name],
                        grouped_inner_layers[name],
                        grouped_inner_layers[name]._weight_quantizer.group_size,
                    )
                    algo_instances[name].free()

            layer_outputs = block_forward(
                layer,
                self.module_kwargs,
                num_batches,
                current_layer_device,
                layer_inputs,
                layer_outputs,
                cache_examples_on_gpu=cache_examples_on_gpu,
            )

            if self._use_original_weight:
                with RestoreOriginalWeights(layer):
                    orig_layer_outputs = block_forward(
                        layer,
                        self.module_kwargs,
                        num_batches,
                        current_layer_device,
                        orig_layer_inputs,
                        orig_layer_outputs,
                        cache_examples_on_gpu=cache_examples_on_gpu,
                    )

            if get_device(layer) != META:
                # if meta, scale and zero point are in execution_device, and weight is in meta, can't change.
                layer = move_to_device(layer, CPU if force_layer_back_to_cpu else current_layer_device)

            del layer, algo_instances, layer_inputs
            self.delete_original_weight_buffer(self.modules[i])
            clear_memory()

            layer_inputs = layer_outputs  # noqa: F841
            if self._use_original_weight:
                orig_layer_inputs = orig_layer_outputs
            orig_layer_outputs, layer_outputs = [], []

        reset_model_kv_cache(self.model, use_cache=forward_pass_use_cache)

    def _collect_statistics(
        self,
        layer: nn.Module,
        grouped_inner_layers: dict[str, nn.Module],
        algo_instances: dict[str, Any],
        layer_inputs: list[torch.Tensor],
        orig_layer_inputs: list[torch.Tensor],
        num_batches: int,
        current_layer_device: torch.device,
    ) -> None:
        """Collect Hessian statistics. Override for algorithm-specific collection."""
        raise NotImplementedError

    @staticmethod
    def register_original_weights(layer: nn.Module) -> None:
        for submodule in layer.modules():
            if hasattr(submodule, "weight"):
                submodule.register_buffer("weight_orig", submodule.weight.detach().clone())

    @staticmethod
    def delete_original_weight_buffer(layer: nn.Module) -> None:
        for submodule in layer.modules():
            if hasattr(submodule, "weight_orig"):
                delattr(submodule, "weight_orig")


class RestoreOriginalWeights:
    """Context manager to temporarily restore original weights."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.quantized_weight_states: list[dict[str, Any]] = []

    def __enter__(self) -> RestoreOriginalWeights:
        for submodule in self.module.modules():
            if hasattr(submodule, "weight") and hasattr(submodule, "weight_orig"):
                self.quantized_weight_states.append({"module": submodule, "current_weight": submodule.weight.data})
                submodule.weight.data = submodule.weight_orig.data
        return self

    def _swap_to_unquantized_weights(self, module: nn.Module) -> None:
        quantized_weight_state = {"module": module, "current_weight": module.weight.data}
        self.quantized_weight_states.append(quantized_weight_state)
        module.weight.data = module.weight_orig.data

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: TracebackType | None
    ) -> None:
        for state in self.quantized_weight_states:
            state["module"].weight.data = state["current_weight"]
        self.quantized_weight_states.clear()
