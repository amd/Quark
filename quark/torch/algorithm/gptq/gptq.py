#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023 潘其威(William)
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import copy
import math
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import GPTQConfig
import fnmatch

from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.blockwise_tuning.blockwise_utils import block_forward
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import get_device, get_named_quant_linears, move_to_device
from quark.torch.algorithm.utils.prepare import init_blockwise_algo, init_device_map, reset_model_kv_cache
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.quantization.config.type import QSchemeType
from quark.torch.quantization.debug import QUARK_GRAPH_DEBUG
from quark.torch.quantization.observer.observer import PerChannelMinMaxObserver
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

logger = ScreenLogger(__name__)

__all__ = ["GptqProcessor"]

CPU = torch.device("cpu")
CUDA = torch.device("cuda")
META = torch.device("meta")

DEFAULT_COLUMNS_PER_GRAPH = 2048
FALLBACK_COLUMNS_PER_GRAPH = [1024, 512, 256, 128, 64]


def record_graphs(kwargs: dict[str, Any], columns: int, device: torch.device) -> list[torch.cuda.CUDAGraph]:
    subgraphs = []

    for name, inp in kwargs.items():
        if isinstance(inp, torch.Tensor):
            if name != "Hinv":
                assert inp.is_contiguous()

            # As the `kwargs[name]` graph input pointer will be overriden with `copy_` operations, it needs not to be a pointer to the first tensors when the graph is recorded, but an other standalone buffer in memory.
            kwargs[name] = inp.clone()

    # Unfortunately, the inner GPTQ loop can be quite large and there is a bug in ROCm
    # when recording large HIP Graphs, whereas captures succeeds on Nvidia devices.
    # Reference: https://github.com/pytorch/pytorch/issues/155720
    # As a workaround, we split the outer GPTQ loop (iterating `block_size` by `block_size` over all `columns`) in several HIP Graphs.
    # Unfortunately, we can not record a single graph and use a different `columns_start_idx` integer input, as dynamically indexing into different input memory
    # locations is forbidden with CUDA/HIP Graphs.
    kwargs["columns_start_idx"] = 0

    for _ in range(columns // kwargs["columns_per_graph"]):
        g = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]

        # Warmup before capture
        s = torch.cuda.Stream(device)  # type: ignore[no-untyped-call]
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(3):
                fasterquant_inner_graph(**kwargs)
        torch.cuda.current_stream(device).wait_stream(s)

        # Captures the graph
        # To allow capture, automatically sets a side stream as the current stream in the context
        with torch.cuda.graph(g, stream=s):
            fasterquant_inner_graph(**kwargs)

        subgraphs.append(g)

        kwargs["columns_start_idx"] += kwargs["columns_per_graph"]

    return subgraphs


def replay_fasterquant_inner_graphs(subgraphs: list[torch.cuda.CUDAGraph], inputs: dict[str, Any]) -> torch.Tensor:
    inputs["columns_start_idx"] = 0

    for subgraph in subgraphs:
        logger.debug(f"replaying subgraph with columns_per_graph={inputs['columns_per_graph']}")

        subgraph.replay()  # type: ignore[no-untyped-call]

        inputs["columns_start_idx"] += inputs["columns_per_graph"]

    # We clone here as the same `inputs["Q"]` memory will be used for all layers, and updated.
    return inputs["Q"].clone()  # type: ignore[no-any-return]


def fasterquant_inner_graph(
    Hinv: torch.Tensor,
    W: torch.Tensor,
    perm: torch.Tensor | None,
    quantizer: ScaledFakeQuantize,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    actorder: bool,
    group_size: int,
    blocksize: int,
    columns_start_idx: int,
    Q: torch.Tensor,
    columns_per_graph: int,
) -> None:
    Q1 = torch.zeros_like(W[:, :blocksize])
    Err1 = torch.zeros_like(W[:, :blocksize])

    for i1 in range(columns_start_idx, columns_start_idx + columns_per_graph, blocksize):
        i2 = i1 + blocksize

        count = i2 - i1

        W1 = W[:, i1:i2]

        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if group_size is not None and group_size > 0:
                idx = i1 + i
                if actorder:
                    # Refer to https://github.com/pytorch/pytorch/issues/155682.
                    # CUDA Graph capture does not accept dynamic indexing, but accepts
                    # dynamic slicing.
                    idx = perm[idx : idx + 1]  # type: ignore

                scale = scales[:, idx // group_size]
                zero_point = zero_points[:, idx // group_size]
                if group_size is not None and group_size > 0 and actorder:
                    # Again, refer to https://github.com/pytorch/pytorch/issues/155682
                    # for this indexing. This is rather hacky.
                    scale = scale[:, 0]
                    zero_point = zero_point[:, 0]
            else:
                scale = scales
                zero_point = zero_points

            q = quantizer.fake_quantize_with_qparams(w.unsqueeze(1), scale=scale, zero_point=zero_point)
            q = q.squeeze(dim=1)

            Q1[:, i] = q

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))

            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])


def fasterquant_inner(
    graphs: dict[str, tuple[list[torch.cuda.CUDAGraph], dict[str, Any]]],
    Hinv: torch.Tensor,
    W: torch.Tensor,
    perm: torch.Tensor | None,
    quantizers: list[ScaledFakeQuantize],
    scales: torch.Tensor | None,
    zero_points: torch.Tensor | None,
    actorder: bool,
    group_size: int,
    columns: int,
    blocksize: int,
    static_groups: bool,
    use_cuda_graphs: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None, list[torch.Tensor] | None]:
    device = W.device

    inputs = {
        "Hinv": Hinv,
        "W": W,
        "Q": torch.zeros_like(W),
        "perm": perm,
        "actorder": actorder,
        "blocksize": blocksize,
        "group_size": group_size,
    }

    if not use_cuda_graphs:
        inputs["columns"] = columns

        Q, Losses, scale, zero_point = fasterquant_inner_eager(
            **inputs,  # type: ignore[arg-type]
            quantizers=quantizers,
            static_groups=static_groups,
        )
    else:
        perm_marker = str(perm.shape) if actorder else "None"  # type: ignore[union-attr]
        # In case we use multiple GPUs, we need to record one graph per GPU as the input
        # need to be static, and we better avoid device <-> device communication.
        graph_spec = (
            str(Hinv.device)
            + str(Hinv.shape)
            + str(Hinv.dtype)
            + str(W.shape)
            + str(W.dtype)
            + perm_marker
            + str(len(quantizers))
            + str(actorder)
            + str(group_size)
            + str(columns)
            + str(blocksize)
        )

        inputs["scales"] = scales
        inputs["zero_points"] = zero_points
        inputs["quantizer"] = quantizers[0]  # type: ignore[assignment]
        inputs["columns_start_idx"] = 0

        for tensor in [Hinv, scales, zero_points]:
            if tensor is not None:
                assert tensor.device == device
        if actorder:
            assert perm.device == device  # type: ignore[union-attr]

        if columns % DEFAULT_COLUMNS_PER_GRAPH == 0:
            inputs["columns_per_graph"] = DEFAULT_COLUMNS_PER_GRAPH
        else:
            if columns < DEFAULT_COLUMNS_PER_GRAPH:
                inputs["columns_per_graph"] = columns
            else:
                for possible_columns_per_graph in FALLBACK_COLUMNS_PER_GRAPH:
                    if columns % possible_columns_per_graph == 0:
                        inputs["columns_per_graph"] = possible_columns_per_graph
                        break
                else:
                    raise ValueError(
                        f"Unsupported columns number in GPTQ using CUDA Graph: {columns}. Only multiples of 64 are supported. Consider using the environment variable `QUARK_GRAPH_DEBUG=1` to avoid using CUDA graphs in GPTQ algorithm - this may result in 3-4x slower quantization."
                    )

        if inputs["columns_per_graph"] % inputs["blocksize"] != 0:  # type: ignore
            raise ValueError(
                f"block_size={inputs['blocksize']} is not supported with CUDA Graph using columns_per_graph={inputs['columns_per_graph']}."
            )

        if graph_spec not in graphs:
            # See the comment below about `record_graph` for the reason to clone the inputs here.
            ref_inputs = {name: inp.clone() if isinstance(inp, torch.Tensor) else inp for name, inp in inputs.items()}

            logger.debug(f"recording graph with columns_per_graph={inputs['columns_per_graph']}")
            subgraphs = record_graphs(inputs, columns=columns, device=W.device)

            graphs[graph_spec] = subgraphs, inputs
        else:
            ref_inputs = inputs

        # We always replay the graph, even in case we went through the `record_graph` step.
        # The reason is that the `record_graph` step has a warmup that modifies in-place
        # the inputs.
        subgraphs, graph_inputs = graphs[graph_spec]

        for name, val in ref_inputs.items():
            if isinstance(val, torch.Tensor):
                if name != "Hinv":
                    assert val.is_contiguous()
                    assert graph_inputs[name].is_contiguous()
                assert graph_inputs[name].device == val.device

                graph_inputs[name].copy_(val)
            elif name == "columns_start_idx":
                pass
            elif not isinstance(val, ScaledFakeQuantize):
                assert graph_inputs[name] == val

        Q = replay_fasterquant_inner_graphs(subgraphs, graph_inputs)

        Losses = None

        scale = None
        zero_point = None

    return Q, Losses, scale, zero_point


def fasterquant_inner_eager(
    Hinv: torch.Tensor,
    W: torch.Tensor,
    Q: torch.Tensor,
    perm: torch.Tensor | None,
    quantizers: list[ScaledFakeQuantize],
    actorder: bool,
    group_size: int,
    columns: int,
    blocksize: int,
    static_groups: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    Losses = torch.zeros_like(W)
    Q1 = torch.zeros_like(W[:, :blocksize])
    Err1 = torch.zeros_like(W[:, :blocksize])
    Losses1 = torch.zeros_like(W[:, :blocksize])

    scale = []
    zero = []
    now_idx = 1

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2]

        if i1 + blocksize > columns:
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)

        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if group_size is not None and group_size > 0:
                if not static_groups:
                    if (i1 + i) % group_size == 0:
                        quantizer = quantizers[0]

                        quantizer.observer.reset_min_max_vals()
                        quantizer.observe(W[:, (i1 + i) : (i1 + i + group_size)])
                    if ((i1 + i) // group_size) - now_idx == -1:
                        scale.append(quantizer.scale)
                        zero.append(quantizer.zero_point)
                        now_idx += 1
                else:
                    idx = i1 + i
                    if actorder:
                        idx = perm[idx]  # type: ignore
                    quantizer = quantizers[idx // group_size]
            else:
                quantizer = quantizers[0]

            q = quantizer.fake_quantize_with_qparams(
                w.unsqueeze(1), scale=quantizer.scale, zero_point=quantizer.zero_point
            )
            q = q.squeeze(dim=1)

            Q1[:, i] = q

            Losses1[:, i] = (w - q) ** 2 / d**2

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))

            Err1[:, i] = err1

        Q[:, i1:i2] = Q1

        Losses[:, i1:i2] = Losses1 / 2

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    return Q, Losses, scale, zero


class GPTQ:
    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.dev = self.layer.weight.device
        if self.dev == META:
            # should be execute_device, When cuda0 is very small, it could be any other value.
            self.dev = self.layer._hf_hook.execution_device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)

        # Transformers might not be in the user environment, hence the class name check instead.
        if "transformers.pytorch_utils.Conv1D" in str(self.layer.__class__):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        # self.H: Optional[torch.Tensor] = torch.zeros((self.columns, self.columns), device=self.dev)
        self.H: torch.Tensor | None = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float)
        self.nsamples = 0
        self.inp1: torch.Tensor | None = None
        self.out1: torch.Tensor | None = None
        self.original_qspec = self.layer._weight_quantizer.observer.qspec
        kwargs: Any = {}
        from quark.torch.quantization.config.config import QuantizationSpec
        from quark.torch.quantization.tensor_quantize import FakeQuantizeBase

        # for per group minmaxobserver: group_size > 1 and group_size == -1
        if self.original_qspec.qscheme == QSchemeType.per_group:
            self.adjusted_qspec = QuantizationSpec(
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
            self.quantizer = FakeQuantizeBase.get_fake_quantize(self.adjusted_qspec, self.dev, **kwargs)
        # pertensor & perchannel
        else:
            self.quantizer = layer._weight_quantizer

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor, name: str) -> None:
        assert self.H is not None
        if os.environ.get("DEBUG"):
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or "transformers.pytorch_utils.Conv1D" in str(self.layer.__class__):
            if len(inp.shape) == 3:
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
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize: int,
        percdamp: float,
        group_size: int,
        actorder: bool,
        static_groups: bool,
        use_cuda_graphs: bool,
        graphs: dict[str, tuple[list[torch.cuda.CUDAGraph], dict[str, Any]]],
    ) -> None:
        assert self.H is not None

        per_group = group_size is not None and group_size > 0

        if get_device(self.layer) == META:  # get from cpu dict
            W = self.layer._hf_hook.weights_map["weight"].data.to(
                self.layer._hf_hook.execution_device
            )  # Need to be cleaned up.
        else:
            W = self.layer.weight.data.clone()
        orig_dtype = W.dtype
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if "transformers.pytorch_utils.Conv1D" in str(self.layer.__class__):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        scale: list[torch.Tensor] = []
        zero: list[torch.Tensor] = []

        quantizers = []
        if per_group:
            if static_groups:
                # only pergroup group_size > 0 need static_group
                # if not static, we will create quantizer for pergroup (groupsize > 0) in the following codes.
                for i in range(0, self.columns, group_size):
                    quantizer = copy.deepcopy(self.quantizer)  # TODO: this is very slow as well!
                    quantizer.observe(W[:, i : (i + group_size)])

                    scale.append(quantizer.scale)
                    zero.append(quantizer.zero_point)
                    quantizers.append(quantizer)
            else:
                quantizers.append(self.quantizer)
        else:
            # per-tensor, per-channel, and group_size = -1 cases.
            self.quantizer.observe(W)

            quantizers.append(self.quantizer)
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero_point)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        else:
            perm = None
            invperm = None

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)

        # TODO: H is not contiguous here as `torch.linalg.cholesky` produces non-contiguous outputs. We might want to fix this in the future, but GPTQ algo somehow yields very slightly different outputs/losses when using contiguous `Hinv` vs non-contiguous `Hinv`. This would need further investigation.
        Hinv = H

        fasterquant_inner_kwargs = {
            "graphs": graphs,
            "Hinv": Hinv,
            "W": W,
            "perm": perm,
            "quantizers": quantizers,
            "actorder": actorder,
            "group_size": group_size,
            "columns": self.columns,
            "blocksize": blocksize,
            "static_groups": static_groups,
            "use_cuda_graphs": use_cuda_graphs,
        }

        if per_group and not static_groups:
            Q, Losses, scale, zero = fasterquant_inner(  # type: ignore[assignment]
                **fasterquant_inner_kwargs, scales=None, zero_points=None
            )
        else:
            if per_group:
                scales = torch.cat([s.view(-1, 1) for s in scale], dim=1)
                zero_points = torch.cat([z.view(-1, 1) for z in zero], dim=1)
            else:
                scales = scale[0]
                zero_points = zero[0]

            Q, Losses, _, _ = fasterquant_inner(**fasterquant_inner_kwargs, scales=scales, zero_points=zero_points)

        if torch.cuda.is_available():
            torch.cuda.synchronize(Q.device)

        logger.info(f"duration: {(time.time() - tick)}")

        if QUARK_GRAPH_DEBUG:
            logger.info(f"avg loss: {torch.sum(Losses).item() / self.nsamples}")  # type: ignore[arg-type]

        group_size_for_order = group_size if per_group else self.columns
        if static_groups and actorder and perm is not None and invperm is not None:
            g_idx = perm // group_size_for_order

            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if "transformers.pytorch_utils.Conv1D" in str(self.layer.__class__):
            Q = Q.t()

        if get_device(self.layer) == META:
            # Directly replace weight in dict with qweight
            self.layer._hf_hook.weights_map["weight"].data = Q.reshape(self.layer.weight.shape).to(orig_dtype).to("cpu")
        else:
            self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)

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


class GptqProcessor(BaseAlgoProcessor):
    def __init__(self, model: nn.Module, quant_algo_config: GPTQConfig, data_loader: DataLoader[torch.Tensor]) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.flags(enabled=True, allow_tf32=False)

        self.model = model
        self.block_size = quant_algo_config.block_size
        self.damp_percent = quant_algo_config.damp_percent
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

        use_cuda = next(iter(model.parameters())).device.type == "cuda"
        self.use_cuda_graphs = not QUARK_GRAPH_DEBUG and self.static_groups and use_cuda

        if self.use_cuda_graphs:
            logger.info("Using CUDA Graph for GPTQ column by column quantization.")
        elif QUARK_GRAPH_DEBUG:
            logger.info(
                f"CUDA Graph are not used for GPTQ column by column quantization as the environment variable QUARK_GRAPH_DEBUG is {QUARK_GRAPH_DEBUG}."
            )
        elif not self.static_groups:
            logger.info(
                "CUDA Graph are not used for GPTQ column by column quantization as `static_groups=False` is not supported in the CUDA Graph implementation."
            )
        elif not use_cuda:
            logger.info(
                "CUDA Graph are not used for GPTQ column by column quantization as the algorithm is running on CPU."
            )

    def apply(self) -> None:
        cache_examples_on_gpu = True
        num_batches = len(self.inps)
        layer_inputs = [inp for inp in self.inps]
        layer_outputs: list[torch.Tensor] = []
        forward_pass_use_cache = reset_model_kv_cache(self.model, use_cache=False)
        for i in tqdm(range(len(self.modules)), desc="GPTQ"):
            logger.info(f"Start quantizing layer {i + 1}/{len(self.modules)}")
            layer = self.modules[i]

            force_layer_back_to_cpu = False
            if get_device(layer) == CPU:
                move_to_device(layer, self.device_map[f"{self.model_decoder_layers}.{i}"])
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer) if not get_device(layer) == META else layer._hf_hook.execution_device

            # full.keys: ['self_attn.k_proj', 'feed_forward.experts.0.gate_proj', 'feed_forward.experts.1.gate_proj', ...]
            full = get_named_quant_linears(layer)
            assert self.inside_layer_modules is not None
            inside_layer_modules: list[str] = self.inside_layer_modules

            # inside_layer_modules: ['self_attn.k_proj', 'down_proj', ...]
            for names in inside_layer_modules:
                # support both dense and moe layers
                matched_names = fnmatch.filter(
                    full.keys(), "*" + names
                )  # e.g., 'self_attn.k_proj' <- '*self_attn.k_proj', 'feed_forward.experts.0.down_proj' <- '*down_proj'
                subset = {
                    name: full[name]
                    for name in matched_names
                    if getattr(full[name], "_weight_quantizer", None) is not None
                }

                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name])

                def add_batch(name: str) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
                    def tmp(_: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
                        gptq[name].add_batch(inp[0].data, out.data, name)

                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                # cal H
                layer_outputs = block_forward(
                    layer,
                    self.module_kwargs,
                    num_batches,
                    cur_layer_device,
                    layer_inputs,
                    layer_outputs,
                    cache_examples_on_gpu,
                )

                layer_outputs = []

                for h in handles:
                    h.remove()

                for name in subset:
                    logger.info(f"Quantizing {name} in layer {i + 1}/{len(self.modules)}...")
                    gptq[name].fasterquant(
                        blocksize=self.block_size,
                        percdamp=self.damp_percent,
                        group_size=subset[name]._weight_quantizer.group_size,
                        actorder=self.act_order,
                        static_groups=self.static_groups,
                        graphs=self.graphs,
                        use_cuda_graphs=self.use_cuda_graphs,
                    )
                    if not self.use_cuda_graphs:
                        gptq[name].free()

            # get whole decoder layer output
            layer_outputs = block_forward(
                layer,
                self.module_kwargs,
                num_batches,
                cur_layer_device,
                layer_inputs,
                layer_outputs,
                cache_examples_on_gpu,
            )

            if get_device(layer) != META:
                # if meta, scale and zero point are in execution_device, and weight is in meta, can't change.
                layer = move_to_device(layer, CPU if force_layer_back_to_cpu else cur_layer_device)
            del layer
            del gptq
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []

            if not self.use_cuda_graphs:
                clear_memory()

        reset_model_kv_cache(self.model, use_cache=forward_pass_use_cache)
