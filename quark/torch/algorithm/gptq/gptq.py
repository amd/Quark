#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023 潘其威(William)
# SPDX-License-Identifier: MIT
#

from __future__ import annotations
import math
import os
import time
from typing import Any, Callable, List, Optional, Tuple, cast, TYPE_CHECKING
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
if TYPE_CHECKING:
    from quark.torch.quantization.config.config import GPTQConfig
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import (get_device, get_named_quant_linears, move_to_device)
from quark.torch.algorithm.utils.prepare import init_device_map, init_blockwise_algo
from quark.torch.algorithm.blockwise_tuning.blockwise_utils import block_forward
from quark.torch.algorithm.utils.utils import clear_memory
from quark.shares.utils.log import ScreenLogger
from quark.torch.quantization.config.type import QSchemeType
from quark.torch.quantization.observer.observer import PerChannelMinMaxObserver

logger = ScreenLogger(__name__)

__all__ = ["GptqProcessor"]

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


class GPTQ:

    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)

        # Transformers might not be in the user environment, hence the class name check instead.
        if "transformers.pytorch_utils.Conv1D" in str(self.layer.__class__):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H: Optional[torch.Tensor] = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.inp1: Optional[torch.Tensor] = None
        self.out1: Optional[torch.Tensor] = None
        self.original_qspec = self.layer._weight_quantizer.observer.qspec
        kwargs: Any = {}
        from quark.torch.quantization.config.config import QuantizationSpec
        from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize
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
                mx_element_dtype=self.original_qspec.mx_element_dtype)
            self.quantizer = ScaledFakeQuantize(self.adjusted_qspec, **kwargs).to(
                self.layer._weight_quantizer.fake_quant_enabled.device)  # need copy
        # pertensor & perchannel
        else:
            self.quantizer = layer._weight_quantizer

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
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
            unfold = nn.Unfold(self.layer.kernel_size,
                               dilation=self.layer.dilation,
                               padding=self.layer.padding,
                               stride=self.layer.stride)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(self,
                    blocksize: int = 128,
                    percdamp: float = .01,
                    group_size: int = -1,
                    actorder: bool = False,
                    static_groups: bool = False) -> torch.Tensor:
        assert self.H is not None
        W = self.layer.weight.data.clone()
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

        g_idx = []
        scale = []
        zero = []
        now_idx = 1
        # for per group > 1
        if group_size is not None and group_size > 0:
            # only pergroup group_size > 0 need static_group
            # if not static, we will create quantizer for pergroup (groupsize > 0) in the following codes.
            if static_groups:
                import copy
                groups = []
                for i in range(0, self.columns, group_size):
                    quantizer = copy.deepcopy(self.quantizer)
                    quantizer.disable_fake_quant()
                    quantizer.enable_observer()
                    quantizer(W[:, i:(i + group_size)])
                    quantizer.disable_observer()
                    quantizer.enable_fake_quant()
                    scale.append(quantizer.scale)
                    zero.append(quantizer.zero_point)
                    groups.append(quantizer)

        # for per tensor, per channel(perchannel & pergroupsize=-1)
        else:
            self.quantizer.disable_fake_quant()
            self.quantizer.enable_observer()
            self.quantizer(W)
            self.quantizer.disable_observer()
            self.quantizer.enable_fake_quant()

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # only for per group > 0
                if group_size is not None and group_size > 0:
                    if not static_groups:
                        if (i1 + i) % group_size == 0:  # Changes when it is an integer multiple of group_size.
                            # clear min, max
                            self.quantizer.observer.reset_min_max_vals()
                            self.quantizer.disable_fake_quant()
                            self.quantizer.enable_observer()
                            self.quantizer(W[:, (i1 + i):(i1 + i + group_size)])
                            self.quantizer.disable_observer()
                            self.quantizer.enable_fake_quant()
                        if ((i1 + i) // group_size) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero_point)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = cast(int, perm[idx])
                        self.quantizer = groups[idx // group_size]
                    # pergroup_size > 0
                    q = self.quantizer(w.unsqueeze(1))
                else:
                    # pertensor, perchannel, pergroupsize == -1
                    q = self.quantizer(w.unsqueeze(1))
                q = q.squeeze(dim=1)
                Q1[:, i] = q

                Losses1[:, i] = (w - q)**2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1

            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info(f'duration: {(time.time() - tick)}')
        logger.info(f'avg loss: {torch.sum(Losses).item() / self.nsamples}')

        group_size_for_order = group_size if group_size is not None and group_size > 0 else self.columns
        if static_groups and actorder:
            g_idx = [perm[i] // group_size_for_order for i in range(self.columns)]
        else:
            g_idx = [i // group_size_for_order for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if "transformers.pytorch_utils.Conv1D" in str(self.layer.__class__):
            Q = Q.t()
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
        return g_idx

    def free(self) -> None:
        self.H = None
        clear_memory()


class GptqProcessor(BaseAlgoProcessor):

    def __init__(self, model: nn.Module, quant_algo_config: GPTQConfig, data_loader: DataLoader[torch.Tensor]) -> None:
        self.model = model
        self.damp_percent = quant_algo_config.damp_percent
        self.act_order = quant_algo_config.desc_act
        self.static_groups = quant_algo_config.static_groups
        self.true_sequential = quant_algo_config.true_sequential
        self.inside_layer_modules = quant_algo_config.inside_layer_modules
        self.model_decoder_layers = quant_algo_config.model_decoder_layers
        self.data_loader = data_loader
        self.device_map = init_device_map(self.model)
        self.modules, self.module_kwargs, self.inps = init_blockwise_algo(self.model, self.model_decoder_layers,
                                                                          self.data_loader)

    def apply(self) -> None:
        cache_examples_on_gpu = True
        num_batches = len(self.inps)
        layer_inputs = [inp for inp in self.inps]
        layer_outputs: List[torch.Tensor] = []
        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        for i in tqdm(range(len(self.modules)), desc="GPTQ"):
            logger.info(f"Start quantizing layer {i + 1}/{len(self.modules)}")
            layer = self.modules[i]

            force_layer_back_to_cpu = False
            if get_device(layer) == CPU:
                move_to_device(layer, self.device_map[f"{self.model_decoder_layers}.{i}"])
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer)

            full = get_named_quant_linears(layer)
            assert self.inside_layer_modules is not None
            inside_layer_modules: List[str] = self.inside_layer_modules
            if not self.true_sequential:
                inside_layer_modules = [''.join(self.inside_layer_modules)]

            for names in inside_layer_modules:
                subset = {names: full[names]}
                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name])

                def add_batch(name: str) -> Callable[[torch.nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:

                    def tmp(_: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
                        gptq[name].add_batch(inp[0].data, out.data)

                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))

                layer_outputs = block_forward(layer, self.module_kwargs, num_batches, cur_layer_device, layer_inputs,
                                              layer_outputs, cache_examples_on_gpu)
                layer_outputs = []

                for h in handles:
                    h.remove()

                for name in subset:
                    logger.info(f'Quantizing {name} in layer {i + 1}/{len(self.modules)}...')
                    g_idx = gptq[name].fasterquant(percdamp=self.damp_percent,
                                                   group_size=subset[name]._weight_quantizer.group_size,
                                                   actorder=self.act_order,
                                                   static_groups=self.static_groups)

                    gptq[name].free()

            layer_outputs = block_forward(layer, self.module_kwargs, num_batches, cur_layer_device, layer_inputs,
                                          layer_outputs, cache_examples_on_gpu)

            layer = move_to_device(layer, CPU if force_layer_back_to_cpu else cur_layer_device)

            del layer
            del gptq
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []
            clear_memory()
        self.model.config.use_cache = forward_pass_use_cache
