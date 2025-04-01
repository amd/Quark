#
# Copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations
import time
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
if TYPE_CHECKING:
    from quark.torch.pruning.config import OSSCARConfig
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import (get_device, get_named_linears, move_to_device)
from quark.torch.algorithm.utils.prepare import init_device_map, init_blockwise_algo
from quark.torch.algorithm.blockwise_tuning.blockwise_utils import block_forward
from quark.torch.algorithm.utils.utils import clear_memory
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ["OsscarProcessor"]

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


class OSSCAR:

    def __init__(self, layer: nn.Module, layer_idx: int) -> None:
        self.layer = layer
        self.layer_idx = layer_idx

        self.dev = self.layer.weight.device
        self.nsamples = 0
        W = layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)

        # Transformers might not be in the user environment, hence the class name check instead.
        if "transformers.pytorch_utils.Conv1D" in str(self.layer.__class__):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.XtX: Optional[torch.Tensor] = torch.zeros((self.columns, self.columns), device=self.dev)

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or "transformers.pytorch_utils.Conv1D" in str(self.layer.__class__):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        inp = inp.float()
        out = out.float()
        self.nsamples += tmp
        self.XtX += (inp).matmul(inp.t()) / tmp

    def prune(
        self,
        mlp_pruning_ratio: float,
        upd_iter: int = 1,
        percdamp: float = .01,
    ) -> None:

        assert self.XtX is not None

        W = self.layer.weight.data.clone()

        if "transformers.pytorch_utils.Conv1D" in str(self.layer.__class__):
            W = W.t()

        W = W.float()

        st_time = time.time()

        dead = torch.diag(self.XtX) == 0
        B = W.t()
        B[dead, :] = 0

        self.XtX += torch.eye(B.shape[0]).to(self.dev) * percdamp * torch.mean(torch.diag(self.XtX))

        self.XtY = (self.XtX @ B)

        pre_time = time.time() - st_time
        st_time = time.time()

        num_cin = B.shape[0]

        logger.info(f'mlp pruning ratio is : {(mlp_pruning_ratio)}')
        logger.info(f'input channel of layer is : {(num_cin)}')

        num_sp_orig = int(num_cin * (1 - mlp_pruning_ratio))

        if num_sp_orig % 128 != 0:
            num_sp = round(num_sp_orig / 128) * 128
            logger.info(
                f'pruned channel is : {(num_sp_orig)}, which can not divisible by 128 and is adjusted to be : {(num_sp)}'
            )
        else:
            num_sp = num_sp_orig
            logger.info(f'pruned channel is : {(num_sp)}')

        B_sol, B_obj = self.OSSCAR_fastprune(B.clone(), self.XtX, self.XtY, num_cin, num_sp, upd_iter)

        run_time = time.time() - st_time

        B = torch.Tensor(B_sol).to(self.dev)

        logger.info(f'pre-processing time: {(pre_time):.4f} seconds.')
        logger.info(f'OSSCAR pruning time: {(run_time):.4f} seconds.')

        if "transformers.pytorch_utils.Conv1D" in str(self.layer.__class__):
            self.layer.weight.data = B.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        else:
            self.layer.weight.data = B.t().reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        return

    def OSSCAR_fastprune(self,
                         W: torch.Tensor,
                         XTX: torch.Tensor,
                         XTY: torch.Tensor,
                         num_cin: int,
                         num_sp: int,
                         update_iter: int = 1,
                         blocksize: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:

        DEV = W.device
        totp, num_cout = W.shape
        ksize = int(totp / num_cin)

        H = torch.linalg.cholesky(XTX)
        H = torch.cholesky_inverse(H)
        XTX_inv = torch.linalg.cholesky(H, upper=True)

        if blocksize == -1:
            blocksize = num_cout

        W_reshaped = W.reshape(num_cin, ksize, num_cout)

        sum_across_cout = torch.sum(W_reshaped, dim=2)
        sum_across_ksize = torch.sum(sum_across_cout, dim=1)

        num_prune = torch.sum(torch.abs(sum_across_ksize) <= 1e-12)

        prune_list = torch.abs(sum_across_ksize) <= 1e-12

        if num_prune:
            upd_idx = torch.cat([torch.arange(i * ksize, (i + 1) * ksize) for i in range(num_cin) if prune_list[i]])
            XTX_inv[upd_idx, :] = 0
            XTX_inv[:, upd_idx] = 0

        W = XTX_inv @ XTY

        if int(num_cin - num_sp - num_prune) <= 0:
            upd_it = 0
        else:
            upd_it = int((num_cin - num_sp - num_prune) / update_iter)
            if upd_it == 0:
                upd_it = 1
            quo, rem = divmod(int(num_cin - num_sp - num_prune), int(upd_it))
            update_ten = torch.full((upd_it, ), quo, dtype=torch.int).to(DEV)
            update_ten[:rem] += 1

        for i1 in range(upd_it):
            obj_mat = torch.zeros_like(W)

            obj_mat = (1 / (prune_list + torch.diag(XTX_inv)))[:, None] * W / 2

            obj_cha = W * obj_mat
            obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
            obj_sum = torch.sum(torch.sum(obj_cha, dim=2), dim=1)

            idx = torch.argsort(obj_sum + 1e20 * (prune_list))

            upd_idx = torch.cat([
                torch.arange(idx[i].item() * ksize, (idx[i].item() + 1) * ksize)
                for i in range(int(update_ten[i1].item()))
            ])

            Xinv_tmp = torch.linalg.inv(XTX_inv[upd_idx[:, None], upd_idx])

            W -= XTX_inv[:, upd_idx] @ Xinv_tmp @ W[upd_idx, :]
            W = W.reshape(num_cin, ksize, num_cout)
            W[idx[:update_ten[i1]], :, :] = 0
            W = W.reshape(totp, num_cout)

            XTX_inv -= XTX_inv[:, upd_idx] @ Xinv_tmp @ XTX_inv[upd_idx, :]
            XTX_inv[upd_idx, :] = 0
            XTX_inv[:, upd_idx] = 0

            prune_list[idx[:update_ten[i1]]] = True

        W_sol = torch.zeros_like(W)
        nzi = torch.nonzero(W[:, 0], as_tuple=True)[0]
        W_sol[nzi, :] = torch.linalg.inv(XTX[nzi[:, None], nzi]) @ XTY[nzi, :]

        return W_sol, torch.sum(-W_sol * XTY + (1 / 2) * W_sol * (XTX @ W_sol))

    def free(self) -> None:
        self.XtX = None
        self.XtY = None
        clear_memory()


class OsscarProcessor(BaseAlgoProcessor):

    def __init__(self, model: nn.Module, pruning_algo_config: OSSCARConfig,
                 data_loader: DataLoader[torch.Tensor]) -> None:
        self.model = model
        self.damp_percent = pruning_algo_config.damp_percent
        self.true_sequential = pruning_algo_config.true_sequential
        self.inside_layer_modules = pruning_algo_config.inside_layer_modules
        self.mlp_pruning_modules = pruning_algo_config.mlp_pruning_modules
        self.mlp_pruning_ratio = pruning_algo_config.mlp_pruning_ratio
        self.model_decoder_layers = pruning_algo_config.model_decoder_layers
        self.data_loader = data_loader
        self.device_map = init_device_map(self.model)
        self.modules, self.module_kwargs, self.inps = init_blockwise_algo(self.model, self.model_decoder_layers,
                                                                          self.data_loader)
        assert self.mlp_pruning_modules is not None and self.inside_layer_modules is not None, "mlp_pruning_modules and inside_layer_modules must not be None"

    def apply(self) -> None:
        cache_examples_on_gpu = True
        num_batches = len(self.inps)
        layer_inputs = [inp for inp in self.inps]
        layer_outputs: List[torch.Tensor] = []
        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        for i in range(len(self.modules)):
            self.modules[i] = self.modules[i].to("cpu")
        clear_memory()

        for i in tqdm(range(len(self.modules)), desc="OSSCAR"):
            logger.info(f"Start pruning layer {i + 1}/{len(self.modules)}")
            layer = self.modules[i]

            force_layer_back_to_cpu = False
            if get_device(layer) == CPU:
                move_to_device(layer, self.device_map[f"{self.model_decoder_layers}.{i}"])
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer)

            full = get_named_linears(layer)

            inside_layer_modules = self.inside_layer_modules

            if not self.true_sequential:
                inside_layer_modules = [''.join(self.inside_layer_modules)]

            for names in inside_layer_modules:
                if isinstance(names, list):
                    subset = {n: full[n] for n in names if n in full}
                elif isinstance(names, str):
                    subset = {names: full[names]}
                else:
                    raise ValueError("inside_layer_modules set error")

                osscar = {}
                for name in subset:
                    if name in self.mlp_pruning_modules:
                        osscar[name] = OSSCAR(subset[name], i)

                def add_batch(name: str) -> Callable[[torch.nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:

                    def tmp(_: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
                        osscar[name].add_batch(inp[0].data, out.data)

                    return tmp

                handles = []
                for name in subset:
                    if name in self.mlp_pruning_modules:
                        handles.append(subset[name].register_forward_hook(add_batch(name)))

                # collect linear input data to calculate Hessian
                layer_outputs = block_forward(layer, self.module_kwargs, num_batches, cur_layer_device, layer_inputs,
                                              layer_outputs, cache_examples_on_gpu)
                layer_outputs = []

                for h in handles:
                    h.remove()

                for name in subset:
                    if name in self.mlp_pruning_modules:
                        logger.info(f'Pruning {name} in layer {i + 1}/{len(self.modules)}...')
                        osscar[name].prune(
                            mlp_pruning_ratio=self.mlp_pruning_ratio,
                            percdamp=self.damp_percent,
                        )
                        osscar[name].free()

            layer_outputs = block_forward(layer, self.module_kwargs, num_batches, cur_layer_device, layer_inputs,
                                          layer_outputs, cache_examples_on_gpu)

            layer = move_to_device(layer, CPU if force_layer_back_to_cpu else cur_layer_device)

            del layer
            del osscar
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []
            clear_memory()
        self.model.config.use_cache = forward_pass_use_cache
