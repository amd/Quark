#
# Copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import torch
import math
import numpy as np
from tqdm.auto import tqdm
import copy

import onnx
import onnxruntime
from onnxruntime.transformers.onnx_model import OnnxModel

from typing import Any, Dict, List, Tuple, Union
from collections import OrderedDict
from onnx import ModelProto, NodeProto, TensorProto
from numpy.typing import NDArray

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class GPTQ():

    def __init__(self, weight: NDArray[Any], extra_options: Dict[str, Any]) -> None:
        self.W = weight
        self.H = np.zeros((self.W.shape[0], self.W.shape[0]))
        self.scale: NDArray[Any] = np.zeros(1)
        self.zero: NDArray[Any] = np.zeros(1)
        self.maxq: NDArray[Any] = np.array(0)
        self.nsamples: int = 0
        self.extra_options = extra_options

    def add_batch(self, inp: NDArray[np.float32]) -> None:
        tmp = inp.shape[0]
        inp = np.reshape(inp, (-1, inp.shape[-1]))
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.astype(np.float32)
        self.H += np.matmul(np.transpose(inp), inp)

    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = .01,
        groupsize: int = -1,
        actorder: bool = False,
    ) -> Tuple[NDArray[np.float32], NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64]]:

        W = self.W.copy()
        if not self.ready():
            self.find_params(W)

        H = self.H
        del self.H
        dead = np.diag(H) == 0
        H[dead, dead] = 1
        W[dead, :] = 0

        if actorder:
            perm = np.argsort(-np.diag(H))
            W = W[perm, :]
            H = H[perm, :][:, perm]

        Losses = np.zeros_like(W)
        Q = np.zeros_like(W)
        Q_int = np.zeros_like(W)

        damp = percdamp * np.mean(np.diag(H))
        diag = np.arange(W.shape[0])

        H[diag, diag] += damp
        H = np.linalg.cholesky(H)
        H = np.linalg.inv(H)
        H = np.linalg.cholesky(H.T @ H).T
        Hinv = H

        for i1 in range(0, W.shape[0], blocksize):
            i2 = min(i1 + blocksize, W.shape[0])
            count = i2 - i1

            W1 = copy.deepcopy(W[i1:i2, :])
            Q1 = np.zeros_like(W1)
            Q1_int = np.zeros_like(W1)
            Err1 = np.zeros_like(W1)
            Losses1 = np.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[i, :]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.find_params(W[(i1 + i):(i1 + i + groupsize), :])

                q_int = self.quantize_int(w, self.scale, self.zero, self.maxq).flatten()
                q = self.scale * (q_int - self.zero)

                Q1[i, :] = q
                Q1_int[i, :] = q_int
                Losses1[i, :] = (w - q)**2 / d**2

                err1 = (w - q) / d
                W1[i:, :] -= np.matmul(np.expand_dims(Hinv1[i:, i], axis=1), np.expand_dims(err1, axis=0))
                Err1[i, :] = err1

            Q[i1:i2, :] = Q1
            Q_int[i1:i2, :] = Q1_int
            Losses[i1:i2, :] = Losses1 / 2

            W[i2:, :] -= np.matmul(Hinv[i2:, i1:i2], Err1)

        if actorder:
            invperm = np.argsort(perm)
            Q = Q[invperm, :]
            Q_int = Q_int[invperm, :]

        Q = Q.reshape(self.W.shape)
        Q_int = Q_int.reshape(self.W.shape).astype(np.uint8)
        self.scale = self.scale.astype(np.float32)
        self.zero = self.zero.astype(Q_int.dtype)

        del W
        return Q, Q_int, self.scale, self.zero

    def configure(
        self,
        bits: int = 8,
        perchannel: bool = False,
        sym: bool = False,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = .8,
        trits: bool = False,
    ) -> None:
        self.maxq = np.array(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x: NDArray[Any]) -> None:
        shape = x.shape
        if not self.perchannel:
            x = np.expand_dims(x.flatten(), axis=1)

        tmp = np.zeros(x.shape[1], dtype=x.dtype)
        xmin = np.minimum(np.min(x, axis=0), tmp)
        xmax = np.maximum(np.max(x, axis=0), tmp)

        if self.sym:
            xmax = np.maximum(np.abs(xmin), xmax)
            tmp = xmin < 0
            if np.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = np.ones(self.scale.shape) * (self.maxq + 1) / 2
            else:
                self.zero = np.round(-xmin / self.scale)
        if self.mse:
            best = np.full([x.shape[1]], float('inf'))
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = np.round(-xmin1 / scale1) if not self.sym else self.zero
                q = self.quantize_real(x, np.expand_dims(scale1, axis=0), np.expand_dims(zero1, axis=0), self.maxq)
                q -= x
                q = np.abs(q)
                q = np.power(q, self.norm)
                err = np.sum(q, 0)
                tmp = err < best
                if np.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            tmp = shape[1]
            self.scale = np.repeat(self.scale, tmp)
            self.zero = np.repeat(self.zero, tmp)
        shape = tuple([-1] + [1] * (len(shape) - 1))
        self.scale = np.reshape(self.scale, shape)
        self.zero = np.reshape(self.zero, shape)

        if len(shape) == 2:
            self.scale = np.squeeze(self.scale, axis=1)
            self.zero = np.squeeze(self.zero, axis=1)

        return

    def ready(self) -> bool:
        return bool(np.all(self.scale != 0))

    def quantize_int(self, x: NDArray[Any], scale: NDArray[np.float64], zero: NDArray[np.float64],
                     maxq: NDArray[Any]) -> Any:
        if maxq < 0:
            return ((x > scale / 2.0) * scale + (x < zero / 2.0) * zero)
        q = np.clip(np.round(x / scale) + zero, 0, maxq).astype(x.dtype)
        return q

    def quantize_real(self, x: NDArray[Any], scale: NDArray[np.float64], zero: NDArray[np.float64],
                      maxq: NDArray[Any]) -> Any:
        if maxq < 0:
            return ((x > scale / 2.0) * scale + (x < zero / 2.0) * zero)
        q = np.clip(np.round(x / scale) + zero, 0, maxq).astype(x.dtype)
        return scale * (q - zero)


class GptqProcessor():

    def __init__(
            self,
            model_path: str,
            float_model: Union[ModelProto, str],
            quantized_model: Union[ModelProto, str],
            dataloader: torch.utils.data.DataLoader,  # type: ignore
            extra_options: Dict[str, Any],
            is_large: bool = True,
            providers: List[str] = ["CPUExecutionProvider"]) -> None:
        self.float_model = onnx.load(float_model) if isinstance(float_model, str) else float_model
        self.quantized_model = onnx.load(quantized_model) if isinstance(quantized_model, str) else quantized_model
        self.onnx_model_float = OnnxModel(self.float_model)
        self.onnx_model_quant = OnnxModel(self.quantized_model)

        self.quant_node_list: List[NodeProto] = []
        self.ln_outputs: List[str] = []
        self.out_dict: Dict[str, TensorProto] = {}
        self.extend_output_nodes: List[str] = []
        self.dataloader = dataloader
        self.base_dir = os.path.dirname(model_path)
        self.gptq_model_path = os.path.join(self.base_dir, "decoder_model_gptq.onnx")
        self.tmp_model_path = os.path.join(self.base_dir, "decoder_model_tmp.onnx")
        self.is_large = is_large
        self.providers = providers
        self.extra_options = extra_options

        self.bits = self.extra_options.get('GPTQParams', {}).get('Bits', 8)
        self.blocksize = self.extra_options.get('GPTQParams', {}).get('BlockSize', 128)
        self.percdamp = self.extra_options.get('GPTQParams', {}).get('PercDamp', 0.01)
        self.groupsize = self.extra_options.get('GPTQParams', {}).get('GroupSize', -1)
        self.actorder = self.extra_options.get('GPTQParams', {}).get('ActOrder', False)
        self.perchannel = self.extra_options.get('GPTQParams', {}).get('PerChannel', False)
        self.sym = self.extra_options.get('GPTQParams', {}).get('WeightSymmetric', True)
        self.mse = self.extra_options.get('GPTQParams', {}).get('MSE', False)

        self.accuracy_level = 0

    def init_gptq_quant(self) -> None:
        matmul_node_list = self.onnx_model_float.get_nodes_by_op_type("MatMul")
        node_list = matmul_node_list
        self.quant_node_list = []
        initializer_names = {init.name for init in self.onnx_model_float.model.graph.initializer}

        # determine whether matmul op has weight
        for node in node_list:
            if node.input[1] in initializer_names:
                self.quant_node_list.append(node)
        for node in self.quant_node_list:
            if node.input[0] not in self.ln_outputs:
                self.ln_outputs.append(node.input[0])
                self.float_model.graph.output.extend([onnx.ValueInfoProto(name=node.input[0])])
                self.extend_output_nodes.append(node.input[0])

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.is_large:
            self.onnx_model_float.save_model_to_file(self.tmp_model_path,
                                                     use_external_data_format=True,
                                                     all_tensors_to_one_file=True)
            session = onnxruntime.InferenceSession(self.tmp_model_path, sess_options, providers=self.providers)
        else:
            session = onnxruntime.InferenceSession(self.float_model.SerializeToString(),
                                                   sess_options,
                                                   providers=self.providers)

        for inputs in tqdm(self.dataloader):
            inputs_dict = inputs
            ort_outs = session.run(self.ln_outputs, inputs_dict)
            self.out_dict = OrderedDict(zip(self.ln_outputs, ort_outs))
            break

    def remove_extend_output_node(self) -> None:
        for node in self.extend_output_nodes:
            if onnx.ValueInfoProto(name=node) in self.float_model.graph.output:
                self.float_model.graph.output.remove(onnx.ValueInfoProto(name=node))

    def get_gptq_model(self) -> ModelProto:
        model = OnnxModel(onnx.load(self.gptq_model_path))
        return model  # type:ignore

    def apply(self) -> ModelProto:
        # pre-processing
        self.init_gptq_quant()

        for node in self.quant_node_list:
            out_dict_emd = self.out_dict[node.input[0]]
            weight_init = self.onnx_model_float.get_initializer(node.input[1])
            weight_data = onnx.numpy_helper.to_array(weight_init, self.base_dir)
            quantized_per_channel_data_list = []

            # GPTQ process
            gptq = GPTQ(weight_data, self.extra_options)
            gptq.configure(bits=8, perchannel=self.perchannel, sym=self.sym, mse=self.mse, trits=False)
            gptq.add_batch(out_dict_emd)
            quantized_weights, quantized_weights_int, scale, zero = gptq.fasterquant(blocksize=self.blocksize,
                                                                                     percdamp=self.percdamp,
                                                                                     groupsize=-1,
                                                                                     actorder=self.actorder)
            quantized_per_channel_data_list.append(quantized_weights_int)
            quantized_weights_int = np.concatenate(quantized_per_channel_data_list, axis=1)

            quantized_weights_init = onnx.numpy_helper.from_array(quantized_weights_int,
                                                                  weight_init.name + '_quantized')
            weight_init_quant = self.onnx_model_quant.get_initializer(weight_init.name + '_quantized')
            if weight_init_quant is not None:
                self.onnx_model_quant.model.graph.initializer.remove(weight_init_quant)
                self.onnx_model_quant.model.graph.initializer.append(quantized_weights_init)

            scale_init = onnx.numpy_helper.from_array(scale, weight_init.name + '_scale')
            scale_init_quant = self.onnx_model_quant.get_initializer(weight_init.name + '_scale')
            if scale_init_quant is not None:
                self.onnx_model_quant.model.graph.initializer.remove(scale_init_quant)
                self.onnx_model_quant.model.graph.initializer.append(scale_init)

            zero_init = onnx.numpy_helper.from_array(zero, weight_init.name + '_zero_point')
            zero_init_quant = self.onnx_model_quant.get_initializer(weight_init.name + '_zero_point')
            if zero_init_quant is not None:
                self.onnx_model_quant.model.graph.initializer.remove(zero_init_quant)
                self.onnx_model_quant.model.graph.initializer.append(zero_init)

        self.remove_extend_output_node()

        return self.onnx_model_quant.model  # type: ignore

    def prepare_matmul4bits_node(self, node: NodeProto, quantized_weights: NDArray[Any], scale: NDArray[Any],
                                 zero: NDArray[Any]) -> Tuple[NodeProto, List[TensorProto]]:
        bits = self.extra_options.get('MatMulNBitsParams', {}).get('Bits', 4)
        kwargs: Dict[str, Any] = {}
        rows, cols = quantized_weights.shape
        kwargs["K"] = rows
        kwargs["N"] = cols
        kwargs["bits"] = bits
        group_size = self.groupsize if self.groupsize != -1 else quantized_weights.shape[0]
        kwargs["block_size"] = group_size
        blob_size = group_size // 2
        k_blocks = (rows + group_size - 1) // group_size

        # Add paddings if needed
        pad_len = k_blocks * group_size - quantized_weights.shape[0]
        if pad_len > 0:
            quantized_weights = np.pad(quantized_weights, ((0, pad_len), (0, 0)), "constant")

        # prepare weight
        quantized_weights = np.reshape(quantized_weights.T, (-1, group_size))
        min_q = np.min(quantized_weights, axis=1, keepdims=True)
        max_q = np.max(quantized_weights, axis=1, keepdims=True)
        range_q = np.maximum(np.abs(min_q), np.abs(max_q))
        mask = range_q > 0
        new_scale = np.ones(max_q.shape)

        # prepare scale and zero
        if self.sym:
            new_scale[mask] = (range_q[mask] * 2.0).astype(np.float64) / (2**bits - 1)
            new_zero = np.ones(max_q.shape).astype(np.uint8) * (1 << (bits - 1))
        else:
            new_scale[max_q != min_q] = np.array(
                [float(q) / (2**bits - 1) for q in (max_q - min_q)[max_q != min_q].flatten().tolist()])
            new_zero = np.maximum(0,
                                  np.minimum(2**bits - 1,
                                             ((np.zeros(new_scale.shape) - min_q) / new_scale).round())).astype("uint8")
        new_scale = new_scale.astype(scale.dtype)
        quantized_weights = np.clip(np.round(quantized_weights / new_scale + new_zero), 0, 2**bits - 1).astype(np.uint8)

        # create weight tensor
        packed_weight = np.zeros((quantized_weights.shape[0], blob_size)).astype(np.uint8)
        pack_weight_pair = (quantized_weights[:, ::2]) | (quantized_weights[:, 1::2] << 4)
        packed_weight[:, :] = pack_weight_pair[:, :blob_size]
        packed_weight = np.reshape(packed_weight, (-1, k_blocks, blob_size))
        weight_tensor = onnx.numpy_helper.from_array(packed_weight)  # noqa: N806
        weight_tensor.name = node.input[1] + "_Q4"

        # create scale tensor
        new_scale = np.reshape(new_scale, (-1, k_blocks))
        scale_tensor = onnx.numpy_helper.from_array(new_scale)
        scale_tensor.name = node.input[1] + "_scales"

        # create zero tensor
        packed_zero = np.full((new_zero.shape[0], 1), 136, dtype="uint8")
        packed_zero[:packed_zero.shape[0] // 2, :] = (new_zero[::2, :]) | (new_zero[1::2, :] << 4)
        zero_tensor = onnx.numpy_helper.from_array(packed_zero)
        zero_tensor.name = node.input[1] + "_zero_points"

        # create matmul4bits node
        input_names = [node.input[0], weight_tensor.name, scale_tensor.name]
        new_inits = [weight_tensor, scale_tensor]
        if not self.sym:
            input_names.append(zero_tensor.name)
            new_inits.append(zero_tensor)

        matmul_q4_node = onnx.helper.make_node(
            "MatMulNBits",
            inputs=input_names,
            outputs=[node.output[0]],
            name=node.name + "_Q4" if node.name else "",
            domain="com.microsoft",
            **kwargs,
        )

        return matmul_q4_node, new_inits

    def apply_matmul4bits(self) -> ModelProto:
        # pre-processing
        self.init_gptq_quant()
        graph = self.onnx_model_float.model.graph
        new_nodes = []

        for node in graph.node:
            if (node.op_type != 'MatMul') or (node not in self.quant_node_list):
                new_nodes.append(node)
                continue

            out_dict_emd = self.out_dict[node.input[0]]
            weight_init = self.onnx_model_float.get_initializer(node.input[1])
            weight_data = onnx.numpy_helper.to_array(weight_init, self.base_dir).copy()
            quantized_per_channel_data_list = []

            # GPTQ process
            gptq = GPTQ(weight_data, self.extra_options)
            gptq.configure(bits=4, perchannel=self.perchannel, sym=self.sym, mse=self.mse, trits=False)
            gptq.add_batch(out_dict_emd)
            quantized_weights, quantized_weights_int, scale, zero = gptq.fasterquant(blocksize=self.blocksize,
                                                                                     percdamp=self.percdamp,
                                                                                     groupsize=self.groupsize,
                                                                                     actorder=self.actorder)
            quantized_per_channel_data_list.append(quantized_weights)
            quantized_weights = np.concatenate(quantized_per_channel_data_list, axis=1)
            graph.initializer.remove(weight_init)

            matmul_q4_node, new_inits = self.prepare_matmul4bits_node(node, quantized_weights, scale, zero)
            graph.initializer.extend(new_inits)
            new_nodes.append(matmul_q4_node)

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        self.remove_extend_output_node()

        self.onnx_model_float.save_model_to_file(self.gptq_model_path,
                                                 use_external_data_format=self.is_large,
                                                 all_tensors_to_one_file=True)

        return self.get_gptq_model()
