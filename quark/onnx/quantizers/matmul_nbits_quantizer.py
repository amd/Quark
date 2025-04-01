#
# Modifications copyright(c) 2023 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

from typing import Optional, Tuple, Any, Dict, List, Union

import numpy as np
import numpy.typing as npt
import onnx
import torch
from pathlib import Path

from onnx.onnx_pb import GraphProto, ModelProto, NodeProto, TensorProto

from onnxruntime.capi._pybind_state import quantize_matmul_4bits

from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import attribute_to_kwarg

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def get_initializer(name: str, graph_path: list[GraphProto]) -> tuple[Optional[TensorProto], Optional[GraphProto]]:
    for gid in range(len(graph_path) - 1, -1, -1):
        graph = graph_path[gid]
        for tensor in graph.initializer:
            if tensor.name == name:
                return tensor, graph
    return None, None


class WeightOnlyQuantConfig:

    def __init__(self, algorithm: str) -> None:
        """This is the Base class for Weight Only Quant Configuration.

        Args:
            algorithm:
                weight only quantize algorithm name.
        """
        self.algorithm = algorithm


class DefaultWeightOnlyQuantConfig(WeightOnlyQuantConfig):

    def __init__(
        self,
        block_size: int = 128,
        is_symmetric: bool = False,
        bits: int = 4,
        accuracy_level: int | None = None,
    ):
        super().__init__(algorithm="DEFAULT")
        self.block_size = block_size
        self.is_symmetric = is_symmetric
        self.bits = bits
        self.accuracy_level = accuracy_level


class HQQWeightOnlyQuantConfig(WeightOnlyQuantConfig):

    def __init__(
        self,
        block_size: int = 128,
        bits: int = 4,
        axis: int = 1,
    ):
        super().__init__(algorithm="HQQ", )
        self.block_size = block_size
        self.bits = bits
        self.axis = axis


class GPTQWeightOnlyQuantConfig(WeightOnlyQuantConfig):

    def __init__(
            self,
            calibration_data_reader: torch.utils.data.DataLoader,  # type: ignore
            percdamp: float = 0.01,
            block_size: int = 128,
            actorder: bool = False,
            mse: bool = False,
            perchannel: bool = True):
        super().__init__(algorithm="GPTQ")
        self.calibration_data_reader = calibration_data_reader
        self.percdamp = percdamp
        self.block_size = block_size

        self.actorder = actorder
        self.mse = mse
        self.perchannel = perchannel


def get_onnx_initializer(name: str, graph_path: list[GraphProto]) -> Tuple[Optional[TensorProto], Any]:
    for gid in range(len(graph_path) - 1, -1, -1):
        graph = graph_path[gid]
        for tensor in graph.initializer:
            if tensor.name == name:
                return tensor, graph
    return None, None


class DefaultWeightOnlyQuantizer:

    def __init__(self, config: DefaultWeightOnlyQuantConfig):
        self.config = config

    def int4_block_quant(
            self, fp32weight: Any) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """4b quantize fp32 weight to a blob"""

        if len(fp32weight.shape) != 2:
            raise ValueError("Current int4 block quantization only supports 2D tensors!")
        rows, cols = fp32weight.shape
        block_size = self.config.block_size
        blob_size = block_size // 2
        k_blocks = (rows + block_size - 1) // block_size
        padded_rows = k_blocks * block_size
        pad_len = padded_rows - rows
        if pad_len > 0:
            fp32weight = np.pad(fp32weight, ((0, pad_len), (0, 0)), "constant")

        # block wise quantization, each block comes from a single column
        packed = np.zeros((cols, k_blocks, blob_size), dtype="uint8")
        scales = np.zeros((cols * k_blocks), dtype=fp32weight.dtype)
        zero_point = np.zeros(cols * ((k_blocks + 1) // 2), dtype="uint8")
        quantize_matmul_4bits(packed, fp32weight, scales, zero_point, block_size, cols, rows, self.config.is_symmetric)

        return (packed, scales, zero_point)

    def quantize(self, node: NodeProto, graph_stack: list[GraphProto]) -> NodeProto:
        """If the node is MatMul with fp32 const weight, quantize the weight with int4, and return the new node"""

        if node.op_type != "MatMul":
            return node  # only care about MatMul for now

        logger.info(f"start to quantize {node.name} ...")
        inputB = node.input[1]  # noqa: N806
        B, Bs_graph = get_onnx_initializer(inputB, graph_stack)  # noqa: N806
        if B is None:
            logger.info("MatMul doesn't have const weight. Skip to quantize")
            return node  # only care about constant weight

        B_array = onnx.numpy_helper.to_array(B)  # noqa: N806
        if len(B_array.shape) != 2:
            logger.info("MatMul weight is not 2D. Skip to quantize")
            return node  # can only process 2-D matrix

        packed, scales, zero_points = self.int4_block_quant(B_array)
        B_quant = onnx.numpy_helper.from_array(packed)  # noqa: N806
        B_quant.name = B.name + "_Q4"
        for input_ in Bs_graph.input:
            if input_.name == inputB:
                Bs_graph.input.remove(input_)
                break

        scales_tensor = onnx.numpy_helper.from_array(scales)
        scales_tensor.name = B.name + "_scales"
        Bs_graph.initializer.extend([B_quant, scales_tensor])

        input_names = [node.input[0], B_quant.name, scales_tensor.name]
        if not self.config.is_symmetric:
            zp_tensor = onnx.numpy_helper.from_array(zero_points)
            zp_tensor.name = B.name + "_zero_points"
            Bs_graph.initializer.extend([zp_tensor])
            input_names.append(zp_tensor.name)

        kwargs: Dict[str, Any] = {}
        rows, cols = B_array.shape
        kwargs["K"] = rows
        kwargs["N"] = cols
        kwargs["bits"] = self.config.bits
        kwargs["block_size"] = self.config.block_size
        if self.config.accuracy_level is not None:
            kwargs["accuracy_level"] = self.config.accuracy_level

        matmul_q4_node = onnx.helper.make_node(
            "MatMulNBits",
            inputs=input_names,
            outputs=[node.output[0]],
            name=node.name + "_Q4" if node.name else "",
            domain="com.microsoft",
            **kwargs,
        )

        logger.info(f"complete quantization of {node.name} ...")

        return matmul_q4_node


class MatMulNBitsQuantizer:
    """Perform 4b quantization of constant MatMul weights"""

    def __init__(self,
                 model: ModelProto | str,
                 block_size: int = 128,
                 is_symmetric: bool = False,
                 bits: int = 4,
                 accuracy_level: int | None = None,
                 nodes_to_exclude: Optional[List[str]] = None,
                 algo_config: Optional[WeightOnlyQuantConfig] = None,
                 extra_options: Dict[str, Any] = {}):
        if nodes_to_exclude is None:
            nodes_to_exclude = []
        self.model = ONNXModel(onnx.load(model)) if isinstance(model, str) else ONNXModel(model)
        self.model_gptq = onnx.load(model) if isinstance(model, str) else model
        self.model_path = model if isinstance(model, str) else None
        self.block_size = block_size
        self.is_symmetric = is_symmetric
        self.accuracy_level = accuracy_level
        self.nodes_to_exclude = set(nodes_to_exclude)
        if algo_config is None:
            algo_config = DefaultWeightOnlyQuantConfig(block_size=block_size,
                                                       is_symmetric=is_symmetric,
                                                       bits=bits,
                                                       accuracy_level=accuracy_level)
        self.algo_config = algo_config
        if self.algo_config.algorithm == "HQQ":
            self.node_quantizer = HQQWeightOnlyQuantizer(self.algo_config)  # type: ignore
        elif self.algo_config.algorithm == "DEFAULT":
            self.node_quantizer = DefaultWeightOnlyQuantizer(self.algo_config)  # type: ignore
        self.extra_options = extra_options

    def _process_subgraph(self, graph_stack: list[GraphProto]) -> GraphProto:
        new_nodes = []
        graph = graph_stack[-1]

        for node in graph.node:
            # TODO: The support of subgraph need to be verified.
            graph_attrs = [
                attr for attr in node.attribute
                if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS
            ]
            if len(graph_attrs):
                kwargs = {}
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        # recursive call to take care of sub-graph
                        graph_stack.append(attr.g)
                        kv: Any = {attr.name: self._process_subgraph(graph_stack)}
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        value = []
                        for subgraph in attr.graphs:
                            # recursive call to take care of sub-graph
                            graph_stack.append(subgraph)
                            value.extend([self._process_subgraph(graph_stack)])
                        kv = {attr.name: value}
                    else:
                        kv = attribute_to_kwarg(attr)
                    kwargs.update(kv)
                node = onnx.helper.make_node(  # noqa: PLW2901
                    node.op_type, node.input, node.output, name=node.name, **kwargs)
            out_node = None
            if node.name in self.nodes_to_exclude:
                logger.info(f"exclude to quantize {node.name} as specified by nodes_to_exclude...")
                out_node = node
            else:
                out_node = self.node_quantizer.quantize(node, graph_stack)
            new_nodes.append(out_node)

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        graph_stack.pop()
        return graph

    def quantize_model(self) -> None:
        if self.algo_config.algorithm in ["HQQ", "DEFAULT"]:
            # use a stack to keep track of sub-graphs
            graph_stack = [self.model.graph()]
            opset_import = self.model.opset_import()

            has_ms_domain = False
            for opset in opset_import:
                if opset.domain == "com.microsoft":
                    has_ms_domain = True
            if not has_ms_domain:
                opset_import.extend([onnx.helper.make_opsetid("com.microsoft", 1)])
            self._process_subgraph(graph_stack)
            self.model.clean_initializers()
        elif self.algo_config.algorithm in ["GPTQ"]:
            from quark.onnx.gptq.gptq import GptqProcessor
            import tempfile
            gptq_path = tempfile.TemporaryDirectory(prefix="vai.quant.")
            gptq_model_output = Path(gptq_path.name).joinpath("gptq_model.onnx").as_posix()
            gptq_processor = GptqProcessor(
                gptq_model_output,
                self.model_gptq,
                self.model_gptq,
                self.algo_config.calibration_data_reader,  # type: ignore[attr-defined]
                self.extra_options)
            self.model = gptq_processor.apply_matmul4bits()


class HQQWeightOnlyQuantizer:

    def __init__(
        self,
        config: HQQWeightOnlyQuantConfig,
    ):
        self.config = config

    @staticmethod
    def optimize_weights(
        tensor: TensorProto,
        scale: torch.Tensor,
        zero: torch.Tensor,
        min_max: list[int],
        axis: int = 0,
        opt_params: Optional[Dict[str, Union[float, int]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        opt_params = {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20} if opt_params is None else opt_params
        lp_norm, beta, kappa, iters = (
            opt_params["lp_norm"],
            opt_params["beta"],
            opt_params["kappa"],
            opt_params["iters"],
        )

        dtype = torch.float16 if tensor.is_cuda else torch.float32
        w_f = tensor.to(dtype)
        scale = scale.to(dtype)
        zero = zero.to(dtype)

        def shrink_op(x: torch.Tensor, beta: float, p: float = lp_norm) -> torch.Tensor:
            return torch.sign(x) * torch.nn.functional.relu(
                torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x) + 1e-8, p - 1))

        best_error = 1e4
        for i in range(int(iters)):
            w_q = torch.round(w_f * scale + zero).clamp(min_max[0], min_max[1])
            w_r = (w_q - zero) / scale
            w_e = shrink_op(w_f - w_r, beta)
            zero = torch.mean(w_q - (w_f - w_e) * scale, axis=axis, keepdim=True)  # type: ignore
            beta *= kappa

            current_error = float(torch.abs(w_f - w_r).mean())
            if current_error < best_error:
                best_error = current_error
            else:
                break

        del w_f, w_q, w_r, w_e

        return scale, zero

    @staticmethod
    def pack_on_row_fast_248bit(pack_tensor: torch.Tensor, ori_int_tensor: torch.Tensor, bits: int) -> None:
        if pack_tensor.shape[0] == ori_int_tensor.shape[0]:
            ori_int_tensor = ori_int_tensor.T
            pack_tensor = pack_tensor.T
        if bits in [2, 4, 8]:
            compress_ratio = pack_tensor.element_size() * 8 // bits
            for j in range(compress_ratio):
                pack_tensor[0:] |= ori_int_tensor[j::compress_ratio] << (bits * (j))
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")

    def quantize_internal(self,
                          tensor: torch.Tensor,
                          bits: int = 4,
                          channel_wise: bool = True,
                          group_size: int = 64,
                          optimize: bool = True,
                          round_zero: bool = True,
                          axis: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        weight = tensor.float()
        ori_shape = weight.shape

        pad_len = (group_size - ori_shape[axis] % group_size) % group_size
        weight = torch.nn.functional.pad(weight, (0, pad_len), "constant", 0)
        shape = weight.shape

        if (group_size is not None) and channel_wise:
            weight = weight.reshape([-1, group_size]) if (axis == 1) else weight.reshape([group_size, -1])

        _min = weight.min(axis=axis, keepdim=True)[0]  # type: ignore
        _max = weight.max(axis=axis, keepdim=True)[0]  # type: ignore

        max_v = 2**bits - 1
        min_v = 0
        min_max = [min_v, max_v]

        scale = (max_v / (_max - _min)).clamp(max=2e4)
        min_max_axis = _max - _min
        zero = -_min * scale

        if round_zero:
            zero = torch.round(zero)

        if optimize:
            scale, zero = self.optimize_weights(
                tensor=weight,  # type: ignore
                scale=scale,
                zero=zero,
                min_max=min_max,
                axis=axis)

        w_q = torch.round(weight * scale + zero).clamp(min_max[0], min_max[1])
        w_q = w_q.reshape(shape).int()

        scale = 1.0 / scale
        if axis == 1:
            scale = scale.reshape(shape[0], -1)
            zero = zero.reshape(shape[0], -1)
        else:
            scale = scale.reshape(-1, shape[-1])
            zero = zero.reshape(-1, shape[-1])
        del weight, _min, _max

        return w_q, scale.to(tensor.dtype), zero.to(tensor.dtype)

    def quantize(self, node: NodeProto, graph_stack: list[GraphProto]) -> NodeProto:
        """If the node is MatMul with fp32 const weight, quantize the weight with int4, and return the new node"""
        if node.op_type != "MatMul":
            return node

        logger.info(f"start to quantize {node.name} ...")
        inputB = node.input[1]
        b_pb, bs_graph = get_initializer(inputB, graph_stack)
        if b_pb is None:
            logger.info("MatMul doesn't have const weight. Skip to quantize")
            return node

        b_array = onnx.numpy_helper.to_array(b_pb)
        if len(b_array.shape) != 2:
            logger.info("MatMul weight is not 2D. Skip to quantize")
            return node
        b_array_torch = torch.from_numpy(b_array)
        if torch.cuda.is_available():
            b_array_torch = b_array_torch.cuda()
        quant_weight_torch, scales_torch, zero_points_torch = self.quantize_internal(b_array_torch.T,
                                                                                     bits=self.config.bits,
                                                                                     group_size=self.config.block_size)
        quant_weight_torch = quant_weight_torch.contiguous()
        scales_torch = scales_torch.contiguous()
        zero_points_torch = zero_points_torch.contiguous()

        packed_torch = torch.zeros(
            (quant_weight_torch.shape[0], quant_weight_torch.shape[1] // 2),
            dtype=torch.uint8,
            device=quant_weight_torch.device,
        )
        self.pack_on_row_fast_248bit(packed_torch, quant_weight_torch, self.config.bits)
        scales = scales_torch.cpu().numpy()
        zero_points = zero_points_torch.cpu().numpy()

        scales = scales.reshape(-1)
        zero_points = zero_points.reshape(-1)
        rows, cols = b_array_torch.shape
        block_size = self.config.block_size
        blob_size = block_size // 2
        k_blocks = (rows + block_size - 1) // block_size
        packed_torch = packed_torch.reshape(cols, k_blocks, blob_size)

        b_quant = onnx.numpy_helper.from_array(packed_torch.cpu().numpy())
        b_quant.name = b_pb.name + "_Q4"
        if bs_graph is not None:
            for input in bs_graph.input:
                if input.name == inputB:
                    bs_graph.input.remove(input)
                    break

        scales_tensor = onnx.numpy_helper.from_array(scales)
        scales_tensor.name = b_pb.name + "_scales"
        if bs_graph is not None:
            bs_graph.initializer.extend([b_quant, scales_tensor])

        input_names = [node.input[0], b_quant.name, scales_tensor.name]
        zp_tensor = onnx.numpy_helper.from_array(zero_points)
        zp_tensor.name = b_pb.name + "_zero_points"
        if bs_graph is not None:
            bs_graph.initializer.extend([zp_tensor])
        input_names.append(zp_tensor.name)

        kwargs = {}
        rows, cols = b_array.shape
        kwargs["K"] = rows
        kwargs["N"] = cols
        kwargs["bits"] = self.config.bits
        kwargs["block_size"] = self.config.block_size

        matmul_q4_node = onnx.helper.make_node(
            "MatMulNBits",
            inputs=input_names,
            outputs=[node.output[0]],
            name=node.name + "_Q4" if node.name else "",
            domain="com.microsoft",
            **kwargs,  # type: ignore
        )

        logger.info(f"complete quantization of {node.name} ...")

        return matmul_q4_node
