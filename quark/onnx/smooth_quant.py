#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
import onnx
import onnxruntime
import numpy as np
from onnx import numpy_helper, helper, onnx_pb
from onnxruntime.transformers.onnx_model import OnnxModel
from tqdm.auto import tqdm
from collections import OrderedDict
from typing import List, Dict, Tuple, Any
import os


class SmoothQuant():
    """
    A class for model smooth
    Args:
        onnx_model_path (str): The ONNX model path to be smoothed.
        input_model (onnx.ModelProto): The ONNX model to be smoothed.
        dataloader (torch.utils.data.DataLoader): The dataloader used for calibrate.
        alpha (float): The extent to which the difficulty of quantification is shifted from activation to weighting.
        is_large (bool): True if the model size is larger than 2GB.
    """

    def __init__(
            self,
            onnx_model_path: str,
            input_model: onnx.ModelProto,
            dataloader: torch.utils.data.DataLoader,  # type:ignore
            alpha: float,
            is_large: bool = True,
            providers: List[str] = ["CPUExecutionProvider"]):
        self.onnx_model_path = onnx_model_path
        self.dataloader = dataloader
        self.is_large = is_large
        self.alpha = alpha
        self.providers = providers
        self.base_dir = os.path.dirname(self.onnx_model_path)
        self.smoothed_model_path = os.path.join(self.base_dir, "decoder_model_smoothed.onnx")
        self.tmp_model_path = os.path.join(self.base_dir, "decoder_model_tmp.onnx")

        self.model = input_model

        self.onnx_model = OnnxModel(self.model)

        self.output_num = len(self.onnx_model.get_graphs_output_names())

        self.linear_dic: Dict[str, List[onnx.NodeProto]] = {}
        self.ln_outputs: List[str] = []
        self.act_scales: Dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
        self.extend_output_nodes: List[str] = []
        self.smooth_nodes: List[str] = []

    def match_matmul_output(self) -> None:
        matmul_node_list = self.onnx_model.get_nodes_by_op_type("MatMul")
        smooth_matmul_node_list = []
        # determine whether matmul op has parameters
        for node in matmul_node_list:
            for init in self.onnx_model.model.graph.initializer:
                if init.name == node.input[1]:
                    smooth_matmul_node_list.append(node)
        for node in smooth_matmul_node_list:
            if node.input[0] not in self.ln_outputs:
                self.ln_outputs.append(node.input[0])
                self.model.graph.output.extend([onnx.ValueInfoProto(name=node.input[0])])
                self.extend_output_nodes.append(node.input[0])
            if node.input[0] not in self.linear_dic:
                self.linear_dic[node.input[0]] = [node]
            else:
                self.linear_dic[node.input[0]].append(node)

    def get_act_scale(self) -> None:
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.is_large:
            self.onnx_model.save_model_to_file(self.tmp_model_path,
                                               use_external_data_format=True,
                                               all_tensors_to_one_file=True)
            session = onnxruntime.InferenceSession(self.tmp_model_path, sess_options, providers=self.providers)
        else:
            session = onnxruntime.InferenceSession(self.model.SerializeToString(),
                                                   sess_options,
                                                   providers=self.providers)

        # calculate act_scale
        for inputs in tqdm(self.dataloader):
            inputs_dict = inputs
            ort_outs = session.run(self.ln_outputs, inputs_dict)
            out_dict = OrderedDict(zip(self.ln_outputs, ort_outs))

            for output in self.ln_outputs:
                hidden_dim = out_dict[output].shape[-1]
                tensor = np.absolute(out_dict[output].reshape(-1, hidden_dim))
                comming_max = np.max(tensor, axis=0)
                if output in self.act_scales:
                    self.act_scales[output] = np.where(self.act_scales[output] > comming_max, self.act_scales[output],
                                                       comming_max)
                else:
                    self.act_scales[output] = comming_max

    def get_initializer_tensor(self, init_name: str) -> Tuple[onnx.TensorProto, np.ndarray[Any, np.dtype[np.float32]]]:
        weight_tensor_proto = [init for init in self.onnx_model.model.graph.initializer if init.name == init_name][0]
        weight_tensor = numpy_helper.to_array(weight_tensor_proto, self.base_dir)
        return weight_tensor_proto, weight_tensor

    def smooth_ln_linear(self) -> None:
        for output in self.ln_outputs:
            linear = self.linear_dic[output]
            act_scale = self.act_scales[output]

            # calculate weight scale
            # linear in attention
            for node in linear:
                linear_weight_init, linear_weight = self.get_initializer_tensor(node.input[1])
                weight_scale = np.max(abs(linear_weight), axis=1)
                scale = np.power(act_scale, self.alpha) / np.power(weight_scale + 1e-9, (1 - self.alpha))
                self.insert_smooth_mul_op(scale, output, node)
                linear_weight = np.multiply(scale.reshape(-1, 1), linear_weight)
                linear_weight_init.CopyFrom(numpy_helper.from_array(linear_weight, linear_weight_init.name))

    def insert_smooth_mul_op(self, scale: np.ndarray[Any, np.dtype[np.float32]], input_name: str,
                             node: onnx.NodeProto) -> None:
        scale_factor = 1.0 / (scale + 1e-9)

        scale_tensor = helper.make_tensor(name=input_name + "_" + node.name + "_" + "smooth_scale",
                                          data_type=onnx_pb.TensorProto.FLOAT,
                                          dims=scale_factor.shape,
                                          vals=scale_factor.flatten().tolist())

        self.mul_output_name = input_name + "_" + node.name + "_smooth_output"
        mul_node = helper.make_node("Mul",
                                    inputs=[input_name, input_name + "_" + node.name + "_" + "smooth_scale"],
                                    outputs=[self.mul_output_name],
                                    name=input_name + "_" + node.name + "_smooth_mul")
        self.smooth_nodes.append(mul_node.name)

        self.onnx_model.add_node(mul_node)
        self.onnx_model.add_initializer(scale_tensor)
        self.onnx_model.remove_node(node)
        self.onnx_model.replace_node_input(node, node.input[0], self.mul_output_name)
        self.onnx_model.add_node(node)

    def remove_extend_output_node(self) -> None:
        for node in self.extend_output_nodes:
            if onnx.ValueInfoProto(name=node) in self.model.graph.output:
                self.model.graph.output.remove(onnx.ValueInfoProto(name=node))

    def transform(self) -> None:
        self.match_matmul_output()
        self.get_act_scale()
        self.smooth_ln_linear()
        self.remove_extend_output_node()

        self.onnx_model.save_model_to_file(self.smoothed_model_path,
                                           use_external_data_format=self.is_large,
                                           all_tensors_to_one_file=True)

    def get_smooth_node(self) -> List[str]:
        return self.smooth_nodes

    def get_smooth_path(self) -> str:
        return self.smoothed_model_path

    def get_smooth_model(self) -> onnx.ModelProto:
        return self.onnx_model.model  # type:ignore


def smooth_transforms(
        onnx_model_path: str,
        input_model: onnx.ModelProto,
        dataloader: torch.utils.data.DataLoader,  # type:ignore
        alpha: float = 0.5) -> onnx.ModelProto:
    smooth_ = SmoothQuant(onnx_model_path, input_model, dataloader, alpha=alpha)
    smooth_.transform()
    return smooth_.get_smooth_model()
