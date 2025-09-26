#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import json
import unittest

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader

from quark.onnx import ModelQuantizer
from quark.onnx.quant_utils import is_version_below
from quark.onnx.quantization import QuarotConfig
from quark.onnx.quantization.config.config import QConfig
from quark.onnx.quantization.config.spec import Int8Spec, QLayerConfig

input_tensor = np.array(
    [
        [-0.36716485, 0.01129738, 0.07908165, 0.35021877, -0.6665454, 0.13556856, -0.01129738, 0.02259476],
        [0.20900153, -0.09037904, -0.15251462, -0.15816331, -0.14686593, 0.6947889, 0.53097683, -0.17510939],
    ]
).astype(np.float32)

output_tensor_correct = np.array(
    [
        [-1.2185992, 0.781493, 0.47684318, 1.4835122, -0.6490366, 1.4702665, -0.27815852, 0.49008882],
        [-1.6821967, 0.6755279, -0.01324564, 1.2583362, -0.37087804, 1.2185992, -0.70201916, 0.31789547],
    ]
).astype(np.float32)

output_tensor_whole = np.array(
    [
        [0.1373996, 0.14839157, 0.2802952, -0.08243977, 0.14839157, 0.21434338, -0.70348597, 0.45616668],
        [0.12091165, 0.15938354, 0.21434338, -0.11541566, 0.13190362, 0.26930323, -0.64852613, 0.5001345],
    ]
).astype(np.float32)

mlp_dim = 16


class CorrectDummyModel(nn.Module):
    def __init__(self, emb_size, mlp_dim):
        super(CorrectDummyModel, self).__init__()
        self.layer1 = nn.Linear(emb_size, mlp_dim)  # From definition, in_feat, out_feat. Weight, out_feat, in_feat
        self.layer2 = nn.Linear(mlp_dim, emb_size, bias=False)
        self.layer3 = nn.Linear(emb_size, mlp_dim, bias=False)
        self.layer4 = nn.Linear(mlp_dim, emb_size)
        self.tmp_factors = nn.Parameter(torch.randn(mlp_dim))

        self._initialize_params()

    def _initialize_params(self):
        for name, param in self.named_parameters():
            nn.init.normal_(param, mean=0.0, std=0.2)

    def forward(self, x):
        x1 = self.layer1(x)
        t = x1 * 10
        x2 = self.layer2(t)
        x3 = self.layer3(x2) + x1
        x4 = self.layer4(x3)
        return x4


class WholeDummyModel(nn.Module):
    def __init__(self, emb_size, mlp_dim):
        super(WholeDummyModel, self).__init__()
        self.layer1 = nn.Linear(emb_size, mlp_dim)  # From definition, in_feat, out_feat. Weight, out_feat, in_feat
        # self.norm1 = nn.LayerNorm(mlp_dim, bias=False)
        self.norm1 = nn.LayerNorm(mlp_dim, bias=True)
        self.layer2 = nn.Linear(mlp_dim, emb_size, bias=False)
        self.layer3 = nn.Linear(emb_size, mlp_dim, bias=False)
        self.norm2 = nn.LayerNorm(mlp_dim, bias=True)
        self.layer4 = nn.Linear(mlp_dim, emb_size)
        self.tmp_factors = nn.Parameter(torch.randn(mlp_dim))

        self._initialize_params()

    def _initialize_params(self):
        for name, param in self.named_parameters():
            nn.init.normal_(param, mean=0.0, std=0.2)

    def forward(self, x):
        x1 = self.layer1(x)
        t1 = self.norm1(x1)
        x2 = self.layer2(t1)
        x3 = self.layer3(x2) + x1
        t2 = self.norm2(x3)
        x4 = self.layer4(t2)
        return x4


class DataReader(CalibrationDataReader):
    def __init__(self, input_tensor):
        self.data = [input_tensor]
        self.input_name = "input"
        self.index = 0

    def get_next(self):
        if self.index < len(self.data):
            input_dict = {self.input_name: self.data[self.index]}
            self.index += 1
            return input_dict
        else:
            return None

    def rewind(self):
        self.index = 0


def prepare_model(gen_type="correct"):
    torch.manual_seed(42)
    model_classes = {"correct": CorrectDummyModel, "whole": WholeDummyModel}

    emb_size = 8
    ModelClass = model_classes.get(gen_type, WholeDummyModel)
    model = ModelClass(emb_size=emb_size, mlp_dim=mlp_dim)

    dummy_input = torch.randn(2, emb_size)

    onnx_model_path = "dm_model.onnx"
    torch.onnx.export(
        model, dummy_input, onnx_model_path, input_names=["input"], output_names=["output"], opset_version=17
    )

    # Further remove intializers from input. Because we need a formal name of each intializer.
    sim_model_path = "dm_simplified_model.onnx"
    fast_rename_initializers(onnx_model_path, sim_model_path)

    return sim_model_path, "dm_quantized_model.onnx"


def fast_rename_initializers(model_input, sim_model_path):
    model = onnx.load(model_input)

    keywords = ["MatMul", "Mul"]
    for initializer in model.graph.initializer:
        if any(keyword in initializer.name for keyword in keywords):
            old_name = initializer.name
            initializer.name = old_name + ".weight"
            # Refactor corresponding node input
            for node in model.graph.node:
                for i, input_name in enumerate(node.input):
                    if input_name == old_name:
                        node.input[i] = initializer.name
                for i, output_name in enumerate(node.output):
                    if output_name == old_name:
                        node.output[i] = initializer.name
    onnx.save_model(model, sim_model_path)
    print(f"Model has been saved to {sim_model_path}")


def dump_roatation_config(rconfig_path, gen_type="correct"):
    if gen_type == "correct":
        data = {
            "R1_pairs": [
                {"prev_nodes": ["/layer1/Gemm"], "next_nodes": ["/layer2/MatMul"]},
                {"prev_nodes": ["/layer3/MatMul"], "next_nodes": ["/layer4/Gemm"]},
            ]
        }
    elif gen_type == "whole":
        data = {
            "R1_pairs": [
                {
                    "prev_nodes": ["/layer1/Gemm"],
                    "norm_node": "/norm1/LayerNormalization",
                    "next_nodes": ["/layer2/MatMul"],
                },
                {
                    "prev_nodes": ["/layer3/MatMul"],
                    "norm_node": "/norm2/LayerNormalization",
                    "next_nodes": ["/layer4/Gemm"],
                },
            ]
        }
    elif gen_type == "wrong_node_name":
        data = {"R1_pairs": [{"prev_nodes": ["/fake"]}]}
    elif gen_type == "wrong_node_type":
        data = {"R1_pairs": [{"prev_nodes": ["/Add"]}]}
    else:
        raise NotImplementedError

    # Dump json
    with open(rconfig_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON file '{rconfig_path}' has been create.")


def prepare_config(gen_type="correct", use_rconfig=True, use_random_had=False):
    rconfig_path = "dm_tmp_rconfig.json"
    dump_roatation_config(rconfig_path, gen_type)
    if use_rconfig:
        quarot_algo = [QuarotConfig(r_matrix_dim=mlp_dim, use_random_had=use_random_had, r_config_path=rconfig_path)]
    else:
        quarot_algo = [QuarotConfig(r_matrix_dim=mlp_dim, use_random_had=use_random_had)]
    quant_config = QConfig(
        global_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec()),
        layer_type_config={None: ["LayerNormalization", "Mul", "Add"]},
        algo_config=quarot_algo,
        ActivationSymmetric=True,
        SimplifyModel=False,
        OptimizeModel=False,
    )

    return quant_config


def prepare_data():
    data_reader = DataReader(input_tensor)
    return data_reader


def prepare_quantizer(quant_config):
    quantizer = ModelQuantizer(quant_config)
    return quantizer


def quantize_static(quantizer, input_model_path, output_model_path, data_reader):
    quantizer.quantize_model(input_model_path, output_model_path, data_reader)
    print("Quantized the ONNX model and saved it at:", output_model_path)
    return output_model_path


def infer_quantized_model(quantized_model_path):
    sess = onnxruntime.InferenceSession(quantized_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize_rotation(gen_type="correct", use_rconfig=True, use_random_had=False):
    input_model_path, output_model_path = prepare_model(gen_type)
    data_reader = prepare_data()
    quant_config = prepare_config(gen_type, use_rconfig, use_random_had)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_rotation_abnormal1():
    input_model_path, output_model_path = prepare_model()
    data_reader = prepare_data()
    quant_config = prepare_config(use_rconfig=False)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_rotation_abnormal2():
    input_model_path, output_model_path = prepare_model()
    data_reader = prepare_data()
    quant_config = prepare_config(use_random_had=True)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_rotation_abnormal3():
    input_model_path, output_model_path = prepare_model()
    data_reader = prepare_data()
    quant_config = prepare_config(gen_type="wrong_node_name")
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    def test_quantize_correct(self):
        if not is_version_below(onnxruntime, "1.18.0"):
            torch.manual_seed(42)
            output = tensor_quantize_rotation("correct")
            comp_equal = np.allclose(output, output_tensor_correct, atol=1e-1)
            self.assertEqual(comp_equal, True)

    def test_quantize_whole(self):
        if not is_version_below(onnxruntime, "1.18.0"):
            torch.manual_seed(42)
            output = tensor_quantize_rotation("whole")
            comp_equal = np.allclose(output, output_tensor_whole, atol=1e-1)
            self.assertEqual(comp_equal, True)

    def test_quantize_abnormal(self):
        if not is_version_below(onnxruntime, "1.18.0"):
            # Trigger no rotation config
            torch.manual_seed(42)
            with self.assertRaises(AssertionError) as context:
                output = tensor_quantize_rotation(use_rconfig=False)
            self.assertIn("Error! Please specify the rotation config", str(context.exception))
            # Trigger random hadamard assertion
            with self.assertRaises(AssertionError) as context:
                output = tensor_quantize_rotation(use_random_had=True)
            self.assertIn("Error! The dim of the target R1 matrix is not support", str(context.exception))
            # Trigger wrong node name
            with self.assertRaises(ValueError) as context:
                output = tensor_quantize_rotation("wrong_node_name")
            self.assertIn("Can not get the corresponding node!", str(context.exception))
            # Trigger wrong node type
            with self.assertRaises(ValueError) as context:
                output = tensor_quantize_rotation("wrong_node_type")
            self.assertIn('do not have a input which has a name include "weight"!', str(context.exception))


if __name__ == "__main__":
    unittest.main()
