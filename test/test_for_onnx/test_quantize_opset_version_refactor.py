#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader

from quark.onnx import Int8Spec, ModelQuantizer, QConfig, QLayerConfig, XInt8Spec
from quark.shares.utils.testing_utils import use_temporary_directory

input_data = np.array(
    [
        [
            [
                [0.9296161, 0.31637555, 0.18391882, 0.20456028],
                [0.567725, 0.5955447, 0.9645145, 0.6531771],
                [0.7489066, 0.6535699, 0.7477148, 0.96130675],
                [0.0083883, 0.10644437, 0.2987037, 0.6564112],
            ],
            [
                [0.80981255, 0.87217593, 0.9646476, 0.7236853],
                [0.6424753, 0.7174536, 0.467599, 0.32558468],
                [0.4396446, 0.72968906, 0.99401456, 0.6768737],
                [0.7908225, 0.17091426, 0.02684928, 0.8003702],
            ],
            [
                [0.9037225, 0.02467621, 0.49174732, 0.5262552],
                [0.596366, 0.05195754, 0.8950895, 0.7282662],
                [0.81835, 0.50022274, 0.8101894, 0.09596852],
                [0.21895005, 0.25871906, 0.46810576, 0.4593732],
            ],
        ]
    ]
).astype(np.float32)

fuse_gelu_input_data = np.array([1.0], dtype=np.float32)

fuse_layer_norm_golden_output = np.array(
    [
        [
            0.13433924,
            0.6985641,
            -0.7321489,
            -0.8127524,
            0.23509368,
            0.43660256,
            0.12090532,
            0.900073,
            -0.11418836,
            -0.7321489,
        ]
    ]
).astype(np.float32)

fuse_gelu_golden_output = np.array([1.5], dtype=np.float32)


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


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, emb_size), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Fuse_Layer_Norm_Model(nn.Module):
    def __init__(self, img_size=4, patch_size=2, in_channels=3, emb_size=64, num_heads=4, mlp_dim=128, num_classes=10):
        super(Fuse_Layer_Norm_Model, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, emb_size))
        self.transformer = TransformerBlock(emb_size, num_heads, mlp_dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(emb_size), nn.Linear(emb_size, num_classes))

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0])
        return x


class Fuse_Gelu_Model(nn.Module):
    def __init__(self):
        super(Fuse_Gelu_Model, self).__init__()
        self.fc = nn.Linear(1, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        return self.gelu(x)


def prepare_model(input_shape, model_type, output_dir: str):
    torch.manual_seed(42)
    onnx_model_path = Path(output_dir, model_type + ".onnx").as_posix()
    quant_onnx_model_path = Path(output_dir, model_type + "_quantized.onnx").as_posix()

    if model_type == "fuse_layer_norm":
        opset_version = 13
        model = Fuse_Layer_Norm_Model()
        onnx_model_path = Path(output_dir, "vit_model_opset_13.onnx").as_posix()
        quant_onnx_model_path = Path(output_dir, "vit_model_opset_17_quantized.onnx").as_posix()

    if model_type == "fuse_gelu":
        opset_version = 17
        model = Fuse_Gelu_Model()
        onnx_model_path = Path(output_dir, "gelu_opset_17.onnx").as_posix()
        quant_onnx_model_path = Path(output_dir, "gelu_opset_20_quantized.onnx").as_posix()

    dummy_input = torch.randn(input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        opset_version=opset_version,
        dynamo=False,
    )

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, quant_onnx_model_path


def prepare_config():
    quant_config = QConfig(
        global_config=QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()), ConvertOpsetVersion=20
    )
    return quant_config


def prepare_config_fuse_layer_norm():
    quant_config = QConfig(
        global_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec()),
        SimplifyModel=False,
        ConvertOpsetVersion=17,
        OpTypesToQuantize=["MatMul", "Gemm"],
        OptimizeModel=False,
    )
    return quant_config


def prepare_data(input_tensor):
    data_reader = DataReader(input_tensor)
    return data_reader


def prepare_quantizer(quant_config):
    quantizer = ModelQuantizer(quant_config)
    return quantizer


def quantize_static(quantizer, input_model_path, output_model_path, data_reader):
    quantizer.quantize_model(input_model_path, output_model_path, data_reader)
    print("Quantized the ONNX model and saved it at:", output_model_path)
    return output_model_path


def infer_quantized_model(input_data, quantized_model_path):
    # Disabling ORT Graph Optimization to achieve reproducible golden numbers across different servers
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(quantized_model_path, sess_options=so)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def contains_op_type(model_path, target_op_type):
    model = onnx.load(model_path)
    return any(node.op_type == target_op_type for node in model.graph.node)


def tensor_quantize(input_data, input_shape, model_type, output_dir: str):
    data_reader = prepare_data(input_data)
    input_model_path, output_model_path = prepare_model(input_shape, model_type=model_type, output_dir=output_dir)
    if model_type == "fuse_layer_norm":
        quant_config = prepare_config_fuse_layer_norm()
    else:
        quant_config = prepare_config()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(input_data, quantized_model_path)
    return output, quantized_model_path


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_optimize_fuse_layer_norm(self, tmpdir: str):
        output, quantized_model_path = tensor_quantize(input_data, [1, 3, 4, 4], "fuse_layer_norm", tmpdir)
        has_target_op_type = contains_op_type(quantized_model_path, "LayerNormalization")
        comp_equal = np.allclose(output, fuse_layer_norm_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)
        self.assertEqual(has_target_op_type, True)

    @use_temporary_directory
    def test_optimize_fuse_gelu(self, tmpdir: str):
        input_shape = [1]
        input_data = np.resize(fuse_gelu_input_data, input_shape)
        output, quantized_model_path = tensor_quantize(input_data, input_shape, "fuse_gelu", tmpdir)
        contains_gelu = contains_op_type(quantized_model_path, "Gelu")
        comp_equal = np.allclose(output, fuse_gelu_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)
        self.assertEqual(contains_gelu, True)


if __name__ == "__main__":
    unittest.main()
