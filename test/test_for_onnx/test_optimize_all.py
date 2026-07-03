#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import Config, ModelQuantizer
from quark.onnx.quantization.config.custom_config import INT8_TRANSFORMER_DEFAULT_CONFIG, XINT8_CONFIG

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

split_large_kernel_pool_init_data = np.array(
    [
        [
            [
                [0.9296161, 0.31637555, 0.18391882, 0.20456028, 0.567725],
                [0.5955447, 0.9645145, 0.6531771, 0.7489066, 0.6535699],
                [0.7477148, 0.96130675, 0.0083883, 0.10644437, 0.2987037],
                [0.6564112, 0.80981255, 0.87217593, 0.9646476, 0.7236853],
                [0.6424753, 0.7174536, 0.467599, 0.32558468, 0.4396446],
            ],
            [
                [0.72968906, 0.99401456, 0.6768737, 0.7908225, 0.17091426],
                [0.02684928, 0.8003702, 0.9037225, 0.02467621, 0.49174732],
                [0.5262552, 0.596366, 0.05195754, 0.8950895, 0.7282662],
                [0.81835, 0.50022274, 0.8101894, 0.09596852, 0.21895005],
                [0.25871906, 0.46810576, 0.4593732, 0.7095098, 0.178053],
            ],
            [
                [0.5314499, 0.16774222, 0.7688139, 0.92817056, 0.6094937],
                [0.1501835, 0.4896267, 0.37734497, 0.8486014, 0.9110972],
                [0.38384873, 0.3154959, 0.5683941, 0.18781804, 0.12584154],
                [0.6875958, 0.79960674, 0.5735366, 0.97323, 0.63405436],
                [0.8884217, 0.49541476, 0.35161653, 0.71423036, 0.50392914],
            ],
        ]
    ]
).astype(np.float32)

fuse_gelu_input_data = np.array([1.0], dtype=np.float32)

split_large_kernel_pool_with_input_shape_23x23_golden_output = np.array(
    [
        [
            [[0.28125]],
            [[0.2265625]],
            [[-0.1875]],
            [[-0.328125]],
            [[-0.625]],
            [[0.015625]],
            [[-0.09375]],
            [[-0.0078125]],
            [[-0.3046875]],
            [[-0.171875]],
            [[-0.421875]],
            [[0.34375]],
            [[-0.5859375]],
            [[-0.140625]],
            [[-0.046875]],
            [[-0.1953125]],
        ]
    ]
).astype(np.float32)

split_large_kernel_pool_with_input_shape_25x25_golden_output = np.array(
    [
        [
            [[0.28125]],
            [[0.2265625]],
            [[-0.1875]],
            [[-0.328125]],
            [[-0.625]],
            [[0.015625]],
            [[-0.09375]],
            [[-0.0078125]],
            [[-0.3046875]],
            [[-0.171875]],
            [[-0.421875]],
            [[0.34375]],
            [[-0.5859375]],
            [[-0.140625]],
            [[-0.046875]],
            [[-0.1953125]],
        ]
    ]
).astype(np.float32)

fold_batch_norm_after_concat_golden_output = np.array(
    [
        [
            -0.09765625,
            0.28515625,
            0.125,
            -0.2421875,
            -0.0078125,
            0.109375,
            0.14453125,
            -0.296875,
            0.06640625,
            -0.0234375,
            -0.03125,
            0.04296875,
            0.03515625,
            -0.03125,
            0.015625,
            0.0703125,
        ]
    ]
)

fold_batch_norm_golden_output = np.array(
    [[0.08984375, 0.37109375, 0.03515625, -0.05078125, -0.01171875, 0.078125]]
).astype(np.float32)

convert_split_to_slice_golden_output = np.array(
    [
        [
            [
                [0.2890625, 0.6171875, 0.453125, 0.0],
                [0.625, 0.5859375, 0.6015625, 0.203125],
                [0.4609375, 0.4453125, 0.53125, 0.5234375],
                [0.296875, 0.375, 0.453125, 0.2265625],
            ],
            [
                [0.0078125, 0.5703125, 0.125, -0.046875],
                [0.3359375, 0.3984375, 0.2890625, 0.359375],
                [0.1015625, 0.0703125, 0.3203125, 0.0703125],
                [0.1328125, -0.0859375, -0.0234375, 0.0625],
            ],
        ]
    ]
).astype(np.float32)

convert_reduce_mean_to_global_avg_pool_golden_output = np.array([[[[0.5390625]], [[0.6328125]], [[0.4921875]]]]).astype(
    np.float32
)

convert_bn_to_conv_golden_output = np.array(
    [
        [
            [
                [-0.0078125, -0.0078125, 0.08203125, 0.08984375],
                [-0.16015625, -0.1484375, -0.07421875, -0.06640625],
                [-0.1171875, -0.19140625, -0.16796875, -0.12890625],
                [-0.0234375, -0.06640625, -0.0859375, -0.015625],
            ],
            [
                [-0.15234375, -0.26171875, -0.3984375, -0.3203125],
                [-0.3203125, -0.125, -0.2109375, -0.29296875],
                [-0.1484375, -0.3203125, -0.3671875, -0.32421875],
                [-0.23828125, -0.1796875, -0.37109375, -0.3671875],
            ],
        ]
    ]
).astype(np.float32)

convert_clip_to_relu_golden_output = np.array(
    [
        [
            [
                [0.2890625, 0.6171875, 0.453125, 0.0],
                [0.625, 0.5859375, 0.6015625, 0.203125],
                [0.4609375, 0.4453125, 0.53125, 0.5234375],
                [0.296875, 0.375, 0.453125, 0.2265625],
            ],
            [
                [0.0078125, 0.5703125, 0.125, 0.0],
                [0.3359375, 0.3984375, 0.2890625, 0.359375],
                [0.1015625, 0.0703125, 0.3203125, 0.0703125],
                [0.1328125, 0.0, 0.0, 0.0625],
            ],
        ]
    ]
).astype(np.float32)

fuse_layer_norm_golden_output = np.array(
    [
        [
            0.13433924,
            0.69856405,
            -0.7321488,
            -0.8127524,
            0.23509367,
            0.43660253,
            0.12090532,
            0.90007293,
            -0.11418835,
            -0.7321488,
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


class Convert_Bn_To_Conv_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv_transpose1 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.conv_transpose2 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x1, x2 = torch.split(x, [1, 1], 1)
        x1 = self.conv_transpose1(x1)
        x2 = self.conv_transpose2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.bn3(x)
        return x


class Convert_Clip_To_Relu_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.clip(x, 0, 1)
        return x


class Convert_Reduce_Mean_To_Global_Avg_Pool_Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        return x


class Fold_Batch_Norm_After_Concat_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(6)

        self.conv_transpose1 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=0)
        self.conv_transpose2 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=0)
        self.conv_transpose3 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)

        self.conv1 = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(4)

        self.gemm1 = nn.Linear(8, 8)
        self.gemm2 = nn.Linear(8, 8, bias=False)
        self.bn3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x1, x2, x3 = torch.split(x, [2, 2, 2], 1)
        x1 = self.conv_transpose1(x1)
        x2 = self.conv_transpose2(x2)
        x3 = self.conv_transpose3(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn1(x)
        x1, x2 = torch.split(x, [3, 3], 1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.bn2(x)
        x1, x2 = torch.split(x, [2, 2], 1)
        x1 = x1.reshape(x1.size(0), -1)
        x2 = x2.reshape(x2.size(0), -1)
        x1 = self.gemm1(x1)
        x2 = self.gemm2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.bn3(x)
        return x


class ConvBN_ConvTransposeBN_GemmBN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv_transpose = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1, padding=0)
        self.gemm = nn.Linear(108, 6)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)
        self.bn3 = nn.BatchNorm1d(6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.conv_transpose(x)
        x = self.bn2(x)
        x = x.reshape(x.size(0), -1)
        x = self.gemm(x)
        x = self.bn3(x)
        return x


class Split_Large_Kernel_Pool_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class Convert_Split_To_Slice_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x1, x2 = torch.split(x, [1, 1], 1)
        x = torch.cat([x1, x2], dim=1)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.fc = nn.Linear(1, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        return self.gelu(x)


def prepare_model(input_shape, model_type, output_dir: str):
    torch.manual_seed(42)
    opset_version = 17

    onnx_model_path = Path(output_dir, model_type + ".onnx").as_posix()
    quant_onnx_model_path = Path(output_dir, model_type + "_quantized.onnx").as_posix()

    if model_type == "fold_batch_norm_after_concat":
        model = Fold_Batch_Norm_After_Concat_Model()
    elif model_type == "fold_batch_norm":
        model = ConvBN_ConvTransposeBN_GemmBN_Model()
    elif model_type == "split_large_kernel_pool":
        model = Split_Large_Kernel_Pool_Model()
    elif model_type == "convert_split_to_slice":
        model = Convert_Split_To_Slice_Model()
    elif model_type == "convert_reduce_mean_to_global_avg_pool":
        model = Convert_Reduce_Mean_To_Global_Avg_Pool_Model()
    elif model_type == "convert_bn_to_conv":
        model = Convert_Bn_To_Conv_Model()
    elif model_type == "convert_clip_to_relu":
        model = Convert_Clip_To_Relu_Model()

    if model_type in ["fuse_layer_norm", "fuse_layer_norm_opset_17"]:
        opset_version = 13  # For opset 13, the exported model includes a scattered LayerNorm op
        model = Fuse_Layer_Norm_Model()
        onnx_model_path = Path(output_dir, "vit_model_opset_13.onnx").as_posix()
        quant_onnx_model_path = Path(output_dir, "vit_model_opset_13_quantized.onnx").as_posix()

    if model_type == "fuse_gelu":
        opset_version = 17
        model = Fuse_Gelu_Model()
        onnx_model_path = Path(output_dir, "gelu_opset_17.onnx").as_posix()
        onnx_model_update_opset_path = Path(output_dir, "gelu_opset_20.onnx").as_posix()
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

    if model_type == "fuse_layer_norm_opset_17":
        model = onnx.load(onnx_model_path)
        model.opset_import[0].version = 17  # Updating the opset to 17 for compatibility with the quantization process
        onnx_model_path = Path(output_dir, "vit_model_opset_17.onnx").as_posix()
        onnx.save(model, onnx_model_path)
        quant_onnx_model_path = Path(output_dir, "vit_model_opset_17_quantized.onnx").as_posix()
        print(f"Model has been saved to {onnx_model_path}")

    if model_type == "fuse_gelu":
        model = onnx.load(onnx_model_path)
        model.opset_import[0].version = 20
        onnx.save(model, onnx_model_update_opset_path)
        onnx_model_path = onnx_model_update_opset_path
        print(f"Model has been saved to {onnx_model_update_opset_path}")

    # Note: this part cannot change the original model
    if model_type == "fold_batch_norm_after_concat":
        onnx_model = onnx.load(onnx_model_path)
        for node in onnx_model.graph.node:
            if node.op_type == "MatMul":
                node.op_type = "Gemm"
        onnx.save_model(onnx_model, onnx_model_path)
    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, quant_onnx_model_path


def prepare_config(config):
    quant_config = Config(global_quant_config=config)
    return quant_config


def prepare_config_convert_bn_to_conv(config):
    config_copy = copy.deepcopy(config)
    config_copy.optimize_model = False
    config_copy.extra_options["SimplifyModel"] = False
    config_copy.extra_options["FoldBatchNorm"] = True
    config_copy.extra_options["ConvertBNToConv"] = True
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_config_convert_clip_to_relu(config):
    config_copy = copy.deepcopy(config)
    config_copy.extra_options["ConvertClipToRelu"] = True
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_config_fuse_layer_norm(config):
    config_copy = copy.deepcopy(config)
    config_copy.extra_options["SimplifyModel"] = False
    quant_config = Config(global_quant_config=config_copy)
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
    if model_type == "convert_bn_to_conv":
        quant_config = prepare_config_convert_bn_to_conv(XINT8_CONFIG)
    elif model_type == "convert_clip_to_relu":
        quant_config = prepare_config_convert_clip_to_relu(XINT8_CONFIG)
    elif model_type in ["fuse_layer_norm", "fuse_layer_norm_opset_17"]:
        quant_config = prepare_config_fuse_layer_norm(INT8_TRANSFORMER_DEFAULT_CONFIG)
    else:
        quant_config = prepare_config(XINT8_CONFIG)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(input_data, quantized_model_path)
    return output, quantized_model_path


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_optimize_convert_bn_to_conv(self, tmpdir: str):
        output, quantized_model_path = tensor_quantize(input_data, [1, 3, 4, 4], "convert_bn_to_conv", tmpdir)
        has_target_op_type = contains_op_type(quantized_model_path, "BatchNormalization")

        comp_equal = np.allclose(output, convert_bn_to_conv_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)
        self.assertEqual(has_target_op_type, False)

    @use_temporary_directory
    def test_optimize_convert_clip_to_relu(self, tmpdir: str):
        output, quantized_model_path = tensor_quantize(input_data, [1, 3, 4, 4], "convert_clip_to_relu", tmpdir)
        has_target_op_type = contains_op_type(quantized_model_path, "Relu")

        comp_equal = np.allclose(output, convert_clip_to_relu_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)
        self.assertEqual(has_target_op_type, True)

    @use_temporary_directory
    def test_optimize_convert_reduce_mean_to_global_avg_pool(self, tmpdir: str):
        output, quantized_model_path = tensor_quantize(
            input_data, [1, 3, 4, 4], "convert_reduce_mean_to_global_avg_pool", tmpdir
        )
        contains_average_pool = contains_op_type(quantized_model_path, "GlobalAveragePool")

        comp_equal = np.allclose(output, convert_reduce_mean_to_global_avg_pool_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)
        self.assertEqual(contains_average_pool, True)

    @use_temporary_directory
    def test_optimize_convert_split_to_slice(self, tmpdir: str):
        output, quantized_model_path = tensor_quantize(input_data, [1, 3, 4, 4], "convert_split_to_slice", tmpdir)
        has_target_op_type = contains_op_type(quantized_model_path, "Slice")

        comp_equal = np.allclose(output, convert_split_to_slice_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)
        self.assertEqual(has_target_op_type, True)

    @use_temporary_directory
    def test_optimize_fold_batch_norm(self, tmpdir: str):
        output, quantized_model_path = tensor_quantize(input_data, [1, 3, 4, 4], "fold_batch_norm", tmpdir)
        has_target_op_type = contains_op_type(quantized_model_path, "BatchNormalization")

        comp_equal = np.allclose(output, fold_batch_norm_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)
        self.assertEqual(has_target_op_type, False)

    @use_temporary_directory
    def test_optimize_fold_batch_norm_after_concat(self, tmpdir: str):
        output, quantized_model_path = tensor_quantize(input_data, [1, 3, 4, 4], "fold_batch_norm_after_concat", tmpdir)
        has_target_op_type = contains_op_type(quantized_model_path, "BatchNormalization")

        comp_equal = np.allclose(output, fold_batch_norm_after_concat_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)
        self.assertEqual(has_target_op_type, False)

    @use_temporary_directory
    def test_optimize_fuse_layer_norm(self, tmpdir: str):
        output, quantized_model_path = tensor_quantize(input_data, [1, 3, 4, 4], "fuse_layer_norm", tmpdir)
        has_target_op_type = contains_op_type(quantized_model_path, "LayerNormalization")

        comp_equal = np.allclose(output, fuse_layer_norm_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)
        self.assertEqual(has_target_op_type, False)

    @use_temporary_directory
    def test_optimize_fuse_layer_norm_opset_17(self, tmpdir: str):
        output, quantized_model_path = tensor_quantize(input_data, [1, 3, 4, 4], "fuse_layer_norm_opset_17", tmpdir)
        has_target_op_type = contains_op_type(quantized_model_path, "LayerNormalization")

        comp_equal = np.allclose(output, fuse_layer_norm_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)
        self.assertEqual(has_target_op_type, True)

    @use_temporary_directory
    def test_optimize_split_large_kernel_pool_with_input_shape_23x23(self, tmpdir: str):
        input_shape = [1, 3, 23, 23]
        input_data = np.resize(split_large_kernel_pool_init_data, input_shape)

        output, quantized_model_path = tensor_quantize(input_data, input_shape, "split_large_kernel_pool", tmpdir)
        contains_average_pool = contains_op_type(quantized_model_path, "AveragePool")

        comp_equal = np.allclose(output, split_large_kernel_pool_with_input_shape_23x23_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)
        self.assertEqual(contains_average_pool, False)

    @use_temporary_directory
    def test_optimize_split_large_kernel_pool_with_input_shape_25x25(self, tmpdir: str):
        input_shape = [1, 3, 25, 25]
        input_data = np.resize(split_large_kernel_pool_init_data, input_shape)

        output, quantized_model_path = tensor_quantize(input_data, input_shape, "split_large_kernel_pool", tmpdir)
        contains_average_pool = contains_op_type(quantized_model_path, "AveragePool")

        comp_equal = np.allclose(output, split_large_kernel_pool_with_input_shape_25x25_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)
        self.assertEqual(contains_average_pool, True)

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
