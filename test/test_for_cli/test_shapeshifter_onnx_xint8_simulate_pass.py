#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for onnx_xint8_simulate pass: export Torch DPU-ops model to ONNX, quantize, run pass, assert conversions."""

import unittest
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import yaml
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.experimental.cli.main import main as cli
from quark.onnx import ModelQuantizer, QConfig, QLayerConfig, XInt8Spec

np.random.seed(123456)
# Single calibration sample: NCHW (1, 3, 32, 32)
INPUT_DATA = np.random.randn(1, 3, 32, 32).astype(np.float32) * 0.1


class DataReader(CalibrationDataReader):
    """Single-batch calibration data reader for quantization."""

    def __init__(self, input_tensor: np.ndarray):
        self.data = [input_tensor]
        self.input_name = "input"
        self.index = 0

    def get_next(self):
        if self.index < len(self.data):
            self.index += 1
            return {self.input_name: self.data[self.index - 1]}
        return None

    def rewind(self):
        self.index = 0


class DPUOpsModel(nn.Module):
    """Torch model with all ops converted by onnx_xint8_simulate: LeakyReLU, Sigmoid, Hardsigmoid, AvgPool, Softmax, InstanceNorm2d, mean, Clamp."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.leaky = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.hardsigmoid = nn.Hardsigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.instnorm = nn.InstanceNorm2d(8)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky(x)
        x = self.sigmoid(x)
        x = self.hardsigmoid(x)
        x = self.avgpool(x)
        x = self.softmax(x)
        x = self.instnorm(x)
        x = x.mean(dim=(2, 3), keepdim=True)
        x = torch.clamp(x, -10.0, 10.0)
        return x


def export_float_onnx(output_dir: str) -> str:
    """Export DPUOpsModel to float ONNX; returns path to dpu_ops_float.onnx."""
    torch.manual_seed(42)
    model = DPUOpsModel()
    model.eval()
    path = Path(output_dir, "dpu_ops_float.onnx").as_posix()
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        keep_initializers_as_inputs=False,
        do_constant_folding=False,
        opset_version=17,
        dynamo=False,
    )
    return path


def quantize_model(output_dir: str, input_data: np.ndarray) -> str:
    """Quantize float ONNX with BFloat16 QDQ; returns path to dpu_ops_quant.onnx."""
    float_path = export_float_onnx(output_dir)
    quant_path = Path(output_dir, "dpu_ops_quant.onnx").as_posix()
    reader = DataReader(input_data)
    config = QConfig(
        global_config=QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()),
        extra_options={
            "ConvertReduceMeanToGlobalAvgPool": False,
            "ConvertLeakyReluToDPUVersion": False,
            "ConvertSigmoidToHardSigmoid": False,
            "ConvertReduceMeanToDPUVersion": False,
            "ConvertAvgPoolToDPUVersion": False,
        },
    )
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(float_path, quant_path, reader)
    return quant_path


def prepare_yaml(
    output_dir: str,
    input_model_path: str,
    output_model_path: str,
) -> str:
    """Write shapeshifter YAML with onnx_xint8_simulate pass enabled; returns YAML path."""
    yaml_path = Path(output_dir, "xint8_simulate.yaml").as_posix()
    config = {
        "input_model_path": input_model_path,
        "passes": {
            "onnx_xint8_simulate": {
                "xint8_simulate": True,
            }
        },
        "output_model_path": output_model_path,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def count_op_types(model: onnx.ModelProto) -> dict[str, int]:
    """Return op_type -> count for all nodes in graph."""
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def check_model_valid(path: str) -> bool:
    """Load model and run onnx.checker.check_model; return True iff valid."""
    try:
        m = onnx.load(path)
        onnx.checker.check_model(m)
        return True
    except Exception:
        return False


class TestONNXAdapterONNXXInt8SimulatePass(unittest.TestCase):
    """Run onnx_xint8_simulate on quantized DPU-ops model; assert log messages and op replacements."""

    @use_temporary_directory
    def test_xint8_simulate_pass_on_quantized_dpu_ops_model(self, tmpdir: str) -> None:
        """Quantize model, run pass via CLI, check logs and output op counts."""
        quant_path = quantize_model(tmpdir, INPUT_DATA)
        out_path = Path(tmpdir, "dpu_ops_simulated.onnx").as_posix()

        with self.assertLogs("quark.shapeshifter.passes.onnx_xint8_simulate_screen", level="INFO") as cm:
            yaml_path = prepare_yaml(tmpdir, quant_path, out_path)
            cli(["shapeshifter", yaml_path])
            self.assertTrue(
                any(
                    "Found Leaky ReLU node /leaky/LeakyRelu with alpha=0.20000000298023224. Replacing with new alpha=0.19921875."
                    in message
                    for message in cm.output
                )
            )
            self.assertTrue(
                any(
                    "Found Sigmoid node /sigmoid/Sigmoid. Replacing with HardSigmoid." in message
                    for message in cm.output
                )
            )
            self.assertTrue(
                any(
                    "Found HardSigmoid node /sigmoid/Sigmoid with alpha=0.16666666666666666. Convert to DPU version."
                    in message
                    for message in cm.output
                )
            )
            self.assertTrue(
                any(
                    "Rescale AveragePool /avgpool/AveragePool with factor 0.984375 to simulate DPU behavior." in message
                    for message in cm.output
                )
            )
            self.assertTrue(
                any(
                    "Rescale ReduceMean /ReduceMean with factor 1.00250244140625 to simulate DPU behavior." in message
                    for message in cm.output
                )
            )
            self.assertTrue(
                any(
                    "Softmax /softmax/Softmax to simulate DPU behavior under opset 17." in message
                    for message in cm.output
                )
            )
            self.assertTrue(
                any(
                    "InstanceNormalization node /instnorm/InstanceNormalization to simulate DPU behavior by ExtendedInstanceNormalization."
                    in message
                    for message in cm.output
                )
            )
            self.assertTrue(
                any(
                    "Clip node '/Clip' is converted to DPU version, min is -10.0, max is 10.0." in message
                    for message in cm.output
                )
            )

        self.assertTrue(check_model_valid(out_path), "Output model should be valid after onnx_xint8_simulate")

        model = onnx.load(out_path)
        counts = count_op_types(model)
        self.assertEqual(counts.get("Sigmoid", 0), 0, "Sigmoid should be replaced by HardSigmoid")
        self.assertEqual(counts.get("Softmax", 0), 0, "Softmax should be replaced by DPU subgraph")
        self.assertEqual(
            counts.get("InstanceNormalization", 0),
            0,
            "InstanceNormalization should be replaced by ExtendedInstanceNormalization",
        )
        self.assertEqual(
            counts.get("ExtendedInstanceNormalization", 0),
            1,
            "Expect ExtendedInstanceNormalization from DPU simulation",
        )
        self.assertGreaterEqual(
            counts.get("HardSigmoid", 0) + counts.get("Mul", 0),
            1,
            "Expect HardSigmoid or inserted Mul from DPU simulation",
        )


if __name__ == "__main__":
    unittest.main()
