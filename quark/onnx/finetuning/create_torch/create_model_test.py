#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
import numpy as np
import onnx
import onnxruntime

import torch

from quark.onnx.finetuning.create_torch.create_model import TorchModel


def onnx_to_torch(onnx_model_path: str) -> TorchModel:
    onnx_model = onnx.load(onnx_model_path)

    return TorchModel(onnx_model)


def main() -> None:
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        print("Please specify onnx model path")
        sys.exit(1)

    # Get input data
    so = onnxruntime.SessionOptions()
    sess = onnxruntime.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    data_shape = sess.get_inputs()[0].shape
    print(f"model {model_path} input: name {input_name} shape {data_shape}")

    input_data = np.random.randint(low=0, high=2, size=data_shape).astype(np.float32)

    # Run ONNX model
    onnx_output = sess.run(None, {input_name: input_data})[0]
    print("onnx model inference succeed!", onnx_output.min(), onnx_output.mean(), onnx_output.max())

    # Convert to Torch
    torch_model = onnx_to_torch(model_path)
    print("onnx to torch succeed!")

    # Run Torch model
    torch_output = torch_model(torch.from_numpy(input_data)).detach().numpy()
    print("torch model inference succeed!", torch_output.min(), torch_output.mean(), torch_output.max())


if __name__ == "__main__":
    main()
