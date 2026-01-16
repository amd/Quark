# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort

import ryzenai_onnx_utils
import ryzenai_onnx_utils.builder.graphs as graph_builder
import ryzenai_onnx_utils.partitioner
import ryzenai_onnx_utils.utils


def test_chain_matmulnbits(tmp_path: Path, dll_path: Path, num_ops: int = 1) -> None:
    golden_providers = ["CPUExecutionProvider"]
    test_providers = ["CPUExecutionProvider"]
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(dll_path)

    graph = graph_builder.chain_matmulnbits.build_default(num_ops)

    opsets = [
        onnx.OperatorSetIdProto(domain="ai.onnx", version=14),
        onnx.OperatorSetIdProto(domain="com.microsoft", version=1),
    ]

    model = onnx.helper.make_model(graph, opset_imports=opsets)
    model.ir_version = onnx.IR_VERSION_2023_5_5

    onnx.save_model(model, os.path.join(tmp_path, "golden_chain_matmulnbits.onnx"))

    rng_data = ryzenai_onnx_utils.utils.get_rng_data(graph.input, -1, 1)
    output_names = [x.name for x in graph.output]

    golden_session = ort.InferenceSession(
        model.SerializeToString(),
        providers=golden_providers,
    )
    golden_outputs: list[npt.NDArray[Any]] = golden_session.run(output_names, rng_data)

    extractor = ryzenai_onnx_utils.matcher.get_extractor(model)

    passes = [
        "change_domain",
    ]

    xclbin = ""
    op_namespace = ""
    params = ryzenai_onnx_utils.ReplaceParams(
        {"xclbins": xclbin, "domains": "com.ryzenai", "op_namespaces": op_namespace},
        Path(),
        Path(),
        tmp_path,
    )

    new_model, replaced_num = ryzenai_onnx_utils.partitioner.partition(extractor, passes, params, 0)
    assert replaced_num == num_ops

    onnx.save_model(new_model, os.path.join(tmp_path, "test_chain_matmulnbits.onnx"))

    test_session = ort.InferenceSession(
        new_model.SerializeToString(),
        providers=test_providers,
        sess_options=session_options,
    )
    test_outputs = test_session.run(output_names, rng_data)

    assert len(golden_outputs) == len(test_outputs)
    for golden, test in zip(golden_outputs, test_outputs, strict=True):
        assert np.allclose(test, golden, atol=0.1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp_path", type=str, required=True, help="Path to save generated onnx models")
    parser.add_argument("--dll_path", type=str, required=True, help="Path to dll with custom ops")
    parser.add_argument("--num_ops", type=int, default=1, help="How many matmulnbits to chain")
    args = parser.parse_args()

    test_chain_matmulnbits(args.tmp_path, args.dll_path, args.num_ops)
