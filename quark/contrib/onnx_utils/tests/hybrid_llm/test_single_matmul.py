# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import pytest

import ryzenai_onnx_utils
import ryzenai_onnx_utils.builder.graphs as graph_builder
import ryzenai_onnx_utils.partitioner
import ryzenai_onnx_utils.utils


def test_single_matmul(request: pytest.FixtureRequest, tmp_path: Path, dll_path: Path) -> None:
    if dll_path is None:
        pytest.skip("--dll-path missing")
    golden_providers = ["CPUExecutionProvider"]
    test_providers = ["CPUExecutionProvider"]
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(dll_path)

    graph = graph_builder.matmul.build_default()

    opsets = [
        onnx.OperatorSetIdProto(domain="ai.onnx", version=14),
        onnx.OperatorSetIdProto(domain="com.microsoft", version=1),
    ]

    model = onnx.helper.make_model(graph, opset_imports=opsets)
    model.ir_version = onnx.IR_VERSION_2023_5_5

    onnx.save_model(model, tmp_path / "golden.onnx")

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
    assert replaced_num == 1

    onnx.save_model(new_model, tmp_path / "test.onnx")

    test_session = ort.InferenceSession(
        new_model.SerializeToString(),
        providers=test_providers,
        sess_options=session_options,
    )
    test_outputs = test_session.run(output_names, rng_data)

    assert len(golden_outputs) == len(test_outputs)
    for golden, test in zip(golden_outputs, test_outputs, strict=True):
        assert np.allclose(test, golden, atol=0.1)

    root_path = request.config.rootpath
    test_path = Path(request.path).relative_to(root_path).parent
    exe_path = root_path / "build" / test_path / "Release" / "test_single_matmul.exe"
    cmd: list[str] = [
        exe_path.as_posix(),
        (tmp_path / "golden.onnx").as_posix(),
        (tmp_path / "test.onnx").as_posix(),
        Path(dll_path).absolute().as_posix(),
    ]

    subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
