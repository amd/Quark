# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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


@pytest.mark.skip("Skipping temporarily until proper test filtering can be done")
def test_group_norm_matmul(tmp_path: Path, dll_path: Path, dd_root: Path) -> None:
    providers = ort.get_available_providers()
    if "RyzenAIExecutionProvider" not in providers:
        pytest.skip("No RyzenAI EP found")
    if dll_path is None or dd_root is None:
        pytest.skip("--dll-path or --dd-root missing a value")
    golden_providers = ["CPUExecutionProvider", "RyzenAIExecutionProvider"]
    test_providers = ["RyzenAIExecutionProvider"]
    session_options = ort.SessionOptions()
    session_options.add_session_config_entry("dd_cache", str(tmp_path))
    session_options.add_session_config_entry("dd_root", dd_root)
    session_options.add_session_config_entry("model_name", "UNET")
    session_options.register_custom_ops_library(dll_path)

    graph = graph_builder.groupnorm_reshape_matmul.build_default()

    opsets = [
        onnx.OperatorSetIdProto(domain="ai.onnx", version=14),
        onnx.OperatorSetIdProto(domain="com.microsoft", version=1),
    ]

    model = onnx.helper.make_model(graph, opset_imports=opsets)
    model.ir_version = onnx.IR_VERSION_2023_5_5

    rng_data = ryzenai_onnx_utils.utils.get_rng_data(graph.input, -1, 1)
    output_names = [x.name for x in graph.output]

    golden_session = ort.InferenceSession(
        model.SerializeToString(),
        providers=golden_providers,
        sess_options=session_options,
    )
    golden_outputs: list[npt.NDArray[Any]] = golden_session.run(output_names, rng_data)

    extractor = ryzenai_onnx_utils.matcher.get_extractor(model)

    passes = [
        "matmul_noqdq.matmul_to_matmul_noqdq",
        "groupnorm_to_groupnorm_noqdq",
        "remove_reshapes.groupnorm_noqdq_matmul_noqdq",
        "dd",
    ]

    xclbin = "/xclbin/stx/sdxl_unet_vae_combined_new.xclbin"
    op_namespace = "sdxlt"
    params = ryzenai_onnx_utils.ReplaceParams(
        {"xclbins": xclbin, "domains": "com.ryzenai", "op_namespaces": op_namespace},
        Path(),
        Path(),
        tmp_path,
    )

    new_model, replaced_num = ryzenai_onnx_utils.partitioner.partition(extractor, passes, params, 0)
    assert replaced_num == 5

    test_session = ort.InferenceSession(
        new_model.SerializeToString(),
        providers=test_providers,
        sess_options=session_options,
    )
    test_outputs = test_session.run(output_names, rng_data)

    assert len(golden_outputs) == len(test_outputs)
    for golden, test in zip(golden_outputs, test_outputs, strict=True):
        assert np.allclose(test, golden, atol=0.3)
