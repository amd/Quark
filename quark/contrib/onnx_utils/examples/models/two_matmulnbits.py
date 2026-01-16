# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.builder.graphs as graph_builder


def main() -> None:
    graph = graph_builder.two_matmulnbits.build_default()
    opsets = [
        onnx.OperatorSetIdProto(domain="ai.onnx", version=14),
        onnx.OperatorSetIdProto(domain="com.microsoft", version=1),
    ]

    model = onnx.helper.make_model(graph, opset_imports=opsets)
    onnx.save_model(model, "two_matmulnbits.onnx")


if __name__ == "__main__":
    main()
