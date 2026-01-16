# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.builder.graphs as graph_builder


def run_chain_matmulnbits(num_ops: int = 1, dtype: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT) -> None:
    graph = graph_builder.chain_matmulnbits.build_default(num_ops, dtype)
    opsets = [
        onnx.OperatorSetIdProto(domain="ai.onnx", version=14),
        onnx.OperatorSetIdProto(domain="com.microsoft", version=1),
    ]

    model = onnx.helper.make_model(graph, opset_imports=opsets)
    onnx.save_model(model, f"chain_matmulnbits_{num_ops}.onnx")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ops", type=int, default=1, help="How many matmulnbits to chain")
    parser.add_argument(
        "--dtype",
        type=str,
        default="Float32",
        choices=["Float32", "Float16"],
        help="Float32 or Float16 activation",
    )
    args = parser.parse_args()

    dtype = None

    if args.dtype == "Float32":
        dtype = onnx.TensorProto.FLOAT
    elif args.dtype == "Float16":
        dtype = onnx.TensorProto.FLOAT16
    else:
        raise ValueError(f"dtype {args.dtype} unexpected")

    run_chain_matmulnbits(args.num_ops, dtype)
