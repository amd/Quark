# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse
from typing import Any

import onnx

import ryzenai_onnx_utils


def get_fusion_shapes(model_path: str) -> None:
    extractor = ryzenai_onnx_utils.matcher.load_extractor(model_path, False)

    attrs: dict[str, Any] = {
        "down_proj_input": None,
        "gate_proj_output": None,
        "group_size": None,
        "hidden_size": None,
        "kv_num_heads": None,
        "matmul_dim": None,
        "num_heads": None,
        "qk_dim": None,
        "seq_len": 1,
        "v_proj_output": None,
    }

    for node in extractor.graph.node:
        if node.op_type == "GroupQueryAttention" and attrs["num_heads"] is None:
            attrs["num_heads"] = onnx.helper.get_node_attr_value(node, "num_heads")
            attrs["kv_num_heads"] = onnx.helper.get_node_attr_value(node, "kv_num_heads")

            past_key_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[3], extractor)
            attrs["hidden_size"] = past_key_shape[-1]

        elif node.op_type == "MatMulNBits":
            if "q_proj" in node.name and attrs["qk_dim"] is None:
                q_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
                attrs["matmul_dim"] = q_shape[-1]

                q_w_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[1], extractor)
                attrs["qk_dim"] = q_w_shape[0]

            elif "k_proj" in node.name and attrs["group_size"] is None:
                k_w_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[1], extractor)
                attrs["qk_dim"] += k_w_shape[0]
                attrs["group_size"] = onnx.helper.get_node_attr_value(node, "block_size")

            elif "v_proj" in node.name and attrs["v_proj_output"] is None:
                v_shape = ryzenai_onnx_utils.matcher.get_shape(node.output[0], extractor)
                attrs["v_proj_output"] = v_shape[-1]

            elif "gate_proj" in node.name and attrs["gate_proj_output"] is None:
                gate_shape = ryzenai_onnx_utils.matcher.get_shape(node.output[0], extractor)
                attrs["gate_proj_output"] = gate_shape[-1]

            elif "down_proj" in node.name and attrs["down_proj_input"] is None:
                down_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
                attrs["down_proj_input"] = down_shape[-1]

    assert all(value is not None for value in attrs.values())

    print(
        f"mha_a16bfacc16bf_{attrs['num_heads']}_{attrs['kv_num_heads']}_{attrs['seq_len']}_maxSeqLen_{attrs['hidden_size']}.bin"
    )
    print(f"mlp_a16bfw3acc16bf_{attrs['seq_len']}_{attrs['matmul_dim']}_{attrs['gate_proj_output']}.bin")
    print(f"rms_add_a16bfw16bfacc16bf__{attrs['matmul_dim']}__{attrs['matmul_dim']}.bin")
    print(
        f"mladf_2x4x4_v1_a16fw3acc16f_{attrs['seq_len']}_{attrs['matmul_dim']}_{attrs['matmul_dim']}_{attrs['group_size']}.bin"
    )
    print(
        f"mladf_2x4x4_v1_a16fw3acc16f_{attrs['seq_len']}_{attrs['matmul_dim']}_{attrs['qk_dim']}_{attrs['group_size']}.bin"
    )
    print(
        f"mladf_2x4x4_v1_a16fw3acc16f_{attrs['seq_len']}_{attrs['down_proj_input']}_{attrs['matmul_dim']}_{attrs['group_size']}.bin"
    )
    print(
        f"mladf_2x4x4_v1_a16fw3acc16f_{attrs['seq_len']}_{attrs['matmul_dim']}_{attrs['v_proj_output']}_{attrs['group_size']}_maxSeqLen_gm.bin"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()

    get_fusion_shapes(args.model_path)
