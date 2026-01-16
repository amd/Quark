# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import csv
import importlib
import importlib.resources
import io
import json
import os
import shutil
import sys
from collections import defaultdict
from json import dumps
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse

import onnx
import onnxruntime as ort


# TODO(varunsh): this should be updated to inherit parser arguments from the others somehow
def configure_parser(
    subparser: argparse._SubParsersAction[Any],
) -> argparse.ArgumentParser:
    vaiml_parser: argparse.ArgumentParser = subparser.add_parser(
        "vaiml", description="Summarize ONNX model and dump plugin config"
    )

    vaiml_parser.add_argument(
        "--plugin_name",
        type=str,
        required=True,
        help="Plugin name used for dumping JSON",
    )
    vaiml_parser.add_argument(
        "--ops",
        nargs="+",
        help="List of op types to include. If not set, all ops are included.",
    )
    vaiml_parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile all ops, if not passed only generates op json file",
    )
    vaiml_parser.add_argument(
        "--ops_type",
        type=str,
        choices=["bf16", "bfp16"],
        default="bf16",
        help='Type of ops to generate: "bf16" (default) or "bfp16"',
    )
    input_group = vaiml_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--ops-json",
        type=str,
        help="An ops.json file with operator names and shapes already specified",
    )
    input_group.add_argument(
        "--model-dir",
        type=str,
        help="Path to folder containing model.onnx and genai_config.json",
    )
    group = vaiml_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--context_length",
        type=int,
        choices=[2048, 3072, 4096],
        default=2048,
        help="Max context length (M). Allowed: 2048 (default), 3072, 4096",
    )
    group.add_argument(
        "--exact_length",
        type=int,
        help="Use an exact M value (e.g. --exact_length 123 → M = [123] for all ops)",
    )

    return vaiml_parser


def simplify_float(f: float) -> str:
    if f == 0:
        return "0"
    s = f"{f:.1e}"  # e.g. "1.0e-05"
    _, exp_str = s.split("e")
    return f"e{int(exp_str)}"


def get_attr_tuple(attr: onnx.AttributeProto) -> tuple[str, Any]:
    if attr.type == onnx.AttributeProto.INT:
        return (attr.name, attr.i)
    elif attr.type == onnx.AttributeProto.FLOAT:
        return (attr.name, simplify_float(attr.f))
    elif attr.type == onnx.AttributeProto.STRING:
        return (attr.name, attr.s.decode("utf-8"))
    elif attr.type == onnx.AttributeProto.INTS:
        return (attr.name, tuple(attr.ints))
    elif attr.type == onnx.AttributeProto.FLOATS:
        return (attr.name, tuple(simplify_float(f) for f in attr.floats))
    elif attr.type == onnx.AttributeProto.STRINGS:
        return (attr.name, tuple(s.decode("utf-8") for s in attr.strings))
    else:
        return (attr.name, f"(unsupported type {attr.type})")


def get_dummy_model(cache_dir: Path) -> Path:
    model = """
    <
   ir_version: 10,
   opset_import: ["" : 21],
   producer_name: "simple_model_producer"
>
simple_model (float[1,3,224,224] input) => (float[1,3,224,224] output) {
   output = Identity (input)
}
    """
    dummy_path = cache_dir / "llm-dummy.onnx"
    onnx.save(onnx.load(io.BytesIO(model.encode()), format="onnxtxt"), dummy_path)
    return dummy_path


def generate_ops_json(args, model_dir):
    all_supported_ops = [
        "bmm1",
        "bmm2",
        "softmax",
        "rope",
        "add",
        "rmsnorm",
        "silu",
        "mul",
        "gemm",
    ]

    selected_ops = set(args.ops) if args.ops else set(all_supported_ops)
    invalid_ops = selected_ops - set(all_supported_ops)
    if invalid_ops:
        print(f"❌ Error: Unsupported ops found: {', '.join(invalid_ops)}")
        print(f"✅ Supported ops are: {', '.join(all_supported_ops)}")
        sys.exit(1)

    if args.exact_length:
        if not (128 <= args.exact_length <= 4096):
            print("❌ Error: --exact_length must be between 128 and 4096")
            sys.exit(1)

        mha_M_list = [args.exact_length]
        mlp_M_list = [args.exact_length]
    else:
        full_M_list = [128, 256, 512, 1024, 2048, 3072, 4096]
        mha_M_list = [m for m in full_M_list if m <= args.context_length]
        mlp_M_list = [1] + mha_M_list

    ordered_ops = sorted(selected_ops)
    plugin_dir_name = args.plugin_name
    if args.ops:
        ordered_ops = sorted(selected_ops)
        plugin_name = plugin_dir_name + "_" + "_".join(ordered_ops)
    else:
        plugin_name = plugin_dir_name

    model_path = os.path.join(model_dir, "model.onnx")
    config_path = os.path.join(model_dir, "genai_config.json")

    if not os.path.isfile(model_path):
        print(f"Error: 'model.onnx' not found in {model_dir}")
        sys.exit(1)

    model = onnx.load(model_path)
    graph = model.graph

    output_dir = os.path.join(os.getcwd(), plugin_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    op_attr_summary: defaultdict[str, defaultdict[Any, int]] = defaultdict(lambda: defaultdict(int))

    num_heads = None
    kv_num_heads = None
    o_proj_N = None
    gate_proj_N = None
    head_size = None

    matmulnbits_table: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

    q_proj_n = None
    k_proj_n = None
    v_proj_n = None
    qkv_k_val = None
    qkv_blk_size = None

    for node in graph.node:
        op_type = node.op_type
        attr_signature = tuple(sorted(get_attr_tuple(attr) for attr in node.attribute))
        op_attr_summary[op_type][attr_signature] += 1

        if op_type == "GroupQueryAttention":
            for attr in node.attribute:
                if attr.name == "num_heads":
                    num_heads = attr.i
                elif attr.name == "kv_num_heads":
                    kv_num_heads = attr.i

        if op_type == "MatMulNBits":
            k_val = n_val = blk_size = None
            for attr in node.attribute:
                if attr.name == "K":
                    k_val = attr.i
                elif attr.name == "N":
                    n_val = attr.i
                elif attr.name == "block_size":
                    blk_size = attr.i
                if attr.name == "N" and "o_proj" in node.name and o_proj_N is None:
                    o_proj_N = attr.i
                if attr.name == "N" and "gate_proj" in node.name and gate_proj_N is None:
                    gate_proj_N = attr.i

            if any(proj in node.name for proj in ["q_proj", "k_proj", "v_proj"]):
                if "q_proj" in node.name and q_proj_n is None:
                    q_proj_n = n_val
                elif "k_proj" in node.name and k_proj_n is None:
                    k_proj_n = n_val
                elif "v_proj" in node.name and v_proj_n is None:
                    v_proj_n = n_val

                if qkv_k_val is None:
                    qkv_k_val = k_val
                    qkv_blk_size = blk_size

                continue

            if k_val is not None and n_val is not None and blk_size is not None:
                matmulnbits_table[(k_val, blk_size)].add(n_val)

    if q_proj_n is not None and k_proj_n is not None and v_proj_n is not None:
        total_qkv_n = q_proj_n + k_proj_n + v_proj_n
        assert qkv_k_val is not None and qkv_blk_size is not None, "qkv_k_val and qkv_blk_size must be set"
        matmulnbits_table[(qkv_k_val, qkv_blk_size)].add(total_qkv_n)

    if os.path.isfile(config_path):
        with open(config_path) as f:
            try:
                cfg = json.load(f)
                head_size = cfg.get("model", {}).get("decoder", {}).get("head_size", None)

            except json.JSONDecodeError:
                print("Warning: Could not parse genai_config.json")
    else:
        print("Warning: genai_config.json not found")

    node_summary_csv = os.path.join(output_dir, "model_node_summary.csv")
    with open(node_summary_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        for op_type, attr_group in op_attr_summary.items():
            for attr_sig, count in attr_group.items():
                row = [op_type, f"#{count}"]
                for k, v in attr_sig:
                    v_str = "[" + ",".join(str(x) for x in v) + "]" if isinstance(v, tuple) else str(v)
                    row.append(f"{k}={v_str}")
                writer.writerow(row)

    key_attr_csv = os.path.join(output_dir, "model_key_attributes.csv")
    with open(key_attr_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_heads", "kv_num_heads", "o_proj_N", "gate_proj_N", "head_size"])
        writer.writerow([num_heads, kv_num_heads, o_proj_N, gate_proj_N, head_size])

    kn_summary_csv = os.path.join(output_dir, "model_matmulnbits_kn_summary.csv")
    with open(kn_summary_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "block_size", "N_values"])
        for (k_val, blk_size), n_set in sorted(matmulnbits_table.items()):
            n_list = sorted(n_set)
            writer.writerow([k_val, blk_size, str(n_list)])

    # print(f"✅ Model Summary written to: {output_dir}")
    # print("  - model_node_summary.csv")
    # print("  - model_key_attributes.csv")
    # print("  - model_matmulnbits_kn_summary.csv")

    overlay_json = importlib.resources.files("ryzenai_onnx_utils") / "data/vaiml/oga_2x4x4_overlay.json"

    with importlib.resources.as_file(overlay_json) as path:
        if not path.exists():
            raise FileNotFoundError(f"The overlay JSON {overlay_json} does not exist")

    json_filename = f"{plugin_name}.json"
    json_path = os.path.join(output_dir, json_filename)

    ops_json = None

    if None in (num_heads, kv_num_heads, head_size, o_proj_N, gate_proj_N):
        print(
            "Warning: Cannot generate OPS JSON because one or more required key attributes (num_heads, kv_num_heads, head_size, o_proj_N, gate_proj_N) are missing."
        )
    else:
        is_v2 = args.ops_type == "bfp16"
        ops_json = {
            "xclbin": "llama2_mladf_2x4x4_v1_gemmbfp16_silu_mul_mha_rms_rope.xclbin",
            "overlay_json": f"{overlay_json}",
            "overlay": "2x4x4",
            "plugin_name": plugin_name,
            "enable_v2": is_v2,
            "op_types": [],
        }

        if "bmm1" in selected_ops:
            ops_json["op_types"].append(
                {
                    "op_type": "bmm1",
                    "B0": [num_heads],
                    "M": mha_M_list,
                    "K": [head_size],
                    "B1": [kv_num_heads],
                    "N": [-1],
                }
            )

        if "bmm2" in selected_ops:
            ops_json["op_types"].append(
                {
                    "op_type": "bmm2",
                    "B0": [num_heads],
                    "M": mha_M_list,
                    "K": [-1],
                    "B1": [kv_num_heads],
                    "N": [head_size],
                }
            )

        if "softmax" in selected_ops:
            ops_json["op_types"].append(
                {
                    "op_type": "masked_softmax",
                    "B0": [num_heads],
                    "M": mha_M_list,
                    "K": [-1],
                    "HS": [head_size],
                }
            )

        if "rope" in selected_ops:
            b0_rope = [num_heads] if num_heads == kv_num_heads else [num_heads, kv_num_heads]
            ops_json["op_types"].append(
                {
                    "op_type": "rope_mbk_bmk",
                    "B0": b0_rope,
                    "M": mha_M_list,
                    "K": [head_size],
                }
            )

        if "add" in selected_ops:
            ops_json["op_types"].append({"op_type": "add", "M": mlp_M_list, "K": [o_proj_N]})

        if "rmsnorm" in selected_ops:
            ops_json["op_types"].append({"op_type": "rmsnorm", "M": mlp_M_list, "K": [o_proj_N]})

        if "silu" in selected_ops:
            ops_json["op_types"].append({"op_type": "silu", "M": mlp_M_list, "K": [gate_proj_N]})

        if "mul" in selected_ops:
            ops_json["op_types"].append({"op_type": "mul", "M": mlp_M_list, "K": [gate_proj_N]})

        if "gemm" in selected_ops:
            for (k_val, blk_size), n_set in sorted(matmulnbits_table.items()):
                op = {
                    "op_type": "gemm_w4abfp16" if is_v2 else "gemm_w4abf16",
                    "M": mlp_M_list,
                    "K": [k_val],
                    "N": sorted(n_set),
                    "G": [blk_size],
                    "S": [1],
                }
                ops_json["op_types"].append(op)

    with open(json_path, "w") as jf:
        raw = json.dumps(ops_json, indent=4)
        import re

        compacted = re.sub(
            r"\[\s+([\d,\s\-]+?)\s+\]",
            lambda m: "[ " + " ".join(m.group(1).split()) + " ]",
            raw,
        )
        jf.write(compacted)

    print(f"✅  OPS JSON written to: {json_path}")
    return json_path


def main(args: argparse.Namespace) -> None:
    if args.ops_json is None:
        model_dir = args.model_dir
        json_path = generate_ops_json(args, model_dir)

        if not args.compile:
            print("OPS JSON created, Returning")
            return
    else:
        json_path = args.ops_json
        if not os.path.isfile(json_path):
            print(f"❌ Error: ops.json file '{json_path}' not found")
            sys.exit(1)

    ep_config_file = Path("vitisai_config.json")

    ep_config_file.write_text(
        dumps(
            {
                "passes": [
                    {"name": "init", "plugin": "vaip-pass_init"},
                    {
                        "name": "vaiml_llm_custom_op",
                        "plugin": "vaip-pass_vaiml_llm_custom_op",
                        "vaiml_config": {
                            "verbose": False,
                            "device": "stx",
                            "fatal_error_on_exception": True,
                            "llm_ops_config": f"{json_path}",
                            "run_archive": True,
                        },
                    },
                ],
                "target": "VAIML",
                "targets": [{"name": "VAIML", "pass": ["init", "vaiml_llm_custom_op"]}],
            }
        )
    )

    cache_dir = Path.cwd() / "ops_cache_dir"
    if cache_dir.exists() and cache_dir.is_dir():
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    dummy_model = get_dummy_model(cache_dir)

    with importlib.resources.as_file(dummy_model) as path:
        if not path.exists():
            raise FileNotFoundError(f"The dummy model {dummy_model} does not exist")

        ort.InferenceSession(
            path,
            providers=["VitisAIExecutionProvider"],
            provider_options=[
                {
                    "config_file": ep_config_file,
                    "cache_dir": cache_dir,
                    "enable_preemption": "0",
                }
            ],
        )
