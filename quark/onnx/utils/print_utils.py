#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import platform
import textwrap
from dataclasses import fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from pprint import pformat
from typing import Any

import onnx
import onnxruntime as ort
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import DEQUANT_OP_NAME, QuantFormat, QuantType

from quark.common.utils.log import ScreenLogger
from quark.onnx.quantization.config.config import QConfig
from quark.onnx.quantization.config.legacy import QuantizationConfig
from quark.onnx.quantization.config.utils import config_to_dict, is_algo_config
from quark.onnx.quantization.quant_utils import (
    COP_BFP_OP_NAME,
    COP_DEQUANT_OP_NAME,
    COP_MX_OP_NAME,
    DEQUANT_OP_TYPES,
    FN_OP_TYPES,
    QUANT_OP_TYPES,
    ExtendedQuantFormat,
    ExtendedQuantType,
    __version__,
)
from quark.onnx.utils.file_utils import save_quantized_info

logger = ScreenLogger(__name__)


def _value_sort_key(item: tuple[str, Any]) -> tuple[int, str]:
    label, v = item
    if v is True:
        prio = 0
    elif v is False:
        prio = 1
    elif isinstance(v, int | float):
        prio = 2
    elif isinstance(v, list | dict | tuple | str):
        prio = 3
    elif v is None:
        prio = 4
    else:
        prio = 5
    return (prio, label)


def _log_param_section(
    title: str,
    rows: list[tuple[str, Any]],
    description: str | None = None,
    label_width: int = 50,
    sort_by_type: bool = False,
) -> None:
    continuation = " " * (label_width + 1)
    lines: list[str] = [title]
    if description:
        lines.extend(textwrap.wrap(description, width=100, initial_indent="    ", subsequent_indent="    "))
        lines.append("")
    ordered = sorted(rows, key=_value_sort_key) if sort_by_type else rows
    for label, value in ordered:
        text = _format_param_value(value)
        label_cell = f"{f'{label} ---':>{label_width}}"
        if "\n" in text:
            head, *rest = text.split("\n")
            lines.append(f"{label_cell} {head}")
            for r in rest:
                lines.append(f"{continuation}{r}")
        else:
            lines.append(f"{label_cell} {text}")
    logger.info("\n".join(lines))


def _log_time_info() -> None:
    """
    Log current time information.

    Returns:
        None
    """
    logger.info(f"Time information:\n{datetime.now()}")


def _log_os_cpu_info() -> None:
    """
    Log OS and CPU information.

    Returns:
        None
    """
    _log_param_section(
        "OS and CPU information:",
        [
            ("system", platform.system()),
            ("node", platform.node()),
            ("release", platform.release()),
            ("version", platform.version()),
            ("machine", platform.machine()),
            ("processor", platform.processor()),
        ],
    )


def _log_tools_version_info() -> None:
    """
    Log tools version information including Python, ONNX, ONNX Runtime, and Quark ONNX versions.

    Returns:
        None
    """
    _log_param_section(
        "Tools version information:",
        [
            ("python", platform.python_version()),
            ("onnx", onnx.__version__),  # type: ignore[attr-defined]
            ("onnxruntime", ort.__version__),
            ("quark.onnx", __version__),
        ],
    )


def _log_quantized_config_info(**config: Any) -> None:
    _log_param_section("Quantized Configuration information:", list(config.items()))


def print_quantize_static_info(
    model_input: str | Path | onnx.ModelProto,
    model_output: str | Path | None,
    calibration_data_reader: CalibrationDataReader | None,
    calibration_data_path: str | None,
    quant_format: QuantFormat | ExtendedQuantFormat,
    input_nodes: list[str] | None,
    output_nodes: list[str] | None,
    op_types_to_quantize: list[str] | None,
    extra_op_types_to_quantize: list[str] | None,
    per_channel: bool,
    reduce_range: bool,
    activation_type: QuantType | ExtendedQuantType,
    weight_type: QuantType | ExtendedQuantType,
    nodes_to_quantize: list[str],
    nodes_to_exclude: list[str],
    subgraphs_to_exclude: list[tuple[list[str]]],
    optimize_model: bool,
    use_external_data_format: bool,
    calibrate_method: CalibrationMethod | Any,
    execution_providers: list[str] | None,
    enable_npu_cnn: bool,
    enable_npu_transformer: bool,
    specific_tensor_precision: bool,
    debug_mode: bool,
    crypto_mode: bool,
    convert_fp16_to_fp32: bool,
    convert_nchw_to_nhwc: bool,
    include_cle: bool,
    include_sq: bool,
    include_rotation: bool,
    include_fast_ft: bool,
    extra_options: dict[str, Any],
) -> None:
    """Flat dump of the legacy ``Config`` / ``QuantizationConfig`` parameter
    set. Used by the legacy ``ModelQuantizer`` path that does not opt into the
    categorized effective-config summary.
    """
    if crypto_mode:
        return

    try:
        _log_time_info()
        _log_os_cpu_info()
        _log_tools_version_info()
        _log_quantized_config_info(
            model_input=type(model_input) if isinstance(model_input, onnx.ModelProto) else model_input,
            model_output=model_output,
            calibration_data_reader=calibration_data_reader,
            calibration_data_path=calibration_data_path,
            quant_format=quant_format,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            op_types_to_quantize=op_types_to_quantize,
            extra_op_types_to_quantize=extra_op_types_to_quantize,
            per_channel=per_channel,
            reduce_range=reduce_range,
            activation_type=activation_type,
            weight_type=weight_type,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            subgraphs_to_exclude=subgraphs_to_exclude,
            optimize_model=optimize_model,
            use_external_data_format=use_external_data_format,
            calibrate_method=calibrate_method,
            execution_providers=execution_providers,
            enable_npu_cnn=enable_npu_cnn,
            enable_npu_transformer=enable_npu_transformer,
            specific_tensor_precision=specific_tensor_precision,
            debug_mode=debug_mode,
            convert_fp16_to_fp32=convert_fp16_to_fp32,
            convert_nchw_to_nhwc=convert_nchw_to_nhwc,
            include_cle=include_cle,
            include_sq=include_sq,
            include_rotation=include_rotation,
            include_fast_ft=include_fast_ft,
            extra_options=extra_options,
        )
    except Exception:
        pass


def print_quantize_dynamic_info(
    model_input: str | Path | onnx.ModelProto,
    model_output: str | Path | None,
    op_types_to_quantize: list[str] | None,
    per_channel: bool,
    reduce_range: bool,
    weight_type: QuantType | ExtendedQuantType,
    nodes_to_quantize: list[str],
    nodes_to_exclude: list[str],
    subgraphs_to_exclude: list[tuple[list[str]]],
    use_external_data_format: bool,
    debug_mode: bool,
    crypto_mode: bool,
    extra_options: dict[str, Any],
) -> None:
    """
    print os_cpu, time, tool_version, quantized_configuration information.
    """

    if crypto_mode:
        return  # Print nothing in crypto mode

    try:
        _log_time_info()
        _log_os_cpu_info()
        _log_tools_version_info()
        _log_quantized_config_info(
            model_input=type(model_input) if isinstance(model_input, onnx.ModelProto) else model_input,
            model_output=model_output,
            op_types_to_quantize=op_types_to_quantize,
            per_channel=per_channel,
            reduce_range=reduce_range,
            weight_type=weight_type,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            subgraphs_to_exclude=subgraphs_to_exclude,
            use_external_data_format=use_external_data_format,
            debug_mode=debug_mode,
            extra_options=extra_options,
        )
    except Exception:
        pass


def print_fp32_nodes(fp32_nodes_dict: dict[str, int], output_model_path: str | Path | None) -> None:
    try:
        fp32_nodes_list = list(fp32_nodes_dict.keys())

        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table()
        table.add_column("Op Type")
        table.add_column("Float Model", style="bold green1")

        for node_op_type in fp32_nodes_list:
            node_fp32_count = fp32_nodes_dict[node_op_type]
            table.add_row(node_op_type, str(node_fp32_count))
        table.add_section()
        if output_model_path is not None:
            output_path = output_model_path.as_posix() if isinstance(output_model_path, Path) else output_model_path
            table.add_row("Quantized model path", output_path)

        logger.info(
            "The operation types and their corresponding quantities of the input float model is shown in the table below."
        )
        console.print(table)

    except Exception:
        pass


def check_weights_in_node(model: onnx.ModelProto, node: onnx.NodeProto) -> bool:
    weights_in_node = False
    initializer_names = {init.name for init in model.graph.initializer}
    for input_ in node.input:
        if input_ in initializer_names:
            weights_in_node = True
    return weights_in_node


def print_quantized_info(
    model_quant: str | Path | onnx.ModelProto, debug_mode: bool, shared_init_optypes: list[str] | None
) -> None:
    try:
        data_type_dict = {
            0: "",
            1: "FLOAT",
            2: "UINT8",
            3: "INT8",
            4: "UINT16",
            5: "INT16",
            6: "INT32",
            7: "INT64",
            8: "STR",
            9: "BOOL",
            10: "FLOAT16",
            11: "DOUBLE",
            12: "UINT32",
            13: "UINT64",
            16: "BFLOAT16",
            17: "FP8E4M3",
            18: "FP8E4M3UZ",
            19: "FP8E5M2",
            20: "FP8E5M2UZ",
            23: "FP4E2M1",
            40: "BFP",
            41: "MX",
        }
        qdq_ops = QUANT_OP_TYPES + DEQUANT_OP_TYPES + FN_OP_TYPES

        op_type_with_weights_bias = [
            "MatMul",
            "Conv",
            "ConvTranspose",
            "Gemm",
            "LayerNormalization",
            "EmbedLayerNormalization",
            "InstanceNormalization",
            "PRelu",
        ]
        quantized_data = []

        quantized_model = model_quant if isinstance(model_quant, onnx.ModelProto) else onnx.load(model_quant)
        onnx_model = ONNXModel(quantized_model)

        tensor_to_node_dict = {}
        tensor_to_init_dict = {}
        for node in onnx_model.model.graph.node:
            for output in node.output:
                tensor_to_node_dict[output] = node
        for init in onnx_model.model.graph.initializer:
            tensor_to_init_dict[init.name] = init

        nodes_quantized_info_list = []

        for node in onnx_model.model.graph.node:
            if node.op_type in DEQUANT_OP_TYPES + FN_OP_TYPES + QUANT_OP_TYPES:
                continue

            if len(node.input) >= 1:
                if (
                    node.input[0] in tensor_to_node_dict
                    and tensor_to_node_dict[node.input[0]].op_type == DEQUANT_OP_NAME
                ):
                    act_dq_data_type = 0
                    weights_dq_data_type = 0
                    bias_dq_data_type = 0
                    act_dq_node = tensor_to_node_dict[node.input[0]]
                    weights_dq_node = None
                    bias_dq_node = None
                    if len(node.input) >= 2 and node.input[1] in tensor_to_node_dict:
                        weights_dq_node = tensor_to_node_dict[node.input[1]]
                    if len(node.input) >= 3 and node.input[2] in tensor_to_node_dict:
                        bias_dq_node = tensor_to_node_dict[node.input[2]]
                    act_init = tensor_to_init_dict[act_dq_node.input[2]]
                    act_dq_data_type = act_init.data_type
                    weights_init = None
                    bias_init = None
                    if (
                        weights_dq_node is not None
                        and node.op_type in op_type_with_weights_bias
                        and check_weights_in_node(onnx_model.model, weights_dq_node)
                    ):
                        weights_init = tensor_to_init_dict[weights_dq_node.input[2]]
                        weights_dq_data_type = weights_init.data_type
                    if (
                        bias_dq_node is not None
                        and node.op_type in op_type_with_weights_bias
                        and check_weights_in_node(onnx_model.model, bias_dq_node)
                    ):
                        bias_init = tensor_to_init_dict[bias_dq_node.input[2]]
                        bias_dq_data_type = bias_init.data_type
                    nodes_quantized_info_list.append(
                        [node.name, node.op_type, act_dq_data_type, weights_dq_data_type, bias_dq_data_type]
                    )
                elif (
                    len(node.input) >= 2
                    and node.input[1] in tensor_to_node_dict
                    and tensor_to_node_dict[node.input[1]].op_type == DEQUANT_OP_NAME
                ):
                    act_dq_data_type = 0
                    weights_dq_data_type = 0
                    bias_dq_data_type = 0
                    act_dq_node = None
                    weights_dq_node = None
                    bias_dq_node = None
                    if node.input[0] in tensor_to_node_dict:
                        act_dq_node = tensor_to_node_dict[node.input[0]]
                    if len(node.input) >= 2 and node.input[1] in tensor_to_node_dict:
                        weights_dq_node = tensor_to_node_dict[node.input[1]]
                    if len(node.input) >= 3 and node.input[2] in tensor_to_node_dict:
                        bias_dq_node = tensor_to_node_dict[node.input[2]]
                    act_init = None
                    weights_init = None
                    bias_init = None
                    if (
                        act_dq_node is not None
                        and node.op_type in op_type_with_weights_bias
                        and check_weights_in_node(onnx_model.model, act_dq_node)
                    ):
                        if len(act_dq_node.input) >= 3 and act_dq_node.input[2] in tensor_to_init_dict:
                            act_init = tensor_to_init_dict[act_dq_node.input[2]]
                            act_dq_data_type = act_init.data_type
                    if (
                        weights_dq_node is not None
                        and node.op_type in op_type_with_weights_bias
                        and check_weights_in_node(onnx_model.model, weights_dq_node)
                    ):
                        if len(weights_dq_node.input) >= 3 and weights_dq_node.input[2] in tensor_to_init_dict:
                            weights_init = tensor_to_init_dict[weights_dq_node.input[2]]
                        assert weights_init is not None
                        weights_dq_data_type = weights_init.data_type
                    if (
                        bias_dq_node is not None
                        and node.op_type in op_type_with_weights_bias
                        and check_weights_in_node(onnx_model.model, bias_dq_node)
                    ):
                        if len(bias_dq_node.input) >= 3 and bias_dq_node.input[2] in tensor_to_init_dict:
                            bias_init = tensor_to_init_dict[bias_dq_node.input[2]]
                        assert bias_init is not None
                        bias_dq_data_type = bias_init.data_type
                    nodes_quantized_info_list.append(
                        [node.name, node.op_type, act_dq_data_type, weights_dq_data_type, bias_dq_data_type]
                    )
                if (
                    node.input[0] in tensor_to_node_dict
                    and tensor_to_node_dict[node.input[0]].op_type == COP_BFP_OP_NAME
                ):
                    act_dq_node = tensor_to_node_dict[node.input[0]]
                    weights_dq_node = None
                    bias_dq_node = None
                    if len(node.input) >= 2 and node.input[1] in tensor_to_node_dict:
                        weights_dq_node = tensor_to_node_dict[node.input[1]]
                    if len(node.input) >= 3 and node.input[2] in tensor_to_node_dict:
                        bias_dq_node = tensor_to_node_dict[node.input[2]]
                    act_dq_data_type = 0
                    weights_dq_data_type = 0
                    bias_dq_data_type = 0
                    if act_dq_node is not None and act_dq_node.op_type == COP_BFP_OP_NAME:
                        act_dq_data_type = 40
                    if weights_dq_node is not None and weights_dq_node.op_type == COP_BFP_OP_NAME:
                        weights_dq_data_type = 40
                    if bias_dq_node is not None and bias_dq_node.op_type == COP_BFP_OP_NAME:
                        bias_dq_data_type = 40
                    nodes_quantized_info_list.append(
                        [node.name, node.op_type, act_dq_data_type, weights_dq_data_type, bias_dq_data_type]
                    )
                if (
                    node.input[0] in tensor_to_node_dict
                    and tensor_to_node_dict[node.input[0]].op_type == COP_MX_OP_NAME
                ):
                    act_dq_node = tensor_to_node_dict[node.input[0]]
                    weights_dq_node = None
                    bias_dq_node = None
                    if len(node.input) >= 2 and node.input[1] in tensor_to_node_dict:
                        weights_dq_node = tensor_to_node_dict[node.input[1]]
                    if len(node.input) >= 3 and node.input[2] in tensor_to_node_dict:
                        bias_dq_node = tensor_to_node_dict[node.input[2]]
                    act_dq_data_type = 0
                    weights_dq_data_type = 0
                    bias_dq_data_type = 0
                    if act_dq_node is not None and act_dq_node.op_type == COP_MX_OP_NAME:
                        act_dq_data_type = 41
                    if weights_dq_node is not None and weights_dq_node.op_type == COP_MX_OP_NAME:
                        weights_dq_data_type = 41
                    if bias_dq_node is not None and bias_dq_node.op_type == COP_MX_OP_NAME:
                        bias_dq_data_type = 41
                    nodes_quantized_info_list.append(
                        [node.name, node.op_type, act_dq_data_type, weights_dq_data_type, bias_dq_data_type]
                    )
                if (
                    node.input[0] in tensor_to_node_dict
                    and tensor_to_node_dict[node.input[0]].op_type == COP_DEQUANT_OP_NAME
                ):
                    act_dq_node = tensor_to_node_dict[node.input[0]]
                    weights_dq_node = None
                    bias_dq_node = None
                    if len(node.input) >= 2 and node.input[1] in tensor_to_node_dict:
                        weights_dq_node = tensor_to_node_dict[node.input[1]]
                    if len(node.input) >= 3 and node.input[2] in tensor_to_node_dict:
                        bias_dq_node = tensor_to_node_dict[node.input[2]]
                    act_dq_data_type = 0
                    weights_dq_data_type = 0
                    bias_dq_data_type = 0
                    act_init = tensor_to_init_dict[act_dq_node.input[2]]
                    act_dq_data_type = act_init.data_type
                    weights_init = None
                    bias_init = None
                    if (
                        weights_dq_node is not None
                        and node.op_type in op_type_with_weights_bias
                        and check_weights_in_node(onnx_model.model, weights_dq_node)
                    ):
                        weights_init = tensor_to_init_dict[weights_dq_node.input[2]]
                        weights_dq_data_type = weights_init.data_type
                    if (
                        bias_dq_node is not None
                        and node.op_type in op_type_with_weights_bias
                        and check_weights_in_node(onnx_model.model, bias_dq_node)
                    ):
                        bias_init = tensor_to_init_dict[bias_dq_node.input[2]]
                        bias_dq_data_type = bias_init.data_type
                    nodes_quantized_info_list.append(
                        [node.name, node.op_type, act_dq_data_type, weights_dq_data_type, bias_dq_data_type]
                    )
                elif (
                    len(node.input) >= 2
                    and node.input[1] in tensor_to_node_dict
                    and tensor_to_node_dict[node.input[1]].op_type == COP_DEQUANT_OP_NAME
                ):
                    act_dq_node = None
                    weights_dq_node = None
                    bias_dq_node = None
                    if node.input[0] in tensor_to_node_dict:
                        act_dq_node = tensor_to_node_dict[node.input[0]]
                    if len(node.input) >= 2 and node.input[1] in tensor_to_node_dict:
                        weights_dq_node = tensor_to_node_dict[node.input[1]]
                    if len(node.input) >= 3 and node.input[2] in tensor_to_node_dict:
                        bias_dq_node = tensor_to_node_dict[node.input[2]]
                    act_dq_data_type = 0
                    weights_dq_data_type = 0
                    bias_dq_data_type = 0
                    act_init = tensor_to_init_dict[act_dq_node.input[2]]
                    act_dq_data_type = act_init.data_type
                    weights_init = None
                    bias_init = None
                    if (
                        weights_dq_node is not None
                        and node.op_type in op_type_with_weights_bias
                        and check_weights_in_node(onnx_model.model, weights_dq_node)
                    ):
                        weights_init = tensor_to_init_dict[weights_dq_node.input[2]]
                        weights_dq_data_type = weights_init.data_type
                    if (
                        bias_dq_node is not None
                        and node.op_type in op_type_with_weights_bias
                        and check_weights_in_node(onnx_model.model, bias_dq_node)
                    ):
                        bias_init = tensor_to_init_dict[bias_dq_node.input[2]]
                        bias_dq_data_type = bias_init.data_type
                    nodes_quantized_info_list.append(
                        [node.name, node.op_type, act_dq_data_type, weights_dq_data_type, bias_dq_data_type]
                    )
                else:
                    if node.op_type not in qdq_ops:
                        act_dq_data_type = 1
                        weights_dq_data_type = 0
                        bias_dq_data_type = 0
                        if len(node.input) >= 2 and node.op_type in op_type_with_weights_bias:
                            weights_dq_data_type = 1
                        if len(node.input) >= 3 and node.op_type in op_type_with_weights_bias:
                            bias_dq_data_type = 1
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table()
        table.add_column("Node Name")
        table.add_column("Op Type")
        table.add_column("Activation", style="bold green1")
        table.add_column("Weights", style="bold green1")
        table.add_column("Bias", style="bold green1")
        quantized_data.append(["Node Name", "Op Type", "Activation", "Weights", "Bias"])

        for node_quantized_info in nodes_quantized_info_list:
            table.add_row(
                node_quantized_info[0],
                node_quantized_info[1],
                data_type_dict[node_quantized_info[2]],
                data_type_dict[node_quantized_info[3]],
                data_type_dict[node_quantized_info[4]],
            )
            quantized_data.append(
                [
                    node_quantized_info[0],
                    node_quantized_info[1],
                    data_type_dict[node_quantized_info[2]],
                    data_type_dict[node_quantized_info[3]],
                    data_type_dict[node_quantized_info[4]],
                ]
            )
        if debug_mode:
            logger.info("The quantized information for all nodes is shown in the table below.")
            console.print(table)

        op_types_dict: Any = {}
        for node_quantized_info in nodes_quantized_info_list:
            op_type = node_quantized_info[1]
            if op_type not in op_types_dict:
                op_types_dict[op_type] = {"act": {}, "weights": {}, "bias": {}}
            if data_type_dict[node_quantized_info[2]] not in op_types_dict[op_type]["act"]:
                op_types_dict[op_type]["act"][data_type_dict[node_quantized_info[2]]] = 0
            if data_type_dict[node_quantized_info[3]] not in op_types_dict[op_type]["weights"]:
                op_types_dict[op_type]["weights"][data_type_dict[node_quantized_info[3]]] = 0
            if data_type_dict[node_quantized_info[4]] not in op_types_dict[op_type]["bias"]:
                op_types_dict[op_type]["bias"][data_type_dict[node_quantized_info[4]]] = 0
            op_types_dict[op_type]["act"][data_type_dict[node_quantized_info[2]]] += 1
            op_types_dict[op_type]["weights"][data_type_dict[node_quantized_info[3]]] += 1
            op_types_dict[op_type]["bias"][data_type_dict[node_quantized_info[4]]] += 1

        console = Console()

        table = Table()
        table.add_column("Op Type")
        table.add_column("Activation", style="bold green1")
        table.add_column("Weights", style="bold green1")
        table.add_column("Bias", style="bold green1")
        quantized_data.append([])
        quantized_data.append(["Op Type", "Activation", "Weights", "Bias"])

        for op_type in op_types_dict:
            act_list = []
            weights_list = []
            bias_list = []
            for data_type in op_types_dict[op_type]["act"]:
                if data_type != "":
                    act_list.append(data_type + "(" + str(op_types_dict[op_type]["act"][data_type]) + ")")
            act_list.sort()
            act_str = " ".join(act_list)
            for data_type in op_types_dict[op_type]["weights"]:
                if data_type != "":
                    weights_list.append(data_type + "(" + str(op_types_dict[op_type]["weights"][data_type]) + ")")
            weights_list.sort()
            weights_str = " ".join(weights_list)
            for data_type in op_types_dict[op_type]["bias"]:
                if data_type != "":
                    bias_list.append(data_type + "(" + str(op_types_dict[op_type]["bias"][data_type]) + ")")
            bias_list.sort()
            bias_str = " ".join(bias_list)
            table.add_row(op_type, act_str, weights_str, bias_str)
            quantized_data.append([op_type, act_str, weights_str, bias_str])
        if not debug_mode:
            logger.info("The quantized information for all operation types is shown in the table below.")
            logger.info(
                "The discrepancy between the operation types in the quantized model and the float model is due to the application of graph optimization."
            )
            console.print(table)
            if shared_init_optypes is not None:
                logger.info(
                    "Note: Due to NPU limitations, some shared parameters in certain models may need to be duplicated, which could lead to an increase in the model size after quantization."
                )

        save_quantized_info([[]])
        save_quantized_info(quantized_data)
        save_quantized_info([[]])

    except Exception:
        pass


# Quantizer-family labels for the (not effective on …) annotation. These mirror
# the dispatch in quark/onnx/quantizers/interface.py::create_static_quantizer.
# Options consumed only by NPU-CNN (XINT8QDQQuantizer) or ExtendedQDQQuantizer:
# other quantizer paths instantiate with default values but never act on them,
# so the summary annotates them as "not effective" on those paths.
_NPU_OR_EXTENDED: frozenset[str] = frozenset({"XINT8QDQQuantizer", "ExtendedQDQQuantizer"})
_NPU_CNN_ONLY: frozenset[str] = frozenset({"XINT8QDQQuantizer"})
_EXTENDED_QDQ_ONLY: frozenset[str] = frozenset({"ExtendedQDQQuantizer"})

# Human-readable label for each consumed_by set, used in the
# "only takes effect under <label> config" annotation.
_CONSUMED_BY_LABEL: dict[frozenset[str], str] = {
    _NPU_OR_EXTENDED: "NPU-CNN (XINT8 QDQ) or Extended QDQ",
    _NPU_CNN_ONLY: "NPU-CNN (XINT8 QDQ)",
    _EXTENDED_QDQ_ONLY: "Extended QDQ",
}

# PascalCase extra_options key → lowercase calibrator-internal key (subset of
# quark.onnx.calibration.interface.extra_options_keys_mapping that the summary
# reflects). The summary uses this to map the resolve_calibrator_extra_defaults
# overlay back into Quark's user-facing names.
_CALIB_KEY_TO_LOWER: tuple[tuple[str, str], ...] = (
    ("CalibTensorRangeSymmetric", "symmetric"),
    ("CalibMovingAverage", "moving_average"),
    ("CalibMovingAverageConstant", "averaging_constant"),
    ("Percentile", "percentile"),
    ("LWPMetric", "lwp_metric"),
    ("PercentileCandidates", "percentile_candidates"),
    ("MinMSEModePof2Scale", "minmse_mode"),
    ("CalibOptimizeMem", "optimize_mem"),
    ("CalibOptimizeDisk", "optimize_disk"),
    ("CalibWorkerNum", "worker_num"),
    ("NumBins", "num_bins"),
    ("NumQuantizedBins", "num_quantized_bins"),
    ("Scenario", "scenario"),
)


def _resolve_active_quantizer(
    quant_format: Any,
    calibrate_method: Any,
    enable_npu_cnn: bool,
) -> str:
    """Mirror create_static_quantizer dispatch from effective settings."""
    from quark.onnx.calibration.methods import Int16Method, LayerWiseMethod, PowerOfTwoMethod

    enable_npu_cnn = bool(enable_npu_cnn)
    is_floatscale = (calibrate_method in CalibrationMethod) or (calibrate_method in LayerWiseMethod)
    is_pow2_or_int16 = (calibrate_method in PowerOfTwoMethod) or (calibrate_method in Int16Method)

    if quant_format is QuantFormat.QOperator:
        return "ONNXQuantizer" if is_floatscale else "ExtendedONNXQuantizer"
    if quant_format is QuantFormat.QDQ:
        if is_pow2_or_int16 and enable_npu_cnn:
            return "XINT8QDQQuantizer"
        return "BaseExtendedQDQQuantizer"
    if quant_format is ExtendedQuantFormat.QDQ:
        return "ExtendedQDQQuantizer"
    # No match: mirror create_static_quantizer, which raises in this case.
    # We return "" so the summary can degrade gracefully (skip consumed_by
    # annotations) instead of fabricating a fake quantizer name.
    return ""


# Category schema: (key, default, consumed_by).
# default=None → print effective_extra value as-is; consumed_by=None → all quantizers consume.
_QCFG_CATEGORIES: tuple[tuple[str, str, tuple[tuple[str, Any, frozenset[str] | None], ...]], ...] = (
    (
        "1. Preprocessing & Graph Optimization",
        "Model simplification, operator fusion (LayerNorm, Gelu, InstanceNorm), BatchNorm folding, "
        "and format/layout conversion steps applied to the graph before quantization begins.",
        (
            ("PreprocessYAML", None, None),
            ("SkipPreprocess", False, None),
            ("OptimizeModel", True, None),
            ("SimplifyModel", True, None),
            ("SimplifyModelOptions", {}, None),
            ("ConvertFP16ToFP32", None, None),
            ("ConvertNCHWToNHWC", None, None),
            ("ConvertOpsetVersion", None, None),
            ("ConvertBNToConv", False, None),
            ("ConvertReduceMeanToGlobalAvgPool", False, None),
            ("SplitLargeKernelPool", False, None),
            ("ConvertSplitToSlice", False, None),
            ("ConvertClipToRelu", False, None),
            ("FuseInstanceNorm", True, None),
            ("FuseL2Norm", True, None),
            ("FuseGelu", True, None),
            ("FuseLayerNorm", True, None),
            ("FoldBatchNorm", None, None),
            ("FoldRelu", False, None),
            ("ReplaceClip6Relu", False, None),
        ),
    ),
    (
        "2. Quantization Target Selection",
        "Determines the scope of quantization: which operator types, named nodes, or subgraphs are "
        "targeted, and whether to quantize weights only or all tensors.",
        (
            ("OpTypesToQuantize", None, None),
            ("NodesToQuantize", None, None),
            # NodesToExclude is a quantize_static() positional parameter, not an
            # extra_options key: the QConfig path writes it as mapping
            # ["nodes_to_exclude"] (config/maps.py:619), never into extra_options.
            # Listing it here would always render an empty default and mislead.
            ("ExtraOpTypesToQuantize", None, None),
            ("QuantizeAllOpTypes", False, None),
            ("WeightsOnly", False, None),
            ("MatMulConstBOnly", None, None),
            ("EnableSubgraph", False, None),
        ),
    ),
    (
        "3. QDQ Node Management",
        "Controls insertion style of QuantizeLinear/DequantizeLinear (QDQ) pairs and removal of "
        "redundant QDQ nodes between adjacent operators, especially for DPU/NPU fusion patterns.",
        (
            ("AddQDQPairToWeight", None, None),
            ("ForceQuantizeNoInputCheck", False, None),
            ("DedicatedQDQPair", False, None),
            ("EnableDualQuantNodePairs", False, None),
            ("OpTypesToExcludeOutputQuantization", [], None),
            ("NodesToExcludeOutputQuantization", [], None),
            ("QDQOpTypePerChannelSupportToAxis", {}, None),
            ("RemoveQDQConvClip", True, None),
            ("RemoveQDQConvRelu", True, None),
            ("RemoveQDQConvLeakyRelu", True, None),
            ("RemoveQDQConvPRelu", True, None),
            ("RemoveQDQConvGelu", False, None),
            ("RemoveQDQMulAdd", False, None),
            ("RemoveQDQBetweenOps", None, None),
            ("RemoveQDQInstanceNorm", False, None),
            ("RemoveFusedQDQ", False, None),
        ),
    ),
    (
        "4. Calibration Configuration",
        "Tunes calibration behaviour: range symmetry, moving-average smoothing, percentile thresholds, "
        "random-data input, per-tensor overrides, parallelism, memory/disk trade-offs, and "
        "save-restore checkpointing. Defaults for CalibTensorRangeSymmetric / Percentile / "
        "CalibOptimizeMem / CalibOptimizeDisk / PercentileCandidates depend on calibrate_method and "
        "come from resolve_calibrator_extra_defaults().",
        (
            ("CalibTensorRangeSymmetric", False, None),
            ("CalibMovingAverage", False, None),
            ("CalibMovingAverageConstant", 0.01, None),
            ("Percentile", 99.999, None),
            ("LWPMetric", "mae", None),
            ("PercentileCandidates", [99.99, 99.999, 99.99999], None),
            ("UseRandomData", False, None),
            ("RandomDataReaderInputShape", {}, None),
            ("RandomDataReaderInputDataRange", None, None),
            ("MinMSEModePof2Scale", "All", None),
            ("WeightCalibrateMethod", None, None),
            ("MinMSEModeFloatScale", None, None),
            ("CalibDataSize", None, None),
            ("CalibWorkerNum", 1, None),
            ("CalibOptimizeMem", True, None),
            ("CalibOptimizeDisk", False, None),
            ("NumBins", None, None),
            ("NumQuantizedBins", None, None),
            ("Scenario", None, None),
            ("SaveTensorHistFig", False, None),
            ("CalibPassthroughOpTypes", [], None),
            ("TensorQuantOverrides", {}, None),
            ("SaveAndRestore", None, None),
        ),
    ),
    (
        "5. Bias Handling",
        "Specifies whether and how bias tensors are quantized (int32, int16, or same as weight) "
        "and whether the bias scale is aligned to input × weight scale.",
        (
            ("QuantizeBias", True, None),
            ("Int32Bias", None, None),
            ("Int16Bias", False, None),
            # AdjustBiasScale is read only by ExtendedQDQQuantizer
            # (quark/onnx/quantizers/extended_quantizer.py:565+).
            ("AdjustBiasScale", True, _EXTENDED_QDQ_ONLY),
        ),
    ),
    (
        "6. Scale & Numeric Type",
        "Controls scale representation format (float32, float16, int16/power-of-two) and whether "
        "activations or weights are mapped to a reduced numeric range before casting.",
        (
            ("Int16Scale", False, None),
            ("QuantizeFP16", False, None),
            ("UseFP32Scale", None, None),
            ("ActivationScaled", None, None),
            ("WeightScaled", None, None),
            ("UseUnsignedReLU", False, None),
        ),
    ),
    (
        "7. DPU/NPU Hardware Adaptation",
        "Replaces operators with DPU-compatible approximations (SimulateDPU) and iteratively adjusts "
        "quantization positions (shift, alignment) to satisfy NPU hardware constraints. Effective only "
        "on XINT8QDQQuantizer (enable_npu_cnn=True) or, for AlignConcat/Pool/Pad/Slice and "
        "AlignTranspose/Reshape, ExtendedQDQQuantizer.",
        (
            ("SimulateDPU", True, _NPU_OR_EXTENDED),
            ("ConvertLeakyReluToDPUVersion", False, _NPU_CNN_ONLY),
            ("ConvertSigmoidToHardSigmoid", False, _NPU_CNN_ONLY),
            ("ConvertHardSigmoidToDPUVersion", False, _NPU_CNN_ONLY),
            ("ConvertAvgPoolToDPUVersion", False, _NPU_CNN_ONLY),
            ("ConvertClipToDPUVersion", False, _NPU_CNN_ONLY),
            ("ConvertReduceMeanToDPUVersion", False, _NPU_CNN_ONLY),
            ("ConvertSoftmaxToDPUVersion", False, _NPU_CNN_ONLY),
            ("NPULimitationCheck", True, _NPU_OR_EXTENDED),
            ("MaxLoopNum", 5, _NPU_OR_EXTENDED),
            ("AdjustShiftCut", True, _NPU_CNN_ONLY),
            ("AdjustShiftBias", True, _NPU_CNN_ONLY),
            ("AdjustShiftRead", True, _NPU_CNN_ONLY),
            ("AdjustShiftWrite", True, _NPU_CNN_ONLY),
            ("AdjustHardSigmoid", True, _NPU_CNN_ONLY),
            ("AdjustShiftSwish", True, _NPU_CNN_ONLY),
            ("AlignConcat", None, _NPU_OR_EXTENDED),
            ("AlignPool", None, _NPU_OR_EXTENDED),
            ("AlignPad", None, _NPU_OR_EXTENDED),
            ("AlignSlice", None, _NPU_OR_EXTENDED),
            ("AlignTranspose", False, _EXTENDED_QDQ_ONLY),
            ("AlignReshape", False, _EXTENDED_QDQ_ONLY),
        ),
    ),
    (
        "8. BF16 / BFP / MX Quantization",
        "Options specific to floating-point block quantization formats (BFloat16, BFP16, MX): "
        "boundary clipping, replacing QDQ with Cast for inference speed, block-axis refinement, "
        "and Vaiml compiler export.",
        (
            ("BF16WithClip", False, None),
            ("BF16QDQToCast", None, None),
            ("RefineBlockAxis", None, None),
            ("EnableVaimlBF16", False, None),
        ),
    ),
    (
        "9. MatMul NBits (LLM Weight-Only)",
        "Enables n-bit (typically 4-bit) weight-only quantization for MatMul ops, with algorithm "
        "choice (DEFAULT / GPTQ / HQQ), group size, and accuracy-level sub-parameters.",
        (
            ("UseMatMulNBits", False, None),
            ("MatMulNBitsParams", {}, None),
        ),
    ),
    (
        "10. Runtime & Infrastructure",
        "Configures ORT execution providers for calibration, initializer deduplication, custom op "
        "libraries, fixed output shapes, temp directories, and cryptographic protection of model data.",
        (
            ("ExecutionProviders", None, None),
            ("RemoveInputInit", True, None),
            ("CopySharedInit", None, None),
            ("CopyBiasInit", ["Conv", "ConvTranspose", "Gemm"], None),
            ("AlignEltwiseQuantType", None, None),
            ("FixShapes", None, None),
            ("TmpDir", None, None),
            ("UserCustomOpLibPath", None, None),
            ("EncryptionAlgorithm", None, None),
            ("CryptoMode", False, None),
        ),
    ),
    (
        "11. Debug, Logging & Evaluation",
        "Controls debug output, and optionally evaluates quantization quality by computing "
        "cosine similarity and L2 loss against the original float model.",
        (
            ("DebugMode", False, None),
            ("PrintSummary", True, None),
            ("IgnoreWarnings", True, None),
            ("EvalMetrics", None, None),
            ("EvalDataReader", None, None),
        ),
    ),
)


def _dump_algo_config_full(obj: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"name": getattr(obj, "name", type(obj).__name__)}
    for k, v in obj.__dict__.items():
        if k.startswith("_") or callable(v) or k == "name":
            continue
        d[k] = config_to_dict(v)
    return d


def _format_param_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if is_algo_config(value):
        value = _dump_algo_config_full(value)
    elif isinstance(value, list | tuple) and value and all(is_algo_config(x) for x in value):
        value = [_dump_algo_config_full(x) for x in value]
    if isinstance(value, dict) and any(k is not None and not isinstance(k, str | int | float | bool) for k in value):
        value = [[config_to_dict(k), v] for k, v in value.items()]
    try:
        return pformat(config_to_dict(value), width=80, sort_dicts=False)
    except Exception:
        return repr(value)


def print_user_supplied_configuration(
    *,
    q_config_source: QConfig | QuantizationConfig | None,
    user_extra: dict[str, Any],
    quant_format: QuantFormat | ExtendedQuantFormat,
    activation_type: Any,
    weight_type: Any,
    calibrate_method: Any,
    optimize_model: bool,
    model_input: str | Path | onnx.ModelProto | None,
    model_output: str | Path | None,
    calibration_data_reader: CalibrationDataReader | None,
) -> None:
    """Print the user-supplied configuration block BEFORE calibration runs.

    Values shown are the user's input as forwarded to ``quantize_static`` —
    they may still be rewritten by downstream normalization (FP16 detection,
    ``TensorQuantOverrides`` upgrade, ``Int16Scale`` calibrate_method flip,
    etc.). The final effective ``extra_options`` is printed after
    ``run_static_quantization`` completes via
    :func:`print_effective_quantization_summary`.

    Only invoked from the QConfig path.
    """
    if user_extra.get("CryptoMode", False):
        return
    if not user_extra.get("PrintSummary", True):
        return

    # EnableNPUCnn is a legacy/internal extra_options key that is not part of
    # the public documentation, so we do not surface it as its own summary
    # row. We still derive enable_npu_cnn from it because
    # _resolve_active_quantizer needs it to pick the correct quantizer class.
    enable_npu_cnn = bool(user_extra.get("EnableNPUCnn", False))
    active_quantizer = _resolve_active_quantizer(quant_format, calibrate_method, enable_npu_cnn)

    model_input_display = type(model_input).__name__ if isinstance(model_input, onnx.ModelProto) else model_input
    crd_display = type(calibration_data_reader).__name__ if calibration_data_reader is not None else None
    main_rows: list[tuple[str, Any]] = [
        ("model_input", _format_param_value(model_input_display)),
        ("model_output", _format_param_value(model_output)),
        ("calibration_data_reader", _format_param_value(crd_display)),
        ("quant_format", _format_param_value(quant_format)),
        ("activation_type", _format_param_value(activation_type)),
        ("weight_type", _format_param_value(weight_type)),
        ("calibrate_method", _format_param_value(calibrate_method)),
        ("optimize_model", _format_param_value(optimize_model)),
        ("active_quantizer", _format_param_value(active_quantizer)),
    ]
    if q_config_source is not None:
        existing = {label for label, _ in main_rows}
        for f in fields(q_config_source):
            if f.name in existing or f.name == "extra_options":
                continue
            main_rows.append((f.name, _format_param_value(getattr(q_config_source, f.name, None))))
    main_rows.append(("extra_options", _format_param_value(user_extra)))

    _log_time_info()
    _log_os_cpu_info()
    _log_tools_version_info()
    _log_param_section("Quantized Configuration information:", main_rows)


def print_effective_quantization_summary(
    *,
    user_extra: dict[str, Any],
    effective_extra: dict[str, Any],
    effective_quant_format: QuantFormat | ExtendedQuantFormat,
    effective_calibrate_method: Any,
    effective_enable_npu_cnn: bool = False,
    effective_activation_type: Any = None,
    effective_weight_type: Any = None,
    exception_context: str | None = None,
) -> None:
    """Print the 11-category **effective** ``extra_options`` summary AFTER
    ``run_static_quantization`` finishes (or after an exception inside it).

    Single source of truth: this printer reads ``effective_extra`` directly
    instead of replicating ``quantize_static``'s default-resolution logic.
    Calibrator-internal defaults come from
    :func:`quark.onnx.calibration.calibrators.resolve_calibrator_extra_defaults`
    — the same helper the calibrator factories use — so summary and runtime
    can never drift.

    ``exception_context`` is a short tag (e.g. exception class name) appended
    to the banner when the printer is invoked from a failure path, so a
    summary printed for a failed run is not mistaken for a successful one.
    """
    if effective_extra.get("CryptoMode", False):
        return
    if not effective_extra.get("PrintSummary", True):
        return

    # Calibrator-internal defaults are stored under lowercase keys (matching
    # what create_calibrator_* expects). Map them back to the PascalCase names
    # the summary categories use; user-supplied PascalCase values win.
    from quark.onnx.calibration.calibrators import resolve_calibrator_extra_defaults

    overlay = resolve_calibrator_extra_defaults(effective_calibrate_method, effective_extra, emit_warnings=False)
    display_extra: dict[str, Any] = dict(effective_extra)
    for pascal_key, lower_key in _CALIB_KEY_TO_LOWER:
        if lower_key in overlay and pascal_key not in display_extra:
            display_extra[pascal_key] = overlay[lower_key]

    active_quantizer = _resolve_active_quantizer(
        effective_quant_format,
        effective_calibrate_method,
        effective_enable_npu_cnn,
    )

    if exception_context is None:
        logger.info("The categories below reflect the effective extra_options consumed by the active quantizer.")
    else:
        logger.info(
            "The categories below reflect extra_options at the point of failure "
            f"(exception: {exception_context}) — values may be partially mutated."
        )

    # consumed_by mismatch: user_set → loud "(not effective; ignored by X)",
    # default → quiet "(default; ignored by X)".
    # CopyBiasInit: preproc.py:110-126 — only runs for int8/int16 + native CalibrationMethod.
    from onnxruntime.quantization.calibrate import CalibrationMethod as _CM
    from onnxruntime.quantization.quant_utils import QuantType as _QT

    from quark.onnx.quantization.quant_utils import ExtendedQuantType as _EQT

    _copy_bias_ok_types = {
        _QT.QUInt8,
        _QT.QInt8,
        _QT.QUInt16,
        _QT.QInt16,
        _EQT.QInt8,
        _EQT.QUInt8,
        _EQT.QInt16,
        _EQT.QUInt16,
    }

    # consumed_by annotations only make sense when we know the active quantizer.
    annotate = bool(active_quantizer)
    for title, description, items in _QCFG_CATEGORIES:
        rows: list[tuple[str, Any]] = []
        for key, default, consumed_by in items:
            value = display_extra.get(key, default)
            if key == "SaveAndRestore":
                if (
                    value is not None
                    and "SaveAndRestore" not in user_extra
                    and user_extra.get("TensorsRangeFile") is not None
                ):
                    rows.append((key, f"{_format_param_value(value)}  (from deprecated TensorsRangeFile alias)"))
                else:
                    rows.append((key, value))
                continue
            if key == "CopyBiasInit":
                copy_bias_active = (
                    value is not None
                    and effective_weight_type in _copy_bias_ok_types
                    and effective_activation_type in _copy_bias_ok_types
                    and effective_calibrate_method in _CM
                )
                if copy_bias_active:
                    rows.append((key, _format_param_value(value)))
                elif key in user_extra:
                    rows.append(
                        (
                            key,
                            f"{_format_param_value(value)}  "
                            "(only takes effect under int8/int16 dtypes with "
                            "calibrate_method in {MinMax, Entropy, Percentile, Distribution})",
                        )
                    )
                # else: user did not set it AND it is a no-op for this config → skip.
                continue
            not_consumed = annotate and consumed_by is not None and active_quantizer not in consumed_by
            if not_consumed and key in user_extra:
                label = _CONSUMED_BY_LABEL.get(consumed_by, ", ".join(sorted(consumed_by)))
                rows.append((key, f"{_format_param_value(value)}  (only takes effect under {label} config)"))
            elif not_consumed:
                # User did not set it AND current quantizer does not consume it → skip the row entirely.
                continue
            else:
                rows.append((key, value))
        if not rows:
            continue
        _log_param_section(title, rows, description=description, sort_by_type=True)
