#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import DEQUANT_OP_NAME, QuantFormat, QuantType

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
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


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
    """
    print os_cpu, time, tool_version, quantized_configuration information.
    """

    def _print_time_info() -> None:
        """
        print time information.
        """
        now = datetime.now()
        print("[QUARK_INFO]: Time information:")
        print(now)

    def _print_os_cpu_info() -> None:
        """
        print os_cpu information.
        """
        system_info = platform.system()
        node_info = platform.node()
        release_info = platform.release()
        version_info = platform.version()
        machine_info = platform.machine()
        processor_info = platform.processor()
        print("[QUARK_INFO]: OS and CPU information:")
        print("{:>50}".format("system ---"), system_info)
        print("{:>50}".format("node ---"), node_info)
        print("{:>50}".format("release ---"), release_info)
        print("{:>50}".format("version ---"), version_info)
        print("{:>50}".format("machine ---"), machine_info)
        print("{:>50}".format("processor ---"), processor_info)

    def _print_tools_version_info() -> None:
        """
        print tools version information.
        """
        python_version = platform.python_version()
        onnx_version = onnx.__version__  # type: ignore[attr-defined]
        onnxruntime_version = ort.__version__
        quark_onnx_version = __version__
        print("[QUARK_INFO]: Tools version information:")
        print("{:>50}".format("python ---"), python_version)
        print("{:>50}".format("onnx ---"), onnx_version)
        print("{:>50}".format("onnxruntime ---"), onnxruntime_version)
        print("{:>50}".format("quark.onnx ---"), quark_onnx_version)

    def _print_quantized_config_info() -> None:
        """
        print quantized configuration information.
        """
        print("[QUARK_INFO]: Quantized Configuration information:")
        print(
            "{:>50}".format("model_input ---"),
            type(model_input) if isinstance(model_input, onnx.ModelProto) else model_input,
        )
        print("{:>50}".format("model_output ---"), model_output)
        print("{:>50}".format("calibration_data_reader ---"), calibration_data_reader)
        print("{:>50}".format("calibration_data_path ---"), calibration_data_path)
        print("{:>50}".format("quant_format ---"), quant_format)
        print("{:>50}".format("input_nodes ---"), input_nodes)
        print("{:>50}".format("output_nodes ---"), output_nodes)
        print("{:>50}".format("op_types_to_quantize ---"), op_types_to_quantize)
        print("{:>50}".format("extra_op_types_to_quantize ---"), extra_op_types_to_quantize)
        print("{:>50}".format("per_channel ---"), per_channel)
        print("{:>50}".format("reduce_range ---"), reduce_range)
        print("{:>50}".format("activation_type ---"), activation_type)
        print("{:>50}".format("weight_type ---"), weight_type)
        print("{:>50}".format("nodes_to_quantize ---"), nodes_to_quantize)
        print("{:>50}".format("nodes_to_exclude ---"), nodes_to_exclude)
        print("{:>50}".format("subgraphs_to_exclude ---"), subgraphs_to_exclude)
        print("{:>50}".format("optimize_model ---"), optimize_model)
        print("{:>50}".format("use_external_data_format ---"), use_external_data_format)
        print("{:>50}".format("calibrate_method ---"), calibrate_method)
        print("{:>50}".format("execution_providers ---"), execution_providers)
        print("{:>50}".format("enable_npu_cnn ---"), enable_npu_cnn)
        print("{:>50}".format("enable_npu_transformer ---"), enable_npu_transformer)
        print("{:>50}".format("specific_tensor_precision ---"), specific_tensor_precision)
        print("{:>50}".format("debug_mode ---"), debug_mode)
        print("{:>50}".format("convert_fp16_to_fp32 ---"), convert_fp16_to_fp32)
        print("{:>50}".format("convert_nchw_to_nhwc ---"), convert_nchw_to_nhwc)
        print("{:>50}".format("include_cle ---"), include_cle)
        print("{:>50}".format("include_sq ---"), include_sq)
        print("{:>50}".format("include_rotation ---"), include_rotation)
        print("{:>50}".format("include_fast_ft ---"), include_fast_ft)
        print("{:>50}".format("extra_options ---"), extra_options)

    if crypto_mode:
        return  # Print nothing in crypto mode

    try:
        _print_time_info()
        _print_os_cpu_info()
        _print_tools_version_info()
        _print_quantized_config_info()
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

    def _print_time_info() -> None:
        """
        print time information.
        """
        now = datetime.now()
        print("[QUARK_INFO]: Time information:")
        print(now)

    def _print_os_cpu_info() -> None:
        """
        print os_cpu information.
        """
        system_info = platform.system()
        node_info = platform.node()
        release_info = platform.release()
        version_info = platform.version()
        machine_info = platform.machine()
        processor_info = platform.processor()
        print("[QUARK_INFO]: OS and CPU information:")
        print("{:>50}".format("system ---"), system_info)
        print("{:>50}".format("node ---"), node_info)
        print("{:>50}".format("release ---"), release_info)
        print("{:>50}".format("version ---"), version_info)
        print("{:>50}".format("machine ---"), machine_info)
        print("{:>50}".format("processor ---"), processor_info)

    def _print_tools_version_info() -> None:
        """
        print tools version information.
        """
        python_version = platform.python_version()
        onnx_version = onnx.__version__  # type: ignore[attr-defined]
        onnxruntime_version = ort.__version__
        quark_onnx_version = __version__
        print("[QUARK_INFO]: Tools version information:")
        print("{:>50}".format("python ---"), python_version)
        print("{:>50}".format("onnx ---"), onnx_version)
        print("{:>50}".format("onnxruntime ---"), onnxruntime_version)
        print("{:>50}".format("quark.onnx ---"), quark_onnx_version)

    def _print_quantized_config_info() -> None:
        """
        print quantized configuration information.
        """
        print("[QUARK_INFO]: Quantized Configuration information:")
        print(
            "{:>50}".format("model_input ---"),
            type(model_input) if isinstance(model_input, onnx.ModelProto) else model_input,
        )
        print("{:>50}".format("model_output ---"), model_output)
        print("{:>50}".format("op_types_to_quantize ---"), op_types_to_quantize)
        print("{:>50}".format("per_channel ---"), per_channel)
        print("{:>50}".format("reduce_range ---"), reduce_range)
        print("{:>50}".format("weight_type ---"), weight_type)
        print("{:>50}".format("nodes_to_quantize ---"), nodes_to_quantize)
        print("{:>50}".format("nodes_to_exclude ---"), nodes_to_exclude)
        print("{:>50}".format("subgraphs_to_exclude ---"), subgraphs_to_exclude)
        print("{:>50}".format("use_external_data_format ---"), use_external_data_format)
        print("{:>50}".format("debug_mode ---"), debug_mode)
        print("{:>50}".format("extra_options ---"), extra_options)

    if crypto_mode:
        return  # Print nothing in crypto mode

    try:
        _print_time_info()
        _print_os_cpu_info()
        _print_tools_version_info()
        _print_quantized_config_info()
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
            21: "UINT4",
            22: "INT4",
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
