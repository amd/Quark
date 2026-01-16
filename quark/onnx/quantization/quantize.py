#
# Modifications copyright(c) 2023 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import time
from pathlib import Path
from typing import Any

import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import (
    QuantFormat,
    QuantType,
    model_has_pre_process_metadata,
    save_and_reload_model_with_shape_infer,
)

from quark.onnx.calibration import (
    CachedDataReader,
    Int16Method,
    PowerOfTwoMethod,
    fake_calibration,
    get_data_reader,
    load_tensors_range,
    run_calibration,
    save_tensors_range,
)
from quark.onnx.postprocess import apply_post_process
from quark.onnx.preprocess import apply_pre_process
from quark.onnx.quantization.input_check import (
    check_crypto_mode_arguments,
    check_fast_fintune_arguments,
    check_static_quant_arguments,
)
from quark.onnx.quantization.output_eval import eval_metrics
from quark.onnx.quantizers import (
    get_dynamic_op_types,
    get_static_op_types,
    run_dynamic_quantization,
    run_matmul_nbits_quantization,
    run_static_quantization,
)
from quark.onnx.utils.file_utils import save_quantized_info, update_crypto_mode
from quark.onnx.utils.model_utils import (
    cache_onnx_model_and_infer_shapes,
    check_onnx_model,
    run_onnx_model,
    save_onnx_model_with_external_data,
    update_user_custom_op_lib_paths,
)
from quark.onnx.utils.print_utils import (
    print_fp32_nodes,
    print_quantize_dynamic_info,
    print_quantize_static_info,
    print_quantized_info,
)
from quark.onnx.utils.system_utils import (
    create_tmp_dir,
    update_tmp_dir,
)
from quark.shares.utils.log import ScreenLogger, log_errors

from .quant_utils import (
    ExtendedQuantFormat,
    ExtendedQuantType,
    VitisQuantFormat,
    VitisQuantType,
    check_model_is_fp16,
    check_model_quantizable,
    fp32_nodes,
    get_all_target_nodes,
    get_eltwise_op,
    get_exclude_nodes,
    get_matmul_nodes_without_weights,
    skip_node_with_inf_tensor,
)

logger = ScreenLogger(__name__)


@log_errors
def quantize_static(
    model_input: str | Path | onnx.ModelProto,
    model_output: str | Path | None = None,
    calibration_data_reader: CalibrationDataReader | None = None,
    calibration_data_path: str | None = None,
    quant_format: QuantFormat | ExtendedQuantFormat = QuantFormat.QDQ,
    calibrate_method: CalibrationMethod | PowerOfTwoMethod | Int16Method = CalibrationMethod.MinMax,
    input_nodes: list[str] | None = [],
    output_nodes: list[str] | None = [],
    op_types_to_quantize: list[str] | None = [],
    extra_op_types_to_quantize: list[str] = [],
    per_channel: bool = False,
    reduce_range: bool = False,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    subgraphs_to_exclude: list[tuple[list[str]]] = [],
    optimize_model: bool = True,
    use_external_data_format: bool = False,
    execution_providers: list[str] | None = ["CPUExecutionProvider"],
    enable_dpu: bool = False,
    enable_npu_cnn: bool = False,
    enable_npu_transformer: bool = False,
    specific_tensor_precision: bool = False,
    convert_fp16_to_fp32: bool = False,
    convert_nchw_to_nhwc: bool = False,
    debug_mode: bool = False,
    crypto_mode: bool = False,
    include_cle: bool = True,
    include_sq: bool = False,
    include_rotation: bool = False,
    include_fast_ft: bool = False,
    include_auto_mp: bool = False,
    print_summary: bool = True,
    extra_options: dict[str, Any] | None = {},
) -> onnx.ModelProto | None:
    """Qantize a given onnx model using static quantization. This api will return an onnx.ModelProto format quantized model
    if the argument 'model_output' is None or 'crypto_mode' is True.
    """

    update_crypto_mode(crypto_mode)
    e2e_quantize_start_time = time.perf_counter()
    model_input_info = type(model_input) if isinstance(model_input, onnx.ModelProto) else model_input
    save_quantized_info(
        [
            [f"{model_input_info} quantization quantization info", time.strftime("%Y-%m-%d %H:%M:%S")],
            ["quantization stage", "time consumed(s)", "sub stage", "time consumed(s)"],
        ],
        write_mode="w",
    )

    if nodes_to_quantize is None:
        nodes_to_quantize = []
    if nodes_to_exclude is None:
        nodes_to_exclude = []
    if subgraphs_to_exclude is None:
        subgraphs_to_exclude = []
    if extra_options is None:
        extra_options = {}

    update_tmp_dir(extra_options.get("TmpDir"))
    update_user_custom_op_lib_paths(extra_options.get("UserCustomOpLibPath"))

    float_model: onnx.ModelProto = model_input if isinstance(model_input, onnx.ModelProto) else onnx.load(model_input)
    quant_model: onnx.ModelProto = onnx.ModelProto()  # the quantized model

    if not use_external_data_format:
        if float_model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
            use_external_data_format = True
            logger.warning("The model size is bigger than 2GB, have set use_external_data_format to True.")

    check_static_quant_arguments(
        float_model, quant_format, activation_type, weight_type, calibrate_method, extra_options
    )

    if include_fast_ft and include_auto_mp is False:
        check_fast_fintune_arguments(activation_type, weight_type, extra_options)

    if crypto_mode:
        check_crypto_mode_arguments(model_input, use_external_data_format, extra_options)
        if optimize_model:
            optimize_model = False
            logger.warning("Can not optimize the model since we can't save exposed data to disk in crypto mode.")

    encrypt_algo = extra_options.get("EncryptionAlgorithm") if crypto_mode else None
    secret_key = os.urandom(48) if crypto_mode else None  # It's used to encrypt and decrypt data

    cache_dir = create_tmp_dir(prefix="quark_onnx.quant.")
    cache_path = Path(cache_dir.name).joinpath("cache_model.onnx").as_posix()
    float_model = cache_onnx_model_and_infer_shapes(
        float_model, cache_path, use_external_data_format, encrypt_algo, secret_key
    )

    if not convert_fp16_to_fp32 and not extra_options.get("QuantizeFP16", False):
        if check_model_is_fp16(float_model):
            extra_options["QuantizeFP16"] = True
            logger.warning(
                "Detected that the input model is an FP16 model. "
                "It will proceed with quantization based on the FP16 model."
            )
    quantize_fp16 = extra_options.get("QuantizeFP16", False)
    if quantize_fp16 and optimize_model:
        optimize_model = False
        logger.warning(
            "The parameter optimize_model is set to False automatically when the parameter QuantizeFP16 is set to True."
        )

    if isinstance(quant_format, VitisQuantFormat):
        if quant_format == VitisQuantFormat.BFPFixNeuron:
            weight_type = ExtendedQuantType.QBFP
            activation_type = ExtendedQuantType.QBFP
        elif quant_format == VitisQuantFormat.MXFixNeuron:
            weight_type = ExtendedQuantType.QMX
            activation_type = ExtendedQuantType.QMX
        quant_format = ExtendedQuantFormat.QDQ
        logger.warning("VitisQuantFormat will be deprecated in future versions, use ExtendedQuantFormat instead.")

    if isinstance(weight_type, VitisQuantType):
        weight_type = ExtendedQuantType(weight_type.value)
        logger.warning("VitisQuantType will be deprecated in future versions, use ExtendedQuantType instead.")
    if isinstance(activation_type, VitisQuantType):
        activation_type = ExtendedQuantType(activation_type.value)
        logger.warning("VitisQuantType will be deprecated in future versions, use ExtendedQuantType instead.")

    if enable_dpu:
        logger.warning("The 'enable_dpu' will be deprecated in future versions. Please use 'enable_npu_cnn' instead.")
        enable_npu_cnn = enable_dpu

    print_quantize_static_info(
        model_input,
        model_output,
        calibration_data_reader,
        calibration_data_path,
        quant_format,
        input_nodes,
        output_nodes,
        op_types_to_quantize,
        extra_op_types_to_quantize,
        per_channel,
        reduce_range,
        activation_type,
        weight_type,
        nodes_to_quantize,
        nodes_to_exclude,
        subgraphs_to_exclude,
        optimize_model,
        use_external_data_format,
        calibrate_method,
        execution_providers,
        enable_npu_cnn,
        enable_npu_transformer,
        specific_tensor_precision,
        debug_mode,
        crypto_mode,
        convert_fp16_to_fp32,
        convert_nchw_to_nhwc,
        include_cle,
        include_sq,
        include_rotation,
        include_fast_ft,
        extra_options,
    )

    check_onnx_model(float_model)

    fp32_nodes_dict = fp32_nodes(float_model)

    nodes_to_exclude = get_all_target_nodes(float_model, nodes_to_exclude + subgraphs_to_exclude)

    if input_nodes or output_nodes:
        if nodes_to_exclude:
            nodes_to_exclude += get_exclude_nodes(float_model, input_nodes, output_nodes)
        else:
            nodes_to_exclude = get_exclude_nodes(float_model, input_nodes, output_nodes)

    if extra_options.get("MatMulConstBOnly", enable_npu_transformer):
        nodes_to_exclude += get_matmul_nodes_without_weights(float_model)

    skip_node_with_inf_tensor_list = skip_node_with_inf_tensor(float_model)
    nodes_to_exclude.extend(skip_node_with_inf_tensor_list)

    op_types_to_quantize = get_static_op_types(
        float_model,
        op_types_to_quantize,
        extra_op_types_to_quantize,
        enable_npu_cnn,
        enable_npu_transformer,
        quant_format,
        extra_options,
    )

    if not check_model_quantizable(float_model, op_types_to_quantize, nodes_to_exclude):
        logger.warning("No quantizable ops in this model, quantization is skipped.")
        if model_output is None or crypto_mode:
            return float_model
        else:
            save_onnx_model_with_external_data(
                float_model, model_output, save_as_external_data=use_external_data_format
            )
            return None

    if extra_options.get("AlignEltwiseQuantType"):
        if (
            enable_npu_cnn is False
            and enable_npu_transformer is False
            and enable_dpu is False
            and quant_format == ExtendedQuantFormat.QDQ
        ):
            if extra_options.get("TensorQuantOverrides") is None:
                extra_options["TensorQuantOverrides"] = {}
            eltwise_tensors = get_eltwise_op(float_model)
            for tensor_name in eltwise_tensors:
                if tensor_name in extra_options["TensorQuantOverrides"]:
                    for override in extra_options["TensorQuantOverrides"][tensor_name]:
                        override["quant_type"] = activation_type
                else:
                    extra_options["TensorQuantOverrides"][tensor_name] = [{"quant_type": activation_type}]
            logger.info(
                "The parameter AlignEltwiseQuantType takes effect, "
                "the weights of nodes will be quantized with the activation quant type "
                "if the operation type is in [Mul, Div, Add, Sub, Min, Max]."
            )
        else:
            logger.warning(
                "The parameter AlignEltwiseQuantType only takes effect "
                "when quant_format is ExtendedQuantFormat.QDQ and enable_npu_cnn is False "
                "and enable_npu_transformer is False and enable_dpu is False."
            )

    if extra_options.get("TensorQuantOverrides") and quant_format is QuantFormat.QDQ:
        logger.warning(
            "The option 'TensorQuantOverrides' is enabled, the quant_format will be forced to ExtendedQuantFormat.QDQ, "
            "and flags enable_npu_cnn and enable_npu_transformer will be unavailable."
        )
        quant_format = ExtendedQuantFormat.QDQ

    # TODO: to remove this patch
    if (
        enable_npu_cnn
        or enable_npu_transformer
        or (
            quant_format is ExtendedQuantFormat.QDQ
            and not extra_options.get("BF16QDQToCast", False)
            and not extra_options.get("EnableVaimlBF16", False)
        )
    ):
        if "ConvertSplitToSlice" not in extra_options:
            extra_options["ConvertSplitToSlice"] = True
        if "ConvertBNToConv" not in extra_options:
            extra_options["ConvertBNToConv"] = True
        if "ConvertReduceMeanToGlobalAvgPool" not in extra_options:
            extra_options["ConvertReduceMeanToGlobalAvgPool"] = True
        if "SplitLargeKernelPool" not in extra_options:
            extra_options["SplitLargeKernelPool"] = True

    data_reader = get_data_reader(float_model, calibration_data_reader, calibration_data_path, extra_options)
    cached_data_reader = CachedDataReader(data_reader, None, convert_nchw_to_nhwc, quantize_fp16)

    float_model = apply_pre_process(
        float_model,
        Path(cache_path),
        cached_data_reader,
        calibrate_method=calibrate_method,
        activation_type=activation_type,
        weight_type=weight_type,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        op_types_to_quantize=op_types_to_quantize,
        use_external_data_format=use_external_data_format,
        convert_fp16_to_fp32=convert_fp16_to_fp32,
        convert_nchw_to_nhwc=convert_nchw_to_nhwc,
        optimize_model_flag=optimize_model and not crypto_mode,
        include_cle=include_cle,
        include_sq=include_sq,
        include_rotation=include_rotation,
        extra_options=extra_options,
    )

    cached_data_reader.reset_iter()

    topo_model = ONNXModel(float_model)
    topo_model.topological_sort()
    float_model = cache_onnx_model_and_infer_shapes(
        topo_model.model, cache_path, use_external_data_format, encrypt_algo, secret_key
    )

    tensors_range_file = extra_options.get("TensorsRangeFile")
    skip_calibration = False
    if (
        extra_options.get("UseMatMulNBits", False)
        or (
            activation_type
            in [ExtendedQuantType.QBFloat16, ExtendedQuantType.QFloat16, ExtendedQuantType.QBFP, ExtendedQuantType.QMX]
            and not extra_options.get("ActivationScaled", False)
        )
        or (tensors_range_file is not None and os.path.exists(tensors_range_file) and (not crypto_mode))
    ):
        skip_calibration = True
    else:
        try:
            run_onnx_model(float_model, cached_data_reader)
            cached_data_reader.reset_iter()
        except Exception as e:
            logger.error(f"Run the float model failed due to an error: {e}, please check your model and data reader.")
            return None

    if not skip_calibration:
        tensors_range = run_calibration(
            float_model,
            cached_data_reader,
            op_types_to_quantize,
            activation_type,
            calibrate_method,
            use_external_data_format,
            execution_providers,
            extra_options,
        )
        cached_data_reader.reset_iter()

        if tensors_range_file is not None and not crypto_mode:
            save_tensors_range(tensors_range, tensors_range_file)
    else:
        if tensors_range_file is not None and not crypto_mode:
            tensors_range = load_tensors_range(tensors_range_file)
        else:
            tensors_range = fake_calibration(float_model)

    if extra_options.get("UseMatMulNBits", False):
        quant_model = run_matmul_nbits_quantization(float_model, cached_data_reader, extra_options)
        cached_data_reader.reset_iter()
    else:
        if extra_options.get("Int16Scale", False):
            if enable_npu_cnn:
                logger.warning("Int16Scale cannot be used simultaneously with enable_npu_cnn=True")
            else:
                calibrate_method = Int16Method.MinMax

        quant_model = run_static_quantization(
            float_model,
            tensors_range,
            per_channel,
            reduce_range,
            weight_type,
            activation_type,
            enable_npu_cnn,
            enable_npu_transformer,
            quant_format,
            calibrate_method,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )

    float_model = topo_model.model
    quant_model = apply_post_process(
        float_model,
        quant_model,
        cached_data_reader,
        calibrate_method=calibrate_method,
        activation_type=activation_type,
        weight_type=weight_type,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        op_types_to_quantize=op_types_to_quantize,
        use_external_data_format=use_external_data_format,
        include_auto_mp=include_auto_mp,
        include_fast_ft=include_fast_ft,
        extra_options=extra_options,
    )
    cached_data_reader.reset_iter()

    e2e_quantized_end_time = time.perf_counter()
    e2e_quantize_time_consumed = e2e_quantized_end_time - e2e_quantize_start_time
    save_quantized_info([["e2e", e2e_quantize_time_consumed]], write_mode="a")
    if not crypto_mode:
        logger.info(f"Quark_latency_profiler: e2e quantization time consumed:{e2e_quantize_time_consumed:1f}")

    if print_summary and fp32_nodes_dict and not crypto_mode:
        shared_init_optypes = extra_options.get("CopySharedInit")
        print_fp32_nodes(fp32_nodes_dict, model_output)
        print_quantized_info(quant_model, debug_mode, shared_init_optypes)

    if "EvalMetrics" in extra_options:
        if "EvalDataReader" in extra_options:
            eval_data_reader = extra_options["EvalDataReader"]
        else:
            eval_data_reader = cached_data_reader

        eval_metrics(model_input, quant_model, eval_data_reader, execution_providers, use_external_data_format)

    if model_output is None or crypto_mode:
        quant_model = onnx.shape_inference.infer_shapes(quant_model)
        return quant_model

    quant_model = save_and_reload_model_with_shape_infer(quant_model)
    save_onnx_model_with_external_data(quant_model, model_output, save_as_external_data=use_external_data_format)
    return None


def quantize_dynamic(
    model_input: str | Path | onnx.ModelProto,
    model_output: str | Path | None = None,
    op_types_to_quantize: list[str] | None = [],
    per_channel: bool = False,
    reduce_range: bool = False,
    weight_type: QuantType = QuantType.QInt8,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    subgraphs_to_exclude: list[tuple[list[str]]] = [],
    use_external_data_format: bool = False,
    debug_mode: bool = False,
    crypto_mode: bool = False,
    extra_options: dict[str, Any] | None = {},
) -> onnx.ModelProto | None:
    """Qantize a given onnx model using dynamic quantization. This api will return an onnx.ModelProto format quantized model
       if the argument 'model_output' is None or 'crypto_mode' is True.

    Args:
        model_input: file path of model or ModelProto to quantize
        model_output: file path of quantized model
        op_types_to_quantize:
            specify the types of operators to quantize, like ['Conv'] to quantize Conv only.
            It quantizes all supported operators by default.
        per_channel: quantize weights per channel
        reduce_range:
            quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine,
            especially for per-channel mode
        weight_type:
            quantization data type of weight. Please refer to
            https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
        nodes_to_quantize:
            List of nodes names to quantize. When this list is not None only the nodes in this list
            are quantized.
            example:
            [
                'Conv__224',
                'Conv__252'
            ]
        nodes_to_exclude:
            List of nodes names to exclude. The nodes in this list will be excluded from quantization
            when it is not None.
        subgraphs_to_exclude:
            List of start and end nodes names of subgraphs to exclude. The nodes matched by the subgraphs will be excluded from quantization
            when it is not None.
        use_external_data_format: option used for large size (>2GB) model. Set to False by default.
        extra_options:
            key value pair dictionary for various options in different case. Current used:
                extra.Sigmoid.nnapi = True/False  (Default is False)
                ActivationSymmetric = True/False: symmetrize calibration data for activations (default is False).
                WeightSymmetric = True/False: symmetrize calibration data for weights (default is True).
                EnableSubgraph = True/False :
                    Default is False. If enabled, subgraph will be quantized. Dynamic mode currently is supported. Will
                    support more in the future.
                ForceQuantizeNoInputCheck = True/False :
                    By default, some latent operators like maxpool, transpose, do not quantize if their input is not
                    quantized already. Setting to True to force such operator always quantize input and so generate
                    quantized output. Also the True behavior could be disabled per node using the nodes_to_exclude.
                MatMulConstBOnly = True/False:
                    Default is True for dynamic mode. If enabled, only MatMul with const B will be quantized.
    """

    extra_options = extra_options or {}
    nodes_to_exclude = nodes_to_exclude or []
    subgraphs_to_exclude = subgraphs_to_exclude or []
    nodes_to_quantize = nodes_to_quantize or []
    op_types_to_quantize = op_types_to_quantize or []

    update_tmp_dir(extra_options.get("TmpDir"))

    float_model: onnx.ModelProto = model_input if isinstance(model_input, onnx.ModelProto) else onnx.load(model_input)
    quant_model: onnx.ModelProto = onnx.ModelProto()  # the quantized model

    if not use_external_data_format:
        if float_model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
            use_external_data_format = True
            logger.warning("The model size is bigger than 2GB, have set use_external_data_format to True.")

    if crypto_mode:
        check_crypto_mode_arguments(model_input, use_external_data_format, extra_options)

    encrypt_algo = extra_options.get("EncryptionAlgorithm", None) if crypto_mode else None
    secret_key = os.urandom(48) if crypto_mode else None  # It's used to encrypt and decrypt data

    cache_dir = create_tmp_dir(prefix="quark_onnx.quant.")
    cache_path = Path(cache_dir.name).joinpath("cache_model.onnx").as_posix()
    float_model = cache_onnx_model_and_infer_shapes(
        float_model, cache_path, use_external_data_format, encrypt_algo, secret_key
    )

    op_types_to_quantize = get_dynamic_op_types(op_types_to_quantize)

    print_quantize_dynamic_info(
        model_input,
        model_output,
        op_types_to_quantize,
        per_channel,
        reduce_range,
        weight_type,
        nodes_to_quantize,
        nodes_to_exclude,
        subgraphs_to_exclude,
        use_external_data_format,
        debug_mode,
        crypto_mode,
        extra_options,
    )

    nodes_to_exclude = get_all_target_nodes(float_model, nodes_to_exclude + subgraphs_to_exclude)

    pre_processed: bool = model_has_pre_process_metadata(float_model)
    if not pre_processed:
        logger.warning(
            "Please consider to run pre-processing before quantization. Refer to example: "
            "https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification"
            "/cpu/ReadMe.md "
        )

    if "MatMulConstBOnly" not in extra_options:
        extra_options["MatMulConstBOnly"] = True

    quant_model = run_dynamic_quantization(
        float_model,
        per_channel,
        reduce_range,
        weight_type,
        QuantType.QUInt8,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options,
    )

    if model_output is None or crypto_mode:
        quant_model = onnx.shape_inference.infer_shapes(quant_model)
        return quant_model

    quant_model = save_and_reload_model_with_shape_infer(quant_model)
    save_onnx_model_with_external_data(quant_model, model_output, save_as_external_data=use_external_data_format)
    return None
