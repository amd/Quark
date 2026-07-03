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

from quark import __version__
from quark.common.utils.log import ScreenLogger, log_errors
from quark.onnx.calibration import (
    CachedDataReader,
    Int16Method,
    PowerOfTwoMethod,
    fake_calibration,
    get_data_reader,
    nearest_non_passthrough_ancestor_mapping,
    run_calibration,
    update_tensors_range_with_dependencies,
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
from quark.onnx.utils.file_utils import (
    save_and_restore_func,
    update_crypto_mode,
)
from quark.onnx.utils.model_utils import (
    cache_onnx_model_and_infer_shapes,
    check_onnx_model,
    check_shared_initializers,
    run_onnx_model,
    save_onnx_model_with_external_data,
    update_user_custom_op_lib_paths,
)
from quark.onnx.utils.print_utils import (
    print_effective_quantization_summary,
    print_fp32_nodes,
    print_quantize_dynamic_info,
    print_quantize_static_info,
    print_quantized_info,
)
from quark.onnx.utils.system_utils import (
    create_tmp_dir,
    update_tmp_dir,
)

from .quant_utils import (
    ExtendedQuantFormat,
    ExtendedQuantType,
    VitisQuantFormat,
    VitisQuantType,
    check_model_is_fp16,
    check_model_quantizable,
    fp32_nodes,
    get_all_target_nodes,
    get_all_tensor_names,
    get_eltwise_op,
    get_exclude_nodes,
    get_matmul_nodes_without_weights,
    get_pre_defined_preprocess_config,
    skip_node_with_inf_tensor,
)

logger = ScreenLogger(__name__)


def prune_stale_tensor_quant_overrides(
    model: onnx.ModelProto,
    extra_options: dict[str, Any],
) -> list[str]:
    """Drop ``TensorQuantOverrides`` entries whose tensor name is not present in ``model``.

    Pruning keeps the override map consistent with the graph and avoids the
    onnxruntime error
    ``Tensor 'X' in TensorQuantOverrides is not present in the model``. Since
    pre-process honors user-supplied override keys (the referenced tensors are
    not folded away), a stale entry is almost always a typo in the override key.

    :param model: the (post-pre-process) model to validate against.
    :param extra_options: ``quantize_static`` extra_options dict; mutated in-place.
    :return: list of tensor names that were dropped (empty if nothing to prune).
    """
    overrides = extra_options.get("TensorQuantOverrides")
    if not overrides:
        return []

    valid_tensor_names = get_all_tensor_names(model)
    stale = [name for name in list(overrides) if name not in valid_tensor_names]
    if not stale:
        return []

    for name in stale:
        del overrides[name]

    sample = ", ".join(stale[:5]) + (" ..." if len(stale) > 5 else "")
    logger.warning(
        f"Dropped {len(stale)} TensorQuantOverrides entr{'y' if len(stale) == 1 else 'ies'} "
        f"whose tensor(s) are not present in the model "
        f"(likely a typo in the override key): [{sample}]."
    )
    return stale


def apply_align_eltwise_quant_type(
    model: onnx.ModelProto,
    extra_options: dict[str, Any],
    *,
    enable_npu_cnn: bool,
    enable_npu_transformer: bool,
    enable_dpu: bool,
    quant_format: Any,
    activation_type: Any,
) -> None:
    """Inject ``activation_type`` overrides for eltwise op inputs when ``AlignEltwiseQuantType`` is set.

    No-op when ``AlignEltwiseQuantType`` is not enabled. When enabled but the
    surrounding configuration does not support it (any of ``enable_npu_cnn`` /
    ``enable_npu_transformer`` / ``enable_dpu`` is True, or ``quant_format`` is
    not ``ExtendedQuantFormat.QDQ``), emit a warning and return without
    injecting anything.

    Otherwise, for every eltwise input tensor in ``model``, set its
    ``TensorQuantOverrides`` quant_type to ``activation_type`` (updating any
    existing user-supplied override or inserting a new entry).

    Must be called AFTER ``apply_pre_process`` so that ``get_eltwise_op``
    reflects the final tensor names; otherwise tensors that get folded away
    during pre-process would be injected as dangling override keys.

    :param model: the (post-pre-process) model to walk for eltwise inputs.
    :param extra_options: ``quantize_static`` extra_options dict; mutated in-place.
    :param enable_npu_cnn: NPU CNN flag from ``quantize_static``.
    :param enable_npu_transformer: NPU transformer flag from ``quantize_static``.
    :param enable_dpu: DPU flag from ``quantize_static``.
    :param quant_format: active ``QuantFormat`` / ``ExtendedQuantFormat``.
    :param activation_type: activation quant_type to write into the overrides.
    """
    if not extra_options.get("AlignEltwiseQuantType"):
        return

    if enable_npu_cnn or enable_npu_transformer or enable_dpu or quant_format != ExtendedQuantFormat.QDQ:
        logger.warning(
            "The parameter AlignEltwiseQuantType only takes effect "
            "when quant_format is ExtendedQuantFormat.QDQ and enable_npu_cnn is False "
            "and enable_npu_transformer is False and enable_dpu is False."
        )
        return

    if extra_options.get("TensorQuantOverrides") is None:
        extra_options["TensorQuantOverrides"] = {}
    eltwise_tensors = get_eltwise_op(model)
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
    # Private kwargs used by ModelQuantizer to feed print_effective_quantization_summary.
    # Not part of the public quantize_static() contract; do not rely on them externally.
    #   _print_effective_summary: True  -> categorized 11-section summary printed
    #                                      after run_static_quantization (on both
    #                                      success and failure), used by the
    #                                      QConfig path.
    #                             False -> flat legacy print_quantize_static_info
    #                                      dump printed before calibration (legacy
    #                                      Config / direct-call path, matches main).
    #   _user_extra_snapshot:     deepcopy of extra_options taken before algorithm
    #                             mutation, so the effective summary can show the
    #                             original user input alongside the normalized
    #                             effective values.
    _print_effective_summary: bool = False,
    _user_extra_snapshot: dict[str, Any] | None = None,
    extra_options: dict[str, Any] | None = {},
) -> onnx.ModelProto | None:
    """Qantize a given onnx model using static quantization. This api will return an onnx.ModelProto format quantized model
    if the argument 'model_output' is None or 'crypto_mode' is True.
    """

    update_crypto_mode(crypto_mode)

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

    skip_pre_process_graph_optimization = extra_options.get("SkipPreprocess", False)
    pre_process_yaml_path = extra_options.get("PreprocessYAML")
    if pre_process_yaml_path is not None:
        from quark.shapeshifter import Engine, LoadConfigFromFileOrDict

        skip_pre_process_graph_optimization = True

        if pre_process_yaml_path.endswith(".yaml"):
            engine_config = LoadConfigFromFileOrDict(pre_process_yaml_path).data
        else:
            engine_config = get_pre_defined_preprocess_config(pre_process_yaml_path)

        engine = Engine(config=engine_config)
        engine.initialize()
        float_model = engine.run(float_model=float_model)  # type: ignore

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

    # Legacy summary (flat dump). Used by direct quantize_static callers and by
    # the ModelQuantizer legacy Config / QuantizationConfig path. The QConfig
    # path opts into the categorized effective-config summary instead (printed
    # after normalization, just before run_static_quantization).
    if not _print_effective_summary:
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
    check_shared_initializers(float_model)

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
        skip_pre_process_graph_optimization=skip_pre_process_graph_optimization,
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

    # Drop TensorQuantOverrides entries whose tensor names are not present in
    # the (post-pre-process) graph. Almost always typos in user-supplied keys.
    prune_stale_tensor_quant_overrides(float_model, extra_options)

    # AlignEltwiseQuantType eltwise-input override injection. Run AFTER pre-process
    # (and after prune) so that get_eltwise_op() sees the final tensor names;
    # otherwise tensors folded away during pre-process (e.g., Shape->Gather
    # feeding an Add) would be injected as dangling override keys.
    apply_align_eltwise_quant_type(
        float_model,
        extra_options,
        enable_npu_cnn=enable_npu_cnn,
        enable_npu_transformer=enable_npu_transformer,
        enable_dpu=enable_dpu,
        quant_format=quant_format,
        activation_type=activation_type,
    )

    save_and_restore = extra_options.get("SaveAndRestore")
    if save_and_restore is None:
        logger.warning(
            'WARNING: "TensorsRangeFile" is deprecated and will be removed in a future release. Please use the "SaveAndRestore" API instead.'
        )
        save_and_restore = extra_options.get("TensorsRangeFile")
    skip_calibration = False
    if (
        extra_options.get("UseMatMulNBits", False)
        or (
            activation_type
            in [ExtendedQuantType.QBFloat16, ExtendedQuantType.QFloat16, ExtendedQuantType.QBFP, ExtendedQuantType.QMX]
            and not extra_options.get("ActivationScaled", False)
            and not extra_options.get("TensorQuantOverrides", {})
        )
        or (save_and_restore is not None and os.path.exists(save_and_restore) and (not crypto_mode))
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
        calib_passthrough_op_types = extra_options.get("CalibPassthroughOpTypes", [])
        op_types_to_pass_through = []
        for item_op_type in op_types_to_quantize:
            if item_op_type in calib_passthrough_op_types:
                op_types_to_quantize.remove(item_op_type)
                op_types_to_pass_through.append(item_op_type)
        if calib_passthrough_op_types:
            dependencies_dict, missing_types = nearest_non_passthrough_ancestor_mapping(
                float_model, op_types_to_pass_through
            )
            if missing_types:
                op_types_to_quantize += missing_types

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

        if calib_passthrough_op_types:
            tensors_range = update_tensors_range_with_dependencies(tensors_range, dependencies_dict)

        if save_and_restore is not None and not crypto_mode:
            save_and_restore_func(
                save_and_restore=save_and_restore,
                command_type="save",
                save_content=__version__,
                stage_name="quark_version",
            )
            save_and_restore_func(
                save_and_restore=save_and_restore,
                command_type="save",
                save_content=tensors_range,
                stage_name="tensors_range",
            )
    else:
        if save_and_restore is not None and not crypto_mode:
            tensors_range = save_and_restore_func(
                save_and_restore=save_and_restore, command_type="restore", stage_name="tensors_range"
            )
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

        # Run quantization wrapped in try/finally so the 11-category effective
        # extra_options summary is printed on BOTH success and failure paths.
        # The failure-path print is tagged with the exception class name so a
        # summary from a failed run is not mistaken for a successful one. The
        # printer reads extra_options/quant_format/calibrate_method by
        # reference, so on failure it captures the state at the failure point.
        _effective_print_exc: str | None = None
        try:
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
        except Exception as exc:
            _effective_print_exc = type(exc).__name__
            raise
        finally:
            # Print extra_options as-is. We deliberately do NOT mirror runtime
            # default-resolution here: any such mirror is a second source of
            # truth that drifts as soon as the real default changes. The
            # summary shows what was passed in plus whatever the pipeline
            # explicitly wrote back into extra_options (e.g. FP16 auto-detect
            # at quantize.py:312-318, TensorQuantOverrides upgrade); unset
            # keys are rendered from the schema's static defaults inside
            # print_effective_quantization_summary.
            if _print_effective_summary and not crypto_mode:
                print_effective_quantization_summary(
                    user_extra=_user_extra_snapshot if _user_extra_snapshot is not None else dict(extra_options),
                    effective_extra=dict(extra_options),
                    effective_quant_format=quant_format,
                    effective_calibrate_method=calibrate_method,
                    effective_enable_npu_cnn=enable_npu_cnn,
                    effective_activation_type=activation_type,
                    effective_weight_type=weight_type,
                    exception_context=_effective_print_exc,
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
