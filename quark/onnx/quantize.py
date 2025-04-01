#
# Modifications copyright(c) 2023 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from quark.shares.utils.log import ScreenLogger, log_errors
import tempfile
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Tuple

import time
import onnx
from onnxruntime.quantization.calibrate import (CalibrationDataReader, CalibrationMethod)
from onnxruntime.quantization.quant_utils import (ms_domain, QuantizationMode, QuantType, QuantFormat)
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxsim import simplify
from .calibrate import (create_calibrator_power_of_two, create_calibrator_float_scale)
from .optimize import optimize
from .equalization import cle_transforms, replace_all_clip6_to_relu
from .quant_utils import (VAI_DOMAIN, COP_DOMAIN, VitisQuantType, VitisQuantFormat, get_exclude_nodes,
                          get_matmul_nodes_without_weights, CachedDataReader, RandomDataReader, check_onnx_model,
                          run_onnx_model, print_quantize_info, print_quantize_dynamic_info, is_ort_version_below,
                          Int16Method, PathDataReader, remove_initializer_from_input, fp32_nodes, print_fp32_nodes,
                          check_model_quantizable, save_tensor_hist_fig, PowerOfTwoMethod, customqdq_to_contribqdq,
                          skip_node_with_inf_tensor, add_or_update_opset_import, check_ir_version, check_opset_version,
                          check_qdq_model, print_quantized_info, check_extra_quant_op_types, convert_fp16_scale_to_fp32,
                          get_eltwise_op, get_fake_tensor_range, match_exclude_subgraphs, check_model_is_fp16)

from .registry import (QLinearOpsRegistry, QDQRegistry, NPUCnnRegistry, NPUTransformerRegistry)
from .bias_correction import bias_correction
from .smooth_quant import smooth_transforms
from .quarot import rotation_transforms
if is_ort_version_below("1.18.0"):
    from .cpu_quantizer import VitisQDQCPUQuantizer
    from .onnx_quantizer import VitisONNXQuantizer, ONNXQuantizer
    from .qdq_quantizer import VitisExtendedQuantizer, VitisQDQQuantizer, VitisQDQNPUCNNQuantizer, VitisBFPQuantizer, QDQNPUTransformerQuantizer
else:
    from .quantizers.cpu_quantizer import VitisQDQCPUQuantizer  # type: ignore
    from .quantizers.onnx_quantizer import ONNXQuantizer, VitisONNXQuantizer
    from .quantizers.qdq_quantizer import VitisQDQQuantizer
    from .quantizers.npu_cnn_quantizer import VitisQDQNPUCNNQuantizer
    from .quantizers.npu_transformer_quantizer import QDQNPUTransformerQuantizer
    from .quantizers.extended_quantizer import VitisExtendedQuantizer
    from .quantizers.bfp_quantizer import VitisBFPQuantizer
from .quantizers.matmul_nbits_quantizer import (MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig,
                                                GPTQWeightOnlyQuantConfig, HQQWeightOnlyQuantConfig)

logger = ScreenLogger(__name__)


@log_errors
def check_static_quant_arguments(quant_format: VitisQuantFormat, activation_type: Union[QuantType, VitisQuantType],
                                 weight_type: Union[QuantType, VitisQuantType],
                                 calibrate_method: Union[CalibrationMethod, PowerOfTwoMethod, Int16Method]) -> None:

    vitis_qwb_types = [VitisQuantType.QInt32, VitisQuantType.QUInt32, VitisQuantType.QFloat16, VitisQuantType.QBFloat16]
    ort_int4_types = []
    if not is_ort_version_below("1.19.0"):
        ort_int4_types = [QuantType.QInt4, QuantType.QUInt4]
    if (activation_type in vitis_qwb_types or weight_type in vitis_qwb_types) and quant_format != VitisQuantFormat.QDQ:
        raise ValueError("Only VitisQuantFormat.QDQ supports wide bits quantization types.")

    elif (activation_type in ort_int4_types
          or weight_type in ort_int4_types) and (not isinstance(calibrate_method, CalibrationMethod)
                                                 or not isinstance(quant_format, QuantFormat)):
        raise ValueError(
            "Only MinMax, Percentile Method and QuantFormat supports int4/uint4 quantization types or onnxruntime version below 1.19.0"
        )


@log_errors
def check_fast_fintune_arguments(extra_options: Dict[str, Any], activation_type: Union[QuantType, VitisQuantType],
                                 weight_type: Union[QuantType, VitisQuantType]) -> None:

    if not is_ort_version_below("1.19.0"):
        int_types = [QuantType.QInt4, QuantType.QUInt4]
        if activation_type in int_types or weight_type in int_types:
            raise ValueError("Fast finetune does not support int4 or uint4.")

    if weight_type in [VitisQuantType.QFloat16, VitisQuantType.QBFloat16]:
        if "AddQDQPairToWeight" in extra_options and not extra_options["AddQDQPairToWeight"]:
            logger.warning("Fast finetune requires not to fold QuantizeLinear for weights.")
        extra_options["AddQDQPairToWeight"] = True
    else:
        if "AddQDQPairToWeight" in extra_options and extra_options["AddQDQPairToWeight"]:
            logger.warning("Fast finetune requires folding QuantizeLinear for weights.")
        extra_options["AddQDQPairToWeight"] = False


@log_errors
def quantize_static(
    model_input: str,
    model_output: str,
    calibration_data_reader: CalibrationDataReader,
    calibration_data_path: Optional[str] = None,
    quant_format: Union[QuantFormat, VitisQuantFormat] = QuantFormat.QDQ,
    calibrate_method: Union[CalibrationMethod, PowerOfTwoMethod, Int16Method] = PowerOfTwoMethod.MinMSE,
    input_nodes: Optional[List[str]] = [],
    output_nodes: Optional[List[str]] = [],
    op_types_to_quantize: Optional[List[str]] = [],
    extra_op_types_to_quantize: List[str] = [],
    per_channel: bool = False,
    reduce_range: bool = False,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
    nodes_to_quantize: List[str] = [],
    nodes_to_exclude: List[str] = [],
    subgraphs_to_exclude: List[Tuple[List[str]]] = [],
    optimize_model: bool = True,
    use_external_data_format: bool = False,
    execution_providers: Optional[List[str]] = ['CPUExecutionProvider'],
    enable_dpu: bool = False,
    enable_npu_cnn: bool = False,
    enable_npu_transformer: bool = False,
    specific_tensor_precision: bool = False,
    convert_fp16_to_fp32: bool = False,
    convert_nchw_to_nhwc: bool = False,
    debug_mode: bool = False,
    include_cle: bool = False,
    include_sq: bool = False,
    include_rotation: bool = False,
    include_fast_ft: bool = False,
    include_auto_mp: bool = False,
    print_summary: bool = True,
    extra_options: Optional[Dict[str, Any]] = {},
) -> None:
    if nodes_to_quantize is None:
        nodes_to_quantize = []
    if nodes_to_exclude is None:
        nodes_to_exclude = []
    if subgraphs_to_exclude is None:
        subgraphs_to_exclude = []
    if extra_options is None:
        extra_options = {}
    if not convert_fp16_to_fp32 and not extra_options.get("QuantizeFP16", False):
        model_is_fp16 = check_model_is_fp16(model_input)
        if model_is_fp16:
            logger.warning(
                "Detected that the input model is an FP16 model. It will proceed with quantization based on the FP16 model. "
            )
            extra_options["QuantizeFP16"] = True

    if enable_dpu:
        logger.warning(
            "The 'enable_dpu' parameter will be deprecated in future versions. Please use 'enable_npu_cnn' instead.")
        enable_npu_cnn = enable_dpu

    print_quantize_info(model_input, model_output, calibration_data_reader, calibration_data_path, quant_format,
                        input_nodes, output_nodes, op_types_to_quantize, extra_op_types_to_quantize, per_channel,
                        reduce_range, activation_type, weight_type, nodes_to_quantize, nodes_to_exclude,
                        subgraphs_to_exclude, optimize_model, use_external_data_format, calibrate_method,
                        execution_providers, enable_npu_cnn, enable_npu_transformer, specific_tensor_precision,
                        debug_mode, convert_fp16_to_fp32, convert_nchw_to_nhwc, include_cle, include_sq,
                        include_rotation, include_fast_ft, extra_options)

    fp32_nodes_dict = fp32_nodes(model_input)

    if extra_options.get("QuantizeAllOpTypes", False):
        all_op_types = list(fp32_nodes_dict.keys())
        extra_op_types_to_quantize.extend(all_op_types)

    if subgraphs_to_exclude:
        nodes_to_exclude += match_exclude_subgraphs(model_input, subgraphs_to_exclude)
        nodes_to_exclude = list(set(nodes_to_exclude))

    if "ConvertOpsetVersion" in extra_options:
        opset_version = extra_options["ConvertOpsetVersion"]
        from .tools.convert_opset_version import convert_opset_version
        model = onnx.load(model_input)
        model_update_opset_version = convert_opset_version(model, opset_version)
        model_update_opset_version_path = tempfile.TemporaryDirectory(prefix="vai.tools.")
        model_input = Path(model_update_opset_version_path.name).joinpath("update_opset_version.onnx").as_posix()
        onnx.save_model(model_update_opset_version, model_input, save_as_external_data=use_external_data_format)

    skip_calibration = False
    if extra_options.get("UseMatMulNBits",
                         False) or (activation_type in [VitisQuantType.QBFloat16, VitisQuantType.QFloat16]
                                    and not extra_options.get("ActivationScaled", False)) or quant_format in [
                                        VitisQuantFormat.BFPFixNeuron, VitisQuantFormat.MXFixNeuron
                                    ]:
        skip_calibration = True

    if convert_fp16_to_fp32:
        logger.info(f"Start converting {model_input} to float32 model.")
        from .tools import float16
        fp16_model = onnx.load(model_input)
        try:
            fp32_model = float16.convert_float16_to_float(fp16_model)
            try:
                model_simp, check = simplify(fp32_model)
                assert check, "Simplified ONNX model could not be validated"
                logger.info(f"Convert {model_input} to float32 model sucessfully")
            except Exception as e2:
                logger.warning(f"Fail to Simplify ONNX model because of {e2}.")
                model_simp = fp32_model
        except Exception as e:
            logger.warning(f"Fail to convert fp16 to fp32 beacuse {e}, "
                           "skip fp16 to fp32 conversion.")
            model_simp = fp16_model

        fp32_path = tempfile.TemporaryDirectory(prefix="vai.tools.")
        model_input = Path(fp32_path.name).joinpath("fp32.onnx").as_posix()
        onnx.save_model(model_simp, model_input, save_as_external_data=use_external_data_format)

    mode = QuantizationMode.QLinearOps

    quantize_fp16 = extra_options.get("QuantizeFP16", False)
    use_fp32_scale = extra_options.get("UseFP32Scale", quantize_fp16)
    if quantize_fp16:
        optimize_model = False
        if is_ort_version_below("1.18.0"):
            logger.warning(
                "The parameter QuantizeFP16 only takes effect in onnxruntime 1.18 and above. It will output a model same as the input model if onnxruntime version is 1.17 or lower."
            )
        logger.info(
            "The parameter optimize_model is set to False automatically when the parameter QuantizeFP16 is set to True."
        )
    if not quantize_fp16 and use_fp32_scale:
        logger.warning("The parameter UseFP32Scale could be True only if the parameter QuantizeFP16 is True.")

    if not skip_calibration:
        if calibration_data_reader is not None and calibration_data_path is not None:
            logger.warning(
                "Both calibration_data_reader and calibration_data_path are provided. Will use the calibration_data_reader for calibration."
            )
        elif calibration_data_reader is None and calibration_data_path is not None:
            logger.info(f"calibration_data_reader is None, using {calibration_data_path} to do calibration.")
            calibration_data_reader = PathDataReader(model_input, calibration_data_path)

        check_onnx_model(model_input)

    if calibration_data_reader is None:
        if not extra_options.get("UseRandomData", False):
            raise ValueError(
                'A calibration data reader is required for static quantization, but none was provided. Please provide a calibration data reader, or alternatively enable random data generation for calibration by setting `config.global_quant_config.extra_options["UseRandomData"]` to `True`.'
            )
        else:
            calibration_data_reader = RandomDataReader(model_input,
                                                       input_shape=extra_options.get("RandomDataReaderInputShape", {}),
                                                       input_data_range=extra_options.get(
                                                           "RandomDataReaderInputDataRange", None))

    if not check_ir_version(model_input):
        ('The ir version of input model is below 4. It is recommended to upgrade ir version to 7 or higher.')
    if not check_opset_version(model_input):
        logger.warning(
            'The opset version of input model is below 10. It is recommended to upgrade opset version to 17 or higher.')
    if check_qdq_model(model_input):
        raise RuntimeError(
            "The input model is already a quantized model. Please make sure that input model is a float model.")
    cached_data_reader = CachedDataReader(calibration_data_reader, None, convert_nchw_to_nhwc, quantize_fp16)

    is_save_hist = False
    if "SaveTensorHistFig" in extra_options and extra_options["SaveTensorHistFig"]:
        is_save_hist = True
    if is_save_hist:
        with tempfile.TemporaryDirectory(prefix="vai.quant.") as quant_tmp_dir:
            hist_calibrator = create_calibrator_float_scale(
                Path(model_input),
                op_types_to_quantize,
                augmented_model_path=Path(quant_tmp_dir).joinpath("augmented_model.onnx").as_posix(),
                calibrate_method=CalibrationMethod.Percentile,
                use_external_data_format=use_external_data_format,
                execution_providers=execution_providers,
                extra_options={"symmetric": False},
            )
            save_tensor_hist_fig(hist_calibrator, cached_data_reader, extra_options)
    if not skip_calibration:
        cached_data_reader.reset_iter()

    if input_nodes or output_nodes:
        if nodes_to_exclude:
            nodes_to_exclude += get_exclude_nodes(model_input, input_nodes, output_nodes)
        else:
            nodes_to_exclude = get_exclude_nodes(model_input, input_nodes, output_nodes)

    if extra_options.get("MatMulConstBOnly", enable_npu_transformer):
        nodes_to_exclude += get_matmul_nodes_without_weights(model_input)

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        if enable_npu_transformer:
            op_types_to_quantize = list(NPUTransformerRegistry.keys())
        else:
            q_linear_ops = list(QLinearOpsRegistry.keys())
            qdq_ops = list(QDQRegistry.keys())
            if enable_npu_cnn or quant_format is VitisQuantFormat.QDQ:
                dpu_ops = list(NPUCnnRegistry.keys())
                qdq_ops = list(set(dpu_ops + qdq_ops))
            op_types_to_quantize = list(set(q_linear_ops + qdq_ops))

    check_extra_quant_op_types(model_input, extra_op_types_to_quantize)

    op_types_to_quantize += extra_op_types_to_quantize
    op_types_to_quantize = list(set(op_types_to_quantize))

    remove_input_init = True
    if "RemoveInputInit" in extra_options:
        remove_input_init = extra_options["RemoveInputInit"]
    if remove_input_init:
        try:
            model = onnx.load(model_input)
            model_rm_input_init = remove_initializer_from_input(model)
            model_rm_input_init_path = tempfile.TemporaryDirectory(prefix="vai.tools.")
            model_input = Path(model_rm_input_init_path.name).joinpath("rm_input_init.onnx").as_posix()
            onnx.save_model(model_rm_input_init, model_input, save_as_external_data=use_external_data_format)
            logger.info("Removed initializers from input")
        except Exception as e:
            logger.debug(f"Fail to remove init from input because {e}")

    if extra_options.get("SimplifyModel", True) and not extra_options.get("UseMatMulNBits", False):
        try:
            model = onnx.load(model_input)
            model_simp, check = simplify(model)
            assert check, "Simplified ONNX model could not be validated"
            model_simp_path = tempfile.TemporaryDirectory(prefix="vai.simp.")
            model_input = Path(model_simp_path.name).joinpath("model_simp.onnx").as_posix()
            onnx.save_model(model_simp, model_input, save_as_external_data=use_external_data_format)
            logger.info("Simplified model sucessfully")
        except Exception as e:
            logger.warning(f"Fail to Simplify ONNX model because of {e}.")

    shared_init_optypes = extra_options.get("CopySharedInit", [])
    if shared_init_optypes is not None:
        from quark.onnx.tools import convert_shared_initializer_to_unique
        try:
            model = onnx.load(model_input)
            model_simp_path = tempfile.TemporaryDirectory(prefix="vai.cpinit.")
            model_input = Path(model_simp_path.name).joinpath("model_cpinit.onnx").as_posix()
            model_copyinit = convert_shared_initializer_to_unique.convert(model, shared_init_optypes)
            onnx.save_model(model_copyinit, model_input, save_as_external_data=use_external_data_format)
            logger.info(
                "Duplicate the shared initializers in the model for separate quantization use across different nodes!")
        except Exception as e:
            logger.warning(f"Fail to duplicate the shared initializers in the ONNX model because of {e}.")

    logger.info("Loading model...")
    fold_batch_norm = optimize_model
    from onnxruntime.quantization.quant_utils import load_model_with_shape_infer
    if optimize_model and not use_external_data_format:
        from onnxruntime.quantization.quant_utils import optimize_model as om
        try:
            quant_opt_tmp_dir = tempfile.TemporaryDirectory(prefix="vai.quant.opt.")
            opt_model_path = Path(quant_opt_tmp_dir.name).joinpath("model.onnx").as_posix()

            om(Path(model_input), Path(opt_model_path))
            model = load_model_with_shape_infer(Path(opt_model_path))
        except Exception as e:
            logger.warning(f"Failed to run quantization preprocessing with error of {e}. "
                           "Using original model. Please check.")
            try:
                model = load_model_with_shape_infer(Path(model_input))
            except Exception as e:
                raise RuntimeError(f"Model loading failed as {e}"
                                   "Shape inference needs write access to the model input directory."
                                   "Please verify permissions of the model input directory.")
                return
    else:
        try:
            model = load_model_with_shape_infer(Path(model_input))
        except Exception as e:
            raise RuntimeError(f"Model loading failed as {e}"
                               "Shape inference needs write access to the model input directory."
                               "Please verify permissions of the model input directory.")
            return
    if convert_nchw_to_nhwc:
        from .utils.model_utils import convert_nchw_to_nhwc as convert_func
        logger.info(f"Start converting {model_input} ncwh to nhwc model.")
        try:
            model = convert_func(model)
            converted_path = tempfile.TemporaryDirectory(prefix="vai.tools.")
            model_input = Path(converted_path.name).joinpath("converted.onnx").as_posix()
            onnx.save_model(model, model_input, save_as_external_data=use_external_data_format)
        except Exception as e:
            logger.warning(f"Failed to convert nchw to nhwc beacuse {e}, ")

    if not skip_calibration:
        run_onnx_model(model_input, cached_data_reader)
        cached_data_reader.reset_iter()

    if not check_model_quantizable(model, op_types_to_quantize, nodes_to_exclude):
        onnx.save_model(model, model_output, save_as_external_data=use_external_data_format)
        logger.warning("No quantizable ops in this model, quantization is skipped.")
        return

    clip6_to_relu6 = False
    if "ReplaceClip6Relu" in extra_options:
        clip6_to_relu6 = extra_options['ReplaceClip6Relu']

    if clip6_to_relu6:
        model = replace_all_clip6_to_relu(model, op_types_to_quantize, nodes_to_quantize, nodes_to_exclude)
        clip6relu_path = tempfile.TemporaryDirectory(prefix="vai.quant.")
        clip6relu_model_output = Path(clip6relu_path.name).joinpath("clip6relu_model.onnx").as_posix()
        onnx.save_model(model, clip6relu_model_output, save_as_external_data=use_external_data_format)
        model_input = clip6relu_model_output
        topo_model = ONNXModel(onnx.load(clip6relu_model_output))
        topo_model.topological_sort()
        model = topo_model.model

    if include_cle:
        cle_balance_method = "max"
        cle_steps = 1
        cle_weight_threshold = 0.5
        cle_scale_append_bias = True
        cle_scale_use_threshold = True
        cle_total_layer_diff_threshold = 2e-7

        if "CLEBalanceMethod" in extra_options:
            cle_balance_method = extra_options['CLEBalanceMethod']
        if "CLEWeightThreshold" in extra_options:
            cle_weight_threshold = extra_options['CLEWeightThreshold']
        if "CLEScaleUseThreshold" in extra_options:
            cle_scale_use_threshold = extra_options['CLEScaleUseThreshold']
        if "CLEScaleAppendBias" in extra_options:
            cle_scale_append_bias = extra_options['CLEScaleAppendBias']
        if "CLESteps" in extra_options:
            cle_steps = extra_options['CLESteps']
        if "CLETotalLayerDiffThreshold" in extra_options:
            cle_total_layer_diff_threshold = extra_options['CLETotalLayerDiffThreshold']
        if nodes_to_exclude is None:
            nodes_to_exclude = []
        model = cle_transforms(
            model,
            op_types_to_quantize,
            nodes_to_quantize,
            nodes_to_exclude,
            cle_steps,
            cle_balance_method,
            cle_weight_threshold,
            cle_scale_append_bias,
            cle_scale_use_threshold,
            cle_total_layer_diff_threshold,
        )

        cle_path = tempfile.TemporaryDirectory(prefix="vai.quant.")
        cle_model_output = Path(cle_path.name).joinpath("cle_model.onnx").as_posix()
        onnx.save_model(model, cle_model_output, save_as_external_data=use_external_data_format)
        model_input = cle_model_output

    if include_rotation:
        from quark.torch.algorithm.rotation.rotation_utils import get_rotation_matrix
        logger.info("Start rotation ....")
        hidden_size = extra_options.get('RMatrixDim', 4096)
        random_had = extra_options.get('UseRandomHad', False)
        rotation_config_file = extra_options.get('RConfigPath', None)
        assert rotation_config_file is not None, "Error! Please specify the rotation config file via extra_options[\"RConfigPath\"]"

        try:
            r1_matrix = get_rotation_matrix(num_channels=hidden_size, random=random_had)
        except AssertionError as e:
            raise AssertionError(
                "Error! The dim of the target R1 matrix is not support while requiring random_had as true.")
        r1_matrix_np = r1_matrix.numpy()

        r_matrixs = {"R1": r1_matrix_np}

        rotation_path = tempfile.TemporaryDirectory(prefix="vai.quant.")
        rotation_model_output = Path(rotation_path.name).joinpath("rotated_model.onnx").as_posix()
        model = rotation_transforms(rotation_model_output, model, r_matrixs, rotation_config_file)
        onnx.save_model(model, rotation_model_output, save_as_external_data=use_external_data_format)

        model_input = rotation_model_output
        model = onnx.load_model(rotation_model_output)
        logger.info("Rotation complete!")

    if include_sq:
        smooth_alpha = 0.5
        if "SmoothAlpha" in extra_options:
            smooth_alpha = extra_options['SmoothAlpha']
        logger.info(f"Start smoothing model, the smooth alpha was set as {smooth_alpha}")
        smooth_path = tempfile.TemporaryDirectory(prefix="vai.quant.")
        smooth_model_output = Path(smooth_path.name).joinpath("smooth_model.onnx").as_posix()
        model = smooth_transforms(smooth_model_output, model, cached_data_reader, alpha=smooth_alpha)
        onnx.save_model(model, smooth_model_output, save_as_external_data=use_external_data_format)
        cached_data_reader.reset_iter()
        model_input = smooth_model_output
        model = onnx.load_model(smooth_model_output)

    skip_node_with_inf_tensor_list = skip_node_with_inf_tensor(model)
    nodes_to_exclude.extend(skip_node_with_inf_tensor_list)

    int16_scale = False
    if "Int16Scale" in extra_options:
        int16_scale = extra_options["Int16Scale"]
    if int16_scale:
        if enable_npu_cnn:
            raise ValueError("Int16Scale is an experimental feature"
                             "and cannot be used simultaneously with enable_npu_cnn")

    add_or_update_opset_import(model, ms_domain, 1)
    if quant_format == VitisQuantFormat.QDQ:
        add_or_update_opset_import(model, VAI_DOMAIN, 1)
    if quant_format in [VitisQuantFormat.QDQ, VitisQuantFormat.BFPFixNeuron, VitisQuantFormat.MXFixNeuron]:
        add_or_update_opset_import(model, COP_DOMAIN, 1)

    fuse_instance_norm = True
    fuse_l2_norm = True
    fuse_gelu = True
    fuse_layer_norm = True
    convert_split_to_slice = False
    convert_bn_to_conv = False
    convert_reduce_mean_to_global_avg_pool = False
    split_large_kernel_pool = False

    # TODO: Refactor logics of optimization for xcompiler and vaiml in the future.
    if (enable_npu_cnn or enable_npu_transformer
            or (quant_format is VitisQuantFormat.QDQ and not extra_options.get('BF16QDQToCast', False)
                and not extra_options.get('EnableVaimlBF16', False))):
        logger.info("optimize the model for better hardware compatibility.")
        convert_split_to_slice = True
        convert_bn_to_conv = True
        convert_reduce_mean_to_global_avg_pool = True
        split_large_kernel_pool = True

    if "FoldBatchNorm" in extra_options:
        fold_batch_norm = extra_options["FoldBatchNorm"]
    if "FuseInstanceNorm" in extra_options:
        fuse_instance_norm = extra_options["FuseInstanceNorm"]
    if "FuseL2Norm" in extra_options:
        fuse_l2_norm = extra_options["FuseL2Norm"]
    if "FuseGelu" in extra_options:
        fuse_gelu = extra_options["FuseGelu"]
    if "FuseLayerNorm" in extra_options:
        fuse_layer_norm = extra_options["FuseLayerNorm"]
    if "ConvertSplitToSlice" in extra_options:
        convert_split_to_slice = extra_options["ConvertSplitToSlice"]
    if "ConvertBNToConv" in extra_options:
        convert_bn_to_conv = extra_options["ConvertBNToConv"]
    if "ConvertReduceMeanToGlobalAvgPool" in extra_options:
        convert_reduce_mean_to_global_avg_pool = extra_options["ConvertReduceMeanToGlobalAvgPool"]
    if "SplitLargeKernelPool" in extra_options:
        split_large_kernel_pool = extra_options["SplitLargeKernelPool"]

    if (fuse_instance_norm or fuse_l2_norm or fuse_gelu or fuse_layer_norm or convert_bn_to_conv
            or convert_reduce_mean_to_global_avg_pool or split_large_kernel_pool or convert_split_to_slice
            or fold_batch_norm):
        model = optimize(model,
                         op_types_to_quantize,
                         nodes_to_quantize,
                         nodes_to_exclude,
                         convert_bn_to_conv,
                         convert_reduce_mean_to_global_avg_pool,
                         split_large_kernel_pool,
                         convert_split_to_slice,
                         fuse_instance_norm,
                         fuse_l2_norm,
                         fuse_gelu,
                         fuse_layer_norm,
                         fold_batch_norm,
                         convert_clip_to_relu=False,
                         fold_batch_norm_after_concat=fold_batch_norm)

        from onnxruntime.quantization.quant_utils import save_and_reload_model_with_shape_infer
        model = save_and_reload_model_with_shape_infer(model)

    calib_extra_options_keys = [
        ("CalibTensorRangeSymmetric", "symmetric"),
        ("CalibMovingAverage", "moving_average"),
        ("CalibMovingAverageConstant", "averaging_constant"),
        ("MinMSEMode", "minmse_mode"),
        ("Percentile", "percentile"),
        ("NumBins", "num_bins"),
        ("NumQuantizedBins", "num_quantized_bins"),
    ]
    calib_extra_options = {
        key: extra_options.get(name)
        for (name, key) in calib_extra_options_keys if name in extra_options
    }

    quantized_tensor_type = {}
    if specific_tensor_precision:
        if extra_options.get("MixedPrecisionTensor"):
            for k, v in extra_options["MixedPrecisionTensor"].items():
                for t in v:
                    quantized_tensor_type[t] = k
            if quantized_tensor_type:
                logger.info("In the specific_tensor_precision mode, "
                            "the quant_format will use VitisQuantFormat.QDQ")
                quant_format = VitisQuantFormat.QDQ

    if extra_options.get("AlignEltwiseQuantType"):
        if enable_npu_cnn is False and enable_npu_transformer is False and enable_dpu is False and quant_format == VitisQuantFormat.QDQ:
            eltwise_tensors = get_eltwise_op(model_input)
            for tensor_name in eltwise_tensors:
                quantized_tensor_type[tensor_name] = activation_type
            logger.info(
                "The parameter AlignEltwiseQuantType takes effect, the weights of nodes will be quantized with the activation quant type if the operation type is in [Mul, Div, Add, Sub, Min, Max]."
            )
        else:
            logger.warning(
                "The parameter AlignEltwiseQuantType only takes effect when quant_format is VitisQuantFormat.QDQ and enable_npu_cnn is False and enable_npu_transformer is False and enable_dpu is False"
            )

    if extra_options.get("UseMatMulNBits", False):
        matmul_nbits_quantize_dict = extra_options.get("MatMulNBitsParams", {})
        assert isinstance(matmul_nbits_quantize_dict,
                          dict), "The parameter 'MatMulNBitsParams' in extra_options must be a dict."
        if "GroupSize" in matmul_nbits_quantize_dict:
            matmul_nbits_group_size = matmul_nbits_quantize_dict["GroupSize"]
        else:
            matmul_nbits_group_size = 128
        if "Symmetric" in matmul_nbits_quantize_dict:
            matmul_nbits_symmetric = matmul_nbits_quantize_dict["Symmetric"]
        else:
            matmul_nbits_symmetric = True
        if "Bits" in matmul_nbits_quantize_dict:
            matmul_nbits_bits = matmul_nbits_quantize_dict["Bits"]
        else:
            matmul_nbits_bits = 4
        if "AccuracyLevel" in matmul_nbits_quantize_dict:
            matmul_nbits_accuracy_level = matmul_nbits_quantize_dict["AccuracyLevel"]
        else:
            matmul_nbits_accuracy_level = 0

        algo_config: Union[DefaultWeightOnlyQuantConfig, GPTQWeightOnlyQuantConfig, HQQWeightOnlyQuantConfig,
                           None] = None
        if extra_options.get("MatMulNBitsParams", {}).get("Algorithm", "DEFAULT") == "GPTQ":
            algo_config = GPTQWeightOnlyQuantConfig(calibration_data_reader=cached_data_reader,
                                                    percdamp=extra_options.get('GPTQParams', {}).get('PercDamp', 0.01),
                                                    block_size=extra_options.get('GPTQParams',
                                                                                 {}).get('BlockSize', 128),
                                                    actorder=extra_options.get('GPTQParams', {}).get('ActOrder', False),
                                                    mse=extra_options.get('GPTQParams', {}).get('MSE', False),
                                                    perchannel=extra_options.get('GPTQParams',
                                                                                 {}).get('PerChannel', False))
        elif extra_options.get("MatMulNBitsParams", {}).get("Algorithm", "DEFAULT") == "HQQ":
            algo_config = HQQWeightOnlyQuantConfig(block_size=matmul_nbits_group_size, bits=matmul_nbits_bits)
        else:
            algo_config = DefaultWeightOnlyQuantConfig(block_size=matmul_nbits_group_size,
                                                       is_symmetric=matmul_nbits_symmetric,
                                                       bits=matmul_nbits_bits,
                                                       accuracy_level=matmul_nbits_accuracy_level)

        quantizer = MatMulNBitsQuantizer(model,
                                         matmul_nbits_group_size,
                                         matmul_nbits_symmetric,
                                         matmul_nbits_bits,
                                         accuracy_level=matmul_nbits_accuracy_level,
                                         algo_config=algo_config,
                                         extra_options=extra_options)
        quantizer.quantize_model()
        quantizer.model.save_model_to_file(model_output, use_external_data_format)
        cached_data_reader.reset_iter()

    # TODO: Refactor logics for quantize.py in the future.
    optimized_path = tempfile.TemporaryDirectory(prefix="vai.quant.")
    model_input = Path(optimized_path.name).joinpath("opt_model.onnx").as_posix()
    topo_model = ONNXModel(model)
    topo_model.topological_sort()
    model = topo_model.model
    onnx.save_model(model, model_input, save_as_external_data=use_external_data_format)

    if not skip_calibration:
        logger.info("Start calibration...")
        start_time = time.perf_counter()
        # Get calib data reader with specified data size
        calib_data_size = extra_options.get('CalibDataSize', None)
        if calib_data_size is not None:
            logger.info(f'CalibDataSize is {calib_data_size}, use the {calib_data_size} data for calibration')
            calib_data_size = int(calib_data_size)
            calib_dr = CachedDataReader(cached_data_reader, calib_data_size, convert_nchw_to_nhwc, quantize_fp16)
        else:
            calib_dr = cached_data_reader
        # Do calibration
        if isinstance(calibrate_method, PowerOfTwoMethod):
            with tempfile.TemporaryDirectory(prefix="vai.quant.") as quant_tmp_dir:
                calibrator = create_calibrator_power_of_two(
                    Path(model_input),
                    op_types_to_quantize,
                    augmented_model_path=Path(quant_tmp_dir).joinpath("augmented_model.onnx").as_posix(),
                    activation_type=activation_type,
                    method=calibrate_method,
                    use_external_data_format=use_external_data_format,
                    execution_providers=execution_providers,
                    quantized_tensor_type=quantized_tensor_type,
                    extra_options=calib_extra_options,
                )
                logger.info(
                    "Start collecting data, runtime depends on your model size and the number of calibration dataset.")
                calibrator.collect_data(calib_dr)
                if calibrate_method == PowerOfTwoMethod.MinMSE:
                    tensors_range = calibrator.compute_range()
                    from onnxruntime.quantization.calibrate import TensorsData
                    tensors_range = TensorsData(CalibrationMethod.MinMax, tensors_range)
                else:
                    tensors_range = calibrator.compute_data()
                del calibrator
        else:
            with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
                calibrator = create_calibrator_float_scale(
                    Path(model_input),
                    op_types_to_quantize,
                    augmented_model_path=Path(quant_tmp_dir).joinpath("augmented_model.onnx").as_posix(),
                    calibrate_method=calibrate_method,
                    use_external_data_format=use_external_data_format,
                    execution_providers=execution_providers,
                    extra_options=calib_extra_options,
                )
                logger.info(
                    "Start collecting data, runtime depends on your model size and the number of calibration dataset.")
                calibrator.collect_data(calib_dr)
                tensors_range = calibrator.compute_data()
                del calibrator
        end_time = time.perf_counter()
        calib_time = end_time - start_time
        logger.info(f"Finished the calibration of {calibrate_method} which costs {calib_time:.1f}s")
        cached_data_reader.reset_iter()
    else:
        from onnxruntime.quantization.calibrate import TensorsData
        fake_tensor_range = get_fake_tensor_range(model)
        tensors_range = TensorsData(CalibrationMethod.MinMax, fake_tensor_range)

    if not extra_options.get("UseMatMulNBits", False):
        from onnxruntime.quantization.quant_utils import load_model_with_shape_infer
        model = load_model_with_shape_infer(Path(model_input))

    from .quant_utils import remove_qdq_op_type, annotate_op_type
    if extra_options.get("RemoveQDQConvClip", True):
        remove_qdq_op_type.append("Clip")
    if extra_options.get("RemoveQDQConvRelu", True):
        remove_qdq_op_type.append("Relu")
    if extra_options.get("RemoveQDQConvLeakyRelu", True):
        remove_qdq_op_type.append("LeakyRelu")
    if extra_options.get("RemoveQDQConvPRelu", True):
        remove_qdq_op_type.append("PRelu")
    if extra_options.get("RemoveQDQConvGelu", False):
        remove_qdq_op_type.append("Gelu")
    if extra_options.get("RemoveQDQInstanceNorm", False):
        annotate_op_type.append("InstanceNormalization")

    check_static_quant_arguments(quant_format, activation_type, weight_type, calibrate_method)
    if include_fast_ft and include_auto_mp is False:
        check_fast_fintune_arguments(extra_options, activation_type, weight_type)

    if int16_scale:
        calibrate_method = Int16Method.MinMax

    if not extra_options.get("UseMatMulNBits", False):
        # BFP and MX quantization don't need calibration, so they are not sensitive to calibration method
        if quant_format in [VitisQuantFormat.BFPFixNeuron, VitisQuantFormat.MXFixNeuron]:
            quantizer = VitisBFPQuantizer(
                model,
                per_channel,
                reduce_range,
                mode,
                quant_format,
                True,
                weight_type,
                activation_type,
                tensors_range,
                nodes_to_quantize,
                nodes_to_exclude,
                op_types_to_quantize,
                calibrate_method,
                quantized_tensor_type,
                extra_options,
            )
        elif calibrate_method in CalibrationMethod:
            if quant_format is QuantFormat.QOperator:
                quantizer = ONNXQuantizer(
                    model,
                    per_channel,
                    reduce_range,
                    mode,
                    True,  # static
                    weight_type,
                    activation_type,
                    tensors_range,
                    nodes_to_quantize,
                    nodes_to_exclude,
                    op_types_to_quantize,
                    extra_options,
                )
            elif quant_format is QuantFormat.QDQ:
                if not enable_npu_transformer:
                    quantizer = VitisQDQCPUQuantizer(
                        model,
                        per_channel,
                        reduce_range,
                        mode,
                        True,  # static
                        weight_type,
                        activation_type,
                        tensors_range,
                        nodes_to_quantize,
                        nodes_to_exclude,
                        op_types_to_quantize,
                        calibrate_method,
                        quantized_tensor_type,
                        extra_options,
                    )
                else:
                    quantizer = QDQNPUTransformerQuantizer(
                        model,
                        per_channel,
                        reduce_range,
                        mode,
                        True,  # static
                        weight_type,
                        activation_type,
                        tensors_range,
                        nodes_to_quantize,
                        nodes_to_exclude,
                        op_types_to_quantize,
                        extra_options,
                    )
            elif quant_format is VitisQuantFormat.QDQ:
                quantizer = VitisExtendedQuantizer(
                    model,
                    per_channel,
                    reduce_range,
                    mode,
                    quant_format,
                    True,
                    weight_type,
                    activation_type,
                    tensors_range,
                    nodes_to_quantize,
                    nodes_to_exclude,
                    op_types_to_quantize,
                    calibrate_method,
                    quantized_tensor_type,
                    extra_options,
                )
            else:
                raise ValueError("No corresponding quantizer for this set of arguments.")
        elif calibrate_method in PowerOfTwoMethod or calibrate_method in Int16Method:
            if quant_format is QuantFormat.QOperator:
                quantizer = VitisONNXQuantizer(
                    model,
                    per_channel,
                    reduce_range,
                    mode,
                    True,
                    weight_type,
                    activation_type,
                    tensors_range,
                    nodes_to_quantize,
                    nodes_to_exclude,
                    op_types_to_quantize,
                    calibrate_method,
                    quantized_tensor_type,
                    extra_options,
                )
            elif quant_format is QuantFormat.QDQ:
                if not enable_npu_cnn:
                    quantizer = VitisQDQQuantizer(
                        model,
                        per_channel,
                        reduce_range,
                        mode,
                        True,
                        weight_type,
                        activation_type,
                        tensors_range,
                        nodes_to_quantize,
                        nodes_to_exclude,
                        op_types_to_quantize,
                        calibrate_method,
                        quantized_tensor_type,
                        extra_options,
                    )
                else:
                    quantizer = VitisQDQNPUCNNQuantizer(
                        model,
                        per_channel,
                        reduce_range,
                        mode,
                        True,
                        weight_type,
                        activation_type,
                        tensors_range,
                        nodes_to_quantize,
                        nodes_to_exclude,
                        op_types_to_quantize,
                        calibrate_method,
                        quantized_tensor_type,
                        extra_options,
                    )
            elif quant_format is VitisQuantFormat.QDQ:
                quantizer = VitisExtendedQuantizer(
                    model,
                    per_channel,
                    reduce_range,
                    mode,
                    quant_format,
                    True,
                    weight_type,
                    activation_type,
                    tensors_range,
                    nodes_to_quantize,
                    nodes_to_exclude,
                    op_types_to_quantize,
                    calibrate_method,
                    quantized_tensor_type,
                    extra_options,
                )
            else:
                raise ValueError("No corresponding quantizer for this set of arguments.")
        quantizer.quantize_model()

        if extra_options.get('RemoveQDQMulAdd', False):
            from .tools.remove_qdq_mul_add import remove_qdq_mul_add
            remove_qdq_mul_add(quantizer.model.model)

        if 'RemoveQDQBetweenOps' in extra_options:
            from .tools.remove_qdq_between_ops import remove_qdq_between_ops

            between_ops = extra_options.get('RemoveQDQBetweenOps')
            if not (isinstance(between_ops, list) and all(
                    isinstance(item, tuple) and len(item) == 2 and all(isinstance(elem, str) for elem in item)
                    for item in between_ops)):
                logger.warning(f"'RemoveQDQBetweenOps' should be a list of (str, str) tuples. Actual: {between_ops}")

            remove_qdq_between_ops(quantizer.model.model, between_ops)

        if extra_options.get('BF16QDQToCast', extra_options.get('EnableVaimlBF16', False)):
            from .tools.replace_bfloat16_qdq_cast import replace_bfloat16_qdq_cast
            quantizer.model.model = replace_bfloat16_qdq_cast(quantizer.model.model)

        if extra_options.get('EnableVaimlBF16', False):
            from .tools.remove_bf16_cast import remove_bf16_cast
            quantizer.model.model = remove_bf16_cast(quantizer.model.model)

        quantizer.model.save_model_to_file(model_output, use_external_data_format)

    if quantize_fp16 and use_fp32_scale:
        convert_fp16_scale_to_fp32(model_output, use_external_data_format)

    if quant_format is VitisQuantFormat.QDQ:
        # Since the ONNXRuntime 1.17.0 starts supportting 16bit quantization,
        # we convert our custom QDQs to the MSFT contributed QDQs by default
        customqdq_to_contribqdq(model_output, use_external_data_format)

    bias_corr = False
    if 'BiasCorrection' in extra_options:
        bias_corr = extra_options['BiasCorrection']
    if bias_corr:
        quant_model = bias_correction(model_input, model_output, use_external_data_format, cached_data_reader,
                                      activation_type, calibrate_method, extra_options)
        onnx.save(quant_model, model_output)

    if 'FixShapes' in extra_options:
        from .tools.convert_dynamic_to_fixed import fix_shapes
        fix_name_shape = extra_options['FixShapes']
        model = onnx.load(model_output)
        model = fix_shapes(model, fix_name_shape)
        onnx.save(model, model_output)

    if include_auto_mp:
        from quark.onnx.mprecision.auto_mixprecision import auto_mixprecision
        cached_data_reader.reset_iter()
        model = auto_mixprecision(model_input, model_output, cached_data_reader, activation_type, weight_type,
                                  extra_options)
        onnx.save(model, model_output)

    if include_fast_ft:
        from quark.onnx.finetuning.fast_finetune import fast_finetune
        cached_data_reader.reset_iter()
        model = fast_finetune(model_input, model_output, cached_data_reader, extra_options)
        onnx.save(model, model_output)

    use_gptq = False
    if 'UseGPTQ' in extra_options:
        use_gptq = extra_options['UseGPTQ']
    if use_gptq:
        from .gptq.gptq import GptqProcessor
        gptq_path = tempfile.TemporaryDirectory(prefix="vai.quant.")
        gptq_model_output = Path(gptq_path.name).joinpath("gptq_model.onnx").as_posix()
        cached_data_reader.reset_iter()
        gptq_processor = GptqProcessor(gptq_model_output, model_input, model_output, cached_data_reader, extra_options)
        model = gptq_processor.apply()
        onnx.save(model, model_output)

    if extra_options.get('BF16WithClip', False):
        from .tools.insert_clip_bfloat16_qdq import insert_clip_bfloat16_qdq
        model = insert_clip_bfloat16_qdq(model)
        onnx.save(model, model_output)

    # This optimization should after calibration
    convert_clip_to_relu = False
    if "ConvertClipToRelu" in extra_options:
        convert_clip_to_relu = extra_options["ConvertClipToRelu"]
    # This is a post processing of quantization
    dedicate_dq_node = False
    if "DedicateDQNode" in extra_options:
        dedicate_dq_node = extra_options["DedicateDQNode"]
    if convert_clip_to_relu or dedicate_dq_node:
        model = optimize(
            model,
            op_types_to_quantize,
            nodes_to_quantize,
            nodes_to_exclude,
            convert_bn_to_conv=False,
            convert_reduce_mean_to_global_avg_pool=False,
            split_large_kernel_pool=False,
            convert_split_to_slice=False,
            fuse_instance_norm=False,
            fuse_l2_norm=False,
            fuse_gelu=False,
            convert_clip_to_relu=convert_clip_to_relu,
            dedicate_dq_node=dedicate_dq_node,
        )
        from onnxruntime.quantization.quant_utils import save_and_reload_model_with_shape_infer
        model = save_and_reload_model_with_shape_infer(model)

        onnx.save(model, model_output)

    if print_summary and fp32_nodes_dict:
        print_fp32_nodes(fp32_nodes_dict, model_output)
        print_quantized_info(model_output, debug_mode)
        if not extra_options.get("UseMatMulNBits", False):
            if not check_ir_version(model_input):
                print(
                    'WARNING: The ir version of input model is below 4. It is recommended to upgrade ir version to 7 or higher.'
                )
            if not check_opset_version(model_input):
                print(
                    'WARNING: The opset version of input model is below 10. It is recommended to upgrade opset version to 17 or higher.'
                )
            if check_qdq_model(model_input):
                print(
                    "ERROR: The input model is already a quantized model. Please make sure that input model is a float model."
                )


def quantize_dynamic(
    model_input: Union[str, Path, onnx.ModelProto],
    model_output: Union[str, Path],
    op_types_to_quantize: Union[List[str], None] = [],
    per_channel: bool = False,
    reduce_range: bool = False,
    weight_type: QuantType = QuantType.QInt8,
    nodes_to_quantize: List[str] = [],
    nodes_to_exclude: List[str] = [],
    subgraphs_to_exclude: List[Tuple[List[str]]] = [],
    use_external_data_format: bool = False,
    debug_mode: bool = False,
    extra_options: Optional[Dict[str, Any]] = {},
) -> None:
    """Given an onnx model, create a quantized onnx model and save it into a file

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
    from onnxruntime.quantization.registry import IntegerOpsRegistry
    from onnxruntime.quantization.quant_utils import load_model_with_shape_infer, model_has_pre_process_metadata, save_and_reload_model_with_shape_infer

    extra_options = extra_options or {}
    nodes_to_exclude = nodes_to_exclude or []
    subgraphs_to_exclude = subgraphs_to_exclude or []
    nodes_to_quantize = nodes_to_quantize or []
    op_types_to_quantize = op_types_to_quantize or []

    mode = QuantizationMode.IntegerOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(IntegerOpsRegistry.keys())

    print_quantize_dynamic_info(model_input, model_output, op_types_to_quantize, per_channel, reduce_range, weight_type,
                                nodes_to_quantize, nodes_to_exclude, subgraphs_to_exclude, use_external_data_format,
                                debug_mode, extra_options)

    if not subgraphs_to_exclude:
        nodes_to_exclude += match_exclude_subgraphs(model_input, subgraphs_to_exclude)
        nodes_to_exclude = list(set(nodes_to_exclude))

    model = (save_and_reload_model_with_shape_infer(model_input)
             if isinstance(model_input, onnx.ModelProto) else load_model_with_shape_infer(Path(model_input)))

    pre_processed: bool = model_has_pre_process_metadata(model)
    if not pre_processed:
        logger.warning(
            "Please consider to run pre-processing before quantization. Refer to example: "
            "https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification"
            "/cpu/ReadMe.md ")

    if "MatMulConstBOnly" not in extra_options:
        extra_options["MatMulConstBOnly"] = True

    quantizer = ONNXQuantizer(
        model,
        per_channel,
        reduce_range,
        mode,
        False,  # static
        weight_type,
        QuantType.QUInt8,  # dynamic activation only supports uint8
        None,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options,
    )

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)
