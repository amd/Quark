#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from quark.shares.utils.log import ScreenLogger
import numpy as np
import onnx
import os
from quark.onnx.finetuning.onnx_subgraph import Subgraph
from quark.onnx.finetuning.torch_utils import optimize_module, setup_seed
from quark.onnx.finetuning.onnx_evaluate import inference_model, average_L2
from tqdm import tqdm
from typing import Any, Optional
import time

logger = ScreenLogger(__name__)


def fast_finetune(float_model_path: str, quant_model_path: str, dr: Any, extra_options: Any) -> Any:
    """ Fast finetune the quantized model to improving its accracy."""

    def _update_optimized_param(qmodel: Any, param_name: str, opt_param: Any) -> Any:
        for init in qmodel.graph.initializer:
            if init.name != param_name or opt_param is None:
                continue

            ori_param = onnx.numpy_helper.to_array(init)
            opt_param = opt_param.astype(ori_param.dtype)
            new_init = onnx.numpy_helper.from_array(opt_param, name=param_name)
            init.CopyFrom(new_init)

            return ori_param
        return None

    selective_update = extra_options.get('FastFinetune', {}).get('SelectiveUpdate', False)
    dynamic_batch = extra_options.get('FastFinetune', {}).get('DynamicBatch', False)
    parallel = extra_options.get('FastFinetune', {}).get('Parallel', False)
    data_size = extra_options.get('FastFinetune', {}).get('DataSize', None)
    output_index = extra_options.get('FastFinetune', {}).get('OutputIndex', None)
    rmodel_path = extra_options.get('FastFinetune', {}).get('RefModelPath', None)

    if rmodel_path is not None and os.path.exists(rmodel_path):
        reference_model_path = rmodel_path
    else:
        reference_model_path = float_model_path

    if selective_update:
        float_results = inference_model(reference_model_path, dr, data_size, output_index)
        quant_results = inference_model(quant_model_path, dr, data_size, output_index)
        l2_distance = average_L2(float_results, quant_results)
        logger.info(f"Selective update for fast finetune, initial average L2 distance {l2_distance}")

    fixed_seed = extra_options.get('FastFinetune', {}).get('FixedSeed', None)
    if fixed_seed is None:
        fixed_seed = int(time.time())
    setup_seed(fixed_seed)

    logger.info(f"Start running fast finetune with seed {fixed_seed} ...")
    sg = Subgraph(reference_model_path, quant_model_path, dr, extra_options)

    if parallel:
        assert (len(sg.subgraph_qmodel_list) == len(sg.subgraph_fmodel_list) == len(sg.f_weight_list) == len(
            sg.f_input_data_list) == len(sg.f_output_data_list) == len(sg.q_input_data_list))
    else:
        assert (len(sg.subgraph_qmodel_list) == len(sg.subgraph_fmodel_list) == len(sg.f_weight_list) == len(
            sg.f_input_data_list) == len(sg.f_output_data_list))

    onnx_inference_time = 0.0
    torch_training_time = 0.0

    # TODO: MUL GEMM shape

    for i, module in tqdm(enumerate(sg.subgraph_qmodel_list), total=len(sg.subgraph_qmodel_list)):

        # Prepare input and output data for fast finetune
        start_time = time.perf_counter()
        if parallel:
            q_input_data = np.array(sg.q_input_data_list[i])
        else:
            q_input_data = np.array(sg.get_q_input_data(i))
        end_time = time.perf_counter()
        onnx_inference_time += (end_time - start_time)

        f_input_data = np.array(sg.f_input_data_list[i])
        f_output_data = np.array(sg.f_output_data_list[i])
        f_input_data = f_input_data.reshape((-1, *f_input_data.shape[2:]))
        f_output_data = f_output_data.reshape((-1, *f_output_data.shape[2:]))
        q_input_data = q_input_data.reshape((-1, *q_input_data.shape[2:]))
        if not q_input_data.shape == f_input_data.shape:
            logger.warning(f"Input shape for quantized module {q_input_data.shape} "
                           f"is different with the float module {f_input_data.shape}. "
                           "Skip this module.")
            continue

        # Optimize weight and bias for this module
        f_weight = np.array(sg.f_weight_list[i])
        f_bias = None if sg.f_bias_list[i] is None else np.array(sg.f_bias_list[i]).reshape(-1)

        try:
            start_time = time.perf_counter()
            opt_weight, opt_bias = optimize_module(module, f_weight, f_bias, q_input_data, f_input_data, f_output_data,
                                                   extra_options)
            end_time = time.perf_counter()
            torch_training_time += (end_time - start_time)
        except RuntimeError as e:
            logger.warning(f"Optimize #{i} module failed due to encountering RuntimeError: {e} ")
            continue
        except Exception as e:
            logger.warning(f"Optimize #{i} module failed due to encountering unsupported ops or OOM: {e}")
            continue

        ori_weight: onnx.TensorProto = _update_optimized_param(sg.qmodel, sg.q_weight_name_list[i], opt_weight)
        ori_bias: Optional[onnx.TensorProto] = _update_optimized_param(sg.qmodel, sg.q_bias_name_list[i], opt_bias)

        # If the L2 distance increased, restore the weight and bias
        if selective_update:
            quant_results = inference_model(sg.qmodel, dr, data_size, output_index)
            l2_distance_new = average_L2(float_results, quant_results)

            if l2_distance_new < l2_distance:
                logger.info(f"The average L2 distance is from {l2_distance} to {l2_distance_new}.")
                l2_distance = l2_distance_new
            else:
                logger.info(f"The average L2 distance is from {l2_distance} to {l2_distance_new},"
                            " the optimized weight and bias will be dropped.")
                _update_optimized_param(sg.qmodel, sg.q_weight_name_list[i], ori_weight)
                _update_optimized_param(sg.qmodel, sg.q_bias_name_list[i], ori_bias)

    logger.info(f"ONNX inference costs {onnx_inference_time:.1f}s and Torch training costs {torch_training_time:.1f}s")

    logger.info(f"Finished running fast finetune for {len(sg.subgraph_qmodel_list)} modules.")
    if dynamic_batch:
        return sg.convert_qmodel_batch_size()
    else:
        return sg.qmodel
