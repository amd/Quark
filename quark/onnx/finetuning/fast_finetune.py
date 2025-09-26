#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import onnx
from tqdm import tqdm

from quark.onnx.finetuning.onnx_evaluate import average_L2, inference_model
from quark.onnx.finetuning.onnx_subgraph import Subgraph
from quark.onnx.finetuning.torch_utils import optimize_module, setup_seed
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def fast_finetune(
    float_model: Union[str, Path, onnx.ModelProto],
    quant_model: Union[str, Path, onnx.ModelProto],
    use_external_data_format: bool,
    data_reader: Any,
    extra_options: Any,
) -> Any:
    """Fast finetune the quantized model to improving its accracy."""

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

    selective_update = extra_options.get("FastFinetune", {}).get("SelectiveUpdate", False)
    data_size = extra_options.get("FastFinetune", {}).get("DataSize", None)
    output_index = extra_options.get("FastFinetune", {}).get("OutputIndex", None)
    ref_model_path = extra_options.get("FastFinetune", {}).get("RefModelPath", None)

    if ref_model_path is not None and isinstance(ref_model_path, str) and os.path.exists(ref_model_path):
        reference_model = onnx.load(ref_model_path)
    else:
        reference_model = float_model if isinstance(float_model, onnx.ModelProto) else onnx.load(float_model)

    if selective_update:
        float_results = inference_model(reference_model, data_reader, data_size, output_index)
        quant_results = inference_model(quant_model, data_reader, data_size, output_index)
        l2_distance = average_L2(float_results, quant_results)
        logger.info(f"Selective update for fast finetune, initial average L2 distance {l2_distance}")

    # Fix the seed to guarantee that finetuned model could be reproduced
    fixed_seed = extra_options.get("FastFinetune", {}).get("FixedSeed", 1705472343)
    setup_seed(fixed_seed)

    logger.info(f"Start running fast finetune with seed {fixed_seed} ...")
    sg = Subgraph(reference_model, quant_model, use_external_data_format, data_reader, extra_options)

    assert len(sg.subgraph_qmodel_list) == len(sg.subgraph_fmodel_list) == len(sg.f_weight_list), (
        "The quantized model or float model has an incorrect number of subgraphs"
    )

    onnx_inference_time = 0.0
    torch_training_time = 0.0

    # TODO: MUL GEMM shape

    for i, module in tqdm(enumerate(sg.subgraph_qmodel_list), total=len(sg.subgraph_qmodel_list)):
        # Prepare input and output data for the training
        got_data_flag: bool = False
        start_time = time.perf_counter()

        try:
            q_input_data, f_input_data, f_output_data = sg.get_training_data(i)
            got_data_flag = True

        except OSError as e:
            logger.error(f"Encountered an OSError: {e}")

            if "space" in str(e) or "written" in str(e):
                logger.warning("This error occurs due to insufficient disk space for the temporary directory.")

                if sg.mem_opt_level == 2:
                    logger.warning(
                        "Will automatically set the 'MemOptLevel' option to 1 to prevent caching data on disk."
                    )
                    sg.mem_opt_level = 1
                else:
                    logger.warning(
                        "Please provide another temporary directory with sufficient disk space via the option 'TmpDir'."
                    )

        except Exception as e:
            logger.error(f"Encountered an error: {e}")

            if "memory" in str(e) or "alloc" in str(e):
                logger.warning("This error occurs due to an out of memory issue.")

                if sg.mem_opt_level != 2:
                    logger.warning(
                        "Will automatically set the 'MemOptLevel' option to 2 to cache data on disk instead of memory."
                    )
                    sg.mem_opt_level = 2
                else:
                    logger.warning("Please reduce the data size for the fine-tuning via the option 'DataSize'.")

        finally:
            end_time = time.perf_counter()
            onnx_inference_time += end_time - start_time

            if not got_data_flag:
                logger.warning(f"Optimizing #{i} module failed in getting data, skip it and continue to the next one.")
                continue  # noqa: B012

        # Optimize weight and bias for this module
        f_weight = np.array(sg.f_weight_list[i])
        f_bias = None if sg.f_bias_list[i] is None else np.array(sg.f_bias_list[i]).reshape(-1)

        # Optimize this module
        optimized_module_flag: bool = False
        start_time = time.perf_counter()

        try:
            opt_weight, opt_bias = optimize_module(
                module, f_weight, f_bias, q_input_data, f_input_data, f_output_data, extra_options
            )
            optimized_module_flag = True

        except (MemoryError, RuntimeError) as e:
            logger.error(f"Encountered an error: {e}")

            if "out of memory" in str(e):  # The error message should be "HIP out of memory" or "CUDA out of memory"
                logger.warning("According to the error message, likely there is a GPU out of memory issue.")

                if sg.mem_opt_level != 2:
                    logger.warning(
                        "Will automatically set the 'MemOptLevel' option to 2 to cache a batch data on GPU only."
                    )
                    sg.mem_opt_level = 2
                else:
                    logger.warning("Please reduce the data size for the fine-tuning via the option 'DataSize'.")

        except Exception as e:
            logger.error(f"Encountered an error: {e}")

        finally:
            end_time = time.perf_counter()
            torch_training_time += end_time - start_time

            if not optimized_module_flag:
                logger.warning(
                    f"Optimizing #{i} module failed in optimizing module, skip it and continue to the next one."
                )
                continue  # noqa: B012

        # Update the module's weight and bias
        ori_weight: onnx.TensorProto = _update_optimized_param(sg.qmodel, sg.q_weight_name_list[i], opt_weight)
        ori_bias: onnx.TensorProto | None = _update_optimized_param(sg.qmodel, sg.q_bias_name_list[i], opt_bias)

        # If the L2 distance increased, restore the weight and bias
        if selective_update:
            quant_results = inference_model(sg.qmodel, data_reader, data_size, output_index)
            l2_distance_new = average_L2(float_results, quant_results)

            if l2_distance_new < l2_distance:
                logger.info(f"The average L2 distance is from {l2_distance} to {l2_distance_new}.")
                l2_distance = l2_distance_new
            else:
                logger.info(
                    f"The average L2 distance is from {l2_distance} to {l2_distance_new},"
                    " the optimized weight and bias will be dropped."
                )
                _update_optimized_param(sg.qmodel, sg.q_weight_name_list[i], ori_weight)
                _update_optimized_param(sg.qmodel, sg.q_bias_name_list[i], ori_bias)

    logger.info(f"ONNX inference costs {onnx_inference_time:.1f}s and Torch training costs {torch_training_time:.1f}s")
    finetuned_model = sg.convert_qmodel_batch_size() if sg.dynamic_batch else sg.qmodel

    logger.info(f"Finished running fast finetune for {len(sg.subgraph_qmodel_list)} modules.")
    sg.clean_up()

    return finetuned_model
