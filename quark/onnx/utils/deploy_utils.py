#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader

from quark.onnx.calibration import RandomDataReader
from quark.onnx.quant_utils import create_infer_session_for_onnx_model
from quark.shares.utils.log import ScreenLogger, log_errors

logger = ScreenLogger(__name__)


@log_errors
def dump_model(
    model_input: Union[str, Path, onnx.ModelProto],
    dump_data_reader: object | None = None,
    random_data_reader_input_shape: dict[str, list[int]] = {},
    dump_float: bool = False,
    output_dir: str = "./dump_results",
) -> None:
    """
    This function dumps the simulation results of the quantized model,
    including weights and activation results.

    :param Union[str, Path, onnx.ModelProto] model_input: path or ModelProto of the input model
    :param Optional[object] dump_data_reader: data reader for dumpping. Defaults to ``None``.
    :param Dict[str, List[int]] random_data_reader_input_shape: if use internal random data reader, this is used to configure input node's shape. Defaults to ``{}``.
    :param bool dump_float: dump results of the float model or not. Defaults to ``False``.
    :param str output_dir: output directory for results. Defaults to ``'./dump_results'``.
    """
    model = model_input if isinstance(model_input, onnx.ModelProto) else onnx.load(model_input)

    # Modify_output_nodes, currently it supports FixNeuron quantized model only
    fn_node_pos = {}
    has_fixneuron = False
    for n in model.graph.node:
        if n.op_type == "FixNeuron":
            fn_node_pos[n.output[0]] = 2 ** int(n.attribute[1].s)
            has_fixneuron = True
    if not has_fixneuron:
        if not dump_float:
            raise ValueError(
                "No FixNeuron node detected in the model, the results of the quantized tensor values will not be saved. "
                "Please use the parameter quant_format=VitisQuantFormat.FixNeuron to quantize the float model."
            )
            return
        else:
            logger.warning(
                "No FixNeuron node detected in the model, the results of the quantized tensor values will not be saved. "
                "Please use the parameter quant_format=VitisQuantFormat.FixNeuron to quantize the float model "
                "if you want to dump the quantized tensor value."
            )
            logger.info("The float output results of each node in the model will be saved. ")

    node_output = []
    model.graph.ClearField("output")
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
            node_output.append(output)

    so = onnxruntime.SessionOptions()
    # TODO: register_custom_ops_library(so)
    sess = create_infer_session_for_onnx_model(model, so)

    if dump_data_reader is None:
        dump_data_reader = RandomDataReader(model, input_shape=random_data_reader_input_shape)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(dump_data_reader, CalibrationDataReader):
        input = dump_data_reader.get_next()
        if not input:
            raise ValueError("dump_data_reader returned None, please confirm if the dump_data_reader is correct")
        else:
            logger.info("Dumping activations and weights...")
            results_outputs = sess.run(None, input)
            for node, res in zip(node_output, results_outputs, strict=False):
                filename = os.path.join(output_dir, node.replace("/", "_"))
                res = res.flatten()
                if node in fn_node_pos:
                    res_q = np.round(res * fn_node_pos[node])
                    res_q = res_q.clip(-128, 127)
                    res_q.astype(np.int8).tofile(filename + ".bin")
                    np.savetxt(filename + ".txt", res_q.astype(np.int8), fmt="%s", delimiter=",")
                if dump_float:
                    res.tofile(filename + "_float.bin")
                    np.savetxt(filename + "_float.txt", res, fmt="%s", delimiter=",")
    else:
        raise ValueError(
            "dump_data_reader is used for the dumping process. It should be an instance of CalibrationDataReader."
        )
