#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Quantize a float model without calibration dataset.

Use the random_quantize.py to quantize a float model without calibration dataset:

```
python randome_quantize.py --input_model_path $FLOAT_MODEL_PATH --output_model_path $QUANTIZED_MODEL_PATH
```

'''

import re
import copy
from typing import List, Tuple
from argparse import ArgumentParser, Namespace

from quark.onnx.quant_utils import PowerOfTwoMethod
from quark.onnx.quantization.config.config import Config
from quark.onnx.quantization.config.custom_config import get_default_config
from quark.onnx.quantization.api import ModelQuantizer


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input_model_path", type=str, default="", help="input onnx model file path.")
    parser.add_argument("--quantized_model_path", type=str, default="", help="output quant model file path.")
    parser.add_argument("--config",
                        type=str,
                        default="XINT8",
                        help="The configuration for quantization",
                        required=False)
    parser.add_argument("--exclude_nodes", type=str, default='', help="The names of excluding nodes", required=False)
    parser.add_argument("--exclude_subgraphs",
                        type=str,
                        default='',
                        help="The lists of excluding subgraphs",
                        required=False)
    parser.add_argument('--save_as_external_data', action='store_true')
    args, _ = parser.parse_known_args()
    return args


def parse_subgraphs_list(exclude_subgraphs: str) -> List[Tuple[List[str]]]:
    subgraphs_list = []
    tuples = exclude_subgraphs.split(";")
    for tup in tuples:
        tup = tup.strip()
        pattern = r'\[.*?\]'
        matches = re.findall(pattern, tup)
        assert len(matches) == 2
        start_nodes = matches[0].strip("[").strip("]").split(",")
        start_nodes = [node.strip() for node in start_nodes]
        end_nodes = matches[1].strip("[").strip("]").split(",")
        end_nodes = [node.strip() for node in end_nodes]
        subgraphs_list.append((start_nodes, end_nodes))
    return subgraphs_list  # type: ignore


def main(args: Namespace) -> None:
    # Prepare quantization config
    quant_config = get_default_config(args.config)
    config_copy = copy.deepcopy(quant_config)
    config_copy.extra_options["UseRandomData"] = True
    config_copy.use_external_data_format = args.save_as_external_data
    if args.config == "XINT8":
        config_copy.calibrate_method = PowerOfTwoMethod.NonOverflow
    if args.exclude_nodes:
        exclude_nodes = args.exclude_nodes.split(";")
        exclude_nodes = [node_name.strip() for node_name in exclude_nodes]
        config_copy.nodes_to_exclude = exclude_nodes
    if args.exclude_subgraphs:
        exclude_subgraphs = parse_subgraphs_list(args.exclude_subgraphs)
        config_copy.subgraphs_to_exclude = exclude_subgraphs

    # Cablibration datareader is None
    calib_datareader = None

    # Run the quantization
    quant_config = Config(global_quant_config=config_copy)
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(args.input_model_path, args.quantized_model_path, calib_datareader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
