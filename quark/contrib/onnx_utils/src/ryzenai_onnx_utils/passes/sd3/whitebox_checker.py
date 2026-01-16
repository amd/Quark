# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar

import onnx
import ryzenai_dynamic_dispatch as dd
import sympy as sp

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import get_attribute, has_attribute
from ryzenai_onnx_utils.partitioner import get_check_shapes
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.typing import PatternType

_logger = logging.getLogger(__name__)
_logged_shape_failures: dict[str, set[tuple]] = {}

##
# @class WhiteboxChecker
# @brief A generator for ops-config.json files.
#
# This class reads static-shape or dynamic-shape ONNX graphs extractor, scans all nodes that may not have corresponding TXNs,
# and pushes their configuration into an ops-config.json file. The generated ops-config.json acts as an intermediate
# bridge and is passed to the unified tiler for the next step of TXN generation.
# The WhiteboxChecker class is first designed to support automatic generation of SD3 transformer and vae model with 25 resolutions.
#
# The main functionalities include:
# 1. Parse model nodes to extract input/output shapes and relevant attributes for various SD operators,
#    and filter out nodes that already have corresponding TXNs. For SDConv, SDGemm, SDGemmConcat, where wts-shuffle involved, we
#    force regenerate the TXN for the nodes to ensure the wts-shuffle alignment across different resolutions.
# 2. For each operator type, invoking the corresponding shape check function to determine if the shape matches supported patterns.
# 3. For supported operators, organizing their configuration information, avoiding duplicates, and generating a unified config json file.
#
# Main responsibilities:
# - Scan ONNX graph nodes and collect those without TXNs.
# - Generate ops-config.json containing operator configuration information.
##

PASS_REGISTRY: dict[str, type["WhiteboxBasePass"]] = {}
T = TypeVar("T", bound="WhiteboxBasePass")


def register_whitebox_pass(op_type: str) -> Callable[[T], T]:
    """
    Decorator to register a Pass class under a given op_type.
    Ensures no duplicate keys.

    Example:
        @register_withebox_pass("SDMul")
        class SDMulPass(WhiteboxBasePass):
            ...
    """

    def decorator(pass_cls: T) -> T:
        if op_type in PASS_REGISTRY:
            raise ValueError(
                f"Pass for '{op_type}' already registered: {PASS_REGISTRY[op_type].__name__} vs {pass_cls.__name__}"
            )
        PASS_REGISTRY[op_type] = pass_cls
        _logger.debug(f"Registered: '{op_type}' -> {pass_cls.__name__}")
        return pass_cls

    return decorator


class WhiteboxBasePass:
    """
    Base class for all Whitebox Passes.
    Provides common interface for Checker to call.

    Subclasses should override:
        - is_supported_shape()
        - get_input_output_shapes()

     Attributes:
        - force_whitelist: if True, always generate config.
        - whitebox_flow_op_type: the op type seen by unified tiler.
    """

    whitebox_flow_op_type: str = None
    force_whitelist: bool = False  # Optional, default is False

    def __init_subclass__(cls: type[T], **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "whitebox_flow_op_type", None):
            raise TypeError(f"[WhiteboxBasePass] Subclass '{cls.__name__}' must define 'whitebox_flow_op_type'.")

    @classmethod
    def is_supported_shape(cls: type[T], op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        raise NotImplementedError(f"[WhiteboxBasePass] Subclass '{cls.__name__}' must override 'is_supported_shape()'.")

    @classmethod
    def get_input_output_shapes(cls: type[T], node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        raise NotImplementedError(
            f"[WhiteboxBasePass] Subclass '{cls.__name__}' must override 'get_input_output_shapes()'."
        )


class WhiteboxChecker:
    def __init__(self, extractor: onnx.utils.Extractor, params: ryzenai_onnx_utils.ReplaceParams) -> None:
        self.extractor: onnx.utils.Extractor = extractor
        self.params: ryzenai_onnx_utils.ReplaceParams = params
        self.op_namespace: str = params.attributes["op_namespaces"]
        self.whitebox_checker_level = (
            int(params.attributes["whitebox_checker_level"]) if "whitebox_checker_level" in params.attributes else 1
        )

    @staticmethod
    def prepare_op_info(node: onnx.NodeProto, op_type: str, inputs: list[list], outputs: list) -> dict[str, Any]:
        if op_type == "multiheadattention" and get_attribute(node, "op_version").decode("utf-8") in ["v1.1", "v2"]:
            num_heads: list[int] = [input[1] for input in inputs]
            assert all(v == num_heads[0] for v in num_heads), (
                f"All values in num_heads must be equal, but got: {num_heads}"
            )
            inputs: list[list[int]] = [[input[0], input[2], input[1] * input[3]] for input in inputs]
            inputs.append([num_heads[0]])
        ops_type: dict[str, Any] = {
            "op_type": op_type,
            "attrs": [dd.onnx_graph.extract_op_attrs(node)],
            "inputs": [inputs],
            "outputs": [outputs],
        }
        return ops_type

    @staticmethod
    def check_op_info_exist(op_type: str, op_info: dict[str, Any], op_infos: list[dict[str, Any]]) -> bool:
        if len(op_infos) == 0:
            return False
        if op_type == "multiheadattention":  # num_heads
            num_heads: int = op_info["attrs"][0]["num_heads"]["value"][0]
            exist_num_heads: list[int] = [elem["attrs"][0]["num_heads"]["value"][0] for elem in op_infos]
            if num_heads not in exist_num_heads:
                return False
        return op_info in op_infos

    @staticmethod
    def generate_ops_config(op_infos_dict: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        ops_config: list[dict[str, Any]] = []
        for _op_type, op_infos in op_infos_dict.items():
            update_op_infos: dict[str, Any] = op_infos[0]
            for i in range(1, len(op_infos)):
                update_op_infos["attrs"].append(op_infos[i]["attrs"][0])
                update_op_infos["inputs"].append(op_infos[i]["inputs"][0])
                update_op_infos["outputs"].append(op_infos[i]["outputs"][0])
            ops_config.append(update_op_infos)
        return ops_config

    @staticmethod
    def sympy_converter(o: Any) -> int | float:
        if isinstance(o, sp.Basic):
            return float(o) if o.is_Float else int(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    @staticmethod
    def save_op_info_to_json(path: str, xclbin: str, model_type: str, op_infos: list[dict[str, Any]]) -> None:
        data: dict[str, Any] = {
            "xclbin": xclbin,
            "overlay_json": "",
            "overlay": "2x4x4",
            "comment": "S: Sign",
            "backend_target": "DD",
            "model_type": model_type,
            "op_types": op_infos,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=WhiteboxChecker.sympy_converter)

    def shape_checker(self, node: onnx.NodeProto, pass_cls: type[WhiteboxBasePass]) -> list[dict[str, Any]]:
        check_fn: Callable[[str, dict[str, Any]], bool] = pass_cls.is_supported_shape
        shape_lists: dict[str, Any] = pass_cls.get_input_output_shapes(node, self.extractor)
        check_shapes: list[dict[str, Any]] = get_check_shapes(shape_lists, self.params.attributes)
        # The `force_whitelist` is enabled only when wts shuffle alignment is required in dynamic mode.
        force_whitelist: bool = getattr(pass_cls, "force_whitelist", False)
        force_whitelist &= "dynamic_shape_list" in self.params.attributes
        whitebox_op_type: str = pass_cls.whitebox_flow_op_type

        def to_tuple(s):
            return tuple(s) if isinstance(s, list | tuple) else (s,)

        op_infos: list[dict[str, Any]] = []
        for shape in check_shapes:
            input_shape: list[list[int]] = shape["input_shape"]
            output_shape: list[int] = shape["output_shape"][0]
            passed: bool = check_fn(self.op_namespace, shape)
            if (self.whitebox_checker_level == 1 and (not passed or force_whitelist)) or (
                self.whitebox_checker_level == 2
            ):
                if not passed:
                    shape_pair = (tuple(to_tuple(x) for x in input_shape), tuple(to_tuple(x) for x in output_shape))
                    op_type = node.op_type
                    if shape_pair not in _logged_shape_failures.get(op_type, set()):
                        _logger.debug(
                            f"[WhiteboxChecker] Shape check failed for '{node.name}' "
                            f"(op_type={op_type}, whitebox_op_type={whitebox_op_type}): input_shape={input_shape}, output_shape={output_shape}"
                        )
                        _logged_shape_failures.setdefault(op_type, set()).add(shape_pair)

                op_info: dict[str, Any] = self.prepare_op_info(
                    node,
                    whitebox_op_type,
                    input_shape,
                    output_shape,
                )
                if not self.check_op_info_exist(whitebox_op_type, op_info, op_infos):
                    op_infos.append(op_info)

        return op_infos

    def run(self, hints_key: str) -> None:
        if not self.whitebox_checker_level:
            return  # Whitebox Checker is disabled; skip processing
        op_infos_dict: dict[str, list[dict[str, Any]]] = {}
        missing_op_types: set[str] = set()
        for node in self.extractor.model.graph.node:
            if not node.op_type.startswith(hints_key):
                continue

            op_type: str = node.op_type
            if op_type == "SDGemm" and has_attribute(node, "trans_head"):
                op_type = "SDGemmTrans"
            if op_type == "SDGemm_bfp" and has_attribute(node, "trans_head"):
                op_type = "SDGemmTrans_bfp"
            if op_type == "SDFlatMHA" or op_type == "SDMHA_VAE":
                op_type = "SDMHA"
            if op_type == "SDFlatMHA_bfp":
                op_type = "SDMHA_bfp"
            if op_type not in PASS_REGISTRY:
                if op_type not in missing_op_types:
                    _logger.warning(f"[WhiteboxChecker] No Pass registered for: {op_type}")
                    missing_op_types.add(op_type)
                continue
            pass_cls: type[WhiteboxBasePass] = PASS_REGISTRY[op_type]
            whitebox_op_type: str = pass_cls.whitebox_flow_op_type
            op_infos: list[dict[str, Any]] = self.shape_checker(node, pass_cls)
            if not op_infos:
                continue
            if whitebox_op_type not in op_infos_dict:
                op_infos_dict[whitebox_op_type] = op_infos
            else:
                for info in op_infos:
                    if not self.check_op_info_exist(whitebox_op_type, info, op_infos_dict[whitebox_op_type]):
                        op_infos_dict[whitebox_op_type].append(info)

        if op_infos_dict:
            op_infos: list[dict[str, Any]] = self.generate_ops_config(op_infos_dict)
            # SD30_VAE is a special case, it can be either SD30_ENCODER or SD30_DECODER
            # when the input channel is 3, it is SD30_ENCODER, otherwise it is SD30_DECODER
            if self.params.attributes["model_type"] == "SD30_VAE":
                model_type = (
                    "SD30_ENCODER"
                    if self.extractor.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 3
                    else "SD30_DECODER"
                )
            else:
                model_type = self.params.attributes["model_type"]
            self.save_op_info_to_json(
                self.params.abs_dd_files_path / self.params.attributes["op_config"],
                self.params.attributes["xclbins"],
                model_type,
                op_infos,
            )


@global_pass
def whitebox_checker(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    hints_key: str = "SD"
    checker: WhiteboxChecker = WhiteboxChecker(extractor, params)
    checker.run(hints_key)


PATTERN: PatternType = []
REPLACEMENT = whitebox_checker
