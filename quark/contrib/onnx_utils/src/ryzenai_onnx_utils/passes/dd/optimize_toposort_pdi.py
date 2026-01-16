# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import json
import os
import subprocess
import tempfile
from pathlib import Path, PurePath
from typing import Any, cast

import onnx

import ryzenai_onnx_utils
import ryzenai_onnx_utils.partitioner
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.transform.topo_sort import optimize_toposort
from ryzenai_onnx_utils.typing import is_sequence_of


def get_pdi_mapping_for_fast_pm(xclbin_json: dict[str, Any]) -> dict[str, list[str]]:
    fast_pm_info = xclbin_json["SD"]["FastPMLoad"]
    assert "Op_PM_PDI_Map" in fast_pm_info
    raw_data = fast_pm_info["Op_PM_PDI_Map"]
    ret_pdi_mapping = {}
    for op_type, info in raw_data.items():
        pdi = info["PDI_NAME"]
        if pdi not in ret_pdi_mapping:
            ret_pdi_mapping[pdi] = [op_type]
        else:
            ret_pdi_mapping[pdi].append(op_type)
    return ret_pdi_mapping


def parse_xclbin_pdi_mapping(params: ryzenai_onnx_utils.ReplaceParams) -> dict[str, list[str]]:
    if "XRT_DIR" in os.environ:
        XRT_DIR = os.environ.get("XRT_DIR")
    else:
        raise ValueError("XRT_DIR is not set, cant find xclbinutil")
    if "DD_ROOT" in os.environ:
        DD_ROOT = os.environ.get("DD_ROOT")
    else:
        raise ValueError("cant find DD_ROOT, set DD_ROOT")
    assert XRT_DIR, "XRT_DIR is not set"
    assert DD_ROOT, "DD_ROOT is not set"
    xclbinutil_path = Path(XRT_DIR) / "xclbinutil"
    key = "VENDER_METADATA"
    if not os.path.exists(xclbinutil_path):
        raise FileNotFoundError(f"cant find xclbinutils in env XRT_DIR: {xclbinutil_path}")

    xclbins = params._get_properties("xclbins")
    assert is_sequence_of(xclbins, str)
    if isinstance(xclbins, list):
        assert len(xclbins) == 1
        assert len(params.get_op_namespaces()) == 1
        with tempfile.TemporaryDirectory() as temp_dir:
            xclbin_file = cast(str, xclbins[0])
            xclbin_path = Path(DD_ROOT) / PurePath(xclbin_file).as_posix().lstrip("/")
            if not os.path.exists(xclbin_path):
                raise FileExistsError(f"cant find xclbin_path in DD_ROOT/xclbin {xclbin_path}")
            output_json = os.path.join(temp_dir, Path(xclbin_file).stem + ".json")
            command = f"{xclbinutil_path} --dump-section {key}:RAW:{output_json} --input {xclbin_path}"
            subprocess.run(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(output_json) as f:
                json_file = json.load(f)

            ret_pdi_mapping = {}
            if "SD" in json_file and "FastPMLoad" in json_file["SD"]:
                ret_pdi_mapping = get_pdi_mapping_for_fast_pm(json_file)
            else:
                assert params.get_op_namespaces()[0].upper() in json_file
                for op_type, pdis in json_file[params.get_op_namespaces()[0].upper()].items():
                    if op_type == "DD PDI Metadata version":
                        continue
                    for pdi in pdis:
                        if pdi["pdi"] not in ret_pdi_mapping:
                            ret_pdi_mapping[pdi["pdi"]] = [op_type]
                        else:
                            ret_pdi_mapping[pdi["pdi"]].append(op_type)
            return ret_pdi_mapping
    elif isinstance(xclbins, dict):
        raise NotImplementedError("xclbin doesn't support dict now, only support list.")
    raise NotImplementedError("xclbin doesn't support dict now, only support list.")


@global_pass
def generate_optimize_graph(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    pdi_mapping = parse_xclbin_pdi_mapping(params)
    if pdi_mapping is None:
        return
    optimize_method = "PDI_first"
    new_topo_graph = optimize_toposort(extractor, optimize_method, pdi_mapping)
    del extractor.model.graph.node[:]
    extractor.model.graph.node.extend(new_topo_graph)


PATTERN: list[str] = []
REPLACEMENT = generate_optimize_graph
