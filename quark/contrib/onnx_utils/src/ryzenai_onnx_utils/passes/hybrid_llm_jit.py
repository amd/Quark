# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass extracts certain weights from custom operators and moves them to a
separate binary file for JIT loading. This is used to reduce memory usage so
ORT doesn't load all weights as the model is loaded. Depending on attributes,
different weights may be extracted.
"""

import os
import re
from collections.abc import Callable, Generator, Sequence
from pathlib import Path

import onnx
import onnx.external_data_helper

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.proto as proto
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.typing import PatternType


def get_layer_id(node: onnx.NodeProto) -> int | None:
    # this matches on something like layers.0 being present in the name
    regex_pattern = r".*layers.(\d+)"
    matched = re.match(regex_pattern, node.name)
    if matched:
        return int(matched.group(1))
    # Note: this relies on hybrid passes renaming node to follow this convention
    #       typically {op_type}_{pass_id}_{layer_id}
    #       e.g. MatMulNBits_5_0
    index = node.name.split("_")[-1]
    try:
        layer_id = int(index)
    except ValueError:
        return None
    return layer_id


def node_gen(graph: onnx.GraphProto) -> Generator[onnx.NodeProto, None, None]:
    for node in graph.node:
        if node.domain != "com.ryzenai":
            continue
        if node.op_type in ["Cast", "CastAvx"]:
            continue
        yield node


def is_npu_weight(initializer_name: str) -> bool:
    return initializer_name.endswith(".packed")


def is_gpu_weight(initializer_name: str) -> bool:
    return "MatMulNBits" in initializer_name and not is_npu_weight(initializer_name)


def is_embedding(initializer_name: str, op_type: str) -> bool:
    return op_type == "LUT" and initializer_name.endswith("embed_tokens.weight")


def is_qmoe_op(op_type: str) -> bool:
    return op_type.startswith("QMoE")


def is_qmoe_weight(initializer_name: str, op_type: str) -> bool:
    return is_qmoe_op(op_type) and initializer_name.endswith(".packed.qexperts")


def is_npu_offloaded(initializer_name: str, enable_jit: bool) -> bool:
    return is_npu_weight(initializer_name) and enable_jit


def is_gpu_offloaded(initializer_name: str, enable_jit: bool) -> bool:
    return is_gpu_weight(initializer_name) and enable_jit


def is_offloaded(initializer_name: str, enable_jit_npu: bool, enable_jit_gpu: bool) -> bool:
    return is_npu_offloaded(initializer_name, enable_jit_npu) or is_gpu_offloaded(initializer_name, enable_jit_gpu)


def save_weights(
    header: proto.external_data.Header,
    enable_jit: bool,
    jit_enabled: Callable[[str], bool],
    jit_enabled_any: Callable[[str], bool],
    extractor: onnx.utils.Extractor,
    weights_file: Path,
    offset: int,
    layers: Sequence[int | None],
) -> int:
    nodes = node_gen(extractor.graph)
    last_layer_index = -1
    inputs_to_clear = set()
    for node in nodes:
        layer_id = get_layer_id(node)
        layer_index = layers.index(layer_id) if layer_id is not None else last_layer_index + 1
        # for some models, especially test models, we can have (unfused) node
        # names that don't match the expected names for identifying layer info.
        # However, this layer data is only actually needed if we're using JIT
        # (more specifically GPU JIT currently since the NPU is not using layer
        # data). So there's no JIT, we can relax this assertion.
        if enable_jit:
            assert layer_index >= last_layer_index, (
                f"Layer_index {layer_index}, last_layer_index {last_layer_index}, node_name {node.name}"
            )
        if layer_index > last_layer_index:
            header.layers[layer_index].offset = offset
            header.layers[layer_index].size = 0
        if node.name not in header.layers[layer_index].operators:
            header.layers[layer_index].operators.append(node.name)
        last_layer_index = layer_index

        input_index = 0
        for input_name in node.input:
            if ryzenai_onnx_utils.matcher.is_initializer(input_name, extractor):
                # only update the header and remove the weight from the model if:
                #   - it's an NPU weight and JIT NPU is enabled
                #   - it's not an NPU weight and JIT GPU is enabled
                if not jit_enabled(input_name):
                    if jit_enabled_any(input_name):
                        input_index += 1
                    continue

                tensor_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(
                    input_name, extractor, False
                ).tobytes()
                header.layers[layer_index].size += len(tensor_data)
                header.operators[node.name].data[input_index].offset = (
                    os.path.getsize(weights_file) if os.path.exists(weights_file) else 0
                )
                dtype = header.operators[node.name].data[input_index].data_type

                with open(weights_file, "ab") as f:
                    f.write(tensor_data)

                input_index += 1
                offset += len(tensor_data)

                inputs_to_clear.add((input_name, dtype))

    for input_name, dtype in inputs_to_clear:
        extractor.wmap[input_name] = onnx.helper.make_tensor(input_name, dtype, [0], [])
    return offset


def save_qmoe(
    header: proto.external_data.Header,
    offload_qmoe: bool,
    extractor: onnx.utils.Extractor,
    weights_file: Path,
    offset: int,
) -> int:
    if not offload_qmoe:
        return offset

    nodes = node_gen(extractor.graph)
    inputs_to_clear = set()

    # when mmap'ing a file there is an alignment constraint from OS
    # this is to help with that so we dont have to mmap unneeded regions
    VIRTUAL_ADDR_ALIGNMENT = 1 << 16

    for node in nodes:
        # only consider QMoE operators
        if not is_qmoe_op(node.op_type):
            continue

        input_index = 0
        for input_name in node.input:
            if ryzenai_onnx_utils.matcher.is_initializer(input_name, extractor) and is_qmoe_weight(
                input_name, node.op_type
            ):
                tensor_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(
                    input_name, extractor, False
                ).tobytes()

                curr_offset = os.path.getsize(weights_file) if os.path.exists(weights_file) else 0
                assert offset == curr_offset, f"Mismatch in offset {offset} and file size {curr_offset}"

                pad_size = (VIRTUAL_ADDR_ALIGNMENT - (offset % VIRTUAL_ADDR_ALIGNMENT)) % VIRTUAL_ADDR_ALIGNMENT

                header.operators[node.name].data[input_index].offset = pad_size + offset

                pad = bytes(pad_size)
                dtype = header.operators[node.name].data[input_index].data_type

                with open(weights_file, "ab") as f:
                    f.write(pad)
                    f.write(tensor_data)

                input_index += 1
                offset += pad_size + len(tensor_data)

                inputs_to_clear.add((input_name, dtype))

    # need to remove from onnx.data, otherwise ORT will load this
    for input_name, dtype in inputs_to_clear:
        extractor.wmap[input_name] = onnx.helper.make_tensor(input_name, dtype, [0], [])

    return offset


def save_embedding(
    header: proto.external_data.Header,
    offload_embedding: bool,
    extractor: onnx.utils.Extractor,
    weights_file: Path,
    offset: int,
) -> int:
    if not offload_embedding:
        return offset

    nodes = node_gen(extractor.graph)
    inputs_to_clear = set()

    for node in nodes:
        # only consider LUT operators
        if node.op_type != "LUT":
            continue

        input_index = 0
        for input_name in node.input:
            if ryzenai_onnx_utils.matcher.is_initializer(input_name, extractor):
                assert input_index == 0, "only expect 1 input initializer"

                tensor_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(
                    input_name, extractor, False
                ).tobytes()

                header.operators[node.name].data[input_index].offset = (
                    os.path.getsize(weights_file) if os.path.exists(weights_file) else 0
                )
                dtype = header.operators[node.name].data[input_index].data_type

                with open(weights_file, "ab") as f:
                    f.write(tensor_data)

                input_index += 1
                offset += len(tensor_data)

                inputs_to_clear.add((input_name, dtype))

    # need to remove from onnx.data, otherwise ORT will load this
    for input_name, dtype in inputs_to_clear:
        extractor.wmap[input_name] = onnx.helper.make_tensor(input_name, dtype, [0], [])

    return offset


def save_jit_weights(
    extractor: onnx.utils.Extractor,
    output_model_path: Path,
    jit_gpu: bool,
    jit_npu: bool,
    offload_embedding: bool,
    offload_qmoe: bool,
) -> None:
    enable_jit_any = jit_gpu or jit_npu

    layers = []
    header = proto.external_data.Header()

    header_file_name = f"{output_model_path.stem}.pb.bin"
    weights_file_name = f"{output_model_path.stem}.bin"

    if enable_jit_any or offload_embedding or offload_qmoe:
        header.external_data.filename = weights_file_name
        header.external_data.gpu = jit_gpu
        header.external_data.npu = jit_npu

        # graph must be topologically sorted for JIT
        try:
            ryzenai_onnx_utils.matcher.graph_topological_sort(extractor, True)
        except RuntimeError as e:
            print("Failed to topologically sort graph, JIT is disabled:", e)
            return
    else:
        header.external_data.filename = ""
        header.external_data.gpu = False
        header.external_data.npu = False

    nodes = node_gen(extractor.graph)
    for node in nodes:
        layer_id = get_layer_id(node)
        if layer_id is None or layer_id not in layers:
            layers.append(layer_id)

        operator = proto.external_data.Operator(op_type=node.op_type)

        for input_name in node.input:
            if ryzenai_onnx_utils.matcher.is_initializer(input_name, extractor):
                if node.op_type not in header.op_metadata:
                    metadata = header.op_metadata.get_or_create(node.op_type)
                    metadata.max_npu_buffer_size = 0
                    metadata.first = node.name
                    metadata.last = node.name
                op_metadata = header.op_metadata[node.op_type]
                op_metadata.last = node.name

                # only update the header and remove the weight from the model if:
                #   - it's an NPU weight and JIT NPU is enabled
                #   - it's not an NPU weight and JIT GPU is enabled
                if is_npu_weight(input_name):
                    if not jit_npu:
                        continue
                elif is_gpu_weight(input_name):
                    if not jit_gpu:
                        continue
                elif is_embedding(input_name, node.op_type):
                    if not offload_embedding:
                        continue
                elif is_qmoe_weight(input_name, node.op_type):
                    if not offload_qmoe:
                        continue
                else:
                    continue

                np_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(input_name, extractor)
                dtype = ryzenai_onnx_utils.matcher.get_dtype(input_name, extractor)
                operator.data.add(
                    offset=0,
                    size=np_data.nbytes,
                    shape=np_data.shape,
                    data_type=int(dtype),
                )

                if is_npu_weight(input_name):
                    op_metadata.max_npu_buffer_size = max(op_metadata.max_npu_buffer_size, np_data.nbytes)

        header.operators.get_or_create(node.name).MergeFrom(operator)

    for _ in layers:
        header.layers.add()

    weights_file = output_model_path.parent / weights_file_name
    weights_file.unlink(True)
    offset = 0

    offset = save_weights(
        header,
        enable_jit_any,
        lambda name: is_npu_weight(name) and jit_npu,
        lambda name: is_offloaded(name, jit_npu, jit_gpu),
        extractor,
        weights_file,
        offset,
        layers,
    )
    # GPU has to be after NPU because currently, we're using the GPU tensors to
    # define the layer offset and sizes. So the NPU call above will assign one
    # set of values which this call will overwrite
    offset = save_weights(
        header,
        enable_jit_any,
        lambda name: is_gpu_offloaded(name, jit_gpu),
        lambda name: is_offloaded(name, jit_npu, jit_gpu),
        extractor,
        weights_file,
        offset,
        layers,
    )

    offset_w_qmoe = save_qmoe(
        header,
        offload_qmoe,
        extractor,
        weights_file,
        offset,
    )

    header.external_data.qmoe = offset_w_qmoe != offset

    # For now have embeddings at very end
    # During runtime embeddings will either be mmap'd or loaded entirely into memory
    offset_w_embedding = save_embedding(
        header,
        offload_embedding,
        extractor,
        weights_file,
        offset_w_qmoe,
    )

    header.external_data.embedding = offset_w_embedding != offset_w_qmoe

    proto.save_header(header, output_model_path.parent / header_file_name)


@global_pass
def extract_jit_weights(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    npu_jit = params.get_bool_attr("npu_jit", False)
    gpu_jit = params.get_bool_attr("gpu_jit", False)
    offload_embedding = params.get_bool_attr("offload_embedding", False)
    offload_qmoe = params.get_bool_attr("offload_qmoe", False)

    save_jit_weights(extractor, params.output_path, gpu_jit, npu_jit, offload_embedding, offload_qmoe)


PATTERN: PatternType = []
REPLACEMENT = extract_jit_weights
