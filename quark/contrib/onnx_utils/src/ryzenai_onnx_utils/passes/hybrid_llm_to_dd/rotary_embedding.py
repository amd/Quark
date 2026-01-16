# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import ml_dtypes
import numpy as np
import numpy.typing as npt
import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as transform_cast
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.typing import PassOutputArgs, StrictPassOutputArgs


def trig_cache_builder(
    cos_data: npt.NDArray[Any], sin_data: npt.NDArray[Any], rotary_interleaved: bool = False
) -> tuple[npt.NDArray[np.dtype("bfloat16")], npt.NDArray[np.dtype("bfloat16")]]:
    cos_shape_0, cos_shape_1 = cos_data.shape
    m_stride = cos_shape_1 * 4
    diff_offset = cos_shape_1 * 2

    # Flatten for indexing
    cos_data_flat = cos_data.reshape(-1).astype(np.float32)
    sin_data_flat = sin_data.reshape(-1).astype(np.float32)

    # Preallocate output buffers
    total_len = cos_shape_0 * m_stride  # include diff_offset
    cos_cache = np.zeros((total_len,), dtype=ml_dtypes.bfloat16)
    sin_cache = np.zeros((total_len,), dtype=ml_dtypes.bfloat16)

    for i in range(cos_shape_0):
        offset = i * m_stride
        for j in range(cos_shape_1):
            idx = i * cos_shape_1 + j

            cos_val = cos_data_flat[idx]
            sin_val = sin_data_flat[idx]

            cos_bf16 = ml_dtypes.bfloat16(cos_val)
            sin_bf16 = ml_dtypes.bfloat16(sin_val)

            cos_trunc = np.float32(cos_bf16)
            sin_trunc = np.float32(sin_bf16)

            loss_cos = ml_dtypes.bfloat16(cos_val - cos_trunc)
            loss_sin = ml_dtypes.bfloat16(sin_val - sin_trunc)

            neg_sin_bf16 = ml_dtypes.bfloat16(0) if sin_bf16 == 0 else -sin_bf16
            neg_loss_sin = ml_dtypes.bfloat16(0) if loss_sin == 0 else -loss_sin

            if not rotary_interleaved:
                cos_cache[offset + j] = cos_bf16
                cos_cache[offset + cos_shape_1 + j] = cos_bf16
                cos_cache[offset + diff_offset + j] = loss_cos
                cos_cache[offset + diff_offset + cos_shape_1 + j] = loss_cos

                sin_cache[offset + j] = sin_bf16
                sin_cache[offset + cos_shape_1 + j] = sin_bf16
                sin_cache[offset + diff_offset + j] = loss_sin
                sin_cache[offset + diff_offset + cos_shape_1 + j] = loss_sin
            else:
                cos_cache[offset + 2 * j] = cos_bf16
                cos_cache[offset + 2 * j + 1] = cos_bf16
                cos_cache[offset + diff_offset + 2 * j] = loss_cos
                cos_cache[offset + diff_offset + 2 * j + 1] = loss_cos

                sin_cache[offset + 2 * j] = neg_sin_bf16
                sin_cache[offset + 2 * j + 1] = sin_bf16
                sin_cache[offset + diff_offset + 2 * j] = neg_loss_sin
                sin_cache[offset + diff_offset + 2 * j + 1] = loss_sin

    cos_cache = cos_cache.reshape(cos_shape_0, -1)
    sin_cache = sin_cache.reshape(cos_shape_0, -1)
    return cos_cache, sin_cache


def create_sin_cos_cache(
    extractor: onnx.utils.Extractor, ends_node: str
) -> tuple[list[onnx.NodeProto], list[onnx.ValueInfoProto]]:
    new_nodes = []
    new_tvis = []

    sin_cache = ryzenai_onnx_utils.matcher.get_initializer_as_numpy("sin_cache", extractor)
    cos_cache = ryzenai_onnx_utils.matcher.get_initializer_as_numpy("cos_cache", extractor)
    cache_size = cos_cache.shape[1]
    head_size = cache_size * 2
    cos_cache_new, sin_cache_new = trig_cache_builder(cos_cache, sin_cache)
    sin_cos_cache = np.concatenate([cos_cache_new, sin_cache_new], axis=0)
    sin_cos_cache = sin_cos_cache.reshape(2, -1, head_size * 2)

    sin_cos_cache_tensor = onnx.helper.make_tensor(
        "sin_cos_cache_prefill",
        onnx.TensorProto.BFLOAT16,
        sin_cos_cache.shape,
        sin_cos_cache.tobytes(),
        True,
    )

    sin_cos_cache_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["sin_cos_cache_prefill"],
        value=sin_cos_cache_tensor,
        name="sin_cos_cache",
    )
    new_tvi = onnx.helper.make_tensor_value_info(
        "sin_cos_cache_prefill",
        onnx.TensorProto.BFLOAT16,
        sin_cos_cache.shape,
    )
    new_nodes.append(sin_cos_cache_node)
    new_tvis.append(new_tvi)

    ends_name = f"{ends_node}_shape"
    shape_node = onnx.helper.make_node("Shape", inputs=[ends_node], outputs=[ends_name], start=1)
    new_nodes.append(shape_node)
    new_tvis.append(
        onnx.helper.make_tensor_value_info(
            ends_name,
            onnx.TensorProto.INT64,
            (1,),
        )
    )

    const_0, const_0_tvi = ryzenai_onnx_utils.matcher.get_integer_const_by_value(0, extractor)
    if const_0 is not None and const_0_tvi is not None:
        new_nodes.append(const_0)
        new_tvis.append(const_0_tvi)

    const_1, const_1_tvi = ryzenai_onnx_utils.matcher.get_integer_const_by_value(0, extractor)
    if const_1 is not None and const_1_tvi is not None:
        new_nodes.append(const_1)
        new_tvis.append(const_1_tvi)

    sin_cos_cache_slice_node = onnx.helper.make_node(
        "Slice",
        inputs=["sin_cos_cache_prefill", "const_0", ends_name, "const_1", "const_1"],
        outputs=["sin_cos_cache_prefill_slice"],
        name="sin_cos_cache_slice",
    )

    new_tvi_2 = onnx.helper.make_tensor_value_info(
        "sin_cos_cache_prefill_slice",
        onnx.TensorProto.BFLOAT16,
        (2, "sequence_length_padded", head_size * 2),
    )
    new_nodes.append(sin_cos_cache_slice_node)
    new_tvis.append(new_tvi_2)

    return new_nodes, new_tvis


def create_rope(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    rope: onnx.NodeProto,
    params: ryzenai_onnx_utils.ReplaceParams,
    output_override: str | None = None,
    create_cache: bool = True,
    is_k: bool = False,
) -> StrictPassOutputArgs:
    domain = params.get_domain("MLADFMHAROPE")

    new_nodes = []
    new_tvis = []
    new_initializers = []
    input_shape = ryzenai_onnx_utils.matcher.get_shape(rope.input[0], extractor)
    assert isinstance(input_shape[2], int)
    head_size = ryzenai_onnx_utils.matcher.get_shape(rope.input[2], extractor)[-1] * 2
    if ryzenai_onnx_utils.matcher.has_attribute(rope, "num_heads"):
        num_heads = onnx.helper.get_node_attr_value(rope, "num_heads")
        if num_heads > 0:  # phi4 rope_dim is not head_size
            head_size = int(input_shape[2] / num_heads)
    assert isinstance(head_size, int) and head_size > 0, "Head size must be a positive integer"
    rope_shape = [int(input_shape[2] / head_size), input_shape[1], head_size]
    input_cast, input_cast_tvis = transform_cast.add_cast_dtype_to_bfloat16_auto(
        rope.input[0], pass_id, domain, extractor, rope_shape
    )
    new_nodes.extend(input_cast)
    new_tvis.extend(input_cast_tvis)

    new_outputs = ["placeholder"]

    if output_override is None:
        # if is_k:
        # rope_shape = [int(input_shape[2] / head_size), 2368, head_size]

        output_cast, output_cast_tvis = transform_cast.add_cast_bfloat16_to_dtype_auto(
            rope.output[0], pass_id, domain, extractor, rope_shape
        )
        new_nodes.extend(output_cast)
        new_tvis.extend(output_cast_tvis)

        new_outputs[0] = output_cast[0].input[0]
    else:
        output_name = f"tmp_{pass_id}"
        new_outputs[0] = output_name
        # new_outputs.append(f"dummy_k_{pass_id}")

        reshape_node, reshape_tvis, reshape_initializer = add_reshape(
            output_name,
            f"reshape_{pass_id}",
            output_override + f"_{pass_id}",
            onnx.TensorProto.BFLOAT16,
            rope_shape,
            (1, *rope_shape),
        )
        new_nodes.append(reshape_node)
        new_tvis.extend(reshape_tvis)

        pad_tensor = onnx.helper.make_tensor(
            f"pad_tensor_{pass_id}",
            onnx.TensorProto.INT64,
            [8],
            [0, 0, 128, 0, 0, 0, 0, 0],
        )
        new_initializers.append(pad_tensor)
        pad = onnx.helper.make_node(
            "Pad",
            inputs=[output_override + f"_{pass_id}", f"pad_tensor_{pass_id}"],
            outputs=[output_override],
            name=output_name + ".pad",
        )
        new_nodes.append(pad)
        new_initializers.append(reshape_initializer)

    # use a common tensor across the whole model
    sin_cos_exists = ryzenai_onnx_utils.matcher.find_consts("sin_cos_cache_prefill", extractor.graph)
    if not sin_cos_exists and create_cache:
        sin_cos_cache, sin_cos_cache_tvi = create_sin_cos_cache(extractor, "input_ids_padded")
        new_nodes.extend(sin_cos_cache)
        new_tvis.extend(sin_cos_cache_tvi)

    new_node = onnx.helper.make_node(
        "MLADFMHAROPE",
        inputs=[input_cast[0].output[0], "sin_cos_cache_prefill_slice"],
        outputs=new_outputs,
        name=f"rope_{pass_id}",
        domain=domain,
    )

    ryzenai_onnx_utils.matcher.add_attribute(new_node, "transpose", "input")
    ryzenai_onnx_utils.matcher.add_attribute(new_node, "op_version", "v2")
    if is_k:
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "use_gm", "v2")
        total_seq_len = int(params.attributes["total_seq_len"])
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "total_seq_len", total_seq_len)

    ryzenai_onnx_utils.matcher.add_attribute(new_node, "head_dim", head_size)
    new_nodes.append(new_node)

    return new_nodes, new_initializers, new_tvis


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    return create_rope(extractor, pass_id, subgraph[0], params)


PATTERN = ["RotaryEmbedding([?,?,?,?], ?)"]
REPLACEMENT = replacement
