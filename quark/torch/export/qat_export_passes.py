#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# QAT-specific post-export ONNX passes.
#
# These passes are called from export_onnx_model_optimization() in onnx.py
# and apply only to QAT exports via quark.torch.  They are not used by the
# PTQ path (quark.onnx).
#
# All passes operate on a saved ONNX file (path: str) and overwrite it in place.

from collections import Counter

import numpy as np
import onnx
from onnx import numpy_helper
from onnxslim import slim

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__: list[str] = ["fold_quantizers_for_weight", "fold_constant_reshape_after_dequant", "merge_consecutive_slices"]


# Numpy integer types (and their value ranges) that a constant QuantizeLinear may target.
# int32 is intentionally excluded: int32 constants (bias) are handled by fold_quantizers_for_bias.
_WEIGHT_QUANT_NP_DTYPES = (np.int8, np.uint8)


def _topological_sort(graph: onnx.GraphProto) -> None:
    """Reorder graph.node so every node appears after the nodes producing its inputs.

    Passes here may append freshly created nodes (e.g. cloned DequantizeLinear) after the
    consumers that already reference them. The trailing onnxslim call normally re-sorts the
    graph, but if that slim fails we must not save a non-topologically-sorted (invalid) graph.
    Initializers and graph inputs are treated as already available.
    """
    available = {init.name for init in graph.initializer}
    available.update(inp.name for inp in graph.input)

    remaining = list(graph.node)
    ordered: list[onnx.NodeProto] = []
    # Kahn-style: repeatedly emit any node whose inputs are all available.
    while remaining:
        progressed = False
        still: list[onnx.NodeProto] = []
        for node in remaining:
            if all((not t) or t in available for t in node.input):
                ordered.append(node)
                available.update(node.output)
                progressed = True
            else:
                still.append(node)
        remaining = still
        if not progressed:
            # Cyclic or dangling input we cannot resolve; keep the rest in place to avoid
            # dropping nodes, and stop (checker will surface any real problem).
            ordered.extend(remaining)
            break

    del graph.node[:]
    graph.node.extend(ordered)


def fold_quantizers_for_weight(model_path: str) -> None:
    """
    Fold a constant ``float -> QuantizeLinear -> DequantizeLinear`` into
    ``int -> DequantizeLinear`` for weights and other quantized constants.

    QAT keeps weights (and PRelu slopes, constant Add/Div operands) as float and
    exports them as ``float_initializer -> QuantizeLinear -> DequantizeLinear``.
    PTQ ships the same constants pre-quantized as ``int_initializer -> DequantizeLinear``
    with no QuantizeLinear. This pass pre-computes the int constant and removes the
    QuantizeLinear so the QAT graph matches PTQ.

    The match is by pattern only (a QuantizeLinear whose ``input[0]`` is an
    initializer and whose zero-point is int8/uint8), not by consumer op type, so it
    covers Conv/ConvTranspose weights, PRelu slopes and constant arithmetic operands
    alike. int32 constants (bias) are left to :func:`fold_quantizers_for_bias`.

    Before:
        QuantizeLinear (fp32_const, zp(int8), scale(fp32))
                |
        DequantizeLinear (zp(int8), scale(fp32))
                |
    After:
        DequantizeLinear (int8_const, zp(int8), scale(fp32))
                |
    """
    model = onnx.load(model_path)
    try:
        model = slim(model)
    except Exception:
        logger.warning("During fold weight, simplify onnx model failed, skip fold_quantizers_for_weight")
        return

    graph = model.graph
    name_to_initializer = {init.name: init for init in graph.initializer}
    input_0_to_dequant_node = {node.input[0]: node for node in graph.node if node.op_type == "DequantizeLinear"}

    # Consumers per tensor and graph outputs, so we only fold a private Q -> DQ path and never
    # delete a tensor another node (or a graph output) still reads.
    tensor_consumers: dict[str, int] = {}
    for node in graph.node:
        for tensor in node.input:
            tensor_consumers[tensor] = tensor_consumers.get(tensor, 0) + 1
    graph_outputs = {out.name for out in graph.output}

    # Step 1: find each constant QuantizeLinear (int8/uint8) followed by a DequantizeLinear.
    fold_targets = []
    for node in graph.node:
        if node.op_type != "QuantizeLinear" or node.input[0] not in name_to_initializer:
            continue
        quant_node = node
        zero_point_name = quant_node.input[2]
        if zero_point_name not in name_to_initializer:
            continue
        zero_point = numpy_helper.to_array(name_to_initializer[zero_point_name])
        if zero_point.dtype.type not in _WEIGHT_QUANT_NP_DTYPES:
            continue

        dequant_node = input_0_to_dequant_node.get(quant_node.output[0])
        if dequant_node is None or dequant_node.op_type != "DequantizeLinear":
            continue

        # The QuantizeLinear output must be private: consumed only by this DequantizeLinear and
        # not exposed as a graph output. Otherwise removing the QuantizeLinear would leave a
        # dangling or invalid tensor.
        if tensor_consumers.get(quant_node.output[0], 0) != 1 or quant_node.output[0] in graph_outputs:
            continue

        fold_targets.append((quant_node, dequant_node))

    if len(fold_targets) == 0:
        return

    # Step 2: pre-compute the int constant, rewire the DequantizeLinear, drop the QuantizeLinear.
    # onnxslim may dedupe identical float constants into a single shared initializer, so one
    # float constant can feed several QuantizeLinear nodes. Compute the int constant once per
    # (float_const, quant_node) target with a unique name, and remove each float initializer at
    # most once.
    removed_nodes = []
    removed_float_names = []
    for idx, (quant_node, dequant_node) in enumerate(fold_targets):
        float_const_name = quant_node.input[0]
        scale = numpy_helper.to_array(name_to_initializer[quant_node.input[1]])
        zero_point = numpy_helper.to_array(name_to_initializer[quant_node.input[2]])
        float_const = numpy_helper.to_array(name_to_initializer[float_const_name])

        # Reshape scale/zero_point to broadcast along the quant axis for per-channel quant.
        axis_attrs = [attr.i for attr in quant_node.attribute if attr.name == "axis"]
        if axis_attrs and scale.ndim == 1:
            axis = axis_attrs[0] % float_const.ndim
            broadcast_shape = [1] * float_const.ndim
            broadcast_shape[axis] = scale.shape[0]
            scale = scale.reshape(broadcast_shape)
            zero_point = zero_point.reshape(broadcast_shape)

        numpy_dtype = zero_point.dtype.type
        dtype_info = np.iinfo(numpy_dtype)
        quantized = np.round(float_const / scale + zero_point)
        quantized = np.clip(quantized, dtype_info.min, dtype_info.max).astype(numpy_dtype)

        # Unique per target so a shared float constant does not collide on the int name.
        int_const_name = f"{float_const_name}_quantized_{idx}"
        graph.initializer.append(numpy_helper.from_array(quantized, name=int_const_name))
        dequant_node.input[0] = int_const_name

        removed_nodes.append(quant_node)
        removed_float_names.append(float_const_name)

    for node in removed_nodes:
        graph.node.remove(node)
    # Remove each float initializer once, and only when every node that read it was folded away
    # (all its consumers were QuantizeLinear targets we just removed). A shared constant still
    # referenced by a non-folded node is kept to avoid leaving a dangling input.
    folded_per_float = Counter(removed_float_names)
    for float_name in dict.fromkeys(removed_float_names):
        if float_name in name_to_initializer and folded_per_float[float_name] >= tensor_consumers.get(float_name, 0):
            graph.initializer.remove(name_to_initializer[float_name])

    folded_num = len(fold_targets)
    try:
        model = slim(model)
    except Exception:
        logger.warning("After fold weight, simplify onnx model failed, saving unsimplified model")
    onnx.save_model(model, model_path)
    logger.info(
        f"Fold constant weight QuantizeLinear into DequantizeLinear to match PTQ packaging, total convert: {folded_num}"
    )
    return


def _get_axes(node: onnx.NodeProto, name_to_initializer: dict[str, onnx.TensorProto]) -> list[int] | None:
    """Return the Unsqueeze axes, from the attribute (opset < 13) or the
    axes input initializer (opset >= 13), or None if not statically known."""
    for attr in node.attribute:
        if attr.name == "axes":
            return list(attr.ints)
    if len(node.input) > 1 and node.input[1] in name_to_initializer:
        return [int(a) for a in numpy_helper.to_array(name_to_initializer[node.input[1]])]
    return None


def fold_constant_reshape_after_dequant(model_path: str) -> None:
    """
    Fold a constant ``int -> DequantizeLinear -> Unsqueeze`` into
    ``reshaped_int -> DequantizeLinear`` so the residual reshape node disappears.

    A PRelu slope declared as ``nn.PReLU(C)`` has a 1-D ``(C,)`` weight; torch.onnx
    export inserts an Unsqueeze to broadcast it to ``(1, C, 1, 1)`` before the PRelu.
    PTQ instead ships the slope already shaped, so it has no such Unsqueeze. This
    pass reshapes the (already int, per-tensor) constant to the unsqueezed shape and
    rewires consumers straight to the DequantizeLinear, matching PTQ.

    Only per-tensor DequantizeLinear (scalar scale/zero-point) is folded, so that
    reshaping the constant before dequantization is numerically identical:
    ``Unsqueeze(DQ(x)) == DQ(Unsqueeze(x))``.

    Before:
        int_const -> DequantizeLinear (scalar scale/zp) -> Unsqueeze -> consumer
    After:
        reshaped_int_const -> DequantizeLinear (scalar scale/zp) -> consumer
    """
    model = onnx.load(model_path)
    try:
        model = slim(model)
    except Exception:
        logger.warning("During fold reshape, simplify onnx model failed, skip fold_constant_reshape_after_dequant")
        return

    graph = model.graph
    name_to_initializer = {init.name: init for init in graph.initializer}
    producer = {out: node for node in graph.node for out in node.output}

    # Consumers per tensor and graph outputs. A single slope DequantizeLinear may be shared by
    # many Unsqueeze->PRelu branches, so we never mutate a shared DequantizeLinear in place;
    # instead each branch gets its own cloned DequantizeLinear reading the reshaped constant,
    # leaving sibling branches untouched. Redundant clones are merged by onnxslim afterwards.
    graph_outputs = {out.name for out in graph.output}

    removed_nodes = []
    added_nodes: list[onnx.NodeProto] = []
    reshaped_const_added: set[str] = set()
    clone_idx = 0
    folded_num = 0
    for node in graph.node:
        if node.op_type != "Unsqueeze":
            continue
        # A graph-output Unsqueeze cannot be folded away without leaving a dangling output.
        if node.output[0] in graph_outputs:
            continue
        dequant_node = producer.get(node.input[0])
        if dequant_node is None or dequant_node.op_type != "DequantizeLinear":
            continue
        # The dequantized constant must originate from an int initializer.
        int_const_name = dequant_node.input[0]
        if int_const_name not in name_to_initializer:
            continue
        # Only per-tensor (scalar scale/zero-point) is safe to reshape before dequant.
        scale = numpy_helper.to_array(name_to_initializer[dequant_node.input[1]])
        if scale.ndim != 0:
            continue
        axes = _get_axes(node, name_to_initializer)
        if axes is None:
            continue

        int_const = numpy_helper.to_array(name_to_initializer[int_const_name])
        reshaped = np.expand_dims(int_const, axis=tuple(axes))

        # Key the reshaped constant on the axes too: the same source constant may feed
        # several Unsqueeze with different axes, which produce different shapes. Keying
        # only on the source name would make later branches reuse the first branch's shape.
        axes_tag = "_".join(str(a) for a in axes)
        reshaped_name = f"{int_const_name}_unsqueezed_{axes_tag}"
        if reshaped_name not in reshaped_const_added:
            graph.initializer.append(numpy_helper.from_array(reshaped, name=reshaped_name))
            reshaped_const_added.add(reshaped_name)

        # Clone a private DequantizeLinear for this branch instead of rewriting the (possibly
        # shared) original. The clone reads the reshaped constant and copies scale/zp/attrs.
        clone_output = f"{dequant_node.output[0]}_reshaped_{clone_idx}"
        clone_idx += 1
        clone_dq = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=[reshaped_name, *list(dequant_node.input[1:])],
            outputs=[clone_output],
        )
        clone_dq.attribute.extend(dequant_node.attribute)
        added_nodes.append(clone_dq)

        # Route only this Unsqueeze's consumers to the cloned DequantizeLinear.
        for consumer in graph.node:
            for idx, tensor in enumerate(consumer.input):
                if tensor == node.output[0]:
                    consumer.input[idx] = clone_output

        removed_nodes.append(node)
        folded_num += 1

    if folded_num == 0:
        return

    graph.node.extend(added_nodes)
    for node in removed_nodes:
        graph.node.remove(node)

    try:
        model = slim(model)
    except Exception:
        logger.warning("After fold reshape, simplify onnx model failed, saving unsimplified model")
        # slim would have re-sorted the graph; do it ourselves so the cloned DequantizeLinear
        # nodes (appended after their consumers) do not leave a non-topologically-sorted graph.
        _topological_sort(model.graph)
    onnx.save_model(model, model_path)
    logger.info(
        f"Fold constant Unsqueeze after DequantizeLinear into the constant to match PTQ, total convert: {folded_num}"
    )
    return


def _get_slice_params(
    slice_node: onnx.NodeProto, name_to_initializer: dict[str, onnx.TensorProto]
) -> dict[int, tuple[int, int, int]] | None:
    """Return {axis: (start, end, step)} for a Slice whose starts/ends/axes/steps are
    all constant initializers, or None if any is dynamic / missing.

    ONNX Slice inputs: data(0), starts(1), ends(2), axes(3, optional), steps(4, optional).
    """
    if len(slice_node.input) < 3:
        return None
    starts_name, ends_name = slice_node.input[1], slice_node.input[2]
    if starts_name not in name_to_initializer or ends_name not in name_to_initializer:
        return None
    starts = numpy_helper.to_array(name_to_initializer[starts_name])
    ends = numpy_helper.to_array(name_to_initializer[ends_name])

    if len(slice_node.input) >= 4 and slice_node.input[3]:
        if slice_node.input[3] not in name_to_initializer:
            return None
        axes = numpy_helper.to_array(name_to_initializer[slice_node.input[3]])
    else:
        axes = np.arange(len(starts))

    if len(slice_node.input) >= 5 and slice_node.input[4]:
        if slice_node.input[4] not in name_to_initializer:
            return None
        steps = numpy_helper.to_array(name_to_initializer[slice_node.input[4]])
    else:
        steps = np.ones(len(starts), dtype=np.int64)

    params = {}
    for axis, start, end, step in zip(axes, starts, ends, steps, strict=False):
        params[int(axis)] = (int(start), int(end), int(step))
    return params


def merge_consecutive_slices(model_path: str) -> None:
    """
    Merge two consecutive per-axis Slices, separated only by a transparent Q/DQ
    pair, into a single multi-axis Slice — matching the PTQ graph.

    QAT keeps each Slice bracketed by its own Q/DQ, so two per-axis Slices export as
    ``DequantizeLinear -> Slice(A) -> QuantizeLinear -> DequantizeLinear -> Slice(B) -> QuantizeLinear``.
    PTQ folds the pair on the float graph before quantizing, ending up with one
    multi-axis Slice. This pass reproduces that on the exported QAT graph WITHOUT
    changing QAT training (the intermediate quantizer is only dropped here, post-export).

    Before:
        ... -> Slice(A) -> QuantizeLinear(Q) -> DequantizeLinear(DQ) -> Slice(B) -> ...
    After:
        ... -> Slice(A+B, multi-axis) -> ...

    Only merges when it is safe and lossless:
      - the intermediate Q and DQ share the same (scalar) scale and zero-point, so
        they are a matched pair;
      - the tensor entering Slice A is already on that same quant grid, i.e. Slice A's
        data input is produced by a DequantizeLinear with the same scalar scale and
        zero-point. Then Slice A's output is still on-grid (slicing does not move values
        off the grid), so the intermediate Q->DQ is an exact round-trip and dropping it
        does not change any value. Equal scale/zp alone does NOT prove this: if the tensor
        were off-grid the Q->DQ would round it, and removing it would change the output;
      - Slice A's output, Q's output and DQ's output are each single-consumer, so no
        other branch (e.g. a Concat fed by Slice A) is detached;
      - both Slices' starts/ends/axes/steps are constant initializers and act on
        disjoint axes (so the two can be composed into one multi-axis Slice).
    """
    model = onnx.load(model_path)
    try:
        model = slim(model)
    except Exception:
        logger.warning("During merge slice, simplify onnx model failed, skip merge_consecutive_slices")
        return

    graph = model.graph
    name_to_initializer = {init.name: init for init in graph.initializer}
    producer = {out: node for node in graph.node for out in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(node)

    graph_output_names = {out.name for out in graph.output}

    def single_consumer(tensor: str) -> bool:
        # A private edge: exactly one node consumes it and it is not a graph output.
        return len(consumers.get(tensor, [])) == 1 and tensor not in graph_output_names

    def scales_equal(quant_node: onnx.NodeProto, dequant_node: onnx.NodeProto) -> bool:
        # scale must be equal scalar, zero-point must be equal.
        q_scale = name_to_initializer.get(quant_node.input[1])
        d_scale = name_to_initializer.get(dequant_node.input[1])
        if q_scale is None or d_scale is None:
            return False
        q_scale_arr = numpy_helper.to_array(q_scale)
        d_scale_arr = numpy_helper.to_array(d_scale)
        if q_scale_arr.ndim != 0 or d_scale_arr.ndim != 0:
            return False
        if not np.array_equal(q_scale_arr, d_scale_arr):
            return False
        q_zp = numpy_helper.to_array(name_to_initializer[quant_node.input[2]]) if len(quant_node.input) > 2 else None
        d_zp = (
            numpy_helper.to_array(name_to_initializer[dequant_node.input[2]]) if len(dequant_node.input) > 2 else None
        )
        return np.array_equal(q_zp, d_zp)

    def scalar_scale_zp(node: onnx.NodeProto) -> tuple[np.ndarray, np.ndarray | None] | None:
        # Return (scale, zero_point) as scalars for a Q/DQ node, or None if not scalar.
        scale_init = name_to_initializer.get(node.input[1])
        if scale_init is None:
            return None
        scale = numpy_helper.to_array(scale_init)
        if scale.ndim != 0:
            return None
        zp = numpy_helper.to_array(name_to_initializer[node.input[2]]) if len(node.input) > 2 else None
        return (scale, zp)

    def input_on_quant_grid(slice_a: onnx.NodeProto, quant_node: onnx.NodeProto) -> bool:
        # The intermediate Q->DQ is only a lossless round-trip if the tensor entering
        # Slice A is already on the same quant grid. That holds when Slice A's data input
        # is produced by a DequantizeLinear whose scalar scale/zero-point equal the
        # intermediate quantizer's. Slicing keeps values on-grid, so Slice A's output is
        # on-grid and the intermediate Q->DQ cannot change any value.
        upstream = producer.get(slice_a.input[0])
        if upstream is None or upstream.op_type != "DequantizeLinear":
            return False
        up = scalar_scale_zp(upstream)
        qn = scalar_scale_zp(quant_node)
        if up is None or qn is None:
            return False
        return np.array_equal(up[0], qn[0]) and np.array_equal(up[1], qn[1])

    removed_nodes = []
    merged_count = 0
    for slice_b in graph.node:
        if slice_b.op_type != "Slice":
            continue
        # slice_b input traces back: DQ <- Q <- Slice(A)
        dequant_node = producer.get(slice_b.input[0])
        if dequant_node is None or dequant_node.op_type != "DequantizeLinear":
            continue
        quant_node = producer.get(dequant_node.input[0])
        if quant_node is None or quant_node.op_type != "QuantizeLinear":
            continue
        slice_a = producer.get(quant_node.input[0])
        if slice_a is None or slice_a.op_type != "Slice":
            continue

        # fan-out guard: the whole A -> Q -> DQ -> B chain must be a private path.
        if not (
            single_consumer(slice_a.output[0])
            and single_consumer(quant_node.output[0])
            and single_consumer(dequant_node.output[0])
        ):
            continue

        # transparent intermediate quantizer.
        if not scales_equal(quant_node, dequant_node):
            continue

        # lossless guard: the tensor entering Slice A must already be on the same quant
        # grid, otherwise dropping the intermediate Q->DQ would change values.
        if not input_on_quant_grid(slice_a, quant_node):
            continue

        params_a = _get_slice_params(slice_a, name_to_initializer)
        params_b = _get_slice_params(slice_b, name_to_initializer)
        if params_a is None or params_b is None:
            continue
        # must act on disjoint axes to compose into one multi-axis Slice.
        if set(params_a) & set(params_b):
            continue

        merged = {**params_a, **params_b}
        axes_sorted = sorted(merged)
        starts = np.array([merged[axis][0] for axis in axes_sorted], dtype=np.int64)
        ends = np.array([merged[axis][1] for axis in axes_sorted], dtype=np.int64)
        axes = np.array(axes_sorted, dtype=np.int64)
        steps = np.array([merged[axis][2] for axis in axes_sorted], dtype=np.int64)

        prefix = slice_b.output[0] + "_merged"
        graph.initializer.extend(
            [
                numpy_helper.from_array(starts, name=prefix + "_starts"),
                numpy_helper.from_array(ends, name=prefix + "_ends"),
                numpy_helper.from_array(axes, name=prefix + "_axes"),
                numpy_helper.from_array(steps, name=prefix + "_steps"),
            ]
        )
        # rewrite slice_b in place into the merged multi-axis Slice reading slice_a's input.
        del slice_b.input[:]
        slice_b.input.extend(
            [slice_a.input[0], prefix + "_starts", prefix + "_ends", prefix + "_axes", prefix + "_steps"]
        )

        removed_nodes.extend([slice_a, quant_node, dequant_node])
        merged_count += 1

    if merged_count == 0:
        return

    for node in removed_nodes:
        graph.node.remove(node)

    try:
        model = slim(model)
    except Exception:
        logger.warning("After merge slice, simplify onnx model failed, saving unsimplified model")
    onnx.save_model(model, model_path)
    logger.info(f"Merge consecutive Slice pairs into one multi-axis Slice to match PTQ, total merged: {merged_count}")
    return


def merge_equivalent_constant_dequantizers(model_path: str) -> None:
    """Merge identical PRelu-slope DequantizeLinear nodes into a single shared one.

    fold_constant_reshape_after_dequant clones a private DequantizeLinear per PRelu branch to
    avoid mutating a shared node, so a slope shared by N PRelu becomes N identical
    ``int_const -> DequantizeLinear`` nodes. PTQ ships a single shared DequantizeLinear feeding
    all N PRelu. This pass deduplicates exactly those slope dequantizers: it only considers a
    constant-fed DequantizeLinear whose every consumer is a PRelu reading it as the slope
    (input[1]). Nodes with equal inputs (constant, scale, zero-point) and attributes are
    collapsed onto the first one. Conv/Linear weight dequantizers and activation quantizers are
    never touched, so distinct weights that happen to share quant params stay separate.
    """
    model = onnx.load(model_path)
    graph = model.graph
    name_to_initializer = {init.name: init for init in graph.initializer}
    graph_outputs = {out.name for out in graph.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(node)

    def _is_prelu_slope_dequant(node: onnx.NodeProto) -> bool:
        # A DequantizeLinear whose output is a graph output must not be merged away.
        if node.output[0] in graph_outputs:
            return False
        downstream = consumers.get(node.output[0], [])
        # Every consumer must be a PRelu using this tensor as its slope (input[1]).
        return bool(downstream) and all(
            c.op_type == "PRelu" and len(c.input) > 1 and c.input[1] == node.output[0] for c in downstream
        )

    def _attr_key(node: onnx.NodeProto) -> tuple[tuple[str, int], ...]:
        return tuple(sorted((a.name, a.i) for a in node.attribute))

    # Group slope DequantizeLinear nodes by (inputs, attributes).
    groups: dict[tuple[tuple[str, ...], tuple[tuple[str, int], ...]], list[onnx.NodeProto]] = {}
    for node in graph.node:
        if node.op_type != "DequantizeLinear" or node.input[0] not in name_to_initializer:
            continue
        if not _is_prelu_slope_dequant(node):
            continue
        key = (tuple(node.input), _attr_key(node))
        groups.setdefault(key, []).append(node)

    output_remap: dict[str, str] = {}
    removed_nodes = []
    for members in groups.values():
        if len(members) < 2:
            continue
        keeper = members[0]
        for dup in members[1:]:
            output_remap[dup.output[0]] = keeper.output[0]
            removed_nodes.append(dup)

    if not removed_nodes:
        return

    for node in graph.node:
        for i, tensor in enumerate(node.input):
            if tensor in output_remap:
                node.input[i] = output_remap[tensor]
    for node in removed_nodes:
        graph.node.remove(node)

    try:
        model = slim(model)
    except Exception:
        logger.warning("After merging equivalent dequantizers, simplify onnx model failed, saving unsimplified model")
    onnx.save_model(model, model_path)
    logger.info(f"Merge equivalent constant DequantizeLinear nodes to match PTQ, total merged: {len(removed_nodes)}")
