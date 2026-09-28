#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import ModelProto, NodeProto
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)

# Shape-only ops that a scalar divisor commutes through unchanged.
PASS_THROUGH = {"Transpose", "Reshape", "Squeeze", "Unsqueeze", "Flatten", "Identity"}


@register_pass
class ONNXFoldDivIntoMatMulPass(ONNXPass):
    """ONNX pass to fold a scalar ``Div`` into an upstream projection's weight.

    The projection may be a ``MatMul(const W)``, a ``MatMul`` followed by an
    ``Add(const bias)``, or a ``Gemm(A, W[, bias])`` (graph simplification such as
    onnxslim commonly fuses ``MatMul + Add`` into a ``Gemm``). Two patterns are
    handled (``c`` is a scalar constant divisor):

    (A) ``... -> Proj -> (shape ops) -> Div(c) -> MatMul/Gemm``
        The ``Div`` scales an input of a matmul. Since ``(X/c) @ Y == X @ (W/c)``
        for the projection ``X = A @ W [+ b]``, the whole projection weight (and
        bias) is divided by ``c``.

    (B) ``... -> Proj -> [Split] -> (shape ops) -> q``, ``q @ k^T -> Div(c) -> Softmax``
        (attention ``QK^T / sqrt(d)``). The ``Div`` scales the output of a
        two-activation ``MatMul``. Since ``(Q @ K^T)/c == (Q/c) @ K^T``, only the
        query projection's output slice (weight + bias) is divided by ``c``. Fused
        QKV projections are supported by scaling only the query partition selected
        by the ``Split`` (shape ops between the ``Split`` and the projection are
        traversed).

    In both cases the ``Div`` node is removed and its consumers are rewired to its
    numerator input. The transform is exact when ``c`` is a power of two.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return default pass configuration.

        Returns:
            dict[str, PassConfigParam]: Configuration controlling whether the scalar
            Div is folded into the upstream MatMul/Gemm projection.
        """
        config = {
            "fold_div_into_matmul": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to fold a scalar Div (e.g. attention QK^T/sqrt(d) scaling) "
                "into an upstream MatMul/Gemm projection weight.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_fold_div_into_matmul(self, model: ModelProto) -> ModelProto:
        """Fold scalar Div nodes into their upstream projection inside an ONNX model.

        Args:
            model (ModelProto): The ONNX model.

        Returns:
            ModelProto: Updated model with foldable Div nodes removed.
        """
        onnx_model = ONNXModel(model)
        graph = onnx_model.model.graph

        inits = {i.name: i for i in graph.initializer}
        producer = {o: n for n in graph.node for o in n.output}
        consumers: dict[str, list[NodeProto]] = {}
        for node in graph.node:
            for inp in node.input:
                consumers.setdefault(inp, []).append(node)

        def get_const(name: str) -> NDArray[np.float32] | None:
            if name in inits:
                return onnx.numpy_helper.to_array(inits[name])
            prod = producer.get(name)
            if prod is not None and prod.op_type == "Constant":
                for attr in prod.attribute:
                    if attr.name == "value":
                        return onnx.numpy_helper.to_array(attr.t)
            return None

        def get_attr_i(node: NodeProto, name: str, default: int) -> int:
            for attr in node.attribute:
                if attr.name == name:
                    return int(attr.i)
            return default

        def match_projection(node: NodeProto | None) -> dict[str, Any] | None:
            """If ``node`` is a projection, return a dict describing where to scale:
            ``w_node``/``w_name`` (weight to scale) and its ``w_axis`` (output-feature
            axis, 0 or -1), ``b_node``/``b_name`` (bias to scale, or None), and
            ``out_dim`` (size of the output-feature axis). Handles MatMul, MatMul+Add
            and Gemm."""
            if node is None:
                return None
            if node.op_type == "Add":
                consts = [i for i in node.input if get_const(i) is not None]
                acts = [i for i in node.input if get_const(i) is None]
                if (
                    len(consts) == 1
                    and len(acts) == 1
                    and producer.get(acts[0]) is not None
                    and producer[acts[0]].op_type == "MatMul"
                ):
                    mm = producer[acts[0]]
                    w_names = [i for i in mm.input if get_const(i) is not None]
                    if len(w_names) != 1:
                        return None
                    w_arr = get_const(w_names[0])
                    assert w_arr is not None
                    return {
                        "w_node": mm,
                        "w_name": w_names[0],
                        "w_axis": -1,
                        "b_node": node,
                        "b_name": consts[0],
                        "out_dim": int(w_arr.shape[-1]),
                    }
                return None
            if node.op_type == "MatMul":
                w_names = [i for i in node.input if get_const(i) is not None]
                if len(w_names) != 1:
                    return None
                w_arr = get_const(w_names[0])
                assert w_arr is not None
                return {
                    "w_node": node,
                    "w_name": w_names[0],
                    "w_axis": -1,
                    "b_node": None,
                    "b_name": None,
                    "out_dim": int(w_arr.shape[-1]),
                }
            if node.op_type == "Gemm":
                w_arr = get_const(node.input[1]) if len(node.input) >= 2 else None
                if w_arr is None:
                    return None
                # Gemm: Y = alpha*(A' @ B') + beta*C. With transB=1, B is [N, K] and the
                # output-feature axis of the weight is 0; otherwise B is [K, N] (axis -1).
                w_axis = 0 if get_attr_i(node, "transB", 0) == 1 else -1
                b_name = None
                if len(node.input) >= 3 and node.input[2] and get_const(node.input[2]) is not None:
                    b_name = node.input[2]
                return {
                    "w_node": node,
                    "w_name": node.input[1],
                    "w_axis": w_axis,
                    "b_node": node if b_name is not None else None,
                    "b_name": b_name,
                    "out_dim": int(w_arr.shape[w_axis]),
                }
            return None

        def trace_projection(start: str, allow_split: bool) -> tuple[dict[str, Any], int, int] | None:
            """Walk back from tensor ``start`` through shape ops (and one Split if
            allow_split) to a projection. Returns (projection_info, col_start, col_end)
            where the column range is the query partition (whole weight if no Split)."""
            cur = start
            split_info = None  # (q_idx, split_node)
            for _ in range(32):
                prod = producer.get(cur)
                if prod is None:
                    return None
                if prod.op_type in PASS_THROUGH:
                    cur = prod.input[0]
                    continue
                if allow_split and prod.op_type == "Split" and split_info is None:
                    split_info = (list(prod.output).index(cur), prod)
                    cur = prod.input[0]
                    continue
                proj = match_projection(prod)
                if proj is None:
                    return None
                out_dim = proj["out_dim"]
                if split_info is None:
                    return proj, 0, out_dim
                q_idx, split_node = split_info
                splits = None
                split_arr = get_const(split_node.input[1]) if len(split_node.input) > 1 else None
                if split_arr is not None:
                    splits = split_arr.astype(int).tolist()
                else:
                    for attr in split_node.attribute:
                        if attr.name == "split":
                            splits = list(attr.ints)
                if splits is None:
                    n_out = len(split_node.output)
                    splits = [out_dim // n_out] * n_out
                col_start = int(sum(splits[:q_idx]))
                col_end = int(col_start + splits[q_idx])
                return proj, col_start, col_end
            return None

        def scaled_clone(name: str, c: float, col_start: int, col_end: int, axis: int) -> str:
            src = get_const(name)
            assert src is not None
            arr = src.astype(np.float32).copy()
            if axis == 0:
                arr[col_start:col_end, ...] = arr[col_start:col_end, ...] / np.float32(c)
            else:
                arr[..., col_start:col_end] = arr[..., col_start:col_end] / np.float32(c)
            new_name = name + "_divfold"
            new_init = onnx.numpy_helper.from_array(arr, new_name)
            onnx_model.add_initializer(new_init)
            inits[new_name] = new_init
            return new_name

        remove_nodes = []

        for div in [n for n in graph.node if n.op_type == "Div"]:
            divisor = get_const(div.input[1])
            if divisor is None or divisor.size != 1:
                continue
            c = float(np.asarray(divisor).reshape(-1)[0])
            if c in (0.0, 1.0):
                continue

            traced = None
            if any(x.op_type in ("MatMul", "Gemm") for x in consumers.get(div.output[0], [])):
                # Pattern A: Div scales a matmul input.
                traced = trace_projection(div.input[0], allow_split=False)
            else:
                # Pattern B: Div scales the output of a two-activation MatMul.
                numerator = producer.get(div.input[0])
                if (
                    numerator is not None
                    and numerator.op_type == "MatMul"
                    and not any(i in inits for i in numerator.input)
                ):
                    traced = trace_projection(numerator.input[0], allow_split=True)

            if traced is None:
                continue
            proj, col_start, col_end = traced

            w_name = proj["w_name"]
            if w_name not in inits:
                continue
            b_name = proj["b_name"]
            if b_name is not None and b_name not in inits:
                continue

            new_w = scaled_clone(w_name, c, col_start, col_end, proj["w_axis"])
            proj["w_node"].input[:] = [new_w if x == w_name else x for x in proj["w_node"].input]
            if b_name is not None:
                new_b = scaled_clone(b_name, c, col_start, col_end, 0)
                proj["b_node"].input[:] = [new_b if x == b_name else x for x in proj["b_node"].input]

            # Splice out the Div: make the tensor feeding it (numerator) carry the Div's
            # output name, redirecting any other consumers. This preserves the case where
            # the Div output is a graph output (which node-only rewiring would orphan).
            num_name = div.input[0]
            out_name = div.output[0]
            num_producer = producer.get(num_name)
            if num_producer is not None:
                for other in consumers.get(num_name, []):
                    if other is div:
                        continue
                    other.input[:] = [out_name if i == num_name else i for i in other.input]
                num_producer.output[:] = [out_name if o == num_name else o for o in num_producer.output]
            else:
                for x in consumers.get(out_name, []):
                    x.input[:] = [num_name if i == out_name else i for i in x.input]
            remove_nodes.append(div)
            logger.info(f"Folded Div {div.name} (scale {c}) into {proj['w_node'].op_type} {proj['w_node'].name}.")

        if remove_nodes:
            onnx_model.remove_nodes(remove_nodes)
            onnx_model.clean_initializers()
            onnx_model.topological_sort()

        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """Run this pass using the provided configuration.

        Args:
            model (ModelProto): The ONNX model to process.
            config (dict[str, PassConfigParam]): Configuration dict.

        Returns:
            ModelProto: Processed model with foldable Div nodes removed if enabled.
        """
        if "fold_div_into_matmul" in config and config["fold_div_into_matmul"]:
            model = self._onnx_fold_div_into_matmul(model)
        else:
            logger.warning(
                "Please ensure that the onnx_fold_div_into_matmul pass contains the "
                "fold_div_into_matmul parameter and it is True."
            )
        return model
