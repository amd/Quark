#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unified MixingStrategy: promotes/demotes quantization precision for a set of nodes.

Handles QDQ initializer surgery, BFP custom-op replacement, and MX custom-op
replacement.  Inspects both the activation *and* weight specs at promote() time
so that mixed configurations (e.g. MX weights with INT8 activations) are handled
correctly.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
from onnx import onnx_pb as onnx_proto
from onnxruntime.quantization.calibrate import CalibrationMethod, TensorsData
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import DEQUANT_OP_NAME, QUANT_OP_NAME, ms_domain

from quark.common.utils.log import ScreenLogger
from quark.onnx.calibration.methods import PowerOfTwoMethod
from quark.onnx.quantization.config.maps import _map_activation_calibration_method
from quark.onnx.quantization.config.spec import (
    BFP16Spec,
    MX4Spec,
    MX6Spec,
    MX9Spec,
    MXFP4E2M1Spec,
    MXFP6E2M3Spec,
    MXFP6E3M2Spec,
    MXFP8E4M3Spec,
    MXFP8E5M2Spec,
)
from quark.onnx.quantization.quant_utils import (
    BFP_OP_DEFAULT_ATTRS,
    COP_BFP_OP_NAME,
    COP_DEQUANT_OP_NAME,
    COP_DOMAIN,
    COP_MX_OP_NAME,
    COP_QUANT_OP_NAME,
    DEQUANT_OP_TYPES,
    MX_OP_DEFAULT_ATTRS,
    ONNX_FP_QTYPES_LIST,
    QUANT_OP_TYPES,
    ExtendedQuantType,
    compute_scale_zp,
    compute_scale_zp_fp,
    get_qmin_qmax_for_qType,
    get_tensor_type_from_qType,
)
from quark.onnx.utils.model_utils import ONNXQuantizedModel

if TYPE_CHECKING:
    from quark.onnx.quantization.config.spec import QLayerConfig

logger = ScreenLogger(__name__)


class MixingStrategy:
    """Unified precision promotion/demotion for quantized ONNX models.

    :param target_layer_config: Required. One of three forms:

        - **Single** :class:`QLayerConfig` — applied to every candidate.
        - **Dict** ``{QLayerConfig: list[str]}`` — maps each config to the
          candidate names that should use it.  One entry may map to ``[]``
          to act as the global fallback for unlisted candidates.
        - **List** ``[QLayerConfig, ...]`` — multi-config mode.  Sensitivity
          analysis scores every config per candidate and the one yielding the
          smallest score is selected for mixing.
    :param str shared_param_mode: How to handle scale/zp initializers that are shared
        between the promoted Q/DQ pair and other nodes (e.g. Q/DQ nodes at input and
        output of Transpose). ``"propagate"`` (default) keeps the shared initializer and
        updates the op_type/domain of every node that references it so that all users
        remain consistent with the new dtype. ``"unshare"`` gives the promoted pair
        its own copy of the initializer and leaves the original shared one untouched.
    :param dict[str, Any] extra_options: Extra options for the mixing strategy.
        The options used in the mixing strategy are:
        - **bool** ``"WeightSymmetric"``: Whether to use symmetric quantization for weight.
        - **bool** ``"ActivationSymmetric"``: Whether to use symmetric quantization for activation.
    """

    def __init__(
        self,
        target_layer_config: QLayerConfig | dict[QLayerConfig, list[str]] | list[QLayerConfig],
        shared_param_mode: str = "propagate",
        extra_options: dict[str, Any] = {},
    ) -> None:
        self._target_config_list: list[QLayerConfig] = []

        self._candidate_to_config: dict[str, QLayerConfig] = {}
        if isinstance(target_layer_config, dict):
            # An entry with an empty list [] is the global fallback config.
            global_cfg: QLayerConfig | None = None
            for cfg, names in target_layer_config.items():
                if not names:
                    if global_cfg is not None:
                        logger.warning(
                            "target_layer_config dict must have at most one entry with an empty list (global fallback)."
                        )
                        continue
                    global_cfg = cfg
                else:
                    for name in names:
                        self._candidate_to_config[name] = cfg
            self._target_config_list = [global_cfg if global_cfg is not None else next(iter(target_layer_config))]
        elif isinstance(target_layer_config, list):
            # Multi-config mode: sensitivity analysis scores all configs per
            # candidate and selects the best one (smallest score) for mixing.
            if not target_layer_config:
                raise ValueError("target_layer_config list must not be empty.")
            self._target_config_list = list(target_layer_config)
        else:
            self._target_config_list = [target_layer_config]

        assert len(self._target_config_list) > 0, "Target config list is empty in the strategy."
        self._load_specs_from_config(self._target_config_list[0])

        self._shared_param_mode = shared_param_mode
        self._extra_options = extra_options

    @property
    def target_config_list(self) -> list[QLayerConfig]:
        """The list of config candidates."""
        return self._target_config_list

    @property
    def candidate_to_config(self) -> dict[str, QLayerConfig]:
        """The dictionary of candidate names to config."""
        return self._candidate_to_config

    def _load_specs_from_config(self, cfg: QLayerConfig) -> None:
        """Populate the active spec/format/tensor-type attributes from *cfg*."""
        # Get the data type of input, weight, bias, and output tensors.
        # The spec is a data class, like Int8Spec, UInt8Spec, etc.
        self.input_spec = cfg.activation if cfg.activation else cfg.input_tensors
        self.weight_spec = cfg.weight
        self.bias_spec = cfg.bias
        self.output_spec = cfg.activation if cfg.activation else cfg.output_tensors

        # Convert the data type to format, like QuantType.QInt8, QuantType.QUInt8, etc.
        # It maybe custom format, like ExtendedQuantType.QBFP or ExtendedQuantType.QMX, etc.
        self.input_format = self.input_spec.data_type.map_onnx_format if self.input_spec is not None else None  # type: ignore
        self.weight_format = self.weight_spec.data_type.map_onnx_format if self.weight_spec is not None else None  # type: ignore
        self.bias_format = self.bias_spec.data_type.map_onnx_format if self.bias_spec is not None else None  # type: ignore
        self.output_format = self.output_spec.data_type.map_onnx_format if self.output_spec is not None else None  # type: ignore

        # Convert the format to tensor type, like TensorProto.UINT8, TensorProto.INT8, etc.
        # For custom formats, like ExtendedQuantType.QBFP or ExtendedQuantType.QMX, etc., return TensorProto.UNDEFINED.
        self.input_tensor_type = get_tensor_type_from_qType(self.input_format) if self.input_format is not None else 0
        self.weight_tensor_type = (
            get_tensor_type_from_qType(self.weight_format) if self.weight_format is not None else 0
        )
        self.bias_tensor_type = get_tensor_type_from_qType(self.bias_format) if self.bias_format is not None else 0
        self.output_tensor_type = (
            get_tensor_type_from_qType(self.output_format) if self.output_format is not None else 0
        )

    def promote(
        self,
        work_model: ONNXModel,
        candidate_nodes: list[str],
        tensors_range: TensorsData,
        layer_config: QLayerConfig,
    ) -> set[str]:
        """Apply target precision to *candidate_nodes* in *work_model*."""
        promoted_tensors: set[str] = set()

        for node_name in candidate_nodes:
            config = self._candidate_to_config.get(node_name, layer_config)
            self._load_specs_from_config(config)

            if not (self.input_format or self.weight_format or self.bias_format or self.output_format):
                logger.warning(f"No target layer types for mixing in the config for node '{node_name}'; skipping.")
                continue

            node_struct = self._get_node_struct(work_model, node_name)
            if not isinstance(node_struct, dict) or not node_struct.get("node"):
                logger.warning(f"Node '{node_name}' not found or has no QDQ structure; skipping.")
                continue
            promoted_tensors |= self._handle_target_qdqs(work_model, node_struct, tensors_range)

        return promoted_tensors

    def demote(
        self,
        work_model: ONNXModel,
        prev_proto: onnx.ModelProto,
    ) -> None:
        """Revert *work_model* to *prev_proto*, undoing the last promotion.

        The caller is responsible for passing a snapshot taken just before the
        last :meth:`promote` call so that only that candidate's changes are
        rolled back, preserving all earlier successful promotions.
        """
        work_model.model.CopyFrom(prev_proto)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_node_struct(self, work_model: ONNXModel, node_name: str) -> dict[str, Any] | None:
        """Locate *node_name* in *work_model* and return its QDQ structure dict."""
        target_node = None

        for node in work_model.model.graph.node:
            if node.name == node_name:
                target_node = node
                break

        if target_node is None:
            return None

        parser = ONNXQuantizedModel(work_model.model)
        return parser.get_target_node_struct(target_node)

    @staticmethod
    def _qdq_name_domain(tensor_type: int) -> tuple[str, str, str]:
        if tensor_type in (onnx_proto.TensorProto.UINT8, onnx_proto.TensorProto.INT8):
            # Standard ONNX domain is "" (empty string) in node attributes.
            # "ai.onnx" is documentation shorthand only — using it as node.domain
            # causes shape inference to fail ("No opset import for domain ai.onnx").
            return "QuantizeLinear", "DequantizeLinear", ""
        elif tensor_type in (onnx_proto.TensorProto.UINT16, onnx_proto.TensorProto.INT16, onnx_proto.TensorProto.INT32):
            return QUANT_OP_NAME, DEQUANT_OP_NAME, ms_domain
        else:
            return COP_QUANT_OP_NAME, COP_DEQUANT_OP_NAME, COP_DOMAIN

    @staticmethod
    def _custom_op_params(fmt: Any, spec: Any = None) -> tuple[str, dict[str, Any]]:
        """Return the op name and attribute dict for a custom format."""
        if fmt == ExtendedQuantType.QBFP:
            fn_type, attrs = COP_BFP_OP_NAME, copy.deepcopy(BFP_OP_DEFAULT_ATTRS)
            if spec is not None and isinstance(spec, BFP16Spec):
                attrs["bfp_method"] = "to_bfp"
                attrs["bit_width"] = 16
                attrs["block_size"] = 8
                attrs["rounding_mode"] = 2
            elif spec is not None and isinstance(spec, MX4Spec):
                attrs["bfp_method"] = "to_bfp_prime"
                attrs["bit_width"] = 11
                attrs["block_size"] = 16
                attrs["rounding_mode"] = 2
            elif spec is not None and isinstance(spec, MX6Spec):
                attrs["bfp_method"] = "to_bfp_prime"
                attrs["bit_width"] = 13
                attrs["block_size"] = 16
                attrs["rounding_mode"] = 2
            elif spec is not None and isinstance(spec, MX9Spec):
                attrs["bfp_method"] = "to_bfp_prime"
                attrs["bit_width"] = 16
                attrs["block_size"] = 16
                attrs["rounding_mode"] = 2
        else:
            fn_type, attrs = COP_MX_OP_NAME, copy.deepcopy(MX_OP_DEFAULT_ATTRS)
            # The default attrs are for MXINT8Spec,
            # so we need to override the default attributes for other MX formats.
            if spec is not None and isinstance(spec, MXFP4E2M1Spec):
                attrs["element_dtype"] = "fp4_e2m1"
            elif spec is not None and isinstance(spec, MXFP6E3M2Spec):
                attrs["element_dtype"] = "fp6_e3m2"
            elif spec is not None and isinstance(spec, MXFP6E2M3Spec):
                attrs["element_dtype"] = "fp6_e2m3"
            elif spec is not None and isinstance(spec, MXFP8E5M2Spec):
                attrs["element_dtype"] = "fp8_e5m2"
            elif spec is not None and isinstance(spec, MXFP8E4M3Spec):
                attrs["element_dtype"] = "fp8_e4m3"
            attrs["rounding_mode"] = 2
        return fn_type, attrs

    # ------------------------------------------------------------------
    # QDQ Converter
    # ------------------------------------------------------------------

    def _dequantize_weight(self, work_model: ONNXModel, dq: Any) -> np.ndarray[Any, Any] | None:
        """Return float weight data by dequantizing a DQ-only weight node.

        Returns ``None`` if the DQ input is a runtime tensor (not an initializer),
        which is the case for activation DQ nodes.
        """
        quant_init = work_model.get_initializer(dq.input[0])
        if quant_init is None:
            return None  # Not a quantized weight (bias) node
        scale_init = work_model.get_initializer(dq.input[1])
        zp_init = work_model.get_initializer(dq.input[2])
        if scale_init is None or zp_init is None:
            return None
        quant_data = onnx.numpy_helper.to_array(quant_init).astype(np.float32)
        scale = onnx.numpy_helper.to_array(scale_init).astype(np.float32)
        zp = onnx.numpy_helper.to_array(zp_init).astype(np.float32)
        return (quant_data - zp) * scale

    def _scale_zp_from_range(
        self,
        rmin: np.ndarray[Any, Any],
        rmax: np.ndarray[Any, Any],
        quant_type: int,
        symmetric: bool,
        calibration_method: Any,
    ) -> tuple[Any, Any]:
        """Compute (scale_np, zp_np) scalars from a [rmin, rmax] float range."""
        qmin, qmax = get_qmin_qmax_for_qType(quant_type, reduce_range=False, symmetric=symmetric)
        if quant_type in ONNX_FP_QTYPES_LIST:
            # Float-type "quantization" (bfloat16, float16) is a direct precision reduction:
            # scale is always 1.0, zero_point is 0. compute_scale_zp_fp enforces this.
            zero_point, scale = compute_scale_zp_fp(
                rmin,
                rmax,
                qmin,
                qmax,
                element_type=quant_type,
                method=calibration_method,
                symmetric=symmetric,
            )
        else:
            # Here we did not get the settings from extra options,
            # just use the calibration method to determine use_pof2s.
            use_pof2s = isinstance(calibration_method, PowerOfTwoMethod)
            zero_point, scale = compute_scale_zp(
                rmin,
                rmax,
                qmin,
                qmax,
                element_type=quant_type,
                method=calibration_method,
                symmetric=symmetric,
                use_pof2s=use_pof2s,
            )
        scale_np = np.asarray(scale, dtype=rmin.dtype).reshape(())
        zp_np = np.asarray(zero_point, dtype=onnx.helper.tensor_dtype_to_np_dtype(quant_type)).reshape(())
        return scale_np, zp_np

    def _apply_scale_zp(
        self,
        work_model: ONNXModel,
        dq: Any,
        q: Any,
        scale_np: Any,
        zp_np: Any,
        quant_type: int,
        symmetric: bool = False,
    ) -> None:
        """Write new scale/zp into Q/DQ initializers.

        For DQ-only nodes (Q was constant-folded), also re-quantizes the weight
        initializer with the new parameters.
        """
        # Get float data before modifying scale/zp (DQ-only path needs old params)
        # This is usually used for weight and bias, not for activation
        float_data = self._dequantize_weight(work_model, dq) if q is None else None

        def _set(name: str, value: Any) -> None:
            old = work_model.get_initializer(name)
            if old is not None:
                work_model.remove_initializer(old)
            work_model.add_initializer(onnx.numpy_helper.from_array(value, name))

        q_op_name, dq_op_name, domain = self._qdq_name_domain(quant_type)

        # Update op_type and domain so the node is compatible with the new quant type.
        # e.g. "ai.onnx" QuantizeLinear does not support INT16; must switch to ms_domain.
        dq.op_type = dq_op_name
        dq.domain = domain
        if q is not None:
            q.op_type = q_op_name
            q.domain = domain

        # Scale/zp initializers may be shared with nodes outside this Q/DQ pair
        # (e.g. input or output Q/DQ pairs of Transpose or Reshape). Writing new
        # values in-place would silently corrupt those other nodes. Two strategies:
        #   "unshare"   — give the promoted pair its own copy; leave the original.
        #   "propagate" — keep the shared initializer; update every sharing node's
        #                 op_type/domain so all users stay consistent with new dtype.
        promoted_ids = {id(n) for n in (dq, q) if n is not None}
        promoted_nodes = [n for n in (dq, q) if n is not None]

        if self._shared_param_mode == "unshare":

            def _unshare_init(input_idx: int) -> None:
                name = dq.input[input_idx]
                users = [
                    n
                    for n in work_model.model.graph.node
                    if len(n.input) > input_idx and n.input[input_idx] == name and id(n) not in promoted_ids
                ]
                if not users:
                    return
                init = work_model.get_initializer(name)
                if init is None:
                    return
                existing = {i.name for i in work_model.model.graph.initializer}
                new_name, counter = name + "_mp", 0
                while new_name in existing:
                    counter += 1
                    new_name = f"{name}_mp{counter}"
                new_init = onnx.TensorProto()
                new_init.CopyFrom(init)
                new_init.name = new_name
                work_model.add_initializer(new_init)
                for node in promoted_nodes:
                    if len(node.input) > input_idx and node.input[input_idx] == name:
                        node.input[input_idx] = new_name

            _unshare_init(1)  # scale
            _unshare_init(2)  # zero point

        else:  # "propagate"

            def _propagate_domain(input_idx: int) -> None:
                name = dq.input[input_idx]
                for other in work_model.model.graph.node:
                    if id(other) in promoted_ids:
                        continue
                    if other.op_type not in QUANT_OP_TYPES and other.op_type not in DEQUANT_OP_TYPES:
                        continue
                    if len(other.input) <= input_idx or other.input[input_idx] != name:
                        continue
                    is_q_node = other.op_type in QUANT_OP_TYPES
                    other.op_type = q_op_name if is_q_node else dq_op_name
                    other.domain = domain

            _propagate_domain(1)  # scale
            _propagate_domain(2)  # zero point

        _set(dq.input[1], scale_np)
        _set(dq.input[2], zp_np)
        if q is not None:
            if q.input[1] != dq.input[1]:
                _set(q.input[1], scale_np)
            if q.input[2] != dq.input[2]:
                _set(q.input[2], zp_np)

        if float_data is not None:
            if quant_type in ONNX_FP_QTYPES_LIST:
                # FP types: cast directly to the target float dtype — no integer rounding
                new_quant = float_data.astype(onnx.helper.tensor_dtype_to_np_dtype(quant_type))
            else:
                qmin, qmax = get_qmin_qmax_for_qType(quant_type, symmetric=symmetric)
                new_quant = np.clip(np.round(float_data / scale_np + zp_np), qmin, qmax).astype(
                    onnx.helper.tensor_dtype_to_np_dtype(quant_type)
                )
            weight_init = work_model.get_initializer(dq.input[0])
            if weight_init is not None:
                work_model.remove_initializer(weight_init)
                work_model.add_initializer(onnx.numpy_helper.from_array(new_quant, dq.input[0]))

    def _refine_bias_scale(self, work_model: ONNXModel, node_name: str) -> None:
        """Recompute bias scale as act_scale * weight_scale after a precision change."""
        onnx_model = ONNXQuantizedModel(work_model.model)
        for node in work_model.model.graph.node:
            if node.name == node_name and len(node.input) == 3:
                dq, q = onnx_model._find_node_input_qdq(node, node.input[2])
                if dq is None:
                    logger.debug(f"Not found DQ for bias of node {node_name}")
                    break
                if len(dq.input) < 3:
                    logger.debug("No need to update bias scale: DQ has no zero_point (float-type quant)")
                    break
                zp_init = work_model.get_initializer(dq.input[2])
                if zp_init is None or zp_init.data_type != onnx_proto.TensorProto.INT32:
                    logger.debug("No need to update bias scale for non-Int32")
                    break

                input_dq, _ = onnx_model._find_node_input_qdq(node, node.input[0])
                if input_dq is None:
                    logger.debug(f"Not found DQ for input of node {node_name}")
                    break

                input_scale_init = work_model.get_initializer(input_dq.input[1])
                if input_scale_init is None:
                    logger.debug(f"Not found scale initializer for input DQ of node {node_name}")
                    break
                input_scale = onnx.numpy_helper.to_array(input_scale_init)

                weight_dq, _ = onnx_model._find_node_input_qdq(node, node.input[1])
                if weight_dq is None:
                    logger.debug(f"Not found DQ for weight of node {node_name}")
                    break

                weight_scale_init = work_model.get_initializer(weight_dq.input[1])
                if weight_scale_init is None:
                    logger.debug(f"Not found scale initializer for weight DQ of node {node_name}")
                    break
                weight_scale = onnx.numpy_helper.to_array(weight_scale_init)

                new_bias_scale = (input_scale * weight_scale).astype(input_scale.dtype)

                if q is None:
                    # If the q node of bias is folded, we need to update the quantized bias tensor
                    bias_scale_init = work_model.get_initializer(dq.input[1])
                    old_bias_scale = onnx.numpy_helper.to_array(bias_scale_init)

                    bias_init = work_model.get_initializer(dq.input[0])
                    bias = onnx.numpy_helper.to_array(bias_init).astype(np.float32)
                    bias = bias * old_bias_scale / new_bias_scale
                    # Suppress "invalid value encountered in cast": when the rescaled bias
                    # overflows int32 (common for INT8→INT16 promotions where the scale
                    # ratio is ~66000×), NumPy's float32→int32 cast produces a
                    # platform-defined value on overflow.  np.errstate silences the
                    # warning without altering the arithmetic or the cast result.
                    with np.errstate(invalid="ignore", over="ignore"):
                        bias = bias.astype(np.int32)

                    new_bias_init = onnx.numpy_helper.from_array(bias, dq.input[0])
                    bias_init.CopyFrom(new_bias_init)
                else:
                    # If the q node of bias is not folded, we need to update the scale of the q node
                    if q.input[1] != dq.input[1]:
                        new_bias_scale_init = onnx.numpy_helper.from_array(new_bias_scale, q.input[1])
                        work_model.get_initializer(q.input[1]).CopyFrom(new_bias_scale_init)

                # Update the scale of the bias dq node
                new_bias_scale_init = onnx.numpy_helper.from_array(new_bias_scale, dq.input[1])
                work_model.get_initializer(dq.input[1]).CopyFrom(new_bias_scale_init)

                logger.debug(f"Updated bias scale for node {node_name} to meet scale_b = scale_x * scale_w")
                return None

    def _compute_scale_zp(
        self,
        work_model: ONNXModel,
        float_tensor_name: str | None,
        dq: Any,
        quant_type: int,
        tensors_range: TensorsData,
        symmetric: bool | None = None,
        calibration_method: Any = None,
    ) -> tuple[Any, Any] | None:
        """Compute (scale_np, zp_np) for the target quant_type.

        Tries three sources in priority order:

        1. **tensors_range** — calibration range for the tensor (activations).
        2. **Float initializer** at *float_tensor_name* — weight with Q node.
        3. **Dequantized weight** from *dq* — folded-Q weight (DQ-only node).

        Returns ``None`` if none of the sources are available (e.g. a runtime
        activation tensor that was not calibrated).
        """
        if symmetric is None:
            symmetric = quant_type in (
                onnx_proto.TensorProto.INT8,
                onnx_proto.TensorProto.INT16,
                onnx_proto.TensorProto.INT32,
                onnx_proto.TensorProto.FLOAT16,
                onnx_proto.TensorProto.BFLOAT16,
            )
        if calibration_method is None:
            calibration_method = CalibrationMethod.MinMax

        # 1. Activation: calibration range
        if float_tensor_name and tensors_range is not None and float_tensor_name in tensors_range:
            td = tensors_range[float_tensor_name]
            rmin, rmax = td.range_value[0], td.range_value[1]
            # If the existing DQ scale is float16, preserve that dtype so the
            # new scale stays float16 (needed for float16 base models where
            # DequantizeLinear must output float16 to match Conv weight dtype).
            if dq is not None and len(dq.input) > 1:
                existing_scale_init = work_model.get_initializer(dq.input[1])
                if existing_scale_init is not None:
                    existing_scale = onnx.numpy_helper.to_array(existing_scale_init)
                    if existing_scale.dtype != rmin.dtype:
                        rmin = rmin.astype(existing_scale.dtype)
                        rmax = rmax.astype(existing_scale.dtype)
            return self._scale_zp_from_range(rmin, rmax, quant_type, symmetric, calibration_method)

        # 2. Weight with Q: float initializer
        if float_tensor_name:
            float_init = work_model.get_initializer(float_tensor_name)
            if float_init is not None:
                float_data = onnx.numpy_helper.to_array(float_init)
                rmin = np.asarray(float_data.min(), dtype=float_data.dtype)
                rmax = np.asarray(float_data.max(), dtype=float_data.dtype)
                return self._scale_zp_from_range(rmin, rmax, quant_type, symmetric, calibration_method)

        # 3. DQ-only weight: dequantize existing quantized initializer
        if dq is not None:
            float_data = self._dequantize_weight(work_model, dq)  # type: ignore
            if float_data is not None:
                rmin = np.asarray(float_data.min(), dtype=float_data.dtype)
                rmax = np.asarray(float_data.max(), dtype=float_data.dtype)
                return self._scale_zp_from_range(rmin, rmax, quant_type, symmetric, calibration_method)

        return None

    # ------------------------------------------------------------------
    # Core functions
    # ------------------------------------------------------------------

    def _promote_fn_slot(
        self,
        work_model: ONNXModel,
        target_node: Any,
        fn_node: Any,
        target_spec: Any,
        target_format: Any,
        quant_type: int,
        tensors_range: TensorsData,
        symmetric: bool | None = None,
        calibration_method: Any = None,
        is_output: bool = False,
    ) -> None:
        """Promote a slot whose current quantizer is a BFP/MX fn node.

        +----------+----------+---------------------------------------------------+
        | target   | slot     | action                                            |
        +==========+==========+===================================================+
        | BFP / MX | any      | swap fn_node for new BFP/MX fn node               |
        | INT      | any      | swap fn_node for Q/DQ pair (scale/zp from data)   |
        +----------+----------+---------------------------------------------------+
        """
        is_custom_target = target_format in (ExtendedQuantType.QBFP, ExtendedQuantType.QMX)

        if is_custom_target:
            fn_type, fn_attrs = self._custom_op_params(target_format, target_spec)

            def _make_fn(inputs: list[str], outputs: list[str], name: str) -> Any:
                node = onnx.helper.make_node(fn_type, inputs=inputs, outputs=outputs, name=name, domain=COP_DOMAIN)
                for k, v in fn_attrs.items():
                    node.attribute.append(onnx.helper.make_attribute(k, v))
                return node

            work_model.add_node(_make_fn([fn_node.input[0]], [fn_node.output[0]], fn_node.name + "_Mixed"))
            work_model.remove_node(fn_node)
        else:
            model_output_names = [output.name for output in work_model.model.graph.output]
            float_tensor_name = fn_node.output[0] if fn_node.output[0] in model_output_names else fn_node.input[0]
            result = self._compute_scale_zp(
                work_model, float_tensor_name, fn_node, quant_type, tensors_range, symmetric, calibration_method
            )
            if result is None:
                logger.warning(
                    f"Cannot promote the quant node '{fn_node.name}' to QDQ nodes: no calibration range or initializer found."
                )
                return None

            q_op_name, dq_op_name, domain = self._qdq_name_domain(quant_type)
            scale_np, zp_np = result
            base = fn_node.name + "_Mixed"
            scale_name, zp_name, q_out = base + "_scale", base + "_zero_point", base + "_quantized"
            work_model.add_initializer(onnx.numpy_helper.from_array(scale_np, scale_name))
            work_model.add_initializer(onnx.numpy_helper.from_array(zp_np, zp_name))
            work_model.add_node(
                onnx.helper.make_node(
                    q_op_name, [fn_node.input[0], scale_name, zp_name], [q_out], base + "_Q", domain=domain
                )
            )
            work_model.add_node(
                onnx.helper.make_node(
                    dq_op_name, [q_out, scale_name, zp_name], [fn_node.output[0]], base + "_DQ", domain=domain
                )
            )
            work_model.remove_node(fn_node)

    def _promote_qdq_with_fn(
        self,
        work_model: ONNXModel,
        target_node: Any,
        dq: Any,
        q: Any,
        target_spec: Any,
        target_format: Any,
        quant_type: int,
        is_output: bool = False,
    ) -> None:
        """Promote a slot whose current quantizer is a Q/DQ pair to BFP/MX format.

        Only called when *target_format* is QBFP or QMX (INT→INT promotion is
        handled by ``_dispatch``).

        +---------+----------+----------------------------------------------------------+
        | mode    | slot     | action                                                   |
        +=========+==========+==========================================================+
        | replace | output   | remove Q/DQ; fn takes target-node's raw output           |
        | replace | input    | keep Q/DQ; fn appended after DQ, takes DQ's float output |
        | insert  | any      | keep Q/DQ; fn appended at the upstream tensor            |
        +---------+----------+----------------------------------------------------------+
        """
        fn_type, fn_attrs = self._custom_op_params(target_format, target_spec)

        def _make_fn(inputs: list[str], outputs: list[str], name: str) -> Any:
            node = onnx.helper.make_node(fn_type, inputs=inputs, outputs=outputs, name=name, domain=COP_DOMAIN)
            for k, v in fn_attrs.items():
                node.attribute.append(onnx.helper.make_attribute(k, v))
            return node

        fn_node_name = dq.name + "_Mixed_fn"
        if is_output:
            original_output = dq.output[0]
            model_output_names = {o.name for o in work_model.model.graph.output}
            if original_output in model_output_names:
                # The DQ output is a graph output — the fn node must produce
                # that same name directly so the graph output declaration is
                # left unchanged.
                fn_output = original_output
            else:
                fn_output = fn_node_name + "_output"
                work_model.replace_input_of_all_nodes(original_output, fn_output)
            work_model.add_node(_make_fn([q.input[0]], [fn_output], fn_node_name))
            work_model.remove_node(q)
            work_model.remove_node(dq)
        else:
            if q is not None:
                upstream = q.input[0]
            else:
                # Q was constant-folded: dq.input[0] is a quantized (e.g. int8) initializer.
                # BFP/MX fn nodes require float32 input, so reconstruct the float weight.
                float_data = self._dequantize_weight(work_model, dq)
                if float_data is not None:
                    upstream = dq.input[0] + "_float_Mixed"
                    work_model.add_initializer(onnx.numpy_helper.from_array(float_data, upstream))
                else:
                    upstream = dq.input[0]
            work_model.add_node(_make_fn([upstream], [dq.output[0]], fn_node_name))
            if q is not None:
                work_model.remove_node(q)
            work_model.remove_node(dq)

    def _promote_qdq_with_qdq(
        self,
        work_model: ONNXModel,
        target_node: Any,
        dq: Any,
        q: Any,
        quant_type: int,
        tensors_range: TensorsData,
        symmetric: bool | None = None,
        calibration_method: Any = None,
    ) -> None:
        """Promote a Q/DQ pair to a different quant type."""
        model_output_names = [output.name for output in work_model.model.graph.output]
        if dq and dq.output[0] in model_output_names:
            float_tensor_name = dq.output[0]
        else:
            float_tensor_name = q.input[0] if q is not None else None
        result = self._compute_scale_zp(
            work_model, float_tensor_name, dq, quant_type, tensors_range, symmetric, calibration_method
        )
        if result is None:
            logger.warning(
                f"Cannot promote the quant node '{dq.name}' to another: no calibration range or initializer found."
            )
            return None

        self._apply_scale_zp(work_model, dq, q, result[0], result[1], quant_type, symmetric or False)

    def _handle_target_qdqs(
        self,
        work_model: ONNXModel,
        node_struct: dict[str, Any],
        tensors_range: TensorsData,
    ) -> set[str]:
        """Replace quantization ops in *node_struct* with the target precision.
        Returns the set of promoted tensor names.
        """

        target_node = node_struct["node"]
        input_qdqs = node_struct.get("input_qdqs", [])
        output_qdqs = node_struct.get("output_qdqs", [])

        initializer_names = {init.name for init in work_model.model.graph.initializer}
        promoted_tensors = set()

        def _is_initializer(qdq_tuple: tuple[Any, Any]) -> bool:
            """Return True when the data tensor for this slot is a model initializer."""
            if len(qdq_tuple) == 1 and qdq_tuple[0] is not None:
                fn_node = qdq_tuple[0]
                return bool(fn_node.input) and fn_node.input[0] in initializer_names
            if len(qdq_tuple) == 2:
                dq, q = qdq_tuple
                ref = q if q is not None else dq
                return ref is not None and bool(ref.input) and ref.input[0] in initializer_names
            return False

        def _get_symmetric(spec: Any, role: str = "activation") -> bool | None:
            base = getattr(spec, "symmetric", None)
            override: bool | None = base
            if role == "weight":
                override = self._extra_options.get("WeightSymmetric", None)
            elif role == "activation":
                override = self._extra_options.get("ActivationSymmetric", None)
            else:
                override = None  # bias: always defer to spec
            return override if override is not None else base

        def _get_calib_method(spec: Any) -> Any:
            cm = getattr(spec, "calibration_method", None)
            st = getattr(spec, "scale_type", None)
            if cm is None or st is None:
                return None
            return _map_activation_calibration_method(cm, st)

        def _handle_slot(
            qdq_tuple: tuple[Any, Any],
            target_spec: Any,
            target_format: Any,
            quant_type: int,
            symmetric: bool | None = None,
            calibration_method: Any = None,
            is_output: bool = False,
        ) -> None:
            if target_spec is None or target_format is None:
                return None

            if len(qdq_tuple) == 1 and qdq_tuple[0] is not None:
                fn_node = qdq_tuple[0]
                self._promote_fn_slot(
                    work_model,
                    target_node,
                    fn_node,
                    target_spec,
                    target_format,
                    quant_type,
                    tensors_range,
                    symmetric,
                    calibration_method,
                    is_output,
                )
                promoted_tensors.add(fn_node.input[0])
            elif len(qdq_tuple) == 2 and qdq_tuple[0] is not None:
                dq, q = qdq_tuple
                if target_format in (ExtendedQuantType.QBFP, ExtendedQuantType.QMX):
                    self._promote_qdq_with_fn(
                        work_model, target_node, dq, q, target_spec, target_format, quant_type, is_output
                    )
                else:
                    self._promote_qdq_with_qdq(
                        work_model, target_node, dq, q, quant_type, tensors_range, symmetric, calibration_method
                    )
                if q is not None:
                    promoted_tensors.add(q.input[0])

        if len(input_qdqs) >= 1:
            _handle_slot(
                input_qdqs[0],
                self.input_spec,
                self.input_format,
                self.input_tensor_type,
                symmetric=_get_symmetric(self.input_spec, "activation"),
                calibration_method=_get_calib_method(self.input_spec),
            )

        if len(input_qdqs) >= 2:
            if _is_initializer(input_qdqs[1]):
                _handle_slot(
                    input_qdqs[1],
                    self.weight_spec,
                    self.weight_format,
                    self.weight_tensor_type,
                    symmetric=_get_symmetric(self.weight_spec, "weight"),
                    calibration_method=_get_calib_method(self.weight_spec),
                )
            else:
                _handle_slot(
                    input_qdqs[1],
                    self.input_spec,
                    self.input_format,
                    self.input_tensor_type,
                    symmetric=_get_symmetric(self.input_spec, "activation"),
                    calibration_method=_get_calib_method(self.input_spec),
                )

        if len(input_qdqs) >= 3:
            if _is_initializer(input_qdqs[2]):
                _handle_slot(
                    input_qdqs[2],
                    self.bias_spec,
                    self.bias_format,
                    self.bias_tensor_type,
                    symmetric=_get_symmetric(self.bias_spec, "bias"),
                    calibration_method=_get_calib_method(self.bias_spec),
                )
                self._refine_bias_scale(work_model, target_node.name)
            else:
                _handle_slot(
                    input_qdqs[2],
                    self.input_spec,
                    self.input_format,
                    self.input_tensor_type,
                    symmetric=_get_symmetric(self.input_spec, "activation"),
                    calibration_method=_get_calib_method(self.input_spec),
                )

        for qdq_tuple in input_qdqs[3:]:
            _handle_slot(
                qdq_tuple,
                self.input_spec,
                self.input_format,
                self.input_tensor_type,
                symmetric=_get_symmetric(self.input_spec, "activation"),
                calibration_method=_get_calib_method(self.input_spec),
            )

        for qdq_tuple in output_qdqs:
            _handle_slot(
                qdq_tuple,
                self.output_spec,
                self.output_format,
                self.output_tensor_type,
                symmetric=_get_symmetric(self.output_spec, "activation"),
                calibration_method=_get_calib_method(self.output_spec),
                is_output=True,
            )

        return promoted_tensors


__all__ = ["MixingStrategy"]
