#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any, Literal

from onnx import AttributeProto, ModelProto, NodeProto, helper

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


def _onnx_attr_type_name(attr_type: int) -> str:
    """Return the ONNX enum name for an ``AttributeProto`` type field.

    Args:
        attr_type (int): Raw ``AttributeProto.type`` value.

    Returns:
        str: Enum name from ``AttributeProto.AttributeType``, or ``str(attr_type)`` on failure.
    """
    try:
        return AttributeProto.AttributeType.Name(attr_type)
    except (ValueError, AttributeError):
        return str(attr_type)


@register_pass
class ONNXSetNodeAttributesPass(ONNXPass):
    """A pass that updates existing ``NodeProto`` attributes selected by node name.

    This pass rewrites attribute values on named graph nodes (for example ``alpha`` on
    LeakyRelu or ``axis`` on custom Q/DQ ops) without re-exporting from the original
    framework. It is typically used after quantization or when tuning a graph for deployment.

    Only attributes that already exist on a matched node are considered; missing names are
    skipped with a warning and new attributes are never inserted. Only ONNX primitive
    attribute kinds (INT, FLOAT, STRING, INTS, FLOATS, STRINGS) may be updated; other kinds
    (for example TENSOR or GRAPH) are left unchanged with a warning. For supported kinds,
    the configured value must match the existing ONNX type; mismatches are skipped with a
    warning.
    """

    # ONNX AttributeProto.type -> (primitive kind key, whether the attribute is repeated).
    _PRIMITIVE_ATTR_SPEC: dict[int, tuple[str, bool]] = {
        AttributeProto.INT: ("i", False),
        AttributeProto.FLOAT: ("f", False),
        AttributeProto.STRING: ("s", False),
        AttributeProto.INTS: ("i", True),
        AttributeProto.FLOATS: ("f", True),
        AttributeProto.STRINGS: ("s", True),
    }

    @staticmethod
    def _user_value_matches_primitive_kind(kind: str, as_list: bool, value: Any) -> bool:
        """Check whether ``value`` is acceptable for the given primitive kind.

        Args:
            kind (str): One of ``i``, ``f``, or ``s`` (see ``_PRIMITIVE_ATTR_SPEC``).
            as_list (bool): True for INTS, FLOATS, STRINGS; False for scalar INT, FLOAT, STRING.
            value (Any): Candidate value from configuration.

        Returns:
            bool: True if ``value`` matches; False otherwise.
        """
        if not as_list:
            if kind == "i":
                return type(value) is int
            if kind == "f":
                return type(value) is float
            return isinstance(value, str)

        if not isinstance(value, list | tuple):
            return False
        if kind == "i":
            return all(type(x) is int for x in value)
        if kind == "f":
            return all(type(x) is float for x in value)
        return all(isinstance(x, str) for x in value)

    @staticmethod
    def _normalized_primitive_value(kind: str, as_list: bool, value: Any) -> Any:
        """Convert a validated value into the form expected by ``helper.make_attribute``.

        Args:
            kind (str): Primitive kind key: ``i``, ``f``, or ``s``.
            as_list (bool): Whether the ONNX attribute is a repeated field.
            value (Any): Value that already passed ``_user_value_matches_primitive_kind``.

        Returns:
            Any: Scalar as-is, or a ``list`` copy for repeated attributes.
        """
        if not as_list:
            return value
        return list(value)

    @staticmethod
    def _update_existing_attribute_on_node(
        node: NodeProto, attribute_name: str, value: Any
    ) -> Literal["updated", "missing", "type_mismatch", "unsupported"]:
        """Update one attribute on ``node`` if it exists and the value type is valid.

        Args:
            node (NodeProto): Graph node whose ``attribute`` list may be modified in place.
            attribute_name (str): Name of the attribute to replace.
            value (Any): New value from configuration.

        Returns:
            Literal["updated", "missing", "type_mismatch", "unsupported"]:
                ``updated`` if the attribute was replaced; ``missing`` if no attribute with
                that name exists; ``type_mismatch`` if the attribute exists but ``value`` does
                not match the expected Python type for that ONNX kind; ``unsupported`` if the
                existing attribute is not INT, FLOAT, STRING, INTS, FLOATS, or STRINGS.
        """
        old = next((a for a in node.attribute if a.name == attribute_name), None)
        if old is None:
            return "missing"

        spec = ONNXSetNodeAttributesPass._PRIMITIVE_ATTR_SPEC.get(old.type)
        if spec is None:
            logger.warning(
                f"Attribute {attribute_name!r} on node {node.name!r} has ONNX kind "
                f"{_onnx_attr_type_name(old.type)}; only INT, FLOAT, STRING, INTS, FLOATS, and "
                "STRINGS can be updated — skipping."
            )
            return "unsupported"

        kind, as_list = spec
        if not ONNXSetNodeAttributesPass._user_value_matches_primitive_kind(kind, as_list, value):
            logger.warning(
                f"The value type for attribute {attribute_name!r} on node {node.name!r} does not match "
                f"the existing ONNX attribute type ({_onnx_attr_type_name(old.type)}); "
                "skipping update."
            )
            return "type_mismatch"
        coerced = ONNXSetNodeAttributesPass._normalized_primitive_value(kind, as_list, value)

        for i in range(len(node.attribute) - 1, -1, -1):
            if node.attribute[i].name == attribute_name:
                del node.attribute[i]
        node.attribute.append(helper.make_attribute(attribute_name, coerced))
        return "updated"

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return the default configuration for the pass.

        Returns:
            dict[str, PassConfigParam]: Configuration parameters, including
            ``node_attribute_updates`` for per-node attribute rewrites.
        """
        config = {
            "node_attribute_updates": PassConfigParam(
                type_=list,
                default_value=[],
                required=False,
                description="List of dicts with keys 'node_name' and 'attributes'; only attributes "
                "already on the node are updated. Example: "
                "[{'node_name': 'Gemm_0', 'attributes': {'transA': 1}}]. "
                "Each new value's Python type must match the existing ONNX attribute kind "
                "(e.g. int for INT, float for scalar FLOAT, str for STRING; list/tuple of "
                "int/float/str for INTS/FLOATS/STRINGS with no scalar shorthand). "
                "TENSOR/GRAPH and other non-primitive attributes are not modified. "
                "Mismatches are skipped with a warning.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_apply_node_attribute_updates(self, model: ModelProto, updates: list[dict[str, Any]]) -> ModelProto:
        """Apply ``node_attribute_updates`` entries to matching nodes in ``model``.

        Args:
            model (ModelProto): The ONNX model to modify in place.
            updates (list[dict[str, Any]]): Parsed ``node_attribute_updates`` from configuration.

        Returns:
            ModelProto: The same ``model`` instance after applying updates.
        """
        for idx, spec in enumerate(updates):
            if not isinstance(spec, dict):
                logger.warning(f"node_attribute_updates[{idx}] is not a dict, skipped.")
                continue
            node_name = spec.get("node_name")
            if not node_name:
                logger.warning(f"node_attribute_updates[{idx}] missing 'node_name', skipped.")
                continue
            attributes = spec.get("attributes")
            if not attributes:
                logger.warning(f"No 'attributes' for node_name={node_name!r}, skipped.")
                continue
            if not isinstance(attributes, dict):
                logger.warning(f"'attributes' for node_name={node_name!r} must be a mapping, skipped.")
                continue

            matched = False
            for node in model.graph.node:
                if node.name != node_name:
                    continue
                matched = True
                updated: list[str] = []
                skipped_missing: list[str] = []
                for attr_name, attr_val in attributes.items():
                    try:
                        outcome = self._update_existing_attribute_on_node(node, attr_name, attr_val)
                        if outcome == "updated":
                            updated.append(attr_name)
                        elif outcome == "missing":
                            skipped_missing.append(attr_name)
                    except Exception as e:
                        logger.warning(
                            f"Failed to update attribute {attr_name!r} on node {node_name!r}: {e}. "
                            "Check value type (int/float/str/list) for ONNX."
                        )
                if skipped_missing:
                    logger.warning(
                        f"Node {node_name!r} has no existing attribute(s) named {skipped_missing!r}; "
                        "only existing attributes are modified — skipped."
                    )
                if updated:
                    logger.info(f"Updated attribute value(s) on node {node_name!r}: {updated}")
            if not matched:
                logger.warning(f"No graph node with name {node_name!r} found; nothing updated.")
        return model

    def _run_for_config(self, model: ModelProto, config: dict[str, Any]) -> ModelProto:
        """Run the pass according to ``config``.

        Args:
            model (ModelProto): The ONNX model to process.
            config (dict[str, Any]): Runtime configuration; may contain ``node_attribute_updates``.

        Returns:
            ModelProto: The model after applying attribute updates, or the unchanged model when
            ``node_attribute_updates`` is missing or empty.
        """
        updates = config.get("node_attribute_updates") or []
        if not updates:
            logger.warning("onnx_set_node_attributes: 'node_attribute_updates' is empty or missing; pass is a no-op.")
            return model
        return self._onnx_apply_node_attribute_updates(model, updates)
