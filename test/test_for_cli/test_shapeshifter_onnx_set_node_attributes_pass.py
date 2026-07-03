#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for the ``onnx_set_node_attributes`` Shapeshifter pass.

Covers CLI YAML workflows (LeakyRelu ``alpha``, BFP Q/DQ ``axis``) and direct unit tests for
``_update_existing_attribute_on_node`` strict typing (primitive types, lists, tensors).
"""

import unittest
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import yaml
from onnx import AttributeProto, TensorProto, helper, numpy_helper

from quark.common.utils.testing_utils import use_temporary_directory
from quark.experimental.cli.main import main as cli
from quark.onnx.quantization.quant_utils import COP_BFP_OP_NAME, COP_DOMAIN
from quark.shapeshifter.passes.onnx_set_node_attributes import ONNXSetNodeAttributesPass


def build_leaky_relu_model(onnx_model_path: str) -> None:
    """Write a tiny LeakyRelu graph used to verify ``alpha`` updates via CLI."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2])
    leaky = helper.make_node("LeakyRelu", ["x"], ["y"], name="leaky_target", alpha=0.01)
    graph = helper.make_graph([leaky], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, onnx_model_path)


def build_bfp_qdq_like_model(onnx_model_path: str) -> None:
    """Minimal graph with Quark BFPQuantizeDequantize (same op type as post-quant BFP Q/DQ)."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])
    bfp = helper.make_node(
        COP_BFP_OP_NAME,
        ["x"],
        ["y"],
        name="bfp_qdq_after_quant",
        domain=COP_DOMAIN,
        axis=1,
        bfp_method="to_bfp_prime",
        bit_width=13,
        block_size=16,
        rounding_mode=2,
        sub_block_size=2,
        sub_block_shift_bits=1,
        convert_to_bfloat_before_bfp=0,
    )
    graph = helper.make_graph([bfp], "g", [x], [y])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(COP_DOMAIN, 1)],
    )
    # Custom op: strict checker may reject unknown schema; structure is enough for this pass.
    onnx.save(model, onnx_model_path)


def get_node_attr(model_path: str, node_name: str, attr_name: str) -> Any:
    model = onnx.load(model_path)
    for n in model.graph.node:
        if n.name != node_name:
            continue
        for a in n.attribute:
            if a.name == attr_name:
                return helper.get_attribute_value(a)
    return None


def get_node_attr_kind(model_path: str, node_name: str, attr_name: str) -> int | None:
    model = onnx.load(model_path)
    for n in model.graph.node:
        if n.name != node_name:
            continue
        for a in n.attribute:
            if a.name == attr_name:
                return int(a.type)
    return None


def prepare_yaml(output_dir: str, onnx_in: str, onnx_out: str) -> str:
    yaml_path = Path(output_dir, "set_node_attributes.yaml").as_posix()
    config = {
        "input_model_path": onnx_in,
        "passes": {
            "onnx_set_node_attributes": {
                "node_attribute_updates": [
                    {
                        "node_name": "leaky_target",
                        "attributes": {"alpha": 0.05},
                    }
                ],
            }
        },
        "output_model_path": onnx_out,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def prepare_yaml_bfp_axis_string_value(output_dir: str, onnx_in: str, onnx_out: str) -> str:
    """``axis`` is a YAML string; must be skipped (no string-to-int coercion)."""
    yaml_path = Path(output_dir, "set_bfp_axis_str.yaml").as_posix()
    config = {
        "input_model_path": onnx_in,
        "passes": {
            "onnx_set_node_attributes": {
                "node_attribute_updates": [
                    {
                        "node_name": "bfp_qdq_after_quant",
                        "attributes": {"axis": "0"},
                    }
                ],
            }
        },
        "output_model_path": onnx_out,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def prepare_yaml_bfp_axis(output_dir: str, onnx_in: str, onnx_out: str) -> str:
    yaml_path = Path(output_dir, "set_bfp_axis.yaml").as_posix()
    config = {
        "input_model_path": onnx_in,
        "passes": {
            "onnx_set_node_attributes": {
                "node_attribute_updates": [
                    {
                        "node_name": "bfp_qdq_after_quant",
                        "attributes": {"axis": 0},
                    }
                ],
            }
        },
        "output_model_path": onnx_out,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


class TestOnnxSetNodeAttributesPass(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_set_node_attributes_pass(self, tmpdir: str) -> None:
        onnx_in = Path(tmpdir, "leaky.onnx").as_posix()
        onnx_out = Path(tmpdir, "leaky_updated.onnx").as_posix()
        build_leaky_relu_model(onnx_in)
        self.assertAlmostEqual(get_node_attr(onnx_in, "leaky_target", "alpha"), 0.01, places=5)

        yaml_path = prepare_yaml(tmpdir, onnx_in, onnx_out)
        cli(["shapeshifter", yaml_path])

        self.assertAlmostEqual(get_node_attr(onnx_out, "leaky_target", "alpha"), 0.05, places=5)

    @use_temporary_directory
    def test_onnx_set_node_attributes_bfp_qdq_axis(self, tmpdir: str) -> None:
        """Quantized-model scenario: update ``axis`` on ``BFPQuantizeDequantize`` (BFP Q/DQ) node."""
        onnx_in = Path(tmpdir, "bfp_qdq.onnx").as_posix()
        onnx_out = Path(tmpdir, "bfp_qdq_updated.onnx").as_posix()
        build_bfp_qdq_like_model(onnx_in)
        self.assertEqual(get_node_attr(onnx_in, "bfp_qdq_after_quant", "axis"), 1)

        yaml_path = prepare_yaml_bfp_axis(tmpdir, onnx_in, onnx_out)
        cli(["shapeshifter", yaml_path])

        self.assertEqual(get_node_attr(onnx_out, "bfp_qdq_after_quant", "axis"), 0)

    @use_temporary_directory
    def test_onnx_set_node_attributes_skips_int_attr_when_config_value_is_str(self, tmpdir: str) -> None:
        """String config for an INT attribute is skipped; model is unchanged."""
        onnx_in = Path(tmpdir, "bfp_qdq_str.onnx").as_posix()
        onnx_out = Path(tmpdir, "bfp_qdq_str_updated.onnx").as_posix()
        build_bfp_qdq_like_model(onnx_in)
        self.assertEqual(get_node_attr_kind(onnx_in, "bfp_qdq_after_quant", "axis"), AttributeProto.INT)
        self.assertEqual(get_node_attr(onnx_in, "bfp_qdq_after_quant", "axis"), 1)

        yaml_path = prepare_yaml_bfp_axis_string_value(tmpdir, onnx_in, onnx_out)
        cli(["shapeshifter", yaml_path])

        self.assertEqual(get_node_attr_kind(onnx_out, "bfp_qdq_after_quant", "axis"), AttributeProto.INT)
        self.assertEqual(get_node_attr(onnx_out, "bfp_qdq_after_quant", "axis"), 1)


class TestOnnxSetNodeAttributesStrictTypes(unittest.TestCase):
    """Exercise ``ONNXSetNodeAttributesPass._update_existing_attribute_on_node`` on a minimal node."""

    def _try_set(self, attr_name: str, initial: Any, new_value: Any) -> tuple[str, Any]:
        node = helper.make_node("X", [], [], name="n", **{attr_name: initial})
        outcome = ONNXSetNodeAttributesPass._update_existing_attribute_on_node(node, attr_name, new_value)
        val = helper.get_attribute_value(next(a for a in node.attribute if a.name == attr_name))
        return outcome, val

    def test_update_int(self) -> None:
        self.assertEqual(self._try_set("k", 0, 7), ("updated", 7))

    def test_skip_int_when_value_is_str(self) -> None:
        self.assertEqual(self._try_set("k", 0, "0"), ("type_mismatch", 0))

    def test_skip_int_when_value_is_bool(self) -> None:
        self.assertEqual(self._try_set("k", 0, True), ("type_mismatch", 0))

    def test_update_float(self) -> None:
        out = self._try_set("k", 0.0, 0.25)
        self.assertEqual(out[0], "updated")
        self.assertAlmostEqual(out[1], 0.25)

    def test_skip_float_when_value_is_int(self) -> None:
        self.assertEqual(self._try_set("k", 0.0, 1), ("type_mismatch", 0.0))

    def test_skip_float_when_value_is_str(self) -> None:
        self.assertEqual(self._try_set("k", 0.0, "1.5"), ("type_mismatch", 0.0))

    def test_update_string(self) -> None:
        self.assertEqual(self._try_set("k", "x", "hello"), ("updated", b"hello"))

    def test_skip_string_when_value_is_int(self) -> None:
        self.assertEqual(self._try_set("k", "x", 42), ("type_mismatch", b"x"))

    def test_skip_string_when_value_is_binary(self) -> None:
        self.assertEqual(self._try_set("k", "x", b"\x00\xff"), ("type_mismatch", b"x"))

    def test_update_ints(self) -> None:
        self.assertEqual(self._try_set("k", [1, 2], [3, 4]), ("updated", [3, 4]))
        self.assertEqual(self._try_set("k", [1, 2], [9]), ("updated", [9]))

    def test_skip_ints_when_value_is_scalar(self) -> None:
        self.assertEqual(self._try_set("k", [1, 3], 9), ("type_mismatch", [1, 3]))

    def test_skip_ints_when_element_is_str(self) -> None:
        self.assertEqual(self._try_set("k", [1, 2], ["10", "3"]), ("type_mismatch", [1, 2]))

    def test_update_floats(self) -> None:
        self.assertEqual(self._try_set("k", [1.0, 2.0], [0.5, 1.5]), ("updated", [0.5, 1.5]))
        self.assertEqual(self._try_set("k", [1.0, 2.0], [2.0, 3.0]), ("updated", [2.0, 3.0]))

    def test_skip_floats_when_element_is_int(self) -> None:
        self.assertEqual(self._try_set("k", [1.0, 2.0], [2, 3]), ("type_mismatch", [1.0, 2.0]))

    def test_skip_floats_when_element_is_str(self) -> None:
        self.assertEqual(self._try_set("k", [1.0, 2.0], ["2", "3"]), ("type_mismatch", [1.0, 2.0]))

    def test_update_strings(self) -> None:
        self.assertEqual(self._try_set("k", ["a", "b"], ["x", "y"]), ("updated", [b"x", b"y"]))
        self.assertEqual(self._try_set("k", ["a", "b"], ["solo"]), ("updated", [b"solo"]))

    def test_skip_strings_when_value_is_scalar(self) -> None:
        self.assertEqual(self._try_set("k", ["a", "b"], "solo"), ("type_mismatch", [b"a", b"b"]))

    def test_skip_strings_when_element_is_binary(self) -> None:
        self.assertEqual(self._try_set("k", ["a"], [b"no"]), ("type_mismatch", [b"a"]))

    def test_tensor_attribute_not_updated(self) -> None:
        t0 = numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64))
        t1 = numpy_helper.from_array(np.array([9], dtype=np.int64))
        got = self._try_set("t", t0, t1)
        self.assertEqual(got[0], "unsupported")
        np.testing.assert_array_equal(numpy_helper.to_array(got[1]), numpy_helper.to_array(t0))

    def test_update_existing_attribute_on_node_primitive_types(self) -> None:
        node = helper.make_node(
            "Custom",
            [],
            [],
            name="n",
            i_attr=1,
            f_attr=1.0,
            s_attr="old",
            ints_attr=[1, 2],
            floats_attr=[1.0],
            strs_attr=["a"],
        )
        self.assertEqual(ONNXSetNodeAttributesPass._update_existing_attribute_on_node(node, "i_attr", 2), "updated")
        self.assertEqual(ONNXSetNodeAttributesPass._update_existing_attribute_on_node(node, "f_attr", 0.5), "updated")
        self.assertEqual(ONNXSetNodeAttributesPass._update_existing_attribute_on_node(node, "s_attr", "new"), "updated")
        self.assertEqual(
            ONNXSetNodeAttributesPass._update_existing_attribute_on_node(node, "ints_attr", [3, 4]), "updated"
        )
        self.assertEqual(
            ONNXSetNodeAttributesPass._update_existing_attribute_on_node(node, "floats_attr", [2.0]), "updated"
        )
        self.assertEqual(
            ONNXSetNodeAttributesPass._update_existing_attribute_on_node(node, "strs_attr", ["b", "c"]), "updated"
        )

        by_name = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        self.assertEqual(by_name["i_attr"], 2)
        self.assertAlmostEqual(by_name["f_attr"], 0.5)
        self.assertEqual(by_name["s_attr"], b"new")
        self.assertEqual(by_name["ints_attr"], [3, 4])
        self.assertEqual(by_name["floats_attr"], [2.0])
        self.assertEqual(by_name["strs_attr"], [b"b", b"c"])


if __name__ == "__main__":
    unittest.main()
