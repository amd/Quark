# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Tests for _apply_weight_converters in the file-to-file quantization pipeline.

Uses lightweight mock objects that implement the same duck-typed interface as
WeightConverter / Chunk / Concatenate, so the tests are self-contained and do
not require ``transformers`` or ``weight_convert`` to be installed.
"""

from concurrent.futures import Future
from dataclasses import dataclass, field

import pytest
import torch

from quark.torch.quantization.file2file_quantization import _apply_weight_converters
from quark.torch.quantization.weight_convert import (
    Chunk,
    Concatenate,
    LoadStateDictInfo,
    SkipParameters,
    SplitFusedExperts,
    WeightConverter,
    WeightTransform,
    log_conversion_errors,
    process_target_pattern,
)


# ---------------------------------------------------------------------------
# Minimal mock operations (same interface as ConversionOps subclasses)
# ---------------------------------------------------------------------------
class MockChunk:
    """Mock of ``Chunk(dim=0)`` — splits a tensor into equal parts along dim."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        chunks = torch.chunk(tensor, len(target_patterns), dim=self.dim)
        return dict(zip(target_patterns, chunks, strict=False))


class MockConcatenate:
    """Mock of ``Concatenate(dim=0)`` — concatenates tensors along dim."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        all_tensors = []
        for source_pattern in source_patterns:
            tensor_or_list = input_dict[source_pattern]
            if isinstance(tensor_or_list, list):
                all_tensors.extend(tensor_or_list)
            else:
                all_tensors.append(tensor_or_list)
        return {target_patterns[0]: torch.cat(all_tensors, dim=self.dim)}


@dataclass
class MockWeightConverter:
    """
    Mock of ``WeightConverter`` with the three attributes that
    ``_apply_weight_converters`` relies on: ``source_patterns``,
    ``target_patterns``, and ``operations``.
    """

    source_patterns: list[str] = field(default_factory=list)
    target_patterns: list[str] = field(default_factory=list)
    operations: list = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.source_patterns, str):
            self.source_patterns = [self.source_patterns]
        if isinstance(self.target_patterns, str):
            self.target_patterns = [self.target_patterns]


class IdentityOperation:
    """Return the first source tensor under the first target pattern."""

    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):  # type: ignore[no-untyped-def]
        tensors = input_dict[source_patterns[0]]
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_patterns[0]: tensor}

    @property
    def reverse_op(self):
        """Return itself for reverse-transform tests."""
        return self


class AddOneOperation:
    """Add one to every tensor in the intermediate mapping."""

    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):  # type: ignore[no-untyped-def]
        return {key: value + 1 for key, value in input_dict.items()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestApplyWeightConverters:
    def test_chunk_gate_up_proj_into_two(self):
        """Split a fused gate_up_proj weight into gate_proj and up_proj."""
        gate_up = torch.randn(256, 128)
        tensors = {"model.layers.0.mlp.experts.5.gate_up_proj.weight": gate_up}

        converter = MockWeightConverter(
            source_patterns="gate_up_proj.weight",
            target_patterns=["gate_proj.weight", "up_proj.weight"],
            operations=[MockChunk(dim=0)],
        )

        result = _apply_weight_converters(tensors, [converter])

        assert "model.layers.0.mlp.experts.5.gate_proj.weight" in result
        assert "model.layers.0.mlp.experts.5.up_proj.weight" in result
        assert "model.layers.0.mlp.experts.5.gate_up_proj.weight" not in result
        assert result["model.layers.0.mlp.experts.5.gate_proj.weight"].shape == (128, 128)
        assert result["model.layers.0.mlp.experts.5.up_proj.weight"].shape == (128, 128)
        torch.testing.assert_close(
            torch.cat(
                [
                    result["model.layers.0.mlp.experts.5.gate_proj.weight"],
                    result["model.layers.0.mlp.experts.5.up_proj.weight"],
                ],
                dim=0,
            ),
            gate_up,
        )

    def test_chunk_qkv_proj_into_three(self):
        """Split a fused qkv_proj weight into q_proj, k_proj, v_proj."""
        qkv = torch.randn(768, 256)
        tensors = {"model.layers.3.self_attn.qkv_proj.weight": qkv}

        converter = MockWeightConverter(
            source_patterns="qkv_proj.weight",
            target_patterns=["q_proj.weight", "k_proj.weight", "v_proj.weight"],
            operations=[MockChunk(dim=0)],
        )

        result = _apply_weight_converters(tensors, [converter])

        assert len(result) == 3
        for key_suffix in ["q_proj.weight", "k_proj.weight", "v_proj.weight"]:
            full_key = f"model.layers.3.self_attn.{key_suffix}"
            assert full_key in result
            assert result[full_key].shape == (256, 256)

    def test_unmatched_tensors_pass_through(self):
        """Tensors that don't match any converter are kept as-is."""
        bias = torch.randn(64)
        norm_weight = torch.randn(128)
        tensors = {
            "model.layers.0.self_attn.q_proj.bias": bias,
            "model.layers.0.input_layernorm.weight": norm_weight,
        }

        converter = MockWeightConverter(
            source_patterns="gate_up_proj.weight",
            target_patterns=["gate_proj.weight", "up_proj.weight"],
            operations=[MockChunk(dim=0)],
        )

        result = _apply_weight_converters(tensors, [converter])

        assert result["model.layers.0.self_attn.q_proj.bias"] is bias
        assert result["model.layers.0.input_layernorm.weight"] is norm_weight

    def test_mixed_matched_and_unmatched(self):
        """Matched tensors are converted while unmatched ones pass through."""
        gate_up = torch.randn(256, 128)
        bias = torch.randn(64)
        tensors = {
            "model.layers.0.mlp.gate_up_proj.weight": gate_up,
            "model.layers.0.mlp.down_proj.bias": bias,
        }

        converter = MockWeightConverter(
            source_patterns="gate_up_proj.weight",
            target_patterns=["gate_proj.weight", "up_proj.weight"],
            operations=[MockChunk(dim=0)],
        )

        result = _apply_weight_converters(tensors, [converter])

        assert len(result) == 3
        assert "model.layers.0.mlp.gate_proj.weight" in result
        assert "model.layers.0.mlp.up_proj.weight" in result
        assert result["model.layers.0.mlp.down_proj.bias"] is bias

    def test_multiple_converters(self):
        """Multiple converters each match different tensors."""
        gate_up = torch.randn(256, 128)
        qkv = torch.randn(768, 256)
        tensors = {
            "model.layers.0.mlp.gate_up_proj.weight": gate_up,
            "model.layers.0.self_attn.qkv_proj.weight": qkv,
        }

        converters = [
            MockWeightConverter(
                source_patterns="gate_up_proj.weight",
                target_patterns=["gate_proj.weight", "up_proj.weight"],
                operations=[MockChunk(dim=0)],
            ),
            MockWeightConverter(
                source_patterns="qkv_proj.weight",
                target_patterns=["q_proj.weight", "k_proj.weight", "v_proj.weight"],
                operations=[MockChunk(dim=0)],
            ),
        ]

        result = _apply_weight_converters(tensors, converters)

        assert len(result) == 5
        assert "model.layers.0.mlp.gate_proj.weight" in result
        assert "model.layers.0.mlp.up_proj.weight" in result
        assert "model.layers.0.self_attn.q_proj.weight" in result
        assert "model.layers.0.self_attn.k_proj.weight" in result
        assert "model.layers.0.self_attn.v_proj.weight" in result

    def test_multiple_shards_with_same_converter(self):
        """Same converter correctly handles tensors from different layers."""
        tensors = {
            "model.layers.0.mlp.gate_up_proj.weight": torch.randn(256, 128),
            "model.layers.1.mlp.gate_up_proj.weight": torch.randn(256, 128),
            "model.layers.2.mlp.gate_up_proj.weight": torch.randn(256, 128),
        }

        converter = MockWeightConverter(
            source_patterns="gate_up_proj.weight",
            target_patterns=["gate_proj.weight", "up_proj.weight"],
            operations=[MockChunk(dim=0)],
        )

        result = _apply_weight_converters(tensors, [converter])

        assert len(result) == 6
        for layer_index in range(3):
            assert f"model.layers.{layer_index}.mlp.gate_proj.weight" in result
            assert f"model.layers.{layer_index}.mlp.up_proj.weight" in result

    def test_empty_converters_is_identity(self):
        """Empty converter list leaves tensors unchanged."""
        original = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(128, 128),
        }

        result = _apply_weight_converters(original, [])

        assert result.keys() == original.keys()
        for key in original:
            assert result[key] is original[key]

    def test_multi_source_converter_rejected(self):
        """Reject converters with more than one source pattern in file-to-file apply."""
        converter = MockWeightConverter(
            source_patterns=["gate_proj.weight", "up_proj.weight"],
            target_patterns=["gate_up_proj.weight"],
            operations=[MockConcatenate(dim=0)],
        )
        tensors = {"layer.gate_proj.weight": torch.randn(4, 8)}

        with pytest.raises(ValueError, match="one source pattern per converter only"):
            _apply_weight_converters(tensors, [converter])

    def test_invalid_source_patterns_type_rejected(self):
        """Reject converter objects whose source patterns are not sized sequences."""
        converter = MockWeightConverter(
            source_patterns="gate_proj.weight",
            target_patterns=["gate_up_proj.weight"],
            operations=[MockConcatenate(dim=0)],
        )
        converter.source_patterns = object()  # type: ignore[assignment]

        with pytest.raises(ValueError, match="source_patterns must be a str or sequence"):
            _apply_weight_converters({"layer.gate_proj.weight": torch.randn(4, 8)}, [converter])

    def test_chunk_preserves_data(self):
        """Verify that chunked data exactly matches the original fused tensor."""
        gate_up = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        tensors = {"layer.gate_up_proj.weight": gate_up}

        converter = MockWeightConverter(
            source_patterns="gate_up_proj.weight",
            target_patterns=["gate_proj.weight", "up_proj.weight"],
            operations=[MockChunk(dim=0)],
        )

        result = _apply_weight_converters(tensors, [converter])

        torch.testing.assert_close(result["layer.gate_proj.weight"], gate_up[:4])
        torch.testing.assert_close(result["layer.up_proj.weight"], gate_up[4:])

    def test_chunk_along_dim1(self):
        """Chunk operation along dim=1 works correctly."""
        fused = torch.randn(64, 256)
        tensors = {"block.fused_proj.weight": fused}

        converter = MockWeightConverter(
            source_patterns="fused_proj.weight",
            target_patterns=["proj_a.weight", "proj_b.weight"],
            operations=[MockChunk(dim=1)],
        )

        result = _apply_weight_converters(tensors, [converter])

        assert result["block.proj_a.weight"].shape == (64, 128)
        assert result["block.proj_b.weight"].shape == (64, 128)
        torch.testing.assert_close(
            torch.cat([result["block.proj_a.weight"], result["block.proj_b.weight"]], dim=1),
            fused,
        )


# ---------------------------------------------------------------------------
# Tests using real WeightConverter / Chunk from weight_convert module
# ---------------------------------------------------------------------------
class TestRealWeightConverter:
    """Tests using the actual WeightConverter and Chunk from quark.torch.quantization.weight_convert."""

    def test_chunk_gate_up_proj_with_real_converter(self):
        """Split gate_up_proj using real WeightConverter + Chunk."""
        gate_up = torch.randn(256, 128)
        tensors = {"model.layers.0.mlp.experts.5.gate_up_proj.weight": gate_up}

        converter = WeightConverter(
            "gate_up_proj.weight",
            ["gate_proj.weight", "up_proj.weight"],
            operations=[Chunk(dim=0)],
        )

        result = _apply_weight_converters(tensors, [converter])

        assert "model.layers.0.mlp.experts.5.gate_proj.weight" in result
        assert "model.layers.0.mlp.experts.5.up_proj.weight" in result
        assert result["model.layers.0.mlp.experts.5.gate_proj.weight"].shape == (128, 128)
        assert result["model.layers.0.mlp.experts.5.up_proj.weight"].shape == (128, 128)
        torch.testing.assert_close(
            torch.cat(
                [
                    result["model.layers.0.mlp.experts.5.gate_proj.weight"],
                    result["model.layers.0.mlp.experts.5.up_proj.weight"],
                ],
                dim=0,
            ),
            gate_up,
        )

    def test_chunk_qkv_proj_with_real_converter(self):
        """Split qkv_proj into q/k/v using real WeightConverter + Chunk."""
        qkv = torch.randn(768, 256)
        tensors = {"model.layers.3.self_attn.qkv_proj.weight": qkv}

        converter = WeightConverter(
            "qkv_proj.weight",
            ["q_proj.weight", "k_proj.weight", "v_proj.weight"],
            operations=[Chunk(dim=0)],
        )

        result = _apply_weight_converters(tensors, [converter])

        assert len(result) == 3
        for key_suffix in ["q_proj.weight", "k_proj.weight", "v_proj.weight"]:
            full_key = f"model.layers.3.self_attn.{key_suffix}"
            assert full_key in result
            assert result[full_key].shape == (256, 256)

    def test_multiple_real_converters(self):
        """Multiple real WeightConverters applied to different tensors."""
        gate_up = torch.randn(256, 128)
        qkv = torch.randn(768, 256)
        bias = torch.randn(64)
        tensors = {
            "model.layers.0.mlp.gate_up_proj.weight": gate_up,
            "model.layers.0.self_attn.qkv_proj.weight": qkv,
            "model.layers.0.self_attn.o_proj.bias": bias,
        }

        converters = [
            WeightConverter(
                "gate_up_proj.weight",
                ["gate_proj.weight", "up_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                "qkv_proj.weight",
                ["q_proj.weight", "k_proj.weight", "v_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
        ]

        result = _apply_weight_converters(tensors, converters)

        assert len(result) == 6
        assert "model.layers.0.mlp.gate_proj.weight" in result
        assert "model.layers.0.mlp.up_proj.weight" in result
        assert "model.layers.0.self_attn.q_proj.weight" in result
        assert "model.layers.0.self_attn.k_proj.weight" in result
        assert "model.layers.0.self_attn.v_proj.weight" in result
        assert result["model.layers.0.self_attn.o_proj.bias"] is bias

    def test_real_converter_preserves_data(self):
        """Verify chunked data from real Chunk exactly matches original."""
        gate_up = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        tensors = {"layer.gate_up_proj.weight": gate_up}

        converter = WeightConverter(
            "gate_up_proj.weight",
            ["gate_proj.weight", "up_proj.weight"],
            operations=[Chunk(dim=0)],
        )

        result = _apply_weight_converters(tensors, [converter])

        torch.testing.assert_close(result["layer.gate_proj.weight"], gate_up[:4])
        torch.testing.assert_close(result["layer.up_proj.weight"], gate_up[4:])

    def test_multi_source_real_weight_converter_rejected(self):
        """Real ``WeightConverter`` with many-to-one merge is rejected in file-to-file apply."""
        converter = WeightConverter(
            ["layer.q_proj.weight", "layer.k_proj.weight"],
            ["layer.qk_proj.weight"],
            operations=[Concatenate(dim=0)],
        )
        tensors = {"layer.q_proj.weight": torch.randn(8, 4)}

        with pytest.raises(ValueError, match="one source pattern per converter only"):
            _apply_weight_converters(tensors, [converter])


class TestWeightConvertUtilities:
    """Tests for standalone weight conversion helpers and operations."""

    def test_load_state_dict_info_missing_and_mismatched(self):
        """Verify missing and mismatched keys are combined for diagnostics."""
        info = LoadStateDictInfo(
            missing_keys={"missing.weight"},
            unexpected_keys=set(),
            mismatched_keys={("shape.weight", (1,), (2,))},
            error_msgs=[],
            conversion_errors={},
        )

        assert info.missing_and_mismatched() == {"missing.weight", "shape.weight"}

    def test_concat_convert_and_reverse_ops(self):
        """Verify real concat/chunk reverse operations and target validation."""
        concat = Concatenate(dim=0)
        result = concat.convert(
            {
                "a.weight": [torch.ones(1, 2)],
                "b.weight": torch.zeros(1, 2),
            },
            source_patterns=["a.weight", "b.weight"],
            target_patterns=["ab.weight"],
        )

        torch.testing.assert_close(result["ab.weight"], torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
        assert isinstance(concat.reverse_op, Chunk)
        assert isinstance(Chunk(dim=1).reverse_op, Concatenate)
        with pytest.raises(ValueError, match="Undefined Operation"):
            concat.get_target_pattern(["a.weight", "b.weight"])
        with pytest.raises(ValueError, match="Undefined Operation"):
            Chunk().get_target_patterns({"a.weight": [torch.ones(1)], "b.weight": [torch.ones(1)]}, ["out.weight"])

    def test_process_and_rename_patterns_with_backrefs(self):
        """Verify regex capture groups are reused when renaming source keys."""
        processed_pattern, captured_group = process_target_pattern(r"target.(\d+).weight$")
        assert processed_pattern == r"target.\1.weight"
        assert captured_group == r"(\d+)"

        transform = WeightTransform(
            source_patterns=r"source.\1.weight",
            target_patterns=r"target.(\d+).weight$",
        )
        renamed_key, matched_pattern = transform.rename_source_key("model.source.7.weight")
        assert renamed_key == "model.target.7.weight"
        assert matched_pattern == r"source.(\d+).weight"
        assert transform.rename_source_key("model.other.weight") == ("model.other.weight", None)

        with pytest.raises(ValueError, match="contains \\\\1 backreference"):
            WeightTransform(source_patterns=r"source\.\1\.weight", target_patterns="target.weight")

    def test_weight_transform_materializes_futures_callables_and_direct_tensors(self):
        """Verify tensor collection materializes all supported deferred tensor forms."""
        transform = WeightTransform("src.weight", "dst.weight")
        future: Future[torch.Tensor | None] = Future()
        future.set_result(torch.tensor([1.0]))

        transform.add_tensor("layer.dst.weight", "layer.src.weight", "src.weight", future)
        transform.collected_tensors["callable.weight"].append(lambda: torch.tensor([2.0]))  # type: ignore[arg-type]
        transform.collected_tensors["direct.weight"].append(torch.tensor([3.0]))  # type: ignore[arg-type]

        materialized = transform.materialize_tensors()

        torch.testing.assert_close(materialized["src.weight"][0], torch.tensor([1.0]))
        torch.testing.assert_close(materialized["callable.weight"][0], torch.tensor([2.0]))
        torch.testing.assert_close(materialized["direct.weight"][0], torch.tensor([3.0]))
        assert transform.layer_targets["layer.dst.weight"] == {"layer.src.weight"}

    def test_log_conversion_errors_records_all_message_shapes(self):
        """Verify conversion error logging covers tuple, string, op-only, and generic messages."""
        info = LoadStateDictInfo(set(), set(), set(), [], {})

        with pytest.raises(RuntimeError, match="no-loading-info"), log_conversion_errors("no_info.weight", None):
            raise RuntimeError("no-loading-info")

        with (
            pytest.raises(SkipParameters),
            log_conversion_errors("tuple.weight", info, (2, "target.weight"), [Chunk(), None, Concatenate()]),
        ):
            raise RuntimeError("tuple-error")
        assert "Chunk, Concatenate" in info.conversion_errors["tuple.weight"]
        assert "target.weight" in info.conversion_errors["tuple.weight"]

        with pytest.raises(SkipParameters), log_conversion_errors("string.weight", info, "parameter.weight", Chunk()):
            raise RuntimeError("string-error")
        assert "via Chunk" in info.conversion_errors["string.weight"]
        assert "parameter.weight" in info.conversion_errors["string.weight"]

        with pytest.raises(SkipParameters), log_conversion_errors("op.weight", info, None, Concatenate()):
            raise RuntimeError("op-error")
        assert info.conversion_errors["op.weight"] == "Concatenate: op-error"

        with pytest.raises(SkipParameters), log_conversion_errors("generic.weight", info, {"extra": "context"}):
            raise RuntimeError("generic-error")
        assert info.conversion_errors["generic.weight"] == "{'extra': 'context'} |Error: generic-error"


class TestSplitFusedExperts:
    """Tests for the SplitFusedExperts ConversionOp."""

    def test_split_gate_up_unfuses_experts_and_halves(self):
        """3D fused gate_up tensor → per-expert gate/up halves, keys prefixed with expert index."""
        num_experts, intermediate, hidden = 4, 8, 6
        fused = torch.arange(num_experts * 2 * intermediate * hidden, dtype=torch.float32).reshape(
            num_experts, 2 * intermediate, hidden
        )
        op = SplitFusedExperts()

        result = op.convert(
            {"gate_up_proj": fused},
            source_patterns=["gate_up_proj"],
            target_patterns=["gate_proj.weight", "up_proj.weight"],
        )

        assert len(result) == 2 * num_experts
        for expert_index in range(num_experts):
            gate_key = f"{expert_index}.gate_proj.weight"
            up_key = f"{expert_index}.up_proj.weight"
            assert result[gate_key].shape == (intermediate, hidden)
            assert result[up_key].shape == (intermediate, hidden)
            torch.testing.assert_close(result[gate_key], fused[expert_index, :intermediate, :])
            torch.testing.assert_close(result[up_key], fused[expert_index, intermediate:, :])

    def test_split_down_proj_unfuses_experts_only(self):
        """3D fused down_proj → per-expert tensors, no further halving."""
        num_experts, out_dim, in_dim = 3, 6, 8
        fused = torch.arange(num_experts * out_dim * in_dim, dtype=torch.float32).reshape(num_experts, out_dim, in_dim)
        op = SplitFusedExperts()

        result = op.convert(
            {"down_proj": fused},
            source_patterns=["down_proj"],
            target_patterns=["down_proj.weight"],
        )

        assert len(result) == num_experts
        for expert_index in range(num_experts):
            key = f"{expert_index}.down_proj.weight"
            assert result[key].shape == (out_dim, in_dim)
            torch.testing.assert_close(result[key], fused[expert_index])

    def test_rejects_non_3d_input(self):
        op = SplitFusedExperts()
        with pytest.raises(ValueError, match="Expected a 3D fused expert tensor"):
            op.convert(
                {"down_proj": torch.zeros(4, 6)},
                source_patterns=["down_proj"],
                target_patterns=["down_proj.weight"],
            )

    def test_rejects_odd_intermediate_for_gate_up(self):
        op = SplitFusedExperts()
        with pytest.raises(ValueError, match="even size along axis"):
            op.convert(
                {"gate_up_proj": torch.zeros(2, 7, 4)},
                source_patterns=["gate_up_proj"],
                target_patterns=["gate_proj.weight", "up_proj.weight"],
            )

    def test_rejects_too_many_target_patterns(self):
        op = SplitFusedExperts()
        with pytest.raises(ValueError, match="Expected 1 or 2 target patterns"):
            op.convert(
                {"src": torch.zeros(2, 4, 4)},
                source_patterns=["src"],
                target_patterns=["a.weight", "b.weight", "c.weight"],
            )

    def test_end_to_end_with_apply_weight_converters_gate_up(self):
        """Full file2file path: tensor name prefix is preserved, experts unfused under the prefix."""
        num_experts, intermediate, hidden = 2, 4, 3
        fused = torch.arange(num_experts * 2 * intermediate * hidden, dtype=torch.float32).reshape(
            num_experts, 2 * intermediate, hidden
        )
        tensors = {"model.layers.0.mlp.experts.gate_up_proj": fused}

        converter = WeightConverter(
            "gate_up_proj",
            ["gate_proj.weight", "up_proj.weight"],
            operations=[SplitFusedExperts()],
        )

        result = _apply_weight_converters(tensors, [converter])

        expected_keys = {
            f"model.layers.0.mlp.experts.{i}.{name}"
            for i in range(num_experts)
            for name in ("gate_proj.weight", "up_proj.weight")
        }
        assert set(result) == expected_keys
        torch.testing.assert_close(
            result["model.layers.0.mlp.experts.1.up_proj.weight"],
            fused[1, intermediate:, :],
        )

    def test_end_to_end_with_apply_weight_converters_down_proj(self):
        """Full file2file path for the down_proj (single target) variant."""
        num_experts, out_dim, in_dim = 3, 5, 4
        fused = torch.arange(num_experts * out_dim * in_dim, dtype=torch.float32).reshape(num_experts, out_dim, in_dim)
        tensors = {"model.layers.7.mlp.experts.down_proj": fused}

        converter = WeightConverter(
            "down_proj",
            ["down_proj.weight"],
            operations=[SplitFusedExperts()],
        )

        result = _apply_weight_converters(tensors, [converter])

        assert set(result) == {f"model.layers.7.mlp.experts.{i}.down_proj.weight" for i in range(num_experts)}
        torch.testing.assert_close(result["model.layers.7.mlp.experts.2.down_proj.weight"], fused[2])

    def test_fused_vs_split_moe_forward_is_bit_exact(self):
        """End-to-end equivalence: a SwiGLU MoE forward must produce identical
        outputs whether it is computed from the fused (3D) expert weights or
        from the per-expert weights produced by ``SplitFusedExperts``.

        Splitting is pure slicing, so the two forward passes must match
        exactly (``torch.equal``), not just be numerically close.
        """
        torch.manual_seed(0)
        num_experts, intermediate, hidden = 4, 5, 7
        num_tokens = 6
        top_k = 2

        fused_gate_up = torch.randn(num_experts, 2 * intermediate, hidden)
        fused_down = torch.randn(num_experts, hidden, intermediate)
        hidden_states = torch.randn(num_tokens, hidden)
        # Deterministic routing: each token picks `top_k` distinct experts.
        topk_experts = torch.stack(
            [torch.randperm(num_experts)[:top_k] for _ in range(num_tokens)],
            dim=0,
        )
        topk_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

        def swiglu_moe(gate_proj_fn, up_proj_fn, down_proj_fn):
            """Generic MoE forward; the three *_fn return the per-expert weight."""
            out = torch.zeros_like(hidden_states)
            for token_index in range(num_tokens):
                x = hidden_states[token_index]
                for k in range(top_k):
                    expert_index = int(topk_experts[token_index, k])
                    weight = topk_weights[token_index, k]
                    gate = x @ gate_proj_fn(expert_index).T
                    up = x @ up_proj_fn(expert_index).T
                    activated = torch.nn.functional.silu(gate) * up
                    out[token_index] += weight * (activated @ down_proj_fn(expert_index).T)
            return out

        fused_out = swiglu_moe(
            gate_proj_fn=lambda i: fused_gate_up[i, :intermediate, :],
            up_proj_fn=lambda i: fused_gate_up[i, intermediate:, :],
            down_proj_fn=lambda i: fused_down[i],
        )

        tensors = {
            "model.layers.0.mlp.experts.gate_up_proj": fused_gate_up,
            "model.layers.0.mlp.experts.down_proj": fused_down,
        }
        converters = [
            WeightConverter(
                "gate_up_proj",
                ["gate_proj.weight", "up_proj.weight"],
                operations=[SplitFusedExperts()],
            ),
            WeightConverter(
                "down_proj",
                ["down_proj.weight"],
                operations=[SplitFusedExperts()],
            ),
        ]
        split = _apply_weight_converters(tensors, converters)

        prefix = "model.layers.0.mlp.experts"
        split_out = swiglu_moe(
            gate_proj_fn=lambda i: split[f"{prefix}.{i}.gate_proj.weight"],
            up_proj_fn=lambda i: split[f"{prefix}.{i}.up_proj.weight"],
            down_proj_fn=lambda i: split[f"{prefix}.{i}.down_proj.weight"],
        )

        assert torch.equal(fused_out, split_out)


class TestWeightTransformAndConverter:
    """Tests for conversion object validation, reversing, and conversion flows."""

    def test_weight_converter_validation_and_reverse_transform(self):
        """Verify validation errors and reverse operation construction."""
        with pytest.raises(ValueError, match="you can only have one to many"):
            WeightConverter(
                ["q.weight", "k.weight"],
                ["q_out.weight", "k_out.weight"],
                operations=[IdentityOperation()],
            )
        with pytest.raises(ValueError, match="requires at least one operation"):
            WeightConverter("src.weight", "dst.weight")

        converter = WeightConverter("src.weight", ["a.weight", "b.weight"], operations=[Chunk(dim=0)])
        reversed_converter = converter.reverse_transform()
        assert reversed_converter.source_patterns == ["a.weight", "b.weight"]
        assert reversed_converter.target_patterns == ["src.weight"]
        assert isinstance(reversed_converter.operations[0], Concatenate)

        converter.quantization_operation = AddOneOperation()
        with pytest.raises(ValueError, match="Cannot reverse"):
            converter.reverse_transform()

    def test_weight_converter_convert_expands_layer_prefix(self):
        """Verify conversion expands output keys using the matched layer prefix."""
        converter = WeightConverter("src.weight", ["a.weight", "b.weight"], operations=[Chunk(dim=0)])
        converter.collected_tensors["src.weight"].append(torch.arange(8, dtype=torch.float32).reshape(4, 2))

        result = converter.convert("decoder.a.weight")

        assert set(result) == {"decoder.a.weight", "decoder.b.weight"}
        torch.testing.assert_close(result["decoder.a.weight"], torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
        torch.testing.assert_close(result["decoder.b.weight"], torch.tensor([[4.0, 5.0], [6.0, 7.0]]))

    def test_weight_converter_convert_handles_wildcard_layer_name_and_quantization_op(self):
        """Verify wildcard layer names and optional quantization operation are applied."""
        converter = WeightConverter("src.weight", "target.weight", operations=[IdentityOperation()])
        converter.collected_tensors["src.weight"].append(torch.tensor([1.0]))

        wildcard_result = converter.convert("layers.*.target.weight")
        assert set(wildcard_result) == {"layers.0.target.weight"}
        torch.testing.assert_close(wildcard_result["layers.0.target.weight"], torch.tensor([1.0]))

        quant_converter = WeightConverter("src.weight", "target.weight", operations=[IdentityOperation()])
        quant_converter.quantization_operation = AddOneOperation()
        quant_converter.collected_tensors["src.weight"].append(torch.tensor([2.0]))

        quantized_result = quant_converter.convert("unmatched.layer", hf_quantizer=object())
        torch.testing.assert_close(quantized_result["target.weight"], torch.tensor([3.0]))
