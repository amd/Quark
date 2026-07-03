#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Test that Quark preprocess classes produce same forward output as their HF counterparts."""

import ast
import importlib
import inspect
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from quark.common.utils.import_utils import (
    is_transformers_available,
    is_transformers_version_higher_or_equal,
)
from quark.common.utils.testing_utils import torch_device

if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    # Import to trigger Quark class registration
    from quark.torch.utils.llm.module_replacement import quark_experts  # noqa: F401
    from quark.torch.utils.llm.module_replacement.preprocess_registry import PREPROCESS_REGISTRY
    from quark.torch.utils.llm.module_replacement.quark_experts import QuarkExperts, _moe_experts_sync_weights_to_linear

_BROAD_EQ_IGNORED_EXPERTS = {
    # Covered by dedicated GPT-OSS tests in this file.
    ("transformers.models.gpt_oss.modeling_gpt_oss", "GptOssExperts"),
    # Excluded per request.
    ("transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeExperts"),
    ("transformers.models.llama4.modeling_llama4", "Llama4TextExperts"),
}


def _get_hf_module(model: torch.nn.Module, module_type: type) -> tuple[str, torch.nn.Module] | None:
    """Find first module of given type in model. Returns (name, module) or None."""
    for name, module in model.named_modules():
        if isinstance(module, module_type):
            return (name, module)
    return None


class _DummyExpertGateUp(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=True)
        self.down_proj = nn.Linear(intermediate, hidden, bias=True)


class _DummyExpertSplit(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=True)
        self.up_proj = nn.Linear(hidden, intermediate, bias=True)
        self.down_proj = nn.Linear(intermediate, hidden, bias=True)


class _DummyReplaceModule(nn.Module):
    def __init__(self, experts: list[nn.Module]):
        super().__init__()
        self.num_experts = len(experts)
        self._weights_synced = False
        for idx, expert in enumerate(experts):
            setattr(self, str(idx), expert)


class _DummyOriginalModule(nn.Module):
    def __init__(self, gate_up: torch.Tensor | None, down: torch.Tensor | None, gate_up_bias=None, down_bias=None):
        super().__init__()
        if gate_up is not None:
            self.gate_up_proj = gate_up
        if down is not None:
            self.down_proj = down
        self.gate_up_proj_bias = gate_up_bias
        self.down_proj_bias = down_bias


@pytest.mark.skipif(
    not is_transformers_available() or not is_transformers_version_higher_or_equal("5.0.0"),
    reason="transformers >= 5.0.0 required",
)
class TestMoeExpertsSyncWeightsToLinear:
    def test_returns_true_if_already_synced(self):
        replace_module = _DummyReplaceModule([_DummyExpertGateUp(hidden=3, intermediate=2)])
        replace_module._weights_synced = True
        original_module = _DummyOriginalModule(
            gate_up=torch.randn(1, 4, 3),
            down=torch.randn(1, 3, 2),
        )
        assert _moe_experts_sync_weights_to_linear(replace_module, original_module, "qwen3_moe")

    def test_returns_false_if_missing_fused_weights(self):
        replace_module = _DummyReplaceModule([_DummyExpertGateUp(hidden=3, intermediate=2)])
        original_module = _DummyOriginalModule(gate_up=None, down=None)
        assert not _moe_experts_sync_weights_to_linear(replace_module, original_module, "qwen3_moe")

    def test_returns_false_for_empty_fused_weights(self):
        replace_module = _DummyReplaceModule([_DummyExpertGateUp(hidden=3, intermediate=2)])
        original_module = _DummyOriginalModule(
            gate_up=torch.empty(0),
            down=torch.empty(0),
        )
        assert not _moe_experts_sync_weights_to_linear(replace_module, original_module, "qwen3_moe")

    def test_copies_gate_up_and_down_weights_and_bias(self):
        hidden, intermediate, num_experts = 3, 2, 2
        replace_module = _DummyReplaceModule(
            [_DummyExpertGateUp(hidden=hidden, intermediate=intermediate) for _ in range(num_experts)]
        )
        gate_up = torch.randn(num_experts, 2 * intermediate, hidden)
        gate_up_bias = torch.randn(num_experts, 2 * intermediate)
        down = torch.randn(num_experts, hidden, intermediate)
        down_bias = torch.randn(num_experts, hidden)
        original_module = _DummyOriginalModule(
            gate_up=gate_up, down=down, gate_up_bias=gate_up_bias, down_bias=down_bias
        )

        assert _moe_experts_sync_weights_to_linear(replace_module, original_module, "qwen3_moe")
        for idx in range(num_experts):
            expert = getattr(replace_module, str(idx))
            torch.testing.assert_close(expert.gate_up_proj.weight, gate_up[idx])
            torch.testing.assert_close(expert.gate_up_proj.bias, gate_up_bias[idx])
            torch.testing.assert_close(expert.down_proj.weight, down[idx])
            torch.testing.assert_close(expert.down_proj.bias, down_bias[idx])

    def test_copies_split_gate_and_up_projections_with_bias(self):
        hidden, intermediate = 3, 2
        replace_module = _DummyReplaceModule([_DummyExpertSplit(hidden=hidden, intermediate=intermediate)])
        gate_up = torch.randn(1, 2 * intermediate, hidden)
        gate_up_bias = torch.randn(1, 2 * intermediate)
        down = torch.randn(1, hidden, intermediate)
        down_bias = torch.randn(1, hidden)
        original_module = _DummyOriginalModule(
            gate_up=gate_up, down=down, gate_up_bias=gate_up_bias, down_bias=down_bias
        )

        assert _moe_experts_sync_weights_to_linear(replace_module, original_module, "qwen3_moe")
        expert = getattr(replace_module, "0")
        torch.testing.assert_close(expert.gate_proj.weight, gate_up[0][:intermediate, :])
        torch.testing.assert_close(expert.up_proj.weight, gate_up[0][intermediate:, :])
        torch.testing.assert_close(expert.gate_proj.bias, gate_up_bias[0][:intermediate])
        torch.testing.assert_close(expert.up_proj.bias, gate_up_bias[0][intermediate:])
        torch.testing.assert_close(expert.down_proj.weight, down[0])
        torch.testing.assert_close(expert.down_proj.bias, down_bias[0])

    def test_gpt_oss_transposes_weights(self):
        hidden, intermediate = 3, 2
        replace_module = _DummyReplaceModule([_DummyExpertGateUp(hidden=hidden, intermediate=intermediate)])
        gate_up = torch.randn(1, hidden, 2 * intermediate)
        down = torch.randn(1, intermediate, hidden)
        original_module = _DummyOriginalModule(gate_up=gate_up, down=down)

        assert _moe_experts_sync_weights_to_linear(replace_module, original_module, "gpt_oss")
        expert = getattr(replace_module, "0")
        torch.testing.assert_close(expert.gate_up_proj.weight, gate_up[0].t())
        torch.testing.assert_close(expert.down_proj.weight, down[0].t())

    def test_raises_when_expert_has_no_supported_projection_attributes(self):
        class _UnsupportedExpert(nn.Module):
            def __init__(self, hidden: int, intermediate: int):
                super().__init__()
                self.down_proj = nn.Linear(intermediate, hidden, bias=True)

        replace_module = _DummyReplaceModule([_UnsupportedExpert(hidden=3, intermediate=2)])
        original_module = _DummyOriginalModule(
            gate_up=torch.randn(1, 4, 3),
            down=torch.randn(1, 3, 2),
        )
        with pytest.raises(AttributeError, match="neither 'gate_up_proj' nor 'gate_proj'/'up_proj'"):
            _moe_experts_sync_weights_to_linear(replace_module, original_module, "qwen3_moe")


@pytest.mark.skipif(
    not is_transformers_available() or not is_transformers_version_higher_or_equal("5.0.0"),
    reason="transformers >= 5.0.0 required for QuarkExperts branch coverage tests",
)
class TestQuarkExpertsBranchCoverage:
    class _DummyGatedHFExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 1
            self.has_gate = True
            self.has_bias = True
            self.is_transposed = True
            self.act_fn = torch.sigmoid
            # Stored transposed to exercise transpose branches.
            self.gate_up_proj = torch.tensor(
                [[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 300.0, 400.0]]]
            )
            self.gate_up_proj_bias = torch.tensor([[0.5, 1.5, 2.5, 3.5]])
            self.down_proj = torch.tensor([[[7.0, 8.0, 9.0], [70.0, 80.0, 90.0]]])
            self.down_proj_bias = torch.tensor([[4.0, 5.0, 6.0]])

    class _DummyNonGatedHFExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 1
            self.has_gate = False
            self.has_bias = True
            self.is_transposed = True
            self.act_fn = torch.relu
            # Stored transposed to exercise transpose branches.
            self.up_proj = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
            self.up_proj_bias = torch.tensor([[0.25, 0.75]])
            self.down_proj = torch.tensor([[[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]]])
            self.down_proj_bias = torch.tensor([[0.1, 0.2, 0.3]])

    def test_default_apply_gate_splits_and_multiplies(self):
        gate_up = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
        out = quark_experts._default_apply_gate(torch.sigmoid, gate_up)
        expected = torch.sigmoid(gate_up[:, :2]) * gate_up[:, 2:]
        torch.testing.assert_close(out, expected)

    def test_init_copies_transposed_gated_and_non_gated_weights_with_bias(self):
        # __init__ builds per-expert nn.Linear modules on the meta device and then
        # assigns each ``linear.weight`` to a (sliced + transposed) view of the
        # source fused tensor — no copy. The per-expert linears therefore share
        # storage with hf_experts.gate_up_proj / down_proj / up_proj. That sharing
        # is intentional: __init__ stays cheap, and ``from_hf`` is what later
        # ``del``s the fused source attributes (Hotspot 1 in #5413) so that, once
        # downstream code replaces the per-expert linears with quantized ones, the
        # original storage can finally be freed.
        gated_hf_experts = self._DummyGatedHFExperts()
        gated_quark = QuarkExperts(gated_hf_experts)
        gated_expert = getattr(gated_quark, "0")
        expected_gated = gated_hf_experts.gate_up_proj[0].transpose(0, 1)
        expected_gate_bias = gated_hf_experts.gate_up_proj_bias[0]
        torch.testing.assert_close(gated_expert.gate_proj.weight, expected_gated[:2])
        torch.testing.assert_close(gated_expert.up_proj.weight, expected_gated[2:])
        torch.testing.assert_close(gated_expert.gate_proj.bias, expected_gate_bias[:2])
        torch.testing.assert_close(gated_expert.up_proj.bias, expected_gate_bias[2:])
        expected_down = gated_hf_experts.down_proj[0].transpose(0, 1)
        torch.testing.assert_close(gated_expert.down_proj.weight, expected_down)
        torch.testing.assert_close(gated_expert.down_proj.bias, gated_hf_experts.down_proj_bias[0])
        gated_source_weight_storage_pointer = gated_hf_experts.gate_up_proj.untyped_storage().data_ptr()
        assert gated_expert.gate_proj.weight.untyped_storage().data_ptr() == gated_source_weight_storage_pointer
        assert gated_expert.up_proj.weight.untyped_storage().data_ptr() == gated_source_weight_storage_pointer
        assert (
            gated_expert.down_proj.weight.untyped_storage().data_ptr()
            == gated_hf_experts.down_proj.untyped_storage().data_ptr()
        )

        nongated_hf_experts = self._DummyNonGatedHFExperts()
        nongated_quark = QuarkExperts(nongated_hf_experts)
        nongated_expert = getattr(nongated_quark, "0")
        expected_up = nongated_hf_experts.up_proj[0].transpose(0, 1)
        expected_up_bias = nongated_hf_experts.up_proj_bias[0]
        torch.testing.assert_close(nongated_expert.up_proj.weight, expected_up)
        torch.testing.assert_close(nongated_expert.up_proj.bias, expected_up_bias)
        expected_down_nongated = nongated_hf_experts.down_proj[0].transpose(0, 1)
        torch.testing.assert_close(nongated_expert.down_proj.weight, expected_down_nongated)
        torch.testing.assert_close(nongated_expert.down_proj.bias, nongated_hf_experts.down_proj_bias[0])
        assert (
            nongated_expert.up_proj.weight.untyped_storage().data_ptr()
            == nongated_hf_experts.up_proj.untyped_storage().data_ptr()
        )
        assert (
            nongated_expert.down_proj.weight.untyped_storage().data_ptr()
            == nongated_hf_experts.down_proj.untyped_storage().data_ptr()
        )

    def test_apply_gate_prefers_custom_callable(self):
        quark = QuarkExperts(self._DummyGatedHFExperts())
        quark._custom_apply_gate = lambda gate_up_out: gate_up_out + 2.0
        gate_up = torch.randn(3, 4)
        torch.testing.assert_close(quark._apply_gate(gate_up), gate_up + 2.0)

    def test_apply_gate_raises_without_act_fn_or_custom(self):
        hf_experts = self._DummyGatedHFExperts()
        hf_experts.act_fn = None
        quark = QuarkExperts(hf_experts)
        quark._custom_apply_gate = None
        with pytest.raises(RuntimeError, match="`act_fn` is required"):
            quark._apply_gate(torch.randn(2, 4))

    def test_forward_non_gated_uses_activation_path(self):
        quark = QuarkExperts(self._DummyNonGatedHFExperts())
        hidden_states = torch.randn(5, 3)
        top_k_index = torch.zeros((5, 1), dtype=torch.long)
        top_k_weights = torch.ones((5, 1), dtype=hidden_states.dtype)
        out = quark(hidden_states, top_k_index, top_k_weights)
        assert out.shape == hidden_states.shape
        assert torch.isfinite(out).all()

    def test_from_hf_releases_gated_fused_weights(self):
        gated_hf_experts = self._DummyGatedHFExperts()
        assert hasattr(gated_hf_experts, "gate_up_proj")
        assert hasattr(gated_hf_experts, "down_proj")

        quark = QuarkExperts.from_hf(gated_hf_experts)

        # Per-expert linears still exist with the original weights baked in.
        assert getattr(quark, "0").gate_proj.weight.numel() > 0
        assert getattr(quark, "0").down_proj.weight.numel() > 0
        # Fused source weights on hf_module are dropped so Hotspot 1 (#5413) cannot recur.
        assert not hasattr(gated_hf_experts, "gate_up_proj")
        assert not hasattr(gated_hf_experts, "down_proj")

    def test_from_hf_releases_nongated_fused_weights(self):
        nongated_hf_experts = self._DummyNonGatedHFExperts()
        assert hasattr(nongated_hf_experts, "up_proj")
        assert hasattr(nongated_hf_experts, "down_proj")

        quark = QuarkExperts.from_hf(nongated_hf_experts)

        assert getattr(quark, "0").up_proj.weight.numel() > 0
        assert getattr(quark, "0").down_proj.weight.numel() > 0
        assert not hasattr(nongated_hf_experts, "up_proj")
        assert not hasattr(nongated_hf_experts, "down_proj")

    def test_forward_skips_sentinel_expert_index(self, monkeypatch):
        quark = QuarkExperts(self._DummyNonGatedHFExperts())
        original_one_hot = torch.nn.functional.one_hot

        def _one_hot_with_sentinel(index: torch.Tensor, num_classes: int = -1):
            one_hot = original_one_hot(index, num_classes + 1)
            one_hot[..., num_classes] = 1
            return one_hot

        monkeypatch.setattr(torch.nn.functional, "one_hot", _one_hot_with_sentinel)
        hidden_states = torch.randn(4, 3)
        top_k_index = torch.zeros((4, 1), dtype=torch.long)
        top_k_weights = torch.ones((4, 1), dtype=hidden_states.dtype)
        out = quark(hidden_states, top_k_index, top_k_weights)
        assert out.shape == hidden_states.shape
        assert torch.isfinite(out).all()


@pytest.mark.skipif(
    not is_transformers_available() or not is_transformers_version_higher_or_equal("5.0.0"),
    reason="transformers >= 5.0.0 required for Qwen3Moe",
)
class TestQuarkQwen3MoeTopKRouter:
    """Test QuarkQwen3MoeTopKRouter forward matches Qwen3MoeTopKRouter."""

    def test_forward_equivalence(self):
        from transformers import AutoModelForCausalLM, Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

        from quark.torch.utils.llm.module_replacement.quark_experts import QuarkQwen3MoeTopKRouter

        config = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=2,
            moe_intermediate_size=64,
            num_experts_per_tok=1,
            experts_implementation="eager",
        )
        model = AutoModelForCausalLM.from_config(config).to(torch_device).eval()
        result = _get_hf_module(model, Qwen3MoeTopKRouter)
        assert result is not None, "Qwen3MoeTopKRouter not found in model"
        _, hf_router = result
        quark_router = QuarkQwen3MoeTopKRouter.from_hf(hf_router)

        dtype = next(hf_router.parameters()).dtype
        torch.manual_seed(42)
        batch, seq, hidden = 2, 4, config.hidden_size
        hidden_states = torch.randn(batch, seq, hidden, device=torch_device, dtype=dtype)

        with torch.no_grad():
            hf_out = hf_router(hidden_states)
            quark_out = quark_router(hidden_states)

        for hf_o, q_o in zip(hf_out, quark_out, strict=False):
            torch.testing.assert_close(q_o, hf_o, rtol=1e-5, atol=1e-5)

    def test_forward_with_norm_topk_prob(self):
        from transformers import Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

        from quark.torch.utils.llm.module_replacement.quark_experts import QuarkQwen3MoeTopKRouter

        hidden_dim, num_experts, top_k = 8, 4, 2
        router = (
            Qwen3MoeTopKRouter(
                Qwen3MoeConfig(
                    hidden_size=hidden_dim,
                    num_experts=num_experts,
                    num_experts_per_tok=top_k,
                    norm_topk_prob=True,
                )
            )
            .to(torch_device)
            .eval()
        )
        quark_router = QuarkQwen3MoeTopKRouter.from_hf(router)
        hidden_states = torch.randn(3, hidden_dim, device=torch_device, dtype=router.weight.dtype)
        with torch.no_grad():
            _, router_scores, _ = quark_router(hidden_states)

        score_sums = router_scores.sum(dim=-1)
        torch.testing.assert_close(score_sums, torch.ones_like(score_sums), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not is_transformers_available() or not is_transformers_version_higher_or_equal("5.0.0"),
    reason="transformers >= 5.0.0 required for GPT-OSS",
)
class TestQuarkGptOssTopKRouter:
    """Test QuarkGptOssTopKRouter forward matches GptOssTopKRouter semantics."""

    def test_forward_equivalence(self):
        from transformers import AutoModelForCausalLM, GptOssConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter

        config = GptOssConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_local_experts=4,
            num_experts_per_tok=2,
        )
        model = AutoModelForCausalLM.from_config(config).to(torch_device).eval()
        result = _get_hf_module(model, GptOssTopKRouter)
        assert result is not None, "GptOssTopKRouter not found in model"
        _, hf_router = result

        quark_router = PREPROCESS_REGISTRY[GptOssTopKRouter].from_hf(hf_router)
        dtype = hf_router.weight.dtype
        torch.manual_seed(42)
        batch, seq, hidden = 2, 4, config.hidden_size
        hidden_states = torch.randn(batch, seq, hidden, device=torch_device, dtype=dtype)
        hidden_states_flat = hidden_states.reshape(-1, hidden)

        with torch.no_grad():
            hf_logits, hf_scores_topk, hf_indices = hf_router(hidden_states_flat)
            quark_logits, quark_scores_dense, quark_indices = quark_router(hidden_states_flat)

        torch.testing.assert_close(quark_logits, hf_logits, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(quark_indices, hf_indices)
        torch.testing.assert_close(quark_scores_dense, hf_scores_topk, rtol=1e-5, atol=1e-5)

    def test_weight_and_bias_properties_proxy_linear_parameters(self):
        from quark.torch.utils.llm.module_replacement.quark_experts import QuarkGptOssTopKRouter

        router = QuarkGptOssTopKRouter(
            hidden_dim=8,
            num_experts=4,
            top_k=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert router.weight is router.linear.weight
        assert router.bias is router.linear.bias

    def test_state_dict_uses_prefix_and_destination(self):
        from quark.torch.utils.llm.module_replacement.quark_experts import QuarkGptOssTopKRouter

        router = QuarkGptOssTopKRouter(
            hidden_dim=8,
            num_experts=4,
            top_k=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        destination = {"existing": torch.tensor(1)}

        state = router.state_dict(destination=destination, prefix="router.")

        assert state is destination
        assert "existing" in state
        assert "router.weight" in state
        assert "router.bias" in state
        assert "weight" not in state
        assert "bias" not in state
        torch.testing.assert_close(state["router.weight"], router.linear.weight)
        torch.testing.assert_close(state["router.bias"], router.linear.bias)

    def test_load_state_dict_loads_legacy_flat_keys(self):
        """A flat (no '.linear.') checkpoint loads via the public load_state_dict path.

        The router exposes its inner linear's params as '<prefix>weight'/'<prefix>bias'
        in state_dict(); load_state_dict must accept the same naming and route it
        to self.linear correctly, with strict=True passing.
        """
        from quark.torch.utils.llm.module_replacement.quark_experts import QuarkGptOssTopKRouter

        class Parent(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = QuarkGptOssTopKRouter(
                    hidden_dim=8,
                    num_experts=4,
                    top_k=2,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )

        dst = Parent()
        expected_weight = torch.randn_like(dst.router.linear.weight)
        expected_bias = torch.randn_like(dst.router.linear.bias)
        legacy_state_dict = {
            "router.weight": expected_weight.clone(),
            "router.bias": expected_bias.clone(),
        }

        result = dst.load_state_dict(legacy_state_dict, strict=True)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        torch.testing.assert_close(dst.router.linear.weight, expected_weight)
        torch.testing.assert_close(dst.router.linear.bias, expected_bias)

    def test_full_model_load_state_dict_strict_round_trip(self):
        """Strict round-trip via the real model.load_state_dict path.

        Wraps the router under a parent module with a sibling module so the
        loader's recursive prefix filtering is exercised end-to-end. Verifies
        that:
          - state_dict()/load_state_dict() round-trip succeeds with strict=True,
          - no missing or unexpected keys are reported,
          - every router parameter is actually overwritten by the checkpoint.
        """
        from quark.torch.utils.llm.module_replacement.quark_experts import QuarkGptOssTopKRouter

        class Parent(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate = QuarkGptOssTopKRouter(
                    hidden_dim=8,
                    num_experts=4,
                    top_k=2,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                # Sibling submodule at the same prefix level -- ensures the
                # loader's prefix recursion still works after the rewrite hook.
                self.sibling = nn.Linear(8, 8, bias=True)

        src = Parent()
        dst = Parent()

        # Sanity check the exported keys use the legacy (no ".linear.") naming
        # for the router but normal naming for the sibling.
        state_dict = src.state_dict()
        assert "gate.weight" in state_dict
        assert "gate.bias" in state_dict
        assert "gate.linear.weight" not in state_dict
        assert "gate.linear.bias" not in state_dict
        assert "sibling.weight" in state_dict
        assert "sibling.bias" in state_dict

        # Make dst params differ from src so we can verify they actually load.
        with torch.no_grad():
            for p in dst.parameters():
                p.add_(1.0)
        assert not torch.equal(dst.gate.linear.weight, src.gate.linear.weight)

        result = dst.load_state_dict(state_dict, strict=True)
        assert result.missing_keys == []
        assert result.unexpected_keys == []

        torch.testing.assert_close(dst.gate.linear.weight, src.gate.linear.weight)
        torch.testing.assert_close(dst.gate.linear.bias, src.gate.linear.bias)
        torch.testing.assert_close(dst.sibling.weight, src.sibling.weight)
        torch.testing.assert_close(dst.sibling.bias, src.sibling.bias)

    def test_full_model_load_state_dict_strict_with_linear_subtree(self):
        """Strict load when self.linear has its own submodule (e.g. a quantizer).

        After Quark patches the router's linear with a quantized linear, the
        linear gains submodules with their own parameters (weight_quantizer.scale,
        weight_quantizer.zero_point, ...) but the exported state_dict still drops
        the ".linear." segment for everything in the subtree. The fix must
        rewrite ALL keys at the router's prefix (not just weight/bias) so that
        PyTorch's recursion into self.linear.weight_quantizer also resolves.
        """
        from quark.torch.utils.llm.module_replacement.quark_experts import QuarkGptOssTopKRouter

        class FakeWeightQuantizer(nn.Module):
            def __init__(self, num_experts: int) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.rand(num_experts))
                self.zero_point = nn.Parameter(torch.randn(num_experts))

        class FakeQuantizedLinear(nn.Linear):
            def __init__(self, in_features: int, out_features: int) -> None:
                super().__init__(in_features, out_features, bias=True)
                self.weight_quantizer = FakeWeightQuantizer(out_features)

        class Parent(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate = QuarkGptOssTopKRouter(
                    hidden_dim=8,
                    num_experts=4,
                    top_k=2,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                # Patch the inner linear with one that owns a quantizer subtree.
                self.gate.linear = FakeQuantizedLinear(8, 4)
                self.sibling = nn.Linear(8, 8, bias=True)

        src = Parent()
        dst = Parent()

        state_dict = src.state_dict()
        # The router subtree must be exported in flat (no ".linear.") form for
        # both the linear params AND the quantizer submodule's params.
        assert "gate.weight" in state_dict
        assert "gate.bias" in state_dict
        assert "gate.weight_quantizer.scale" in state_dict
        assert "gate.weight_quantizer.zero_point" in state_dict
        assert "gate.linear.weight" not in state_dict
        assert "gate.linear.weight_quantizer.scale" not in state_dict

        with torch.no_grad():
            for p in dst.parameters():
                p.add_(1.0)

        result = dst.load_state_dict(state_dict, strict=True)
        assert result.missing_keys == []
        assert result.unexpected_keys == []

        torch.testing.assert_close(dst.gate.linear.weight, src.gate.linear.weight)
        torch.testing.assert_close(dst.gate.linear.bias, src.gate.linear.bias)
        torch.testing.assert_close(
            dst.gate.linear.weight_quantizer.scale,
            src.gate.linear.weight_quantizer.scale,
        )
        torch.testing.assert_close(
            dst.gate.linear.weight_quantizer.zero_point,
            src.gate.linear.weight_quantizer.zero_point,
        )
        torch.testing.assert_close(dst.sibling.weight, src.sibling.weight)
        torch.testing.assert_close(dst.sibling.bias, src.sibling.bias)

    def test_full_model_load_state_dict_accepts_new_naming(self):
        """A checkpoint that already uses '<prefix>linear.<name>' must still load."""
        from quark.torch.utils.llm.module_replacement.quark_experts import QuarkGptOssTopKRouter

        class Parent(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate = QuarkGptOssTopKRouter(
                    hidden_dim=8,
                    num_experts=4,
                    top_k=2,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )

        src = Parent()
        dst = Parent()

        new_style_state_dict = {
            "gate.linear.weight": src.gate.linear.weight.detach().clone(),
            "gate.linear.bias": src.gate.linear.bias.detach().clone(),
        }
        with torch.no_grad():
            for p in dst.parameters():
                p.add_(1.0)

        result = dst.load_state_dict(new_style_state_dict, strict=True)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        torch.testing.assert_close(dst.gate.linear.weight, src.gate.linear.weight)
        torch.testing.assert_close(dst.gate.linear.bias, src.gate.linear.bias)

    def test_full_model_load_state_dict_mixed_naming_prefers_new_key(self):
        """When both '<prefix>X' and '<prefix>linear.X' are present, the new-style
        key must win and the legacy key must NOT clobber it."""
        from quark.torch.utils.llm.module_replacement.quark_experts import QuarkGptOssTopKRouter

        class Parent(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate = QuarkGptOssTopKRouter(
                    hidden_dim=8,
                    num_experts=4,
                    top_k=2,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )

        dst = Parent()
        new_weight = torch.randn_like(dst.gate.linear.weight)
        new_bias = torch.randn_like(dst.gate.linear.bias)
        legacy_weight = torch.randn_like(dst.gate.linear.weight)
        legacy_bias = torch.randn_like(dst.gate.linear.bias)
        # Both naming styles in the same checkpoint; new-style should win.
        mixed_state_dict = {
            "gate.linear.weight": new_weight.clone(),
            "gate.linear.bias": new_bias.clone(),
            "gate.weight": legacy_weight.clone(),
            "gate.bias": legacy_bias.clone(),
        }

        # strict=False because the legacy keys are intentionally left over and
        # will be reported as unexpected.
        result = dst.load_state_dict(mixed_state_dict, strict=False)
        assert "gate.weight" in result.unexpected_keys
        assert "gate.bias" in result.unexpected_keys
        torch.testing.assert_close(dst.gate.linear.weight, new_weight)
        torch.testing.assert_close(dst.gate.linear.bias, new_bias)


@pytest.mark.skipif(
    not is_transformers_available() or not is_transformers_version_higher_or_equal("5.0.0"),
    reason="transformers >= 5.0.0 required for GPT-OSS",
)
class TestQuarkGptOssExperts:
    """Test QuarkGptOssExperts forward matches GptOssExperts."""

    def test_forward_equivalence(self):
        from transformers import AutoModelForCausalLM, GptOssConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts, GptOssTopKRouter

        config = GptOssConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_local_experts=4,
            num_experts_per_tok=2,
        )
        model = AutoModelForCausalLM.from_config(config).to(torch_device).eval()
        experts_result = _get_hf_module(model, GptOssExperts)
        assert experts_result is not None, "GptOssExperts not found in model"
        _, hf_experts = experts_result
        router_result = _get_hf_module(model, GptOssTopKRouter)
        assert router_result is not None, "GptOssTopKRouter not found in model"
        _, hf_router = router_result

        quark_experts_mod = PREPROCESS_REGISTRY[GptOssExperts].from_hf(hf_experts)
        dtype = next(hf_experts.parameters()).dtype
        torch.manual_seed(42)
        batch, seq, hidden = 2, 4, config.hidden_size
        hidden_states = torch.randn(batch, seq, hidden, device=torch_device, dtype=dtype)
        hidden_states_flat = hidden_states.reshape(-1, hidden)

        with torch.no_grad():
            _, top_k_weights, top_k_index = hf_router(hidden_states_flat)
            hf_out = hf_experts(hidden_states_flat, top_k_index, top_k_weights)
            quark_out = quark_experts_mod(hidden_states, top_k_index, top_k_weights)

        torch.testing.assert_close(quark_out, hf_out.view(batch, seq, hidden), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not is_transformers_available() or not is_transformers_version_higher_or_equal("5.0.0"),
    reason="transformers >= 5.0.0 required for GraniteMoeHybrid",
)
class TestQuarkGraniteMoeHybridMoE:
    """Test QuarkGraniteMoeHybridMoE forward matches GraniteMoeHybridMoE output."""

    def test_forward_equivalence(self):
        from transformers import AutoModelForCausalLM
        from transformers.models.granitemoehybrid.configuration_granitemoehybrid import GraniteMoeHybridConfig
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import GraniteMoeHybridMoE

        config = GraniteMoeHybridConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=2,
            num_experts_per_tok=1,
            shared_intermediate_size=16,
            mamba_n_heads=64,
            layer_types=["attention"],
        )
        model = AutoModelForCausalLM.from_config(config).to(torch_device).eval()
        result = _get_hf_module(model, GraniteMoeHybridMoE)
        assert result is not None, "GraniteMoeHybridMoE not found in model"
        _, hf_moe = result

        quark_moe = PREPROCESS_REGISTRY[GraniteMoeHybridMoE].from_hf(hf_moe)
        dtype = next(hf_moe.parameters()).dtype
        torch.manual_seed(42)
        batch, seq, hidden = 2, 4, config.hidden_size
        hidden_states = torch.randn(batch, seq, hidden, device=torch_device, dtype=dtype)

        with torch.no_grad():
            hf_out = hf_moe(hidden_states)
            quark_out = quark_moe(hidden_states)

        hf_hidden = hf_out[0] if isinstance(hf_out, tuple) else hf_out
        torch.testing.assert_close(quark_out, hf_hidden, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not is_transformers_available() or not is_transformers_version_higher_or_equal("5.0.0"),
    reason="transformers required for DbrxExperts",
)
class TestQuarkDbrxExperts:
    """Test QuarkDbrxExperts conversion and forward compatibility."""

    def test_forward_equivalence(self):
        from transformers.models.dbrx.configuration_dbrx import DbrxConfig, DbrxFFNConfig
        from transformers.models.dbrx.modeling_dbrx import DbrxExperts

        ffn_config = DbrxFFNConfig(
            hidden_size=16,
            ffn_hidden_size=16,
            moe_num_experts=2,
            moe_top_k=1,
        )
        config = DbrxConfig(
            d_model=16,
            n_heads=4,
            n_layers=1,
            max_seq_len=32,
            vocab_size=64,
            ffn_config=ffn_config,
        )
        # Keep compatibility with implementations that read flattened attrs directly.
        config.hidden_size = config.d_model
        config.ffn_hidden_size = ffn_config.ffn_hidden_size
        config.moe_num_experts = ffn_config.moe_num_experts
        config.moe_top_k = ffn_config.moe_top_k
        config.ffn_act_fn = ffn_config.ffn_act_fn

        torch.manual_seed(7)
        hf_experts = DbrxExperts(config).to(torch_device).eval()
        for param in hf_experts.parameters():
            nn.init.uniform_(param, a=-0.1, b=0.1)
        hf_experts_for_quark = DbrxExperts(config).to(torch_device).eval()
        hf_experts_for_quark.load_state_dict(hf_experts.state_dict())
        dtype = hf_experts.mlp.w1.dtype
        quark_dbrx_experts = PREPROCESS_REGISTRY[DbrxExperts].from_hf(hf_experts_for_quark)
        torch.manual_seed(42)
        batch, seq, hidden = 2, 4, config.hidden_size
        hidden_states = torch.randn(batch, seq, hidden, device=torch_device, dtype=dtype)
        top_k_index = torch.randint(
            0,
            config.moe_num_experts,
            (batch * seq, config.moe_top_k),
            device=torch_device,
        )
        top_k_weights = torch.softmax(
            torch.randn(batch * seq, config.moe_top_k, device=torch_device, dtype=dtype), dim=-1
        )

        with torch.no_grad():
            quark_out = quark_dbrx_experts(hidden_states, top_k_index, top_k_weights)

        assert quark_out.shape == (batch, seq, hidden)
        assert quark_out.dtype == dtype
        assert torch.isfinite(quark_out).all()


@pytest.mark.skipif(
    not is_transformers_available() or not is_transformers_version_higher_or_equal("5.0.0"),
    reason="transformers >= 5.0.0 required for GPT-OSS error-path tests",
)
class TestQuarkGptOssExpertsErrorPaths:
    def test_forward_raises_if_router_inputs_missing(self):
        from quark.torch.utils.llm.module_replacement.quark_experts import QuarkGptOssExperts

        mod = QuarkGptOssExperts(
            num_experts=2,
            hidden_size=8,
            intermediate_size=4,
            limit=1.0,
            alpha=1.0,
            device=torch_device,
            dtype=torch.float32,
        )
        hidden_states = torch.randn(2, 3, 8, device=torch_device)
        with pytest.raises((TypeError, AssertionError)):
            mod(hidden_states)

    def test_from_hf_raises_when_weights_are_meta(self):
        from quark.torch.utils.llm.module_replacement.quark_experts import QuarkGptOssExperts

        class _DummyGptOssExperts:
            num_experts = 1
            hidden_size = 3
            intermediate_size = 2
            limit = 1.0
            alpha = 1.0
            gate_up_proj = torch.empty(1, 4, 3, device="meta")
            down_proj = torch.empty(1, 3, 2, device="meta")

        with pytest.raises((RuntimeError, AssertionError), match="weights are on 'meta'"):
            QuarkGptOssExperts.from_hf(_DummyGptOssExperts())


@pytest.mark.skipif(
    not is_transformers_available() or not is_transformers_version_higher_or_equal("5.0.0"),
    reason="transformers >= 5.0.0 required for broad experts equivalence tests",
)
class TestQuarkExpertsBroadEquivalence:
    @staticmethod
    def _get_transformers_models_root() -> Path:
        import transformers

        package_root = Path(transformers.__file__).resolve().parent
        # Installed package layout.
        models_root = package_root / "models"
        if models_root.exists():
            return models_root
        # Source checkout layout.
        src_models_root = package_root.parents[2] / "src" / "transformers" / "models"
        if src_models_root.exists():
            return src_models_root
        raise RuntimeError("Could not locate transformers models directory")

    @staticmethod
    def _find_decorated_experts_classes() -> list[tuple[str, str]]:
        classes: list[tuple[str, str]] = []
        models_root = TestQuarkExpertsBroadEquivalence._get_transformers_models_root()
        transformers_root = models_root.parent

        for file_path in models_root.rglob("modeling_*.py"):
            rel = file_path.relative_to(transformers_root)
            module_path = ".".join(("transformers", *rel.with_suffix("").parts))
            module_ast = ast.parse(file_path.read_text(encoding="utf-8"))
            for node in module_ast.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                has_decorator = False
                for dec in node.decorator_list:
                    if (
                        isinstance(dec, ast.Name)
                        and dec.id == "use_experts_implementation"
                        or isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Name)
                        and dec.func.id == "use_experts_implementation"
                    ):
                        has_decorator = True
                if has_decorator:
                    classes.append((module_path, node.name))

        return sorted(item for item in classes if item not in _BROAD_EQ_IGNORED_EXPERTS)

    @staticmethod
    def _pick_config_class(module):
        from transformers.configuration_utils import PretrainedConfig

        candidates = []
        for _, obj in vars(module).items():
            if not inspect.isclass(obj):
                continue
            if not issubclass(obj, PretrainedConfig) or obj is PretrainedConfig:
                continue
            if not obj.__name__.endswith("Config"):
                continue
            candidates.append(obj)

        if len(candidates) == 1:
            return candidates[0]

        module_name = module.__name__
        for candidate in candidates:
            if (
                candidate.__module__.split(".")[-1].startswith("configuration_")
                and module_name.split(".")[-1].replace("modeling_", "") in candidate.__module__
            ):
                return candidate

        if candidates:
            return candidates[0]

        raise RuntimeError(f"No config class found in module: {module.__name__}")

    @staticmethod
    def _iter_candidate_configs(module):
        from transformers.configuration_utils import PretrainedConfig

        config_classes = []
        for _, obj in vars(module).items():
            if not inspect.isclass(obj):
                continue
            if not issubclass(obj, PretrainedConfig) or obj is PretrainedConfig:
                continue
            if not obj.__name__.endswith("Config"):
                continue
            config_classes.append(obj)

        preferred = TestQuarkExpertsBroadEquivalence._pick_config_class(module)
        ordered = [preferred] + [cfg_cls for cfg_cls in config_classes if cfg_cls is not preferred]

        for cfg_cls in ordered:
            try:
                root_cfg = cfg_cls()
            except Exception:
                continue

            queue = [root_cfg]
            seen = set()
            while queue:
                cfg = queue.pop(0)
                if id(cfg) in seen:
                    continue
                seen.add(id(cfg))
                yield cfg

                for value in vars(cfg).values():
                    if isinstance(value, PretrainedConfig):
                        queue.append(value)

    @staticmethod
    def _prepare_config_for_experts(cfg, module_path: str, _class_name: str):
        # Keep dimensions small so broad parity testing remains fast and memory-safe.
        for attr, value in (
            ("hidden_size", 64),
            ("intermediate_size", 128),
            ("moe_intermediate_size", 128),
            ("n_routed_experts", 8),
            ("num_local_experts", 8),
            ("num_experts", 8),
        ):
            if hasattr(cfg, attr):
                setattr(cfg, attr, value)
        for attr, value in (("num_experts_per_tok", 2), ("num_experts_per_token", 2)):
            if hasattr(cfg, attr):
                setattr(cfg, attr, value)

        if module_path == "transformers.models.dots1.modeling_dots1":
            if getattr(cfg, "n_routed_experts", None) is None:
                cfg.n_routed_experts = 8
            if getattr(cfg, "num_experts_per_tok", None) is None:
                cfg.num_experts_per_tok = 2

        if module_path == "transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe":
            moe_intermediate_size = getattr(cfg, "moe_intermediate_size", None)
            if isinstance(moe_intermediate_size, list) and len(moe_intermediate_size) > 0:
                cfg.moe_intermediate_size = int(moe_intermediate_size[0])

        cfg._experts_implementation = "eager"
        return cfg

    @staticmethod
    def _build_experts_instance(module_path: str, class_name: str):
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # pragma: no cover - optional dependency/environment-specific
            pytest.skip(f"Cannot import {module_path}: {exc!r}")

        experts_cls = getattr(module, class_name)
        errors = []

        for cfg in TestQuarkExpertsBroadEquivalence._iter_candidate_configs(module):
            cfg = TestQuarkExpertsBroadEquivalence._prepare_config_for_experts(cfg, module_path, class_name)
            try:
                experts = experts_cls(cfg).eval()
                return experts, cfg
            except Exception as exc:
                errors.append(f"{type(cfg).__name__}: {exc!r}")

        raise RuntimeError(
            f"Could not instantiate {module_path}:{class_name} with any config candidate.\n" + "\n".join(errors)
        )

    @staticmethod
    def _infer_input_dim(experts_module: torch.nn.Module) -> int:
        has_gate = getattr(experts_module, "has_gate", hasattr(experts_module, "gate_up_proj"))
        is_transposed = getattr(experts_module, "is_transposed", False)

        if has_gate:
            weight = experts_module.gate_up_proj[0]
        else:
            weight = experts_module.up_proj[0]
        if is_transposed:
            return int(weight.shape[0])
        return int(weight.shape[-1])

    def test_discovered_all_experts_classes(self):
        discovered = self._find_decorated_experts_classes()
        assert discovered, "No experts classes decorated with use_experts_implementation were discovered"

    @torch.no_grad()
    def test_quark_experts_matches_original_eager(self):
        discovered = self._find_decorated_experts_classes()
        assert discovered, "No experts classes decorated with use_experts_implementation were discovered"

        for module_path, class_name in discovered:
            torch.manual_seed(abs(hash((module_path, class_name))) % (2**31))
            experts, config = self._build_experts_instance(module_path, class_name)
            for param in experts.parameters():
                param.copy_(torch.randn_like(param))
            quark = QuarkExperts(experts).eval()

            num_experts = int(experts.num_experts)
            input_dim = self._infer_input_dim(experts)
            top_k_raw = getattr(config, "num_experts_per_tok", None)
            if top_k_raw is None:
                top_k_raw = getattr(config, "num_experts_per_token", None)
            if top_k_raw is None:
                top_k_raw = 2
            top_k = max(1, min(int(top_k_raw), num_experts))
            num_tokens = 11

            hidden_states = torch.randn(num_tokens, input_dim, dtype=torch.float32)
            top_k_index = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.long)
            top_k_weights = torch.rand(num_tokens, top_k, dtype=torch.float32)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

            expected = experts(hidden_states, top_k_index, top_k_weights)
            actual = quark(hidden_states, top_k_index, top_k_weights)

            assert expected.shape == actual.shape
            assert torch.allclose(expected, actual, rtol=1e-5, atol=1e-5), (
                f"Mismatch for {module_path}:{class_name} max_abs_diff={(expected - actual).abs().max().item():.6e}"
            )
