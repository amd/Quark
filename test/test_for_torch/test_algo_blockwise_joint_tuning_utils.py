#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from quark.torch.algorithm.blockwise_joint_tuning.utils import (
    _align_attention_mask_for_input,
    _build_layer_forward_kwargs,
    _move_nested_to_device,
    block_batch_forward,
    block_forward,
    blockwise_joint_training,
    enable_learnable_qparams_requires_grad,
    enable_weight_requires_grad,
    set_requires_grad,
)

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleLinear(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _LinearWithKwargs(nn.Module):
    """Accepts a fixed set of kwargs so unknown keys are filtered out."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        return self.linear(x) * scale


class _TupleOutputModule(nn.Module):
    """Returns (tensor, extra_value) to exercise tuple-unwrapping in block_batch_forward."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        return self.linear(x), 42


class _BadTupleOutputModule(nn.Module):
    """Returns (non-tensor, ...) to trigger the ValueError branch."""

    def forward(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:  # noqa: ARG002
        return 99, x


class _BadOutputModule(nn.Module):
    """Returns a plain int to trigger the other ValueError branch."""

    def forward(self, x: torch.Tensor) -> int:  # noqa: ARG002
        return 99


class _ModelWithScaleParam(nn.Module):
    """Mimics a quantized model that exposes scale/zero_point parameters."""

    def __init__(self):
        super().__init__()
        self.attn_weight = nn.Parameter(torch.randn(4, 4))
        self.attn_scale = nn.Parameter(torch.ones(4))
        self.attn_zero_point = nn.Parameter(torch.zeros(4))
        self.ffn_weight = nn.Parameter(torch.randn(4, 4))
        self.ffn_scale = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# set_requires_grad
# ---------------------------------------------------------------------------


class TestSetRequiresGrad:
    def test_disable_then_enable(self):
        model = _SimpleLinear()
        set_requires_grad(model, False)
        assert all(not p.requires_grad for p in model.parameters())

        set_requires_grad(model, True)
        assert all(p.requires_grad for p in model.parameters())

    def test_module_with_no_params(self):
        empty = nn.Sequential()
        set_requires_grad(empty, True)  # should not raise

    def test_only_affects_passed_module(self):
        m1 = _SimpleLinear()
        m2 = _SimpleLinear()
        set_requires_grad(m1, False)
        # m2 should be unaffected (default requires_grad=True)
        assert all(p.requires_grad for p in m2.parameters())


# ---------------------------------------------------------------------------
# enable_weight_requires_grad
# ---------------------------------------------------------------------------


class TestEnableWeightRequiresGrad:
    def test_returns_weight_params_only(self):
        model = _SimpleLinear()
        set_requires_grad(model, False)
        params, names = enable_weight_requires_grad(model, trainable_modules=[])
        assert len(params) == 1
        assert names[0] == "linear.weight"
        assert params[0].requires_grad is True

    def test_trainable_modules_filter_matches(self):
        model = _SimpleLinear()
        set_requires_grad(model, False)
        params, names = enable_weight_requires_grad(model, trainable_modules=["linear"])
        assert len(params) == 1
        assert "linear" in names[0]

    def test_trainable_modules_filter_no_match(self):
        model = _SimpleLinear()
        set_requires_grad(model, False)
        params, names = enable_weight_requires_grad(model, trainable_modules=["nonexistent"])
        assert len(params) == 0
        assert len(names) == 0

    def test_bias_not_included(self):
        model = nn.Linear(4, 4, bias=True)
        set_requires_grad(model, False)
        params, names = enable_weight_requires_grad(model, trainable_modules=[])
        # Only "weight", not "bias"
        assert all("weight" in n for n in names)
        assert not any("bias" in n for n in names)


# ---------------------------------------------------------------------------
# enable_learnable_qparams_requires_grad
# ---------------------------------------------------------------------------


class TestEnableLearnableQparamsRequiresGrad:
    def test_enables_scale_and_zero_point(self):
        model = _ModelWithScaleParam()
        for p in model.parameters():
            p.requires_grad = False
        params, names = enable_learnable_qparams_requires_grad(model, [], [])
        param_names_set = set(names)
        assert "attn_scale" in param_names_set
        assert "attn_zero_point" in param_names_set
        assert "ffn_scale" in param_names_set
        assert all(p.requires_grad for p in params)

    def test_trainable_modules_filter(self):
        model = _ModelWithScaleParam()
        for p in model.parameters():
            p.requires_grad = False
        params, names = enable_learnable_qparams_requires_grad(model, ["attn"], [])
        assert all("attn" in n for n in names)
        assert not any("ffn" in n for n in names)

    def test_quant_trainable_modules_filter(self):
        model = _ModelWithScaleParam()
        for p in model.parameters():
            p.requires_grad = False
        params, names = enable_learnable_qparams_requires_grad(model, [], ["zero_point"])
        assert all("zero_point" in n for n in names)

    def test_non_scale_params_excluded(self):
        model = _ModelWithScaleParam()
        for p in model.parameters():
            p.requires_grad = False
        _, names = enable_learnable_qparams_requires_grad(model, [], [])
        assert not any("weight" in n for n in names)


# ---------------------------------------------------------------------------
# _move_nested_to_device
# ---------------------------------------------------------------------------


class TestMoveNestedToDevice:
    def test_none(self):
        assert _move_nested_to_device(None, CPU) is None

    def test_tensor(self):
        t = torch.randn(2, 3)
        out = _move_nested_to_device(t, CPU)
        assert isinstance(out, torch.Tensor)
        assert out.device == t.device

    def test_tensor_values_preserved(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        out = _move_nested_to_device(t, CPU)
        assert torch.equal(out, t)

    def test_dict(self):
        d = {"a": torch.randn(2), "b": torch.randn(3)}
        out = _move_nested_to_device(d, CPU)
        assert set(out.keys()) == {"a", "b"}
        assert all(isinstance(v, torch.Tensor) for v in out.values())

    def test_dict_values_preserved(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        out = _move_nested_to_device({"a": a, "b": b}, CPU)
        assert torch.equal(out["a"], a)
        assert torch.equal(out["b"], b)

    def test_tuple(self):
        tup = (torch.randn(2), torch.randn(3))
        out = _move_nested_to_device(tup, CPU)
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_tuple_values_preserved(self):
        t0 = torch.tensor([7.0, 8.0])
        t1 = torch.tensor([9.0])
        out = _move_nested_to_device((t0, t1), CPU)
        assert torch.equal(out[0], t0)
        assert torch.equal(out[1], t1)

    def test_list(self):
        lst = [torch.randn(2), torch.randn(3)]
        out = _move_nested_to_device(lst, CPU)
        assert isinstance(out, list)
        assert len(out) == 2

    def test_list_values_preserved(self):
        t0 = torch.tensor([1.0, 2.0])
        out = _move_nested_to_device([t0], CPU)
        assert torch.equal(out[0], t0)

    def test_passthrough_non_tensor(self):
        val = 42
        assert _move_nested_to_device(val, CPU) == 42

    def test_nested_dict_in_list(self):
        nested = [{"k": torch.randn(2)}, torch.randn(3)]
        out = _move_nested_to_device(nested, CPU)
        assert isinstance(out[0], dict)
        assert isinstance(out[0]["k"], torch.Tensor)


# ---------------------------------------------------------------------------
# _align_attention_mask_for_input
# ---------------------------------------------------------------------------


class TestAlignAttentionMaskForInput:
    def test_none_passthrough(self):
        inp = torch.randn(2, 4)
        assert _align_attention_mask_for_input(None, inp) is None

    def test_non_tensor_passthrough(self):
        inp = torch.randn(2, 4)
        assert _align_attention_mask_for_input("mask", inp) == "mask"

    def test_batch_expansion(self):
        mask = torch.ones(1, 4)
        inp = torch.randn(3, 4)
        out = _align_attention_mask_for_input(mask, inp)
        assert out.shape[0] == 3

    def test_batch_expansion_values_match_original_row(self):
        row = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        inp = torch.randn(3, 4)
        out = _align_attention_mask_for_input(row, inp)
        # Every expanded row must equal the original row
        assert torch.equal(out[0], row[0])
        assert torch.equal(out[1], row[0])
        assert torch.equal(out[2], row[0])

    def test_batch_mismatch_clips_to_first(self):
        # Build a mask where each row is distinct so we can verify row[0] was picked
        mask = torch.stack(
            [
                torch.tensor([1.0, 1.0, 1.0, 1.0]),
                torch.tensor([2.0, 2.0, 2.0, 2.0]),
                torch.tensor([3.0, 3.0, 3.0, 3.0]),
                torch.tensor([4.0, 4.0, 4.0, 4.0]),
                torch.tensor([5.0, 5.0, 5.0, 5.0]),
            ]
        )
        inp = torch.randn(3, 4)
        out = _align_attention_mask_for_input(mask, inp)
        assert out.shape[0] == 3
        # All output rows should be copies of mask[0] (the first row)
        assert torch.all(out == 1.0)

    def test_dtype_alignment_for_float_mask(self):
        mask = torch.ones(2, 4, dtype=torch.float32)
        inp = torch.randn(2, 4, dtype=torch.float64)
        out = _align_attention_mask_for_input(mask, inp)
        assert out.dtype == torch.float64

    def test_integer_mask_dtype_unchanged(self):
        mask = torch.ones(2, 4, dtype=torch.int64)
        inp = torch.randn(2, 4, dtype=torch.float32)
        out = _align_attention_mask_for_input(mask, inp)
        assert out.dtype == torch.int64

    def test_same_batch_no_expansion(self):
        mask = torch.ones(2, 4)
        inp = torch.randn(2, 4)
        out = _align_attention_mask_for_input(mask, inp)
        assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# _build_layer_forward_kwargs
# ---------------------------------------------------------------------------


class TestBuildLayerForwardKwargs:
    def test_filters_unknown_kwargs(self):
        layer = _LinearWithKwargs()
        kwargs = {"scale": 2.0, "unknown_key": torch.ones(1)}
        inp = torch.randn(1, 4)
        out = _build_layer_forward_kwargs(layer, kwargs, inp, CPU)
        assert "scale" in out
        assert "unknown_key" not in out

    def test_kwarg_value_preserved(self):
        layer = _LinearWithKwargs()
        kwargs = {"scale": 3.14}
        inp = torch.randn(1, 4)
        out = _build_layer_forward_kwargs(layer, kwargs, inp, CPU)
        assert out["scale"] == pytest.approx(3.14)

    def test_var_kwargs_accepts_everything(self):
        # A layer whose forward accepts **kwargs should pass all keys through
        class _VarKwargsLayer(nn.Module):
            def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
                return x

        layer = _VarKwargsLayer()
        kwargs = {"anything": torch.ones(1), "foo": 42}
        inp = torch.randn(1, 4)
        out = _build_layer_forward_kwargs(layer, kwargs, inp, CPU)
        assert "anything" in out
        assert "foo" in out

    def test_past_key_value_set_to_none(self):
        class _PkvLayer(nn.Module):
            def forward(self, x: torch.Tensor, past_key_value=None) -> torch.Tensor:
                return x

        layer = _PkvLayer()
        kwargs = {"past_key_value": torch.ones(2)}
        inp = torch.randn(1, 4)
        out = _build_layer_forward_kwargs(layer, kwargs, inp, CPU)
        assert out["past_key_value"] is None

    def test_past_key_values_set_to_none(self):
        class _PkvsLayer(nn.Module):
            def forward(self, x: torch.Tensor, past_key_values=None) -> torch.Tensor:
                return x

        layer = _PkvsLayer()
        kwargs = {"past_key_values": [torch.ones(2)]}
        inp = torch.randn(1, 4)
        out = _build_layer_forward_kwargs(layer, kwargs, inp, CPU)
        assert out["past_key_values"] is None

    def test_attention_mask_aligned(self):
        class _AttnLayer(nn.Module):
            def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
                return x

        layer = _AttnLayer()
        mask = torch.ones(1, 4)
        inp = torch.randn(3, 4)
        kwargs = {"attention_mask": mask}
        out = _build_layer_forward_kwargs(layer, kwargs, inp, CPU)
        assert out["attention_mask"].shape[0] == 3


# ---------------------------------------------------------------------------
# block_batch_forward
# ---------------------------------------------------------------------------


class TestBlockBatchForward:
    def test_tensor_output(self):
        layer = _SimpleLinear()
        inp = torch.randn(2, 4)
        out = block_batch_forward(layer, {}, inp, CPU)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 4)

    def test_output_values_match_direct_forward(self):
        # Fix weight to identity matrix so output == input (no bias)
        layer = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(4))
        inp = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        out = block_batch_forward(layer, {}, inp, CPU)
        assert torch.allclose(out, inp)

    def test_tuple_output_unwrapped(self):
        layer = _TupleOutputModule()
        inp = torch.randn(2, 4)
        out = block_batch_forward(layer, {}, inp, CPU)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 4)

    def test_tuple_output_values_match_direct_forward(self):
        # Verify the first element of the tuple is returned, not the second
        layer = _TupleOutputModule()
        torch.manual_seed(0)
        inp = torch.randn(2, 4)
        out_bbf = block_batch_forward(layer, {}, inp, CPU)
        expected = layer(inp)[0]
        assert torch.equal(out_bbf, expected)

    def test_bad_tuple_raises(self):
        layer = _BadTupleOutputModule()
        inp = torch.randn(2, 4)
        with pytest.raises(ValueError, match="Unexpected layer output\\[0\\] type"):
            block_batch_forward(layer, {}, inp, CPU)

    def test_bad_output_raises(self):
        layer = _BadOutputModule()
        inp = torch.randn(2, 4)
        with pytest.raises(ValueError, match="Unexpected layer output type"):
            block_batch_forward(layer, {}, inp, CPU)


# ---------------------------------------------------------------------------
# block_forward
# ---------------------------------------------------------------------------


class TestBlockForward:
    def test_accumulates_outputs(self):
        layer = _SimpleLinear()
        inp_list = [torch.randn(2, 4), torch.randn(2, 4)]
        outputs: list[torch.Tensor] = []
        result = block_forward(
            layer=layer,
            module_kwargs={},
            num_batches=2,
            device=CPU,
            layer_inputs=inp_list,
            outputs_acc=outputs,
            cache_examples_on_gpu=False,
        )
        assert len(result) == 2
        assert all(isinstance(t, torch.Tensor) for t in result)

    def test_output_values_match_block_batch_forward(self):
        # Fix weight to identity so output == input
        layer = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(4))
        inp0 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        inp1 = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
        result = block_forward(
            layer=layer,
            module_kwargs={},
            num_batches=2,
            device=CPU,
            layer_inputs=[inp0, inp1],
            outputs_acc=[],
            cache_examples_on_gpu=False,
        )
        assert torch.allclose(result[0], inp0)
        assert torch.allclose(result[1], inp1)

    def test_output_on_cpu_when_not_caching_on_gpu(self):
        layer = _SimpleLinear()
        inp_list = [torch.randn(2, 4)]
        outputs: list[torch.Tensor] = []
        result = block_forward(
            layer=layer,
            module_kwargs={},
            num_batches=1,
            device=CPU,
            layer_inputs=inp_list,
            outputs_acc=outputs,
            cache_examples_on_gpu=False,
        )
        assert result[0].device == CPU

    def test_appends_to_existing_acc(self):
        layer = _SimpleLinear()
        existing = [torch.randn(2, 4)]
        new_inputs = [torch.randn(2, 4)]
        result = block_forward(
            layer=layer,
            module_kwargs={},
            num_batches=1,
            device=CPU,
            layer_inputs=new_inputs,
            outputs_acc=existing,
            cache_examples_on_gpu=False,
        )
        assert len(result) == 2

    def test_layer_inputs_must_not_be_plain_tensor(self):
        layer = _SimpleLinear()
        with pytest.raises(AssertionError):
            block_forward(
                layer=layer,
                module_kwargs={},
                num_batches=1,
                device=CPU,
                layer_inputs=torch.randn(2, 4),  # type: ignore[arg-type]
                outputs_acc=[],
                cache_examples_on_gpu=False,
            )


# ---------------------------------------------------------------------------
# blockwise_joint_training
# ---------------------------------------------------------------------------


class _TinyWeightLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


class _TinyQparamLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        # name contains "scale" so enable_learnable_qparams_requires_grad can pick it
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ (self.weight * self.scale)


class _DummyScaler:
    """CPU-friendly scaler used to avoid cuda/amp side effects in unit tests."""

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad: float | None = None,  # noqa: ARG002
        parameters: object = None,  # noqa: ARG002
        create_graph: bool = False,
        update_grad: bool = True,
        retain_graph: bool = False,
    ) -> None:
        loss.backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            optimizer.step()
        return None


class TestBlockwiseJointTraining:
    def test_updates_weight_when_weight_lr_positive(self):
        layer = _TinyWeightLayer()
        before = layer.weight.detach().clone()

        quant_inputs = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        # Different target to force non-zero loss/gradient
        fp_targets = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

        with (
            patch(
                "quark.torch.algorithm.blockwise_joint_tuning.utils.NativeScalerWithGradNormCount",
                return_value=_DummyScaler(),
            ),
            patch(
                "quark.torch.algorithm.blockwise_joint_tuning.utils.torch.amp.autocast",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            patch("quark.torch.algorithm.blockwise_joint_tuning.utils.clear_memory"),
        ):
            blockwise_joint_training(
                layer=layer,
                module_kwargs={},
                trainable_modules=[],
                quant_trainable_modules=[],
                layer_inputs=[quant_inputs],
                fp_layer_outputs=[fp_targets],
                quant_val_inps=[],
                fp_val_outputs=[],
                device=CPU,
                epochs=1,
                quant_lr=0.0,
                weight_lr=0.1,
                min_lr_factor=10.0,
                weight_decay=0.0,
                qparam_weight_decay=0.0,
                max_grad_norm=0.0,
                layer_index=0,
                loss_func=nn.MSELoss(),
            )

        assert not torch.allclose(layer.weight.detach(), before)

    def test_updates_qparam_scale_when_quant_lr_positive(self):
        layer = _TinyQparamLayer()
        before = layer.scale.detach().clone()

        quant_inputs = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        fp_targets = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

        with (
            patch(
                "quark.torch.algorithm.blockwise_joint_tuning.utils.NativeScalerWithGradNormCount",
                return_value=_DummyScaler(),
            ),
            patch(
                "quark.torch.algorithm.blockwise_joint_tuning.utils.torch.amp.autocast",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            patch("quark.torch.algorithm.blockwise_joint_tuning.utils.clear_memory"),
        ):
            blockwise_joint_training(
                layer=layer,
                module_kwargs={},
                trainable_modules=[],
                quant_trainable_modules=[],
                layer_inputs=[quant_inputs],
                fp_layer_outputs=[fp_targets],
                quant_val_inps=[],
                fp_val_outputs=[],
                device=CPU,
                epochs=1,
                quant_lr=0.1,
                weight_lr=0.0,
                min_lr_factor=10.0,
                weight_decay=0.0,
                qparam_weight_decay=0.0,
                max_grad_norm=0.0,
                layer_index=0,
                loss_func=nn.MSELoss(),
            )

        assert not torch.allclose(layer.scale.detach(), before)

    def test_raises_on_nan_loss(self):
        class _NaNLoss(nn.Module):
            def forward(self, fp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
                return torch.tensor(float("nan"), dtype=torch.float32)

        layer = _TinyWeightLayer()
        quant_inputs = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        fp_targets = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

        with (
            patch(
                "quark.torch.algorithm.blockwise_joint_tuning.utils.NativeScalerWithGradNormCount",
                return_value=_DummyScaler(),
            ),
            patch(
                "quark.torch.algorithm.blockwise_joint_tuning.utils.torch.amp.autocast",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            patch("quark.torch.algorithm.blockwise_joint_tuning.utils.clear_memory"),
            pytest.raises(RuntimeError, match="Loss is NaN/Inf"),
        ):
            blockwise_joint_training(
                layer=layer,
                module_kwargs={},
                trainable_modules=[],
                quant_trainable_modules=[],
                layer_inputs=[quant_inputs],
                fp_layer_outputs=[fp_targets],
                quant_val_inps=[],
                fp_val_outputs=[],
                device=CPU,
                epochs=1,
                quant_lr=0.1,
                weight_lr=0.1,
                min_lr_factor=10.0,
                weight_decay=0.0,
                qparam_weight_decay=0.0,
                max_grad_norm=0.0,
                layer_index=0,
                loss_func=_NaNLoss(),
            )

    def test_returns_early_when_no_trainable_params_match(self):
        layer = _TinyQparamLayer()
        before_weight = layer.weight.detach().clone()
        before_scale = layer.scale.detach().clone()

        quant_inputs = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        fp_targets = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

        with (
            patch(
                "quark.torch.algorithm.blockwise_joint_tuning.utils.NativeScalerWithGradNormCount",
                return_value=_DummyScaler(),
            ) as mock_scaler_cls,
            patch(
                "quark.torch.algorithm.blockwise_joint_tuning.utils.torch.amp.autocast",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            patch("quark.torch.algorithm.blockwise_joint_tuning.utils.clear_memory"),
        ):
            blockwise_joint_training(
                layer=layer,
                module_kwargs={},
                trainable_modules=["not_exist_keyword"],
                quant_trainable_modules=["also_not_exist"],
                layer_inputs=[quant_inputs],
                fp_layer_outputs=[fp_targets],
                quant_val_inps=[],
                fp_val_outputs=[],
                device=CPU,
                epochs=1,
                quant_lr=0.1,
                weight_lr=0.1,
                min_lr_factor=10.0,
                weight_decay=0.0,
                qparam_weight_decay=0.0,
                max_grad_norm=0.0,
                layer_index=0,
                loss_func=nn.MSELoss(),
            )

        # No matching params -> return before building optimizer/scaler
        mock_scaler_cls.assert_not_called()
        assert torch.allclose(layer.weight.detach(), before_weight)
        assert torch.allclose(layer.scale.detach(), before_scale)
        assert not layer.weight.requires_grad
        assert not layer.scale.requires_grad
