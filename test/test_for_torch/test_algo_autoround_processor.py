#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

# Real-model GPU integration tests below build a small LlamaForCausalLM and run the full
# AutoRoundProcessor blockwise-tuning loop -- correctness-wise cheap on a tiny model, but the
# first CUDA/ROCm op in a test process pays a one-time context-init + kernel-selection cost
# (10s+) regardless of which test triggers it. Gated behind QUARK_RUN_SLOW, matching
# test_smoke_autoround.py's convention, so a default `pytest` run stays fast; set
# QUARK_RUN_SLOW=1 on a GPU host to run them.
SLOW = os.environ.get("QUARK_RUN_SLOW") == "1"


class TwoLinearBlock(nn.Module):
    """A tiny decoder-style block: two linears in sequence, tuple-returning forward."""

    def __init__(self, dim=64):
        super().__init__()
        self.a = nn.Linear(dim, dim, bias=False)
        self.b = nn.Linear(dim, dim, bias=False)

    def forward(self, x, **kw):
        return self.a(self.b(x))


def _build_wrapped_block(dim=64, n_batches=4, enable_minmax_tuning=False, seed=0):
    """Fresh TwoLinearBlock with TEST-ONLY int4 fake-quant attached and wrapped, plus the
    full-precision reconstruction target -- shared setup for the signed-SGD unit tests below."""
    from quark.experimental.torch.autoround.wrapper import WrapperLinear, attach_int4_fakequant

    torch.manual_seed(seed)
    block = TwoLinearBlock(dim)
    fp_block = copy.deepcopy(block)
    xs = [torch.randn(8, dim) for _ in range(n_batches)]
    with torch.no_grad():
        target = [fp_block(x) for x in xs]

    attach_int4_fakequant(block.a, group_size=32, bits=4)
    attach_int4_fakequant(block.b, group_size=32, bits=4)
    wrappers = {
        "a": WrapperLinear(block.a, enable_minmax_tuning=enable_minmax_tuning),
        "b": WrapperLinear(block.b, enable_minmax_tuning=enable_minmax_tuning),
    }
    block.a = wrappers["a"]
    block.b = wrappers["b"]
    return block, wrappers, xs, target


class TestAutoRoundOptimizer(unittest.TestCase):
    """Fast unit test of the signed-SGD inner loop, no real model / no init_blockwise_algo."""

    def test_signed_sgd_reduces_block_reconstruction_mse(self):
        from quark.experimental.torch.autoround.autoround import optimize_wrappers_signed_sgd

        block, wrappers, xs, target = _build_wrapped_block()

        def eval_mse():
            with torch.no_grad():
                losses = [torch.nn.functional.mse_loss(block(x), t).item() for x, t in zip(xs, target, strict=False)]
            return sum(losses) / len(losses)

        mse_v0 = eval_mse()  # V == 0 baseline
        optimize_wrappers_signed_sgd(block, wrappers, {}, xs, target, torch.device("cpu"), iters=30, lr=1.0 / 30)
        mse_after = eval_mse()

        # Best-V tracking must still beat V=0 despite linear lr decay shrinking late steps.
        self.assertLess(mse_after, mse_v0)

    def test_linear_lr_decay_applied_to_both_param_groups(self):
        """Linear lr decay (paper §4.1) must apply to BOTH param groups (V and clip):
        lr starts at the full value and decays linearly toward 0 over `iters` steps."""
        from quark.experimental.torch.autoround.autoround import optimize_wrappers_signed_sgd

        # enable_minmax_tuning => a SECOND param group (clip) exists.
        block, wrappers, xs, target = _build_wrapped_block(enable_minmax_tuning=True)

        # Record the per-group lrs used at each optimization step (scheduler.step() runs
        # AFTER opt.step(), so the lr seen at opt.step() is the step's actual lr).
        recorded: list[list[float]] = []
        orig_step = torch.optim.SGD.step

        def recording_step(self, *a, **k):
            # lr is a torch.Tensor (matches the official auto-round repo's `torch.tensor(lr)`,
            # see autoround.py) -- .item() to a plain float for comparison below.
            recorded.append([g["lr"].item() if torch.is_tensor(g["lr"]) else g["lr"] for g in self.param_groups])
            return orig_step(self, *a, **k)

        iters = 10
        lr = 1.0 / iters
        torch.optim.SGD.step = recording_step
        try:
            optimize_wrappers_signed_sgd(
                block,
                wrappers,
                {},
                xs,
                target,
                torch.device("cpu"),
                iters=iters,
                lr=lr,
                minmax_lr=lr,
            )
        finally:
            torch.optim.SGD.step = orig_step

        self.assertEqual(len(recorded), iters)
        # Two param groups every step (V + clip).
        self.assertTrue(all(len(groups) == 2 for groups in recorded))
        # First step at full lr; both groups equal.
        self.assertAlmostEqual(recorded[0][0], lr, places=6)
        self.assertAlmostEqual(recorded[0][1], lr, places=6)
        # Monotone non-increasing across steps, for both groups.
        for gi in (0, 1):
            seq = [r[gi] for r in recorded]
            self.assertTrue(all(seq[i + 1] <= seq[i] + 1e-12 for i in range(len(seq) - 1)))
        # Last step lr strictly below the first (decay actually happened), for both groups.
        self.assertLess(recorded[-1][0], recorded[0][0])
        self.assertLess(recorded[-1][1], recorded[0][1])

    def test_amp_autocast_disabled_on_cpu(self):
        """AMP is CUDA-only: on CPU, autocast must be entered with enabled=False so the
        fast CPU unit test path is byte-for-byte unaffected."""
        from quark.experimental.torch.autoround import autoround as ar_mod
        from quark.experimental.torch.autoround.autoround import optimize_wrappers_signed_sgd

        block, wrappers, xs, target = _build_wrapped_block()

        seen: list[tuple[str, bool]] = []
        orig_autocast = torch.autocast

        def recording_autocast(device_type, *a, **k):
            seen.append((device_type, k.get("enabled", True)))
            return orig_autocast(device_type, *a, **k)

        ar_mod.torch.autocast = recording_autocast
        try:
            optimize_wrappers_signed_sgd(block, wrappers, {}, xs, target, torch.device("cpu"), iters=3, lr=1.0 / 3)
        finally:
            ar_mod.torch.autocast = orig_autocast

        self.assertTrue(len(seen) > 0, "autocast context was never entered")
        self.assertTrue(all(enabled is False for _, enabled in seen), "autocast must be disabled on CPU")

    def test_reconstruction_loss_computed_in_fp32(self):
        """Regression test: the reconstruction MSE must be computed on fp32 tensors, matching
        the official auto-round repo's quantizer.py (which explicitly casts pred/ref to float32
        before mse_loss even though the forward ran under autocast). An earlier version computed
        F.mse_loss directly on the (possibly bf16-autocast) forward output -- sign-SGD is
        sensitive to sign flips from rounding noise, so this precision gap is a real fidelity
        issue, not just a style preference."""
        import torch.nn.functional as F

        from quark.experimental.torch.autoround.autoround import optimize_wrappers_signed_sgd

        block, wrappers, xs, target = _build_wrapped_block()

        seen_dtypes: list[tuple[torch.dtype, torch.dtype]] = []
        orig_mse_loss = F.mse_loss

        def recording_mse_loss(pred, tgt, *a, **k):
            seen_dtypes.append((pred.dtype, tgt.dtype))
            return orig_mse_loss(pred, tgt, *a, **k)

        import quark.experimental.torch.autoround.autoround as ar_mod

        ar_mod.F.mse_loss = recording_mse_loss
        try:
            optimize_wrappers_signed_sgd(block, wrappers, {}, xs, target, torch.device("cpu"), iters=2, lr=1.0 / 2)
        finally:
            ar_mod.F.mse_loss = orig_mse_loss

        self.assertTrue(len(seen_dtypes) > 0, "mse_loss was never called")
        self.assertTrue(
            all(pred_dtype == torch.float32 and tgt_dtype == torch.float32 for pred_dtype, tgt_dtype in seen_dtypes),
            f"mse_loss must receive fp32 tensors, got {seen_dtypes}",
        )


class TestAutoRoundCacheConfigCompatibility(unittest.TestCase):
    """AutoRound must support both legacy flat and newer nested Transformers configs."""

    @staticmethod
    def _config(use_cache: bool, nested: bool) -> SimpleNamespace:
        if nested:
            return SimpleNamespace(text_config=SimpleNamespace(use_cache=use_cache))
        return SimpleNamespace(use_cache=use_cache)

    @staticmethod
    def _get_use_cache(config: SimpleNamespace, nested: bool) -> bool:
        return config.text_config.use_cache if nested else config.use_cache

    def _assert_cache_config_supported(self, nested: bool) -> None:
        from quark.experimental.torch.autoround.autoround import AutoRoundProcessor

        model_ref = SimpleNamespace(config=self._config(use_cache=True, nested=nested))
        fp_model_ref = SimpleNamespace(config=self._config(use_cache=False, nested=nested))

        processor = AutoRoundProcessor.__new__(AutoRoundProcessor)
        processor.model = model_ref
        processor.fp_model = fp_model_ref
        processor.inps = []
        processor.modules = []
        processor.modules_fp = []

        observed_cache_values = []

        def record_cache_values():
            observed_cache_values.append(
                (
                    self._get_use_cache(model_ref.config, nested),
                    self._get_use_cache(fp_model_ref.config, nested),
                )
            )

        with patch(
            "quark.experimental.torch.autoround.autoround.clear_memory",
            side_effect=record_cache_values,
        ):
            processor._apply()

        self.assertEqual(observed_cache_values[0], (False, False))
        self.assertEqual(self._get_use_cache(model_ref.config, nested), True)
        self.assertEqual(self._get_use_cache(fp_model_ref.config, nested), False)

    def test_apply_disables_and_restores_cache_for_flat_and_nested_configs(self):
        for nested in (False, True):
            with self.subTest(nested=nested):
                self._assert_cache_config_supported(nested)


def _build_int4_quant_config(group_size=64):
    from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
    from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
    from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver

    weight_spec = QTensorConfig(
        dtype=Dtype.uint4,
        observer_cls=PerGroupMinMaxObserver,
        symmetric=False,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        qscheme=QSchemeType.per_group,
        ch_axis=1,
        is_dynamic=False,
        group_size=group_size,
    )
    return QConfig(global_quant_config=QLayerConfig(weight=weight_spec), exclude=["lm_head", "*lm_head"])


def _build_mxfp4_quant_config():
    from quark.torch.quantization.config.config import OCP_MXFP4Spec, QConfig, QLayerConfig

    weight_spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
    return QConfig(global_quant_config=QLayerConfig(weight=weight_spec), exclude=["lm_head", "*lm_head"])


_REAL_MODEL_INSIDE_LAYER_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


@unittest.skipUnless(SLOW and torch.cuda.is_available(), "set QUARK_RUN_SLOW=1 on a GPU host")
class TestAutoRoundRealLlama(unittest.TestCase):
    """Integration test of the REAL path: real weight_quantizer + init_blockwise_algo, run once
    each for INT4 and MXFP4 (hidden_size/intermediate_size divide evenly by both group sizes).
    Verifies (1) V/minmax tuning actually changes the baked weight vs the pure-round (V=0)
    baseline, and (2) the KEY consistency property -- with enable_minmax_tuning, the scale/
    zero_point written back into weight_quantizer at freeze makes the frozen linear's normal
    forward reproduce the tuned wrapper output on a fixed probe input, with the scale itself
    having actually changed."""

    def _run(self, quant_config, expected_dtype_str=None):
        from transformers import LlamaConfig, LlamaForCausalLM

        from quark.experimental.torch.autoround.autoround import AutoRoundProcessor
        from quark.torch import ModelQuantizer
        from quark.torch.quantization.config.config import AutoRoundConfig

        torch.manual_seed(0)
        cfg = LlamaConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=256,
            max_position_embeddings=128,
        )
        device = torch.device("cuda")
        model = LlamaForCausalLM(cfg).to(device).eval()
        fp_model = copy.deepcopy(model)

        model = ModelQuantizer(quant_config).quantize_model(model, None)
        model = model.to(device).eval()

        seq_len = 16
        dataloader = [torch.randint(0, cfg.vocab_size, (1, seq_len), device=device) for _ in range(3)]
        ar_cfg = AutoRoundConfig(
            iters=5,
            lr=1.0 / 5,
            enable_minmax_tuning=True,
            model_decoder_layers="model.layers",
            inside_layer_modules=_REAL_MODEL_INSIDE_LAYER_MODULES,
        )

        target_name = "self_attn.q_proj"
        q_proj = model.model.layers[0].self_attn.q_proj
        in_features = q_proj.weight.shape[1]
        wq = q_proj.weight_quantizer
        if expected_dtype_str is not None:
            self.assertEqual(str(wq.dtype), expected_dtype_str)
        with torch.no_grad():
            baseline_qweight = wq(q_proj.weight.detach().clone())
        # Fixed probe input for the linear, and pre-tuning scale snapshot.
        probe_x = torch.randn(4, in_features, device=device)
        scale_before = wq.scale.detach().clone()

        # Capture the tuned wrapper output on the probe input BEFORE freeze by intercepting
        # _freeze_block (which receives the block with the still-wrapped linears).
        captured: dict[str, torch.Tensor] = {}
        orig_freeze = AutoRoundProcessor._freeze_block

        def capturing_freeze(self, block, wrappers):  # type: ignore[no-untyped-def]
            wrapper = wrappers[target_name]
            with torch.no_grad():
                captured["tuned"] = wrapper(probe_x).detach().clone()
            orig_freeze(self, block, wrappers)

        AutoRoundProcessor._freeze_block = capturing_freeze
        try:
            proc = AutoRoundProcessor(fp_model, model, ar_cfg, dataloader)
            proc.apply()
        finally:
            AutoRoundProcessor._freeze_block = orig_freeze

        # Completed without error and produced a per-block MSE.
        self.assertIsNotNone(proc.last_block_mse)
        self.assertIn("tuned", captured)

        # (1) The baked weight should differ from the pure-round (V=0) baseline: V learned something.
        learned_weight = model.model.layers[0].self_attn.q_proj.weight.detach()
        self.assertEqual(learned_weight.shape, baseline_qweight.shape)
        self.assertFalse(
            torch.allclose(learned_weight, baseline_qweight.to(learned_weight.device), atol=1e-6),
            "baked weights identical to V=0 baseline — AutoRound learned nothing",
        )

        # (2) After freeze: the plain quantized linear is restored; its normal forward
        # re-quantizes weight via the written-back scale/zp. The processor may have moved
        # the block back to CPU, so run on the linear's actual device and compare on CPU.
        frozen_q_proj = model.model.layers[0].self_attn.q_proj
        frozen_device = frozen_q_proj.weight.device
        with torch.no_grad():
            frozen_out = frozen_q_proj(probe_x.to(frozen_device)).detach().cpu()

        self.assertTrue(
            torch.allclose(frozen_out, captured["tuned"].cpu(), atol=1e-5, rtol=1e-4),
            "frozen inference output does not reproduce tuned output — scale/zp writeback is wrong",
        )

        scale_after = frozen_q_proj.weight_quantizer.scale.detach().cpu()
        self.assertEqual(scale_after.shape, scale_before.shape)
        self.assertFalse(
            torch.allclose(scale_after, scale_before.cpu(), atol=1e-8),
            "weight_quantizer.scale unchanged — minmax tuning / writeback did not take effect",
        )

    def test_int4_learns_and_freeze_writeback_reproduces_tuned_output(self):
        self._run(_build_int4_quant_config())

    def test_mxfp4_learns_and_freeze_writeback_reproduces_tuned_output(self):
        self._run(_build_mxfp4_quant_config(), expected_dtype_str="Dtype.fp4")


class TestAutoRoundCheckpointRoundTrip(unittest.TestCase):
    """AutoRound-tuned weights/qparams must survive a real freeze -> export_safetensors ->
    import_model_from_safetensors round trip, not just the in-memory freeze writeback covered by
    TestAutoRoundRealLlama. Runs on CPU (no GPU/QUARK_RUN_SLOW gate) since it only needs a handful
    of tuning iters on a tiny model to exercise the pipeline, not to actually converge."""

    def test_int4_autoround_survives_export_import(self):
        import tempfile

        from transformers import LlamaConfig, LlamaForCausalLM

        from quark.torch import ModelQuantizer, export_safetensors, import_model_from_safetensors
        from quark.torch.algorithm.api import blockwise_tuning_algo
        from quark.torch.quantization.config.config import AutoRoundConfig
        from quark.torch.utils import getattr_recursive

        torch.manual_seed(0)
        cfg = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=128,
            max_position_embeddings=64,
        )
        model = LlamaForCausalLM(cfg).eval()
        fp_model = copy.deepcopy(model)

        model = ModelQuantizer(_build_int4_quant_config()).quantize_model(model, None)

        seq_len = 8
        dataloader = [torch.randint(0, cfg.vocab_size, (1, seq_len)) for _ in range(3)]
        ar_cfg = AutoRoundConfig(
            iters=3,
            lr=1.0 / 3,
            enable_minmax_tuning=True,
            model_decoder_layers="model.layers",
            inside_layer_modules=_REAL_MODEL_INSIDE_LAYER_MODULES,
        )
        model = blockwise_tuning_algo(fp_model, model, ar_cfg, is_accelerate=False, dataloader=dataloader)

        quantizer = ModelQuantizer(_build_int4_quant_config())
        quant_model = quantizer.freeze(model)
        state_dict_post_freeze = quant_model.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_safetensors(model=quant_model, output_dir=tmpdir, weight_format="real_quantized")
            original_model = LlamaForCausalLM(cfg)
            reloaded_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

        target_name = "model.layers.0.self_attn.q_proj"
        param_reload = reloaded_model.state_dict()[f"{target_name}.weight"]
        param_frozen = state_dict_post_freeze[f"{target_name}.weight"]

        reload_quantizer = getattr_recursive(reloaded_model, f"{target_name}.weight_quantizer")
        weight_dequantized = reload_quantizer(param_reload)

        self.assertEqual(weight_dequantized.dtype, param_frozen.dtype)
        self.assertLess(
            (weight_dequantized - param_frozen).abs().max().item(),
            5e-4,
            "AutoRound-tuned weight does not survive export_safetensors -> import_model_from_safetensors",
        )


if __name__ == "__main__":
    unittest.main()
