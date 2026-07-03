#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Test that preprocess_for_quantization works correctly."""

import os

import pytest
import torch.nn as nn

from quark.common.utils.import_utils import (
    is_transformers_available,
    is_transformers_version_higher_or_equal,
)
from quark.torch.utils.llm import preprocess_for_quantization

if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    from quark.torch.utils.llm.module_replacement import QuarkExperts, QuarkQwen3MoeTopKRouter
from quark.torch.utils.llm.module_replacement.preprocess_registry import PREPROCESS_REGISTRY


@pytest.mark.parametrize("model_type", ["qwen3_5"])
def test_legacy_prepare_for_moe_quant_rejects_unsupported_model_type(model_type):
    import quark.torch.utils.llm.model_preparation as model_preparation

    class DummyConfig:
        pass

    cfg = DummyConfig()
    cfg.model_type = model_type

    model = nn.Linear(2, 2)
    model.config = cfg

    with pytest.raises(ValueError, match="not yet supported"):
        model_preparation._legacy_prepare_for_moe_quant(model)


def test_prepare_for_moe_quant_logs_deprecation_and_uses_legacy_path(monkeypatch):
    import quark.torch.utils.llm.model_preparation as model_preparation

    warning_messages = []
    legacy_call_args = {}

    monkeypatch.setattr(
        model_preparation,
        "is_transformers_version_higher_or_equal",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        model_preparation,
        "_legacy_prepare_for_moe_quant",
        lambda model, reload=False: legacy_call_args.update({"model": model, "reload": reload}),
    )
    monkeypatch.setattr(
        model_preparation.logger,
        "warning",
        lambda msg, *args, **kwargs: warning_messages.append(msg),
    )

    dummy_model = nn.Linear(2, 2)
    model_preparation.prepare_for_moe_quant(dummy_model, reload=True)

    assert legacy_call_args == {"model": dummy_model, "reload": True}
    assert len(warning_messages) == 1
    assert "deprecated" in warning_messages[0]
    assert "preprocess_for_quantization" in warning_messages[0]


@pytest.mark.skipif(
    not is_transformers_available() or not is_transformers_version_higher_or_equal("5.0.0"),
    reason="transformers >= 5.0.0 required for Qwen3Moe",
)
class TestPreprocessForQuantization:
    """Test that preprocess_for_quantization works correctly."""

    def test_qwen3_moe(self):
        from transformers import AutoConfig, AutoModelForCausalLM

        config_path = os.path.join(os.path.dirname(__file__), "configs", "moe_model", "qwen3")
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config=config)
        preprocess_for_quantization(model)

        found_module_replacement = False
        for _, module in model.named_modules():
            if isinstance(module, QuarkQwen3MoeTopKRouter | QuarkExperts):
                found_module_replacement = True
                break
        assert found_module_replacement, "Qwen3MoeTopKRouter or Qwen3MoeExperts not found in model"

    def test_moe_quant_skips_when_model_has_no_config(self):
        from transformers import Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

        class ModelWithoutConfig(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = Qwen3MoeTopKRouter(Qwen3MoeConfig(hidden_size=8, num_experts=4, num_experts_per_tok=2))

        model = ModelWithoutConfig()
        preprocess_for_quantization(model)
        assert isinstance(model.router, Qwen3MoeTopKRouter)

    def test_moe_quant_raises_when_registered_replacement_fails(self):
        class DummyConfig:
            model_type = "qwen3_moe"

        class DummyHFModule(nn.Module):
            pass

        class DummyContainerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = DummyConfig()
                self.block = DummyHFModule()

        class BrokenReplacement(nn.Module):
            @classmethod
            def from_hf(cls, module, reload: bool = False):  # noqa: ARG003
                raise RuntimeError("intentional preprocess failure")

        model = DummyContainerModel()
        PREPROCESS_REGISTRY[DummyHFModule] = BrokenReplacement
        try:
            with pytest.raises(RuntimeError, match="intentional preprocess failure"):
                preprocess_for_quantization(model)
        finally:
            PREPROCESS_REGISTRY.pop(DummyHFModule, None)

    def test_preprocess_uses_legacy_path_when_transformers_lt5(self, monkeypatch):
        import quark.torch.utils.llm.model_preparation as model_preparation

        call_args = {}

        def fake_legacy(model, reload: bool = False):
            call_args["model"] = model
            call_args["reload"] = reload

        monkeypatch.setattr(model_preparation, "is_transformers_available", lambda: True)
        monkeypatch.setattr(
            model_preparation, "is_transformers_version_higher_or_equal", lambda *_args, **_kwargs: False
        )
        monkeypatch.setattr(model_preparation, "_legacy_prepare_for_moe_quant", fake_legacy)
        monkeypatch.setattr(
            model_preparation,
            "_prepare_for_moe_quant",
            lambda *_args, **_kwargs: pytest.fail("new preprocess path should not be used"),
        )

        dummy_model = nn.Linear(2, 2)
        preprocess_for_quantization(dummy_model, reload=True)
        assert call_args == {"model": dummy_model, "reload": True}
