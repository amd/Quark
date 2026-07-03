#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from quark.torch.utils.diffusers.calibration import (
    _InputCapture,
    _passthrough_collate,
    get_calib_dataloader,
)


class _TinyModule(nn.Module):
    """Minimal module that accepts (x, scale=1.0) for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.some_attribute = "hello"

    def forward(self, x, scale=1.0):
        return self.linear(x) * scale


class _ModuleWithDictKwargs(nn.Module):
    """Stand-in for diffusion submodules that accept dict-valued kwargs
    (e.g. ``cross_attention_kwargs`` on UNet2DConditionModel)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x, attention_kwargs=None):
        return self.linear(x)


def _make_mock_pipeline(target_module: nn.Module) -> MagicMock:
    """Create a mock pipeline that calls *target_module* on each invocation."""
    pipe = MagicMock()

    def side_effect(*args, **kwargs):
        x = torch.randn(1, 4)
        target_module(x, scale=1.0)

    pipe.side_effect = side_effect
    return pipe


def test_dataloader_returns_iterable():
    target = _TinyModule()
    pipe = _make_mock_pipeline(target)

    dataloader = get_calib_dataloader(pipe, target, prompts=["test"], n_steps=1, device="cpu")

    assert hasattr(dataloader, "__iter__")
    items = list(dataloader)
    assert len(items) > 0


def test_dataloader_capture_count():
    target = _TinyModule()
    pipe = _make_mock_pipeline(target)

    dataloader = get_calib_dataloader(pipe, target, prompts=["p1", "p2"], n_steps=1, device="cpu")

    items = list(dataloader)
    assert len(items) == 2


def test_dataloader_yields_dict_keyed_by_param_name():
    """Capture maps args to forward parameter names."""
    target = _TinyModule()
    pipe = _make_mock_pipeline(target)
    dataloader = get_calib_dataloader(pipe, target, prompts=["test"], n_steps=1, device="cpu")
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    assert "x" in data and "scale" in data


def test_dataloader_binds_positional_args_to_param_names():
    """bind_partial maps positional args to the forward signature's parameter names."""
    target = _TinyModule()
    pipe = MagicMock()

    def side_effect(*args, **kwargs):
        target(torch.randn(1, 4), 2.0)  # positional scale

    pipe.side_effect = side_effect
    dataloader = get_calib_dataloader(pipe, target, prompts=["test"], n_steps=1, device="cpu")
    data = next(iter(dataloader))
    assert data["scale"] == 2.0
    assert torch.is_tensor(data["x"])


def test_target_module_callable_with_double_star_unpack():
    """ModelQuantizer._do_calibration uses model(**data); the dict must unpack cleanly."""
    target = _TinyModule()
    pipe = _make_mock_pipeline(target)
    dataloader = get_calib_dataloader(pipe, target, prompts=["test"], n_steps=1, device="cpu")
    data = next(iter(dataloader))
    out = target(**data)
    assert torch.is_tensor(out)


def test_dataloader_forwards_pipe_kwargs():
    target = _TinyModule()
    pipe = MagicMock()

    def side_effect(*args, **kwargs):
        target(torch.randn(1, 4))

    pipe.side_effect = side_effect

    get_calib_dataloader(
        pipe,
        target,
        prompts=["test"],
        n_steps=5,
        device="cpu",
        guidance_scale=8.0,
        height=512,
    )

    call_kwargs = pipe.call_args.kwargs
    assert call_kwargs["guidance_scale"] == 8.0
    assert call_kwargs["height"] == 512


def test_dataloader_raises_on_pipeline_error():
    target = _TinyModule()
    pipe = MagicMock(side_effect=RuntimeError("pipeline error"))

    with pytest.raises(RuntimeError, match="pipeline error"):
        get_calib_dataloader(pipe, target, prompts=["test"], n_steps=1, device="cpu")


def test_dataloader_raises_when_target_never_called():
    target = _TinyModule()
    pipe = MagicMock()  # default return value; does not call target

    with pytest.raises(RuntimeError, match="No calibration samples"):
        get_calib_dataloader(pipe, target, prompts=["test"], n_steps=1, device="cpu")


def test_dataloader_tensors_on_cpu():
    target = _TinyModule()
    pipe = _make_mock_pipeline(target)

    dataloader = get_calib_dataloader(pipe, target, prompts=["test"], n_steps=1, device="cpu")

    for data in dataloader:
        for v in data.values():
            if torch.is_tensor(v):
                assert v.device == torch.device("cpu")


def test_passthrough_collate_rejects_wrong_batch_size():
    with pytest.raises(ValueError, match="Expected batch_size=1"):
        _passthrough_collate([1, 2])


def test_input_capture_remove_is_idempotent():
    capturer = _InputCapture()
    capturer.register(_TinyModule())
    capturer.remove()
    capturer.remove()


def test_register_rejects_var_positional():
    class _VarArgsModule(nn.Module):
        def forward(self, *args):
            return None

    with pytest.raises(RuntimeError, match="VAR_POSITIONAL"):
        _InputCapture().register(_VarArgsModule())


def test_register_rejects_var_keyword():
    class _VarKwargsModule(nn.Module):
        def forward(self, x, **kwargs):
            return x

    with pytest.raises(RuntimeError, match="VAR_KEYWORD"):
        _InputCapture().register(_VarKwargsModule())


def test_capture_handles_dict_valued_kwargs():
    target = _ModuleWithDictKwargs()
    pipe = MagicMock()

    def side_effect(*args, **kwargs):
        target(
            torch.randn(1, 4),
            attention_kwargs={"scale_tensor": torch.randn(2), "scalar": 0.5},
        )

    pipe.side_effect = side_effect

    dataloader = get_calib_dataloader(pipe, target, prompts=["test"], n_steps=1, device="cpu")
    items = list(dataloader)
    assert len(items) == 1

    data = items[0]
    assert "attention_kwargs" in data
    nested = data["attention_kwargs"]
    assert isinstance(nested, dict)
    assert torch.is_tensor(nested["scale_tensor"])
    assert nested["scale_tensor"].device == torch.device("cpu")
    assert nested["scalar"] == 0.5
