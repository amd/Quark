#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Model-loading compatibility is local to mixed precision."""

from contextlib import nullcontext

import pytest

from quark.experimental.torch.mix_precision import model_loading


@pytest.mark.parametrize("load_fails", [False, True])
def test_moved_output_recorder_is_scoped_to_loading(monkeypatch, load_fails):
    import sys
    from types import ModuleType

    from transformers.utils import generic

    recorder = object()
    moved = ModuleType("transformers.utils.output_capturing")
    moved.OutputRecorder = recorder
    monkeypatch.setitem(sys.modules, moved.__name__, moved)
    monkeypatch.delattr(generic, "OutputRecorder", raising=False)

    expectation = pytest.raises(RuntimeError, match="remote model import failed") if load_fails else nullcontext()
    with expectation, model_loading._transformers_output_recorder_compatibility():
        assert generic.OutputRecorder is recorder
        if load_fails:
            raise RuntimeError("remote model import failed")
    assert not hasattr(generic, "OutputRecorder")


def test_existing_output_recorder_is_preserved(monkeypatch):
    from transformers.utils import generic

    recorder = object()
    monkeypatch.setattr(generic, "OutputRecorder", recorder, raising=False)
    with model_loading._transformers_output_recorder_compatibility():
        assert generic.OutputRecorder is recorder
    assert generic.OutputRecorder is recorder
