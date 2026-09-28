#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from unittest.mock import MagicMock

from quark.experimental.torch.quant_perf.orchestration import backend_probe
from quark.experimental.torch.quant_perf.session.spec import Spec


def _spec(session_dir: str) -> Spec:
    return Spec(
        model_dir="/model",
        base_model="/model",
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="gfx950",
        isl=128,
        osl=32,
        quant_strategy="mxfp4",
        model_arch="unsupported",
        session_dir=session_dir,
    )


def test_run_backend_probe_persists_unsupported_result_without_workspaces(
    monkeypatch,
    tmp_path,
):
    session_dir = tmp_path / "session"
    quant_dir = session_dir / "quant"
    quant_dir.mkdir(parents=True)
    (quant_dir / "config.json").write_text("{}")
    spec = _spec(str(session_dir))
    checkpoint = MagicMock()
    checkpoint.state = {
        "quant_ckpt_dir": str(quant_dir),
        "run_spec": spec.to_dict(),
        "runtime_context": {},
    }
    orchestrator = MagicMock()

    monkeypatch.setattr(backend_probe.Checkpoint, "load", lambda _: checkpoint)
    monkeypatch.setattr(backend_probe, "Orchestrator", lambda: orchestrator)
    monkeypatch.setattr(
        backend_probe,
        "_probe_supported",
        lambda *_: (False, "unsupported_architecture"),
    )

    result = backend_probe.run_backend_probe(session_dir)

    assert result["status"] == "unsupported"
    assert result["reason"] == "unsupported_architecture"
    assert result["selected_backend"] == "triton"
    assert result["artifact"]
    orchestrator._prepare_managed_workspaces.assert_not_called()
    checkpoint.save.assert_called_once()
