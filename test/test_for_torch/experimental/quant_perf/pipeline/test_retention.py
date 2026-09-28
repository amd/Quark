#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from quark.experimental.torch.quant_perf.pipeline import retention
from quark.experimental.torch.quant_perf.session.spec import StageError


def test_restore_repo_revision_fails_closed_when_rebuild_fails(monkeypatch):
    reset = MagicMock()
    monkeypatch.setattr(retention, "reset_hard_to", reset)
    monkeypatch.setattr(
        retention,
        "rebuild_framework",
        MagicMock(return_value=(False, "compiler failed")),
    )

    with pytest.raises(StageError, match="rollback rebuild failed: compiler failed"):
        retention._restore_repo_revision(
            "/repo",
            "base-sha",
            requires_rebuild=True,
        )

    reset.assert_called_once_with("/repo", "base-sha")
