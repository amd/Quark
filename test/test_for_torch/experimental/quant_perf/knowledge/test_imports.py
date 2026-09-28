#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Verify startup imports without pytest's already-loaded package state."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["knowledge.store", "repair.service"])
def test_knowledge_and_repair_import_in_fresh_process(module):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib; importlib.import_module('quark.experimental.torch.quant_perf.{module}')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
