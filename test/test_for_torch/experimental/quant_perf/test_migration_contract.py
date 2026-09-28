#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Migration contract tests for the experimental quantization-performance pipeline."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from importlib.resources import files
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_quant_perf_package_is_available() -> None:
    assert importlib.util.find_spec("quark.experimental.torch.quant_perf") is not None


def test_quant_perf_responsibility_packages_are_available() -> None:
    modules = (
        "quark.experimental.torch.quant_perf.orchestration.orchestrator",
        "quark.experimental.torch.quant_perf.orchestration.intake",
        "quark.experimental.torch.quant_perf.orchestration.spec_factory",
        "quark.experimental.torch.quant_perf.session.spec",
        "quark.experimental.torch.quant_perf.session.progress",
        "quark.experimental.torch.quant_perf.runtime.backends",
        "quark.experimental.torch.quant_perf.runtime.inventory",
        "quark.experimental.torch.quant_perf.runtime.recovery",
        "quark.experimental.torch.quant_perf.reporting.service",
    )

    assert all(importlib.util.find_spec(module) is not None for module in modules)


def test_quant_perf_console_script_is_registered() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["quark-quant-perf"] == ("quark.experimental.torch.quant_perf.cli:main")


def test_quant_perf_runtime_resources_are_available() -> None:
    package = files("quark.experimental.torch.quant_perf")

    assert package.joinpath("README.md").is_file()
    assert package.joinpath("AGENTS.md").is_file()
    assert package.joinpath("CLAUDE.md").is_file()
    assert package.joinpath("agent-instructions/PROJECT_RULES.md").is_file()
    assert package.joinpath("evaluation/policies/gsm8k.json").is_file()
    assert package.joinpath("knowledge/records/quantization/quant-mxfp4-format.yaml").is_file()


def test_importing_quark_does_not_load_quant_perf() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import sys; import quark; assert 'quark.experimental.torch.quant_perf' not in sys.modules"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
