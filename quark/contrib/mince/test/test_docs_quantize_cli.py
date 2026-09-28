#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Contract tests for the ``quantize_quark.py`` invocation documented by MINCE.

``docs/example_quark_torch_mince_quantized.rst`` tells users to produce a
quantized checkpoint with a specific ``quantize_quark.py`` command line, then
score it on a frozen MINCE subset. That command is part of Quark's ``examples/``
tree, which MINCE does not own, so nothing otherwise stops it from drifting out
from under the documentation.

These tests assert only that the documented surface still exists -- the flags,
the quantization scheme, the calibration dataset, the export format, and the
algorithm. They deliberately do not quantize anything: Quark's tests cover that.
What matters for MINCE usage is whether the flags this
documentation tells users to type are still real.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Every flag used by the documented command. Keep in sync with the "Quantize"
# step of example_quark_torch_mince_quantized.rst.
DOCUMENTED_FLAGS = (
    "--model_dir",
    "--output_dir",
    "--quant_scheme",
    "--quant_algo",
    "--dataset",
    "--num_calib_data",
    "--seq_len",
    "--model_export",
)

# Argument values the documented command passes.
DOCUMENTED_SCHEME = "int4_wo_128"
DOCUMENTED_ALGO = "awq"
DOCUMENTED_DATASET = "pileval_for_awq_benchmark"
DOCUMENTED_EXPORT_FORMAT = "hf_format"

# The documented walkthrough calibrates on Llama-3.1-8B-Instruct, so the AWQ
# algorithm has to be registered for this transformers ``model_type``.
DOCUMENTED_MODEL_TYPE = "llama"

_SCRIPT_RELPATH = Path("examples") / "torch" / "language_modeling" / "llm_ptq" / "quantize_quark.py"


def _quantize_quark_script() -> Path:
    """Resolve ``quantize_quark.py`` in the source checkout, or skip.

    ``examples/`` is not shipped inside the wheel, so this skips rather than
    fails when the tests run against an installed package.
    """
    repo_root = Path(__file__).resolve().parents[4]
    script = repo_root / _SCRIPT_RELPATH
    if not script.is_file():
        pytest.skip(f"{_SCRIPT_RELPATH} not found under {repo_root}; examples/ absent from this install")
    return script


def test_documented_scheme_is_supported() -> None:
    """``--quant_scheme int4_wo_128`` must still resolve to a built-in scheme."""
    from quark.torch.quantization.config.template import LLMTemplate

    supported = LLMTemplate.get_supported_schemes()
    assert DOCUMENTED_SCHEME in supported, (
        f"MINCE docs quantize with --quant_scheme {DOCUMENTED_SCHEME}, which is no longer a "
        f"supported scheme. Supported: {sorted(supported)}"
    )


def test_documented_algo_is_registered_for_documented_model() -> None:
    """``--quant_algo awq`` must still be registered for the documented model family."""
    from quark.torch.quantization.config.algo_configs import ALGORITHM_CONFIG_MAPS

    assert DOCUMENTED_ALGO in ALGORITHM_CONFIG_MAPS, (
        f"MINCE docs quantize with --quant_algo {DOCUMENTED_ALGO}, which is no longer a "
        f"registered algorithm. Registered: {sorted(ALGORITHM_CONFIG_MAPS)}"
    )
    by_model_type = ALGORITHM_CONFIG_MAPS[DOCUMENTED_ALGO]
    assert DOCUMENTED_MODEL_TYPE in by_model_type, (
        f"MINCE docs apply {DOCUMENTED_ALGO} to a '{DOCUMENTED_MODEL_TYPE}' model, but that "
        f"model_type has no {DOCUMENTED_ALGO} config. Available: {sorted(by_model_type)}"
    )


def test_documented_cli_flags_and_choices_exist() -> None:
    """``quantize_quark.py --help`` must still advertise every documented flag and value.

    The parser is built inside the script's ``__main__`` block, so it cannot be
    imported; a subprocess is the only way to exercise the real argparse. This
    also catches the script failing to start at all, which would break the
    documented workflow just as surely as a renamed flag.
    """
    script = _quantize_quark_script()

    result = subprocess.run(
        [sys.executable, script.name, "--help"],
        cwd=script.parent,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"`{script.name} --help` exited {result.returncode}; the documented entry point "
        f"does not run.\nstderr:\n{result.stderr}"
    )

    help_text = result.stdout
    missing = [flag for flag in DOCUMENTED_FLAGS if flag not in help_text]
    assert not missing, f"Flags used by the MINCE docs are gone from {script.name}: {missing}"

    # argparse renders `choices` into the help output, so a removed choice shows up here.
    for value, flag in (
        (DOCUMENTED_EXPORT_FORMAT, "--model_export"),
        (DOCUMENTED_DATASET, "--dataset"),
        (DOCUMENTED_SCHEME, "--quant_scheme"),
    ):
        assert value in help_text, f"MINCE docs pass `{flag} {value}`, but '{value}' is no longer an accepted choice."
