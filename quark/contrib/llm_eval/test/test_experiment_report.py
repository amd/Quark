#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import argparse

from quark.contrib.llm_eval.experiment_report import build_experiment_dict


def test_build_experiment_dict_timestamp_uses_utc() -> None:
    """build_experiment_dict must produce an ISO-8601 UTC timestamp ending with 'Z'.

    This exercises the datetime.UTC import (line 10) and the datetime.now(UTC) call
    (line 105) introduced when the ruff target-version was bumped from py310 to py311.
    """
    args = argparse.Namespace(
        model_dir="/models/my-model",
        quant_scheme="w_int4_per_group",
        quant_algo="awq",
        dataset="pileval",
        num_calib_data=512,
        seq_len=512,
        batch_size=1,
        model_export="hf_format",
    )
    result = build_experiment_dict(args)
    timestamp = result["timestamp"]
    assert isinstance(timestamp, str)
    assert timestamp.endswith("Z"), f"expected UTC timestamp ending with 'Z', got {timestamp!r}"
    # Basic ISO-8601 shape: YYYY-MM-DDTHH:MM:SSZ
    assert len(timestamp) == 20, f"unexpected timestamp length: {timestamp!r}"
