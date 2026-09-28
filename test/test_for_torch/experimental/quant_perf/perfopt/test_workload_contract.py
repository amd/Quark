#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json

import pytest

from quark.experimental.torch.quant_perf.perfopt.workload_contract import WorkloadContract


def _contract() -> WorkloadContract:
    return WorkloadContract(
        spec_path="workload.json",
        schema="workload-v1",
        kernel_name="kernel",
        cases=(
            {
                "dims": [
                    [32, 2048],
                    [256, 1024, 1024],
                    [256, 2048, 256],
                ],
                "count": 440,
            },
        ),
        logical_semantics={},
        managed_sources=(),
    )


def _write_measurement(tmp_path, *, dims, count=440):
    (tmp_path / "baseline_timing.json").write_text(
        json.dumps(
            {
                "workload_aligned": True,
                "test_cases": [
                    {
                        "dims": dims,
                        "count": count,
                    }
                ],
            }
        )
    )


def test_alignment_accepts_kernel_operand_projection(tmp_path):
    _write_measurement(
        tmp_path,
        dims=[
            [32, 2048],
            [256, 2048, 256],
        ],
    )

    valid, _ = _contract().validate_alignment(str(tmp_path))

    assert valid is True


@pytest.mark.parametrize(
    ("dims", "count"),
    [
        ([[32, 2048], [1, 2, 3]], 440),
        ([[32, 2048], [256, 2048, 256]], 1),
    ],
)
def test_alignment_rejects_measurements_outside_trace_contract(
    tmp_path,
    dims,
    count,
):
    _write_measurement(tmp_path, dims=dims, count=count)

    valid, reason = _contract().validate_alignment(str(tmp_path))

    assert valid is False
    assert "dimensions/counts" in reason
    assert f"dims={dims}" in reason
    assert f"count={count}" in reason
    assert "expected=" in reason
    assert str(_contract().cases[0]["dims"]) in reason


@pytest.mark.parametrize("latency", [None, 0, -1, float("nan"), float("inf"), True, "invalid", 0.0066, 0.0049, 0.0225])
@pytest.mark.parametrize("workload_aligned", [True, False])
def test_trace_timing_requires_a_valid_matching_measurement(tmp_path, latency, workload_aligned):
    contract = _contract()
    contract.cases[0]["baseline_latency_ms"] = 0.0066
    _write_measurement(tmp_path, dims=contract.cases[0]["dims"])
    path = tmp_path / "baseline_timing.json"
    timing = json.loads(path.read_text())
    timing["workload_aligned"] = workload_aligned
    if latency is not None:
        timing["test_cases"][0]["latency_ms"] = latency
    path.write_text(json.dumps(timing))

    valid, reason = contract.validate_alignment(str(tmp_path))

    assert valid is (latency == 0.0066 and workload_aligned)
    if latency is None or latency in (0.0049, 0.0225):
        assert f"dims={contract.cases[0]['dims']}" in reason
        assert "count=440" in reason
        assert "trace=6.600000 us" in reason
    if latency in (0.0049, 0.0225):
        assert f"measured={latency * 1000:.6f} us" in reason
        assert f"deviation={(latency / 0.0066 - 1) * 100:+.4f}%" in reason
        assert "tolerance=+/-25%" in reason


@pytest.mark.parametrize("workload_aligned", [True, False])
def test_alignment_rejects_malformed_case_evidence(tmp_path, workload_aligned):
    (tmp_path / "baseline_timing.json").write_text(
        json.dumps({"workload_aligned": workload_aligned, "test_cases": [None]})
    )
    valid, reason = _contract().validate_alignment(str(tmp_path))
    assert valid is False
    assert "dimensions/counts" in reason
