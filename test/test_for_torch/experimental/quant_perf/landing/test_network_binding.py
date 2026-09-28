#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Network binding tests for Quant-Perf landing adapters."""

from unittest.mock import MagicMock, patch

import pytest

from quark.experimental.torch.quant_perf.landing.atom_adapter import start_atom_server
from quark.experimental.torch.quant_perf.landing.base import ensure_port_available, wait_ready
from quark.experimental.torch.quant_perf.landing.vllm_adapter import start_vllm_server


@patch("quark.experimental.torch.quant_perf.landing.base.socket.socket")
def test_port_check_defaults_to_loopback(mock_socket):
    ensure_port_available(9000)

    mock_socket.return_value.bind.assert_called_once_with(("127.0.0.1", 9000))


@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
def test_vllm_server_uses_configured_host(mock_popen):
    mock_popen.return_value = MagicMock()

    start_vllm_server(
        model_dir="/quant_ckpt",
        tp=1,
        port=9000,
        kv_cache_scheme=None,
        host="10.0.0.8",
    )

    command = mock_popen.call_args.args[0]
    assert command[command.index("--host") + 1] == "10.0.0.8"


@patch("quark.experimental.torch.quant_perf.landing.atom_adapter.subprocess.Popen")
def test_atom_server_uses_configured_host(mock_popen):
    mock_popen.return_value = MagicMock()

    start_atom_server(
        model_dir="/quant_ckpt",
        tp=1,
        port=9000,
        kv_cache_scheme=None,
        host="10.0.0.8",
    )

    command = mock_popen.call_args.args[0]
    assert command[command.index("--host") + 1] == "10.0.0.8"


@patch("quark.experimental.torch.quant_perf.landing.base.requests")
@patch("quark.experimental.torch.quant_perf.landing.base.time")
@pytest.mark.parametrize(
    ("server_host", "request_host"),
    [
        ("10.0.0.8", "10.0.0.8"),
        ("0.0.0.0", "127.0.0.1"),
    ],
)
def test_readiness_uses_configured_host(mock_time, mock_requests, server_host, request_host):
    mock_time.time.side_effect = [0, 1]
    health_response = MagicMock(status_code=200)
    models_response = MagicMock()
    models_response.json.return_value = {"data": [{"id": "model"}]}
    mock_requests.get.side_effect = [health_response, models_response]
    mock_requests.post.return_value = MagicMock(status_code=200)

    assert wait_ready(9000, host=server_host) is True
    assert mock_requests.get.call_args_list[0].args[0] == f"http://{request_host}:9000/health"
