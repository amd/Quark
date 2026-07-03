#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import importlib
import json
import sys
import types
from pathlib import Path


def _load_analyze_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    fake_dashboard_pkg = types.ModuleType("quark_dashboard")
    fake_dashboard_pkg.__path__ = []
    fake_dashboard_api = types.ModuleType("quark_dashboard.api")
    fake_dashboard_api.upload_data = lambda *args, **kwargs: True
    fake_dashboard_pkg.api = fake_dashboard_api

    monkeypatch.setitem(sys.modules, "quark_dashboard", fake_dashboard_pkg)
    monkeypatch.setitem(sys.modules, "quark_dashboard.api", fake_dashboard_api)
    sys.modules.pop("tools.ci.analyze_onnx_reg_results", None)

    return importlib.import_module("tools.ci.analyze_onnx_reg_results")


def _write_payload_json(tmp_path):
    payload = {
        "version": 1,
        "data": [
            {
                "experiment": {
                    "name": "crypto_mode",
                    "model": "Crypto_Mode",
                    "settings": {"quantization_scheme": "A8W8"},
                },
                "metrics": [{"name": "Prec@1", "value": 73.68}],
                "metadata": {"duration": "6:45:00"},
            }
        ],
    }
    json_path = tmp_path / "crypto_mode.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return json_path, payload


def test_upload_results_to_dashboard_uploads_parsed_json(tmp_path, monkeypatch, capsys):
    module = _load_analyze_module(monkeypatch)
    json_path, payload = _write_payload_json(tmp_path)

    upload_calls = []

    def fake_upload_data(report_name, json_content, api_url):
        upload_calls.append((report_name, json_content, api_url))
        return True

    monkeypatch.setattr(module.dashboard_api, "upload_data", fake_upload_data)

    module.upload_results_to_dashboard(str(tmp_path))

    assert upload_calls == [("ONNX Quantization", payload, "http://quark.amd.com/dashboard")]

    stdout = capsys.readouterr().out
    assert "Uploading results to dashboard from" in stdout
    assert f"Successfully uploaded {json_path} to dashboard" in stdout


def test_upload_results_to_dashboard_reports_failed_upload(tmp_path, monkeypatch, capsys):
    module = _load_analyze_module(monkeypatch)
    json_path, payload = _write_payload_json(tmp_path)

    upload_calls = []

    def fake_upload_data(report_name, json_content, api_url):
        upload_calls.append((report_name, json_content, api_url))
        return False

    monkeypatch.setattr(module.dashboard_api, "upload_data", fake_upload_data)

    module.upload_results_to_dashboard(str(tmp_path))

    assert upload_calls == [("ONNX Quantization", payload, "http://quark.amd.com/dashboard")]

    stdout = capsys.readouterr().out
    assert "Uploading results to dashboard from" in stdout
    assert f"Failed to upload {json_path} to dashboard" in stdout
