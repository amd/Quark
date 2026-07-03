#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import importlib
import json
import sys
from pathlib import Path


def _load_analyze_module():
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    sys.modules.pop("tools.ci.analyze_vision_result", None)
    return importlib.import_module("tools.ci.analyze_vision_result")


def _patch_to_markdown(monkeypatch, module):
    monkeypatch.setattr(module.pd.DataFrame, "to_markdown", lambda self, index=True: self.to_csv(index=index))


def test_analyze_vision_results_writes_merged_json_with_missing_values(tmp_path, monkeypatch):
    module = _load_analyze_module()
    _patch_to_markdown(monkeypatch, module)

    golden_csv = tmp_path / "golden.csv"
    log_dir = tmp_path / "logs"
    merged_json = tmp_path / "vision_merged.json"
    log_dir.mkdir()

    golden_csv.write_text(
        "model,ptq,qat\nresnet18,70.0000,71.0000\nmobilenetv2,60.0000,74.0000\n",
        encoding="utf-8",
    )
    (log_dir / "resnet18.log").write_text(
        "* Acc@1 70.000 some tokens\n* Acc@1 71.000 more tokens\n",
        encoding="utf-8",
    )

    result = module.analyze_vision_results(
        str(golden_csv),
        log_dir=str(log_dir),
        output_prefix=str(tmp_path / "vision"),
        merged_json=str(merged_json),
    )

    assert (tmp_path / "vision_results.csv").exists()
    assert (tmp_path / "vision_overview.md").exists()
    assert merged_json.exists()
    assert len(result["results_df"]) == 4

    payload = json.loads(merged_json.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["data"]) == 4

    resnet_ptq = next(item for item in payload["data"] if item["experiment"]["name"] == "resnet18/ptq")
    assert resnet_ptq["metrics"][0]["value"] == 70.0
    assert resnet_ptq["metrics"][0]["metadata"] == ""

    missing_case = next(item for item in payload["data"] if item["experiment"]["name"] == "mobilenetv2/ptq")
    assert missing_case["metrics"][0]["value"] is None
    assert missing_case["metrics"][0]["metadata"] == "FAILED - no result found"
