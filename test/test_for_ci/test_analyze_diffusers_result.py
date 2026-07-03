#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import importlib
import json
import sys
from pathlib import Path

import pandas as pd


def _load_analyze_module():
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    sys.modules.pop("tools.ci.analyze_diffusers_result", None)
    return importlib.import_module("tools.ci.analyze_diffusers_result")


def _patch_to_markdown(monkeypatch, module):
    monkeypatch.setattr(module.pd.DataFrame, "to_markdown", lambda self, index=True: self.to_csv(index=index))


def test_analyze_diffusers_results_writes_merged_json_and_reports_errors(tmp_path, monkeypatch):
    module = _load_analyze_module()
    _patch_to_markdown(monkeypatch, module)

    golden_csv = tmp_path / "golden.csv"
    log_dir = tmp_path / "logs"
    merged_json = tmp_path / "diffusers_merged.json"
    log_dir.mkdir()

    df_gold = pd.DataFrame(
        {
            "float": {
                "stabilityai/stable-diffusion-xl-base-1.0": "[0.8, 10.0]",
                "runwayml/stable-diffusion-v1-5": "[0.7, 20.0]",
            },
            "w_int8_a_int8": {
                "stabilityai/stable-diffusion-xl-base-1.0": "[0.79, 10.1]",
                "runwayml/stable-diffusion-v1-5": "",
            },
            "w_fp8_a_fp8": {
                "stabilityai/stable-diffusion-xl-base-1.0": "",
                "runwayml/stable-diffusion-v1-5": "[0.69, 20.1]",
            },
        }
    )
    df_gold.to_csv(golden_csv)

    (log_dir / "diffusers_stabilityai_stable-diffusion-xl-base-1.0_float.log").write_text(
        "clip_score: 0.8000\nfid: 10.0000\n",
        encoding="utf-8",
    )
    (log_dir / "diffusers_stabilityai_stable-diffusion-xl-base-1.0_w_int8_a_int8.log").write_text(
        "clip_score: 0.7000\nfid: 9.0000\n",
        encoding="utf-8",
    )
    (log_dir / "diffusers_runwayml_stable-diffusion-v1-5_w_fp8_a_fp8.log").write_text(
        "clip_score: 0.6900\nfid: 20.1000\n",
        encoding="utf-8",
    )

    result = module.analyze_diffusers_results(
        str(golden_csv),
        log_dir=str(log_dir),
        output_prefix=str(tmp_path / "diffusers"),
        merged_json=str(merged_json),
    )

    assert (tmp_path / "diffusers_results.csv").exists()
    assert (tmp_path / "diffusers_overview.md").exists()
    assert merged_json.exists()
    assert len(result["results_df"]) == 4
    assert len(result["errors_df"]) == 1
    assert len(result["diffs_df"]) == 1

    payload = json.loads(merged_json.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["data"]) == 4

    passing_case = next(
        item
        for item in payload["data"]
        if item["experiment"]["name"] == "stabilityai/stable-diffusion-xl-base-1.0/float"
    )
    assert passing_case["metrics"][0]["value"] == 0.8
    assert passing_case["metrics"][1]["value"] == 10.0

    missing_case = next(
        item for item in payload["data"] if item["experiment"]["name"] == "runwayml/stable-diffusion-v1-5/float"
    )
    assert missing_case["metrics"][0]["value"] is None
    assert missing_case["metrics"][0]["metadata"] == "FAILED - no result found"
    assert missing_case["metrics"][1]["value"] is None
