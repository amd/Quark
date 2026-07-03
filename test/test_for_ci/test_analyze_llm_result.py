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

    sys.modules.pop("tools.ci.analyze_llm_result", None)
    return importlib.import_module("tools.ci.analyze_llm_result")


def _write_llm_golden_csv(path):
    path.write_text(
        "config_name,model_a,model_b,model_c\ncfg1,10,20,30\n",
        encoding="utf-8",
    )


def _write_llm_config_csv(path):
    path.write_text(
        "config_name,quant_scheme,base_params,hf_models,quant_models,onnx_models,gguf_models\n"
        'cfg1,w_uint4_per_group_asym,"--quant_algo awq",model_a,model_b,model_c,model_c\n',
        encoding="utf-8",
    )


def test_compare_single_acc_and_metadata_enrichment_cover_thresholds():
    module = _load_analyze_module()

    assert module.compare_single_acc("N/A", "10") == "ERROR"
    assert module.compare_single_acc("10", "10") == "="
    assert module.compare_single_acc("10", "10.005") == "~"
    assert module.compare_single_acc("10", "10.020") == "0.0020 > 0.001"
    assert module.compare_single_acc("10", "9.980") == "-0.0020 < -0.001"

    exact_entry = {"metrics": [{"name": "Perplexity", "value": 10.0}]}
    module._enrich_metric_metadata(exact_entry, "cfg1", "model_a", {"cfg1": {"model_a": "10.0000"}})
    assert json.loads(exact_entry["metrics"][0]["metadata"]) == {"golden": "10.0000", "diff": "no change"}

    rise_entry = {"metrics": [{"name": "Perplexity", "value": 11.0}]}
    module._enrich_metric_metadata(rise_entry, "cfg1", "model_a", {"cfg1": {"model_a": "10.0000"}})
    assert json.loads(rise_entry["metrics"][0]["metadata"]) == {"golden": "10.0000", "diff": "rises by 10.0000%"}

    drop_entry = {"metrics": [{"name": "Perplexity", "value": 9.0}]}
    module._enrich_metric_metadata(drop_entry, "cfg1", "model_a", {"cfg1": {"model_a": "10.0000"}})
    assert json.loads(drop_entry["metrics"][0]["metadata"]) == {"golden": "10.0000", "diff": "drops by 10.0000%"}

    none_entry = {"metrics": [{"name": "Perplexity", "value": None}]}
    module._enrich_metric_metadata(none_entry, "cfg1", "model_a", {"cfg1": {"model_a": "10.0000"}})
    assert json.loads(none_entry["metrics"][0]["metadata"]) == {"golden": "10.0000", "diff": "N/A"}

    missing_golden_entry = {"metrics": [{"name": "Perplexity", "value": 10.0}]}
    module._enrich_metric_metadata(missing_golden_entry, "cfg1", "unknown_model", {"cfg1": {"model_a": "10.0000"}})
    assert json.loads(missing_golden_entry["metrics"][0]["metadata"]) == {"golden": "N/A", "diff": "N/A"}

    zero_golden_entry = {"metrics": [{"name": "Perplexity", "value": 10.0}]}
    module._enrich_metric_metadata(zero_golden_entry, "cfg1", "model_z", {"cfg1": {"model_z": "0.0000"}})
    assert json.loads(zero_golden_entry["metrics"][0]["metadata"]) == {"golden": "0.0000", "diff": "N/A"}

    invalid_value_entry = {"metrics": [{"name": "Perplexity", "value": "not-a-number"}]}
    module._enrich_metric_metadata(invalid_value_entry, "cfg1", "model_a", {"cfg1": {"model_a": "10.0000"}})
    assert json.loads(invalid_value_entry["metrics"][0]["metadata"]) == {"golden": "10.0000", "diff": "N/A"}


def test_analyze_llm_results_passes_merge_inputs(monkeypatch):
    module = _load_analyze_module()

    fake_result = {
        "errors_df": pd.DataFrame([{"config_name": "cfg1"}]),
        "diffs_df": pd.DataFrame([{"config_name": "cfg1"}]),
    }
    calls = []

    def fake_generate(config_csv_path, results_csv_path, golden_csv_path, overview_md_path, log_dir, cli):
        calls.append(
            (
                "generate",
                config_csv_path,
                results_csv_path,
                golden_csv_path,
                overview_md_path,
                log_dir,
                cli,
            )
        )
        return fake_result

    def fake_merge(
        config_csv_path,
        log_dir,
        output_json_path,
        cli=False,
        errors_df=None,
        diffs_df=None,
        golden_csv_path=None,
        report_period="nightly",
    ):
        calls.append(
            (
                "merge",
                config_csv_path,
                log_dir,
                output_json_path,
                cli,
                errors_df,
                diffs_df,
                golden_csv_path,
                report_period,
            )
        )

    monkeypatch.setattr(module, "generate_results_csv", fake_generate)
    monkeypatch.setattr(module, "merge_experiment_reports", fake_merge)

    result = module.analyze_llm_results(
        "config.csv",
        "golden.csv",
        log_dir="logs",
        output_prefix="artifacts/llm",
        cli=True,
        merged_json="merged.json",
        report_period="weekly",
    )

    assert result is fake_result
    assert calls[0] == (
        "generate",
        "config.csv",
        "artifacts/llm_results.csv",
        "golden.csv",
        "artifacts/llm_overview.md",
        "logs",
        True,
    )
    assert calls[1] == (
        "merge",
        "config.csv",
        "logs",
        "merged.json",
        True,
        fake_result["errors_df"],
        fake_result["diffs_df"],
        "golden.csv",
        "weekly",
    )


def test_merge_experiment_reports_handles_existing_corrupt_and_missing_json(tmp_path, capsys):
    module = _load_analyze_module()

    config_csv = tmp_path / "config.csv"
    golden_csv = tmp_path / "golden.csv"
    log_dir = tmp_path / "logs"
    output_json = tmp_path / "merged.json"
    log_dir.mkdir()

    _write_llm_config_csv(config_csv)
    _write_llm_golden_csv(golden_csv)

    existing_report = {
        "version": 1,
        "data": [
            {
                "experiment": {"name": "model_a/cfg1", "model": "model_a", "settings": {}},
                "metrics": [{"name": "Perplexity", "value": 10.123456, "metadata": ""}],
                "metadata": {},
            }
        ],
    }
    (log_dir / "experiment_report_llm_ptq_model_a_cfg1.json").write_text(
        json.dumps(existing_report),
        encoding="utf-8",
    )
    (log_dir / "experiment_report_llm_ptq_model_b_cfg1.json").write_text("{", encoding="utf-8")

    result = module.merge_experiment_reports(
        str(config_csv),
        str(log_dir),
        str(output_json),
        errors_df=pd.DataFrame([{"config_name": "cfg1"}]),
        diffs_df=pd.DataFrame([{"config_name": "cfg1"}]),
        golden_csv_path=str(golden_csv),
        report_period="weekly",
    )

    assert output_json.exists()
    assert len(result["data"]) == 4

    assert result["data"][0]["metrics"][0]["value"] == 10.1235
    first_metric_metadata = json.loads(result["data"][0]["metrics"][0]["metadata"])
    assert first_metric_metadata == {"golden": "10.0000", "diff": "rises by 1.2350%"}

    corrupt_stub = result["data"][1]
    assert corrupt_stub["experiment"]["name"] == "model_b/w_uint4_per_group_asym/awq"
    assert corrupt_stub["experiment"]["timestamp"]
    assert corrupt_stub["experiment"]["timestamp"].endswith("Z")
    assert corrupt_stub["metrics"][0]["value"] is None

    missing_stub = result["data"][2]
    assert missing_stub["experiment"]["name"] == "model_c/w_uint4_per_group_asym/awq"
    assert missing_stub["experiment"]["timestamp"]
    assert missing_stub["experiment"]["timestamp"].endswith("Z")
    assert missing_stub["metrics"][0]["value"] is None

    summary = result["data"][-1]
    assert summary["experiment"] == {
        "name": "quark-summary/weekly",
        "model": "quark-summary/weekly",
        "settings": {"source": "summary", "period": "weekly"},
        "timestamp": summary["experiment"]["timestamp"],
    }
    assert summary["experiment"]["timestamp"].endswith("Z")
    assert summary["metadata"] == {
        "total_test_cases": 3,
        "passed": 1,
        "total failed": 2,
        "runtime_failed": 1,
        "accuracy_failed": 1,
        "other_failed": 0,
    }

    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved == result

    stdout = capsys.readouterr().out
    assert "Warning: failed to read" in stdout
    assert "Missing JSON (test likely failed): experiment_report_llm_ptq_model_c_cfg1.json" in stdout
    assert "Merged experiment report written to:" in stdout
