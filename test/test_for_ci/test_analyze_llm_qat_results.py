#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import csv
import importlib
import json
import os
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_qat_module():
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    sys.modules.pop("tools.ci.analyze_llm_qat_results", None)
    return importlib.import_module("tools.ci.analyze_llm_qat_results")


def _patch_markdown(monkeypatch):
    monkeypatch.setattr(
        pd.DataFrame,
        "to_markdown",
        lambda self, index=False: f"table(rows={len(self)}, cols={list(self.columns)})",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_log(path: Path, *lines: str) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_qat_result_json(log_dir: Path, model_name: str, task_name: str, phase: str, payload: dict) -> Path:
    path = log_dir / f"llm_qat_{model_name}_{task_name}_{phase}_eval_results.json"
    _write_json(path, payload)
    return path


def test_helper_functions_cover_common_edges(tmp_path):
    module = _load_qat_module()

    nested_output = tmp_path / "nested" / "dir" / "results.csv"
    module.ensure_parent_dir(str(nested_output))
    assert nested_output.parent.is_dir()
    module.ensure_parent_dir("results.csv")

    assert module.format_float(None) == "N/A"
    assert module.format_float(1.23456) == "1.2346"
    assert module.format_diff_abs(None) == "N/A"
    assert module.format_diff_abs(-0.125) == "-0.1250"

    assert module._parse_float(None) is None
    assert module._parse_float("N/A") is None
    assert module._parse_float("1.25") == 1.25
    assert module._parse_float(object()) is None

    assert module._default_metric_candidates("wikitext") == [
        "word_perplexity",
        "perplexity",
        "byte_perplexity",
        "bits_per_byte",
    ]
    assert module._default_metric_candidates("mmlu")[0] == "acc"
    assert module._metric_candidate_keys("acc,none", "mmlu") == ["acc,none", "acc"]
    assert "acc_norm" in module._metric_candidate_keys(None, "mmlu")

    assert module._infer_higher_is_better("acc") is True
    assert module._infer_higher_is_better("word_perplexity") is False
    assert module._infer_higher_is_better("bits_per_byte") is False
    assert module._supports_perplexity_log_fallback("word_perplexity") is True
    assert module._supports_perplexity_log_fallback("acc") is False

    assert module.compare_metric(None, 1.0, 0.1, True) == ("MISSING_RESULT", None)
    assert module.compare_metric(1.0, None, 0.1, True) == ("MISSING_GOLDEN", None)
    status, diff = module.compare_metric(1.03, 1.0, 0.05, True)
    assert status == "PASS"
    assert diff == pytest.approx(0.03)

    status, diff = module.compare_metric(1.2, 1.0, 0.05, True)
    assert status == "IMPROVED"
    assert diff == pytest.approx(0.2)

    status, diff = module.compare_metric(0.8, 1.0, 0.05, True)
    assert status == "REGRESSION"
    assert diff == pytest.approx(-0.2)

    status, diff = module.compare_metric(1.2, 1.0, 0.05, False)
    assert status == "REGRESSION"
    assert diff == pytest.approx(0.2)

    status, diff = module.compare_metric(0.8, 1.0, 0.05, False)
    assert status == "IMPROVED"
    assert diff == pytest.approx(-0.2)


def test_load_matrix_and_golden_support_new_and_legacy_formats(tmp_path):
    module = _load_qat_module()

    matrix_yaml = tmp_path / "matrix.yaml"
    matrix_yaml.write_text(
        "include:\n  - model_name: THUDM_chatglm3_6b\n    eval_task: wikitext\n",
        encoding="utf-8",
    )
    assert module.load_matrix(str(matrix_yaml)) == [{"model_name": "THUDM_chatglm3_6b", "eval_task": "wikitext"}]

    invalid_matrix_yaml = tmp_path / "invalid_matrix.yaml"
    invalid_matrix_yaml.write_text("include: invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected matrix format"):
        module.load_matrix(str(invalid_matrix_yaml))

    golden_csv = tmp_path / "golden.csv"
    fieldnames = [
        "model_name",
        "eval_task",
        "metric_name",
        "baseline_golden_value",
        "quantized_golden_value",
        "baseline_golden_ppl",
        "quantized_golden_ppl",
        "threshold_abs",
        "baseline_threshold_abs",
        "quantized_threshold_abs",
        "threshold_pct",
        "baseline_threshold_pct",
        "quantized_threshold_pct",
    ]
    with golden_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"model_name": "", "eval_task": "ignored"})
        writer.writerow(
            {
                "model_name": "model_new",
                "eval_task": "wikitext",
                "metric_name": "word_perplexity",
                "baseline_golden_value": "10.5",
                "quantized_golden_value": "11.5",
                "threshold_abs": "0.2",
            }
        )
        writer.writerow(
            {
                "model_name": "model_old",
                "baseline_golden_ppl": "20.5",
                "quantized_golden_ppl": "21.5",
                "baseline_threshold_pct": "0.3",
                "quantized_threshold_pct": "0.4",
            }
        )

    loaded = module.load_golden(str(golden_csv), 0.05)

    assert set(loaded) == {("model_new", "wikitext"), ("model_old", "")}
    assert loaded[("model_new", "wikitext")] == module.GoldenEntry(
        eval_task="wikitext",
        metric_name="word_perplexity",
        baseline_golden_value=10.5,
        quantized_golden_value=11.5,
        baseline_threshold_abs=0.2,
        quantized_threshold_abs=0.2,
    )
    assert loaded[("model_old", "")] == module.GoldenEntry(
        eval_task="",
        metric_name=None,
        baseline_golden_value=20.5,
        quantized_golden_value=21.5,
        baseline_threshold_abs=0.3,
        quantized_threshold_abs=0.4,
    )


def test_log_and_json_extractors_handle_preferred_fallback_and_error_paths(tmp_path):
    module = _load_qat_module()

    missing_log = tmp_path / "missing.log"
    assert module.extract_last_ppl_from_log(str(missing_log)) is None
    assert module.extract_last_time_from_log(str(missing_log)) == "N/A"

    run_log = tmp_path / "run.log"
    _write_log(
        run_log,
        "[INFO]: Perplexity: 10.5",
        "Time elapsed: 00:00:01",
        "[AMD-INFO]: Perplexity: 11.5",
        "Time elapsed: 00:00:02",
    )
    assert module.extract_last_ppl_from_log(str(run_log)) == 11.5
    assert module.extract_last_time_from_log(str(run_log)) == "00:00:02"

    first = tmp_path / "result_a.json"
    second = tmp_path / "result_b.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))
    assert module.find_latest_file(str(tmp_path), "result_*.json") == str(second)
    assert module.find_latest_file(str(tmp_path), "not_found*.json") is None

    direct_payload = {"results": {"wikitext": {"word_perplexity,none": 9.9}}}
    assert module._extract_task_results(direct_payload, "wikitext") == {"word_perplexity,none": 9.9}

    fuzzy_payload = {"results": {"mmlu:5shot": {"acc,none": 0.71}}}
    assert module._extract_task_results(fuzzy_payload, "mmlu") == {"acc,none": 0.71}
    assert module._extract_task_results({"results": []}, "mmlu") is None

    higher_is_better_payload = {"higher_is_better": {"mmlu": {"acc": True}, "wikitext": False}}
    assert module._extract_higher_is_better(higher_is_better_payload, "mmlu", "acc,none") is True
    assert module._extract_higher_is_better(higher_is_better_payload, "wikitext", "word_perplexity") is False
    assert module._extract_higher_is_better({}, "anything", None) is True

    assert module.extract_metric_from_result_json(None, "wikitext", "word_perplexity") == (
        None,
        "word_perplexity",
        False,
    )

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    assert module.extract_metric_from_result_json(str(invalid_json), "wikitext", "word_perplexity") == (
        None,
        "word_perplexity",
        False,
    )

    missing_task_json = tmp_path / "missing_task.json"
    _write_json(missing_task_json, {"results": {"other": {"acc,none": 0.5}}})
    assert module.extract_metric_from_result_json(str(missing_task_json), "wikitext", "word_perplexity") == (
        None,
        "word_perplexity",
        False,
    )

    preferred_json = tmp_path / "preferred.json"
    _write_json(
        preferred_json,
        {
            "results": {"wikitext": {"word_perplexity,none": 12.34}},
            "higher_is_better": {"wikitext": {"word_perplexity": False}},
        },
    )
    assert module.extract_metric_from_result_json(str(preferred_json), "wikitext", None) == (
        12.34,
        "word_perplexity",
        False,
    )

    fallback_json = tmp_path / "fallback.json"
    _write_json(
        fallback_json,
        {
            "results": {"custom_task_v2": {"alias": "custom", "metric_stderr,none": 0.01, "my_metric": 0.88}},
        },
    )
    assert module.extract_metric_from_result_json(str(fallback_json), "custom_task", "metric") == (
        0.88,
        "my_metric",
        True,
    )


def test_lookup_golden_entry_prefers_exact_then_model_default_then_threshold_default():
    module = _load_qat_module()

    golden = {
        ("exact_model", "mmlu"): module.GoldenEntry("mmlu", "acc", 0.71, 0.69, 0.01, 0.02),
        ("fallback_model", ""): module.GoldenEntry("", None, 12.0, 11.0, 0.05, 0.06),
    }

    assert module._lookup_golden_entry(golden, "exact_model", "mmlu", 0.5) == module.GoldenEntry(
        "mmlu",
        "acc",
        0.71,
        0.69,
        0.01,
        0.02,
    )
    assert module._lookup_golden_entry(golden, "fallback_model", "wikitext", 0.5) == module.GoldenEntry(
        "",
        None,
        12.0,
        11.0,
        0.05,
        0.06,
    )
    assert module._lookup_golden_entry(golden, "missing_model", "wikitext", 0.5) == module.GoldenEntry(
        "wikitext",
        None,
        None,
        None,
        0.5,
        0.5,
    )


def test_build_rows_and_phase_rows_cover_perplexity_accuracy_and_missing_golden(tmp_path):
    module = _load_qat_module()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    _write_qat_result_json(
        log_dir,
        "THUDM_chatglm3_6b",
        "wikitext",
        "non_quantized",
        {
            "results": {"wikitext": {"word_perplexity,none": 29.94}},
            "higher_is_better": {"wikitext": {"word_perplexity": False}},
        },
    )
    _write_qat_result_json(
        log_dir,
        "THUDM_chatglm3_6b",
        "wikitext",
        "quantized",
        {
            "results": {"wikitext": {"word_perplexity,none": 9.92}},
            "higher_is_better": {"wikitext": {"word_perplexity": False}},
        },
    )
    _write_qat_result_json(
        log_dir,
        "demo_mmlu_model",
        "mmlu",
        "non_quantized",
        {
            "results": {"mmlu": {"acc,none": 0.71}},
            "higher_is_better": {"mmlu": {"acc": True}},
        },
    )
    _write_qat_result_json(
        log_dir,
        "demo_mmlu_model",
        "mmlu",
        "quantized",
        {
            "results": {"mmlu": {"acc,none": 0.60}},
            "higher_is_better": {"mmlu": {"acc": True}},
        },
    )

    _write_log(log_dir / "llm_qat_THUDM_chatglm3_6b_test_bf16.log", "Time elapsed: 00:00:10")
    _write_log(log_dir / "llm_qat_THUDM_chatglm3_6b_test_finetuned.log", "Time elapsed: 00:00:20")
    _write_log(log_dir / "llm_qat_THUDM_chatglm3_6b_finetune.log", "Time elapsed: 00:00:30")
    _write_log(log_dir / "llm_qat_THUDM_chatglm3_6b_total_time.log", "Time elapsed: 00:01:00")

    _write_log(log_dir / "llm_qat_demo_mmlu_model_test_bf16.log", "Time elapsed: 00:00:11")
    _write_log(log_dir / "llm_qat_demo_mmlu_model_test_finetuned.log", "Time elapsed: 00:00:21")
    _write_log(log_dir / "llm_qat_demo_mmlu_model_finetune.log", "Time elapsed: 00:00:31")
    _write_log(log_dir / "llm_qat_demo_mmlu_model_total_time.log", "Time elapsed: 00:01:01")

    _write_log(
        log_dir / "llm_qat_fallback_model_test_bf16.log",
        "[INFO]: Perplexity: 7.00",
        "Time elapsed: 00:00:12",
    )
    _write_log(
        log_dir / "llm_qat_fallback_model_test_finetuned.log",
        "[INFO]: Perplexity: 6.50",
        "Time elapsed: 00:00:22",
    )
    _write_log(log_dir / "llm_qat_fallback_model_finetune.log", "Time elapsed: 00:00:32")
    _write_log(log_dir / "llm_qat_fallback_model_total_time.log", "Time elapsed: 00:01:02")

    matrix_entries = [
        {"model_name": "THUDM_chatglm3_6b", "eval_task": "wikitext"},
        {"model_name": "demo_mmlu_model", "eval_task": "mmlu"},
        {"model_name": "fallback_model", "eval_task": "wikitext"},
        {"model_name": ""},
    ]
    golden = {
        ("THUDM_chatglm3_6b", "wikitext"): module.GoldenEntry("wikitext", "word_perplexity", 29.93, 9.84, 0.05, 0.05),
        ("demo_mmlu_model", "mmlu"): module.GoldenEntry("mmlu", "acc", 0.70, 0.65, 0.02, 0.02),
        ("fallback_model", ""): module.GoldenEntry("", None, None, None, 0.05, 0.05),
    }

    rows = module.build_rows(matrix_entries, golden, str(log_dir), 0.05)
    assert len(rows) == 3

    by_model = {row["model_name"]: row for row in rows}

    assert by_model["THUDM_chatglm3_6b"]["metric_name"] == "word_perplexity"
    assert by_model["THUDM_chatglm3_6b"]["higher_is_better"] == "false"
    assert by_model["THUDM_chatglm3_6b"]["baseline_status"] == "PASS"
    assert by_model["THUDM_chatglm3_6b"]["quantized_status"] == "REGRESSION"
    assert by_model["THUDM_chatglm3_6b"]["total_time"] == "00:01:00"

    assert by_model["demo_mmlu_model"]["metric_name"] == "acc"
    assert by_model["demo_mmlu_model"]["higher_is_better"] == "true"
    assert by_model["demo_mmlu_model"]["baseline_status"] == "PASS"
    assert by_model["demo_mmlu_model"]["quantized_status"] == "REGRESSION"
    assert by_model["demo_mmlu_model"]["baseline_actual_value"] == "0.7100"

    assert by_model["fallback_model"]["baseline_json"] == ""
    assert by_model["fallback_model"]["metric_name"] == "word_perplexity"
    assert by_model["fallback_model"]["baseline_actual_value"] == "7.0000"
    assert by_model["fallback_model"]["quantized_actual_value"] == "6.5000"
    assert by_model["fallback_model"]["baseline_status"] == "MISSING_GOLDEN"
    assert by_model["fallback_model"]["quantized_status"] == "MISSING_GOLDEN"

    phase_rows = module.build_phase_rows(rows, module.FAILURE_STATUSES)
    assert {(row["model_name"], row["phase"], row["status"]) for row in phase_rows} == {
        ("THUDM_chatglm3_6b", "quantized", "REGRESSION"),
        ("demo_mmlu_model", "quantized", "REGRESSION"),
        ("fallback_model", "baseline", "MISSING_GOLDEN"),
        ("fallback_model", "quantized", "MISSING_GOLDEN"),
    }
    assert all(row["status"] in module.FAILURE_STATUSES for row in phase_rows)


def test_write_results_and_overview_markdown_outputs(tmp_path, monkeypatch):
    module = _load_qat_module()
    _patch_markdown(monkeypatch)

    rows = [
        {
            "model_name": "model_a",
            "eval_task": "wikitext",
            "metric_name": "word_perplexity",
            "higher_is_better": "false",
            "baseline_actual_value": "10.0000",
            "baseline_golden_value": "10.0000",
            "baseline_diff_abs": "+0.0000",
            "baseline_threshold_abs": "0.0500",
            "baseline_status": "PASS",
            "quantized_actual_value": "11.0000",
            "quantized_golden_value": "10.0000",
            "quantized_diff_abs": "+1.0000",
            "quantized_threshold_abs": "0.0500",
            "quantized_status": "REGRESSION",
            "baseline_time": "00:00:10",
            "quantized_time": "00:00:20",
            "finetune_time": "00:00:30",
            "total_time": "00:01:00",
            "baseline_json": "baseline.json",
            "quantized_json": "quantized.json",
            "baseline_log": "baseline.log",
            "quantized_log": "quantized.log",
        }
    ]
    error_rows = module.build_phase_rows(rows, module.FAILURE_STATUSES)

    results_csv = tmp_path / "nested" / "qat_results.csv"
    module.write_results_csv(rows, str(results_csv))
    csv_text = results_csv.read_text(encoding="utf-8")
    assert "model_name,eval_task,metric_name,higher_is_better" in csv_text
    assert "model_a,wikitext,word_perplexity,false" in csv_text

    markdown_with_failures = module.render_overview_markdown(rows, error_rows, "https://example.test/run/1")
    assert "Run URL: https://example.test/run/1" in markdown_with_failures
    assert "## Failed Cases" in markdown_with_failures
    assert "Some tests failed. Threshold breaches are counted as failures." in markdown_with_failures
    assert "table(rows=2" in markdown_with_failures

    markdown_without_failures = module.render_overview_markdown(rows, [], None)
    assert "No failed cases." in markdown_without_failures
    assert "All tests passed." in markdown_without_failures

    overview_md = tmp_path / "nested" / "overview.md"
    module.write_overview_markdown(rows, error_rows, str(overview_md), "https://example.test/run/2")
    assert "Run URL: https://example.test/run/2" in overview_md.read_text(encoding="utf-8")


def test_main_script_executes_end_to_end(tmp_path, monkeypatch, capsys):
    _patch_markdown(monkeypatch)
    repo_root = Path(__file__).resolve().parents[2]

    matrix_yaml = tmp_path / "matrix.yaml"
    matrix_yaml.write_text(
        "include:\n  - model_name: THUDM_chatglm3_6b\n    eval_task: wikitext\n",
        encoding="utf-8",
    )

    golden_csv = tmp_path / "golden.csv"
    golden_csv.write_text(
        "model_name,eval_task,metric_name,baseline_golden_value,quantized_golden_value,baseline_threshold_abs,quantized_threshold_abs\n"
        "THUDM_chatglm3_6b,wikitext,word_perplexity,29.93,9.84,0.05,0.05\n",
        encoding="utf-8",
    )

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_qat_result_json(
        log_dir,
        "THUDM_chatglm3_6b",
        "wikitext",
        "non_quantized",
        {
            "results": {"wikitext": {"word_perplexity,none": 29.93}},
            "higher_is_better": {"wikitext": {"word_perplexity": False}},
        },
    )
    _write_qat_result_json(
        log_dir,
        "THUDM_chatglm3_6b",
        "wikitext",
        "quantized",
        {
            "results": {"wikitext": {"word_perplexity,none": 9.84}},
            "higher_is_better": {"wikitext": {"word_perplexity": False}},
        },
    )
    _write_log(log_dir / "llm_qat_THUDM_chatglm3_6b_test_bf16.log", "Time elapsed: 00:00:10")
    _write_log(log_dir / "llm_qat_THUDM_chatglm3_6b_test_finetuned.log", "Time elapsed: 00:00:20")
    _write_log(log_dir / "llm_qat_THUDM_chatglm3_6b_finetune.log", "Time elapsed: 00:00:30")
    _write_log(log_dir / "llm_qat_THUDM_chatglm3_6b_total_time.log", "Time elapsed: 00:01:00")

    results_csv = tmp_path / "results.csv"
    overview_md = tmp_path / "overview.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_llm_qat_results.py",
            "--matrix_yaml",
            str(matrix_yaml),
            "--golden_csv",
            str(golden_csv),
            "--log_dir",
            str(log_dir),
            "--results_csv",
            str(results_csv),
            "--overview_md",
            str(overview_md),
            "--run_url",
            "https://example.test/run/3",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(repo_root / "tools" / "ci" / "analyze_llm_qat_results.py"), run_name="__main__")
    assert exc.value.code == 0

    assert results_csv.exists()
    assert overview_md.exists()

    stdout = capsys.readouterr().out
    assert f"QAT results CSV written to: {results_csv}" in stdout
    assert f"QAT overview Markdown written to: {overview_md}" in stdout
