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


def _write_golden_csv(tmp_path, config, acc1, acc5):
    golden_file = tmp_path / "golden.csv"
    golden_file.write_text(f",{config}\nacc1,{acc1}\nacc5,{acc5}\n", encoding="utf-8")
    return golden_file


def _patch_to_markdown(monkeypatch, module):
    monkeypatch.setattr(module.pd.DataFrame, "to_markdown", lambda self, index=True: self.to_csv(index=index))


def test_compare_cnn_results_parses_metrics_and_writes_markdown(tmp_path, monkeypatch, capsys):
    module = _load_analyze_module(monkeypatch)
    _patch_to_markdown(monkeypatch, module)
    config = "resnet50"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    output_csv = tmp_path / "current.md"
    golden_file = _write_golden_csv(tmp_path, config, "76.100", "93.000")

    (log_dir / f"validate_{config}.log").write_text(
        "Validation summary: * Prec@1 76.100 top1 delta Prec@5 93.400\n",
        encoding="utf-8",
    )

    err_count, diff_count, err_df, diff_df = module.compare_cnn_results(
        str(log_dir), str(output_csv), [config], str(golden_file)
    )

    assert err_count == 0
    assert diff_count == 1
    assert err_df.empty
    assert len(diff_df) == 1
    assert diff_df.iloc[0].to_dict() == {
        "target": "acc5",
        "config": config,
        "golden": "93.000",
        "current": "93.400",
        "diff": "rises by 0.400",
    }

    markdown = output_csv.read_text(encoding="utf-8")
    assert config in markdown
    assert "76.100" in markdown
    assert "93.400" in markdown

    stdout = capsys.readouterr().out
    assert "Success to compare_cnn_results" in stdout


def test_compare_cnn_results_marks_missing_log_as_error(tmp_path, monkeypatch, capsys):
    module = _load_analyze_module(monkeypatch)
    _patch_to_markdown(monkeypatch, module)
    config = "resnet50"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    output_csv = tmp_path / "current.md"
    golden_file = _write_golden_csv(tmp_path, config, "76.100", "93.000")

    err_count, diff_count, err_df, diff_df = module.compare_cnn_results(
        str(log_dir), str(output_csv), [config], str(golden_file)
    )

    assert err_count == 1
    assert diff_count == 0
    assert diff_df.empty
    assert err_df.at["acc1", config] == "ERROR"
    assert err_df.at["acc5", config] == "ERROR"
    assert output_csv.exists()

    stdout = capsys.readouterr().out
    assert f"Did not found the validate log of {config}" in stdout
    assert "Success to compare_cnn_results" in stdout


def _write_procyon_log(log_dir, model, body):
    log_path = log_dir / f"procyon_{model}.log"
    log_path.write_text(body, encoding="utf-8")
    return log_path


def _write_procyon_golden(tmp_path, rows):
    """rows: list of (metric, {stage: value}) tuples. Columns inferred from union."""
    stages = []
    for _, by_stage in rows:
        for s in by_stage:
            if s not in stages:
                stages.append(s)
    lines = ["," + ",".join(stages)]
    for metric, by_stage in rows:
        cells = [str(by_stage.get(s, "")) for s in stages]
        lines.append(f"{metric}," + ",".join(cells))
    golden_file = tmp_path / "procyon_golden.csv"
    golden_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return golden_file


def test_parse_procyon_log_extracts_resources_and_accuracies(tmp_path, monkeypatch):
    module = _load_analyze_module(monkeypatch)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = _write_procyon_log(
        log_dir,
        "blip",
        (
            "PROCYON_METRIC blip_image_encoder memory_gb 12.50\n"
            "PROCYON_METRIC blip_image_encoder peak_disk_gb 8.30\n"
            "PROCYON_METRIC blip_text_decoder memory_gb 5.75\n"
            "PROCYON_METRIC blip_text_decoder peak_disk_gb 2.10\n"
            "noise that should be skipped\n"
            "Bleu_4: 0.412\n"
            "CIDEr: 1.234\n"
        ),
    )

    resources, accuracies = module.parse_procyon_log(str(log_path))

    assert resources == {
        "blip_image_encoder": {"memory_gb": 12.50, "peak_disk_gb": 8.30},
        "blip_text_decoder": {"memory_gb": 5.75, "peak_disk_gb": 2.10},
    }
    assert accuracies == {"bleu4": 0.412, "cider": 1.234}


def test_within_tolerance_handles_edge_cases(monkeypatch):
    module = _load_analyze_module(monkeypatch)

    assert module._within_tolerance("top1", None, 0.5) is True
    assert module._within_tolerance("top1", "", 0.5) is True
    assert module._within_tolerance("top1", "not-a-number", 0.5) is False
    assert module._within_tolerance("top1", 0, 0.0) is True
    assert module._within_tolerance("top1", 0, 1.0) is False
    assert module._within_tolerance("top1", 1.0, 1.001) is True
    assert module._within_tolerance("top1", 1.0, 1.5) is False
    assert module._within_tolerance("memory_gb", 10.0, 10.9) is True
    assert module._within_tolerance("memory_gb", 10.0, 11.5) is False
    assert module._within_tolerance("unknown_metric", 100.0, 100.5) is True


def test_golden_or_dash_returns_value_or_placeholder(monkeypatch):
    module = _load_analyze_module(monkeypatch)
    import pandas as pd

    assert module._golden_or_dash(pd.DataFrame(), "top1", "convnext") == "-"

    df = pd.DataFrame({"convnext": [0.75, None]}, index=["top1", "memory_gb"])
    assert module._golden_or_dash(df, "missing_metric", "convnext") == "-"
    assert module._golden_or_dash(df, "top1", "missing_stage") == "-"
    assert module._golden_or_dash(df, "memory_gb", "convnext") == "-"
    assert module._golden_or_dash(df, "top1", "convnext") == 0.75


def test_compare_procyon_results_success_and_dashboard_json(tmp_path, monkeypatch, capsys):
    module = _load_analyze_module(monkeypatch)
    _patch_to_markdown(monkeypatch, module)

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_procyon_log(
        log_dir,
        "convnext",
        (
            "PROCYON_METRIC convnext memory_gb 12.500\n"
            "PROCYON_METRIC convnext peak_disk_gb 8.300\n"
            "Top-1 Accuracy: 0.7520\n"
        ),
    )
    golden_file = _write_procyon_golden(
        tmp_path,
        [
            ("memory_gb", {"convnext": 12.5}),
            ("peak_disk_gb", {"convnext": 8.3}),
            ("top1", {"convnext": 0.752}),
        ],
    )
    output_csv = tmp_path / "procyon.md"
    json_output = tmp_path / "procyon.json"

    err_count, diff_count, err_df, diff_df = module.compare_procyon_results(
        str(log_dir), str(output_csv), ["convnext"], str(golden_file), str(json_output)
    )

    assert err_count == 0
    assert diff_count == 0
    assert err_df.empty
    assert diff_df.empty
    assert output_csv.exists()

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["data"]) == 1
    entry = payload["data"][0]
    assert entry["experiment"]["name"] == "procyon/convnext"
    metric_names = sorted(m["name"] for m in entry["metrics"])
    assert metric_names == ["memory_gb", "peak_disk_gb", "top1"]
    for m in entry["metrics"]:
        assert m["metadata"] == ""

    stdout = capsys.readouterr().out
    assert "Success to compare_procyon_results" in stdout
    assert "Procyon dashboard JSON written to" in stdout


def test_compare_procyon_results_flags_out_of_tolerance(tmp_path, monkeypatch):
    module = _load_analyze_module(monkeypatch)
    _patch_to_markdown(monkeypatch, module)

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_procyon_log(
        log_dir,
        "convnext",
        (
            "PROCYON_METRIC convnext memory_gb 20.000\n"
            "PROCYON_METRIC convnext peak_disk_gb 8.300\n"
            "Top-1 Accuracy: 0.5000\n"
        ),
    )
    golden_file = _write_procyon_golden(
        tmp_path,
        [
            ("memory_gb", {"convnext": 12.5}),
            ("peak_disk_gb", {"convnext": 8.3}),
            ("top1", {"convnext": 0.752}),
        ],
    )

    err_count, diff_count, err_df, diff_df = module.compare_procyon_results(
        str(log_dir),
        str(tmp_path / "out.md"),
        ["convnext"],
        str(golden_file),
        str(tmp_path / "out.json"),
    )

    assert err_count == 0
    assert diff_count == 2  # memory_gb and top1 both out of tolerance
    diff_targets = sorted(diff_df["target"].tolist())
    assert diff_targets == ["memory_gb", "top1"]


def test_compare_procyon_results_missing_log_and_missing_markers(tmp_path, monkeypatch, capsys):
    module = _load_analyze_module(monkeypatch)
    _patch_to_markdown(monkeypatch, module)

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    # detr log present but with no markers and no accuracy line
    _write_procyon_log(log_dir, "detr", "Nothing useful here.\n")
    # sam2 log absent entirely
    golden_file = _write_procyon_golden(
        tmp_path,
        [
            ("memory_gb", {"detr_resnet50": 4.0, "sam2_image_encoder": 6.0, "sam2_image_decoder": 6.0}),
            ("peak_disk_gb", {"detr_resnet50": 1.0, "sam2_image_encoder": 2.0, "sam2_image_decoder": 2.0}),
            ("map", {"detr_resnet50": 0.5}),
            ("iou", {"sam2_image_encoder": 0.7, "sam2_image_decoder": 0.7}),
        ],
    )

    err_count, diff_count, err_df, diff_df = module.compare_procyon_results(
        str(log_dir),
        str(tmp_path / "out.md"),
        ["detr", "sam2"],
        str(golden_file),
        str(tmp_path / "out.json"),
    )

    # detr: 2 resource markers missing + 1 accuracy regex missing = 3 errors
    # sam2: 1 missing log error
    assert err_count == 4
    assert diff_count == 0
    assert "MISSING" in err_df["current"].tolist()

    stdout = capsys.readouterr().out
    assert "Did not find a unique procyon log for sam2" in stdout


def test_compare_procyon_results_skips_nan_golden_entries(tmp_path, monkeypatch):
    module = _load_analyze_module(monkeypatch)
    _patch_to_markdown(monkeypatch, module)

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_procyon_log(
        log_dir,
        "convnext",
        (
            "PROCYON_METRIC convnext memory_gb 12.500\n"
            "PROCYON_METRIC convnext peak_disk_gb 8.300\n"
            "Top-1 Accuracy: 0.7520\n"
        ),
    )
    # Golden has NaN entries (empty cells) for memory_gb and top1, so the
    # comparison should be skipped for those metrics without raising or
    # registering a diff.
    golden_file = _write_procyon_golden(
        tmp_path,
        [
            ("memory_gb", {"convnext": ""}),
            ("peak_disk_gb", {"convnext": 8.3}),
            ("top1", {"convnext": ""}),
        ],
    )

    err_count, diff_count, err_df, diff_df = module.compare_procyon_results(
        str(log_dir),
        str(tmp_path / "out.md"),
        ["convnext"],
        str(golden_file),
        str(tmp_path / "out.json"),
    )

    assert err_count == 0
    assert diff_count == 0


def test_compare_procyon_results_handles_missing_golden_file(tmp_path, monkeypatch, capsys):
    module = _load_analyze_module(monkeypatch)
    _patch_to_markdown(monkeypatch, module)

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_procyon_log(
        log_dir,
        "convnext",
        (
            "PROCYON_METRIC convnext memory_gb 12.500\n"
            "PROCYON_METRIC convnext peak_disk_gb 8.300\n"
            "Top-1 Accuracy: 0.7520\n"
        ),
    )

    err_count, diff_count, err_df, diff_df = module.compare_procyon_results(
        str(log_dir),
        str(tmp_path / "out.md"),
        ["convnext"],
        str(tmp_path / "does_not_exist.csv"),
        str(tmp_path / "out.json"),
    )

    assert err_count == 0
    assert diff_count == 0
    assert err_df.empty
    assert diff_df.empty

    stdout = capsys.readouterr().out
    assert "procyon golden file not found" in stdout


def test_get_overview_result_nightly_only(tmp_path, monkeypatch, capsys):
    module = _load_analyze_module(monkeypatch)
    _patch_to_markdown(monkeypatch, module)
    import pandas as pd

    overview_csv = tmp_path / "overview.md"
    cnn_opts = {
        "cnn_config_list": ["c1", "c2"],
        "cnn_err_count": 0,
        "cnn_diff_count": 0,
        "cnn_err_df": pd.DataFrame(columns=["target", "config", "golden", "current", "diff"]),
        "cnn_diff_df": pd.DataFrame(columns=["target", "config", "golden", "current", "diff"]),
    }

    module.get_overview_result({}, cnn_opts, str(overview_csv), str(tmp_path))

    body = overview_csv.read_text(encoding="utf-8")
    assert "CNN Models" in body
    assert "LLM Models" not in body
    assert "Procyon Models" not in body
    assert "failed cases" not in body
    assert "different results" not in body

    stdout = capsys.readouterr().out
    assert "Success to get_overview_result" in stdout


def test_get_overview_result_weekly_with_procyon_reports_errs_and_diffs(tmp_path, monkeypatch):
    module = _load_analyze_module(monkeypatch)
    _patch_to_markdown(monkeypatch, module)
    import pandas as pd

    def _df(rows):
        df = pd.DataFrame(columns=["target", "config", "golden", "current", "diff"])
        for r in rows:
            df.loc[len(df)] = r
        return df

    overview_csv = tmp_path / "overview.md"
    llm_opts = {
        "llm_model_list": module.llm_model_list,
        "llm_config_list": module.llm_config_list,
        "llm_err_count": 1,
        "llm_diff_count": 1,
        "llm_err_df": _df(
            [{"target": "opt-125m", "config": "gptq", "golden": "1.0", "current": "ERROR", "diff": "ERROR"}]
        ),
        "llm_diff_df": _df(
            [
                {
                    "target": "opt-125m",
                    "config": "smooth_quant",
                    "golden": "1.0",
                    "current": "1.2",
                    "diff": "rises by 0.2",
                }
            ]
        ),
    }
    cnn_opts = {
        "cnn_config_list": ["bfp", "crypto_mode"],
        "cnn_err_count": 1,
        "cnn_diff_count": 1,
        "cnn_err_df": _df([{"target": "acc1", "config": "bfp", "golden": "75.0", "current": "ERROR", "diff": "ERROR"}]),
        "cnn_diff_df": _df(
            [{"target": "acc5", "config": "crypto_mode", "golden": "90.0", "current": "91.0", "diff": "rises by 1.0"}]
        ),
        "procyon_err_count": 1,
        "procyon_diff_count": 1,
        "procyon_err_df": _df(
            [{"target": "log", "config": "detr_resnet50", "golden": "-", "current": "MISSING", "diff": "ERROR"}]
        ),
        "procyon_diff_df": _df(
            [{"target": "memory_gb", "config": "convnext", "golden": "12.5", "current": "20.0", "diff": "+60.00%"}]
        ),
    }

    module.get_overview_result(llm_opts, cnn_opts, str(overview_csv), str(tmp_path))

    body = overview_csv.read_text(encoding="utf-8")
    assert "LLM Models" in body
    assert "CNN Models" in body
    assert "Procyon Models" in body
    assert "tools/ci/procyon" in body
    assert "failed cases" in body
    assert "different results" in body
    # err rows from each bucket are emitted
    assert "ERROR" in body
    assert "MISSING" in body
    # diff rows from each bucket are emitted
    assert "rises by 0.2" in body
    assert "rises by 1.0" in body
    assert "+60.00%" in body


def test_get_overview_result_weekly_without_procyon(tmp_path, monkeypatch):
    module = _load_analyze_module(monkeypatch)
    _patch_to_markdown(monkeypatch, module)
    import pandas as pd

    overview_csv = tmp_path / "overview.md"
    llm_opts = {
        "llm_model_list": module.llm_model_list,
        "llm_config_list": module.llm_config_list,
        "llm_err_count": 0,
        "llm_diff_count": 0,
        "llm_err_df": pd.DataFrame(columns=["target", "config", "golden", "current", "diff"]),
        "llm_diff_df": pd.DataFrame(columns=["target", "config", "golden", "current", "diff"]),
    }
    cnn_opts = {
        "cnn_config_list": ["bfp"],
        "cnn_err_count": 0,
        "cnn_diff_count": 0,
        "cnn_err_df": pd.DataFrame(columns=["target", "config", "golden", "current", "diff"]),
        "cnn_diff_df": pd.DataFrame(columns=["target", "config", "golden", "current", "diff"]),
    }

    module.get_overview_result(llm_opts, cnn_opts, str(overview_csv), str(tmp_path))

    body = overview_csv.read_text(encoding="utf-8")
    assert "LLM Models" in body
    assert "CNN Models" in body
    assert "Procyon Models" not in body
    assert "failed cases" not in body
    assert "different results" not in body
