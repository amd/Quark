#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import importlib
import sys
from pathlib import Path


def _load_timm_module():
    """Load the analyze_timm_results module."""
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    sys.modules.pop("tools.ci.analyze_timm_results", None)
    return importlib.import_module("tools.ci.analyze_timm_results")


def _write_analyze_log(path: Path, total: int, failed: int, failed_models: list[str]) -> None:
    """Write a mock analyze_timm.log file."""
    lines = []
    for idx, model_name in enumerate(failed_models, start=1):
        lines.append(f"index: {idx} name: {model_name}  save log to: /path/to/{model_name}.txt")
    lines.append(f"Total model: {total}  and amond them  {failed}  model failed")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_analyze_log_with_failures(tmp_path):
    """Test parsing analyze_timm.log with failed models."""
    module = _load_timm_module()
    log_file = tmp_path / "analyze_timm.log"

    # Write log with 3 failed models out of 100 total
    _write_analyze_log(log_file, total=100, failed=3, failed_models=["resnet50", "vgg16", "efficientnet_b0"])

    total, failed, models = module.parse_analyze_log(str(log_file))

    assert total == 100
    assert failed == 3
    assert len(models) == 3
    assert models == ["resnet50", "vgg16", "efficientnet_b0"]


def test_parse_analyze_log_no_failures(tmp_path):
    """Test parsing analyze_timm.log with no failed models."""
    module = _load_timm_module()
    log_file = tmp_path / "analyze_timm.log"

    # Write log with 0 failed models
    _write_analyze_log(log_file, total=50, failed=0, failed_models=[])

    total, failed, models = module.parse_analyze_log(str(log_file))

    assert total == 50
    assert failed == 0
    assert len(models) == 0
    assert models == []


def test_parse_analyze_log_file_not_exist(tmp_path, capsys):
    """Test parsing when log file doesn't exist."""
    module = _load_timm_module()
    log_file = tmp_path / "nonexistent.log"

    total, failed, models = module.parse_analyze_log(str(log_file))

    assert total == 0
    assert failed == 0
    assert models == []

    captured = capsys.readouterr()
    assert "does not exist" in captured.out


def test_parse_analyze_log_empty_file(tmp_path):
    """Test parsing an empty log file."""
    module = _load_timm_module()
    log_file = tmp_path / "empty.log"
    log_file.write_text("", encoding="utf-8")

    total, failed, models = module.parse_analyze_log(str(log_file))

    assert total == 0
    assert failed == 0
    assert models == []


def test_parse_analyze_log_count_mismatch(tmp_path, capsys):
    """Test parsing when summary count doesn't match actual failed model entries."""
    module = _load_timm_module()
    log_file = tmp_path / "mismatch.log"

    # Write 2 failed model entries but summary says 3
    lines = [
        "index: 1 name: model_a  save log to: /path/to/model_a.txt",
        "index: 2 name: model_b  save log to: /path/to/model_b.txt",
        "Total model: 100  and amond them  3  model failed",  # Mismatch: says 3 but only 2 entries
    ]
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total, failed, models = module.parse_analyze_log(str(log_file))

    assert total == 100
    assert failed == 3  # From summary line
    assert len(models) == 2  # Actual entries

    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "Summary shows 3 failed models, but found 2 in log" in captured.out


def test_parse_analyze_log_with_extra_content(tmp_path):
    """Test parsing log with extra unrelated content."""
    module = _load_timm_module()
    log_file = tmp_path / "extra_content.log"

    lines = [
        "Some random log output",
        "index: 1 name: model_x  save log to: /path/to/model_x.txt",
        "More random content",
        "index: 2 name: model_y  save log to: /path/to/model_y.txt",
        "Even more content",
        "Total model: 200  and amond them  2  model failed",
        "Trailing content after summary",
    ]
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total, failed, models = module.parse_analyze_log(str(log_file))

    assert total == 200
    assert failed == 2
    assert models == ["model_x", "model_y"]


def test_generate_html_body_with_failures():
    """Test HTML generation with failed models."""
    module = _load_timm_module()

    html = module.generate_html_body(
        total_models=100,
        failed_count=3,
        failed_models=["resnet50", "vgg16", "efficientnet_b0"],
        run_url="https://github.com/test/repo/actions/runs/12345",
    )

    # Check basic structure
    assert "<!DOCTYPE html>" in html
    assert "<html>" in html
    assert "</html>" in html

    # Check content
    assert "Weekly Timm Examples Regression Test Results" in html
    assert "Total Models:</strong> 100" in html
    assert "Passed Models:</strong> 97" in html
    assert "Failed Models:</strong> 3" in html
    assert "Success Rate:</strong> 97.0%" in html
    assert "❌ 3 model(s) failed" in html
    assert "status-fail" in html

    # Check failed models list
    assert "❌ Failed Models" in html
    assert "<li>resnet50</li>" in html
    assert "<li>vgg16</li>" in html
    assert "<li>efficientnet_b0</li>" in html
    assert "Please download the artifacts" in html

    # Check URL
    assert 'href="https://github.com/test/repo/actions/runs/12345"' in html
    assert "View Full Results and Logs" in html

    # Should not have success message
    assert "Congratulations" not in html


def test_generate_html_body_all_passed():
    """Test HTML generation when all models passed."""
    module = _load_timm_module()

    html = module.generate_html_body(
        total_models=50, failed_count=0, failed_models=[], run_url="https://github.com/test/repo/actions/runs/67890"
    )

    # Check basic structure
    assert "<!DOCTYPE html>" in html
    assert "<html>" in html
    assert "</html>" in html

    # Check content
    assert "Total Models:</strong> 50" in html
    assert "Passed Models:</strong> 50" in html
    assert "Failed Models:</strong> 0" in html
    assert "Success Rate:</strong> 100.0%" in html
    assert "✅ All models passed" in html
    assert "status-pass" in html

    # Check success message
    assert "🎉 Congratulations! All models passed the quantization test." in html

    # Should not have failed models section
    assert "❌ Failed Models" not in html
    assert "Please download the artifacts" not in html


def test_generate_html_body_zero_total_models():
    """Test HTML generation with zero total models (edge case)."""
    module = _load_timm_module()

    html = module.generate_html_body(
        total_models=0, failed_count=0, failed_models=[], run_url="https://github.com/test/repo/actions/runs/11111"
    )

    # Should handle division by zero gracefully
    assert "Success Rate:</strong> 0.0%" in html
    assert "Total Models:</strong> 0" in html


def test_generate_html_body_escaping():
    """Test that model names with special characters are properly handled."""
    module = _load_timm_module()

    # Model names with potential HTML-breaking characters
    html = module.generate_html_body(
        total_models=10,
        failed_count=2,
        failed_models=["model<script>", "model&test"],
        run_url="https://github.com/test/repo/actions/runs/99999",
    )

    # Model names should appear in the HTML
    # Note: We're not doing HTML escaping in the current implementation,
    # but the test documents current behavior
    assert "model<script>" in html or "model&lt;script&gt;" in html
    assert "model&test" in html or "model&amp;test" in html


def test_main_end_to_end(tmp_path, monkeypatch, capsys):
    """Test the main() function end-to-end."""
    module = _load_timm_module()

    # Prepare input files
    analyze_log = tmp_path / "analyze_timm.log"
    _write_analyze_log(
        analyze_log, total=150, failed=5, failed_models=["model1", "model2", "model3", "model4", "model5"]
    )

    output_html = tmp_path / "output" / "email.html"
    run_url = "https://github.com/test/repo/actions/runs/123"

    # Mock sys.argv
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_timm_results.py",
            "--analyze_log",
            str(analyze_log),
            "--output_html",
            str(output_html),
            "--run_url",
            run_url,
        ],
    )

    # Run main
    exit_code = module.main()

    # Check exit code
    assert exit_code == 0

    # Check output file was created
    assert output_html.exists()

    # Check HTML content
    html_content = output_html.read_text(encoding="utf-8")
    assert "Total Models:</strong> 150" in html_content
    assert "Failed Models:</strong> 5" in html_content
    assert "<li>model1</li>" in html_content
    assert "<li>model5</li>" in html_content

    # Check console output
    captured = capsys.readouterr()
    assert "Total models: 150" in captured.out
    assert "Failed models: 5" in captured.out
    assert "Failed model list: model1, model2, model3, model4, model5" in captured.out
    assert "Email HTML written to:" in captured.out


def test_main_all_passed(tmp_path, monkeypatch, capsys):
    """Test main() when all models pass."""
    module = _load_timm_module()

    analyze_log = tmp_path / "analyze_timm.log"
    _write_analyze_log(analyze_log, total=75, failed=0, failed_models=[])

    output_html = tmp_path / "email.html"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_timm_results.py",
            "--analyze_log",
            str(analyze_log),
            "--output_html",
            str(output_html),
            "--run_url",
            "https://github.com/test/repo/actions/runs/456",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    assert output_html.exists()

    html_content = output_html.read_text(encoding="utf-8")
    assert "✅ All models passed" in html_content
    assert "🎉 Congratulations!" in html_content

    captured = capsys.readouterr()
    assert "Total models: 75" in captured.out
    assert "Failed models: 0" in captured.out
    # Should not have "Failed model list" line
    assert "Failed model list:" not in captured.out


def test_main_creates_nested_directories(tmp_path, monkeypatch):
    """Test that main() creates nested output directories."""
    module = _load_timm_module()

    analyze_log = tmp_path / "analyze_timm.log"
    _write_analyze_log(analyze_log, total=10, failed=1, failed_models=["test_model"])

    # Output to a deeply nested path that doesn't exist yet
    output_html = tmp_path / "deeply" / "nested" / "path" / "email.html"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_timm_results.py",
            "--analyze_log",
            str(analyze_log),
            "--output_html",
            str(output_html),
            "--run_url",
            "https://example.com",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    assert output_html.exists()
    assert output_html.parent.is_dir()


def test_main_with_missing_log_file(tmp_path, monkeypatch, capsys):
    """Test main() when analyze log file is missing."""
    module = _load_timm_module()

    analyze_log = tmp_path / "missing.log"  # File doesn't exist
    output_html = tmp_path / "email.html"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_timm_results.py",
            "--analyze_log",
            str(analyze_log),
            "--output_html",
            str(output_html),
            "--run_url",
            "https://example.com",
        ],
    )

    exit_code = module.main()

    # Should still complete successfully with zeros
    assert exit_code == 0
    assert output_html.exists()

    html_content = output_html.read_text(encoding="utf-8")
    assert "Total Models:</strong> 0" in html_content

    captured = capsys.readouterr()
    assert "does not exist" in captured.out
    assert "Total models: 0" in captured.out
