#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Tests for quark.torch.utils.llm.compatibility module."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from quark.torch.utils.llm.compatibility import (
    CompatibilityIssue,
    TransformersCompatibilityChecker,
    check_compatibility_before_quantization,
)


# =============================================================================
# Test Models
# =============================================================================
class SimpleModel(nn.Module):
    """A simple model without Transformers config."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ModelWithDummyInputs(nn.Module):
    """A model that provides dummy_inputs."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    @property
    def dummy_inputs(self):
        return {"x": torch.randn(1, 64)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ModelWithDeprecatedAPI(nn.Module):
    """A model that uses deprecated seen_tokens API in source code."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor, cache=None) -> torch.Tensor:
        # Actual attribute access that AST can detect
        if cache is not None:
            _ = cache.seen_tokens
        return self.linear(x)


class ModelWithGetMaxLength(nn.Module):
    """A model that uses deprecated get_max_length API."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor, cache=None) -> torch.Tensor:
        # Actual attribute access that AST can detect
        if cache is not None:
            _ = cache.get_max_length()
        return self.linear(x)


# =============================================================================
# Test: TransformersCompatibilityChecker
# =============================================================================
class TestTransformersCompatibilityChecker:
    """Tests for TransformersCompatibilityChecker class."""

    def test_init(self):
        """Test checker initialization."""
        checker = TransformersCompatibilityChecker()
        assert checker.issues == []
        assert checker.transformers_version is not None

    def test_check_model_compatibility_with_dummy_inputs(self):
        """Test compatibility check with a model that has dummy_inputs."""
        checker = TransformersCompatibilityChecker()
        model = ModelWithDummyInputs()

        issues = checker.check_model_compatibility(model)

        assert len(issues) == 0

    def test_print_issues_no_issues(self, capsys):
        """Test print_issues when no issues found."""
        checker = TransformersCompatibilityChecker()
        checker.issues = []

        checker.print_issues()
        # Should not raise any errors

    def test_print_issues_with_issues(self, capsys):
        """Test print_issues when issues are present."""
        checker = TransformersCompatibilityChecker()
        checker.issues = [
            CompatibilityIssue(
                category="api_change",
                message="Test issue 1",
                suggestion="Test suggestion 1",
            ),
            CompatibilityIssue(
                category="version_mismatch",
                message="Test issue 2",
                suggestion="Test suggestion 2",
            ),
        ]

        checker.print_issues()
        # Should not raise any errors

    def test_dry_run_check_success(self):
        """Test dry-run check with successful forward pass."""
        checker = TransformersCompatibilityChecker()
        model = ModelWithDummyInputs()

        checker._dry_run_check(model)

        assert len(checker.issues) == 0

    def test_dry_run_check_attribute_error(self):
        """Test dry-run check catches AttributeError."""
        checker = TransformersCompatibilityChecker()

        class AttrErrorModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, **kwargs):
                raise AttributeError("'Cache' object has no attribute 'seen_tokens'")

        model = AttrErrorModel()
        checker._dry_run_check(model)

        assert len(checker.issues) >= 1

    def test_dry_run_check_generic_error(self):
        """Test dry-run check handles generic errors."""
        checker = TransformersCompatibilityChecker()

        class ErrorModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, **kwargs):
                raise RuntimeError("Some error during forward")

        model = ErrorModel()
        checker._dry_run_check(model)

        assert len(checker.issues) >= 1

    def test_check_deprecated_api_usage_detects_seen_tokens(self):
        """Test that _check_deprecated_api_usage detects seen_tokens when version > 4.53.3."""
        checker = TransformersCompatibilityChecker()
        checker.transformers_version = "4.54.0"  # Higher than max supported 4.53.3

        model = ModelWithDeprecatedAPI()
        checker._check_deprecated_api_usage(model)

        assert len(checker.issues) == 1
        assert "seen_tokens" in checker.issues[0].message
        assert "4.53.3" in checker.issues[0].suggestion

    def test_check_deprecated_api_usage_detects_get_max_length(self):
        """Test that _check_deprecated_api_usage detects get_max_length when version > 4.48.3."""
        checker = TransformersCompatibilityChecker()
        checker.transformers_version = "4.49.0"  # Higher than max supported 4.48.3

        model = ModelWithGetMaxLength()
        checker._check_deprecated_api_usage(model)

        assert len(checker.issues) == 1
        assert "get_max_length" in checker.issues[0].message
        assert "4.48.3" in checker.issues[0].suggestion

    def test_check_deprecated_api_usage_no_issue_when_version_supported(self):
        """Test that no issue is raised when transformers version is supported."""
        checker = TransformersCompatibilityChecker()
        checker.transformers_version = "4.48.0"  # Lower than max supported for get_max_length

        model = ModelWithGetMaxLength()
        checker._check_deprecated_api_usage(model)

        assert len(checker.issues) == 0


# =============================================================================
# Test: check_compatibility_before_quantization function
# =============================================================================
class TestCheckCompatibilityBeforeQuantization:
    """Tests for check_compatibility_before_quantization function."""

    def test_compatible_model(self):
        """Test function returns True for compatible model."""
        model = ModelWithDummyInputs()

        result = check_compatibility_before_quantization(model)

        assert result is True

    @patch.object(TransformersCompatibilityChecker, "check_model_compatibility")
    def test_raise_on_error_true_with_issues(self, mock_check):
        """Test function raises RuntimeError when raise_on_error=True and issues found."""
        mock_check.return_value = [
            CompatibilityIssue(
                category="api_change",
                message="Test error",
                suggestion="Test suggestion",
            )
        ]

        model = SimpleModel()

        with pytest.raises(RuntimeError) as exc_info:
            check_compatibility_before_quantization(model, raise_on_error=True)

        assert "Test error" in str(exc_info.value)

    @patch.object(TransformersCompatibilityChecker, "check_model_compatibility")
    def test_no_raise_when_compatible(self, mock_check):
        """Test function does not raise when model is compatible."""
        mock_check.return_value = []

        model = SimpleModel()

        result = check_compatibility_before_quantization(model, raise_on_error=True)
        assert result is True


# =============================================================================
# Test: Integration tests
# =============================================================================
class TestIntegration:
    """Integration tests for the compatibility module."""

    def test_full_workflow_with_dummy_inputs(self):
        """Test full workflow with a model that has dummy_inputs."""
        model = ModelWithDummyInputs()
        checker = TransformersCompatibilityChecker()

        issues = checker.check_model_compatibility(model)
        checker.print_issues()

        assert len(issues) == 0

    def test_checker_reuse(self):
        """Test that checker can be reused for multiple models."""
        checker = TransformersCompatibilityChecker()

        model1 = ModelWithDummyInputs()
        model2 = ModelWithDummyInputs()

        issues1 = checker.check_model_compatibility(model1)
        issues2 = checker.check_model_compatibility(model2)

        assert len(issues1) == 0
        assert len(issues2) == 0
