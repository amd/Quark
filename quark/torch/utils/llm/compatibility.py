#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import transformers
from packaging import version

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

# Mapping of deprecated APIs to their max supported transformers versions
DEPRECATED_API_MAX_VERSION: dict[str, str] = {
    "seen_tokens": "4.53.3",
    "get_max_length": "4.48.3",
    "get_usable_length": "4.53.3",
}


@dataclass
class CompatibilityIssue:
    """Describes a compatibility issue found during checks."""

    category: str  # "api_change", "version_mismatch"
    message: str
    suggestion: str


class TransformersCompatibilityChecker:
    """
    Compatibility checker for Transformers models.

    Detects compatibility issues between models and Transformers versions.
    """

    def __init__(self) -> None:
        self.issues: list[CompatibilityIssue] = []
        self.transformers_version = transformers.__version__

    def _get_version_hint(self, model: nn.Module) -> str:
        """Get version hint based on model's config.transformers_version."""
        if hasattr(model, "config") and hasattr(model.config, "transformers_version"):
            return f"Use transformers=={model.config.transformers_version} as specified in model's config.json"
        return "Use a transformers version compatible with this model"

    def check_model_compatibility(self, model: nn.Module) -> list[CompatibilityIssue]:
        """Check model compatibility with current Transformers version."""
        self.issues = []
        self._check_deprecated_api_usage(model)

        if hasattr(model, "config") and getattr(model.config, "quantization_config", None) is not None:
            logger.info(
                "Skipping dry-run forward pass: the model is a "
                "pre-quantized model. A dry-run forward would trigger "
                "in-place weight decompression in compressed modules "
                "(e.g. compressed-tensors quantized linear), which would irreversibly change "
                "the original pre-quantized weights from low-precision "
                "(e.g. FP8) to high-precision (e.g. bfloat16) in-place"
            )
        else:
            self._dry_run_check(model)

        return self.issues

    def _check_deprecated_api_usage(self, model: nn.Module) -> None:
        """Check if model uses deprecated APIs that are not supported in current transformers version."""
        current_version = version.parse(self.transformers_version)
        checked_classes: set[type] = set()
        found_apis: set[str] = set()

        for module in model.modules():
            cls = module.__class__
            if cls in checked_classes:
                continue
            checked_classes.add(cls)

            try:
                source = inspect.getsource(cls)
                tree = ast.parse(source)
            except (OSError, TypeError, SyntaxError):
                continue

            # Find all attribute accesses in actual code (not comments)
            used_apis: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    used_apis.add(node.attr)

            for api, max_version in DEPRECATED_API_MAX_VERSION.items():
                if api in used_apis and current_version > version.parse(max_version) and api not in found_apis:
                    found_apis.add(api)
                    self.issues.append(
                        CompatibilityIssue(
                            category="api_change",
                            message=f"Model may use deprecated API '{api}' (not supported in transformers>{max_version})",
                            suggestion=(
                                f"Suggestions:\n"
                                f"  1. Use transformers<={max_version} which supports '{api}'\n"
                                f"  2. Update model code to use new API (get_seq_length())\n"
                            ),
                        )
                    )

    def _dry_run_check(self, model: nn.Module) -> None:
        """Execute a dry-run forward pass to catch runtime errors."""
        try:
            logger.info("Running compatibility dry-run test...")
            model.eval()

            # Determine model device
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = torch.device("cpu")

            # Prepare dummy inputs
            if hasattr(model, "dummy_inputs"):
                dummy_inputs = {
                    k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in model.dummy_inputs.items()
                }
            else:
                dummy_inputs = {"input_ids": torch.ones((1, 8), dtype=torch.long, device=model_device)}

            with torch.no_grad():
                _ = model(**dummy_inputs)

            logger.info("Dry-run test passed")

        except AttributeError as e:
            error_msg = str(e)
            version_hint = self._get_version_hint(model)

            # Find which deprecated API caused the error
            matched_api = next((api for api in DEPRECATED_API_MAX_VERSION if api in error_msg), None)
            if matched_api:
                max_version = DEPRECATED_API_MAX_VERSION[matched_api]
                self.issues.append(
                    CompatibilityIssue(
                        category="api_change",
                        message=f"Deprecated API '{matched_api}' caused runtime error (not supported in transformers {self.transformers_version})",
                        suggestion=(
                            f"Suggestions:\n"
                            f"  1. Use transformers<={max_version} which supports '{matched_api}'\n"
                            f"  2. Update model code to use new API (get_seq_length())\n"
                        ),
                    )
                )
            else:
                self.issues.append(
                    CompatibilityIssue(
                        category="api_change",
                        message=f"API compatibility error: {error_msg}",
                        suggestion=f"The model may be incompatible with Transformers {self.transformers_version}. "
                        f"Try: {version_hint}",
                    )
                )

        except Exception as e:
            self.issues.append(
                CompatibilityIssue(
                    category="version_mismatch",
                    message=f"Model forward failed during compatibility check: {type(e).__name__}",
                    suggestion=f"This may indicate a compatibility issue. {self._get_version_hint(model)}",
                )
            )

    def print_issues(self) -> None:
        """Print all compatibility issues."""
        if not self.issues:
            logger.info("No compatibility issues found")
            return

        logger.warning(f"Found {len(self.issues)} compatibility issue(s):")
        for i, issue in enumerate(self.issues, 1):
            logger.warning(f"Issue {i} [{issue.category}]\n   {issue.message}\n   {issue.suggestion}")


def check_compatibility_before_quantization(model: nn.Module, raise_on_error: bool = False) -> bool:
    """
    Check model compatibility before quantization.

    Args:
        model: Model to check
        raise_on_error: Whether to raise exception on issues

    Returns:
        is_compatible: Whether the model is compatible

    Raises:
        RuntimeError: If raise_on_error=True and issues are found
    """
    checker = TransformersCompatibilityChecker()
    issues = checker.check_model_compatibility(model)

    if issues:
        checker.print_issues()
        if raise_on_error:
            raise RuntimeError("Model compatibility check failed:\n" + "\n".join(issue.message for issue in issues))

    return len(issues) == 0
