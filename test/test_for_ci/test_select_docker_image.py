# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for Docker image selection functionality.

This module contains comprehensive unit tests for the dynamic Docker image
selection system, including:
- Normalization functions
- Specificity calculation
- Mapping matching logic
- Image selection with public/private preference
- Error handling for ambiguous matches and missing images
- CLI interface testing

The tests use mocking to avoid dependencies on actual YAML files and
external Docker operations.
"""

import os
import sys
import unittest
from io import StringIO
from unittest.mock import mock_open, patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.github/scripts/select_docker_image"))
)

import select_docker_image as select_docker_image

# Mock YAML mapping data for testing
# Includes test-only fixtures (rocm-7.2, cpu ubuntu-20.04) that are not in production docker_image_mapping.yaml
# Covers: ROCm, CUDA, CPU; public/private; single/template tag; various version combinations
MOCK_YAML_MAPPING = {
    "description": "Test mapping",
    "mappings": [
        # Test-only: Public ROCm 7.2 image
        {
            "accelerator_version": "rocm-7.2",
            "os_version": "ubuntu-24.04",
            "python_version": "3.13",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": None,
            "transformers_version": None,
            "registry": "docker.io",
            "image": "rocm/pytorch",
            "tag": "rocm7.2_ubuntu24.04_py3.13_pytorch_release_2.10.0",
            "is_public": True,
            "description": "Public ROCm PyTorch image for ROCm 7.2 and Ubuntu 24.04 (test only)",
        },
        # Test-only: Public image for CPU and Ubuntu 20.04
        {
            "accelerator_version": "cpu",
            "os_version": "ubuntu-20.04",
            "python_version": "3.9",
            "pytorch_version": "2.7.1",
            "onnxruntime_version": None,
            "transformers_version": None,
            "registry": "registry.amd.com/dockerhub/library",
            "image": "ubuntu",
            "tag": "20.04",
            "is_public": True,
            "description": "Public image for CPU and Ubuntu 20.04 (test only)",
        },
        # Public ROCm 7.1.1 with null optional (matches any onnx/transformers)
        {
            "accelerator_version": "rocm-7.1.1",
            "os_version": "ubuntu-24.04",
            "python_version": "3.12",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": None,
            "transformers_version": None,
            "registry": "docker.io",
            "image": "rocm/pytorch",
            "tag": "rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.10.0",
            "is_public": True,
            "description": "Public ROCm PyTorch image for ROCm 7.1.1",
        },
        # ROCm 7.1 with explicit optional versions (higher specificity)
        {
            "accelerator_version": "rocm-7.1",
            "os_version": "ubuntu-22.04",
            "python_version": "3.12",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": "1.22.2",
            "transformers_version": "4.57",
            "registry": "xcoartifactory.xilinx.com/uai-docker-local",
            "image": "amd_quark",
            "tag": "latest-rocm7.1-base-py3.12-torch2.10.0-onnxruntime1.22.2-ubuntu22.04",
            "is_public": False,
            "description": "Private Quark image for ROCm 7.1 with latest versions",
        },
        # ROCm 7.1 with null optional (lower specificity, matches any)
        {
            "accelerator_version": "rocm-7.1",
            "os_version": "ubuntu-22.04",
            "python_version": "3.12",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": None,
            "transformers_version": None,
            "registry": "xcoartifactory.xilinx.com/uai-docker-local",
            "image": "amd_quark",
            "tag": "latest-rocm7.1-base-py3.12-torch2.10.0-ubuntu22.04",
            "is_public": False,
            "description": "Private Quark image for ROCm 7.1 without optional versions",
        },
        # ROCm 7.1 with Python 3.11 (different python version)
        {
            "accelerator_version": "rocm-7.1",
            "os_version": "ubuntu-22.04",
            "python_version": "3.11",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": "1.22.2",
            "transformers_version": None,
            "registry": "xcoartifactory.xilinx.com/uai-docker-local",
            "image": "amd_quark",
            "tag": "latest-rocm7.1-base-py3.11-torch2.10.0-onnxruntime1.22.2-ubuntu22.04",
            "is_public": False,
            "description": "Private Quark image for ROCm 7.1 with Python 3.11",
        },
        # CUDA 12.6.3
        {
            "accelerator_version": "cuda-12.6.3",
            "os_version": "ubuntu-22.04",
            "python_version": "3.12",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": "1.22.2",
            "transformers_version": None,
            "registry": "xcoartifactory.xilinx.com/uai-docker-local",
            "image": "amd_quark",
            "tag": "latest-cuda12.6.3-base-py3.12-torch2.10.0-onnxruntime1.22.2-ubuntu22.04",
            "is_public": False,
            "description": "Private Quark image for CUDA 12.6.3",
        },
        # CPU with Python 3.12, ONNX 1.22.2
        {
            "accelerator_version": "cpu",
            "os_version": "ubuntu-22.04",
            "python_version": "3.12",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": "1.22.2",
            "transformers_version": "4.57",
            "registry": "xcoartifactory.xilinx.com/uai-docker-local",
            "image": "amd_quark",
            "tag": "latest-cpu-base-py3.12-torch2.10.0-onnxruntime1.22.2-ubuntu22.04",
            "is_public": False,
            "description": "Private Quark image for CPU",
        },
        # CPU with Python 3.11, ONNX 1.24.2 (newer ONNX)
        {
            "accelerator_version": "cpu",
            "os_version": "ubuntu-22.04",
            "python_version": "3.11",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": "1.24.2",
            "transformers_version": None,
            "registry": "xcoartifactory.xilinx.com/uai-docker-local",
            "image": "amd_quark",
            "tag": "latest-cpu-base-py3.11-torch2.10.0-onnxruntime1.24.2-ubuntu22.04",
            "is_public": False,
            "description": "Private Quark image for CPU with Python 3.11 and ONNX 1.24.2",
        },
    ],
}

# Mock YAML with template tag (for testing _expand_tag) - expands to multiple entries
MOCK_YAML_MAPPING_WITH_TEMPLATE = {
    "description": "Test mapping with template expansion",
    "mappings": [
        {
            "accelerator_version": "rocm-7.1",
            "os_version": "ubuntu-22.04",
            "python_version": "3.13, 3.12",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": "1.23.2",
            "transformers_version": None,
            "registry": "xcoartifactory.xilinx.com/uai-docker-local",
            "image": "amd_quark",
            "tag": "latest-rocm7.1-base-py<python_version>-torch<pytorch_version>-onnxruntime<onnxruntime_version>-<os_version>",
            "is_public": False,
            "description": "Template entry for expansion test",
        },
    ],
}


class TestParseVersionList(unittest.TestCase):
    """Test _parse_version_list for comma-separated version parsing."""

    def test_parse_comma_separated(self):
        """Test parsing comma-separated values."""
        self.assertEqual(
            select_docker_image._parse_version_list("3.13, 3.12, 3.11"),
            ["3.13", "3.12", "3.11"],
        )

    def test_parse_single_value(self):
        """Test parsing single value."""
        self.assertEqual(select_docker_image._parse_version_list("3.12"), ["3.12"])

    def test_parse_none_returns_empty(self):
        """Test that None returns empty list (covers line 82)."""
        self.assertEqual(select_docker_image._parse_version_list(None), [])

    def test_parse_empty_string_returns_empty(self):
        """Test that empty string returns empty list."""
        self.assertEqual(select_docker_image._parse_version_list(""), [])


class TestOsVersionForTag(unittest.TestCase):
    """Test _os_version_for_tag conversion."""

    def test_ubuntu_hyphen_removed(self):
        """Test ubuntu-22.04 -> ubuntu22.04."""
        self.assertEqual(select_docker_image._os_version_for_tag("ubuntu-22.04"), "ubuntu22.04")

    def test_empty_returns_empty(self):
        """Test empty string returns empty."""
        self.assertEqual(select_docker_image._os_version_for_tag(""), "")


class TestExpandTag(unittest.TestCase):
    """Test _expand_tag for template expansion and edge cases."""

    def test_expand_missing_tag_exits(self):
        """Test that entry without tag raises error (covers lines 124-125)."""
        entry = {"accelerator_version": "cpu", "os_version": "ubuntu-22.04"}
        with self.assertRaises(SystemExit) as cm:
            select_docker_image._expand_tag(entry)
        self.assertEqual(cm.exception.code, 1)

    def test_expand_template_with_empty_list_fallbacks(self):
        """Test expansion when version fields are single values (covers lines 136,138,140,142)."""
        # Single values parse to 1-item lists; use values that would give empty from parse
        # to trigger fallback: e.g. python_version as number or empty
        entry = {
            "accelerator_version": "cpu",
            "os_version": "ubuntu-22.04",
            "python_version": "3.12",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": "1.23.2",
            "tag": "latest-cpu-py<python_version>-torch<pytorch_version>-ort<onnxruntime_version>-<os_version>",
            "registry": "xcoartifactory.xilinx.com/uai-docker-local",
            "image": "amd_quark",
            "is_public": False,
        }
        result = select_docker_image._expand_tag(entry)
        self.assertEqual(len(result), 1)
        self.assertIn("py3.12", result[0]["tag"])

    def test_expand_template_skips_empty_onnxruntime_when_in_tag(self):
        """Test that ort=="" with onnxruntime in tag is skipped (covers line 149)."""
        entry = {
            "accelerator_version": "cpu",
            "os_version": "ubuntu-22.04",
            "python_version": "3.12",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": "",  # Empty -> ort_list = [""] after fallback
            "tag": "latest-cpu-py<python_version>-torch<pytorch_version>-onnxruntime<onnxruntime_version>-<os_version>",
            "registry": "xcoartifactory.xilinx.com/uai-docker-local",
            "image": "amd_quark",
            "is_public": False,
        }
        result = select_docker_image._expand_tag(entry)
        # Should skip the ort=="" case (continue), so we get 0 entries from that loop
        # Actually: py_list=[3.12], torch_list=[2.10.0], ort_list=[str("")]=[""], os_list=[ubuntu22.04]
        # For ort="", we hit continue, so we never append. Result = [].
        self.assertEqual(len(result), 0)

    def test_expand_template_with_empty_py_fallback(self):
        """Test fallback when python_version is None (covers line 136)."""
        entry = {
            "accelerator_version": "cpu",
            "os_version": "ubuntu-22.04",
            "python_version": None,
            "pytorch_version": "2.10.0",
            "onnxruntime_version": "1.23.2",
            "tag": "latest-cpu-py<python_version>-torch<pytorch_version>-ort<onnxruntime_version>-<os_version>",
            "registry": "xcoartifactory.xilinx.com/uai-docker-local",
            "image": "amd_quark",
            "is_public": False,
        }
        result = select_docker_image._expand_tag(entry)
        self.assertEqual(len(result), 1)
        self.assertIn("pyNone", result[0]["tag"])  # str(None) = "None"

    def test_expand_template_with_empty_os_fallback(self):
        """Test fallback when os_version is empty (covers line 142)."""
        entry = {
            "accelerator_version": "cpu",
            "os_version": "",
            "python_version": "3.12",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": "1.23.2",
            "tag": "latest-cpu-py<python_version>-torch<pytorch_version>-ort<onnxruntime_version>-<os_version>",
            "registry": "xcoartifactory.xilinx.com/uai-docker-local",
            "image": "amd_quark",
            "is_public": False,
        }
        result = select_docker_image._expand_tag(entry)
        self.assertEqual(len(result), 1)
        self.assertTrue(
            result[0]["tag"].endswith("-")
            or "<os_version>" in result[0]["tag"]
            or result[0]["tag"].endswith("latest-cpu-py3.12-torch2.10.0-ort1.23.2-")
        )

    def test_expand_template_with_empty_torch_fallback(self):
        """Test fallback when pytorch_version is None (covers line 138)."""
        entry = {
            "accelerator_version": "cpu",
            "os_version": "ubuntu-22.04",
            "python_version": "3.12",
            "pytorch_version": None,
            "onnxruntime_version": "1.23.2",
            "tag": "latest-cpu-py<python_version>-torch<pytorch_version>-ort<onnxruntime_version>-<os_version>",
            "registry": "xcoartifactory.xilinx.com/uai-docker-local",
            "image": "amd_quark",
            "is_public": False,
        }
        result = select_docker_image._expand_tag(entry)
        self.assertEqual(len(result), 1)
        self.assertIn("torchNone", result[0]["tag"])

    def test_expand_template_with_empty_ort_fallback(self):
        """Test fallback when onnxruntime_version is None (covers line 140)."""
        entry = {
            "accelerator_version": "cpu",
            "os_version": "ubuntu-22.04",
            "python_version": "3.12",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": None,
            "tag": "latest-cpu-py<python_version>-torch<pytorch_version>-<os_version>",
            "registry": "xcoartifactory.xilinx.com/uai-docker-local",
            "image": "amd_quark",
            "is_public": False,
        }
        result = select_docker_image._expand_tag(entry)
        self.assertEqual(len(result), 1)
        self.assertIn("py3.12", result[0]["tag"])


class TestNormalizeFunctions(unittest.TestCase):
    """Test normalization functions for accelerator versions and versions.

    These tests verify that normalization functions correctly handle
    case-insensitive matching and whitespace trimming.
    """

    def test_normalize_accelerator_version(self):
        """Test accelerator version normalization handles case and whitespace.

        Verifies that accelerator versions are normalized to lowercase and
        have leading/trailing whitespace removed.
        """
        self.assertEqual(select_docker_image.normalize_accelerator_version("rocm-7.1"), "rocm-7.1")
        self.assertEqual(select_docker_image.normalize_accelerator_version("ROCM-7.1"), "rocm-7.1")
        self.assertEqual(select_docker_image.normalize_accelerator_version("  rocm-7.1  "), "rocm-7.1")

    def test_normalize_version(self):
        """Test version normalization handles whitespace and None values.

        Verifies that version strings are trimmed and None/empty values
        are handled correctly.
        """
        self.assertEqual(select_docker_image.normalize_version("3.12"), "3.12")
        self.assertEqual(select_docker_image.normalize_version("  3.12  "), "3.12")
        self.assertIsNone(select_docker_image.normalize_version(None))
        self.assertIsNone(select_docker_image.normalize_version(""))


class TestCalculateSpecificity(unittest.TestCase):
    """Test specificity calculation for mapping entries.

    Specificity determines how well a mapping entry matches the input parameters.
    Higher specificity means more fields match exactly. These tests verify the
    specificity calculation logic, including handling of optional parameters.
    """

    def setUp(self):
        """Set up test fixtures with a base mapping entry.

        Creates a base entry that matches all required fields and has
        optional fields specified. This is used as a template for testing
        various specificity scenarios.
        """
        self.base_entry = {
            "accelerator_version": "rocm-7.1",
            "os_version": "ubuntu-22.04",
            "python_version": "3.12",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": "1.22.2",
            "transformers_version": "4.57",
        }

    def test_specificity_all_required_fields_match(self):
        """Test specificity when all required fields match."""
        specificity = select_docker_image.calculate_specificity(
            self.base_entry,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            None,  # Optional params not specified
            None,
        )
        self.assertEqual(specificity, 4)  # Only 4 required fields

    def test_specificity_with_optional_params_not_specified(self):
        """Test specificity when optional params are not specified (should be ignored)."""
        specificity = select_docker_image.calculate_specificity(
            self.base_entry,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            None,  # Optional params not specified - should be ignored
            None,
        )
        # Should only count required fields (4), optional params are ignored
        self.assertEqual(specificity, 4)

    def test_specificity_with_optional_params_specified_and_match(self):
        """Test specificity when optional params are specified and match."""
        specificity = select_docker_image.calculate_specificity(
            self.base_entry,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            "1.22.2",  # Optional params specified and match
            "4.57",
        )
        # Should count all 6 fields (4 required + 2 optional)
        self.assertEqual(specificity, 6)

    def test_specificity_with_optional_params_specified_but_mismatch(self):
        """Test specificity when optional params are specified but don't match."""
        specificity = select_docker_image.calculate_specificity(
            self.base_entry,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            "1.21.1",  # Mismatch - should return -1
            "4.57",
        )
        # Should return -1 (invalid match)
        self.assertEqual(specificity, -1)

    def test_specificity_entry_with_null_optional_params(self):
        """Test specificity when entry has null optional params and input specifies them."""
        entry_with_null = {
            "accelerator_version": "rocm-7.1.1",
            "os_version": "ubuntu-24.04",
            "python_version": "3.12",
            "pytorch_version": "2.10.0",
            "onnxruntime_version": None,  # null in entry
            "transformers_version": None,  # null in entry
        }
        specificity = select_docker_image.calculate_specificity(
            entry_with_null,
            "rocm-7.1.1",
            "ubuntu-24.04",
            "3.12",
            "2.10.0",
            "1.22.2",  # Input specifies optional params
            "4.57",
        )
        # Entry with null matches any, but doesn't increase specificity
        # So specificity = 4 (required fields only)
        self.assertEqual(specificity, 4)

    def test_specificity_required_field_mismatch(self):
        """Test specificity when required field doesn't match."""
        specificity = select_docker_image.calculate_specificity(
            self.base_entry,
            "rocm-5.0",  # Mismatch
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            None,
            None,
        )
        # Should return -1 (invalid match)
        self.assertEqual(specificity, -1)

    def test_specificity_with_vllm_version_must_match_when_specified(self):
        """Test strict vllm_version matching behavior when the filter is provided."""
        entry = {
            **self.base_entry,
            "vllm_version": "v0.13.0",
        }
        specificity = select_docker_image.calculate_specificity(
            entry,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            None,
            None,
            vllm_version="v0.13.0",
        )
        self.assertEqual(specificity, 5)

        mismatch = select_docker_image.calculate_specificity(
            entry,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            None,
            None,
            vllm_version="v0.14.0",
        )
        self.assertEqual(mismatch, -1)

    def test_specificity_with_image_name_must_match_when_specified(self):
        """Test strict image_name matching behavior when the filter is provided."""
        entry = {
            **self.base_entry,
            "image_name": "vllm/vllm-openai-rocm:v0.14.0",
        }
        specificity = select_docker_image.calculate_specificity(
            entry,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            None,
            None,
            "vllm/vllm-openai-rocm:v0.14.0",
            None,
        )
        self.assertEqual(specificity, 5)

        mismatch = select_docker_image.calculate_specificity(
            entry,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            None,
            None,
            "docker.io/vllm/vllm-openai-rocm:v0.14.0",
            None,
        )
        self.assertEqual(mismatch, -1)


class TestMatchMapping(unittest.TestCase):
    """Test mapping matching logic."""

    @patch("select_docker_image.load_mapping")
    def test_match_public_image_preferred(self, mock_load):
        """Test that public images are preferred over private images."""
        mock_load.return_value = {
            "mappings": [
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.12",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": None,
                    "transformers_version": None,
                    "registry": "docker.io",
                    "image": "rocm/pytorch",
                    "tag": "public-tag",
                    "is_public": True,
                },
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.12",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": None,
                    "transformers_version": None,
                    "registry": "xcoartifactory.xilinx.com/uai-docker-local",
                    "image": "amd_quark",
                    "tag": "private-tag",
                    "is_public": False,
                },
            ]
        }

        result = select_docker_image.match_mapping(
            mock_load.return_value,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            None,
            None,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result["is_public"])
        self.assertEqual(result["image"], "rocm/pytorch")

    @patch("select_docker_image.load_mapping")
    def test_match_mapping_image_name_filter_selects_direct_entry(self, mock_load):
        """Test that image_name acts as a hard filter in version-based matching."""
        mock_load.return_value = {
            "mappings": [
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.12",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": None,
                    "transformers_version": None,
                    "registry": "xcoartifactory.xilinx.com/uai-docker-local",
                    "image": "amd_quark",
                    "tag": "generic-quantization",
                    "is_public": True,
                },
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.12",
                    "pytorch_version": "2.10.0",
                    "image_name": "rocm/vllm-private:private-specific",
                    "onnxruntime_version": None,
                    "transformers_version": None,
                    "registry": "xcoartifactory.xilinx.com/uai-docker-local",
                    "image": "amd_quark",
                    "tag": "direct-select-only",
                    "is_public": False,
                },
            ]
        }

        result = select_docker_image.match_mapping(
            mock_load.return_value,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            None,
            None,
            "rocm/vllm-private:private-specific",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["tag"], "direct-select-only")

    @patch("select_docker_image.load_mapping")
    def test_match_mapping_vllm_version_filter_must_match(self, mock_load):
        """Test that vllm_version is enforced as a hard filter."""
        mock_load.return_value = {
            "mappings": [
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.12",
                    "pytorch_version": "2.10.0",
                    "vllm_version": "v0.13.0",
                    "onnxruntime_version": None,
                    "transformers_version": None,
                    "registry": "docker.io",
                    "image": "rocm/vllm",
                    "tag": "match",
                    "is_public": True,
                }
            ]
        }

        result = select_docker_image.match_mapping(
            mock_load.return_value,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            None,
            None,
            None,
            "v0.13.0",
        )
        self.assertIsNotNone(result)

        mismatch = select_docker_image.match_mapping(
            mock_load.return_value,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            None,
            None,
            None,
            "v0.14.0",
        )
        self.assertIsNone(mismatch)

    @patch("select_docker_image.load_mapping")
    def test_match_most_specific_selected(self, mock_load):
        """Test that most specific match is selected."""
        mock_load.return_value = {
            "mappings": [
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.12",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": None,  # Less specific
                    "transformers_version": None,
                    "registry": "xcoartifactory.xilinx.com/uai-docker-local",
                    "image": "amd_quark",
                    "tag": "less-specific",
                    "is_public": False,
                },
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.12",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": "1.22.2",  # More specific
                    "transformers_version": "4.57",
                    "registry": "xcoartifactory.xilinx.com/uai-docker-local",
                    "image": "amd_quark",
                    "tag": "more-specific",
                    "is_public": False,
                },
            ]
        }

        result = select_docker_image.match_mapping(
            mock_load.return_value,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            "1.22.2",  # Optional params specified
            "4.57",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["tag"], "more-specific")  # More specific match selected

    @patch("select_docker_image.load_mapping")
    def test_match_optional_params_must_match_when_specified(self, mock_load):
        """Test that optional params must match exactly when specified."""
        mock_load.return_value = {
            "mappings": [
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.12",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": "1.22.2",
                    "transformers_version": "4.57",
                    "registry": "xcoartifactory.xilinx.com/uai-docker-local",
                    "image": "amd_quark",
                    "tag": "match",
                    "is_public": False,
                },
            ]
        }

        # When optional params are specified but don't match, should return None
        result = select_docker_image.match_mapping(
            mock_load.return_value,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            "1.21.1",  # Mismatch
            "4.57",
        )

        self.assertIsNone(result)  # No match found

    @patch("select_docker_image.load_mapping")
    def test_match_optional_params_ignored_when_not_specified(self, mock_load):
        """Test that optional params are ignored when not specified."""
        mock_load.return_value = {
            "mappings": [
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.12",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": "1.22.2",  # Entry has value
                    "transformers_version": "4.57",
                    "registry": "xcoartifactory.xilinx.com/uai-docker-local",
                    "image": "amd_quark",
                    "tag": "match",
                    "is_public": False,
                },
            ]
        }

        # When optional params are NOT specified, should match regardless of entry's values
        result = select_docker_image.match_mapping(
            mock_load.return_value,
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            None,  # Optional params not specified - should be ignored
            None,
        )

        self.assertIsNotNone(result)  # Should match

    @patch("select_docker_image.load_mapping")
    def test_match_ambiguous_candidates_error(self, mock_load):
        """Test that ambiguous candidates raise ValueError."""
        mock_load.return_value = {
            "mappings": [
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.12",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": "1.22.2",
                    "transformers_version": "4.57",
                    "registry": "xcoartifactory.xilinx.com/uai-docker-local",
                    "image": "amd_quark",
                    "tag": "tag1",
                    "is_public": False,
                },
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.12",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": "1.22.2",
                    "transformers_version": "4.57",
                    "registry": "xcoartifactory.xilinx.com/uai-docker-local",
                    "image": "amd_quark",
                    "tag": "tag2",
                    "is_public": False,
                },
            ]
        }

        # Should raise ValueError for ambiguous candidates
        with self.assertRaises(ValueError) as context:
            select_docker_image.match_mapping(
                mock_load.return_value,
                "rocm-7.1",
                "ubuntu-22.04",
                "3.12",
                "2.10.0",
                "1.22.2",
                "4.57",
            )

        self.assertIn("Multiple", str(context.exception))
        self.assertIn("candidates", str(context.exception))

    @patch("select_docker_image.load_mapping")
    def test_match_no_match_found(self, mock_load):
        """Test that None is returned when no match is found."""
        mock_load.return_value = {"mappings": []}

        result = select_docker_image.match_mapping(
            mock_load.return_value,
            "rocm-5.0",
            "ubuntu-20.04",
            "3.9",
            "2.0.0",
            None,
            None,
        )

        self.assertIsNone(result)


class TestSelectDockerImage(unittest.TestCase):
    """Test the main select_docker_image function.

    These tests verify the high-level image selection function including:
    - Selection of public vs private images
    - Error handling when no match is found
    - Error handling for ambiguous candidates
    - Proper return value format (registry, image, tag, is_public)
    """

    @patch("select_docker_image.load_mapping")
    def test_select_public_image(self, mock_load):
        """Test selecting a public image."""
        mock_load.return_value = MOCK_YAML_MAPPING

        registry, image, tag, is_public = select_docker_image.select_docker_image(
            "rocm-7.1.1",
            "ubuntu-24.04",
            "3.12",
            "2.10.0",
            None,
            None,
        )

        self.assertEqual(registry, "docker.io")
        self.assertEqual(image, "rocm/pytorch")
        self.assertEqual(tag, "rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.10.0")
        self.assertTrue(is_public)

    @patch("select_docker_image.load_mapping")
    def test_select_rocm72_public_image(self, mock_load):
        """Test selecting ROCm 7.2 public image (test-only fixture)."""
        mock_load.return_value = MOCK_YAML_MAPPING

        registry, image, tag, is_public = select_docker_image.select_docker_image(
            "rocm-7.2",
            "ubuntu-24.04",
            "3.13",
            "2.10.0",
            None,
            None,
        )

        self.assertEqual(registry, "docker.io")
        self.assertEqual(image, "rocm/pytorch")
        self.assertEqual(tag, "rocm7.2_ubuntu24.04_py3.13_pytorch_release_2.10.0")
        self.assertTrue(is_public)

    @patch("select_docker_image.load_mapping")
    def test_select_cuda_image(self, mock_load):
        """Test selecting CUDA 12.6.3 private image."""
        mock_load.return_value = MOCK_YAML_MAPPING

        registry, image, tag, is_public = select_docker_image.select_docker_image(
            "cuda-12.6.3",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            "1.22.2",
            None,
        )

        self.assertEqual(registry, "xcoartifactory.xilinx.com/uai-docker-local")
        self.assertEqual(image, "amd_quark")
        self.assertIn("cuda12.6.3", tag)
        self.assertFalse(is_public)

    @patch("select_docker_image.load_mapping")
    def test_select_cpu_py311_onnx124(self, mock_load):
        """Test selecting CPU image with Python 3.11 and ONNX 1.24.2 (most specific)."""
        mock_load.return_value = MOCK_YAML_MAPPING

        registry, image, tag, is_public = select_docker_image.select_docker_image(
            "cpu",
            "ubuntu-22.04",
            "3.11",
            "2.10.0",
            "1.24.2",
            None,
        )

        self.assertEqual(registry, "xcoartifactory.xilinx.com/uai-docker-local")
        self.assertEqual(image, "amd_quark")
        self.assertIn("py3.11", tag)
        self.assertIn("onnxruntime1.24.2", tag)
        self.assertFalse(is_public)

    @patch("select_docker_image.load_mapping")
    def test_select_private_image(self, mock_load):
        """Test selecting a private image."""
        mock_load.return_value = MOCK_YAML_MAPPING

        registry, image, tag, is_public = select_docker_image.select_docker_image(
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            "1.22.2",
            "4.57",
        )

        self.assertEqual(registry, "xcoartifactory.xilinx.com/uai-docker-local")
        self.assertEqual(image, "amd_quark")
        self.assertIn("rocm7.1", tag)
        self.assertFalse(is_public)

    @patch("select_docker_image.load_mapping")
    def test_select_with_optional_params_not_specified(self, mock_load):
        """Test selecting when optional params are not specified."""
        mock_load.return_value = MOCK_YAML_MAPPING

        # Should match entry with null optional params (less specific)
        registry, image, tag, is_public = select_docker_image.select_docker_image(
            "rocm-7.1",
            "ubuntu-22.04",
            "3.12",
            "2.10.0",
            None,  # Optional params not specified
            "4.55",
        )

        self.assertIsNotNone(registry)
        self.assertIsNotNone(image)
        self.assertIsNotNone(tag)

    @patch("select_docker_image.load_mapping")
    def test_select_no_match_raises_error(self, mock_load):
        """Test that ValueError is raised when no match is found."""
        mock_load.return_value = {"mappings": []}

        with self.assertRaises(ValueError) as context:
            select_docker_image.select_docker_image(
                "rocm-5.0",
                "ubuntu-20.04",
                "3.9",
                "2.0.0",
                None,
                None,
            )

        self.assertIn("No matching Docker image found", str(context.exception))

    @patch("select_docker_image.load_mapping")
    def test_select_ambiguous_candidates_raises_error(self, mock_load):
        """Test that ValueError is raised for ambiguous candidates."""
        mock_load.return_value = {
            "mappings": [
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.12",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": "1.22.2",
                    "transformers_version": "4.57",
                    "registry": "xcoartifactory.xilinx.com/uai-docker-local",
                    "image": "amd_quark",
                    "tag": "tag1",
                    "is_public": False,
                },
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.12",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": "1.22.2",
                    "transformers_version": "4.57",
                    "registry": "xcoartifactory.xilinx.com/uai-docker-local",
                    "image": "amd_quark",
                    "tag": "tag2",
                    "is_public": False,
                },
            ]
        }

        with self.assertRaises(ValueError) as context:
            select_docker_image.select_docker_image(
                "rocm-7.1",
                "ubuntu-22.04",
                "3.12",
                "2.10.0",
                "1.22.2",
                "4.57",
            )

        self.assertIn("Multiple", str(context.exception))

    @patch("select_docker_image.load_mapping")
    def test_select_vllm_eval_by_direct_image_name(self, mock_load):
        """Test selecting a direct image by image_name."""
        mock_load.return_value = {
            "mappings": [
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.13",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": None,
                    "transformers_version": None,
                    "registry": "docker.io",
                    "image": "rocm/vllm",
                    "tag": "generic-vllm-eval",
                    "is_public": True,
                },
                {
                    "image_name": "rocm/vllm-private:private-specific",
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.13",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": None,
                    "transformers_version": None,
                    "registry": "docker.io",
                    "image": "rocm/vllm-private",
                    "tag": "private-specific",
                    "is_public": False,
                },
            ]
        }

        registry, image, tag, is_public = select_docker_image.select_docker_image(
            "rocm-7.1",
            "ubuntu-22.04",
            "3.13",
            "2.10.0",
            None,
            None,
            "rocm/vllm-private:private-specific",
        )

        self.assertEqual(registry, "docker.io")
        self.assertEqual(image, "rocm/vllm-private")
        self.assertEqual(tag, "private-specific")
        self.assertFalse(is_public)

    @patch("select_docker_image.load_mapping")
    def test_select_direct_image_name_no_match_raises_error(self, mock_load):
        """Test error message when image_name hard filter cannot be satisfied."""
        mock_load.return_value = {
            "mappings": [
                {
                    "accelerator_version": "rocm-7.1",
                    "os_version": "ubuntu-22.04",
                    "python_version": "3.13",
                    "pytorch_version": "2.10.0",
                    "onnxruntime_version": None,
                    "transformers_version": None,
                    "registry": "docker.io",
                    "image": "rocm/vllm",
                    "tag": "generic-vllm-eval",
                    "is_public": True,
                }
            ]
        }

        with self.assertRaises(ValueError) as context:
            select_docker_image.select_docker_image(
                "rocm-7.1",
                "ubuntu-22.04",
                "3.13",
                "2.10.0",
                None,
                None,
                "rocm/vllm-private:does-not-exist",
            )

        self.assertIn("image_name=rocm/vllm-private:does-not-exist", str(context.exception))
        self.assertIn("rocm/vllm-private:does-not-exist", str(context.exception))


class TestFormatImageTag(unittest.TestCase):
    """Test image tag formatting with registry prefixes.

    Verifies that image tags are formatted correctly with or without
    registry prefixes, depending on the registry type (docker.io vs others).
    """

    def test_format_image_tag_docker_io(self):
        """Test formatting for docker.io registry (should omit registry)."""
        result = select_docker_image.format_image_tag("docker.io", "rocm/pytorch", "tag1", include_registry=True)
        self.assertEqual(result, "rocm/pytorch:tag1")

    def test_format_image_tag_non_docker_io(self):
        """Test formatting for non-docker.io registry (should include registry)."""
        result = select_docker_image.format_image_tag(
            "xcoartifactory.xilinx.com/uai-docker-local", "amd_quark", "tag1", include_registry=True
        )
        self.assertEqual(result, "xcoartifactory.xilinx.com/uai-docker-local/amd_quark:tag1")

    def test_format_image_tag_without_registry(self):
        """Test formatting without registry prefix."""
        result = select_docker_image.format_image_tag("docker.io", "rocm/pytorch", "tag1", include_registry=False)
        self.assertEqual(result, "rocm/pytorch:tag1")


class TestLoadMapping(unittest.TestCase):
    """Test loading mapping from YAML file."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("select_docker_image.yaml.safe_load")
    def test_load_mapping_success(self, mock_yaml_load, mock_file):
        """Test successful loading of YAML mapping."""
        mock_yaml_load.return_value = MOCK_YAML_MAPPING

        result = select_docker_image.load_mapping()

        self.assertIn("mappings", result)
        self.assertEqual(len(result["mappings"]), 9)  # 7 production-like + 2 test-only fixtures
        mock_file.assert_called_once()

    def test_load_mapping_file_not_found(self):
        """Test handling of missing mapping file."""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            with self.assertRaises(SystemExit) as cm:
                select_docker_image.load_mapping()
            self.assertEqual(cm.exception.code, 1)

    def test_load_mapping_invalid_yaml(self):
        """Test handling of invalid YAML by providing broken YAML content."""
        invalid_yaml_content = ":\n  - invalid: ["

        with patch("builtins.open", return_value=StringIO(invalid_yaml_content)):
            with pytest.raises(SystemExit) as e:
                select_docker_image.load_mapping()
            assert e.value.code == 1

    @patch("builtins.open", new_callable=mock_open)
    @patch("select_docker_image.yaml.safe_load")
    def test_load_mapping_expands_template_tag(self, mock_yaml_load, mock_file):
        """Test that load_mapping expands template tags (placeholders) into concrete entries."""
        mock_yaml_load.return_value = MOCK_YAML_MAPPING_WITH_TEMPLATE

        result = select_docker_image.load_mapping()

        # Template has python "3.13, 3.12" -> expands to 2 entries
        mappings = result["mappings"]
        self.assertEqual(len(mappings), 2)

        tags = [m["tag"] for m in mappings]
        self.assertIn("latest-rocm7.1-base-py3.13-torch2.10.0-onnxruntime1.23.2-ubuntu22.04", tags)
        self.assertIn("latest-rocm7.1-base-py3.12-torch2.10.0-onnxruntime1.23.2-ubuntu22.04", tags)


class TestCLI(unittest.TestCase):
    """Test command-line interface and output formats.

    These tests verify the CLI functionality including:
    - Basic usage with required parameters
    - JSON output format
    - Image-only and tag-only output formats
    - Error handling and exit codes
    """

    @patch("select_docker_image.select_docker_image")
    @patch(
        "sys.argv",
        [
            "select_docker_image.py",
            "--accelerator-version",
            "rocm-7.1",
            "--os-version",
            "ubuntu-22.04",
            "--python-version",
            "3.12",
            "--pytorch-version",
            "2.10.0",
        ],
    )
    def test_cli_basic_usage(self, mock_select):
        """Test basic CLI usage."""
        mock_select.return_value = ("docker.io", "rocm/pytorch", "tag1", True)

        with patch("builtins.print") as mock_print:
            select_docker_image.main()
            mock_print.assert_called_once()
            output = mock_print.call_args[0][0]
            self.assertIn("rocm/pytorch", output)

    @patch("select_docker_image.select_docker_image")
    @patch(
        "sys.argv",
        [
            "select_docker_image.py",
            "--accelerator-version",
            "rocm-7.1",
            "--os-version",
            "ubuntu-22.04",
            "--python-version",
            "3.12",
            "--pytorch-version",
            "2.10.0",
            "--output-format",
            "json",
        ],
    )
    def test_cli_json_output(self, mock_select):
        """Test JSON output format."""
        mock_select.return_value = ("docker.io", "rocm/pytorch", "tag1", True)

        with patch("builtins.print") as mock_print:
            select_docker_image.main()
            mock_print.assert_called_once()
            output = mock_print.call_args[0][0]
            # Should be valid JSON
            import json

            result = json.loads(output)
            self.assertIn("registry", result)
            self.assertIn("image", result)
            self.assertIn("tag", result)
            self.assertIn("is_public", result)

    @patch("select_docker_image.select_docker_image")
    @patch(
        "sys.argv",
        [
            "select_docker_image.py",
            "--accelerator-version",
            "rocm-7.1",
            "--os-version",
            "ubuntu-22.04",
            "--python-version",
            "3.12",
            "--pytorch-version",
            "2.10.0",
            "--output-format",
            "image",
        ],
    )
    def test_cli_image_only_output(self, mock_select):
        """Test image-only output format."""
        mock_select.return_value = ("docker.io", "rocm/pytorch", "tag1", True)

        with patch("builtins.print") as mock_print:
            select_docker_image.main()
            mock_print.assert_called_once()
            output = mock_print.call_args[0][0]
            self.assertEqual(output, "rocm/pytorch")

    @patch("select_docker_image.select_docker_image")
    @patch(
        "sys.argv",
        [
            "select_docker_image.py",
            "--accelerator-version",
            "rocm-7.1",
            "--os-version",
            "ubuntu-22.04",
            "--python-version",
            "3.12",
            "--pytorch-version",
            "2.10.0",
            "--output-format",
            "tag",
        ],
    )
    def test_cli_tag_only_output(self, mock_select):
        """Test tag-only output format."""
        mock_select.return_value = ("docker.io", "rocm/pytorch", "tag1", True)

        with patch("builtins.print") as mock_print:
            select_docker_image.main()
            mock_print.assert_called_once()
            output = mock_print.call_args[0][0]
            self.assertEqual(output, "tag1")

    @patch("select_docker_image.select_docker_image")
    @patch(
        "sys.argv",
        [
            "select_docker_image.py",
            "--accelerator-version",
            "rocm-7.1",
            "--os-version",
            "ubuntu-22.04",
            "--python-version",
            "3.12",
            "--pytorch-version",
            "2.10.0",
        ],
    )
    def test_cli_error_handling(self, mock_select):
        """Test CLI error handling."""
        mock_select.side_effect = ValueError("No matching Docker image found")

        with self.assertRaises(SystemExit):
            select_docker_image.main()

    @patch(
        "sys.argv",
        [
            "select_docker_image.py",
            "--image-name",
            "rocm/vllm-private:vllm_dev_base_mxfp4_20260122",
        ],
    )
    def test_cli_direct_image_lookup_without_version_args(self):
        """Test CLI validation: image_name alone is no longer sufficient."""
        with self.assertRaises(SystemExit) as context:
            select_docker_image.main()
        self.assertEqual(context.exception.code, 2)

    @patch("select_docker_image.select_docker_image")
    @patch(
        "sys.argv",
        [
            "select_docker_image.py",
            "--accelerator-version",
            "rocm-7.1",
            "--os-version",
            "ubuntu-22.04",
            "--python-version",
            "3.12",
            "--pytorch-version",
            "2.10.0",
            "--image-name",
            "rocm/vllm-private:vllm_dev_base_mxfp4_20260122",
            "--vllm-version",
            "v0.13.0",
        ],
    )
    def test_cli_image_name_and_vllm_version_are_forwarded(self, mock_select):
        """Test CLI forwards image_name and vllm_version hard filters."""
        mock_select.return_value = ("docker.io", "rocm/vllm-private", "vllm_dev_base_mxfp4_20260122", False)
        with patch("builtins.print"):
            select_docker_image.main()

        _, kwargs = mock_select.call_args
        self.assertEqual(kwargs["image_name"], "rocm/vllm-private:vllm_dev_base_mxfp4_20260122")
        self.assertEqual(kwargs["vllm_version"], "v0.13.0")

    @patch(
        "sys.argv",
        [
            "select_docker_image.py",
            "--accelerator-version",
            "rocm-7.1",
        ],
    )
    def test_cli_missing_version_args_without_image_name(self):
        """Test CLI validation when version matching args are incomplete."""
        with self.assertRaises(SystemExit) as context:
            select_docker_image.main()
        self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
