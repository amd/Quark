#!/usr/bin/env python3
#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import sys
from unittest.mock import MagicMock, patch

from quark.common.utils.detect_pytorch_extension_dir import detect_pytorch_extension_dir, main

_MODULE = "quark.common.utils.detect_pytorch_extension_dir"


class MockVersionInfo:
    """Mock version_info that supports both attribute access and tuple comparison."""

    def __init__(self, major: int, minor: int, micro: int = 0, releaselevel: str = "final", serial: int = 0):
        self.major = major
        self.minor = minor
        self.micro = micro
        self.releaselevel = releaselevel
        self.serial = serial

    def __getitem__(self, index: int):
        return (self.major, self.minor, self.micro, self.releaselevel, self.serial)[index]

    def __ge__(self, other):
        return (self.major, self.minor, self.micro) >= other[:3]

    def __le__(self, other):
        return (self.major, self.minor, self.micro) <= other[:3]

    def __lt__(self, other):
        return (self.major, self.minor, self.micro) < other[:3]

    def __gt__(self, other):
        return (self.major, self.minor, self.micro) > other[:3]

    def __eq__(self, other):
        return (self.major, self.minor, self.micro) == other[:3]


class TestDetectPytorchExtensionDirOldTorch:
    """Tests for torch <= 2.10 behavior (CUDA priority, HIP treated as CPU)."""

    def test_cuda_with_valid_version(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=11))

        mock_version = MagicMock()
        mock_version.cuda = "12.1.0"
        mock_version.hip = None

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=True),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py311_cu121"

    def test_cuda_with_two_part_version(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=10))

        mock_version = MagicMock()
        mock_version.cuda = "11.8"
        mock_version.hip = None

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=True),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py310_cu118"

    def test_cuda_with_single_part_version(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=9))

        mock_version = MagicMock()
        mock_version.cuda = "12"
        mock_version.hip = None

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=True),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py39_cpu"

    def test_cuda_empty_string(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=10))

        mock_version = MagicMock()
        mock_version.cuda = ""
        mock_version.hip = ""

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=True),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py310_cpu"

    def test_hip_treated_as_cpu(self, monkeypatch):
        """On old torch, HIP falls back to CPU regardless of platform."""
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=11))

        mock_version = MagicMock()
        mock_version.cuda = None
        mock_version.hip = "5.7.0"

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=True),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py311_cpu"

    def test_no_cuda_no_hip(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=12))

        mock_version = MagicMock()
        delattr(mock_version, "cuda")
        delattr(mock_version, "hip")

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=True),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py312_cpu"

    def test_cuda_none_value(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=11))

        mock_version = MagicMock()
        mock_version.cuda = None
        mock_version.hip = None

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=True),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py311_cpu"

    def test_cuda_priority_over_hip(self, monkeypatch):
        """On old torch, CUDA is checked first and HIP is ignored."""
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=11))

        mock_version = MagicMock()
        mock_version.cuda = "12.1"
        mock_version.hip = "5.7.0"

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=True),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py311_cu121"

    def test_different_python_versions(self, monkeypatch):
        test_cases = [
            (3, 8, "py38"),
            (3, 9, "py39"),
            (3, 10, "py310"),
            (3, 11, "py311"),
            (3, 12, "py312"),
        ]

        mock_version = MagicMock()
        mock_version.cuda = None
        mock_version.hip = None

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=True),
        ):
            for major, minor, expected_prefix in test_cases:
                monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=major, minor=minor))
                result = detect_pytorch_extension_dir()
                assert result == f"{expected_prefix}_cpu"

    def test_cuda_version_formats(self, monkeypatch):
        test_cases = [
            ("11.7", "py311_cu117"),
            ("11.8.0", "py311_cu118"),
            ("12.0", "py311_cu120"),
            ("12.1.1", "py311_cu121"),
        ]

        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=11))

        for cuda_version, expected_subdir in test_cases:
            mock_version = MagicMock()
            mock_version.cuda = cuda_version
            mock_version.hip = None

            with (
                patch(f"{_MODULE}.torch.version", mock_version),
                patch(f"{_MODULE}.is_package_lower_or_equal", return_value=True),
            ):
                result = detect_pytorch_extension_dir()
                assert result == expected_subdir

    def test_main_function_output(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=11))

        mock_version = MagicMock()
        mock_version.cuda = "12.1"
        mock_version.hip = None

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=True),
        ):
            main()
            captured = capsys.readouterr()
            assert captured.out.strip() == "py311_cu121"


class TestDetectPytorchExtensionDirNewTorch:
    """Tests for torch >= 2.11 behavior (HIP priority, rocm{version} naming)."""

    def test_hip_produces_rocm_subdir(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=11))

        mock_version = MagicMock()
        mock_version.cuda = None
        mock_version.hip = "6.3.0"

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=False),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py311_rocm630"

    def test_hip_priority_over_cuda(self, monkeypatch):
        """On new torch, HIP takes priority over CUDA."""
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=11))

        mock_version = MagicMock()
        mock_version.cuda = "12.1"
        mock_version.hip = "6.3.0"

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=False),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py311_rocm630"

    def test_cuda_still_works(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=11))

        mock_version = MagicMock()
        mock_version.cuda = "12.1"
        mock_version.hip = None

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=False),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py311_cu121"

    def test_cpu_fallback(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=11))

        mock_version = MagicMock()
        mock_version.cuda = None
        mock_version.hip = None

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=False),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py311_cpu"

    def test_hip_two_part_version(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=11))

        mock_version = MagicMock()
        mock_version.cuda = None
        mock_version.hip = "6.3"

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=False),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py311_rocm63"

    def test_cuda_with_single_part_version(self, monkeypatch):
        """Test CUDA with single-part version falls back to CPU (covers line 69)."""
        monkeypatch.setattr(sys, "version_info", MockVersionInfo(major=3, minor=11))

        mock_version = MagicMock()
        mock_version.cuda = "12"
        mock_version.hip = None

        with (
            patch(f"{_MODULE}.torch.version", mock_version),
            patch(f"{_MODULE}.is_package_lower_or_equal", return_value=False),
        ):
            result = detect_pytorch_extension_dir()
            assert result == "py311_cpu"
