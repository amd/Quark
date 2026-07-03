#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for ``quark.common.utils.collect_env``.

The module shells out to ``rocminfo`` / ``nvidia-smi`` / ``lscpu`` /
``lsb_release`` / ``pip`` / ``conda`` and reads ROCm version files under
``/opt/rocm``.  All external interactions are mocked here so the suite
runs identically on CPU, ROCm and CUDA runners.
"""

import json
import platform
import subprocess
import unittest
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import quark.common.utils.collect_env as ce
from quark.common.utils.collect_env import (
    HardwareInfoCollector,
    SoftwareInfoCollector,
    collect_environment,
    print_environment,
    save_environment_to_json,
)

ROCMINFO_MI300X = """\
ROCk module is loaded
=====================
HSA System Attributes
=====================
Runtime Version:         1.15
=====================
HSA Agents
=====================
*******
Agent 1
*******
  Name:                    AMD Ryzen Threadripper PRO
  Marketing Name:          AMD Ryzen Threadripper PRO
  Vendor Name:             CPU
*******
Agent 2
*******
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  Vendor Name:             AMD
*******
Agent 3
*******
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  Vendor Name:             AMD
*** Done ***
"""

ROCMINFO_NO_INSTINCT = """\
ROCk module is loaded
Agent 1
  Name:                    gfx-stub
  Marketing Name:          Some Other GPU
"""

LSCPU_AMD_EPYC = """\
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Byte Order:                         Little Endian
Address sizes:                      52 bits physical, 57 bits virtual
CPU(s):                             96
Model name:                         AMD EPYC 9474F 48-Core Processor
"""

LSCPU_NO_MODEL_LINE = """\
Architecture:                       x86_64
CPU(s):                             1
"""

OS_RELEASE_UBUNTU2204 = """\
NAME="Ubuntu"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
ID=ubuntu
PRETTY_NAME="Ubuntu 22.04.5 LTS"
malformed-line-no-equals
"""

OS_RELEASE_NO_PRETTY = """\
NAME="MinimalOS"
ID=minimal
"""

PIP_FREEZE_SAMPLE = """\
# pip freeze output

absl-py==2.1.0
torch==2.10.0+rocm7.1
amd-quark==0.12
bogus-line-no-equals
name @ git+https://example.com/repo
"""

CONDA_LIST_SAMPLE = """\
# packages in environment at /opt/conda/envs/quark-env:
#
# Name                    Version                   Build  Channel

absl-py                   2.1.1                    pypi_0    pypi
torch                     2.10.0           py3.13_rocm7.1    rocm
single_token
"""


def _which_factory(*available: str):
    """Return a ``shutil.which`` side-effect that pretends only the given
    binaries are on PATH."""

    def _which(cmd: str) -> str | None:
        return f"/usr/bin/{cmd}" if cmd in available else None

    return _which


# ---------------------------------------------------------------------------
# HardwareInfoCollector
# ---------------------------------------------------------------------------


class TestHardwareInfoCollector(unittest.TestCase):
    """Cover every branch of ``HardwareInfoCollector``."""

    # ---- _get_accelerator_type ----

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("rocminfo"))
    @patch("quark.common.utils.collect_env.subprocess.check_output", return_value=ROCMINFO_MI300X)
    def test_accelerator_rocm_instinct(self, _mock_run: MagicMock, _mock_which: MagicMock) -> None:
        c = HardwareInfoCollector()
        self.assertEqual(c.accelerator_type, "AMD Instinct MI300X")
        self.assertEqual(c._get_gpu_models(), "gfx942/gfx942")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("rocminfo"))
    @patch("quark.common.utils.collect_env.subprocess.check_output", return_value=ROCMINFO_NO_INSTINCT)
    def test_accelerator_rocm_no_instinct_line(self, _mock_run: MagicMock, _mock_which: MagicMock) -> None:
        c = HardwareInfoCollector()
        self.assertIsNone(c.accelerator_type)

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("rocminfo", "nvidia-smi"))
    @patch(
        "quark.common.utils.collect_env.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "rocminfo"),
    )
    def test_accelerator_rocminfo_raises_then_nvidia(self, _mock_run: MagicMock, _mock_which: MagicMock) -> None:
        c = HardwareInfoCollector()
        # rocminfo blew up; falls through to nvidia-smi
        self.assertEqual(c.accelerator_type, "NVIDIA GPU")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("nvidia-smi"))
    @patch(
        "quark.common.utils.collect_env.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
    )
    def test_accelerator_nvidia_only(self, _mock_run: MagicMock, _mock_which: MagicMock) -> None:
        # Mock check_output so the in-__init__ ``_get_gpu_models`` call doesn't
        # shell out to a real nvidia-smi on the host. The test only asserts
        # accelerator_type, which is set from ``_get_accelerator_type`` and
        # doesn't depend on check_output (the "nvidia-smi" branch returns the
        # constant string before any subprocess call).
        c = HardwareInfoCollector()
        self.assertEqual(c.accelerator_type, "NVIDIA GPU")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())  # nothing on PATH
    def test_accelerator_none(self, _mock_which: MagicMock) -> None:
        c = HardwareInfoCollector()
        self.assertIsNone(c.accelerator_type)
        self.assertIsNone(c._get_gpu_models())

    # ---- _get_gpu_models additional branches ----

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("rocminfo"))
    @patch("quark.common.utils.collect_env.subprocess.check_output", return_value=ROCMINFO_NO_INSTINCT)
    def test_gpu_models_rocminfo_returns_gfx_stub(self, _mock_run: MagicMock, _mock_which: MagicMock) -> None:
        # ROCMINFO_NO_INSTINCT has one "Name:" line containing "gfx-stub", which
        # counts as a gfx hit (the production code checks for the "gfx" substring),
        # so _get_gpu_models() returns it even though there is no Instinct GPU.
        c = HardwareInfoCollector()
        self.assertEqual(c._get_gpu_models(), "gfx-stub")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("rocminfo"))
    @patch(
        "quark.common.utils.collect_env.subprocess.check_output",
        return_value="Agent 1\n  Name:                    not-a-gpu\n",
    )
    def test_gpu_models_rocminfo_no_match(self, _mock_run: MagicMock, _mock_which: MagicMock) -> None:
        c = HardwareInfoCollector()
        self.assertIsNone(c._get_gpu_models())

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("rocminfo"))
    @patch(
        "quark.common.utils.collect_env.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "rocminfo"),
    )
    def test_gpu_models_rocminfo_raises(self, _mock_run: MagicMock, _mock_which: MagicMock) -> None:
        # No nvidia-smi present in this scenario; falls through to None.
        c = HardwareInfoCollector()
        self.assertIsNone(c._get_gpu_models())

    def test_gpu_models_nvidia_success(self) -> None:
        """nvidia-smi present + valid CSV stdout → joined GPU names."""
        with (
            patch(
                "quark.common.utils.collect_env.shutil.which",
                side_effect=_which_factory("nvidia-smi"),
            ),
            patch(
                "quark.common.utils.collect_env.subprocess.check_output",
                return_value="NVIDIA H100 80GB HBM3\nNVIDIA H100 80GB HBM3\n",
            ),
        ):
            c = HardwareInfoCollector()
            self.assertEqual(
                c._get_gpu_models(),
                "NVIDIA H100 80GB HBM3/NVIDIA H100 80GB HBM3",
            )

    def test_gpu_models_nvidia_empty_stdout(self) -> None:
        """nvidia-smi present + empty stdout → None (the falsy branch of the join)."""
        with (
            patch(
                "quark.common.utils.collect_env.shutil.which",
                side_effect=_which_factory("nvidia-smi"),
            ),
            patch(
                "quark.common.utils.collect_env.subprocess.check_output",
                return_value="",
            ),
        ):
            c = HardwareInfoCollector()
            self.assertIsNone(c._get_gpu_models())

    def test_gpu_models_nvidia_check_output_raises(self) -> None:
        """nvidia-smi present but check_output raises → None (exception swallowed)."""
        with (
            patch(
                "quark.common.utils.collect_env.shutil.which",
                side_effect=_which_factory("nvidia-smi"),
            ),
            patch(
                "quark.common.utils.collect_env.subprocess.check_output",
                side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
            ),
        ):
            c = HardwareInfoCollector()
            self.assertIsNone(c._get_gpu_models())

    # ---- _get_cpu_model ----

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("lscpu"))
    @patch("quark.common.utils.collect_env.subprocess.run")
    def test_cpu_model_from_lscpu(self, mock_run: MagicMock, _mock_which: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout=LSCPU_AMD_EPYC)
        c = HardwareInfoCollector()
        self.assertEqual(c.cpu_model, "AMD EPYC 9474F 48-Core Processor")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("lscpu"))
    @patch("quark.common.utils.collect_env.subprocess.run")
    @patch("quark.common.utils.collect_env.platform.processor", return_value="x86_64")
    def test_cpu_model_lscpu_no_model_line(
        self,
        _mock_proc: MagicMock,
        mock_run: MagicMock,
        _mock_which: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout=LSCPU_NO_MODEL_LINE)
        c = HardwareInfoCollector()
        self.assertEqual(c.cpu_model, "x86_64")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("lscpu"))
    @patch(
        "quark.common.utils.collect_env.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="lscpu", timeout=3),
    )
    @patch("quark.common.utils.collect_env.platform.processor", return_value="cpu-from-platform")
    def test_cpu_model_lscpu_timeout(
        self,
        _mock_proc: MagicMock,
        _mock_run: MagicMock,
        _mock_which: MagicMock,
    ) -> None:
        c = HardwareInfoCollector()
        self.assertEqual(c.cpu_model, "cpu-from-platform")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    @patch("quark.common.utils.collect_env.platform.processor", return_value="no-lscpu-fallback")
    def test_cpu_model_no_lscpu(self, _mock_proc: MagicMock, _mock_which: MagicMock) -> None:
        c = HardwareInfoCollector()
        self.assertEqual(c.cpu_model, "no-lscpu-fallback")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("lscpu"))
    @patch("quark.common.utils.collect_env.subprocess.run")
    @patch("quark.common.utils.collect_env.platform.processor", return_value="returncode-nonzero")
    def test_cpu_model_lscpu_nonzero_returncode(
        self,
        _mock_proc: MagicMock,
        mock_run: MagicMock,
        _mock_which: MagicMock,
    ) -> None:
        # Covers the `if result.returncode == 0:` False branch (line 74 -> 80).
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        c = HardwareInfoCollector()
        self.assertEqual(c.cpu_model, "returncode-nonzero")

    # ---- _get_cuda_version ----

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_cuda_version_present(self, _mock_which: MagicMock) -> None:
        with patch("quark.common.utils.collect_env.torch") as mock_torch:
            mock_torch.version.cuda = "12.4"
            c = HardwareInfoCollector()
            self.assertEqual(c.cuda_version, "12.4")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_cuda_version_attr_missing(self, _mock_which: MagicMock) -> None:
        with patch("quark.common.utils.collect_env.torch") as mock_torch:
            # spec=[] => mock has no attributes; hasattr(torch.version, "cuda") is False
            mock_torch.version = MagicMock(spec=[])
            c = HardwareInfoCollector()
            self.assertIsNone(c.cuda_version)

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_cuda_version_attr_none(self, _mock_which: MagicMock) -> None:
        with patch("quark.common.utils.collect_env.torch") as mock_torch:
            mock_torch.version.cuda = None
            c = HardwareInfoCollector()
            self.assertIsNone(c.cuda_version)

    # ---- _get_rocm_version ----

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_rocm_version_from_info_file(self, _mock_which: MagicMock) -> None:
        m = mock_open(read_data="7.1.0\n")
        with (
            patch.dict("quark.common.utils.collect_env.os.environ", {"ROCM_PATH": "/custom/rocm"}, clear=False),
            patch("quark.common.utils.collect_env.os.path.exists") as mock_exists,
            patch("quark.common.utils.collect_env.open", m, create=True),
        ):
            mock_exists.side_effect = lambda p: p == "/custom/rocm/.info/version"
            c = HardwareInfoCollector()
        self.assertEqual(c.rocm_version, "7.1.0")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_rocm_version_from_version_txt_default_path(self, _mock_which: MagicMock) -> None:
        """ROCM_PATH unset -> defaults to /opt/rocm; only version.txt exists.

        Note on env handling: we want ROCM_PATH absent for the duration of
        this test, but we must NOT use ``clear=True`` -- that would briefly
        empty the process-global ``os.environ`` (stripping HOME / PATH /
        LD_LIBRARY_PATH / COVERAGE_FILE / ...), visible to any concurrent
        reader during the window between ``clear()`` and re-population.
        Instead, ``patch.dict(..., clear=False)`` snapshots the dict on
        entry and restores it on exit, so popping ROCM_PATH inside the
        with-block is automatically reverted with no other side effects.
        """
        m = mock_open(read_data="6.4.1\n")
        with (
            patch.dict("quark.common.utils.collect_env.os.environ", clear=False),
            patch("quark.common.utils.collect_env.os.path.exists") as mock_exists,
            patch("quark.common.utils.collect_env.open", m, create=True),
        ):
            ce.os.environ.pop("ROCM_PATH", None)
            mock_exists.side_effect = lambda p: p == "/opt/rocm/version.txt"
            c = HardwareInfoCollector()
        self.assertEqual(c.rocm_version, "6.4.1")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    @patch("quark.common.utils.collect_env.os.path.exists", return_value=False)
    def test_rocm_version_no_files(self, _mock_exists: MagicMock, _mock_which: MagicMock) -> None:
        c = HardwareInfoCollector()
        self.assertIsNone(c.rocm_version)

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_rocm_version_open_raises_falls_through(self, _mock_which: MagicMock) -> None:
        """All three version files report existing but ``open`` raises every time.

        Validates that an unreadable version file is treated as "missing":
        ``_get_rocm_version`` swallows the OSError, moves on to the next
        candidate path, and ultimately returns None when none of the three
        can be read.
        """
        with (
            patch("quark.common.utils.collect_env.os.path.exists", return_value=True),
            patch("quark.common.utils.collect_env.open", side_effect=OSError("perm"), create=True),
        ):
            c = HardwareInfoCollector()
        # All three files raised -> falls through to return None.
        self.assertIsNone(c.rocm_version)

    # ---- to_dict ----

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    @patch("quark.common.utils.collect_env.os.path.exists", return_value=False)
    def test_to_dict_shape(self, _mock_exists: MagicMock, _mock_which: MagicMock) -> None:
        with patch("quark.common.utils.collect_env.torch") as mock_torch:
            mock_torch.version.cuda = None
            c = HardwareInfoCollector()
        self.assertEqual(
            set(c.to_dict().keys()),
            {"accelerator_type", "gpu_models", "cpu_model", "cuda_version", "rocm_version"},
        )


# ---------------------------------------------------------------------------
# SoftwareInfoCollector
# ---------------------------------------------------------------------------


class TestSoftwareInfoCollector(unittest.TestCase):
    """Cover every branch of ``SoftwareInfoCollector``."""

    # ---- _get_os ----

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("lsb_release"))
    @patch("quark.common.utils.collect_env.subprocess.run")
    def test_os_from_lsb_release(self, mock_run: MagicMock, _mock_which: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout='"Ubuntu 22.04.5 LTS"\n')
        c = SoftwareInfoCollector()
        self.assertEqual(c.os, "Ubuntu 22.04.5 LTS")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("lsb_release"))
    @patch("quark.common.utils.collect_env.subprocess.run")
    def test_os_lsb_release_nonzero_fallback_to_os_release(self, mock_run: MagicMock, _mock_which: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        m = mock_open(read_data=OS_RELEASE_UBUNTU2204)
        with patch("quark.common.utils.collect_env.open", m, create=True):
            c = SoftwareInfoCollector()
        self.assertEqual(c.os, "Ubuntu 22.04.5 LTS")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("lsb_release"))
    @patch(
        "quark.common.utils.collect_env.subprocess.run",
        side_effect=FileNotFoundError("lsb_release vanished"),
    )
    def test_os_lsb_release_raises_fallback_to_os_release(self, _mock_run: MagicMock, _mock_which: MagicMock) -> None:
        m = mock_open(read_data=OS_RELEASE_UBUNTU2204)
        with patch("quark.common.utils.collect_env.open", m, create=True):
            c = SoftwareInfoCollector()
        self.assertEqual(c.os, "Ubuntu 22.04.5 LTS")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_os_no_lsb_no_pretty_falls_to_platform(self, _mock_which: MagicMock) -> None:
        # /etc/os-release exists but has no PRETTY_NAME -> drops to platform.platform()
        m = mock_open(read_data=OS_RELEASE_NO_PRETTY)
        with (
            patch("quark.common.utils.collect_env.open", m, create=True),
            patch("quark.common.utils.collect_env.platform.platform", return_value="Linux-fake"),
        ):
            c = SoftwareInfoCollector()
        self.assertEqual(c.os, "Linux-fake")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_os_etc_os_release_open_raises(self, _mock_which: MagicMock) -> None:
        with (
            patch("quark.common.utils.collect_env.open", side_effect=OSError("boom"), create=True),
            patch("quark.common.utils.collect_env.platform.platform", return_value="Linux-fallback"),
        ):
            c = SoftwareInfoCollector()
        self.assertEqual(c.os, "Linux-fallback")

    # ---- python_version exposed via to_dict() ----

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_python_version_in_to_dict_is_dotted_three_part(self, _mock_which: MagicMock) -> None:
        """`to_dict()["python"]` is a dotted-int string ``major.minor.micro``
        and matches the currently-running interpreter.

        Asserts the *format* (a behavioral contract -- downstream consumers
        like ``contrib/llm_eval/experiment_report.py`` parse this field)
        against an independent source (``platform.python_version()``), not
        against a copy-paste of the production expression. So if the
        implementation is refactored to use ``platform.python_version()``,
        a different format string, or anything else that breaks the
        ``M.m.p`` contract, the test catches it.
        """
        with patch("quark.common.utils.collect_env.open", side_effect=OSError, create=True):
            c = SoftwareInfoCollector()
        # Stub package-list collection so the test doesn't depend on the
        # host's pip/conda state and to_dict() returns only fields populated
        # by SoftwareInfoCollector itself.
        with patch.object(c, "collect_packages", return_value={}):
            d = c.to_dict()
        self.assertIn("python", d, "to_dict() must expose 'python' for downstream consumers")
        py = d["python"]
        self.assertRegex(py, r"^\d+\.\d+\.\d+$")
        # Cross-check against a different source of truth.
        self.assertEqual(py, platform.python_version())

    # ---- _get_cuda_driver_version ----

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("nvidia-smi"))
    @patch("quark.common.utils.collect_env.subprocess.check_output", return_value="555.42.02\n")
    def test_cuda_driver_version_present(self, _mock_run: MagicMock, _mock_which: MagicMock) -> None:
        with patch("quark.common.utils.collect_env.open", side_effect=OSError, create=True):
            c = SoftwareInfoCollector()
        self.assertEqual(c.cuda_driver_version, "555.42.02")

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("nvidia-smi"))
    @patch("quark.common.utils.collect_env.subprocess.check_output", return_value="\n")
    def test_cuda_driver_version_empty_stdout(self, _mock_run: MagicMock, _mock_which: MagicMock) -> None:
        with patch("quark.common.utils.collect_env.open", side_effect=OSError, create=True):
            c = SoftwareInfoCollector()
        self.assertIsNone(c.cuda_driver_version)

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("nvidia-smi"))
    @patch(
        "quark.common.utils.collect_env.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
    )
    def test_cuda_driver_version_raises(self, _mock_run: MagicMock, _mock_which: MagicMock) -> None:
        with patch("quark.common.utils.collect_env.open", side_effect=OSError, create=True):
            c = SoftwareInfoCollector()
        self.assertIsNone(c.cuda_driver_version)

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_cuda_driver_version_no_nvidia_smi(self, _mock_which: MagicMock) -> None:
        with patch("quark.common.utils.collect_env.open", side_effect=OSError, create=True):
            c = SoftwareInfoCollector()
        self.assertIsNone(c.cuda_driver_version)

    # ---- parse_pip_freeze ----

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_pip_freeze_no_pip(self, _mock_which: MagicMock) -> None:
        with patch("quark.common.utils.collect_env.open", side_effect=OSError, create=True):
            c = SoftwareInfoCollector()
        self.assertEqual(c.parse_pip_freeze(), {})

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("pip"))
    @patch("quark.common.utils.collect_env.subprocess.run")
    def test_pip_freeze_mixed_output(self, mock_run: MagicMock, _mock_which: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout=PIP_FREEZE_SAMPLE)
        with patch("quark.common.utils.collect_env.open", side_effect=OSError, create=True):
            c = SoftwareInfoCollector()
        pkgs = c.parse_pip_freeze()
        self.assertEqual(pkgs["absl-py"], {"version": "2.1.0"})
        self.assertEqual(pkgs["torch"], {"version": "2.10.0+rocm7.1"})
        self.assertEqual(pkgs["amd-quark"], {"version": "0.12"})
        # Garbage lines must NOT appear.
        self.assertNotIn("bogus-line-no-equals", pkgs)
        self.assertNotIn("name", pkgs)  # the `name @ git+...` form has no `==`

    # ---- parse_conda_list ----

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_conda_list_no_conda(self, _mock_which: MagicMock) -> None:
        with patch("quark.common.utils.collect_env.open", side_effect=OSError, create=True):
            c = SoftwareInfoCollector()
        self.assertEqual(c.parse_conda_list(), {})

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory("conda"))
    @patch("quark.common.utils.collect_env.subprocess.run")
    def test_conda_list_mixed_output(self, mock_run: MagicMock, _mock_which: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout=CONDA_LIST_SAMPLE)
        with patch("quark.common.utils.collect_env.open", side_effect=OSError, create=True):
            c = SoftwareInfoCollector()
        pkgs = c.parse_conda_list()
        self.assertEqual(pkgs["absl-py"], {"version": "2.1.1"})
        self.assertEqual(pkgs["torch"], {"version": "2.10.0"})
        # Single-token line and comments must be dropped.
        self.assertNotIn("single_token", pkgs)

    # ---- collect_packages ----

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_collect_packages_merges_conda_over_pip(self, _mock_which: MagicMock) -> None:
        with patch("quark.common.utils.collect_env.open", side_effect=OSError, create=True):
            c = SoftwareInfoCollector()
        with (
            patch.object(c, "parse_pip_freeze", return_value={"absl-py": {"version": "1.0.0"}}),
            patch.object(
                c,
                "parse_conda_list",
                return_value={
                    "absl-py": {"version": "2.0.0"},
                    "torch": {"version": "2.10.0"},
                },
            ),
        ):
            merged = c.collect_packages()
        # conda wins on overlap; new keys are added.
        self.assertEqual(merged["absl-py"], {"version": "2.0.0"})
        self.assertEqual(merged["torch"], {"version": "2.10.0"})

    # ---- to_dict ----

    @patch("quark.common.utils.collect_env.shutil.which", side_effect=_which_factory())
    def test_to_dict_includes_os_driver_and_packages(self, _mock_which: MagicMock) -> None:
        with patch("quark.common.utils.collect_env.open", side_effect=OSError, create=True):
            c = SoftwareInfoCollector()
        c.os = "TestOS 1.0"
        c.python_version = "9.9.9"
        c.cuda_driver_version = "555.42.02"
        with patch.object(
            c,
            "collect_packages",
            return_value={
                "torch": {"version": "2.10.0"},
                "amd-quark": {"version": "0.12"},
                "broken": {},  # missing "version" key -> skipped
            },
        ):
            d = c.to_dict()
        self.assertEqual(d["os"], "TestOS 1.0")
        self.assertEqual(d["python"], "9.9.9")
        self.assertEqual(d["cuda_driver_version"], "555.42.02")
        self.assertEqual(d["torch"], "2.10.0")
        self.assertEqual(d["amd-quark"], "0.12")
        self.assertNotIn("broken", d)


# ---------------------------------------------------------------------------
# Top-level helpers
# ---------------------------------------------------------------------------


class TestCollectEnvironment(unittest.TestCase):
    @patch("quark.common.utils.collect_env.SoftwareInfoCollector")
    @patch("quark.common.utils.collect_env.HardwareInfoCollector")
    def test_collect_environment_shape(self, mock_hw_cls: MagicMock, mock_sw_cls: MagicMock) -> None:
        mock_hw_cls.return_value.to_dict.return_value = {"hw": 1}
        mock_sw_cls.return_value.to_dict.return_value = {"sw": 2}
        env = collect_environment()
        self.assertEqual(env, {"hardware": {"hw": 1}, "software": {"sw": 2}})


class TestSaveEnvironmentToJson(unittest.TestCase):
    def test_save_environment_to_json_happy_path(self) -> None:
        import tempfile
        from pathlib import Path

        env: dict[str, Any] = {"hardware": {"a": 1}, "software": {"b": 2}}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "env.json"
            save_environment_to_json(str(out), env, indent=4)
            loaded = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(loaded, env)

    def test_save_environment_to_json_open_raises(self) -> None:
        with (
            patch("quark.common.utils.collect_env.open", side_effect=OSError("disk full"), create=True),
            self.assertRaises(RuntimeError) as ctx,
        ):
            save_environment_to_json("/anywhere.json", {"x": 1})
        self.assertIn("Failed to save environment to JSON", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, OSError)


class TestPrintEnvironment(unittest.TestCase):
    def test_print_full_env(self) -> None:
        env: dict[str, Any] = {
            "hardware": {
                "accelerator_type": "AMD Instinct MI300X",
                "cpu_model": "AMD EPYC",
                "cuda_version": None,
                "rocm_version": "7.1.0",
            },
            "software": {
                "os": "Ubuntu 22.04.5 LTS",
                "python": "3.13.0",
                "torch": "2.10.0",
                "amd-quark": "0.12",
                "cuda_driver_version": None,
                "zzz-extra": "1.0",
                "another-pkg": "2.0",
            },
        }
        from contextlib import redirect_stdout
        from io import StringIO

        buf = StringIO()
        with redirect_stdout(buf):
            print_environment(env)
        text = buf.getvalue()

        # Hardware section assertions
        self.assertIn("Hardware Information", text)
        self.assertIn("accelerator_type: AMD Instinct MI300X", text)
        self.assertIn("rocm_version: 7.1.0", text)

        # Priority keys come before "Installed packages:"
        prio_idx = text.index("os: Ubuntu 22.04.5 LTS")
        extras_idx = text.index("Installed packages:")
        self.assertLess(prio_idx, extras_idx)
        for prio in ("os:", "python:", "torch:", "amd-quark:", "cuda_driver_version:"):
            self.assertIn(prio, text)

        # Other packages are listed sorted (another-pkg before zzz-extra).
        a_idx = text.index("another-pkg: 2.0")
        z_idx = text.index("zzz-extra: 1.0")
        self.assertLess(a_idx, z_idx)

    def test_print_with_empty_hardware_and_no_software(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO

        buf = StringIO()
        with redirect_stdout(buf):
            print_environment({})
        text = buf.getvalue()
        self.assertIn("Hardware Information", text)
        self.assertIn("Software Information", text)
        self.assertIn("Installed packages:", text)

    def test_print_only_priority_keys(self) -> None:
        # No "other" packages at all -> priority printed, "Installed packages:" still rendered.
        from contextlib import redirect_stdout
        from io import StringIO

        env = {
            "hardware": {},
            "software": {
                "os": "Ubuntu",
                "python": "3.13",
                "torch": "2.10",
                "amd-quark": "0.12",
                "cuda_driver_version": "555.42.02",
            },
        }
        buf = StringIO()
        with redirect_stdout(buf):
            print_environment(env)
        text = buf.getvalue()
        self.assertIn("cuda_driver_version: 555.42.02", text)
        # The header for the "other packages" list is unconditional.
        self.assertIn("Installed packages:", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
