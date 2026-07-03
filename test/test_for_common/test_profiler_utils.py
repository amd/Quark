#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import subprocess
import unittest
from unittest.mock import MagicMock, patch

from quark.common.profiler.utils import (
    Hardware,
    detect_platform,
    get_cpu_memory,
    get_gpu_memory,
    init_gpu_profiling,
    init_psutil_profiling,
)


class TestHardwareEnum(unittest.TestCase):
    def test_hardware_enum_values(self):
        self.assertEqual(Hardware.ROCM, "rocm")
        self.assertEqual(Hardware.CUDA, "cuda")
        self.assertEqual(Hardware.CPU, "cpu")


class TestDetectPlatform(unittest.TestCase):
    def setUp(self):
        # Clear the cache before each test
        detect_platform.cache_clear()

    def tearDown(self):
        # Clear the cache after each test
        detect_platform.cache_clear()

    @patch("quark.common.profiler.utils.torch")
    def test_detect_rocm_platform(self, mock_torch):
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.cuda = None
        result = detect_platform()
        self.assertEqual(result, Hardware.ROCM)

    @patch("quark.common.profiler.utils.torch")
    def test_detect_cuda_platform(self, mock_torch):
        mock_torch.version.hip = None
        mock_torch.version.cuda = "12.1"
        result = detect_platform()
        self.assertEqual(result, Hardware.CUDA)

    @patch("quark.common.profiler.utils.torch")
    def test_detect_cpu_platform(self, mock_torch):
        mock_torch.version.hip = None
        mock_torch.version.cuda = None
        result = detect_platform()
        self.assertEqual(result, Hardware.CPU)

    @patch("quark.common.profiler.utils.torch")
    def test_detect_platform_cached(self, mock_torch):
        # First call
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.cuda = None
        result1 = detect_platform()
        self.assertEqual(result1, Hardware.ROCM)

        # Change mock values
        mock_torch.version.hip = None
        mock_torch.version.cuda = "12.1"

        # Second call should return cached value (ROCM), not new value
        result2 = detect_platform()
        self.assertEqual(result2, Hardware.ROCM)


class TestGetCPUMemory(unittest.TestCase):
    def setUp(self):
        # Reset cached process instance
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None

    def tearDown(self):
        # Reset cached process instance
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None

    @patch("quark.common.profiler.utils.psutil.Process")
    def test_get_cpu_memory_no_children(self, mock_process_class):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.children.return_value = []
        mock_process_class.return_value = mock_process

        result = get_cpu_memory()
        self.assertEqual(result, 100 * 1024 * 1024)

    @patch("quark.common.profiler.utils.psutil.Process")
    def test_get_cpu_memory_with_children(self, mock_process_class):
        mock_parent = MagicMock()
        mock_parent.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB

        mock_child1 = MagicMock()
        mock_child1.memory_info.return_value.rss = 50 * 1024 * 1024  # 50 MB

        mock_child2 = MagicMock()
        mock_child2.memory_info.return_value.rss = 30 * 1024 * 1024  # 30 MB

        mock_parent.children.return_value = [mock_child1, mock_child2]
        mock_process_class.return_value = mock_parent

        result = get_cpu_memory()
        self.assertEqual(result, 180 * 1024 * 1024)  # 100 + 50 + 30
        mock_parent.children.assert_called_with(recursive=True)

    @patch("quark.common.profiler.utils.psutil.Process")
    def test_get_cpu_memory_child_access_denied(self, mock_process_class):
        import psutil

        mock_parent = MagicMock()
        mock_parent.memory_info.return_value.rss = 100 * 1024 * 1024

        mock_good_child = MagicMock()
        mock_good_child.memory_info.return_value.rss = 50 * 1024 * 1024

        mock_bad_child = MagicMock()
        mock_bad_child.memory_info.side_effect = psutil.AccessDenied(1234, "test", "denied")

        mock_parent.children.return_value = [mock_good_child, mock_bad_child]
        mock_process_class.return_value = mock_parent

        result = get_cpu_memory()
        # Should only count parent + good_child
        self.assertEqual(result, 150 * 1024 * 1024)

    @patch("quark.common.profiler.utils.psutil.Process")
    def test_get_cpu_memory_child_no_such_process(self, mock_process_class):
        import psutil

        mock_parent = MagicMock()
        mock_parent.memory_info.return_value.rss = 100 * 1024 * 1024

        mock_good_child = MagicMock()
        mock_good_child.memory_info.return_value.rss = 50 * 1024 * 1024

        mock_terminated_child = MagicMock()
        mock_terminated_child.memory_info.side_effect = psutil.NoSuchProcess(1234, "test", "terminated")

        mock_parent.children.return_value = [mock_good_child, mock_terminated_child]
        mock_process_class.return_value = mock_parent

        result = get_cpu_memory()
        # Should only count parent + good_child
        self.assertEqual(result, 150 * 1024 * 1024)

    @patch("quark.common.profiler.utils.psutil.Process")
    def test_get_cpu_memory_uses_cached_process(self, mock_process_class):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process.children.return_value = []
        mock_process_class.return_value = mock_process

        # First call
        result1 = get_cpu_memory()
        # Second call
        result2 = get_cpu_memory()

        # Process should only be created once
        mock_process_class.assert_called_once()
        self.assertEqual(result1, 100 * 1024 * 1024)
        self.assertEqual(result2, 100 * 1024 * 1024)


class TestGetGPUMemoryROCm(unittest.TestCase):
    def setUp(self):
        detect_platform.cache_clear()

    def tearDown(self):
        detect_platform.cache_clear()

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=True)
    def test_rocm_single_gpu(self, mock_run, mock_torch):
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.cuda = None
        mock_torch.cuda.is_available.return_value = True

        # Mock rocm-smi output
        mock_result = MagicMock()
        mock_result.stdout = "GPU[0] VRAM Total Used Memory (B): 1073741824"  # 1GB
        mock_run.return_value = mock_result

        result = get_gpu_memory()
        self.assertEqual(result, 1073741824)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][0][0], "rocm-smi")
        self.assertIn("-d", call_args[0][0])
        self.assertIn("0", call_args[0][0])

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"HIP_VISIBLE_DEVICES": "1,2"}, clear=True)
    def test_rocm_multiple_gpus_hip_visible(self, mock_run, mock_torch):
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.cuda = None
        mock_torch.cuda.is_available.return_value = True

        # Mock rocm-smi outputs for GPU 1 and 2
        mock_results = [
            MagicMock(stdout="GPU[1] VRAM Total Used Memory (B): 536870912"),  # 512MB
            MagicMock(stdout="GPU[2] VRAM Total Used Memory (B): 1073741824"),  # 1GB
        ]
        mock_run.side_effect = mock_results

        result = get_gpu_memory()
        self.assertEqual(result, 536870912 + 1073741824)
        self.assertEqual(mock_run.call_count, 2)

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {}, clear=True)
    def test_rocm_no_env_var_uses_current_device(self, mock_run, mock_torch):
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.cuda = None
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 3

        mock_result = MagicMock()
        mock_result.stdout = "GPU[3] VRAM Total Used Memory (B): 2147483648"  # 2GB
        mock_run.return_value = mock_result

        result = get_gpu_memory()
        self.assertEqual(result, 2147483648)
        mock_torch.cuda.current_device.assert_called_once()

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=True)
    def test_rocm_subprocess_timeout(self, mock_run, mock_torch):
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.cuda = None
        mock_torch.cuda.is_available.return_value = True

        mock_run.side_effect = subprocess.TimeoutExpired("rocm-smi", 5)

        result = get_gpu_memory()
        self.assertEqual(result, 0)

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=True)
    def test_rocm_subprocess_error(self, mock_run, mock_torch):
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.cuda = None
        mock_torch.cuda.is_available.return_value = True

        mock_run.side_effect = subprocess.CalledProcessError(1, "rocm-smi")

        result = get_gpu_memory()
        self.assertEqual(result, 0)

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=True)
    def test_rocm_file_not_found(self, mock_run, mock_torch):
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.cuda = None
        mock_torch.cuda.is_available.return_value = True

        mock_run.side_effect = FileNotFoundError("rocm-smi not found")

        result = get_gpu_memory()
        self.assertEqual(result, 0)

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=True)
    def test_rocm_parsing_error(self, mock_run, mock_torch):
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.cuda = None
        mock_torch.cuda.is_available.return_value = True

        # Invalid output format
        mock_result = MagicMock()
        mock_result.stdout = "Invalid output format"
        mock_run.return_value = mock_result

        result = get_gpu_memory()
        # Should return 0 when GPU[0] line is not found
        self.assertEqual(result, 0)

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0, 1, 2"}, clear=True)
    def test_rocm_spaces_in_device_list(self, mock_run, mock_torch):
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.cuda = None
        mock_torch.cuda.is_available.return_value = True

        mock_results = [
            MagicMock(stdout="GPU[0] VRAM Total Used Memory (B): 100"),
            MagicMock(stdout="GPU[1] VRAM Total Used Memory (B): 200"),
            MagicMock(stdout="GPU[2] VRAM Total Used Memory (B): 300"),
        ]
        mock_run.side_effect = mock_results

        result = get_gpu_memory()
        self.assertEqual(result, 600)
        self.assertEqual(mock_run.call_count, 3)


class TestGetGPUMemoryCUDA(unittest.TestCase):
    def setUp(self):
        detect_platform.cache_clear()

    def tearDown(self):
        detect_platform.cache_clear()

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=True)
    def test_cuda_single_gpu(self, mock_run, mock_torch):
        mock_torch.version.hip = None
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.is_available.return_value = True

        # nvidia-smi returns memory in MiB
        mock_result = MagicMock()
        mock_result.stdout = "1024\n"  # 1024 MiB
        mock_run.return_value = mock_result

        result = get_gpu_memory()
        self.assertEqual(result, 1024 * 1024 * 1024)  # Convert to bytes

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][0][0], "nvidia-smi")
        self.assertIn("-i", call_args[0][0])
        self.assertIn("0", call_args[0][0])

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2"}, clear=True)
    def test_cuda_multiple_gpus(self, mock_run, mock_torch):
        mock_torch.version.hip = None
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.is_available.return_value = True

        # nvidia-smi returns one line per GPU
        mock_result = MagicMock()
        mock_result.stdout = "512\n1024\n2048\n"  # 512, 1024, 2048 MiB
        mock_run.return_value = mock_result

        result = get_gpu_memory()
        expected = (512 + 1024 + 2048) * 1024 * 1024  # Convert to bytes
        self.assertEqual(result, expected)

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {}, clear=True)
    def test_cuda_no_env_var_uses_current_device(self, mock_run, mock_torch):
        mock_torch.version.hip = None
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 2

        mock_result = MagicMock()
        mock_result.stdout = "2048\n"  # 2048 MiB
        mock_run.return_value = mock_result

        result = get_gpu_memory()
        self.assertEqual(result, 2048 * 1024 * 1024)
        mock_torch.cuda.current_device.assert_called_once()

        # Verify nvidia-smi was called with device ID "2"
        call_args = mock_run.call_args
        self.assertIn("2", call_args[0][0])

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=True)
    def test_cuda_subprocess_timeout(self, mock_run, mock_torch):
        mock_torch.version.hip = None
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.is_available.return_value = True

        mock_run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 5)

        result = get_gpu_memory()
        self.assertEqual(result, 0)

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=True)
    def test_cuda_subprocess_error(self, mock_run, mock_torch):
        mock_torch.version.hip = None
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.is_available.return_value = True

        mock_run.side_effect = subprocess.CalledProcessError(1, "nvidia-smi")

        result = get_gpu_memory()
        self.assertEqual(result, 0)

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=True)
    def test_cuda_file_not_found(self, mock_run, mock_torch):
        mock_torch.version.hip = None
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.is_available.return_value = True

        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

        result = get_gpu_memory()
        self.assertEqual(result, 0)

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=True)
    def test_cuda_parsing_error(self, mock_run, mock_torch):
        mock_torch.version.hip = None
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.is_available.return_value = True

        # Invalid output format
        mock_result = MagicMock()
        mock_result.stdout = "not_a_number\n"
        mock_run.return_value = mock_result

        result = get_gpu_memory()
        self.assertEqual(result, 0)

    @patch("quark.common.profiler.utils.torch")
    @patch("quark.common.profiler.utils.subprocess.run")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=True)
    def test_cuda_empty_lines_ignored(self, mock_run, mock_torch):
        mock_torch.version.hip = None
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.is_available.return_value = True

        # Output with empty lines
        mock_result = MagicMock()
        mock_result.stdout = "512\n\n1024\n\n"  # Empty lines should be ignored
        mock_run.return_value = mock_result

        result = get_gpu_memory()
        expected = (512 + 1024) * 1024 * 1024
        self.assertEqual(result, expected)


class TestGetGPUMemoryGeneral(unittest.TestCase):
    def setUp(self):
        detect_platform.cache_clear()

    def tearDown(self):
        detect_platform.cache_clear()

    @patch("quark.common.profiler.utils.torch")
    def test_gpu_not_available(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False

        result = get_gpu_memory()
        self.assertEqual(result, 0)

    @patch("quark.common.profiler.utils.torch")
    def test_cpu_platform_returns_zero(self, mock_torch):
        mock_torch.version.hip = None
        mock_torch.version.cuda = None
        mock_torch.cuda.is_available.return_value = True

        result = get_gpu_memory()
        self.assertEqual(result, 0)


class TestInitPsutilProfiling(unittest.TestCase):
    def setUp(self):
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None

    def tearDown(self):
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None

    @patch("quark.common.profiler.utils.psutil.Process")
    def test_init_psutil_success(self, mock_process_class):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process.children.return_value = []
        mock_process_class.return_value = mock_process

        enabled, process, base_memory = init_psutil_profiling()

        self.assertTrue(enabled)
        self.assertIsNotNone(process)
        self.assertEqual(base_memory, 100 * 1024 * 1024)

    @patch("quark.common.profiler.utils.psutil.Process")
    def test_init_psutil_failure(self, mock_process_class):
        mock_process_class.side_effect = Exception("psutil not available")

        enabled, process, base_memory = init_psutil_profiling()

        self.assertFalse(enabled)
        self.assertIsNone(process)
        self.assertEqual(base_memory, 0)


class TestInitGPUProfiling(unittest.TestCase):
    def setUp(self):
        detect_platform.cache_clear()

    def tearDown(self):
        detect_platform.cache_clear()

    @patch("quark.common.profiler.utils.torch")
    def test_init_gpu_not_available(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False

        result = init_gpu_profiling()
        self.assertFalse(result)

    @patch("quark.common.profiler.utils.torch")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=True)
    def test_init_gpu_rocm_with_cuda_visible(self, mock_torch):
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.cuda = None
        mock_torch.cuda.is_available.return_value = True

        result = init_gpu_profiling()
        self.assertTrue(result)

    @patch("quark.common.profiler.utils.torch")
    @patch.dict("os.environ", {"HIP_VISIBLE_DEVICES": "1,2"}, clear=True)
    def test_init_gpu_rocm_with_hip_visible(self, mock_torch):
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.cuda = None
        mock_torch.cuda.is_available.return_value = True

        result = init_gpu_profiling()
        self.assertTrue(result)

    @patch("quark.common.profiler.utils.torch")
    @patch.dict("os.environ", {}, clear=True)
    def test_init_gpu_rocm_no_env_var(self, mock_torch):
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.cuda = None
        mock_torch.cuda.is_available.return_value = True

        result = init_gpu_profiling()
        # Should still return True, but would have logged a warning
        self.assertTrue(result)

    @patch("quark.common.profiler.utils.torch")
    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=True)
    def test_init_gpu_cuda_with_visible_devices(self, mock_torch):
        mock_torch.version.hip = None
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.is_available.return_value = True

        result = init_gpu_profiling()
        self.assertTrue(result)

    @patch("quark.common.profiler.utils.torch")
    @patch.dict("os.environ", {}, clear=True)
    def test_init_gpu_cuda_no_env_var(self, mock_torch):
        mock_torch.version.hip = None
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.is_available.return_value = True

        result = init_gpu_profiling()
        # Should still return True, but would have logged a warning
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
