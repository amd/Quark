#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

from quark.common.profiler import (
    CheckpointMetric,
    GlobalProfiler,
    MetricContext,
    ProfileStep,
    SummaryMetric,
    profile_scope,
)
from quark.common.profiler.metrics.disk_checkpoint import DiskReadMbMetric, DiskWriteMbMetric
from quark.common.profiler.metrics.disk_summary import TotalDiskReadMbMetric, TotalDiskWriteMbMetric
from quark.common.profiler.utils import get_disk_io, init_disk_io_profiling
from quark.common.utils.testing_utils import require_torch_cuda


class TestProfiler(unittest.TestCase):
    def setUp(self):
        # Reset singleton before each test
        GlobalProfiler._instance = None
        # Reset cached process instance from utils
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None
        self.test_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.test_dir, "test_profile.yaml")

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        GlobalProfiler._instance = None
        # Reset cached process instance from utils
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None

    def test_singleton(self):
        p1 = GlobalProfiler()
        p2 = GlobalProfiler()
        self.assertIs(p1, p2)

    @patch.dict("os.environ", {"QUARK_PROFILING": "0"}, clear=True)
    def test_disabled_by_env(self):
        profiler = GlobalProfiler()
        self.assertFalse(profiler.enabled)

        profiler._checkpoint("step1")

        self.assertEqual(len(profiler.records), 0)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_enabled_usage_flow(self):
        # Mock psutil to ensure it returns predictable values
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)
            self.assertTrue(profiler.enabled)
            self.assertTrue(profiler._has_psutil)

            # Profiler auto-starts
            self.assertNotEqual(profiler.start_time, None)
            self.assertEqual(len(profiler.records), 1)  # Start
            self.assertEqual(profiler.records[0]["step"], "Start")
            self.assertEqual(profiler.records[0]["cpu_memory_mb"], 10.0)

            # Log step 1
            profiler._checkpoint("step_1")
            self.assertEqual(len(profiler.records), 2)
            self.assertEqual(profiler.records[1]["step"], "step_1")
            self.assertEqual(profiler.records[1]["cpu_memory_mb"], 10.0)

            # Change memory to 20MB
            mock_process.memory_info.return_value.rss = 20 * 1024 * 1024
            profiler._checkpoint("step_2")
            self.assertEqual(len(profiler.records), 3)
            self.assertEqual(profiler.records[2]["step"], "step_2")
            self.assertEqual(profiler.records[2]["cpu_memory_mb"], 20.0)

            profiler.stop()

            self.assertTrue(os.path.exists(self.output_file))

            with open(self.output_file) as f:
                content = f.read()
                self.assertIn("total_quantization_time_seconds", content)
                self.assertTrue("step: step_1" in content or 'step: "step_1"' in content)
                self.assertIn("cpu_memory_mb: 10.0", content)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_psutil_missing(self):
        # Simulate psutil initialization failure
        with patch("quark.common.profiler.utils.psutil.Process", side_effect=Exception("psutil not available")):
            profiler = GlobalProfiler()
            self.assertFalse(profiler.enabled)

            profiler._checkpoint("test")
            self.assertEqual(len(profiler.records), 0)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_peak_memory_linux(self):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        # Mock file reading in CPUPeakMemoryMetric._get_peak_memory
        mock_file_data = "Name: python\nVmHWM:    10240 kB\nVmRSS: 5000 kB\n"
        m_open = mock_open(read_data=mock_file_data)

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.summary.sys.platform", "linux"),
            patch("quark.common.profiler.metrics.summary.open", m_open),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)
            profiler.stop()

        # Read the actual output file
        with open(self.output_file) as f:
            content = f.read()
            self.assertIn("peak_memory_mb:", content)
            self.assertIn("10.0", content)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_peak_memory_windows(self):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.memory_info.return_value.peak_wset = 20 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.summary.sys.platform", "win32"),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)
            profiler.stop()

            with open(self.output_file) as f:
                content = f.read()
                self.assertIn("peak_memory_mb:", content)
                self.assertIn("20.0", content)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_child_process_memory_tracking(self):
        mock_parent = MagicMock()
        mock_parent.memory_info.return_value.rss = 10 * 1024 * 1024

        mock_child1 = MagicMock()
        mock_child1.memory_info.return_value.rss = 5 * 1024 * 1024
        mock_child1.pid = 1001

        mock_child2 = MagicMock()
        mock_child2.memory_info.return_value.rss = 8 * 1024 * 1024
        mock_child2.pid = 1002

        mock_parent.children.return_value = [mock_child1, mock_child2]

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_parent),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)
            self.assertTrue(profiler.enabled)

            self.assertEqual(profiler.records[0]["cpu_memory_mb"], 23.0)

            profiler._checkpoint("with_children")
            self.assertEqual(profiler.records[1]["cpu_memory_mb"], 23.0)

            profiler.stop()

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_child_process_memory_with_no_children(self):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            self.assertEqual(profiler.records[0]["cpu_memory_mb"], 10.0)

            profiler._checkpoint("no_children")
            self.assertEqual(profiler.records[1]["cpu_memory_mb"], 10.0)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_child_process_memory_recursive(self):
        mock_parent = MagicMock()
        mock_parent.memory_info.return_value.rss = 10 * 1024 * 1024

        mock_child = MagicMock()
        mock_child.memory_info.return_value.rss = 5 * 1024 * 1024
        mock_child.pid = 1001

        mock_grandchild = MagicMock()
        mock_grandchild.memory_info.return_value.rss = 3 * 1024 * 1024
        mock_grandchild.pid = 1002

        mock_parent.children.return_value = [mock_child, mock_grandchild]

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_parent),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            self.assertEqual(profiler.records[0]["cpu_memory_mb"], 18.0)

            mock_parent.children.assert_called_with(recursive=True)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_child_process_memory_changes_over_time(self):
        mock_parent = MagicMock()
        mock_parent.memory_info.return_value.rss = 10 * 1024 * 1024

        mock_child = MagicMock()
        mock_child.pid = 1001
        mock_child.memory_info.return_value.rss = 5 * 1024 * 1024

        mock_parent.children.return_value = [mock_child]

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_parent),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            self.assertEqual(profiler.records[0]["cpu_memory_mb"], 15.0)

            mock_child.memory_info.return_value.rss = 15 * 1024 * 1024
            profiler._checkpoint("child_allocated")
            self.assertEqual(profiler.records[1]["cpu_memory_mb"], 25.0)

            mock_parent.children.return_value = []
            profiler._checkpoint("child_terminated")
            self.assertEqual(profiler.records[2]["cpu_memory_mb"], 10.0)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_child_process_access_denied_is_skipped(self):
        import psutil

        real_process = psutil.Process()

        mock_good_child = MagicMock()
        mock_good_child.pid = 9001
        mock_good_child_info = MagicMock()
        mock_good_child_info.rss = 5 * 1024 * 1024
        mock_good_child.memory_info.return_value = mock_good_child_info

        mock_bad_child = MagicMock()
        mock_bad_child.pid = 9002
        mock_bad_child.memory_info.side_effect = psutil.AccessDenied(9002, "test_process", "Access denied")

        with (
            patch.object(real_process, "children", return_value=[mock_good_child, mock_bad_child]),
            patch("quark.common.profiler.utils.psutil.Process", return_value=real_process),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            first_memory = profiler.records[0]["cpu_memory_mb"]
            self.assertGreater(first_memory, 5.0)

            profiler.stop()

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_child_process_no_such_process_is_skipped(self):
        import psutil

        real_process = psutil.Process()

        mock_good_child = MagicMock()
        mock_good_child.pid = 8001
        mock_good_child_info = MagicMock()
        mock_good_child_info.rss = 8 * 1024 * 1024
        mock_good_child.memory_info.return_value = mock_good_child_info

        mock_terminated_child = MagicMock()
        mock_terminated_child.pid = 8002
        mock_terminated_child.memory_info.side_effect = psutil.NoSuchProcess(8002, "test_process", "Process terminated")

        with (
            patch.object(real_process, "children", return_value=[mock_good_child, mock_terminated_child]),
            patch("quark.common.profiler.utils.psutil.Process", return_value=real_process),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            first_memory = profiler.records[0]["cpu_memory_mb"]
            self.assertGreater(first_memory, 8.0)

            profiler.stop()

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_all_children_inaccessible(self):
        import psutil

        # Use a mock process with predictable memory value
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024

        mock_child1 = MagicMock()
        mock_child1.pid = 7001
        mock_child1.memory_info.side_effect = psutil.AccessDenied(7001, "test_process", "Access denied")

        mock_child2 = MagicMock()
        mock_child2.pid = 7002
        mock_child2.memory_info.side_effect = psutil.NoSuchProcess(7002, "test_process", "Process terminated")

        mock_process.children.return_value = [mock_child1, mock_child2]

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            # Should count only parent memory since all children are inaccessible
            first_memory = profiler.records[0]["cpu_memory_mb"]
            self.assertEqual(first_memory, 10.0)

            profiler.stop()

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_metric_definitions_in_output(self):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            profiler._checkpoint("test_step")
            profiler.stop()

            self.assertTrue(os.path.exists(self.output_file))

            with open(self.output_file) as f:
                content = f.read()

                # Verify metric definitions section is present
                self.assertIn("# Metric Definitions:", content)

                # Verify checkpoint metric definitions
                self.assertIn("# - step:", content)
                self.assertIn("# - timestamp:", content)
                self.assertIn("# - relative_time_secs:", content)
                self.assertIn("# - cpu_memory_mb:", content)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_register_custom_checkpoint_metric(self):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        # Define a custom checkpoint metric
        class CustomCounterMetric(CheckpointMetric):
            def __init__(self):
                self._counter = 0

            @property
            def name(self) -> str:
                return "custom_counter"

            def get_definition(self) -> str:
                return "A custom counter that increments at each checkpoint."

            def collect(self, context: MetricContext) -> int:
                self._counter += 1
                return self._counter

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            # Register custom metric
            custom_metric = CustomCounterMetric()
            profiler.checkpoint_metrics.append(custom_metric)

            # Trigger a new checkpoint to collect custom metric
            profiler._checkpoint("step_1")
            self.assertEqual(profiler.records[1]["custom_counter"], 1)

            profiler._checkpoint("step_2")
            self.assertEqual(profiler.records[2]["custom_counter"], 2)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_register_custom_summary_metric(self):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        # Define a custom summary metric
        class CustomSummaryMetric(SummaryMetric):
            @property
            def name(self) -> str:
                return "custom_summary"

            def get_definition(self) -> str:
                return "A custom summary metric for testing."

            def collect(self, context: MetricContext) -> str:
                return "test_value"

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            # Register custom metric
            profiler.summary_metrics.append(CustomSummaryMetric())

            profiler._checkpoint("test_step")
            profiler.stop()

            # Verify custom metric is in output
            with open(self.output_file) as f:
                content = f.read()
                self.assertIn("custom_summary", content)
                self.assertIn("test_value", content)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_default_metrics_registered(self):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            # Check that default summary metrics are registered
            summary_names = [m.name for m in profiler.summary_metrics]
            self.assertIn("total_quantization_time_seconds", summary_names)
            self.assertIn("peak_memory_mb", summary_names)
            self.assertIn("peak_gpu_memory_mb", summary_names)

            # Check that default checkpoint metrics are registered
            checkpoint_names = [m.name for m in profiler.checkpoint_metrics]
            self.assertIn("step", checkpoint_names)
            self.assertIn("timestamp", checkpoint_names)
            self.assertIn("relative_time_secs", checkpoint_names)
            self.assertIn("cpu_memory_mb", checkpoint_names)
            self.assertIn("gpu_memory_mb", checkpoint_names)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1", "CUDA_VISIBLE_DEVICES": "0"}, clear=True)
    def test_gpu_memory_tracking(self):
        # Test GPU memory tracking when GPU is available
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=True),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.profiler.get_gpu_memory", return_value=100 * 1024 * 1024),  # 100 MB
        ):
            profiler = GlobalProfiler(output_path=self.output_file)
            self.assertTrue(profiler._gpu_available)

            # Check that GPU memory is tracked in first checkpoint
            self.assertIn("gpu_memory_mb", profiler.records[0])
            self.assertEqual(profiler.records[0]["gpu_memory_mb"], 100.0)

            profiler.stop()

            # Verify GPU peak memory is in output
            with open(self.output_file) as f:
                content = f.read()
                self.assertIn("peak_gpu_memory_mb:", content)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_gpu_memory_not_available(self):
        # Test that GPU metrics are None when GPU is not available
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)
            self.assertFalse(profiler._gpu_available)

            # GPU memory should not be in records when GPU not available
            self.assertNotIn("gpu_memory_mb", profiler.records[0])

            profiler.stop()

    @patch.dict("os.environ", {"QUARK_PROFILING": "1", "CUDA_VISIBLE_DEVICES": "0"}, clear=True)
    def test_gpu_peak_memory_tracking(self):
        # Test that GPU peak memory is tracked across multiple checkpoints
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        # Need extra values: Start (auto), step_2, step_3, End (auto in stop())
        gpu_memory_values = [100 * 1024 * 1024, 200 * 1024 * 1024, 150 * 1024 * 1024, 140 * 1024 * 1024]
        gpu_memory_iter = iter(gpu_memory_values)

        def mock_gpu_memory():
            return next(gpu_memory_iter)

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=True),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.profiler.get_gpu_memory", side_effect=mock_gpu_memory),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            # First checkpoint (auto "Start"): 100 MB
            self.assertEqual(profiler.gpu_peak_memory, 100 * 1024 * 1024)

            # Second checkpoint: 200 MB (new peak)
            profiler._checkpoint("step_2")
            self.assertEqual(profiler.gpu_peak_memory, 200 * 1024 * 1024)

            # Third checkpoint: 150 MB (peak should stay at 200 MB)
            profiler._checkpoint("step_3")
            self.assertEqual(profiler.gpu_peak_memory, 200 * 1024 * 1024)

            profiler.stop()

            # Verify peak is 200 MB in output (peak should still be 200 MB even after End checkpoint with 140 MB)
            with open(self.output_file) as f:
                content = f.read()
                self.assertIn("peak_gpu_memory_mb: 200.0", content)

    @require_torch_cuda
    def test_log_torch_memory(self):
        GlobalProfiler.log_torch_memory("test_tag")
        GlobalProfiler.log_torch_memory()

    @patch("torch.cuda.is_available", return_value=False)
    def test_log_torch_memory_no_cuda(self, _mock_available: MagicMock) -> None:
        GlobalProfiler.log_torch_memory("test_tag")

    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3))
    @patch("torch.cuda.memory_reserved", return_value=2 * 1024**3)
    @patch("torch.cuda.memory_allocated", return_value=1 * 1024**3)
    @patch("torch.cuda.is_available", return_value=True)
    def test_log_torch_memory_mocked(
        self, _mock_available, _mock_allocated, _mock_reserved, _mock_mem_info, _mock_device_count
    ):
        GlobalProfiler.log_torch_memory("test_tag")
        GlobalProfiler.log_torch_memory()

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_get_global_profiler(self):
        # Test that GlobalProfiler returns the singleton instance
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler1 = GlobalProfiler(output_path=self.output_file)
            profiler2 = GlobalProfiler(output_path="different_path.yaml")  # Should ignore this

            # Should be the same instance
            self.assertIs(profiler1, profiler2)

            # Should use the first output path
            self.assertEqual(profiler1.output_path, self.output_file)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_scope_context_manager(self):
        # Test the scope() context manager
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            # Use scope context manager
            with profiler.scope("Test Operation"):
                pass

            # Should have Start, "Test Operation Start", and "Test Operation End" checkpoints
            self.assertEqual(len(profiler.records), 3)
            self.assertEqual(profiler.records[0]["step"], "Start")
            self.assertEqual(profiler.records[1]["step"], "Test Operation Start")
            self.assertEqual(profiler.records[2]["step"], "Test Operation End")

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_nested_scopes(self):
        # Test nested scope() context managers
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            with profiler.scope("Outer"), profiler.scope("Inner"):
                pass

            # Should have: Start, Outer Start, Inner Start, Inner End, Outer End
            self.assertEqual(len(profiler.records), 5)
            self.assertEqual(profiler.records[0]["step"], "Start")
            self.assertEqual(profiler.records[1]["step"], "Outer Start")
            self.assertEqual(profiler.records[2]["step"], "Inner Start")
            self.assertEqual(profiler.records[3]["step"], "Inner End")
            self.assertEqual(profiler.records[4]["step"], "Outer End")

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_profile_scope_decorator(self):
        # Test the profile_scope decorator
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            # Define a function with the decorator
            @profile_scope("Test Function")
            def my_function(x, y):
                return x + y

            result = my_function(2, 3)

            # Verify function works correctly
            self.assertEqual(result, 5)

            # Get the global profiler
            profiler = GlobalProfiler()

            # Should have: Start, "Test Function Start", "Test Function End"
            self.assertEqual(len(profiler.records), 3)
            self.assertEqual(profiler.records[1]["step"], "Test Function Start")
            self.assertEqual(profiler.records[2]["step"], "Test Function End")

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_profile_step_constants(self):
        # Test using ProfileStep constants
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            with profiler.scope(ProfileStep.MODEL_LOADING):
                pass

            # Verify ProfileStep constant was used correctly
            self.assertEqual(profiler.records[1]["step"], "Model Loading Start")
            self.assertEqual(profiler.records[2]["step"], "Model Loading End")

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_real_time_file_streaming(self):
        # Test that checkpoints are written to file in real-time
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
            patch(
                "quark.common.profiler.profiler.init_psutil_profiling",
                return_value=(True, mock_process, 10 * 1024 * 1024),
            ),
            patch("quark.common.profiler.profiler.get_cpu_memory", return_value=10 * 1024 * 1024),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            # Add a checkpoint
            profiler._checkpoint("step_1")

            # Flush file handle to ensure data is written
            if profiler._file_handle:
                profiler._file_handle.flush()

            # Read file without calling stop()
            with open(self.output_file) as f:
                content = f.read()

            # Verify checkpoint was written in real-time
            # Note: YAML format uses quotes for strings
            self.assertTrue('step: "Start"' in content or "step: Start" in content)
            self.assertTrue('step: "step_1"' in content or "step: step_1" in content)

            # Note: Summary metrics won't be there yet since stop() hasn't been called
            self.assertNotIn("total_quantization_time_seconds", content)

            profiler.stop()

    @patch.dict("os.environ", {"QUARK_PROFILING": "0"}, clear=True)
    def test_disabled_profiler_scope_no_op(self):
        # Test that scope() is a no-op when profiler is disabled
        profiler = GlobalProfiler()
        self.assertFalse(profiler.enabled)

        with profiler.scope("Test"):
            pass

        # Should have no records
        self.assertEqual(len(profiler.records), 0)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_relative_time_metric(self):
        # Test that relative time is calculated correctly
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            # First checkpoint should have relative_time_secs near 0
            self.assertAlmostEqual(profiler.records[0]["relative_time_secs"], 0.0, delta=0.1)

            # Add some delay
            import time

            time.sleep(0.1)

            profiler._checkpoint("step_1")

            # Second checkpoint should have positive relative_time_secs
            self.assertGreater(profiler.records[1]["relative_time_secs"], 0.0)
            self.assertLess(profiler.records[1]["relative_time_secs"], 1.0)  # Should be less than 1 second

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_total_time_metric(self):
        # Test that total_quantization_time_seconds is calculated correctly
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            import time

            time.sleep(0.1)

            profiler.stop()

            # Read output and verify total time
            with open(self.output_file) as f:
                content = f.read()
                self.assertIn("total_quantization_time_seconds:", content)

                # Extract the value
                import re

                match = re.search(r"total_quantization_time_seconds:\s+([\d.]+)", content)
                self.assertIsNotNone(match)
                total_time = float(match.group(1))
                self.assertGreater(total_time, 0.0)
                self.assertLess(total_time, 1.0)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_user_defined_scope_with_user_msg(self):
        """Test ProfileStep.USER_DEFINED with custom user_msg parameter."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            # Test context manager with user_msg
            with profiler.scope(ProfileStep.USER_DEFINED, user_msg="Custom Data Loading"):
                pass

            with profiler.scope(ProfileStep.USER_DEFINED, user_msg="Custom Validation"):
                pass

            profiler.stop()

            self.assertTrue(os.path.exists(self.output_file))

            with open(self.output_file) as f:
                content = f.read()

                # Verify custom messages appear in output
                self.assertIn('step: "Custom Data Loading Start"', content)
                self.assertIn('step: "Custom Data Loading End"', content)
                self.assertIn('step: "Custom Validation Start"', content)
                self.assertIn('step: "Custom Validation End"', content)

                # Verify USER_DEFINED constant itself doesn't appear
                self.assertNotIn('step: "User-Defined Start"', content)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_profile_scope_decorator_with_user_msg(self):
        """Test @profile_scope decorator with user_msg parameter."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            @profile_scope(ProfileStep.USER_DEFINED, user_msg="Custom Function")
            def custom_function():
                return "test"

            result = custom_function()
            self.assertEqual(result, "test")

            profiler.stop()

            self.assertTrue(os.path.exists(self.output_file))

            with open(self.output_file) as f:
                content = f.read()

                # Verify custom message appears in output
                self.assertIn('step: "Custom Function Start"', content)
                self.assertIn('step: "Custom Function End"', content)


class TestGetDiskIO(unittest.TestCase):
    """Unit tests for get_disk_io() in quark.common.profiler.utils."""

    def setUp(self):
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None

    def tearDown(self):
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None

    def test_get_disk_io_returns_read_and_write_bytes(self):
        """get_disk_io() returns a (read_bytes, write_bytes) tuple from the process."""
        mock_process = MagicMock()
        mock_process.io_counters.return_value.read_bytes = 100
        mock_process.io_counters.return_value.write_bytes = 200
        mock_process.children.return_value = []

        with patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process):
            read, write = get_disk_io()

        self.assertEqual(read, 100)
        self.assertEqual(write, 200)

    def test_get_disk_io_includes_child_process_counters(self):
        """get_disk_io() sums I/O from main process and all children."""
        mock_process = MagicMock()
        mock_process.io_counters.return_value.read_bytes = 100
        mock_process.io_counters.return_value.write_bytes = 50

        mock_child = MagicMock()
        mock_child.io_counters.return_value.read_bytes = 40
        mock_child.io_counters.return_value.write_bytes = 20
        mock_process.children.return_value = [mock_child]

        with patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process):
            read, write = get_disk_io()

        self.assertEqual(read, 140)
        self.assertEqual(write, 70)

    def test_get_disk_io_skips_terminated_child(self):
        """get_disk_io() skips children that raise NoSuchProcess and still returns parent counts."""
        import psutil

        mock_process = MagicMock()
        mock_process.io_counters.return_value.read_bytes = 100
        mock_process.io_counters.return_value.write_bytes = 50

        dead_child = MagicMock()
        dead_child.io_counters.side_effect = psutil.NoSuchProcess(9999, "dead")
        mock_process.children.return_value = [dead_child]

        with patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process):
            read, write = get_disk_io()

        self.assertEqual(read, 100)
        self.assertEqual(write, 50)

    def test_get_disk_io_skips_access_denied_child(self):
        """get_disk_io() skips children that raise AccessDenied and still returns parent counts."""
        import psutil

        mock_process = MagicMock()
        mock_process.io_counters.return_value.read_bytes = 200
        mock_process.io_counters.return_value.write_bytes = 80

        denied_child = MagicMock()
        denied_child.io_counters.side_effect = psutil.AccessDenied(9998, "denied")
        mock_process.children.return_value = [denied_child]

        with patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process):
            read, write = get_disk_io()

        self.assertEqual(read, 200)
        self.assertEqual(write, 80)

    def test_get_disk_io_access_denied_on_main_process_returns_zero(self):
        """get_disk_io() returns (0, 0) when the main process io_counters() raises AccessDenied (e.g. macOS)."""
        import psutil

        mock_process = MagicMock()
        mock_process.io_counters.side_effect = psutil.AccessDenied(os.getpid(), "test")

        with patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process):
            read, write = get_disk_io()

        self.assertEqual(read, 0)
        self.assertEqual(write, 0)

    def test_get_disk_io_not_implemented_returns_zero(self):
        """get_disk_io() returns (0, 0) when io_counters() raises NotImplementedError."""
        mock_process = MagicMock()
        mock_process.io_counters.side_effect = NotImplementedError("not supported on this OS")

        with patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process):
            read, write = get_disk_io()

        self.assertEqual(read, 0)
        self.assertEqual(write, 0)


class TestInitDiskIOProfiling(unittest.TestCase):
    """Unit tests for init_disk_io_profiling() in quark.common.profiler.utils."""

    def setUp(self):
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None

    def tearDown(self):
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None

    def test_init_disk_io_profiling_available(self):
        """Returns (True, baseline_read, baseline_write) when io_counters() succeeds."""
        mock_process = MagicMock()
        mock_process.io_counters.return_value.read_bytes = 1000
        mock_process.io_counters.return_value.write_bytes = 500
        mock_process.children.return_value = []

        with patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process):
            available, baseline_read, baseline_write = init_disk_io_profiling()

        self.assertTrue(available)
        self.assertEqual(baseline_read, 1000)
        self.assertEqual(baseline_write, 500)

    def test_init_disk_io_profiling_access_denied_returns_false(self):
        """Returns (False, 0, 0) when io_counters() raises AccessDenied (e.g. macOS without root)."""
        import psutil

        mock_process = MagicMock()
        mock_process.io_counters.side_effect = psutil.AccessDenied(os.getpid(), "test")

        with patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process):
            available, baseline_read, baseline_write = init_disk_io_profiling()

        self.assertFalse(available)
        self.assertEqual(baseline_read, 0)
        self.assertEqual(baseline_write, 0)

    def test_init_disk_io_profiling_not_implemented_returns_false(self):
        """Returns (False, 0, 0) when io_counters() raises NotImplementedError."""
        mock_process = MagicMock()
        mock_process.io_counters.side_effect = NotImplementedError("not supported")

        with patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process):
            available, baseline_read, baseline_write = init_disk_io_profiling()

        self.assertFalse(available)
        self.assertEqual(baseline_read, 0)
        self.assertEqual(baseline_write, 0)

    def test_init_disk_io_profiling_unexpected_error_returns_false(self):
        """Returns (False, 0, 0) when io_counters() raises an unexpected exception."""
        mock_process = MagicMock()
        mock_process.io_counters.side_effect = RuntimeError("unexpected")

        with patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process):
            available, baseline_read, baseline_write = init_disk_io_profiling()

        self.assertFalse(available)
        self.assertEqual(baseline_read, 0)
        self.assertEqual(baseline_write, 0)


class TestDiskCheckpointMetrics(unittest.TestCase):
    """Unit tests for DiskReadMbMetric and DiskWriteMbMetric."""

    def _make_context(self, disk_io_available, baseline_read, baseline_write, current_read, current_write):
        return MetricContext(
            disk_io_available=disk_io_available,
            baseline_disk_read_bytes=baseline_read,
            baseline_disk_write_bytes=baseline_write,
            current_disk_read_bytes=current_read,
            current_disk_write_bytes=current_write,
        )

    def test_disk_read_mb_basic(self):
        """DiskReadMbMetric computes (current_read - baseline_read) / 1 MB."""
        metric = DiskReadMbMetric()
        context = self._make_context(True, 0, 0, 5 * 1024 * 1024, 0)
        self.assertEqual(metric.collect(context), 5.0)

    def test_disk_write_mb_basic(self):
        """DiskWriteMbMetric computes (current_write - baseline_write) / 1 MB."""
        metric = DiskWriteMbMetric()
        context = self._make_context(True, 0, 0, 0, 10 * 1024 * 1024)
        self.assertEqual(metric.collect(context), 10.0)

    def test_disk_read_mb_subtracts_baseline(self):
        """DiskReadMbMetric correctly subtracts the baseline from the current counter."""
        metric = DiskReadMbMetric()
        # baseline = 1 MB, current = 3 MB → delta = 2 MB
        context = self._make_context(True, 1 * 1024 * 1024, 0, 3 * 1024 * 1024, 0)
        self.assertEqual(metric.collect(context), 2.0)

    def test_disk_write_mb_subtracts_baseline(self):
        """DiskWriteMbMetric correctly subtracts the baseline from the current counter."""
        metric = DiskWriteMbMetric()
        # baseline = 4 MB, current = 6 MB → delta = 2 MB
        context = self._make_context(True, 0, 4 * 1024 * 1024, 0, 6 * 1024 * 1024)
        self.assertEqual(metric.collect(context), 2.0)

    def test_disk_read_mb_not_available_returns_none(self):
        """DiskReadMbMetric returns None when disk_io_available is False."""
        metric = DiskReadMbMetric()
        context = self._make_context(False, 0, 0, 5 * 1024 * 1024, 0)
        self.assertIsNone(metric.collect(context))

    def test_disk_write_mb_not_available_returns_none(self):
        """DiskWriteMbMetric returns None when disk_io_available is False."""
        metric = DiskWriteMbMetric()
        context = self._make_context(False, 0, 0, 0, 5 * 1024 * 1024)
        self.assertIsNone(metric.collect(context))

    def test_disk_read_mb_negative_delta_clipped_to_zero(self):
        """DiskReadMbMetric clips negative deltas (counter wrap/reset) to 0.0."""
        metric = DiskReadMbMetric()
        # current (5 MB) < baseline (10 MB) → should not go negative
        context = self._make_context(True, 10 * 1024 * 1024, 0, 5 * 1024 * 1024, 0)
        self.assertEqual(metric.collect(context), 0.0)

    def test_disk_read_mb_fractional_value_rounded_to_two_decimals(self):
        """DiskReadMbMetric rounds to 2 decimal places."""
        metric = DiskReadMbMetric()
        # 1.5 MB = 1,572,864 bytes
        context = self._make_context(True, 0, 0, 1572864, 0)
        self.assertEqual(metric.collect(context), 1.5)


class TestDiskSummaryMetrics(unittest.TestCase):
    """Unit tests for TotalDiskReadMbMetric and TotalDiskWriteMbMetric."""

    def test_total_disk_read_mb_basic(self):
        """TotalDiskReadMbMetric subtracts baseline from the final read counter."""
        metric = TotalDiskReadMbMetric()
        context = MetricContext(
            disk_io_available=True,
            baseline_disk_read_bytes=2 * 1024 * 1024,
        )
        # Final counter = 7 MB → total during session = 5 MB
        with patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(7 * 1024 * 1024, 0)):
            result = metric.collect(context)
        self.assertEqual(result, 5.0)

    def test_total_disk_write_mb_basic(self):
        """TotalDiskWriteMbMetric subtracts baseline from the final write counter."""
        metric = TotalDiskWriteMbMetric()
        context = MetricContext(
            disk_io_available=True,
            baseline_disk_write_bytes=3 * 1024 * 1024,
        )
        # Final counter = 8 MB → total during session = 5 MB
        with patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 8 * 1024 * 1024)):
            result = metric.collect(context)
        self.assertEqual(result, 5.0)

    def test_total_disk_read_mb_not_available_returns_none(self):
        """TotalDiskReadMbMetric returns None when disk I/O is unavailable."""
        metric = TotalDiskReadMbMetric()
        context = MetricContext(disk_io_available=False)
        self.assertIsNone(metric.collect(context))

    def test_total_disk_write_mb_not_available_returns_none(self):
        """TotalDiskWriteMbMetric returns None when disk I/O is unavailable."""
        metric = TotalDiskWriteMbMetric()
        context = MetricContext(disk_io_available=False)
        self.assertIsNone(metric.collect(context))

    def test_total_disk_read_mb_negative_delta_clipped_to_zero(self):
        """TotalDiskReadMbMetric clips to 0.0 if final counter is below baseline."""
        metric = TotalDiskReadMbMetric()
        context = MetricContext(
            disk_io_available=True,
            baseline_disk_read_bytes=10 * 1024 * 1024,
        )
        with patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(5 * 1024 * 1024, 0)):
            result = metric.collect(context)
        self.assertEqual(result, 0.0)


class TestDiskIOProfilerIntegration(unittest.TestCase):
    """Integration tests for disk I/O profiling within GlobalProfiler."""

    def setUp(self):
        GlobalProfiler._instance = None
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None
        self.test_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.test_dir, "test_profile.yaml")

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        GlobalProfiler._instance = None
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None

    def _base_patches(self, disk_io_available=True, get_disk_io_return=(0, 0), disk_summary_return=(0, 0)):
        """Return the standard set of patches used by most integration tests."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []
        return mock_process, [
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch(
                "quark.common.profiler.profiler.init_disk_io_profiling",
                return_value=(disk_io_available, 0, 0),
            ),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=get_disk_io_return),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=disk_summary_return),
        ]

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_disk_io_checkpoint_metrics_present_when_available(self):
        """disk_read_mb and disk_write_mb keys appear in checkpoint records when disk I/O is available."""
        _, patches = self._base_patches(
            disk_io_available=True,
            get_disk_io_return=(1 * 1024 * 1024, 512 * 1024),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            profiler = GlobalProfiler(output_path=self.output_file)

            self.assertIn("disk_read_mb", profiler.records[0])
            self.assertIn("disk_write_mb", profiler.records[0])

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_disk_io_checkpoint_metrics_absent_when_unavailable(self):
        """disk_read_mb and disk_write_mb are absent from records when disk I/O is unavailable."""
        _, patches = self._base_patches(disk_io_available=False)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            profiler = GlobalProfiler(output_path=self.output_file)

            self.assertNotIn("disk_read_mb", profiler.records[0])
            self.assertNotIn("disk_write_mb", profiler.records[0])

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_disk_io_values_increase_over_checkpoints(self):
        """disk_read_mb and disk_write_mb grow as I/O counters advance between checkpoints."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        # Sequence: Start, step_1, step_2
        disk_io_sequence = iter(
            [
                (0, 0),  # Start (establishes baseline = 0)
                (5 * 1024 * 1024, 2 * 1024 * 1024),  # step_1: 5 MB read, 2 MB write
                (10 * 1024 * 1024, 4 * 1024 * 1024),  # step_2: 10 MB read, 4 MB write
            ]
        )

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", side_effect=disk_io_sequence),
            patch(
                "quark.common.profiler.metrics.disk_summary.get_disk_io",
                return_value=(10 * 1024 * 1024, 4 * 1024 * 1024),
            ),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

            # Start checkpoint: baseline = (0, 0), current = (0, 0) → delta = 0
            self.assertEqual(profiler.records[0]["disk_read_mb"], 0.0)
            self.assertEqual(profiler.records[0]["disk_write_mb"], 0.0)

            profiler._checkpoint("step_1")
            self.assertEqual(profiler.records[1]["disk_read_mb"], 5.0)
            self.assertEqual(profiler.records[1]["disk_write_mb"], 2.0)

            profiler._checkpoint("step_2")
            self.assertEqual(profiler.records[2]["disk_read_mb"], 10.0)
            self.assertEqual(profiler.records[2]["disk_write_mb"], 4.0)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_disk_io_summary_metrics_in_yaml_output(self):
        """total_disk_read_mb and total_disk_write_mb appear with correct values in the YAML file."""
        _, patches = self._base_patches(
            disk_io_available=True,
            get_disk_io_return=(0, 0),
            disk_summary_return=(3 * 1024 * 1024, 1 * 1024 * 1024),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            profiler = GlobalProfiler(output_path=self.output_file)
            profiler.stop()

        with open(self.output_file) as f:
            content = f.read()

        self.assertIn("total_disk_read_mb: 3.0", content)
        self.assertIn("total_disk_write_mb: 1.0", content)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_disk_io_summary_absent_when_unavailable(self):
        """total_disk_read_mb and total_disk_write_mb value lines are not written when disk I/O is unavailable.

        The metric definitions comments (# - total_disk_read_mb: ...) are still written unconditionally,
        but the actual data lines (total_disk_read_mb: <value>) must be absent.
        """
        _, patches = self._base_patches(disk_io_available=False)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            profiler = GlobalProfiler(output_path=self.output_file)
            profiler.stop()

        with open(self.output_file) as f:
            content = f.read()

        # Value lines start with the metric name at column 0 (no leading "# -")
        import re

        self.assertIsNone(re.search(r"^total_disk_read_mb:", content, re.MULTILINE))
        self.assertIsNone(re.search(r"^total_disk_write_mb:", content, re.MULTILINE))

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_disk_io_metric_definitions_in_yaml_output(self):
        """The YAML output contains definitions for all four disk I/O metrics."""
        _, patches = self._base_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            profiler = GlobalProfiler(output_path=self.output_file)
            profiler.stop()

        with open(self.output_file) as f:
            content = f.read()

        self.assertIn("# - disk_read_mb:", content)
        self.assertIn("# - disk_write_mb:", content)
        self.assertIn("# - total_disk_read_mb:", content)
        self.assertIn("# - total_disk_write_mb:", content)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_default_disk_io_metrics_registered(self):
        """GlobalProfiler registers disk I/O metrics in both checkpoint_metrics and summary_metrics by default."""
        _, patches = self._base_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            profiler = GlobalProfiler(output_path=self.output_file)

        checkpoint_names = [m.name for m in profiler.checkpoint_metrics]
        self.assertIn("disk_read_mb", checkpoint_names)
        self.assertIn("disk_write_mb", checkpoint_names)

        summary_names = [m.name for m in profiler.summary_metrics]
        self.assertIn("total_disk_read_mb", summary_names)
        self.assertIn("total_disk_write_mb", summary_names)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_disk_io_baseline_captured_at_init(self):
        """GlobalProfiler stores the baseline disk I/O counters returned by init_disk_io_profiling."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []

        with (
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch(
                "quark.common.profiler.profiler.init_disk_io_profiling",
                return_value=(True, 50 * 1024 * 1024, 20 * 1024 * 1024),
            ),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(50 * 1024 * 1024, 20 * 1024 * 1024)),
            patch(
                "quark.common.profiler.metrics.disk_summary.get_disk_io",
                return_value=(50 * 1024 * 1024, 20 * 1024 * 1024),
            ),
        ):
            profiler = GlobalProfiler(output_path=self.output_file)

        self.assertTrue(profiler._disk_io_available)
        self.assertEqual(profiler.baseline_disk_read, 50 * 1024 * 1024)
        self.assertEqual(profiler.baseline_disk_write, 20 * 1024 * 1024)

        # Checkpoint delta = current - baseline = 0 when counters haven't advanced
        self.assertEqual(profiler.records[0]["disk_read_mb"], 0.0)
        self.assertEqual(profiler.records[0]["disk_write_mb"], 0.0)


class TestCacheDirDiskMonitor(unittest.TestCase):
    """Unit tests for CacheDirDiskMonitor."""

    def _write_bytes(self, path: str, size_bytes: int) -> None:
        with open(path, "wb") as f:
            f.write(b"x" * size_bytes)

    def test_sample_sums_directory_contents(self):
        """_sample() returns the total size of files under the monitored directory."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_bytes(os.path.join(tmpdir, "a.bin"), 1 * 1024 * 1024)
            self._write_bytes(os.path.join(tmpdir, "b.bin"), 512 * 1024)

            monitor = CacheDirDiskMonitor(tmpdir)
            result = monitor._sample()

        self.assertEqual(result, 1.5)

    def test_sample_includes_nested_directories(self):
        """_sample() includes files from nested subdirectories."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested")
            os.makedirs(nested_dir)
            self._write_bytes(os.path.join(nested_dir, "c.bin"), 256 * 1024)

            monitor = CacheDirDiskMonitor(tmpdir)
            result = monitor._sample()

        self.assertEqual(result, 0.25)

    def test_sample_accepts_file_path(self):
        """_sample() returns the size of a file when the monitored path is a file."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "single.bin")
            self._write_bytes(file_path, 2 * 1024 * 1024)

            monitor = CacheDirDiskMonitor(file_path)
            result = monitor._sample()

        self.assertEqual(result, 2.0)

    def test_sample_missing_path_returns_none(self):
        """_sample() returns None when the monitored path does not exist."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = CacheDirDiskMonitor(os.path.join(tmpdir, "missing"))
            result = monitor._sample()
        self.assertIsNone(result)

    def test_sample_unexpected_exception_returns_none(self):
        """_sample() returns None on any unexpected exception."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = CacheDirDiskMonitor(tmpdir)
            with patch.object(monitor, "_get_path_size_bytes", side_effect=RuntimeError("unexpected error")):
                result = monitor._sample()
        self.assertIsNone(result)

    def test_peak_mb_initial_value_is_zero(self):
        """peak_mb starts at 0.0 before any samples are taken."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = CacheDirDiskMonitor(tmpdir)
            self.assertEqual(monitor.peak_mb, 0.0)
            self.assertEqual(monitor.initial_mb, 0.0)

    def test_start_records_initial_disk_usage(self):
        """start() stores the initial disk usage before background sampling begins."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = CacheDirDiskMonitor(tmpdir)
            with patch.object(monitor, "_run"), patch.object(monitor, "_sample", return_value=12.5):
                monitor.start()

            self.assertEqual(monitor.initial_mb, 12.5)
            self.assertEqual(monitor.peak_mb, 0.0)

    def test_run_tracks_peak_across_samples_relative_to_initial_usage(self):
        """_run() updates peak_mb to the maximum increase above the initial disk usage."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = CacheDirDiskMonitor(tmpdir, interval=0.001)
            monitor.initial_mb = 100.0
            sample_sequence = [100.0, 250.0, 150.0]
            call_count = [0]

            def fake_sample():
                if call_count[0] < len(sample_sequence):
                    val = sample_sequence[call_count[0]]
                    call_count[0] += 1
                    return val
                monitor._stop_event.set()
                return None

            with patch.object(monitor, "_sample", side_effect=fake_sample):
                monitor._run()

            self.assertEqual(monitor.peak_mb, 150.0)

    def test_stop_takes_final_sample_and_updates_peak_relative_to_initial_usage(self):
        """stop() updates peak_mb using the final sample minus the initial disk usage."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = CacheDirDiskMonitor(tmpdir)
            monitor.peak_mb = 50.0
            monitor.initial_mb = 80.0

            with patch.object(monitor, "_run"):  # thread body is a no-op
                with patch.object(monitor, "_sample", return_value=80.0):
                    monitor.start()
                with patch.object(monitor, "_sample", return_value=200.0):
                    monitor.stop()

            self.assertEqual(monitor.peak_mb, 120.0)

    def test_stop_does_not_decrease_peak(self):
        """stop() does not lower peak_mb when the final sample is below the current peak."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = CacheDirDiskMonitor(tmpdir)
            monitor.peak_mb = 300.0
            monitor.initial_mb = 50.0

            with patch.object(monitor, "_run"):
                with patch.object(monitor, "_sample", return_value=50.0):
                    monitor.start()
                with patch.object(monitor, "_sample", return_value=100.0):
                    monitor.stop()

            self.assertEqual(monitor.peak_mb, 300.0)

    def test_stop_handles_none_final_sample(self):
        """stop() does not change peak_mb when the final _sample() call returns None."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = CacheDirDiskMonitor(tmpdir)
            monitor.peak_mb = 150.0
            monitor.initial_mb = 20.0

            with patch.object(monitor, "_run"):
                with patch.object(monitor, "_sample", return_value=20.0):
                    monitor.start()
                with patch.object(monitor, "_sample", return_value=None):
                    monitor.stop()

            self.assertEqual(monitor.peak_mb, 150.0)

    def test_thread_is_daemon(self):
        """The background monitoring thread is a daemon thread."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = CacheDirDiskMonitor(tmpdir)
            self.assertTrue(monitor._thread.daemon)


class TestPeakCacheDirDiskUsageMbMetric(unittest.TestCase):
    """Unit tests for PeakCacheDirDiskUsageMbMetric."""

    def test_metric_name(self):
        """Metric name is 'peak_cache_dir_disk_usage_mb'."""
        from quark.common.profiler.metrics.cache_dir_summary import PeakCacheDirDiskUsageMbMetric

        metric = PeakCacheDirDiskUsageMbMetric()
        self.assertEqual(metric.name, "peak_cache_dir_disk_usage_mb")

    def test_collect_returns_context_value(self):
        """collect() returns the cache_dir_peak_disk_mb value from the context."""
        from quark.common.profiler.metrics.cache_dir_summary import PeakCacheDirDiskUsageMbMetric

        metric = PeakCacheDirDiskUsageMbMetric()
        context = MetricContext(cache_dir_peak_disk_mb=512.5)
        self.assertEqual(metric.collect(context), 512.5)

    def test_collect_returns_none_when_not_set(self):
        """collect() returns None when cache_dir_peak_disk_mb is None (no cache dir monitored)."""
        from quark.common.profiler.metrics.cache_dir_summary import PeakCacheDirDiskUsageMbMetric

        metric = PeakCacheDirDiskUsageMbMetric()
        context = MetricContext(cache_dir_peak_disk_mb=None)
        self.assertIsNone(metric.collect(context))

    def test_get_definition_mentions_scandir(self):
        """get_definition() mentions os.scandir() so users know how the metric is measured."""
        from quark.common.profiler.metrics.cache_dir_summary import PeakCacheDirDiskUsageMbMetric

        metric = PeakCacheDirDiskUsageMbMetric()
        self.assertIn("os.scandir()", metric.get_definition())


class TestPeakCacheDirIntegration(unittest.TestCase):
    """Integration tests for peak cache dir disk usage within GlobalProfiler."""

    def setUp(self):
        GlobalProfiler._instance = None
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None
        self.test_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.test_dir, "test_profile.yaml")

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        GlobalProfiler._instance = None
        import quark.common.profiler.utils

        quark.common.profiler.utils._PROCESS_INSTANCE = None

    def _base_patches(self):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 10 * 1024 * 1024
        mock_process.children.return_value = []
        return mock_process, [
            patch("quark.common.profiler.utils.psutil.Process", return_value=mock_process),
            patch("quark.common.profiler.profiler._is_package_available", return_value=(True, "5.9.0")),
            patch("quark.common.profiler.profiler.init_gpu_profiling", return_value=False),
            patch("quark.common.profiler.profiler.init_disk_io_profiling", return_value=(True, 0, 0)),
            patch("quark.common.profiler.profiler.get_disk_io", return_value=(0, 0)),
            patch("quark.common.profiler.metrics.disk_summary.get_disk_io", return_value=(0, 0)),
        ]

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_peak_cache_dir_metric_registered_in_summary_metrics(self):
        """PeakCacheDirDiskUsageMbMetric is registered in GlobalProfiler.summary_metrics by default."""
        _, patches = self._base_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            profiler = GlobalProfiler(output_path=self.output_file)

        summary_names = [m.name for m in profiler.summary_metrics]
        self.assertIn("peak_cache_dir_disk_usage_mb", summary_names)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_start_cache_dir_monitoring_creates_and_starts_monitor(self):
        """start_cache_dir_monitoring() creates a CacheDirDiskMonitor and calls start() on it."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        _, patches = self._base_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            profiler = GlobalProfiler(output_path=self.output_file)

            mock_monitor = MagicMock(spec=CacheDirDiskMonitor)
            with (
                tempfile.TemporaryDirectory() as test_dir,
                patch("quark.common.profiler.profiler.CacheDirDiskMonitor", return_value=mock_monitor),
            ):
                profiler.start_cache_dir_monitoring(test_dir)

        mock_monitor.start.assert_called_once()

    @patch.dict("os.environ", {"QUARK_PROFILING": "0"}, clear=True)
    def test_start_cache_dir_monitoring_is_noop_when_disabled(self):
        """start_cache_dir_monitoring() is a no-op when profiling is disabled."""
        profiler = GlobalProfiler(output_path=self.output_file)
        self.assertFalse(profiler.enabled)

        with tempfile.TemporaryDirectory() as test_dir:
            profiler.start_cache_dir_monitoring(test_dir)

        self.assertEqual(profiler._cache_dir_monitors, [])

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_start_cache_dir_monitoring_tracks_multiple_monitors(self):
        """start_cache_dir_monitoring() starts a monitor for each created cache directory."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        _, patches = self._base_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            profiler = GlobalProfiler(output_path=self.output_file)

            first_monitor = MagicMock(spec=CacheDirDiskMonitor)
            second_monitor = MagicMock(spec=CacheDirDiskMonitor)
            test_dir_1 = tempfile.TemporaryDirectory()
            test_dir_2 = tempfile.TemporaryDirectory()

            try:
                with patch(
                    "quark.common.profiler.profiler.CacheDirDiskMonitor", side_effect=[first_monitor, second_monitor]
                ):
                    profiler.start_cache_dir_monitoring(test_dir_1.name)
                    profiler.start_cache_dir_monitoring(test_dir_2.name)
            finally:
                test_dir_1.cleanup()
                test_dir_2.cleanup()

        first_monitor.start.assert_called_once()
        second_monitor.start.assert_called_once()
        self.assertEqual(profiler._cache_dir_monitors, [first_monitor, second_monitor])

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_stop_writes_highest_peak_cache_dir_metric_to_yaml(self):
        """stop() writes the highest peak_cache_dir_disk_usage_mb to the YAML file."""
        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        _, patches = self._base_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            profiler = GlobalProfiler(output_path=self.output_file)

            first_monitor = MagicMock(spec=CacheDirDiskMonitor)
            first_monitor.peak_mb = 1024.0
            second_monitor = MagicMock(spec=CacheDirDiskMonitor)
            second_monitor.peak_mb = 2048.0
            test_dir_1 = tempfile.TemporaryDirectory()
            test_dir_2 = tempfile.TemporaryDirectory()

            try:
                with patch(
                    "quark.common.profiler.profiler.CacheDirDiskMonitor", side_effect=[first_monitor, second_monitor]
                ):
                    profiler.start_cache_dir_monitoring(test_dir_1.name)
                    profiler.start_cache_dir_monitoring(test_dir_2.name)
            finally:
                test_dir_1.cleanup()
                test_dir_2.cleanup()

            profiler.stop()

        with open(self.output_file) as f:
            content = f.read()
        self.assertIn("peak_cache_dir_disk_usage_mb: 2048.0", content)

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_stop_omits_peak_cache_dir_metric_when_no_monitor(self):
        """stop() omits peak_cache_dir_disk_usage_mb from YAML when no cache dir was monitored."""
        import re

        _, patches = self._base_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            profiler = GlobalProfiler(output_path=self.output_file)
            profiler.stop()

        with open(self.output_file) as f:
            content = f.read()
        self.assertIsNone(re.search(r"^peak_cache_dir_disk_usage_mb:", content, re.MULTILINE))

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_stop_omits_metric_when_peak_is_zero(self):
        """stop() omits peak_cache_dir_disk_usage_mb when all monitor peaks are 0."""
        import re

        from quark.common.profiler.cache_dir_disk_monitor import CacheDirDiskMonitor

        _, patches = self._base_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            profiler = GlobalProfiler(output_path=self.output_file)

            mock_monitor = MagicMock(spec=CacheDirDiskMonitor)
            mock_monitor.peak_mb = 0.0

            with (
                tempfile.TemporaryDirectory() as test_dir,
                patch("quark.common.profiler.profiler.CacheDirDiskMonitor", return_value=mock_monitor),
            ):
                profiler.start_cache_dir_monitoring(test_dir)

            profiler.stop()

        with open(self.output_file) as f:
            content = f.read()
        self.assertIsNone(re.search(r"^peak_cache_dir_disk_usage_mb:", content, re.MULTILINE))

    @patch.dict("os.environ", {"QUARK_PROFILING": "1"}, clear=True)
    def test_metric_definition_in_yaml(self):
        """The YAML output contains a definition comment for peak_cache_dir_disk_usage_mb."""
        _, patches = self._base_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            profiler = GlobalProfiler(output_path=self.output_file)
            profiler.stop()

        with open(self.output_file) as f:
            content = f.read()
        self.assertIn("# - peak_cache_dir_disk_usage_mb:", content)


if __name__ == "__main__":
    unittest.main()
