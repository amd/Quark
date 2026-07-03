#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from quark.onnx.utils.system_utils import create_tmp_dir, update_tmp_dir


class TestCreateTmpDir(unittest.TestCase):
    def setUp(self):
        update_tmp_dir(None)

    def test_created_under_cur_dir(self, path: str = "unittest.TestCreateTmpDir."):
        update_tmp_dir(".")
        with create_tmp_dir(prefix=path) as tmp_dir:
            abs_path = os.path.join(os.getcwd(), tmp_dir)
            self.assertTrue(os.path.exists(abs_path), f"tmp_dir {abs_path} is NOT created properly.")

    def test_created_under_assigned_dir(self, path: str = "unittest.TestCreateTmpDir."):
        parent_path = str(Path(os.getcwd()).parent)
        update_tmp_dir(parent_path)
        with create_tmp_dir(prefix=path) as tmp_dir:
            self.assertTrue(os.path.exists(tmp_dir), f"tmp_dir {tmp_dir} is NOT created properly.")

    def test_registers_cache_dir_monitor_when_tmp_dir_is_none(self, path: str = "unittest.TestCreateTmpDir."):
        profiler = MagicMock()
        with (
            patch("quark.common.profiler.GlobalProfiler", return_value=profiler),
            create_tmp_dir(prefix=path) as tmp_dir,
        ):
            self.assertTrue(os.path.exists(tmp_dir), f"tmp_dir {tmp_dir} is NOT created properly.")

        profiler.start_cache_dir_monitoring.assert_called_once_with(tmp_dir)

    def test_registers_cache_dir_monitor_when_tmp_dir_is_assigned(self, path: str = "unittest.TestCreateTmpDir."):
        update_tmp_dir(".")
        profiler = MagicMock()
        with (
            patch("quark.common.profiler.GlobalProfiler", return_value=profiler),
            create_tmp_dir(prefix=path) as tmp_dir,
        ):
            self.assertTrue(os.path.exists(tmp_dir), f"tmp_dir {tmp_dir} is NOT created properly.")

        profiler.start_cache_dir_monitoring.assert_called_once_with(tmp_dir)


if __name__ == "__main__":
    unittest.main()
