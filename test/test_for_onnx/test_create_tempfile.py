#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import unittest
from pathlib import Path

from quark.onnx.utils.system_utils import create_tmp_dir, update_tmp_dir


class TestCreateTmpDir(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
