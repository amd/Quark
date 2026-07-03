#
# Copyright (C) 2025 - 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Tests for load_coco_prompts in examples/torch/diffusers/quantize_diffusers.py.

Verifies safe handling of malformed lines in COCO2014 caption TSV files.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "examples", "torch", "diffusers"),
)

from quantize_diffusers import load_coco_prompts  # noqa: E402


class TestLoadCocoPrompts(unittest.TestCase):
    """Tests for load_coco_prompts (column-2 captions, header skipped, stripped)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def _create_file(self, content):
        path = os.path.join(self.temp_dir, "captions.tsv")
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_valid_file_with_tabs(self):
        """Parse a valid TSV: column 2 captions, header skipped, whitespace stripped."""
        content = "header1\theader2\theader3\nfield1\tfield2\tprompt1\nfield1\tfield2\tprompt2\n"
        path = self._create_file(content)

        self.assertEqual(load_coco_prompts(path), ["prompt1", "prompt2"])

    def test_malformed_line_with_insufficient_fields(self):
        """Lines with fewer than 3 tab-separated fields are skipped, not errors."""
        content = (
            "header1\theader2\theader3\nfield1\tfield2\tprompt1\nmalformed_line_no_tabs\nfield1\tfield2\tprompt2\n"
        )
        path = self._create_file(content)

        self.assertEqual(load_coco_prompts(path), ["prompt1", "prompt2"])

    def test_empty_line(self):
        """Empty lines are skipped."""
        content = "header1\theader2\theader3\nfield1\tfield2\tprompt1\n\nfield1\tfield2\tprompt2\n"
        path = self._create_file(content)

        self.assertEqual(load_coco_prompts(path), ["prompt1", "prompt2"])

    def test_skips_header_line(self):
        """The first line (header) is skipped."""
        content = "header1\theader2\theader_prompt\nfield1\tfield2\tprompt1\n"
        path = self._create_file(content)

        self.assertEqual(load_coco_prompts(path), ["prompt1"])

    def test_all_malformed_lines(self):
        """A file whose data lines are all malformed yields no prompts."""
        content = "header1\theader2\theader3\nno_tabs_here\nalso_no_tabs\n"
        path = self._create_file(content)

        self.assertEqual(load_coco_prompts(path), [])

    def test_limit(self):
        """The limit argument caps the number of returned prompts."""
        content = "h1\th2\th3\nf\tf\tp1\nf\tf\tp2\nf\tf\tp3\n"
        path = self._create_file(content)

        self.assertEqual(load_coco_prompts(path, limit=2), ["p1", "p2"])


if __name__ == "__main__":
    unittest.main()
