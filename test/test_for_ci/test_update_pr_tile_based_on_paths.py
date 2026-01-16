import os
import unittest
from contextlib import contextmanager
from unittest.mock import mock_open, patch

import update_pr_title_based_on_paths

# This is a sample of what the YAML config would be loaded as.
# We use this to simulate Github Actions PR creation events
MOCK_YAML_CONFIG = [
    {"path": ".github/workflows/", "prefix": "[CI]", "label": "ci"},
    {"path": "docs/", "prefix": "[docs]", "label": "documentation"},
    {"path": "src/backend/", "prefix": "[BE]", "label": "backend"},
    {"path": "src/contrib/", "prefix": "[contrib]"},
    {"path": "src/contrib/external/", "prefix": "[external]", "label": "contrib: external"},
    {"path": "src/frontend/", "prefix": "[FE]", "label": "frontend"},
    {"path": "tests/", "prefix": "[tests]", "label": "tests"},
]


class TestUpdatePR(unittest.TestCase):
    def setUp(self):
        """Set up a mock for the GITHUB_OUTPUT file."""
        self.mock_github_output = ""

    @contextmanager
    def mock_context(self, mock_env, mock_config):
        # This function will be the new implementation for open()
        def _mock_open(path, *args, **kwargs):
            """This function will be the new implementation for open()

            When the script tries to open the .github/config/pr_metadata_path_map.yml map file, returns a mock map config.
            This is how we mock the `path: prefix` mapping to the unit tests.

            When the script tries to open the GITHUB_OUTPUT file to write the output, use a custom mock
            that lets us capture what's written to it. This is how we can check the script returned result.
            """
            if path == update_pr_title_based_on_paths.MAP_CONFIG_PATH:
                return mock_open(read_data=mock_config)()
            elif path == os.getenv("GITHUB_OUTPUT"):

                class MockFile:
                    def __enter__(self):
                        return self

                    def __exit__(self, exc_type, exc_val, exc_tb):
                        pass

                    def write(self, content):
                        self.test_instance.mock_github_output += content

                # We need to link the test instance to the mock file
                mock_file = MockFile()
                mock_file.test_instance = self
                return mock_file
            else:
                # For any other file, fall back to the default mock_open behavior
                return mock_open()()

        # Add GITHUB_OUTPUT to the mock environment if not already present
        # this is just a placeholder path since we are mocking open()
        if "GITHUB_OUTPUT" not in mock_env:
            mock_env["GITHUB_OUTPUT"] = "/tmp/github_output"

        with (
            patch.dict("os.environ", mock_env),
            patch("yaml.safe_load", return_value=MOCK_YAML_CONFIG),
            patch("builtins.open", side_effect=_mock_open),
        ):
            yield


class TestUpdateTitle(TestUpdatePR):
    def _update_pr_title_with_mocks(self, mock_env, mock_config):
        """Helper function to run the update_pr_based_on_paths with mocked dependencies."""

        # Reset mock output at the beginning of each call to ensure isolation
        self.mock_github_output = ""

        # Mock the environment variables, the yaml.safe_load, and the open function
        with self.mock_context(mock_env, mock_config):
            update_pr_title_based_on_paths.main("title")

        # Return the captured output title
        assert "title=" in self.mock_github_output, "No title= found in github output"
        return self.mock_github_output.split("=")[1].strip()

    def _update_pr_labels_with_mocks(self, mock_env, mock_config):
        """Helper function to run the update_pr_based_on_paths with mocked dependencies."""

        # Reset mock output at the beginning of each call to ensure isolation
        self.mock_github_output = ""

        # Mock the environment variables, the yaml.safe_load, and the open function
        with self.mock_context(mock_env, mock_config):
            update_pr_title_based_on_paths.main("labels")

        # Return the captured output title
        assert "labels=" in self.mock_github_output, "No labels= found in github output"
        return self.mock_github_output.split("=")[1].strip()

    def test_add_single_prefix(self):
        """Adds a single prefix to a title with no prefixes."""
        env = {"MODIFIED_FILES": "docs/guide.md", "PR_TITLE": "Add new guide"}
        new_title = self._update_pr_title_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_title, "[docs] Add new guide")
        new_labels = self._update_pr_labels_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_labels, "documentation")

    def test_add_multiple_prefixes(self):
        """Adds multiple, sorted prefixes to a title with no prefixes."""
        env = {"MODIFIED_FILES": "src/frontend/app.js docs/api.md", "PR_TITLE": "Full stack feature"}
        new_title = self._update_pr_title_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_title, "[FE][docs] Full stack feature")
        new_labels = self._update_pr_labels_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_labels, "documentation,frontend")

    def test_merge_with_unrelated_prefix(self):
        """Adds required prefixes while preserving a manual one like [WIP]."""
        env = {"MODIFIED_FILES": "src/backend/server.py", "PR_TITLE": "[WIP] New API endpoint"}
        new_title = self._update_pr_title_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_title, "[BE] [WIP] New API endpoint")
        new_labels = self._update_pr_labels_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_labels, "backend")

    def test_merge_with_unrelated_prefix_without_separator(self):
        """Adds required prefixes while preserving a manual one like [my-tmp] without a separator."""
        env = {"MODIFIED_FILES": "src/backend/server.py", "PR_TITLE": "[my-tmp]New API endpoint"}
        new_title = self._update_pr_title_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_title, "[BE] [my-tmp] New API endpoint")
        new_labels = self._update_pr_labels_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_labels, "backend")

    def test_add_missing_prefix_to_existing_block(self):
        """Adds a missing prefix to an existing block of managed prefixes."""
        env = {"MODIFIED_FILES": "src/backend/server.py docs/api.md", "PR_TITLE": "[docs] Update docs and API"}
        new_title = self._update_pr_title_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_title, "[BE][docs] Update docs and API")
        new_labels = self._update_pr_labels_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_labels, "backend,documentation")

    def test_no_change_if_prefixes_are_correct_with_upper_case_prefix(self):
        """Does not change the title if all required prefixes are already present."""
        env = {"MODIFIED_FILES": "src/frontend/app.js", "PR_TITLE": "[FE] Update component"}
        new_title = self._update_pr_title_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_title, "[FE] Update component")
        new_labels = self._update_pr_labels_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_labels, "frontend")

    def test_no_change_if_prefixes_are_correct_with_lower_case_prefix(self):
        env = {
            "MODIFIED_FILES": "docs/source/index.rst pyproject.toml quark/experimental/cli/main.py quark/experimental/cli/requirements.txt",
            "PR_TITLE": "[CLI][docs][feat][onnx] Add first pass for quark onnx adapter",
        }
        new_title = self._update_pr_title_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_title, "[docs] [CLI][feat][onnx] Add first pass for quark onnx adapter")
        new_labels = self._update_pr_labels_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_labels, "documentation")

    def test_no_change_if_multiple_prefixes_are_correct(self):
        """Does not change the title if multiple prefixes are present and correct."""
        env = {"MODIFIED_FILES": "src/frontend/app.js docs/api.md", "PR_TITLE": "[docs][FE] Full stack feature"}
        new_title = self._update_pr_title_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_title, "[docs][FE] Full stack feature")
        new_labels = self._update_pr_labels_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_labels, "documentation,frontend")

    def test_multiple_matched_prefixes(self):
        """Adds all required prefixes for a directory with multiple matching prefixes."""
        env = {"MODIFIED_FILES": "src/contrib/external/app.js", "PR_TITLE": "[my_tag] external update"}
        new_title = self._update_pr_title_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_title, "[contrib][external] [my_tag] external update")
        new_labels = self._update_pr_labels_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_labels, "contrib: external")

    def test_no_relevant_files_changed(self):
        """Does not change the title if no files match the config rules."""
        original_title = "Just a refactor in a non-mapped area"
        env = {"MODIFIED_FILES": "internal/utils/helpers.py", "PR_TITLE": original_title}
        new_title = self._update_pr_title_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_title, original_title)
        new_labels = self._update_pr_labels_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_labels, "")

    def test_empty_modified_files_string(self):
        """Does not change the title if the modified_files string is empty."""
        original_title = "Some Title"
        env = {"MODIFIED_FILES": "", "PR_TITLE": original_title}
        new_title = self._update_pr_title_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_title, original_title)
        new_labels = self._update_pr_labels_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_labels, "")

    def test_sort_order_is_consistent(self):
        """Ensures prefixes are always sorted, regardless of input order."""
        env = {"MODIFIED_FILES": "docs/api.md src/frontend/app.js", "PR_TITLE": "Full stack feature"}
        new_title = self._update_pr_title_with_mocks(env, "config_yaml_content")
        # Note the alphabetical sort order of prefixes
        self.assertEqual(new_title, "[FE][docs] Full stack feature")
        new_labels = self._update_pr_labels_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_labels, "documentation,frontend")

        env = {"MODIFIED_FILES": "src/frontend/app.js docs/api.md", "PR_TITLE": "Full stack feature"}
        new_title = self._update_pr_title_with_mocks(env, "config_yaml_content")
        # Note the alphabetical sort order of prefixes
        self.assertEqual(new_title, "[FE][docs] Full stack feature")
        new_labels = self._update_pr_labels_with_mocks(env, "config_yaml_content")
        self.assertEqual(new_labels, "documentation,frontend")


if __name__ == "__main__":
    unittest.main()
