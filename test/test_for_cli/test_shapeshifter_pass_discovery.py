#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Tests for shapeshifter pass discovery module."""

import importlib
from unittest.mock import MagicMock, patch

import pytest


def test_import_pass_modules():
    """Test that import_pass_modules discovers and imports modules correctly."""
    from quark.shapeshifter.pass_discovery import import_pass_modules

    # Test with empty directory (no modules to import)
    package_path = []
    package_name = "test_package"

    imported_modules = import_pass_modules(package_path, package_name)

    # Should return empty dict when no modules exist
    assert isinstance(imported_modules, dict)
    assert len(imported_modules) == 0


def test_discover_pass_classes():
    """Test that discover_pass_classes finds pass classes correctly."""
    from quark.shapeshifter.pass_discovery import discover_pass_classes

    # Create a mock module with a mock pass class
    mock_module = MagicMock()
    mock_module.__name__ = "test_module"

    class TestPass:
        pass

    TestPass.__module__ = "test_module"

    # Mock the inspect.getmembers to return our test class
    with patch("quark.shapeshifter.pass_discovery.inspect.getmembers") as mock_getmembers:
        mock_getmembers.return_value = [("TestPass", TestPass)]

        imported_modules = {"test_module": mock_module}
        pass_classes = discover_pass_classes(imported_modules)

        # Should discover the TestPass class
        assert isinstance(pass_classes, dict)
        assert "TestPass" in pass_classes
        assert pass_classes["TestPass"] == TestPass


def test_discover_pass_classes_skips_non_pass_classes():
    """Test that classes not ending with 'Pass' are skipped."""
    from quark.shapeshifter.pass_discovery import discover_pass_classes

    mock_module = MagicMock()
    mock_module.__name__ = "test_module"

    class Helper:
        """A helper class that doesn't end with 'Pass'."""

        pass

    Helper.__module__ = "test_module"

    with patch("quark.shapeshifter.pass_discovery.inspect.getmembers") as mock_getmembers:
        mock_getmembers.return_value = [("Helper", Helper)]

        imported_modules = {"test_module": mock_module}
        pass_classes = discover_pass_classes(imported_modules)

        # Should not discover classes that don't end with "Pass"
        assert isinstance(pass_classes, dict)
        assert len(pass_classes) == 0


def test_register_pass_classes():
    """Test pass registration in namespace."""
    from quark.shapeshifter.pass_discovery import register_pass_classes

    class TestPass:
        pass

    pass_classes = {"TestPass": TestPass}
    namespace = {}

    # Should register without errors
    all_passes = register_pass_classes(pass_classes, namespace)

    assert isinstance(all_passes, list)
    assert all_passes == ["TestPass"]
    assert "TestPass" in namespace
    assert namespace["TestPass"] == TestPass


def test_register_multiple_pass_classes():
    """Test registering multiple pass classes."""
    from quark.shapeshifter.pass_discovery import register_pass_classes

    class FirstPass:
        pass

    class SecondPass:
        pass

    pass_classes = {"FirstPass": FirstPass, "SecondPass": SecondPass}
    namespace = {}

    all_passes = register_pass_classes(pass_classes, namespace)

    assert len(all_passes) == 2
    assert sorted(all_passes) == ["FirstPass", "SecondPass"]
    assert namespace["FirstPass"] == FirstPass
    assert namespace["SecondPass"] == SecondPass


def test_community_passes_use_shared_discovery():
    """Test that community passes use the shared discovery functions."""
    import quark.contrib.shapeshifter_community_passes as community_passes

    # Should NOT have private functions (they're in shared module now)
    assert not hasattr(community_passes, "_import_pass_modules")
    assert not hasattr(community_passes, "_discover_pass_classes")
    assert not hasattr(community_passes, "_register_pass_classes_with_conflict_detection")

    # Should have __all__ attribute (result of using shared functions)
    assert hasattr(community_passes, "__all__")
    assert isinstance(community_passes.__all__, list)


def test_core_passes_use_shared_discovery():
    """Test that core passes use the shared discovery functions."""
    import quark.shapeshifter.passes as core_passes

    # Should NOT have private functions (they're in shared module now)
    assert not hasattr(core_passes, "_import_pass_modules")
    assert not hasattr(core_passes, "_discover_pass_classes")
    assert not hasattr(core_passes, "_register_pass_classes")

    # Should have __all__ attribute (result of using shared functions)
    assert hasattr(core_passes, "__all__")
    assert isinstance(core_passes.__all__, list)


def test_community_passes_optional_import():
    """Test that community passes are optional and don't break core shapeshifter."""
    # This test verifies that the try/except in shapeshifter/__init__.py works
    import quark.shapeshifter

    # Reload to trigger the import logic
    importlib.reload(quark.shapeshifter)

    # Should succeed without errors even if community passes fail to load
    assert hasattr(quark.shapeshifter, "shapeshifter")


def test_duplicate_pass_name_detection():
    """Test that duplicate pass names are detected and prevented."""
    from quark.shapeshifter.pass_base import REGISTRY, PytorchPass, register_pass

    # Save the current registry state
    original_registry = REGISTRY.copy()

    try:
        # Create a test pass and register it
        @register_pass
        class TestDuplicateCheckPass(PytorchPass):
            def _default_config(self):
                return {}

            def _run_for_config(self, model, config):
                return model

        # Verify it was registered
        assert "test_duplicate_pass_name_detection.<locals>.TestDuplicateCheckPass" in str(
            REGISTRY.get(list(REGISTRY.keys())[-1])
        )

        # Now try to register another pass with the same filename
        # We need to mock the __file__ to simulate a duplicate filename
        with patch("quark.shapeshifter.pass_base.Path") as mock_path:
            # Get the last registered pass name to simulate duplicate
            last_pass_name = list(REGISTRY.keys())[-1]
            mock_path.return_value.stem = last_pass_name

            # Should raise ValueError about duplicate
            with pytest.raises(ValueError, match="already registered"):

                @register_pass
                class AnotherTestPass(PytorchPass):
                    def _default_config(self):
                        return {}

                    def _run_for_config(self, model, config):
                        return model

    finally:
        # Restore registry to avoid affecting other tests
        REGISTRY.clear()
        REGISTRY.update(original_registry)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
