#
# Copyright (C) 2024 - 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys


def _install_pkg_resources_shim() -> None:
    """Provide a minimal ``pkg_resources`` shim backed by ``importlib.metadata``.

    Background
    ----------
    ``brevitas/__init__.py`` (master branch) still does, at module top::

        from pkg_resources import DistributionNotFound
        from pkg_resources import get_distribution

    ``pkg_resources`` was removed from ``setuptools`` in v82.0.0 (released
    2026-02-08), so importing brevitas on any env that picks up a modern
    setuptools fails at collection time with::

        ModuleNotFoundError: No module named 'pkg_resources'

    Rather than pin our env to an old setuptools, this shim provides only the
    two symbols brevitas actually consumes, implemented on top of the stdlib
    ``importlib.metadata`` (which is the official migration target).

    The shim is registered at ``sys.modules['pkg_resources']`` *before* any
    test module in this directory is collected, so the brevitas import
    succeeds without us ever installing the deprecated package.

    Scope: local to this conftest only. Nothing outside
    ``test/test_for_torch/extensions/brevitas/`` sees the shim.

    Remove this shim once brevitas migrates upstream
    (https://github.com/Xilinx/brevitas, src/brevitas/__init__.py).
    Tracking: https://github.com/pypa/setuptools/issues/5174
    """
    # If a real pkg_resources is already installed (e.g. setuptools < 82),
    # don't shadow it.
    import importlib.util

    if importlib.util.find_spec("pkg_resources") is not None:
        return

    import types
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _md_version

    shim = types.ModuleType("pkg_resources")

    class DistributionNotFound(Exception):
        """pkg_resources.DistributionNotFound stub."""

    class _Distribution:
        """Minimal stand-in for ``pkg_resources.Distribution``.

        brevitas only reads ``.version``; other pkg_resources attributes are
        intentionally omitted. Extend if a future brevitas release needs them.
        """

        def __init__(self, project_name: str, version: str) -> None:
            self.project_name = project_name
            self.version = version

    def get_distribution(name: str) -> _Distribution:
        try:
            return _Distribution(name, _md_version(name))
        except PackageNotFoundError as exc:
            raise DistributionNotFound(name) from exc

    shim.DistributionNotFound = DistributionNotFound
    shim.get_distribution = get_distribution
    sys.modules["pkg_resources"] = shim


# brevitas does not support Python 3.13+ — skip the whole directory rather than
# fight an import we cannot reasonably patch.
# https://github.com/Xilinx/brevitas/issues/1450
if sys.version_info >= (3, 13):
    collect_ignore_glob = ["test_*.py"]
else:
    _install_pkg_resources_shim()
    collect_ignore_glob = []
