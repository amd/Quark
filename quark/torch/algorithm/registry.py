#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""The algorithm registry: every :py:class:`~quark.torch.algorithm.algorithm.QuarkAlgorithm` Quark
knows about, and the accessor core looks them up through.
"""

from __future__ import annotations

from quark.torch.algorithm.algorithm import QuarkAlgorithm

__all__ = ["ALGORITHM_REGISTRY"]


class _AlgorithmRegistry:
    """The registry behind :py:data:`ALGORITHM_REGISTRY`, which is its only instance.

    The algorithms it holds are private to it, reachable through ``register``, ``get`` and
    ``get_algorithms``: there is no second way to add an entry, and no way to remove or replace
    one.
    """

    def __init__(self) -> None:
        self._algorithms: dict[str, QuarkAlgorithm] = {}

    def register(self, algorithm: QuarkAlgorithm) -> None:
        """Register an algorithm, making it visible to every core lookup.

        Registering the same object twice is a no-op, so a module that registers at import time is
        safe to import more than once. Registering a *different* algorithm under a name already
        taken raises: silently replacing one would change the behaviour of every lookup site at
        once, and which of the two won would depend on import order.

        ``QuarkAlgorithm`` validates itself on construction, so by the time it reaches here the
        config class and processor are already known-good.

        .. code-block:: python

            ALGORITHM_REGISTRY.register(
                QuarkAlgorithm(name="myalgo", algo_config=MyAlgoConfig, algo_processor=MyAlgoProcessor)
            )

        :param QuarkAlgorithm algorithm: The algorithm to register.
        :raises TypeError: If ``algorithm`` is not a :py:class:`QuarkAlgorithm`.
        :raises ValueError: If a different algorithm is already registered under the same name.
        """
        if not isinstance(algorithm, QuarkAlgorithm):
            raise TypeError(f"register expects a QuarkAlgorithm, got {algorithm!r}.")

        registered = self._algorithms.get(algorithm.name)
        if registered is algorithm:
            return

        if registered is not None:
            raise ValueError(
                f"An algorithm named {algorithm.name!r} is already registered ({registered!r}); "
                f"cannot register {algorithm!r} under the same name."
            )

        self._algorithms[algorithm.name] = algorithm

    def get(self, name: str) -> QuarkAlgorithm | None:
        """Look up the :py:class:`~quark.torch.algorithm.algorithm.QuarkAlgorithm` called ``name``.

        Matched case-insensitively. ``None`` if no registered algorithm claims the name, which is
        core's signal to fall back to its own tables.

        :param str name: The algorithm name, as it appears in a config's ``name`` field.
        :return: The registered algorithm, or ``None``.
        :rtype: QuarkAlgorithm | None
        """
        return self._algorithms.get(name.strip().lower())

    def get_algorithms(self) -> tuple[QuarkAlgorithm, ...]:
        """Return every registered algorithm, in registration order.

        A tuple, not a view: the caller gets a snapshot it cannot use to add, remove or reorder
        the registry's entries.

        :return: The registered algorithms.
        :rtype: tuple[QuarkAlgorithm, ...]
        """
        return tuple(self._algorithms.values())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(self._algorithms)})"


ALGORITHM_REGISTRY = _AlgorithmRegistry()
