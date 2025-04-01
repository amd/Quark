#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import abc
import inspect
from typing import Optional
from typing import Callable, List, Any
from torch.fx.graph_module import GraphModule
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ['OptPassManager', 'OptPassBase']


class OptPassBase(abc.ABC):
    """
    Base interface for implementing passes.

    It is required to implement the `call` function so that we can directly
    pass instances of the Pass directly to the PassManager and call them as a
    function.

    We can directly pass an instance of a class implementing this interface into
    the PassManager's `passes` attribute.
    """

    def __call__(self, graph_module: GraphModule) -> GraphModule:
        """
        Runs the precondition check, the pass itself, and the postcondition check.
        """
        self.requires(graph_module)
        res = self.call(graph_module)
        self.ensures(graph_module)
        return res

    @abc.abstractmethod
    def call(self, graph_module: GraphModule) -> GraphModule:
        """
        The pass that is run through the given graph module. To implement a
        pass, it is required to implement this function.

        Args:
            graph_module: The graph module we will run a pass on
        """
        pass

    def requires(self, graph_module: GraphModule) -> None:  # noqa: B027
        """
        This function will be called before the pass is run and will check that
        the given graph module contains the preconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            graph_module: The graph module we will run checks on
        """
        pass

    def ensures(self, graph_module: GraphModule) -> None:  # noqa: B027
        """
        This function will be called after the pass is run and will check that
        the given graph module contains the postconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            graph_module: The graph module we will run checks on
        """
        pass


# ref: torch/fx/passes/infra/pass_manager.py
class OptPassManager:
    """
    Construct a OPTPassManager.

    Collects passes. This defines the pass schedule

    Args:
        passes (Optional[List[Callable]]): List of passes. A pass is a
            callable which modifies an object and returns a PassResu
    """

    def __init__(self, passes: Optional[List[Callable[[GraphModule], GraphModule]]] = None) -> None:
        self.passes = passes if passes is not None else []

    def add_pass(self, _pass: Callable[[GraphModule], GraphModule]) -> None:
        """
        Adds a pass into the current list of passes.
        """
        self.passes.append(_pass)

    def add_checks(self, check: Callable[[Any], Any]) -> None:
        """
        Adds a function which takes runs various checks on a given graph module.
        This function is run before and after each pass if the
        `run_checks_after_each_pass` flag is enabled.
        """
        pass
        # TODO
        # sig = inspect.signature(check)

        # if len(list(sig.parameters.values())) != 1:
        #     raise TypeError("PassManager check function should only take in one variable, a module")

        # setattr(self, "check", check)  # noqa: B010

    def check(self, module: GraphModule) -> None:
        pass

    def __call__(self, model: GraphModule) -> GraphModule:
        """
        Runs a list of passes in the order based on `self.passes` on the given
        graph module. Each time a pass is run, checks and linting will be run on
        the graph module if `run_checks_after_each_pass` is set.

        If the module is a graph module, we will run the list of passes until
        the graph stops changing, or until `steps` number of times.
        """
        # Check graph invariants
        self.check(model)
        # Run the set of passes on the graph module
        for i, fn in enumerate(self.passes):
            fn_name = fn.__name__ if inspect.isfunction(fn) else type(fn).__name__
            logger.info("Running pass '%s'", fn_name)
            model = fn(model)
        model: GraphModule = GraphModule(model, model.graph)
        return model
