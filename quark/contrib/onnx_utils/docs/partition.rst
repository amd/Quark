..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Partition
=========

You can partition an :term:`ONNX` model to transform it into a new model.

.. image:: figures/partition_flow.png
    :alt: Picture showing the transformation from the original model to DynamicDispatch

Usage
-----

See the :ref:`command-line arguments <cli:partition>` for the arguments.

Overview
--------

Partitioning takes an input ONNX model, applies a configurable set of passes on it, and saves a new model.
In particular, this process allows you to offload supported ops to the NPU using DynamicDispatch.

The set of passes that exist are defined in ``src/ryzenai_onnx_utils/passes`` while the built-in partitioning strategies—the set and order of passes to run—is defined in ``src/ryzenai_onnx_utils/data/partition_strategies``.
The goal of the passes is to transform the ONNX model in a step-by-step process by replacing small subgraphs at a time to convert the original model into something more desirable.
For offloading to the NPU, this means converting the original ONNX ops into ops that are DD ops.
Then, these DD ops can be pattern matched and replaced with a ``DynamicDispatch`` custom op that executes the pattern with fusion as a single operator.

Passes
------

Passes are defined in ``src/ryzenai_onnx_utils/passes``.
To add a new pass, you would create a file there.

Below, you can see an example of a pass that has been annotated to add additional commentary.
Note that the imports and some helper functions have been omitted from this sample.
The full pass is in ``src/ryzenai_onnx_utils/passes/add_to_elwadd_noqdq.py``.

.. literalinclude:: ../src/ryzenai_onnx_utils/passes/add_to_elwadd_noqdq.py
    :start-after: +start:
    :end-before: -end:
    :language: python

Aggregate Passes
^^^^^^^^^^^^^^^^

There are also some directories in ``src/ryzenai_onnx_utils/passes``.
These represent aggregate passes.
It allows you to specify the directory name in the strategy and ``ryzenai_onnx_utils`` will automatically run all the passes under that directory.
The only different thing in aggregate passes is that the ``__init__.py`` file in the directories will automatically import all passes that are there.

Aggregated passes are run in descending order of pattern size so larger patterns are matched first before smaller ones.
However, there is no guaranteed order between passes that have patterns of the same size.
If you need to run a subset of aggregate passes or run them in a specific order, you have to specify them explicitly in the strategy with ``directory.pass_name``.

To make a new aggregate pass, create a directory with a meaningful name, copy an ``__init__.py`` from an existing aggregate pass and add new passes in the directory.
Then, in your strategy, you can specify the name of the directory to run all your passes or specify individual passes to run manually as any other pass.


Dynamic passes
^^^^^^^^^^^^^^

Traditionally, passes match on static :term:`patterns <Pattern>`.
You can also match on a *dynamic pattern* where you match on a function that returns a list of static patterns.
The dynamic pattern in ``src/ryzenai_onnx_utils/passes/dd/dynamic_sd_nodes_to_dd.py`` is one example of this style of matching.
