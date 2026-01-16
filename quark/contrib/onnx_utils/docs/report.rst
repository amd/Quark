..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Report
======

You can analyze ONNX models and generate predefined reports from them.
The list of available reports are below.

Usage
-----

Reporting requires:

1. *report* - Name of the report to generate
2. *input_path* - path to the input ONNX model to analyze

Reports also have optional flags like ``--summarize`` and ``--comparison-path``.
The exact behavior of these two flags are dependent on the report type and not all reports will use all options.

See the :ref:`command-line arguments <cli:report>` for the full list of arguments and options.

Reports
-------

The currently supported reports are described in more detail here.

``dd_offload``
^^^^^^^^^^^^^^

This report prints the number of operators that are and are not offloaded to the NPU using :term:`DD`.
It prints the operators, input and output shapes and data types for these operators.
This report can help guide you on how to offload most, if not all, of the model to NPU.

.. code-block:: console

    onnx_utils report dd_offload /path/to/onnx.model [--summarize] [--comparison-path]

.. tip::

    If you're starting from a new model, generate the report on the preprocessed model.
    Then, as you offload more operators, you can iteratively run the report as you go.

.. using the heading "adjacency" breaks in Firefox: it's interpreted as a CSS property
.. and hides this section

``op_adjacency``
^^^^^^^^^^^^^^^^

This report analyzes the given ONNX model and prints the frequency of nodes having certain parent node combinations.
It can be used to identify operators that are frequently preceding others.
The ``--depth`` flag defines how far up to search and defaults to 1.

.. code-block:: console

    onnx_utils report op_adjacency /path/to/onnx.model [--depth <number>]

In this sample report, you can see that there are 19 ``Transpose`` nodes in the model and 53.85% of them have an ``Add`` node as a direct parent.
``Level 1`` indicates direct ancestor nodes while ``Level 2`` and beyond show the ancestry of the node from the parent node to the specified level.
For higher levels, each entry shows one continuous chain of nodes from the initial node.
For example, it shows that 23.08% of Transpose nodes have ``Add`` node(s) as direct ancestors, which in turn have ``Add`` and ``Transpose`` ancestors, respectively.
This result can mean that there are two separate ``Add`` nodes that are parents of the ``Transpose`` or the same parent node which in turn has two parents.
These two possible cases are represented in :numref:`op_adjacency` and both would have the same representation in the report.
From knowledge of this op in particular, since ``Transpose`` takes one argument, the latter is the true graph.
However, this deduction may not always be possible if an op can have variable numbers of inputs.

.. code-block:: text
    :caption: Sample output for the ``op_adjacency`` report

    Op: Transpose (19)
        Level: 1
            (('Add',),): 53.85%
            (('NhwcConv',),): 38.46%
            (('Resize',),): 7.69%
        Level: 2
            (('NhwcConv', 'GroupNorm'),): 38.46%
            (('Add', 'Add'), ('Add', 'Transpose')): 23.08%
            (('Add', 'NhwcConv'), ('Add', 'NhwcConv')): 7.69%
            (('Add', 'Mul'),): 7.69%
            (('Add', 'MatMul'),): 7.69%
            (('Add', 'Reshape'), ('Add', 'Transpose')): 7.69%
            (('Resize', 'Add'),): 7.69%
    ...

.. figure:: figures/op_adjacency.png
    :name: op_adjacency
    :alt: Picture showing two alternative graphs that would have the same ancestry in the ``op_adjacency`` report

    Potential graphs with same Level 2 ancestry in the ``op_adjacency`` report

``subgraph_heuristic``
^^^^^^^^^^^^^^^^^^^^^^

This report analyzes the given ONNX model and the names of nodes to extract subgraphs and save them as individual ONNX models.
It can be used to pull out potentially repeating subgraphs.
This name-based heuristic is very dependent on the original model.
If the model has many nodes that can't be inferred as being part of the same subgraph, then the resulting subgraphs will be incomplete.
It will also not work well for extremely large models with many nodes.

.. code-block:: console

    onnx_utils report subgraph_heuristic /path/to/onnx.model [--output-dir]
