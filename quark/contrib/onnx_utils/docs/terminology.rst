..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Terminology
===========

This page presents some important terminology to frame the rest of the documentation.

.. glossary::
    :sorted:

    Partition
        Partitioning refers to transforming an :term:`ONNX` model to a new ONNX model by running :term:`passes <Pass>` on it

    ONNX
        A Microsoft-backed ecosystem for machine-learning models. Stands for Open Neural Network Exchange.

    ONNXRuntime
        A Microsoft-backed machine learning model accelerator to run models on a variety of backends

    Pass
        A pass is made up of a :term:`pattern` and a replacement function. The pattern defines which nodes in an :term:`ONNX` graph to match on, and the replacement function defines how the matched subgraph should be transformed. A transformation could be to replace the subgraph with a new one, for example.

    Pattern
        A pattern is a representation of a graph of :term:`ONNX` nodes. An example of a simple pattern is ``["Add([?, ?], a0)", "Reshape(a0,[?])"]``. It specifies the name of the ONNX operator followed by parentheses. Inside the parentheses, it defines a list or a name for the input/output. You have to use lists if there are multiple input/outputs but it's optional if there's only one. Using the same name in the output of one operator and the input of an another shows that the two operators are connected. At the outer edges of the graph, use question marks to denote unnamed values.

    Dynamic Pattern
        Similar to the :term: `Pattern`, a dynamic pattern is a representation of a graph of :term:`ONNX` nodes, but it does not provide constant patterns but dynamically generates the patterns by calling a function. See more information about :ref:`dynamic pattern <partition:Dynamic passes>`

    DD
        DynamicDispatch. See :term:`DynamicDispatch`

    DynamicDispatch
        DynamicDispatch (DD) allows you to run ops on the NPU using its fusion runtime. See more information about DD `online <https://gitenterprise.xilinx.com/VitisAI/dynamicdispatch>`__.

    Strategy
        A strategy defines how to partition a model. It is made up of a domain (or domains), a xclbin (or xclbins), and a list of passes to run.
