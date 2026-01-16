..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Preprocess
==========

Preprocessing refers to preparing an off-the-shelf model for partitioning.
Depending on the model, this process could be not necessary, very simple or complex.

Given an ONNX model, preprocessing will:

1. Prompt you to fix dynamic input dimensions to static fixed values
2. Run an optional optimization script
3. Infer shapes for all operators

Usage
-----

Preprocessing requires:

1. *input_path* - path to the original ONNX model to preprocess
2. *output_path* - path to the output ONNX model to write

To run an optimization script, use ``--optimize <name>``.
If you pass a model name, it's assumed to refer to ``src/ryzenai_onnx_utils/model_preprocessing/<name>.py``.
You can use your own optimization script by passing an absolute path to an optimization script.
Your custom optimization script should have an ``optimize()`` function that matches the function signature of the built-in functions.

.. warning::

    Preprocessing commands must be run in the same working directory as the original model if the model uses external data.
    Unlike other commands, preprocessing requires loading external data and saving intermediate ONNX models.
    If you are not in the same directory, ONNX will be unable to find external data.


See the :ref:`command-line arguments <cli:preprocess>` for the full list of arguments and options.

Example
-------

Existing preprocessing scripts are stored in ``src/ryzenai_onnx_utils/model_preprocessing``.
As an example, the :onnxUtilsTree:`preprocessing script <src/ryzenai_onnx_utils/model_preprocessing/unet.py>` for *unet* from Stable Diffusion infers Gelu, LayerNormalization and MHA ops from the original model and adds these as single operators into the graph.
It also runs ORT's default optimization passes to add constant folding and other basic optimizations.

Multiple graphs with If
-----------------------

The first prompt for preprocessing will ask you to specify the number of graphs.
Entering ``1`` will perform the standard case: the input dimensions of the graph will be fixed to what you specify.
If you enter a number more than one, the original graph of the model will be duplicated that many times and each copy will be placed under an ONNX ``If`` node.
The prompts for the input dimensions will ask you to specify that many values for each dimension so each copy of the graph can be fixed to a different value if needed.
In the graph now, the runtime dynamic shape is checked against these fixed values and the appropriate copy of the graph is invoked.

.. warning::

    This is an experimental feature and does not work with the rest of the partitioning flow yet. Set the number of graphs to ``1`` for now if you want to use the preprocessed model.

.. image:: figures/if_block.png
    :alt: Picture showing the transformation from the original model to the model with If nodes
