..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Match
=====

You can use an :term:`ONNX` model, or part of an ONNX model, to generate a valid :term:`pattern` for it.
If you want to print a pattern based on a subgraph of a model, you'd pass ``--extract-inputs`` and ``--extract-outputs`` along with a list of edges that define the subgraph you want to match on.
You can also provide a strategy to run before matching with ``--strategy``.

Matching will:

1. If ``--extract-inputs`` and ``--extract-outputs`` are provided, extract a subgraph based on the provided edges
2. Print the pattern based on this subgraph
3. If ``--strategy`` is provided, run the strategy, and print the pattern on the transformed graph as well
4. If ``--hints-key`` is provided, it will use the hint to generate dynamic patterns and print those

If you're trying to match patterns for DD replacement, you will need to add ``remove_io_casts`` as the last pass in your strategy to remove Cast operators from the pattern.

Usage
-----

Matching requires:

1. *input_path* - path to the original ONNX model to match on

See the :ref:`command-line arguments <cli:match>` for the full list of arguments and options.

Example
-------

For SDXL-Turbo, matching on *unet* was with:

.. code-block:: console

    onnx_utils match /path/to/optimized.model --strategy sdxl_turbo_unet_dd_matcher.yaml --extract-inputs /Concat_3_output_0 /time_proj/Concat_1_output_0 NhwcConv_0_out-/conv_in/Conv_output_0 encoder_hidden_states --extract-outputs NhwcConv_50_out-out_sample
