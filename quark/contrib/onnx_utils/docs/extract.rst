..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Extract
=======

You can extract a subgraph from an :term:`ONNX` model for debugging purposes.
ONNX has a built-in method in ``onnx.utils.extract_model`` which the *extract* function in ``ryzenai_onnx_utils`` wraps and extends.
It will:

1. Run the original ONNX model with user-provided data
2. Extract all (or the first) matching subgraph(s) from the model
3. For each extracted subgraph, save the input and expected output data and create a standalone test case.

The resulting files can be run with ``tools/run.py``.

Usage
-----

Extraction requires:

1. *input_path* - path to the original ONNX model to extract from
2. *output_path* - path to a directory to save files in
3. *model_name* - a name to use as a prefix. If the tensor data in the model has a prefix, this should match that.
4. *pattern* - a pattern or pattern file to use for matching the graphs of interest

See the :ref:`command-line arguments <cli:extract>` for the full list of arguments and options.

Saving input data
-----------------

To generate individual ONNX graphs and their input/output data, you need to first save the inputs to the original model and pass them in with ``--tensor-data``.
Presumably, you can run your entire model already.
When you invoke your original model, you pass the inputs to the ONNX session.
There, you can save the inputs with something like this:

.. code-block:: python

    import json
    import numpy as np

    def save_inputs(filename: str, onnx_inputs: dict)
        # this helper class converts numpy arrays to lists for JSON
        class NumpyArrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        with open(filename, "w") as write_file:
            json.dump(onnx_inputs, write_file, cls=NumpyArrayEncoder)

    # this is the input to your ONNX model. It should be a dict mapping to numpy arrays or lists
    onnx_inputs = {
        "input_0": np.ndarray(...),
        "input_1": np.ndarray(...),
        ...
    }
    input_data_filename = "input_data.json"
    save_inputs(input_data_filename, onnx_inputs)

If you don't have real data, you can also make a dictionary with random data of the right shape/data type and save that as above.
Doing this may result in errors though if the input data has values that don't work with the model.

Pattern matching
----------------

You can match on either a predefined pattern (use ``--pattern <name>``) or a custom pattern (use ``--pattern-file``).
There are two kinds of patterns that you can define: a node-based one and an pattern-based one.

A node-based pattern uses a function like:

.. code-block:: python
    :caption: my_pattern.py

    import onnx

    def user_pattern(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> bool:
        # body


It operates on each node in the graph in succession and returns true or false whether a given node should be matched or not.

An pattern-based pattern uses a function like:

.. code-block:: python
    :caption: my_pattern.py

    import textwrap

    def user_pattern() -> str:
        return textwrap.dedent(
            """
            Conv(?, ?)
            """
        )

It can match a single node or a complex set of nodes.

Predefined patterns
^^^^^^^^^^^^^^^^^^^

The predefined patterns are in ``src/ryzenai_onnx_utils/data/extract_patterns``.

The special predefined pattern, ``op``, matches any single named op.
To use this pattern from the command line, use ``--pattern op-<name>``, where ``<name>`` matches the ONNX operator you're interested in (case-sensitive).

Another special pattern is ``dd`` which uses DynamicDispatch's ``is_supported()`` API to see if a given node is supported under DD.
If so, it matches, otherwise it doesn't.

Pattern file
^^^^^^^^^^^^

To use your own pattern matching logic, you can write it in a Python file and pass it with ``--pattern-file``.
The pattern file must contain a function called ``user_pattern()`` which contains a valid pattern: a node-based or an pattern-based one.

Examples
--------

Extracting the first MatMul op:

.. code-block:: console

    onnx_utils extract /path/to/model.onnx . matmul --pattern "op-MatMul" --extract-first
