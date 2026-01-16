..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Offload a new model
===================

This guide describes the suggested process for offloading a new model to the NPU using this repository.
It is intended for kernel developers who will be writing new kernels or adding new shape support for existing kernels.

Get
---

The first step is to download the model you want to offload to the NPU.
If the model is not in ONNX format, convert it to ONNX.
You will need the ONNX model and external data file, if it exists.
This will be at ``/path/to/model.onnx``

Preprocess
----------

With your model, you should :ref:`preprocess <preprocess:Preprocess>` the model to be more amenable for the NPU.
This preprocessing script will be model-specific and specific to how you implement kernels.
You can see references to existing preprocessing scripts at the link above.
After preprocessing, you should have an ONNX model that has the operators you intend to offload, perhaps in a data format that is convenient (NCHW vs NHWC for example).

.. code-block:: console

    cd /path/to
    onnx_utils preprocess model.onnx optimized.onnx [--optimize <id>] [--save-as-external]

First report
------------

With your preprocessed model, the next step is to determine which operators need offloading.
With the :ref:`report <report:Report>` command, you can print which operators haven't been offloaded to :term:`DynamicDispatch`.

.. code-block:: console

    onnx_utils report dd_offload "/path/to/preprocessed_model" --summarize

This will print out a table similar to:

.. code-block:: text

                        DynamicDispatch Offload - not offloaded
    +--------------------+-------+-------------------------+------------------------+
    | Op Type            | Count | Inputs                  | Outputs                |
    +====================+=======+=========================+========================+
    | Add                | 5     | [1,64,64,320] - FLOAT   | [1,64,64,320] - FLOAT  |
    |                    |       | [1,1,1,320] - FLOAT     |                        |
    | Add                | 10    | [1,64,64,320] - FLOAT   | [1,64,64,320] - FLOAT  |
    |                    |       | [1,64,64,320] - FLOAT   |                        |
    | Add                | 15    | [320] - FLOAT           | [1,4096,320] - FLOAT   |
    |                    |       | [1,4096,320] - FLOAT    |                        |
    ...

In this table, note the title.
These are the operators that are not offloaded to DD.
It lists the type of the operator, the input/output shapes, the data types, and the count (how many times this operator with this configuration appears).
Based on this table and examining the model in Netron, you can choose the priority of operators to offload.

Extract a test
--------------

Once you select a specific operator + shape to offload, :ref:`extract <extract:Extract>` a test case for it to work with.
If you were intending to offload all the Add operators for example, you may want to extract all the Adds in one shot.

.. code-block:: console

    onnx_utils extract /path/to/optimized.model . /path/to/input_data.json add --pattern op-Add

If there are a lot of matching ops in the model though, this may result in a lot of files.
You can add ``--extract-first`` to only extract the first one.
Another option is to extract more specific operators with a pattern file and a node-based method:

.. code-block:: python
    :caption: my_pattern.py

    import onnx

    def user_pattern(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> bool:
        if ...
            return True
        else:
            return False

If you need to extract a subgraph with more than one operator (for example, if you are replacing a set of operators with a single DD op), use a pattern-based method. For example:

.. code-block:: python
    :caption: my_pattern.py

    import textwrap

    def user_pattern() -> str:
        return textwrap.dedent(
            """
            MatMul([?, ?], ?),
            Add([?, ?], ?),
            """
        )

Extracting will produce a small ONNX model(s) with your matched patterns, as well as, input and output data.
To run this extracted model, you can use ``tools/run.py`` as a template.
Duplicate this file into your directory with your extracted model and update the global variables as appropriate.
Confirm that you can call it and it prints "Success".
If it fails, you may need to adjust the tolerance for the comparison.
This confirms your model works as expected.
Then, rename this model to save it with a different name.

Implement your operator
-----------------------

In DD, add your operator with a unique name.
DD operators inherit from a common base class and implement an eager and fusion interface.
While eager may be useful for testing, this repository only uses fusion.
Your operator will be added as a PR into this repository so ensure you have test cases that test both the eager and fusion modes.
See DD documentation and existing tests for more information about this.

Write a pass
------------

With your test case, you need to add a :ref:`pass <partition:passes>` under ``src/ryzenai_onnx_utils/passes`` to match on your operator and replace it with a new op that represents your kernel.
Add your new op(s) to ``_single_nodes`` in ``src/ryzenai_onnx_utils/passes/dd/single_noqdq_to_dd.py``.
This will enable your single op to be replaced by a single DD fusion op.
If you replaced a set of ops with a single fusion op, then you'll need to add a new file in the ``src/ryzenai_onnx_utils/passes/dd`` directory to match on your pattern and replace it with a single DD fusion op.
Create a new strategy that includes your new pass in your working directory:

.. code-block:: yaml

    # domain to use for new ops
    domains: com.ryzenai
    # relative path to your XCLBIN in DD
    xclbins: "/xclbin/stx/sdxl_unet_vae_combined_new.xclbin"
    # define the passes to run. These passes are defined in passes/ and are executed
    # in order
    passes:
        - your_pass

Then, you can :ref:`partition <partition:Partition>` your extracted model:

.. code-block:: console

    onnx_utils partition add_0.onnx . /abs/path/to/your/strategy.yaml -v --force

This will generate ``replaced.onnx`` in this directory, along with a ``.cache`` directory containing the DD files.
Rename ``replaced.onnx`` to the name of the original model you were running before.

Build the custom op shared library
----------------------------------

To run your new model, you need the custom op shared library that defines the ``DynamicDispatch`` operator that was inserted into the model.
Follow the :ref:`build instructions <installation:custom ops shared library>` to build the library.

Run the replaced model
----------------------

Update your ``run.py`` so the path to the location of this shared library is set.
Then, when you run ``run.py`` again, it should run your new replaced model.
If everything went well, it should run, the output should still match and it will print "Success".

If it succeeded, you can go back to your list of operators and repeat these steps to offload more operators.

Extending to larger patterns
----------------------------

Once single op replacement or small op replacements work with fusion and your test cases can validate it's correct, you will slowly build up a library of operators that you can offload to DD along with a set of passes that transform the model this way.
If you run your passes on the full optimized model, you will see many ``DynamicDispatch`` operators, perhaps back-to-back, if you open your partitioned model in Netron.
These are candidates for larger op fusion.

To match larger patterns, use :ref:`match <match:Match>` to print a pattern for your target subgraph.
First, find the edges that define the inputs and outputs of the subgraph you want to fuse.
If you're doing this from the replaced model, look for places where the ``DynamicDispatch`` operator appears back-to-back.
Note the inputs and outputs of these operators, including any casts that may have been added before or after them.
Duplicate your strategy file and add the special pass ``remove_io_casts`` to remove Cast operators from your pattern.
Then, to print the pattern for these ops:

.. code-block:: console

    onnx_utils match "optimized.onnx" --strategy your_strategy.yaml --extract-inputs input_0 input_1 ... --extract-outputs output_0 output_1 ...


Troubleshooting
---------------

Through this process, you may run into some common issues.
Here's what you could do about them.

Accuracy errors
^^^^^^^^^^^^^^^

If the test case with your partitioned model fails because of mismatched data, then it indicates your op replacement changed the behavior of the model.
This could be an issue with your operator implementation.

DD out-of-memory
^^^^^^^^^^^^^^^^

If you fuse too many single DD ops in a large graph, you may run out of memory for DD.
To avoid this, you need to fuse fewer ops or fuse larger patterns.
