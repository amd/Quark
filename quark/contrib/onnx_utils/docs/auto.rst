..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Auto
====

Auto refers to Automatic Graph Compilation Flow.

Given an ONNX model, auto will:

1. Preprocess the model based on the optimize function denoted in the command line. Generate a preprocessed model.
2. Partition the model based on the strategy denoted in the command line. Generate a partitioned model.
3. Generate a report on the partitioned model showing the supported/unsupported ops and shapes.
4. Depend on the flag, this script can run the partitioned model on the NPU with random input data.

Usage
-----

Preprocessing requires:

1. *input_path* - path to the original ONNX model
2. *output_path* - path to the output ONNX model to write
3. *optimize* - optimization script to run
4. *strategy* - strategy to partition the model

Optional:

1. *execute* - flag to run the partitioned model on the NPU with random input data

.. warning::

    Auto ('s Preprocessing) must be run in the same working directory as the original model if the model uses external data.
    Unlike other commands, preprocessing requires loading external data and saving intermediate ONNX models.
    If you are not in the same directory, ONNX will be unable to find external data.


See the :ref:`command-line arguments <cli:auto>` for the full list of arguments and options.

Example
-------

As an example, the :onnxUtilsTree:`auto --model-name replaced --optimize sd15_unet <path/to/input/model.onnx>  <path/to/input/output> sd15_unet.yaml --load-external-data --save-as-external --execute`
preprocesses the model with the optimization script sd15_unet and partitions the model with the strategy sd15_unet.yaml. The partitioned model is then run on the NPU with random input data. And a report of the partitioned model is generated.
