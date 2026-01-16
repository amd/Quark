..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Introduction
============

RyzenAI ONNX Utils is a collection of tools to work with ONNX models and deploy them to RyzenAI hardware.
It's made up of a command-line Python executable, *onnx_utils*, and a shared library for custom ONNX ops.
Using the *onnx_utils* executable, you can process and offline partition any ONNX model for RyzenAI by inserting custom ops representing RyzenAI operations.
For the NPU, this compute is handled using `DynamicDispatch <https://gitenterprise.xilinx.com/VitisAI/dynamicdispatch>`__ and its fusion runtime.

Features
--------

The core feature set is in the *onnx_utils* executable and its components:

* *match* - Given an ONNX model, print an *onnx_utils* compatible pattern from the model or a subset of it
* *preprocess* - Given an ONNX model, fix its dynamic shapes, infer all shapes and run any model-specific optimizations
* *partition* - Run a set of passes on an ONNX model to pattern match on selected nodes and transform them
* *extract* - Using a specified pattern, extract subgraphs from an ONNX model and save input and output data to create a self-contained test case
* *report* - Print different reports from an ONNX model

The shared library for custom ops allows injecting custom operators into arbitrary ONNXRuntime execution providers.
These custom operators can execute custom code to leverage hardware accelerators like GPU and NPU.

Documentation overview
----------------------

The remainder of this documentation is organized as follows:

* The **Getting Started** section continues to talk about the project at a high-level
* The **Functions** section discusses the different features of the project in more detail
* The **Projects** section provides more commentary on specific projects that leverage this project
* The **Developers** section has useful information for developers who want to dive into the implementation details
* The **Related Projects** section presents links to other projects that are related to this one


Support
-------

This documentation is your best source of support.
You can also raise issues on `Github <https://gitenterprise.xilinx.com/varunsh/onnx_utils/issues>`__ if you run into a bug or have a question.
