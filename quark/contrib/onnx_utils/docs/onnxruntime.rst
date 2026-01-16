..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

ONNXRuntime
===========

ONNXRuntime (ORT) is a Microsoft-backed machine learning model accelerator to run models on a variety of backends.
See more information about the project on their `documentation <https://onnxruntime.ai/docs/>`__ or `Github <https://github.com/microsoft/onnxruntime>`__.

Custom Ops
----------

ONNXRuntime allows you to add `custom operators <https://onnxruntime.ai/docs/reference/operators/add-custom-op.html>`__ to an execution provider with minimal modifications to a user's ONNXRuntime installation.
These are supported by standard ORT APIs.
Custom ops should be added to a custom domain to avoid clashing with built-in operators.
The standard usage model for custom ops is to compile them to a shared library that can be dynamically loaded by ORT.

.. tabs::

    .. code-tab:: c++ C++

        #include "onnxruntime_cxx_api.h"

        Ort::SessionOptions session_options;
        session_options.RegisterCustomOpsLibrary("/path/to/shared/library")

    .. code-tab:: python Python

        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.register_custom_ops_library("/path/to/shared/library")

Then, when ORT encounters an operator that is implemented by a custom operator, it will execute the custom operator.
