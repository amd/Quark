..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

ONNXRuntime_IPSP
================

ONNXRuntime_IPSP is an internal fork of :ref:`onnxruntime:ONNXRuntime` (ORT).
It provides the ``RyzenAIExecutionProvider`` but otherwise behaves similarly to public ORT.
See more information about the project on `GitEnterprise <https://gitenterprise.xilinx.com/IPSP/onnxruntime_internal/tree/dd-ryzenai>`__.

Usage
-----

These instructions are provided as a convenience.
The canon instructions are in the README for this project.

Get the repository and set up the environment:

.. code-block:: bash

    # setup custom ORT
    git clone --recursive https://gitenterprise.xilinx.com/IPSP/onnxruntime_internal.git -b dd-ryzenai
    # this will be <ORT_DIR>
    cd onnxruntime_internal
    conda env create --file=ryzenai_diffusers.yaml
    conda activate ryzenai-diffusers
    set XRT_PATH=<path_to_mcdm_driver>
    # use setup.bat on cmd
    setup.ps1

To build:

.. code-block:: bash

    # the install prefix tells ORT to put the install files in the build tree under a directory called "install"
    build.bat --enable_pybind --build_shared_lib --skip_tests --parallel --build_wheel --compile_no_warning_as_error --use_dml --use_ryzenai --config=Release --target install --cmake_extra_defines CMAKE_INSTALL_PREFIX=install
    # this produces DLL_PATH=build/Windows/Release/install/bin/onnx_custom_ops.dll
    # if you run without install target, it will be under build/Windows/Release/_deps/onnxutils-build/src/onnx_custom_ops or similar

To install:

.. code-block:: bash

    # install the DynamicDispatch python library, same version as used in ORT, this will be <DD_ROOT>
    pip install build/Windows/Release/_deps/dynamicdispatch-src
    # install the ORT python library. The name will be different based on your platform
    pip install build/Windows/Release/Release/dist/onnxruntime_ryzenai-1.18.1-cp39-cp39-win_amd64.whl --force-reinstall
    # install onnx_utils
    pip install build/Windows/Release/_deps/onnxutils-src
