..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Build and install
=================

There are two components to this repository: the Python package and the custom ops shared library.
The Python package contains the command-line executable to work with and :term:`partition` an :term:`ONNX` model while the custom ops shared library is only needed when you want to run a partitioned ONNX model on hardware.

Python package
--------------

If you're just a consumer, you can install the Python package into your environment directly from Github:

.. code-block:: console

    pip install git+https://gitenterprise.xilinx.com/varunsh/onnx_utils.git

If you're going to be editing the tool or changing passes, you should clone the repository and use an editable build with dev dependencies:

.. code-block:: console

    git clone https://gitenterprise.xilinx.com/varunsh/onnx_utils.git
    cd onnx_utils
    pip install -e .[dev]

To build a shareable wheel, clone the repository as above and run:

.. code-block:: console

    pip wheel --no-deps .

Additional dependencies
^^^^^^^^^^^^^^^^^^^^^^^

Standard Python dependencies from PyPI are installed automatically with one exception: ``onnxruntime``.
This is omitted from automatic installation to allow you to install custom versions of ``onnxruntime`` as needed without the installation of ``ryzenai_onnx_utils`` uninstalling your version with the standard one.

The other dependency which is not automatically captured is :term:`DD`'s Python library.
For consistency, you should use the same version of DD's Python library as the C++ version you're using for the custom ops (see below).
That said, DD's Python library changes slower than the C++ version so you can get away with not keeping them in sync all the time.
To install DD's Python library:

.. code-block:: console

    cd /path/to/DD/repo
    pip install .

If you already have the DD repository cloned locally that you're using, you can install it from there.
When you build the custom ops shared library, ``ryzenai_onnx_utils`` will clone it for you if you don't specify otherwise (see below).

Custom ops shared library
-------------------------

You will need a modern CMake (3.28+) to build the shared library.

ONNXRuntime
^^^^^^^^^^^

You need a directory containing the install files from :term:`ONNXRuntime`.
If you already have this locally, you can pass this to CMake: ``-DORT_INSTALL_DIRS=/path/to/ORT/installation``.
Otherwise, you can download a release for your platform from `Github <https://github.com/microsoft/onnxruntime/releases>`__ and unzip it to ``external/onnxruntime``.
The directory structure should be:

.. code-block:: text

    external/
    ├─ onnxruntime/
    │  ├─ include/
    │  │  ├─ onnxruntime_cxx_api.h
    │  │  ├─ ...
    │  ├─ lib/
    │  │  ├─ ...

If you're building ORT from source, then add ``--target install --cmake_extra_defines CMAKE_INSTALL_PREFIX=install`` to the ``build.bat`` in ORT to create an install directory in the build tree.
Then, you can set ``ORT_INSTALL_DIRS`` to the path to this directory.

DynamicDispatch
^^^^^^^^^^^^^^^

If you're using :term:`DD`, you will need DD's dependencies.
The easiest way to resolve this is to use ``conda`` and build an environment based on `DD's specifications <https://gitenterprise.xilinx.com/VitisAI/DynamicDispatch/blob/main/env.yml>`__.

If you already have DD locally, you can pass this to CMake: ``-DDYNAMIC_DISPATCH_SRC=/path/to/DD``.
Otherwise, ``ryzenai_onnx_utils`` will automatically clone it during the build based on ``cmake/FindDynamicDispatch.cmake``.
In this case, it will be located in ``build/_deps/dynamicdispatch-src``.
You can enable this by passing ``-DONNX_UTILS_ENABLE_PROJECT_DD=ON`` or using a preset that has this set.

In your environment, set ``DD_ROOT`` to ``/path/to/DD``

Build
^^^^^

Then, to build:

.. code-block:: bash

    # use the appropriate preset for your project or your own
    cmake --preset default --fresh <more CMake args>
    cmake --build --preset release

By default, the CMake install directory is set to ``build/install`` so any files marked for install, including the custom ops shared library, get placed in this directory for easy reference.

For convenience, you should save any custom CMake args as a `user preset <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html#id3>`__.
To do this, make a file called ``CMakeUserPresets.json`` in the root directory.
Then, you can save your arguments and run your preset instead of the default one.
For example, one user preset file could look like:

.. code-block:: json

    {
        "$schema": "https://cmake.org/cmake/help/v3.28/_downloads/3e2d73bff478d88a7de0de736ba5e361/schema.json",
        "version": 8,
        "configurePresets": [
            {
                "name": "my-preset",
                "inherits": "ninja",
                "cacheVariables": {
                    "ONNX_UTILS_ENABLE_PROJECT_DD": "ON",
                    "DYNAMIC_DISPATCH_SRC": "/path/to/DD",
                    "ORT_INCLUDE_DIRS": "/path/to/ORT"
                }
            }
        ]
    }

You can enter the DD directory and do ``pip install .`` to install the Python library based on this version.
Otherwise, you should install the Python library from the DD path you provided to CMake.
If you don't have DynamicDispatch source code or want to use some other version in the Python library, you can `clone the repository <https://gitenterprise.xilinx.com/VitisAI/DynamicDispatch>`__, checkout the appropriate version and do ``pip install .`` in that directory.

CMake Options
^^^^^^^^^^^^^

The available CMake options for configuration are:

.. csv-table:: CMake Variables
    :header: "Variable", "Meaning", "Default value"

    ``DYNAMIC_DISPATCH_SRC``,Path to Dynamic Dispatch source directory,N/A
    ``ONNX_UTILS_BUILD_EXAMPLES``,Build examples, ``OFF``
    ``ONNX_UTILS_BUILD_TESTS``,Build tests, ``OFF``
    ``ONNX_UTILS_ENABLE_PROJECT_DD``, Enable DynamicDispatch-based custom ops, ``OFF``
    ``ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM``, Enable custom ops for the hybrid GPU/NPU project, ``OFF``
    ``ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU``, Enable the CPU implementations for the hybrid GPU/NPU project, ``OFF``
    ``ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU``, Enable the GPU implementations for the hybrid GPU/NPU project, ``OFF``
    ``ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU``, Enable the NPU implementations for for the hybrid GPU/NPU project, ``OFF``
    ``ONNX_UTILS_EP``, Assign custom ops to this EP, ``CPUExecutionProvider``
    ``ORT_BIN_DIRS``, Path to the ORT bin directory, ``${ORT_INSTALL_DIRS}/bin``
    ``ORT_INCLUDE_DIRS``, Path to the ORT include directory, ``${ORT_INSTALL_DIRS}/include``
    ``ORT_INSTALL_DIRS``, Path to the ORT install directory, ``${PROJECT_SOURCE_DIR}/external/onnxruntime``
    ``ORT_LIB_DIRS``, Path to the ORT lib directory, ``${ORT_INSTALL_DIRS}/lib``
