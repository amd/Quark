.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Installation Guide
==================

Recommended First Time User Installation
----------------------------------------

If this is your first time setting up Quark, and want to try it out, this section will get you set up quickly with something
that will run on a laptop without a GPU.
When you are comfortable with the basic concepts you can come back to the following sections on this page for more advanced set up options.

The suggested installation of AMD Quark is in a Python environment such as `Miniforge <https://github.com/conda-forge/miniforge>`_, but you can also use `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/install>`_, or `Anaconda <https://www.anaconda.com/docs/getting-started/anaconda/install>`_.
For example, you can perform a typical interactive user installation of Miniconda, and then create and activate an environment for Quark and its dependencies with:

.. code-block:: bash

   wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   conda create -y -n AMD_Quark python=3.13
   conda activate AMD_Quark

You may then use ``pip`` to install Quark into your Python environment from PyPI, and all dependencies.

.. note::

   On Windows it is common for developers to use a Linux environment with *WSL* (Windows Subsystem for Linux), by installing Ubuntu *via* the Microsoft Store.
   In that case Linux installation instructions apply, and Miniforge is installed on Ubuntu.
   We suggest you do this for your first installation of Quark.
   It is also possible to install Quark on Windows directly.
   To do so, make sure to install the required dependencies as outlined in `Advanced Installation <#advanced>`_.

We need to install a C++ compiler for ONNX, such as ``g++``. In an Ubuntu environment you can install that with:

.. code-block:: bash

   sudo apt install build-essential

Next we will install PyTorch and Quark itself.
We've selected the CPU wheel of PyTorch here so that Quark will run on laptops without GPUs, which is slower, but fine for trying out Quark.
We will install Quark from PyPI, which will pull in required dependencies.

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Next, if you are using the ONNX-to-ONNX flow in Quark, please install ONNX Runtime. We recommend using an ONNX Runtime version ≥ 1.22.2 and ≤ 1.25.1 for compatibility.

.. code-block:: bash

   pip install "onnxruntime>=1.22.2,<=1.25.1"

Next, if you are using the OnnxRuntime Gen AI (OGA) Flow for LLM models, please install ONNX Runtime Gen AI.

.. code-block:: bash

   pip install onnxruntime-genai

That's it! You should now be able to move on to the *Getting started* guides in the side bar to try different workflows in Quark.
You can return to this guide later when you'd like to try a more advanced set up, for example, a new conda environment with a PyTorch
wheel supporting your GPU.


.. _advanced:

Advanced Installation
---------------------

The rest of this guide gives more specific installation instructions for:

* Set-up on other operating systems, such as a Microsoft Windows install, using Microsoft Visual Studio.
* Installation of PyTorch with GPU back-ends for better performance.
* Installing Quark with usage examples.
* Pre-compiling kernels and operators.
* Validating the installation.


Install Python
^^^^^^^^^^^^^^

Python 3.11, 3.12 or 3.13 is required. *Python 3.14 is not currently supported* by Quark's dependencies.

On all platforms we recommend installing Python with an environment such as `Miniforge <https://github.com/conda-forge/miniforge>`_,
which will simplify installation of dependencies.

.. note::

   On Windows consider adding Conda to your ``PATH`` in the install options,
   which will let you activate your environments easily, from within the *Developer Command Prompt*.

If you are using a Python environment, make sure that you create and activate an environment for Quark,
and that you install the dependencies from the following subsections into that environment. e.g.

.. code-block:: bash

   conda create -n AMD_Quark python=3.13
   conda activate AMD_Quark

Verify your Python version is one of those supported with:

.. code-block:: bash

   python --version

You should see the version number returned e.g. ``Python 3.13.0``, which is fine.

If you are using an conda-based environment, such as Miniforge,
we recommend that you install the following dependencies with ``pip install``.


Install PyTorch with GPU Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch 2.2.0 or later is required.

Windows
"""""""

To install **PyTorch with CUDA** 12.6 GPU support, in a Python environment using ``pip``:

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

If CUDA is not available, install PyTorch without GPU support:

.. code-block:: bash

   pip install torch torchvision torchaudio


Linux
"""""
.. note::

   The commands below assume **ROCm 7.1**, but for a different ROCm version or further options, consult the `PyTorch <https://pytorch.org/get-started/locally/>`__ install guide.

To install **PyTorch with ROCm** 7.1 GPU support, in a Python environment using ``pip``:

.. code-block:: bash

   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1

To install **PyTorch with CUDA** 13.0 GPU support:

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

If neither of these combinations is available on your system, you may install without GPU support:

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


Install ONNX Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ONNX Runtime version >=1.22.2 and <=1.25.1 is required.

Windows
"""""""
.. note::
   ROCm support on Windows is under active development and will be made available in a future release.

To install **ONNX Runtime with CUDA** 13.X GPU support, in a Python environment using ``pip``:

.. code-block:: bash

   pip install onnxruntime-gpu

If CUDA is not available, install ONNX Runtime without GPU support:

.. code-block:: bash

   pip install onnxruntime


Linux
"""""
.. note::

   The commands below assume **CPU**, but for a different accelerator or further options, consult the `ONNX Runtime <https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-rocm>`__ install guide.

.. To install **ONNX Runtime with ROCm 7.1** GPU support, in a Python environment using ``pip``:

.. .. code-block:: bash

..    pip install onnxruntime-rocm

To install **ONNX Runtime with CUDA** 13.X GPU support:

.. code-block:: bash

   pip install onnxruntime-gpu

If neither of these combinations is available on your system, you may install without GPU support:

.. code-block:: bash

   pip install onnxruntime


Install a C++ Compiler
^^^^^^^^^^^^^^^^^^^^^^

Windows
"""""""

When installing on Windows, `Visual Studio <https://visualstudio.microsoft.com/vs/community/>`_ is required,
with Visual Studio 2022 being the minimum required version.
During the compilation process, you can either use the *Developer Command Prompt*, or add paths to environment variables.

When installing Visual Studio, ensure that you choose the *Desktop development with C++* workload.

Alternatively, if you prefer not to use the *Developer Command Prompt*, you may set paths to the build tools by running the
`developer command file <https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170#developer_command_file_locations>`_
from your existing command prompt or within a batch file.

Linux
"""""

On Ubuntu the ``g++`` compiler is installed with the ``build-essential`` package.
Note that while many versions of C++ compiler may work, we currently only confirm support for g++ version 13.3,
which installs with the ``build-essential`` package on Ubuntu 24.04.


Install Quark
^^^^^^^^^^^^^

.. note::

   The AMD Quark package distribution name has been renamed to ``amd-quark``. Please use the new package name for releases newer than 0.6.0.

We recommend new users install Quark from PyPI with ``pip``. It's also possible to install from a ZIP download, which contains additional examples.


Quick Install Selector
""""""""""""""""""""""

Select your preferences and run the generated command. AMD Quark is distributed in two wheel
flavors, hosted on **different package indexes**:

* The **universal wheel** (``py3-none-any``) is the **recommended default for most users**, and is
  published to `PyPI <https://pypi.org/project/amd-quark/>`__. It works on every operating system, Python
  version, and accelerator, regardless of your PyTorch version, so it is the most portable and reliable
  choice. It does not ship pre-compiled ``_C`` extensions; instead, the *fast quantization kernels* and
  *ONNX custom operators library* are compiled the first time they are imported, which requires a C++
  compiler (and ``nvcc`` / ``hipcc`` for GPU builds) on your machine — see *Install a C++ Compiler* above.
* The **pre-built wheels** are an optional optimization for PyTorch 2.10 and newer, and are hosted on
  the **AMD package index** rather than PyPI. They ship pre-compiled C++ extensions, which means:

  * **No C++ compiler is needed** — you can skip the *Install a C++ Compiler* step entirely, with no
    ``g++`` / ``build-essential``, Visual Studio, ``nvcc``, or ``hipcc`` required.
  * **No first-run compilation** — the first ``import quark`` no longer triggers a one-time JIT build,
    so the *Compile Fast Quantization Kernels* and *Compile Custom Operators Library* steps below are
    unnecessary.

  Pre-built CPU and CUDA wheels are published for both Linux and Windows; ROCm 7.1 and ROCm 7.2 wheels
  are Linux-only. All pre-built wheels target Python 3.11–3.13.

Because the two flavors live on different indexes, the one you get depends on which index you install
from. A plain ``pip install amd-quark`` pulls the universal wheel from PyPI. To get a pre-built wheel,
point ``pip`` at the AMD package index. When in doubt, the universal wheel from PyPI is
the safe, preferred choice.

.. The container below is a static fallback for renderers that strip the interactive
   ``raw:: html`` widget (e.g. GitHub's file viewer, which sanitizes embedded HTML/CSS/JS).
   In the Sphinx-built docs it is hidden once qk-install-selector.js builds the selector
   (the script adds a ``qk-js-ready`` class that the stylesheet keys off), so it stays
   visible if those assets fail to load.

.. container:: qk-static-fallback

   If the interactive selector below does not appear (for example, when viewing this file
   on a Git host that strips embedded HTML), use the commands in this table directly:

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Wheel / compute platform
        - Command
      * - Universal (PyPI) — recommended, any OS / Python / accelerator
        - ``pip install amd-quark``
      * - Pre-built, CPU
        - ``pip install amd-quark --extra-index-url https://pypi.amd.com/quark/cpu/simple``
      * - Pre-built, CUDA 12.8
        - ``pip install amd-quark --extra-index-url https://pypi.amd.com/quark/cu128/simple``
      * - Pre-built, ROCm 7.1 (Linux only)
        - ``pip install amd-quark --extra-index-url https://pypi.amd.com/quark/rocm71/simple``
      * - Pre-built, ROCm 7.2 (Linux only)
        - ``pip install amd-quark --extra-index-url https://pypi.amd.com/quark/rocm72/simple``

   Pre-built wheels require PyTorch 2.10+ and Python 3.11–3.13; otherwise use the universal
   wheel.

.. raw:: html

   <!-- The interactive "Quick Install Selector" mounts here. Its markup, styles,
        and behaviour live in docs/source/_static/qk-install-selector.{css,js},
        loaded only by the Sphinx build (see conf.py setup()). Renderers that do
        not load those assets (e.g. GitHub's file viewer) show only the static
        fallback table above, instead of unstyled markup or raw CSS/JS text. -->
   <div class="qk-selector" id="qk-selector"></div>


Install Quark + Quark Examples from Download
""""""""""""""""""""""""""""""""""""""""""""

Download and unzip 📥*amd_quark-\*.zip*, which has a wheel package in it.
You can also download the wheel package 📥*amd_quark-\*.whl* directly.
We strongly recommend downloading the ZIP file, as it includes examples compatible with the wheel package version.

- `📥amd_quark.zip release_version (recommended) <https://download.amd.com/opendownload/Quark/amd_quark-@version@.zip>`__
- `📥amd_quark.whl release_version <https://download.amd.com/opendownload/Quark/amd_quark-@version@-py3-none-any.whl>`__

Directory Structure of the zip file:

::

   + amd_quark.zip
      + amd_quark.whl
      + examples    # Examples of using Quark.
      + docs        # Off-line documentation of Quark.
      + README.md

Then install the quark wheel package by running the following command:

.. code-block:: bash

   pip install amd_quark*.whl

.. note::

   If your Quark ``pip`` install fails with dependency version mismatches, check that you are running a supported version of Python.

.. note::

   If your Windows ``pip`` installation of ONNX dependencies with Quark is failing on a long generated path name, you may enable long path name support in Windows
   in the *Group Policy Editor*. In the Windows *Start Menu*, type ``GPEDIT.MSC`` in search box, and open the editor. Navigate to:
   *Computer Configuration -> Administrative Templates -> System -> Filesystem*, and change the setting for *Enable Win32 long paths*
   to *Enabled*. You will need to restart your terminal for changes to take effect.


Installation Verification (Optional)
------------------------------------

Verify the installation by running:

.. code-block:: bash

   python -c "import quark"

If no error is reported, then the installation was successful.


Compile Fast Quantization Kernels (Optional)
--------------------------------------------

When using Quark's quantization APIs for the first time, it compiles the *fast quantization kernels* using your installed PyTorch with GPU support, if available.
This process might take a few minutes, but the subsequent quantization calls are then much faster.
This process requires the `Transformers <https://huggingface.co/docs/transformers/en/index>`__ package from Hugging Face.
To invoke this compilation now, and check if it is successful, run the following commands:

.. code-block:: bash

   pip install transformers
   python -c "import quark.torch.kernel"

If the kernel cannot be loaded successfully with Quark on Windows with GPU support, follow the steps below to troubleshoot:

- Ensure a C++ compiler is installed, and can be invoked from the command line.
- Check GPU support with a Python call to ``torch.cuda.is_available()`` returning ``True``.
- Verify that ``nvcc``, for CUDA, or ``hipcc``, for ROCm HIP, can be invoked from the command line.
- If the compilation is successful, but Python fails to load the DLL, locate the Quark build directory (the build path will be printed in the log), and check the dependencies using ``dumpbin /DEPENDENTS kernel_ext.pyd``.
- Check that the required DLL is included in the system path.
- Check that the version of the dependent DLL is correct.
- Check that the Python version of the dependent DLL is correct.


Compile Custom Operators Library (Optional)
-------------------------------------------

When using Quark's ONNX custom operators for the first time, it compiles the *custom operators library* using your local environment.
To invoke this compilation now, and check if it is successful, run the following command:

.. code-block:: bash

   python -c "import quark.onnx.operators.custom_ops"

Please note that ``ROCM_PATH`` or ``CUDA_HOME`` needs to be set to the local installation directory on the server (typically under ``/opt/rocm``
or ``/usr/local/cuda`` on Linux) for building GPU version library. This allows the compiler to locate the necessary header files.

Setting logging level (Optional)
--------------------------------

It is possible to control the logging level used by AMD Quark with the environment variable ``QUARK_LOG_LEVEL``:

- ``QUARK_LOG_LEVEL=debug``: Display all logs (useful for debugging).
- ``QUARK_LOG_LEVEL=info``: Display logs of level info and above (default).
- ``QUARK_LOG_LEVEL=warning``: Display logs of level warning and above.
- ``QUARK_LOG_LEVEL=error``: Display logs of level error and above.
- ``QUARK_LOG_LEVEL=critical``: Display logs of level critical only.


Previous Versions of AMD Quark
------------------------------

**Note**: The following links are for older versions of AMD Quark, before the package distribution name was renamed to ``amd-quark``.

-  `quark_0.12.zip <https://download.amd.com/opendownload/Quark/amd_quark-0.12.zip>`__
-  `quark_0.11.zip <https://download.amd.com/opendownload/Quark/amd_quark-0.11.zip>`__
-  `quark_0.10.zip <https://download.amd.com/opendownload/Quark/amd_quark-0.10.zip>`__
-  `quark_0.9.zip <https://download.amd.com/opendownload/Quark/amd_quark-0.9.zip>`__
-  `quark_0.8.2.zip <https://download.amd.com/opendownload/Quark/amd_quark-0.8.2.zip>`__
-  `quark_0.8.1.zip <https://download.amd.com/opendownload/Quark/amd_quark-0.8.1.zip>`__
-  `quark_0.8.zip <https://www.xilinx.com/bin/public/openDownload?filename=amd_quark-0.8.zip>`__
-  `quark_0.7.zip <https://www.xilinx.com/bin/public/openDownload?filename=amd_quark-0.7.zip>`__
-  `quark_0.6.0.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.6.0.zip>`__
-  `quark_0.5.1.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.5.1+88e60b456.zip>`__
-  `quark_0.5.0.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.5.0+fae64a406.zip>`__
-  `quark_0.2.0.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.2.0+6af1bac23.zip>`__
-  `quark_0.1.0.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.1.0+a9827f5.zip>`__
