..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

SDXL-Turbo
==========

The standard ONNX model for SDXL-Turbo is available on `HuggingFace <https://huggingface.co/stabilityai/sdxl-turbo>`__.
For Computex 2024, parts of SDXL-Turbo were offloaded to the NPU: *unet* and *vae_decoder*.

To start, clone the repository or download the *unet* and *vae_decoder* models.
Large files are stored with git lfs so make sure you download the actual large files and not just lfs placeholder files.
These instructions assume you've cloned the repository to ``/path/to/sdxl-turbo``.

Preprocess
----------

First, you preprocess the default models. For *unet*:

.. code-block:: console

    cd /path/to/sdxl-turbo/unet
    onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize unet --save-as-external

And *vae_decoder*:

.. code-block:: console

    cd /path/to/sdxl-turbo/vae_decoder
    onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize vae

For fixing the input dimensions, use these values:

* *unet*
    * batch_size: 1
    * num_channels: 4
    * height: 64
    * width: 64
    * steps: 1
    * sequence_length: 77
* *vae_decoder*
    * batch_size: 1
    * num_channels_latent: 4
    * height_latent: 64
    * width_latent: 64

These steps will generate ``optimized.onnx`` models in the respective directories and an external data file for unet in its directory.

.. attention::

    You will need the ``RyzenAIExecutionProvider`` to preprocess the model. To install this, build and install this :ref:`onnxruntime_ipsp:ONNXRuntime_IPSP` in your Python environment. You can download pre-optimized models for SDXL-Turbo from `OneDrive <https://amdcloud-my.sharepoint.com/:f:/r/personal/chiz_amd_com/Documents/sdxl_onnx_turbo?csf=1&web=1&e=lW5Bem>`__. Here, they're named ``model_nhwc02a.onnx`` and ``model_nhwc02a.onnx_data`` but are the same as above.

    The ``replaced.onnx`` model in this location is old so it may not work.

Partition
---------

With the optimized models, you can partition them for NPU. For *unet*:

.. code-block:: console

    onnx_utils partition "/path/to/sdxl-turbo/unet/optimized.onnx" "/path/to/sdxl-turbo/unet" "sdxl_turbo_bfp_unet.yaml" -v --combine-dd --force

And *vae_decoder*:

.. code-block:: console

    onnx_utils partition "/path/to/sdxl-turbo/vae_decoder/optimized.onnx" "/path/to/sdxl-turbo/vae_decoder" "sdxl_turbo_bfp_vae_decoder.yaml" -v --combine-dd --force

These steps will generate a ``replaced.onnx`` models in the respective directories and a ``cache`` directory containing DD metadata files.
SDXL-Turbo is offloaded to the NPU using kernels from Mrinal's team (reach out to Akshay Jain for more information).

Run
---

Then to run, you need to :ref:`build the custom op library <installation:Custom ops shared library>` and load it in ORT.

For SDXL-Turbo, you need to use the following ORT session configurations:

.. code-block:: python

    import onnxruntime as ort

    decoder_session_options = ort.SessionOptions()
    decoder_session_options.add_session_config_entry("model_name", "DECODER")
    decoder_session_options.add_session_config_entry("dd_cache", "/path/to/sdxl-turbo/vae_decoder/cache")
    decoder_session_options.add_session_config_entry("dd_root", "/path/to/DD/source")
    # this should be the last step
    decoder_session_options.register_custom_ops_library("/path/to/shared/library")

    unet_session_options = ort.SessionOptions()
    unet_session_options.add_session_config_entry("model_name", "UNET")
    unet_session_options.add_session_config_entry("dd_cache", "/path/to/sdxl-turbo/unet/cache")
    unet_session_options.add_session_config_entry("dd_root", "/path/to/DD/source")
    # this should be the last step
    unet_session_options.register_custom_ops_library("/path/to/shared/library")

    ...

One way to run the SDXL-Turbo model is to use this `Python script <https://gitenterprise.xilinx.com/IPSP/RyzenAI_diffusers/blob/main/onnx/scripting/run_sdxl_turbo_fusion_vae%2Bunet.py>`__.
In this script, update the ``MODEL_DIR`` to your ``/path/to/sdxl-turbo`` and ``DD_ROOT`` to your DD source directory.
For these fully partitioned models, you can also update the providers to use ``CPUExecutionProvider`` or another standard EP.
Update the ``custom_op_path`` to the path to your custom ops shared library.
Then:

.. code-block:: console

    python ./run_sdxl_turbo_fusion_vae+unet.py
