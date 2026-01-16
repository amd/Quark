..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Stable Diffusion 1.5
====================

Parts of SD 1.5 were offloaded to the NPU: UNet, ControlNet and vae_decoder.

To start, clone the repository or download the *UNet* and *vae_decoder* models.
Large files are stored with git lfs so make sure you download the actual large files and not just lfs placeholder files.
These instructions assume you've cloned the repository to ``/path/to/sd1.5``.

Preprocess
----------

First, you preprocess the default models. For *UNet*:

.. code-block:: console

    cd /path/to/sd1.5/unet
    onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize sd15_unet --save-as-external

*Controlnet*

.. code-block:: console

    cd /path/to/sd1.5/controlnet
    onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize sd15_controlnet --save-as-external

And *vae_decoder*

.. code-block:: console

    cd /path/to/vae_decoder
    onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize vae

These steps will generate ``optimized.onnx`` models in the respective directories and an external data file for UNet and
Controlnet in the respective directories.

.. attention::

    You will need the ``RyzenAIExecutionProvider`` to preprocess the model.
    To install this, build and install this :ref:`onnxruntime_ipsp:ONNXRuntime_IPSP`
    in your Python environment.

    You can download pre-optimized models for SD1.5 from
    `OneDrive <https://amdcloud-my.sharepoint.com/:f:/r/personal/varunsh_amd_com/Documents/Shared/stable_diffusion/sd-1.5>`__.


    1. For UNet, the pre-optimized models at: `OneDrive <https://amdcloud-my.sharepoint.com/:f:/r/personal/varunsh_amd_com/Documents/Shared/stable_diffusion/sd-1.5/unet>`__.

    Including:

    - ``optimized_nhwc_matmul_v3.onnx``
    - ``optimized_nhwc_matmul_v3.onnx_data``

    2. For vae_decoder, the pre-optimized models at: `OneDrive <https://amdcloud-my.sharepoint.com/:f:/r/personal/varunsh_amd_com/Documents/Shared/stable_diffusion/sd-1.5/vae_decoder>`__.

    Including:

    - ``vae_decoder_optimized_nhwc_mha_v2.onnx``

    The project also support SD1.5 ControlNet.

    1. For controlnet, the pre-optimized models at: `OneDrive <https://amdcloud-my.sharepoint.com/:f:/r/personal/varunsh_amd_com/Documents/Shared/stable_diffusion/sd1.5-controlnet/controlnet>`__.

    Including:

    - ``controlnet_bs2_512x512.onnx``
    - ``controlnet_bs2_512x512.onnx_data``

    2. For unet with controlnet, the pre-optimized models at: `OneDrive <https://amdcloud-my.sharepoint.com/:f:/r/personal/varunsh_amd_com/Documents/Shared/stable_diffusion/sd1.5-controlnet/unet>`__.

    Including:

    - ``unet_bs2_512x512.onnx``
    - ``unet_bs2_512x512.onnx_data``


Partition
---------

With the optimized models, you can partition them for NPU. For *UNet*:

.. code-block:: console

    onnx_utils partition "/path/to/sd1.5/unet/optimized.onnx" "/path/to/sd1.5/unet/" "sd15_unet.yaml" -v --force

*ControlNet*:

.. code-block:: console

    onnx_utils partition "/path/to/sd1.5/controlnet/optimized.onnx" "path/to/sd1.5/controlnet/" "sd15_unet.yaml" -v --force

And *vae_decoder*:

.. code-block:: console

    onnx_utils partition "/path/to/sd1.5/vae_decoder/optimized.onnx" "/path/to/sd1.5/vae_decoder" "sd15_vae_decoder.yaml" -v --force


These steps will generate a ``replaced.onnx`` models in the respective directories and a ``.cache`` directory containing DD metadata files.
SD1.5 is offloaded to the NPU using kernels from Tianping's team and others.

Run
---

Then to run, you need to :ref:`build the custom op library <installation:Custom ops shared library>` and load it in ORT.

For SD1.5, you need to use the following ORT session configurations:

.. code-block:: python

    import onnxruntime as ort

    decoder_session_options = ort.SessionOptions()
    decoder_session_options.add_session_config_entry("model_name", "SD15_DECODER")
    decoder_session_options.add_session_config_entry("dd_cache", "/path/to/sd1.5/vae_decoder/.cache")
    decoder_session_options.add_session_config_entry("dd_root", "/path/to/DD/source")
    # this should be the last step
    decoder_session_options.register_custom_ops_library("/path/to/shared/library")

    unet_session_options = ort.SessionOptions()
    unet_session_options.add_session_config_entry("model_name", "SD15_UNET")
    unet_session_options.add_session_config_entry("dd_cache", "/path/to/sd1.5/unet/.cache")
    unet_session_options.add_session_config_entry("dd_root", "/path/to/DD/source")
    # this should be the last step
    unet_session_options.register_custom_ops_library("/path/to/shared/library")

    controlnet_session_options = ort.SessionOptions()
    controlnet_session_options.add_session_config_entry("model_name", "SD15_UNET")
    controlnet_session_options.add_session_config_entry("dd_cache", "/path/to/sd1.5/controlnet/.cache")
    controlnet_session_options.add_session_config_entry("dd_root", "/path/to/DD/source")
    # this should be the last step
    controlnet_session_options.register_custom_ops_library("/path/to/shared/library")

    ...

Reports
-------

The latest reports for this SD 1.5 are here for reference.
Each submodel in SD 1.5 has its own section below.

unet (without controlnet)
^^^^^^^^^^^^^^^^^^^^^^^^^

The latest (version: optimized_nhwc_matmul_v3.onnx) reports the UNET model are here for reference.

.. collapse:: Op adjacency report

    .. code-block::

        Op: Add (212)
            Level: 1
                (('MatMul',),): 49.06%
                (('Add',), ('Add',)): 15.09%
                (('NhwcConv',), ('Reshape',)): 10.38%
                (('Add',), ('NhwcConv',)): 10.38%
                (('NhwcConv',), ('NhwcConv',)): 7.55%
                (('Add',), ('Reshape',)): 7.55%
            Level: 2
                (('MatMul', 'Mul'),): 18.40%
                (('MatMul', 'MultiHeadAttention'),): 15.09%
                (('MatMul', 'LayerNormalization'),): 15.09%
                (('NhwcConv', 'Mul'), ('Reshape', 'Add')): 10.38%
                (('Add', 'MatMul'), ('Reshape', 'NhwcConv')): 7.55%
                (('Add', 'Add'), ('Add', 'MatMul'), ('Add', 'Reshape')): 7.55%
                (('Add', 'Add'), ('Add', 'Add'), ('Add', 'MatMul')): 7.55%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('NhwcConv', 'Reshape')): 5.66%
                (('NhwcConv', 'Concat'), ('NhwcConv', 'Mul')): 5.66%
                (('Add', 'Add'), ('Add', 'NhwcConv'), ('NhwcConv', 'Mul')): 2.36%
                (('Add', 'Add'), ('Add', 'NhwcConv'), ('NhwcConv', 'Reshape')): 1.89%
                (('NhwcConv', 'Mul'), ('NhwcConv', 'NhwcConv')): 0.94%
                (('MatMul', 'Concat'),): 0.47%
                (('NhwcConv', 'Mul'), ('NhwcConv', 'Transpose')): 0.47%
                (('NhwcConv', 'Add'), ('NhwcConv', 'Mul')): 0.47%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('NhwcConv', 'Mul')): 0.47%
        Op: Cast (1)
            Level: 1
                (('Reshape',),): 100.00%
        Op: Concat (13)
            Level: 1
                (('Add',), ('NhwcConv',)): 53.85%
                (('Add',), ('Add',)): 38.46%
                (('Cos',), ('Sin',)): 7.69%
            Level: 2
                (('Add', 'Add'), ('Add', 'Add'), ('Add', 'NhwcConv'), ('Add', 'NhwcConv')): 30.77%
                (('Add', 'Add'), ('Add', 'NhwcConv'), ('NhwcConv', 'Resize')): 23.08%
                (('Add', 'Add'), ('Add', 'NhwcConv'), ('NhwcConv', 'Add')): 15.38%
                (('Cos', 'Mul'), ('Sin', 'Mul')): 7.69%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('Add', 'NhwcConv')): 7.69%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('NhwcConv', 'Add')): 7.69%
                (('Add', 'Add'), ('Add', 'NhwcConv'), ('NhwcConv', 'Transpose')): 7.69%
        Op: Cos (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Cast'),): 100.00%
        Op: Gelu (16)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'MatMul'),): 100.00%
        Op: GroupNorm (61)
            Level: 1
                (('Add',),): 73.77%
                (('Concat',),): 19.67%
                (('NhwcConv',),): 6.56%
            Level: 2
                (('Add', 'NhwcConv'), ('Add', 'Reshape')): 36.07%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv')): 21.31%
                (('Add', 'Add'), ('Add', 'NhwcConv')): 16.39%
                (('Concat', 'Add'), ('Concat', 'NhwcConv')): 11.48%
                (('Concat', 'Add'), ('Concat', 'Add')): 8.20%
                (('NhwcConv', 'Add'),): 4.92%
                (('NhwcConv', 'Transpose'),): 1.64%
        Op: LayerNormalization (48)
            Level: 1
                (('Add',),): 66.67%
                (('Reshape',),): 33.33%
            Level: 2
                (('Reshape', 'NhwcConv'),): 33.33%
                (('Add', 'Add'), ('Add', 'Reshape')): 33.33%
                (('Add', 'Add'), ('Add', 'Add')): 33.33%
        Op: MatMul (200)
            Level: 1
                (('LayerNormalization',),): 57.14%
                (('Mul',),): 23.21%
                (('MultiHeadAttention',),): 19.05%
                (('Concat',),): 0.60%
            Level: 2
                (('LayerNormalization', 'Reshape'),): 28.57%
                (('LayerNormalization', 'Add'),): 28.57%
                (('MultiHeadAttention', 'MatMul'), ('MultiHeadAttention', 'MatMul'), ('MultiHeadAttention', 'MatMul')): 19.05%
                (('Mul', 'Add'), ('Mul', 'Sigmoid')): 13.69%
                (('Mul', 'Add'), ('Mul', 'Gelu')): 9.52%
                (('Concat', 'Cos'), ('Concat', 'Sin')): 0.60%
        Op: Mul (64)
            Level: 1
                (('GroupNorm',), ('Sigmoid',)): 70.31%
                (('Add',), ('Gelu',)): 25.00%
                (('Add',), ('Sigmoid',)): 3.12%
                (('Cast',),): 1.56%
            Level: 2
                (('GroupNorm', 'Add'), ('Sigmoid', 'GroupNorm')): 45.31%
                (('Add', 'MatMul'), ('Gelu', 'Add')): 25.00%
                (('GroupNorm', 'Concat'), ('Sigmoid', 'GroupNorm')): 18.75%
                (('GroupNorm', 'NhwcConv'), ('Sigmoid', 'GroupNorm')): 6.25%
                (('Add', 'MatMul'), ('Sigmoid', 'Add')): 3.12%
                (('Cast', 'Reshape'),): 1.56%
        Op: MultiHeadAttention (32)
            Level: 1
                (('MatMul',), ('MatMul',), ('MatMul',)): 100.00%
            Level: 2
                (('MatMul', 'LayerNormalization'), ('MatMul', 'LayerNormalization'), ('MatMul', 'LayerNormalization')): 50.00%
                (('MatMul', 'LayerNormalization'),): 50.00%
        Op: NhwcConv (98)
            Level: 1
                (('Mul',),): 45.92%
                (('GroupNorm',),): 16.33%
                (('Reshape',),): 16.33%
                (('Concat',),): 12.24%
                (('Add',),): 3.06%
                (('Resize',),): 3.06%
                (('NhwcConv',),): 2.04%
                (('Transpose',),): 1.02%
            Level: 2
                (('Mul', 'GroupNorm'), ('Mul', 'Sigmoid')): 46.39%
                (('GroupNorm', 'Add'),): 16.49%
                (('Reshape', 'Add'),): 16.49%
                (('Concat', 'Add'), ('Concat', 'NhwcConv')): 7.22%
                (('Concat', 'Add'), ('Concat', 'Add')): 5.15%
                (('Add', 'Add'), ('Add', 'NhwcConv')): 3.09%
                (('Resize', 'Add'),): 3.09%
                (('NhwcConv', 'Add'),): 2.06%
        Op: Reshape (55)
            Level: 1
                (('Add',),): 70.37%
                (('NhwcConv',),): 29.63%
                (('NhwcConv',),): 29.63%
            Level: 2
                (('Add', 'MatMul'),): 40.74%
                (('NhwcConv', 'GroupNorm'),): 29.63%
                (('Add', 'Add'), ('Add', 'Add')): 29.63%
        Op: Resize (3)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'Add'), ('Add', 'NhwcConv')): 66.67%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv')): 33.33%
        Op: Sigmoid (47)
            Level: 1
                (('GroupNorm',),): 95.74%
                (('Add',),): 4.26%
            Level: 2
                (('GroupNorm', 'Add'),): 61.70%
                (('GroupNorm', 'Concat'),): 25.53%
                (('GroupNorm', 'NhwcConv'),): 8.51%
                (('Add', 'MatMul'),): 4.26%
        Op: Sin (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Cast'),): 100.00%
        Op: Transpose (2)
            Level: 1
                (('NhwcConv',),): 100.00%
            Level: 2
                (('NhwcConv', 'Mul'),): 100.00%

The command used for the op_adjacency report was: ``onnx_utils report op_adjacency /path/to/sd1.5/unet/optimized.onnx --depth
2``.

unet (with controlnet)
^^^^^^^^^^^^^^^^^^^^^^

The latest reports (version: unet_bs2_512x512.onnx) the CONTROLNET model are here for reference.

.. collapse:: Op adjacency report

    .. code-block::

        Op: Add (225)
            Level: 1
                (('MatMul',),): 46.22%
                (('Add',), ('Add',)): 14.22%
                (('NhwcConv',), ('Reshape',)): 9.78%
                (('Add',), ('NhwcConv',)): 9.78%
                (('NhwcConv',), ('NhwcConv',)): 7.11%
                (('Add',), ('Reshape',)): 7.11%
                (('Add',), ('Transpose',)): 4.00%
                (('NhwcConv',), ('Transpose',)): 1.78%
            Level: 2
                (('MatMul', 'Mul'),): 17.33%
                (('MatMul', 'MultiHeadAttention'),): 14.22%
                (('MatMul', 'LayerNormalization'),): 14.22%
                (('NhwcConv', 'Mul'), ('Reshape', 'Add')): 9.78%
                (('Add', 'MatMul'), ('Reshape', 'NhwcConv')): 7.11%
                (('Add', 'Add'), ('Add', 'MatMul'), ('Add', 'Reshape')): 7.11%
                (('Add', 'Add'), ('Add', 'Add'), ('Add', 'MatMul')): 7.11%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('NhwcConv', 'Reshape')): 5.33%
                (('NhwcConv', 'Concat'), ('NhwcConv', 'Mul')): 5.33%
                (('Add', 'Add'), ('Add', 'NhwcConv')): 3.56%
                (('Add', 'Add'), ('Add', 'NhwcConv'), ('NhwcConv', 'Mul')): 2.22%
                (('Add', 'Add'), ('Add', 'NhwcConv'), ('NhwcConv', 'Reshape')): 1.78%
                (('NhwcConv', 'Add'),): 1.33%
                (('NhwcConv', 'Mul'), ('NhwcConv', 'NhwcConv')): 0.89%
                (('NhwcConv', 'Transpose'),): 0.44%
                (('MatMul', 'Concat'),): 0.44%
                (('NhwcConv', 'Mul'), ('NhwcConv', 'Transpose')): 0.44%
                (('NhwcConv', 'Add'), ('NhwcConv', 'Mul')): 0.44%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv')): 0.44%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('NhwcConv', 'Mul')): 0.44%
        Op: Cast (1)
            Level: 1
                (('Reshape',),): 100.00%
        Op: Concat (13)
            Level: 1
                (('Add',), ('Add',)): 69.23%
                (('Add',), ('NhwcConv',)): 23.08%
                (('Cos',), ('Sin',)): 7.69%
            Level: 2
                (('Add', 'Add'), ('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('Add', 'Transpose')): 30.77%
                (('Add', 'Add'), ('Add', 'Transpose'), ('NhwcConv', 'Resize')): 23.08%
                (('Add', 'Add'), ('Add', 'Add'), ('Add', 'NhwcConv'), ('Add', 'Transpose')): 23.08%
                (('Cos', 'Mul'), ('Sin', 'Mul')): 7.69%
                (('Add', 'Add'), ('Add', 'Add'), ('Add', 'Transpose'), ('Add', 'Transpose')): 7.69%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('Add', 'Transpose')): 7.69%
        Op: Cos (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Cast'),): 100.00%
        Op: Gelu (16)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'MatMul'),): 100.00%
        Op: GroupNorm (61)
            Level: 1
                (('Add',),): 73.77%
                (('Concat',),): 19.67%
                (('NhwcConv',),): 6.56%
            Level: 2
                (('Add', 'NhwcConv'), ('Add', 'Reshape')): 36.07%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv')): 21.31%
                (('Add', 'Add'), ('Add', 'NhwcConv')): 16.39%
                (('Concat', 'Add'), ('Concat', 'Add')): 14.75%
                (('NhwcConv', 'Add'),): 4.92%
                (('Concat', 'Add'), ('Concat', 'NhwcConv')): 4.92%
                (('NhwcConv', 'Transpose'),): 1.64%
        Op: LayerNormalization (48)
            Level: 1
                (('Add',),): 66.67%
                (('Reshape',),): 33.33%
            Level: 2
                (('Reshape', 'NhwcConv'),): 33.33%
                (('Add', 'Add'), ('Add', 'Reshape')): 33.33%
                (('Add', 'Add'), ('Add', 'Add')): 33.33%
        Op: MatMul (200)
            Level: 1
                (('LayerNormalization',),): 57.14%
                (('Mul',),): 23.21%
                (('MultiHeadAttention',),): 19.05%
                (('Concat',),): 0.60%
            Level: 2
                (('LayerNormalization', 'Reshape'),): 28.57%
                (('LayerNormalization', 'Add'),): 28.57%
                (('MultiHeadAttention', 'MatMul'), ('MultiHeadAttention', 'MatMul'), ('MultiHeadAttention', 'MatMul')): 19.05%
                (('Mul', 'Add'), ('Mul', 'Sigmoid')): 13.69%
                (('Mul', 'Add'), ('Mul', 'Gelu')): 9.52%
                (('Concat', 'Cos'), ('Concat', 'Sin')): 0.60%
        Op: Mul (64)
            Level: 1
                (('GroupNorm',), ('Sigmoid',)): 70.31%
                (('Add',), ('Gelu',)): 25.00%
                (('Add',), ('Sigmoid',)): 3.12%
                (('Cast',),): 1.56%
            Level: 2
                (('GroupNorm', 'Add'), ('Sigmoid', 'GroupNorm')): 45.31%
                (('Add', 'MatMul'), ('Gelu', 'Add')): 25.00%
                (('GroupNorm', 'Concat'), ('Sigmoid', 'GroupNorm')): 18.75%
                (('GroupNorm', 'NhwcConv'), ('Sigmoid', 'GroupNorm')): 6.25%
                (('Add', 'MatMul'), ('Sigmoid', 'Add')): 3.12%
                (('Cast', 'Reshape'),): 1.56%
        Op: MultiHeadAttention (32)
            Level: 1
                (('MatMul',), ('MatMul',), ('MatMul',)): 100.00%
            Level: 2
                (('MatMul', 'LayerNormalization'), ('MatMul', 'LayerNormalization'), ('MatMul', 'LayerNormalization')): 50.00%
                (('MatMul', 'LayerNormalization'),): 50.00%
        Op: NhwcConv (98)
            Level: 1
                (('Mul',),): 45.92%
                (('GroupNorm',),): 16.33%
                (('Reshape',),): 16.33%
                (('Concat',),): 12.24%
                (('Add',),): 3.06%
                (('Resize',),): 3.06%
                (('NhwcConv',),): 2.04%
                (('Transpose',),): 1.02%
            Level: 2
                (('Mul', 'GroupNorm'), ('Mul', 'Sigmoid')): 46.39%
                (('GroupNorm', 'Add'),): 16.49%
                (('Reshape', 'Add'),): 16.49%
                (('Concat', 'Add'), ('Concat', 'Add')): 9.28%
                (('Add', 'Add'), ('Add', 'NhwcConv')): 3.09%
                (('Resize', 'Add'),): 3.09%
                (('Concat', 'Add'), ('Concat', 'NhwcConv')): 3.09%
                (('NhwcConv', 'Add'),): 2.06%
        Op: Reshape (55)
            Level: 1
                (('Add',),): 70.37%
                (('NhwcConv',),): 29.63%
            Level: 2
                (('Add', 'MatMul'),): 40.74%
                (('NhwcConv', 'GroupNorm'),): 29.63%
                (('Add', 'Add'), ('Add', 'Add')): 29.63%
        Op: Resize (3)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'Add'), ('Add', 'NhwcConv')): 66.67%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv')): 33.33%
        Op: Sigmoid (47)
            Level: 1
                (('GroupNorm',),): 95.74%
                (('Add',),): 4.26%
            Level: 2
                (('GroupNorm', 'Add'),): 61.70%
                (('GroupNorm', 'Concat'),): 25.53%
                (('GroupNorm', 'NhwcConv'),): 8.51%
                (('Add', 'MatMul'),): 4.26%
        Op: Sin (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Cast'),): 100.00%
        Op: Transpose (15)
            Level: 1
                (('NhwcConv',),): 100.00%
            Level: 2
                (('NhwcConv', 'Mul'),): 100.00%


ControlNet
^^^^^^^^^^

The latest reports (version: controlnet_bs2_512x512.onnx) the CONTROLNET model are here for reference.

.. collapse:: Op adjacency report

    .. code-block::

        Op: Add (96)
            Level: 1
                (('MatMul',),): 48.96%
                (('Add',), ('NhwcConv',)): 14.58%
                (('Add',), ('Add',)): 14.58%
                (('NhwcConv',), ('Reshape',)): 10.42%
                (('Add',), ('Reshape',)): 7.29%
                (('NhwcConv',), ('NhwcConv',)): 4.17%
            Level: 2
                (('MatMul', 'Mul'),): 18.75%
                (('MatMul', 'MultiHeadAttention'),): 14.58%
                (('MatMul', 'LayerNormalization'),): 14.58%
                (('NhwcConv', 'Mul'), ('Reshape', 'Add')): 10.42%
                (('Add', 'MatMul'), ('Reshape', 'NhwcConv')): 7.29%
                (('Add', 'Add'), ('Add', 'MatMul'), ('Add', 'Reshape')): 7.29%
                (('Add', 'Add'), ('Add', 'Add'), ('Add', 'MatMul')): 7.29%
                (('Add', 'Add'), ('Add', 'NhwcConv'), ('NhwcConv', 'Reshape')): 5.21%
                (('Add', 'Add'), ('Add', 'NhwcConv'), ('NhwcConv', 'Mul')): 5.21%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('NhwcConv', 'Mul')): 2.08%
                (('NhwcConv', 'Mul'), ('NhwcConv', 'NhwcConv')): 2.08%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('NhwcConv', 'Reshape')): 2.08%
                (('MatMul', 'Concat'),): 1.04%
                (('NhwcConv', 'Mul'), ('NhwcConv', 'Transpose')): 1.04%
                (('NhwcConv', 'Add'), ('NhwcConv', 'Mul')): 1.04%
        Op: Cast (2)
            Level: 1
                (('Reshape',),): 100.00%
        Op: Concat (1)
            Level: 1
                (('Cos',), ('Sin',)): 100.00%
            Level: 2
                (('Cos', 'Mul'), ('Sin', 'Mul')): 100.00%
        Op: Cos (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Cast'),): 100.00%
        Op: Gelu (7)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'MatMul'),): 100.00%
        Op: GroupNorm (27)
            Level: 1
                (('Add',),): 88.89%
                (('NhwcConv',),): 11.11%
            Level: 2
                (('Add', 'NhwcConv'), ('Add', 'Reshape')): 37.04%
                (('Add', 'Add'), ('Add', 'NhwcConv')): 37.04%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv')): 14.81%
                (('NhwcConv', 'Add'),): 11.11%
        Op: LayerNormalization (21)
            Level: 1
                (('Add',),): 66.67%
                (('Reshape',),): 33.33%
            Level: 2
                (('Reshape', 'NhwcConv'),): 33.33%
                (('Add', 'Add'), ('Add', 'Reshape')): 33.33%
                (('Add', 'Add'), ('Add', 'Add')): 33.33%
        Op: MatMul (89)
            Level: 1
                (('LayerNormalization',),): 56.00%
                (('Mul',),): 24.00%
                (('MultiHeadAttention',),): 18.67%
                (('Concat',),): 1.33%
            Level: 2
                (('LayerNormalization', 'Reshape'),): 28.00%
                (('LayerNormalization', 'Add'),): 28.00%
                (('MultiHeadAttention', 'MatMul'), ('MultiHeadAttention', 'MatMul'), ('MultiHeadAttention', 'MatMul')): 18.67%
                (('Mul', 'Add'), ('Mul', 'Sigmoid')): 14.67%
                (('Mul', 'Add'), ('Mul', 'Gelu')): 9.33%
                (('Concat', 'Cos'), ('Concat', 'Sin')): 1.33%
        Op: Mul (51)
            Level: 1
                (('GroupNorm',), ('Sigmoid',)): 39.22%
                (('NhwcConv',), ('Split',)): 23.53%
                (('NhwcConv',), ('Sigmoid',)): 13.73%
                (('Add',), ('Gelu',)): 13.73%
                (('Cast',),): 3.92%
                (('Add',), ('Sigmoid',)): 3.92%
                (('Cast',), ('NhwcConv',)): 1.96%
            Level: 2
                (('GroupNorm', 'Add'), ('Sigmoid', 'GroupNorm')): 34.00%
                (('NhwcConv', 'Add'), ('Split', 'Mul')): 18.00%
                (('Add', 'MatMul'), ('Gelu', 'Add')): 14.00%
                (('NhwcConv', 'Mul'), ('Sigmoid', 'NhwcConv')): 12.00%
                (('NhwcConv', 'NhwcConv'), ('Split', 'Mul')): 6.00%
                (('GroupNorm', 'NhwcConv'), ('Sigmoid', 'GroupNorm')): 6.00%
                (('Add', 'MatMul'), ('Sigmoid', 'Add')): 4.00%
                (('Cast', 'Reshape'),): 2.00%
                (('NhwcConv', 'Transpose'), ('Sigmoid', 'NhwcConv')): 2.00%
                (('NhwcConv', 'Add'),): 2.00%
        Op: MultiHeadAttention (14)
            Level: 1
                (('MatMul',), ('MatMul',), ('MatMul',)): 100.00%
            Level: 2
                (('MatMul', 'LayerNormalization'), ('MatMul', 'LayerNormalization'), ('MatMul', 'LayerNormalization')): 50.00%
                (('MatMul', 'LayerNormalization'),): 50.00%
        Op: NhwcConv (61)
            Level: 1
                (('Mul',),): 44.26%
                (('Add',),): 21.31%
                (('GroupNorm',),): 11.48%
                (('Reshape',),): 11.48%
                (('NhwcConv',),): 8.20%
                (('Transpose',),): 3.28%
            Level: 2
                (('Mul', 'GroupNorm'), ('Mul', 'Sigmoid')): 33.90%
                (('Add', 'Add'), ('Add', 'NhwcConv')): 18.64%
                (('Mul', 'NhwcConv'), ('Mul', 'Sigmoid')): 11.86%
                (('GroupNorm', 'Add'),): 11.86%
                (('Reshape', 'Add'),): 11.86%
                (('NhwcConv', 'Add'),): 8.47%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv')): 3.39%
        Op: Reshape (25)
            Level: 1
                (('Add',),): 70.83%
                (('NhwcConv',),): 29.17%
            Level: 2
                (('Add', 'MatMul'),): 41.67%
                (('NhwcConv', 'GroupNorm'),): 29.17%
                (('Add', 'Add'), ('Add', 'Add')): 29.17%
        Op: Sigmoid (29)
            Level: 1
                (('GroupNorm',),): 68.97%
                (('NhwcConv',),): 24.14%
                (('Add',),): 6.90%
            Level: 2
                (('GroupNorm', 'Add'),): 58.62%
                (('NhwcConv', 'Mul'),): 20.69%
                (('GroupNorm', 'NhwcConv'),): 10.34%
                (('Add', 'MatMul'),): 6.90%
                (('NhwcConv', 'Transpose'),): 3.45%
        Op: Sin (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Cast'),): 100.00%
        Op: Split (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Cast'),): 100.00%
        Op: Transpose (15)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'NhwcConv'), ('Mul', 'Split')): 92.31%
                (('Mul', 'Cast'), ('Mul', 'NhwcConv')): 7.69%

The command used for the op_adjacency report was: ``onnx_utils report op_adjacency /path/to/sd1.5/controlnet/optimized.onnx --depth 2``.

vae_decoder
^^^^^^^^^^^

The latest reports (version: vae_decoder_optimized_nhwc_mha_v2.onnx) the VAE_DECODER model are here for reference.

.. collapse:: Op adjacency report

    .. code-block::

        Op: Add (19)
            Level: 1
                (('Add',), ('NhwcConv',)): 52.63%
                (('NhwcConv',), ('NhwcConv',)): 21.05%
                (('MatMul',),): 21.05%
                (('Add',), ('Reshape',)): 5.26%
            Level: 2
                (('Add', 'Add'), ('Add', 'NhwcConv'), ('NhwcConv', 'Mul')): 31.58%
                (('NhwcConv', 'Mul'), ('NhwcConv', 'NhwcConv')): 15.79%
                (('MatMul', 'Reshape'),): 15.79%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('NhwcConv', 'Mul')): 15.79%
                (('MatMul', 'MultiHeadAttention'),): 5.26%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('Reshape', 'Add')): 5.26%
                (('Add', 'Add'), ('Add', 'Reshape'), ('NhwcConv', 'Mul')): 5.26%
                (('NhwcConv', 'Mul'), ('NhwcConv', 'Resize')): 5.26%
        Op: GroupNorm (30)
            Level: 1
                (('NhwcConv',),): 60.00%
                (('Add',),): 40.00%
            Level: 2
                (('NhwcConv', 'Mul'),): 46.67%
                (('Add', 'Add'), ('Add', 'NhwcConv')): 23.33%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv')): 13.33%
                (('NhwcConv', 'Resize'),): 10.00%
                (('NhwcConv', 'NhwcConv'),): 3.33%
                (('Add', 'Add'), ('Add', 'Reshape')): 3.33%
        Op: MatMul (4)
            Level: 1
                (('Reshape',),): 75.00%
                (('MultiHeadAttention',),): 25.00%
            Level: 2
                (('Reshape', 'GroupNorm'),): 75.00%
                (('MultiHeadAttention', 'Add'), ('MultiHeadAttention', 'Add'), ('MultiHeadAttention', 'Add')): 25.00%
        Op: Mul (29)
            Level: 1
                (('GroupNorm',), ('Sigmoid',)): 100.00%
            Level: 2
                (('GroupNorm', 'NhwcConv'), ('Sigmoid', 'GroupNorm')): 62.07%
                (('GroupNorm', 'Add'), ('Sigmoid', 'GroupNorm')): 37.93%
        Op: MultiHeadAttention (1)
            Level: 1
                (('Add',), ('Add',), ('Add',)): 100.00%
            Level: 2
                (('Add', 'MatMul'), ('Add', 'MatMul'), ('Add', 'MatMul')): 100.00%
        Op: NhwcConv (36)
            Level: 1
                (('Mul',),): 80.56%
                (('NhwcConv',),): 8.33%
                (('Resize',),): 8.33%
                (('Transpose',),): 2.78%
            Level: 2
                (('Mul', 'GroupNorm'), ('Mul', 'Sigmoid')): 82.86%
                (('Resize', 'Add'),): 8.57%
                (('NhwcConv', 'Resize'),): 5.71%
                (('NhwcConv', 'Transpose'),): 2.86%
        Op: Reshape (2)
            Level: 1
                (('GroupNorm',),): 50.00%
                (('Add',),): 50.00%
            Level: 2
                (('GroupNorm', 'Add'),): 50.00%
                (('Add', 'MatMul'),): 50.00%
        Op: Resize (3)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'Add'), ('Add', 'NhwcConv')): 100.00%
        Op: Sigmoid (29)
            Level: 1
                (('GroupNorm',),): 100.00%
            Level: 2
                (('GroupNorm', 'NhwcConv'),): 62.07%
                (('GroupNorm', 'Add'),): 37.93%
        Op: Transpose (2)
            Level: 1
                (('NhwcConv',),): 100.00%
            Level: 2
                (('NhwcConv', 'Mul'),): 100.00%

The command used for the op_adjacency report was: ``onnx_utils report op_adjacency /path/to/sd1.5/vae_decoder/optimized.onnx --depth 2``.
