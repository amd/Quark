..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Stable Diffusion 3.0
====================

The standard ONNX model for SD 3.0 is available on `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers>`__.
Parts of SD 3.0 were offloaded to the NPU: T5, MM-DiT, ControlNet, vae_encoder and vae_decoder.

SD 3.0 also provides a rich set of ControlNet plugins, which can be downloaded from the following links:

- `SD3-Controlnet-Canny <https://huggingface.co/InstantX/SD3-Controlnet-Canny>`_
- `SD3-Controlnet-Tile <https://huggingface.co/InstantX/SD3-Controlnet-Tile>`_
- `SD3-Controlnet-Pose <https://huggingface.co/InstantX/SD3-Controlnet-Pose>`_
- `SD3-Controlnet-Depth <https://huggingface.co/InstantX/SD3-Controlnet-Depth>`_

To start, clone the repository or download the *MM-DiT* and *vae_decoder* models.
Large files are stored with git lfs so make sure you download the actual large files and not just lfs placeholder files.
These instructions assume you've cloned the repository to ``/path/to/sd3``.


Preprocess
----------

First, you preprocess the default models.

*MM-DiT*:
^^^^^^^^^

.. code-block:: console

    cd /path/to/sd3/mmdit
    onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize mmdit --save-as-external

.. csv-table:: parameters for preprocessing *MM-DiT*
   :header: "Parameter", "512x512", "1024x1024"
   :widths: 20, 10, 10

   "graphs", 1, 1
   "batch_1", 2, 2
   "w", 64, 128
   "h", 64, 128
   "batch_2", 2, 2
   "batch_3", 2, 2
   "max_length", 154, 154
   "batch_4", 2, 2
   "batch_size", 2, 2
   "state_dim1", 1024, 4096


*vae_decoder*
^^^^^^^^^^^^^

.. code-block:: console

    cd /path/to/sd3/vae_decoder
    onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize vae

.. csv-table:: parameters for preprocessing *vae_decoder*
   :header: "Parameter", "512x512", "1024x1024"
   :widths: 20, 10, 10

   "graphs", 1, 1
   "batch", 1, 1
   "channels", 16, 16
   "height", 64, 128
   "width", 64, 128

*vae_encoder*
^^^^^^^^^^^^^

.. code-block:: console

    cd /path/to/sd3/vae_encoder
    onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize vae

.. csv-table:: parameters for preprocessing *vae_encoder*
   :header: "Parameter", "512x512", "1024x1024"
   :widths: 20, 10, 10

   "graphs", 1, 1
   "batch", 1, 1
   "channels", 3, 3
   "height", 512, 1024
   "width", 512, 1024

*ControlNet*
^^^^^^^^^^^^

.. code-block:: console

    cd /path/to/sd3/controlnet # canny, pose, tile
    onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize sd3_controlnet

.. csv-table:: parameters for preprocessing *controlnet*
   :header: "Parameter", "512x512", "1024x1024"
   :widths: 20, 10, 10

   "graphs", 1, 1
   "batch_size", 2, 2
   "w", 64, 128
   "h", 64, 128
   "max_length", 154, 154

These steps will generate ``optimized.onnx`` models in the respective directories and an external data file for MM-DiT in its directory.

.. attention::

    You will need the ``DmlExecutionProvider`` to preprocess the model.
    To install this, build and install `onnxruntime-directml <https://pypi.org/project/onnxruntime-directml/>`_ in your Python environment.

    You can download pre-optimized models for SD3 from
    `OneDrive <https://amdcloud-my.sharepoint.com/:f:/r/personal/varunsh_amd_com/Documents/Shared/stable_diffusion/sd-3.0>`__.

    For MM-DiT, the pre-optimized models at: `OneDrive <https://amdcloud-my.sharepoint.com/:f:/r/personal/varunsh_amd_com/Documents/Shared/stable_diffusion/sd-3.0/mmdit>`__.

    Including:

    - ``mmdit_512_preprocess.onnx`` and ``mmdit_512_preprocess.onnx_data``
    - ``mmdit_1024_preprocess.onnx`` and ``mmdit_1024_preprocess.onnx_data``

    For vae_decoder, the pre-optimized models at: `OneDrive <https://amdcloud-my.sharepoint.com/:f:/r/personal/varunsh_amd_com/Documents/Shared/stable_diffusion/sd-3.0/vae_decoder>`__.

    Including:

    - ``vae_512_preprocess.onnx``
    - ``vae_1024_fp32_preprocess.onnx`` and ``vae_1024_fp32_preprocess.onnx_data``

    Furthermore, the project also support SD3 ControlNet.

    For ControlNet, the pre-optimized models at: `OneDrive <https://amdcloud-my.sharepoint.com/:f:/r/personal/varunsh_amd_com/Documents/Shared/stable_diffusion/sd-3.0/controlnet>`__.

    ControlNet includes controlnet-canny, controlnet-depth, controlnet-tile, controlnet-pose, and transformer-controlnet;
    the above types all include the following models:

    - ``model_512x512_preprocess.onnx`` and ``model_512x512_preprocess.onnx_data``
    - ``model_1024x1024_preprocess.onnx`` and ``model_1024x1024_preprocess.onnx_data``

    For vae_encoder, which used to encode control image, the pre-optimized models at: `OneDrive <https://amdcloud-my.sharepoint.com/:f:/r/personal/varunsh_amd_com/Documents/Shared/stable_diffusion/sd-3.0/vae_encoder>`__.

    Including:

    - ``vae_encoder_512x512_preprocess.onnx`` and ``vae_encoder_512x512_preprocess.onnx_data``
    - ``vae_encoder_1024x1024_preprocess.onnx`` and ``vae_encoder_1024x1024_preprocess.onnx_data``

Partition
---------

With the optimized models, you can partition them for NPU.

*MM-DiT*:
^^^^^^^^^

.. code-block:: console

    onnx_utils partition "/path/to/sd3/mmdit/optimized.onnx" "/path/to/sd3/mmdit" "sd3_mmdit.yaml" -v

*vae_decoder*:
^^^^^^^^^^^^^^

.. code-block:: console

    onnx_utils partition "/path/to/sd3/vae_decoder/optimized.onnx" "/path/to/sd3/vae_decoder" "sd3_vae.yaml" -v

*controlnet*:
^^^^^^^^^^^^^

.. code-block:: console

    onnx_utils partition "/path/to/sd3/controlnet/optimized.onnx" "/path/to/sd3/controlnet" "sd3_mmdit.yaml" -v

*vae_encoder*:
^^^^^^^^^^^^^^

.. code-block:: console

    onnx_utils partition "/path/to/sd3/vae_encoder/optimized.onnx" "/path/to/sd3/vae_encoder" "sd3_vae.yaml" -v


These steps will generate a ``replaced.onnx`` models in the respective directories and a ``.cache`` directory containing DD metadata files.
SD3 is offloaded to the NPU using kernels from Tianping's team and others.

Run
---

Then to run, you need to `build the custom op library <installation:Custom ops shared library>` and load it in ORT.

For SD3, you need to use the following ORT session configurations:

.. code-block:: python

    import onnxruntime as ort

    decoder_session_options = ort.SessionOptions()
    decoder_session_options.add_session_config_entry("model_name", "SD30_DECODER")
    decoder_session_options.add_session_config_entry("dd_cache", "/path/to/sd3/vae_decoder/.cache")
    decoder_session_options.add_session_config_entry("dd_root", "/path/to/DD/source")
    decoder_session_options.add_session_config_entry("compile_fusion_rt", "True")
    # this should be the last step
    decoder_session_options.register_custom_ops_library("/path/to/shared/library")

    mmdit_session_options = ort.SessionOptions()
    mmdit_session_options.add_session_config_entry("model_name", "SD30_MMDIT")
    mmdit_session_options.add_session_config_entry("dd_cache", "/path/to/sd3/mmdit/.cache")
    mmdit_session_options.add_session_config_entry("dd_root", "/path/to/DD/source")
    mmdit_session_options.add_session_config_entry("compile_fusion_rt", "True")
    # this should be the last step
    mmdit_session_options.register_custom_ops_library("/path/to/shared/library")

    encoder_session_options = ort.SessionOptions()
    encoder_session_options.add_session_config_entry("model_name", "SD30_DECODER")
    encoder_session_options.add_session_config_entry("dd_cache", "/path/to/sd3/vae_encoder/.cache")
    encoder_session_options.add_session_config_entry("dd_root", "/path/to/DD/source")
    encoder_session_options.add_session_config_entry("compile_fusion_rt", "True")
    # this should be the last step
    encoder_session_options.register_custom_ops_library("/path/to/shared/library")


    controlnet_session_options = ort.SessionOptions()
    controlnet_session_options.add_session_config_entry("model_name", "SD30_MMDIT")
    controlnet_session_options.add_session_config_entry("dd_cache", "/path/to/sd3/controlnet/.cache")
    controlnet_session_options.add_session_config_entry("dd_root", "/path/to/DD/source")
    controlnet_session_options.add_session_config_entry("compile_fusion_rt", "True")
    # this should be the last step
    controlnet_session_options.register_custom_ops_library("/path/to/shared/library")
    ...

Reports
-------

The latest reports for this SD 3.0 are here for reference.
Each submodel in SD 3.0 has its own section below.

mmdit(without controlnet)
^^^^^^^^^^^^^^^^^^^^^^^^^

The latest reports for the MMDIT model are here for reference.

.. collapse:: Op adjacency report

    .. code-block::

        Op: Add (863)
            Level: 1
                (('MatMul',),): 66.86%
                (('Add',), ('Mul',)): 16.34%
                (('Mul',), ('Reshape',)): 11.12%
                (('Tanh',),): 5.45%
                (('Reshape',),): 0.12%
                (('Add',), ('Add',)): 0.12%
            Level: 2
                (('MatMul', 'Mul'),): 38.91%
                (('MatMul', 'Add'),): 22.30%
                (('Mul', 'LayerNormalization'), ('Mul', 'Reshape'), ('Reshape', 'Add')): 11.15%
                (('Add', 'Add'), ('Add', 'Mul'), ('Mul', 'Add'), ('Mul', 'Reshape')): 10.69%
                (('MatMul', 'Slice'),): 5.46%
                (('Add', 'MatMul'), ('Mul', 'Mul')): 5.46%
                (('Tanh', 'Mul'),): 5.46%
                (('Reshape', 'NhwcConv'),): 0.12%
                (('MatMul', 'Concat'),): 0.12%
                (('Add', 'MatMul'), ('Add', 'MatMul')): 0.12%
                (('Add', 'Reshape'), ('Mul', 'Add'), ('Mul', 'Reshape')): 0.12%
                (('Add', 'MatMul'), ('Mul', 'Add'), ('Mul', 'Reshape')): 0.12%
        Op: Concat (73)
            Level: 1
                (('Add',), ('Add',)): 98.63%
                (('Cos',), ('Sin',)): 1.37%
            Level: 2
                (('Add', 'MatMul'), ('Add', 'MatMul')): 98.63%
                (('Cos', 'Mul'), ('Sin', 'Mul')): 1.37%
        Op: Cos (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Reshape'),): 100.00%
        Op: LayerNormalization (96)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'Add'), ('Add', 'Mul')): 97.92%
                (('Add', 'MatMul'),): 1.04%
                (('Add', 'Reshape'),): 1.04%
        Op: MatMul (577)
            Level: 1
                (('Mul',),): 58.26%
                (('Add',),): 33.39%
                (('Slice',),): 8.17%
                (('Concat',),): 0.17%
            Level: 2
                (('Mul', 'Add'), ('Mul', 'Sigmoid')): 50.09%
                (('Add', 'Mul'), ('Add', 'Reshape')): 33.39%
                (('Slice', 'MultiHeadAttention'),): 8.17%
                (('Mul', 'Mul'),): 8.17%
                (('Concat', 'Cos'), ('Concat', 'Sin')): 0.17%
        Op: Mul (476)
            Level: 1
                (('LayerNormalization',), ('Reshape',)): 20.17%
                (('Add',), ('Reshape',)): 19.75%
                (('Add',), ('Add',)): 19.75%
                (('Mul',),): 19.75%
                (('Add',), ('Mul',)): 9.87%
                (('Add',),): 9.87%
                (('Add',), ('Sigmoid',)): 0.63%
                (('Reshape',),): 0.21%
            Level: 2
                (('LayerNormalization', 'Add'), ('Reshape', 'Add')): 20.21%
                (('Add', 'MatMul'), ('Reshape', 'Add')): 19.79%
                (('Add', 'MatMul'), ('Add', 'MatMul')): 9.89%
                (('Add', 'MatMul'), ('Mul', 'Add'), ('Mul', 'Add')): 9.89%
                (('Mul', 'Add'), ('Mul', 'Mul')): 9.89%
                (('Add', 'Add'), ('Add', 'Mul')): 9.89%
                (('Add', 'MatMul'), ('Add', 'Tanh')): 9.89%
                (('Mul', 'Add'), ('Mul', 'Add')): 9.89%
                (('Add', 'MatMul'), ('Sigmoid', 'Add')): 0.42%
                (('Add', 'Add'), ('Add', 'Add'), ('Sigmoid', 'Add')): 0.21%
        Op: MultiHeadAttention (24)
            Level: 1
                (('Concat',), ('Concat',), ('Concat',)): 100.00%
            Level: 2
                (('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add')): 100.00%
        Op: NhwcConv (1)
            Level: 1
                (('Transpose',),): 100.00%
        Op: Reshape (290)
            Level: 1
                (('Add',),): 99.31%
                (('NhwcConv',),): 0.35%
                (('Transpose',),): 0.35%
            Level: 2
                (('Add', 'MatMul'),): 99.31%
                (('NhwcConv', 'Transpose'),): 0.35%
                (('Transpose', 'Reshape'),): 0.35%
        Op: Sigmoid (3)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'MatMul'),): 66.67%
                (('Add', 'Add'), ('Add', 'Add')): 33.33%
        Op: Sin (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Reshape'),): 100.00%
        Op: Slice (47)
            Level: 1
                (('MultiHeadAttention',),): 100.00%
            Level: 2
                (('MultiHeadAttention', 'Concat'), ('MultiHeadAttention', 'Concat'), ('MultiHeadAttention', 'Concat')): 100.00%
        Op: Tanh (47)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Add'),): 100.00%
        Op: Transpose (2)
            Level: 1
                (('Reshape',),): 100.00%
            Level: 2
                (('Reshape', 'Add'),): 100.00%


The command used for the op_adjacency report was: ``onnx_utils report op_adjacency /path/to/mmdit/optimized.onnx --depth 2``.

vae_decoder
^^^^^^^^^^^

The latest reports for the VAE_DECODER model are here for reference.

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
                (('MatMul', 'Reshape'),): 15.79%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('NhwcConv', 'Mul')): 15.79%
                (('NhwcConv', 'Mul'), ('NhwcConv', 'NhwcConv')): 10.53%
                (('NhwcConv', 'Mul'), ('NhwcConv', 'Transpose')): 5.26%
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
                (('NhwcConv', 'Transpose'),): 3.33%
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
        Op: NhwcConv (35)
            Level: 1
                (('Mul',),): 82.86%
                (('Resize',),): 8.57%
                (('NhwcConv',),): 5.71%
                (('Transpose',),): 2.86%
            Level: 2
                (('Mul', 'GroupNorm'), ('Mul', 'Sigmoid')): 85.29%
                (('Resize', 'Add'),): 8.82%
                (('NhwcConv', 'Resize'),): 5.88%
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


The command used for the op_adjacency report was: ``onnx_utils report op_adjacency /path/to/vae_decoder/optimized.onnx --depth 2``.

The latest reports for this SD 3.0 ControlNet are here for reference.
ControlNet Canny Tile and Pose share the same model structure.

controlnet-canny/pose/tile
^^^^^^^^^^^^^^^^^^^^^^^^^^

The latest reports for the CONTROLNET_CANNY/CONTROLNET_POSE/CONTROLNET_TILE model are here for reference.

.. collapse:: Op adjacency report


    .. code-block::

        Op: Add (196)
            Level: 1
                (('MatMul',),): 75.51%
                (('Mul',), ('Reshape',)): 11.73%
                (('Add',), ('Mul',)): 11.22%
                (('Reshape',),): 0.51%
                (('Add',), ('Reshape',)): 0.51%
                (('Add',), ('Add',)): 0.51%
            Level: 2
                (('MatMul', 'Mul'),): 36.08%
                (('MatMul', 'Add'),): 27.32%
                (('Mul', 'LayerNormalization'), ('Mul', 'Reshape'), ('Reshape', 'Add')): 11.86%
                (('Add', 'Add'), ('Add', 'Mul'), ('Mul', 'Add'), ('Mul', 'Reshape')): 10.31%
                (('MatMul', 'Slice'),): 5.67%
                (('MatMul', 'Gelu'),): 5.67%
                (('Reshape', 'NhwcConv'),): 0.52%
                (('Add', 'Reshape'), ('Reshape', 'NhwcConv')): 0.52%
                (('MatMul', 'Cast'),): 0.52%
                (('Add', 'MatMul'), ('Add', 'MatMul')): 0.52%
                (('Add', 'Add'), ('Add', 'Reshape'), ('Mul', 'Add'), ('Mul', 'Reshape')): 0.52%
                (('Add', 'MatMul'), ('Mul', 'Add'), ('Mul', 'Reshape')): 0.52%
        Op: Cast (2)
            Level: 1
                (('Reshape',),): 50.00%
                (('Concat',),): 50.00%
            Level: 2
                (('Concat', 'Cos'), ('Concat', 'Sin')): 100.00%
        Op: Concat (19)
            Level: 1
                (('Add',), ('Add',)): 94.74%
                (('Cos',), ('Sin',)): 5.26%
            Level: 2
                (('Add', 'MatMul'), ('Add', 'MatMul')): 94.74%
                (('Cos', 'Mul'), ('Sin', 'Mul')): 5.26%
        Op: Cos (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Cast'),): 100.00%
        Op: Gelu (11)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'MatMul'),): 100.00%
        Op: LayerNormalization (23)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'Add'), ('Add', 'Mul')): 91.30%
                (('Add', 'MatMul'),): 4.35%
                (('Add', 'Add'), ('Add', 'Reshape')): 4.35%
        Op: MatMul (148)
            Level: 1
                (('Mul',),): 47.95%
                (('Add',),): 36.30%
                (('Slice',),): 7.53%
                (('Gelu',),): 7.53%
                (('Cast',),): 0.68%
            Level: 2
                (('Mul', 'Add'), ('Mul', 'Sigmoid')): 47.95%
                (('Add', 'Mul'), ('Add', 'Reshape')): 32.19%
                (('Slice', 'MultiHeadAttention'),): 7.53%
                (('Gelu', 'Add'),): 7.53%
                (('Add', 'Add'), ('Add', 'Mul')): 4.11%
                (('Cast', 'Concat'),): 0.68%
        Op: Mul (55)
            Level: 1
                (('LayerNormalization',), ('Reshape',)): 41.82%
                (('Add',), ('Reshape',)): 40.00%
                (('Add',),): 10.91%
                (('Add',), ('Sigmoid',)): 5.45%
                (('Cast',),): 1.82%
            Level: 2
                (('LayerNormalization', 'Add'), ('Reshape', 'Add')): 41.82%
                (('Add', 'MatMul'), ('Reshape', 'Add')): 40.00%
                (('Add', 'MatMul'),): 10.91%
                (('Add', 'MatMul'), ('Sigmoid', 'Add')): 3.64%
                (('Cast', 'Reshape'),): 1.82%
                (('Add', 'Add'), ('Add', 'Add'), ('Sigmoid', 'Add')): 1.82%
        Op: MultiHeadAttention (6)
            Level: 1
                (('Concat',), ('Concat',), ('Concat',)): 100.00%
            Level: 2
                (('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add')): 100.00%
        Op: NhwcConv (2)
            Level: 1
                (('Transpose',),): 100.00%
        Op: Reshape (71)
            Level: 1
                (('Add',),): 97.14%
                (('NhwcConv',),): 2.86%
            Level: 2
                (('Add', 'MatMul'),): 97.14%
                (('NhwcConv', 'Transpose'),): 2.86%
        Op: Sigmoid (3)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'MatMul'),): 66.67%
                (('Add', 'Add'), ('Add', 'Add')): 33.33%
        Op: Sin (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Cast'),): 100.00%
        Op: Slice (11)
            Level: 1
                (('MultiHeadAttention',),): 100.00%
            Level: 2
                (('MultiHeadAttention', 'Concat'), ('MultiHeadAttention', 'Concat'), ('MultiHeadAttention', 'Concat')): 100.00%
        Op: Transpose (2)

The command used for the op_adjacency report was: ``onnx_utils report op_adjacency /path/to/controlnet/optimized.onnx --depth 2``.

controlnet-depth
^^^^^^^^^^^^^^^^

The latest reports for the CONTROLNET_DEPTH model are here for reference.

.. collapse:: Op adjacency report


    .. code-block::

        Op: Add (394)
            Level: 1
                (('MatMul',),): 75.63%
                (('Mul',), ('Reshape',)): 11.93%
                (('Add',), ('Mul',)): 11.68%
                (('Reshape',),): 0.25%
                (('Add',), ('Reshape',)): 0.25%
                (('Add',), ('Add',)): 0.25%
            Level: 2
                (('MatMul', 'Mul'),): 36.22%
                (('MatMul', 'Add'),): 27.30%
                (('Mul', 'LayerNormalization'), ('Mul', 'Reshape'), ('Reshape', 'Add')): 11.99%
                (('Add', 'Add'), ('Add', 'Mul'), ('Mul', 'Add'), ('Mul', 'Reshape')): 11.22%
                (('MatMul', 'Slice'),): 5.87%
                (('MatMul', 'Gelu'),): 5.87%
                (('Reshape', 'NhwcConv'),): 0.26%
                (('Add', 'Reshape'), ('Reshape', 'NhwcConv')): 0.26%
                (('MatMul', 'Cast'),): 0.26%
                (('Add', 'MatMul'), ('Add', 'MatMul')): 0.26%
                (('Add', 'Add'), ('Add', 'Reshape'), ('Mul', 'Add'), ('Mul', 'Reshape')): 0.26%
                (('Add', 'MatMul'), ('Mul', 'Add'), ('Mul', 'Reshape')): 0.26%
        Op: Cast (2)
            Level: 1
                (('Reshape',),): 50.00%
                (('Concat',),): 50.00%
            Level: 2
                (('Concat', 'Cos'), ('Concat', 'Sin')): 100.00%
        Op: Concat (37)
            Level: 1
                (('Add',), ('Add',)): 97.30%
                (('Cos',), ('Sin',)): 2.70%
            Level: 2
                (('Add', 'MatMul'), ('Add', 'MatMul')): 97.30%
                (('Cos', 'Mul'), ('Sin', 'Mul')): 2.70%
        Op: Cos (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Cast'),): 100.00%
        Op: Gelu (23)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'MatMul'),): 100.00%
        Op: LayerNormalization (47)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'Add'), ('Add', 'Mul')): 95.74%
                (('Add', 'MatMul'),): 2.13%
                (('Add', 'Add'), ('Add', 'Reshape')): 2.13%
        Op: MatMul (298)
            Level: 1
                (('Mul',),): 47.97%
                (('Add',),): 36.15%
                (('Slice',),): 7.77%
                (('Gelu',),): 7.77%
                (('Cast',),): 0.34%
            Level: 2
                (('Mul', 'Add'), ('Mul', 'Sigmoid')): 47.97%
                (('Add', 'Mul'), ('Add', 'Reshape')): 32.09%
                (('Slice', 'MultiHeadAttention'),): 7.77%
                (('Gelu', 'Add'),): 7.77%
                (('Add', 'Add'), ('Add', 'Mul')): 4.05%
                (('Cast', 'Concat'),): 0.34%
        Op: Mul (109)
            Level: 1
                (('LayerNormalization',), ('Reshape',)): 43.12%
                (('Add',), ('Reshape',)): 42.20%
                (('Add',),): 11.01%
                (('Add',), ('Sigmoid',)): 2.75%
                (('Cast',),): 0.92%
            Level: 2
                (('LayerNormalization', 'Add'), ('Reshape', 'Add')): 43.12%
                (('Add', 'MatMul'), ('Reshape', 'Add')): 42.20%
                (('Add', 'MatMul'),): 11.01%
                (('Add', 'MatMul'), ('Sigmoid', 'Add')): 1.83%
                (('Cast', 'Reshape'),): 0.92%
                (('Add', 'Add'), ('Add', 'Add'), ('Sigmoid', 'Add')): 0.92%
        Op: MultiHeadAttention (12)
            Level: 1
                (('Concat',), ('Concat',), ('Concat',)): 100.00%
            Level: 2
                (('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add')): 100.00%
        Op: NhwcConv (2)
            Level: 1
                (('Transpose',),): 100.00%
        Op: Reshape (143)
            Level: 1
                (('Add',),): 98.59%
                (('NhwcConv',),): 1.41%
            Level: 2
                (('Add', 'MatMul'),): 98.59%
                (('NhwcConv', 'Transpose'),): 1.41%
        Op: Sigmoid (3)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'MatMul'),): 66.67%
                (('Add', 'Add'), ('Add', 'Add')): 33.33%
        Op: Sin (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Cast'),): 100.00%
        Op: Slice (23)
            Level: 1
                (('MultiHeadAttention',),): 100.00%
            Level: 2
                (('MultiHeadAttention', 'Concat'), ('MultiHeadAttention', 'Concat'), ('MultiHeadAttention', 'Concat')): 100.00%
        Op: Transpose (2)

The command used for the op_adjacency report was: ``onnx_utils report op_adjacency /path/to/controlnet-depth/optimized.onnx --depth 2``.

transformer-controlnet
^^^^^^^^^^^^^^^^^^^^^^

The latest reports for the TRANSFORMER_CONTROLNET model are here for reference.

.. collapse:: Op adjacency report


    .. code-block::

        Op: Add (792)
            Level: 1
                (('MatMul',),): 72.85%
                (('Mul',), ('Reshape',)): 12.12%
                (('Add',), ('Mul',)): 11.87%
                (('Add',),): 2.90%
                (('Reshape',),): 0.13%
                (('Add',), ('Add',)): 0.13%
            Level: 2
                (('MatMul', 'Mul'),): 36.46%
                (('MatMul', 'Add'),): 24.30%
                (('Mul', 'LayerNormalization'), ('Mul', 'Reshape'), ('Reshape', 'Add')): 12.15%
                (('Add', 'Add'), ('Add', 'Mul'), ('Mul', 'Add'), ('Mul', 'Reshape')): 8.73%
                (('MatMul', 'Slice'),): 5.95%
                (('MatMul', 'Gelu'),): 5.95%
                (('Add', 'Add'), ('Add', 'Mul')): 2.91%
                (('Add', 'Add'), ('Mul', 'Add'), ('Mul', 'Reshape')): 2.91%
                (('Reshape', 'NhwcConv'),): 0.13%
                (('MatMul', 'Cast'),): 0.13%
                (('Add', 'MatMul'), ('Add', 'MatMul')): 0.13%
                (('Add', 'Reshape'), ('Mul', 'Add'), ('Mul', 'Reshape')): 0.13%
                (('Add', 'MatMul'), ('Mul', 'Add'), ('Mul', 'Reshape')): 0.13%
        Op: Cast (2)
            Level: 1
                (('Reshape',),): 50.00%
                (('Concat',),): 50.00%
            Level: 2
                (('Concat', 'Cos'), ('Concat', 'Sin')): 100.00%
        Op: Concat (73)
            Level: 1
                (('Add',), ('Add',)): 98.63%
                (('Cos',), ('Sin',)): 1.37%
            Level: 2
                (('Add', 'MatMul'), ('Add', 'MatMul')): 98.63%
                (('Cos', 'Mul'), ('Sin', 'Mul')): 1.37%
        Op: Cos (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Cast'),): 100.00%
        Op: Gelu (47)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'MatMul'),): 100.00%
        Op: LayerNormalization (96)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'Add'), ('Add', 'Mul')): 73.96%
                (('Add', 'Add'),): 23.96%
                (('Add', 'MatMul'),): 1.04%
                (('Add', 'Reshape'),): 1.04%
        Op: MatMul (577)
            Level: 1
                (('Mul',),): 50.09%
                (('Add',),): 33.39%
                (('Slice',),): 8.17%
                (('Gelu',),): 8.17%
                (('Cast',),): 0.17%
            Level: 2
                (('Mul', 'Add'), ('Mul', 'Sigmoid')): 50.09%
                (('Add', 'Mul'), ('Add', 'Reshape')): 33.39%
                (('Slice', 'MultiHeadAttention'),): 8.17%
                (('Gelu', 'Add'),): 8.17%
                (('Cast', 'Concat'),): 0.17%
        Op: Mul (194)
            Level: 1
                (('LayerNormalization',), ('Reshape',)): 49.48%
                (('Add',), ('Reshape',)): 48.45%
                (('Add',), ('Sigmoid',)): 1.55%
                (('Cast',),): 0.52%
            Level: 2
                (('LayerNormalization', 'Add'), ('Reshape', 'Add')): 49.48%
                (('Add', 'MatMul'), ('Reshape', 'Add')): 48.45%
                (('Add', 'MatMul'), ('Sigmoid', 'Add')): 1.03%
                (('Cast', 'Reshape'),): 0.52%
                (('Add', 'Add'), ('Add', 'Add'), ('Sigmoid', 'Add')): 0.52%
        Op: MultiHeadAttention (24)
            Level: 1
                (('Concat',), ('Concat',), ('Concat',)): 100.00%
            Level: 2
                (('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add'), ('Concat', 'Add')): 100.00%
        Op: NhwcConv (1)
            Level: 1
                (('Transpose',),): 100.00%
        Op: Reshape (290)
            Level: 1
                (('Add',),): 99.31%
                (('NhwcConv',),): 0.35%
                (('Transpose',),): 0.35%
            Level: 2
                (('Add', 'MatMul'),): 99.31%
                (('NhwcConv', 'Transpose'),): 0.35%
                (('Transpose', 'Reshape'),): 0.35%
        Op: Sigmoid (3)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'MatMul'),): 66.67%
                (('Add', 'Add'), ('Add', 'Add')): 33.33%
        Op: Sin (1)
            Level: 1
                (('Mul',),): 100.00%
            Level: 2
                (('Mul', 'Cast'),): 100.00%
        Op: Slice (47)
            Level: 1
                (('MultiHeadAttention',),): 100.00%
            Level: 2
                (('MultiHeadAttention', 'Concat'), ('MultiHeadAttention', 'Concat'), ('MultiHeadAttention', 'Concat')): 100.00%
        Op: Transpose (2)
            Level: 1
                (('Reshape',),): 100.00%
            Level: 2
                (('Reshape', 'Add'),): 100.00%


The command used for the op_adjacency report was: ``onnx_utils report op_adjacency /path/to/transformer_controlnet/optimized.onnx --depth 2``.

vae_encoder
^^^^^^^^^^^

The latest reports for the VAE_ENCODER model are here for reference.

.. collapse:: Op adjacency report


    .. code-block::

            Op: Add (15)
            Level: 1
                (('Add',), ('NhwcConv',)): 40.00%
                (('NhwcConv',), ('NhwcConv',)): 26.67%
                (('MatMul',),): 26.67%
                (('Add',), ('Reshape',)): 6.67%
            Level: 2
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv'), ('NhwcConv', 'Mul')): 26.67%
                (('MatMul', 'Reshape'),): 20.00%
                (('NhwcConv', 'Mul'), ('NhwcConv', 'NhwcConv')): 13.33%
                (('NhwcConv', 'Mul'), ('NhwcConv', 'Transpose')): 6.67%
                (('NhwcConv', 'Mul'), ('NhwcConv', 'Pad')): 6.67%
                (('Add', 'Add'), ('Add', 'NhwcConv'), ('NhwcConv', 'Mul')): 6.67%
                (('MatMul', 'MultiHeadAttention'),): 6.67%
                (('Add', 'Add'), ('Add', 'NhwcConv'), ('Reshape', 'Add')): 6.67%
                (('Add', 'Add'), ('Add', 'Reshape'), ('NhwcConv', 'Mul')): 6.67%
            Op: GroupNorm (22)
            Level: 1
                (('NhwcConv',),): 63.64%
                (('Add',),): 36.36%
            Level: 2
                (('NhwcConv', 'Mul'),): 45.45%
                (('Add', 'NhwcConv'), ('Add', 'NhwcConv')): 18.18%
                (('NhwcConv', 'Pad'),): 13.64%
                (('Add', 'Add'), ('Add', 'NhwcConv')): 13.64%
                (('NhwcConv', 'Transpose'),): 4.55%
                (('Add', 'Add'), ('Add', 'Reshape')): 4.55%
            Op: MatMul (4)
            Level: 1
                (('Reshape',),): 75.00%
                (('MultiHeadAttention',),): 25.00%
            Level: 2
                (('Reshape', 'GroupNorm'),): 75.00%
                (('MultiHeadAttention', 'Add'), ('MultiHeadAttention', 'Add'), ('MultiHeadAttention', 'Add')): 25.00%
            Op: Mul (21)
            Level: 1
                (('GroupNorm',), ('Sigmoid',)): 100.00%
            Level: 2
                (('GroupNorm', 'NhwcConv'), ('Sigmoid', 'GroupNorm')): 66.67%
                (('GroupNorm', 'Add'), ('Sigmoid', 'GroupNorm')): 33.33%
            Op: MultiHeadAttention (1)
            Level: 1
                (('Add',), ('Add',), ('Add',)): 100.00%
            Level: 2
                (('Add', 'MatMul'), ('Add', 'MatMul'), ('Add', 'MatMul')): 100.00%
            Op: NhwcConv (27)
            Level: 1
                (('Mul',),): 77.78%
                (('Pad',),): 11.11%
                (('NhwcConv',),): 7.41%
                (('Transpose',),): 3.70%
            Level: 2
                (('Mul', 'GroupNorm'), ('Mul', 'Sigmoid')): 80.77%
                (('Pad', 'Add'),): 11.54%
                (('NhwcConv', 'Pad'),): 7.69%
            Op: Pad (3)
            Level: 1
                (('Add',),): 100.00%
            Level: 2
                (('Add', 'Add'), ('Add', 'NhwcConv')): 100.00%
            Op: Reshape (2)
            Level: 1
                (('GroupNorm',),): 50.00%
                (('Add',),): 50.00%
            Level: 2
                (('GroupNorm', 'Add'),): 50.00%
                (('Add', 'MatMul'),): 50.00%
            Op: Sigmoid (21)
            Level: 1
                (('GroupNorm',),): 100.00%
            Level: 2
                (('GroupNorm', 'NhwcConv'),): 66.67%
                (('GroupNorm', 'Add'),): 33.33%
            Op: Transpose (2)
            Level: 1
                (('NhwcConv',),): 100.00%
            Level: 2
                (('NhwcConv', 'Mul'),): 100.00%

The command used for the op_adjacency report was: ``onnx_utils report op_adjacency /path/to/vae_encoder/optimized.onnx --depth 2``.
