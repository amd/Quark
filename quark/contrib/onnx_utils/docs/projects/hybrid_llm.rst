..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Hybrid LLM
==========

This project aims to improve the time to first token (TTFT) when running LLMs on devices with an integrated GPU and NPU.
For LLMs, token generation is broken into two phases that run sequentially: pre-fill phase and token phase.
First, the pre-fill phase processes the entire prompt of variable length from the user.
Subsequently, the token phase starts where the same model runs on one token at a time.

Currently, the NPU is not well optimized to run on the small shapes associated with the token phase while the GPU runs it well.
However, the NPU can effectively run the pre-fill phase.
So the project is to implement a solution that can run a given LLM on the NPU for the pre-fill phase and then on GPU for the token phase.

From this initial scope, the project has grown to run LLMs in a variety of configurations (NPU-only, fusion etc.) on different hardware with different capabilities.

This page aims to document how the ``ryzenai_onnx_utils`` repository fits into the project.
For other project details, refer to `Confluence <https://confluence.amd.com/pages/viewpage.action?pageId=1450571260>`__.

Custom Ops
----------

The current strategy for implementing this project is to use :ref:`custom ops <onnxruntime:Custom Ops>`.
These are in ``src/onnx_custom_ops``.
To build the custom ops shared library:

.. code-block:: bash

    # or use your own preset that extends this one
    cmake --preset hybrid-llm --fresh
    # for NPU-only builds, you can use:
    cmake --preset hybrid-llm-npu --fresh
    # for a custom ops DLL with statically linked DD:
    cmake --preset hybrid-llm-static --fresh

    cmake --build --preset release

This builds the shared library in ``build/install/bin``.
By default, this builds with all backends and shared memory enabled.
If you want to change this, set the appropriate :ref:`CMake options <installation:CMake Options>` in your user CMake preset.

Dependencies
^^^^^^^^^^^^

Depending on which backends you enable, you will have additional dependencies.

If you enable the GPU, download the GPU driver from `Artifactory <http://atlartifactory/artifactory/SMT_Virtual/software/amd%20gpu%20driver/24.20.10-240820a-406991c-ati/binaries/>`__, install and restart the machine. Verify that the GPU shows up in task manager.

If you enable NPU, you add a dependency on :term:`DD` and its build dependencies.
In this case, follow the directions in the :ref:`main installation instructions for DD <installation:DynamicDispatch>` and setting up the environment.
For this project, use the git hash in ``cmake/FindDynamicDispatch.cmake`` or point to your own clone of DD with the CMake option.
Using the NPU also adds a dependency on XRT and you may need ``xrt_coreutil.dll`` for an executable to run.
You may also need ``XRT_DIR`` to be set if CMake can't find XRT by itself.

If you enable shared memory, you will bring in the same dependencies as for NPU.

Model processing
----------------

There are multiple flavors of hybrid model execution

* True hybrid: NPU eager prefill/GPU eager token phases
* NPU-only: NPU eager prefill and token phases
* NPU fusion: NPU eager prefill/NPU fusion token phases
* Full fusion: NPU fusion prefill and token phases

Within these flavors, there are also different options to choose from and each uses a different default strategy file for partitioning that defines which passes to run.
The default values for options can vary between strategy files so double-check the defaults as they may differ from what's documented here.

True hybrid
^^^^^^^^^^^

The standard strategy for partitioning a model for this project is ``hybrid_llm.yaml``.
You can also use ``hybrid_llm_experimental.yaml`` to use v2 BFP16 MatMul kernels but not every model may support this.
Your source model should be a DML compatible FP16 model.
To partition a model:

.. code-block:: console

    onnx_utils partition /path/to/model.onnx /path/to/output/dir <strategy.yaml> -v --save-as-external --model-name output_model [--attributes npu_jit=true gpu_jit=true]

This strategy will change all the supported operators in the model to be in the ``com.ryzenai`` domain, perform fusion of blocks to SSMLP and GQO, and add NPU preprocessed weights while keeping the GPU weights.

To explicitly pull out the weights for JIT loading, add ``npu_jit=true`` and/or ``gpu_jit=true`` to the command or explicitly disable by using the value ``false``.
By default, ``hybrid_llm.yaml`` enables NPU JIT but examine the strategy file you're using to confirm the default values.

NPU-only (eager)
^^^^^^^^^^^^^^^^

The standard strategy for partitioning a model for this project is ``hybrid_llm_npu.yaml``.
Your source model should be an FP16 or FP32 model.
To partition a model:

.. code-block:: console

    onnx_utils partition /path/to/model.onnx /path/to/output/dir hybrid_llm_npu.yaml -v --save-as-external --model-name output_model [--attributes npu_jit=true]

This strategy will change all the supported operators in the model to be in the ``com.ryzenai`` domain, perform fusion of blocks to SSMLP and GQO, and add NPU preprocessed weights and remove the GPU weights.

To explicitly pull out the weights for JIT loading, add ``npu_jit=true`` to the command or explicitly disable by using the value ``false``.
By default, ``hybrid_llm_npu.yaml`` enables NPU JIT but examine the strategy file you're using to confirm the default values.

To run this model, you will need a custom ops DLL that has GPU disabled so that both prefill and token phase remain on NPU.

Token Fusion
^^^^^^^^^^^^

Token fusion is a more complex model to prepare.
Your source model should be a model with FP16 or FP32 activations and quantized weights.
First, you need to fix the dynamic shapes for this model to the ``max_seq_len`` of interest.
The ``max_seq_len`` is used to size the KV cache and is typically a power of 2 \<= 4096 as of writing.
Look in the model for the dynamic shapes ``total_sequence_length`` and ``past_sequence_length``.
These are the values you're fixing with the value you choose for ``max_seq_len``.

.. code-block:: console

    cd /path/to/model_dir
    onnx_utils preprocess ./source_model.onnx ./fixed.onnx --save-as-external

This will prompt you for values for the dynamic shapes.
For LLMs, this may look like:

.. code-block:: console

    Enter fixed value(s) for graphs: 1
    Enter fixed value(s) for batch_size: 1
    Enter fixed value(s) for sequence_length: 1
    Enter fixed value(s) for total_sequence_length: <max_seq_len>
    Enter fixed value(s) for past_sequence_length: <max_seq_len>

The ``sequence_length`` should be set to 1 because it's fixed to 1 in token phase.

First partition the original model for eager NPU prefill:

.. code-block:: console

    onnx_utils partition /path/to/model.onnx /path/to/output/dir hybrid_llm_npu.yaml -v --save-as-external --model-name prefill [--attributes npu_jit=true]

To explicitly pull out the weights for JIT loading, add ``npu_jit=true`` to the command or explicitly disable by using the value ``false``.
By default, ``hybrid_llm_npu.yaml`` enables NPU JIT but examine the strategy file you're using to confirm the default values.

Then, using the fixed shape model, partition it for NPU token fusion:

.. code-block:: console

    onnx_utils partition /path/to/fixed.onnx /path/to/output/dir hybrid_llm_fusion.yaml -v --save-as-external --model-name token

This step will produce the DD cache directory in ``/path/to/output/dir/cache``.
Save the contents of this directory as it will be needed at runtime.

Post-processing is required to combine the prefill and token phase models into one.

.. code-block:: console

    onnx_utils postprocess combine_llm --input-path /path/to/prefill.onnx /path/to/prefill.onnx /path/to/token.onnx --output-path /path/to/fusion.onnx

The model files you need to run are:

* ``fusion.onnx``
* ``fusion.onnx.data``
* ``prefill.pb.bin``
* ``prefill.bin`` (if NPU JIT was enabled)

A minimal example of the genai_config.json session options for this is shown below:

.. code-block::

    "session_options": {
        "log_id": "onnxruntime-genai",
        "custom_ops_library": "/path/to/onnx_custom_ops.dll",
        "external_data_file": "prefill.pb.bin",
        "custom_allocator": "ryzen_mm",
        "amd_options": {
            "dd_cache": "/path/to/cache",
            "model_name": "hybrid",
            "dd_root": "/path/to/dd",
            "compile_fusion_rt": "1",
            "hybrid_opt_token_backend": "npu"
        },
        "provider_options": []
    }

Prefill Fusion and Full Fusion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prefill fusion is the most complex model to prepare due to the potential ways of combining models for prefill and token phases.

As for token fusion, you first need fixed shape models for the prefill.
You need to fix the dynamic shapes for this model to the ``max_seq_len`` of interest.
The ``max_seq_len`` is used to size the KV cache and is typically a power of 2 \<= 4096 as of writing.
Look in the model for the dynamic shapes ``total_sequence_length`` and ``past_sequence_length``.
These are the values you're fixing with the value you choose for ``max_seq_len``.
There are two cases here:

1. Using a single fixed prompt size
2. Using a set of fixed prompt sizes

.. code-block:: shell

    cd /model/dir
    # Option 1: if building for a fixed prompt size
    onnx_utils preprocess ./model.onnx ./model-<max_seq_len>-prompt-<prompt_len>.onnx --save-as-external
    Enter fixed value(s) for graphs: 1
    Enter fixed value(s) for batch_size: 1
    Enter fixed value(s) for sequence_length: <prompt_len>
    Enter fixed value(s) for total_sequence_length: <max_seq_len>
    Enter fixed value(s) for past_sequence_length: <max_seq_len>
    # Option 2: if building for multiple fixed prompt sizes
    onnx_utils preprocess ./model.onnx ./model-<max_seq_len>-prompt-dynamic.onnx --save-as-external
    Enter fixed value(s) for graphs: 1
    Enter fixed value(s) for batch_size: 1
    Enter fixed value(s) for sequence_length: [just press enter]
    Enter fixed value(s) for total_sequence_length: <max_seq_len>
    Enter fixed value(s) for past_sequence_length: <max_seq_len>

For full fusion, you'll also need a fixed shape model for token phase.
If you're using eager for token phase, you don't need this step.

.. code-block:: shell

    onnx_utils preprocess ./model.onnx ./model-<max_seq_len>-prompt-1.onnx --save-as-external
    Enter fixed value(s) for graphs: 1
    Enter fixed value(s) for batch_size: 1
    Enter fixed value(s) for sequence_length: 1
    Enter fixed value(s) for total_sequence_length: <max_seq_len>
    Enter fixed value(s) for past_sequence_length: <max_seq_len>

For prefill and full fusion, you need a reference model that's used to define the inputs and outputs since the partitioned models may have different values.

.. code-block:: shell

    # Option 1: if using GPU eager in token phase
    onnx_utils partition "/model/dir/model.onnx" ./tmp hybrid_llm.yaml -v --force --model-name reference --save-as-external
    # Option 2: if using GPU eager in token phase with experimental settings
    onnx_utils partition "/model/dir/model.onnx" ./tmp hybrid_llm_experimental.yaml -v --force --model-name reference --save-as-external
    # Option 3: otherwise
    onnx_utils partition "/model/dir/model.onnx" ./tmp hybrid_llm_npu.yaml -v --force --model-name reference --save-as-external

Partition the prefill phase model:

.. code-block:: shell

    # Option 1: if using a single fixed prompt
    onnx_utils partition "/path/to/model-<max_seq_len>-prompt-<prompt_len>.onnx" ./tmp hybrid_llm_prefill_fusion.yaml -v --force --model-name prefill_fusion --attributes dynamic_shape_list="[]" --save-as-external
    # Option 2: if using a set of fixed prompts
    onnx_utils partition "/path/to/model-<max_seq_len>-prompt-dynamic.onnx" ./tmp hybrid_llm_prefill_fusion.yaml -v --force --model-name prefill_fusion --save-as-external

    mv ./tmp/.cache ./tmp/.cache_backup

Partition the token phase model:

.. code-block:: shell

    # Option 1: for token fusion
    onnx_utils partition "/path/to/model-<max_seq_len>-prompt-1.onnx" ./tmp hybrid_llm_token_fusion.yaml -v --force --model-name token_fusion --save-as-external
    # Option 2: for eager NPU
    onnx_utils partition "/path/to/model.onnx" ./tmp hybrid_llm_npu.yaml -v --force --model-name token_fusion --save-as-external
    # Option 3: for eager GPU
    onnx_utils partition "/path/to/model.onnx" ./tmp hybrid_llm.yaml -v --force --model-name token_fusion --save-as-external
    # Option 4: for eager GPU with experimental settings
    onnx_utils partition "/path/to/model.onnx" ./tmp hybrid_llm_experimental.yaml -v --force --model-name token_fusion --save-as-external

Finally, combine the two ``.cache`` directories for prefill and token models and postprocess to combine:

.. code-block:: shell

    cd ./tmp

    # merge two .cache directories to one
    cp .cache_backup/* .cache

    onnx_utils postprocess combine_llm --input-path "./reference.onnx" "./prefill_fusion.onnx" "./token_fusion.onnx" --output-path ./fusion.onnx

The model files you need to run are:

* ``fusion.onnx``
* ``fusion.onnx.data``
* ``token_fusion.pb.bin`` (if token phase is eager)
* ``token_fusion.bin`` (if token phase is eager and NPU JIT was enabled)
* ``cache`` directory

A minimal example of the genai_config.json session options for this is shown below:

.. code-block::

    "session_options": {
        "log_id": "onnxruntime-genai",
        "custom_ops_library": "/path/to/onnx_custom_ops.dll",
        "external_data_file": "prefill.pb.bin", // not needed for full fusion
        "custom_allocator": "ryzen_mm",
        "amd_options": {
            "dd_cache": "/path/to/cache",
            "model_name": "hybrid",
            "dd_root": "/path/to/dd",
            "compile_fusion_rt": "1",
            "hybrid_opt_token_backend": "npu" // set depending on what token is running
        },
        "provider_options": []
    }

Execution
---------

The output model (``.onnx``), its external data(s) (``.onnx.data``), header (``*.pb.bin``) and any JIT weights (``*.bin``) should be moved to an existing directory that has the ORT GenAI files (e.g. tokenizer, ``genai_config.json`` etc.).
Then update the ``genai_config.json`` session options depending on what you want to enable.
For any optional arguments, the default values are shown.

.. code-block:: python

    "session_options": {
        # this path depends on where your install of the custom operators is
        "custom_ops_library": "/absolute/path/to/onnx_custom_ops.dll",
        # optional - set to this value to "ryzen_mm" to use a custom allocator
        # for lower memory consumption
        "custom_allocator": "",
        # Set to the proto file. This file, and optionally JIT weights if used,
        # must be in the same directory as the model
        "external_data_file": "<model>.pb.bin",
        "amd_options": {
            # required for fusion - set the path to where the DD files required
            # for fusion are located
            "dd_cache": "/path/to/DD_cache/cache",
            # required for fusion - set the name of the model type for fusion.
            # For newer prefill and token fusion LLM models, it's not needed
            "model_name": "hybrid",
            # optional for fusion - used to set a key for identifying const DD
            # tensors
            "onnx_custom_ops_const_key": "",
            # required for fusion - set the path to DD. Strictly, this is used
            # only for xclbin lookup so the structure should match DD:
            # xclbin/stx/*
            "dd_root": "C:/Users/varunsh/Documents/DynamicDispatch",
            # optional - for fusion, set to 1 to enable compilation. Must be
            # set the first time to compile the metastate from the const file.
            # The compiled metastate can be used in the future
            "compile_fusion_rt": "",
            # optional for fusion: set to 1 to skip copying the KV cache between
            # prefill and token phases if both are fused. Turning this on
            # otherwise will result in bad output
            "fusion_opt_skip_ext_buf_copy": "",
            # optional for fusion: set to a value to set the XRT BO stack size.
            # Defaults to 30 for prefill fusion and default DD value otherwise.
            "fusion_opt_stack_size": "",
            # optional - free memory after prefill phase. This lowers the peak
            # memory at the cost of TTFT for the next prompt.
            "hybrid_opt_free_after_prefill": "0",
            # optional - enable GPU JIT to delay loading GPU weights until the
            # first token phase token. Can lower peak memory. Values range from
            # 0 (no GPU JIT) to 5 (delay all GPU weights to token phase)
            "hybrid_opt_gpu_jit": "0",
            # optional - control when to reallocate NPU JIT buffers. 1.0 means
            # the new buffer size must 100% match the existing one. A lesser
            # factor means if the new buffer size is <x% smaller, don't
            # reallocate. Larger buffer size requests always reallocate.
            "hybrid_opt_dynamic_jit_factor": "1.0",
            # optional - if using a model from memory, pass a pointer to it
            "external_data_blob": "",
            # optional - if using a model from memory, pass the size in bytes
            "external_data_blob_size": "",
            # optional - set to 1 to set the NPU DPM state to high performance
            # during prefill and reset in token phase
            "hybrid_opt_enable_dynamic_dpm": "0",
            # optional - used for debugging, this will ignore NPU exceptions,
            # waits and other errors to run through. Can be used to evaluate
            # which transaction bins are missing
            "hybrid_opt_continue_on_exception" = "0",
            # optional - see MatMulNBits for more detail
            "hybrid_opt_execution_mode" = "",
            # optional - set how far to read ahead for NPU JIT weights. Must be
            # >=2. Any number larger than the number of layers will result in
            # extra memory being used and all NPU weights being loaded in memory
            "hybrid_opt_npu_read_ahead": "3",
            # optional - enable preemption and elf flow
            "hybrid_opt_enable_npu_preemption" = "0",
            # optional - enable qos, requires preemption enabled
            "hybrid_opt_enable_npu_qos" = "0",
            # optional - if using a non-default PDI in xclbin, pass the PDI name
            "hybrid_opt_npu_pdi_name" = "",
            # optional - used to explicitly control whether AIE RoPE is used.
            # By default, AIE RoPE is used if the model supports it
            "hybrid_dbg_use_aie_rope" = ""
            # optional - used to explicitly control whether AIE GQA is used.
            # By default, AIE GQA is used
            "hybrid_dbg_use_aie_gqa" = ""
            # optional - controls where the token phase executes. Defaults to
            # By default, AIE FlashMHA is used if the model supports it
            "hybrid_dbg_use_flash_mha" = ""
            # optional - used to explicitly control whether AIE FlashMHA is used.
            # GPU if it's present.
            "hybrid_opt_token_backend" = "gpu",
            # optional - controls the NPU max seq len which is used to size things
            # like MAX_M for the NPU
            "hybrid_opt_max_seq_length" = "4096",
            # optional - if the embedding Gather has been replaced by a custom
            # op, then this option will enable using a mmap for it to save
            # memory for some performance penalty
            "hybrid_opt_embedding_mmap": "0"
        },
        # use CPU EP, not DML EP
        "provider_options": []
    }

Also remember to update the model name to the new one.
Then, run as usual.


Testing
-------

As we build up more operators, you will probably want to test on synthetic models to start.
One such model is available on `OneDrive <LinkSingleMatMul_>`_.
This is an ONNX model with a single ``MatMul`` operator in the ``com.ryzenai`` domain with input/output data and a ``run.py`` script.
The original ONNX model with the ``MatMul`` in the original domain is also in this archive.
To run this example, you can update the ``CUSTOM_OPS_PATH`` variable in ``run.py`` to the path to your build shared library and then executing the script.
This will invoke the ``MatMul`` operator from the shared library.

You can extract your own models using :ref:`onnx_utils extract <extract:Extract>` and partitioning them as above to run them for this project.

See the links below for other models of interest.

Troubleshooting
---------------

If you encounter issues during partitioning or execution, use this section first to see if there's already a known solution.

Partitioning
^^^^^^^^^^^^

* Bad allocation error or another cryptic error during NPU weight preprocessing: try adding ``--passes-per-iteration 2`` to the partitioning command or freeing some memory from your system.
* Random protobuf errors during model saving: this could be a memory issue. Try closing windows and freeing memory before trying again
* ``from ._DynamicDispatch import *; ImportError: DLL load failed while importing _DynamicDispatch: The specified module could not be found.``: if the DD wheel is built as a dynamic library, you may have dependencies that are not met in your env. As of writing, you need at least ``libprotobuf.dll``, ``fmt.dll``, and ``spdlog.dll`` available in your env and there may be other dependencies as well. To work around this problem, you can either resolve the dependencies (use `Process Monitor <https://learn.microsoft.com/en-us/sysinternals/downloads/procmon>`__ to find the missing DLLs) or build/get a static version of the wheel.

Execution
^^^^^^^^^

* Errors from XRT about not enough video memory: you may get this if the model is large. In this case, you need to use JIT to reduce how many tensors get allocated
* ``Unsupported model IR version: x, max supported IR version: y``: this indicates a mismatch in the Python ORT/ONNX version you used to partition the model and the C++ ORT/ONNX version you're using to execute. Make sure your versions match up or at least are compatible with each other. The ORT vs ONNX version compatibility chart is `here <https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support>`__.

Links
-----

* `Confluence <https://confluence.amd.com/pages/viewpage.action?pageId=1450571260>`__ - overall project planning
* `ONNXRuntime Gen-AI <https://github.com/microsoft/onnxruntime-genai>`__ - OGA used for running LLMs
* Models
    * `Single MatMul <LinkSingleMatMul_>`_
    * `Llama-3-8B-int4-128 <https://amdcloud-my.sharepoint.com/:f:/g/personal/vkjain_amd_com/Ej1F6juClQ5NteJ0NVJWgsQBvYUIbSW5cctUpCH2vZpmhg?e=TLrAHr>`__
    * `Llama3 FP16 <https://amdcloud-my.sharepoint.com/:u:/r/personal/anilm_amd_com/Documents/share/LLM/llama3_quant_blocksize_128_asym.zip?csf=1&web=1&e=nyHIO3>`__
    * `Llama3_quant_blocksize_128_asym_npu_fused <https://amdcloud-my.sharepoint.com/:u:/r/personal/anilm_amd_com/Documents/share/LLM/llama3_quant_blocksize_128_asym_npu_fused.zip?csf=1&web=1&e=snIQLO>`__

.. _LinkSingleMatMul: https://amdcloud-my.sharepoint.com/:u:/g/personal/varunsh_amd_com/EYn8UconNPFBsysnaMOSbMgBlHuH0hVkO0pCzskmy7sxvw
