.. Copyright (C) 2025 - 2026, Advanced Micro Devices, Inc. All rights reserved.

Quantizing Diffusion Models with Quark
======================================

This guide walks through end-to-end quantization of diffusion model
submodules (UNet, transformer, etc.) using AMD Quark.  The examples
below have been validated on SDXL, SD3, and Flux.1-dev.

For most diffusion pipelines, the bulk of the compute is in a single
heavy submodule (``pipe.unet`` for UNet-based pipelines like SDXL, or
``pipe.transformer`` for transformer-based pipelines like SD3 and
Flux), so quantization targets that submodule directly.  Quark's
``get_calib_dataloader`` (in ``quark.torch.utils.diffusers``) hooks
into the pipeline run, captures the submodule's inputs as a
``dict`` keyed by the forward parameter names, and packages them into a
``DataLoader`` that ``ModelQuantizer.quantize_model`` consumes via
``model(**data)`` automatically.

Prerequisites
-------------

.. code-block:: bash

   pip install diffusers transformers accelerate

Pattern
-------

Every diffusion model quantization follows the same two steps:

.. code-block:: python

   from quark.torch.utils.diffusers import get_calib_dataloader

   # Step 1: collect calibration data by running the pipeline.
   dataloader = get_calib_dataloader(pipe, pipe.unet, prompts, n_steps=20, ...)

   # Step 2: quantize (same as LLMs -- ModelQuantizer + QConfig).
   pipe.unet = ModelQuantizer(qconfig).quantize_model(pipe.unet, dataloader)

``get_calib_dataloader`` runs the pipeline, captures the target
submodule's inputs as a dict keyed by the ``forward`` parameter names,
and returns a ``DataLoader``.  ``ModelQuantizer.quantize_model``
consumes those dicts via ``model(**data)`` automatically -- no wrapping
is required.

Example 1: SDXL -- INT8 weight-only
-----------------------------------

Observer-based INT8 quantization on the SDXL UNet.

.. code-block:: python

   import torch
   from diffusers import DiffusionPipeline
   from quark.torch import ModelQuantizer
   from quark.torch.quantization.config.config import Int8PerTensorSpec, QConfig, QLayerConfig
   from quark.torch.utils.diffusers import get_calib_dataloader

   pipe = DiffusionPipeline.from_pretrained(
       "stabilityai/stable-diffusion-xl-base-1.0",
       torch_dtype=torch.float16, variant="fp16",
   )
   pipe.to("cuda")

   prompts = [
       "A serene lake reflecting mountains at sunset",
       "A futuristic city with flying cars at night",
       "A close-up portrait with dramatic lighting",
   ]

   dataloader = get_calib_dataloader(pipe, pipe.unet, prompts, n_steps=20, guidance_scale=8.0)

   weight_spec = Int8PerTensorSpec(
       observer_method="min_max", symmetric=True, scale_type="float",
       round_method="half_even", is_dynamic=False,
   ).to_quantization_spec()
   qconfig = QConfig(global_quant_config=QLayerConfig(weight=weight_spec))

   pipe.unet = ModelQuantizer(qconfig).quantize_model(pipe.unet, dataloader)

   image = pipe("A cat on a windowsill", num_inference_steps=30, guidance_scale=8.0).images[0]
   image.save("sdxl_int8.png")

Example 2: SD3 -- SVDQuant w4a4
-------------------------------

SVDQuant decomposes weights via SVD, adds a low-rank correction branch,
and smooths activations.  Here both weights and activations are
quantized to INT4 (``w4a4``).

.. code-block:: python

   import torch
   from diffusers import StableDiffusion3Pipeline
   from quark.torch import ModelQuantizer
   from quark.torch.quantization.config.config import QConfig, SVDQuantConfig
   from quark.torch.algorithm.svdquant import build_quant_layer_config
   from quark.torch.utils.diffusers import get_calib_dataloader

   pipe = StableDiffusion3Pipeline.from_pretrained(
       "stabilityai/stable-diffusion-3-medium-diffusers",
       torch_dtype=torch.float16,
   )
   pipe.to("cuda")

   prompts = [
       "A serene lake reflecting mountains at sunset",
       "A futuristic city with flying cars at night",
       "A close-up portrait with dramatic lighting",
       "A golden retriever playing in autumn leaves",
       "An astronaut floating above Earth",
   ]

   dataloader = get_calib_dataloader(pipe, pipe.transformer, prompts, n_steps=20)

   qconfig = QConfig(
       global_quant_config=build_quant_layer_config("w4a4"),
       exclude=[
           "*time_text_embed*", "*context_embedder*", "*pos_embed*",
           "*norm_out*", "*proj_out*", "*correction*",
       ],
       algo_config=[SVDQuantConfig(
           svd_rank=32,
           search_alpha=False,
           exclude_patterns=[
               "*time_text_embed*", "*context_embedder*",
               "*pos_embed*", "*norm_out*", "*proj_out*",
           ],
       )],
   )

   pipe.transformer = ModelQuantizer(qconfig).quantize_model(pipe.transformer, dataloader)

   image = pipe("A cat on a windowsill", num_inference_steps=30).images[0]
   image.save("sd3_svdquant_w4a4.png")

Example 3: SDXL -- INT8 weight + activation (w8a8)
--------------------------------------------------

Static INT8 quantization for both weights and activations.  Uses a
min/max observer to collect activation ranges during calibration.

.. code-block:: python

   import torch
   from diffusers import DiffusionPipeline
   from quark.torch import ModelQuantizer
   from quark.torch.quantization.config.config import Int8PerTensorSpec, QConfig, QLayerConfig
   from quark.torch.utils.diffusers import get_calib_dataloader

   pipe = DiffusionPipeline.from_pretrained(
       "stabilityai/stable-diffusion-xl-base-1.0",
       torch_dtype=torch.float16, variant="fp16",
   )
   pipe.to("cuda")

   prompts = [
       "A serene lake reflecting mountains at sunset",
       "A futuristic city with flying cars at night",
       "A close-up portrait with dramatic lighting",
       "A golden retriever playing in autumn leaves",
       "An astronaut floating above Earth",
   ]

   dataloader = get_calib_dataloader(pipe, pipe.unet, prompts, n_steps=20, guidance_scale=8.0)

   int8_spec = Int8PerTensorSpec(
       observer_method="min_max", symmetric=True, scale_type="float",
       round_method="half_even", is_dynamic=False,
   ).to_quantization_spec()
   qconfig = QConfig(global_quant_config=QLayerConfig(weight=int8_spec, input_tensors=int8_spec))

   pipe.unet = ModelQuantizer(qconfig).quantize_model(pipe.unet, dataloader)

   image = pipe("A cat on a windowsill", num_inference_steps=30, guidance_scale=8.0).images[0]
   image.save("sdxl_int8_w8a8.png")

Example 4: Flux.1-dev -- SVDQuant w4a16
---------------------------------------

Flux uses bfloat16 and a different transformer architecture.  Note the
Flux-specific ``pipe_kwargs`` and exclude patterns.

.. code-block:: python

   import torch
   from diffusers import FluxPipeline
   from quark.torch import ModelQuantizer
   from quark.torch.quantization.config.config import QConfig, SVDQuantConfig
   from quark.torch.algorithm.svdquant import build_quant_layer_config
   from quark.torch.utils.diffusers import get_calib_dataloader

   pipe = FluxPipeline.from_pretrained(
       "black-forest-labs/FLUX.1-dev",
       torch_dtype=torch.bfloat16,
       device_map="balanced",
   )

   prompts = [
       "A serene lake reflecting mountains at sunset",
       "A futuristic city with flying cars at night",
       "A close-up portrait with dramatic lighting",
       "A golden retriever playing in autumn leaves",
       "An astronaut floating above Earth",
   ]

   dataloader = get_calib_dataloader(
       pipe, pipe.transformer, prompts, n_steps=20,
       height=1024, width=1024, guidance_scale=3.5, max_sequence_length=512,
   )

   qconfig = QConfig(
       global_quant_config=build_quant_layer_config("w4a16"),
       exclude=[
           "*x_embedder*", "*context_embedder*", "*time_text_embed*",
           "*norm_out*", "*proj_out*", "*correction*",
       ],
       algo_config=[SVDQuantConfig(
           svd_rank=32,
           search_alpha=False,
           exclude_patterns=[
               "*x_embedder*", "*context_embedder*", "*time_text_embed*",
               "*norm_out*", "*proj_out*", "*norm1.linear*", "*norm1_context.linear*",
           ],
       )],
   )

   pipe.transformer = ModelQuantizer(qconfig).quantize_model(pipe.transformer, dataloader)

   image = pipe(
       "A cat on a windowsill", num_inference_steps=50,
       height=1024, width=1024, guidance_scale=3.5, max_sequence_length=512,
   ).images[0]
   image.save("flux_svdquant_w4a16.png")

Example 5: Wan2.2-TI2V-5B -- FP8 text-to-video
----------------------------------------------

Wan2.2 is a text-to-video pipeline.  The heavy submodule is
``pipe.transformer`` (a ``WanTransformer3DModel``); TI2V-5B is
single-transformer (``transformer_2`` is ``None``).  The pipeline runs
in bfloat16 and returns video frames (``result.frames``) in ``[0, 1]``
float, so scale by 255 before writing a video -- and write it with the
bundled ``video_io.save_video`` helper rather than calling
``diffusers.utils.export_to_video`` on the resulting ``uint8`` array:
``export_to_video`` rescales a ``list[np.ndarray]`` by 255 on the
assumption that it is still float, which wraps modulo 256 and
colour-inverts every frame.  ``save_video`` converts to PIL first, which
skips that branch.  This example uses FP8 E4M3 weight + activation,
which needs a short calibration pass.

.. code-block:: python

   import numpy as np
   import torch
   from diffusers import WanPipeline
   from video_io import save_video  # examples/torch/diffusers/video_io.py
   from quark.torch import ModelQuantizer
   from quark.torch.quantization import FP8E4M3PerTensorSpec
   from quark.torch.quantization.config.config import QConfig, QLayerConfig
   from quark.torch.utils.diffusers import get_calib_dataloader

   pipe = WanPipeline.from_pretrained(
       "Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch_dtype=torch.bfloat16,
   ).to("cuda")

   prompts = [
       "A serene lake reflecting mountains at sunset, gentle ripples on the water",
       "A futuristic city with flying cars at night, neon lights",
   ]

   # Capture calibration at a small res / frame count to bound activation memory.
   dataloader = get_calib_dataloader(
       pipe, pipe.transformer, prompts, n_steps=20,
       height=480, width=832, num_frames=17, guidance_scale=5.0,
   )

   fp8_spec = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()
   qconfig = QConfig(
       global_quant_config=QLayerConfig(weight=fp8_spec, input_tensors=fp8_spec),
       exclude=[
           "*patch_embedding*", "*condition_embedder*", "*time_embedder*",
           "*norm_out*", "*proj_out*",
       ],
   )

   pipe.transformer = ModelQuantizer(qconfig).quantize_model(pipe.transformer, dataloader)

   result = pipe(
       "A golden retriever running through a field of autumn leaves",
       height=480, width=832, num_frames=17, num_inference_steps=20, guidance_scale=5.0,
   )
   frames = (np.clip(np.asarray(result.frames[0]), 0.0, 1.0) * 255).round().astype(np.uint8)
   save_video(frames, "wan22_5b_fp8.mp4", fps=16)

The bundled ``quantize_diffusers.py`` wires this end to end (it
auto-detects a Wan checkpoint and runs a generate-and-non-black smoke
check in place of the image-only COCO CLIP/FID harness):

.. code-block:: shell

   python quantize_diffusers.py \
       --model_id Wan-AI/Wan2.2-TI2V-5B-Diffusers \
       --quant_scheme w_fp8_a_fp8 \
       --n_steps 20 --height 480 --width 832 --frames 17 \
       --test --test_size 1

Example 6: Wan2.2-A14B -- packed MXFP4 w4a8 export + native reload (video)
--------------------------------------------------------------------------

Wan2.2-T2V-A14B is a **dual-expert** text-to-video pipeline (two
``WanTransformer3DModel`` submodules, ``transformer`` and ``transformer_2``).
This example quantizes each expert to **w4a8** (per-1x32 MXFP4 weight + static
FP8-e4m3 activation), exports **packed** ``float4_e2m1fn_x2`` weights (~1/4 the
bf16 size) that ``WanPipeline.from_pretrained`` reloads directly into packed
``QParamsLinear``, and runs the reloaded model on the **native FlyDSL / aiter
w4a8** kernels (the packed weight is re-shuffled into the kernel layout at load).

The full, runnable workflow lives in
``examples/torch/diffusers/wan14b_w4a8/`` (``export_wan14b_w4a8.py`` /
``reload_wan14b_w4a8.py`` / ``svd_correction.py``, plus the shared
``examples/torch/diffusers/video_io.py``); see its ``README.md``.

.. code-block:: shell

   cd examples/torch/diffusers/wan14b_w4a8
   export PYTHONPATH="<quark-repo>" HIP_VISIBLE_DEVICES=0

   # quantize both experts + export packed fp4
   python3 export_wan14b_w4a8.py --model /path/to/Wan2.2-T2V-A14B-Diffusers \
       --out ./wan14b_w4a8_export --n_calib 4          # add --svd for SVDQuant

   # reload via from_pretrained -> native w4a8 -> generate a video
   python3 reload_wan14b_w4a8.py --export ./wan14b_w4a8_export \
       --native_linear_mode mxfp4      # or flydsl_a8w4 / flydsl_svdquant / none

Notes specific to Wan2.2:

* **Packed real-quantized export must be requested explicitly** with
  ``weight_format="real_quantized"`` -- diffusers models default to
  ``fake_quantized``, since a packed checkpoint is only readable by a Quark that
  has the packed reload path. With it set (as ``export_wan14b_w4a8.py`` does) the
  on-disk ``.weight`` is packed fp4, and ``config.json`` records
  ``quant_method=quark`` so ``from_pretrained`` auto-dispatches to the Quark
  quantizer and rebuilds packed ``QParamsLinear``.
* **SVDQuant low-rank** is not part of the diffusers module tree, so with
  ``--svd`` it is saved to a side ``svd_correction.safetensors`` and re-attached
  after load (``svd_correction.py``), then converted to the FlyDSL SVDQuant
  kernel. Such an export stores the *residual* ``W - l2 @ l1``, so it also records
  ``svdquant_correction_file: svd_correction.safetensors`` in ``config.json``; the
  reload refuses a checkpoint naming a correction file that is not there, rather
  than quietly serving an uncorrected model.
* Requires gfx950 for the native kernels and a diffusers build with the Quark
  auto-quantizer.

Quick reference
---------------

Which submodule to quantize
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30

   * - Pipeline
     - Target submodule
   * - SDXL, SD 1.5, SD 2.1
     - ``pipe.unet``
   * - Flux, SD3, PixArt
     - ``pipe.transformer``
   * - Wan2.2 (TI2V-5B, T2V-A14B)
     - ``pipe.transformer``

Pipeline-specific kwargs for ``get_calib_dataloader``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Pipeline
     - Recommended kwargs
   * - SDXL
     - ``guidance_scale=8.0``
   * - SD3
     - (defaults)
   * - Flux
     - ``height=1024, width=1024, guidance_scale=3.5, max_sequence_length=512``
   * - Wan2.2 (video)
     - ``height=480, width=832, num_frames=17, guidance_scale=5.0``

Any kwargs accepted by ``pipe(...)`` can be passed -- the utilities
are pipeline-agnostic.

Exclude patterns by model
~~~~~~~~~~~~~~~~~~~~~~~~~

These patterns control which layers are skipped during SVD
decomposition and quantization.  ``*correction*`` must always be
excluded from quantization to protect the SVD low-rank correction
branch.

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Model
     - SVDQuant exclude
     - Quantization exclude
   * - SDXL
     - ``*time_embedding*``, ``*add_time_proj*``, ``*conv_in*``, ``*conv_out*``
     - same as SVDQuant exclude + ``*correction*``
   * - SD3
     - ``*time_text_embed*``, ``*context_embedder*``, ``*pos_embed*``, ``*norm_out*``, ``*proj_out*``
     - same as SVDQuant exclude + ``*correction*``
   * - Flux
     - ``*x_embedder*``, ``*context_embedder*``, ``*time_text_embed*``, ``*norm_out*``, ``*proj_out*``, ``*norm1.linear*``, ``*norm1_context.linear*``
     - same as SVDQuant exclude + ``*correction*``
   * - Wan2.2
     - ``*patch_embedding*``, ``*condition_embedder*``, ``*time_embedder*``, ``*norm_out*``, ``*proj_out*``
     - same as SVDQuant exclude + ``*correction*``

Calibration data sizing
~~~~~~~~~~~~~~~~~~~~~~~

``len(prompts) * n_steps`` = total calibration samples.

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Use case
     - Prompts
     - Steps
     - Samples
   * - Quick test
     - 3
     - 10
     - 30
   * - Standard
     - 5--10
     - 20
     - 100--200
   * - Production
     - 15+
     - 20
     - 300+

Captured tensors are detached and stored on CPU (one copy per
submodule call).  Memory cost is roughly proportional to
``len(prompts) * n_steps`` times the size of one submodule input.
Measure if calibrating with large prompt sets on memory-constrained
hosts.

Using COCO2014 calibration prompts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For production-quality calibration, you can use prompts from the
COCO2014 dataset instead of hand-written prompts.

Setup
^^^^^

Requires ``torchvision`` compatible with your PyTorch version.

.. code-block:: bash

   export DIFFUSERS_ROOT=$PWD
   git clone https://github.com/mlcommons/inference.git
   cd inference
   git checkout 87ba8cb8a6a4f6525f26255fa513d902b17ab060
   cd ./text_to_image/tools/
   sh ./download-coco-2014.sh --num-workers 5
   sh ./download-coco-2014-calibration.sh -n 5
   cd ${DIFFUSERS_ROOT}
   export PYTHONPATH="${DIFFUSERS_ROOT}/inference/text_to_image/:$PYTHONPATH"

Dataset files
^^^^^^^^^^^^^

* **Calibration captions:** ``${DIFFUSERS_ROOT}/inference/text_to_image/coco2014/calibration/captions.tsv``
* **Test captions:** ``${DIFFUSERS_ROOT}/inference/text_to_image/coco2014/captions/captions_source.tsv``

Usage
^^^^^

.. code-block:: python

   def load_coco2014_prompts(coco_dir, max_prompts=None):
       tsv_path = f"{coco_dir}/captions/captions_source.tsv"
       prompts = []
       with open(tsv_path, encoding="utf-8") as f:
           for line in f.readlines()[1:]:  # skip header
               cols = line.split("\t")
               if len(cols) >= 3:
                   prompts.append(cols[2].strip())
       return prompts[:max_prompts] if max_prompts else prompts

   prompts = load_coco2014_prompts("./inference/text_to_image/coco2014", max_prompts=50)

   # Use with get_calib_dataloader as usual.
   dataloader = get_calib_dataloader(pipe, pipe.unet, prompts, n_steps=20, guidance_scale=8.0)

Native inference
~~~~~~~~~~~~~~~~

Quark can convert quantized ``Linear`` layers in Diffusers transformer or UNet modules to native inference kernels during ``ModelQuantizer.freeze``. Use ``RuntimeOptions`` to select the native linear backend after quantization and before running generation.

.. code-block:: python

   from quark.torch.quantization.api import ModelQuantizer
   from quark.torch.quantization.utils import RuntimeOptions

   runtime_options = RuntimeOptions(native_linear_mode="fp8_per_tensor")
   pipe.transformer = ModelQuantizer.freeze(
       pipe.transformer,
       runtime_options=runtime_options,
   )

For MXFP4 quantization, set ``native_linear_mode="mxfp4"``. The ``examples/torch/diffusers/benchmark_flux_fp8_compile.py`` script provides a complete FLUX FP8 native inference benchmark, including optional ``torch.compile``.

.. code-block:: shell

   python benchmark_flux_fp8_compile.py --mode eager

Quantization modes
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 18 22 22 20

   * - Method
     - Weights
     - Activations
     - Observer
     - Config helper
   * - INT8 weight-only
     - INT8 per-tensor
     - fp16/bf16
     - ``min_max``, ``histogram``, ``percentile``, ``MSE``, ``histogrampro``
     - manual ``QLayerConfig(weight=...)``
   * - INT8 w8a8
     - INT8 per-tensor
     - INT8 per-tensor static
     - same as above
     - manual ``QLayerConfig(weight=..., input_tensors=...)``
   * - SVDQuant w4a16
     - INT4 per-group
     - fp16/bf16
     - per-group min/max (fixed)
     - ``build_quant_layer_config("w4a16")``
   * - SVDQuant w4a4
     - INT4 per-group
     - INT4 per-group dynamic
     - per-group min/max (fixed)
     - ``build_quant_layer_config("w4a4")``
   * - MXFP4
     - MXFP4
     - fp16/bf16
     - n/a
     - ``build_quant_layer_config("mxfp4")``
   * - NVFP4
     - FP4 block-16
     - FP4 block-16 dynamic
     - n/a
     - ``build_quant_layer_config("nvfp4")``

For observer-based methods (INT8), the observer determines how
quantization scales are computed from calibration data.  Pass the
observer name via ``observer_method`` in ``Int8PerTensorSpec``:

.. code-block:: python

   Int8PerTensorSpec(observer_method="percentile", ...)
   # other valid values: "min_max", "histogram", "MSE", "histogrampro"

.. note::

   Quark's SmoothQuant algorithm (``SmoothQuantConfig``) currently
   requires LLM-specific layer structure
   (``model_decoder_layers``, ``scaling_layers``) and is not yet
   supported for diffusion models.
