.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

xDiT Inference with Quark Quantization
=======================================

This topic provides examples of using xDiT (Diffusion Transformer Inference) with AMD Quark quantization support. The ``xdit_quark_inference.py`` script provides a standalone interface for running distributed diffusion model inference with optional FP8 or MXFP4 quantization. It supports both single-GPU and multi-GPU distributed execution with torchrun.

Third-party Dependencies
------------------------

This example requires ``xDiT`` and ``AMD Quark`` to be installed:

.. code-block:: shell

   # Install xDiT
   pip install xfuser

   # Install AMD Quark
   pip install amd-quark

   # Optional: Install torchvision if needed
   pip install torchvision

Basic Usage
-----------

You can run the following commands in the ``examples/torch/diffusers`` path.

Single GPU Inference Without Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run FLUX.1-dev model without quantization:

.. code-block:: shell

   python xdit_quark_inference.py --model FLUX.1-dev --prompt "a cat"

Single GPU Inference with Quark Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run with FP8 Quark Quantization
--------------------------------

.. code-block:: shell

   python xdit_quark_inference.py --model FLUX.1-dev --prompt "a cat" --use_quark_quantize

Run with MXFP4 Quark Quantization
----------------------------------

.. code-block:: shell

   python xdit_quark_inference.py --model FLUX.1-dev --prompt "a cat" \
       --use_quark_quantize --quark_quantization_mode mxfp4

Multi-GPU Distributed Inference
--------------------------------

Multi-GPU with FP8 Quantization (2 GPUs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   torchrun --nproc_per_node 2 xdit_quark_inference.py --model FLUX.1-dev --prompt "a cat" \
       --use_quark_quantize --quark_quantization_mode fp8

Multi-GPU with MXFP4 Quantization (4 GPUs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   torchrun --nproc_per_node 4 xdit_quark_inference.py --model FLUX.1-dev --prompt "a cat" \
       --use_quark_quantize --quark_quantization_mode mxfp4

Advanced Usage
--------------

Multi-GPU with Sequence Parallelism (8 GPUs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses Ulysses and Ring sequence parallelism:

.. code-block:: shell

   torchrun --nproc_per_node 8 xdit_quark_inference.py --model FLUX.1-dev --prompt "a cat" \
       --ulysses_degree 2 --ring_degree 2 --use_quark_quantize \
       --quark_quantization_mode fp8

Using Batch Prompts from JSON File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process multiple prompts from a JSON file:

.. code-block:: shell

   python xdit_quark_inference.py --model FLUX.1-dev --prompt_file prompts.json \
       --use_quark_quantize --quark_quantization_mode fp8

The JSON file should follow this format:

.. code-block:: json

   [
       {"id": 1, "prompt": "a cat sitting on a mat"},
       {"id": 2, "prompt": "a dog in the park"},
       {"id": 3, "prompt": "a cityscape at sunset"}
   ]

Command Line Arguments
----------------------

xDiT Arguments
~~~~~~~~~~~~~~

The script inherits all standard xDiT runner arguments via ``xFuserArgs.add_runner_args()``, including:

- ``--model``: Model ID or path (e.g., FLUX.1-dev, stabilityai/stable-diffusion-xl-base-1.0)
- ``--prompt``: Text prompt for image generation
- ``--prompt_file``: Path to JSON file containing multiple prompts
- ``--ulysses_degree``: Ulysses sequence parallelism degree
- ``--ring_degree``: Ring sequence parallelism degree
- ``--profile``: Enable profiling mode

Quark Quantization Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``--use_quark_quantize``: Enable AMD Quark quantization (default: False)
- ``--quark_quantization_mode``: Quantization mode - ``fp8`` (FP8 E4M3) or ``mxfp4`` (OCP MXFP4). Default: ``fp8``
- ``--quark_dynamic_quantization``: Use dynamic quantization (default: True). Set to False for static quantization with calibration.

Native Inference
~~~~~~~~~~~~~~~~

For xDiT integrations that call Quark directly, enable native inference after the transformer or UNet has been quantized and before it is assigned back to the pipeline. This converts eligible quantized ``Linear`` layers to Aiter-backed native inference modules.

.. code-block:: python

   from quark.torch.quantization.api import ModelQuantizer
   from quark.torch.quantization.utils import RuntimeOptions

   quantized_model = quantizer.quantize_model(model_to_quantize)

   runtime_options = RuntimeOptions(native_linear_mode="fp8_per_tensor")
   quantized_model = ModelQuantizer.freeze(
       quantized_model,
       runtime_options=runtime_options,
   )

Use ``native_linear_mode="mxfp4"`` with MXFP4 quantization. The conversion should run after xDiT loads the model and before the first inference call, matching the point where ``xdit_quark_inference.py`` applies Quark quantization.

Quantization Modes
------------------

FP8 Mode
~~~~~~~~

FP8 E4M3 quantization provides:

- Moderate compression with good accuracy preservation
- Fast inference on hardware with FP8 support
- Dynamic quantization by default (no calibration needed)

MXFP4 Mode
~~~~~~~~~~

MXFP4 (Microscaling FP4) quantization provides:

- Higher compression ratio compared to FP8
- Block-wise scaling for better accuracy
- Suitable for memory-constrained environments

Dynamic vs Static Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Dynamic Quantization** (default): Quantization scales are computed at runtime. No calibration required.
- **Static Quantization**: Quantization scales are pre-computed during calibration. Requires a calibration dataloader (not yet implemented in this script).

Notes
-----

1. **Model Support**: The script automatically detects whether the model uses a transformer or UNet architecture and applies quantization accordingly.

2. **Distributed Execution**: When using torchrun with multiple GPUs, the script automatically handles distributed initialization and synchronization.

3. **Quantization Timing**: Quark quantization is applied after the model is loaded but before inference begins.

4. **Fallback Behavior**: If Quark is not installed, the script will warn the user but allow the model to load without quantization if ``--use_quark_quantize`` is not specified.

5. **Output**: Generated images and timing information are saved automatically by the xDiT runner.

Troubleshooting
---------------

ImportError for xDiT
~~~~~~~~~~~~~~~~~~~~

If you encounter an import error for xDiT components:

.. code-block:: shell

   pip install xfuser

ImportError for AMD Quark
~~~~~~~~~~~~~~~~~~~~~~~~~

If quantization fails due to missing Quark:

.. code-block:: shell

   pip install amd-quark

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~

If you encounter OOM errors, try:

1. Use MXFP4 quantization for higher compression: ``--quark_quantization_mode mxfp4``
2. Use more GPUs with torchrun: ``torchrun --nproc_per_node 4 ...``
3. Reduce batch size or image resolution

Example Workflow
----------------

Here's a complete workflow for running quantized inference:

.. code-block:: shell

   # Step 1: Navigate to the examples directory
   cd examples/torch/diffusers

   # Step 2: Run single-GPU inference with FP8 quantization
   python xdit_quark_inference.py \
       --model FLUX.1-dev \
       --prompt "A futuristic city at sunset with flying cars" \
       --use_quark_quantize

   # Step 3: Run multi-GPU inference with MXFP4 quantization
   torchrun --nproc_per_node 4 xdit_quark_inference.py \
       --model FLUX.1-dev \
       --prompt "A futuristic city at sunset with flying cars" \
       --use_quark_quantize \
       --quark_quantization_mode mxfp4

   # Step 4: Check generated images in the output directory
   ls -la output/

Benchmark
---------

**MI355** GPU, xDiT==0.26.3

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - Model Name
     - Quant Config
     - CLIP score
     - HPS score
   * - FLUX.1-dev Base
     - FP16
     - 26.62
     - 30.27
   * - FLUX.1-dev Quantized
     - FP8
     - 26.60
     - 30.15

Reproduce the benchmark
-----------------------

Docker image
~~~~~~~~~~~~

docker pull rocm/pytorch-xdit:v26.3

.. code-block:: shell

   docker run \
       -it --rm \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --user root \
       --device=/dev/kfd \
       --device=/dev/dri \
       --group-add video \
       --ipc=host \
       --network host \
       --privileged \
       --shm-size 128G \
       --name pytorch-xdit \
       -e HSA_NO_SCRATCH_RECLAIM=1 \
       -e OMP_NUM_THREADS=16 \
       -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
       -e HF_HOME=/app/huggingface_models \
       -v /sys:/sys:ro \
       -v $HF_HOME:/app/huggingface_models \
       rocm/pytorch-xdit:v26.3

Generate images
~~~~~~~~~~~~~~~

First, generate the baseline FP16 images (no quantization):

.. code-block:: shell

   python xdit_quark_inference.py \
       --model FLUX.1-dev \
       --csv_prompt_file ./coco2017/coco_prompts.csv \
       --output_dir ./xdit_outputs/fp16

Then, generate images with FP8 Quark quantization:

.. code-block:: shell

   python xdit_quark_inference.py \
       --model FLUX.1-dev \
       --csv_prompt_file ./coco2017/coco_prompts.csv \
       --output_dir ./xdit_outputs/fp8 \
       --use_quark_quantize --quark_quantization_mode fp8

The CSV file should have columns ``image_id``, ``caption_id``, and ``prompt``. Generated images are
saved as ``{image_id}-{caption_id}.png`` in the specified output directory. Point the evaluation
scripts below at the corresponding output directory to compute CLIP and HPS scores.

Clip score evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   import numpy as np
   import torch
   import pandas as pd
   from PIL import Image
   from tqdm import tqdm
   from torchmetrics.multimodal.clip_score import CLIPScore

   GENERATED_IMAGES_DIR = "./xdit_outputs"
   COCO_CSV_PATH = "./coco2017/coco_prompts.csv"
   MODEL_NAME = "openai/clip-vit-large-patch14"
   DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

   def evaluate_clip_score():
       metric = CLIPScore(model_name_or_path=MODEL_NAME).to(DEVICE)

       # Assumes CSV has columns 'image_id', 'caption_id', and 'prompt'
       df = pd.read_csv(COCO_CSV_PATH)

       total_score = 0.0
       valid_count = 0

       print(f"Evaluating {len(df)} images from {GENERATED_IMAGES_DIR}...")

       for _, row in tqdm(df.iterrows(), total=len(df)):
           img_name = f"{row['image_id']}-{row['caption_id']}.png"
           caption = row['prompt']
           img_path = os.path.join(GENERATED_IMAGES_DIR, img_name)

           if not os.path.exists(img_path):
               continue

           image = Image.open(img_path).convert("RGB")
           img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(DEVICE)

           score = metric(img_tensor, caption)
           total_score += score.item()
           valid_count += 1

       if valid_count > 0:
           avg_score = total_score / valid_count
           print(f"\n--- Evaluation Results ---")
           print(f"Model: {MODEL_NAME}")
           print(f"Average CLIP Score: {avg_score:.4f}")
       else:
           print("No valid images found for evaluation.")

   if __name__ == "__main__":
       evaluate_clip_score()

HPS score evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   import torch
   import pandas as pd
   from PIL import Image
   from tqdm import tqdm
   import hpsv2

   GENERATED_IMAGES_DIR = "./xdit_outputs"
   COCO_CSV_PATH = "./coco2017/coco_prompts.csv"

   def evaluate_hps_score():
       # Assumes CSV has columns 'image_id', 'caption_id', and 'prompt'
       df = pd.read_csv(COCO_CSV_PATH)

       total_score = 0.0
       valid_count = 0

       print(f"Evaluating {len(df)} images from {GENERATED_IMAGES_DIR}...")

       for _, row in tqdm(df.iterrows(), total=len(df)):
           img_name = f"{row['image_id']}-{row['caption_id']}.png"
           caption = row['prompt']
           img_path = os.path.join(GENERATED_IMAGES_DIR, img_name)

           if not os.path.exists(img_path):
               continue

           score = hpsv2.score(img_path, caption)
           total_score += score
           valid_count += 1

       if valid_count > 0:
           avg_score = total_score / valid_count
           print(f"\n--- Evaluation Results ---")
           print(f"Average HPS Score: {avg_score:.4f}")
       else:
           print("No valid images found for evaluation.")

   if __name__ == "__main__":
       evaluate_hps_score()
