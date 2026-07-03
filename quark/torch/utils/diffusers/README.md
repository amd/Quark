# Diffusers Quantization Utilities

Calibration data collection for quantizing diffusion model submodules
(UNet, Transformer, VAE decoder, etc.) with Quark.

## Overview

Diffusion model submodules require intermediate pipeline outputs (latents,
timestep embeddings, cross-attention conditioning) as inputs.  These
utilities hook into a diffusers pipeline run, capture the submodule's
inputs, and package them into a `DataLoader` that works with
`ModelQuantizer.quantize_model()` and algorithm processors like
`SVDQuantProcessor`.

One function is provided:

- **`get_calib_dataloader`** -- runs the pipeline, captures submodule
  inputs as a dict keyed by the forward parameter names, returns a
  `DataLoader` consumed by `ModelQuantizer.quantize_model` via
  `model(**data)`

## Quick start

```python
import torch
from diffusers import DiffusionPipeline
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig, Int8PerTensorSpec
from quark.torch.utils.diffusers import get_calib_dataloader

# 1. Load pipeline (use diffusers directly)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("cuda")

# 2. Define calibration prompts
prompts = [
    "A serene lake reflecting mountains at sunset",
    "A futuristic city with flying cars at night",
    "A close-up portrait of a person with dramatic lighting",
]

# 3. Collect calibration data
dataloader = get_calib_dataloader(
    pipe, pipe.unet, prompts, n_steps=20, guidance_scale=8.0,
)

# 4. Quantize
weight_spec = Int8PerTensorSpec(
    observer_method="min_max", symmetric=True, scale_type="float",
    round_method="half_even", is_dynamic=False,
).to_quantization_spec()
qconfig = QConfig(global_quant_config=QLayerConfig(weight=weight_spec))

pipe.unet = ModelQuantizer(qconfig).quantize_model(pipe.unet, dataloader)
```

## Pipeline-specific kwargs

The `**pipe_kwargs` are forwarded directly to `pipe(...)`, so you can
pass any arguments your pipeline accepts:

### SDXL

```python
dataloader = get_calib_dataloader(
    pipe, pipe.unet, prompts, n_steps=20,
    guidance_scale=8.0,
)
```

### Flux

```python
dataloader = get_calib_dataloader(
    pipe, pipe.transformer, prompts, n_steps=20,
    height=1024, width=1024, guidance_scale=3.5, max_sequence_length=512,
)
```

### SD3

```python
dataloader = get_calib_dataloader(
    pipe, pipe.transformer, prompts, n_steps=20,
)
```

### Any other pipeline (PixArt, etc.)

```python
dataloader = get_calib_dataloader(
    pipe, pipe.transformer, prompts, n_steps=20,
    **your_pipeline_specific_kwargs,
)
```

## Calibration data size

Each prompt triggers one pipeline run.  With `n_steps` denoising steps,
the submodule is called `n_steps` times per prompt.  So
`P` prompts x `N` steps = `P * N` calibration samples.

Typical configurations:

- Fast testing: 3 prompts x 10 steps = 30 samples
- Standard: 5-10 prompts x 20 steps = 100-200 samples
- COCO-based: 50+ prompts x 20 steps = 1000+ samples

Captured tensors are stored on CPU (one detached copy per submodule
call).  Memory cost is roughly proportional to `len(prompts) * n_steps`
times the size of one submodule input.  Measure if calibrating with
large prompt sets on memory-constrained hosts.

## SVDQuant

SVDQuant is a primary quantization method for diffusion models.  Pass an
`SVDQuantConfig` in `QConfig.algo_config` and `ModelQuantizer` handles
the rest:

```python
from quark.torch.quantization.config.config import QConfig, SVDQuantConfig
from quark.torch.algorithm.svdquant import build_quant_layer_config

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", ...)
pipe.to("cuda")

dataloader = get_calib_dataloader(pipe, pipe.unet, prompts, n_steps=20, guidance_scale=8.0)

qconfig = QConfig(
    global_quant_config=build_quant_layer_config("w4a16"),
    exclude=["*time_embedding*", "*conv_in*", "*conv_out*"],
    algo_config=[SVDQuantConfig(
        svd_rank=32,
        exclude_patterns=["*time_embedding*", "*add_time_proj*", "*conv_in*", "*conv_out*"],
    )],
)
pipe.unet = ModelQuantizer(qconfig).quantize_model(pipe.unet, dataloader)
```

## Exclude patterns

Exclude patterns are specified in the quantization config, same as for
LLMs:

- `QConfig.exclude` -- layers excluded from quantization
- `SVDQuantConfig.exclude_patterns` -- layers excluded from SVD
  decomposition

See the SVDQuant example above and the
[testSVDQuant.py](../../../../examples/torch/diffusers/testSVDQuant.py)
example for model-specific exclude patterns.
