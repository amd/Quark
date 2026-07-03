# Quantizing Diffusion Models with Quark

This guide walks through end-to-end quantization of diffusion models
using Quark.  The examples below have been validated on SDXL, SD3, and
Flux.1-dev.

## Prerequisites

```bash
pip install diffusers transformers accelerate
```

## Pattern

Every diffusion model quantization follows the same steps:

```python
from quark.torch.utils.diffusers import get_calib_dataloader

# Step 1: Collect calibration data by running the pipeline
dataloader = get_calib_dataloader(pipe, pipe.unet, prompts, n_steps=20, ...)

# Step 2: Quantize (same as LLMs — ModelQuantizer + QConfig)
pipe.unet = ModelQuantizer(qconfig).quantize_model(pipe.unet, dataloader)
```

`get_calib_dataloader` runs the pipeline, captures the target submodule's
inputs as a dict keyed by the forward parameter names, and returns a
`DataLoader`. `ModelQuantizer.quantize_model` consumes those dicts via
`model(**data)` automatically — no wrapping required.

---

## Example 1: SDXL — INT8 Weight-Only

Observer-based INT8 quantization on the SDXL UNet.

```python
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
```

---

## Example 2: SD3 — SVDQuant w4a4

SVDQuant decomposes weights via SVD, adds a low-rank correction branch,
and smooths activations.  Here both weights and activations are
quantized to INT4 (`w4a4`).

```python
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
```

---

## Example 3: SDXL — INT8 Weight + Activation (w8a8)

Static INT8 quantization for both weights and activations.  Uses a
min/max observer to collect activation ranges during calibration.

```python
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
```

---

## Example 4: Flux.1-dev — SVDQuant w4a16

Flux uses bfloat16 and a different transformer architecture.  Note the
Flux-specific `pipe_kwargs` and exclude patterns.

```python
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
```

---

## Quick Reference

### Which submodule to quantize

| Pipeline | Target |
|----------|--------|
| SDXL, SD 1.5, SD 2.1 | `pipe.unet` |
| Flux, SD3, PixArt | `pipe.transformer` |

### Pipeline-specific kwargs for `get_calib_dataloader`

| Pipeline | kwargs |
|----------|--------|
| SDXL | `guidance_scale=8.0` |
| SD3 | (defaults) |
| Flux | `height=1024, width=1024, guidance_scale=3.5, max_sequence_length=512` |

Any kwargs accepted by `pipe(...)` can be passed — the utilities are
pipeline-agnostic.

### Exclude patterns by model

These patterns control which layers are skipped during SVD decomposition
and quantization.  `*correction*` must always be excluded from
quantization to protect the SVD low-rank correction branch.

| Model | SVDQuant exclude | Quantization exclude |
|-------|-----------------|---------------------|
| SDXL | `*time_embedding*`, `*add_time_proj*`, `*conv_in*`, `*conv_out*` | same + `*correction*` |
| SD3 | `*time_text_embed*`, `*context_embedder*`, `*pos_embed*`, `*norm_out*`, `*proj_out*` | same + `*correction*` |
| Flux | `*x_embedder*`, `*context_embedder*`, `*time_text_embed*`, `*norm_out*`, `*proj_out*`, `*norm1.linear*`, `*norm1_context.linear*` | same + `*correction*` |

### Calibration data sizing

`len(prompts) × n_steps` = total calibration samples.

| Use case | Prompts | Steps | Samples |
|----------|---------|-------|---------|
| Quick test | 3 | 10 | 30 |
| Standard | 5–10 | 20 | 100–200 |
| Production | 15+ | 20 | 300+ |

### Using COCO2014 calibration prompts

For production-quality calibration, you can use prompts from the
COCO2014 dataset instead of hand-written prompts.

#### Setup

Requires `torchvision` compatible with your PyTorch version.

```bash
export DIFFUSERS_ROOT=$PWD
git clone https://github.com/mlcommons/inference.git
cd inference
git checkout 87ba8cb8a6a4f6525f26255fa513d902b17ab060
cd ./text_to_image/tools/
sh ./download-coco-2014.sh --num-workers 5
sh ./download-coco-2014-calibration.sh -n 5
cd ${DIFFUSERS_ROOT}
export PYTHONPATH="${DIFFUSERS_ROOT}/inference/text_to_image/:$PYTHONPATH"
```

#### Dataset files

- **Calibration captions:** `${DIFFUSERS_ROOT}/inference/text_to_image/coco2014/calibration/captions.tsv`
- **Test captions:** `${DIFFUSERS_ROOT}/inference/text_to_image/coco2014/captions/captions_source.tsv`

#### Usage

```python
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

# Use with get_calib_dataloader as usual
dataloader = get_calib_dataloader(pipe, pipe.unet, prompts, n_steps=20, guidance_scale=8.0)
```

### Quantization modes

| Method | Weights | Activations | Observer | Config helper |
|--------|---------|-------------|----------|---------------|
| INT8 weight-only | INT8 per-tensor | fp16/bf16 | `min_max`, `histogram`, `percentile`, `MSE`, `histogrampro` | manual `QLayerConfig(weight=...)` |
| INT8 w8a8 | INT8 per-tensor | INT8 per-tensor static | same as above | manual `QLayerConfig(weight=..., input_tensors=...)` |
| SVDQuant w4a16 | INT4 per-group | fp16/bf16 | per-group min/max (fixed) | `build_quant_layer_config("w4a16")` |
| SVDQuant w4a4 | INT4 per-group | INT4 per-group dynamic | per-group min/max (fixed) | `build_quant_layer_config("w4a4")` |
| MXFP4 | MXFP4 | fp16/bf16 | — | `build_quant_layer_config("mxfp4")` |
| NVFP4 | FP4 block-16 | FP4 block-16 dynamic | — | `build_quant_layer_config("nvfp4")` |

For observer-based methods (INT8), the observer determines how
quantization scales are computed from calibration data.  Pass the
observer name via `observer_method` in `Int8PerTensorSpec`:

```python
Int8PerTensorSpec(observer_method="percentile", ...)  # or "min_max", "histogram", "MSE", "histogrampro"
```

> **Note:** Quark's SmoothQuant algorithm (`SmoothQuantConfig`) currently
> requires LLM-specific layer structure (`model_decoder_layers`,
> `scaling_layers`) and is not yet supported for diffusion models.
