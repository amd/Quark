.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Quantization Strategies
=======================

AMD Quark for Pytorch offers three distinct quantization strategies tailored to meet the requirements of various hardware backends:

-  **Post Training Weight-Only Quantization**: The weights are quantized ahead of time, but the activations are not quantized (using the original float data type) during inference.

-  **Post Training Static Quantization**: Quantizes both the weights and activations in the model. To achieve the best results, this process necessitates calibration with a dataset that accurately represents the actual data, which allows for precise determination of the optimal quantization parameters for activations.

- **Post Training Dynamic Quantization**: Quantizes the weights ahead of time, while the activations are quantized dynamically at runtime. This method allows for a more flexible approach, especially when the activation distribution is not well-known or varies significantly during inference.

Here is one sample example for different quant strategies:

.. code:: python

   # 1. Set model
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype="auto")
   model.eval()
   tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

   # 2. Set quantization configuration
   from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
   from quark.torch.quantization.config.config import QConfig, QTensorConfig, QLayerConfig
   from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver, PerChannelMinMaxObserver

   # 2-1. For weight only quantization, please uncomment the following lines.
   DEFAULT_UINT4_PER_GROUP_ASYM_SPEC = QTensorConfig(dtype=Dtype.uint4,
                                                     observer_cls=PerChannelMinMaxObserver,
                                                     symmetric=False,
                                                     scale_type=ScaleType.float,
                                                     round_method=RoundType.half_even,
                                                     qscheme=QSchemeType.per_group,
                                                     ch_axis=0,
                                                     is_dynamic=False,
                                                     group_size=32)
   DEFAULT_W_UINT4_PER_GROUP_CONFIG = QLayerConfig(weight=DEFAULT_UINT4_PER_GROUP_ASYM_SPEC)
   quant_config = QConfig(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG)

   # 2-2. For dynamic quantization, please uncomment the following lines.
   # INT8_PER_TENSER_DYNAMIC_SPEC = QTensorConfig(dtype=Dtype.int8,
   #                                              qscheme=QSchemeType.per_tensor,
   #                                              observer_cls=PerTensorMinMaxObserver,
   #                                              symmetric=True,
   #                                              scale_type=ScaleType.float,
   #                                              round_method=RoundType.half_even,
   #                                              is_dynamic=True)
   # DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG = QLayerConfig(input_tensors=INT8_PER_TENSER_DYNAMIC_SPEC,
   #                                                                weight=INT8_PER_TENSER_DYNAMIC_SPEC)
   # quant_config = QConfig(global_quant_config=DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG)

   # 2-3. For static quantization , please uncomment the following lines.
   # FP8_PER_TENSOR_SPEC = QTensorConfig(dtype=Dtype.fp8_e4m3,
   #                                     qscheme=QSchemeType.per_tensor,
   #                                     observer_cls=PerTensorMinMaxObserver,
   #                                     is_dynamic=False)
   # DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG = QLayerConfig(input_tensors=FP8_PER_TENSOR_SPEC,
   #                                                     weight=FP8_PER_TENSOR_SPEC)
   # quant_config = QConfig(global_quant_config=DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG)

   # 3. Define calibration dataloader (still need this step for weight only and dynamic quantization)
   from torch.utils.data import DataLoader
   text = "Hello, how are you?"
   tokenized_outputs = tokenizer(text, return_tensors="pt")
   calib_dataloader = DataLoader(tokenized_outputs['input_ids'])

   # 4. In-place replacement with quantized modules in model
   from quark.torch import ModelQuantizer
   quantizer = ModelQuantizer(quant_config)
   quant_model = quantizer.quantize_model(model, calib_dataloader)

The strategies share the same user API.
You simply need to set the strategy through the quantization configuration, as demonstrated in the previous example.
For more details about setting quantization configuration, refer to the "Configuring AMD Quark for PyTorch" chapter.

Choosing a strategy for diffusion models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When quantizing activations of diffusion models (the UNet or transformer
submodule), **prefer dynamic activation quantization -- and FP8 in
particular -- over static observer-based activation quantization.**

The quantized submodule is called many times per image, once per denoising
timestep, and its activation magnitudes vary substantially across those steps
(high-noise early steps versus low-noise late steps):

- A **static** scale is calibrated once over all timesteps, so it must cover the
  union of those ranges; most timesteps then use only a fraction of the
  available quantization bins. Static ``min/max`` observers are also
  outlier-sensitive (a single extreme activation inflates the scale for every
  step); ``percentile`` and ``MSE`` observers help but remain static.
- A **dynamic** scale is recomputed per forward pass, adapting to the actual
  range at each timestep, at a small per-forward runtime cost. Dynamic scaling
  also unlocks finer-grained quantization schemes. These include per-channel,
  per-group, and per-token scales. We cannot do these with a single static scale,
  since these schemes consider the range within each forward pass
  rather than relying on one range fixed across all timesteps.
- **FP8 (E4M3)** has a wider dynamic range and non-uniform spacing compared to
  INT8's uniform grid, so it tolerates the per-step variation and outliers
  better.

A good default for diffusion activation quantization is therefore **dynamic
FP8**:

.. code-block:: python

   from quark.torch.quantization import FP8E4M3PerTensorSpec
   from quark.torch.quantization.config.config import QConfig, QLayerConfig

   fp8_dyn = FP8E4M3PerTensorSpec(is_dynamic=True).to_quantization_spec()
   quant_config = QConfig(global_quant_config=QLayerConfig(weight=fp8_dyn, input_tensors=fp8_dyn))

.. note::

   The relative quality of dynamic versus static activation quantization is
   model- and configuration-dependent. Validate the specific configuration
   end-to-end (for example with CLIP/FID on a diffusion model) before relying on
   it in production.

For very low-bit activations (such as INT4 activations, ``w4a4``), use
:doc:`SVDQuant <svdquant>`, whose smoothing and low-rank error correction make
aggressive weight and activation quantization viable. See also
:doc:`Quantizing Diffusion Models with Quark <example_quark_torch_diffusers>`
and :doc:`xDiT Inference with Quark Quantization <example_quark_torch_xdit>`.
