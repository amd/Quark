.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Configuring PyTorch Quantization for Large Language Models
==========================================================

AMD Quark for PyTorch provides a convenient way to configure quantization for Large Language Models (LLMs) through the :py:class:`.LLMTemplate` class. This approach simplifies the configuration process by providing pre-defined settings for popular LLM architectures.

Using LLMTemplate for Quantization Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supported Quantization Schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table shows the quantization schemes supported by :py:class:`.LLMTemplate`, their detailed configurations and the platforms they are supported on:

+-----------------+---------------------------------+------------------------+
| Scheme          | Configuration Details           | Platforms Supported    |
+=================+=================================+========================+
| int4_wo_128     | - Weight-only symmetric INT4    | - RyzenAI              |
|                 | - Per-group quantization        | - ZenDNN               |
|                 | - Group size 128                |                        |
+-----------------+---------------------------------+------------------------+
| int4_wo_64      | - Weight-only symmetric INT4    | - RyzenAI              |
|                 | - Per-group quantization        | - ZenDNN               |
|                 | - Group size 64                 |                        |
+-----------------+---------------------------------+------------------------+
| int4_wo_32      | - Weight-only symmetric INT4    | - RyzenAI              |
|                 | - Per-group quantization        | - ZenDNN               |
|                 | - Group size 32                 |                        |
+-----------------+---------------------------------+------------------------+
| uint4_wo_128    | - Weight-only asymmetric UINT4  | - RyzenAI              |
|                 | - Per-group quantization        | - ZenDNN               |
|                 | - Group size 128                |                        |
+-----------------+---------------------------------+------------------------+
| uint4_wo_64     | - Weight-only asymmetric UINT4  | - RyzenAI              |
|                 | - Per-group quantization        | - ZenDNN               |
|                 | - Group size 64                 |                        |
+-----------------+---------------------------------+------------------------+
| uint4_wo_32     | - Weight-only asymmetric UINT4  | - RyzenAI              |
|                 | - Per-group quantization        | - ZenDNN               |
|                 | - Group size 32                 |                        |
+-----------------+---------------------------------+------------------------+
| int8            | - INT8 quantization             | - RyzenAI              |
|                 | - Per-tensor quantization       | - ZenDNN               |
|                 | - Static quantization           |                        |
+-----------------+---------------------------------+------------------------+
| fp8             | - FP8 E4M3 format               | - AMD MI300 GPU        |
|                 | - Per-tensor quantization       | - AMD MI350 GPU        |
|                 | - Static quantization           | - AMD MI355 GPU        |
+-----------------+---------------------------------+------------------------+
| mxfp4           | - OCP MXFP4 format              | - AMD MI350 GPU        |
|                 | - Per-group quantization        | - AMD MI355 GPU        |
|                 | - Group size 32                 |                        |
|                 | - Dynamic quantization          |                        |
+-----------------+---------------------------------+------------------------+
| mxfp6_e2m3      | - OCP MXFP6E2M3 format          | - AMD MI350 GPU        |
|                 | - Per-group quantization        | - AMD MI355 GPU        |
|                 | - Group size 32                 |                        |
|                 | - Dynamic quantization          |                        |
+-----------------+---------------------------------+------------------------+
| mxfp6_e3m2      | - OCP MXFP6E3M2 format          | - AMD MI350 GPU        |
|                 | - Per-group quantization        | - AMD MI355 GPU        |
|                 | - Group size 32                 |                        |
|                 | - Dynamic quantization          |                        |
+-----------------+---------------------------------+------------------------+
| mx6             | - MX6 format                    | TBD                    |
|                 | - Per-group quantization        |                        |
|                 | - Group size 32                 |                        |
|                 | - Dynamic quantization          |                        |
+-----------------+---------------------------------+------------------------+
| bfp16           | - BFP16 format                  | TBD                    |
|                 | - Per-group quantization        |                        |
|                 | - Group size 8                  |                        |
|                 | - Dynamic quantization          |                        |
+-----------------+---------------------------------+------------------------+

The :py:class:`.LLMTemplate` class offers several methods to create and customize quantization configurations:

1. Using Built-in Templates
---------------------------

AMD Quark includes built-in templates for popular LLM architectures. You can get a list of available templates and use them directly:

.. code-block:: python

    from quark.torch import LLMTemplate

    # List available templates
    templates = LLMTemplate.list_available()
    print(templates)  # ['llama', 'opt', 'qwen', 'mistral', ...]

    # Get a specific template
    llama_template = LLMTemplate.get("llama")

    # Create a basic configuration
    config = llama_template.get_config(scheme="fp8", kv_cache_scheme="fp8")

.. note::

   In the function :py:func:`~quark.torch.quantization.config.template.LLMTemplate.get`, the parameter ``model_type``
   is obtained from the ``model.config.model_type`` attribute. For example, for the model ``facebook/opt-125m``, the ``model_type``
   is ``opt``. See `config.json <https://huggingface.co/facebook/opt-125m/blob/main/config.json#L18>`__.
   When the model_type field is not defined, the ``model.config.architecture[0]`` is assigned as the model_type.

2. Creating Configurations with Advanced Options
------------------------------------------------

The template system supports various quantization options including algorithms, KV cache, attention schemes, layer-wise quantization and exclude_layers, etc.

.. code-block:: python

    from quark.torch import LLMTemplate

    # Get a specific template
    llama_template = LLMTemplate.get("llama")

    # Create configuration with multiple options
    config = llama_template.get_config(
        scheme="int4_wo_128",          # Global quantization scheme
        algorithm="awq",               # Quantization algorithm
        kv_cache_scheme="fp8",         # KV cache quantization
        min_kv_scale=1.0,              # Minimum value of KV Cache scale
        attention_scheme="fp8",        # Attention quantization
        layer_config={                 # Layer-specific configurations
            "*.mlp.gate_proj": "mxfp4",
            "*.mlp.up_proj": "mxfp4",
            "*.mlp.down_proj": "mxfp4"
        },
        layer_type_config={            # Layer type configurations
            nn.LayerNorm: "fp8"
        },
        exclude_layers=["lm_head"]      # Exclude layers from quantization
    )

Notes:

- KV cache quantization is only supported for fp8 now.
- The minimum value of KV Cache scale is 1.0.
- Attention quantization is only supported for fp8 now.
- Algorithm is supported for awq, gptq, smoothquant, autosmoothquant and rotation.
- Layer-wise and layer-type-wise are supported all the quantization schemes.
- Layer-wise and layer-type-wise configurations can override global schemes.

3. Creating New Templates
-------------------------

You can create a new model's template by subclassing :py:class:`.LLMTemplate` and its quantization configuration. Take `moonshotai/Kimi-K2-Instruct <https://huggingface.co/moonshotai/Kimi-K2-Instruct>`__ as an example:

.. code-block:: python

    from quark.torch import LLMTemplate

    # Create a new template
    template = LLMTemplate(
      model_type="kimi_k2",
      kv_layer_name=["*kv_b_proj"],
      exclude_layers=["lm_head"]
    )

    # Register the template to LLMTemplate class (optional, if you want to use the template in other places)
    LLMTemplate.register_template(template)

    # Get the template
    template = LLMTemplate.get("kimi_k2")

    # Create a configuration
    config = template.get_config(
        scheme="fp8",
        kv_cache_scheme="fp8"
    )

4. Registering Custom Schemes
-----------------------------

You can register custom quantization schemes for use with templates:

.. code-block:: python

    from quark.torch.quantization.config.config import Int8PerTensorSpec, QuantizationConfig
    from quark.torch import LLMTemplate

    # Create custom quantization specification
    quant_spec = Int8PerTensorSpec(
        observer_method="min_max",
        symmetric=True,
        scale_type="float",
        round_method="half_even",
        is_dynamic=False
    ).to_quantization_spec()

    # Create and register custom scheme
    global_config = QuantizationConfig(weight=quant_spec)
    LLMTemplate.register_scheme("custom_int8_wo", config=global_config)

    # Get a specific template
    llama_template = LLMTemplate.get("llama")

    # Use custom scheme
    config = llama_template.get_config(scheme="custom_int8_wo")

The template-based configuration system provides a streamlined way to set up quantization for LLMs while maintaining flexibility for customization. It handles common patterns and configurations automatically while allowing for specific adjustments when needed.
