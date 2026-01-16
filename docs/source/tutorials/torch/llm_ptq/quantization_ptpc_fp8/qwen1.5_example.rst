FP8 Quantization with Per-Channel Static Weights and Per-Token Dynamic Activations
==================================================================================

Installation
------------

To get started, install:

``pip install amd-quark transformers``

The resulting model Qwen1.5-0.5B-ptpc is ready to be loaded into vLLM.

Code Overview
-------------

Typically, quantizing a floating-point model with AMD Quark involves the
following steps:

1) Load the original floating-point model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    ckpt_path = "Qwen/Qwen1.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

2) Set the quantization configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from quark.torch import LLMTemplate
    
    model_config_type = model.config.model_type if hasattr(model.config, "model_type") else model.config.architectures[0]
    template = LLMTemplate.get(model_config_type)
    quant_config = template.get_config(scheme="ptpc_fp8", exclude_layers=["lm_head"])

3) Use the AMD Quark API to perform an in-place replacement of the model’s modules with quantized modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from quark.torch import ModelQuantizer
    
    quantizer = ModelQuantizer(quant_config)
    model = quantizer.quantize_model(model)

4) (Optional) Export the quantized model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from quark.torch import export_safetensors
    
    output_dir = ckpt_path.rstrip("/").split("/")[-1] + "-FP8-ptpc"
    model = quantizer.freeze(model)
    export_safetensors(model, output_dir)
    tokenizer.save_pretrained(output_dir)

5) (Optional) Evaluate Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install ``vllm`` and ``lm-evaluation-harness``:

``pip install vllm lm_eval``

Evaluate accuracy with ``lm_eval`` (for example on 200 samples of
``gsm8k``):

::

   lm_eval \
     --model vllm \
     --model_args pretrained=./Qwen1.5-0.5B-FP8-ptpc,add_bos_token=True \
     --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 200
