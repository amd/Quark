Introduction to Auto-SmoothQuant Algorithm
==========================================

1. Auto-SmoothQuant Overview
----------------------------

**Auto-SmoothQuant** is a post-training weight quantization algorithm
for large language models (LLMs) that aims to mitigate the quantization
degradation caused by activation outliers. It builds upon the core ideas
of SmoothQuant by automatically determining and applying optimal scaling
factors to activations and weights.

The fundamental challenge in quantizing LLMs lies in the wide range of
activation values, particularly the presence of “outliers” that can lead
to significant quantization errors when activations are mapped to
low-bit formats. Auto-SmoothQuant addresses this by:

- **Shifting Quantization Difficulty:** It strategically shifts the
  quantization difficulty from the activations to the weights by
  applying channel-wise scaling factors to activations. This makes the
  activations more “quantization-friendly.”
- **Compensating Scales:** To maintain model correctness, these scaling
  factors are compensated for by inversely scaling the corresponding
  weights.
- **Automated Scale Search:** Unlike manual or fixed-parameter
  SmoothQuant, Auto-SmoothQuant automatically searches for the best
  scaling factors for each layer, minimizing the quantization error
  (e.g., MSE, MAE, RMSE) between the floating-point and quantized
  outputs. SmoothQuant uses fixed hyperparameter values in the range of
  0.1 to 1, while Auto-SmoothQuant applies a grid search with a step
  size of 0.1 to efficiently find the optimal parameter.

This automated approach allows Auto-SmoothQuant to achieve a practical
trade-off, delivering significant memory and computational efficiency
gains without incurring unacceptable accuracy degradation in LLM
applications, while also simplifying the quantization process for users.

For technical details on SmoothQuant, please refer to the original
paper: `SmoothQuant: Accurate and Efficient Post-Training Quantization
for Large Language Models <https://arxiv.org/abs/2211.10438>`__

2. Quark Auto-SmoothQuant Workflow
----------------------------------

The Auto-SmoothQuant algorithm in Quark processes the model in a
layer-by-layer fashion. As illustrated in the figure, after identifying
the decoder layers from the model structure and caching the inputs, the
following steps are performed:

1. **Input Feature Extraction:** For each layer, the input activations
   of all linear modules within that layer are first collected.
2. **Scale Search:** For each defined “scaling layers” within the layer
   (a ``prev_op`` and its layers), a grid search is performed to find
   the optimal channel-wise scaling factors. This search minimizes a
   user-defined loss metric (e.g., MSE, MAE, RMSE) between the
   floating-point output and the output from the pseudo-quantized
   weights and scaled/pseudo-quantized inputs.
3. **Scale Application:** Once the best scales are identified, they are
   applied (fused) into the weights of the ``prev_op`` and the target
   layers.

This process is repeated for every decoder layer in the model, ensuring
optimal scaling for robust quantization.

3. Quark Auto-SmoothQuant Config
--------------------------------

Auto-SmoothQuant leverages JSON config files to define the how to scale
layers within each decoder block. Following is the JSON schema:

Config schema
~~~~~~~~~~~~~

.. code:: json

   {
     "$schema": "http://json-schema.org/draft-07/schema#",
     "title": "AutoSmoothQuant Configuration",
     "description": "Schema for configuring the AutoSmoothQuant process.",
     "type": "object",
     "properties": {
       "name": {
         "description": "The name of the configuration.",
         "type": "string"
       },
       "scaling_layers": {
         "description": "A list of dictionaries that defines how to scale layers within each decoder block. Each dictionary specifies a group of layers that should be scaled together, along with their preceding operation and inspection module.",
         "type": "array",
         "items": {
           "type": "object",
           "properties": {
             "prev_op": {
               "description": "The module preceding the layers, which can be either a linear layer or a layer normalization. During the apply-scale phase, the optimal scale is fused into `prev_op`.",
               "type": "string"
             },
             "layers": {
               "description": "The layers typically consist of one or more linear modules. By applying scaling to the inputs of these layers and inversely scaling their weights, we effectively reduce the quantization error within these critical computational units.",
               "type": "array",
               "items": {
                 "type": "string"
               }
             },
             "inp": {
               "description": "The layer whose input is used for calibration and scaling. Typically, `inp` corresponds to one of the modules within `layers`. A PyTorch forward hook is attached to this operator to capture its input tensor, which is then used as the calibration input for `module2inspect`.",
               "type": "string"
             },
             "module2inspect": {
               "description": "The minimal module used for the grid search to find the optimal scale for a `prev_op`–`layers` pair. It is used to compare the loss between the quantized output and the float output, selecting the scale that minimizes the loss. This field must include at least the target layers. Advanced users may expand the scope of `module2inspect` to potentially achieve higher accuracy. When `module2inspect` is left empty, it defaults to the target layers itself. When `layers` is an array, `module2inspect` must be explicitly specified.",
               "type": "string"
             }
           },
           "required": [
             "prev_op",
             "layers",
             "inp",
           ]
         }
       },
       "model_decoder_layers": {
         "description": "The path to decoder blocks within the model (e.g., 'model.layers'). It specifies where the quantization algorithm should identify and process the main computational layers of the LLM.",
         "type": "string"
       },
       "compute_scale_loss": {
         "description": "The loss function used to search for the best scaling factors. It determines how the quantization error is evaluated during the automated scale search.",
         "type": "string",
         "enum": ["MAE", "MSE", "RMSE"]
       }
     },
     "required": [
       "name",
       "scaling_layers",
       "model_decoder_layers",
       "compute_scale_loss"
     ]
   }

A config example for Llama3
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following the schema above, this is a JSON config for Llama3:

.. code:: json

   {
     "name": "autosmoothquant",
     "scaling_layers": [
       {
         "prev_op": "input_layernorm",
         "layers": [
           "self_attn.q_proj",
           "self_attn.k_proj",
           "self_attn.v_proj"
         ],
         "inp": "self_attn.q_proj",
         "module2inspect": "self_attn"
       },
       {
         "prev_op": "self_attn.v_proj",
         "layers": [
           "self_attn.o_proj"
         ],
         "inp": "self_attn.o_proj"
       },
       {
         "prev_op": "post_attention_layernorm",
         "layers": [
           "mlp.gate_proj",
           "mlp.up_proj"
         ],
         "inp": "mlp.gate_proj",
         "module2inspect": "mlp"
       },
       {
         "prev_op": "mlp.up_proj",
         "layers": [
           "mlp.down_proj"
         ],
         "inp": "mlp.down_proj"
       }
     ],
     "model_decoder_layers": "model.layers",
     "compute_scale_loss": "MAE"
   }

How to Write Your Own Auto-SmoothQuant Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Find decoder layer path**

   Use ``print(model)`` to locate the decoder blocks. The decoder is
   typically defined as an ``torch.nn.Sequential``. Specify the
   decoder-layer *module name* as the value of ``model_decoder_layers``.

2. **Identify layers for scaling**

   Locate the linear layers within the attention and MLP components of
   each decoder block that require scaling. Record each linear layer’s
   *module name* as the value of ``layers``.

      **Note:** The *module name* should be written relative to the
      current decoder block. For example, instead of
      ``model.layers[5].self_attn.q_proj``, it should be written as
      ``self_attn.q_proj``.

3. **Define prev_op - layers pair**

   From the layers identified in Step 2, trace upward along the
   computation path to find a linear layer or a layer normalization that
   acts as the ``prev_op``. Ensure that no non-linear layer exists
   between ``prev_op`` and ``layers``. If multiple layers share the same
   ``prev_op``, merge them into a single entry in the ``scaling_layers``
   list.

   Example:

   .. code:: json

      {
        "prev_op": "input_layernorm",
        "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
      }

4. **Define module2inspect and inp**

   - ``module2inspect`` is an ``nn.Module`` that serves as the minimal
     unit to search for the optimal scale of a ``prev_op``–``layers``
     pair.

     - In the config, specify the *module name* of this ``nn.Module``.
     - It must at least include the target layers. You can expand its
       scope for potentially higher accuracy. If left empty, it defaults
       to the target layers itself. If layers is an array,
       ``module2inspect`` must be explicitly specified.

   - ``inp`` denotes the **first operator (module) inside**
     ``module2inspect``. Specify its *module name* in the config. A
     forward hook will be attached to capture its input tensor for
     calibration.

4. Auto Smoothquant end-to-end example
--------------------------------------

This is a simple example of the Auto Smoothquant algorithm which
includes quantization, SafeTensors exporting and a simple testing
routine. Before running this script, make sure that ``amd-quark`` has
been properly installed. Refer to the `AMD Quark
docs <https://quark.docs.amd.com/latest/install.html>`__ for more
installation detail.

Make sure you have the following dependencies installed on your system:

.. code:: shell

   !pip install torch
   !pip install transformers==4.52.1
   !pip install tqdm
   !pip install datasets
   !pip install accelerate

Let’s start with some basic imports that we are going to use through the
example.

.. code:: ipython3

    import json
    from typing import Any, Optional
    
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

The relevant quark imports follow:

.. code:: ipython3

    from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors
    from quark.torch.quantization.config.config import load_quant_algo_config_from_file

Quark provides default Auto Smoothquant configurations for common
models, but advanced users can create their custom configuration by
crafting a config JSON file as specified above. In this example, we
generate an Auto Smoothquant configuration JSON file using Python.

.. code:: ipython3

    # Define the configuration to be written
    autosmoothquant_config = {
        "name": "autosmoothquant",
        "scaling_layers": [
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        "model_decoder_layers": "model.layers",
        "compute_scale_loss": "MAE",
    }
    
    # Write configuration to a JSON file
    with open("custom_autosmoothquant_config.json", "w") as f:
        json.dump(autosmoothquant_config, f, indent=4)
    
    print("custom_autosmoothquant_config.json has been created.")

Next, we implement utility code to load the dataset and tokenizer:

.. code:: ipython3

    def get_pileval(
        tokenizer: PreTrainedTokenizer,
        nsamples: int,
        seqlen: int,
        device: str | None,
        seed: int = 0,
    ) -> torch.Tensor:
        dataset: Any = load_dataset("mit-han-lab/pile-val-backup", split="validation").shuffle(seed=seed)
        samples, n_run = [], 0
    
        for data in dataset:
            line_encoded = tokenizer.encode(data["text"].strip())
            if 0 < len(line_encoded) <= seqlen:
                samples.append(torch.tensor([line_encoded], device=device))
                n_run += 1
            if n_run == nsamples:
                break
    
        cat_samples = torch.cat(samples, dim=1)
        n_split = cat_samples.shape[1] // seqlen
        train_dataset = [cat_samples[:, i * seqlen : (i + 1) * seqlen] for i in range(n_split)]
    
        return torch.cat(train_dataset, dim=0)
    
    
    def get_tokenizer(model_id: str, max_seq_len: int = 512) -> PreTrainedTokenizer:
        print(f"Initializing tokenizer from {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            model_max_length=max_seq_len,
            padding_side="left",
            trust_remote_code=True,
            use_fast=False,
        )
        if tokenizer.pad_token != "<unk>":
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
        assert tokenizer.pad_token is not None, "Pad token cannot be set!"
        return tokenizer
    
    
    def get_dataloader(
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        device: str | None,
        seq_len: int = 512,
    ) -> DataLoader:
        samples: torch.Tensor = get_pileval(tokenizer, nsamples=128, seqlen=seq_len, device=device, seed=42)
        return DataLoader(samples, batch_size=batch_size, shuffle=False, drop_last=True)

The model is also just a few lines of code:

.. code:: ipython3

    def get_model(model_id: str, device: str | None) -> PreTrainedModel:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="eager",
        )
        return model.eval().to(device)

The next step is the central nerve of this tutorial. This is where we
are going to quantize the model using Auto Smoothquant. Note that
``autosmoothquant`` is specified as the ``algorithm`` for the function
``get_config``.

.. code:: ipython3

    def quantize_model_pipeline(
        model: PreTrainedModel,
        calib_dataloader: DataLoader,
    ) -> PreTrainedModel:
        # Load custom Auto Smoothquant config
        custom_autosmoothquant_config = load_quant_algo_config_from_file("custom_autosmoothquant_config.json")
        # If you don’t need a custom Auto Smoothquant config, you can omit it and use the default configuration.
        template = LLMTemplate(
            model_type=model.config.model_type,
            exclude_layers_name=["lm_head"],
            autosmoothquant_config=custom_autosmoothquant_config,
        )
        quant_config = template.get_config(scheme="uint4_wo_128", algorithm=["autosmoothquant"])
    
        quantizer = ModelQuantizer(quant_config, multi_device=True)
        quantized_model: PreTrainedModel = quantizer.quantize_model(model, calib_dataloader)
    
        print("[INFO] Export Quant Model.")
        export_safetensors(model=quantized_model, output_dir="./")
    
        return quantized_model

The following block creates an evaluation function so that we can see
how our quantized model performs:

.. code:: ipython3

    @torch.no_grad()
    def ppl_eval(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str | None,
    ) -> torch.Tensor:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids.to(device)
    
        seqlen_for_eval = 2048
        nsamples = testenc.numel() // seqlen_for_eval
        nlls: list[torch.Tensor] = []
    
        for i in tqdm(range(nsamples)):
            batch = testenc[:, i * seqlen_for_eval : (i + 1) * seqlen_for_eval]
            lm_logits = model(batch)["logits"]
    
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:]
    
            loss = torch.nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(loss.float() * seqlen_for_eval)
    
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen_for_eval))
        return ppl

Now that all the blocks are created, let’s put everything together and
see AMD Quark in action!

.. code:: ipython3

    def run_quark_example() -> None:
        model_id = "Qwen/Qwen2.5-0.5B"
        batch_size, seq_len = 4, 512
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        print(f"[INFO] Loading model: {model_id}")
        model = get_model(model_id, device)
        tokenizer = get_tokenizer(model_id, max_seq_len=seq_len)
        calib_dataloader = get_dataloader(tokenizer, batch_size, device, seq_len)
    
        print("[INFO] Starting quantization...")
        quantized_model = quantize_model_pipeline(model, calib_dataloader)
        print("[INFO] Quantization complete.")
    
        print("[INFO] Simple test PPL with wikitext-2.")
        ppl = ppl_eval(quantized_model, tokenizer, device)
        print(f"[INFO] Perplexity: {ppl.item():.4f}")
    
    
    if __name__ == "__main__":
        with torch.no_grad():
            run_quark_example()

As you may have noticed, after the quantization using Auto Smoothquant,
the perplexity is around 14.85, which is a great result!
