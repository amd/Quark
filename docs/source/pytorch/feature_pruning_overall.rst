.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

LLM Pruning
===========

Overview of LLM Pruning
-----------------------

Deployment of LLMs is constrained by their intensive computational demands.

To make LLMs more accessible and efficient for practical use, LLM pruning shows the ability to reduce the number of parameters or create a Small LLM from a larger one.

Through pruning, either by dropping layers (*depth pruning*) or dropping neurons and attention heads, and embedding channels (*width pruning*).

After pruning, the higher the pruning ratio, the higher the accuracy degradation will be. As a result, Pruning is often accompanied by some fine-tuning or retraining for accuracy recovery.

Mainstream Methods Overview
---------------------------

**Unstructured pruning**:  Concentrate on individual weights, set the individual elements in weight to 0. Meaning that sparsity of rate s% will eliminate s% of the entries in a weight matrix. But the efficiency of the pruned sparse network cannot be realized on general-purpose GPU hardware.

**Semi-structured pruning**: This approach enforces exactly N non-zero values in each block of M consecutive weights. Whether can get GPU acceleration depends on specific hardware architectures.

**Structured pruning**: The pruning methods can be categorized into two different grain levels.

- **Channel-wise**: Based on the weight importance, delete entire rows or columns of weights, providing a more hardware-friendly solution that reduces storage requirements and improves GPU inference performance.
- **Layer-Wise**: Delete the entire decode layer, which reduces the model parameters and model structure more straightforwardly.  As a result, this can bring a more significant GPU inference performance improvement.

By viewing the above short introduction, even though several methods have been proposed by researchers. Considering the actual production environments, Unstructured pruning and Semi-structured pruning may have competitive results in accuracy.  It may not decrease the amount of computation, especially may not bring GPU inference acceleration.

In order to get a real LLM model size reduction and forward computation acceleration. As a result, in the **Quark Pruning tool**, we mainly consider the **structured pruning** methods, which can directly reduce the model size and accelerate GPU inference.

Quark Pruning Tool Feature
--------------------------

Supported Methods Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **OSSCAR Pruner**: A structured pruning method, this method needs to use a small amount of calibration data to perform layer-wise reconstruction and minimize the layer reconstruction error.

  - This method will reduce the intermediate dimension (output channel of the former FC and the input channel of the following FC layer ) in fully connected layers.
  - This method will not change LLM model structure.
  - After pruning, if the remaining channel number is power-of-2, it usually can bring GPU inference time acceleration. (e.g 4096 = 2**12).

- **Depth-Wise Pruner**: Based on the PPL influence, the consecutive decode layers that even after deletion have less influence on the final PPL will be regarded as having less influence on the LLM. These layers can be deleted.

  - Will delete consecutive decode layers based on pre-defined prune ratio.
  - Will directly reduce model size, as it directly deletes the decode layers. This way will improve GPU inference performance.

- For example, the following code block shows a typical ``Llama`` model structure.

  - For ``OSSCAR`` pruner: will decrease the output channel in ``gate_proj`` and ``up_proj``, meanwhile decrease the input channel of ``down_proj``.
  - For depth pruner: will directly delete the entire ``LlamaDecoderLayer`` in the model, for example, after deleting the 10 consecutive layers, this model will remain 70 (80-10) layers, which can directly improve GPU inference performance.

.. code-block:: python

    LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(128256, 8192)
        (layers): ModuleList(
          (0-79): 80 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj): Linear(in_features=8192, out_features=8192, bias=False)
              (k_proj): Linear(in_features=8192, out_features=1024, bias=False)
              (v_proj): Linear(in_features=8192, out_features=1024, bias=False)
              (o_proj): Linear(in_features=8192, out_features=8192, bias=False)
            )
            (mlp): LlamaMLP(
              (gate_proj): Linear(in_features=8192, out_features=28672, bias=False)
              (up_proj): Linear(in_features=8192, out_features=28672, bias=False)
              (down_proj): Linear(in_features=28672, out_features=8192, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm((8192,), eps=1e-05)
            (post_attention_layernorm): LlamaRMSNorm((8192,), eps=1e-05)
          )
        )
        (norm): LlamaRMSNorm((8192,), eps=1e-05)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (lm_head): Linear(in_features=8192, out_features=128256, bias=False)
    )

API Usage
---------

Prepare LLM & Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, padding_side="left")

Init Prune Config & Calibration/Test Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from quark.torch.pruning.config import Config
    from quark.torch.pruning.config import OSSCARConfig, LayerImportancePruneConfig
    # Init Prune config
    # Using OSSCAR
    algo_config = OSSCARConfig.from_dict(json.load(algo_config_file))
    # or Using depth-wise prune
    algo_config = LayerImportancePruneConfig.from_dict((json.load(algo_config_file))

    pruning_config = Config(algo_config=pruning_algo_config)

    # Init the Calibration/Test dataset (Fake code)
    def get_wikitext_dataset(data_dir, tokenizer):
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc

    validation_data = get_wikitext_dataset(data_dir, tokenizer)

Init Pruner & Perform Pruning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from quark.torch import ModelPruner
    model_pruner = ModelPruner(pruning_config)
    model = model_pruner.pruning_model(model, validation_data)

Evaluate the Metric & Save Pruned Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # evaluate
    eval_model(args, model, main_device) # evaluate the PPL on pruned model
    # save to target path
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

Prune Results
-------------

OSSCAR:
~~~~~~~

+-------------------------------------+------------+---------------+-------------------+-----------------------------+----------------------------+
| Model Name                          | Model Size | Pruning Rate  | Pruned Model Size | Before Pruning PPL On Wiki2 | After Pruning PPL On Wiki2 |
+=====================================+============+===============+===================+=============================+============================+
| mistralai/Mixtral-8x7B-Instruct-v0.1| 46.7B      | 9.4838%       | 42.2B             | 4.1370                      | 5.1195                     |
+-------------------------------------+------------+---------------+-------------------+-----------------------------+----------------------------+
| CohereForAI/c4ai-command-r-08-2024  | 32.3B      | 7.4025%       | 29.9B             | 4.5081                      | 6.3794                     |
+-------------------------------------+------------+---------------+-------------------+-----------------------------+----------------------------+
| Qwen/Qwen2.5-14B-Instruct           | 14.8B      | 7.0284%       | 13.7B             | 5.6986                      | 7.5994                     |
+-------------------------------------+------------+---------------+-------------------+-----------------------------+----------------------------+
| meta-llama/Meta-Llama-3-8B          | 8.0B       | 6.8945%       | 7.5B              | 6.1382                      | 8.0755                     |
+-------------------------------------+------------+---------------+-------------------+-----------------------------+----------------------------+
| meta-llama/Llama-2-7b-hf            | 6.7B       | 6.7224%       | 6.2B              | 5.4721                      | 6.2462                     |
+-------------------------------------+------------+---------------+-------------------+-----------------------------+----------------------------+
| facebook/opt-6.7b                   | 6.7B       | 7.5651%       | 6.2B              | 10.8602                     | 11.8958                    |
+-------------------------------------+------------+---------------+-------------------+-----------------------------+----------------------------+
| THUDM/chatglm3-6b                   | 6.2B       | 7.7590%       | 5.6B              | 29.9560                     | 36.0010                    |
+-------------------------------------+------------+---------------+-------------------+-----------------------------+----------------------------+
| microsoft/Phi-3.5-mini-instruct     | 3.8B       | 5.9274%       | 3.6B              | 6.1959                      | 7.8074                     |
+-------------------------------------+------------+---------------+-------------------+-----------------------------+----------------------------+

Depth-Wise Prune:
~~~~~~~~~~~~~~~~~

+-----------------------------+-----------------+------------------------+--------------+-----------------------------------+
| Model                       | PPL (Original)  | PPL (After prune)      | Prune Ratio  | Other Info                        |
+=============================+=================+========================+==============+===================================+
| llama-2-7b-chat-hf          | 6.9419          | 8.139                  | 9.01%        | delete 3 layer [11-13] total 32   |
+-----------------------------+-----------------+------------------------+--------------+-----------------------------------+
| llama-3.1-405B              | 1.85624         | 2.8429                 | 10.3%        | delete 13 layer [18-30] total 126 |
+-----------------------------+-----------------+------------------------+--------------+-----------------------------------+
| Llama-2-70b-chat-h          | 4.6460          | 5.5305                 | 11.16%       | delete 9 layer [27-35] total 80   |
+-----------------------------+-----------------+------------------------+--------------+-----------------------------------+
| Qwen2.5-14B-Instruct        | 5.7010          | 7.0681                 | 9.32%        | delete 5 layers [22-26] total 48  |
+-----------------------------+-----------------+------------------------+--------------+-----------------------------------+
| Mixtral-8x7B-Instruct-v0.1  | 4.1378          | 5.0338                 | 10%          | delete 3 layer [12-14] total 32   |
+-----------------------------+-----------------+------------------------+--------------+-----------------------------------+
| facebook/opt-6.7b           | 10.8605         | 14.4321                | 12.1%        | delete 4 layer [21-24] total 32   |
+-----------------------------+-----------------+------------------------+--------------+-----------------------------------+
| deepseek-moe-16b-chat       | 7.3593          | 8.94327                | 10.76%       | delete 3 layer [11-13] total 28   |
+-----------------------------+-----------------+------------------------+--------------+-----------------------------------+

Example Code
------------

All the example code can be found under ``/examples/torch/language_modeling/llm_pruning``

Other:
------

Reference: `OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization <https://arxiv.org/abs/2403.12983>`_
