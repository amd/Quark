LLM Model Depth-Wise Pruning (beta)
===================================

Pruning is the process of making the model smaller, either by dropping
layers (*depth pruning*) or dropping neurons and attention heads and
embedding channels (*width pruning*).

In this tutorial, we will show how to use Quark Depth-Wise pruning tool
to perform depth-wise pruning on several main streams of LLMs.

Please note that this tutorial requires a GPU compatible with ROCm or
CUDA to run. Before you run the pruning code, users need to get the LLM
from Hugging Face (e.g., download the HF checkpoint).

LLM Pruning
-----------

Pruning is a powerful and well-known technique for reducing model size.
Combined with the model quantization, makes it easier to deploy the LLMs
in a server under computation resource constraints environments.

- **Structured pruning:**

  - Prunes (delete) entire rows or columns of weights;
  - Dropping entire decode layers;
  - Both pruning the entire rows or entire layers can bring certain
    computation acceleration.

- **Unstructured pruning**: select the specific individual weight to 0,
  but it may not bring much computation acceleration.
- **Semi-structured pruning.** Exactly N non-zero values in each block
  of M consecutive weights.

According to the recent research papers and industrial practice.
Structured pruning is a usable and adaptable method, which can actually
bring model size decrease and computation speed acceleration. In this
tutorial, we adopt several famous LLMs as examples to show the
effectiveness of the Quark **Depth-Pruning** tool.

All the Models and the accuracy have been tested under AMD ROCM GPU.
Showing the power of AMD GPU and the affiliated software community.

Depth-Wise Pruning
------------------

Pruning essentially involves two key aspects: **discovering redundancy**
and recovering performance.

We adopt the **PPL influence** as layers importance evaluation method:
As redundant blocks contribute less to the model’s outputs, and their
removal leads to smaller degradation in PPL.

Prerequisite & Some Facts
-------------------------

- GPU: to run pruning, at least a GPU compatible with ROCm or CUDA.
- We support three modes to run the pruning process:

  - **Pure GPU mode**. This will need less time to finish the pruning
    process, but requires larger GPU resources.
  - **GPU CPU mix mode**. During the pruning process, only the layer
    that needs to be computed will be placed on GPU; this may largely
    reduce the GPU memory requirements, but it needs more time.
  - The user can select the proper method based on the actual production
    environments.

Environments prepare
--------------------

For model pruning, we need to use GPUs compatible with ROCm (AMD GPUs)
or CUDA(NVIDIA GPUs), and you need a GPU version of PyTorch. You should
check `PyTorch’s installation
page <https://pytorch.org/get-started/locally/>`__ for instructions. For
example, we used ROCm 6.2.4 in this tutorial, which we installed as
follows:

.. code:: shell

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4

To install Quark. Please refer to the `Installation
Guide <https://quark.docs.amd.com/latest/install.html>`__ for further
information on setup.

Perform Depth-Wise
------------------

Import the necessary packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from typing import Any
    
    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    from quark.torch.pruning.config import Config, LayerImportancePruneConfig

Init the LLM model and dataset for evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the ``wikitext`` dataset to test the ``PPL`` result to decide
which layers can be deleted.

Init the LLM
^^^^^^^^^^^^

.. code:: ipython3

    model_path = "facebook/opt-125m"  # User can assign to the folder where the LLM downloaded.
    
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    
    # if not enough GPU memory to load the entire LLM, user can also load LLM to CPU,
    # but user must make sure they have at least one GPU to perform the forward running.
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cpu', torch_dtype="auto", trust_remote_code=True)
    # NOTE: if LLM is loaded to the CPU, make sure to use save_gpu_memory=true to run.

Init the testdata
^^^^^^^^^^^^^^^^^

Each time the original model deletes several consecutive layers, we use
``wikitext`` to test the model’s ``ppl`` so as to decide layers’
importance.

.. code:: ipython3

    def get_wikitext_dataset(model_dir: str, dev: torch.device, **kwargs: Any):
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        seqlen_for_eval = 2048
        testenc = testenc.input_ids
        nsamples = testenc.numel() // seqlen_for_eval
        batch_data = []
        testenc = testenc.to(dev)
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen_for_eval) : ((i + 1) * seqlen_for_eval)].to(dev)
            batch_data.append(batch)
        return batch_data
    
    
    main_device = model.device
    
    eval_dataloader = get_wikitext_dataset(model_path, main_device)

Prepare the Prune config
~~~~~~~~~~~~~~~~~~~~~~~~

As we use the ``facebook/opt-125m`` as an example, this model has the
following structure: - Decoder layers have name fields:
``model.decoder.layers``. - Final norm layer has name fields:
``model.decoder.final_layer_norm``

The model structure is as follows.

.. code:: ipython3

    print(model)

Init the prune config
^^^^^^^^^^^^^^^^^^^^^

NOTE: - Different models have different model structures, so we prepare
different configurations. - User need to Prepare the ``config.json``
file, the ``config.json`` file shows as follows. The concent in
``config.json`` can be seen as follows.

.. code:: json

   {
       "delete_layer_num": 2,
       "model_decoder_layers": "model.decoder.layers",
       "layer_norm_field": "model.decoder.final_layer_norm",
       "layer_num_field": "num_hidden_layers",
       "save_gpu_memory": false
   }

Param explanations:
'''''''''''''''''''

- **delete_layer_num**: We want to finally delete 2 consecutive decode
  layers.
- **model_decoder_layers**: the decode layers name field in
  ``facebook/opt-125m``;
- **layer_norm_field**: the final norm layer name field.
- **layer_num_field**: In model’s ``config.json``, for example, in
  ``fakebook/opt``, ``num_hidden_layers`` indicates the decode layer
  num.
- **save_gpu_memory**:

  - if ``false``: model fully running in GPU, save time but need more
    GPU mem.
  - if ``true``: during the evaluation, the model is tested layer by
    layer, saving GPU memory but typically needing more time.

Init the prune config and the pruner instance.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Init the prune config
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # Method 1: load from the Json file
    # algo_config_file = "./config.json"  # as describe above
    # with open(algo_config_file, 'r') as file:
    #         algo_config_info = json.load(file)
    # pruning_algo_config = LayerImportanceConfig.from_dict(algo_config_info)
    
    # Method 2: manually set the config
    pruning_config = Config()
    pruning_config.algo_config = LayerImportancePruneConfig()
    pruning_config.algo_config.delete_layer_num = 2
    pruning_config.algo_config.model_decoder_layers = "model.decoder.layers"
    pruning_config.algo_config.layer_norm_field = "model.decoder.final_layer_norm"
    pruning_config.algo_config.layer_num_field = "num_hidden_layers"

Init the pruner instance
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from quark.torch import ModelPruner
    
    model_pruner = ModelPruner(pruning_config)

.. code:: ipython3

    param_num = sum(p.numel() for p in model.parameters())
    print(f"Before pruning, the model with: {param_num} parameters")

Perform the Pruning Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    model = model_pruner.pruning_model(model, eval_dataloader)

Based on the above log, it shows that after deleting the [2, 3] layers,
it has the smallest impact on PPL. The original model has 125239296
parameters; after pruning, the model has 10 layers and 111063552. The
pruning ratio is 11.32%.

.. code:: ipython3

    print(model)
    param_num = sum(p.numel() for p in model.parameters())
    print(f"After pruning, the model with: {param_num} parameters")

Save the pruned model to the specified path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   save_dir = "{YOUR_PATH}/facebook/opt-125m"
   model.save_pretrained(save_dir, safe_serialization=True)
   tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
   tokenizer.save_pretrained(save_dir)

Experiments results (Partly)
----------------------------

We conducted several experiments on different types of models to show
the compatibility of our tools.

+----------------------------+--------------+--------------+---------+-------------------+
| **Model**                  | **PPL        | **PPL**      | **prune | **other info**    |
|                            | (Original)** | **(After     | ratio** |                   |
|                            |              | prune)**     |         |                   |
+============================+==============+==============+=========+===================+
| llama-2-7b-chat-hf         | 6.9419       | 8.139        | 9.01%   | delete 3 layer    |
|                            |              |              |         | [11-13] total 32  |
+----------------------------+--------------+--------------+---------+-------------------+
| llama-3.1-405B             | 1.85624      | 2.8429       | 10.3%   | delete 13 layer   |
|                            |              |              |         | [18-30] total 126 |
+----------------------------+--------------+--------------+---------+-------------------+
| Llama-2-70b-chat-h         | 4.6460       | 5.5305       | 11.16%  | delete 9 layer    |
|                            |              |              |         | [27-35] total 80  |
+----------------------------+--------------+--------------+---------+-------------------+
| Qwen2.5-14B-Instruct       | 5.7010       | 7.0681       | 9.32%   | delete 5 layers   |
|                            |              |              |         | [22-26] total 48  |
+----------------------------+--------------+--------------+---------+-------------------+
| Mixtral-8x7B-Instruct-v0.1 | 4.1378       | 5.0338       | 10%     | delete 3 layer    |
|                            |              |              |         | [12-14] total 32  |
+----------------------------+--------------+--------------+---------+-------------------+
| facebook/opt-6.7b          | 10.8605      | 14.4321      | 12.1%   | delete 4 layer    |
|                            |              |              |         | [21-24] total 32  |
+----------------------------+--------------+--------------+---------+-------------------+
| deepseek-moe-16b-chat      | 7.3593       | 8.94327      | 10.76%  | delete 3 layer    |
|                            |              |              |         | [11-13] total 28  |
+----------------------------+--------------+--------------+---------+-------------------+

From several research papers, the beginning and the ending decode layers
play an important role in LLMs. Through the depth pruning, we also get
the same results that the layers at the beginning and end are the most
important. As a result should not be deleted.

Others
------

We will further update our API design to enable Quark can support more
LLMs and enrich more pruning experiments.

**NOTE** For the entire runnable Python script, the user can find the
script under
``examples/torch/language_modeling/llm_pruning/llm_pruning/main_depth_pruning.py``

The example running command can be as follows.

.. code:: shell

   python main_depth_pruning.py --model_dir={PATH_TO_HF_MODEL}Llama-2-7b-chat-hf --multi_gpu 
