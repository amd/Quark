.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Composed Quantization (Rotation + AutoSmoothQuant + GPTQ)
============================================================

At low bit widths, no single pre-quantization algorithm is enough. Rotation, AutoSmoothQuant
(ASQ) and GPTQ each attack a *different* source of quantization error, and AMD Quark lets you
apply all three to the same model in one pass.

What each algorithm actually fixes
-----------------------------------

The three algorithms are not interchangeable, and they are not redundant. Each targets a
distinct failure mode of low-bit quantization.

**Rotation — smooth activation outliers.**
Quantization error is driven by the *dynamic range* of a tensor. A single outlier channel
forces a coarse scale on every other channel sharing that scale. Rotation multiplies the
hidden state by an orthogonal matrix :math:`R` and its inverse into the next weight, which is
mathematically a no-op in floating point but spreads a concentrated outlier across all
channels. The canonical illustration: the vector :math:`(1, 10)` has a 10x range; rotate it 45
degrees and you get :math:`(7.78, 6.36)`, a 1.2x range. See :doc:`tutorial_rotation` for more
details, and the runnable example at
https://github.com/amd/Quark/tree/HEAD/examples/torch/language_modeling/rotation. Rotation is
applied **offline**, fused into the weights, so it costs nothing at inference time.

**ASQ — rebalances difficulty between activations and weights.**
Rotation makes outliers less extreme but does not equalize activations against weights. In
W4A4 both tensors are quantized, and activations are typically the harder of the two. ASQ
searches for a per-channel scale :math:`s` and rewrites :math:`y = (x / s)(s \cdot W)`: the
activation gets easier to quantize, the weight gets harder, and the product is unchanged. The
scale is folded into the preceding LayerNorm or Linear, so this is also free at inference. See the
runnable :doc:`Auto SmoothQuant tutorial <../tutorials/torch/auto_smoothquant_document_and_example>`,
and :doc:`smoothquant` for the smoothing family ASQ belongs to. Rotation cannot do this — an
orthogonal matrix preserves norms and therefore cannot move difficulty from one tensor to the other.

**GPTQ — minimizes the rounding error that remains.**
After rotation and ASQ have reshaped the tensors, the weights still have to land on the fp4
grid. Round-to-nearest treats each weight independently; GPTQ instead uses second-order
(Hessian) information from calibration data to quantize column by column, compensating each
rounding decision against the ones still to come. It fixes error the first two algorithms
cannot touch, because it operates on the *rounding step itself* rather than on the
distributions being rounded.

They compose because they act at different stages:

.. code-block:: text

   Rotation           ASQ                 GPTQ
   (reshape the  ->   (rebalance      ->  (choose the
    distribution)      act. vs. wt.)       best rounding)

Order matters
^^^^^^^^^^^^^^

What works the best for Qwen3.5-397B-A17B model is: **Rotation → ASQ → GPTQ**

- *Rotation first*, because it is a global change of basis. Any smoothing scale or Hessian
  computed before rotation would describe a tensor that no longer exists afterwards.
- *ASQ before GPTQ*, because ASQ folds a scale into the weights. Running ASQ after GPTQ would
  rescale weights that GPTQ had already snapped onto the fp4 grid, knocking them back off it
  and discarding the entire benefit of the Hessian correction.

Quark applies ``algo_config`` entries in list order, so the order you write is the order you
get.

Target: what gets which treatment
----------------------------------

The recipe is deliberately *not* uniform across the model:

+-------------------------------------------+----------------------------------------+
| Module group                              | Treatment                              |
+===========================================+========================================+
| ``self_attn.{q,k,v,o}_proj``              | Rotation + ASQ + GPTQ, then MXFP4      |
+-------------------------------------------+----------------------------------------+
| ``linear_attn.{in_proj_qkv,in_proj_z,``   | Rotation + ASQ + GPTQ, then MXFP4      |
| ``in_proj_a,in_proj_b,out_proj}``         |                                        |
+-------------------------------------------+----------------------------------------+
| ``mlp.experts.*.{gate,up,down}_proj``     | Rotation, then **MXFP4 only**          |
+-------------------------------------------+----------------------------------------+
| ``mlp.shared_expert.{gate,up,down}_proj`` | Rotation, then **MXFP4 only**          |
+-------------------------------------------+----------------------------------------+
| ``mlp.gate`` (MoE router)                 | **Rotation only** — stays bf16         |
+-------------------------------------------+----------------------------------------+
| ``linear_attn.conv1d``                    | excluded (Conv1d, not a Linear)        |
+-------------------------------------------+----------------------------------------+
| ``mlp.shared_expert_gate``                | excluded (scalar gate, stays bf16)     |
+-------------------------------------------+----------------------------------------+
| ``lm_head``, vision tower, MTP block      | excluded                               |
+-------------------------------------------+----------------------------------------+

Not every projection gets every algorithm
------------------------------------------

The three algorithms do **not** cover the same module set:

- ``conv1d`` is not a Linear, so no algorithm applies.
- ``out_proj`` gets no ASQ because the smoothing scale has nowhere to fold: the gated RMSNorm in
  front of it is sized ``head_v_dim``, but ``out_proj`` takes ``head_v_dim * num_v_heads`` inputs.
- The MoE rows are rotated because they read the rotated residual stream and would break otherwise.
  They skip ASQ and GPTQ because neither is needed for correctness.

+-------------------------------------+----------+----------+--------+
| Module                              | Rotation | ASQ      | GPTQ   |
+=====================================+==========+==========+========+
| ``linear_attn.in_proj_{qkv,z,a,b}`` | yes      | yes      | yes    |
+-------------------------------------+----------+----------+--------+
| ``linear_attn.out_proj``            | yes      | **no**   | yes    |
+-------------------------------------+----------+----------+--------+
| ``linear_attn.conv1d``              | n/a      | **no**   | **no** |
+-------------------------------------+----------+----------+--------+
| ``self_attn.{q,k,v}_proj``          | yes      | yes      | yes    |
+-------------------------------------+----------+----------+--------+
| ``self_attn.o_proj``                | yes      | **no**   | yes    |
+-------------------------------------+----------+----------+--------+
| ``mlp.experts.*.*``                 | yes      | **no**   | **no** |
+-------------------------------------+----------+----------+--------+
| ``mlp.shared_expert.*``             | yes      | **no**   | **no** |
+-------------------------------------+----------+----------+--------+
| ``mlp.gate``                        | yes      | **no**   | **no** |
+-------------------------------------+----------+----------+--------+

Rotation config
^^^^^^^^^^^^^^^^

Rotation needs to know, for each RMSNorm, which modules produce its input and which consume its
output, so that :math:`R` and :math:`R^{-1}` can be fused into the right weights. Each group
names the union of both attention shapes; per-layer filtering keeps whichever exists. The JSON
below is deserialized into :py:class:`.RotationConfig`, whose reference documents every field:

.. code-block:: json

   {
     "name": "rotation",
     "backbone": "model.language_model",
     "model_decoder_layers": "model.language_model.layers",
     "rotation_size": 4096,
     "r1": true, "r2": false, "r3": false, "r4": false,
     "online_r1_rotation": false,
     "scaling_layers": {
       "first_layer": [
         {"prev_modules": ["model.language_model.embed_tokens"],
          "norm_module": "model.language_model.layers.layer_id.input_layernorm",
          "next_modules": ["model.language_model.layers.layer_id.linear_attn.in_proj_qkv",
                           "model.language_model.layers.layer_id.linear_attn.in_proj_z",
                           "model.language_model.layers.layer_id.linear_attn.in_proj_a",
                           "model.language_model.layers.layer_id.linear_attn.in_proj_b",
                           "model.language_model.layers.layer_id.self_attn.q_proj",
                           "model.language_model.layers.layer_id.self_attn.k_proj",
                           "model.language_model.layers.layer_id.self_attn.v_proj"]},
         {"prev_modules": ["model.language_model.layers.layer_id.linear_attn.out_proj",
                           "model.language_model.layers.layer_id.self_attn.o_proj"],
          "norm_module": "model.language_model.layers.layer_id.post_attention_layernorm",
          "next_modules": ["model.language_model.layers.layer_id.mlp.gate",
                           "model.language_model.layers.layer_id.mlp.experts.*.gate_proj",
                           "model.language_model.layers.layer_id.mlp.experts.*.up_proj",
                           "model.language_model.layers.layer_id.mlp.shared_expert.gate_proj",
                           "model.language_model.layers.layer_id.mlp.shared_expert.up_proj",
                           "model.language_model.layers.layer_id.mlp.shared_expert_gate"]}
       ],
       "middle_layers": [
         {"prev_modules": ["model.language_model.layers.pre_layer_id.mlp.experts.*.down_proj",
                           "model.language_model.layers.pre_layer_id.mlp.shared_expert.down_proj"],
          "norm_module": "model.language_model.layers.layer_id.input_layernorm",
          "next_modules": ["model.language_model.layers.layer_id.linear_attn.in_proj_qkv",
                           "model.language_model.layers.layer_id.linear_attn.in_proj_z",
                           "model.language_model.layers.layer_id.linear_attn.in_proj_a",
                           "model.language_model.layers.layer_id.linear_attn.in_proj_b",
                           "model.language_model.layers.layer_id.self_attn.q_proj",
                           "model.language_model.layers.layer_id.self_attn.k_proj",
                           "model.language_model.layers.layer_id.self_attn.v_proj"]},
         {"prev_modules": ["model.language_model.layers.layer_id.linear_attn.out_proj",
                           "model.language_model.layers.layer_id.self_attn.o_proj"],
          "norm_module": "model.language_model.layers.layer_id.post_attention_layernorm",
          "next_modules": ["model.language_model.layers.layer_id.mlp.gate",
                           "model.language_model.layers.layer_id.mlp.experts.*.gate_proj",
                           "model.language_model.layers.layer_id.mlp.experts.*.up_proj",
                           "model.language_model.layers.layer_id.mlp.shared_expert.gate_proj",
                           "model.language_model.layers.layer_id.mlp.shared_expert.up_proj",
                           "model.language_model.layers.layer_id.mlp.shared_expert_gate"]}
       ],
       "last_layer": [
         {"prev_modules": ["model.language_model.layers.layer_id.mlp.experts.*.down_proj",
                           "model.language_model.layers.layer_id.mlp.shared_expert.down_proj"],
          "norm_module": "model.language_model.norm",
          "next_modules": ["lm_head"]}
       ]
     }
   }

Note that ``mlp.gate``, ``mlp.experts.*``, ``mlp.shared_expert.*`` and ``mlp.shared_expert_gate``
all appear here even though two of them are excluded from quantization and none of them receives
an algorithm for offline merging.

ASQ config
^^^^^^^^^^^

Each group names a ``prev_op`` (the module the smoothing scale is folded into) and the
``layers`` that consume its output. Only the attention blocks are listed. The JSON below is
deserialized into :py:class:`.AutoSmoothQuantConfig`, whose reference documents every field:

.. code-block:: json

   {
     "name": "autosmoothquant",
     "model_decoder_layers": "model.language_model.layers",
     "compute_scale_loss": "MAE",
     "scaling_layers": [
       {"prev_op": "input_layernorm",
        "layers": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                   "linear_attn.in_proj_a", "linear_attn.in_proj_b"],
        "inp": "linear_attn.in_proj_qkv", "module2inspect": "linear_attn"},
       {"prev_op": "input_layernorm",
        "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "inp": "self_attn.q_proj", "module2inspect": "self_attn"}
     ]
   }

Unlike rotation, ASQ resolves module names with ``fnmatch``, so a group whose modules do not
exist in a given layer is genuinely skipped without any code support. Listing both attention
variants is safe: each layer matches exactly one of the two.

Two omissions are deliberate: ``linear_attn.out_proj`` has no foldable ``prev_op``, and the
``v_proj -> o_proj`` group for ``self_attn.o_proj`` is left out pending verification. Both are
explained in `Not every projection gets every algorithm`_.

GPTQ config
^^^^^^^^^^^^

GPTQ just needs the list of Linear modules to correct, covering both attention shapes (full- and
linear-attention). The JSON below is deserialized into :py:class:`.GPTQConfig`, whose reference
documents every field; see :doc:`user_guide_config_description` for how algorithm configs are
built in Python and for the constraints GPTQ places on the quantization scheme:

.. code-block:: json

   {
     "name": "gptq",
     "model_decoder_layers": "model.language_model.layers",
     "damp_percent": 0.01, "desc_act": true, "static_groups": true, "block_size": 128,
     "inside_layer_modules": [
       "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
       "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
       "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.out_proj"
     ]
   }

.. important::

   **R2 must be disabled for this model family** (``"r2": false``). R2 rotates the v_proj output and relies on o_proj
   applying the exact inverse. Qwen3.5 attention always multiplies an elementwise sigmoid gate into the attention output
   before o_proj (attn_output = attn_output * torch.sigmoid(gate)), and an elementwise gate does not commute with a
   dense rotation — so the inverse no longer cancels and the model is silently corrupted. R1 alone is safe.

Running Quantization
--------------------

The three configuration files are passed to the stock ``quantize_quark.py`` entry point with
``--quant_algo`` selecting the composition and ``--quant_algo_config_file`` supplying each
config:

.. code-block:: bash

   cd examples/torch/language_modeling/llm_ptq

   export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   export QUARK_MXFP4_IMPL=triton

   python3 quantize_quark.py \
       --model_dir Qwen/Qwen3.5-397B-A17B \
       --data_type bfloat16 --device cuda --multi_gpu balanced \
       --model_attn_implementation sdpa \
       --quant_scheme mxfp4 \
       --quant_algo rotation,autosmoothquant,gptq \
       --quant_algo_config_file rotation        ./rotation.json \
       --quant_algo_config_file autosmoothquant ./autosmoothquant.json \
       --quant_algo_config_file gptq            ./gptq.json \
       --exclude_layers "lm_head" "*visual*" "*mtp*" "*conv1d*" \
                        "*mlp.gate" "*shared_expert_gate" \
       --dataset pileval --num_calib_data 128 --seq_len 2048 --batch_size 1 \
       --skip_evaluation \
       --model_export hf_format \
       --output_dir Qwen3.5-397B-A17B-MXFP4-Rot-ASQ-GPTQ

Serving the export
^^^^^^^^^^^^^^^^^^^

SGLang is not part of this repository. Pull a ROCm build from
`lmsysorg/sglang-rocm <https://hub.docker.com/r/lmsysorg/sglang-rocm>`_ and launch the server
from the image. The MXFP4 export is ~213 GB, so **two** MI355X GPUs are enough:

.. code-block:: bash

   MODEL=/path/to/Qwen3.5-397B-A17B-MXFP4-Rot-ASQ-GPTQ
   IMAGE=lmsysorg/sglang-rocm:v0.5.18-rocm720-mi35x-20260903

   docker run -d --rm --name serve-mxfp4 \
       --device=/dev/kfd --device=/dev/dri --group-add video \
       --ipc=host --shm-size 64g --network host \
       -e HIP_VISIBLE_DEVICES=0,1 \
       -v "$MODEL":"$MODEL" \
       --entrypoint python3 "$IMAGE" \
       -m sglang.launch_server \
         --model-path "$MODEL" \
         --served-model-name Qwen3.5-397B-A17B-MXFP4-Rot-ASQ-GPTQ \
         --tp 2 --context-length 131072 \
         --reasoning-parser qwen3-thinking \
         --host 0.0.0.0 --port 8000 --trust-remote-code

Results
--------

MMLU-Pro measures whether quality survives into multi-step reasoning, which is far more
sensitive to quantization damage than perplexity. All quantized arms are MXFP4 W4A4 with
identical calibration (128 pileval samples, sequence length 2048).

MMLU-Pro (0-shot, chat, full split)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------------------------+---------------+
| Configuration                  | Accuracy      |
+================================+===============+
| No quantization                | 88.33 %       |
+--------------------------------+---------------+
| Rotation only                  | 62.72%        |
+--------------------------------+---------------+
| Rotation + ASQ                 | 82.65%        |
+--------------------------------+---------------+
| Rotation + ASQ + GPTQ          | 87.45%        |
+--------------------------------+---------------+

Reproduce the MMLU-Pro numbers by serving each export and running ``lm_eval`` against it.
Install the API extra first (``pip install 'lm_eval[api]'``) — the plain package cannot talk to
an OpenAI-compatible endpoint and fails at start-up with a missing ``tenacity``:

.. code-block:: bash

   lm_eval --model local-chat-completions \
      --tasks mmlu_pro_chat \
      --model_args "model=Qwen3.5-397B-A17B-MXFP4-Rot-ASQ-GPTQ,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600" \
      --num_fewshot 0 \
      --apply_chat_template \
      --output_path results.json \
      --seed 42 \
      --gen_kwargs "do_sample=true,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=0.0,repetition_penalty=1.0,seed=42"

See also
---------

- :doc:`tutorial_rotation` — rotation and QuaRot in depth
- :doc:`smoothquant` — the smoothing family that ASQ belongs to
