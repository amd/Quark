.. Copyright (C) 2025 - 2026, Advanced Micro Devices, Inc. All rights reserved.

Automatic Mixed Precision (AMP)
================================

Automatic Mixed Precision (AMP) is a post-training accuracy-recovery technique that
selectively promotes individual layers or subgraphs from a low-precision quantized
model to a higher precision, using sensitivity analysis to identify which promotions
yield the greatest accuracy improvement with the smallest quality cost.

The AMP pipeline consists of two phases:

1. **Sensitivity analysis** — Each candidate layer (or subgraph) is temporarily
   promoted to the target precision and the metric between the float and the
   promoted model is measured. Candidates are ranked from most sensitive (closest
   to float) to least sensitive.
2. **Greedy promotion** — Starting from the most-sensitive candidate, layers are
   permanently promoted one at a time. When ``metric_threshold`` is non-zero, the
   process stops as soon as the metric exceeds the threshold, which prevents
   over-promoting and keeps the model close to the original quantized size. When
   ``metric_threshold`` is ``0`` (the default), all candidates are promoted
   unconditionally. When ``metric_threshold`` is ``None``, the promotion step is
   skipped entirely and only the sensitivity table is produced — useful for
   inspecting per-layer sensitivity scores before deciding on a threshold.

Basic Usage
-----------

The example below shows mixing a baseline A8W8 model with A16W8 layers using the AMP:

.. code-block:: python

    from quark.onnx import ModelQuantizer, QConfig, QLayerConfig, AutoMixprecisionConfig
    from quark.onnx.quantization.config.spec import Int8Spec, Int16Spec

    base_config = QLayerConfig(activation=Int8Spec(), weight=Int8Spec())
    amp_config = AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int16Spec()),
        target_op_type = ("Conv", "ConvTranspose", "Gemm", "MatMul"),
        data_size = 10,
        metric_optimize_object = "quality",
        sensitivity_cache_file = "model_sensitivity.json",
    )

    quant_config = QConfig(global_config=base_config, algo_config=[amp_config])

    quantizer = ModelQuantizer(quant_config)

    quantizer.quantize_model(float_model_path, quant_model_path, calib_data_reader)

Arguments
---------

The following arguments can be passed to ``AutoMixprecisionConfig``.

  - **target_layer_config**: (QLayerConfig | Dict[QLayerConfig, List[str]] | List[QLayerConfig]) **Required.**
    The precision spec to promote candidate nodes to. Three forms are supported:

    - **Single config** (most common) — a ``QLayerConfig`` applied to every
      candidate. For example, ``QLayerConfig(activation=Int16Spec(), weight=Int16Spec())``
      promotes both activations and weights to INT16 for all candidates.

    - **Per-candidate dict** — a ``dict`` mapping each ``QLayerConfig`` to the
      list of candidate node names that should be promoted with that config.
      This allows different candidates to be promoted to different precision in a
      single AMP run.

      Exactly one entry may map to an empty list ``[]``; that config becomes the
      **global fallback** for any candidate not explicitly listed in another entry.
      If no entry maps to ``[]``, the first config in iteration order is used as
      the fallback.

      Example — promote all candidates to INT8 by default, but promote ``/matmul1``
      to INT16:

      .. code-block:: python

          target_layer_config={
              QLayerConfig(activation=Int8Spec(), weight=Int8Spec()): [],  # global fallback
              QLayerConfig(activation=Int16Spec(), weight=Int16Spec()): ["/matmul1"],
          }

    - **Multi-config list** — a ``list`` of ``QLayerConfig`` candidates.
      During sensitivity analysis, every candidate node is scored against **each
      config** in the list; the config that yields the smallest score (closest to
      float) is automatically selected for that candidate during the mixing phase.
      Per-config scores and the selected index are stored in the sensitivity cache
      so the selection can be replayed without re-running analysis.

      Example — let AMP choose per-layer between INT8 and INT16:

      .. code-block:: python

          target_layer_config=[
              QLayerConfig(activation=Int8Spec(), weight=Int8Spec()),
              QLayerConfig(activation=Int16Spec(), weight=Int16Spec()),
          ]

  - **target_op_type**: (Tuple[str, ...] | List[str]) ONNX op types that are considered as
    promotion candidates. The default value is
    ``("Conv", "ConvTranspose", "Gemm", "MatMul")``.

  - **subgraph_json**: (str | Path | None) Path to a JSON file that groups model
    nodes into named subgraphs. When provided, the sensitivity score is computed
    per subgraph instead of per layer. See :ref:`subgraph-wise-analysis-and-mixing` for the
    file format. The default value is ``None`` (layer-wise analysis).

  - **include_layers**: (List[str] | None) Allowlist of node names eligible for
    promotion. When empty or ``None``, all nodes matching ``target_op_type`` are
    considered. The default value is ``None``.

  - **exclude_layers**: (List[str] | None) Denylist of node names that must not
    be promoted regardless of their sensitivity score. The default value is ``None``.

  - **data_size**: (int) Number of calibration samples to use during sensitivity
    analysis and the promotion loop. ``0`` uses all available samples. The default
    value is ``0``.

  - **metric_output_index**: (int) Index of the model output to use when computing
    the metric. The default value is ``0``.

  - **metric_distance_fn**: (Callable | None) A custom distance function
    ``(float_output, quant_output) -> float`` where a **lower** value means the
    quantized model is closer to float. Takes priority over ``metric_default``
    when provided. Mutually exclusive with ``metric_evaluate_fn``. The default
    value is ``None``.

  - **metric_evaluate_fn**: (Callable | None) A custom evaluation function
    ``(model_output) -> float`` where a **higher** value is better (for example,
    top-1 accuracy). Internally converted to a distance by subtracting the
    quantized score from the float score. Takes priority over ``metric_default``
    when provided. Mutually exclusive with ``metric_distance_fn``. The default
    value is ``None``.

  - **metric_default**: (str) Name of the built-in distance metric to use when
    neither ``metric_distance_fn`` nor ``metric_evaluate_fn`` is provided. Supported
    values:

    - ``"l2"`` *(default)* — mean L2 norm of element-wise differences between
      float and quantized outputs, averaged over all (sample, output) pairs.
    - ``"kl"`` — mean KL divergence KL(P_float ‖ P_quant) where each output array
      is treated as an unnormalized probability distribution. Values are shifted to
      be non-negative and normalized before the divergence is computed.
    - ``"cosine"`` — mean cosine distance (1 − cosine_similarity) between flat
      float and quantized output vectors, averaged over all pairs. Returns 0 for
      identical outputs and 2 for perfectly opposite vectors.
    - ``"sqnr"`` — mean **negative** SQNR (Signal-to-Quantization-Noise Ratio) in
      dB (−dB), measured at the **network output** so that error propagation through
      subsequent layers is captured. Set ``metric_threshold`` to a **negative** value
      when using this metric; for example, ``metric_threshold=-30`` stops mixing when
      SQNR drops below 30 dB.
    - ``"psnr"`` — mean **negative** PSNR in dB (−dB), so that the lower-is-better
      convention is preserved. Set ``metric_threshold`` to a **negative** value when
      using this metric; for example, ``metric_threshold=-20`` stops mixing when
      PSNR drops below 20 dB.

    An unknown name raises ``ValueError`` at construction time listing the valid
    options.

  - **metric_threshold**: (float | None) The accuracy threshold for the promotion loop.
    Its role depends on ``metric_optimize_object``.

    - When set to ``None``, only sensitivity analysis is performed and the mixing
      (promotion) step is skipped entirely — useful for inspecting per-layer
      sensitivity scores without modifying the model.
    - When set to ``0`` (default), the threshold is disabled and all candidates are
      promoted unconditionally.
    - When set to a non-zero float, the loop stops as soon as the metric crosses
      the threshold (direction depends on ``metric_optimize_object``).

  - **metric_optimize_object**: (str) Controls the optimization objective of the
    promotion loop. Two values are supported:

    - ``"speed"`` *(default)* — targets a **high-precision baseline** (e.g.
      Int16) and mixes in **lower-precision** layers (e.g. Int8) to maximize
      hardware performance. The baseline metric must satisfy
      ``score <= metric_threshold`` to proceed. The loop promotes candidates in
      ascending sensitivity order (least impactful first) and stops — reverting
      the last candidate — as soon as ``score > metric_threshold``.

    - ``"quality"`` — targets a **low-precision baseline** (e.g. Int8) and mixes
      in **higher-precision** layers (e.g. Int16) to recover accuracy. The
      baseline metric must satisfy ``score > metric_threshold`` to proceed. The
      loop promotes candidates in ascending sensitivity order (most impactful
      first) and stops — without reverting — as soon as
      ``score <= metric_threshold``.

  - **sensitivity_cache_file**: (str | Path | None) Path to a JSON cache file for
    sensitivity analysis results. If the file exists the analysis is skipped and
    results are loaded from disk; otherwise analysis runs and results are saved.
    The default value is ``None`` (no caching).

  - **worker_num**: (int) Number of parallel workers for sensitivity analysis.
    Each worker scores one candidate spec independently, using its own copy of
    the model and the mixing strategy. Parallelism is thread-based (joblib
    ``threading`` backend), so it is most effective when the bottleneck is ONNX
    Runtime inference rather than Python logic. Cannot exceed the number of
    available CPU cores. The default value is ``1`` (serial execution).

  - **dual_quant_nodes**: (bool) When ``True``, Q/DQ node pairs are inserted at
    every precision boundary in the final mixed-precision model so that downstream
    runtimes can handle the transition between precision regions. The default value
    is ``False``.

  - **no_input_qdq_shared**: (bool) When ``True``, nodes whose input activation
    Q/DQ node is shared with other nodes are skipped, avoiding graph topology
    inconsistencies. The default value is ``False``.

  - **shared_param_mode**: (str) Controls how scale/zp initializers that are shared
    between the promoted Q/DQ pair and other nodes (e.g. Q/DQ nodes at input and
    output of Transpose) are handled during precision promotion.

    - ``"propagate"`` *(default)* — keeps the shared initializer and updates the
      ``op_type`` and ``domain`` of every node that references it so that all
      users remain consistent with the new data type.

    - ``"unshare"`` — gives the promoted Q/DQ pair its own copy of the
      initializer (named with a ``_mp`` suffix) and leaves the original shared
      initializer and all other referencing nodes untouched.

    The default ``"propagate"`` is suitable for most models. Use ``"unshare"``
    if you want to preserve the original graph sharing structure and handle
    domain consistency through other means (e.g. ``dual_quant_nodes``).

.. _subgraph-wise-analysis-and-mixing:

Subgraph-wise Analysis and Mixing
---------------------------------

By default, AMP runs a **layer-wise** sensitivity analysis where each candidate
node is scored independently and mixed precision where each candidate node is promoted
one by one. For models with architectural blocks (residual connections, attention heads,
etc.) it is often better to score an entire block as a single unit — this is called
**subgraph-wise** analysis and mixing and is enabled by providing a ``subgraph_json`` file.

How to obtain the JSON file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are three ways to produce the subgraph partition JSON file.

**Manual authoring**

For small models or when you have detailed knowledge of the model architecture,
you can write the JSON by hand.  Open the model in `Netron <https://netron.app>`_
(or inspect the ONNX graph with ``onnx.load``), identify the node names that
mark the start and end of each logical block, and write the corresponding JSON
entries.

**EP-based graph partitioning**

Some execution providers (for example, CPUExecutionProvider) expose graph partitioning
APIs that can automatically group nodes into deployment units.  The partition output can
be post-processed into the subgraph JSON format expected by AMP.  Refer to your
execution provider's documentation for details.

**AI agent (recommended)**

Our experiments consistently show that an AI agent produces the best-quality
partitions, balancing architectural semantics with practical constraints such as
avoiding subgraphs that are too coarse or too fine-grained.  AMD Quark ships a
dedicated skill for this purpose:

.. code-block:: text

    /quark-onnx-subgraph-partitioner

Invoke this skill in a Claude Code session with your model path and any
architecture hints (for example, the number of attention heads or the layer
naming convention).  The skill will load the model, analyse its topology, and
emit a ready-to-use ``subgraph.json`` file.

.. note::

   Using the AI agent is the recommended approach when working with large or
   complex models such as Transformers, diffusion backbones, or detection models
   with multi-scale feature pyramids.

   Why the agent approach works best?

   Rule-based partitioning tools must rely on a fixed vocabulary of patterns —
   regular expressions over node names, hard-coded op-type sequences, or execution
   provider-specific heuristics.  These rules generalise poorly across model
   families and often produce partitions that are either too coarse (entire backbone
   as one block) or too fine (one op per subgraph).

   An LLM-based agent has a qualitatively different capability: it can reason
   jointly over graph structure and the semantic information embedded in entity
   names.  Node names such as ``/encoder/layer.3/attention/self/query/MatMul``,
   initializer names such as ``model.backbone.layer1.0.conv1.weight``, and tensor
   names such as ``attention_mask`` all carry rich semantic signal that a language
   model can interpret without any model-specific rules.  By combining this semantic
   understanding with structural graph traversal, the agent can:

   - identify logical boundaries (for example, between a self-attention sublayer and
     the subsequent feed-forward sublayer) that would be invisible to a purely
     topology-based tool;
   - respect architectural conventions across diverse model families (Transformers,
     CNNs, detection heads) without requiring family-specific rule sets;
   - adapt partition granularity to the user's stated goals — for example,
     grouping aggressive blocks together when the user asks for coarse-grained
     analysis, or splitting residual branches apart when fine-grained scoring is
     requested.

What is the JSON file structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The JSON file has two top-level fields and a ``"subgraphs"`` array:

.. code-block:: none

    {
      "quantized":     false,
      "num_subgraphs": 2,
      "subgraphs": [
        {
          "name": "<subgraph_name>",
          "description": "<optional human-readable description>",
          "start_nodes": ["<node_name>", ...],
          "end_nodes":   ["<node_name>", ...]
        }
      ]
    }

**Top-level fields:**

- **quantized** — *(optional, default* ``false`` *)* When ``true``, the entire
  partition was generated from a *quantized* ONNX model. The parser validates
  all boundary node names against the quantized graph and resolves subgraph
  topology on the quantized model directly. When ``false`` (default), boundary
  names are validated against the float model; resolved nodes are subsequently
  filtered to those present in the quantized model.
- **num_subgraphs** — *(optional)* The number of subgraph entries. When present
  it must match the actual length of the ``"subgraphs"`` array; a mismatch
  raises a ``ValueError``. Use it as a quick consistency check.

**Per-subgraph fields (inside** ``"subgraphs"`` **):**

- **name** — A human-readable label for the subgraph, used in logs and the
  sensitivity cache.
- **description** — *(optional)* A free-text description of what the subgraph
  represents (for example, ``"Encoder self-attention block, layer 0"``). The
  parser ignores this field at runtime; it is purely for documentation purposes
  inside the JSON file and has no effect on analysis results.
- **start_nodes** — One or more node names at which graph traversal begins.
- **end_nodes** — One or more node names at which graph traversal terminates (the
  end nodes themselves are included in the subgraph).

The resolved subgraph consists of all nodes reachable by walking forward from every
``start_node`` up to and including the ``end_nodes``.

Behaviour rules
^^^^^^^^^^^^^^^

- If a node name in ``start_nodes`` or ``end_nodes`` does not exist in the
  reference model (float model when the top-level ``"quantized"`` is ``false``,
  quantized model when it is ``true``), a ``ValueError`` is raised.
- If a boundary node exists in the float model but was removed by graph
  optimisation in the quantized model, a warning is emitted and all nodes of
  that subgraph are moved to ``__ungrouped__`` automatically.
- If the same node appears in two subgraphs, a warning is emitted and the
  overlapping node is removed from the **later** subgraph definition.
- Model nodes that are not covered by any subgraph definition are automatically
  collected into a synthetic subgraph named ``"__ungrouped__"`` and included in
  the sensitivity analysis.

Example
^^^^^^^

Consider a simple four-node model with topology
``Conv_0 → Relu_1 → Conv_2 → MatMul_3``.
The following JSON groups ``Conv_0`` and ``Relu_1`` as one block and scores
``Conv_2`` with ``MatMul_3`` as another:

.. code-block:: json

    {
      "subgraphs": [
        {
          "name": "encoder_block",
          "description": "First two ops: stem Conv and activation",
          "start_nodes": ["Conv_0"],
          "end_nodes":   ["Relu_1"]
        },
        {
          "name": "decoder_block",
          "description": "Second Conv followed by the final MatMul (from quantized graph)",
          "start_nodes": ["Conv_2"],
          "end_nodes":   ["MatMul_3"],
          "quantized":   true
        }
      ]
    }

A more realistic example for a Transformer with two attention layers might look
like this:

.. code-block:: json

    {
      "subgraphs": [
        {
          "name": "attention_0",
          "description": "Self-attention block for encoder layer 0",
          "start_nodes": ["/encoder/layer.0/attention/self/query/MatMul"],
          "end_nodes":   ["/encoder/layer.0/attention/output/dense/MatMul"]
        },
        {
          "name": "attention_1",
          "description": "Self-attention block for encoder layer 1",
          "start_nodes": ["/encoder/layer.1/attention/self/query/MatMul"],
          "end_nodes":   ["/encoder/layer.1/attention/output/dense/MatMul"]
        }
      ]
    }

Save the file (for example, ``subgraphs.json``) and pass its path to the config:

.. code-block:: python

    amp_config = AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int16Spec(), weight=Int16Spec()),
        subgraph_json="subgraphs.json",
        metric_threshold=0.01,
    )

Sensitivity Analysis Result Caching
-----------------------------------

Sensitivity analysis can be expensive because each candidate is promoted,
evaluated, and reverted in turn. The ``sensitivity_cache_file`` parameter lets you
persist the ranked results to disk so that subsequent runs skip the analysis phase
entirely:

.. code-block:: python

    amp_config = AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int16Spec(), weight=Int16Spec()),
        sensitivity_cache_file="amp_sensitivity.json",
        metric_threshold=0.01,
    )

On the **first run** the sensitivity scores are computed, sorted, and written to
the file. On **all subsequent runs** with the same path the file is loaded and the
analysis is skipped. This makes iterative threshold tuning much faster — you can
vary ``metric_threshold`` as many times as you like without re-running inference.

.. warning::

   **The final mixing result depends entirely on the sensitivity ranking.**
   Whether the ranking comes from a freshly computed analysis or a cache file,
   the greedy promotion loop uses it as-is. A stale cache produces a stale mix:

   - If the **calibration data** changes, sensitivity scores shift and the old
     ranking no longer reflects the true sensitivity order.
   - If the **metric function or threshold** changes meaning, a ranking built
     under the old metric may not optimise what you care about now.
   - If the **subgraph partition** (``subgraph_json``) changes, the new grouping
     is silently ignored: the cached ``candidate_nodes`` already encode which
     nodes belong to which candidate, so the updated partition has no effect.

   Changes to the **model structure**, **target configuration**
   (``target_layer_config``, ``target_op_type``), and **layer filters**
   (``include_layers``, ``exclude_layers``) are detected automatically via the
   ``cache_key`` fingerprint embedded in the file — see `Details of the caching file format`_ below.
   When a mismatch is detected, the cache is discarded and sensitivity analysis
   runs again.

Details of the caching file format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cache file is a JSON object. The ``results`` array contains one element per
candidate (a single layer in layer-wise mode, or a named subgraph in subgraph-wise
mode), sorted in **ascending score order** (most-sensitive / closest to float first):

.. code-block:: json

    {
      "version": "0.13.0",
      "cache_key": "b20a7264d22ca0eb...",
      "results": [
        {
          "name": "/layer0/Conv",
          "candidate_nodes": ["/layer0/Conv"],
          "score": 0.0031,
          "all_config_scores": [0.0031],
          "best_config_index": 0,
          "enabled": true
        },
        {
          "name": "attention_block_0",
          "candidate_nodes": [
            "/encoder/layer.0/attention/self/query/MatMul",
            "/encoder/layer.0/attention/self/key/MatMul"
          ],
          "score": 0.0147,
          "all_config_scores": [0.0312, 0.0147],
          "best_config_index": 1,
          "enabled": true
        },
        {
          "name": "/layer2/MatMul",
          "candidate_nodes": ["/layer2/MatMul"],
          "score": 0.0512,
          "all_config_scores": [0.0512, 0.0891],
          "best_config_index": 0,
          "enabled": true
        }
      ]
    }

**Top-level fields:**

- **version** — (string) The Quark version that wrote the file (for example
  ``"0.13.0"``). Printed in the info log when the cache is loaded so you can
  confirm which release produced the results.
- **cache_key** — (string) A 64-character ``Secure Hash Algorithm-256`` hex
  digest computed from the quantized model's graph topology (node names, op types,
  and connectivity), ``target_op_type``, ``target_layer_config``, ``include_layers``,
  and ``exclude_layers``. On every subsequent run Quark recomputes this fingerprint
  and compares it against the stored value. If they differ, the cache is
  discarded and sensitivity analysis runs again automatically.

**Per-result fields (inside** ``"results"`` **):**

- **name** — (string) The candidate identifier. In layer-wise mode this is the
  node name. In subgraph-wise mode this is the ``"name"`` field from the subgraph
  JSON.
- **candidate_nodes** — (array of strings) The ONNX node names covered by this
  candidate. Contains exactly one element in layer-wise mode; may contain multiple
  elements in subgraph-wise mode.
- **score** — (number) The best metric distance across all config candidates —
  the minimum of ``all_config_scores``. A lower score means the promoted candidate
  is closer to the float model, i.e. it benefits more from precision promotion.
- **all_config_scores** — (array of numbers) One score per entry in
  ``target_config_list``, in the same order. When ``target_layer_config`` is a
  single ``QLayerConfig`` or a dict, this array always has exactly one
  element and equals ``[score]``. When ``target_layer_config`` is a list of
  configs, each element records the metric measured under the corresponding config
  so that the best one can be replayed during mixing without re-running inference.
- **best_config_index** — (integer) Zero-based index into ``all_config_scores``
  (and into ``target_config_list``) of the config that produced the smallest
  score. Used by the mixing executor to select the right config for each candidate
  without re-running sensitivity analysis.
- **enabled** — (boolean, default ``true``) When ``false``, the candidate is excluded
  from the greedy promotion loop even though its score is recorded. Set this field to
  ``false`` for any layer or subgraph you want to pin at its original quantization
  precision regardless of sensitivity.

.. note::

   The ``results`` array is human-editable: you can open the file in any text
   editor, inspect the sensitivity scores, manually reorder entries, set
   ``"enabled": false`` on specific candidates to exclude them, or remove
   candidates you do not want to consider. The promotion loop respects whatever
   order and ``enabled`` flags are present when the file is loaded. If you edit
   the file, update or clear the ``cache_key`` field so Quark does not discard
   your edits on the next run (any value that differs from the recomputed key
   will cause a cache miss — simply delete the ``cache_key`` field or set it to
   an empty string to force re-analysis, or leave it matching the current key to
   reuse the edited file as-is).

Metric Functions
----------------

The metric controls how to measure a candidate's sensitivity and when the promotion loop stops.
Three styles are supported.

**Built-in named metric** — select one of the four built-in metrics by name via
``metric_default``:

.. list-table::
   :header-rows: 1
   :widths: 12 45 43

   * - Name
     - Description
     - Threshold guidance
   * - ``"l2"`` *(default)*
     - Mean L2 norm of element-wise differences, averaged over all
       (sample, output) pairs.
     - Set to a small positive float, e.g. ``0.05``.
   * - ``"kl"``
     - Mean KL divergence KL(P_float ‖ P_quant) treating each output as an
       unnormalized distribution.
     - Set to a small positive float, e.g. ``0.01``.
   * - ``"cosine"``
     - Mean cosine distance (1 − cosine_similarity). Returns 0 for identical
       outputs, 2 for perfectly opposite vectors.
     - Set to a small positive float, e.g. ``0.02``.
   * - ``"sqnr"``
     - Mean **negative** SQNR (Signal-to-Quantization-Noise Ratio) in dB,
       measured at the **network output** so that error propagation through
       subsequent layers is captured. Higher SQNR means less distortion.
     - Set to a **negative** float, e.g. ``-30`` to stop when SQNR drops
       below 30 dB.
   * - ``"psnr"``
     - Mean **negative** PSNR in dB so the lower-is-better convention is
       preserved.
     - Set to a **negative** float, e.g. ``-20`` to stop when PSNR drops
       below 20 dB.

Example — use KL divergence and stop when it exceeds 0.01:

.. code-block:: python

    amp_config = AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int16Spec(), weight=Int16Spec()),
        metric_default="kl",
        metric_threshold=0.01,
    )

Example — use SQNR and stop when it drops below 30 dB:

.. code-block:: python

    amp_config = AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int16Spec(), weight=Int16Spec()),
        metric_default="sqnr",
        metric_threshold=-30,
    )

Example — use PSNR and stop when it drops below 30 dB:

.. code-block:: python

    amp_config = AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int16Spec(), weight=Int16Spec()),
        metric_default="psnr",
        metric_threshold=-30,
    )

**Custom distance metric** (lower is better) — measures how far the quantized output
drifts from the float output. Takes priority over ``metric_default``:

.. code-block:: python

    import numpy as np

    def my_l1_distance(float_out, quant_out):
        return float(np.mean(np.abs(float_out - quant_out)))

    amp_config = AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int16Spec(), weight=Int16Spec()),
        metric_distance_fn=my_l1_distance,
        metric_threshold=0.05,
    )

**Custom evaluation metric** (higher is better) — computes a quality score such as
top-1 accuracy for the full model. Takes priority over ``metric_default``:

.. code-block:: python

    def my_top1(model_output):
        # model_output is the raw numpy array from inference
        return compute_accuracy(model_output, ground_truth_labels)

    amp_config = AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int16Spec(), weight=Int16Spec()),
        metric_evaluate_fn=my_top1,
        metric_threshold=0.02,   # tolerate up to 2 pp accuracy drop
    )

``metric_distance_fn`` and ``metric_evaluate_fn`` are mutually exclusive. When
both are ``None``, ``metric_default`` (default: ``"l2"``) is used.

Example
-------

.. note::

   For information on accessing AMD Quark ONNX examples, refer to
   :doc:`Accessing ONNX Examples <../onnx_examples>`.
