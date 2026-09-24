.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Using Quark Agent Skills for PyTorch
====================================

AMD Quark provides Agent Skills for Claude Code, Cursor, and Codex.
The skills turn plain-language PyTorch and Hugging Face requests into planned, confirmation-gated Quark workflows.

Prerequisites
-------------

- Install and start Claude Code, Cursor, or Codex.
- For the complete skill suite, check out this repository and start the agent from the repository root.
- Install the runtime required by the selected workflow as described in :ref:`torch-skills-runtime`.

Discovery and invocation
------------------------

The canonical skill tree is the repository-root ``skills/`` directory.
Claude Code discovers it through real-file adapters in ``.claude/skills``, while Cursor and Codex use the adapters in ``.agents/skills``.

The self-contained public entries are ``quark-install``, ``quark-torch-ptq``, and ``quark-torch-quant-perf``.
``quark-install`` and ``quark-torch-ptq`` can be copied independently.
A copied ``quark-torch-quant-perf`` entry supports automatic search, while fixed ``--quant-strategy`` also requires a Quark checkout containing ``skills/quark-torch-ptq``.
The other 16 public entries delegate to ``skills/_legacy_impl/`` and must remain with that tree.
The ``amd-quark`` wheel bundles the complete skill tree; after installation, run ``quark-skills install --agent {claude-code,cursor,codex,all}`` to copy it into a workspace.
The command targets the current directory by default, and ``--target <path>`` selects another existing project directory.

Describe the task in natural language:

.. code-block:: text

   Quantize Qwen/Qwen3-8B to FP8 and validate the result.

You can also name the public entry:

.. code-block:: text

   Use quark-torch-ptq to quantize Qwen/Qwen3-8B to FP8.

Claude Code may expose the entry as ``/quark-torch-ptq``.
Route Hugging Face repository IDs, ``config.json`` plus SafeTensors checkpoints, and ``torch`` or ``transformers`` workflows to Torch skills.
If the model format is ambiguous, confirm it before routing, and never send an ONNX model through a Torch workflow.

Available skills
----------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Skill
     - Purpose
   * - ``quark-env-preflight``
     - Report OS, Python, accelerator, and container facts before installation or quantization.
   * - ``quark-install``
     - Install or verify ``amd-quark`` and the selected runtime capabilities.
   * - ``quark-torch-ptq``
     - Run confirmed end-to-end PTQ for a PyTorch or Hugging Face LLM and verify the quantized output.
   * - ``quark-torch-quant-perf``
     - Run managed quantization with an accuracy gate, optional throughput and performance optimization, and final reports.
   * - ``quark-torch-install``
     - Install or verify the PyTorch build that matches the accelerator.
   * - ``quark-torch-model-intake``
     - Inspect a Hugging Face or SafeTensors model and produce facts for PTQ planning.
   * - ``quark-torch-result-validator``
     - Validate quantized SafeTensors, auxiliary files, configuration, and excluded tensor identity.
   * - ``quark-torch-llm-eval``
     - Evaluate LLM accuracy on AMD ROCm through supported serving and benchmark frameworks.
   * - ``quark-torch-file2file-quantization``
     - Quantize very large sharded SafeTensors checkpoints without loading the whole model.
   * - ``quark-torch-shrink-model``
     - Create a small representative SafeTensors model for faster debugging and workflow tests.

.. _torch-skills-runtime:

Runtime requirements
--------------------

``quark-torch-ptq`` requires ``amd-quark[cli]``, ``datasets``, and a separately installed PyTorch build that matches the accelerator.
Those packages provide ``quark-cli torch-llm-ptq`` and its runtime dependencies, but they do not install the skill directory.

``quark-torch-quant-perf`` requires ``amd-quark[quant_perf]`` plus its documented accelerator, vLLM, and optional backend/tool stack.
Automatic search does not require a Quark source checkout solely to locate a PTQ skill.
Fixed ``--quant-strategy`` requires a checkout containing ``skills/quark-torch-ptq``; Quant-Perf discovers one from an editable package or the current directory and its ancestors, and ``QUARK_ROOT`` selects one explicitly when needed.

PTQ workflow
------------

The self-contained ``quark-torch-ptq`` workflow always uses these four checkpoints:

1. **Model analysis:** inspect configuration without loading model weights, write ``model_analysis.json``, and ask the user to accept or correct the findings.
2. **Quantization plan:** write and validate ``quant_plan.json``, then ask the user to confirm the scheme, exclusions, algorithms, and calibration choices.
3. **Execution:** write ``run_manifest.yaml``, show the exact command and affected paths, and wait for explicit approval before running PTQ.
4. **Verification:** inspect the output files, update the manifest with observed status, and ask the user to accept the verified result.

A typical execution command is:

.. code-block:: bash

   quark-cli torch-llm-ptq \
     --model_dir "Qwen/Qwen3-8B" \
     --output_dir "$PWD/qwen3-8b-fp8" \
     --quant_scheme fp8 \
     --device cuda \
     --no_trust_remote_code \
     --skip_evaluation

Plan approval is not execution approval, and an exit code alone is not proof that the expected model, configuration, and tokenizer files were produced.

Post-quantization checks
------------------------

Validation and accuracy evaluation are separate follow-up skills rather than a combined recipe.
Use ``quark-torch-result-validator`` to inspect the quantized output, then use ``quark-torch-llm-eval`` when ROCm accuracy evaluation is requested.
Each follow-up applies its own environment checks, execution gates, and recovery rules.
