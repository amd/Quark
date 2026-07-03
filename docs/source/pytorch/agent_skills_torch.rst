.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Using Quark Agent Skills for PyTorch (Claude Code)
==================================================

.. note::

    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of
    reference. When you encounter the term "Quark" without the "AMD" prefix, it refers to the AMD Quark
    quantizer unless otherwise stated.

AMD Quark ships a set of **agent skills** for `Claude Code <https://claude.com/claude-code>`__. They let
you drive the Quark PyTorch quantization flow with plain-language requests instead of memorizing CLI
flags and script layouts. You describe the goal — *"quantize Qwen3-8B to FP8"* — and Claude Code
routes to the right skill, inspects your model, plans the run, and (with your confirmation) executes it.

This page covers the **PyTorch / Hugging Face** skills only. It is a usage guide for end users; you do
not need to understand how the skills are built or layered internally to use them.

Prerequisites
-------------

- `Claude Code <https://claude.com/claude-code>`__ installed and running.
- This repository checked out locally. The skills live under ``.claude/skills/`` and are
  **auto-discovered** by Claude Code when you start it from the repository root — there is no extra
  registration step.
- AMD Quark and a compatible PyTorch build installed. If you are not sure, just ask Claude Code to set
  it up (see :ref:`torch-skills-common-tasks`), or follow :doc:`../install`.

How to invoke a skill
---------------------

There are two ways to trigger a skill:

1. **Describe the task in natural language.** Claude Code reads your intent and routes to the matching
   skill automatically. For example:

   .. code-block:: text

      Quantize Qwen/Qwen3-8B to FP8 and validate the result.

2. **Call a skill explicitly** with a slash command when you already know which one you want:

   .. code-block:: text

      /quark-torch-ptq

Skills are routed by **input artifact type**. A Hugging Face repo id, a ``config.json`` +
``*.safetensors`` checkpoint, or a ``transformers`` / ``torch`` workflow routes to the
``quark-torch-*`` skills described here. A ``.onnx`` file routes to the ONNX skills instead — the two
backends are never silently mixed. If your request is ambiguous (for example, *"quantize my model"*
with no path), Claude Code asks before routing.

Torch skills at a glance
------------------------

The table below lists the PyTorch-side skills plus the two backend-agnostic helpers most users need
first. You rarely call these by name — describing the task is usually enough — but the trigger phrases
show what each one responds to.

+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| Skill                                  | What it does                                             | Say this to trigger                                    |
+========================================+==========================================================+========================================================+
| ``quark-env-preflight``                | Reports your OS, Python, GPU, and CUDA / ROCm state      | "check my environment", "what GPU do I have"           |
|                                        | before any install or quantization step.                 |                                                        |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-install``                      | Installs or verifies the ``amd-quark`` package and its   | "install Quark", "pip install amd-quark"               |
|                                        | core dependencies.                                       |                                                        |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-torch-model-intake``           | Inspects a Hugging Face / safetensors checkpoint and     | "analyze my model", "is this model supported"          |
|                                        | reports architecture, quant targets, and risks.          |                                                        |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-torch-ptq``                    | Runs the full PTQ pipeline: intake, planning, script     | "quantize my model", "run PTQ", "quantize Qwen3        |
|                                        | generation, and optional execution. Stops at the         | with FP8/INT4"                                         |
|                                        | quantized output.                                        |                                                        |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-torch-llm-ptq-eval``           | The full lifecycle: quantize, then validate the output,  | "quantize and validate", "quantize and evaluate", "PTQ |
|                                        | then run opt-in accuracy evaluation in one flow.         | with accuracy check"                                   |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-torch-result-validator``       | Checks an exported model for structural correctness      | "validate quantization result", "verify exported       |
|                                        | (config diff, byte-identity on excluded tensors).        | weights"                                               |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-torch-llm-eval``               | Runs LLM accuracy evaluation (perplexity, ``lm-eval``    | "evaluate this model", "run gsm8k/mmlu", "measure      |
|                                        | tasks) on the quantized model.                           | perplexity"                                            |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-torch-export``                 | Exports a quantized run to Hugging Face safetensors,     | "export model", "convert to GGUF"                      |
|                                        | GGUF, or ONNX.                                           |                                                        |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-torch-file2file-quantization`` | Low-memory file-to-file quantization for very large LLMs | "run file2file quantization", "quantize without        |
|                                        | that cannot be loaded whole.                             | loading the model"                                     |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+

Walkthrough: quantize a Hugging Face LLM end to end
---------------------------------------------------

This example quantizes ``Qwen/Qwen3-8B`` to FP8 and checks the result. You only type
the requests in the ``You`` blocks; Claude Code drives the rest.

**1. (Optional) Confirm your environment is ready.**

.. code-block:: text

   You: Check my environment and make sure Quark is ready to quantize an 8B model.

Claude Code runs ``quark-env-preflight`` (and ``quark-install`` if anything is missing), then reports
your GPU, the detected CUDA / ROCm stack, and whether ``amd-quark`` imports cleanly.

**2. Ask for the quantization.**

.. code-block:: text

   You: Quantize Qwen/Qwen3-8B to FP8.

Claude Code routes to ``quark-torch-ptq``. It first inspects the checkpoint (architecture, layer count,
which modules are quantization targets, which to exclude such as ``lm_head``), then proposes a plan and
the exact command it intends to run — for example:

.. code-block:: bash

   python3 quantize_quark.py --model_dir Qwen/Qwen3-8B \
                             --quant_scheme fp8 \
                             --output_dir qwen3-8b-fp8

**3. Confirm execution.**

Quantization is a high-cost step, so the skill **waits for your explicit approval** before running.
After you confirm, it executes the command and reports where the quantized model was written
(``--output_dir``).

**4. Validate (and optionally evaluate).**

If you had asked for *"quantize and validate"* or *"quantize and evaluate"* up front, Claude Code would
have routed to ``quark-torch-llm-ptq-eval`` instead, which chains all three stages automatically:

- **Quantize** — delegates to the same ``quark-torch-ptq`` pipeline.
- **Validate** — runs ``quark-torch-result-validator`` to confirm the export is structurally sound
  (config diff, byte-identity on excluded tensors).
- **Evaluate** *(opt-in)* — runs ``quark-torch-llm-eval`` for perplexity or ``lm-eval`` tasks such as
  ``gsm8k`` / ``mmlu``.

  .. code-block:: text

     You: Quantize Qwen/Qwen3-8B to INT4 with AWQ, validate it, and report mmlu.

**Use ``quark-torch-ptq`` when** you only want the quantized model. **Use ``quark-torch-llm-ptq-eval``
when** you also want the output checked and its accuracy measured.

.. _torch-skills-common-tasks:

Common tasks and which skill handles them
-----------------------------------------

- **Check hardware / readiness** → ``quark-env-preflight``
- **Install Quark** → ``quark-install``
- **See whether a model is supported and what will be quantized** → ``quark-torch-model-intake``
- **Quantize only (stop at the quantized output)** → ``quark-torch-ptq``
- **Quantize, validate, and measure accuracy in one flow** → ``quark-torch-llm-ptq-eval``
- **Quantize a model too large to load whole** → ``quark-torch-file2file-quantization``
- **Export an existing run to safetensors / GGUF / ONNX** → ``quark-torch-export``
- **Check an exported model is structurally correct** → ``quark-torch-result-validator``
- **Run accuracy eval on a quantized model** → ``quark-torch-llm-eval``

You do not have to memorize this list: describe the goal and Claude Code picks the skill. The mapping is
here so you know what is happening under the hood.
