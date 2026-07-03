.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Using Quark Agent Skills for ONNX (Claude Code)
===============================================

.. note::

    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of
    reference. When you encounter the term "Quark" without the "AMD" prefix, it refers to the AMD Quark
    quantizer unless otherwise stated.

The AMD Quark distribution bundles a collection of **agent skills** for
`Claude Code <https://claude.com/claude-code>`__. Instead of recalling the exact ``ModelQuantizer``
arguments, calibration boilerplate, and execution-provider settings, you simply state what you want —
*"quantize my resnet50.onnx to XINT8"* — and Claude Code picks the matching skill, examines the model,
prepares the run, and carries it out once you give the go-ahead.

Everything documented here is specific to the **ONNX** backend. Read it as a practical how-to for end
users: the internal design and layering of the skills are out of scope and you do not need to know them
to get work done.

Prerequisites
-------------

- `Claude Code <https://claude.com/claude-code>`__ installed and available on your machine.
- A local clone of this repository. Claude Code **discovers the skills automatically** from
  ``.claude/skills/`` the moment it is launched from the repository root, so there is nothing to register
  by hand.
- A working AMD Quark install plus a matching ONNX Runtime build. If you are unsure, let Claude Code
  handle the setup for you (see :ref:`onnx-skills-common-tasks`), or walk through :doc:`../install`.

How to invoke a skill
---------------------

You can start a skill in either of two ways:

1. **State the task in natural language.** Claude Code interprets your request and dispatches the right
   skill on its own. For example:

   .. code-block:: text

      Quantize my resnet50.onnx to XINT8 and validate the result.

2. **Name the skill directly** with a slash command whenever you already know which one you need:

   .. code-block:: text

      /quark-onnx-ptq

Which skill runs is decided by the **kind of input you hand over**. Anything that signals ONNX work — a
``.onnx`` file (optionally paired with an ``.onnx_data`` external-weights sidecar), code that calls
``quantize_static`` / ``ModelQuantizer``, or a reference to an ONNX Runtime execution provider — is
served by the ``quark-onnx-*`` skills on this page. A Hugging Face repo id or a ``config.json`` +
``*.safetensors`` checkpoint goes to the PyTorch skills instead. Quark keeps the two backends strictly
apart, and whenever the intent is unclear (for instance, *"quantize my model"* with no file named) it
checks with you first.

ONNX skills at a glance
-----------------------

The table below summarizes the ONNX skills together with the two backend-neutral helpers that most users
reach for first. You will rarely call any of them by name — a plain description of your goal is normally
enough — but the example phrases show what each skill listens for.

+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| Skill                                  | What it does                                             | Say this to trigger                                    |
+========================================+==========================================================+========================================================+
| ``quark-env-preflight``                | Surveys your OS, Python, GPU, and CUDA / ROCm setup      | "check my environment", "what GPU do I have"           |
|                                        | before anything is installed or quantized.               |                                                        |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-install``                      | Sets up or confirms the ``amd-quark`` package and the    | "install Quark", "pip install amd-quark"               |
|                                        | dependencies it relies on.                               |                                                        |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-onnx-install``                 | Installs and verifies the correct ``onnxruntime`` /      | "install onnxruntime", "onnxruntime-gpu vs             |
|                                        | ``onnxruntime-gpu`` / ROCm build and matching ``onnx``   | onnxruntime", "onnx version mismatch"                  |
|                                        | package for your accelerator.                            |                                                        |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-onnx-model-intake``            | Inspects a ``.onnx`` model (opset / IR version, I/O      | "analyze my ONNX model", "what opset is this", "is my  |
|                                        | shapes and dtypes, op-type histogram, quantizable-op     | model NPU-compatible", "does my model already have     |
|                                        | count, >2 GB external-data check) and assesses target    | QDQ"                                                   |
|                                        | compatibility (CPU / CUDA / ROCm / AMD NPU).             |                                                        |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-onnx-ptq``                     | Runs the full ONNX PTQ pipeline: intake, planning,       | "quantize my .onnx", "quantize yolov8/resnet50 with    |
|                                        | calibration-script generation, manifest, and confirmed   | XINT8/A8W8/BFP16"                                      |
|                                        | execution for schemes such as XINT8, A8W8, BFP16, and    |                                                        |
|                                        | MXFP\*.                                                  |                                                        |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-onnx-autosearch-pro``          | Drives ``quark.onnx.AutoSearchPro`` (Optuna-based        | "auto search", "tune quantization", "find best quant   |
|                                        | hyperparameter search) to find the best activation /     | config", "ADVANCED_SEARCH / XINT8_SEARCH / A8W8_SEARCH |
|                                        | weight spec, calibration method, and tuning params.      | / A16W8_SEARCH for my .onnx"                           |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-onnx-result-validator``        | Inspects a quantized ``model.onnx`` (and                 | "validate ONNX quantization result", "verify ONNX      |
|                                        | ``model.onnx_data``) to confirm QDQ insertion and        | initializers", "did QDQ insertion happen"              |
|                                        | initializer byte-identity on non-quantized tensors.      |                                                        |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+
| ``quark-onnx-debug``                   | Diagnoses failed install, calibration, quantization,     | "Quark ONNX error", "quantize_static failed",          |
|                                        | custom-op compilation, or export attempts.               | "CUDAExecutionProvider not available", "custom op      |
|                                        |                                                          | library load failed"                                   |
+----------------------------------------+----------------------------------------------------------+--------------------------------------------------------+

Walkthrough: quantize an ONNX model end to end
----------------------------------------------

In this example we take ``resnet50.onnx``, quantize it to ``XINT8``, and verify the output. Only the
lines prefixed with ``You`` are things you type; Claude Code takes care of the rest.

**1. (Optional) Confirm your environment is ready.**

.. code-block:: text

   You: Check my environment and make sure Quark is ready to quantize an ONNX model.

Claude Code invokes ``quark-env-preflight`` (pulling in ``quark-install`` / ``quark-onnx-install`` to
fill any gaps), then summarizes your GPU, the CUDA / ROCm stack it detected, the ``onnxruntime`` build in
place, and whether ``amd-quark`` imports without errors.

**2. (Optional) Analyze the model.**

.. code-block:: text

   You: Analyze resnet50.onnx and tell me whether it can be quantized.

This triggers ``quark-onnx-model-intake``. The skill returns the opset / IR version, the shapes and
dtypes of every input and output, a histogram of op types, the count of quantizable ops, whether
external-data crosses the 2 GB threshold, whether QDQ nodes are already in the graph, and the deployment
targets (CPU / CUDA / ROCm / AMD NPU) the model supports.

**3. Ask for the quantization.**

.. code-block:: text

   You: Quantize resnet50.onnx to XINT8.

Here Claude Code hands off to ``quark-onnx-ptq``. After studying the model it lays out a plan plus the
precise calibration script it intends to run — covering the calibration data, the quant scheme, the
chosen execution provider, and the output location — and then pauses for your approval.

**4. Confirm execution.**

Because quantization is an expensive step, the skill **holds for your explicit confirmation** before it
proceeds. Once you approve, it runs the script and tells you where the quantized model was written.

**5. Validate the result.**

.. code-block:: text

   You: Validate the quantized model.

This routes to ``quark-onnx-result-validator``, which checks that the output is well-formed: it looks for
QDQ / ``com.amd.quark`` custom ops, confirms the auxiliary files were copied consistently, compares model
metadata after dropping quantization-only opset entries, and verifies that the non-quantized initializers
are byte-for-byte unchanged.

When a single fixed scheme is not enough and you want to push accuracy as far as it will go, ask for an
automated search instead:

.. code-block:: text

   You: Auto search the best quant config for resnet50.onnx targeting XINT8.

That request goes to ``quark-onnx-autosearch-pro``, which performs an Optuna-driven exploration across
activation / weight specs, calibration methods, and tuning knobs (CLE, AdaRound / AdaQuant, FastFinetune),
drawing on built-in presets such as ``ADVANCED_SEARCH``, ``XINT8_SEARCH``, ``A8W8_SEARCH``, and
``A16W8_SEARCH``.

.. _onnx-skills-common-tasks:

Common tasks and which skill handles them
-----------------------------------------

- **Check hardware / readiness** → ``quark-env-preflight``
- **Install Quark** → ``quark-install``
- **Install / fix the ONNX Runtime build** → ``quark-onnx-install``
- **See whether a model is supported and what will be quantized** → ``quark-onnx-model-intake``
- **Quantize a .onnx model (XINT8 / A8W8 / BFP16 / MXFP\*)** → ``quark-onnx-ptq``
- **Find the best quant config automatically** → ``quark-onnx-autosearch-pro``
- **Check a quantized model is structurally correct** → ``quark-onnx-result-validator``
- **Diagnose a failed install / calibration / quantization run** → ``quark-onnx-debug``

There is no need to commit this table to memory: describe what you are after and Claude Code selects the
skill. It is laid out here only so the routing stays transparent.
