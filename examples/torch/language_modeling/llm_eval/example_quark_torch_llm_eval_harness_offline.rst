LM-Evaluation Harness (Offline)
===============================

We provide a multi-step flow to run LM-Evaluation Harness metrics offline for ONNX models. Offline mode is used to evaluate models generations on specific hardware (i.e., NPUs). The offline mode is invoked through ``llm_eval.py --mode offline``. Currently, only the below generation tasks are supported in offline mode.

Supported Tasks
---------------

[``gsm8k``, ``tinyGSM8k``]

Step-by-Step Process
--------------------

Below are the steps on how to use the offline mode.
Please make sure ``--num_fewshot`` is set to 0 to allow for fair comparisons from
OGA model generations.


1. Retrieve dataset from LM-Eval-Harness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``--retrieve_dataset`` to save dataset inputs.json and references.json.
Example shown with 20 samples of gsm8k:

.. code-block:: bash

    python llm_eval.py \
        --mode offline \
        --retrieve_dataset \
        --tasks gsm8k \
        --limit 20 \
        --num_fewshot 0

2. Export pre-trained Model-Of-Interest to ONNX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use OGA Model Builder to save ONNX Pretrained Model.
See `here <https://github.com/microsoft/onnxruntime-genai/tree/main/examples/python>`_ for how to use OGA Model Builder.

3. Retrieve OGA references for pre-trained ONNX Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``--oga_references`` to save the OGA references for a particular pre-trained model.
Example shown with 20 samples of gsm8k for pre-trained Phi3.5-mini-instruct ONNX Model:

.. code-block:: bash

    python llm_eval.py \
        --mode offline \
        --oga_references \
        --inputs [path to inputs.json] \
        --import_model_dir [path to Phi3.5-mini-instruct ONNX Model] \
        --import_file_format onnx_format \
        --tasks gsm8k \
        --limit 20 \
        --num_fewshot 0 \
        --eor "<EOR>"

4. Get Baseline Evaluation Scores on pre-trained ONNX Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``--eval_mode`` to compare the pre-trained model's references to the dataset references.
Example shown with comparing Phi3.5-mini-instruct ONNX model references to GSM8k references.

.. code-block:: bash

    python llm_eval.py \
        --mode offline \
        --eval_mode \
        --outputs_path [path to Phi3.5-mini-instruct OGA references.txt] \
        --tasks gsm8k \
        --limit 20 \
        --num_fewshot 0 \
        --eor "<EOR>"

5. Evaluate an optimized ONNX Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now use ``--eval_mode`` to compare an optimized model to the dataset references.
Example shown with comparing a quantized Phi3.5-mini-instruct ONNX model predictions to GSM8k references.

.. code-block:: bash

    python llm_eval.py \
        --mode offline \
        --eval_mode \
        --outputs_path [path to quantized model predictions.txt] \
        --tasks gsm8k \
        --limit 20 \
        --num_fewshot 0 \
        --eor "<EOR>"

Note: predictions.txt should follow the same format as references.txt from (4). This means, each model output must
be separated by a end-of-response delimiter such as ``"<EOR>"``. See example below of the formatting:

::

    This would be the first model output.
    <EOR>
    This would be the second model output
    <EOR>

1. Compare Scores from Step 4 and Step 5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute the percent error between Step 4 and 5 to understand how the quantized model
compares to the original pre-trained model.

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
