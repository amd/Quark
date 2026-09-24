Quark ONNX ShapeShifter Tutorial (ResNet50): CLI and ``ShapeShifterYaml``
=========================================================================

.. container:: alert alert-block alert-info

   NOTE This tutorial can be downloaded for local execution on a Jupyter
   Notebook environment. Click here to download the source file.

**ShapeShifter** is Quark’s pass-based graph-transformation framework.
Each *pass* takes a model, applies one transformation (fold BatchNorm,
simplify, align Q/DQ scales, …), and returns the transformed model;
passes are composable and run sequentially. ONNX passes use the
``onnx_`` prefix and split into two stages:

-  **Preprocessing passes** — run on the *float* model **before**
   quantization to clean up and canonicalize the graph (constant
   folding, operator fusion, BatchNorm folding, opset conversion, …).
-  **Postprocessing passes** — run on the *quantized* Q/DQ model
   **after** quantization to adapt it for a deployment target (Q/DQ
   scale alignment, bias-scale correction, XINT8/NPU simulation, …).

This tutorial shows the two ways to drive ShapeShifter ONNX passes,
using ResNet50 as the example model:

1. **The ShapeShifter CLI** (``quark-cli shapeshifter config.yaml``) — a
   standalone, file-in / file-out graph transform, independent of
   quantization.
2. **``ShapeShifterYaml``** — driving the same passes *from inside a
   quantization script*, so preprocessing and postprocessing run
   automatically as part of ``quantize_model``.

For the full pass catalog and framework internals, see `Quark
ShapeShifter <https://quark.docs.amd.com/latest/quark_shapeshifter.html>`__
and `ONNX Model
Passes <https://quark.docs.amd.com/latest/quark_shapeshifter_onnx_passes.html>`__.

1) Install the necessary Python packages
----------------------------------------

In addition to Quark (installed as `documented
here <https://quark.docs.amd.com/latest/install.html>`__), this tutorial
needs a few extra packages. The ShapeShifter **CLI** (``quark-cli``)
loads all of its subcommands at startup, so it also requires Quark’s
optional CLI dependencies (``pip install amd-quark[cli]``); these are
included in ``requirements.txt``.

.. code:: ipython3

    %pip install amd-quark
    %pip install -r ./requirements.txt

2) Download the ResNet50 model
------------------------------

We use ``resnet50-v1-12``, publicly available from the ONNX model zoo.
It ships as opset 12 and contains 53 ``BatchNormalization`` nodes — a
good fit for demonstrating BatchNorm folding and opset conversion.

.. code:: ipython3

    !mkdir -p models
    !wget -O models/resnet50-v1-12.onnx https://github.com/onnx/models/raw/new-models/vision/classification/resnet/model/resnet50-v1-12.onnx

Parse optional command-line arguments (used when this notebook is
executed headless in CI). When run interactively you can ignore these —
sensible defaults are applied.

.. code:: ipython3

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace_dir", default="")
    parser.add_argument("--calib_n_samples", type=int, default=None)
    args, _ = parser.parse_known_args()
    
    FLOAT_MODEL = "models/resnet50-v1-12.onnx"
    # Number of (demo) calibration samples. Small by default so the tutorial runs quickly.
    NUM_CALIB_DATA = args.calib_n_samples if args.calib_n_samples is not None else 8
    print("Float model:", FLOAT_MODEL)
    print("Calibration samples:", NUM_CALIB_DATA)

A small helper to count operator types in an ONNX graph. We use it
throughout to *observe* what each pass does to the graph.

.. code:: ipython3

    from collections import Counter
    
    import onnx
    
    
    def op_counts(model_path: str) -> Counter:
        model = onnx.load(model_path)
        return Counter(node.op_type for node in model.graph.node)
    
    
    def show_counts(model_path: str, keys=("total", "Conv", "BatchNormalization", "QuantizeLinear", "DequantizeLinear")):
        counts = op_counts(model_path)
        total = sum(counts.values())
        row = {"total": total, **{k: counts.get(k, 0) for k in keys if k != "total"}}
        print(model_path)
        print("  " + ", ".join(f"{k}={v}" for k, v in row.items()))
        return row
    
    
    float_counts = show_counts(FLOAT_MODEL)

3) Part A — the ShapeShifter ONNX CLI
-------------------------------------

The CLI runs a list of passes described in a YAML file and writes the
transformed model to disk — no quantization involved. The YAML has three
top-level keys:

-  ``input_model_config`` — the model to load. For ONNX, set
   ``model_type: onnx`` and ``input_model_path``.
-  ``passes`` — an ordered mapping of ``pass_name: { params }``. Pass
   names match the ShapeShifter registry (the filenames under
   ``quark/shapeshifter/passes/``).
-  ``output_model_path`` — where to write the result.

Here we apply three of the most frequently used **preprocessing** passes
for CNNs:

+--------------------+-------------------------------------------------+
| Pass               | What it does                                    |
+====================+=================================================+
| ``onnx_conve       | Upgrades the model opset (ResNet50-v1-12 ships  |
| rt_opset_version`` | as opset 12) so later passes and fusions that   |
|                    | require a newer opset are available.            |
+--------------------+-------------------------------------------------+
| ``onnx_simplify``  | General graph cleanup via ``onnxslim``          |
|                    | (constant folding, Identity removal,            |
|                    | redundant-node elimination).                    |
+--------------------+-------------------------------------------------+
| ``onnx             | Folds each ``BatchNormalization`` into the      |
| _fold_batch_norm`` | weights/bias of the preceding                   |
|                    | ``Conv``/``Gemm``, removing the BN nodes        |
|                    | entirely.                                       |
+--------------------+-------------------------------------------------+

.. code:: ipython3

    cli_yaml = """
    input_model_config:
      model_type: onnx
      input_model_path: models/resnet50-v1-12.onnx
    passes:
      onnx_convert_opset_version:
        target_opset_version: 21
      onnx_simplify:
        simplify: true
      onnx_fold_batch_norm:
        fold_batch_norm: true
    output_model_path: models/resnet50_preprocessed.onnx
    """
    
    with open("shapeshifter_cli.yaml", "w") as f:
        f.write(cli_yaml)
    
    print(cli_yaml)

Run the transform. The CLI prints each pass as it executes
(``Running pass: ...``).

.. code:: ipython3

    !quark-cli shapeshifter shapeshifter_cli.yaml

Compare the graph before and after. The 53 ``BatchNormalization`` nodes
are folded away and the total node count drops, while the 53 ``Conv``
nodes are preserved (the BN parameters are now baked into their
weights/bias).

.. code:: ipython3

    print("Before:")
    show_counts(FLOAT_MODEL)
    print("\nAfter:")
    show_counts("models/resnet50_preprocessed.onnx")

4) Part B — ``ShapeShifterYaml`` inside a quantization script
-------------------------------------------------------------

Instead of running the CLI as a separate step, you can hand ShapeShifter
passes to the quantizer directly through the ``ShapeShifterYaml`` entry
in ``extra_options``. The quantizer then runs them at the right points
in the pipeline:

-  **``preprocess_passes``** run on the float model **before**
   calibration/quantization.
-  **``postprocess_passes``** run on the quantized Q/DQ model **after**
   quantization (in addition to the built-in postprocessing).

Either group may be omitted. Setting ``ShapeShifterYaml`` automatically
forces ``SkipPreprocess=True``, so ShapeShifter fully owns preprocessing
— you don’t need to (and should not) also enable the individual
preprocessing flags.

   The older ``PreprocessYAML`` option is deprecated; it is now a
   preprocess-only alias of ``ShapeShifterYaml``.

When a ``preprocess_passes`` group runs, the quantizer also **saves the
preprocessed float model** (after the preprocess passes, before
quantization) so you can inspect or reuse it. The destination is the
optional ``preprocessed_model_path`` field in the YAML; if you omit it,
the model is written to ``<input_model_name>_preprocessed.onnx`` next to
the input model.

The YAML below uses three common preprocessing passes and two common
postprocessing passes:

+-----------------+---------------+------------------------------------+
| Stage           | Pass          | What it does                       |
+=================+===============+====================================+
| pre             | ``on          | Upgrade opset to 21.               |
|                 | nx_convert_op |                                    |
|                 | set_version`` |                                    |
+-----------------+---------------+------------------------------------+
| pre             | ``on          | Graph cleanup via ``onnxslim``.    |
|                 | nx_simplify`` |                                    |
+-----------------+---------------+------------------------------------+
| pre             | ``onnx_fold   | Fold BatchNorm into Conv/Gemm.     |
|                 | _batch_norm`` |                                    |
+-----------------+---------------+------------------------------------+
| post            | ``onnx_       | Align Q/DQ scale & zero-point      |
|                 | align_scale`` | across pooling ops (``MaxPool``,   |
|                 |               | ``GlobalAveragePool``) so a        |
|                 |               | downstream compiler sees matching  |
|                 |               | quantization parameters on their   |
|                 |               | inputs and outputs.                |
+-----------------+---------------+------------------------------------+
| post            | ``onnx_adjust | Ensure each int32                  |
|                 | _bias_scale`` | ``Conv``/``Gemm`` bias scale       |
|                 |               | equals                             |
|                 |               | ``                                 |
|                 |               | activation_scale × weight_scale``, |
|                 |               | preventing accuracy loss from      |
|                 |               | inconsistent bias scaling.         |
+-----------------+---------------+------------------------------------+

.. code:: ipython3

    shapeshifter_yaml = """
    # preprocess_passes: applied to the FLOAT model, before quantization.
    preprocess_passes:
      onnx_convert_opset_version:
        target_opset_version: 21
      onnx_simplify:
        simplify: true
      onnx_fold_batch_norm:
        fold_batch_norm: true
    
    # Optional: where to save the float model AFTER the preprocess passes (before
    # quantization). If omitted, it is saved as <input_model_name>_preprocessed.onnx
    # next to the input model.
    preprocessed_model_path: models/resnet50_preprocessed_by_yaml.onnx
    
    # postprocess_passes: applied to the QUANTIZED (Q/DQ) model, after quantization.
    postprocess_passes:
      onnx_align_scale:
        align_scale: [MaxPool, GlobalAveragePool]
      onnx_adjust_bias_scale:
        adjust_bias_scale: true
    """
    
    with open("shapeshifter_passes.yaml", "w") as f:
        f.write(shapeshifter_yaml)
    
    print("ShapeShifter passes for quantization (written to shapeshifter_passes.yaml):")
    print(shapeshifter_yaml)

Now quantize ResNet50 with the XINT8 configuration and point it at the
YAML.

For a self-contained demo we calibrate with **random** data so the
notebook runs without an ImageNet download. This exercises the
ShapeShifter pre/postprocessing pipeline end-to-end but will **not**
produce a meaningful accuracy number — replace
``RandomCalibrationDataReader`` with a reader over real preprocessed
images for that. See the `ResNet50 on Ryzen AI
tutorial <https://quark.docs.amd.com/latest/tutorials/onnx/ryzen_ai/resnet50/onnx_ryzen_ai_resnet50_tutorial.html>`__
for a full accuracy evaluation on the ImageNet validation set.

.. code:: ipython3

    import numpy as np
    from onnxruntime.quantization import CalibrationDataReader
    
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import Config, get_default_config
    
    
    class RandomCalibrationDataReader(CalibrationDataReader):
        # Feeds random NCHW tensors as calibration data (demo only).
    
        def __init__(self, input_name: str, num_samples: int = 8, shape=(1, 3, 224, 224)):
            self.input_name = input_name
            self._data = [{input_name: np.random.rand(*shape).astype(np.float32)} for _ in range(num_samples)]
            self._iter = iter(self._data)
    
        def get_next(self):
            return next(self._iter, None)
    
        def rewind(self):
            self._iter = iter(self._data)
    
    
    # First graph input name of the model (ResNet50-v1-12 calls it "data").
    input_name = onnx.load(FLOAT_MODEL, load_external_data=False).graph.input[0].name
    calib_reader = RandomCalibrationDataReader(input_name, num_samples=NUM_CALIB_DATA)
    
    # Start from the XINT8 (power-of-two scale) CNN config and hand graph transforms to ShapeShifter.
    quant_config = get_default_config("XINT8")
    quant_config.extra_options["ShapeShifterYaml"] = "shapeshifter_passes.yaml"
    
    QUANT_MODEL = "models/resnet50_xint8_shapeshifter.onnx"
    quantizer = ModelQuantizer(Config(global_quant_config=quant_config))
    quantizer.quantize_model(FLOAT_MODEL, QUANT_MODEL, calib_reader)
    print("Quantized model saved to:", QUANT_MODEL)

In the log above you can see the preprocessing passes run before
calibration (``Running pass: onnx_convert_opset_version`` →
``onnx_simplify`` → ``onnx_fold_batch_norm``) and the postprocessing
passes run after quantization (``Running pass: onnx_align_scale`` →
``onnx_adjust_bias_scale``).

Confirm Q/DQ insertion in the quantized model.

.. code:: ipython3

    onnx.checker.check_model(onnx.load(QUANT_MODEL))
    show_counts(QUANT_MODEL)

5) Summary and expected results
-------------------------------

Running the two flows on ``resnet50-v1-12`` produces (numbers may vary
slightly by environment):

**Part A — CLI preprocessing** (``models/resnet50_preprocessed.onnx``):

====================== =========== ===================
\                      Float model After preprocessing
====================== =========== ===================
Total nodes            175         122
``BatchNormalization`` 53          0
``Conv``               53          53
====================== =========== ===================

**Part B — quantization with ``ShapeShifterYaml``**
(``models/resnet50_xint8_shapeshifter.onnx``): the float graph is
preprocessed, quantized to XINT8, and postprocessed — the result is a
Q/DQ model with ~74 ``QuantizeLinear`` / ~182 ``DequantizeLinear`` nodes
and the 53 ``Conv`` nodes intact.

**Key takeaways:**

-  Use the **CLI** (``quark-cli shapeshifter config.yaml``) for a
   standalone, file-in/file-out graph transform — handy for inspecting
   or preparing a model outside the quantizer.
-  Use **``ShapeShifterYaml``** to run the same passes as an integrated
   part of quantization; the ``preprocess_passes`` group runs on the
   float model and the ``postprocess_passes`` group runs on the
   quantized model. Setting it forces ``SkipPreprocess=True``. The
   preprocessed float model is saved automatically to
   ``preprocessed_model_path`` (or
   ``<input_model_name>_preprocessed.onnx``).
-  Pick passes to match your graph and target: ``onnx_fold_batch_norm``
   / ``onnx_simplify`` / ``onnx_convert_opset_version`` are
   near-universal for CNN preprocessing, while ``onnx_align_scale`` /
   ``onnx_adjust_bias_scale`` are common postprocessing steps for NPU /
   XINT8 deployment.

**Further reading:**

-  `Quark
   ShapeShifter <https://quark.docs.amd.com/latest/quark_shapeshifter.html>`__
   — framework overview, CLI/API usage, and how to add new passes.
-  `ONNX Model
   Passes <https://quark.docs.amd.com/latest/quark_shapeshifter_onnx_passes.html>`__
   — the full catalog of preprocessing and postprocessing passes and
   their parameters.
