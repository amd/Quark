Quark ONNX FP8 Quantization Tutorial (Qwen1.5-0.5B)
===================================================

.. container:: alert alert-block alert-info

   NOTE This tutorial can be downloaded for local execution on a Jupyter
   Notebook environment. Click here to download the source file.

This tutorial demonstrates **FP8 (8-bit floating point) Post-Training
Quantization (PTQ)** with Quark ONNX, using the LLM **Qwen1.5-0.5B** as
a running example. Unlike the INT8/INT16 flows shown in the ResNet50
tutorial, FP8 uses a floating-point representation (sign + exponent +
mantissa) that better matches the near-Gaussian weight distributions
found in transformer models.

The tutorial has the following parts:

-  FP8 support status in Quark ONNX
-  Install requirements
-  Export Qwen1.5-0.5B to ONNX
-  Prepare calibration / evaluation data (WikiText-2)
-  How to set up the FP8 quantization config

   -  E4M3FN vs E5M2
   -  Per-tensor vs per-channel weights
   -  Weights-only via the built-in ``WeightsOnly`` option

-  Quantize the model with different FP8 configs
-  Evaluate perplexity and compare

1) FP8 Support Status in Quark ONNX
-----------------------------------

FP8 has two IEEE-style variants defined in the ONNX standard (opset
21+):

+-------------+-----------+---------------------------+---------------+
| Variant     | Layout    | Max finite value          | Best for      |
+=============+===========+===========================+===============+
| **E4M3FN**  | 1 sign, 4 | 448.0                     | Weights &     |
| (``FLOA     | exponent, |                           | activations   |
| T8E4M3FN``) | 3         |                           | (more         |
|             | mantissa  |                           | precision)    |
+-------------+-----------+---------------------------+---------------+
| **E5M2**    | 1 sign, 5 | 57344.0                   | Wide dynamic  |
| (``FL       | exponent, |                           | range (more   |
| OAT8E5M2``) | 2         |                           | range, less   |
|             | mantissa  |                           | precision)    |
+-------------+-----------+---------------------------+---------------+

Key points to understand before quantizing:

-  **Quantizer support** — Quark ONNX computes FP8 scales with a
   symmetric MinMax rule ``scale = absmax / fp8_max`` and stores a
   zero-point of ``0`` in the FP8 dtype. Weight quantization is
   available in both **per-tensor** (one scale per weight tensor) and
   **per-channel** (one scale per output channel) modes. Activation
   quantization is always **per-tensor**.
-  **Weights-only mode** — Quark exposes a built-in ``WeightsOnly``
   extra option. When enabled, only weight initializers receive Q/DQ
   nodes; activations are left untouched in FP16. This is the
   recommended mode for LLMs (see below) and avoids any manual graph
   surgery.
-  **Scale dtype** — With ``QuantizeFP16=True`` and
   ``UseFP32Scale=False``, Quark stores the Q/DQ scales as FP16. Per the
   ONNX spec, the ``DequantizeLinear`` output dtype equals the scale
   dtype, so an FP16 base model stays FP16 end-to-end.
-  **Automatic opset conversion** — FP8 ``DequantizeLinear`` only loads
   in ONNX Runtime at **opset >= 21**. Quark automatically converts a
   lower-opset input model up to opset 21 before inserting FP8 Q/DQ
   nodes, so you no longer need to bump the opset manually. (E4M3FN has
   partial support at opset 19; E5M2 strictly needs >= 21.)
-  **Runtime (ORT) support** — ONNX Runtime’s execution providers (CPU /
   CUDA / ROCm) do **not** provide hardware FP8 compute kernels. FP8
   Q/DQ nodes are executed in software (values are up-cast to FP16/FP32
   before the MatMul). FP8 is therefore primarily a **model-size /
   bandwidth** optimization in ORT today; the compute speedup requires
   an FP8-capable backend such as the TensorRT EP on NVIDIA Ada/Hopper.
-  **QLinearConv fusion caveat (Conv/Gemm activation quantization)** —
   If you quantize *activations* on ``Conv``/``Gemm`` ops (i.e. not
   weights-only), do **not** create the inference session with
   ``ORT_ENABLE_ALL``. At session build time ORT fuses the
   ``Q -> DQ -> Conv`` pattern into a single ``QLinearConv`` kernel,
   which only supports ``int8``/``uint8`` — never FP8. On an FP32 model
   this raises
   ``INVALID_GRAPH: Type 'tensor(float8e5m2)' ... of operator (QLinearConv) is invalid``;
   on an FP16 model it loads but silently produces garbage (**0%
   accuracy**). Fix: set
   ``sess_options.graph_optimization_level = ORT_DISABLE_ALL`` (or
   ``ORT_ENABLE_BASIC``) so the FP8 Q/DQ nodes stay unfused. The
   weights-only recipe below sidesteps this entirely.

Recommended recipe for LLMs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

LLM attention (``softmax(QKᵀ/√d)``) is extremely sensitive to activation
precision — quantizing activations to FP8 scrambles the attention
distribution and destroys accuracy (perplexity in the millions). The
recommended recipe is **weights-only FP8**: set ``WeightsOnly=True`` so
only the weight tensors are quantized and activations stay in FP16.

2) Install the Necessary Python Packages
----------------------------------------

In addition to Quark that must be installed as
`documented <https://quark.docs.amd.com/latest/install.html>`__, extra
packages are required for this tutorial.

.. code:: python

    %pip install torch --index-url https://download.pytorch.org/whl/cpu
    %pip install amd-quark
    %pip install -r ./requirements.txt

3) Export Qwen1.5-0.5B to ONNX
------------------------------

The FP8 flow operates on an ONNX graph. We export the HuggingFace
Qwen1.5-0.5B checkpoint to a FP16 ONNX model using Quark Torch’s
``export_onnx`` helper. The model is exported with a small static
sequence length (``SEQ_LEN=12``) purely to keep this tutorial fast to
run on CPU; production exports would use dynamic shapes and a KV-cache.

The exporter may emit a low opset (e.g. 17). That is fine — Quark
automatically converts the model to opset 21 during FP8 quantization
(Section 1), so **no manual opset bump is required**.

Set ``MODEL_DIR`` to your local Qwen1.5-0.5B checkpoint (or a
HuggingFace repo id).

.. code:: python

    import os
    
    import torch
    
    from quark.torch import export_onnx
    from quark.torch.utils.llm import get_model, get_tokenizer
    
    MODEL_DIR = os.environ.get("QWEN_MODEL_DIR", "Qwen/Qwen1.5-0.5B")
    OUTPUT_DIR = "models"
    SEQ_LEN = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tokenizer = get_tokenizer(MODEL_DIR, max_seq_len=SEQ_LEN, model_type="qwen2")
    model, _ = get_model(MODEL_DIR, data_type="float16", device=DEVICE, attn_implementation="eager", trust_remote_code=True)
    model.eval()
    
    encoded = tokenizer(
        ["Hello, this is a test sentence for ONNX export."],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SEQ_LEN,
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    
    with torch.inference_mode():
        export_onnx(
            model=model,
            output_dir=OUTPUT_DIR,
            input_args=(input_ids,),
            opset_version=17,
            input_names=["input_ids"],
            output_names=["logits"],
            uint4_int4_flag=False,
            dynamo=True,
        )
    
    BASELINE = os.path.join(OUTPUT_DIR, "quark_model.onnx")
    print(f"Exported FP16 baseline: {BASELINE} ({os.path.getsize(BASELINE) / 1e6:.1f} MB)")

4) Prepare Data (WikiText-2)
----------------------------

We use the WikiText-2 dataset for both calibration (FP8 weight
statistics) and evaluation (perplexity). The calibration reader yields
fixed-length token windows matching the model’s static sequence length.

.. code:: python

    import numpy as np
    import onnx
    from datasets import load_dataset
    from onnxruntime.quantization.calibrate import CalibrationDataReader
    from transformers import AutoTokenizer
    
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    
    class WikiTextCalibDataReader(CalibrationDataReader):
        """Yields SEQ_LEN-length token windows from the WikiText-2 train set."""
    
        def __init__(self, limit=8):
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            ids = hf_tokenizer.encode("\n\n".join(d["text"] for d in ds))
            self._samples = []
            for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN):
                self._samples.append({"input_ids": np.array([ids[i : i + SEQ_LEN]], dtype=np.int64)})
                if len(self._samples) >= limit:
                    break
            print(f"  Calibration: {len(self._samples)} windows (seq_len={SEQ_LEN})")
            self._iter = None
    
        def get_next(self):
            if self._iter is None:
                self._iter = iter(self._samples)
            return next(self._iter, None)
    
        def rewind(self):
            self._iter = None
    
        def reset(self):
            self._iter = None

5) How to Set Up the FP8 Quantization Config
--------------------------------------------

FP8 quantization uses the legacy ``QuantizationConfig`` API together
with ORT’s ``QuantType.QFLOAT8E4M3FN``. The config covers the design
axes from Section 1:

-  **``_fp8_quant_type(fp8_type)``** — returns the quant type object.
   ``QuantType`` natively exposes ``QFLOAT8E4M3FN``; for E5M2 we return
   a small stub carrying ``tensor_type = onnx.TensorProto.FLOAT8E5M2``
   (Quark only reads ``.tensor_type``).
-  **``per_channel``** — ``True`` gives one FP8 scale per weight output
   channel; ``False`` gives one scale per tensor.
-  **``WeightsOnly``** — the built-in extra option that quantizes only
   weight tensors and leaves activations in FP16. No manual graph
   post-processing is needed. This is the recommended LLM recipe.

Note that we quantize only ``MatMul`` / ``Gemm`` — the compute-heavy LLM
ops.

.. code:: python

    import types
    
    from onnxruntime.quantization.calibrate import CalibrationMethod
    from onnxruntime.quantization.quant_utils import QuantFormat, QuantType
    
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config.config import Config
    from quark.onnx.quantization.config.legacy import QuantizationConfig
    
    
    def _fp8_quant_type(fp8_type: str = "e4m3"):
        """Return a QuantType-compatible object for the requested FP8 variant.
    
        Any value other than "e5m2" defaults to E4M3FN.
        """
        if fp8_type == "e5m2":
            return types.SimpleNamespace(tensor_type=onnx.TensorProto.FLOAT8E5M2)  # E5M2 stub
        return QuantType.QFLOAT8E4M3FN  # default: E4M3FN (native ORT type)
    
    
    def _fp8_config(fp8_type: str, per_channel: bool, weights_only: bool = True) -> QuantizationConfig:
        quant_type = _fp8_quant_type(fp8_type)
        return QuantizationConfig(
            calibrate_method=CalibrationMethod.MinMax,
            quant_format=QuantFormat.QDQ,
            activation_type=quant_type,
            weight_type=quant_type,
            op_types_to_quantize=["MatMul", "Gemm"],  # the compute-heavy LLM ops
            per_channel=per_channel,
            include_cle=False,
            use_external_data_format=True,
            extra_options={
                "WeightsOnly": weights_only,  # built-in weights-only switch
                "ActivationSymmetric": True,  # FP8 zero-point is always 0
                "QuantizeFP16": True,  # model is already FP16
                "UseFP32Scale": False,  # keep Q/DQ scales in FP16
                "QuantizeBias": False,
            },
        )

Define a single ``quantize()`` helper. Quark auto-converts the opset for
FP8, so this is just: build config, calibrate, run.

.. code:: python

    def quantize(fp8_type: str, per_channel: bool, weights_only: bool = True) -> str:
        gran = "perchannel" if per_channel else "pertensor"
        suffix = "_wo" if weights_only else ""
        out = os.path.join(OUTPUT_DIR, f"qwen_fp8{fp8_type}_{gran}{suffix}.onnx")
        mode = "weights-only" if weights_only else "act+weight"
        print(f"\n=== FP8 {fp8_type.upper()} | {gran} | {mode} ===")
    
        cfg = _fp8_config(fp8_type, per_channel, weights_only)
        quantizer = ModelQuantizer(Config(global_quant_config=cfg))
        quantizer.quantize_model(BASELINE, out, WikiTextCalibDataReader(limit=8))
    
        m = onnx.load(out)
        q = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")
        dq = sum(1 for n in m.graph.node if n.op_type == "DequantizeLinear")
        opset = [o.version for o in m.opset_import if o.domain in ("", "ai.onnx")][0]
        print(f"  [done] {os.path.basename(out)} ({os.path.getsize(out) / 1e6:.1f} MB) | opset={opset} Q={q} DQ={dq}")
        return out

6) Quantize With Different FP8 Configs
--------------------------------------

We produce weights-only FP8 models for both variants and both weight
granularities. With ``WeightsOnly=True``, the output contains only
weight ``DequantizeLinear`` nodes (no activation Q/DQ).

.. code:: python

    model_paths = {}
    for fp8_type in ["e4m3", "e5m2"]:
        for per_channel in [False, True]:
            label = f"FP8 {fp8_type.upper()} {'per-channel' if per_channel else 'per-tensor'} (wo)"
            model_paths[label] = quantize(fp8_type, per_channel, weights_only=True)

7) Evaluation (WikiText-2 Perplexity)
-------------------------------------

Perplexity is computed over sliding ``SEQ_LEN``-token windows (no KV
cache, so context is limited to ``SEQ_LEN`` tokens). The absolute PPL is
high because of the short context — what matters is the **relative** gap
between FP16 and each FP8 model. ``MAX_WINDOWS`` caps the run for a
quick CPU evaluation.

E5M2 models require the runtime to have a float8e5m2
``DequantizeLinear`` kernel; if your ORT build lacks it, that model is
skipped gracefully.

.. code:: python

    import math
    import time
    
    import onnxruntime as ort
    
    MAX_WINDOWS = 200  # increase for a more stable PPL estimate
    STRIDE = SEQ_LEN - 1
    
    test_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    test_ids = hf_tokenizer.encode("\n\n".join(d["text"] for d in test_ds))
    
    
    def log_softmax(x):
        x = x - x.max()
        return x - np.log(np.exp(x).sum())
    
    
    def compute_ppl(path):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            sess = ort.InferenceSession(
                path, sess_options=opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        except Exception as e:
            print(f"  [SKIP] cannot load: {str(e)[:80]}")
            return None
        total_nll, n, i, w = 0.0, 0, 0, 0
        t0 = time.time()
        while i + SEQ_LEN <= len(test_ids) and w < MAX_WINDOWS:
            win = test_ids[i : i + SEQ_LEN]
            logits = sess.run(None, {"input_ids": np.array([win], dtype=np.int64)})[0][0].astype(np.float32)
            for j in range(STRIDE):
                total_nll -= log_softmax(logits[j])[win[j + 1]]
                n += 1
            i += STRIDE
            w += 1
        return math.exp(min(total_nll / n, 30)), time.time() - t0
    
    
    results = {}
    print("[FP16 baseline]")
    results["FP16 baseline"] = compute_ppl(BASELINE)
    for label, path in model_paths.items():
        print(f"[{label}]")
        results[label] = compute_ppl(path)

.. code:: python

    base = results["FP16 baseline"][0]
    print(f"{'Model':<32} {'PPL':>10} {'ΔPPL':>10}")
    print("-" * 54)
    for label, r in results.items():
        if r is None:
            print(f"{label:<32} {'SKIPPED':>10}")
            continue
        ppl = r[0]
        d = "—" if label == "FP16 baseline" else f"{ppl - base:+.2f}"
        print(f"{label:<32} {ppl:>10.2f} {d:>10}")

8) Expected Results and Takeaways
---------------------------------

Representative weights-only results on WikiText-2 (500 windows,
seq_len=12). Absolute PPL is high due to the short-context export; focus
on the relative gap. Exact numbers vary by machine and window count.

========================= ====== ==============
Model                     PPL    ΔPPL vs FP16
========================= ====== ==============
FP16 baseline             209.74 —
FP8 E4M3 per-channel (wo) 221.08 +11.3 (+5.4%)
FP8 E4M3 per-tensor (wo)  222.22 +12.5 (+5.9%)
FP8 E5M2 per-channel (wo) 227.45 +17.6 (+8.4%)
FP8 E5M2 per-tensor (wo)  243.01 +33.1 (+15.8%)
========================= ====== ==============

For contrast, quantizing **activations + weights**
(``WeightsOnly=False``) yields PPL in the millions — the failure case
that motivates the weights-only recipe.

**Takeaways:**

-  **Weights-only is essential for LLMs.** ``WeightsOnly=True`` keeps
   attention in FP16 and brings the degradation from catastrophic
   (×80,000+) down to a few percent.
-  **E4M3FN > E5M2 for weight quantization.** The extra mantissa bit (3
   vs 2) represents near-Gaussian transformer weights more accurately.
   E5M2’s wider exponent range is not needed for weights.
-  **Per-channel > per-tensor.** One scale per output channel isolates
   outlier weights, and the benefit is larger for E5M2 (which has less
   mantissa precision to absorb outliers).
-  **Opset conversion is automatic.** Quark bumps a low-opset input
   model to opset 21 before FP8 QDQ insertion — no manual step needed.
-  **ORT runs FP8 in software.** In ONNX Runtime today FP8 is a
   size/bandwidth win, not a compute win; hardware FP8 acceleration
   needs an FP8-capable backend (e.g. TensorRT EP on Ada/Hopper).
