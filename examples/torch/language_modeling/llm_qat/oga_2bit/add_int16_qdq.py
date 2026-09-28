#!/usr/bin/env python3
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Add static INT16 activation Q-DQ to an OGA-translated 2-bit (``MatMulNBits``) ONNX model.

An OGA (onnxruntime-genai) builder run translates a standard-Quark uint2 model into an
ONNX graph whose weight matmuls are ``MatMulNBits`` (2-bit weights, float zero-point).
This tool adds **static QDQ activation quantization** (UINT16 activations, UINT8 weights)
on top, so the model becomes W2A16 for deployment (e.g. AMD Ryzen AI NPU transformer),
while leaving the packed 2-bit weights untouched.

How the ``MatMulNBits`` activation gets quantized
-------------------------------------------------
``MatMulNBits`` (``com.microsoft``) is *excluded* from ``op_types_to_quantize`` so the
quantizer never rewrites its inputs (its weight stays packed uint8 2-bit). Because every
*other* op is quantized, a Q/DQ pair is inserted on the **output of the op that feeds**
``MatMulNBits`` input 0 -- i.e. the activation flowing into ``MatMulNBits`` is covered,
while the weight is not. ``GatherBlockQuantized`` (packed embedding) and
``GroupQueryAttention`` (attention kernel) are excluded for the same reason.

Backends
--------
* ``quark`` (default): ``quark.onnx.quantize_static`` -- the recommended path. It is a
  superset of the ORT quantizer and adds AMD Ryzen AI options (NPU-transformer mode,
  power-of-two scales, INT16 calibration). Requires ``optimize_model=False`` for the
  OGA GQA/MatMulNBits graph (ORT's post-quant shape-inference reload otherwise fails).
* ``ort``: ``onnxruntime.quantization.quantize_static`` -- a simpler fallback with no
  Quark dependency and free (non-power-of-two) scales.

Usage
-----
    python add_int16_qdq.py --input-model <oga_out_dir> --output-dir <qdq_out_dir>
    python add_int16_qdq.py -i <dir> -o <dir> --backend ort
    python add_int16_qdq.py -i <dir> -o <dir> --npu-transformer   # Ryzen AI (quark)

``--input-model`` may be a directory containing ``model.onnx`` + ``genai_config.json``
(the OGA output layout) or a direct path to a ``model.onnx``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import onnx

_SKIP_QUANT_TYPES = {"MatMulNBits", "GatherBlockQuantized", "GroupQueryAttention"}
_AUX_FILES = (
    "genai_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
    "config.json",
)
_ONNX_TO_NP = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.FLOAT16: np.float16,
}


def _resolve_paths(input_model: str, output_dir: str):
    """Resolve the input model dir, the ``model.onnx`` path, and the output ``model.onnx`` path.

    ``input_model`` may be an OGA output directory or a direct ``model.onnx`` path; the output
    directory is created if missing.
    """
    if os.path.isdir(input_model):
        model_dir = input_model
        model_path = os.path.join(input_model, "model.onnx")
    else:
        model_dir = os.path.dirname(os.path.abspath(input_model))
        model_path = input_model
    os.makedirs(output_dir, exist_ok=True)
    return model_dir, model_path, os.path.join(output_dir, "model.onnx")


def _read_decoder_meta(model_dir: str):
    """Read decoder geometry (num_hidden_layers, num_key_value_heads, head_size) from genai_config.json.

    These drive the empty KV-cache tensors fed to the model during calibration.
    """
    cfg_path = os.path.join(model_dir, "genai_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"genai_config.json not found in {model_dir}. Point --input-model at the OGA output directory."
        )
    with open(cfg_path) as f:
        dec = json.load(f)["model"]["decoder"]
    return dec["num_hidden_layers"], dec["num_key_value_heads"], dec["head_size"]


def _kv_cache_np_dtype(model_path: str) -> np.dtype:
    """Detect the past_key_values dtype from the ONNX graph inputs (fp32 for CPU builds)."""
    model = onnx.load(model_path, load_external_data=False)
    for inp in model.graph.input:
        if inp.name.startswith("past_key_values"):
            elem = inp.type.tensor_type.elem_type
            if elem == onnx.TensorProto.BFLOAT16:
                try:
                    import ml_dtypes  # noqa: F401

                    return np.dtype("bfloat16")
                except ImportError:
                    return np.float32
            return np.dtype(_ONNX_TO_NP.get(elem, np.float32))
    return np.dtype(np.float32)


def _collect_op_types(model_path: str) -> list[str]:
    """Return the sorted op types to quantize: every op present in the graph minus the excluded set.

    Excluding ``MatMulNBits``/``GatherBlockQuantized``/``GroupQueryAttention`` (see module docstring)
    keeps their packed weights untouched while still covering the activations feeding them.
    """
    proto = onnx.load(model_path, load_external_data=False)
    op_types = sorted({n.op_type for n in proto.graph.node} - _SKIP_QUANT_TYPES)
    del proto
    return op_types


def _build_calib_reader(model_dir, model_path, tokenizer_id, dataset_name, dataset_subset, split, num_samples, seq_len):
    """Build a CalibrationDataReader that yields prefill-style feeds from a text dataset.

    Each sample tokenizes one dataset row (capped at ``seq_len``, short rows skipped) and pairs it
    with empty (zero-length) past-KV tensors so the model runs a prefill step during calibration.
    """
    from datasets import load_dataset
    from onnxruntime.quantization import CalibrationDataReader
    from transformers import AutoTokenizer

    num_layers, num_kv_heads, head_size = _read_decoder_meta(model_dir)
    kv_dtype = _kv_cache_np_dtype(model_path)
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    ds = load_dataset(dataset_name, dataset_subset, split=split)

    min_len = min(32, seq_len // 4)

    class _Reader(CalibrationDataReader):
        def __init__(self):
            samples = []
            for item in ds:
                text = item["text"].strip()
                if not text:
                    continue
                ids = tok.encode(text, add_special_tokens=False)[:seq_len]
                if len(ids) < min_len:
                    continue
                n = len(ids)
                feed = {
                    "input_ids": np.array([ids], dtype=np.int64),
                    "attention_mask": np.ones((1, n), dtype=np.int64),
                }
                for i in range(num_layers):
                    feed[f"past_key_values.{i}.key"] = np.zeros((1, num_kv_heads, 0, head_size), dtype=kv_dtype)
                    feed[f"past_key_values.{i}.value"] = np.zeros((1, num_kv_heads, 0, head_size), dtype=kv_dtype)
                samples.append(feed)
                if len(samples) >= num_samples:
                    break
            print(f"  prepared {len(samples)} calibration samples (seq_len<={seq_len}, kv_dtype={kv_dtype})")
            self._it = iter(samples)

        def get_next(self):
            return next(self._it, None)

    return _Reader()


def _quantize_quark(model_path, output_path, reader, op_types, npu_transformer, power_of_two):
    """Quantize activations to UINT16 (weights UINT8, QDQ) via ``quark.onnx.quantize_static``.

    The Ryzen AI transformer / power-of-two paths switch calibration to MinMSE power-of-two scales;
    ``optimize_model=False`` avoids ORT's post-quant shape-inference reload that fails on the OGA graph.
    """
    from onnxruntime.quantization import CalibrationMethod

    from quark.onnx import QuantFormat, QuantType, quantize_static
    from quark.onnx.calibration.methods import PowerOfTwoMethod

    calibrate_method = CalibrationMethod.MinMax
    if power_of_two or npu_transformer:
        # Ryzen AI transformer path prefers hardware-aligned (power-of-two) INT16 scales.
        calibrate_method = PowerOfTwoMethod.MinMSE

    print("  backend        : quark.onnx.quantize_static")
    print(f"  npu_transformer: {npu_transformer}  power_of_two: {power_of_two}  calibrate: {calibrate_method}")
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        calibrate_method=calibrate_method,
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=op_types,
        use_external_data_format=True,
        execution_providers=["CPUExecutionProvider"],
        enable_npu_transformer=npu_transformer,
        # ORT's post-quant shape-inference reload fails on the OGA GQA/MatMulNBits graph;
        # skipping model optimization avoids that reload while keeping QDQ insertion intact.
        optimize_model=False,
        include_cle=False,
        print_summary=False,
        extra_options={"CalibTensorRangeSymmetric": False},
    )


def _quantize_ort(model_path, output_path, reader, op_types):
    """Quantize activations to UINT16 (weights UINT8, QDQ) via ``onnxruntime.quantization.quantize_static``.

    The dependency-free fallback to the Quark backend; uses free (non-power-of-two) scales.
    """
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    print("  backend        : onnxruntime.quantization.quantize_static")
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=op_types,
        use_external_data_format=True,
        calibration_providers=["CPUExecutionProvider"],
        extra_options={"CalibTensorRangeSymmetric": False},
    )


def _summarize(output_path: str):
    """Print a post-quantization summary: Q/DQ node counts, preserved MatMulNBits, and zero-point dtypes."""
    from collections import Counter

    m = onnx.load(output_path, load_external_data=False)
    counts = Counter(n.op_type for n in m.graph.node)
    inits = {i.name: i for i in m.graph.initializer}
    zp_dtypes = set()
    for n in m.graph.node:
        if n.op_type == "QuantizeLinear" and len(n.input) >= 3 and n.input[2] in inits:
            zp_dtypes.add(inits[n.input[2]].data_type)
    dtype_name = {2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16"}
    print(
        f"  QuantizeLinear={counts.get('QuantizeLinear', 0)} "
        f"DequantizeLinear={counts.get('DequantizeLinear', 0)} "
        f"MatMulNBits={counts.get('MatMulNBits', 0)} (preserved)"
    )
    print(f"  zero-point dtypes: {sorted(dtype_name.get(d, d) for d in zp_dtypes)}")


def main():
    """CLI entry point: parse args, calibrate, run INT16 QDQ quantization, and copy aux files.

    Resolves input/output paths and tokenizer, builds the calibration reader, dispatches to the
    quark or ort backend, then copies the OGA aux files so the output dir is a self-contained model.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "-i",
        "--input-model",
        required=True,
        help="OGA output dir (with model.onnx + genai_config.json) or a model.onnx path",
    )
    ap.add_argument(
        "-o", "--output-dir", required=True, help="Directory to write the INT16-QDQ model + copied aux files"
    )
    ap.add_argument("--backend", choices=["quark", "ort"], default="quark", help="Quantizer backend (default: quark)")
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer id/path for calibration (default: the input model dir's own tokenizer)",
    )
    ap.add_argument("--calib-dataset", default="wikitext", help="HF dataset for calibration (default: wikitext)")
    ap.add_argument(
        "--calib-subset", default="wikitext-2-raw-v1", help="HF dataset subset (default: wikitext-2-raw-v1)"
    )
    ap.add_argument("--calib-split", default="train", help="HF dataset split (default: train)")
    ap.add_argument("--calib-samples", type=int, default=64, help="Number of calibration samples (default: 64)")
    ap.add_argument("--seq-len", type=int, default=1024, help="Max tokens per calibration sample (default: 1024)")
    ap.add_argument(
        "--npu-transformer",
        action="store_true",
        help="[quark] enable AMD Ryzen AI NPU-transformer mode (power-of-two INT16 scales)",
    )
    ap.add_argument("--power-of-two", action="store_true", help="[quark] use power-of-two INT16 calibration (MinMSE)")
    a = ap.parse_args()

    model_dir, model_path, output_path = _resolve_paths(a.input_model, a.output_dir)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.onnx not found at {model_path}")

    tokenizer_id = a.tokenizer
    if tokenizer_id is None:
        # OGA output dirs ship their own tokenizer, so calibrate with it -- this
        # works for any model (phi-4, QwQ, ...) without a hardcoded id. Error out
        # if the directory has none, rather than miscalibrating with a wrong tokenizer.
        if os.path.exists(os.path.join(model_dir, "tokenizer.json")):
            tokenizer_id = model_dir
        else:
            raise FileNotFoundError(
                f"No tokenizer.json found in {model_dir}. Pass --tokenizer <hf id or path> "
                "so calibration uses the correct tokenizer for this model."
            )

    print(f"Input model : {model_path}")
    print(f"Output dir  : {a.output_dir}")
    op_types = _collect_op_types(model_path)
    print(f"op_types_to_quantize ({len(op_types)}): {op_types}")
    print(f"excluded from quantization: {sorted(_SKIP_QUANT_TYPES)}")

    reader = _build_calib_reader(
        model_dir,
        model_path,
        tokenizer_id,
        a.calib_dataset,
        a.calib_subset,
        a.calib_split,
        a.calib_samples,
        a.seq_len,
    )

    print("\nQuantizing (activation=QUInt16, weight=QUInt8, format=QDQ) ...")
    if a.backend == "quark":
        _quantize_quark(model_path, output_path, reader, op_types, a.npu_transformer, a.power_of_two)
    else:
        _quantize_ort(model_path, output_path, reader, op_types)

    # Copy aux files so the output dir is a self-contained OGA model.
    for fname in _AUX_FILES:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, a.output_dir)

    print("\nResult:")
    _summarize(output_path)
    print(f"\nDone. INT16-QDQ model written to: {a.output_dir}")


if __name__ == "__main__":
    main()
