#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""SVDQuant low-rank correction I/O for the diffusers reload path.

The diffusers checkpoint (``diffusers.from_pretrained``) only reconstructs the
quantized *residual* linear (packed MXFP4 weight as ``QParamsLinear``). SVDQuant's
low-rank correction branch (an :class:`ErrorCorrectedModule` wrapping the residual with
``l1``/``l2`` and an optional ``smooth_factor``) is NOT part of the diffusers module tree
and is therefore lost on reload.

Rather than teach the diffusers loader about ``ErrorCorrectedModule``, we save the
low-rank factors to a **separate** ``svd_correction.safetensors`` beside the transformer
subfolder and re-attach them after ``from_pretrained`` by wrapping each reloaded residual
back into an :class:`ErrorCorrectedModule`. Its forward is simply
``layer(x) + l2(l1(x))`` (see ``quark/torch/algorithm/svdquant/svdquant.py``), so the
reattached module reproduces the in-process SVDQuant math. Run this BEFORE
``enable_native_inference`` so the native converter sees a proper ECM (which it converts
to the FlyDSL SVDQuant kernel).
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file, save_file

from quark.torch.algorithm.svdquant.svdquant import ErrorCorrectedModule, LowRankCorrectionModule
from quark.torch.utils import getattr_recursive, setattr_recursive

_CORRECTION_FILE = "svd_correction.safetensors"
# config.json key recording that the checkpoint weights are the SVDQuant *residual*
# R = W - l2 @ l1, and naming the file that completes them. Holding the filename rather
# than a bare ``true`` keeps the loader from hardcoding it and makes the requirement
# legible to anyone reading config.json.
_MARKER = "svdquant_correction_file"


def _configured_file(load_dir: str) -> str | None:
    """Correction filename recorded in ``config.json``, or None if not marked.

    Any truthy marker means "required"; a non-string value (e.g. a legacy ``true``) falls
    back to the default filename, so a stale marker still fails loudly rather than being
    ignored.
    """
    path = os.path.join(load_dir, "config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        value = json.load(f).get(_MARKER)
    if not value:
        return None
    return value if isinstance(value, str) else _CORRECTION_FILE


def save_svd_correction(model: torch.nn.Module, out_dir: str) -> int:
    """Collect every ErrorCorrectedModule's low-rank factors into one file.

    Keys: ``<module_path>.l1``, ``<module_path>.l2``, and ``<module_path>.smooth_factor``
    (when present). Returns the number of ECMs saved. Call on the quantized (pre-freeze)
    model, whose SVDQuant layers are still ErrorCorrectedModule wrappers.
    """
    tensors: dict[str, torch.Tensor] = {}
    n = 0
    for name, module in model.named_modules():
        if not isinstance(module, ErrorCorrectedModule):
            continue
        # .contiguous() is required: safetensors refuses non-contiguous tensors, and the
        # SVD factors can be views (e.g. l2 from a transposed decomposition).
        tensors[f"{name}.l1"] = module.correction.l1.weight.detach().to(torch.bfloat16).cpu().contiguous()
        tensors[f"{name}.l2"] = module.correction.l2.weight.detach().to(torch.bfloat16).cpu().contiguous()
        sf = getattr(module, "smooth_factor", None)
        if isinstance(sf, torch.Tensor):
            tensors[f"{name}.smooth_factor"] = sf.detach().to(torch.bfloat16).cpu().contiguous()
        n += 1
    if n:
        os.makedirs(out_dir, exist_ok=True)
        save_file(tensors, os.path.join(out_dir, _CORRECTION_FILE))
    return n


def unwrap_error_corrected(model: torch.nn.Module) -> int:
    """Replace every ErrorCorrectedModule with its inner residual ``layer``.

    Call AFTER :func:`save_svd_correction` and BEFORE freeze/export. SVDQuant leaves the
    model wrapped as ``<name> = ErrorCorrectedModule(correction, layer)``, which serializes
    as ``<name>.layer.weight`` / ``<name>.correction.l*.weight``. The diffusers reload path
    rebuilds a plain module tree and maps ``QParamsLinear`` onto ``<name>`` directly, so
    those nested keys match nothing and the quantizer buffers are left on the meta device
    ("Buffer 'scale' ... still on the meta device"). Unwrapping makes the exported
    checkpoint identical in layout to a plain (non-SVD) export; the low-rank branch travels
    in its own file and is re-attached by :func:`attach_svd_correction` after load.
    """
    names = [n for n, m in model.named_modules() if isinstance(m, ErrorCorrectedModule)]
    for name in names:
        ecm = getattr_recursive(model, name)
        setattr_recursive(model, name, ecm.layer)
    return len(names)


def has_correction_file(load_dir: str) -> bool:
    return os.path.isfile(os.path.join(load_dir, _configured_file(load_dir) or _CORRECTION_FILE))


def mark_correction_required(export_dir: str) -> None:
    """Record ``svdquant_correction_file: <name>`` in the exported ``config.json``.

    :func:`unwrap_error_corrected` deliberately makes an SVD checkpoint byte-layout
    identical to a plain w4a8 one -- which means nothing on disk distinguishes "weights are
    the full W" from "weights are the residual W - l2@l1, the rest is in a separate file".
    Load the second without the correction and you get a silently wrong model: no missing
    key, no error, just a transformer that has lost its low-rank branch. This marker closes
    that gap; :func:`require_correction_file` refuses to load such a checkpoint bare.

    Call AFTER ``export_safetensors`` (which writes ``config.json``). Diffusers ignores the
    extra top-level key on ``from_pretrained`` (it logs it as an unexpected config
    attribute).
    """
    path = os.path.join(export_dir, "config.json")
    with open(path) as f:
        cfg = json.load(f)
    cfg[_MARKER] = _CORRECTION_FILE
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def require_correction_file(load_dir: str) -> None:
    """Raise if ``config.json`` marks this export as SVDQuant but the file is missing.

    No-op for a plain (non-SVD) export, and for an SVD export whose correction is present.
    """
    name = _configured_file(load_dir)
    if name and not os.path.isfile(os.path.join(load_dir, name)):
        raise FileNotFoundError(
            f"{load_dir} was exported with --svd ({_MARKER}={name!r} in config.json) but "
            f"{name} is missing. Its weights are the SVDQuant residual; loading them "
            "without the low-rank correction produces a silently wrong model. Re-export, "
            "or restore the correction file next to the checkpoint."
        )


def attach_svd_correction(model: torch.nn.Module, load_dir: str) -> int:
    """Re-wrap each residual named in the correction file into an ErrorCorrectedModule.

    Loads ``svd_correction.safetensors`` from ``load_dir`` and, for every saved module
    path, builds a ``LowRankCorrectionModule`` from ``l1``/``l2``, then replaces the
    reloaded residual linear with ``ErrorCorrectedModule(correction, residual, smooth)``.
    Returns the number of modules re-wrapped. Call after ``from_pretrained`` and before
    ``enable_native_inference``.
    """
    path = os.path.join(load_dir, _configured_file(load_dir) or _CORRECTION_FILE)
    if not os.path.isfile(path):
        return 0
    tensors = load_file(path)
    # Group the flat keys back by module path.
    names = sorted({k.rsplit(".", 1)[0] for k in tensors if k.endswith(".l1")})
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n = 0
    for name in names:
        residual = getattr_recursive(model, name)
        l1 = tensors[f"{name}.l1"].to(device=dev, dtype=dtype)
        l2 = tensors[f"{name}.l2"].to(device=dev, dtype=dtype)
        rank, in_features = l1.shape
        out_features = l2.shape[0]
        correction = LowRankCorrectionModule(in_features, out_features, rank).to(device=dev, dtype=dtype)
        with torch.no_grad():
            correction.l1.weight.copy_(l1)
            correction.l2.weight.copy_(l2)
        smooth = tensors.get(f"{name}.smooth_factor")
        smooth = smooth.to(device=dev, dtype=dtype) if smooth is not None else None
        setattr_recursive(model, name, ErrorCorrectedModule(correction, residual, smooth_factor=smooth))
        n += 1
    return n
