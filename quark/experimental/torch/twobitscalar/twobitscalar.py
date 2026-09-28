#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from quark.common.utils.log import ScreenLogger

from .config import TwoBitScalarConfig

logger = ScreenLogger(__name__)


@dataclass
class _LinearCalib:
    in_amax: torch.Tensor  # [in_features]
    X: torch.Tensor  # [N, in_features] (float32)


def _name_excluded(name: str, exclude_patterns: list[str] | None) -> bool:
    """True if ``name`` matches any of the wildcard ``exclude_patterns``."""
    return any(fnmatch(name, p) for p in (exclude_patterns or []))


def _snap_quantize_group(W: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """MSE-optimal per-row snap of a weight group to the 4 ``levels`` (scale search)."""
    Wf = W.float()
    amax = Wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    best_q, best_mse = None, None
    for frac in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
        s = amax * frac
        x = (Wf / s).clamp(-1.5, 1.5)
        d = (x.unsqueeze(-1) - levels.to(Wf.device).view(1, 1, -1)).abs()
        idx = d.argmin(dim=-1)
        q = levels.to(Wf.device)[idx] * s
        mse = (Wf - q).pow(2).sum(dim=1)
        if best_mse is None:
            best_mse, best_q = mse, q
        else:
            imp = mse < best_mse
            if imp.any():
                best_mse = torch.where(imp, mse, best_mse)
                best_q = torch.where(imp.unsqueeze(1), q, best_q)
    return best_q


def snap_to_2bit(
    model: nn.Module,
    group_size: int = 64,
    use_lloyd_max: bool = True,
    exclude_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Post-load conversion: snap weights to visible 4-level 2-bit values.

    For every nn.Linear weight (except excluded), each group of ``group_size``
    columns is re-quantized in-place so that every row contains exactly::

        weight_value = scale_per_row × level[index]

    where level is one of the Lloyd-Max levels ``{-1.0, -0.2998, +0.2998, +1.0}``
    (``use_lloyd_max=True``) or the linear levels ``{-1.0, -1/3, +1/3, +1.0}``
    (``use_lloyd_max=False``).
    The forward pass remains standard matmul.  LoRA adapters work unchanged.

    Returns per-layer stats dict with SQNR and shape.
    """
    levels = (
        torch.tensor([-1.0, -0.2998, 0.2998, 1.0])
        if use_lloyd_max
        else torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
    )

    exclude_patterns = exclude_patterns or []
    stats = {}
    processed = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or _name_excluded(name, exclude_patterns):
            continue
        W = mod.weight.data
        rows, cols = W.shape
        if cols < group_size:
            continue

        Wf = W.float()
        Wq = torch.zeros_like(Wf)
        ng = cols // group_size
        for gi in range(ng):
            gs, ge = gi * group_size, (gi + 1) * group_size
            Wq[:, gs:ge] = _snap_quantize_group(Wf[:, gs:ge], levels)
        if cols - ng * group_size > 0:
            Wq[:, ng * group_size :] = _snap_quantize_group(Wf[:, ng * group_size :], levels)

        noise = (Wf - Wq).pow(2).mean()
        signal = Wf.pow(2).mean()
        sqnr = (10 * torch.log10(signal / noise.clamp_min(1e-20))).item()

        mod.weight.data = Wq.to(W.dtype)
        processed += 1
        stats[name] = {"sqnr_db": round(sqnr, 2), "shape": list(W.shape)}

    logger.info(f"[snap_to_2bit] Snapped {processed} layers to 4-level 2-bit (group_size={group_size})")
    return stats


class AWQScaledLinear(nn.Module):
    """Drop-in Linear replacement that divides input by per-channel AWQ scales
    before matmul.  The weight is stored in AWQ-scaled domain (4 Lloyd-Max levels).
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, awq_scales: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.register_buffer("awq_scales", awq_scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = x / self.awq_scales.to(x.device, x.dtype)
        out = nn.functional.linear(x_scaled, self.weight, self.bias)
        return out


def snap_to_2bit_awq(
    model: nn.Module,
    dataloader: Any,
    group_size: int = 64,
    alpha: float = 0.5,
    clip_min: float = 0.01,
    clip_max: float = 100.0,
    calib_batches: int = 8,
    max_rows_per_batch: int = 256,
    exclude_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """AWQ-scaled 2-bit snap: collect activation stats, apply AWQ scaling,
    quantize to 4 Lloyd-Max levels, and replace Linear with AWQScaledLinear.

    Unlike plain snap_to_2bit, this uses activation-aware scaling (AWQ) before
    quantizing, giving ~7-8 dB SQNR instead of ~6.3 dB.  The AWQ scales are
    NOT undone — instead, AWQScaledLinear divides the input by scales at runtime
    (a cheap per-channel operation).

    Returns per-layer stats dict with SQNR, shape, and scale norm.
    """
    levels = torch.tensor([-1.0, -0.2998, 0.2998, 1.0])
    exclude_patterns = exclude_patterns or []

    # --- Step 1: Collect per-layer activation amax ---
    logger.info("[snap_to_2bit_awq] Collecting activation statistics...")
    layer_amax: dict[str, torch.Tensor] = {}
    hooks = []

    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear) or _name_excluded(name, exclude_patterns):
            continue
        if m.weight.shape[1] < group_size:
            continue

        def make_hook(layer_name: str) -> Callable[..., None]:
            @torch.no_grad()
            def hook(mod: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
                x = inp[0]
                if not torch.is_tensor(x):  # pragma: no cover  (defensive input guard)
                    return
                a = x.detach()
                if a.dim() > 2:
                    a = a.view(-1, a.shape[-1])
                if a.dim() != 2:  # pragma: no cover  (defensive input guard)
                    return
                if a.shape[0] > max_rows_per_batch:
                    a = a[:max_rows_per_batch]
                in_amax = a.abs().max(dim=0).values
                if layer_name not in layer_amax:
                    layer_amax[layer_name] = in_amax
                else:
                    layer_amax[layer_name] = torch.maximum(layer_amax[layer_name], in_amax)

            return hook

        hooks.append(m.register_forward_hook(make_hook(name)))

    model.eval()
    dev = next(model.parameters()).device
    batches = 0
    for batch in dataloader:
        batches += 1
        if isinstance(batch, torch.Tensor):
            model(batch.to(dev))
        elif isinstance(batch, dict):
            batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
            model(**batch)
        if batches >= calib_batches:
            break

    for h in hooks:
        h.remove()
    logger.info(f"[snap_to_2bit_awq] Calibration done: {len(layer_amax)} layers, {batches} batches")

    # --- Step 2: AWQ scale + quantize + replace ---
    stats = {}
    processed = 0

    # Cache the name->module map once. We only replace leaf nn.Linear children
    # (never their container parents), so the parent entries stay valid across
    # the in-place replacements below — avoids rebuilding the dict per layer.
    modules_by_name = dict(model.named_modules())

    for name, mod in list(modules_by_name.items()):
        if not isinstance(mod, nn.Linear) or _name_excluded(name, exclude_patterns) or name not in layer_amax:
            continue
        W = mod.weight.data
        rows, cols = W.shape
        if cols < group_size:
            continue

        in_amax = layer_amax[name].to(W.device).float()
        a = in_amax.clamp_min(1e-8)
        a = a / a.mean().clamp_min(1e-8)
        s_vec = torch.pow(a, alpha).clamp(clip_min, clip_max)

        Wf = W.float()
        Wf_orig = Wf.clone()
        Wf_scaled = Wf * s_vec.view(1, -1)

        Wq = torch.zeros_like(Wf_scaled)
        ng = cols // group_size
        for gi in range(ng):
            gs, ge = gi * group_size, (gi + 1) * group_size
            Wq[:, gs:ge] = _snap_quantize_group(Wf_scaled[:, gs:ge], levels)
        if cols - ng * group_size > 0:
            Wq[:, ng * group_size :] = _snap_quantize_group(Wf_scaled[:, ng * group_size :], levels)

        Wq_unscaled = Wq / s_vec.view(1, -1)
        noise = (Wf_orig - Wq_unscaled).pow(2).mean()
        signal = Wf_orig.pow(2).mean()
        sqnr = (10 * torch.log10(signal / noise.clamp_min(1e-20))).item()

        # Replace nn.Linear with AWQScaledLinear
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = modules_by_name[parent_name] if parent_name else model
        new_mod = AWQScaledLinear(Wq.to(W.dtype), mod.bias.data if mod.bias is not None else None, s_vec)
        setattr(parent, child_name, new_mod)

        processed += 1
        stats[name] = {"sqnr_db": round(sqnr, 2), "shape": list(W.shape), "scale_norm": round(s_vec.norm().item(), 4)}

    logger.info(f"[snap_to_2bit_awq] Snapped {processed} layers to AWQ-scaled 4-level 2-bit (group_size={group_size})")
    return stats


class TwoBitScalarProcessor:
    """
    TwoBitScalar 2-bit PTQ pass (fast-ish, offline-only):

    For each selected nn.Linear:
      - Collect calibration inputs X and in_amax
      - For each input-group of columns:
          (optional) offline AWQ scaling (W *= s, later undo)
          (optional) SRHT incoherence rotation within group
          (optional) blockwise Hessian preconditioning (Cholesky of cov of transformed X)
          groupwise 2-bit quant (levels = {-1, -1/3, 1/3, 1})
          map back (undo precondition, undo rotation, undo scaling)
      - Write quantized-dequantized weights back into the existing module weight tensor

    IMPORTANT:
      - No runtime wrappers, so no dtype mismatch issues in forward.
      - Still "fake quantized" (weights remain bf16/fp16), but quant error reflects 2-bit.

    Notes on Hessian block:
      - Uses covariance C = E[(x - mean)(x - mean)^T] in the *current basis*
        (after scaling and rotation). Mean-centering is important for stability.
      - Precondition uses: Wp = Wg @ L  where C = L L^T.
        Then quantize Wp, and map back: Wg_q = Wp_q @ inv(L)  (RIGHT solve).
      - Guardrails:
          * only apply if width >= hess_min_group_size
          * optional subsampling of groups via hess_every_n_groups
          * fallback to non-Hessian quant if Hessian path looks numerically unsafe
    """

    def __init__(self, model: nn.Module, config: TwoBitScalarConfig, dataloader: Any = None) -> None:
        self.model = model
        self.config = config
        self.dataloader = dataloader

        self._include = list(config.include_layers or [])
        self._exclude = list(config.exclude_layers or [])

        self._calib: dict[str, _LinearCalib] = {}
        self._processed: list[str] = []
        self._skipped_excluded: list[str] = []

        # Cache SRHT matrices to avoid rebuilding per group

        # Stats
        self._hess_applied = 0
        self._hess_skipped = 0
        self._hess_failed = 0
        self._hess_fallback = 0

    def _match_any(self, name: str, patterns: Iterable[str]) -> bool:
        return any(fnmatch(name, p) for p in patterns)

    def _should_process(self, name: str) -> bool:
        if self._include and not self._match_any(name, self._include):
            return False
        if self._exclude and self._match_any(name, self._exclude):
            return False
        return True

    @torch.no_grad()
    def apply(self) -> None:
        if self.dataloader is None:
            raise ValueError("[TwoBitScalar] Calibration dataloader must be provided (dataloader=None).")

        # reset stats
        self._hess_applied = 0
        self._hess_skipped = 0
        self._hess_failed = 0
        self._hess_fallback = 0

        logger.info(
            f"[TwoBitScalar] start | bits={self.config.bits} group={self.config.group_size} "
            f"alpha={self.config.act_scale_alpha} calib_batches={self.config.calib_batches} "
            f"incoherence={self.config.enable_incoherence} "
            f"hess_damp={self.config.hess_damp} hess_max_rows={self.config.hess_max_rows} "
            f"include={len(self._include)} exclude={len(self._exclude)}"
        )

        self._collect_calibration()

        if getattr(self.config, "do_quantize", True):
            self._apply_quant()
            self._disable_weight_quantizers()
        else:
            logger.info("[TwoBitScalar] do_quantize=False -> skipping weight overwrite (calibration only).")

        logger.info(
            f"[TwoBitScalar] done | processed_linear={len(self._processed)} excluded_linear={len(self._skipped_excluded)} "
            f"| hess_applied={self._hess_applied} hess_skipped={self._hess_skipped} "
            f"hess_failed={self._hess_failed} hess_fallback={self._hess_fallback}"
        )

        if getattr(self.config, "do_quantize", True) and not self._processed:
            logger.warning(
                "[TwoBitScalar] processed_linear=0 — NO layers were quantized; the exported "
                "checkpoint is effectively UNQUANTIZED. TwoBitScalar only matches nn.Linear "
                "modules, so architectures with fused/custom weight tensors (e.g. some MoE "
                "experts) are not handled. Check the model architecture and the "
                "include/exclude patterns / min_in_features settings."
            )

        if self.config.verbose:
            if self._processed:
                logger.info("[TwoBitScalar] processed (up to 80):")
                for n in self._processed[:80]:
                    logger.info(f"  - {n}")
            if self._skipped_excluded:
                logger.info("[TwoBitScalar] excluded (up to 80):")
                for n in self._skipped_excluded[:80]:
                    logger.info(f"  - {n}")

        # Release calibration buffers now that quantization is done. With
        # enable_blockwise_hessian=True these hold the full per-layer activation
        # matrices (10s of GB on large models); don't keep them resident after apply().
        self._calib.clear()

    def _disable_weight_quantizers(self) -> None:
        """Disable the framework's fake-quantizers for TwoBitScalar-processed layers.

        Disables weight, input, and output quantizers to prevent the Quark pipeline
        from adding BFP16 (or any scheme) quantization noise on top of the
        carefully optimized 2-bit TwoBitScalar weights.  The input/output quantizers add
        activation noise at every forward pass which compounds across layers.
        """
        w_disabled = 0
        act_disabled = 0
        for name, m in self.model.named_modules():
            if name not in self._processed:
                continue
            for attr in ("_weight_quantizer", "_input_quantizer", "_output_quantizer"):
                q = getattr(m, attr, None)
                if q is not None:
                    if hasattr(q, "disable_fake_quant"):
                        q.disable_fake_quant()
                    if hasattr(q, "disable_observer"):
                        q.disable_observer()
                    if attr == "_weight_quantizer":
                        w_disabled += 1
                    else:
                        act_disabled += 1
        logger.info(
            f"[TwoBitScalar] disabled quantizers: {w_disabled} weight, {act_disabled} activation "
            f"on {len(self._processed)} processed layers"
        )

    # ----------------------------
    # Calibration
    # ----------------------------

    def _collect_calibration(self) -> None:
        self._calib.clear()
        self._processed.clear()
        self._skipped_excluded.clear()

        # The full per-layer activation matrix X is only consumed by the
        # Hessian-weighted scale search and the blockwise-Hessian path. When both
        # are off (the default), only per-channel `in_amax` is needed, so we skip
        # retaining X entirely — this avoids holding max_samples_per_layer ×
        # in_features × fp32 for every layer at once (tens of GB on 32B models).
        needs_x = bool(
            getattr(self.config, "use_hessian_weighted_scale", False)
            or getattr(self.config, "enable_blockwise_hessian", False)
        )

        hooks = []

        for name, m in self.model.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            if m.in_features < self.config.min_in_features:
                continue
            if not self._should_process(name):
                self._skipped_excluded.append(name)
                continue

            def make_hook(layer_name: str) -> Callable[..., None]:
                @torch.no_grad()
                def hook(mod: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
                    x = inp[0]
                    if not torch.is_tensor(x):  # pragma: no cover  (defensive input guard)
                        return

                    a = x.detach()
                    if a.dim() > 2:
                        a = a.view(-1, a.shape[-1])
                    elif a.dim() != 2:  # pragma: no cover  (defensive input guard)
                        return

                    if a.shape[0] > self.config.max_rows_per_batch:
                        a = a[: self.config.max_rows_per_batch]

                    in_amax = a.abs().max(dim=0).values

                    if layer_name not in self._calib:
                        if needs_x:
                            X = a.float().clone()
                            if X.shape[0] > self.config.max_samples_per_layer:
                                X = X[: self.config.max_samples_per_layer]
                        else:
                            X = a.new_empty((0, a.shape[1]), dtype=torch.float32)
                        self._calib[layer_name] = _LinearCalib(in_amax=in_amax, X=X)
                    else:
                        cur = self._calib[layer_name]
                        cur.in_amax = torch.maximum(cur.in_amax, in_amax)

                        if needs_x and cur.X.shape[0] < self.config.max_samples_per_layer:
                            need = self.config.max_samples_per_layer - cur.X.shape[0]
                            add = a.float()
                            if add.shape[0] > need:
                                add = add[:need]
                            cur.X = torch.cat([cur.X, add], dim=0)

                return hook

            hooks.append(m.register_forward_hook(make_hook(name)))

        self.model.eval()

        batches = 0
        for batch in self.dataloader:
            batches += 1
            self._forward_batch(batch)
            if self.config.calib_batches is not None and batches >= self.config.calib_batches:
                break

        for h in hooks:
            h.remove()

        logger.info(
            f"[TwoBitScalar] calibration collected: layers={len(self._calib)} batches={batches} "
            f"(caps: max_rows_per_batch={self.config.max_rows_per_batch}, "
            f"max_samples_per_layer={self.config.max_samples_per_layer}, "
            f"hess_max_rows={self.config.hess_max_rows})"
        )

    def _forward_batch(self, batch: Any) -> None:
        dev = next(self.model.parameters()).device

        if isinstance(batch, torch.Tensor):
            self.model(batch.to(dev))
            return

        if isinstance(batch, dict):
            kwargs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            self.model(**kwargs)
            return

        if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict):
            for item in batch:
                kwargs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in item.items()}
                self.model(**kwargs)
            return

        self.model(batch)  # pragma: no cover  (fallback for unrecognized batch types)

    # ----------------------------
    # Linear algebra helpers
    # ----------------------------

    def _right_solve_lower_triangular(self, L: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Solve X @ L = B for X, i.e. X = B @ inv(L) (RIGHT solve)."""
        try:
            return torch.linalg.solve_triangular(L, B, upper=False, left=False)
        except TypeError:  # pragma: no cover  (older torch API fallback)
            Xt = torch.linalg.solve_triangular(L.transpose(0, 1), B.transpose(0, 1), upper=True)
            return Xt.transpose(0, 1)

    def _adaptive_damp(self, C: torch.Tensor, base_damp: float) -> float:
        n = C.shape[0]
        damp = float(base_damp)
        if n <= 8:
            damp = max(damp, 0.10)
        elif n <= 32:
            damp = max(damp, 0.03)
        else:
            damp = max(damp, 0.01)
        return damp

    def _covariance(self, Xg: torch.Tensor) -> torch.Tensor:
        """Covariance with mean-centering: C = E[(x - mu)(x - mu)^T]."""
        if Xg.numel() == 0:  # pragma: no cover  (defensive: empty calibration group)
            return torch.empty((Xg.shape[1], Xg.shape[1]), device=Xg.device, dtype=Xg.dtype)
        mu = Xg.mean(dim=0, keepdim=True)
        Xc = Xg - mu
        N = max(1, Xc.shape[0])
        return (Xc.transpose(0, 1) @ Xc) / float(N)

    def _group_is_hess_eligible(self, width: int, gi: int) -> tuple[bool, str]:
        every = int(getattr(self.config, "hess_every_n_groups", 1) or 1)
        min_g = int(getattr(self.config, "hess_min_group_size", 1) or 1)
        if width < min_g:
            return False, f"width<{min_g}"
        if every > 1 and (gi % every) != 0:
            return False, f"gi%{every}!=0"
        return True, "ok"

    def _safe_group_output(self, Wg_ref: torch.Tensor, Wg_candidate: torch.Tensor) -> bool:
        if not torch.isfinite(Wg_candidate).all():  # pragma: no cover  (numerical guard)
            return False
        ref = Wg_ref.to(torch.float32)
        cand = Wg_candidate.to(torch.float32)
        ref_rms = ref.pow(2).mean().sqrt().clamp_min(1e-12)
        cand_rms = cand.pow(2).mean().sqrt()
        ratio = (cand_rms / ref_rms).item()
        if ratio > 10.0 or ratio < 0.1:
            return False
        return True

    # ----------------------------
    # Quantization pass
    # ----------------------------

    def _quantize_2bit_sub_groups(
        self, Wg: torch.Tensor, sub_g: int, importance: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Quantize with per-row scale at sub-group granularity within a (possibly
        larger) SRHT-rotated group.  This gives the decorrelation benefit of the
        larger group while keeping per-row scales tight."""
        _, width = Wg.shape
        if sub_g <= 0 or sub_g >= width:
            return self._quantize_2bit_per_row(Wg, importance=importance)
        Wq = torch.zeros_like(Wg)
        for si in range(0, width, sub_g):
            se = min(si + sub_g, width)
            imp_slice = importance[si:se] if importance is not None else None
            Wq[:, si:se] = self._quantize_2bit_per_row(Wg[:, si:se], importance=imp_slice)
        return Wq

    @torch.no_grad()
    def _apply_quant(self) -> None:
        targets: dict[str, nn.Linear] = {}
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear) and name in self._calib and self._should_process(name):
                targets[name] = m

        g = int(self.config.group_size)
        if g <= 0:
            raise ValueError("[TwoBitScalar] group_size must be > 0")

        sub_g = int(self.config.sub_group_size) if self.config.sub_group_size > 0 else g
        if sub_g > g:
            sub_g = g
        logger.info(f"[TwoBitScalar] group_size={g} sub_group_size={sub_g}")

        # Optional sidecar dump: per-layer (awq_scale, srht_perm, srht_signs) that
        # reconstruct the input-side transform x_rot = SRHT(x / awq_scale) for
        # downstream "factored" / shared-rotation exports.
        dump_sidecar = bool(getattr(self.config, "dump_sidecar_dir", None))
        sidecar: dict[str, dict[str, torch.Tensor]] = {
            "W_q_levels": {},
            "group_scale": {},
            "awq_scale": {},
            "srht_perm": {},
            "srht_signs": {},
        }
        if dump_sidecar:
            if not self.config.enable_incoherence:
                logger.warning(
                    "[TwoBitScalar] dump_sidecar_dir set but enable_incoherence=False; "
                    "no SRHT rotation to record — sidecar will contain awq_scale only."
                )
            if int(self.config.num_hadamard_passes) != 1:
                logger.warning(
                    "[TwoBitScalar] dump_sidecar_dir requires num_hadamard_passes=1 for a single "
                    f"perm/sign per layer; got {self.config.num_hadamard_passes}. Sidecar disabled."
                )
                dump_sidecar = False

        pbar = tqdm(targets.items(), desc="[TwoBitScalar] Quantizing layers", unit="layer")
        for name, lin in pbar:
            pbar.set_postfix_str(".".join(name.split(".")[-2:]))
            t0 = time.time()
            calib = self._calib[name]
            W = lin.weight.data
            orig_dtype = W.dtype
            device = W.device

            Wf = W.float()
            Wf_orig = Wf.clone()
            _, in_dim = Wf.shape

            s_vec = None
            if self.config.act_scale_alpha and self.config.act_scale_alpha > 0.0:
                s_vec = self._compute_scale_vector(
                    calib.in_amax.to(device=device),
                    alpha=float(self.config.act_scale_alpha),
                    clip_min=float(self.config.scale_clip_min),
                    clip_max=float(self.config.scale_clip_max),
                )

            X = calib.X
            if X.shape[0] > self.config.hess_max_rows:
                X = X[: self.config.hess_max_rows]
            X = X.to(device=device, dtype=torch.float32)

            # --- AWQ scaling (full layer) ---
            if s_vec is not None:
                Wf = Wf * s_vec.view(1, -1)
                X = X / s_vec.view(1, -1)

            # --- Output (row) rotation for full incoherence ---
            row_rot = None
            if getattr(self.config, "enable_output_rotation", False):
                seed = int(self.config.incoherence_seed)
                Wf_T = Wf.t().contiguous()
                Wf_T, row_perm, row_signs = self._global_rotate(Wf_T, seed=seed + 999, device=device)
                Wf = Wf_T.t().contiguous()
                row_rot = (row_perm, row_signs)

            # --- Global incoherence rotation (full layer) ---
            global_rot = None
            if self.config.enable_incoherence:
                Wf, g_perm, g_signs = self._global_rotate(Wf, seed=int(self.config.incoherence_seed), device=device)
                X, _, _ = self._global_rotate(X, seed=int(self.config.incoherence_seed), device=device)
                global_rot = (g_perm, g_signs)

            # --- Sidecar capture (single-pass perm/sign + awq scale) ---
            if dump_sidecar:
                sidecar["awq_scale"][name] = (
                    s_vec.detach().to(dtype=torch.float32, device="cpu")
                    if s_vec is not None
                    else torch.ones(in_dim, dtype=torch.float32)
                )
                if global_rot is not None:
                    g_perm_l, g_signs_l = global_rot
                    sidecar["srht_perm"][name] = g_perm_l[0].detach().to(dtype=torch.int64, device="cpu")
                    sidecar["srht_signs"][name] = g_signs_l[0].detach().to(dtype=torch.float32, device="cpu")

            # --- Per-group quantization (sub-group scales) ---
            Wq = Wf.clone()

            use_hess_importance = getattr(self.config, "use_hessian_weighted_scale", False)

            num_groups = (in_dim + g - 1) // g
            for gi in range(num_groups):
                gs = gi * g
                ge = min((gi + 1) * g, in_dim)
                width = ge - gs
                if width <= 0:
                    continue

                Wg = Wq[:, gs:ge]
                Xg = X[:, gs:ge]

                imp = None
                if use_hess_importance and Xg.shape[0] > 0:
                    imp = (Xg**2).mean(dim=0)  # [width]

                if self.config.enable_blockwise_hessian and Xg.shape[0] > 0:
                    eligible, reason = self._group_is_hess_eligible(width, gi)
                    if not eligible:
                        self._hess_skipped += 1
                        Wg_q = self._quantize_2bit_sub_groups(Wg, sub_g, importance=imp)
                    else:
                        C = self._covariance(Xg)
                        diag = torch.diag(C)
                        diag_mean = diag.mean().clamp_min(1e-12)

                        if not torch.isfinite(diag_mean) or diag_mean.item() <= 0.0:  # pragma: no cover
                            self._hess_skipped += 1
                            Wg_q = self._quantize_2bit_sub_groups(Wg, sub_g, importance=imp)
                        else:
                            damp = self._adaptive_damp(C, float(self.config.hess_damp))
                            C = C + torch.eye(width, device=device, dtype=C.dtype) * (damp * diag_mean)

                            if not torch.isfinite(C).all():  # pragma: no cover
                                self._hess_skipped += 1
                                Wg_q = self._quantize_2bit_sub_groups(Wg, sub_g, importance=imp)
                            else:
                                try:
                                    L = torch.linalg.cholesky(C)
                                    Wp = (Wg.to(torch.float32) @ L).to(Wg.dtype)
                                    Wp_q = self._quantize_2bit_sub_groups(Wp, sub_g, importance=imp)
                                    Wg_q_candidate = self._right_solve_lower_triangular(L, Wp_q.to(torch.float32)).to(
                                        Wg.dtype
                                    )

                                    if not self._safe_group_output(  # pragma: no cover
                                        self._quantize_2bit_sub_groups(Wg, sub_g, importance=imp), Wg_q_candidate
                                    ):
                                        self._hess_fallback += 1
                                        Wg_q = self._quantize_2bit_sub_groups(Wg, sub_g, importance=imp)
                                    else:
                                        Wg_q = Wg_q_candidate
                                        self._hess_applied += 1

                                except Exception:  # pragma: no cover
                                    self._hess_failed += 1
                                    Wg_q = self._quantize_2bit_sub_groups(Wg, sub_g, importance=imp)
                else:
                    Wg_q = self._quantize_2bit_sub_groups(Wg, sub_g, importance=imp)

                Wq[:, gs:ge] = Wg_q

            # --- Sidecar: rotated-domain 4-level weight + per-group scale ---
            # Captured here (before undoing the rotation) so the factored /
            # shared-rotation export can keep weights in the compressible rotated
            # domain. group_scale = per-(row, group) max|Wq| — an element at the
            # group amax always snaps to level ±1, so this recovers the exact scale;
            # W_q_levels = Wq / group_scale ∈ {-1,-1/3,1/3,1} (or the Lloyd-Max grid).
            # Read-only w.r.t. Wq, so the exported (undone) weights are unchanged.
            if dump_sidecar:
                num_g = (in_dim + g - 1) // g
                gscale = torch.empty((Wq.shape[0], num_g), dtype=torch.float32, device=Wq.device)
                wlevels = torch.empty_like(Wq, dtype=torch.float32)
                for gi in range(num_g):
                    gs = gi * g
                    ge = min((gi + 1) * g, in_dim)
                    blk = Wq[:, gs:ge].float()
                    sc = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
                    gscale[:, gi] = sc.squeeze(1)
                    wlevels[:, gs:ge] = blk / sc
                sidecar["W_q_levels"][name] = wlevels.detach().to(device="cpu")
                sidecar["group_scale"][name] = gscale.detach().to(device="cpu")

            # --- Undo global rotation ---
            if global_rot is not None:
                g_perm, g_signs = global_rot
                Wq = self._global_unrotate(Wq, g_perm, g_signs)

            # --- Undo output (row) rotation ---
            if row_rot is not None:
                rp, rs = row_rot
                Wq_T = Wq.t().contiguous()
                Wq_T = self._global_unrotate(Wq_T, rp, rs)
                Wq = Wq_T.t().contiguous()

            # --- Undo AWQ scaling ---
            if s_vec is not None:
                Wq = Wq / s_vec.view(1, -1)

            if self.config.debug_dump_per_layer:
                err = (Wf_orig - Wq).pow(2).mean().sqrt().item()
                w_rms = Wf_orig.pow(2).mean().sqrt().item()
                logger.info(
                    f"[TwoBitScalar] {name}: RMS_err={err:.6e} | W_rms={w_rms:.6e} | rel={err / (w_rms + 1e-12):.6e}"
                )

            lin.weight.data = Wq.to(orig_dtype) if self.config.preserve_weight_dtype else Wq
            self._processed.append(name)
            elapsed = time.time() - t0
            pbar.set_postfix_str(f"{'.'.join(name.split('.')[-2:])} ({elapsed:.1f}s)")

        pbar.close()

        if dump_sidecar and sidecar["awq_scale"]:
            import os as _os

            from safetensors.torch import save_file as _save_file

            out_dir = str(self.config.dump_sidecar_dir)
            _os.makedirs(out_dir, exist_ok=True)
            for field, tensors in sidecar.items():
                if tensors:
                    _save_file(tensors, _os.path.join(out_dir, f"{field}.safetensors"))
            logger.info(
                f"[TwoBitScalar] wrote sidecar ({len(sidecar['awq_scale'])} layers) to {out_dir}: "
                f"{[f + '.safetensors' for f in sidecar if sidecar[f]]}"
            )

    # ----------------------------
    # Quant helpers
    # ----------------------------

    def _compute_scale_vector(
        self, in_amax: torch.Tensor, *, alpha: float, clip_min: float, clip_max: float
    ) -> torch.Tensor:
        a = in_amax.float().clamp_min(1e-8)
        a = a / a.mean().clamp_min(1e-8)
        s = torch.pow(a, alpha)
        return s.clamp(clip_min, clip_max)

    def _quantize_2bit_per_row(self, W: torch.Tensor, importance: torch.Tensor | None = None) -> torch.Tensor:
        """Per-row symmetric 2-bit quant with MSE-optimal scale search.

        Levels are {-1, -1/3, 1/3, 1} by default, or Lloyd-Max Gaussian-optimal
        {-1, -0.2998, 0.2998, 1} when use_lloyd_max_levels is enabled.

        When adaptive_grid_levels is set, a 2D grid search over (scale, inner_level)
        is performed per row to find the truly optimal 4-level symmetric quantizer.

        When `importance` (Hessian diagonal, shape [cols]) is provided, the
        scale search minimises importance-weighted MSE instead of raw MSE.
        """
        device = W.device
        dtype = W.dtype

        if getattr(self.config, "adaptive_grid_levels", False):
            return self._quantize_2bit_adaptive(W, importance)

        if getattr(self.config, "use_lloyd_max_levels", False):
            levels = torch.tensor([-1.0, -0.2998, 0.2998, 1.0], device=device, dtype=torch.float32)
        else:
            levels = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], device=device, dtype=torch.float32)

        Wf = W.float()
        amax = Wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)

        imp = None
        if importance is not None:
            imp = importance.float().view(1, -1)

        best_q = None
        best_mse = None
        if getattr(self.config, "fine_scale_grid", False):
            scale_fracs = [0.50 + i * 0.0125 for i in range(41)]  # 0.50 to 1.0, step 0.0125
        else:
            scale_fracs = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        for frac in scale_fracs:
            scale = amax * frac
            x = (Wf / scale).clamp(-1.5, 1.5)
            dist = (x.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
            idx = dist.argmin(dim=-1)
            q = levels[idx] * scale
            err_sq = (Wf - q).pow(2)
            if imp is not None:
                mse = (err_sq * imp).sum(dim=1)
            else:
                mse = err_sq.sum(dim=1)
            if best_mse is None:
                best_mse = mse
                best_q = q
            else:
                improved = mse < best_mse
                if improved.any():
                    best_mse = torch.where(improved, mse, best_mse)
                    best_q = torch.where(improved.unsqueeze(1), q, best_q)

        assert best_q is not None
        return best_q.to(dtype)

    def _quantize_2bit_adaptive(self, W: torch.Tensor, importance: torch.Tensor | None = None) -> torch.Tensor:
        """2D grid search over (scale_frac, inner_level) for optimal per-row quantization.

        Searches inner level from 0.20 to 0.45 (13 values) x scale_frac (fine or coarse),
        picking the (scale, inner_level) pair that minimizes MSE for each row independently.
        """
        device = W.device
        dtype = W.dtype
        Wf = W.float()
        amax = Wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)

        imp = None
        if importance is not None:
            imp = importance.float().view(1, -1)

        if getattr(self.config, "fine_scale_grid", False):
            scale_fracs = [0.50 + i * 0.025 for i in range(21)]  # coarser for 2D (21 pts)
        else:
            scale_fracs = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

        inner_levels = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.45]

        best_q = None
        best_mse = None

        for inner in inner_levels:
            levels = torch.tensor([-1.0, -inner, inner, 1.0], device=device, dtype=torch.float32)
            for frac in scale_fracs:
                scale = amax * frac
                x = (Wf / scale).clamp(-1.5, 1.5)
                dist = (x.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
                idx = dist.argmin(dim=-1)
                q = levels[idx] * scale
                err_sq = (Wf - q).pow(2)
                if imp is not None:
                    mse = (err_sq * imp).sum(dim=1)
                else:
                    mse = err_sq.sum(dim=1)
                if best_mse is None:
                    best_mse = mse
                    best_q = q
                else:
                    improved = mse < best_mse
                    if improved.any():
                        best_mse = torch.where(improved, mse, best_mse)
                        best_q = torch.where(improved.unsqueeze(1), q, best_q)

        assert best_q is not None
        return best_q.to(dtype)

    # ----------------------------
    # SRHT helpers
    # ----------------------------

    @staticmethod
    def _largest_pow2_factor(n: int) -> int:
        """Largest power of 2 that divides n."""
        return n & (-n)

    def _global_rotate(
        self, W: torch.Tensor, seed: int, device: torch.device
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Multi-pass global incoherence rotation.

        Each pass applies (random permute -> random signs -> block Hadamard).
        Multiple passes with different permutations break cross-block
        correlations that a single pass leaves behind, achieving near-global
        decorrelation even when Hadamard blocks are small.

        Returns (W_rotated, perms_list, signs_list) for undo.
        """
        _, in_dim = W.shape
        blk = min(self._largest_pow2_factor(in_dim), 1024)
        n_passes = getattr(self.config, "num_hadamard_passes", 1)
        perms, signs_list = [], []

        for p_idx in range(n_passes):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed + p_idx * 77)

            perm = torch.randperm(in_dim, generator=gen)
            W = W[:, perm]
            perms.append(perm)

            s = torch.empty(in_dim, dtype=torch.float32)
            s.bernoulli_(0.5, generator=gen)
            s = (s * 2.0 - 1.0).to(device=device, dtype=W.dtype)
            W = W * s.unsqueeze(0)
            signs_list.append(s)

            if blk >= 2:
                nblocks = in_dim // blk
                W = self._hadamard(W.reshape(W.shape[0], nblocks, blk)) / (blk**0.5)
                W = W.reshape(W.shape[0], in_dim)

        return W, perms, signs_list

    def _global_unrotate(
        self, W: torch.Tensor, perms: list[torch.Tensor], signs_list: list[torch.Tensor]
    ) -> torch.Tensor:
        """Undo multi-pass global rotation in reverse order."""
        _, in_dim = W.shape
        blk = min(self._largest_pow2_factor(in_dim), 1024)

        n_passes = getattr(self.config, "num_hadamard_passes", 1)

        for p_idx in reversed(range(n_passes)):
            if blk >= 2:
                nblocks = in_dim // blk
                W = self._hadamard(W.reshape(W.shape[0], nblocks, blk)) / (blk**0.5)
                W = W.reshape(W.shape[0], in_dim)

            W = W * signs_list[p_idx].unsqueeze(0)

            inv_perm = torch.empty_like(perms[p_idx])
            inv_perm[perms[p_idx]] = torch.arange(in_dim, device=perms[p_idx].device)
            W = W[:, inv_perm]

        return W

    def _is_power_of_two(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def _hadamard(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        if not self._is_power_of_two(n):
            raise ValueError("Hadamard requires power-of-two length.")

        orig_shape = x.shape
        y = x.reshape(-1, n)

        h = 1
        while h < n:
            y = y.view(-1, n // (2 * h), 2, h)
            a = y[:, :, 0, :]
            b = y[:, :, 1, :]
            y = torch.stack((a + b, a - b), dim=2).view(-1, n)
            h *= 2

        return y.view(*orig_shape)
