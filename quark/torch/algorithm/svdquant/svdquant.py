from __future__ import annotations

import fnmatch
import gc
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from quark.common.utils.import_utils import is_accelerate_available
from quark.common.utils.log import ScreenLogger
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.utils.accelerate_helper import clone_align_devices_hook

if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import QLayerConfig

logger = ScreenLogger(__name__)

# ---------------------------------------------------------------------------
# GPTQ helpers for residual weight quantisation
# ---------------------------------------------------------------------------


class HessianCollector:
    """Collects input Hessian matrices (H = X^T X) for linear layers via
    forward hooks using a numerically stable running average."""

    def __init__(self) -> None:
        self.hessians: dict[str, torch.Tensor] = {}
        self.nsamples: dict[str, int] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def register_hooks(self, model: nn.Module, target_names: set[str]) -> None:
        for name, module in model.named_modules():
            if name in target_names and isinstance(module, nn.Linear):
                self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook_fn(module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            inp = input[0].detach()
            if inp.dim() == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            inp = inp.float()

            n = inp.shape[0]
            cols = inp.shape[1]

            if name not in self.hessians:
                self.hessians[name] = torch.zeros((cols, cols), device=inp.device, dtype=torch.float32)
                self.nsamples[name] = 0

            self.hessians[name] *= self.nsamples[name] / (self.nsamples[name] + n)
            self.nsamples[name] += n
            inp = math.sqrt(2.0 / self.nsamples[name]) * inp
            self.hessians[name] += inp.t() @ inp

        return hook_fn

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def adjust_for_smoothing(self, smooth_factors: dict[str, torch.Tensor]) -> None:
        """Transform H -> diag(1/s) H diag(1/s) to account for smoothing."""
        for name, s in smooth_factors.items():
            if name in self.hessians:
                inv_s = (1.0 / s.float().to(self.hessians[name].device)).clamp(min=1e-10)
                self.hessians[name] = self.hessians[name] * inv_s.unsqueeze(0) * inv_s.unsqueeze(1)


def gptq_quantize_residual(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    n_bits: int = 4,
    symmetric: bool = True,
    group_size: int = -1,
    blocksize: int = 128,
    percdamp: float = 0.01,
    actorder: bool = False,
) -> torch.Tensor:
    """Column-wise GPTQ quantisation of a single weight matrix using Hessian information."""
    W = weight.float().clone()
    H = hessian.float().clone()
    orig_dtype = weight.dtype
    device = W.device
    rows, columns = W.shape

    # --- handle dead columns ------------------------------------------------
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # --- activation ordering ------------------------------------------------
    perm = invperm = None
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

    # --- Cholesky-based inverse ---------------------------------------------
    damp = percdamp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(columns, device=device)
    H[diag_idx, diag_idx] += damp
    try:
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)
    except torch.linalg.LinAlgError:
        logger.warning(
            "GPTQ: Cholesky decomposition failed — falling back to "
            "pseudo-inverse.  Results may be slightly less accurate."
        )
        Hinv = torch.linalg.pinv(H)
        Hinv = torch.linalg.cholesky(Hinv + 1e-6 * torch.eye(columns, device=device), upper=True)

    # --- quantisation grid --------------------------------------------------
    if symmetric:
        qmin = -(2 ** (n_bits - 1))
        qmax = 2 ** (n_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2**n_bits - 1

    per_group = group_size > 0

    if per_group:
        n_groups = (columns + group_size - 1) // group_size
        group_scales: list[torch.Tensor] = []
        group_zeros: list[torch.Tensor] = []
        for g in range(n_groups):
            gs = g * group_size
            ge = min(gs + group_size, columns)
            w_g = W[:, gs:ge]
            if symmetric:
                s = w_g.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / qmax
                z = torch.zeros(rows, 1, device=device)
            else:
                lo = w_g.amin(dim=1, keepdim=True)
                hi = w_g.amax(dim=1, keepdim=True)
                s = ((hi - lo) / qmax).clamp(min=1e-10)
                z = (-lo / s).round()
            group_scales.append(s)
            group_zeros.append(z)
    else:
        if symmetric:
            ch_scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / qmax
            ch_zero = torch.zeros(rows, 1, device=device)
        else:
            lo = W.amin(dim=1, keepdim=True)
            hi = W.amax(dim=1, keepdim=True)
            ch_scale = ((hi - lo) / qmax).clamp(min=1e-10)
            ch_zero = (-lo / ch_scale).round()

    # --- GPTQ column-by-column loop ----------------------------------------
    Q = torch.zeros_like(W)

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if per_group:
                col_idx = i1 + i
                orig_col = perm[col_idx].item() if (actorder and perm is not None) else col_idx
                g = orig_col // group_size
                s = group_scales[g]
                z = group_zeros[g]
            else:
                s = ch_scale
                z = ch_zero

            q = ((w.unsqueeze(1) / s + z).round().clamp(qmin, qmax) - z) * s
            q = q.squeeze(1)
            Q1[:, i] = q

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    if actorder and invperm is not None:
        Q = Q[:, invperm]

    return Q.to(orig_dtype)


QUANT_MODE_TO_SCHEME: dict[str, str] = {
    "w4a16": "int4_wo_64",
    "w4a4": "int4_wa_64",
    "mxfp4": "mxfp4",
    "nvfp4": "nvfp4",
}


def build_quant_layer_config(mode: str) -> QLayerConfig:
    """Return a ``QLayerConfig`` for the requested quantization mode.

    Supported modes are listed in ``QUANT_MODE_TO_SCHEME``.
    Raises ``ValueError`` for unknown modes.
    """
    from quark.torch.quantization.config.template import QuantizationSchemeCollection

    mode = mode.lower()
    if mode not in QUANT_MODE_TO_SCHEME:
        raise ValueError(f"Unknown quant mode '{mode}'. Available: {sorted(QUANT_MODE_TO_SCHEME)}")
    collection = QuantizationSchemeCollection()
    return collection.get_scheme(QUANT_MODE_TO_SCHEME[mode]).config


class LowRankCorrectionModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_features, rank, bias=False)
        self.l2 = nn.Linear(rank, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.l1(x))


class ErrorCorrectedModule(nn.Module):
    def __init__(
        self,
        correction: LowRankCorrectionModule,
        layer: nn.Module,
        smooth_factor: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.correction = correction
        self.layer = layer
        if smooth_factor is not None:
            self.register_buffer("smooth_factor", smooth_factor)
        else:
            self.smooth_factor = None

        if hasattr(layer, "in_features"):
            self.in_features = layer.in_features
        if hasattr(layer, "out_features"):
            self.out_features = layer.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.layer, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        if self.smooth_factor is not None:
            x = (x / self.smooth_factor).to(original_dtype)
        x = x.contiguous()

        if original_dtype == torch.float16:
            # float16 matmul outputs can overflow (max 65504) when fan-in is
            # large and activations have grown across transformer blocks.
            # Compute in float32 and cast back.
            layer_out = F.linear(
                x.float(),
                self.layer.weight.float(),
                self.layer.bias.float() if self.layer.bias is not None else None,
            )
            corr_out = F.linear(x.float(), self.correction.l1.weight.float())
            corr_out = F.linear(corr_out, self.correction.l2.weight.float())
            return (layer_out + corr_out).to(original_dtype)

        return self.layer(x) + self.correction(x)


def _svd_decompose(
    weight: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    W = weight.float()
    max_rank = min(W.shape)
    rank = min(rank, max_rank)

    U, S, V = torch.linalg.svd(W, full_matrices=False)

    L1 = torch.diag(S[:rank]) @ V[:rank, :]
    L2 = U[:, :rank]
    R = W - L2 @ L1

    return L1.to(weight.dtype), L2.to(weight.dtype), R.to(weight.dtype)


class InputCache:
    """Caches a bounded number of input activations per linear layer for alpha search."""

    def __init__(self, max_samples: int = 8) -> None:
        self.max_samples = max_samples
        self.inputs: dict[str, list[torch.Tensor]] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def register_hooks(self, model: nn.Module, target_names: set[str]) -> None:
        for name, module in model.named_modules():
            if name in target_names and isinstance(module, nn.Linear):
                self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook_fn(module: nn.Module, input: tuple[Any, ...], output: Any) -> None:
            if name in self.inputs and len(self.inputs[name]) >= self.max_samples:
                return
            inp = input[0].detach()
            if inp.dim() == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            self.inputs.setdefault(name, []).append(inp.cpu())

        return hook_fn

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _simulate_quant(
    tensor: torch.Tensor,
    n_bits: int = 4,
    symmetric: bool = True,
    group_size: int = -1,
) -> torch.Tensor:
    """Fake-quantize a tensor to *n_bits* (round-to-nearest, symmetric)."""
    W = tensor.float()
    qmax = 2 ** (n_bits - 1) - 1

    if group_size > 0:
        rows, cols = W.shape
        n_groups = (cols + group_size - 1) // group_size
        Q = torch.zeros_like(W)
        for g in range(n_groups):
            gs = g * group_size
            ge = min(gs + group_size, cols)
            w_g = W[:, gs:ge]
            s = w_g.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / qmax
            Q[:, gs:ge] = (w_g / s).round().clamp(-qmax - 1, qmax) * s
        return Q.to(tensor.dtype)

    if symmetric:
        s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / qmax
        return ((W / s).round().clamp(-qmax - 1, qmax) * s).to(tensor.dtype)

    qmax_unsigned = 2**n_bits - 1
    lo = W.amin(dim=1, keepdim=True)
    hi = W.amax(dim=1, keepdim=True)
    s = ((hi - lo) / qmax_unsigned).clamp(min=1e-10)
    z = (-lo / s).round()
    return ((W / s + z).round().clamp(0, qmax_unsigned).sub(z) * s).to(tensor.dtype)


class ActivationSmoother:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.activation_max: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def register_hooks(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook_fn(module: nn.Module, input: tuple[Any, ...], output: Any) -> None:
            inp = input[0]
            if inp.dim() == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            channel_max = inp.abs().amax(dim=0).detach()
            if name in self.activation_max:
                self.activation_max[name] = torch.max(self.activation_max[name], channel_max)
            else:
                self.activation_max[name] = channel_max

        return hook_fn

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def compute_smooth_factors(
        self,
        model: nn.Module,
        exclude_patterns: list[str],
        per_layer_alpha: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        smooth_factors = {}
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(fnmatch.fnmatch(name, p) for p in exclude_patterns):
                continue
            if name not in self.activation_max:
                continue
            alpha = per_layer_alpha[name] if per_layer_alpha and name in per_layer_alpha else self.alpha
            W = module.weight.data.float()
            act_max = self.activation_max[name].float().to(W.device)
            weight_max = W.abs().amax(dim=0).clamp(min=1e-5)
            s = act_max.clamp(min=1e-8).pow(alpha) / weight_max.pow(1.0 - alpha)
            s = s.clamp(min=1e-5, max=1e5)
            smooth_factors[name] = s
        return smooth_factors


def apply_smoothing(
    model: nn.Module,
    smooth_factors: dict[str, torch.Tensor],
    keep_float32: bool = False,
) -> dict[str, torch.Tensor]:
    applied = {}
    for name, module in model.named_modules():
        if name in smooth_factors:
            s = smooth_factors[name].to(module.weight.device)
            smoothed = module.weight.data.float() * s.unsqueeze(0)
            if keep_float32:
                module.weight.data = smoothed
            else:
                module.weight.data = smoothed.to(module.weight.dtype)
            applied[name] = s
    return applied


def _replace_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    old_module = getattr(parent, parts[-1])
    if hasattr(old_module, "_hf_hook"):
        quark_hook = clone_align_devices_hook(old_module._hf_hook)
        add_hook_to_module(new_module, quark_hook)

    setattr(parent, parts[-1], new_module)


class SVDQuantProcessor(BaseAlgoProcessor):
    """SVD-based low-rank error correction for ``nn.Linear`` layers.

    Decomposes each eligible weight matrix via SVD, keeps a low-rank
    correction branch, and smooths activations — works for both diffusion
    models and LLMs.
    """

    def __init__(
        self,
        model: nn.Module,
        quant_algo_config: Any,
        calib_data: (
            DataLoader[torch.Tensor]
            | DataLoader[list[dict[str, torch.Tensor]]]
            | DataLoader[dict[str, torch.Tensor]]
            | None
        ) = None,
    ) -> None:
        self.model = model
        self.config = quant_algo_config
        self.data_loader = calib_data
        self.smoother = ActivationSmoother(alpha=self.config.smooth_alpha)
        self.hessian_collector: HessianCollector | None = None
        self.input_cache: InputCache | None = None

    def _should_quantize(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if any(fnmatch.fnmatch(name, p) for p in self.config.exclude_patterns):
            return False
        if module.weight.dim() != 2:
            return False
        if min(module.weight.shape) < self.config.min_layer_size:
            return False
        return True

    def _calibrate(self) -> None:
        """Collect activation statistics and optionally Hessian matrices for calibration.

        Registers forward hooks to gather activation statistics via the smoother.
        When alpha search is enabled, also caches a small number of input
        activations per layer.  If GPTQ is enabled in the config, also collects
        Hessian matrices for eligible layers.  Runs the model in evaluation mode
        on all calibration data samples, then removes the hooks.

        Side effects:
            - Populates self.smoother.activation_max with per-layer statistics
            - If alpha search enabled, populates self.input_cache.inputs
            - If GPTQ enabled, populates self.hessian_collector.hessians
            - Sets model to evaluation mode
        """
        logger.info("SVDQuant: collecting activation statistics ...")
        self.smoother.register_hooks(self.model)

        target_names: set[str] = set()
        for name, module in self.model.named_modules():
            if self._should_quantize(name, module):
                target_names.add(name)

        if getattr(self.config, "search_alpha", False):
            max_samples = getattr(self.config, "alpha_search_max_samples", 8)
            logger.info(f"SVDQuant: alpha search enabled — caching up to {max_samples} inputs per layer")
            self.input_cache = InputCache(max_samples=max_samples)
            self.input_cache.register_hooks(self.model, target_names)

        if getattr(self.config, "use_gptq", False):
            logger.info("SVDQuant: GPTQ enabled — also collecting Hessian matrices")
            self.hessian_collector = HessianCollector()
            self.hessian_collector.register_hooks(self.model, target_names)
            logger.info(f"SVDQuant: registered Hessian hooks on {len(target_names)} layers")

        self.model.eval()
        n_samples = 0
        assert self.data_loader is not None, "_calibrate called without data_loader"
        with torch.no_grad():
            for data in self.data_loader:
                if isinstance(data, dict):
                    self.model(**data)
                elif isinstance(data, tuple | list):
                    self.model(*data)
                else:
                    self.model(data)

                n_samples += 1

        self.smoother.remove_hooks()
        if self.input_cache is not None:
            self.input_cache.remove_hooks()
        if self.hessian_collector is not None:
            self.hessian_collector.remove_hooks()

        logger.info(
            f"SVDQuant: collected activation stats for "
            f"{len(self.smoother.activation_max)} layers "
            f"over {n_samples} calibration samples"
        )
        if self.input_cache is not None:
            logger.info(f"SVDQuant: cached inputs for {len(self.input_cache.inputs)} layers")
        if self.hessian_collector is not None:
            logger.info(f"SVDQuant: collected Hessian matrices for {len(self.hessian_collector.hessians)} layers")

    def _search_alpha(
        self,
        name: str,
        module: nn.Linear,
        cached_inputs: list[torch.Tensor],
        alpha_candidates: list[float],
        svd_rank: int,
    ) -> float:
        """Search for the best smoothing alpha for a single layer.

        For each candidate alpha, simulates smooth -> SVD -> fake-quantize of
        the residual and measures output MSE against the original layer output
        on the cached calibration inputs.  Returns the alpha that yields the
        lowest total MSE.
        """
        W = module.weight.data.float()
        device = W.device
        act_max = self.smoother.activation_max[name].float().to(device)
        weight_max = W.abs().amax(dim=0).clamp(min=1e-5)

        best_alpha = alpha_candidates[0]
        best_mse = float("inf")

        for alpha in alpha_candidates:
            s = act_max.clamp(min=1e-8).pow(alpha) / weight_max.pow(1.0 - alpha)
            s = s.clamp(min=1e-5, max=1e5)

            W_smooth = W * s.unsqueeze(0)
            L1, L2, R = _svd_decompose(W_smooth, rank=svd_rank)
            Q_R = _simulate_quant(R, n_bits=4, symmetric=True)
            W_recon = Q_R + L2 @ L1

            total_mse = 0.0
            for X_cpu in cached_inputs:
                X = X_cpu.float().to(device)
                Y_orig = X @ W.t()
                Y_approx = (X / s) @ W_recon.t()
                total_mse += (Y_orig - Y_approx).pow(2).sum().item()

            if total_mse < best_mse:
                best_mse = total_mse
                best_alpha = alpha

        return best_alpha

    def apply(self) -> None:
        config = self.config

        if self.data_loader is not None:
            self._calibrate()
        else:
            logger.warning("SVDQuant: no calibration data provided – activation smoothing will be skipped")
            if getattr(config, "use_gptq", False):
                logger.warning("SVDQuant: GPTQ requires calibration data — falling back to RTN")

        per_layer_alpha: dict[str, float] | None = None
        if getattr(config, "search_alpha", False) and self.input_cache is not None and self.input_cache.inputs:
            candidates_cfg = getattr(config, "alpha_candidates", None)
            if candidates_cfg:
                alpha_candidates = list(candidates_cfg)
            else:
                alpha_candidates = [i / 20.0 for i in range(1, 20)]

            logger.info(
                f"SVDQuant: searching per-layer alpha "
                f"({len(alpha_candidates)} candidates, "
                f"{len(self.input_cache.inputs)} layers) ..."
            )
            per_layer_alpha = {}
            for name, module in tqdm(
                list(self.model.named_modules()),
                desc="SVDQuant alpha search",
            ):
                if not self._should_quantize(name, module):
                    continue
                if name not in self.input_cache.inputs:
                    continue
                if name not in self.smoother.activation_max:
                    continue

                best = self._search_alpha(
                    name=name,
                    module=module,
                    cached_inputs=self.input_cache.inputs[name],
                    alpha_candidates=alpha_candidates,
                    svd_rank=config.svd_rank,
                )
                per_layer_alpha[name] = best

            logger.info(f"SVDQuant: found per-layer alphas for {len(per_layer_alpha)} layers")
            if per_layer_alpha:
                vals = list(per_layer_alpha.values())
                logger.info(
                    f"SVDQuant: alpha stats — "
                    f"min={min(vals):.3f}, max={max(vals):.3f}, "
                    f"mean={sum(vals) / len(vals):.3f}"
                )

            self.input_cache.inputs.clear()

        logger.info("SVDQuant: computing smooth factors ...")
        smooth_factors = self.smoother.compute_smooth_factors(
            self.model, config.exclude_patterns, per_layer_alpha=per_layer_alpha
        )

        pre_smooth_dtypes: dict[str, torch.dtype] = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "weight") and name in smooth_factors:
                pre_smooth_dtypes[name] = module.weight.dtype

        applied = apply_smoothing(self.model, smooth_factors, keep_float32=True)
        logger.info(f"SVDQuant: applied {len(applied)} smooth factors")

        if self.hessian_collector is not None:
            self.hessian_collector.adjust_for_smoothing(applied)
            logger.info("SVDQuant: adjusted Hessians for activation smoothing")

        logger.info("SVDQuant: performing SVD decomposition ...")
        replacements: dict[str, ErrorCorrectedModule] = {}

        for name, module in list(self.model.named_modules()):
            if not self._should_quantize(name, module):
                continue

            W = module.weight.data
            orig_dtype = pre_smooth_dtypes.get(name, W.dtype)
            if orig_dtype == torch.float32:
                orig_dtype = torch.bfloat16
            out_features, in_features = W.shape

            L1, L2, R = _svd_decompose(W, rank=config.svd_rank)

            sf = applied.get(name, None)
            if sf is not None:
                # Absorb 1/s into weights so no runtime division is needed.
                # nn.Linear computes x @ W^T, so layer(x/s) = x @ diag(1/s) @ R^T
                # = x @ (R @ diag(1/s))^T.  Same logic for L1.
                inv_s = (1.0 / sf.float()).to(R.device)
                R = R * inv_s.unsqueeze(0)
                L1 = L1 * inv_s.unsqueeze(0)

            module.weight = nn.Parameter(R.to(orig_dtype))

            correction = LowRankCorrectionModule(in_features, out_features, rank=config.svd_rank)
            correction.l1.weight.data = L1.to(orig_dtype)
            correction.l2.weight.data = L2.to(orig_dtype)
            correction = correction.to(dtype=orig_dtype, device=W.device)

            replacements[name] = ErrorCorrectedModule(correction, module, smooth_factor=None)

        logger.info(f"SVDQuant: replacing {len(replacements)} modules")
        for name, new_module in replacements.items():
            _replace_module(self.model, name, new_module)

        if (
            getattr(config, "use_gptq", False)
            and self.hessian_collector is not None
            and self.hessian_collector.hessians
        ):
            self._apply_gptq(replacements)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("SVDQuant: done")

    def _apply_gptq(self, replacements: dict[str, ErrorCorrectedModule]) -> None:
        """Quantise residual weights with GPTQ using collected Hessians."""
        config = self.config
        n_applied = 0
        n_skipped = 0

        logger.info(
            f"SVDQuant-GPTQ: quantising residual weights "
            f"(n_bits={config.gptq_n_bits}, sym={config.gptq_symmetric}, "
            f"group_size={config.gptq_group_size}, "
            f"blocksize={config.gptq_blocksize}, "
            f"percdamp={config.gptq_percdamp}, "
            f"actorder={config.gptq_actorder})"
        )

        assert self.hessian_collector is not None
        hessians = self.hessian_collector.hessians

        for name, ecm in tqdm(
            replacements.items(),
            desc="SVDQuant-GPTQ quantising residual weights",
        ):
            if name not in hessians:
                logger.debug(f"SVDQuant-GPTQ: no Hessian for '{name}' — skipping")
                n_skipped += 1
                continue

            layer = ecm.layer
            H = hessians[name]
            W = layer.weight.data

            Q = gptq_quantize_residual(
                weight=W,
                hessian=H,
                n_bits=config.gptq_n_bits,
                symmetric=config.gptq_symmetric,
                group_size=config.gptq_group_size,
                blocksize=config.gptq_blocksize,
                percdamp=config.gptq_percdamp,
                actorder=config.gptq_actorder,
            )

            layer.weight.data = Q
            n_applied += 1

        self.hessian_collector.hessians.clear()
        self.hessian_collector.nsamples.clear()

        logger.info(f"SVDQuant-GPTQ: applied to {n_applied} layers, skipped {n_skipped}")
