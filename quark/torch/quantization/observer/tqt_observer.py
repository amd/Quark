#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from quark.torch.quantization.observer.observer import UniformScalingObserver
from quark.torch.quantization.config.config import QuantizationSpec, TQTSpec
from quark.torch.quantization.config.type import TQTThresholdInitMeth
from quark.torch.quantization.utils import get_num_bits
import numpy as np
import torch
from typing import Tuple, Any, List, Optional


# TODO: @Ruiying Add TQTObserver here
class TQTObserver(UniformScalingObserver):

    def __init__(self, qspec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__(qspec)
        _num_bits = get_num_bits(qspec.dtype)
        assert isinstance(_num_bits, int)
        self.register_buffer("_bitwidth", torch.tensor([_num_bits], dtype=torch.uint8))
        self.register_buffer("_domain", torch.tensor([2**(_num_bits - 1)]).float())
        self.register_buffer("_warmup_enable", torch.tensor([1], dtype=torch.uint8))
        self._zero_point = torch.tensor([0], dtype=torch.int)
        self._log_threshold = torch.nn.Parameter(torch.tensor([0.0]))
        assert isinstance(qspec.qat_spec, TQTSpec)
        self.threshold_init_meth = qspec.qat_spec.threshold_init_meth

    def forward(self, x_orig: torch.Tensor) -> None:
        if self._warmup_enable[0] == 1:
            data = x_orig.cpu().numpy()
            if self.threshold_init_meth == TQTThresholdInitMeth._3SD:
                self._log_threshold.data[0] = torch.log2(
                    torch.tensor(self._3SD(data), dtype=x_orig.dtype, device=x_orig.device))
            elif self.threshold_init_meth == TQTThresholdInitMeth._KL_J:
                self._log_threshold.data[0] = torch.log2(
                    torch.tensor(self._KL_J(data, self._bitwidth), dtype=x_orig.dtype, device=x_orig.device))
            self._log_threshold.data = self._log_threshold.to(x_orig.device)
            self._domain.data = self._domain.to(x_orig.device)
            self._warmup_enable[0] = 0

    def _3SD(self, x: np.ndarray[Any, np.dtype[Any]]) -> Any:
        y = x.astype(np.float32) if x.dtype == np.float16 else x
        return np.abs(np.mean(y + 1e-6)) + 3 * np.std(y)

    def _KL_J(self, x: np.ndarray[Any, np.dtype[Any]], bitwidth: torch.Tensor) -> Any:
        """
        Ref paper (Algorithm 1):
        "Quantizing Convolutional Neural Networks for Low-Power
        High-Throughput Inference Engines" - Sean Settle et al.
        https://arxiv.org/pdf/1805.07941.pdf.
        """

        def calculate_kl_j(x: np.ndarray[Any, np.dtype[Any]], y: np.ndarray[Any, np.dtype[Any]]) -> Any:
            return np.sum((x - y) * np.log2(x / y))

        mn = 0
        mx = np.max(np.abs(x))
        y = x.astype(np.float32) if x.dtype == np.float16 else x
        hist, bin_edges = np.histogram((np.abs(y)), 'sqrt', range=(mn, mx), density=True)
        hist = hist.astype(x.dtype)
        bin_edges = bin_edges.astype(x.dtype)
        pdf = hist / np.sum(hist)
        cdf = np.cumsum(pdf)
        n = pow(2, bitwidth.item() - 1)
        threshold: List[Any] = []
        d: List[Any] = []
        if n + 1 > len(bin_edges) - 1:
            return bin_edges[(-1)]
        else:
            for i in range(int(n + 1), len(bin_edges), 1):
                threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
                threshold = np.concatenate((threshold, [threshold_tmp]))
                p = np.copy(cdf)
                p[i - 1:] = 1
                x = np.linspace(0.0, 1.0, int(n))
                xp = np.linspace(0.0, 1.0, i)
                fp = p[:i]
                p_interp = np.interp(x, xp, fp)
                x = np.linspace(0.0, 1.0, i)
                xp = np.linspace(0.0, 1.0, int(n))
                fp = p_interp
                q_interp = np.interp(x, xp, fp)
                q = np.copy(p)
                q[:i] = q_interp
                d_tmp = calculate_kl_j(cdf[np.nonzero(cdf)], q[np.nonzero(cdf)])
                d = np.concatenate((d, [d_tmp]))

            return threshold[np.argmin(d)]

    def get_fix_position(self) -> int:
        """
        (1) TQT: qx = clip(round(fx / scale)) * scale, scale = 2^ceil(log2t) / 2^(b-1)
        (2) NndctFixNeron: qx = clip(round(fx * scale)) * (1 / scale), scale = 2^fp
        Let (1) equals (2), we can get
        (3): 2^(b-1) / 2^ceil(log2t) = 2^fp
         => fp = b - 1 - ceil(log2t)

        For more details, see nndct/include/cuda/nndct_fix_kernels.cuh::_fix_neuron_v2_device
        """
        bitwidth = self._bitwidth.item()
        ceil_log2t = torch.ceil(self._log_threshold).item()
        return int(bitwidth - 1 - ceil_log2t)

    def _calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        fp = self.get_fix_position()
        scale = torch.tensor([2**(-fp)], dtype=torch.float)
        return scale, self._zero_point

    @property
    def domain(self) -> Any:
        return self._domain

    @property
    def log_threshold(self) -> torch.Tensor:
        return self._log_threshold

    @log_threshold.setter
    def log_threshold(self, scale: torch.Tensor) -> None:
        fix_position = torch.log2(1 / scale)
        self._log_threshold.data = torch.tensor([self.bitwidth - 1 - fix_position], dtype=self._log_threshold.dtype)
        self._warmup_enable[0] = 0
