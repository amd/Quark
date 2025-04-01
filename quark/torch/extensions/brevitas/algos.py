#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Pre-quantization optimization and post quantization algorithms for Brevitas API."""
from dataclasses import dataclass
from typing import Optional
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

try:
    import brevitas.graph.equalize  # type: ignore[import-not-found]
    import brevitas.graph.calibrate  # type: ignore[import-not-found]
    import brevitas.graph.gpfq  # type: ignore[import-not-found]
    import brevitas.graph.gptq  # type: ignore[import-not-found]
    import brevitas.graph.quantize  # type: ignore[import-not-found]
except ModuleNotFoundError:
    logger.warning("Brevitas is not installed.")


@dataclass
class PreQuantOptConfig:
    name = ""

    def apply(self,
              model: torch.nn.Module,
              calib_loader: Optional[torch.utils.data.DataLoader] = None) -> torch.nn.Module:  # type: ignore[type-arg]
        raise NotImplementedError('Apply functionality has not been implemented.')


@dataclass
class AlgoConfig:
    name = ""

    def apply(self,
              model: torch.nn.Module,
              calib_loader: Optional[torch.utils.data.DataLoader] = None) -> torch.nn.Module:  # type: ignore[type-arg]
        raise NotImplementedError('Apply functionality has not been implemented.')


class Preprocess(PreQuantOptConfig):
    """
    Preprocesses the model to make it easier to quantize.
    """

    name = "Pre-Processing"

    def __init__(self,
                 trace_model: bool = True,
                 equalize_iterations: int = 20,
                 equalize_merge_bias: bool = True,
                 merge_batch_norm: bool = True,
                 channel_splitting_ratio: float = 0.0,
                 channel_splitting_split_input: bool = False) -> None:
        self.trace_model = trace_model
        self.equalize_iterations = equalize_iterations
        self.equalize_merge_bias = equalize_merge_bias
        self.merge_batch_norm = merge_batch_norm
        self.channel_splitting_ratio = channel_splitting_ratio
        self.channel_splitting_split_input = channel_splitting_split_input

    def apply(self,
              model: torch.nn.Module,
              calib_loader: Optional[torch.utils.data.DataLoader] = None) -> torch.nn.Module:  # type: ignore[type-arg]

        model = brevitas.graph.quantize.preprocess_for_quantize(
            model,
            trace_model=self.trace_model,
            equalize_iters=self.equalize_iterations,
            equalize_merge_bias=self.equalize_merge_bias,
            merge_bn=self.merge_batch_norm,
            channel_splitting_ratio=self.channel_splitting_ratio,
            channel_splitting_split_input=self.channel_splitting_split_input)

        return model  # type: ignore[no-any-return]


class ActivationEqualization(PreQuantOptConfig):
    """
    Activation Equalization from the paper "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" by Nagel et al.

    - `is_layerwise`: Whether the model having ActivationEqualization applied to it is using Backend.layerwise for its quantization or not.
    """
    name = "Activation Equalization"

    def __init__(self, is_layerwise: bool = True, alpha: float = 0.5):
        self.is_layerwise = is_layerwise
        self.alpha = alpha

    def apply(self,
              model: torch.nn.Module,
              calib_loader: Optional[torch.utils.data.DataLoader] = None) -> torch.nn.Module:  # type: ignore[type-arg]
        if calib_loader is None:
            raise ValueError("Activation Equalization requires calibration data.")

        model.eval()
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device

        with torch.no_grad():
            with brevitas.graph.equalize.activation_equalization_mode(model,
                                                                      alpha=self.alpha,
                                                                      layerwise=self.is_layerwise,
                                                                      add_mul_node=self.is_layerwise):
                for i, (images, target) in enumerate(tqdm(calib_loader)):
                    images = images.to(device)
                    images = images.to(dtype)
                    model(images)

        return model


class GPFQ(AlgoConfig):
    """
    GPFQ or Greedy Path Following Quantization from the papers
    - "Post-training Quantization for Neural Networks with Provable Guarantees" by Zhang et al. and
    - "A Greedy Algorithm for Quantizing Neural Networks" by Lybrand et al.
    """

    name = "GPFQ"

    def __init__(self, act_order: bool = False, percentage_of_processed_inputs: float = 1.0) -> None:
        self.act_order = act_order
        self.percentage_of_processed_inputs = percentage_of_processed_inputs

    def apply(self,
              model: torch.nn.Module,
              calib_loader: Optional[torch.utils.data.DataLoader] = None) -> torch.nn.Module:  # type: ignore[type-arg]
        if calib_loader is None:
            raise ValueError("GPFQ requires calibration data.")

        model.eval()
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device

        with torch.no_grad():
            with brevitas.graph.gpfq.gpfq_mode(model,
                                               p=self.percentage_of_processed_inputs,
                                               use_quant_activations=True,
                                               act_order=self.act_order,
                                               use_gpfa2q=False,
                                               accumulator_bit_width=None) as gpfq:
                gpfq_model = gpfq.model
                for i in tqdm(range(gpfq.num_layers)):
                    for i, (images, target) in enumerate(calib_loader):
                        images = images.to(device)
                        images = images.to(dtype)
                        gpfq_model(images)
                    gpfq.update()

        return model


class GPFA2Q(AlgoConfig):
    """
    Extension of GPFQ using A2Q or Accumulator-Aware Quantization from the paper "A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance" by Colbert et al.
    """

    name = "GPFA2Q"

    def __init__(self,
                 act_order: bool = False,
                 percentage_of_processed_inputs: float = 1.0,
                 accumulator_bit_width: int = 16) -> None:
        self.act_order = act_order
        self.percentage_of_processed_inputs = percentage_of_processed_inputs
        self.accumulator_bit_width = accumulator_bit_width

    def apply(self,
              model: torch.nn.Module,
              calib_loader: Optional[torch.utils.data.DataLoader] = None) -> torch.nn.Module:  # type: ignore[type-arg]
        if calib_loader is None:
            raise ValueError("GPFA2Q requires calibration data.")

        model.eval()
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device

        with torch.no_grad():
            with brevitas.graph.gpfq.gpfq_mode(model,
                                               p=self.percentage_of_processed_inputs,
                                               use_quant_activations=True,
                                               act_order=self.act_order,
                                               use_gpfa2q=True,
                                               accumulator_bit_width=self.accumulator_bit_width) as gpfq:
                gpfq_model = gpfq.model
                for i in tqdm(range(gpfq.num_layers)):
                    for i, (images, target) in enumerate(calib_loader):
                        images = images.to(device)
                        images = images.to(dtype)
                        gpfq_model(images)
                    gpfq.update()

        return model


class GPTQ(AlgoConfig):
    """
    GPTQ or Generative Pre-Trained Transformers Quantization from the paper "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" by Frantar et al.
    """

    name = "GPTQ"

    def __init__(self, act_order: bool = False) -> None:
        self.act_order = act_order

    def apply(self,
              model: torch.nn.Module,
              calib_loader: Optional[torch.utils.data.DataLoader] = None) -> torch.nn.Module:  # type: ignore[type-arg]
        if calib_loader is None:
            raise ValueError("GPTQ calibration requires calibration data.")

        model.eval()
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        with torch.no_grad():
            with brevitas.graph.gptq.gptq_mode(model, act_order=self.act_order, use_quant_activations=False) as gptq:
                gptq_model = gptq.model
                for i in tqdm(range(gptq.num_layers)):
                    for i, (images, target) in enumerate(calib_loader):
                        images = images.to(device)
                        images = images.to(dtype)
                        gptq_model(images)
                    gptq.update()

        return model


class CalibrateBatchNorm(AlgoConfig):
    name = "Calibrate Batch Norm"

    def apply(self,
              model: torch.nn.Module,
              calib_loader: Optional[torch.utils.data.DataLoader] = None) -> torch.nn.Module:  # type: ignore[type-arg]
        if calib_loader is None:
            raise ValueError("Batch normalization calibration requires calibration data.")

        model.eval()
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        with torch.no_grad():
            with brevitas.graph.calibrate.norm_correction_mode(model):
                for i, (images, target) in enumerate(tqdm(calib_loader)):
                    images = images.to(device)
                    images = images.to(dtype)
                    model(images)

        return model


class BiasCorrection(AlgoConfig):
    """
    Bias correction from the paper "Data-Free Quantization Through Weight Equalization and Bias Correction" by Nagel et al.
    """
    name = "Bias Correction"

    def apply(self,
              model: torch.nn.Module,
              calib_loader: Optional[torch.utils.data.DataLoader] = None) -> torch.nn.Module:  # type: ignore[type-arg]
        if calib_loader is None:
            raise ValueError("Bias correction requires calibration data.")

        model.eval()
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        with torch.no_grad():
            with brevitas.graph.calibrate.bias_correction_mode(model):
                for i, (images, target) in enumerate(tqdm(calib_loader)):
                    images = images.to(device)
                    images = images.to(dtype)
                    model(images)

        return model


def _calibrate(
        calib_loader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
        model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with brevitas.graph.calibrate.calibration_mode(model):
            for i, (images, target) in enumerate(tqdm(calib_loader)):
                images = images.to(device)
                images = images.to(dtype)
                model(images)
    return model
