#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization Algorithm Config API for ONNX"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from quark.common.utils.log import ScreenLogger

from .utils import config_to_dict

logger = ScreenLogger(__name__)


class AlgoConfig(ABC):
    @abstractmethod
    def _get_config(self, extra_options: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError()

    def to_dict(self) -> dict[str, Any]:
        return config_to_dict(self)


class SmoothQuantConfig(AlgoConfig):
    """Configuration for the Smooth Quant algorithm, which is originally proposed in the following paper:
    "Guangxuan Xiao et al., SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models,
    arXiv:2211.10438, 2022."

    SmoothQuant is a PTQ algorithm designed to reduce the accuracy drop when quantizing
    large language models (LLMs), especially for transformer architectures. It tackles
    one of the key issues in activation quantization: the mismatch in dynamic ranges
    between weights and activations across different layers.

    The core idea is to smooth out the activation and weight ranges by inserting a
    scaling factor that shifts some of the variation in activations into the weights.

    SmoothQuant requires only a small set of calibration data and no model retraining.
    By aligning the quantization ranges, it minimizes information loss in layers like
    attention or MLP, leading to much better accuracy retention. It has proven particularly
    effective for large models such as OPT, BLOOM, and GPT-like architectures under INT8 quantization.

    :param float alpha: A parameter in SmoothQuant that controls the trade-off between shifting activation range into weights and preserving the original distribution,
                        enabling optimal balancing for quantization accuracy. Defaults to 0.5.
    """

    def __init__(self, alpha: float = 0.5):
        self.name: str = "smooth_quant"
        self.alpha = alpha

    def _get_config(self, extra_options: dict[str, Any]) -> dict[str, Any]:
        smooth_quant_config = dict()
        if "SmoothAlpha" not in extra_options:
            smooth_quant_config["SmoothAlpha"] = self.alpha
        return smooth_quant_config


class CLEConfig(AlgoConfig):
    """Configuration for the CLE algorithm, which is originally proposed in the following paper:
    "Markus Nagel et al., Data-Free Quantization Through Weight Equalization and Bias Correction,
    arXiv:1906.04721, 2019."

    CLE (Cross-Layer Equalization) is a pre-processing technique used in PTQ that improves
    the quantization robustness of deep neural networks by reducing the range imbalance across layers.
    It operates by scaling the weights of adjacent layers in such a way that their output distributions
    become more uniform, minimizing the dynamic range mismatch that often causes quantization errors.

    The core idea behind CLE is that certain operations (like ReLU activations) are scale-invariant,
    meaning you can scale the output of one layer and inversely scale the next without affecting
    the final output. CLE leverages this property to propagate scale adjustments across consecutive layers,
    typically convolutional or linear layers followed by batch norm or ReLU.

    CLE does not require retraining, and it’s particularly effective when applied to networks that have large
    layer-wise scale imbalances. By smoothing out these differences before quantization, CLE helps
    preserve accuracy and stabilizes quantized inference in a lightweight, calibration-only pipeline.

    :param str cle_balance_method: The balance method of CLE. Defaults to "max".
    :param int cle_steps: The steps for CrossLayerEqualization execution. When set to -1, an adaptive CrossLayerEqualization will be conducted. Defaults to 1.
    :param float cle_weight_threshold: The threshold of the scale of the weights when calculating them. Defulats to 0.5.
    :param bool cle_scale_append_bias: Whether the bias be included when calculating the scale of the weights. Defaults to True.
    :param bool cle_scale_use_threshold: Whether use the threshold when calculating the scale of the wegiths. Defaults to True.
    :param float cle_total_layer_diff_threshold: The threshold represents the sum of mean transformations of CrossLayerEqualization transformations across all layers. Defaults to 1.9e-7.
    """

    def __init__(
        self,
        cle_balance_method: str = "max",
        cle_steps: int = 1,
        cle_weight_threshold: float = 0.5,
        cle_scale_append_bias: bool = True,
        cle_scale_use_threshold: bool = True,
        cle_total_layer_diff_threshold: float = 1.9e-7,
    ) -> None:
        self.name: str = "cle"
        self.cle_balance_method = cle_balance_method
        self.cle_steps = cle_steps
        self.cle_weight_threshold = cle_weight_threshold
        self.cle_scale_append_bias = cle_scale_append_bias
        self.cle_scale_use_threshold = cle_scale_use_threshold
        self.cle_total_layer_diff_threshold = cle_total_layer_diff_threshold

    def _get_config(self, extra_options: dict[str, Any]) -> dict[str, Any]:
        cle_config: dict[str, Any] = dict()
        if "CLEBalanceMethod" not in extra_options:
            cle_config["CLEBalanceMethod"] = self.cle_balance_method
        if "CLESteps" not in extra_options:
            cle_config["CLESteps"] = self.cle_steps
        if "CLEWeightThreshold" not in extra_options:
            cle_config["CLEWeightThreshold"] = self.cle_weight_threshold
        if "CLEScaleAppendBias" not in extra_options:
            cle_config["CLEScaleAppendBias"] = self.cle_scale_append_bias
        if "CLEScaleUseThreshold" not in extra_options:
            cle_config["CLEScaleUseThreshold"] = self.cle_scale_use_threshold
        if "CLETotalLayerDiffThreshold" not in extra_options:
            cle_config["CLETotalLayerDiffThreshold"] = self.cle_total_layer_diff_threshold
        return cle_config


class BiasCorrectionConfig(AlgoConfig):
    """Configuration for the Bias Correction algorithm, which is originally proposed in the following paper:
    "Markus Nagel et al., Data-Free Quantization Through Weight Equalization and Bias Correction,
    arXiv:1906.04721, 2019."

    Bias Correction is a PTQ technique designed to reduce the quantization-induced shift in
    a neural network's output by adjusting the bias terms in layers like convolution or linear.
    It computes the difference (bias error) between the original float model and the quantized model outputs
    using a small calibration dataset. It then adjusts the biases of the affected layers so that
    the quantized model better matches the float model’s behavior, particularly at the layer output level.

    This method is simple, data-efficient (requiring no retraining), and effective at improving
    accuracy—especially for models that are sensitive to quantization noise, such as those with
    small activations or low-bit quantization like INT8.

    """

    def __init__(self) -> None:
        self.name: str = "bias_correction"

    def _get_config(self, extra_options: dict[str, Any]) -> dict[str, Any]:
        bias_correction_config: dict[str, Any] = dict()
        bias_correction_config["BiasCorrection"] = True
        return bias_correction_config


class GPTQConfig(AlgoConfig):
    """Configuration for the GPTQ algorithm, which is originally proposed in the following paper:
    "Elias Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers,
    arXiv:2210.17323, 2022."

    GPTQ is an efficient PTQ algorithm for compressing LLMs. It quantizes weights layer-by-layer and
    column-by-column within each layer. Crucially, when quantizing one column, it calculates the error and
    updates subsequent unquantized columns using an approximate Hessian matrix to minimize output distortion.
    This error correction step preserves accuracy far better than simple rounding.

    The result is near-original model accuracy at ultra-low precision (e.g., 4-bit) with fast,
    single-GPU quantization. This makes GPTQ a key technique for efficient LLM deployment.

    :param int bits: The quantization bits used in GPTQ. Defaults to 8.
    :param int block_size: The block size in GPTQ determines how many columns of weights will be quantized for one update. Defaults to 128.
    :param int group_size: The group size in GPTQ determines how many columns of weights share one set of scale and zero-point. Defaults is -1.
    :param float perc_damp: Percent of the average Hessian diagonal to use for dampening. Defaults to 0.01.
    :param bool act_order: Whether to re-order Hessian matrix according the values of diag. Defulats to False.
    :param bool per_channel: Whether to perform per-channel quantization in GPTQ. Defaults to False.
    :param bool mse: Whether to use MSE method to do data calibration in GPTQ. Defaults to False.
    :param bool weight_symmetric: Whether to only quantize weights of the model. Defaults to True.
    """

    def __init__(
        self,
        bits: int = 8,
        block_size: int = 128,
        group_size: int = -1,
        perc_damp: float = 0.01,
        act_order: bool = False,
        per_channel: bool = False,
        mse: bool = False,
        weight_symmetric: bool = True,
    ) -> None:
        self.name: str = "gptq"
        self.bits = bits
        self.block_size = block_size
        self.group_size = group_size
        self.perc_damp = perc_damp
        self.act_order = act_order
        self.per_channel = per_channel
        self.mse = mse
        self.weight_symmetric = weight_symmetric

    def _get_config(self, extra_options: dict[str, Any]) -> dict[str, Any]:
        gptq_config: dict[str, Any] = dict()
        gptq_config["UseGPTQ"] = True
        gptq_config["GPTQParams"] = {}
        if "GPTQParams" not in extra_options:
            extra_options["GPTQParams"] = {}
        if "Bits" not in extra_options["GPTQParams"]:
            gptq_config["GPTQParams"]["Bits"] = self.bits
        if "BlockSize" not in extra_options["GPTQParams"]:
            gptq_config["GPTQParams"]["BlockSize"] = self.block_size
        if "PercDamp" not in extra_options["GPTQParams"]:
            gptq_config["GPTQParams"]["PercDamp"] = self.perc_damp
        if "GroupSize" not in extra_options["GPTQParams"]:
            gptq_config["GPTQParams"]["GroupSize"] = self.group_size
        if "ActOrder" not in extra_options["GPTQParams"]:
            gptq_config["GPTQParams"]["ActOrder"] = self.act_order
        if "PerChannel" not in extra_options["GPTQParams"]:
            gptq_config["GPTQParams"]["PerChannel"] = self.per_channel
        if "WeightSymmetric" not in extra_options["GPTQParams"]:
            gptq_config["GPTQParams"]["WeightSymmetric"] = self.weight_symmetric
        if "MSE" not in extra_options["GPTQParams"]:
            gptq_config["GPTQParams"]["MSE"] = self.mse
        return gptq_config


class AutoMixprecisionConfig(AlgoConfig):
    """Configuration for the automatic mixed precision.

    Mixed precision is a highly effective technique in the field of quantization. When low-bit quantization
    leads to poor accuracy, quantizing part of the tensors or layers with higher bit-width can often
    significantly improve the overall quantization accuracy.

    Automatic mixed-precision algorithms can automatically identify tensors or layers that suffer from
    low-bit quantization errors and replace them with higher-bit quantization, thereby enhancing the
    final model performance.

    :param int data_size: The size of the data used for mix-precision. Defaults to 10000000.
    :param Tuple[str, ...] target_op_type: The user defined op type set for mix-precision. Defaults to (‘Conv’, ‘ConvTranspose’, ‘Gemm’, ‘MatMul’).
    :param QuantType target_quant_type: Activation data type to be mixed in the model if 'act_target_quant_type' is not given.
                                        Error will be raised if 'target_quant_type', 'act_target_quant_type' and 'weight_target_quant_type' are not given.
    :param QuantType act_target_quant_type: Activation data type to be mixed in the model. If both 'act_target_quant_type' and 'weight_target_quant_type' are not specified,
                                            the 'act_target_quant_type' will be same as 'target_quant_type'.
                                            If only 'act_target_quant_type' is not specified, it will be the original activation_type.
    :param QuantType weight_target_quant_type: Weight data type to be mixed in the model. If both 'act_target_quant_type' and 'weight_target_quant_type' are not specified,
                                               the 'weight_target_quant_type' will be same as 'target_quant_type'.
                                               If only 'weight_target_quant_type' is not specified, it will be the original weight_type.
    :param QuantType bias_target_quant_type: Bias data type to be mixed in the model. If 'bias_target_quant_type' is not specified and Int32Bias is True,
                                             the 'bias_target_quant_type' will be int32. If 'bias_target_quant_type' is not specified and Int32Bias is False,
                                             the 'bias_target_quant_type' will be same as 'weight_target_quant_type'.
    :param bool dual_quant_nodes: Some backend compilers require that two types of quantization nodes exist simultaneously on the tensors which connect two different precision nodes,
                                  for example, they require the tensor that connects BFP16 Conv and BF16 Reshape has a BFP node and a QDQ pair both. Defaults to False.
    :param int output_index: The index of model output to be calculated for loss. Defaults to 0.
    :param float l2_target: The L2 metric as a target. Defaults to 0.5.
    :param Optional[float] top1_acc_target: The Top1 accuracy as a target. Defaults to None.
    :param Any evaluate_function: The function to measure top1 accuracy loss. Input of the function is model output(numpy tensor),
                                  output of the function is top1 accuracy(between 0~1).
                                  If 'evaluate_function' is not specified while 'top1_acc_target' is given, error will be raised.
    :param int num_target: The number of nodes for mix-precision to minimize the loss. Defaults to 0.
    :param List[str] target_tensors: The names of nodes to mix into the target quant type. Defaults to [].
    :param List[str] target_indices: The indices (based on sensitivity analysis results) of the nodes to mix into the target quant type. Defaults to [].
    :param List[str] exclude_indices: The indices (based on sensitivity analysis results) of the nodes not to mix into the target quant type. Defaults to [].
    :param bool no_input_qdq_shared: Whether to skip the nodes who shared the input Q/DQ pair with other nodes. Defaults to True.
    :param bool auto_mix_use_fast_ft: Whether to perform fast finetune to improve accuracy after mixed a layer. Defaults to False.
    """

    def __init__(
        self,
        data_size: int = 10000000,
        target_op_type: tuple[str, ...] = ("Conv", "ConvTranspose", "Gemm", "MatMul"),
        target_quant_type: Any = None,
        act_target_quant_type: Any = None,
        weight_target_quant_type: Any = None,
        bias_target_quant_type: Any = None,
        dual_quant_nodes: bool = False,
        output_index: int = 0,
        l2_target: float = 0.5,
        top1_acc_target: float | None = None,
        evaluate_function: Any = None,
        num_target: int = 0,
        target_tensors: list[str] = [],
        target_indices: list[Any] = [],
        exclude_indices: list[Any] = [],
        no_input_qdq_shared: bool = True,
        auto_mix_use_fast_ft: bool = False,
    ) -> None:
        self.name: str = "auto_mixprecision"
        self.data_size = data_size
        self.target_op_type = target_op_type
        self.target_quant_type = target_quant_type
        self.act_target_quant_type = act_target_quant_type
        self.weight_target_quant_type = weight_target_quant_type
        self.bias_target_quant_type = bias_target_quant_type
        self.dual_quant_nodes = dual_quant_nodes
        self.output_index = output_index
        self.l2_target = l2_target
        self.top1_acc_target = top1_acc_target
        self.evaluate_function = evaluate_function
        self.num_target = num_target
        self.target_tensors = target_tensors
        self.target_indices = target_indices
        self.exclude_indices = exclude_indices
        self.no_input_qdq_shared = no_input_qdq_shared
        self.auto_mix_use_fast_ft = auto_mix_use_fast_ft

    def _get_config(self, extra_options: dict[str, Any]) -> dict[str, Any]:
        auto_mixprecision_config: dict[str, Any] = dict()
        auto_mixprecision_config["AutoMixprecision"] = {}
        if "AutoMixprecision" not in extra_options:
            extra_options["AutoMixprecision"] = {}
        if "DataSize" not in extra_options["AutoMixprecision"]:
            auto_mixprecision_config["AutoMixprecision"]["DataSize"] = self.data_size
        if "TargetOpType" not in extra_options["AutoMixprecision"]:
            auto_mixprecision_config["AutoMixprecision"]["TargetOpType"] = self.target_op_type
        if "TargetQuantType" not in extra_options["AutoMixprecision"]:
            if self.target_quant_type is not None:
                auto_mixprecision_config["AutoMixprecision"]["TargetQuantType"] = self.target_quant_type.map_onnx_format
            else:
                auto_mixprecision_config["AutoMixprecision"]["TargetQuantType"] = self.target_quant_type
        if "ActTargetQuantType" not in extra_options["AutoMixprecision"]:
            if self.act_target_quant_type is not None:
                auto_mixprecision_config["AutoMixprecision"]["ActTargetQuantType"] = (
                    self.act_target_quant_type.map_onnx_format
                )
            else:
                auto_mixprecision_config["AutoMixprecision"]["ActTargetQuantType"] = self.act_target_quant_type
        if "WeightTargetQuantType" not in extra_options["AutoMixprecision"]:
            if self.weight_target_quant_type is not None:
                auto_mixprecision_config["AutoMixprecision"]["WeightTargetQuantType"] = (
                    self.weight_target_quant_type.map_onnx_format
                )
            else:
                auto_mixprecision_config["AutoMixprecision"]["WeightTargetQuantType"] = self.weight_target_quant_type
        if "BiasTargetQuantType" not in extra_options["AutoMixprecision"]:
            if self.bias_target_quant_type is not None:
                auto_mixprecision_config["AutoMixprecision"]["BiasTargetQuantType"] = (
                    self.bias_target_quant_type.map_onnx_format
                )
            else:
                auto_mixprecision_config["AutoMixprecision"]["BiasTargetQuantType"] = self.bias_target_quant_type
        if "DualQuantNodes" not in extra_options["AutoMixprecision"]:
            auto_mixprecision_config["AutoMixprecision"]["DualQuantNodes"] = self.dual_quant_nodes
        if "OutputIndex" not in extra_options["AutoMixprecision"]:
            auto_mixprecision_config["AutoMixprecision"]["OutputIndex"] = self.output_index
        if "L2Target" not in extra_options["AutoMixprecision"]:
            auto_mixprecision_config["AutoMixprecision"]["L2Target"] = self.l2_target
        if "Top1AccTarget" not in extra_options["AutoMixprecision"]:
            auto_mixprecision_config["AutoMixprecision"]["Top1AccTarget"] = self.top1_acc_target
        if "EvaluateFunction" not in extra_options["AutoMixprecision"]:
            auto_mixprecision_config["AutoMixprecision"]["EvaluateFunction"] = self.evaluate_function
        if "NumTarget" not in extra_options["AutoMixprecision"]:
            auto_mixprecision_config["AutoMixprecision"]["NumTarget"] = self.num_target
        if "TargetTensors" not in extra_options["AutoMixprecision"]:
            auto_mixprecision_config["AutoMixprecision"]["TargetTensors"] = self.target_tensors
        if "TargetIndices" not in extra_options["AutoMixprecision"]:
            auto_mixprecision_config["AutoMixprecision"]["TargetIndices"] = self.target_indices
        if "ExcludeIndices" not in extra_options["AutoMixprecision"]:
            auto_mixprecision_config["AutoMixprecision"]["ExcludeIndices"] = self.exclude_indices
        if "NoInputQDQShared" not in extra_options["AutoMixprecision"]:
            auto_mixprecision_config["AutoMixprecision"]["NoInputQDQShared"] = self.no_input_qdq_shared
        if "AutoMixUseFastFT" not in extra_options["AutoMixprecision"]:
            auto_mixprecision_config["AutoMixprecision"]["AutoMixUseFastFT"] = self.auto_mix_use_fast_ft
        return auto_mixprecision_config


class AdaRoundConfig(AlgoConfig):
    """Configuration for the AdaRound algorithm, which is originally proposed in the following paper:
    "Markus Nagel et al., Up or Down? Adaptive Rounding for Post-Training Quantization,
    arXiv:2004.10568, 2020."

    AdaRound (Adaptive Rounding) is a post-training quantization method that
    aims to mitigate the accuracy degradation caused by rounding during quantization.
    Traditional quantization methods often use a simple rounding scheme (e.g.,
    round-to-nearest) to convert floating-point values to their quantized integer
    representation. This can lead to a significant loss of information, especially
    in deep neural networks.

    AdaRound addresses this by treating the rounding decision as a learnable
    parameter. Instead of deterministically rounding up or down, it introduces a
    soft rounding function and optimizes the rounding direction for each weight.
    The optimization is performed using a limited amount of unlabeled data
    (calibration data) to minimize the difference between the floating-point model's
    output and the quantized model's output. The objective function typically
    includes a reconstruction loss term to minimize the L2 distance between the
    original and quantized weight tensors, and a regularization term that
    encourages the soft rounding parameters to converge to either 0 or 1,
    corresponding to rounding down or up, respectively.

    The key idea behind AdaRound is to find the optimal rounding decisions for each
    weight, such that the overall model's performance is preserved after quantization.

    :param str optim_device: The device for optimization. Defaults to "cpu".
    :param str infer_device: The device for inference. Defaults to "cpu".
    :param int fixed_seed: A fixed seed for reproducibility. Defaults to 1705472343.
    :param int data_size: The total size of the dataset. Defaults to 1000000000.
    :param int batch_size: The batch size for optimization. Defaults to 1.
    :param int num_batches: The number of batches for optimization. Defaults to 1.
    :param int num_iterations: The number of optimization iterations. Defaults to 1000.
    :param float learning_rate: The learning rate for optimization. Defaults to 1e-1.
    :param bool early_stop: Whether to use early stopping. Defaults to False.
    :param int output_index: The index of the model's output to use for loss calculation. Defaults to 0.
    :param Optional[Tuple[float, float]] lr_adjust: Learning rate adjustment parameters. Defaults to None.
    :param List[str] target_op_type: List of operator types to be quantized. Defaults to ["Conv", "ConvTranspose", "Gemm", "MatMul", "InstanceNormalization", "LayerNormalization"].
    :param bool selective_update: Whether to selectively update weights. Defaults to False.
    :param bool update_bias: Whether to update the bias terms. Defaults to False.
    :param bool output_qdq: Whether to output QDQ format. Defaults to False.
    :param float drop_ratio: The ratio of weights to drop. Defaults to 1.0.
    :param int mem_opt_level: Memory optimization level. Defaults to 1.
    :param Optional[str] cache_dir: Directory for caching. Defaults to None.
    :param int log_period: Logging period. Defaults to 100.
    :param Optional[str] ref_model_path: Path to the reference model. Defaults to None.
    :param bool dynamic_batch: Whether to use dynamic batching. Defaults to False.
    :param bool parallel: Whether to use parallel processing. Defaults to False.
    :param float reg_param: The regularization parameter for the rounding loss.
                            This controls the trade-off between minimizing the reconstruction error and forcing the rounding parameters to be binary.
                            Defaults to 0.01.
    :param Tuple[float, float] beta_range: The range of the temperature parameter 'beta'.
                                           the 'beta' controls the sharpness of the soft rounding function.
                                           It is annealed from the first value to the second value over the course of optimization.
                                           A high 'beta' at the beginning allows for more exploration,
                                           while a low 'beta' at the end encourages convergence to a binary solution. Defaults to (20, 2).
    :param float warm_start: The fraction of total iterations for the "warm start" phase.
                             During this phase, only the reconstruction loss is used, and the regularization term is gradually introduced.
                             This helps to find a good initial state before forcing the rounding decisions to be binary. Defaults to 0.2.
    :param bool select_max_mem_layer: Whether to select the layer with largest estimated memory usage to run.
    :param int num_workers: Number of subprocesses used for data loading.
                            - 0 means the data will be loaded in the main process.
                            - >0 enables multi-process data loading, which can significantly speed up data pipeline when dataset and transforms are heavy.
                            Note: Using multiple workers increases CPU usage and may require careful handling of worker-safe code.
    :param bool pin_memory: If True, the DataLoader will copy tensors into CUDA pinned memory before returning them.
    :param bool use_gds: If True, with optim_device equals to 'cuda' and mem_opt_level equals to 2  on special GPU, the GDS (GPU Direct Storage) pipeline will be built to speedup the fine tuning process.
    """

    def __init__(
        self,
        optim_device: str = "cpu",
        infer_device: str = "cpu",
        fixed_seed: int = 1705472343,
        data_size: int = 1000000000,
        batch_size: int = 1,
        num_batches: int = 1,
        num_iterations: int = 1000,
        learning_rate: float = 1e-1,
        early_stop: bool = False,
        output_index: int = 0,
        lr_adjust: tuple[float, float] | None = None,
        target_op_type: list[str] = [
            "Conv",
            "ConvTranspose",
            "Gemm",
            "MatMul",
            "InstanceNormalization",
            "LayerNormalization",
        ],
        selective_update: bool = False,
        update_bias: bool = False,
        output_qdq: bool = False,
        drop_ratio: float = 1.0,
        mem_opt_level: int = 1,
        cache_dir: str | None = None,
        log_period: int = 100,
        ref_model_path: str | None = None,
        dynamic_batch: bool = False,
        parallel: bool = False,
        reg_param: float = 0.01,
        beta_range: tuple[float, float] = (20, 2),
        warm_start: float = 0.2,
        select_max_mem_layer: bool = False,
        num_workers: int = 1,
        pin_memory: bool = False,
        use_gds: bool = False,
    ) -> None:
        self.name: str = "adaround"
        self.optim_device = optim_device
        self.infer_device = infer_device
        self.fixed_seed = fixed_seed
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.early_stop = early_stop
        self.output_index = output_index
        self.lr_adjust = lr_adjust
        self.target_op_type = target_op_type
        self.selective_update = selective_update
        self.update_bias = update_bias
        self.output_qdq = output_qdq
        self.drop_ratio = drop_ratio
        self.mem_opt_level = mem_opt_level
        self.cache_dir = cache_dir
        self.log_period = log_period
        self.ref_model_path = ref_model_path
        self.dynamic_batch = dynamic_batch
        self.parallel = parallel
        self.reg_param = reg_param
        self.beta_range = beta_range
        self.warm_start = warm_start
        self.select_max_mem_layer = select_max_mem_layer
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_gds = use_gds

    def _get_config(self, extra_options: dict[str, Any]) -> dict[str, Any]:
        adaround_config: dict[str, Any] = dict()
        adaround_config["FastFinetune"] = {}
        if "FastFinetune" not in extra_options:
            extra_options["FastFinetune"] = {}
        if "OptimAlgorithm" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["OptimAlgorithm"] = self.name
        if "OptimDevice" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["OptimDevice"] = self.optim_device
        if "InferDevice" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["InferDevice"] = self.infer_device
        if "FixedSeed" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["FixedSeed"] = self.fixed_seed
        if "DataSize" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["DataSize"] = self.data_size
        if "BatchSize" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["BatchSize"] = self.batch_size
        if "NumBatches" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["NumBatches"] = self.num_batches
        if "NumIterations" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["NumIterations"] = self.num_iterations
        if "LearningRate" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["LearningRate"] = self.learning_rate
        if "EarlyStop" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["EarlyStop"] = self.early_stop
        if "LRAdjust" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["LRAdjust"] = self.lr_adjust
        if "TargetOpType" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["TargetOpType"] = self.target_op_type
        if "SelectiveUpdate" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["SelectiveUpdate"] = self.selective_update
        if "UpdateBias" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["UpdateBias"] = self.update_bias
        if "OutputQDQ" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["OutputQDQ"] = self.output_qdq
        if "DropRatio" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["DropRatio"] = self.drop_ratio
        if "MemOptLevel" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["MemOptLevel"] = self.mem_opt_level
        if "CacheDir" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["CacheDir"] = self.cache_dir
        if "LogPeriod" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["LogPeriod"] = self.log_period
        if "SelectMaxMemLayer" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["SelectMaxMemLayer"] = self.select_max_mem_layer
        if "NumWorkers" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["NumWorkers"] = self.num_workers
        if "PinMemory" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["PinMemory"] = self.pin_memory
        if "UseGDS" not in extra_options["FastFinetune"]:
            adaround_config["FastFinetune"]["UseGDS"] = self.use_gds
        return adaround_config


class AdaQuantConfig(AlgoConfig):
    """Configuration for the AdaQuant algorithm, which is originally proposed in the following paper:
    "Itay Hubara et al., Improving Post Training Neural Quantization: Layer-wise Calibration and Integer Programming,
    arXiv:2006.10518, 2020."

    AdaQuant (Adaptive Quantization) is a PTQ algorithm that adaptively adjusts
    quantization parameters based on calibration data. Rather than relying on
    fixed statistics, it performs lightweight optimization to minimize the
    difference between the original and quantized model activations, leading to
    better accuracy retention.

    The core idea is to minimize loss metrics such as L2 distance between
    original and quantized activation distributions. Like Adaround, AdaQuant
    doesn't require labeled data or full retraining, making it suitable for
    deployment-time optimization. Its adaptive nature makes it more robust than
    static quantization, especially when quantizing large or sensitive models.

    :param str optim_device: The device for optimization. Defaults to "cpu".
    :param str infer_device: The device for inference. Defaults to "cpu".
    :param int fixed_seed: A fixed seed for reproducibility. Defaults to 1705472343.
    :param int data_size: The total size of the dataset. Defaults to 1000000000.
    :param int batch_size: The batch size for optimization. Defaults to 1.
    :param int num_batches: The number of batches for optimization. Defaults to 1.
    :param int num_iterations: The number of optimization iterations. Defaults to 3000.
    :param float learning_rate: The learning rate for optimization. Defaults to 1e-5.
    :param bool early_stop: Whether to use early stopping. Defaults to False.
    :param int output_index: The index of the model's output to use for loss calculation. Defaults to 0.
    :param Optional[Tuple[float, float]] lr_adjust: Learning rate adjustment parameters. Defaults to None.
    :param List[str] target_op_type: List of operator types to be quantized. Defaults to ["Conv", "ConvTranspose", "Gemm", "MatMul", "InstanceNormalization", "LayerNormalization"].
    :param bool selective_update: Whether to selectively update weights. Defaults to False.
    :param bool update_bias: Whether to update the bias terms. Defaults to False.
    :param bool output_qdq: Whether to output QDQ format. Defaults to False.
    :param float drop_ratio: The ratio of weights to drop. Defaults to 1.0.
    :param int mem_opt_level: Memory optimization level. Defaults to 1.
    :param Optional[str] cache_dir: Directory for caching. Defaults to None.
    :param int log_period: Logging period. Defaults to 100.
    :param Optional[str] ref_model_path: Path to the reference model. Defaults to None.
    :param bool dynamic_batch: Whether to use dynamic batching. Defaults to False.
    :param bool parallel: Whether to use parallel processing. Defaults to False.
    :param float reg_param: The regularization parameter for the rounding loss.
                            This controls the trade-off between minimizing the reconstruction error and forcing the rounding parameters to be binary.
                            Defaults to 0.01.
    :param Tuple[float, float] beta_range: The range of the temperature parameter 'beta'.
                                           the 'beta' controls the sharpness of the soft rounding function.
                                           It is annealed from the first value to the second value over the course of optimization.
                                           A high 'beta' at the beginning allows for more exploration,
                                           while a low 'beta' at the end encourages convergence to a binary solution. Defaults to (20, 2).
    :param float warm_start: The fraction of total iterations for the "warm start" phase.
                             During this phase, only the reconstruction loss is used, and the regularization term is gradually introduced.
                             This helps to find a good initial state before forcing the rounding decisions to be binary. Defaults to 0.2.
    :param bool select_max_mem_layer: Whether to select the layer with largest estimated memory usage to run.
    :param int num_workers: Number of subprocesses used for data loading.
                            - 0 means the data will be loaded in the main process.
                            - >0 enables multi-process data loading, which can significantly speed up data pipeline when dataset and transforms are heavy.
                            Note: Using multiple workers increases CPU usage and may require careful handling of worker-safe code.
    :param bool pin_memory: If True, the DataLoader will copy tensors into CUDA pinned memory before returning them.
    :param bool use_gds: If True, with optim_device equals to 'cuda' and mem_opt_level equals to 2  on special GPU, the GDS (GPU Direct Storage) pipeline will be built to speedup the fine tuning process.
    """

    def __init__(
        self,
        optim_device: str = "cpu",
        infer_device: str = "cpu",
        fixed_seed: int = 1705472343,
        data_size: int = 1000000000,
        batch_size: int = 1,
        num_batches: int = 1,
        num_iterations: int = 3000,
        learning_rate: float = 1e-5,
        early_stop: bool = False,
        output_index: int = 0,
        lr_adjust: tuple[float, float] | None = None,
        target_op_type: list[str] = [
            "Conv",
            "ConvTranspose",
            "Gemm",
            "MatMul",
            "InstanceNormalization",
            "LayerNormalization",
        ],
        selective_update: bool = False,
        update_bias: bool = False,
        output_qdq: bool = False,
        drop_ratio: float = 1.0,
        mem_opt_level: int = 1,
        cache_dir: str | None = None,
        log_period: int = 100,
        ref_model_path: str | None = None,
        dynamic_batch: bool = False,
        parallel: bool = False,
        reg_param: float = 0.01,
        beta_range: tuple[float, float] = (20, 2),
        warm_start: float = 0.2,
        select_max_mem_layer: bool = False,
        num_workers: int = 1,
        pin_memory: bool = False,
        use_gds: bool = False,
    ) -> None:
        self.name: str = "adaquant"
        self.optim_device = optim_device
        self.infer_device = infer_device
        self.fixed_seed = fixed_seed
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.early_stop = early_stop
        self.output_index = output_index
        self.lr_adjust = lr_adjust
        self.target_op_type = target_op_type
        self.selective_update = selective_update
        self.update_bias = update_bias
        self.output_qdq = output_qdq
        self.drop_ratio = drop_ratio
        self.mem_opt_level = mem_opt_level
        self.cache_dir = cache_dir
        self.log_period = log_period
        self.ref_model_path = ref_model_path
        self.dynamic_batch = dynamic_batch
        self.parallel = parallel
        self.reg_param = reg_param
        self.beta_range = beta_range
        self.warm_start = warm_start
        self.select_max_mem_layer = select_max_mem_layer
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_gds = use_gds

    def _get_config(self, extra_options: dict[str, Any]) -> dict[str, Any]:
        adaquant_config: dict[str, Any] = dict()
        adaquant_config["FastFinetune"] = {}
        if "FastFinetune" not in extra_options:
            extra_options["FastFinetune"] = {}
        if "OptimAlgorithm" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["OptimAlgorithm"] = self.name
        if "OptimDevice" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["OptimDevice"] = self.optim_device
        if "InferDevice" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["InferDevice"] = self.infer_device
        if "FixedSeed" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["FixedSeed"] = self.fixed_seed
        if "DataSize" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["DataSize"] = self.data_size
        if "BatchSize" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["BatchSize"] = self.batch_size
        if "NumBatches" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["NumBatches"] = self.num_batches
        if "NumIterations" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["NumIterations"] = self.num_iterations
        if "LearningRate" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["LearningRate"] = self.learning_rate
        if "EarlyStop" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["EarlyStop"] = self.early_stop
        if "LRAdjust" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["LRAdjust"] = self.lr_adjust
        if "TargetOpType" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["TargetOpType"] = self.target_op_type
        if "SelectiveUpdate" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["SelectiveUpdate"] = self.selective_update
        if "UpdateBias" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["UpdateBias"] = self.update_bias
        if "OutputQDQ" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["OutputQDQ"] = self.output_qdq
        if "DropRatio" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["DropRatio"] = self.drop_ratio
        if "MemOptLevel" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["MemOptLevel"] = self.mem_opt_level
        if "CacheDir" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["CacheDir"] = self.cache_dir
        if "LogPeriod" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["LogPeriod"] = self.log_period
        if "SelectMaxMemLayer" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["SelectMaxMemLayer"] = self.select_max_mem_layer
        if "NumWorkers" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["NumWorkers"] = self.num_workers
        if "PinMemory" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["PinMemory"] = self.pin_memory
        if "UseGDS" not in extra_options["FastFinetune"]:
            adaquant_config["FastFinetune"]["UseGDS"] = self.use_gds
        return adaquant_config


class QuarotConfig(AlgoConfig):
    """Configuration for the Quarot algorithm, which is originally proposed in the following paper:
    "Saleh Ashkboos et al., QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs,
    arXiv:2404.00456, 2024."

    Quarot is a PTQ algorithm that enhances model robustness and accuracy by applying
    a rotation to the weight matrices before quantization. Instead of quantizing
    weights directly in their original basis, Quarot learns an optimal rotation
    that aligns the weights with a more quantization-friendly direction. This process
    reduces the quantization error without requiring full retraining.

    The algorithm works by factorizing a rotation matrix (e.g., using SVD or low-rank
    approximations) and optimizing it on unlabeled calibration data. The rotated
    weights are quantized, and the inverse rotation is fused back cleverly so that
    the final computation remains efficient and accurate.

    By leveraging the structure of the weight distribution and introducing minimal
    additional overhead, Quarot significantly improves quantization performance—especially
    in low-bit regimes. It’s particularly effective for transformer-based
    models or MLPs, where preserving fine-grained relationships between weights is
    crucial for maintaining performance.

    :param int r_matrix_dim: The dimension of constructing rotation matrix. Defaults to 4096.
    :param bool use_random_had: If True, the rotation matrix will be generated by the random Hadamard scheme. Defaults to False.
    :param Optional[str] r_config_path: The path of rotation config file. This is necessary when using QuaRot. Defaults to None.
    """

    def __init__(
        self, r_matrix_dim: int = 4096, use_random_had: bool = False, r_config_path: str | None = None
    ) -> None:
        self.name: str = "quarot"
        self.r_matrix_dim = r_matrix_dim
        self.use_random_had = use_random_had
        self.r_config_path = r_config_path

    def _get_config(self, extra_options: dict[str, Any]) -> dict[str, Any]:
        quarot_config: dict[str, Any] = dict()
        if "RMatrixDim" not in extra_options:
            quarot_config["RMatrixDim"] = self.r_matrix_dim
        if "UseRandomHad" not in extra_options:
            quarot_config["UseRandomHad"] = self.use_random_had
        if "RConfigPath" not in extra_options:
            quarot_config["RConfigPath"] = self.r_config_path
        return quarot_config


def _algo_flag(algorithms: list[AlgoConfig], algo_config: type[AlgoConfig]) -> bool:
    return any(isinstance(algo, algo_config) for algo in algorithms)


def _resolove_algo_conflict(algorithms: list[AlgoConfig]) -> list[AlgoConfig]:
    new_algorithms = set()
    ada_count = 0
    for algo in algorithms:
        if isinstance(algo, AdaRoundConfig | AdaQuantConfig):
            ada_count += 1
        if ada_count >= 2:
            logger.warning(f"Only one of the AdaRound and AdaQuant can be selected. {algo.name} has been removed.")  # type: ignore
            ada_count -= 1
            continue
        new_algorithms.add(algo)
    return list(new_algorithms)


ALGO_NAME_TO_CLASS = {
    "smooth_quant": SmoothQuantConfig,
    "cle": CLEConfig,
    "bias_correction": BiasCorrectionConfig,
    "gptq": GPTQConfig,
    "auto_mixprecision": AutoMixprecisionConfig,
    "adaround": AdaRoundConfig,
    "adaquant": AdaQuantConfig,
    "quarot": QuarotConfig,
}


def from_dict(d: dict[str, Any]) -> AlgoConfig:
    """
    Convert a dictionary into an algorithm configuration object.

    The dictionary must contain a valid algorithm name that maps
    to a supported algorithm configuration class.

    :param dict[str, Any] d: Dictionary representation of an algorithm config.
    :return: Constructed AlgoConfig instance.
    :raises ValueError: If the algorithm name is missing or unknown.
    """
    if "name" not in d:
        raise ValueError("Algo config dict must contain field 'name'.")

    name = d["name"]

    if name not in ALGO_NAME_TO_CLASS:
        raise ValueError(f"Unknown algo name: {name}")

    Cls = ALGO_NAME_TO_CLASS[name]

    kwargs = {k: v for k, v in d.items() if k != "name"}

    return Cls(**kwargs)


def add_method_to_subclasses(base_cls: type[AlgoConfig], method_name: str, method: Callable[..., Any]) -> None:
    for cls in base_cls.__subclasses__():
        setattr(cls, method_name, method)
        add_method_to_subclasses(cls, method_name, method)


add_method_to_subclasses(AlgoConfig, "from_dict", from_dict)  # type: ignore[type-abstract]
