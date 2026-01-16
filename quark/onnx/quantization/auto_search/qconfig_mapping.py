#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
from typing import Any

from quark.onnx.quantization.config.algorithm import AdaQuantConfig, AdaRoundConfig, CLEConfig
from quark.onnx.quantization.config.data_type import BFP16, BFloat16, Int8, Int16, Int32, UInt8, UInt16, UInt32
from quark.onnx.quantization.config.spec import (
    BFloat16Spec,
    BFP16Spec,
    CalibMethod,
    Int8Spec,
    Int16Spec,
    QuantGranularity,
    ScaleType,
    UInt8Spec,
    UInt16Spec,
    XInt8Spec,
)

ONNX_ACTIVATION_WEIGHT = {
    "XInt8Spec": XInt8Spec,
    "Int8Spec": Int8Spec,
    "UInt8Spec": UInt8Spec,
    "Int16Spec": Int16Spec,
    "UInt16Spec": UInt16Spec,
    "BFloat16Spec": BFloat16Spec,
    "BFP16Spec": BFP16Spec,
}

SCALE_TYPE = {"Float32": ScaleType.Float32, "PowerOf2": ScaleType.PowerOf2, "Int16": ScaleType.Int16}

QUANT_GRANULARITY = {
    "Tensor": QuantGranularity.Tensor,
    "Channel": QuantGranularity.Channel,
    "Group": QuantGranularity.Group,
}

DATA_TYPE = {
    "Int8": Int8,
    "UInt8": UInt8,
    "Int16": Int16,
    "UInt16": UInt16,
    "BFP16": BFP16,
    "BFloat16": BFloat16,
    "Int32": Int32,
    "UInt32": UInt32,
}

ONNX_CALIBMETHOD = {
    "MinMSE": CalibMethod.MinMSE,
    "MinMax": CalibMethod.MinMax,
    "Percentile": CalibMethod.Percentile,
    "Entropy": CalibMethod.Entropy,
    "Distribution": CalibMethod.Distribution,
    "LayerwisePercentile": CalibMethod.LayerwisePercentile,
}

ONNX_ALGORITHM = {
    "cle": CLEConfig,
    "adaround": AdaRoundConfig,
    "adaquant": AdaQuantConfig,
}

# accurate but slow
ADVANCED_SEARCH_CONFIG = {
    "search_space": {
        "activation": ["Int8Spec"],
        "activation_params": {
            "symmetric": [True, False],
            "calibration_method": ["MinMax", "Percentile", "LayerwisePercentile"],
            "scale_type": ["Float32"],
            "quant_granularity": ["Tensor"],
            "only_if": "activation",
        },
        "weight": ["Int8Spec"],
        "weight_params": {
            "symmetric": [True],
            "calibration_method": ["MinMax", "MinMSE"],
            "only_if": "weight",
        },
        "cle_algo": ["cle", None],
        "cle_params": {"cle_steps": [-1, 1], "only_if": {"cle_algo": "cle"}},
        "algorithms": ["adaround", "adaquant"],
        "adaquant_params": {
            "num_iterations": {"type": "int", "low": 3000, "high": 30000, "step": 3000},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
            "optim_device": ["cuda:0"],
            "infer_device": ["cuda:0"],
            "data_size": [3000],
            "batch_size": [2],
            "early_stop": [True],
            "only_if": {"algorithms": "adaquant"},
        },
        "adaround_params": {
            "num_iterations": {"type": "int", "low": 3000, "high": 30000, "step": 3000},
            "learning_rate": {"type": "float", "low": 1e-2, "high": 1e-1, "log": True},
            "optim_device": ["cuda:0"],
            "infer_device": ["cuda:0"],
            "data_size": [3000],
            "batch_size": [2],
            "early_stop": [True],
            "only_if": {"algorithms": "adaround"},
        },
        "specific_layer_config": [None],
        "layer_type_config": [None],
        "exclude": [None],
        "use_external_data_format": [False],
        "OptimizeModel": [True],
    },
    "n_trials": 20,
    "n_jobs": 1,
    "output_dir": "./output",
    "temp_dir": "./temp_dir",
    "search_algo": "TPE",
    "search_evaluator": None,  # Custom or built-in function
    "search_metric": None,
    "direction": "minimize",
    "base_framework": "onnx",
    "study_storage_db": "auto_search.db",
    "load_study_if_exists": True,
    "study_name": "AutoSearch",
    "model_input": None,
    "calib_data_reader": None,
    "eval_data_reader": None,
    "two_stage_search": True,
    "plot_results": False,
}

XINT8_SEARCH_CONFIG = {
    "search_space": {
        "activation": ["XInt8Spec"],
        "activation_params": {
            "symmetric": [True],
            "calibration_method": ["MinMSE"],
            "only_if": "activation",
        },
        "weight": ["XInt8Spec"],
        "weight_params": {
            "symmetric": [True],
            "quant_granularity": ["Tensor"],
            "only_if": "weight",
        },
        "cle_algo": ["cle", None],
        "cle_params": {"cle_steps": [-1, 1], "only_if": {"cle_algo": "cle"}},
        "algorithms": ["adaround", "adaquant"],  # all default algorithms: adaround and adaquant
        "adaquant_params": {
            "num_iterations": {"type": "int", "low": 3000, "high": 20000, "step": 3000},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
            "optim_device": ["cuda:0"],
            "infer_device": ["cuda:0"],
            "data_size": [1000],
            "batch_size": [2],
            "early_stop": [True],
            "only_if": {"algorithms": "adaquant"},
        },
        "adaround_params": {
            "num_iterations": {"type": "int", "low": 3000, "high": 20000, "step": 3000},
            "learning_rate": {"type": "float", "low": 1e-2, "high": 1e-1, "log": True},
            "only_if": {"algorithms": "adaround"},
        },
        "specific_layer_config": [None],
        "layer_type_config": [None],
        "exclude": [None],
        "use_external_data_format": [False],
        "OptimizeModel": [True],
    },
    # 20 trials for FastFinetune search because of applying two_stage_search, there are 4 + 20 quantization config totally
    "n_trials": 20,
    "n_jobs": 1,
    "output_dir": "./output",
    "temp_dir": "./temp_dir",
    "search_algo": "TPE",  # TPE search
    "search_evaluator": None,  # Custom or built-in function
    "search_metric": None,
    "direction": "minimize",
    "base_framework": "onnx",
    "study_storage_db": "auto_search.db",
    "load_study_if_exists": True,
    "study_name": "AutoSearch",
    "model_input": None,
    "calib_data_reader": None,
    "eval_data_reader": None,
    "two_stage_search": True,  # apply two_stage_search
    "plot_results": False,
}

A8W8_SEARCH_CONFIG = {
    "search_space": {
        "activation": ["Int8Spec"],  # activation: ["Int8Spec"]
        "activation_params": {
            "symmetric": [True, False],  # activation symmetric: [True, False]
            "calibration_method": ["MinMax", "Percentile", "LayerwisePercentile"],  # three main calibration methods
            "only_if": "activation",
        },
        "weight": ["Int8Spec"],  # weight: ["Int8Spec"]
        "weight_params": {
            "symmetric": [True],
            "quant_granularity": ["Tensor"],  # weight granularity: ["Tensor"]
            "only_if": "weight",
        },
        "algorithms": ["adaround", "adaquant"],  # all default algorithms
        "adaquant_params": {
            "num_iterations": {"type": "int", "low": 3000, "high": 20000, "step": 3000},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
            "optim_device": ["cuda:0"],
            "infer_device": ["cuda:0"],
            "data_size": [1000],
            "batch_size": [2],
            "early_stop": [True],
            "only_if": {"algorithms": "adaquant"},
        },
        "adaround_params": {
            "num_iterations": {"type": "int", "low": 3000, "high": 20000, "step": 3000},
            "learning_rate": {"type": "float", "low": 1e-2, "high": 1e-1, "log": True},
            "only_if": {"algorithms": "adaround"},
        },
        "cle_algo": ["cle", None],
        "cle_params": {"cle_steps": [-1, 1], "only_if": {"cle_algo": "cle"}},
        "specific_layer_config": [None],
        "layer_type_config": [None],
        "exclude": [None],
        "use_external_data_format": [False],
        "OptimizeModel": [True],
        "AlignSlice": [False],
        "FoldRelu": [True],
        "AlignConcat": [True],
        "PercentileCandidates": [[99.99, 99.999, 99.9999], [99.0, 99.9, 99.99, 99.999, 99.9999]],
    },
    # 20 trials for FastFinetune search because of applying two_stage_search, there are 12 + 20 quantization config totally
    "n_trials": 20,
    "n_jobs": 1,
    "output_dir": "./output",
    "temp_dir": "./temp_dir",
    "search_algo": "TPE",  # grid search
    "search_evaluator": None,  # Custom or built-in function
    "search_metric": None,
    "direction": "minimize",
    "base_framework": "onnx",
    "study_storage_db": "auto_search.db",
    "load_study_if_exists": True,
    "study_name": "AutoSearch",
    "model_input": None,
    "calib_data_reader": None,
    "eval_data_reader": None,
    "two_stage_search": True,  # apply two_stage_search
    "plot_results": False,
}

A16W8_SEARCH_CONFIG = {
    "search_space": {
        "activation": ["Int16Spec"],  # activation: ["Int16Spec"]
        "activation_params": {
            "symmetric": [True, False],  # activation symmetric: [True, False]
            "calibration_method": ["MinMax", "Percentile"],  # "calibration_method": ["MinMax", "Percentile"]
            "only_if": "activation",
        },
        "weight": ["Int8Spec"],  # weight: ["Int8Spec"]
        "weight_params": {
            "symmetric": [True],
            "quant_granularity": ["Tensor"],  # weight granularity: ["Tensor"]
            "only_if": "weight",
        },
        "cle_algo": ["cle", None],
        "cle_params": {"cle_steps": [-1, 1], "only_if": {"cle_algo": "cle"}},
        "algorithms": ["adaround", "adaquant"],
        "adaquant_params": {
            "num_iterations": {"type": "int", "low": 3000, "high": 20000, "step": 3000},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
            "optim_device": ["cuda:0"],
            "infer_device": ["cuda:0"],
            "data_size": [1000],
            "batch_size": [2],
            "early_stop": [True],
            "only_if": {"algorithms": "adaquant"},
        },
        "adaround_params": {
            "num_iterations": {"type": "int", "low": 3000, "high": 20000, "step": 3000},
            "learning_rate": {"type": "float", "low": 1e-2, "high": 1e-1, "log": True},
            "only_if": {"algorithms": "adaround"},
        },
        "specific_layer_config": [None],
        "layer_type_config": [None],
        "exclude": [None],
        "use_external_data_format": [False],
        "OptimizeModel": [True],
        "AlignSlice": [False],
        "FoldRelu": [True],
        "AlignConcat": [True],
        "AlignEltwiseQuantType": [True],
    },
    "n_trials": 20,  # grid search will execute every config in caliration stage, 32+20 configs here totally
    "n_jobs": 1,
    "output_dir": "./output",
    "temp_dir": "./temp_dir",
    "search_algo": "TPE",  # TPE search
    "search_evaluator": None,  # Custom or built-in function
    "search_metric": None,
    "direction": "minimize",
    "base_framework": "onnx",
    "study_storage_db": "auto_search.db",
    "load_study_if_exists": True,
    "study_name": "AutoSearch",
    "model_input": None,
    "calib_data_reader": None,
    "eval_data_reader": None,
    "two_stage_search": True,
    "plot_results": False,
}

AUTO_SEARCH_CONFIGS = {
    "ADVANCED_SEARCH": ADVANCED_SEARCH_CONFIG,
    "XINT8_SEARCH": XINT8_SEARCH_CONFIG,
    "A8W8_SEARCH": A8W8_SEARCH_CONFIG,
    "A16W8_SEARCH": A16W8_SEARCH_CONFIG,
}


def get_sampler_dict(logger: logging.Logger) -> dict[str, Any]:
    try:
        import optuna  # type: ignore
    except ImportError:
        logger.error("optuna is not detected. Please install optuna and try again.")

    SAMPLER_DICT = {
        "TPE": optuna.samplers.TPESampler(),
        "Random": optuna.samplers.RandomSampler(),
        "CmaEs": optuna.samplers.CmaEsSampler(),
        "GPS": optuna.samplers.GPSampler(),
        "NSGAII": optuna.samplers.NSGAIISampler(),
        "QMC": optuna.samplers.QMCSampler(),
    }
    return SAMPLER_DICT


def get_auto_search_config(config_name: str) -> dict[str, Any]:
    return AUTO_SEARCH_CONFIGS.get(config_name, "Default")
