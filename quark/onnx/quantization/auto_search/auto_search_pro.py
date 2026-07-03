#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import copy
import difflib
import json
import logging
import multiprocessing
import os
import shutil
from typing import Any

import numpy as np

# avoid circular import issue
from quark.onnx.calibration import CachedDataReader
from quark.onnx.quantization.api import ModelQuantizer as ONNXModelQuantizer
from quark.onnx.quantization.config import QConfig
from quark.onnx.quantization.config.maps import QCONFIG_ALL_PARAMS
from quark.onnx.quantization.config.spec import QLayerConfig
from quark.onnx.quantization.output_eval import (
    calculate_cos,
    calculate_l1_distance,
    calculate_l2_distance,
    calculate_psnr,
    calculate_ssim,
)

from .config_generator import generate_all_configs
from .qconfig_mapping import (
    ADVANCED_SEARCH_CONFIG as DEFAULT_CONFIG,
)
from .qconfig_mapping import (
    DATA_TYPE,
    ONNX_ACTIVATION_WEIGHT,
    ONNX_ALGORITHM,
    ONNX_CALIBMETHOD,
    QUANT_GRANULARITY,
    SCALE_TYPE,
    get_sampler_dict,
)
from .utils import buildin_eval_func, validate_search_space

METRICS = {
    "L2": calculate_l2_distance,
    "L1": calculate_l1_distance,
    "cos": calculate_cos,
    "psnr": calculate_psnr,
    "ssim": calculate_ssim,
}


def validate_keys(
    reference_keys: list[str], input_keys: list[str], logger_ins: logging.Logger, cutoff: float = 0.6
) -> dict[str, str]:
    matched_res = {}
    for key in input_keys:
        if key not in reference_keys:
            suggestions = difflib.get_close_matches(key, reference_keys, n=1, cutoff=cutoff)
            matched_res[key] = suggestions[0]
            if suggestions:
                logger_ins.warning(f"Warning: '{key}' is not a valid key. Did you mean '{suggestions[0]}'?")
                logger_ins.info(f"{key} will be replaced with {suggestions[0]}")
            else:
                logger_ins.warning(f"Warning: '{key}' is not a valid key and no similar key was found.")
    return matched_res


def replace_keys(input_dict: dict[str, Any], repalced_dict: dict[str, str]) -> dict[str, Any]:
    for replace_key in repalced_dict:
        replace_val = input_dict[replace_key]
        input_dict[repalced_dict[replace_key]] = replace_val
        del input_dict[replace_key]
    return input_dict


class AutoSearchPro:
    def __init__(self, config: dict[str, Any]):
        """
        Parse auto search config
        """
        self.config = config

        self.output_dir = self.config.get("output_dir", DEFAULT_CONFIG["output_dir"])
        output_dir_exist = False
        if os.path.exists(self.output_dir):
            output_dir_exist = True
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = self._prepare_logging()
        matched_config_keys = validate_keys(list(DEFAULT_CONFIG.keys()), list(self.config.keys()), self.logger)
        if matched_config_keys != {}:
            self.config = replace_keys(self.config, matched_config_keys)

        if output_dir_exist:
            self.logger.warning(f"output_dir {self.output_dir} has existed! It's better to reset a empty one!")
        self.temp_dir = self.config.get("temp_dir", DEFAULT_CONFIG["temp_dir"])
        if os.path.exists(self.temp_dir):
            self.logger.warning(f"temp_dir {self.temp_dir} has existed! It will be removed!")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.search_space = self.config.get("search_space", DEFAULT_CONFIG["search_space"])
        if "search_space" in self.config:
            QCONFIG_ALL_PARAMS_SEARCH = list(QCONFIG_ALL_PARAMS) + [
                "weight",
                "weight_params",
                "activation",
                "activation_params",
                "algorithms",
                "cle_algo",
                "cle_params",
                "adaround_params",
                "adaquant_params",
            ]
            matched_search_space_keys = validate_keys(
                QCONFIG_ALL_PARAMS_SEARCH, list(self.search_space.keys()), self.logger
            )
            if matched_search_space_keys != {}:
                self.search_space = replace_keys(self.search_space, matched_search_space_keys)
        self.sampler_algo = self.config.get("search_algo", DEFAULT_CONFIG["search_algo"])
        self.evaluator = self.config.get("search_evaluator", None)
        self.metric = self.config.get("search_metric", None)
        if self.metric is not None:
            matched_metric = validate_keys(list(METRICS.keys()), [self.metric], self.logger)
            self.metric = METRICS[list(matched_metric.values())[0]] if matched_metric != {} else METRICS[self.metric]
        else:
            self.metric = METRICS["L2"]

        self.direction = self.config.get("direction", DEFAULT_CONFIG["direction"])
        self.n_trials = self.config.get("n_trials", DEFAULT_CONFIG["n_trials"])
        self.n_jobs = self.config.get("n_jobs", DEFAULT_CONFIG["n_jobs"])

        self.base_framework = self.config.get("base_framework", DEFAULT_CONFIG["base_framework"])
        self.model_input = self.config.get("model_input", None)
        if self.model_input is None:
            self.logger.error("Please set the model input!")
            raise ValueError("Please input the onnx model! 'model_input' can not be None.")

        if "calib_data_reader" not in self.config:
            raise ValueError("calib_data_reader in Auto Search Pro can not None!")
        self.calib_data_reader = CachedDataReader(self.config.get("calib_data_reader", None))
        eval_data_reader = self.config.get("eval_data_reader", None)
        if eval_data_reader is None:
            self.eval_data_reader = self.calib_data_reader
            self.logger.warning("Set eval_data_reader to be calib_data_reader!")
        else:
            self.eval_data_reader = CachedDataReader(eval_data_reader)
        self.study_storage_db = self.config.get("study_storage_db", None)
        self.load_study_if_exists = self.config.get("load_study_if_exists", False)
        if isinstance(self.study_storage_db, str):
            if not self.study_storage_db.endswith(".db"):
                self.logger.warning("study storage path should be endswith '.db'!")
        else:
            self.logger.warning("Please set the storage path a string and endswith '.db'!")

        self.two_stage_search = self.config.get("two_stage_search", False)

        validate_search_space_res = validate_search_space(self.search_space)
        discrete_space_size = validate_search_space_res["discrete_space_size"]
        contains_continuous = validate_search_space_res["contains_continuous"]
        if self.n_trials > discrete_space_size and (not contains_continuous):
            self.logger.warning(
                f"Your search space does not contain conituous trail and setting n_trials {self.n_trials} larger than discrete_space_size! n_trials set to be discrete_space_size {discrete_space_size}!"
            )
            self.n_trials = discrete_space_size

        self.study_name = self.config.get("study_name", "AutoSearch")
        self.plot_results = self.config.get("plot_results", False)

    def _prepare_logging(self, logging_name: str = "auto_search.log") -> logging.Logger:
        logger = logging.getLogger(logging_name)
        log_path = os.path.join(self.output_dir, logging_name)
        if not logger.handlers:
            logger.setLevel(level=logging.INFO)
            handler = logging.FileHandler(log_path, encoding="UTF-8")
            handler.setLevel(logging.INFO)
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            logger.addHandler(console)
        return logger

    def _get_sampler(self) -> Any:
        SAMPLER_DICT = get_sampler_dict(self.logger)
        if self.sampler_algo in list(SAMPLER_DICT.keys()):
            return SAMPLER_DICT[self.sampler_algo]
        elif self.sampler_algo in ["PartialFixed"]:
            raise NotImplementedError(f"{self.sampler_algo} support not implemented.")
        else:
            raise ValueError(f"Unsupported search_algo: {self.sampler_algo}")

    @staticmethod
    def process_joint_numiter_lr(input_dict: dict[str, Any]) -> dict[str, Any]:
        numiter_lr_name = "NumIterLR"
        if numiter_lr_name in list(input_dict.keys()):
            item = input_dict[numiter_lr_name]
            assert len(item) == 2
            input_dict["num_iterations"] = item[0]
            input_dict["learning_rate"] = item[1]
            del input_dict[numiter_lr_name]
        return input_dict

    def _onnx_quantize_model(self, params: dict[str, Any], model_output: str) -> None:
        # Parse params
        # Parse activation
        activation = params.pop("activation", None)
        activation_params = params.pop("activation_params", {})
        if "calibration_method" in activation_params:
            activation_params["calibration_method"] = ONNX_CALIBMETHOD[activation_params["calibration_method"]]
        if "scale_type" in activation_params:
            activation_params["scale_type"] = SCALE_TYPE[activation_params["scale_type"]]
        if "quant_granularity" in activation_params:
            activation_params["quant_granularity"] = QUANT_GRANULARITY[activation_params["quant_granularity"]]
        if "data_type" in activation_params:
            activation_params["data_type"] = DATA_TYPE[activation_params["data_type"]]
        activation_spec = ONNX_ACTIVATION_WEIGHT[activation](**activation_params)

        # Parse weight
        weight = params.pop("weight", None)
        weight_params = params.pop("weight_params", {})
        if "calibration_method" in weight_params:
            weight_params["calibration_method"] = ONNX_CALIBMETHOD[weight_params["calibration_method"]]
        if "scale_type" in weight_params:
            weight_params["scale_type"] = SCALE_TYPE[weight_params["scale_type"]]
        if "quant_granularity" in weight_params:
            weight_params["quant_granularity"] = QUANT_GRANULARITY[weight_params["quant_granularity"]]
        if "data_type" in weight_params:
            weight_params["data_type"] = DATA_TYPE[weight_params["data_type"]]
        weight_spec = ONNX_ACTIVATION_WEIGHT[weight](**weight_params)

        # Parese algorithms
        algo_conf = []
        algo = params.pop("algorithms", None)
        algo_params = {}
        if algo == "adaround":
            algo_params = params.pop("adaround_params", {})
        if algo == "adaquant":
            algo_params = params.pop("adaquant_params", {})
        if algo is not None:
            if algo not in ONNX_ALGORITHM:
                self.logger.warning(f"Input algorithm {algo} not supported!")
            else:
                algo_params = self.process_joint_numiter_lr(algo_params)
                algo_ins = ONNX_ALGORITHM[algo](**algo_params)
                algo_conf.append(algo_ins)

        # Parse cle
        cle_algo = params.pop("cle_algo", None)
        if cle_algo == "cle":
            cle_params = params.pop("cle_params", {})
            cle_ins = ONNX_ALGORITHM[cle_algo](**cle_params)
            algo_conf.append(cle_ins)

        # Parse other params
        specific_layer_config = params.pop("specific_layer_config", None)
        layer_type_config = params.pop("layer_type_config", None)
        exclude = (params.pop("exclude", None),)
        use_external_data_format = params.pop("use_external_data_format", False)

        # Build QConfig
        onnx_qconfig = QConfig(
            QLayerConfig(activation=activation_spec, weight=weight_spec),
            specific_layer_config=specific_layer_config,
            layer_type_config=layer_type_config,
            exclude=exclude,
            use_external_data_format=use_external_data_format,
            algo_config=algo_conf,
            **params,
        )

        # Build ONNXModelQuantizer
        onnx_quantizer = ONNXModelQuantizer(onnx_qconfig)

        # Quantize the model
        self.calib_data_reader.reset_iter()
        onnx_quantizer.quantize_model(
            model_input=self.model_input, model_output=model_output, calibration_data_reader=self.calib_data_reader
        )

    def _cal_diff(self, temp_output_path: str) -> float:
        # Evalute the quantized model
        if self.evaluator is None:
            prefix = "model_output"
            quantized_model_res_path = os.path.join(self.temp_dir, "quantized_" + prefix)
            if os.path.exists(quantized_model_res_path):
                self.logger.warning(f"{quantized_model_res_path} already exists and will be emptied...")
                shutil.rmtree(quantized_model_res_path)
            os.makedirs(quantized_model_res_path)
            self.eval_data_reader.reset_iter()
            try:
                self.quantized_model_output_path = buildin_eval_func(
                    temp_output_path, self.eval_data_reader, save_path=quantized_model_res_path, save_prefix=prefix
                )
            except Exception as e:
                self.logger.error(f"Quantized model inference error {e}!")
        else:
            self.quantized_model_output_metric = self.evaluator(temp_output_path)

        # calculate the metric difference
        diff: float = 99999.0
        if self.evaluator is None:
            diffs = []
            fp_output_files = os.listdir(self.base_model_output_path)
            for file in fp_output_files:
                fp_file = os.path.join(self.base_model_output_path, file)
                quantized_file = os.path.join(self.quantized_model_output_path, file)
                fp_output = np.load(fp_file)
                quantized_output = np.load(quantized_file)
                diff_item = self.metric(fp_output, quantized_output)
                diffs.append(diff_item)
            # remove the temporay quantized output
            shutil.rmtree(self.quantized_model_output_path)
            diff = np.mean(diffs)
            metric_name = self.config.get("search_metric", "L2")
        else:
            diff = self.base_model_output_metric - self.quantized_model_output_metric
            metric_name = "customer definded"

        self.logger.info(f"{metric_name} distance is:{diff}")
        return diff

    def grid_search(self, grid_search_space: dict[str, Any], search_len: int | None = None) -> dict[str, Any]:
        grid_configs = generate_all_configs(grid_search_space)
        if search_len is None:
            search_len = len(grid_configs)
        searched_scores = []
        for idx in range(search_len):
            temp_model_output = os.path.join(self.output_dir, f"quantized_model{idx}.onnx")
            temp_params = copy.deepcopy(grid_configs[idx])
            self._onnx_quantize_model(temp_params, temp_model_output)
            temp_score = self._cal_diff(temp_model_output)
            searched_scores.append(temp_score)
            self.logger.info(f"Grid search index:{idx}")
            self.logger.info(f"Grid search params:{grid_configs[idx]}")
            self.logger.info(f"Grid search metric:{temp_score:.6f}")

        sorted_idx = np.argsort(searched_scores)
        self.logger.info("-----------------------------------------------------")
        self.logger.info("---------------Grid search sorted result-------------")
        self.logger.info("-----------------------------------------------------")
        for idx in sorted_idx:
            self.logger.info(f"grid search index:{idx}, metric:{searched_scores[idx]:.6f}")
        self.logger.info(f"The best config is:{grid_configs[sorted_idx[0]]}")
        return grid_configs[sorted_idx[0]]

    def _evaluate_float_model(
        self,
    ) -> str:
        prefix = "model_output"
        fp_model_res_path = os.path.join(self.temp_dir, "fp_" + prefix)
        if self.evaluator is None:
            if os.path.exists(fp_model_res_path):
                self.logger.warning(f"{fp_model_res_path} already exists and will be emptied...")
                shutil.rmtree(fp_model_res_path)
            os.makedirs(fp_model_res_path)
            try:
                self.base_model_output_path = buildin_eval_func(
                    self.model_input, self.eval_data_reader, save_path=fp_model_res_path, save_prefix=prefix
                )
                self.eval_data_reader.reset_iter()
            except Exception as e:
                self.logger.error(f"Float model inference error {e}!")
        else:
            self.base_model_output_metric = self.evaluator(self.model_input)
            self.logger.info(f"float model baseline metric:{self.base_model_output_metric}")
        return fp_model_res_path

    def _two_stage_search(
        self,
    ) -> dict[str, Any]:
        fastft_params = {}
        algorithms = self.search_space.pop("algorithms", None)
        adaround_params = self.search_space.pop("adaround_params", None)
        adaquant_params = self.search_space.pop("adaquant_params", None)
        if algorithms is not None:
            fastft_params["algorithms"] = algorithms
        else:
            self.logger.warning("FastFinetune search space is empty!")
        if adaround_params is not None:
            fastft_params["adaround_params"] = adaround_params
        if adaquant_params is not None:
            fastft_params["adaquant_params"] = adaquant_params
        # Grid search for caliration
        calib_best_config = self.grid_search(self.search_space)
        self.logger.info(f"Best calib config in two_stage_search is:{calib_best_config}")

        if algorithms is None:
            return calib_best_config

        for key, value in calib_best_config.items():
            if isinstance(value, str):
                fastft_params[key] = [value]
            elif isinstance(value, dict):
                fastft_params[key] = {searched_key: [searched_value] for searched_key, searched_value in value.items()}

        if "activation_params" in fastft_params:
            fastft_params["activation_params"]["only_if"] = "activation"
        if "weight_params" in fastft_params:
            fastft_params["weight_params"]["only_if"] = "weight"
        if "cle_params" in fastft_params:
            fastft_params["cle_params"]["only_if"] = {"cle_algo": "cle"}

        return fastft_params

    def run(self) -> dict[str, Any]:
        try:
            import optuna  # type: ignore
        except ImportError:
            self.logger.error("optuna is not detected. Please install optuna and try again.")

        #  evaluate float onnx model
        fp_model_res_path = self._evaluate_float_model()

        # two_stage_search
        if self.two_stage_search:
            algorithms = self.search_space.get("algorithms", None)
            two_stage_params = self._two_stage_search()
            if algorithms is not None:
                # Fastfinetune
                self.search_space = two_stage_params
            else:
                return two_stage_params

        # support for Grid search
        if self.sampler_algo == "Grid":
            all_configs = generate_all_configs(self.search_space)
            search_len = self.n_trials if self.n_trials < len(all_configs) else len(all_configs)
            best_config = self.grid_search(self.search_space, search_len)
            return best_config

        def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
            params: dict[str, Any] = {}
            for key, value in self.search_space.items():
                if isinstance(value, dict) and "only_if" in value:
                    cond = value["only_if"]
                    if isinstance(cond, str):
                        if cond not in params:
                            continue
                    elif isinstance(cond, dict):
                        k, v = list(cond.items())[0]
                        if params.get(k) != v:
                            continue

                    sub_params: dict[str, Any] = {}
                    for subkey, subval in value.items():
                        # in order to build different params with same name under different scope, add prefix for subkey
                        subkey_save = key + "_" + subkey
                        if subkey == "only_if":
                            continue
                        if isinstance(subval, list):
                            sub_params[subkey] = trial.suggest_categorical(subkey_save, subval)
                        elif isinstance(subval, dict):
                            if subval["type"] == "int":
                                sub_params[subkey] = trial.suggest_int(
                                    subkey_save, subval["low"], subval["high"], step=subval.get("step", 1)
                                )
                            elif subval["type"] == "float":
                                sub_params[subkey] = trial.suggest_float(
                                    subkey_save, subval["low"], subval["high"], log=subval.get("log", False)
                                )

                    if sub_params != {}:
                        params[key] = copy.deepcopy(sub_params)
                else:
                    if isinstance(value, list):
                        params[key] = trial.suggest_categorical(key, value)

            return params

        def default_objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial)
            self.logger.info(f"[Trial {trial.number}] Params: {params}")
            quant_prefix = "quantized_model_"
            if self.base_framework == "onnx":
                temp_model_output = os.path.join(self.output_dir, f"{quant_prefix}{str(trial.number)}.onnx")
                self._onnx_quantize_model(params, temp_model_output)
            score = self._cal_diff(temp_output_path=temp_model_output)
            return score

        # create_study
        storage = None if self.study_storage_db is None else f"sqlite:///{self.output_dir}/{self.study_storage_db}"
        study = optuna.create_study(
            direction=self.direction,
            study_name=self.study_name,
            storage=storage,
            load_if_exists=self.load_study_if_exists,
            sampler=self._get_sampler(),
        )

        # start search
        if self.config["n_jobs"] > 1:
            with multiprocessing.Pool(self.config["n_jobs"]) as pool:
                pool.map(lambda _: study.optimize(default_objective, n_trials=1), range(self.n_trials))
        else:
            study.optimize(default_objective, n_trials=self.n_trials)

        if self.evaluator is None:
            shutil.rmtree(fp_model_res_path)

        # save result
        best = study.best_trial
        self.logger.info(f"Best Trial {best.number}: Value={best.value}, Params={json.dumps(best.params)}")

        output_dir = self.config["output_dir"]
        with open(os.path.join(output_dir, "best_params.json"), "w") as f:
            json.dump(best.params, f, indent=2)

        if self.plot_results:
            try:
                from optuna import visualization

                visualization.plot_optimization_history(study).write_html(os.path.join(output_dir, "opt_history.html"))
                visualization.plot_param_importances(study).write_html(
                    os.path.join(output_dir, "param_importance.html")
                )
            except ValueError as e:
                self.logger.warning(f"Please set more trials! {e}")

        return study.best_params
