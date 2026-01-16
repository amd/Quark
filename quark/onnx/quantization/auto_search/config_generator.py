#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
from itertools import product
from typing import Any

Config = dict[str, Any]
SearchSpace = dict[str, list[Any] | dict[str, Any]]


def is_condition_met(config: Config, condition: str | dict[str, Any]) -> bool:
    """
    Check if the 'only_if' condition is satisfied given a config.
    """
    if isinstance(condition, str):
        return config.get(condition) is not None
    if isinstance(condition, dict):
        return all(config.get(k) == v for k, v in condition.items())
    return False


def validate_search_space(search_space: SearchSpace) -> None:
    """
    Validate that the search space follows the expected structure:
      - Base fields must have list values.
      - Conditional fields must be dicts containing 'only_if' and all parameter values must be lists.

    Raises:
        ValueError: If the search space contains any invalid format.
    """
    for key, value in search_space.items():
        if isinstance(value, list):
            continue  # Valid base field
        elif isinstance(value, dict):
            if "only_if" not in value:
                raise ValueError(f"Conditional field '{key}' is missing required 'only_if' key.")
            for param_key, param_val in value.items():
                if param_key == "only_if":
                    continue
                if not isinstance(param_val, list):
                    raise ValueError(f"In conditional field '{key}', parameter '{param_key}' must be a list.")
        else:
            raise ValueError(
                f"Field '{key}' has unsupported type: {type(value).__name__}. "
                "Only list (base) or dict with 'only_if' (conditional) are allowed."
            )


def split_search_space(search_space: SearchSpace) -> tuple[dict[str, list[Any]], dict[str, dict[str, Any]]]:
    """
    Split the search space into base and conditional fields.
    """
    base_fields = {}
    conditional_fields = {}

    for key, value in search_space.items():
        if isinstance(value, list):
            base_fields[key] = value
        else:
            conditional_fields[key] = value

    return base_fields, conditional_fields


def generate_param_combinations(param_dict: dict[str, Any], context_config: Config) -> list[dict[str, Any]]:
    """
    Generate all valid combinations of a conditional parameter group,
    if the 'only_if' condition is satisfied.
    """
    param_dict = copy.deepcopy(param_dict)
    only_if = param_dict.pop("only_if", None)

    if only_if and not is_condition_met(context_config, only_if):
        return [{}]

    keys = list(param_dict.keys())
    value_lists = [param_dict[k] for k in keys]
    combinations = [dict(zip(keys, values, strict=False)) for values in product(*value_lists)]

    return combinations if combinations else [{}]


def generate_all_configs(search_space: SearchSpace) -> list[Config]:
    """
    Generate all valid configurations based on the given discrete search space.
    """
    validate_search_space(search_space)  # Validate before processing

    all_configs: list[Config] = []
    base_fields, conditional_fields = split_search_space(search_space)

    base_keys = list(base_fields.keys())
    base_value_lists = [base_fields[k] for k in base_keys]

    for base_combo in product(*base_value_lists):
        base_config = dict(zip(base_keys, base_combo, strict=False))
        config_list = [base_config]

        for field_name, param_spec in conditional_fields.items():
            next_configs = []
            for cfg in config_list:
                param_combos = generate_param_combinations(param_spec, cfg)
                for combo in param_combos:
                    new_cfg = copy.deepcopy(cfg)
                    if combo:
                        new_cfg[field_name] = combo
                    next_configs.append(new_cfg)
            config_list = next_configs

        all_configs.extend(config_list)

    return all_configs
