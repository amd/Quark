#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Parses and registers a custom LLM template JSON file, used by ``quark-cli torch-llm-ptq
# --template_file`` (QUARK-1075) so that models whose ``model_type`` is not covered by the
# built-in templates can be quantized through the standard CLI interface.
#
# Expected JSON format (only ``model_type`` is required):
#
# .. code-block:: json
#
#     {
#         "model_type": "my_custom_model",                    # required
#         "kv_layers_name": ["*k_proj", "*v_proj"],           # optional
#         "q_layer_name": "*q_proj",                          # optional, string or list of strings
#         "gate_up_layers_name": ["gate_proj", "up_proj"],    # optional
#         "exclude_layers_name": ["lm_head"],                 # optional, defaults to []
#         "algorithm_configs": {                              # optional
#             "awq": {"name": "awq", "model_decoder_layers": "model.layers"},
#             "gptq": {"name": "gptq", "block_size": 128, "damp_percent": 0.01}
#         }
#     }
#
# Each ``algorithm_configs`` entry follows the same dictionary format as
# ``quark.torch.quantization.load_quant_algo_config_from_file``: a JSON object with a ``name``
# field identifying the algorithm configuration (e.g. ``"awq"``, ``"gptq"``, ``"smooth"``,
# ``"rotation"``) plus the algorithm-specific fields.

# Gracefully handle package import, as user may not be aware of dependencies.
try:
    import json
    from pathlib import Path
    from typing import Any
except ImportError:  # pragma: no cover
    print(
        "AMD Quark CLI dependencies need to be installed with `pip install amd-quark[cli]` or `pip install -r requirements-cli.txt`."
    )
    exit(1)

# Gracefully handle imports, as user may not have Quark installed.
try:
    from quark.torch import LLMTemplate
    from quark.torch.quantization.config.algo_configs import get_supported_algorithm_types

    # Dict-level counterpart of the public ``load_quant_algo_config_from_file``;
    # used to parse each per-algorithm entry of the ``algorithm_configs`` JSON section.
    from quark.torch.quantization.config.config import AlgoConfig, _load_quant_algo_config_from_dict
except ImportError as import_error:  # pragma: no cover
    print(f"AMD Quark needs to be installed with e.g. `pip3 install amd-quark`: {import_error}.")
    exit(1)


# Top-level keys allowed in the template JSON file.
_REQUIRED_KEYS = {"model_type"}
_OPTIONAL_KEYS = {
    "kv_layers_name",
    "q_layer_name",
    "gate_up_layers_name",
    "exclude_layers_name",
    "algorithm_configs",
}
_ALLOWED_KEYS = _REQUIRED_KEYS | _OPTIONAL_KEYS

# Algorithm names used in configuration files that differ from the names used
# to select an algorithm from LLMTemplate.
_ALGORITHM_NAME_ALIASES = {
    "smooth": "smoothquant",
    "quarot": "rotation",
}


def _validate_str_list(value: Any, key: str) -> list[str]:
    """Raise ValueError if ``value`` is not a list of strings, otherwise return it."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Template key '{key}' must be a list of strings, got: {value!r}")
    return value


def _parse_algorithm_configs(algorithm_configs_data: dict[str, Any]) -> dict[str, AlgoConfig]:
    """
    Parse the ``algorithm_configs`` section of the template JSON into AlgoConfig objects.

    :param dict algorithm_configs_data: Mapping of algorithm name to algorithm configuration dictionary.
    :return: Mapping of lower-cased algorithm name to the parsed AlgoConfig.
    :raises ValueError: If an algorithm name is unsupported, an entry is not a JSON object
        with a ``name`` field, or an entry contains invalid fields.
    """
    parsed_algorithm_configs: dict[str, AlgoConfig] = {}
    supported_algorithm_names = get_supported_algorithm_types()
    for algorithm_name, algorithm_config_data in algorithm_configs_data.items():
        # JSON object keys are always strings; normalize case to match the supported set.
        normalized_algorithm_name = algorithm_name.lower()
        if normalized_algorithm_name not in supported_algorithm_names:
            raise ValueError(
                f"Unsupported algorithm '{algorithm_name}' in 'algorithm_configs'. "
                f"Supported algorithms: {supported_algorithm_names}."
            )
        if not isinstance(algorithm_config_data, dict):
            raise ValueError(
                f"Each entry in 'algorithm_configs' must be a JSON object, "
                f"got for '{algorithm_name}': {algorithm_config_data!r}"
            )
        if "name" not in algorithm_config_data:
            raise ValueError(
                f"Entry '{algorithm_name}' in 'algorithm_configs' is missing the required 'name' field. "
                "Each entry must follow the format of `quark.torch.quantization.load_quant_algo_config_from_file`, "
                'e.g. {"name": "awq", "model_decoder_layers": "model.layers"}.'
            )
        config_name = algorithm_config_data["name"]
        if not isinstance(config_name, str):
            raise ValueError(
                f"Entry '{algorithm_name}' in 'algorithm_configs' must have a string 'name' field, got: {config_name!r}"
            )
        normalized_config_name = _ALGORITHM_NAME_ALIASES.get(config_name.lower(), config_name.lower())
        if normalized_config_name != normalized_algorithm_name:
            raise ValueError(
                f"Algorithm key '{algorithm_name}' does not match its configuration name '{config_name}'. "
                f"Use '{normalized_algorithm_name}' for both values."
            )
        try:
            parsed_algorithm_configs[normalized_algorithm_name] = _load_quant_algo_config_from_dict(
                dict(algorithm_config_data)
            )
        except (TypeError, ValueError) as parse_error:
            raise ValueError(
                f"Invalid configuration for algorithm '{algorithm_name}' in 'algorithm_configs': {parse_error}"
            ) from parse_error
    return parsed_algorithm_configs


def create_template_from_json_dict(data: dict[str, Any]) -> LLMTemplate:
    """
    Validate a template configuration dictionary and create (but not register) an LLMTemplate.

    :param dict data: Template configuration as parsed from a JSON file.
    :return: The created LLMTemplate.
    :raises ValueError: If the configuration is missing required keys, contains unknown keys,
        or has values of an unexpected type.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Template configuration must be a JSON object, got: {type(data).__name__}.")

    unknown_keys = sorted(set(data) - _ALLOWED_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Unknown key(s) in template configuration: {unknown_keys}. Allowed keys: {sorted(_ALLOWED_KEYS)}."
        )
    missing_keys = sorted(_REQUIRED_KEYS - set(data))
    if missing_keys:
        raise ValueError(f"Missing required key(s) in template configuration: {missing_keys}.")

    model_type = data["model_type"]
    if not isinstance(model_type, str) or not model_type:
        raise ValueError(f"Template key 'model_type' must be a non-empty string, got: {model_type!r}")

    kwargs: dict[str, Any] = {"model_type": model_type}
    if "kv_layers_name" in data:
        kwargs["kv_layers_name"] = _validate_str_list(data["kv_layers_name"], "kv_layers_name")
    if "gate_up_layers_name" in data:
        kwargs["gate_up_layers_name"] = _validate_str_list(data["gate_up_layers_name"], "gate_up_layers_name")
    if "exclude_layers_name" in data:
        kwargs["exclude_layers_name"] = _validate_str_list(data["exclude_layers_name"], "exclude_layers_name")
    if "q_layer_name" in data:
        q_layer_name = data["q_layer_name"]
        is_str_list = isinstance(q_layer_name, list) and all(isinstance(item, str) for item in q_layer_name)
        if not isinstance(q_layer_name, str) and not is_str_list:
            raise ValueError(
                f"Template key 'q_layer_name' must be a string or a list of strings, got: {q_layer_name!r}"
            )
        kwargs["q_layer_name"] = q_layer_name
    if "algorithm_configs" in data:
        algorithm_configs_data = data["algorithm_configs"]
        if not isinstance(algorithm_configs_data, dict):
            raise ValueError(
                f"Template key 'algorithm_configs' must be a JSON object mapping algorithm names to "
                f"configuration objects, got: {type(algorithm_configs_data).__name__}."
            )
        kwargs["algorithm_configs"] = _parse_algorithm_configs(algorithm_configs_data)

    return LLMTemplate(**kwargs)


def parse_template_from_json_file(path: str | Path) -> LLMTemplate:
    """
    Load and validate a template definition from a JSON file, without registering it with
    the LLMTemplate class.

    :param path: Path to the JSON file defining the template.
    :return: The created (but not registered) LLMTemplate.
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the file is not valid JSON or the configuration is invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Template file not found: {path}")

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as json_decode_error:
        raise ValueError(f"Template file {path} is not valid JSON: {json_decode_error}") from json_decode_error

    return create_template_from_json_dict(data)


def load_template_from_json_file(path: str | Path) -> LLMTemplate:
    """
    Load a template definition from a JSON file, create the LLMTemplate, and register it
    with the LLMTemplate class, making it available for subsequent quantization workflows.

    :param path: Path to the JSON file defining the template.
    :return: The registered LLMTemplate.
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the file is not valid JSON or the configuration is invalid.
    """
    template = parse_template_from_json_file(path)
    LLMTemplate.register_template(template)
    return template
