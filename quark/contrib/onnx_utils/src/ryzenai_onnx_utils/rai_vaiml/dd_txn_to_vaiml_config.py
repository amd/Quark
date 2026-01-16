# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import json
import os
import re
from typing import Any, cast

import yaml

VAIML_TEMPLATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vaiml_templates.yaml")


class VAIMLTemplateConfig:
    def __init__(self, model_type: str = "llama2-unified", plugin_name: str | None = None) -> None:
        self.valml_templates_path = VAIML_TEMPLATES_PATH
        self.txn_patterns_config = self._load_yaml_config(self.valml_templates_path)

        # Set up class attributes for easier access
        assert self.txn_patterns_config is not None
        self.transaction_patterns = self.txn_patterns_config[model_type]["TRANSACTION_PATTERNS"]
        self.json_template = self.txn_patterns_config[model_type]["JSON_TEMPLATE"]

        if plugin_name:
            self.json_template["plugin_name"] = plugin_name

    def _load_yaml_config(self, yaml_file_path: str) -> Any:
        try:
            with open(yaml_file_path) as file:
                data = yaml.safe_load(file)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {yaml_file_path}") from None
        except Exception as e:
            raise RuntimeError(f"Error loading YAML file {yaml_file_path}: {str(e)}") from None


def extract_missing_transactions(text: str) -> list[Any]:
    pattern = r"(?:Transaction not found|\s+Invalid transaction binary string):\s+(\S+)"
    return re.findall(pattern, text)


def read_error_file(file_path: str, vaiml_config: VAIMLTemplateConfig) -> list[dict[str, Any]]:
    try:
        with open(file_path, encoding="utf-8") as file:
            content = file.read()

        # Extract and parse transactions
        transaction_ids = extract_missing_transactions(content)
        unique_ids = set(transaction_ids)

        results = [parse_transaction(trans_id, vaiml_config) for trans_id in unique_ids]

        # Sort by operation type then transaction ID
        return sorted(results, key=lambda x: (x["operation"], x["transaction_id"]))

    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found.") from None
    except Exception as e:
        raise RuntimeError(f"Error processing file: {e}") from None


# Returns a dictionary with operation type and parameters as below
# """{
#     "transaction_id": transaction_id,
#     "operation": operation,
#     "template": template,
#     "param1": param1,
#     "param2": param2,
#     ...
# }"""
def parse_transaction(transaction_id: str, vaiml_config: VAIMLTemplateConfig) -> dict[str, Any]:
    result: dict[str, Any] = {
        "transaction_id": transaction_id,
        "operation": "unknown",
        "template": "unknown",
    }

    # Try each pattern until we find a match
    for pattern in vaiml_config.transaction_patterns:
        match = re.match(pattern["regex"], transaction_id)
        if match:
            result["operation"] = pattern["op_type"]
            result["template"] = pattern["naming_template"]

            # Extract parameters based on the pattern
            for i, param_name in enumerate(pattern["params"]):
                result[param_name] = int(match.group(i + 1))
            break
    return result


def print_summary(transactions: list[dict[str, Any]]) -> None:
    if not transactions:
        print("No transactions found.")
        return

    # Print summary header
    print(f"Found {len(transactions)} unique missing transactions\n")

    # Extract all parameter names
    all_params = set()
    for trans in transactions:
        all_params.update([k for k in trans if k not in ["transaction_id", "operation", "template"]])

    # Define column headers and widths
    columns: list[dict[str, str | int]] = [
        {"name": "Transaction ID", "width": 50},
        {"name": "Operation", "width": 15},
        {"name": "Template", "width": 50},
    ]

    # Add parameter columns
    for param in sorted(all_params):
        columns.append({"name": param, "width": 10})

    # Print header row
    header = " | ".join(f"{col['name']:<{col['width']}}" for col in columns)
    print(header)

    # Print separator
    separator = "-+-".join("-" * cast(int, col["width"]) for col in columns)
    print(separator)

    # Print each transaction
    for trans in transactions:
        row = [
            f"{trans['transaction_id']:<{columns[0]['width']}}",
            f"{trans['operation']:<{columns[1]['width']}}",
            f"{trans['template']:<{columns[2]['width']}}",
        ]

        # Add parameter values
        for i, param in enumerate(sorted(all_params)):
            value = trans.get(param, "N/A")
            row.append(f"{value if value != 'N/A' else 'N/A':<{columns[i + 3]['width']}}")

        print(" | ".join(row))


def generate_json_config(
    vaiml_config: VAIMLTemplateConfig,
    transactions: list[dict[str, Any]],
    output_path: str,
) -> None:
    op_params: dict[str, Any] = {}

    for trans in transactions:
        op_type = trans["operation"]

        if op_type == "unknown":
            continue

        if op_type not in op_params:
            op_params[op_type] = {}

        if op_type == "masked_softmax":
            op_params[op_type]["K"] = {-1}

        # Collect all parameters for this operation
        for param, value in trans.items():
            if param not in ["transaction_id", "operation", "template"]:
                if param not in op_params[op_type]:
                    op_params[op_type][param] = set()
                op_params[op_type][param].add(value)

    # Create the JSON structure
    json_config = vaiml_config.json_template.copy()

    # Add operation types with their parameters
    for op_type, params in op_params.items():
        op_config: dict[str, Any] = {"op_type": op_type}

        # Add parameters as sorted lists
        for param, values in params.items():
            op_config[param] = sorted(values)

        json_config["op_types"].append(op_config)

    # Write the configuration to a JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=4)

    print(f"\nJSON configuration written to {output_path}")


def convert(
    transactions: list[dict[str, Any]],
    json_file: str,
    vaiml_config: VAIMLTemplateConfig,
    verbose: bool = True,
) -> None:
    if verbose:
        print_summary(transactions)
    generate_json_config(vaiml_config, transactions, json_file)
