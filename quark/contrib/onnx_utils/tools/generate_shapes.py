# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import csv
import json

CSV_FILE = ""


def format_shape(shape: str) -> list[str]:
    if shape:
        return shape.split("x")
    return []


with open(CSV_FILE) as csvfile:
    fieldnames = (
        "name_x",
        "op_type",
        "in_shape",
        "wgt_shape",
        "out_shape",
        "mult",
        "max_elf",
        "name",
        "layer_cycles",
        "layer_exec_time",
    )
    reader = csv.DictReader(csvfile, fieldnames)
next(reader, None)  # skip header
new_json = {}
for row in reader:
    assert row["name_x"] == row["name"]
    new_json[row["name"]] = {
        "op_type": row["op_type"],
        "in": format_shape(row["in_shape"]),
        "const": format_shape(row["wgt_shape"]),
        "out": format_shape(row["out_shape"]),
    }

with open("shapes.json", "w") as f:
    json.dump(new_json, f)
