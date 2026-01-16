# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
from typing import Any

from . import external_data_pb2 as external_data


def open_header(header_file_name: os.PathLike[Any] | str) -> external_data.Header:
    """Open and parse the header file containing external data information."""
    # Read header file
    with open(header_file_name, "rb") as f:
        header_bytes = f.read()

    header = external_data.Header()
    header.ParseFromString(header_bytes)

    return header


def save_header(header: external_data.Header, path: os.PathLike[Any] | str) -> None:
    header_bytes = header.SerializeToString()
    with open(path, "wb") as f:
        f.write(header_bytes)


def print_header(header: external_data.Header) -> None:
    """Print the content of the header proto in a readable format."""
    print("=" * 60)
    print("HEADER CONTENT")
    print("=" * 60)

    # Print external data info
    print("\nEXTERNAL DATA:")
    print(f"  Filename: {header.external_data.filename}")
    print(f"  NPU: {header.external_data.npu}")
    print(f"  GPU: {header.external_data.gpu}")
    print(f"  Embedding: {header.external_data.embedding}")

    # Print operators
    print(f"\nOPERATORS ({len(header.operators)} total):")
    for name, op in header.operators.items():
        print(f"  {name}:")
        print(f"    Op Type: {op.op_type}")
        print(f"    Data Tensors: {len(op.data)}")
        for i, tensor in enumerate(op.data):
            print(f"      Tensor {i}:")
            print(f"        Offset: {tensor.offset}")
            print(f"        Size: {tensor.size}")
            print(f"        Shape: {list(tensor.shape)}")
            print(f"        Data Type: {tensor.data_type}")

    # Print op metadata
    print(f"\nOP METADATA ({len(header.op_metadata)} total):")
    for op_type, metadata in header.op_metadata.items():
        print(f"  {op_type}:")
        print(f"    Max NPU Buffer Size: {metadata.max_npu_buffer_size}")
        print(f"    First Node: {metadata.first}")
        print(f"    Last Node: {metadata.last}")

    # Print layers
    print(f"\nLAYERS ({len(header.layers)} total):")
    for i, layer in enumerate(header.layers):
        print(f"  Layer {i}:")
        print(f"    Offset: {layer.offset}")
        print(f"    Size: {layer.size}")
        print(f"    Operators: {list(layer.operators)}")


def rename_external_data_file(
    header_path: os.PathLike[Any] | str,
    new_external_data_file: str,
    new_header_path: os.PathLike[Any] | str,
) -> None:
    header = open_header(header_path)
    header.external_data.filename = new_external_data_file
    save_header(header, new_header_path)


__all__ = ["external_data", "open_header", "save_header", "print_header", "rename_external_data_file"]
