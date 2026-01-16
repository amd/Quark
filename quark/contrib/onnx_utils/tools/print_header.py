# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse

import ryzenai_onnx_utils.proto as proto

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--header_file_name",
        type=str,
        required=True,
        help="Path to the header file containing external data information.",
    )
    args = parser.parse_args()

    header = proto.open_header(args.header_file_name)
    proto.print_header(header)
