#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Wrapper for the "onnx-prepare-data" subcommand.
import argparse
import os
import shutil
import sys

from quark.experimental.cli import base_cli


class ONNXPrepareData_CLI(base_cli.BaseQuarkCLICommand):
    """
    Simple flow specifically for preparing ImageNet dataset by putting it into a form expected by Quark.
    Do not use with any other datasets.
    """

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser):
        parser.add_argument("--val_path", help="Path to validation data", required=True)
        parser.add_argument("--cal_path", help="Path to calibration data", required=True)

    def run(self):
        args = self.args

        source_folder = args.val_path
        calib_data_path = args.cal_path

        if not os.path.exists(source_folder):
            print("The provided data path does not exist.")
            sys.exit(1)

        files = os.listdir(source_folder)

        for filename in files:
            if not filename.startswith("ILSVRC2012_val_") or not filename.endswith(".JPEG"):
                continue

            n_identifier = filename.split("_")[-1].split(".")[0]
            folder_name = n_identifier
            folder_path = os.path.join(source_folder, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(source_folder, filename)
            destination = os.path.join(folder_path, filename)
            shutil.move(file_path, destination)

        print("File organization complete.")

        if not os.path.exists(calib_data_path):
            os.makedirs(calib_data_path)

        destination_folder = calib_data_path

        subfolders = os.listdir(source_folder)

        for subfolder in subfolders:
            source_subfolder = os.path.join(source_folder, subfolder)
            destination_subfolder = os.path.join(destination_folder, subfolder)
            os.makedirs(destination_subfolder, exist_ok=True)
            files = os.listdir(source_subfolder)

            if files:
                file_to_copy = files[0]
                source_file = os.path.join(source_subfolder, file_to_copy)
                destination_file = os.path.join(destination_subfolder, file_to_copy)

                shutil.copy(source_file, destination_file)

        print("Creating calibration dataset complete.")
