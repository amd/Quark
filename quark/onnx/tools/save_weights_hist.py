#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
A tool for show the weights distribution for model

    Example : python -m quark.onnx.tools.save_weights_hist --input_model [INPUT_MODEL_PATH] --output [OUTPUT_PATH] --perchannel

'''

import argparse
import onnx
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from quark.shares.utils.log import ScreenLogger
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = ScreenLogger(__name__)


def sanitize_filename(name: str) -> str:
    name = name.replace("/", "_")
    name = name.replace(".", "_")
    name = name.replace(":", "_")
    return name


def main(model_path: str, output_dir: str, perchannel: bool = False) -> None:  # set perchannel default value: False
    # Load the ONNX model
    model = onnx.load(model_path)

    # If no output directory is provided, use the default directory weights_hist
    if output_dir is None:
        output_dir = "weights_hist"

    # Create the output directory (if it does not exist)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tqdm progress bar
    total_tensors = len(model.graph.initializer)
    progress_bar = tqdm(total=total_tensors, desc="Saving Histograms")

    # Iterate over each weight tensor
    for initializer in model.graph.initializer:
        tensor_name = sanitize_filename(initializer.name)
        weight_array = onnx.numpy_helper.to_array(initializer)

        # Assume the weight array has a shape of (out_channels , ...)
        if weight_array.ndim >= 2 and perchannel:  # Ensure the tensor has at least two dimensions and perchannel is True
            out_channels = weight_array.shape[0]

            # Calculate the maximum and minimum values of the weights
            oc_values = weight_array.reshape(out_channels, -1)
            oc_max_values = np.max(oc_values, axis=-1)
            oc_min_values = np.min(oc_values, axis=-1)

            plt.figure(figsize=(12, 6))
            channels = np.arange(out_channels)

            plt.vlines(channels, oc_min_values, oc_max_values, colors='b', linewidth=2, label='Min-Max Values')

            plt.title(f'{tensor_name} per_channel Max and Min Values per Channel')
            plt.xlabel('Channel')
            plt.ylabel('Value')
            plt.legend()

            plt.tight_layout()
            # Clear the current figure
            # Log the save information
            output_path = os.path.join(output_dir, f'{tensor_name}_per_channel.png')
            # Save the histogram
            plt.savefig(output_path)
            plt.clf()
            plt.close()
            # Update progress bar
            progress_bar.update(1)

        else:
            # If perchannel is False or the tensor is not at least 2D, treat it as a single-channel tensor
            weights = weight_array.flatten()
            max_weight = np.max(weights)
            min_weight = np.min(weights)
            logger.debug(f"Tensor {tensor_name} - Max value: {max_weight}, Min value: {min_weight}")
            bins = np.linspace(min_weight, max_weight, 129)
            plt.hist(weights, bins=bins, edgecolor='black')

            # Add title and labels
            plt.title(f'{tensor_name}')
            plt.xlabel('Values')
            plt.ylabel('Frequency')

            output_path = os.path.join(output_dir, f'{tensor_name}.png')
            plt.savefig(output_path)
            plt.clf()
            plt.close()
            logger.debug(f"Histogram for {tensor_name} has been saved to {output_path}")

            # Update progress bar
            progress_bar.update(1)

    # Close progress bar
    progress_bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate histograms for each weight tensor in an ONNX model')
    parser.add_argument('--input_model', type=str, required=True, help='Path to the input ONNX model file')
    parser.add_argument('--output', type=str, help='Directory to save the histograms')
    parser.add_argument('--perchannel', action='store_true', help='Whether to generate histograms per channel')

    args = parser.parse_args()
    main(args.input_model, args.output, args.perchannel)
