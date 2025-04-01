#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Evaluate quantization accuracy and loss between baseline and quantized results folders.

Use the evaluate.py to measure cosine similarity, l2 loss, psnr and vmaf between two results folders:

```
python evaluate.py --baseline_results_folder $BASELINE_RESULTS_FOLDER_PATH --quantized_results_folder $QUANTIZED_RESULTS_FOLDER_PATH
```

'''
import os
import cv2
import numpy as np
from typing import Any
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser("Evaluator")
    parser.add_argument("baseline_results_folder", type=str, help='Path to image folder 1')
    parser.add_argument("quantized_results_folder", type=str, help='Path to image folder 2')
    args, _ = parser.parse_known_args()
    return args


def ssim(img1: np.ndarray[Any, Any], img2: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    mu1 = cv2.GaussianBlur(img1.astype(np.float32), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2.astype(np.float32), (11, 11), 1.5)

    sigma1_sq = cv2.GaussianBlur(img1.astype(np.float32)**2, (11, 11), 1.5)
    sigma2_sq = cv2.GaussianBlur(img2.astype(np.float32)**2, (11, 11), 1.5)
    sigma12 = cv2.GaussianBlur(img1.astype(np.float32) * img2.astype(np.float32), (11, 11), 1.5)

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / (denominator + 1e-10)
    score = np.array(np.mean(ssim_map))
    assert isinstance(score, np.ndarray)
    return score


def calculate_cos(array1: np.ndarray[Any, Any], array2: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    arr1 = array1.astype(np.float32).flatten()
    arr2 = array2.astype(np.float32).flatten()
    dot_product = np.dot(arr1, arr2)
    norm_arr1 = np.linalg.norm(arr1)
    norm_arr2 = np.linalg.norm(arr2)
    cos_sim = dot_product / (norm_arr1 * norm_arr2)
    cos_sim = np.array(cos_sim)
    assert isinstance(cos_sim, np.ndarray)
    return cos_sim


def calculate_l2_distance(array1: np.ndarray[Any, Any], array2: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    l2_distance = np.linalg.norm(array1.astype(np.float32) - array2.astype(np.float32))
    l2_distance = np.array(l2_distance)
    assert isinstance(l2_distance, np.ndarray)
    return l2_distance


def calculate_psnr(reference_image: np.ndarray[Any, Any], noisy_image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:

    reference_image = reference_image.astype(np.float32)
    mse = np.mean((reference_image - noisy_image)**2)
    if mse == np.array(0):
        mse = mse + np.array(1e-10)
    max_pixel_value: np.ndarray[Any, Any] = np.max(reference_image)  # type: ignore
    if max_pixel_value <= np.array(0):
        max_pixel_value = np.array(1e-10)
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    psnr = np.array(psnr)
    assert isinstance(psnr, np.ndarray)
    return psnr


def calculate_vmaf(img1: np.ndarray[Any, Any], img2: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    score = ssim(img1, img2)
    assert isinstance(score, np.ndarray)
    return score


def main(baseline_results_folder: str, quantized_results_folder: str) -> None:

    metric_values_cos_image = []
    metric_values_l2_image = []
    metric_values_psnr_image = []
    metric_values_vmaf_image = []

    img_names = [f for f in os.listdir(baseline_results_folder) if f.endswith('.png') or f.endswith('.jpg')]

    for name in img_names:
        image1 = cv2.imread(os.path.join(baseline_results_folder, name))
        image2 = cv2.imread(os.path.join(quantized_results_folder, name))

        if image1 is None:
            print(f'Failed to load: {os.path.join(baseline_results_folder, name)}')
            continue
        if image2 is None:
            print(f'Failed to load: {os.path.join(quantized_results_folder, name)}')
            continue

        metric_values_cos_image.append(calculate_cos(image1, image2))
        metric_values_l2_image.append(calculate_l2_distance(image1, image2))
        metric_values_psnr_image.append(calculate_psnr(image1, image2))
        metric_values_vmaf_image.append(calculate_vmaf(image1, image2))

    npy_names = [f for f in os.listdir(baseline_results_folder) if f.endswith('.npy')]
    for name in npy_names:
        array1 = np.load(os.path.join(baseline_results_folder, name))
        array2 = np.load(os.path.join(quantized_results_folder, name))

        metric_values_cos_image.append(calculate_cos(array1, array2))
        metric_values_l2_image.append(calculate_l2_distance(array1, array2))
        metric_values_psnr_image.append(calculate_psnr(array1, array2))
        metric_values_vmaf_image.append(calculate_vmaf(array1, array2))

    print(f"Mean cos similarity: {np.mean(metric_values_cos_image)}")
    print(f"Min cos similarity: {np.min(metric_values_cos_image)}")
    print(f"Mean l2 distance: {np.mean(metric_values_l2_image)}")
    print(f"Max l2 distance: {np.max(metric_values_l2_image)}")
    print(f"Mean psnr: {np.mean(metric_values_psnr_image)}")
    print(f"Min psnr: {np.min(metric_values_psnr_image)}")
    print(f"Mean vmaf: {np.mean(metric_values_vmaf_image)}")
    print(f"Min vmaf: {np.min(metric_values_vmaf_image)}")


if __name__ == "__main__":
    args = parse_args()
    main(args.baseline_results_folder, args.quantized_results_folder)
