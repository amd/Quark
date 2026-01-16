#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import csv
import os
import re
import subprocess
import time
from collections import defaultdict

import matplotlib.pyplot as plt


def get_rocm_memory_usage(gpu_ids: list[int] = [6]) -> list[tuple[int, int, float, float]]:
    """Get the memory usage of a specified GPU, returning [(gpu_id, percent, used_MB, total_MB)]."""
    result = subprocess.run(["rocm-smi", "--showmemuse"], capture_output=True, text=True)
    text = result.stdout
    pattern = re.compile(r"GPU\[(\d+)\].*GPU Memory Allocated \(VRAM%\):\s*(\d+)", re.I)
    matches = pattern.findall(text)
    percent_dict = {int(gid): int(percent) for gid, percent in matches}

    if gpu_ids is not None:
        percent_dict = {gid: percent_dict.get(gid, 0) for gid in gpu_ids}

    result2 = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True)
    total_text = result2.stdout
    total_pattern = re.compile(r"GPU\[(\d+)\].*VRAM Total Memory.*:\s*(\d+)", re.I)
    total_matches = total_pattern.findall(total_text)
    total_dict = {int(gid): int(bytes_) / 1024**2 for gid, bytes_ in total_matches}

    usage = []
    for gid, percent in percent_dict.items():
        total_mb = total_dict.get(gid, 0)
        used_mb = total_mb * percent / 100
        usage.append((gid, percent, used_mb, total_mb))
    return usage


def monitor_rocm_memory(
    gpu_ids: list[int],
    interval: float = 2.0,
    duration: float = 30,
    save_path: str = "gpu_memory_plot.png",
    save_csv: bool = False,
) -> None:
    """Monitor the memory usage of a specified GPU in real time and draw an animated graph; save the image after completion."""
    print(f"Monitoring GPUs {gpu_ids} for {duration}s (interval={interval}s)...\n")

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("ROCm GPU Memory Usage Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory Used (MB)")

    gpu_data = defaultdict(list)
    time_data = []

    start_time = time.time()
    lines = {}

    while True:
        t = time.time() - start_time
        time_data.append(t)

        usage = get_rocm_memory_usage(gpu_ids=gpu_ids)
        for i, (gid, percent, used, total) in enumerate(usage):
            gpu_data[gid].append(used)
            if gid not in lines:
                (line,) = ax.plot(time_data, gpu_data[gid], label=f"GPU{gid}")
                lines[gid] = line
            else:
                lines[gid].set_xdata(time_data)
                lines[gid].set_ydata(gpu_data[gid])

            print(f"[{t:5.1f}s] GPU{gid}: {used:.0f} MB / {total:.0f} MB ({percent}%)")

        ax.relim()
        ax.autoscale_view()
        ax.legend(loc="upper left")
        plt.pause(0.01)

        if t >= duration:
            break
        time.sleep(interval)

    plt.ioff()

    # Save image
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"GPU memory usage plot saved to: {os.path.abspath(save_path)}")

    # Optional saving as CSV
    if save_csv:
        csv_path = os.path.splitext(save_path)[0] + ".csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Time(s)"] + [f"GPU{gid}_Used(MB)" for gid in list(gpu_data.keys())]
            writer.writerow(header)
            for i in range(len(time_data)):
                row = [time_data[i]] + [
                    gpu_data[gid][i] if i < len(gpu_data[gid]) else "" for gid in list(gpu_data.keys())
                ]
                writer.writerow(row)
        print(f"Data saved to: {os.path.abspath(csv_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitort the ROCM GPU memory usage.")
    parser.add_argument("--gpu_id", type=int, default=6, help="GPU Node id to be moniterd.")
    parser.add_argument("--interval", type=float, default=0.5, help="Interval for time sampling, default is 0.5s.")
    parser.add_argument("--duration", type=float, default=3600.0, help="Duration for GPU memory monitoring.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="gpu_memory_usage.png",
        help="Path to save the GPU memory usage, default is gpu_memory_usage.png",
    )
    parser.add_argument("--save_csv", action="store_true", help="Wether the monitored data as .csv file")
    args = parser.parse_args()

    monitor_rocm_memory(
        gpu_ids=[args.gpu_id],
        interval=args.interval,
        duration=args.duration,
        save_path=args.save_path,
        save_csv=args.save_csv,
    )
