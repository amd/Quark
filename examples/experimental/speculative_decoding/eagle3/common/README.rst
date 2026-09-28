.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Shared EAGLE-3 Runner
=====================

This directory is the canonical implementation used by every example profile.
It owns the ROCm image, orchestration, on-policy data preparation, TorchSpec
training, checkpoint selection, vLLM export, serving, and benchmark reporting.

Profile directories select configs and runtime defaults, then delegate to
``run_all.sh``. The parent example's legacy paths remain ordinary compatibility
wrappers so source archives and wheels work without symlink support.
