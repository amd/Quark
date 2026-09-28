.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

EAGLE-3 Example Profiles
========================

``qwen3_8b_quick_start``
  Validated Qwen3-8B configuration and thin runner selector.

``minimax_m3_best_recipe``
  Portable large-model baseline demonstrated with the public
  ``amd/MiniMax-M3-MXFP4`` target.

``common``
  Canonical setup, data, training, export, serving, and benchmark assets shared
  by both profiles.

Each profile directory holds a thin ``run.sh`` that selects its assets and
delegates to ``common/run_all.sh``. The Python CLI resolves the profile itself
and calls that wrapper, so a new profile needs a directory here and nothing
above it.
