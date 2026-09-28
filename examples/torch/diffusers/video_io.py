#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Safe video writing for diffusers pipeline output.

**The trap this exists to prevent.** ``diffusers.utils.export_to_video`` assumes a
``list[np.ndarray]`` is float in ``[0,1]`` and rescales it::

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

Hand it frames you have *already* converted to ``uint8`` and ``frame * 255`` wraps modulo
256, i.e. ``x -> 256 - x``: **every frame is colour-inverted**. The result is still
structurally coherent (motion, objects and edges all survive), so it does not look like a
crash -- it looks like a *quantization* failure: orange skies, magenta foliage, flat
patches of impossible neon colour. This cost a full misdiagnosis once; the inverted bf16
reference was blamed on sampling settings.

Passing PIL images avoids the rescale branch entirely and is stable across diffusers
versions (the ndarray branch's semantics have changed between releases).

Shared by the diffusers examples: ``quantize_diffusers.py`` (Wan smoke check) and
``wan14b_w4a8/reload_wan14b_w4a8.py``. Keep the conversion in one place -- two copies
drift, and the drifted copy is the one that inverts.
"""

from __future__ import annotations

import numpy as np


def to_uint8(frames) -> np.ndarray:
    """Normalize pipeline output (float [0,1] or uint8) to a uint8 [N,H,W,3] array."""
    f = np.asarray(frames)
    if f.dtype != np.uint8:
        f = (np.clip(f, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return f


def save_video(frames, path: str, fps: int = 16) -> str:
    """Write ``frames`` [N,H,W,3] uint8 (or float in [0,1]) to ``path`` without rescaling.

    Returns ``path``. Use this instead of calling ``export_to_video`` on a uint8 array.
    """
    from diffusers.utils import export_to_video
    from PIL import Image

    # PIL path: export_to_video does np.array(frame) with no scaling.
    export_to_video([Image.fromarray(f) for f in to_uint8(frames)], path, fps=fps)
    return path
