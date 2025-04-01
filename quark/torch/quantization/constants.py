#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch.nn as nn

QUARK_LAYER_TYPES = {"Conv2d": nn.Conv2d, "Linear": nn.Linear, "ConvTranspose2d": nn.ConvTranspose2d}
