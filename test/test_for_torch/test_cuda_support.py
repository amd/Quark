#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from quark.testing.common_utils import TestCase, skip_if_no_gpu


class TestCudaInfoFromTorch(TestCase):

    @skip_if_no_gpu
    def test_get_cuda_info_from_server(self):
        torch.cuda.is_available()
        torch.cuda.device_count()
        torch.cuda.get_arch_list()
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i))
