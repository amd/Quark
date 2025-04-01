#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer, CalibrationMethod, PowerOfTwoMethod
from quark.onnx import get_library_path
from quark.onnx.quantization.config.custom_config import (
        XINT8_CONFIG, XINT8_ADAROUND_CONFIG, XINT8_ADAQUANT_CONFIG,
        S8S8_AAWS_CONFIG, S8S8_AAWS_ADAROUND_CONFIG, S8S8_AAWS_ADAQUANT_CONFIG,
        U8S8_AAWS_CONFIG, U8S8_AAWS_ADAROUND_CONFIG, U8S8_AAWS_ADAQUANT_CONFIG,
        U8U8_AAWA_CONFIG, S16S8_ASWS_CONFIG, A8W8_CONFIG, A16W8_CONFIG, U16S8_AAWS_CONFIG, U16S8_AAWS_ADAROUND_CONFIG, U16S8_AAWS_ADAQUANT_CONFIG,
        FP16_CONFIG, FP16_ADAQUANT_CONFIG, BF16_CONFIG, BF16_ADAQUANT_CONFIG, BFP16_CONFIG, BFP16_ADAQUANT_CONFIG,
        MX4_CONFIG, MX4_ADAQUANT_CONFIG, MX6_CONFIG, MX6_ADAQUANT_CONFIG, MX9_CONFIG, MX9_ADAQUANT_CONFIG,
        MXFP8E5M2_CONFIG, MXFP8E5M2_ADAQUANT_CONFIG, MXFP8E4M3_CONFIG, MXFP8E4M3_ADAQUANT_CONFIG, MXFP6E3M2_CONFIG, MXFP6E3M2_ADAQUANT_CONFIG,
        MXFP6E2M3_CONFIG, MXFP6E2M3_ADAQUANT_CONFIG, MXFP4E2M1_CONFIG, MXFP4E2M1_ADAQUANT_CONFIG, MXINT8_CONFIG, MXINT8_ADAQUANT_CONFIG,
        BF16_MIXED_BFP16_ADAQUANT_CONFIG, BF16_BFP16_CONFIG, BF16_MXINT8_CONFIG, MX9_INT8_CONFIG,
        INT16_CNN_ACCURATE_CONFIG, INT16_CNN_DEFAULT_CONFIG, INT8_CNN_ACCURATE_CONFIG, INT8_CNN_DEFAULT_CONFIG,
        S16S8_ASWS_ADAQUANT_CONFIG, S16S8_ASWS_ADAROUND_CONFIG, S16S16_MIXED_S8S8_CONFIG)
from testing_utils import prepare_model
from quark.onnx.quantization.config.config import Config
from quark.shares.utils.testing_utils import use_temporary_directory

input_tensor = np.array([[[[0.26921557, 0.79500909, 0.6102178, 0.04375664],
                           [0.06221361, 0.98258356, 0.38635129, 0.06492238],
                           [0.49631707, 0.35442799, 0.51719146, 0.52100111],
                           [0.04145599, 0.88960236, 0.50627326, 0.57204613]],
                          [[0.99185097, 0.93582153, 0.13174529, 0.42896287],
                           [0.14552133, 0.02538564, 0.0732355, 0.25725371],
                           [0.09856916, 0.43015628, 0.55679755, 0.66560074],
                           [0.9439425, 0.45701841, 0.86791293, 0.64728276]],
                          [[0.29159685, 0.79021383, 0.3117182, 0.11342342],
                           [0.16660495, 0.46426165, 0.31348552, 0.143383],
                           [0.96454802, 0.63258874, 0.30295267, 0.96720039],
                           [0.29879457, 0.79916527, 0.02905061, 0.20115725]]]]).astype(np.float32)

XINT8_golden_output = np.array([[[[0.25390625, 0.16015625, 0.0234375, -0.046875],
                                  [0.1484375, -0.015625, 0.13671875,
                                   0.0390625], [0.125, 0.2734375, 0.421875, 0.11328125],
                                  [0.1484375, 0.13671875, 0.234375, 0.28515625]]]]).astype(np.float32)

XINT8_ADAROUND_golden_output = np.array([[[[0.25, 0.16015625, 0.0234375, -0.05078125],
                                           [0.1484375, -0.01953125, 0.13671875, 0.0390625],
                                           [0.12109375, 0.26953125, 0.421875, 0.11328125],
                                           [0.1484375, 0.13671875, 0.234375, 0.28515625]]]]).astype(np.float32)

XINT8_ADAQUANT_golden_output = np.array([[[[0.25, 0.16015625, 0.0234375, -0.05078125],
                                           [0.14453125, -0.01953125, 0.1328125, 0.0390625],
                                           [0.12109375, 0.265625, 0.41796875, 0.11328125],
                                           [0.1484375, 0.13671875, 0.23046875, 0.28515625]]]]).astype(np.float32)

S8S8_AAWS_golden_output = np.array([[[[0.24136764, 0.1602976, 0.02395251, -0.04790503],
                                      [0.14371508, -0.03500752, 0.13818759, 0.03685002],
                                      [0.12344757, 0.26900515, 0.41456276, 0.11423507],
                                      [0.1510851, 0.13818759, 0.23584014, 0.28558767]]]]).astype(np.float32)

S8S8_AAWS_ADAROUND_golden_output = np.array([[[[0.23952514, 0.1602976, 0.02395251, -0.04790503],
                                               [0.14187258, -0.03685002, 0.13818759, 0.03869252],
                                               [0.12160508, 0.26900515, 0.41272026, 0.11423507],
                                               [0.1492426, 0.13818759, 0.23584014, 0.28558767]]]]).astype(np.float32)

S8S8_AAWS_ADAQUANT_golden_output = np.array([[[[0.24136764, 0.1602976, 0.02395251, -0.04790503],
                                               [0.14187258, -0.03685002, 0.13818759, 0.03869252],
                                               [0.12160508, 0.26900515, 0.41456276, 0.11607757],
                                               [0.1492426, 0.13818759, 0.23584014, 0.28558767]]]]).astype(np.float32)

U8S8_AAWS_golden_output = np.array([[[[0.24136764, 0.1602976, 0.02395251, -0.04790503],
                                      [0.14371508, -0.03500752, 0.13818759, 0.03685002],
                                      [0.12344757, 0.26900515, 0.41456276, 0.11423507],
                                      [0.1510851, 0.13818759, 0.23584014, 0.28558767]]]]).astype(np.float32)

U8S8_AAWS_ADAROUND_golden_output = np.array([[[[0.23952514, 0.1602976, 0.02395251, -0.04790503],
                                               [0.14187258, -0.03685002, 0.13818759, 0.03869252],
                                               [0.12160508, 0.26900515, 0.41272026, 0.11423507],
                                               [0.1492426, 0.13818759, 0.23584014, 0.28558767]]]]).astype(np.float32)

U8S8_AAWS_ADAQUANT_golden_output = np.array([[[[0.24136764, 0.1602976, 0.02395251, -0.04790503],
                                               [0.14187258, -0.03685002, 0.13818759, 0.03869252],
                                               [0.12160508, 0.26900515, 0.41456276, 0.11607757],
                                               [0.1492426, 0.13818759, 0.23584014, 0.28558767]]]]).astype(np.float32)

U8U8_AAWA_golden_output = np.array([[[[0.25058016, 0.15845509, 0.02211001, -0.04790503],
                                      [0.14555758, -0.02026751, 0.13450257, 0.03685002],
                                      [0.12160508, 0.26716265, 0.41824776, 0.11239257],
                                      [0.1492426 , 0.13450257, 0.23399764, 0.28374517]]]]).astype(np.float32)

S16S8_ASWS_golden_output = np.array([[[[0.25221714, 0.16079743, 0.02421408, -0.04817111],
                                       [0.1461713, -0.01723518, 0.1378429, 0.03842893],
                                       [0.12316535, 0.26825705, 0.42113736, 0.11437426],
                                       [0.15020698, 0.13766296, 0.23648572, 0.28625053]]]]).astype(np.float32)

U16S8_AAWS_golden_output = np.array([[[[0.2522219, 0.16079944, 0.02420344, -0.04817029],
                                       [0.14616698, -0.01724208, 0.13784346, 0.03842726],
                                       [0.12316081, 0.26826674, 0.42113698, 0.11437846],
                                       [0.15020327, 0.13765706, 0.23647821, 0.28625444]]]]).astype(np.float32)

U16S8_AAWS_ADAROUND_golden_output = np.array([[[[0.2507164, 0.16078511, 0.0235367, -0.04842839],
                                                [0.14599492, -0.01780845, 0.13730577, 0.03827671],
                                                [0.12124661, 0.26816636, 0.42113698, 0.11468673],
                                                [0.14957239, 0.13645263, 0.23558205, 0.28625444]]]]).astype(np.float32)

U16S8_AAWS_ADAQUANT_golden_output = np.array([[[[0.25057298, 0.15959501, 0.02388799, -0.04870082],
                                                [0.1457655, -0.01739981, 0.13859624, 0.03828388],
                                                [0.12164092, 0.2685105, 0.42113698, 0.1141347],
                                                [0.14882678, 0.13829513, 0.23608391, 0.28570238]]]]).astype(np.float32)

FP16_golden_output = np.array([[[[0.25048828, 0.16027832, 0.02336121, -0.04867554],
                                 [0.1463623, -0.01860046, 0.13671875, 0.03826904],
                                 [0.12158203, 0.2680664, 0.42138672, 0.11395264],
                                 [0.14929199, 0.13684082, 0.23522949, 0.28564453]]]]).astype(np.float32)

FP16_ADAQUANT_golden_output = np.array([[[[0.25073242, 0.16027832, 0.02336121, -0.04870605],
                                          [0.1463623, -0.01852417, 0.13696289, 0.03842163],
                                          [0.12158203, 0.2680664, 0.42138672, 0.11401367],
                                          [0.14929199, 0.13684082, 0.23547363, 0.28588867]]]]).astype(np.float32)

BF16_golden_output = np.array([[[[0.25195312, 0.16113281, 0.02368164, -0.04907227],
                                 [0.14648438, -0.01916504, 0.13671875, 0.03759766],
                                 [0.12207031, 0.26757812, 0.421875, 0.11376953],
                                 [0.14941406, 0.13769531, 0.23535156, 0.28515625]]]]).astype(np.float32)

BF16_ADAQUANT_golden_output = np.array([[[[0.25000000, 0.16015625, 0.0234375, -0.05078125],
                                          [0.1484375, -0.01953125, 0.13671875, 0.0390625],
                                          [0.12109375, 0.26953125, 0.421875, 0.11328125],
                                          [0.1484375 , 0.13671875, 0.234375, 0.28515625]]]]).astype(np.float32)

BFP16_golden_output = np.array([[[[0.25390625, 0.16210938, 0.02612305, -0.04785156],
                                  [0.1484375, -0.0168457, 0.13867188, 0.0390625],
                                  [0.12304688, 0.2734375, 0.421875, 0.11425781],
                                  [0.15039062, 0.13867188, 0.23632812, 0.2890625]]]]).astype(np.float32)

BFP16_ADAQUANT_golden_output = np.array([[[[0.2500000, 0.15820312, 0.0222168, -0.04931641],
                                           [0.14453125, -0.02026367, 0.13671875, 0.03759766],
                                           [0.12109375, 0.265625000, 0.42187500, 0.11230469],
                                           [0.1484375, 0.13476562, 0.234375, 0.28515625]]]]).astype(np.float32)

MX4_golden_output = np.array([[[[0.25, 0.125, 0.0234375, -0.046875],
                                [0.125, 0.0234375, 0.09375, 0.00390625],
                                [0.09375, 0.25, 0.375, 0.0625],
                                [0.1875, 0.09375, 0.25, 0.375]]]]).astype(np.float32)

MX4_ADAQUANT_golden_output = np.array([[[[0.25, 0.1875, 0.03125, -0.046875],
                                         [0.125, 0.015625, 0.09375, 0.00390625],
                                         [0.125, 0.25, 0.375, 0.09375],
                                         [0.125, 0.09375, 0.25, 0.25]]]]).astype(np.float32)

MX6_golden_output = np.array([[[[0.25, 0.140625, 0.01171875, -0.05078125],
                                [0.140625, -0.02734375, 0.125, 0.03515625],
                                [0.109375, 0.25, 0.40625, 0.1015625],
                                [0.140625, 0.140625, 0.21875, 0.28125]]]]).astype(np.float32)

MX6_ADAQUANT_golden_output = np.array([[[[0.25, 0.15625, 0.015625, -0.046875],
                                         [0.15625, -0.00585938, 0.15625, 0.05859375],
                                         [0.125, 0.28125, 0.4375, 0.125],
                                         [0.15625, 0.15625, 0.234375, 0.28125]]]]).astype(np.float32)

MX9_golden_output = np.array([[[[0.25, 0.16015625, 0.02563477, -0.04785156],
                                [0.14648438, -0.01708984, 0.13867188, 0.03808594],
                                [0.12207031, 0.26953125, 0.421875, 0.11425781],
                                [0.1484375, 0.13671875, 0.23632812, 0.2890625]]]]).astype(np.float32)

MX9_ADAQUANT_golden_output = np.array([[[[0.25, 0.16015625, 0.0246582, -0.04785156],
                                         [0.14453125, -0.02001953, 0.13867188, 0.0390625],
                                         [0.12109375, 0.26953125, 0.421875, 0.11523438],
                                         [0.1484375, 0.13671875, 0.23632812, 0.2890625]]]]).astype(np.float32)

MXFP8E5M2_golden_output = np.array([[[[0.21875, 0.109375, 0.015625, -0.0546875],
                                      [0.125, 0.00109863, 0.109375, 0.0390625],
                                      [0.109375, 0.21875, 0.375, 0.1015625],
                                      [0.140625, 0.140625, 0.21875, 0.28125]]]]).astype(np.float32)

MXFP8E5M2_ADAQUANT_golden_output = np.array([[[[0.21875, 0.109375, 0.01171875, -0.0390625],
                                               [0.109375, -0.0390625, 0.109375, 0.046875],
                                               [0.109375, 0.21875, 0.4375, 0.109375],
                                               [0.125, 0.125, 0.21875, 0.25]]]]).astype(np.float32)

MXFP8E4M3_golden_output = np.array([[[[0.21875, 0.125, 0.0390625, -0.0390625],
                                      [0.109375, -0.0390625, 0.109375, 0.0390625],
                                      [0.109375, 0.21875, 0.4375, 0.109375],
                                      [0.125, 0.15625, 0.21875, 0.25]]]]).astype(np.float32)

MXFP8E4M3_ADAQUANT_golden_output = np.array([[[[0.25, 0.15625, 0.04296875, -0.046875],
                                               [0.140625, 0.00634766, 0.140625, 0.05078125],
                                               [0.125, 0.28125, 0.40625, 0.125],
                                               [0.15625, 0.15625, 0.21875, 0.28125]]]]).astype(np.float32)

MXFP6E3M2_golden_output = np.array([[[[0.234375, 0.140625, 0.01757812, -0.0546875],
                                      [0.140625, -0.00976562, 0.109375, 0.03515625],
                                      [0.109375, 0.234375, 0.375, 0.09375],
                                      [0.15625, 0.140625, 0.234375, 0.28125]]]]).astype(np.float32)

MXFP6E3M2_ADAQUANT_golden_output = np.array([[[[0.21875, 0.109375, 0.01171875, -0.0390625],
                                               [0.109375, -0.0390625, 0.109375, 0.046875],
                                               [0.109375, 0.21875, 0.4375, 0.109375],
                                               [0.125, 0.125, 0.21875, 0.25]]]]).astype(np.float32)

MXFP6E2M3_golden_output = np.array([[[[0.21875, 0.125, 0.0390625, -0.0390625],
                                      [0.109375, -0.0390625, 0.109375, 0.0390625],
                                      [0.109375, 0.21875, 0.4375, 0.109375],
                                      [0.125, 0.15625, 0.21875, 0.25]]]]).astype(np.float32)

MXFP6E2M3_ADAQUANT_golden_output = np.array([[[[0.234375, 0.15625, 0.02539062, -0.05078125],
                                               [0.140625, -0.00976562, 0.1171875, 0.03515625],
                                               [0.1171875, 0.25, 0.40625, 0.1015625],
                                               [0.15625, 0.140625, 0.234375, 0.28125]]]]).astype(np.float32)

MXFP4E2M1_golden_output = np.array([[[[0.1875, 0.046875, 0.01171875, -0.0625],
                                      [0.09375, 0.046875, 0.09375, 0.046875],
                                      [0.09375, 0.1875, 0.375, 0.09375],
                                      [0.09375, 0.09375, 0.1875, 0.25]]]]).astype(np.float32)

MXFP4E2M1_ADAQUANT_golden_output = np.array([[[[0.1875, 0.09375, 0.0234375, -0.046875],
                                               [0.125, 0.09375, 0.09375, 0.046875],
                                               [0.125, 0.25, 0.375, 0.125],
                                               [0.125, 0.125, 0.25, 0.25]]]]).astype(np.float32)

MXINT8_golden_output = np.array([[[[0.25390625, 0.16210938, 0.02612305, -0.04785156],
                                  [0.1484375, -0.0168457, 0.13867188, 0.0390625],
                                  [0.12304688, 0.2734375, 0.421875, 0.11425781],
                                  [0.15039062, 0.13867188, 0.23632812, 0.2890625]]]]).astype(np.float32)

MXINT8_ADAQUANT_golden_output = np.array([[[[0.25, 0.15820312, 0.0222168, -0.04931641],
                                            [0.14453125, -0.02026367, 0.13671875, 0.03759766],
                                            [0.12109375, 0.265625, 0.421875, 0.11230469],
                                            [0.1484375, 0.13476562, 0.234375, 0.28515625]]]]).astype(np.float32)

BF16_MIXED_BFP16_ADAQUANT_golden_output = np.array([[[[0.24902344, 0.15917969, 0.02087402, -0.05029297],
                                                      [0.14453125, -0.02246094, 0.13378906, 0.03662109],
                                                      [0.11914062, 0.265625, 0.41796875, 0.11279297],
                                                      [0.14941406, 0.13476562, 0.23242188, 0.28320312]]]]).astype(np.float32)

BF16_BFP16_golden_output = np.array([[[[0.25195312, 0.16210938, 0.02490234, -0.04833984],
                                       [0.1484375, -0.01635742, 0.13867188, 0.03881836],
                                       [0.12255859, 0.27148438, 0.421875, 0.11474609],
                                       [0.15039062, 0.13867188, 0.23730469, 0.28710938]]]]).astype(np.float32)

BF16_MXINT8_golden_output = np.array([[[[0.25195312, 0.16210938, 0.02490234, -0.04833984],
                                        [0.1484375, -0.01635742, 0.13867188, 0.03881836],
                                        [0.12255859, 0.27148438, 0.421875, 0.11474609],
                                        [0.15039062, 0.13867188, 0.23730469, 0.28710938]]]]).astype(np.float32)

MX9_INT8_golden_output = np.array([[[[0.25390625, 0.16015625, 0.02514648, -0.04785156],
                                     [0.14648438, -0.01733398, 0.13867188, 0.03808594],
                                     [0.12402344, 0.26953125, 0.42578125, 0.11425781],
                                     [0.15039062, 0.13671875, 0.23632812, 0.28515625]]]]).astype(np.float32)

INT16_CNN_golden_output = np.array([[[[0.25062764, 0.16027145, 0.02336007, -0.04869968],
                                      [0.14636442, -0.01859768, 0.13696876, 0.03838596],
                                      [0.12159142, 0.26808494, 0.42133474, 0.11401751],
                                      [0.14927636, 0.13684683, 0.23538658, 0.28582913]]]]).astype(np.float32)

INT16_CNN_ACCURATE_golden_output = np.array([[[[0.250616, 0.1602044, 0.02335747, -0.04870082],
                                               [0.14628169, -0.01853972, 0.13696164, 0.03838425],
                                               [0.12158357, 0.26808032, 0.42113698, 0.11401282],
                                               [0.14927126, 0.13684693, 0.23538132, 0.28583142]]]]).astype(np.float32)

INT8_CNN_golden_output = np.array([[[[0.23952514, 0.1602976, 0.02395251, -0.04790503],
                                     [0.14187258, -0.03685002, 0.13818759, 0.03869252],
                                     [0.12160508, 0.26900515, 0.41272026, 0.11423507],
                                     [0.1492426, 0.13818759, 0.23584014, 0.28558767]]]]).astype(np.float32)

INT8_CNN_ACCURATE_golden_output = np.array([[[[0.24146867, 0.16036469, 0.02396254, -0.04792508],
                                              [0.14377524, -0.03502217, 0.13824542, 0.03686544],
                                              [0.12349924, 0.26911774, 0.41473624, 0.11428288],
                                              [0.14930505, 0.13824542, 0.23593885, 0.2857072]]]]).astype(np.float32)

S16S8_ASWS_ADAROUND_golden_output = np.array([[[[0.2507134, 0.16078457, 0.02354575, -0.04842816],
                                                [0.14600423, -0.01780069, 0.13730308, 0.03828755],
                                                [0.12125034, 0.26815423, 0.42113736, 0.11468272],
                                                [0.14956436, 0.13645482, 0.23558603, 0.28625053]]]]).astype(np.float32)

S16S8_ASWS_ADAQUANT_golden_output = np.array([[[[0.2507134, 0.16078457, 0.02354575, -0.04842816],
                                                [0.14600423, -0.01780069, 0.13730308, 0.03828755],
                                                [0.12125034, 0.26815423, 0.42113736, 0.11468272],
                                                [0.14956436, 0.13645482, 0.23558603, 0.28625053]]]]).astype(np.float32)

A8W8_golden_output = np.array([[[[0.19242026, 0.15924434, 0.02322313, -0.04644627],
                                 [0.1061629, -0.07298699, 0.13602121, 0.0364935],
                                 [0.11943326, 0.27204242, 0.37157014, 0.11279808],
                                 [0.1426564, 0.13602121, 0.23554893, 0.26872483]]]]).astype(np.float32)

A16W8_golden_output = np.array([[[[0.25223204, 0.1608725, 0.02419967, -0.04816788],
                                  [0.14623955, -0.01730752, 0.13784297, 0.03843401],
                                  [0.12315857, 0.2682537, 0.421334, 0.11437622],
                                  [0.15019996, 0.13766295, 0.23646754, 0.2862556]]]]).astype(np.float32)


S16S16_MIXED_S8S8_golden_output = np.array([[[[0.25254455, 0.06587121, -0.04870082, -0.04870082],
                                              [0.14815287, -0.0401909, -0.04870082, -0.04870082],
                                              [0.12147603, 0.09333666, 0.297503, -0.04870082],
                                              [0.15252613, 0.12260161, -0.04870082, 0.10817704]]]]).astype(np.float32)

class DataReader(CalibrationDataReader):

    def __init__(self, input_tensor):
        self.data = [input_tensor]
        self.input_name = 'input'
        self.index = 0

    def get_next(self):
        if self.index < len(self.data):
            input_dict = {self.input_name: self.data[self.index]}
            self.index += 1
            return input_dict
        else:
            return None

    def rewind(self):
        self.index = 0


def prepare_config(config):
    quant_config = Config(global_quant_config=config)
    return quant_config


def prepare_data():
    data_reader = DataReader(input_tensor)
    return data_reader


def prepare_quantizer(quant_config):
    quantizer = ModelQuantizer(quant_config)
    return quantizer


def quantize_static(quantizer, input_model_path, output_model_path, data_reader):
    quantizer.quantize_model(input_model_path, output_model_path, data_reader)
    print('Quantized the ONNX model and saved it at:', output_model_path)
    return output_model_path


def infer_quantized_model(quantized_model_path):
    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path())
    sess = onnxruntime.InferenceSession(quantized_model_path, sess_options)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f'Model output: {output}')
    return output


def tensor_quantize(config, output_dir):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config(config)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_XINT8(self, tmpdir: str):
        config = XINT8_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, XINT8_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_XINT8_ADAROUND(self, tmpdir: str):
        config = XINT8_ADAROUND_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, XINT8_ADAROUND_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_XINT8_ADAQUANT(self, tmpdir: str):
        config = XINT8_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, XINT8_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_S8S8_AAWS(self, tmpdir: str):
        config = S8S8_AAWS_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, S8S8_AAWS_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_S8S8_AAWS_ADAROUND(self, tmpdir: str):
        config = S8S8_AAWS_ADAROUND_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, S8S8_AAWS_ADAROUND_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_S8S8_AAWS_ADAQUANT(self, tmpdir: str):
        config = S8S8_AAWS_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, S8S8_AAWS_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_U8S8_AAWS(self, tmpdir: str):
        config = U8S8_AAWS_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, U8S8_AAWS_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_U8S8_AAWS_ADAROUND(self, tmpdir: str):
        config = U8S8_AAWS_ADAROUND_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, U8S8_AAWS_ADAROUND_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_U8S8_AAWS_ADAQUANT(self, tmpdir: str):
        config = U8S8_AAWS_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, U8S8_AAWS_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_U8U8_AAWA(self, tmpdir: str):
        config = U8U8_AAWA_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, U8U8_AAWA_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_S16S8_ASWS(self, tmpdir: str):
        config = S16S8_ASWS_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, S16S8_ASWS_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_U16S8_AAWS(self, tmpdir: str):
        config = U16S8_AAWS_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, U16S8_AAWS_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_U16S8_AAWS_ADAROUND(self, tmpdir: str):
        config = U16S8_AAWS_ADAROUND_CONFIG
        config.extra_options['AddQDQPairToWeight'] = True
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, U16S8_AAWS_ADAROUND_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_U16S8_AAWS_ADAQUANT(self, tmpdir: str):
        config = U16S8_AAWS_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, U16S8_AAWS_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_FP16(self, tmpdir: str):
        config = FP16_CONFIG
        config.extra_options['AddQDQPairToWeight'] = False
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, FP16_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_FP16_ADAQUANT(self, tmpdir: str):
        config = FP16_ADAQUANT_CONFIG
        config.calibrate_method = PowerOfTwoMethod.NonOverflow
        config.extra_options['AddQDQPairToWeight'] = False
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, FP16_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_BF16(self, tmpdir: str):
        config = BF16_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, BF16_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_BF16_ADAQUANT(self, tmpdir: str):
        config = BF16_ADAQUANT_CONFIG
        config.calibrate_method = CalibrationMethod.MinMax
        config.extra_options['WeightScaled'] = True
        config.extra_options['ActivationScaled'] = True
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, BF16_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_BFP16(self, tmpdir: str):
        config = BFP16_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, BFP16_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_BFP16_ADAQUANT(self, tmpdir: str):
        config = BFP16_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, BFP16_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MX4(self, tmpdir: str):
        config = MX4_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MX4_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MX4_ADAQUANT(self, tmpdir: str):
        config = MX4_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MX4_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MX6(self, tmpdir: str):
        config = MX6_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MX6_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MX6_ADAQUANT(self, tmpdir: str):
        config = MX6_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MX6_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MX9(self, tmpdir: str):
        config = MX9_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MX9_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MX9_ADAQUANT(self, tmpdir: str):
        config = MX9_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MX9_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXFP8E5M2(self, tmpdir: str):
        config = MXFP8E5M2_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MXFP8E5M2_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXFP8E5M2_ADAQUANT(self, tmpdir: str):
        config = MXFP8E5M2_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MXFP8E5M2_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXFP8E4M3(self, tmpdir: str):
        config = MXFP8E4M3_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MXFP8E4M3_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXFP8E4M3_ADAQUANT(self, tmpdir: str):
        config = MXFP8E4M3_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MXFP8E4M3_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXFP6E3M2(self, tmpdir: str):
        config = MXFP6E3M2_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MXFP6E3M2_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXFP6E3M2_ADAQUANT(self, tmpdir: str):
        config = MXFP6E3M2_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MXFP6E3M2_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXFP6E2M3(self, tmpdir: str):
        config = MXFP6E2M3_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MXFP6E2M3_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXFP6E2M3_ADAQUANT(self, tmpdir: str):
        config = MXFP6E2M3_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MXFP6E2M3_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXFP4E2M1(self, tmpdir: str):
        config = MXFP4E2M1_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MXFP4E2M1_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXFP4E2M1_ADAQUANT(self, tmpdir: str):
        config = MXFP4E2M1_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MXFP4E2M1_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXINT8(self, tmpdir: str):
        config = MXINT8_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MXINT8_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXINT8_ADAQUANT(self, tmpdir: str):
        config = MXINT8_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MXINT8_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_BF16_MIXED_BFP16_ADAQUANT(self, tmpdir: str):
        config = BF16_MIXED_BFP16_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, BF16_MIXED_BFP16_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_BF16_BFP16(self, tmpdir: str):
        config = BF16_BFP16_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, BF16_BFP16_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_BF16_MXINT8(self, tmpdir: str):
        config = BF16_MXINT8_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, BF16_MXINT8_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MX9_INT8(self, tmpdir: str):
        config = MX9_INT8_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, MX9_INT8_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_INT16_CNN(self, tmpdir: str):
        config = INT16_CNN_DEFAULT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, INT16_CNN_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_INT16_CNN_ACCURATE(self, tmpdir: str):
        config = INT16_CNN_ACCURATE_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, INT16_CNN_ACCURATE_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_INT8_CNN(self, tmpdir: str):
        config = INT8_CNN_DEFAULT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, INT8_CNN_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_INT8_CNN_ACCURATE(self, tmpdir: str):
        config = INT8_CNN_ACCURATE_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, INT8_CNN_ACCURATE_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_S16S8_ASWS_ADAROUND(self, tmpdir: str):
        config = S16S8_ASWS_ADAROUND_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, S16S8_ASWS_ADAROUND_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_S16S8_ASWS_ADAQUANT(self, tmpdir: str):
        config = S16S8_ASWS_ADAQUANT_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, S16S8_ASWS_ADAQUANT_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_S16S16_MIXED_S8S8(self, tmpdir: str):
        config = S16S16_MIXED_S8S8_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, S16S16_MIXED_S8S8_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_A8W8(self, tmpdir: str):
        config = A8W8_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, A8W8_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_A16W8(self, tmpdir: str):
        config = A16W8_CONFIG
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, A16W8_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)


if __name__ == '__main__':
    unittest.main()
