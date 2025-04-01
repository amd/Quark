.. raw:: html

   <!--
   ## License
   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->

BFP16 (Block floating point) Quantization
=========================================

Introduction
------------

In this tutorial, we will learn how to use BFP16 data type with Quark.

BFP is short for block floating point. A floating point number consists
of 1 sign bit, 8 exponent bits and 23 mantissa bits. The main idea of
block floating point is a block of numbers sharing one exponent and the
mantissa of each number shifting right accordingly.

This
`paper <https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf>`__
introduces the attempt to apply BFP to deep neural networks(DNNs). The
specific BFP16 is widely used across AI industry. The definition of
BFP16 in quark is a block consisting of 8 numbers, the shared exponent
consisting of 8 bits and the rest of each number consisting of 1 sign
bit and 7 mantissa bits.

How to use BFP16 in Quark
-------------------------

1. Install Quark:
~~~~~~~~~~~~~~~~~

Follow the steps in the :doc:`installation guide <../install>` to
install Quark

2. Set the model:
~~~~~~~~~~~~~~~~~

.. code:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
   model.eval()
   tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

We are retrieving the model from
`HuggingFace <https://huggingface.co/>`__ using their
`Transformers <https://huggingface.co/docs/transformers/index>`__
library. We are using the model facebook/opt-125m as an example

3. Set the quantization configuration:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
   from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig
   from quark.torch.quantization.observer.observer import PerBlockBFPObserver
   DEFAULT_BFP16_PER_BLOCK = QuantizationSpec(dtype=Dtype.int8,
                                              symmetric=True,
                                              observer_cls=PerBlockBFPObserver, # for BFP16 the observer_cls is always PerBlockBFPObserver
                                              qscheme=QSchemeType.per_group, # for BFP16 the qscheme is always QSchemeType.per_group
                                              is_dynamic=True, # this controls whether static or dynamic quantization is performed
                                              ch_axis=-1,
                                              scale_type=ScaleType.float,
                                              group_size=8,
                                              round_method=RoundType.half_even)

   DEFAULT_W_BFP16_PER_BLOCK_CONFIG = QuantizationConfig(weight=DEFAULT_BFP16_PER_BLOCK)
   quant_config = Config(global_quant_config=DEFAULT_W_BFP16_PER_BLOCK_CONFIG)

In Quark, we store the 1 sign bit and 7 mantissa bits as a single int8,
so the dtype should be ``Dtype.int8``. The observer class
PerBlockBFPObserver is used for shared exponent calculation.

4. Do quantization
~~~~~~~~~~~~~~~~~~

we initialize a ModelQuantizer with the quant_config constructed above
and call the method ``quantize_model`` to do quantization:

.. code:: python

   from quark.torch import ModelQuantizer
   from torch.utils.data import DataLoader
   import torch
   calib_dataloader = DataLoader(torch.randint(0, 1000, (1, 64))) # Using random inputs is for demonstration purpose only
   quantizer = ModelQuantizer(quant_config)
   quant_model = quantizer.quantize_model(model, calib_dataloader)

In practice, users should construct meaningful calibration datasets.

How BFP16 works in Quark
------------------------

Quantizing floating point tensor to BFP16 tensor consists of three main
steps: getting shared exponent, shifting mantissas right accordingly and
doing rounding on mantissa.

We use the max exponent in each block as the shared exponent. Then we
shift mantissa of each element right accordingly. Note that in BFP, the
implicit one is included in mantissa. Finally, we do rounding and remove
the trailing mantissa bits. Only the rounding method half_to_even has
been supported by now.
