Microscaling (MX) Example
=========================

.. note::

   For information on accessing Quark ONNX examples, refer to :doc:`Accessing ONNX Examples <onnx_examples>`.
   This example and the relevant files are available at ``/onnx/accuracy_improvement/MX``.

This folder provides an example of quantizing a ResNet50 model using Quark's ONNX quantizer
with Microexponents and Microscaling formats.

Microexponents (abbreviated as MX) extend the Block Floating Point (BFP) concept by
introducing two levels of exponents: a shared exponent for entire blocks and microexponents
for finer-grained sub-blocks. This enables more precise scaling of individual elements
within a block, improving accuracy while retaining computational efficiency. It has three
concrete formats: MX4, MX6, and MX9.

Microscaling (also abbreviated as MX) builds on the BFP approach by allowing small-scale
adjustments for individual elements. It defines independent data formats, such as FP8 (E5M2
and E4M3), FP6 (E3M2 and E2M3), FP4 (E2M1), and INT8, to achieve fine-grained scaling within
blocks. This technique enhances numerical precision, especially for low-precision computations.

The example has the following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Prepare data <#prepare-data>`__
-  `Quantization with MX Formats <#quantization-with-mx-formats>`__
-  `Evaluation <#evaluation>`__


Pip requirements
----------------

Install the necessary Python packages:

.. code-block:: bash

   python -m pip install -r ../utils/requirements.txt

Prepare model
-------------

Download the ONNX float model from the `onnx/models <https://github.com/onnx/models>`__ repo directly:

.. code-block:: bash

   wget -P models https://github.com/onnx/models/raw/new-models/vision/classification/resnet/model/resnet50-v1-12.onnx

Prepare data
------------

ILSVRC 2012, commonly known as 'ImageNet'. This dataset provides access
to ImageNet (ILSVRC) 2012, which is the most commonly used subset of
ImageNet. This dataset spans 1000 object classes and contains 50,000
validation images.

If you already have an ImageNet dataset, you can directly use your
dataset path.

To prepare the test data, please check the download section of the main
website: `imagenet-1k dataset <https://huggingface.co/datasets/imagenet-1k/tree/main/data>`__. You
need to register and download **val_images.tar.gz**.

Then, create the validation dataset and calibration dataset:

.. code-block:: bash

   mkdir val_data && tar -xzf val_images.tar.gz -C val_data
   python ../utils/prepare_data.py val_data calib_data

The storage format of the ``val_data`` of the ImageNet dataset is organized as
follows:

.. code-block::

   -  val_data

      -  n01440764

         -  ILSVRC2012_val_00000293.JPEG
         -  ILSVRC2012_val_00002138.JPEG
         -  …

      -  n01443537

         -  ILSVRC2012_val_00000236.JPEG
         -  ILSVRC2012_val_00000262.JPEG
         -  …

      -  …
      -  n15075141

         -  ILSVRC2012_val_00001079.JPEG
         -  ILSVRC2012_val_00002663.JPEG
         -  …

The storage format of the ``calib_data`` of the ImageNet dataset is organized
as follows:

.. code-block::

   -  calib_data

      -  n01440764

         -  ILSVRC2012_val_00000293.JPEG

      -  n01443537

         -  ILSVRC2012_val_00000236.JPEG

      -  …
      -  n15075141

         -  ILSVRC2012_val_00001079.JPEG

Quantization with MX Formats
----------------------------

The quantizer takes the float model and produces a MX quantized model.

There are several built-in configurations within the quantizer for MX formats, that
are named as 'MX4', 'MX6', 'MX9', 'MXFP8E5M2', 'MXFP8E4M3', 'MXFP6E3M2', 'MXFP6E2M3',
'MXFP4E2M1' and 'MXINT8'. We can choose one of the formats by passing the name of
the configuration to the script.

Here is an example of MXINT8 quantization:

.. code-block:: bash

   python quantize_model.py --input_model_path models/resnet50-v1-12.onnx \
                            --output_model_path models/resnet50-v1-12_quantized.onnx \
                            --calibration_dataset_path calib_data \
                            --config MXINT8

This command will generate a MXINT8 quantized model under the **models** folder.

Evaluation
----------

Test the accuracy of the float model on the ImageNet validation dataset:

.. code-block:: bash

   python onnx_validate.py val_data --batch-size 1 --onnx-input models/resnet50-v1-12.onnx

Test the accuracy of the MX quantized model on the ImageNet
validation dataset:

.. code-block:: bash

   python onnx_validate.py val_data --batch-size 1 --onnx-input models/resnet50-v1-12_quantized.onnx

If you want to run faster with GPU support, you can also execute the following command:

.. code-block:: bash

   python onnx_validate.py val_data --batch-size 1 --onnx-input models/resnet50-v1-12_quantized.onnx --gpu

Here are the comparison results of these data types:

+---------------------+---------------------+---------------------+---------------------+
| Data Type           |     Model Size      |         Top1        |         Top5        |
+=====================+=====================+=====================+=====================+
| Float               |       97.82 MB      |       74.114 %      |       91.716 %      |
+---------------------+---------------------+---------------------+---------------------+
| MX4                 |       97.47 MB      |       0.764 %       |       2.742 %       |
+---------------------+---------------------+---------------------+---------------------+
| MX6                 |       97.47 MB      |       67.642 %      |       88.182 %      |
+---------------------+---------------------+---------------------+---------------------+
| MX9                 |       97.47 MB      |       73.996 %      |       91.658 %      |
+---------------------+---------------------+---------------------+---------------------+
| MXFP8E5M2           |       97.47 MB      |       64.076 %      |       87.248 %      |
+---------------------+---------------------+---------------------+---------------------+
| MXFP8E4M3           |       97.47 MB      |       70.052 %      |       89.922 %      |
+---------------------+---------------------+---------------------+---------------------+
| MXFP6E3M2           |       97.47 MB      |       64.090 %      |       87.256 %      |
+---------------------+---------------------+---------------------+---------------------+
| MXFP6E2M3           |       97.47 MB      |       71.766 %      |       90.684 %      |
+---------------------+---------------------+---------------------+---------------------+
| MXFP4E2M1           |       97.47 MB      |       18.446 %      |       41.512 %      |
+---------------------+---------------------+---------------------+---------------------+
| MXINT8              |       97.47 MB      |       73.920 %      |       91.662 %      |
+---------------------+---------------------+---------------------+---------------------+

.. note::

   Different execution devices can lead to minor variations in the accuracy of the quantized model.

.. raw:: html

   <!--
   ## License
   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
