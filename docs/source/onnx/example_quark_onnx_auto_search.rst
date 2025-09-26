.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

.. raw:: html

   <!-- omit in toc -->

Auto-Search for General Yolov8 ONNX Quantization
================================================

This folder contains an example of Auto search for quantizing a yolov8 model based on the ONNX quantizer of Quark. The example has the following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Prepare data <#prepare-data>`__
-  `Quantization with auto_search <#quantization-with-auto_search>`__

Pip requirements
----------------

Install the necessary python packages:

::

   python -m pip install -r requirements.txt

Prepare model
-------------

To download the YOLOv8 model from Ultralytics, run the following commands:

.. code-block:: python

   # We use yolov8n for this demo. Feel free to change to other YOLO models
   model = YOLO("yolov8n.pt")
   model.export(format="onnx")

Prepare data
------------

COCO 2017 — commonly known simply as “COCO” — contains 5,000 validation images.
Organize your data folder as follows:

.. code-block::

   -  val_data
         -  sample_1.jpg
         -  sample_2.jpg
         -  …

we use this dataset as evaluation dataset and calibration dataset at the same time.

Quantization with auto_search
-----------------------------

The quantization and auto-search configurations use their default settings. You can customize them in auto_search_model.py to suit your requirements.
To start the auto-search, run the following Python script:

.. code-block:: bash

   python auto_search_model.py --model_path yolov8n.onnx --dataset_path val_data

This command generates a series of configurations from the auto_search settings. As long as the stop condition remains false, the instance samples configurations from the entire search space according to the selected search algorithm. Each sampled configuration is then used to quantize the input model with Quark ONNX. The evaluator computes the chosen metric on the quantized model and checks whether it falls within the defined tolerance. Models that meet the tolerance are added to the output dictionary; those that don't are discarded.

To reduce computational load for this demo, we only set to run two search spaces, but we have defined 10 more spaces in the auto_search_model.py. You are welcome to test all of them or define your own search spaces based on their needs.

License
-------

Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
