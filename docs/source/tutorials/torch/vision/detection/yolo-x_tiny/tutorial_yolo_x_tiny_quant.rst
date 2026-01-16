YOLO-X Tiny Quant example
=========================

In this example, we present an Object Detection Model Quantization
workflow. We used YOLO-X Tiny as a demonstration to illustrate the
effectiveness of FX-graph-based QAT and PTQ.

1. We conduct **QAT** (Quantization-Aware Training) experiments and show
   competitive results compared with **PTQ** (Post-Training
   Quantization).
2. The finally exported ONNX model can be used for NPU hardware compile
   and deployment.
3. The detailed code about
   ``YOLO-X Tiny Model <https://github.com/Megvii-BaseDetection/YOLOX>``\ \_
   can be found in Megvii Research.

This repo contains the code for the training, evaluation, etc. In this
quant example code, we adopt the original repo and the majority of the
code to perform the quantization.

Highlight Overview
------------------

- **Quantization schema**: INT8 (quant range [-128, 127]), symmetric,
  power-of-2 scale (e.g., 1/(2**4)) for weight, bias, activation.
- **Hardware friendly**: Step-by-step instructions to deploy in the AMD
  NPU.
- **Satisfied Quant results**: For the original FP32 model, the
  detection results get the 32.8mAP on COCO val dataset. Using the Quark
  FX quant tool, the PTQ model gets 25.2 mAP. After QAT(training), the
  final quantized model gets 30.3 30.3 mAP. This means that even after
  int8 and pow-of-2 format scale quantization, the quantized model can
  recover over 92% of the original floating-point model.

Important Information
---------------------

YOLO-X Tiny is an object detection model in computer vision tasks.
Developed by Megvii. The original GitHub repo can be found here
``YOLOX <https://github.com/Megvii-BaseDetection/YOLOX>``\ \_. We use
code from this repo to perform the quantization and only keep the demand
code.

**Modify the YOLO-X model code** As we adopt the official PyTorch API to
trace the orthodox PyTorch code to get the ``torch.fx.GraphModule``
format computation graph. We need to modify the original model code. As:
- In original repo: 1. In the YOLO-X ``forward`` process, both the
``loss computation code <https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolox.py>``\ \_
and the final
``bounding-boxes decoding <https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py>``\ \_
code are included. However, for quantization, we only need to quantize
the model itself; we should not quantize the loss computation process. -
As a result, we modify the code and let the neural network body as
``base_model``, in ``base_model`` model, contain no loss computation and
bounding-boxes. We only need to trace the ``base_model`` to get the
``fx`` model to insert quantizers to perform the quantization.
Meanwhile, not influence the training procedure. After the modification,
the modified code was saved in ``yolo-x_tiny/models/``. The user can
compare the code to find the difference.

**For better & easier quantization**, we only use one GPU to perform the
quantization, which reduces a lot of complexity in the code. We have
cleaned up the code and reduced the amount of code a lot. We have
cleaned up the code and reduced the amount of code a lot.

**Quantization scope:** In Yolo-X, the model mainly contains two parts,
the model body and the detection head. In the detection head, there are
several constant tensors used for the final bounding box decode. In this
example, we quantized the YOLO-X model body. All weight, bias, and
activation tensors are quantized.. The detection head part is not
quantized, meaning it keeps the FP32 computation. The following image
shows that the detection head is not quantized.

Prepare the environment
-----------------------

To run this tutorial, you must prepare the
``COCO Dataset <https://cocodataset.org/#download>``\ \_ 2017 Dataset: -
User can follow the instructions of the
``YOLO-X <https://github.com/Megvii-BaseDetection/YOLOX/tree/main>``\ \_
repo.

Also, download the pre-trained weights at
``YOLO-X Tiny Weight <https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth>``\ \_

This tutorial uses local caches defined by environment variable
LOCAL_MODEL_CACHE and LOCAL_DATA_CACHE, but if they are not defined on
your environment, please download the required checkpoint and dataset
store them in the current folder!

.. code:: ipython3

    import os
    
    if os.environ.get("LOCAL_MODEL_CACHE") is not None:
        checkpoint_path = os.path.join(
            os.environ["LOCAL_MODEL_CACHE"], "quark_tutorials_cache", "torch", "vision", "detection", "yolo-x_tiny"
        )
    else:
        checkpoint_path = os.getcwd()
    print(f"The dataset cache is {checkpoint_path}")
    
    if os.environ.get("LOCAL_DATA_CACHE") is not None:
        dataset_path = os.path.join(os.environ["LOCAL_DATA_CACHE"], "coco", "2017")
    else:
        dataset_path = os.getcwd()
    print(f"The dataset cache is {dataset_path}")

With the dataset and checkpoint out of the way, let’s import a few basic
modules and do our log configuration

.. code:: ipython3

    import argparse
    import itertools
    import logging
    import sys
    
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)

Next, we import the core modules for our experiment

.. code:: ipython3

    import torch
    from trainer import Trainer
    from yolo_x_tiny_exp import Exp

This is how we defined our CLI interface using ``argparse``:

.. code:: ipython3

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=f"{checkpoint_path}/yolox_tiny.pth", type=str, help="pre train checkpoint")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--random_size_range", type=int, default=3, help="random_size")
    parser.add_argument("--experiment_name", type=str, default="0", help="exp name")
    parser.add_argument("--data_dir", default=f"{dataset_path}", help="Data set directory.")
    
    parser.add_argument("--min_lr_ratio", type=float, default=0.01, help="batch size")
    parser.add_argument("--ema_decay", type=float, default=0.9995, help="ema decay reate.")
    
    parser.add_argument("--output_dir", default="./YOLOX_outputs", help="Experiments results save path.")
    parser.add_argument("--workers", default=4, type=int, help="Number of data loading workers to be used.")
    parser.add_argument("--multiscale_range", default=5, type=int, help="multiscale_range.")
    parser.add_argument("--start_epoch", type=int, default=280, help="batch size")
    args = parser.parse_args([])

Initializing the experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have defined our own ``Trainer`` and ``Exp`` classes which wraps the
Yolo X hyper-parameters.

.. code:: ipython3

    # Uses local cached dataset for this experiment
    args.ckpt = os.path.join(checkpoint_path, "yolox_tiny.pth")
    
    exp = Exp(args)
    trainer = Trainer(exp, args)
    
    logging.info(f"args: {trainer.args}")
    logging.info(f"exp value:\n{trainer.exp}")

Prepare FP32 model & test accuracy
----------------------------------

Next we load the model with its pre-trained weights and evaluator.

.. code:: ipython3

    model = trainer.exp.get_model()
    model.to(trainer.device)
    model = trainer.load_pretrain_weight(model)
    trainer.model = model
    
    trainer.evaluator = trainer.exp.get_evaluator(batch_size=int(trainer.args.batch_size / 2))

Evaluate the full precision model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before quantizing the model, let’s get a baseline through the full
precision model using the COCO validation dataset.

.. code:: ipython3

    *_, summary = trainer.evaluator.evaluate(trainer.model)
    print(summary)

Post training quantization
--------------------------

Prepare Quantization Config & Quantizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that the weight, bias, output and input set to int8, per-tensor,
pow-of-2, symmetric quantization, which is more friendly for AMD NPU
hardware.

.. code:: ipython3

    from quark.torch import ModelQuantizer
    from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
    from quark.torch.quantization.config.type import Dtype, QSchemeType, QuantizationMode, RoundType, ScaleType
    from quark.torch.quantization.observer.observer import PerTensorPowOf2MinMSEObserver

.. code:: ipython3

    INT8_PER_WEIGHT_TENSOR_SPEC = QTensorConfig(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorPowOf2MinMSEObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    quant_config = QLayerConfig(
        weight=INT8_PER_WEIGHT_TENSOR_SPEC,
        input_tensors=INT8_PER_WEIGHT_TENSOR_SPEC,
        output_tensors=INT8_PER_WEIGHT_TENSOR_SPEC,
        bias=INT8_PER_WEIGHT_TENSOR_SPEC,
    )
    quant_config = QConfig(global_quant_config=quant_config, quant_mode=QuantizationMode.fx_graph_mode)
    trainer.quantizer = ModelQuantizer(quant_config)

Prepare calibration dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    calib_data = [x[0].to(trainer.device) for x in list(itertools.islice(trainer.evaluator.dataloader, 1))]
    dummy_input = torch.randn(1, 3, *trainer.exp.input_size).to(trainer.device)
    trainer.model = trainer.model.eval()

Extract FX Graph Model
~~~~~~~~~~~~~~~~~~~~~~

Next we trace the PyTorch code and prepare the FX graph model.

NOTE: Based on the original YOLO_X Tiny repo code, loss calculation and
bounding-boxes decode code are integrated in YOLO_X Tiny ``forward``, we
modify the code and let the ``trainer.model.base_model`` only contain
the backbone network. We only need to quantize this part of the model.

.. code:: ipython3

    graph_model = torch.export.export_for_training(trainer.model.base_model, (dummy_input,)).module()
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    trainer.model.base_model = graph_model

Perform Post Training Quantization (PTQ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    quantized_model = trainer.quantizer.quantize_model(graph_model, calib_data)
    trainer.model.base_model = quantized_model

Evaluate the quantized model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    *_, summary = trainer.evaluator.evaluate(trainer.model)
    print(summary)

Perform QAT based on PTQ results
--------------------------------

1. Based on the PTQ results, we perform the PTQ, through training, and
   adjust the weight/bias. This can get higher results.
2. We adopt the training code from the original YOLO-X Tiny repo, and we
   train the model from 280 epoch. Based on the development time and our
   work focused mainly on the Quark Fx QAT tool, we only tried several
   parameters to perform training. Differently, we using one single GPU
   to perform training to largely reduce the training complexity. The
   user can try other hyperparameters to get higher results.

Prepare the Dataloader & Optimizer etc.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from data import DataPrefetcher
    from trainer import ModelEMA

.. code:: ipython3

    trainer.no_aug = trainer.start_epoch >= trainer.max_epoch - trainer.exp.no_aug_epochs
    trainer.train_loader = trainer.exp.get_data_loader(
        batch_size=trainer.args.batch_size, no_aug=trainer.no_aug, cache_img=None
    )
    logging.info("init prefetcher, this might take one minute or less...")
    trainer.prefetcher = DataPrefetcher(trainer.train_loader)
    
    trainer.max_iter = len(trainer.train_loader)
    trainer.lr_scheduler = trainer.exp.get_lr_scheduler(
        trainer.exp.basic_lr_per_img * trainer.args.batch_size, trainer.max_iter
    )
    trainer.optimizer = trainer.exp.get_optimizer(trainer.args.batch_size)
    #  ------ using ema for better coverage ---
    if trainer.use_model_ema:
        trainer.ema_model = ModelEMA(trainer.model, trainer.args.ema_decay)  # 0.9995
        trainer.ema_model.updates = trainer.max_iter * trainer.start_epoch

Perform training to further improve accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NOTE: We only training one epoch for demonstration

.. code:: ipython3

    logging.info("Training start...")
    # logging.info("\n{}".format(trainer.model))
    trainer.epoch = 280
    logging.info(f"---> start train epoch{trainer.epoch + 1}")

**NOTE**: in function, ``train_in_iter``, 1. We close the observer,
meaning, during training the scale will not change; 2. Based on
experience, we found that during training, we close the ``bn`` update
that can get higher results.

.. code:: ipython3

    trainer.train_in_iter()

Evaluate the model
~~~~~~~~~~~~~~~~~~

To simplify, we directly load the fintuned weight to test accuracy

.. code:: ipython3

    trainer.save_ckpt("best")  # TODO: @meng: is this ok? why didnt we use trainer.evaluate_and_save_model here?
    trainer.model.load_state_dict(torch.load(os.path.join(trainer.file_name, "best_ckpt.pth"), weights_only=False)["model"])
    *_, summary = trainer.evaluator.evaluate(trainer.model)
    print(summary)

The summary may as follows Average Precision (AP) @[ IoU=0.50:0.95 \|
area= all \| maxDets=100 ] = 0.\ **Average Precision (AP) @[ IoU=0.50 \|
area= all \| maxDets=100 ] = 0.** Average Precision (AP) @[ IoU=0.75 \|
area= all \| maxDets=100 ] = 0.\ **Average Precision (AP) @[
IoU=0.50:0.95 \| area= small \| maxDets=100 ] = 0.** Average Precision
(AP) @[ IoU=0.50:0.95 \| area=medium \| maxDets=100 ] = 0.\ **Average
Precision (AP) @[ IoU=0.50:0.95 \| area= large \| maxDets=100 ] = 0.**

Freeze model & export to onnx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Freeze model
^^^^^^^^^^^^

For better deployment in the AMD NPU device, we apply several hardware
optimizations (e.g. adjust the scale, insert multiply nodes to perform
adjustment for hardware)

.. code:: ipython3

    frozen_model = trainer.quantizer.freeze(trainer.model.base_model.eval())
    trainer.model.base_model = frozen_model

Export to ONNX
^^^^^^^^^^^^^^

.. code:: ipython3

    from quark.torch import export_onnx

.. code:: ipython3

    # NOTE for NPU compile, it is better using batch-size = 1 for better compliance
    example_inputs = (torch.rand(1, 3, 416, 416).to(trainer.device),)
    export_onnx(model=trainer.model, output_dir="./export_onnx/", input_args=example_inputs[0])

Simplity the Onnx model and visualize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    import onnx
    from onnxslim import slim
    
    quant_model = onnx.load("./export_onnx/quark_model.onnx")
    model_simp = slim(quant_model)
    onnx.save_model(model_simp, "./export_onnx/sample_quark_model.onnx")

Using ``netron`` to visualize the model (Optional)

.. code:: shell

   $netron  ./export_onnx/sample_quark_model.onnx

Quantization Results
--------------------

The results is get under the image resolution under 416 \* 416. In
addition, the hyperparameter such as ``nmsthre`` and ``test_conf`` will
also influence the test results. We use the default of the YOLO-X repo.

+---------------+-----------------+--------------+
| Model format  | mAP 0.50:0.95   | mAP 0.50     |
+===============+=================+==============+
| FP32          | 32.6            | 50.0         |
+---------------+-----------------+--------------+
| PTQ int8      | 25.5 (78.2%)    | 43.0 (86.0%) |
+---------------+-----------------+--------------+
| QAT int8      | 30.3 (92.9%)    | 48.3 (96.6%) |
+---------------+-----------------+--------------+
