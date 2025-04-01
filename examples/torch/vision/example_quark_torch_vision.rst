Vision Model Quantization Using Quark FX Graph Mode
====================================================

This example demonstrates a vision model quantization workflow. You specify a ``nn.Module`` and transform the model to ``torch.fx.GraphModule`` format using the PyTorch API. During the quantization process, after annotation and insertion of quantizers, this modified ``fx.GraphModule`` can be used to perform PTQ (Post-Training Quantization) and/or QAT (Quantization Aware Training). Demonstration code is provided to show how you can assign ``quant config``.

In this example, we present a vision model quantization workflow. The
user specified a ``nn.Module`` and transformed the model to
``torch.fx.GraphModule`` format by using PyTorch API. During the
quantization process, after annotation and insertion quantizers, this
modified ``fx.GraphModule`` can be used to perform PTQ (Post Training
Quantization), or/and QAT (Quantization Aware Training). We supply a
demonstration code and show how users assign ``quant config``, more
information can be found in User Guide.

Get example code and script
~~~~~~~~~~~~~~~~~~~~~~~~~~~
After unzip ``amd_quark.zip`` (referring to :doc:`Installation Guide <../install>`).
The example folder is in amd_quark.zip. In folder ``/examples/torch/vision``, user can get the detailed explanation of
image classification and object detection quantization demonstration code.
.. note::

   For information on accessing Quark PyTorch examples, refer to `Accessing PyTorch Examples <pytorch_examples>`_.
   This example and the relevant files are available at ``/torch/vision``.

PTQ
~~~

In Post-Training Quantization (PTQ), after inserting ``FakeQuantize``, the ``observer`` is activated during calibration to record the tensor's distribution. Values such as minimum and maximum are recorded to calculate quantization parameters, without performing fake quantization. This ensures all calculations are under FP32 precision. After calibration, you can activate the fake quantizer to perform quantization and evaluation.

QAT
~~~

Similar to Post-Training Quantization (PTQ), after preparing the model, both the ``observer`` and ``fake_quant`` are active during the training process. The ``observer`` records the tensor's distribution, including minimum and maximum values, to calculate quantization parameters. The tensor is then quantized by ``fake_quant``.

TQT
~~~

This method involves uniform symmetric quantizers using standard backpropagation and gradient descent. Unlike Quantization-Aware Training (QAT), Trained Quantization Thresholds (TQT) add a gradient for scale factors. Unlike Learned Step Size Quantization (LSQ), which directly trains scale factors and may encounter stability issues, TQT constrains scale factors to powers of two and uses a gradient formulation to train log-thresholds instead. Theoretically, TQT is superior to LSQ, and LSQ is superior to QAT. For efficient fixed-point implementations, TQT constrains the quantization scheme to use symmetric quantization, per-tensor scaling, and power-of-two scaling. Currently, TQT supports only signed data. More experimental results are forthcoming.

Quick Start
-----------

Perform Post-Training Quantization (PTQ) to obtain the quantized model and export it to ONNX:

.. code-block:: bash

   python3 quantize.py --data_dir [Train and Test Data folder] \
                       --model_name [mobilenetv2 or resnet18] \
                       --pretrained [Pre-trained model file address] \
                       --model_export onnx \
                       --export_dir [directory to save exported model]

You can also choose to perform Quantization-Aware Training (QAT) to further enhance classification accuracy. Typically, some training parameters need to be adjusted for higher accuracy:

.. code-block:: bash

   python3 quantize.py --data_dir [Train and Test Data folder] \
                       --model_name [mobilenetv2 or resnet18] \
                       --pretrained [Pre-trained model file address] \
                       --model_export onnx \
                       --export_dir [directory to save exported model] \
                       --qat True

LSQ and TQT are optimized methods for QAT that can theoretically improve accuracy. The parameters ``--tqt True`` and ``--lsq True`` are available for you to try. Model export is not supported at this time.

Fine-Grained User Guide
-----------------------

**Step 1: Prepare the floating-point model, dataset, and loss function**

.. code-block:: python

   from torchvision.models import resnet18
   float_model = resnet18(pretrained=False)
   float_model.load_state_dict(torch.load(pretrained))
   calib_loader = prepare_calib_dataset(args.data_dir, device, calib_length=args.train_batch_size * 10)
   train_loader, val_loader = prepare_data_loaders(args.data_dir)
   criterion = nn.CrossEntropyLoss().to(device)

**Step 2: Transform the ``torch.nn.Module`` to ``torch.fx.GraphModule``**

.. code-block:: python

   from torch._export import capture_pre_autograd_graph
   example_inputs = (torch.rand(args.train_batch_size, 3, 224, 224).to(device), )
   graph_model = capture_pre_autograd_graph(float_model, example_inputs)

**Step 3: Initialize the quantizer and quantization configuration**

.. code-block:: python

   from quark.torch.quantization.config.config import QuantizationSpec, QuantizationConfig, Config
   from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType, QuantizationMode
   from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
   INT8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                           qscheme=QSchemeType.per_tensor,
                                           observer_cls=PerTensorMinMaxObserver,
                                           symmetric=True,
                                           scale_type=ScaleType.float,
                                           round_method=RoundType.half_even,
                                           is_dynamic=False)
   quant_config = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC,
                                         output_tensors=INT8_PER_TENSOR_SPEC,
                                         weight=INT8_PER_TENSOR_SPEC,
                                         bias=INT8_PER_TENSOR_SPEC)
   quant_config = Config(global_quant_config=quant_config,
                         quant_mode=QuantizationMode.fx_graph_mode)
   quantizer = ModelQuantizer(quant_config)

**Step 4: Generate the quantized graph model by performing calibration**

.. code-block:: python

   quantized_model = quantizer.quantize_model(graph_model, calib_loader)

**Step 5 (Optional): Perform QAT for higher accuracy**

.. code-block:: python

   train(quantized_model, train_loader, val_loader, criterion, device_ids)

**Step 6: Validate model performance and export**

.. code-block:: python

   acc1_quant = validate(val_loader, quantized_model, criterion, device)
   freezed_model = quantizer.freeze(prepared_model)
   acc1_freeze = validate(val_loader, freezed_model, criterion, device)
   # Check whether acc1_quant == acc1_freeze

   # ============== Export to ONNX ==================
   from quark.torch import ModelExporter
   from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
   config = ExporterConfig(json_export_config=JsonExporterConfig())
   exporter = ModelExporter(config=config, export_dir=args.export_dir)
   example_inputs = (torch.rand(batch_size, 3, 224, 224).to(device),)
   exporter.export_onnx_model(freezed_model, example_inputs[0])

   # ========== Export using torch.export ============
   example_inputs = (next(iter(val_loader))[0].to(device),)
   model_file_path = os.path.join(args.export_dir, args.model_name + ".pth")
   exported_model = torch.export.export(freezed_model, example_inputs)
   torch.export.save(exported_model, model_file_path)

Experiment Results
------------------

1. Image Classification Task PTQ/QAT Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We conduct PTQ and QAT on both ResNet-18 and MobileNet-V2. In these models, all weights, biases, and activations are quantized. All types of tensors are quantized in INT8, per-tensor, symmetric (zero point is 0). The scale factor is in float format. The following table shows the validation accuracy on the ImageNet dataset produced by the above script.

.. list-table::
   :header-rows: 1

   * - Method
     - ResNet-18
     - MobileNetV2
   * - Float Model
     - 69.764 / 89.085
     - 71.881 / 90.301
   * - PTQ (INT8)
     - 69.084 / 88.648
     - 65.291 / 86.254
   * - QAT (INT8)
     - 69.469 / 88.872
     - 68.562 / 88.484

2. Object Detection Task PTQ/QAT Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We conduct PTQ and QAT on YOLO-NAS. In this model quantization, we partially quantize the model by assigning the configuration.

.. list-table::
   :header-rows: 1

   * - Metric
     - FP32 model
     - INT8 PTQ
     - INT8 QAT
   * - mAP@0.50
     - 0.6466
     - 0.6236
     - 0.6239
   * - mAP@0.50:0.95
     - 0.4759
     - 0.4537
     - 0.4532
