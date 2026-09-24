ShapeShifter Tutorials
======================

ShapeShifter is Quark's pass-based graph-transformation framework for ONNX (and
PyTorch) models. ONNX passes run in two stages: **preprocessing** passes prepare
the float model before quantization (constant folding, operator fusion, BatchNorm
folding, opset conversion, ...), and **postprocessing** passes adapt the quantized
Q/DQ model for a deployment target (Q/DQ scale alignment, bias-scale correction,
XINT8/NPU simulation, ...). The tutorials below show how to drive these passes.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ShapeShifter on ResNet50 (CLI and ShapeShifterYaml)
      :link: tutorials/onnx/shapeshifter/onnx_shapeshifter_resnet50_tutorial
      :link-type: doc

      Run ShapeShifter ONNX passes on ResNet50 two ways: standalone with the
      ``quark-cli shapeshifter`` command, and integrated into a quantization script
      via the ``ShapeShifterYaml`` option (preprocessing + postprocessing).

.. toctree::
   :hidden:
   :caption: ShapeShifter Tutorials
   :maxdepth: 0

   ShapeShifter on ResNet50<tutorials/onnx/shapeshifter/onnx_shapeshifter_resnet50_tutorial>
