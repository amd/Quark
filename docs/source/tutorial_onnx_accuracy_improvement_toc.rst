Accuracy Improvement Methods
==============================

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Layer-wise Percentile
      :link: tutorials/onnx/accuracy_improvement/layerwise/onnx_layerwise_tutorial
      :link-type: doc

      Improve quantization accuracy by calibrating per-layer clipping ranges using percentile statistics.

   .. grid-item-card:: Cross Layer Equalization (CLE)
      :link: tutorials/onnx/accuracy_improvement/cle/onnx_cle_tutorial
      :link-type: doc

      Equalize weight ranges across adjacent layers to reduce quantization error before calibration.

   .. grid-item-card:: ADAQuant
      :link: tutorials/onnx/accuracy_improvement/adaquant/onnx_adaquant_tutorial
      :link-type: doc

      Apply adaptive rounding of quantized weights to minimize layer-wise reconstruction error.

   .. grid-item-card:: ADARound
      :link: tutorials/onnx/accuracy_improvement/adaround/onnx_adaround_tutorial
      :link-type: doc

      Learn task-aware rounding decisions for quantized weights to recover post-quantization accuracy.

   .. grid-item-card:: Mixed Precision
      :link: tutorials/onnx/accuracy_improvement/mixed_precision/onnx_mixed_precision_tutorial
      :link-type: doc

      Assign different bit-widths to different layers to balance accuracy and performance.

   .. grid-item-card:: Smooth Quant
      :link: tutorials/onnx/accuracy_improvement/smooth_quant/onnx_smooth_quant_tutorial
      :link-type: doc

      Migrate quantization difficulty from activations to weights via mathematically equivalent scaling.

.. toctree::
   :hidden:
   :caption: Accuracy Improvement Methods
   :maxdepth: 0

   Layer-wise Percentile<tutorials/onnx/accuracy_improvement/layerwise/onnx_layerwise_tutorial>
   Cross Layer Equalization (CLE)<tutorials/onnx/accuracy_improvement/cle/onnx_cle_tutorial>
   ADAQuant<tutorials/onnx/accuracy_improvement/adaquant/onnx_adaquant_tutorial>
   ADARound<tutorials/onnx/accuracy_improvement/adaround/onnx_adaround_tutorial>
   Mixed Precision<tutorials/onnx/accuracy_improvement/mixed_precision/onnx_mixed_precision_tutorial>
   Smooth Quant<tutorials/onnx/accuracy_improvement/smooth_quant/onnx_smooth_quant_tutorial>
