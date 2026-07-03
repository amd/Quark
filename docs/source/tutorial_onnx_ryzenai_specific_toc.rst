Ryzen AI-Specific Tutorials
============================

AMD Ryzen AI is AMD's NPU-enabled platform for on-device AI inference. Unlike
general-purpose quantization, Ryzen AI-specific quantization requires that the
model and its quantization configuration satisfy the hardware constraints of the
Ryzen AI NPU — such as supported operators, data types, and graph topology.
Each tutorial below covers how to meet those constraints for a specific model
or use case.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ResNet50 on Ryzen AI
      :link: tutorials/onnx/ryzen_ai/resnet50/onnx_ryzen_ai_resnet50_tutorial
      :link-type: doc

      Quantize and deploy ResNet50 on AMD Ryzen AI hardware using Quark for ONNX.

   .. grid-item-card:: YOLOv8 on Ryzen AI
      :link: tutorials/onnx/ryzen_ai/yolov8/onnx_ryzen_ai_yolov8_tutorial
      :link-type: doc

      Quantize YOLOv8 object detection model and run it on AMD Ryzen AI hardware.

   .. grid-item-card:: Auto Search on MobileNetv2-50 (Ryzen AI)
      :link: tutorials/onnx/ryzen_ai/auto_search_for_ryzen_ai/auto_search_mobilenetv2_50_custom_evaluator/onnx_ryzen_ai_auto_search_mobilenetv2_50_tutorial
      :link-type: doc

      Use Auto Search with a custom evaluator to find optimal quantization config for MobileNetv2-50 on Ryzen AI.

   .. grid-item-card:: Auto Search on ResNet50 (Ryzen AI)
      :link: tutorials/onnx/ryzen_ai/auto_search_for_ryzen_ai/auto_search_resnet50/onnx_ryzen_ai_auto_search_resnet50_tutorial
      :link-type: doc

      Automatically search for the best quantization strategy for ResNet50 targeting Ryzen AI.

   .. grid-item-card:: Auto Search on YOLOv8 (Ryzen AI)
      :link: tutorials/onnx/ryzen_ai/auto_search_for_ryzen_ai/auto_search_yolov8/onnx_ryzen_ai_auto_search_yolov8_tutorial
      :link-type: doc

      Apply Auto Search to find the optimal mixed-precision config for YOLOv8 on Ryzen AI.

.. toctree::
   :hidden:
   :caption: Ryzen AI-Specific Tutorials
   :maxdepth: 0

   ResNet50<tutorials/onnx/ryzen_ai/resnet50/onnx_ryzen_ai_resnet50_tutorial>
   YOLOv8<tutorials/onnx/ryzen_ai/yolov8/onnx_ryzen_ai_yolov8_tutorial>
   Auto Search On MobileNetv2-50<tutorials/onnx/ryzen_ai/auto_search_for_ryzen_ai/auto_search_mobilenetv2_50_custom_evaluator/onnx_ryzen_ai_auto_search_mobilenetv2_50_tutorial>
   Auto Search On ResNet50<tutorials/onnx/ryzen_ai/auto_search_for_ryzen_ai/auto_search_resnet50/onnx_ryzen_ai_auto_search_resnet50_tutorial>
   Auto Search On YOLOv8<tutorials/onnx/ryzen_ai/auto_search_for_ryzen_ai/auto_search_yolov8/onnx_ryzen_ai_auto_search_yolov8_tutorial>
