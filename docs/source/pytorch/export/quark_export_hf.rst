HuggingFace Format
==================

Exporting to HuggingFace Format
-------------------------------

HF format is an optional exporting format for
Quark, and the file list of this exporting format is the same as the
file list of the original HuggingFace model, with quantization
information added to these files. Taking the llama2-7b model as an
example, the exported file list and added information are as below:

+------------------------------+--------------------------------------------------------------------------+
| File name                    | Additional Quantization Information                                      |
+------------------------------+--------------------------------------------------------------------------+
| config.json                  | Quantization configurations                                              |
+------------------------------+--------------------------------------------------------------------------+
| generation_config.json       | \-                                                                       |
+------------------------------+--------------------------------------------------------------------------+
| model*.safetensors           | Quantization info (tensors of scaling factor, zero point)                |
+------------------------------+--------------------------------------------------------------------------+
| model.safetensors.index.json | Mapping information of scaling factor and zero point to safetensors files|
+------------------------------+--------------------------------------------------------------------------+
| special_tokens_map.json      | \-                                                                       |
+------------------------------+--------------------------------------------------------------------------+
| tokenizer_config.json        | \-                                                                       |
+------------------------------+--------------------------------------------------------------------------+
| tokenizer.json               | \-                                                                       |
+------------------------------+--------------------------------------------------------------------------+

Example of HF Format Exporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   export_path = "./output_dir"
   from llm_utils.export_import_hf_model import export_hf_model
   from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig, OnnxExporterConfig
   NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized",
                                              pack_method="reorder")
   export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG, onnx_export_config=OnnxExporterConfig())
   export_hf_model(model, export_config, args.model_dir, args.output_dir, quant_config, custom_mode=args.custom_mode)

By default, ``export_hf_model`` exports models by `model.save_pretrained()` using a Quark-specific format for the checkpoint and ``quantization_config`` format in the ``config.json`` file. This format may not directly be usable by some downstream libraries (AutoAWQ, vLLM).

Until downstream libraries support Quark quantized models, one may export models so that the weight checkpoint and ``config.json`` file targets a specific downstream libraries, using ``custom_mode="awq"`` or ``custom_mode="fp8"``. Example:

.. code:: python

   # `custom_mode="awq"` would e.g. use `qzeros` instead of `weight_zero_point`, `qweight` instead of `weight` in the checkpoint.
   # Moreover, the `quantization_config` in the `config.json` file is custom, and the full quark `Config` is not serialized.
   export_hf_model(model, export_config, args.model_dir, args.output_dir, quant_config, custom_mode="awq")

Importing HuggingFace Format
----------------------------

Quark provides the importing function for `HF format` export files.
In other words, these files can be reloaded into Quark. After reloading,
the weights of the quantized operators in the model are stored in the real_quantized format.

Currently, this importing function supports weight-only, static, and dynamic quantization for
FP8 and AWQ. For other quantization methods, only weight-only and static
quantization are supported.

Example of HF Format Importing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from llm_utils.export_import_hf_model import export_hf_model, import_hf_model
   import_model_dir = "./output_dir"
   model = import_hf_model(model, model_info_dir=import_model_dir)
