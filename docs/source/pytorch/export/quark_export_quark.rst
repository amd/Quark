Quark Format
============

Quark Format Exporting
----------------------

Quark format is a proprietary export format for Quark, and the file list of
this exporting format contains the quantized parameters (pth) such as weight, scale, and zero point and config.json with quantization  configuration.

Note that this model currently only supports exporting linear parts (which is sufficient for general large language modeling)
For other needs using quark export (e.g., exporting embedding layers, convolutional layers), use `Saving & Loading` below.
In fact, we are gradually migrating the `save and load` functionality to ModelExporter in `quark format`.

Example of Quark Format Exporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   export_path = "./output_dir"
   from llm_utils.export_import_hf_model import export_hf_model
   from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig, OnnxExporterConfig
   NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized",
                                              pack_method="reorder")
   export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG, onnx_export_config=OnnxExporterConfig())
   exporter = ModelExporter(config=export_config, export_dir=args.output_dir)
   exporter.export_quark_model(model, quant_config=quant_config, custom_mode=args.custom_mode)

By default, ``ModelExporter.export_quark_model`` exports models using a Quark-specific format for the checkpoint and ``quantization_config`` format in the ``config.json`` file.

This format may not directly be usable by some downstream libraries (vLLM) until downstream libraries support Quark quantized models. But it can be loaded and used by quark itself.

This format supports two forms of weight saving, `fake quantized` will save the high precision weight after quantization , while `real quantized` will save the weights after the real quantization. You can configure this with `weight_format`.

.. code:: python

   NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized", pack_method="reorder")
   export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG, onnx_export_config=OnnxExporterConfig())
   exporter = ModelExporter(config=export_config, export_dir=args.output_dir)
   exporter.export_quark_model(model, quant_config=quant_config, custom_mode=args.custom_mode)


Quark Format Importing
----------------------

Models exported using quark format can be imported directly using quark. Models exported using quark format can be imported directly using quark. quark chooses how to load the weights based on the information in the config.

Example of Quark Format Importing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from quark.torch import ModelImporter
   importer = ModelImporter(model_info_dir=args.import_model_dir)
   model = importer.import_model_info(model)
