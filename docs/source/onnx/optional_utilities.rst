.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Optional Utilities
==================

Exporting PyTorch Models to ONNX
--------------------------------

.. note::
   Skip this step if you already have the ONNX format model.

For PyTorch models, it is recommended to use the TorchScript-based ONNX exporter for exporting ONNX models. Refer to the `PyTorch documentation for guidance <https://pytorch.org/docs/stable/onnx_torchscript.html#torchscript-based-onnx-exporter>`__.

Tips:
-----

1. Before exporting, perform `model.eval()`.
2. Models with opset 21 or higher are recommended.
3. NPU_CNN platforms do not support dynamic input shapes and allow only a batch size of 1. Ensure that the input shape is fixed and the batch dimension is set to 1.

Example code:

.. code-block:: python

   torch.onnx.export(
       model,
       input,
       model_output_path,
       opset_version=21,
       input_names=['input'],
       output_names=['output'],
   )

- **Opset Versions**: Models with opset 21 or higher are recommended. Models must use opset 10 or higher to be quantized. If models use an opset lower than 10, you should reconvert them to ONNX from their original framework using a later opset. Alternatively, refer to the usage of the version converter for the `ONNX Version Converter <https://github.com/onnx/onnx/blob/main/docs/VersionConverter.html>`__. Opset 10 does not support some node fusions and might not achieve the best performance. We recommend updating the model to opset 21 or higher for better performance. Moreover, per-channel quantization is supported for models using opset 13 or higher.

- **Large Models > 2GB**: Because of the 2 GB file size limit of Protobuf, additional data for ONNX models exceeding 2 GB is stored separately. Ensure that the ``.onnx`` file and the data files are placed in the same directory. Also, set the ``use_external_data_format`` parameter to ``True`` for large models when quantizing.


Pre-processing on the Float Model
---------------------------------

Pre-processing is the transformation of a float model to prepare it for quantization. It consists of the following three optional steps:

- **Symbolic shape inference**: This step is best suited for transformer models.
- **Model optimization**: This step uses the ONNX Runtime native library to rewrite the computation graph, including merging computation nodes and eliminating redundancies to improve runtime efficiency.
- **ONNX shape inference**.

The goal of these steps is to improve quantization quality. The ONNX Runtime quantization tool works best when the tensor's shape is known. Both symbolic shape inference and ONNX shape inference help determine tensor shapes. Symbolic shape inference works best with transformer-based models, and ONNX shape inference works with other models.

Model optimization performs certain operator fusions that make the quantization tool's job easier. For instance, a convolution operator followed by batch normalization can be fused into one during optimization, which can be quantized very efficiently.

Unfortunately, a known issue in ONNX Runtime is that model optimization cannot output a model size greater than 2 GB. Therefore, for large models, optimization must be skipped.

The pre-processing API is in the Python module ``onnxruntime.quantization.shape_inference``, function ``quant_pre_process()``.

.. code-block:: python

   from onnxruntime.quantization import shape_inference

   shape_inference.quant_pre_process(
       input_model_path: str,
       output_model_path: str,
       skip_optimization: bool = False,
       skip_onnx_shape: bool = False,
       skip_symbolic_shape: bool = False,
       auto_merge: bool = False,
       int_max: int = 2**31 - 1,
       guess_output_rank: bool = False,
       verbose: int = 0,
       save_as_external_data: bool = False,
       all_tensors_to_one_file: bool = False,
       external_data_location: str = "./",
       external_data_size_threshold: int = 1024,)

**Arguments**

- **input_model_path**: (String) This parameter specifies the file path of the input model that is to be pre-processed for quantization.
- **output_model_path**: (String) This parameter specifies the file path where the pre-processed model is saved.
- **skip_optimization**: (Boolean) This flag indicates whether to skip the model optimization step. If set to True, model optimization is skipped, which may cause ONNX shape inference failure for some models. The default value is False.
- **skip_onnx_shape**: (Boolean) This flag indicates whether to skip the ONNX shape inference step. The symbolic shape inference is most effective with transformer-based models. Skipping all shape inferences may reduce the effectiveness of quantization, as a tensor with an unknown shape cannot be quantized. The default value is False.
- **skip_symbolic_shape**: (Boolean) This flag indicates whether to skip the symbolic shape inference step. Symbolic shape inference is most effective with transformer-based models. Skipping all shape inferences may reduce the effectiveness of quantization, as a tensor with an unknown shape cannot be quantized. The default value is False.
- **auto_merge**: (Boolean) This flag determines whether to automatically merge symbolic dimensions when a conflict occurs during symbolic shape inference. The default value is False.
- **int_max**: (Integer) This parameter specifies the maximum integer value that is to be considered as boundless for operations like slice during symbolic shape inference. The default value is 2**31 - 1.
- **guess_output_rank**: (Boolean) This flag indicates whether to guess the output rank to be the same as input 0 for unknown operations. The default value is False.
- **verbose**: (Integer) This parameter controls the level of detailed information logged during inference. A value of 0 turns off logging, 1 logs warnings, and 3 logs detailed information. The default value is 0.
- **save_as_external_data**: (Boolean) This flag determines whether to save the ONNX model to external data. The default value is False.
- **all_tensors_to_one_file**: (Boolean) This flag indicates whether to save all the external data to one file. The default value is False.
- **external_data_location**: (String) This parameter specifies the file location where the external file is saved. The default value is "./".
- **external_data_size_threshold**: (Integer) This parameter specifies the size threshold for external data. The default value is 1024.

.. _caching-input-as-npy:

Caching Input as NPY
--------------------

This section explains how to perform inference with an ONNX model using floating-point inputs and save the input data in `.npy` format. This approach facilitates data storage and reuse and can serve as a **calibration dataset** during model quantization, provided that the data adequately reflects the typical distribution of the model inputs.

Through an example, we demonstrate how to define a simple dataset class (`InputDataset`), perform inference using an ONNX model, and save input data in `.npy` format to support subsequent model quantization.

Detailed Code
~~~~~~~~~~~~~

.. code-block:: python

    import onnxruntime as ort
    import numpy as np
    import os
    from torch.utils.data import Dataset, DataLoader


    # A simple dataset with two inputs (`input1`, `input2`) and random tensors.
    # Users can customize data generation to match their model's needs.
    class InputDataset(Dataset):
        def __init__(self, num_samples):
            super(InputDataset, self).__init__()
            self.num_samples = num_samples
            self.input1 = [np.random.rand(3, 224, 224).astype(np.float32) for _ in range(num_samples)]
            self.input2 = [np.random.rand(10).astype(np.float32) for _ in range(num_samples)]
            self.labels = [np.random.randint(0, 2) for _ in range(num_samples)]

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {
                "input1": self.input1[idx],
                "input2": self.input2[idx],
                "label": self.labels[idx]
            }


    dataset = InputDataset(num_samples=10)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    onnx_model_path = "path/to/your/float_model.onnx"
    session = ort.InferenceSession(onnx_model_path)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    enable_data_caching = True
    calibration_cache_dir = "calibration_data/"

    if enable_data_caching:
        for name in input_names:
            input_folder_path = os.path.join(calibration_cache_dir, name)
            os.makedirs(input_folder_path, exist_ok=True)

    for batch_idx, batch in enumerate(data_loader):
        input_feed = {}

        for name in input_names:
            input_data = batch[name].numpy()
            input_feed[name] = input_data

            # If `enable_data_caching` is True, save input data as `.npy` files by input name for each batch.
            if enable_data_caching:
                file_path = os.path.join(calibration_cache_dir, name, f"calib_{batch_idx+1:06d}.npy")
                np.save(file_path, input_data)
                print(f"Saved input data for {name} to {file_path}")

        outputs = session.run(output_names, input_feed)

        predictions = np.argmax(outputs[0], axis=1)

        print(f"Predictions for batch {batch_idx}: {predictions}")


The input data saved during ONNX inference can serve as a calibration dataset for model quantization. For instructions on how to use the saved NPY data as a calibration dataset, refer to :doc:`Calibration Data Path for AMD Quark Quantizer <./config/calibration_datasets>`. The output data format saved during inference is as follows:

For Single-Input Models
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

   calibration_data/
     calib_000001.npy
     calib_000002.npy
     calib_000003.npy
     calib_000004.npy
     calib_000005.npy
     ...

For Multi-Input Models
^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

   calibration_data/
     input1_name/
       calib_000001.npy
       calib_000002.npy
       calib_000003.npy
       calib_000004.npy
       calib_000005.npy
       ...
     input2_name/
       calib_000001.npy
       calib_000002.npy
       calib_000003.npy
       calib_000004.npy
       calib_000005.npy
       ...

Evaluating the Quantized Model
------------------------------

If you have scripts to evaluate float models, you can replace the float model file with the quantized model for evaluation.

If BFP/BF16/FP16/int32 data types are used in the quantized model, it is necessary to register the custom operations library to the ONNX Runtime inference session before evaluation. For example:

.. code-block:: python

   import onnxruntime as ort

   so = ort.SessionOptions()
   so.register_custom_ops_library(quark.onnx.get_library_path())
   session = ort.InferenceSession(quantized_model, so)

Dumping the Simulation Results
------------------------------

Sometimes after deploying the quantized model, it is necessary to compare the simulation results on the CPU/GPU and the output values on the DPU. You can use the ``dump_model`` of the AMD Quark ONNX API to dump the simulation results with the quantized_model. Currently, only the models containing FixNeuron nodes support this feature. For models using ``QuantFormat.QDQ``, you can set ``dump_float`` to True to save float data for all nodes' results.

.. code-block:: python

   # This function dumps the simulation results of the quantized model,
   # including weights and input_tensors results.
   quark.onnx.dump_model(
       model,
       dump_data_reader=None,
       random_data_reader_input_shape={},
       dump_float=False,
       output_dir='./dump_results',)

**Arguments**

- **model**: (String or ModelProto) This parameter specifies the file path of or the ModelProto object of the quantized model whose simulation results are to be dumped.
- **dump_data_reader**: (CalibrationDataReader or None) This parameter is a data reader that is used for the dumping process. The first batch is taken as input. If you wish to use random data for a quick test, you can set `dump_data_reader` to None. The default value is None.
- **random_data_reader_input_shape**: (Dict) It is required to use a dict {name: shape} to specify a certain input. For example, `RandomDataReaderInputShape={"image": [1, 3, 224, 224]}` for the input named "image". The default value is an empty dict {}.
- **dump_float**: (Boolean) This flag determines whether to dump the floating-point value of nodes' results. If set to True, the float values are dumped. Note that this may require a lot of storage space. The default value is False.
- **output_dir**: (String) This parameter specifies the directory where the dumped simulation results are saved. After successful execution of the function, dump results are generated in this specified directory. The default value is './dump_results'.

.. note::
   The `batch_size` of the `dump_data_reader` is better set to 1 for DPU debugging.

Dump results of each FixNeuron node (including weights and input_tensors) are generated in ``output_dir`` after the command is successfully executed.

For each quantized node, results are saved in \*.bin and \*.txt formats (\* represents the output name of the node). If ``dump_float`` is set to True, the output of all the nodes is saved in \*_float.bin and \*_float.txt (\* represents the output name of the node), which might require a lot of storage space.

Examples of dumping results are shown in the following table. Because of the storage path considerations, the '/' in the node name is replaced with '\_'.

Table 2. Example of Dumping Results

.. list-table::
   :header-rows: 1

   * - Quantized
     - Node Name
     - Saved Weights or Activations
     -
   * - Yes
     - /conv1/Conv_out
     - {output_dir}/dump_results/\_conv1_Conv_output_0_DequantizeLinear_Output.bin
     - {output_dir}/dump_results/\_conv1_Conv_output_0_DequantizeLinear_Output.txt
   * - Yes
     - onnx::Conv_501_DequantizeLinear
     - {output_dir}/dump_results/onnx::Conv_501_DequantizeLinear_Output.bin
     - {output_dir}/dump_results/onnx::Conv_501_DequantizeLinear_Output.txt
   * - No
     - /avgpool/GlobalAveragePool
     - {output_dir}/dump_results/\_avgpool_GlobalAveragePool_output_0_float.bin
     - {output_dir}/dump_results/\_avgpool_GlobalAveragePool_output_0_float.txt
