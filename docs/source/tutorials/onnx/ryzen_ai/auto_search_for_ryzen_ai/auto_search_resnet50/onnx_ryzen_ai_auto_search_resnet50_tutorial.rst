Quark ONNX Quantization Tutorial For Auto Search
================================================

.. container:: alert alert-block alert-info

   NOTE This tutorial can be downloaded for local execution on a Jupyter
   Notebook environment. Click here to download the source file.

In this tutorial, you will learn how to use the AMD Quark Auto Search
module to automatically find optimal quantization configurations for a
resnet50 model. Quantization is a key step in optimizing deep learning
models for efficient inference, reducing model size and improving
performance without significantly impacting accuracy.

Using Quark’s Auto Search feature, you can streamline the quantization
process by letting the tool explore different parameter combinations and
select the best configuration automatically. This tutorial provides a
step-by-step guide on setting up the environment, preparing the resnet50
model, running the Auto Search process, and evaluating the quantized
model’s performance.

The example has the following parts:

- Install requirements

- Prepare model

- Prepare data

- Run auto search

1) Install The Necessary Python Packages:
-----------------------------------------

In addition to Quark that must be installed as
`documented <https://quark.docs.amd.com/latest/install.html>`__ at here,
extra packages are require for this tutorial.

.. code:: ipython3

    %pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    %pip install amd-quark
    %pip install -r ./requirements.txt

2) Download resnet50-v1-12 Model
--------------------------------

The model is publicly available and can be downloaded from this link:
https://github.com/onnx/models/blob/new-models/vision/classification/resnet/model/resnet50-v1-12.onnx

.. code:: ipython3

    !mkdir -p models
    !wget -O models/resnet50-v1-12.onnx https://raw.githubusercontent.com/onnx/models/new-models/vision/classification/resnet/model/resnet50-v1-12.onnx

3) Prepare data
---------------

ILSVRC 2012, commonly known as ‘ImageNet’. This dataset provides access
to ImageNet (ILSVRC) 2012 which is the most commonly used subset of
ImageNet. This dataset spans 1000 object classes and contains 50,000
validation images.

If you already have an ImageNet datasets, you can directly use your
dataset path.

To prepare the test data, please check the download section of the main
website: https://huggingface.co/datasets/imagenet-1k/tree/main/data. You
need to register and download **val_images.tar.gz** to the current
directory.

Then, create a val_data folder and decompress the .gz file to the
folder.

.. code:: ipython3

    !mkdir -p val_data && tar -xzf val_images.tar.gz -C val_data

If you have a local cache to store the dataset, you can use and
environment variable like ``LOCAL_DATA_CACHE`` to specify its path. This
is useful to organize and store all your datasets for different
experiments in a central place. Otherwise, the current folder is used,
and validation dataset and calibration dataset will be created under
current directory.

.. code:: ipython3

    import os
    import shutil
    import sys
    
    source_folder = "val_data"
    calib_data_path = "calib_data"
    
    if os.environ.get("LOCAL_DATA_CACHE") is not None:
        data_path = os.environ["LOCAL_DATA_CACHE"]
        source_folder = os.path.join(data_path, "Imagenet/val")
        calib_data_path = os.path.join(data_path, "Imagenet/calib_100")
    else:
        files = os.listdir(source_folder)
    
        for filename in files:
            if not filename.startswith("ILSVRC2012_val_") or not filename.endswith(".JPEG"):
                continue
    
            n_identifier = filename.split("_")[-1].split(".")[0]
            folder_name = n_identifier
            folder_path = os.path.join(source_folder, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(source_folder, filename)
            destination = os.path.join(folder_path, filename)
            shutil.move(file_path, destination)
    
        print("File organization complete.")
    
        if not os.path.exists(calib_data_path):
            os.makedirs(calib_data_path)
    
        destination_folder = calib_data_path
    
        subfolders = os.listdir(source_folder)
    
        for subfolder in subfolders:
            source_subfolder = os.path.join(source_folder, subfolder)
            destination_subfolder = os.path.join(destination_folder, subfolder)
            os.makedirs(destination_subfolder, exist_ok=True)
            files = os.listdir(source_subfolder)
    
            if files:
                file_to_copy = files[0]
                source_file = os.path.join(source_subfolder, file_to_copy)
                destination_file = os.path.join(destination_subfolder, file_to_copy)
    
                shutil.copy(source_file, destination_file)
    
        print("Creating calibration dataset complete.")
    
    if not os.path.exists(source_folder):
        print("The provided data path does not exist.")
        sys.exit(1)

The storage format of the val_data of the ImageNet dataset organized as
follows:

- val_data

  - n01440764

    - ILSVRC2012_val_00000293.JPEG
    - ILSVRC2012_val_00002138.JPEG
    - …

  - n01443537

    - ILSVRC2012_val_00000236.JPEG
    - ILSVRC2012_val_00000262.JPEG
    - …

  - …

The storage format of the calib_data of the ImageNet dataset organized
as follows:

- calib_data

  - n01440764

    - ILSVRC2012_val_00000293.JPEG

  - n01443537

    - ILSVRC2012_val_00000236.JPEG

  - …

4) Auto Search Pipeline
-----------------------

The following cell defines an auto search config settings. You can
customize the serch space to suit your requirements.

**Search Tolerance Setting**

The search tolerance is the acceptable margin between the accuracy of
the original floating-point model and the quantized model. When the
quantized model’s accuracy loss exceeds the set tolerance, the Auto
Search framework will stop further searches.

- **Tolerance Threshold**: This is a value representing the maximum
  acceptable accuracy drop from the floating-point model.
- **Auto-Stop Condition**: When the search reaches a configuration with
  accuracy loss below the tolerance threshold, the framework will halt,
  saving the best configuration and corresponding quantized model.

Example: If the floating-point model has 95% accuracy and the tolerance
is set to 1%, the Auto Search will stop if a configuration causes an
accuracy drop greater than 1% (i.e., below 94%).

.. code:: ipython3

    import copy
    import os
    
    import cv2
    import numpy as np
    import onnx
    from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod
    from onnxruntime.quantization.quant_utils import QuantType
    
    from quark.onnx import Config, ExtendedQuantFormat, ExtendedQuantType, PowerOfTwoMethod
    from quark.onnx.quantization import auto_search
    from quark.onnx.quantization.config import get_default_config
    
    
    class AutoSearchConfig_Default:
        # for s8s8 & s16s8 aaws/asws
        search_space: dict[str, any] = {
            "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
            "activation_type": [
                QuantType.QInt8,
                QuantType.QInt16,
            ],
            "weight_type": [
                QuantType.QInt8,
            ],
            "include_cle": [False],
            "include_fast_ft": [False],
            "extra_options": {
                "ActivationSymmetric": [True, False],
                "WeightSymmetric": [True],
                "CalibMovingAverage": [False, True],
                "CalibMovingAverageConstant": [0.01],
            },
        }
    
        # for s8s8 aaws/asws
        search_space_s8s8: dict[str, any] = {
            "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
            "activation_type": [
                QuantType.QInt8,
            ],
            "weight_type": [
                QuantType.QInt8,
            ],
            "include_cle": [False],
            "include_fast_ft": [False],
            "extra_options": {
                "ActivationSymmetric": [True, False],
                "WeightSymmetric": [True],
                "CalibMovingAverage": [False, True],
                "CalibMovingAverageConstant": [0.01],
                "AlignSlice": [False],
                "FoldRelu": [True],
                "AlignConcat": [True],
            },
        }
    
        search_space_s8s8_advanced: dict[str, any] = {
            "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
            "activation_type": [QuantType.QInt8],
            "weight_type": [
                QuantType.QInt8,
            ],
            "include_cle": [False, True],
            "include_fast_ft": [False, True],
            "extra_options": {
                "ActivationSymmetric": [True, False],
                "WeightSymmetric": [True],
                "CalibMovingAverage": [
                    False,
                    True,
                ],
                "CalibMovingAverageConstant": [0.01],
                "AlignSlice": [False],
                "FoldRelu": [True],
                "AlignConcat": [True],
                "FastFinetune": {
                    "DataSize": [
                        200,
                    ],
                    "NumIterations": [1000],
                    "OptimAlgorithm": ["adaround"],
                    "LearningRate": [0.1],
                    "OptimDevice": ["cuda:0"],
                    "InferDevice": ["cuda:0"],
                    "EarlyStop": [False],
                },
            },
        }
    
        search_space_s8s8_advanced2: dict[str, any] = {
            "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
            "activation_type": [
                QuantType.QInt8,
                QuantType.QInt16,
            ],
            "weight_type": [
                QuantType.QInt8,
            ],
            "include_cle": [False, True],
            "include_fast_ft": [False, True],
            "extra_options": {
                "ActivationSymmetric": [True, False],
                "WeightSymmetric": [True],
                "CalibMovingAverage": [
                    False,
                    True,
                ],
                "CalibMovingAverageConstant": [0.01],
                "AlignSlice": [False],
                "FoldRelu": [True],
                "AlignConcat": [True],
                "FastFinetune": {
                    "DataSize": [
                        200,
                    ],
                    "NumIterations": [5000],
                    "OptimAlgorithm": ["adaquant"],
                    "LearningRate": [
                        1e-5,
                    ],
                    "OptimDevice": ["cuda:0"],
                    "InferDevice": ["cuda:0"],
                    "EarlyStop": [False],
                },
            },
        }
    
        # for s16s8 aaws/asws
        search_space_s16s8: dict[str, any] = {
            "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
            "activation_type": [
                QuantType.QInt16,
            ],
            "weight_type": [
                QuantType.QInt8,
            ],
            "include_cle": [False],
            "include_fast_ft": [False],
            "extra_options": {
                "ActivationSymmetric": [True, False],
                "WeightSymmetric": [True],
                "CalibMovingAverage": [False, True],
                "CalibMovingAverageConstant": [0.01],
                "AlignSlice": [False],
                "FoldRelu": [True],
                "AlignConcat": [True],
                "AlignEltwiseQuantType": [True],
            },
        }
    
        search_space_s16s8_advanced: dict[str, any] = {
            "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
            "activation_type": [
                QuantType.QInt16,
            ],
            "weight_type": [
                QuantType.QInt8,
            ],
            "include_cle": [False, True],
            "include_fast_ft": [False, True],
            "extra_options": {
                "ActivationSymmetric": [True, False],
                "WeightSymmetric": [True],
                "CalibMovingAverage": [
                    False,
                    True,
                ],
                "CalibMovingAverageConstant": [0.01],
                "AlignSlice": [False],
                "FoldRelu": [True],
                "AlignConcat": [True],
                "AlignEltwiseQuantType": [True],
                "FastFinetune": {
                    "DataSize": [
                        200,
                    ],
                    "NumIterations": [1000],
                    "OptimAlgorithm": ["adaround"],
                    "LearningRate": [0.1],
                    "OptimDevice": ["cuda:0"],
                    "InferDevice": ["cuda:0"],
                    "EarlyStop": [False],
                },
            },
        }
    
        search_space_s16s8_advanced2: dict[str, any] = {
            "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
            "activation_type": [
                QuantType.QInt16,
            ],
            "weight_type": [
                QuantType.QInt8,
            ],
            "include_cle": [False, True],
            "include_fast_ft": [False, True],
            "extra_options": {
                "ActivationSymmetric": [True, False],
                "WeightSymmetric": [True],
                "CalibMovingAverage": [
                    False,
                    True,
                ],
                "CalibMovingAverageConstant": [0.01],
                "AlignSlice": [False],
                "FoldRelu": [True],
                "AlignConcat": [True],
                "AlignEltwiseQuantType": [True],
                "FastFinetune": {
                    "DataSize": [
                        200,
                    ],
                    "NumIterations": [5000],
                    "OptimAlgorithm": ["adaquant"],
                    "LearningRate": [
                        1e-5,
                    ],
                    "OptimDevice": ["cuda:0"],
                    "InferDevice": ["cuda:0"],
                    "EarlyStop": [False],
                },
            },
        }
    
        # for XINT8
        search_space_XINT8: dict[str, any] = {
            "calibrate_method": [PowerOfTwoMethod.MinMSE],
            "activation_type": [
                QuantType.QUInt8,
            ],
            "weight_type": [
                QuantType.QInt8,
            ],
            "enable_npu_cnn": [True],
            "include_cle": [False],
            "include_fast_ft": [False],
            "extra_options": {
                "ActivationSymmetric": [
                    True,
                ],
            },
        }
    
        search_space_XINT8_advanced: dict[str, any] = {
            "calibrate_method": [PowerOfTwoMethod.MinMSE],
            "activation_type": [
                QuantType.QUInt8,
            ],
            "weight_type": [
                QuantType.QInt8,
            ],
            "enable_npu_cnn": [True],
            "include_cle": [False, True],
            "include_fast_ft": [True],
            "extra_options": {
                "ActivationSymmetric": [
                    True,
                ],
                "WeightSymmetric": [True],
                "CalibMovingAverage": [
                    False,
                    True,
                ],
                "CalibMovingAverageConstant": [0.01],
                "FastFinetune": {
                    "DataSize": [
                        200,
                    ],
                    "NumIterations": [1000],
                    "OptimAlgorithm": ["adaround"],
                    "LearningRate": [
                        0.1,
                    ],
                    "OptimDevice": ["cuda:0"],
                    "InferDevice": ["cuda:0"],
                    "EarlyStop": [False],
                },
            },
        }
    
        search_space_XINT8_advanced2: dict[str, any] = {
            "calibrate_method": [PowerOfTwoMethod.MinMSE],
            "activation_type": [
                QuantType.QUInt8,
            ],
            "weight_type": [
                QuantType.QInt8,
            ],
            "enable_npu_cnn": [True],
            "include_cle": [False, True],
            "include_fast_ft": [True],
            "extra_options": {
                "ActivationSymmetric": [
                    True,
                ],
                "WeightSymmetric": [True],
                "CalibMovingAverage": [
                    False,
                    True,
                ],
                "CalibMovingAverageConstant": [0.01],
                "FastFinetune": {
                    "DataSize": [
                        200,
                    ],
                    "NumIterations": [5000],
                    "OptimAlgorithm": ["adaquant"],
                    "LearningRate": [
                        1e-5,
                    ],
                    "OptimDevice": ["cuda:0"],
                    "InferDevice": ["cuda:0"],
                    "EarlyStop": [False],
                },
            },
        }
    
        # for BF16
        search_space_bf16: dict[str, any] = {
            "calibrate_method": [CalibrationMethod.MinMax],
            "activation_type": [ExtendedQuantType.QBFloat16],
            "weight_type": [ExtendedQuantType.QBFloat16],
            "quant_format": [ExtendedQuantFormat.QDQ],
            "include_cle": [False],
            "include_fast_ft": [False],
        }
    
        search_space_bf16_advanced: dict[str, any] = {
            "calibrate_method": [CalibrationMethod.MinMax],
            "activation_type": [ExtendedQuantType.QBFloat16],
            "weight_type": [ExtendedQuantType.QBFloat16],
            "quant_format": [ExtendedQuantFormat.QDQ],
            "include_cle": [False],
            "include_fast_ft": [True],
            "extra_options": {
                "FastFinetune": {
                    "DataSize": [1000],
                    "FixedSeed": [1705472343],
                    "BatchSize": [2],
                    "NumIterations": [1000],
                    "LearningRate": [0.00001],
                    "OptimAlgorithm": ["adaquant"],
                    "OptimDevice": ["cuda:0"],
                    "InferDevice": ["cuda:0"],
                    "EarlyStop": [False],
                }
            },
        }
    
        #  for BFP16
        search_space_bfp16: dict[str, any] = {
            "calibrate_method": [CalibrationMethod.MinMax],
            "activation_type": [ExtendedQuantType.QBFP],
            "weight_type": [ExtendedQuantType.QBFP],
            "quant_format": [ExtendedQuantFormat.QDQ],
            "include_cle": [False],
            "include_fast_ft": [False],
            "extra_options": {
                "BFPAttributes": [
                    {
                        "bfp_method": "to_bfp",
                        "axis": 1,
                        "bit_width": 16,
                        "block_size": 8,
                        "rounding_mode": 2,
                    }
                ]
            },
        }
    
        search_space_bfp16_advanced: dict[str, any] = {
            "calibrate_method": [CalibrationMethod.MinMax],
            "activation_type": [ExtendedQuantType.QBFP],
            "weight_type": [ExtendedQuantType.QBFP],
            "quant_format": [ExtendedQuantFormat.QDQ],
            "include_cle": [False],
            "include_fast_ft": [True],
            "extra_options": {
                "BFPAttributes": [
                    {
                        "bfp_method": "to_bfp",
                        "axis": 1,
                        "bit_width": 16,
                        "block_size": 8,
                        "rounding_mode": 2,
                    }
                ],
                "FastFinetune": {
                    "DataSize": [1000],
                    "FixedSeed": [1705472343],
                    "BatchSize": [2],
                    "NumIterations": [1000],
                    "LearningRate": [0.00001],
                    "OptimAlgorithm": ["adaquant"],
                    "OptimDevice": ["cuda:0"],
                    "InferDevice": ["cuda:0"],
                    "EarlyStop": [False],
                },
            },
        }
    
        search_metric: str = "L2"
        search_algo: str = "grid_search"  # candidates: "grid_search", "random"
        search_evaluator = None
        search_metric_tolerance: float = 0.60001
        search_cache_dir: str = "./"
        search_output_dir: str = "./"
        search_log_path: str = "./auto_search.log"
    
        search_stop_condition: dict[str, any] = {
            "find_n_candidates": 1,
            "iteration_limit": 10000,
            "time_limit": 1000000.0,  # unit: second
        }

Now, let’s define an image reader for Imagenet dataset

.. code:: ipython3

    def get_model_input_name(input_model_path: str) -> str:
        model = onnx.load(input_model_path)
        model_input_name = model.graph.input[0].name
        return model_input_name
    
    
    class ImageDataReader(CalibrationDataReader):
        def __init__(self, calibration_image_folder: str, input_name: str):
            self.enum_data = None
    
            self.input_name = input_name
    
            self.data_list = self._preprocess_images(calibration_image_folder)
    
        def _preprocess_images(self, image_folder: str):
            data_list = []
            img_names = [f for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")]
            for name in img_names:
                input_image = cv2.imread(os.path.join(image_folder, name))
                # Resize the input image. Because the size of Resnet50 is 224.
                input_image = cv2.resize(input_image, (224, 224))
                input_data = np.array(input_image).astype(np.float32)
                # Customer Pre-Process
                input_data = input_data.transpose(2, 0, 1)
                input_size = input_data.shape
                if input_size[1] > input_size[2]:
                    input_data = input_data.transpose(0, 2, 1)
                input_data = np.expand_dims(input_data, axis=0)
                input_data = input_data / 255.0
                data_list.append(input_data)
    
            return data_list
    
        def get_next(self):
            if self.enum_data is None:
                self.enum_data = iter([{self.input_name: data} for data in self.data_list])
            return next(self.enum_data, None)
    
        def __getitem__(self, idx):
            return self.data_list[idx]
    
        def rewind(self):
            self.enum_data = None
    
        def reset(self):
            self.enum_data = None

The following command generates a series of configurations from the
auto_search settings. As long as the stop condition remains false, the
instance samples configurations from the entire search space according
to the selected search algorithm. Each sampled configuration is then
used to quantize the input model with Quark ONNX. The evaluator computes
the chosen metric on the quantized model and checks whether it falls
within the defined tolerance. Models that meet the tolerance are added
to the output dictionary; those that don’t are discarded.

To reduce computational load for this demo, we only set to run two
search spaces, but we have defined 13 more spaces in the
auto_search_model.py. You are welcome to test all of them or define your
own search spaces based on their needs.

.. code:: ipython3

    input_model_path = "models/resnet50-v1-12.onnx"
    
    model_input_name = get_model_input_name(input_model_path)
    auto_search_config = AutoSearchConfig_Default()
    
    # Get quantization configuration
    quant_config = get_default_config("S8S8_AAWS")
    config_copy = copy.deepcopy(quant_config)
    config_copy.calibrate_method = CalibrationMethod.MinMax
    config = Config(global_quant_config=config_copy)
    print(f"The configuration for quantization is {config}")
    
    # Create auto search instance
    auto_search_ins = auto_search.AutoSearch(
        config=config,
        auto_search_config=auto_search_config,
        model_input=input_model_path,
        calibration_data_reader=ImageDataReader(calib_data_path, model_input_name),
    )
    
    # build search space
    # To reduce computational load for this demo, we have commented out the other predefined search spaces. Users are welcome to modify them based on their needs
    
    # fixed point
    space1 = auto_search_ins.build_all_configs(auto_search_config.search_space_XINT8)
    space2 = auto_search_ins.build_all_configs(auto_search_config.search_space_s8s8)
    space3 = auto_search_ins.build_all_configs(auto_search_config.search_space_s16s8)
    space4 = auto_search_ins.build_all_configs(auto_search_config.search_space_XINT8_advanced)
    space5 = auto_search_ins.build_all_configs(auto_search_config.search_space_XINT8_advanced2)
    space6 = auto_search_ins.build_all_configs(auto_search_config.search_space_s8s8_advanced)
    space7 = auto_search_ins.build_all_configs(auto_search_config.search_space_s8s8_advanced2)
    space8 = auto_search_ins.build_all_configs(auto_search_config.search_space_s16s8_advanced)
    space9 = auto_search_ins.build_all_configs(auto_search_config.search_space_s16s8_advanced2)
    # bf16 and bfp16
    space10 = auto_search_ins.build_all_configs(auto_search_config.search_space_bf16)
    space11 = auto_search_ins.build_all_configs(auto_search_config.search_space_bfp16)
    space12 = auto_search_ins.build_all_configs(auto_search_config.search_space_bf16_advanced)
    space13 = auto_search_ins.build_all_configs(auto_search_config.search_space_bfp16_advanced)
    
    # auto_search_ins.all_configs = space1 + space2 + space3 + space4 + space5 + space6 + space7 + space8 + space9 + space10 + space11 + space12 + space13
    auto_search_ins.all_configs = space1 + space2
    
    # Excute the auto search process
    auto_search_ins.search_model()

The overall runtime of the AMD Quark Auto Search module varies depending
on model complexity, dataset size, and available compute resources. Upon
completion, the module automatically generates a comprehensive log file
summarizing all evaluated configurations. These results are ranked
according to the optimization criteria you specified.
