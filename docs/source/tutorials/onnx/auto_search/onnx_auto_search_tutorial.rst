Quark ONNX Quantization Tutorial For Auto Search
================================================

.. container:: alert alert-block alert-info

   NOTE This tutorial can be downloaded for local execution on a Jupyter
   Notebook environment. Click here to download the source file.

In this tutorial, you will learn how to use the AMD Quark Auto Search
module to automatically find optimal quantization configurations for a
YOLOv8 model. Quantization is a key step in optimizing deep learning
models for efficient inference, reducing model size and improving
performance without significantly impacting accuracy.

Using Quark’s Auto Search feature, you can streamline the quantization
process by letting the tool explore different parameter combinations and
select the best configuration automatically. This tutorial provides a
step-by-step guide on setting up the environment, preparing the YOLOv8
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

2) Prepare YOLO Model
---------------------

We are using YOLOv8-n model from Ultralytics to demonstrate Quark’s auto
search process. First, we need to download the model, then convert it to
onnx format.

.. code:: ipython3

    from ultralytics import YOLO
    
    model = YOLO("yolov8n.pt")
    model.export(format="onnx")

3) Prepare data
---------------

The COCO (Common Objects in Context) dataset is a large-scale benchmark
designed for computer vision tasks such as object detection,
segmentation, keypoint detection, and image captioning. It contains over
330,000 images, with more than 200,000 labeled images and around 1.5
million object instances spanning 80 object categories.

What makes COCO unique is its focus on objects in complex, real-world
scenes, where multiple objects often appear together in varied contexts.
Each image is richly annotated with detailed instance segmentation
masks, bounding boxes, and object category labels, making COCO an
essential dataset for training and evaluating deep learning models in
visual recognition tasks.

Due to its diversity and annotation quality, COCO has become a standard
benchmark for evaluating the performance of modern models.

If you do not have a COCO dataset in your local machine, you can
download from COCO’s official website:
https://cocodataset.org/#download, then decompress it.

.. code:: ipython3

    !wget http://images.cocodataset.org/zips/val2017.zip
    !unzip val2017.zip

If you have a local cache to store the dataset, you can use and
environment variable like ``LOCAL_DATA_CACHE`` to specify its path. This
is useful to organize and store all your datasets for different
experiments in a central place. Otherwise, the current folder is used.

.. code:: ipython3

    import os
    import sys
    
    calibration_dataset_path = "val2017"
    
    if os.environ.get("LOCAL_DATA_CACHE") is not None:
        data_path = os.environ["LOCAL_DATA_CACHE"]
        calibration_dataset_path = os.path.join(data_path, "coco/2017/val2017")
    
    if not os.path.exists(calibration_dataset_path):
        print("The provided data path does not exist.")
        sys.exit(1)

The COCO dataset should be structured as follows:

- val2017

  - sample_1.jpg
  - sample_2.jpg
  - …

4) Auto Search Pipeline
-----------------------

The following cell defines an auto search config settings. You can
customize the serch space to suit your requirements.

.. code:: ipython3

    import copy
    import os
    
    import cv2
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization.calibrate import CalibrationMethod
    from onnxruntime.quantization.quant_utils import QuantType
    
    from quark.onnx import Config, ExtendedQuantFormat, ExtendedQuantType, LayerWiseMethod, PowerOfTwoMethod
    from quark.onnx.quantization import auto_search
    from quark.onnx.quantization.config import get_default_config
    
    
    class AutoSearchConfig_Default:
        # for s8s8 & s16s8 aaws/asws
        search_space: dict[str, any] = {
            "calibrate_method": [
                CalibrationMethod.MinMax,
                CalibrationMethod.Percentile,
                LayerWiseMethod.LayerWisePercentile,
            ],
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
    
        search_space_advanced: dict[str, any] = {
            "calibrate_method": [
                CalibrationMethod.MinMax,
                CalibrationMethod.Percentile,
                LayerWiseMethod.LayerWisePercentile,
            ],
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
                "FastFinetune": {
                    "DataSize": [
                        200,
                    ],
                    "NumIterations": [1000, 5000, 10000],
                    "OptimAlgorithm": ["adaround"],
                    "LearningRate": [0.1, 0.01],
                    # 'OptimDevice': ['cuda:0'],
                    # 'InferDevice': ['cuda:0'],
                    "EarlyStop": [False],
                },
            },
        }
    
        search_space_advanced2: dict[str, any] = {
            "calibrate_method": [
                CalibrationMethod.MinMax,
                CalibrationMethod.Percentile,
                LayerWiseMethod.LayerWisePercentile,
            ],
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
                "FastFinetune": {
                    "DataSize": [
                        200,
                    ],
                    "NumIterations": [1000, 5000, 10000],
                    "OptimAlgorithm": ["adaquant"],
                    "LearningRate": [1e-5, 1e-6],
                    # 'OptimDevice': ['cuda:0'],
                    # 'InferDevice': ['cuda:0'],
                    "EarlyStop": [False],
                },
            },
        }
    
        # for XINT8
        search_space_XINT8: dict[str, any] = {
            "calibrate_method": [PowerOfTwoMethod.MinMSE],
            "activation_type": [QuantType.QUInt8],
            "weight_type": [
                QuantType.QInt8,
            ],
            "enable_npu_cnn": [True],
            "include_cle": [False],
            "include_fast_ft": [False],
            "extra_options": {
                "ActivationSymmetric": [True],
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
                    # 'OptimDevice': ['cuda:0'],
                    # 'InferDevice': ['cuda:0'],
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
                    # 'OptimDevice': ['cuda:0'],
                    # 'InferDevice': ['cuda:0'],
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
                    # 'OptimDevice': ['cuda:0'],
                    # 'InferDevice': ['cuda:0'],
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
                    # 'OptimDevice': ['cuda:0'],
                    # 'InferDevice': ['cuda:0'],
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

Now, let’s define an image reader for COCO dataset

.. code:: ipython3

    class ImageDataReader:
        def __init__(self, model_path: str, calibration_image_folder: str):
            self.enum_data = None
            self.data_list = self._preprocess_images(calibration_image_folder)
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = session.get_inputs()[0].name
    
            self.datasize = len(self.data_list)
    
        def _preprocess_images(self, image_folder: str):
            data_list = []
            img_names = [
                f for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".JPEG")
            ]
            for name in img_names:
                input_image = cv2.imread(os.path.join(image_folder, name))
                input_image = cv2.resize(input_image, (640, 640))
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
            return {self.input_name: self.data_list[idx]}
    
        def __len__(
            self,
        ):
            return self.datasize
    
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
search spaces, but we have defined 10 more spaces in the
auto_search_model.py. You are welcome to test all of them or define your
own search spaces based on their needs.

.. code:: ipython3

    input_model_path = "./yolov8n.onnx"
    
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
        calibration_data_reader=ImageDataReader(
            input_model_path,
            calibration_dataset_path,
        ),
    )
    
    # build search space
    # To reduce computational load for this demo, we have commented out the other predefined search spaces. Users are welcome to modify them based on their needs
    
    # fixed point
    # space1 = auto_search_ins.build_all_configs(auto_search_config.search_space_XINT8)
    # space2 = auto_search_ins.build_all_configs(auto_search_config.search_space)
    # space3 = auto_search_ins.build_all_configs(auto_search_config.search_space_XINT8_advanced)
    # space4 = auto_search_ins.build_all_configs(auto_search_config.search_space_XINT8_advanced2)
    space5 = auto_search_ins.build_all_configs(auto_search_config.search_space_advanced)
    space6 = auto_search_ins.build_all_configs(auto_search_config.search_space_advanced2)
    
    # bf16 and bfp16
    # space7 = auto_search_ins.build_all_configs(auto_search_config.search_space_bf16)
    # space8 = auto_search_ins.build_all_configs(auto_search_config.search_space_bfp16)
    # space9 = auto_search_ins.build_all_configs(auto_search_config.search_space_bf16_advanced)
    # space10 = auto_search_ins.build_all_configs(auto_search_config.search_space_bfp16_advanced)
    # auto_search_ins.all_configs = space1 + space2 + space3 + space4 + space5 + space6 + space7 + space8 + space9 + space10
    auto_search_ins.all_configs = space5 + space6
    
    # Execute the auto search process
    auto_search_ins.search_model()

The overall runtime of the AMD Quark Auto Search module varies depending
on model complexity, dataset size, and available compute resources. Upon
completion, the module automatically generates a comprehensive log file
summarizing all evaluated configurations. These results are ranked
according to the optimization criteria you specified.
