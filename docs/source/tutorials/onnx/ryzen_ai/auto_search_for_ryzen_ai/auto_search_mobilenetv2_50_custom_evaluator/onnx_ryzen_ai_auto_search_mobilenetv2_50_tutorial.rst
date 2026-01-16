Quark ONNX Quantization Tutorial For Auto Search
================================================

In this tutorial, you will learn how to use the AMD Quark Auto Search
module to automatically find optimal quantization configurations for a
mobilenetv2_050 model. Quantization is a key step in optimizing deep
learning models for efficient inference, reducing model size and
improving performance without significantly impacting accuracy.

Using Quark’s Auto Search feature, you can streamline the quantization
process by letting the tool explore different parameter combinations and
select the best configuration automatically. This tutorial provides a
step-by-step guide on setting up the environment, preparing the
mobilenetv2_050 model, running the Auto Search process, and evaluating
the quantized model’s performance.

The example has the following parts:

- Install requirements

- Prepare model

- Prepare data

- Customize evaluator

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

2) Export ONNX Model From mobilenetv2_050.lamb_in1k Torch Model.
----------------------------------------------------------------

You don’t need to download this model manually. If you’re curious about
its source, the corresponding model link is:
https://huggingface.co/timm/mobilenetv2_050.lamb_in1k

Before exporting, let’s create a directory for models:

.. code:: ipython3

    !mkdir -p models

.. code:: ipython3

    import os
    import shutil
    
    import timm
    import torch
    from timm.data import resolve_data_config
    
    model_name = "mobilenetv2_050.lamb_in1k"
    
    model = timm.create_model(model_name, pretrained=True)
    model = model.eval()
    device = torch.device("cpu")
    
    data_config = timm.data.resolve_model_data_config(
        model=model,
        use_test_size=True,
    )
    
    batch_size = 1
    torch.manual_seed(42)
    dummy_input = torch.randn((batch_size,) + tuple(data_config["input_size"])).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        "models/" + model_name + ".onnx",
        export_params=True,
        do_constant_folding=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=False,
        dynamo=False,
    )
    print("Onnx model is saved at models/" + model_name + ".onnx")

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

    import sys
    
    import torch
    
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
    import time
    
    import numpy as np
    import onnxruntime
    import torchvision
    from onnxruntime.quantization.calibrate import CalibrationMethod
    from onnxruntime.quantization.quant_utils import QuantType
    from torchvision import transforms
    
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

    class CalibrationDataReader:
        def __init__(self, dataloader):
            super().__init__()
            self.iterator = iter(dataloader)
    
        def get_next(self) -> dict:
            try:
                return {"input": next(self.iterator)[0].numpy()}
            except Exception:
                return None
    
    
    def load_loader(model_name, data_dir, batch_size, workers):
        timm_model = timm.create_model(
            model_name,
            pretrained=False,
        )
        data_config = resolve_data_config(model=timm_model, use_test_size=True)
        crop_pct = data_config["crop_pct"]
        input_size = data_config["input_size"]
        width = input_size[-1]
        data_transform = transforms.Compose(
            [
                transforms.Resize(int(width / crop_pct)),
                transforms.CenterCrop(width),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        dataset = torchvision.datasets.ImageFolder(data_dir, data_transform)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
        )
        return data_loader

The following cell demonstrates how to customize an evaluator. ImageNet
has 1,000 classes, so normally we report both Prec@1 and Prec@5 to
capture strict and relaxed accuracy. Both metrics are reported as
percentages (higher is better). Prec@1 shows exact single-label
correctness; Prec@5 is useful on large, fine-grained label spaces
because it captures near-misses where the correct class is among the
model’s top candidates. However, we need to limit to one evaluation
metric to determine if pre-defined criteria is met. Therefore, this
evaluator only uses Prec@1 metric.

.. code:: ipython3

    from quark.onnx.operators.custom_ops import get_library_path
    
    
    class AverageMeter:
        """Computes and stores the average and current value"""
    
        def __init__(self):
            self.reset()
    
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
    
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    
    
    def accuracy_np(output, target):
        max_indices = np.argsort(output, axis=1)[:, ::-1]
        top1 = 100 * np.equal(max_indices[:, 0], target).mean()
        return top1
    
    
    def metrics(onnx_model_path, sess_options, providers, data_loader):
        session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=providers)
        input_name = session.get_inputs()[0].name
    
        batch_time = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # run the net and return prediction
            output = session.run([], {input_name: input.data.numpy()})
            output = output[0]
    
            # measure accuracy and record loss
            prec1 = accuracy_np(output, target.numpy())
            top1.update(prec1.item(), input.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
        return top1
    
    
    def evaluator(model_path):
        # model_path = "models/mobilenetv2_050.lamb_in1k.onnx"
        args = {
            "data": source_folder,
            "model_name": "mobilenetv2_050.lamb_in1k",
            "batch_size": 1,
            "workers": 2,
            "gpu": False,
            "print_freq": 1000,
        }
        # Set graph optimization level
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        if args.get("profile"):
            sess_options.enable_profiling = True
        if args.get("onnx_output_opt"):
            sess_options.optimized_model_filepath = args["onnx_output_opt"]
        if args.get("gpu"):
            if "ROCMExecutionProvider" in onnxruntime.get_available_providers():
                device = "ROCM"
                providers = ["ROCMExecutionProvider"]
            elif "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                device = "CUDA"
                providers = ["CUDAExecutionProvider"]
            else:
                device = "CPU"
                providers = ["CPUExecutionProvider"]
                print("Warning: GPU is not available, use CPU instead.")
        else:
            device = "CPU"
            providers = ["CPUExecutionProvider"]
        sess_options.register_custom_ops_library(get_library_path(device))
    
        val_loader = load_loader(args["model_name"], args["data"], args["batch_size"], args["workers"])
        top1 = metrics(model_path, sess_options, providers, val_loader)
        return top1

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

    data_loader = load_loader("mobilenetv2_050.lamb_in1k", source_folder, 1, 1)
    
    auto_search_config = AutoSearchConfig_Default()
    auto_search_config.search_evaluator = evaluator
    
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
        model_input="models/mobilenetv2_050.lamb_in1k.onnx",
        calibration_data_reader=CalibrationDataReader(data_loader),
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
