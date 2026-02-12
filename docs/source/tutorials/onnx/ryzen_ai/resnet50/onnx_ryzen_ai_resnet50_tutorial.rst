Quark ONNX Quantization Tutorial For Resnet50
=============================================

.. container:: alert alert-block alert-info

   NOTE This tutorial can be downloaded for local execution on a Jupyter
   Notebook environment. Click here to download the source file.

This topic outlines best practice for Post-Training Quantization (PTQ)
in Quark ONNX. It provides guidance on fine-tuning your quantization
strategy to meet target quantization accuracy.

The example has the following parts:

- Install requirements

- Prepare model

- Prepare data

- Quantizatize model With different configs

  - XINT8
  - A8W8
  - A16W8
  - BF16
  - BFP16
  - CLE
  - AdaRound
  - AdaQuant
  - Exclude Nodes

- Evaluate model

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
    !wget -O models/resnet50-v1-12.onnx https://github.com/onnx/models/raw/new-models/vision/classification/resnet/model/resnet50-v1-12.onnx

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
    import time
    
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
    
        subfolders = os.listdir(source_folder)
    
        for subfolder in subfolders:
            source_subfolder = os.path.join(source_folder, subfolder)
            destination_subfolder = os.path.join(calib_data_path, subfolder)
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

4) Quantization Procedure
-------------------------

First, create a data reader that gathers calibration statistics from the
target dataset. Next, inside quantize_model, constructing the quantized
model and passing in your configuration.

.. code:: ipython3

    # import copy
    
    # import cv2
    # import numpy as np
    # import onnx
    # import onnxruntime
    # import torch
    # import torchvision
    # from onnxruntime.quantization import CalibrationDataReader
    # from torchvision import transforms
    
    # from quark.onnx import (
    #     AdaQuantConfig,
    #     AdaRoundConfig,
    #     BFloat16Spec,
    #     BFP16Spec,
    #     CalibMethod,
    #     CLEConfig,
    #     Int8Spec,
    #     Int16Spec,
    #     ModelQuantizer,
    #     QConfig,
    #     QLayerConfig,
    #     XInt8Spec,
    # )
    
    
    # class ImageDataReader(CalibrationDataReader):
    #     def __init__(self, calibration_image_folder: str, input_name: str):
    #         self.enum_data = None
    #         self.input_name = input_name
    #         self.data_list = self._preprocess_images(calibration_image_folder)
    
    #     def _preprocess_images(self, image_folder: str):
    #         data_list = []
    #         img_names = [f for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".JPEG")]
    #         for name in img_names:
    #             input_image = cv2.imread(os.path.join(image_folder, name))
    #             # Resize the input image. Because the size of Resnet50 is 224.
    #             input_image = cv2.resize(input_image, (224, 224))
    #             input_data = np.array(input_image).astype(np.float32)
    #             # Customer Pre-Process
    #             input_data = input_data.transpose(2, 0, 1)
    #             input_size = input_data.shape
    #             if input_size[1] > input_size[2]:
    #                 input_data = input_data.transpose(0, 2, 1)
    #             input_data = np.expand_dims(input_data, axis=0)
    #             input_data = input_data / 255.0
    #             data_list.append(input_data)
    
    #         return data_list
    
    #     def get_next(self):
    #         if self.enum_data is None:
    #             self.enum_data = iter([{self.input_name: data} for data in self.data_list])
    #         return next(self.enum_data, None)
    
    #     def rewind(self):
    #         self.enum_data = None

.. code:: ipython3

    # from onnxruntime.quantization import CalibrationDataReader
    
    import copy
    
    import numpy as np
    import onnxruntime
    import torch
    import torchvision
    from onnxruntime.quantization.calibrate import CalibrationDataReader
    from PIL import Image
    from torchvision import transforms
    
    from quark.onnx import (
        AdaQuantConfig,
        AdaRoundConfig,
        BFloat16Spec,
        BFP16Spec,
        CalibMethod,
        CLEConfig,
        Int8Spec,
        Int16Spec,
        ModelQuantizer,
        QConfig,
        QLayerConfig,
        XInt8Spec,
    )
    from quark.onnx.operators.custom_ops import get_library_path
    
    # def get_model_input_name(input_model_path: str) -> str:
    #     model = onnx.load(input_model_path)
    #     model_input_name = model.graph.input[0].name
    #     return model_input_name
    
    
    def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0, batch_size=100):
        """
        Loads a batch of images and preprocess them
        parameter images_folder: path to folder storing images
        parameter height: image height in pixels
        parameter width: image width in pixels
        parameter size_limit: number of images to load. Default is 0 which means all images are picked.
        return: list of matrices characterizing multiple images
        """
        image_path = os.listdir(images_folder)
        image_names = []
        for image_dir in image_path:
            image_name = os.listdir(os.path.join(images_folder, image_dir))
            image_names.append(os.path.join(image_dir, image_name[0]))
        if size_limit > 0 and len(image_names) >= size_limit:
            batch_filenames = [image_names[i] for i in range(size_limit)]
        else:
            batch_filenames = image_names
        unconcatenated_batch_data = []
    
        batch_data = []
        for index, image_name in enumerate(batch_filenames):
            image_filepath = images_folder + "/" + image_name
            pillow_img = Image.new("RGB", (width, height))
            pillow_img.paste(Image.open(image_filepath).resize((width, height)))
            image_array = np.array(pillow_img) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            image_array = image_array - mean
            std = np.array([0.229, 0.224, 0.225])
            nchw_data = image_array / std
            nchw_data = nchw_data.transpose((2, 0, 1))
            nchw_data = np.expand_dims(nchw_data, axis=0)
            nchw_data = nchw_data.astype(np.float32)
            unconcatenated_batch_data.append(nchw_data)
    
            if (index + 1) % batch_size == 0:
                one_batch_data = np.concatenate(unconcatenated_batch_data, axis=0)
                unconcatenated_batch_data.clear()
                batch_data.append(one_batch_data)
    
        return batch_data
    
    
    class ImageDataReader(CalibrationDataReader):
        def __init__(self, calibration_image_folder: str, model_path: str, data_size: int, batch_size: int):
            self.enum_data = None
    
            # Use inference session to get input shape.
            session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            (_, _, height, width) = session.get_inputs()[0].shape
    
            # Convert image to input data
            self.nhwc_data_list = _preprocess_images(calibration_image_folder, height, width, data_size, batch_size)
            self.input_name = session.get_inputs()[0].name
            self.datasize = len(self.nhwc_data_list)
    
        def get_next(self):
            if self.enum_data is None:
                self.enum_data = iter([{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list])
            return next(self.enum_data, None)
    
        def rewind(self):
            self.enum_data = None
    
        def reset(self):
            self.enum_data = None

As we plan to experiment with various methods and combinations, it is
more efficient to organize them into groups and reference them by their
labels when needed.

.. code:: ipython3

    activation_spec_dict = {
        "XINT8": XInt8Spec(),
        "A8W8": Int8Spec(),
        "A16W8": Int16Spec(),
        "BF16": BFloat16Spec(),
        "BFP16": BFP16Spec(),
    }
    weight_spec_dict = {
        "XINT8": XInt8Spec(),
        "A8W8": Int8Spec(),
        "A16W8": Int8Spec(),
        "BF16": BFloat16Spec(),
        "BFP16": BFP16Spec(),
    }
    
    
    DEFAULT_ADAROUND_PARAMS = {
        "DataSize": 1000,
        "FixedSeed": 1705472343,
        "BatchSize": 2,
        "NumIterations": 1000,
        "LearningRate": 0.1,
        "OptimAlgorithm": "adaround",
        "OptimDevice": "cpu",
        "InferDevice": "cpu",
        "EarlyStop": True,
    }
    
    DEFAULT_ADAQUANT_PARAMS = {
        "DataSize": 1000,
        "FixedSeed": 1705472343,
        "BatchSize": 2,
        "NumIterations": 1000,
        "LearningRate": 0.00001,
        "OptimAlgorithm": "adaquant",
        "OptimDevice": "cpu",
        "InferDevice": "cpu",
        "EarlyStop": True,
    }

Define a function to perform quantization based on your config

.. code:: ipython3

    def quantize_model(args):
        activation_spec = activation_spec_dict[args["config"]]
        weight_spec = weight_spec_dict[args["config"]]
        algo_confs = [CLEConfig()]
    
        exclude_info = []
        if args.get("exclude_nodes"):
            exclude_nodes = args["exclude_nodes"].split(";")
            exclude_nodes = [node_name.strip() for node_name in exclude_nodes]
            exclude_info += exclude_nodes
    
        if args["config"] == "XINT8":
            enable_npucnn = True
        else:
            enable_npucnn = False
            activation_spec.set_calibration_method(CalibMethod.MinMax)
            weight_spec.set_calibration_method(CalibMethod.MinMax)
    
        if args.get("learning_rate"):
            lr = args["learning_rate"]
        else:
            if args.get("adround"):
                lr = DEFAULT_ADAROUND_PARAMS["LearningRate"]
            elif args.get("adaquant"):
                lr = DEFAULT_ADAQUANT_PARAMS["LearningRate"]
    
        if args.get("num_iters"):
            num_iter = args["num_iters"]
        else:
            if args.get("adround"):
                num_iter = DEFAULT_ADAROUND_PARAMS["NumIterations"]
            elif args.get("adaquant"):
                num_iter = DEFAULT_ADAQUANT_PARAMS["NumIterations"]
    
        if args.get("adaround"):
            adaround_algo = AdaRoundConfig(
                data_size=DEFAULT_ADAROUND_PARAMS["DataSize"],
                batch_size=DEFAULT_ADAROUND_PARAMS["BatchSize"],
                num_iterations=num_iter,
                learning_rate=lr,
                early_stop=DEFAULT_ADAROUND_PARAMS["EarlyStop"],
            )
            algo_confs.append(adaround_algo)
    
        if args.get("adaquant"):
            adaround_algo = AdaQuantConfig(
                data_size=DEFAULT_ADAQUANT_PARAMS["DataSize"],
                batch_size=DEFAULT_ADAQUANT_PARAMS["BatchSize"],
                num_iterations=num_iter,
                learning_rate=lr,
                early_stop=DEFAULT_ADAQUANT_PARAMS["EarlyStop"],
            )
            algo_confs.append(adaround_algo)
    
        extra_info = {}
        if args["config"] == "A8W8":
            extra_info = {"EnableNPUCnn": enable_npucnn, "AlignSlice": False, "FoldRelu": True, "AlignConcat": True}
        if args["config"] == "A16W8":
            extra_info = {
                "EnableNPUCnn": enable_npucnn,
                "AlignSlice": False,
                "FoldRelu": True,
                "AlignConcat": True,
                "AlignEltwiseQuantType": True,
            }
        if args["config"] == "BF16":
            extra_info = {"EnableNPUCnn": enable_npucnn, "QuantizeAllOpTypes": True, "ForceQuantizeNoInputCheck": True}
    
        quant_config = QConfig(
            global_config=QLayerConfig(activation=activation_spec, weight=weight_spec),
            algo_config=algo_confs,
            use_external_data_format=args.get("save_as_external_data"),
            exclude=exclude_info,
            **extra_info,
        )
    
        # model_input_name = get_model_input_name(args["input_model_path"])
        # calib_datareader = ImageDataReader(args["calib_data_path"], model_input_name)
        calib_datareader = ImageDataReader(
            args["calib_data_path"], args["input_model_path"], args["num_calib_data"], args["batch_size"]
        )
        quantizer = ModelQuantizer(quant_config)
        quantizer.quantize_model(args["input_model_path"], args["output_model_path"], calib_datareader)

Create a basic configuration, then modify it based on each requirement

.. code:: ipython3

    quant_config = {
        "input_model_path": "models/resnet50-v1-12.onnx",
        "calib_data_path": calib_data_path,
        "num_calib_data": 1000,
        "batch_size": 1,
        "num_iters": 1,
        "device": "cpu",
    }

Run XINT8

.. code:: ipython3

    quant_config_with_xint8 = copy.deepcopy(quant_config)
    quant_config_with_xint8["output_model_path"] = "models/resnet152_xint8_quantized.onnx"
    quant_config_with_xint8["config"] = "XINT8"
    
    quantize_model(quant_config_with_xint8)

Run A8W8

.. code:: ipython3

    quant_config_with_a8w8 = copy.deepcopy(quant_config)
    quant_config_with_a8w8["output_model_path"] = "models/resnet152_a8w8_quantized.onnx"
    quant_config_with_a8w8["config"] = "A8W8"
    
    quantize_model(quant_config_with_a8w8)

Run A16W8

.. code:: ipython3

    quant_config_with_a16w8 = copy.deepcopy(quant_config)
    quant_config_with_a16w8["output_model_path"] = "models/resnet152_a16w8_quantized.onnx"
    quant_config_with_a16w8["config"] = "A16W8"
    
    quantize_model(quant_config_with_a16w8)

Run BF16

.. code:: ipython3

    quant_config_with_bf16 = copy.deepcopy(quant_config)
    quant_config_with_bf16["output_model_path"] = "models/resnet152_bf16_quantized.onnx"
    quant_config_with_bf16["config"] = "BF16"
    
    quantize_model(quant_config_with_bf16)

Run XINT8 with CLE

.. code:: ipython3

    quant_config_with_xint8_cle = copy.deepcopy(quant_config)
    quant_config_with_xint8_cle["output_model_path"] = "models/resnet152_xint8_cle_quantized.onnx"
    quant_config_with_xint8_cle["config"] = "XINT8"
    quant_config_with_xint8_cle["cle"] = True
    
    quantize_model(quant_config_with_xint8_cle)

Run XINT8 with AdaRound

.. code:: ipython3

    quant_config_with_xint8_adaround = copy.deepcopy(quant_config)
    quant_config_with_xint8_adaround["output_model_path"] = "models/resnet152_xint8_adaround_quantized.onnx"
    quant_config_with_xint8_adaround["config"] = "XINT8"
    quant_config_with_xint8_adaround["adaround"] = True
    quant_config_with_xint8_adaround["learning_rate"] = 0.1
    quant_config_with_xint8_adaround["num_iters"] = 3000
    
    quantize_model(quant_config_with_xint8_adaround)

Run XINT8 with AdaQuant

.. code:: ipython3

    quant_config_with_xint8_adaquant = copy.deepcopy(quant_config)
    quant_config_with_xint8_adaquant["output_model_path"] = "models/resnet152_xint8_adaquant_quantized.onnx"
    quant_config_with_xint8_adaquant["config"] = "XINT8"
    quant_config_with_xint8_adaquant["adaquant"] = True
    quant_config_with_xint8_adaquant["learning_rate"] = 0.00001
    quant_config_with_xint8_adaquant["num_iters"] = 10000
    
    quantize_model(quant_config_with_xint8_adaquant)

Run XINT8 with Exclude Nodes

.. code:: ipython3

    quant_config_with_xint8_exclude_nodes = copy.deepcopy(quant_config)
    quant_config_with_xint8_exclude_nodes["output_model_path"] = "models/resnet152_xint8_exclude_nodes_quantized.onnx"
    quant_config_with_xint8_exclude_nodes["config"] = "XINT8"
    quant_config_with_xint8_exclude_nodes["exclude_nodes"] = "resnetv17_conv0_fwd; resnetv17_stage1_conv0_fwd"
    
    quantize_model(quant_config_with_xint8_exclude_nodes)

5) Evaluation and Expected Results
----------------------------------

Evaluation is performed on the ImageNet validation set. We compare three
models — (1) full-precision, (2) quantized with minmax calibration, and
(3) quantized with layerwise calibration — to assess layerwise
calibration’s effectiveness. The full-precision model serves as the
baseline for measuring any accuracy change caused by quantization.

ImageNet has 1,000 classes, so we report both Prec@1 and Prec@5 to
capture strict and relaxed accuracy. Both metrics are reported as
percentages (higher is better). Prec@1 shows exact single-label
correctness; Prec@5 is useful on large, fine-grained label spaces
because it captures near-misses where the correct class is among the
model’s top candidates.

.. code:: ipython3

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
    
    
    def load_loader(data_dir, batch_size, workers):
        data_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        dataset = torchvision.datasets.ImageFolder(data_dir, data_transform)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
        )
        return data_loader
    
    
    def accuracy_np(output, target):
        max_indices = np.argsort(output, axis=1)[:, ::-1]
        top5 = 100 * np.equal(max_indices[:, :5], target[:, np.newaxis]).sum(axis=1).mean()
        top1 = 100 * np.equal(max_indices[:, 0], target).mean()
        return top1, top5
    
    
    def metrics(onnx_model_path, sess_options, providers, data_loader, print_freq):
        session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=providers)
        input_name = session.get_inputs()[0].name
    
        batch_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # run the net and return prediction
            output = session.run([], {input_name: input.data.numpy()})
            output = output[0]
    
            # measure accuracy and record loss
            prec1, prec5 = accuracy_np(output, target.numpy())
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % print_freq == 0:
                print(
                    f"Test: [{i}/{len(data_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {input.size(0) / batch_time.avg:.3f}/s, "
                    f"{100 * batch_time.avg / input.size(0):.3f} ms/sample) \t"
                    f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})"
                )
    
        return top1, top5
    
    
    def evaluate(args: dict):
        args["gpu_id"] = 0
    
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
    
        if args.get("onnx_input"):
            val_loader = load_loader(args["data"], args["batch_size"], args["workers"])
            f_top1, f_top5 = metrics(args["onnx_input"], sess_options, providers, val_loader, args["print_freq"])
            print(f" * Prec@1 {f_top1.avg:.3f} ({100 - f_top1.avg:.3f}) Prec@5 {f_top5.avg:.3f} ({100.0 - f_top5.avg:.3f})")
        elif args.get("onnx_float") and args.get("onnx_quant"):
            val_loader = load_loader(args["data"], args["batch_size"], args["workers"])
            f_top1, f_top5 = metrics(args["onnx_float"], sess_options, providers, val_loader, args["print_freq"])
            f_top1 = format(f_top1.avg, ".2f")
            f_top5 = format(f_top5.avg, ".2f")
    
            q_top1, q_top5 = metrics(args["onnx_quant"], sess_options, providers, val_loader, args["print_freq"])
            q_top1 = format(q_top1.avg, ".2f")
            q_top5 = format(q_top5.avg, ".2f")
    
            f_size = format(os.path.getsize(args["onnx_float"]) / (1024 * 1024), ".2f")
            q_size = format(os.path.getsize(args["onnx_quant"]) / (1024 * 1024), ".2f")
            """
            --------------------------------------------------------
            |             | float model    | quantized model |
            --------------------------------------------------------
            | ****        | ****           | ****             |
            --------------------------------------------------------
            | Model Size  | ****           | ****             |
            --------------------------------------------------------
            """
            from rich.console import Console
            from rich.table import Table
    
            console = Console()
    
            table = Table()
            table.add_column("")
            table.add_column("Float Model")
            table.add_column("Quantized Model", style="bold green1")
    
            table.add_row("Model", args["onnx_float"], args["onnx_quant"])
            table.add_row("Model Size", str(f_size) + " MB", str(q_size) + " MB")
            table.add_row("Prec@1", str(f_top1) + " %", str(q_top1) + " %")
            table.add_row("Prec@5", str(f_top5) + " %", str(q_top5) + " %")
    
            console.print(table)
    
        else:
            print("Please specify both model-float and model-quant or model-input for evaluation.")

First, define an evaluation config, and record accuracy of the Full
Precision model on ImageNet val dataset

.. code:: ipython3

    eval_config = {
        "data": source_folder,
        "batch_size": 100,
        "workers": 1,
        "gpu": False,
        "print_freq": 1000,
    }

.. code:: ipython3

    full_precision_eval_config = copy.deepcopy(eval_config)
    full_precision_eval_config["onnx_input"] = "models/resnet50-v1-12.onnx"
    
    evaluate(full_precision_eval_config)

We have quantized several models, but to keep this tutorial simple, we
will focus on evaluating the A8W8 model as an example. If you are
interested in exploring other quantization configurations, you can
follow the provided code and run your own experiments. For now, let’s
specify the path to the A8W8 quantized model and measure its accuracy on
the ImageNet validation dataset.

.. code:: ipython3

    quant_eval_config = copy.deepcopy(eval_config)
    quant_eval_config["onnx_input"] = "models/resnet152_a8w8_quantized.onnx"
    
    evaluate(quant_eval_config)

The following table contains the expected results, but please note that
different machines can lead to minor variations in the accuracy of
quantized model.

========== =========== ===============
\          Float Model Quantized Model
========== =========== ===============
Model Size 97.82 MB    25.62 MB
Prec@1     74.114 %    73.562 %
Prec@5     91.716 %    91.420 %
========== =========== ===============
