Quark ONNX Quantization Tutorial For AdaQuant
=============================================

.. container:: alert alert-block alert-info

   NOTE This tutorial can be downloaded for local execution on a Jupyter
   Notebook environment. Click here to download the source file.

In this tutorial, you will learn how to use AMD Quark to quantize image
classification models from the Hugging Face timm library using three
different quantization configurations: XINT8, A8W8, and A16W8.

Quantization is a powerful optimization technique that reduces model
precision to improve inference speed and reduce memory footprint while
maintaining model accuracy. The tutorial provides a step-by-step guide
on setting up the environment, loading pretrained timm models, and
applying AMD Quark’s quantization workflow.

By the end of this tutorial, you will understand the trade-offs between
XINT8, A8W8, and A16W8 quantization schemes, and learn how to select the
best configuration for achieving optimal performance and accuracy to
suit your needs.

The example has the following parts:

- Install requirements

- Prepare model

- Prepare data

- Quantizatize model

- Evaluate Models

- Apply More Models

1) Install The Necessary Python Packages:
-----------------------------------------

In addition to Quark that must be installed as
`documented <https://quark.docs.amd.com/latest/install.html>`__ at here,
extra packages are require for this tutorial.

.. code:: ipython3

    %pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    %pip install amd-quark
    %pip install -r ./requirements.txt

2) Export ONNX Model From mobilenetv2_100.ra_in1k Model.
--------------------------------------------------------

You don’t need to download this model manually. If you’re curious about
its source, the corresponding model link is:
https://huggingface.co/timm/mobilenetv2_100.ra_in1k

Before exporting, let’s create a directory for models:

.. code:: ipython3

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="")
    parser.add_argument("--workspace_dir", default="")
    args, _ = parser.parse_known_args()
    model_name = args.model_name if args.model_name != "" else "mobilenetv2_100.ra_in1k"
    workspace_dir = args.workspace_dir
    print("Using model:", model_name)
    print("Working root dir: ", workspace_dir)

.. code:: ipython3

    !mkdir -p models

.. code:: ipython3

    import os
    import shutil
    
    import timm
    import torch
    
    
    def timm_to_onnx(model_name):
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
    
    
    timm_to_onnx(model_name)

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
    from datetime import datetime
    
    start_exe_time = datetime.now().replace(second=0, microsecond=0)
    
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

4) Quantization Procedure
-------------------------

First, create a data reader that gathers calibration statistics from the
target dataset. Next, inside quantize_model, construct the quantized
model and pass in your configuration.

.. code:: ipython3

    import numpy as np
    import onnxruntime
    import torchvision
    from timm.data import resolve_data_config
    from timm.models import create_model
    from torchvision import transforms
    
    
    def load_loader(model_name, data_dir, batch_size, workers):
        timm_model = create_model(
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

.. code:: ipython3

    import copy
    
    from quark.onnx import A8W8_QCONFIG, A16W8_QCONFIG, XINT8_QCONFIG, ModelQuantizer
    
    
    class CalibrationDataReader:
        def __init__(self, dataloader):
            super().__init__()
            self.iterator = iter(dataloader)
    
        def get_next(self) -> dict:
            try:
                return {"input": next(self.iterator)[0].numpy()}
            except Exception:
                return None
    
    
    def quantize_model(args: dict) -> None:
        # `dr` (Data Reader) is an instance of ResNet50DataReader, which is a utility class that
        # reads the calibration dataset and prepares it for the quantization process.
        if args["calibration_dataset_path"] == "":
            dr = None
        else:
            data_loader = load_loader(
                args["model_name"], args["calibration_dataset_path"], args["batch_size"], args["workers"]
            )
            dr = CalibrationDataReader(data_loader)
    
        # Get quantization configuration
        if args["config"] == "XINT8":
            config = XINT8_QCONFIG
        elif args["config"] == "A8W8":
            config = A8W8_QCONFIG
        elif args["config"] == "A16W8":
            config = A16W8_QCONFIG
        else:
            raise ValueError(
                "No config is specified in the default quantize_model function. Please adjust it to fit your need."
            )
    
        print(f"The configuration for quantization is {config}")
    
        # Create an ONNX quantizer
        quantizer = ModelQuantizer(config)
    
        # Quantize the ONNX model
        quantizer.quantize_model(args["input_model_path"], args["output_model_path"], dr)

The cell defines a basic config inputs without specifying the
quantization yet

.. code:: ipython3

    quant_config = {
        "model_name": model_name,
        "input_model_path": f"models/{model_name}.onnx",
        "output_model_path": f"models/{model_name}_quantized.onnx",
        "calibration_dataset_path": calib_data_path,
        "batch_size": 1,
        "workers": 1,
    }

Quantize with XINT8 config

.. code:: ipython3

    quant_config_with_xint8 = copy.deepcopy(quant_config)
    quant_config_with_xint8["config"] = "XINT8"
    quant_config_with_xint8["output_model_path"] = f"models/{model_name}_xint8.onnx"
    
    quantize_model(quant_config_with_xint8)

Quantize with A8W8 config

.. code:: ipython3

    quant_config_with_a8w8 = copy.deepcopy(quant_config)
    quant_config_with_a8w8["config"] = "A8W8"
    quant_config_with_a8w8["output_model_path"] = f"models/{model_name}_a8w8.onnx"
    
    quantize_model(quant_config_with_a8w8)

Quantize with A16W8 config

.. code:: ipython3

    quant_config_with_a16w8 = copy.deepcopy(quant_config)
    quant_config_with_a16w8["config"] = "A16W8"
    quant_config_with_a16w8["output_model_path"] = f"models/{model_name}_a16w8.onnx"
    
    quantize_model(quant_config_with_a16w8)

5) Evaluation and Expected Results
----------------------------------

Evaluation is performed on the ImageNet validation set. We compare three
models — (1) full-precision, (2) quantized without AdaQuant, and (3)
quantized with AdaQuant — to assess AdaQuant’s effectiveness. The
full-precision model serves as the baseline for measuring any accuracy
change caused by quantization.

ImageNet has 1,000 classes, so we report both Prec@1 and Prec@5 to
capture strict and relaxed accuracy. Both metrics are reported as
percentages (higher is better). Prec@1 shows exact single-label
correctness; Prec@5 is useful on large, fine-grained label spaces
because it captures near-misses where the correct class is among the
model’s top candidates.

.. code:: ipython3

    import time
    
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
    
        val_loader = load_loader(args["model_name"], args["data"], args["batch_size"], args["workers"])
        f_top1, f_top5 = metrics(args["onnx_input"], sess_options, providers, val_loader, args["print_freq"])
        print(f" * Prec@1 {f_top1.avg:.3f} ({100 - f_top1.avg:.3f}) Prec@5 {f_top5.avg:.3f} ({100.0 - f_top5.avg:.3f})")
        return round(f_top1.avg, 3), round(f_top5.avg, 3)

Let’s define a basic evaluation config inputs, then record accuracy of
each method on ImageNet val dataset

.. code:: ipython3

    eval_config = {
        "data": source_folder,
        "model_name": model_name,
        "batch_size": 1,
        "workers": 1,
        "gpu": False,
        "print_freq": 1000,
    }

Evaluate on the full-precision modeland set it as a measurement base

.. code:: ipython3

    full_precision_eval_config = copy.deepcopy(eval_config)
    full_precision_eval_config["onnx_input"] = f"models/{model_name}.onnx"
    
    top1, top5 = evaluate(full_precision_eval_config)

Evaluate on XINT8 quantized model

.. code:: ipython3

    eval_config_xint8 = copy.deepcopy(eval_config)
    eval_config_xint8["onnx_input"] = f"models/{model_name}_xint8.onnx"
    
    top1, top5 = evaluate(eval_config_xint8)

Evaluate on A8W8 quantized model

.. code:: ipython3

    eval_config_a8w8 = copy.deepcopy(eval_config)
    eval_config_a8w8["onnx_input"] = f"models/{model_name}_a8w8.onnx"
    
    top1, top5 = evaluate(eval_config_a8w8)

Evaluate on A16W8 quantized model

.. code:: ipython3

    eval_config_a16w8 = copy.deepcopy(eval_config)
    eval_config_a16w8["onnx_input"] = f"models/{model_name}_a16w8.onnx"
    
    top1, top5 = evaluate(eval_config_a16w8)

The following table contains the expected results, but please note that
different machines can lead to minor variations in the accuracy of
quantized model with AdaQuant.

+--------+--------+------------------+-----------------+------------------+
|        | Float  | Quantized Model  | Quantized Model | Quantized Model  |
|        | Model  | with XINT8       | with A8W8       | with A16W8       |
+========+========+==================+=================+==================+
| Model  | 15 MB  | 4.0 MB           | 4.0 MB          | 4.0 MB           |
| Size   |        |                  |                 |                  |
+--------+--------+------------------+-----------------+------------------+
| Prec@1 | 72.890 | 66.640 %         | 70.486 %        | 70.562 %         |
|        | %      |                  |                 |                  |
+--------+--------+------------------+-----------------+------------------+
| Prec@5 | 90.996 | 87.122 %         | 89.614 %        | 89.586 %         |
|        | %      |                  |                 |                  |
+--------+--------+------------------+-----------------+------------------+

.. code:: ipython3

    # This cell should have the remove-input tag as we don't want it rendered in the documentation
    # it's creating results for submission to the dashboard
    
    import json
    
    import pandas as pd
    
    benchmarks = {
        "resnet101d.gluon_in1k": {"acc1": 79.428, "acc5": 94.56},
        "vgg19.tv_in1k": {"acc1": 72.108, "acc5": 90.736},
        "mobilenetv2_100.ra_in1k": {"acc1": 72.704, "acc5": 90.894},
        "densenet169.tv_in1k": {"acc1": 72.506, "acc5": 91.126},
        "botnet26t_256.c1_in1k": {"acc1": 78.828, "acc5": 94.382},
        "cs3darknet_focus_l.c2ns_in1k": {"acc1": 75.576, "acc5": 93.016},
        "cspdarknet53.ra_in1k": {"acc1": 79.664, "acc5": 94.824},
        "cspresnext50.ra_in1k": {"acc1": 80.062, "acc5": 95.048},
        "dla102.in1k": {"acc1": 77.304, "acc5": 93.562},
        "dla60_res2net.in1k": {"acc1": 77.924, "acc5": 93.996},
        "dla60_res2next.in1k": {"acc1": 77.778, "acc5": 93.898},
        "eca_nfnet_l0.ra2_in1k": {"acc1": 81.974, "acc5": 96.072},
        "ecaresnet50d.miil_in1k": {"acc1": 80.582, "acc5": 95.452},
        "ecaresnetlight.miil_in1k": {"acc1": 80.522, "acc5": 95.39},
        "lambda_resnet26rpt_256.c1_in1k": {"acc1": 78.234, "acc5": 94.18},
        "nf_resnet50.ra2_in1k": {"acc1": 79.916, "acc5": 95.05},
        "nfnet_l0.ra2_in1k": {"acc1": 82.46, "acc5": 96.346},
        "regnety_004.pycls_in1k": {"acc1": 72.9, "acc5": 91.094},
        "wide_resnet50_2.racm_in1k": {"acc1": 81.948, "acc5": 95.972},
        "cspresnet50": {"acc1": 78.778, "acc5": 94.256},
        "efficientnet_el_pruned.in1k": {"acc1": 79.752, "acc5": 95.022},
        "efficientnet_el.ra_in1k": {"acc1": 80.954, "acc5": 95.42},
        "efficientnet_em.ra2_in1k": {"acc1": 78.978, "acc5": 94.64},
        "inception_v3": {"acc1": 77.182, "acc5": 93.364},
        "mnasnet_100.rmsp_in1k": {"acc1": 73.512, "acc5": 91.514},
        "regnetx_032.tv2_in1k": {"acc1": 78.862, "acc5": 94.408},
        "resnet34d": {"acc1": 78.246, "acc5": 94.178},
        "resnet50d": {"acc1": 81.08, "acc5": 95.592},
        "resnet101d": {"acc1": 79.622, "acc5": 94.862},
        "resnext101_64x4d": {"acc1": 81.014, "acc5": 95.14},
        "resnetv2_101.a1h_in1k": {"acc1": 75.232, "acc5": 92.85},
        "resnet152d": {"acc1": 81.544, "acc5": 95.86},
        "resnet200d": {"acc1": 82.308, "acc5": 96.154},
        "tv_resnet50": {"acc1": 75.646, "acc5": 92.67},
        "tv_resnext50_32x4d": {"acc1": 77.222, "acc5": 93.55},
        "vgg16": {"acc1": 71.27, "acc5": 90.296},
        "xception": {"acc1": 70.536, "acc5": 90.058},
        "mixnet_l.ft_in1k": {"acc1": 67.22, "acc5": 88.112},
        "ese_vovnet19b_dw": {"acc1": 76.986, "acc5": 93.568},
        "dpn107": {"acc1": 79.002, "acc5": 94.438},
        "coat_lite_mini": {"acc1": 76.018, "acc5": 92.74},
        "adv_inception_v3": {"acc1": 71.44, "acc5": 90.152},
    }
    
    
    def get_golden_val(benchmarks_dict: dict, model_name: str):
        golden_df = pd.DataFrame.from_dict(benchmarks_dict, orient="index")
        acc1 = golden_df.at[model_name, "acc1"]
        acc5 = golden_df.at[model_name, "acc5"]
        benchmark = {"Prec@1": acc1, "Prec@5": acc5}
        return benchmark
    
    
    def compare_result(golden: float, current: float):
        result = current - golden
        diff = "no change"
        if golden > current:
            diff = f"drops by {result:.3f}"
        elif golden < current:
            diff = f"rises by {result:.3f}"
        return diff
    
    
    def get_output(model_name: str, benchmark: dict[str, float], current: dict[str, float]):
        try:
            top1, top5 = current["Prec@1"], current["Prec@5"]
            diff1 = compare_result(benchmark["Prec@1"], top1)
            diff5 = compare_result(benchmark["Prec@5"], top5)
        except NameError:
            top1, top5 = "ERROR", "ERROR"
            diff1, diff5 = "ERROR", "ERROR"
        prec1_meta = {"golden": benchmark["Prec@1"], "difference": diff1}
        prec5_meta = {"golden": benchmark["Prec@1"], "difference": diff5}
        result = {
            "version": 1,
            "data": [
                {
                    "experiment": {
                        "name": model_name,
                        "model": model_name.split(".")[0].title(),
                        "timestamp": str(start_exe_time),
                        "settings": {
                            "quantization_scheme": "A8W8",
                            "quantization_algorithm": "",
                        },
                    },
                    "metrics": [
                        {
                            "name": "Prec@1",
                            "value": top1,
                            "metadata": str(prec1_meta),
                        },
                        {
                            "name": "Prec@5",
                            "value": top5,
                            "metadata": str(prec5_meta),
                        },
                    ],
                    "metadata": {"duration": str(datetime.now().replace(second=0, microsecond=0) - start_exe_time)},
                }
            ],
        }
        return result
    
    
    benchmark = get_golden_val(benchmarks, model_name)
    current = {"Prec@1": top1, "Prec@5": top5}
    output = get_output(model_name, benchmark, current)
    
    root_path = f"{workspace_dir}/logs_regression_test/"
    os.makedirs(root_path, exist_ok=True)
    file_path = f"{root_path}/huggingface_{model_name}.json"
    
    with open(file_path, "w") as json_file:
        json.dump(output, json_file)
