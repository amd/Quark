Quark ONNX Quantization Tutorial For Mixed Precision
====================================================

Mixed precision quantization involves using different precision levels
for different parts of a neural network, such as using 8-bit integers
for some layers while retaining higher precision, for example, 16-bit or
32-bit floating point, for others. This approach leverages the fact that
not all parts of a model are equally sensitive to quantization. By
carefully selecting which parts of the model can tolerate lower
precision, you achieve significant computational savings while
minimizing the impact on model accuracy.

The example has the following parts:

- Install requirements

- Prepare model

- Prepare data

- Quantizatize without Mixed Precision

- Quantizatize with Mixed Precision

- Evaluate Models

1) Install The Necessary Python Packages:
-----------------------------------------

In addition to Quark that must be installed as
`documented <https://quark.docs.amd.com/latest/install.html>`__ at here,
extra packages are require for this tutorial.

.. code:: ipython3

    %pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    %pip install amd-quark
    %pip install -r ./requirements.txt

2) Export ONNX Model From Densenet121.ra_in1k Model.
----------------------------------------------------

You don’t need to download this model manually. If you’re curious about
its source, the corresponding model link is:
https://huggingface.co/timm/densenet121.ra_in1k

Before exporting, let’s create a directory for models:

.. code:: ipython3

    !mkdir -p models

.. code:: ipython3

    import os
    import shutil
    
    import timm
    import torch
    
    global model_name
    model_name = "densenet121.ra_in1k"
    
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
    
    global calib_data_path
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
model and pass in your configuration. This example uses MinMax
calibration and INT8 quantization for both weights and activations;
then, it enables the mixed precision procedure to reduce accuracy
degradation during quantization.

.. code:: ipython3

    import numpy as np
    import onnxruntime
    import torch
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
    from typing import Any, Union
    
    import numpy
    import torch
    from timm.data import create_dataset, create_loader
    
    from quark.onnx import AutoMixprecisionConfig, Int8, Int8Spec, Int16Spec, ModelQuantizer, QConfig, QLayerConfig
    
    
    def post_process_top1(output: torch.tensor) -> float:
        _, preds_top1 = torch.max(output, 1)
        return preds_top1
    
    
    def getAccuracy_top1(preds: Union[torch.tensor, list], targets: Union[torch.tensor, list]) -> float:
        assert len(preds) == len(targets)
        assert len(preds) > 0
        count = 0
        for i in range(len(preds)):
            pred = preds[i]
            target = targets[i]
            if pred == target:
                count += 1
        return count / len(preds)
    
    
    def top1_acc(results: list[Union[torch.tensor, list[Any]]]) -> float:
        """
        Calculate the top1 accuracy of the model.
        :param results: the result of the model
        :return: the top1 accuracy
        """
        timm_model_name = model_name
    
        timm_model = create_model(
            timm_model_name,
            pretrained=False,
        )
    
        data_config = resolve_data_config(model=timm_model, use_test_size=True)
    
        loader = create_loader(
            create_dataset("", calib_data_path),
            input_size=data_config["input_size"],
            batch_size=20,
            use_prefetcher=False,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=2,
            crop_pct=data_config["crop_pct"],
        )
        target = []
        for _, labels in loader:
            target.extend(labels.data.tolist())
        outputs_top1 = post_process_top1(torch.tensor(numpy.squeeze(numpy.array(results))))
        top1_acc = getAccuracy_top1(outputs_top1, target)
        return round(top1_acc, 2)
    
    
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
        if args["config"] == "S16S16_MIXED_S8S8":
            activation_spec = Int16Spec()
            weight_spec = Int16Spec()
            algo_config = [
                AutoMixprecisionConfig(
                    l2_target=None,
                    top1_acc_target=0.02,
                    evaluate_function=top1_acc,
                    act_target_quant_type=Int8,
                    weight_target_quant_type=Int8,
                    output_index=0,
                )
            ]
        else:
            activation_spec = Int8Spec()
            weight_spec = Int8Spec()
            activation_spec.set_symmetric(False)
            algo_config = []
    
        config = QConfig(
            global_config=QLayerConfig(activation=activation_spec, weight=weight_spec),
            algo_config=algo_config,
            Percentile=99.9999,
            Int32Bias=False,
            Int16Bias=False,
        )
        print(f"The configuration for quantization is {config}")
    
        # Create an ONNX quantizer
        quantizer = ModelQuantizer(config)
    
        # Quantize the ONNX model
        quantizer.quantize_model(args["input_model_path"], args["output_model_path"], dr)

The cell defines a quantization config with the default S8S8_AAWS
configuration — symmetric INT8 quantization for both weights and
activations, and then generates a quantized model to the models
directory.

.. code:: ipython3

    quant_config = {
        "model_name": "densenet121.ra_in1k",
        "input_model_path": "models/densenet121.ra_in1k.onnx",
        "output_model_path": "models/densenet121.ra_in1k_quantized.onnx",
        "calibration_dataset_path": calib_data_path,
        "config": "S8S8_AAWS",
        "batch_size": 1,
        "workers": 1,
    }

.. code:: ipython3

    quantize_model(quant_config)

The cell applies the same default quantization scheme and changes to the
mixed precision option, then generates the quantized model into the
models directory.

.. code:: ipython3

    quant_config_with_mixed_precision = copy.deepcopy(quant_config)
    quant_config_with_mixed_precision["output_model_path"] = "models/densenet121.ra_in1k_mixed_precision_quantized.onnx"
    quant_config_with_mixed_precision["config"] = "S16S16_MIXED_S8S8"
    
    quantize_model(quant_config_with_mixed_precision)

5) Evaluation and Expected Results
----------------------------------

Evaluation is performed on the ImageNet validation set. We compare three
models — (1) full-precision, (2) quantized without mixed precision, and
(3) quantized with mixed precision — to assess effectiveness of mixed
precision. The full-precision model serves as the baseline for measuring
any accuracy change caused by quantization.

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
    
        if args.get("onnx_input"):
            val_loader = load_loader(args["model_name"], args["data"], args["batch_size"], args["workers"])
            f_top1, f_top5 = metrics(args["onnx_input"], sess_options, providers, val_loader, args["print_freq"])
            print(f" * Prec@1 {f_top1.avg:.3f} ({100 - f_top1.avg:.3f}) Prec@5 {f_top5.avg:.3f} ({100.0 - f_top5.avg:.3f})")
        elif args.get("onnx_float") and args.get("onnx_quant"):
            val_loader = load_loader(args[""], args["data"], args["batch_size"], args["workers"])
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
        "model_name": "densenet121.ra_in1k",
        "batch_size": 1,
        "workers": 1,
        "gpu": False,
        "print_freq": 1000,
    }

.. code:: ipython3

    full_precision_eval_config = copy.deepcopy(eval_config)
    full_precision_eval_config["onnx_input"] = "models/densenet121.ra_in1k.onnx"
    
    evaluate(full_precision_eval_config)

Then, specify the path to the quantized model **without mixed
precision** and record its accuracy on ImageNet val dataset

.. code:: ipython3

    quant_eval_config = copy.deepcopy(eval_config)
    quant_eval_config["onnx_input"] = "models/densenet121.ra_in1k2_quantized.onnx"
    
    evaluate(quant_eval_config)

Last, specify the path to the quantized model **with mixed precision**
and record its accuracy on ImageNet val dataset

.. code:: ipython3

    mixed_precision_eval_config = copy.deepcopy(eval_config)
    mixed_precision_eval_config["onnx_input"] = "models/densenet121.ra_in1k_mixed_precision_quantized.onnx"
    
    evaluate(mixed_precision_eval_config)

The following table contains the expected results, but please note that
different machines can lead to minor variations in the accuracy of
quantized model with Mixed Precision.

+--------+----------+---------------------------+-------------------------+
|        | Float    | Quantized Model without   | Quantized Model with    |
|        | Model    | Mixed Precision           | Mixed Precision         |
+========+==========+===========================+=========================+
| Model  | 33 MB    | 10 MB                     | 17 MB                   |
| Size   |          |                           |                         |
+--------+----------+---------------------------+-------------------------+
| Prec@1 | 76.602 % | 2.642 %                   | 75.216 %                |
+--------+----------+---------------------------+-------------------------+
| Prec@5 | 93.440 % | 7.932 %                   | 92.768 %                |
+--------+----------+---------------------------+-------------------------+
