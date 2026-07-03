Quark ONNX Quantization Tutorial For YOLOv8
===========================================

.. container:: alert alert-block alert-info

   NOTE This tutorial can be downloaded for local execution on a Jupyter
   Notebook environment. Click here to download the source file.

In this tutorial, you will learn how to use AMD Quark, a lightweight and
high-performance quantization framework, to optimize a YOLOv8 model for
image classification tasks. Quantization reduces model size and
computation cost by converting high-precision floating-point parameters
into lower-bit representations, enabling faster inference and lower
memory usage.

The example has the following parts:

- Install requirements

- Prepare model

- Prepare data

- Quantizatize Model

- Evaluate Model

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

Get working directory for json output file

.. code:: ipython3

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace_dir", default="")
    args, _ = parser.parse_known_args()
    workspace_dir = args.workspace_dir
    print("Working root dir: ", workspace_dir)

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
    !wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    !unzip annotations_trainval2017.zip

If you have a local cache to store the dataset, you can use and
environment variable like ``LOCAL_DATA_CACHE`` to specify its path. This
is useful to organize and store all your datasets for different
experiments in a central place. Otherwise, the current folder is used,
and validation dataset and calibration dataset will be created under
current directory.

.. code:: ipython3

    import os
    import sys
    from datetime import datetime
    
    start_exe_time = datetime.now().replace(second=0, microsecond=0)
    
    calib_dataset_path = "val2017"
    annot_path = "annotations/instances_val2017.json"
    
    if os.environ.get("LOCAL_DATA_CACHE") is not None:
        data_path = os.environ["LOCAL_DATA_CACHE"]
        calib_dataset_path = os.path.join(data_path, "coco/2017/val2017")
        annot_path = os.path.join(data_path, "coco/2017/annotations/instances_val2017.json")
    
    if not os.path.exists(calib_dataset_path):
        print("The provided data path does not exist.")
        sys.exit(1)

The COCO dataset should be structured as follows:

- val2017

  - sample_1.jpg
  - sample_2.jpg
  - …

- annotations

  - …
  - instances_train2017.json
  - instances_val2017.json
  - …

4) Quantization Procedure
-------------------------

First, create a data reader that gathers calibration statistics from the
target dataset.

.. code:: ipython3

    import cv2
    import numpy as np
    import onnxruntime as ort
    
    
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

Next, define configuration sets to enable code reuse when running
different setups.

.. code:: ipython3

    import copy
    import re
    
    from quark.onnx import (
        AdaQuantConfig,
        AdaRoundConfig,
        BFloat16Spec,
        BFP16Spec,
        CalibMethod,
        Int8Spec,
        Int16Spec,
        ModelQuantizer,
        QConfig,
        QLayerConfig,
        XInt8Spec,
    )
    
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
    
    
    def parse_subgraphs_list(exclude_subgraphs: str) -> list[tuple[list[str]]]:
        subgraphs_list = []
        tuples = exclude_subgraphs.split(";")
        for tup in tuples:
            tup = tup.strip()
            pattern = r"\[.*?\]"
            matches = re.findall(pattern, tup)
            assert len(matches) == 2
            start_nodes = matches[0].strip("[").strip("]").split(",")
            start_nodes = [node.strip() for node in start_nodes]
            end_nodes = matches[1].strip("[").strip("]").split(",")
            end_nodes = [node.strip() for node in end_nodes]
            subgraphs_list.append((start_nodes, end_nodes))
        return subgraphs_list

Now, define a procedure to run quantizations.

.. code:: ipython3

    def quantize_model(args):
        activation_spec = activation_spec_dict[args["config"]]
        weight_spec = weight_spec_dict[args["config"]]
        algo_confs = []
    
        exclude_info = []
        if args.get("exclude_nodes"):
            exclude_nodes = args["exclude_nodes"].split(";")
            exclude_nodes = [node_name.strip() for node_name in exclude_nodes]
            exclude_info += exclude_nodes
        if args.get("exclude_subgraphs"):
            exclude_subgraphs = parse_subgraphs_list(args["exclude_subgraphs"])
            exclude_info += exclude_subgraphs
    
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
            use_external_data_format=args["save_as_external_data"],
            exclude=exclude_info,
            **extra_info,
        )
    
        calib_datareader = ImageDataReader(args["input_model_path"], args["calib_data_path"])
        quantizer = ModelQuantizer(quant_config)
        quantizer.quantize_model(args["input_model_path"], args["output_model_path"], calib_datareader)

Let’s create a folder “models” to contain all the quantized models.

.. code:: ipython3

    !mkdir -p models

Define your configuration, then perform quantization.

.. code:: ipython3

    quant_config = {
        "input_model_path": "yolov8n.onnx",
        "calib_data_path": calib_dataset_path,
        "cle": False,
        "adaround": False,
        "adaquant": False,
        "learning_rate": 0.1,
        "batch_size": 1,
        "num_iters": 1000,
        "exclude_nodes": "",
        "exclude_subgraphs": "",
        "save_as_external_data": False,
    }

Quantize XINT8

.. code:: ipython3

    quant_config_with_xint8 = copy.deepcopy(quant_config)
    quant_config_with_xint8["output_model_path"] = "models/yolov8_xint8_quantized.onnx"
    quant_config_with_xint8["config"] = "XINT8"
    
    quantize_model(quant_config_with_xint8)

Quantize A8W8

.. code:: ipython3

    quant_config_with_a8w8 = copy.deepcopy(quant_config)
    quant_config_with_a8w8["output_model_path"] = "models/yolov8_a8w8_quantized.onnx"
    quant_config_with_a8w8["config"] = "A8W8"
    
    quantize_model(quant_config_with_a8w8)

Quantize A16W8

.. code:: ipython3

    quant_config_with_a16w8 = copy.deepcopy(quant_config)
    quant_config_with_a16w8["output_model_path"] = "models/yolov8_a16w8_quantized.onnx"
    quant_config_with_a16w8["config"] = "A16W8"
    
    quantize_model(quant_config_with_a16w8)

Quantize BF16

.. code:: ipython3

    quant_config_with_bf16 = copy.deepcopy(quant_config)
    quant_config_with_bf16["output_model_path"] = "models/yolov8_bf16_quantized.onnx"
    quant_config_with_bf16["config"] = "BF16"
    
    quantize_model(quant_config_with_bf16)

Quantize BFP16

.. code:: ipython3

    quant_config_with_bfp16 = copy.deepcopy(quant_config)
    quant_config_with_bfp16["output_model_path"] = "models/yolov8_bfp16_quantized.onnx"
    quant_config_with_bfp16["config"] = "BFP16"
    
    quantize_model(quant_config_with_bfp16)

Quantize XINT8 with CLE

.. code:: ipython3

    quant_config_with_xint8_cle = copy.deepcopy(quant_config)
    quant_config_with_xint8_cle["output_model_path"] = "models/yolov8_xint8_cle_quantized.onnx"
    quant_config_with_xint8_cle["config"] = "XINT8"
    quant_config_with_xint8_cle["include_cle"] = True
    
    quantize_model(quant_config_with_xint8_cle)

Quantize XINT8 with AdaRound

.. code:: ipython3

    quant_config_with_xint8_adaround = copy.deepcopy(quant_config)
    quant_config_with_xint8_adaround["output_model_path"] = "models/yolov8_xint8_adaround_quantized.onnx"
    quant_config_with_xint8_adaround["config"] = "XINT8"
    quant_config_with_xint8_adaround["adaround"] = True
    quant_config_with_xint8_adaround["learning_rate"] = 0.1
    quant_config_with_xint8_adaround["num_iters"] = 3000
    
    quantize_model(quant_config_with_xint8_adaround)

Quantize XINT8 with AdaQuant

.. code:: ipython3

    quant_config_with_xint8_adaquant = copy.deepcopy(quant_config)
    quant_config_with_xint8_adaquant["output_model_path"] = "models/yolov8_xint8_adaquant_quantized.onnx"
    quant_config_with_xint8_adaquant["config"] = "XINT8"
    quant_config_with_xint8_adaquant["adaquant"] = True
    quant_config_with_xint8_adaquant["learning_rate"] = 0.00001
    quant_config_with_xint8_adaquant["num_iters"] = 10000
    
    quantize_model(quant_config_with_xint8_adaquant)

Quantize XINT8 with Exclude Nodes

.. code:: ipython3

    quant_config_with_xint8_exclude_nodes = copy.deepcopy(quant_config)
    quant_config_with_xint8_exclude_nodes["output_model_path"] = "models/yolov8_xint8_exclude_nodes_quantized.onnx"
    quant_config_with_xint8_exclude_nodes["config"] = "XINT8"
    quant_config_with_xint8_exclude_nodes["exclude_nodes"] = "/model.22/Concat_5"
    
    quantize_model(quant_config_with_xint8_exclude_nodes)

Quantize A8W8 with Exclude Graph

.. code:: ipython3

    quant_config_with_a8w8_exclude_subgraphs = copy.deepcopy(quant_config)
    quant_config_with_a8w8_exclude_subgraphs["output_model_path"] = "models/yolov8_a8w8_exclude_subgraphs_quantized.onnx"
    quant_config_with_a8w8_exclude_subgraphs["config"] = "A8W8"
    quant_config_with_a8w8_exclude_subgraphs["exclude_subgraphs"] = "[/model.22/Concat_3], [/model.22/Concat_5]"
    
    quantize_model(quant_config_with_a8w8_exclude_subgraphs)

5) Evaluation and Expected Results
----------------------------------

The following section demonstrates how to implement a custom evaluator
when working with YOLOv8 ONNX models. We will use pycocotools and define
an evaluation function for mean Average Precision (mAP), which is a
standard evaluation metric used in object detection to measure how well
a model identifies and localizes objects in images. It summarizes the
trade-off between precision (how many predicted objects are correct) and
recall (how many ground-truth objects are detected).

.. code:: ipython3

    import json
    
    # from faster_coco_eval import COCO, COCOeval_faster
    import os
    
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from tqdm import tqdm
    
    
    def evaluate_onnx_model(onnx_path, coco_json_path, img_dir, output_file_name=None):
        """Evaluate ONNX YOLOv8 model using COCO API."""
    
        # Load model
        model = YOLO(onnx_path, verbose=False)
    
        # Load COCO annotations
        coco_gt = COCO(coco_json_path)
    
        # IMPORTANT: build mapping from YOLO class index -> real COCO category id
        # COCO category IDs are NOT 1..80
        coco_cat_ids = sorted(coco_gt.getCatIds())
    
        results = []
        img_ids = coco_gt.getImgIds()
    
        for img_id in tqdm(img_ids, desc="Evaluating"):
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info["file_name"])
    
            preds = model(img_path, device="cpu", conf=0.5, verbose=False)
    
            for pred in preds:
                if pred.boxes is None or len(pred.boxes) == 0:
                    continue
    
                conf_s = pred.boxes.conf.cpu().numpy()
                cls_s = pred.boxes.cls.cpu().numpy().astype(int)
                xyxy_s = pred.boxes.xyxy.cpu().numpy()
    
                for i in range(len(cls_s)):
                    cls_id = cls_s[i]
    
                    # Map YOLO class index -> COCO category id
                    coco_cat_id = coco_cat_ids[cls_id]
    
                    x1, y1, x2, y2 = xyxy_s[i]
    
                    results.append(
                        {
                            "image_id": img_id,
                            "category_id": coco_cat_id,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(conf_s[i]),
                        }
                    )
    
        # Save predictions
        res_file = output_file_name if output_file_name else "predictions_coco.json"
    
        with open(res_file, "w") as f:
            json.dump(results, f)
    
        print(f"\nSaved {len(results)} detections to {res_file}")
    
        # Evaluate with COCO API
        coco_dt = coco_gt.loadRes(res_file)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    
        stats = coco_eval.stats
        return stats[0], stats[1]

First, define an evaluation configuration and record the accuracy of the
full-precision model. For simplicity, we use the same dataset for both
calibration and evaluation; however, it is strongly recommended to use a
separate dataset for evaluation to ensure the model’s validity.

.. code:: ipython3

    fp32_map50_95, fp32_map50 = coco_evaluator = evaluate_onnx_model(
        onnx_path="yolov8n.onnx",
        coco_json_path=annot_path,
        img_dir=calib_dataset_path,  # strongly recommend to use a different dataset for evaluation
        output_file_name="full_precision_pred.json",
    )

Now, define an evaluation configuration for the A8W8 quantized model and
record its accuracy. As before, we use the same dataset for both
calibration and evaluation; however, it is strongly recommended to use a
separate dataset for evaluation.

In this example, we use one of the A8W8 quantized models to demonstrate
the evaluation process, but you are encouraged to experiment with other
quantized models as well.

.. code:: ipython3

    quant_map50_95, quant_map50 = evaluate_onnx_model(
        onnx_path="models/yolov8_xint8_adaround_quantized.onnx",
        coco_json_path=annot_path,
        img_dir=calib_dataset_path,  # strongly recommend to use a different dataset for evaluation
    )

The following table contains the expected results, but please note that
different machines can lead to minor variations in the accuracy of
quantized model.

============ =========== ===============
\            Float Model Quantized Model
============ =========== ===============
Model Size   12.26 MB    3.5 MB
mAP@0.5:0.95 0.265       0.206
mAP@0.5      0.341       0.269
mAP@0.75     0.295       0.233
============ =========== ===============

.. code:: ipython3

    # This cell should have the remove-input tag as we don't want it rendered in the documentation
    # it's creating results for submission to the dashboard
    
    import pandas as pd
    
    model_name = "ryzen_ai_yolov8"
    
    benchmarks = {"ryzen_ai_yolov8": {"map50_95": 0.206, "map50": 0.269}}
    
    
    def get_golden_val(benchmarks_dict: dict, model_name: str):
        golden_df = pd.DataFrame.from_dict(benchmarks_dict, orient="index")
        map50_95 = golden_df.at[model_name, "map50_95"]
        map50 = golden_df.at[model_name, "map50"]
        benchmark = {"mAP@0.5:0.95": map50_95, "mAP@0.5": map50}
        return benchmark
    
    
    def compare_result(golden: float, current: float):
        result = current - golden
        diff = "no change"
        if golden > current:
            diff = f"drops by {result:.3f}"
        elif golden < current:
            diff = f"rises by {result:.3f}"
        return diff
    
    
    def get_output(benchmark: dict[str, float], current: dict[str, float]):
        try:
            map50_95, map50 = current["mAP@0.5:0.95"], current["mAP@0.5"]
            diff_map50_95 = compare_result(benchmark["mAP@0.5:0.95"], map50_95)
            diff_map50 = compare_result(benchmark["mAP@0.5"], map50)
        except NameError:
            map50_95, map50 = "ERROR", "ERROR"
            diff_map50_95, diff_map50 = "ERROR", "ERROR"
        map50_95_meta = {"golden": benchmark["mAP@0.5:0.95"], "difference": diff_map50_95}
        map50_meta = {"golden": benchmark["mAP@0.5"], "difference": diff_map50}
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
                            "name": "mAP@0.5:0.95",
                            "value": map50_95,
                            "metadata": str(map50_95_meta),
                        },
                        {
                            "name": "mAP@0.5",
                            "value": map50,
                            "metadata": str(map50_meta),
                        },
                    ],
                    "metadata": {"duration": str(datetime.now().replace(second=0, microsecond=0) - start_exe_time)},
                }
            ],
        }
        return result
    
    
    benchmark = get_golden_val(benchmarks, model_name)
    current = {"mAP@0.5:0.95": quant_map50_95, "mAP@0.5": quant_map50}
    output = get_output(benchmark, current)
    
    root_path = f"{workspace_dir}/logs_regression_test/"
    os.makedirs(root_path, exist_ok=True)
    file_path = f"{root_path}/{model_name}.json"
    
    with open(file_path, "w") as json_file:
        json.dump(output, json_file)
