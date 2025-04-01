# YOLO-NAS FX graph Quantization

In this example, we present an Object Detection Model Quantization workflow. We used YOLO-NAS as a demonstration to illustrate the quantization workflow. Both the PTQ and QAT show the compatible result.

[YOLO-NAS](https://github.com/Deci-AI/super-gradients/tree/master) is an object detection model developed by Deci that achieves SOTA performances compared to YOLOv5, v7, and v8.

## Preparation

1. Feels easy to install the third-party Python packages:

   - ```sh
     pip install super_gradients
     ```

2. Download the COCO 2017 train and val images dataset from [http://cocodataset.org](http://cocodataset.org/#download).

   - Download coco dataset: [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [train2017](http://images.cocodataset.org/zips/train2017.zip), [val2017](http://images.cocodataset.org/zips/val2017.zip)

   - After Unzip, the data directory structure would be the following:

   - ```bash
     coco_data_dir
     ├── annotations
     │      ├─ instances_train2017.json
     │      ├─ instances_val2017.json
     │      └─ ...
     └── images
         ├── train2017
         │   ├─ 000000000001.jpg
         │   └─ ...
         └── val2017
             └─ ...
     ```

## Start from zero to quant YOLO-NAS

We provide a clean and compact code for a quick start, the user can directly run to quant the YOLO-NAS,

`python yolo-nas_quant.py --data_dir={DATA_PATH_TO_COCO}`

We will give a short and clear explanation to help you better understand. We hope that after glancing at the following instructions, you can use Quark's FX-Graph quantization tool more fluently.

For Import necessary packages and init GPU devices, the code can be seen in the yolo-nas_quant.py.

### 1.Prepare data and model

```python
from super_gradients.training.dataloaders import coco2017_val_yolo_nas, coco2017_train_yolo_nas
# ===== prepare the data for training, validation and calibration
valid_dataloader = coco2017_val_yolo_nas(dataset_params={"data_dir": args.data_dir})
train_dataloader = coco2017_train_yolo_nas(dataset_params={"data_dir": args.data_dir})
calib_data_size = 10 # we only use a small part of validation data for calibration(PTQ).
calib_data = [x[0] for x in list(itertools.islice(valid_dataloader, calib_data_size))] # each item shape [b, 3, 640, 640]

# ===== prepare the data for training, validation and calibration
yolo_nas = super_gradients.training.models.get("yolo_nas_s", pretrained_weights="coco").to(device)
yolo_nas.prep_model_for_conversion(input_size=[1, 3, 640, 640])
graph_model = capture_pre_autograd_graph(yolo_nas.eval(), (dummy_input, ))
```

**NOTE**: we need to call `prep_model_for_conversion` for better model optimization.

```python
# Before call prep_model_for_conversion
# The QARepVGGBlock module's forward would be like this: (super_gradients/modules/qarepvgg_block.py)
'''
                     |                                                     |
        |------------|---------|                                           |
        |            |         |                                      3x3conv+bias
     3x3conv    1x1conv+bias   |                                           |
        |            |         |                                           |
    BatchNorm     *alpha       |                                           |
        |            |         |  -> prep_model_for_conversion->       BatchNorm
        |------------+---------|                                           |
                     |                                                     |
                 BatchNorm                                                 |
                     |                                                  Act(ReLU)
                  Act(ReLU)                                                |
                     |                                                     |
                    SE                                                     SE
'''
```

The yolo-mas model structure is specially designed, typically we can not directly perform model optimizations (like folding the Batchnorm layer to the Conv layer). We need to call `prep_model_for_conversion` to prepare the model for easier optimization.

After you call `capture_pre_autograd_graph`, the original `torch.nn.Module` will be translated to `torch.fx.GraphModule`. `torch.fx.GraphModule` is a lower-level graph (at the `torch.ops.aten` operator level), which only contains `torch.ops.aten` operators and is fully functional, without any inplace operators such as `torch.add_`. More information can be found [here](https://pytorch.org/docs/stable/export.html).

**NOTE**: Please make  good use of  `to_folder("./{export_code_dir}")` function to export graph_model to readable Python code. The API can be found [here](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule.to_folder).

```python
graph_model.to_folder("./my_code")
```

The exported Python code `module.py` in folder`my_code` is critical for further graph quantization annotation. Please read this code and understand the relationship with the original yolo-nas `torch.nn.Module`  format code.

### 2.Set quantization Config and Quantizer

```python
# we adopt INT8 quantization for weight, bias and activation.
INT8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                            qscheme=QSchemeType.per_tensor,
                                            observer_cls=PerTensorMinMaxObserver,
                                            symmetric=True,
                                            scale_type=ScaleType.float,
                                            round_method=RoundType.half_even,
                                            is_dynamic=False)
quant_config = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC,
                                      output_tensors=INT8_PER_TENSOR_SPEC,
                                      weight=INT8_PER_TENSOR_SPEC,
                                      bias=INT8_PER_TENSOR_SPEC)
```

#### Select and scope that do not want to quantize (Further Improve in the future)

Read the `module.py` generated by `to_folder()` function. By viewing the `forward()` function, we need to select and decide the scope we don't want to quantize. That is how to set `exclude_node`.

```python
# Prototype function, will be changed in future version.
exclude_node = [ '_param_constant310', '_set_grad_enabled_1',
                 '_param_constant310', 'softmax',
                 '_param_constant323', 'softmax_1',
                 '_param_constant336', 'softmax_2' ]
# 1. All the codes in forward() are default set to quantable.
# 2. The first str pair (['_param_constant310', '_set_grad_enabled_1']) defines the scope that we don't want to quantify.
# 3. The second and later pairs define the scope that except in the previous scope defined by the first str pair, means the operation among these pairs will be quantized.


# The following shows how the exclude_node will effect the quantizable scope in the Yolo-Nas model.
# module.py in my_code folder
#class FxModule(torch.nn.Module):
#    def __init__(self):
#        ...

#    def forward(self, x):
#        ...                     (the code above is quantizable)
#        _param_constant310      (quantizable)
#        ...                     (quantizable)
#        softmax                 (quantizable)
#        ...                     (no quant)
#        _param_constant323      (quantizable)
#        ...                     (quantizable)
#        softmax_1               (quantizable)
#        ...                     (no quant)
#        _param_constant336      (quantizable)
#        ...                     (quantizable)
#        softmax_2               (quantizable)
#        ...                     (no quant)
#        _set_grad_enabled_1     (no quant)
#        ...                     (quantizable)
#        return  out_put
```

After we select the quantizable scope we will define the quant configuration:

```python
quant_config = Config(global_quant_config=quant_config,
                          quant_mode=QuantizationMode.fx_graph_mode,
                          exclude=exclude_node) # exclude_node
quantizer = ModelQuantizer(quant_config)
```

### 3.Calibration (PTQ) / Training (QAT) (Optional)

```python
# PTQ will be performed automatically.
quantized_model = quantizer.quantize_model(graph_model, calib_data)

# QAT  (Optional),
# User needs to define the train code and train the quantized_model directly
train_model(quantized_model)
# Then the weight and quantization parameters will be updated during the training process.
```

NOTE: the training result (QAT) very rely on training parameters.

### 4.Exported to Onnx and other format files

**NOTE:** In [YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/arch_params/yolo_nas_s_arch_params.yaml), the output contain two parts, [decoded_predictions & raw_predictions](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/yolo_nas/dfl_heads.py#L245). Some data in raw_predictions is not fully supported in exported to Onnx model.

```python
# doing the PTQ/QAT(optional) as above
# After PTQ/QAT(optional)
torch.save(quantized_model.state_dict(), "./{Path_to_state_dict_path}")
```

Then user need to modify the code `dfl_heads.py`([links](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/yolo_nas/dfl_heads.py#L245)) as follows:

```python
# original code
#raw_predictions = cls_score_list, reg_distri_list, anchors, anchor_points, num_anchors_list, stride_tensor
raw_predictions = cls_score_list, reg_distri_list, anchors, anchor_points, stride_tensor
```

We then need to re-run the quantization code, to get the quantized_model model. **Note**, because modified the code, we can not / no need to perform training (QAT).

```python
quantized_model.load_state_doct(torch.load("./{Path_to_state_dict_path}"))
freezeded_model = quantizer.freeze(quantized_model.eval())
# export to onnx model
from quark.torch import ModelExporter
from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
config = ExporterConfig(json_export_config=JsonExporterConfig())
exporter = ModelExporter(config=config, export_dir=args.export_dir)
example_inputs = (torch.rand(val_batch_size, 3, 640, 640).to(device), )
exporter.export_onnx_model(freezeded_model, example_inputs[0])
```

## Quantization Result

| Metric              | FP32 model | INT 8 PTQ | INT 8 QAT |
| :------------------ | ---------- | :-------: | :-------: |
| Precision@0.50      | 0.1827     |  0.1693   |  0.1430   |
| Recall@0.50         | 0.8048     |  0.7842   |  0.7933   |
| mAP@0.50            | 0.6466     |  0.6236   |  0.6239   |
| F1@0.50             | 0.2919     |  0.2724   |  0.2375   |
| Precision@0.50:0.95 | 0.1426     |  0.1305   |  0.1100   |
| Recall@0.50:0.95    | 0.6155     |  0.5941   |  0.5971   |
| mAP@0.50:0.95       | 0.4759     |  0.4537   |  0.4532   |
| F1@0.50:0.95        | 0.2269     |  0.2093   |  0.1820   |
