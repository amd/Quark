# DLRM Model Quantization Using Quark

This document provides examples of quantizing and exporting the DLRM models using Quark. Please refer to [DLRM](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch) for more details about the model.

## Preparation

### Environment

The quantization and evaluation scripts run on CPU. Install the dependencies as follows:

```bash

pip install scikit-learn
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    torch==2.13.0 fbgemm-gpu==1.8.0 torchrec==1.8.0

```

`torchrec` pulls in `tensordict`, `torchmetrics`, `pyre-extensions` and `tqdm` itself, so those
do not need to be listed separately. PyPI is given as a secondary index because the PyTorch CPU
index does not carry `tensordict`.

Three things are worth knowing before you deviate from the command above.

**`torch`, `fbgemm-gpu` and `torchrec` are a matched set.** Each `fbgemm-gpu` release targets one
`torch` release, and a mismatch fails at import with an undefined-symbol error rather than
anything self-explanatory. The pairing follows
[FBGEMM's compatibility table](https://docs.pytorch.org/FBGEMM/general/Releases.html#fbgemm-releases-compatibility):
1.8.0 with torch 2.13, 1.6.0 with 2.11, 1.5.0 with 2.10, and so on. Leaving `torch` unpinned is
enough to break it, since pip will happily install a newer torch than the pinned `fbgemm-gpu`
was built for.

**Take `fbgemm-gpu` from the PyTorch CPU index, not from PyPI.** The PyPI package under that
name is the CUDA build; on a CPU-only or ROCm machine it fails to load with
`libtorch_cuda.so: cannot open shared object file`.

**If you already have a `torch` you need to keep, do not install this CPU wheel over it.** That
applies in particular to the GPU build Quark itself is installed against. Instead install only
`fbgemm-gpu` and `torchrec` at the versions matching your existing torch -- the CPU build of
`fbgemm-gpu` works fine against a ROCm torch of the same version, and this example never needs
a GPU anyway:

```bash
# example: an environment that already has torch 2.11
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    fbgemm-gpu==1.6.0 torchrec==1.6.0
```

`fbgemm-gpu` is also what gates the usable Python versions: 1.8.0 ships wheels for CPython 3.10
through 3.14, whereas the 1.0.0 release previously pinned here stopped at 3.12.

### Third-Party Dependencies

The example relies on some code from [inference_results_v3.1](https://github.com/mlcommons/inference_results_v3.1/tree/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python) repo.

```bash
git clone https://github.com/mlcommons/inference_results_v3.1.git
cd inference_results_v3.1
git checkout 951b4a7686692d1a0d9b9067a36a7fc26d72ada5
cp -r closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/. /path/to/Quark/examples/torch/rm/utils/
export PYTHONPATH=$PYTHONPATH:/path/to/Quark/examples/torch/rm/utils
```

The copied code then needs four source edits. Apply them from this directory:

```bash
patch -p1 -d utils --forward < dlrm_third_party.patch
```

`dlrm_third_party.patch` is the authoritative form of these edits -- it is what CI applies too,
so the documented setup and the tested one cannot drift apart. Read it for the exact diff; what
follows is why each edit exists.

**`multihot_criteo.py` -- off-by-one in the EmbeddingBag offsets**
([upstream line](https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/multihot_criteo.py#L418)).
`_get_offsets()` builds `batchsize` offsets where `torch.nn.EmbeddingBag` expects `batchsize + 1`,
which silently drops the last sample of every batch. This one skews the AUC rather than raising,
so it is easy to miss.

**`multihot_criteo.py` -- read the `.npy` header through NumPy's public API**
([upstream line](https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/multihot_criteo.py#L400)).
`_load_from_npz()` calls `np.lib.format._read_array_header`, a private helper that NumPy 2.0
moved out of reach when it split public from private API, so on NumPy 2.x the dataset fails to
load with `AttributeError`. The replacement dispatches on the header version and calls
`read_array_header_1_0` / `read_array_header_2_0`, which are public on both NumPy 1.x and 2.x.

**`model/dlrm_model.py` -- replace `MergedEmbeddingBagCat.forward()`**
([upstream line](https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/model/dlrm_model.py#L204)).
The upstream int8 branch calls `torch.ops.torch_ipex.merged_emb_with_cat`, a fused Intel kernel.
The replacement performs the same lookup with a plain `EmbeddingBag` loop, which is what makes
the example runnable without `intel-extension-for-pytorch` and what lets Quark observe the
individual EmbeddingBags it needs to quantize.

**Both files -- drop the module-level `intel-extension-for-pytorch` imports**
([one](https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/model/dlrm_model.py#L6),
[two](https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/backend_pytorch_native.py#L15)).
This follows from the previous edit: with the fused kernel gone, the only remaining consumer is
`ipex.optimize()` on the `--use-bf16` path, which neither `quark_dlrm.py` nor `quark_dlrm_eva.py`
takes. The patch moves that import into the `--use-bf16` branch, so `intel-extension-for-pytorch`
becomes optional -- install it only if you intend to run the bf16 path.

The copied code also does not use `transformers`, so no `transformers` pin is required here.

### Model weights

For DLRM model, refer to the [README.md](https://github.com/mlcommons/inference/blob/master/recommendation/dlrm_v2/pytorch/README.md) to download the model weights and datasets for calibration. The model weights have multiple files, you need to pack them in a single pt file. Please run the script to get the single pt file

This packing step is the only one that needs the MLPerf loadgen bindings and `torchsnapshot`
(`dump_torch_model.py` imports `mlperf_loadgen` at module level and restores the checkpoint
through `torchsnapshot.Snapshot`). The quantization and evaluation scripts below need neither,
so skip this section entirely if you already have the packed `.pt` file.

```bash
pip install mlcommons-loadgen torchsnapshot==0.1.0
```

`mlcommons-loadgen` provides prebuilt wheels that expose the same `mlperf_loadgen` module as
the in-tree loadgen, which replaces cloning `mlcommons/inference` and building it from source.

```bash
python utils/dump_torch_model.py \
    --model-path=/path/to/model_dir \
    --dataset-path=/path/to/data/Criteo1TBMultiHotPreprocessed
```

## Quantization & Export Scripts

Run the python script as follows

```python3
# The compressed quantized model
python quark_dlrm.py \
    --max-batchsize=64000 \
    --model-path=/path/to/dlrm-multihot-pytorch.pt \
    --int8-model-dir /dir/to/dlrm_quark \
    --int8-model-name DLRM_INT \
    --dataset-path=/path/to/data/Criteo1TBMultiHotPreprocessed \
    --calibration \
    --compressed

# The QDQ model
python quark_dlrm.py \
    --max-batchsize=64000 \
    --model-path=/path/to/dlrm-multihot-pytorch.pt \
    --int8-model-dir /dir/to/dlrm_quark \
    --int8-model-name DLRM_INT \
    --dataset-path=/path/to/data/Criteo1TBMultiHotPreprocessed \
    --calibration
```

## Evaluation

Quark currently uses Area Under the Curve(AUC) as the evaluation metric of dlrm for accuracy loss before and after quantization.The specific AUC algorithm can be referenced [roc_auc_score](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.roc_auc_score.html).

Run the python script as follows to calculate the AUC(Area Under Curve) for the quantized model

```python3
# evaluate the compressed quantized model
python quark_dlrm_eva.py \
    --max-batchsize=64000 \
    --model-path=/path/to/dlrm-multihot-pytorch.pt \
    --int8-model-dir /dir/to/dlrm_quark \
    --int8-model-name DLRM_INT \
    --dataset-path=/path/to/data/Criteo1TBMultiHotPreprocessed \
    --calibration \
    --compressed

# evaluate the QDQ model
python quark_dlrm_eva.py \
    --max-batchsize=64000 \
    --model-path=/path/to/dlrm-multihot-pytorch.pt \
    --int8-model-dir /dir/to/dlrm_quark \
    --int8-model-name DLRM_INT \
    --dataset-path=/path/to/data/Criteo1TBMultiHotPreprocessed \
    --calibration
```

Both commands report a single line to stdout:

```text
Total ROC AUC =  0.8027...
```

With no `--samples-to-aggregate-*` flag set, each row counts as one sample, so the evaluation
sweeps the whole validation split (the first half of `day_23`, roughly 89.1M rows) in
`--max-batchsize` steps -- about 1393 batches at 64000, all on CPU. Pass `--count-samples N`
to truncate the dataset for a quicker smoke run, keeping in mind that a truncated run produces
a different AUC than the full-split numbers tabulated below.

Note that `quark_dlrm_eva.py` always applies `load_params()`, so it only reports the quantized
AUC; the unquantized reference in the table below is not reproducible through this script.

The quantization evaluation results are conducted in pseudo-quantization mode, which may slightly differ from the actual quantized inference accuracy. These results are provided for reference only.

### Evaluation scores

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>dlrm </strong>
   </td>
   <td><strong>dlrm-embeddingbag-uint4-weight-int8(this model)</strong>
   </td>
  </tr>
  <tr>
   <td>AUC-MultihotCriteo
   </td>
   <td>0.8031
   </td>
   <td>0.8027
   </td>
  </tr>
</table>

<!--
## License
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
-->
