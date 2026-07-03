Quark ONNX Quantization Tutorial For GPTQ
=========================================

.. container:: alert alert-block alert-info

   NOTE This tutorial can be downloaded for local execution on a Jupyter
   Notebook environment. Click here to download the source file.

GPTQ (Gradient Post-Training Quantization) is an advanced post-training
quantization method designed to minimize accuracy degradation by
applying a layer-wise error compensation mechanism during quantization.
Instead of requiring access to the full training dataset, GPTQ uses a
small calibration set to estimate quantization impact and optimize
weight adjustments. This makes it particularly effective for large
language models and other architectures sensitive to quantization noise.

By applying GPTQ method within the AMD Quark toolchain, developers can
efficiently deploy highly optimized quantized models with improved
accuracy retention, reduced compute demand, and enhanced performance
across AMD’s hardware ecosystem.

The example has the following parts:

- Install requirements

- Prepare model

- Prepare dataset

- Quantizatize models

  - INT8 only

  - INT8 and GPTQ

  - INT8 and MatMul NBits

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

.. code:: ipython3

    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace_dir", default="")
    args, _ = parser.parse_known_args()
    workspace_dir = args.workspace_dir
    start_exe_time = datetime.now().replace(second=0, microsecond=0)
    print("Working root dir: ", workspace_dir)

2) Export ONNX Model From OPT-125M Model.
-----------------------------------------

| Let’s download necessary files and put it in one folder named
  opt-125m:

| mkdir opt-125m
| wget -P opt-125m
  https://huggingface.co/facebook/opt-125m/resolve/main/pytorch_model.bin
| wget -P opt-125m
  https://huggingface.co/facebook/opt-125m/resolve/main/config.json
| wget -P opt-125m
  https://huggingface.co/facebook/opt-125m/resolve/main/tokenizer_config.json
| wget -P opt-125m
  https://huggingface.co/facebook/opt-125m/resolve/main/vocab.json
| wget -P opt-125m
  https://huggingface.co/facebook/opt-125m/resolve/main/merges.txt
| wget -P opt-125m
  https://huggingface.co/facebook/opt-125m/resolve/main/generation_config.json
| wget -P opt-125m
  https://huggingface.co/facebook/opt-125m/resolve/main/special_tokens_map.json

.. code:: ipython3

    import os
    import shutil
    
    # We have a shared cache directory containing pre-downloaded models. You can manually create a folder and download the above files
    HF_CACHE_MODEL = os.path.join(
        os.environ.get("HF_HOME", ""),
        "hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6",
    )
    
    if os.path.isdir(HF_CACHE_MODEL):
        print(f"CI cache found at: {HF_CACHE_MODEL}")
        print("Copying cached model to ./opt-125m/ ...")
        if os.path.exists("opt-125m"):
            shutil.rmtree("opt-125m")
        shutil.copytree(HF_CACHE_MODEL, "opt-125m", symlinks=False)
        print("Done.")

Now create a folder “models” and convert opt-125m to the onnx format
into the “models” folder.

.. code:: ipython3

    !rm -rf models
    !mkdir -p models
    !optimum-cli export onnx --model ./opt-125m --task text-generation ./models/

Import all the dependencies

.. code:: ipython3

    import copy
    import logging
    import os
    import random
    from typing import Any
    
    import numpy as np
    import onnxruntime as ort
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset, SequentialSampler
    from tqdm import tqdm
    from transformers import AutoTokenizer, GPT2Tokenizer, PreTrainedTokenizer
    
    from quark.onnx import Config, ModelQuantizer
    from quark.onnx.quantization.config import get_default_config

3) Prepare Dataset
------------------

We provide a dataloader that supports three commonly used datasets:
Pileval, CNN DailyMail, and WikiText. In this tutorial, we will be using
the WikiText dataset as an example, but you are encouraged to experiment
with the others to evaluate GPTQ’s effectiveness across different data
domains.

The WikiText-2 dataset is a widely used benchmark for evaluating
language models. It consists of high-quality Wikipedia text curated to
better reflect natural language usage compared to earlier corpora. A key
feature is the preservation of article structure, such as headings and
paragraph organization—information often lost in simpler datasets. This
structure helps language models learn long-range dependencies more
effectively.

.. code:: ipython3

    dataset_cache_dir = os.environ.get("LOCAL_DATA_CACHE", "data_cache")
    
    
    def ensure_padding_token(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
        # CI can load exported GPT2/OPT tokenizer folders without a pad token.
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer must define pad_token or eos_token for padded calibration batches.")
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    
    
    def resolve_exported_onnx_model_path(model_dir: str) -> str:
        preferred_model_names = (
            "decoder_model_merged.onnx",
            "decoder_model.onnx",
            "model.onnx",
            "decoder_with_past_model.onnx",
        )
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Export directory {model_dir} does not exist.")
    
        onnx_files = sorted(file_name for file_name in os.listdir(model_dir) if file_name.endswith(".onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX model files found in {model_dir}.")
        if len(onnx_files) == 1:
            return os.path.join(model_dir, onnx_files[0])
    
        for model_name in preferred_model_names:
            if model_name in onnx_files:
                return os.path.join(model_dir, model_name)
    
        raise FileNotFoundError(f"Unable to determine exported ONNX model path in {model_dir}. Found: {onnx_files}")
    
    
    exported_model_path = resolve_exported_onnx_model_path("models")
    exported_model_filename = os.path.basename(exported_model_path)
    print(f"Detected exported ONNX model: {exported_model_path}")
    
    
    def get_calib_dataloader(
        dataset_name: str, **kwargs: Any
    ) -> DataLoader[torch.Tensor] | DataLoader[list[dict[str, torch.Tensor]]] | DataLoader[dict[str, torch.Tensor]]:
        if dataset_name in ["pileval", "cnn_dailymail"]:
            return get_calib_dataloader_to_tensor(dataset_name, **kwargs)
        elif dataset_name in ["pileval_for_awq_benchmark", "wikitext_for_gptq_benchmark"]:
            return get_calib_dataloader_to_list(dataset_name, **kwargs)
        else:
            raise NotImplementedError
    
    
    def get_pileval(
        tokenizer: PreTrainedTokenizer, nsamples: int, seqlen: int, device: str | None, seed: int = 0
    ) -> list[dict[str, torch.Tensor]]:
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation", cache_dir=dataset_cache_dir)
        dataset = dataset.shuffle(seed=seed)
        samples = []
        n_run = 0
        for data in dataset:
            line = data["text"]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
            if len(line_encoded) > 512:
                continue
            sample = torch.tensor([line_encoded])
            if sample.numel() == 0:
                continue
            sample = sample.to(device)
            samples.append(sample)
            n_run += 1
            if n_run == nsamples:
                break
        # now concatenate all samples and split according to block size
        cat_samples = torch.cat(samples, dim=1)
        n_split = cat_samples.shape[1] // seqlen
        logging.debug(f" * Split into {n_split} blocks")
        traindataset = []
        for i in range(n_split):
            traindataset.append({"input_ids": cat_samples[:, i * seqlen : (i + 1) * seqlen]})
        return traindataset
    
    
    def get_wikitext2(
        tokenizer: PreTrainedTokenizer, nsamples: int, seqlen: int, device: str | None, seed: int = 0
    ) -> list[dict[str, torch.Tensor]]:
        traindata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", cache_dir=dataset_cache_dir)
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        trainenc = trainenc.to(device)
    
        import random
    
        random.seed(seed)
        torch.random.manual_seed(seed)
    
        traindataset = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            attention_mask = torch.ones_like(inp)
            traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        return traindataset
    
    
    def get_calib_dataloader_to_list(
        dataset_name: str = "pileval_for_awq_benchmark",
        tokenizer: AutoTokenizer = None,
        batch_size: int = 1,
        num_calib_data: int = 128,
        seqlen: int = 2048,
        device: str = "cpu",
    ) -> DataLoader[list[dict[str, torch.Tensor]]]:
        if dataset_name == "pileval_for_awq_benchmark":
            samples = get_pileval(tokenizer, num_calib_data, seqlen, device, seed=42)
        elif dataset_name == "wikitext_for_gptq_benchmark":
            samples = get_wikitext2(tokenizer, num_calib_data, seqlen, device)
        else:
            raise NotImplementedError
    
        calib_dataloader: DataLoader[list[dict[str, torch.Tensor]]] = DataLoader(samples, batch_size=None, shuffle=False)  # type: ignore
    
        return calib_dataloader
    
    
    def get_calib_dataloader_to_tensor(
        dataset_name: str = "cnn_dailymail",
        tokenizer: AutoTokenizer = None,
        batch_size: int = 1,
        num_calib_data: int = 512,
        seqlen: int = 512,
        device: str | None = None,
    ) -> DataLoader[torch.Tensor]:
        if dataset_name == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation", cache_dir=dataset_cache_dir)
            text_data = dataset["text"][:num_calib_data]
        elif dataset_name == "cnn_dailymail":
            dataset = load_dataset("abisee/cnn_dailymail", name="3.0.0", split="train", cache_dir=dataset_cache_dir)
            text_data = dataset["article"][:num_calib_data]
        elif dataset_name == "wikitext":
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", cache_dir=dataset_cache_dir)
            text_data = dataset["text"][:num_calib_data]
        else:
            raise NotImplementedError
    
        tokenizer = ensure_padding_token(tokenizer)
        batch_encoded = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True, max_length=seqlen)
        if device:
            batch_encoded = batch_encoded.to(device)
        batch_encoded = batch_encoded["input_ids"]
    
        calib_dataloader = DataLoader(batch_encoded, batch_size=batch_size, shuffle=False)
    
        return calib_dataloader
    
    
    def get_calib_dataloader_to_dict(
        dataset_name: str = "cnn_dailymail",
        tokenizer: AutoTokenizer = None,
        batch_size: int = 1,
        num_calib_data: int = 512,
        seqlen: int = 512,
        device: str | None = None,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        tokenizer = ensure_padding_token(tokenizer)
    
        def make_data_block(
            examples: dict[str, list[str]],
            tokenizer: AutoTokenizer = None,
            prompt_col_name: str = "",
            max_length: int = 512,
        ) -> dict[str, list[list[torch.Tensor]]]:
            res: dict[str, list[list[torch.Tensor]]] = tokenizer(
                examples[prompt_col_name], padding=True, truncation=True, max_length=max_length
            )
            return res
    
        def my_collate_fn(blocks: list[dict[str, list[list[str]]]]) -> dict[str, torch.Tensor]:
            data_batch = {}
            data_batch["input_ids"] = torch.Tensor([block["input_ids"] for block in blocks])
            if device:
                data_batch["input_ids"] = data_batch["input_ids"].to(device)
            return data_batch
    
        if dataset_name == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation", cache_dir=dataset_cache_dir)
            prompt_col_name = "text"
        elif dataset_name == "cnn_dailymail":
            dataset = load_dataset("abisee/cnn_dailymail", name="3.0.0", split="train", cache_dir=dataset_cache_dir)
            prompt_col_name = "article"
        elif dataset_name == "wikitext":
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", cache_dir=dataset_cache_dir)
            prompt_col_name = "text"
        else:
            raise NotImplementedError
    
        dataset = dataset.select(
            indices=list(range(min(len(dataset), num_calib_data))),
            keep_in_memory=True,
        )
        tokenized_datasets = dataset.map(
            make_data_block,
            batched=True,
            batch_size=len(dataset),
            num_proc=1,
            remove_columns=dataset.column_names,
            keep_in_memory=True,
            fn_kwargs={"tokenizer": tokenizer, "prompt_col_name": prompt_col_name, "max_length": seqlen},
        )
    
        calib_dataloader = DataLoader(tokenized_datasets, batch_size=batch_size, collate_fn=my_collate_fn)
    
        return calib_dataloader

Let’s create a data reader to load the target dataset.

.. code:: ipython3

    class CalibrationDataReader:
        def __init__(self, dataloader, required_input_names: list[str] | None = None):
            super().__init__()
            self.iterator = iter(dataloader)
            self.required_input_names = set(required_input_names or ["input_ids", "attention_mask"])
    
        def get_next(self) -> dict:
            try:
                inputs = next(self.iterator)[0]
                input_array = inputs.numpy().reshape(1, -1)
                input_dict = {}
                if "input_ids" in self.required_input_names:
                    input_dict["input_ids"] = input_array
                if "attention_mask" in self.required_input_names:
                    input_dict["attention_mask"] = np.ones_like(input_array)
                if "position_ids" in self.required_input_names:
                    input_dict["position_ids"] = np.arange(input_array.shape[-1], dtype=input_array.dtype).reshape(1, -1)
                return input_dict
            except StopIteration:
                return None

4) Quantization Procedure
-------------------------

In this section, we compare three quantization configurations – INT8
baseline, INT8 enhanced with GPTQ, and MalMul-NBits (which is for
transformer-based models with 4-bits GPTQ Quant) – to illustrate how
GPTQ improves accuracy retention during quantization. These
configurations allow us to evaluate the trade-offs between model size,
computational efficiency, and overall accuracy when deploying quantized
models using AMD Quark.

.. code:: ipython3

    def quantize_model(args: dict) -> None:
        # `input_model_path` is the path to the original, unquantized ONNX model.
        input_model_path = args["input_model_path"]
    
        # `output_model_path` is the path where the quantized model will be saved.
        output_model_path = args["output_model_path"]
    
        model_session = ort.InferenceSession(input_model_path, providers=["CPUExecutionProvider"])
        model_input_names = [model_input.name for model_input in model_session.get_inputs()]
        del model_session
    
        tokenizer_class = GPT2Tokenizer
        tokenizer = tokenizer_class.from_pretrained(
            os.path.dirname(input_model_path),
            do_lower_case=False,
            cache_dir=None,
        )
        tokenizer = ensure_padding_token(tokenizer)
    
        # `dr` (Data Reader) is an instance of DataReader, which is a utility class that
        # reads the calibration dataset and prepares it for the quantization process.
        calib_dataloader = get_calib_dataloader(
            dataset_name="pileval", tokenizer=tokenizer, batch_size=1, seqlen=512, device=args["device"]
        )
        calib_dataloader = CalibrationDataReader(calib_dataloader, required_input_names=model_input_names)
        # Get quantization configuration
        quant_config = get_default_config(args["config"])
        config_copy = copy.deepcopy(quant_config)
        config_copy.extra_options["OpTypesToExcludeOutputQuantization"] = ["MatMul", "Gemm"]
    
        if args["config"] == "INT8_TRANSFORMER_DEFAULT":
            config_copy.extra_options["UseGPTQ"] = args["use_gptq"]
        elif args["config"] == "MATMUL_NBITS":
            config_copy.extra_options["MatMulNBitsParams"]["AccuracyLevel"] = 0
            config_copy.extra_options["MatMulNBitsParams"]["Algorithm"] = "GPTQ"
        config_copy.include_cle = False
        config_copy.extra_options["GPTQParams"] = {"MSE": False, "GroupSize": 128, "ActOrder": False, "PerChannel": True}
        config = Config(global_quant_config=config_copy)
        print(f"The configuration for quantization is {config}")
    
        # Create an ONNX quantizer
        quantizer = ModelQuantizer(config)
    
        # Quantize the ONNX model
        quantizer.quantize_model(input_model_path, output_model_path, calib_dataloader)

Create a dedicated folder for INT8 baseline to prevent interference with
other quantization configurations. Then, define a base config for INT8
and apply quantization.

.. code:: ipython3

    !rm -rf quantized_models
    !cp -r models quantized_models
    !rm -f quantized_models/{exported_model_filename}

.. code:: ipython3

    quant_config_int8_only = {
        "input_model_path": exported_model_path,
        "output_model_path": f"quantized_models/{exported_model_filename}",
        "use_gptq": False,
        "num_calib_data": 1000,
        "config": "INT8_TRANSFORMER_DEFAULT",
        "device": "cpu",
        "batch_size": 1,
        "workers": 1,
    }

.. code:: ipython3

    quantize_model(quant_config_int8_only)

Now try INT8 with GPTQ. Create a dedicated folder to prevent
interference with other quantization configurations. Then, define GPTQ
config and apply quantization.

.. code:: ipython3

    !rm -rf gptq_quantized_models
    !cp -r models gptq_quantized_models
    !rm -f gptq_quantized_models/{exported_model_filename}

.. code:: ipython3

    quant_config_int8_gptq = copy.deepcopy(quant_config_int8_only)
    quant_config_int8_gptq["output_model_path"] = f"gptq_quantized_models/{exported_model_filename}"
    quant_config_int8_gptq["use_gptq"] = True
    
    quantize_model(quant_config_int8_gptq)

Last, try MalMul-NBits for INT4 with GPTQ. Again, create a dedicated
folder to prevent interference with other quantization configurations.
Then, define MatMul config and apply quantization.

.. code:: ipython3

    !rm -rf gptq_MATMUL_NBITS_quantized_models
    !cp -r models gptq_MATMUL_NBITS_quantized_models
    !rm -f gptq_MATMUL_NBITS_quantized_models/{exported_model_filename}

.. code:: ipython3

    quant_config_matmul = copy.deepcopy(quant_config_int8_only)
    quant_config_matmul["output_model_path"] = f"gptq_MATMUL_NBITS_quantized_models/{exported_model_filename}"
    quant_config_matmul["config"] = "MATMUL_NBITS"
    
    quantize_model(quant_config_matmul)

5) Evaluation and Expected Results
----------------------------------

Evaluation is performed on the WikiText2 dataset. We compare four models
— (1) full-precision, (2) quantized with INT8, (3) quantized with INT8
and GPTQ, and (4) quantized with MatMul (INT4 and GPTQ). The
full-precision model serves as the baseline for measuring any accuracy
change caused by quantization.

The evaluation metric is Perplexity, which is a standard metric used to
assess how well a language model predicts a sequence of words. It
effectively measures how “surprised” the model is by the test data: -
Low perplexity → model predicts the text well (less surprised) - High
perplexity → model struggles to predict the text (more surprised)

You can think of perplexity as the “average branching factor”—how many
choices the model is effectively considering at each prediction step.

.. code:: ipython3

    from transformers import OPTConfig, OPTForCausalLM, PreTrainedTokenizer
    
    WEIGHTS_NAME = "pytorch_model.bin"
    logger = logging.getLogger(__name__)
    
    MODEL_CLASSES = {
        "opt": (OPTConfig, OPTForCausalLM, GPT2Tokenizer),
    }
    
    
    class TextDataset(Dataset):
        def __init__(self, tokenizer, args, block_size=512):
            testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", cache_dir=dataset_cache_dir)
            text = ""
            for i in testdata:
                text += i["text"]
            self.examples = []
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
    
        def __len__(self):
            return len(self.examples)
    
        def __getitem__(self, item):
            return torch.tensor(self.examples[item])
    
    
    def load_and_cache_examples(args, tokenizer, evaluate=True):
        dataset = TextDataset(
            tokenizer,
            args,
            block_size=args["block_size"],
        )
        return dataset
    
    
    def set_seed(args: str) -> None:
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
    
    
    def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, args["mlm_probability"])
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
    
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
    
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    
    def evaluate_onnx(args, model, tokenizer, prefix=""):
        from torch.nn import CrossEntropyLoss
    
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", cache_dir=dataset_cache_dir)
        test_data = ""
        for i in testdata:
            test_data += i["text"]
    
        eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    
        args["eval_batch_size"] = args["per_gpu_eval_batch_size"]
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])
    
        logger.info(f"***** Running evaluation {prefix} *****")
        eval_loss = 0.0
        nb_eval_steps = 0
    
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, labels = (batch, batch)
            with torch.no_grad():
                outputs = model(input_ids=inputs, attention_mask=inputs.new_ones(inputs.shape))
    
                # Shift so that tokens < n predict n
                lm_logits = outputs[0]
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                lm_loss = loss_fct(shift_logits.float().view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1
    
        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
    
        result = {"perplexity": perplexity}
    
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    
        return result

.. code:: ipython3

    def evaluate(args: dict) -> None:
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    
        # Set seed
        set_seed(args)
    
        # Load pretrained model and tokenizer
        _, _, tokenizer_class = MODEL_CLASSES[args["model_type"]]
    
        tokenizer = tokenizer_class.from_pretrained(
            args["tokenizer_name"] if args.get("tokenizer_name") else args["model_name_or_path"],
            do_lower_case=False,
            cache_dir=None,
        )
        tokenizer = ensure_padding_token(tokenizer)
        tokenizer.add_bos_token = False
        if args["block_size"] <= 0:
            args["block_size"] = (
                tokenizer.max_len_single_sentence
            )  # Our input block size will be the max possible for the model
    
        if args["do_onnx_eval"]:
            logger.info("Evaluate the following onnx model: %s", args["model_name_or_path"])
            prefix = "onnx"
    
            from optimum.onnxruntime import ORTModelForCausalLM
    
            if args.get("no_cuda"):
                provider = "CPUExecutionProvider"
            else:
                provider = "CUDAExecutionProvider"
            model = ORTModelForCausalLM.from_pretrained(
                args["model_name_or_path"], provider=provider, use_cache=False, use_io_binding=False
            )
            result = evaluate_onnx(args, model, tokenizer, prefix=prefix)
            perplexity_value = result["perplexity"].item()
            return perplexity_value

First, define an evaluation config, and record accuracy of the Full
Precision model

.. code:: ipython3

    eval_config = {
        "model_type": "opt",
        "mlm_probability": 0.15,
        "block_size": 2048,
        "per_gpu_eval_batch_size": 1,
        "no_cuda": True,
        "seed": 42,
        "do_onnx_eval": True,
        "eval_data_file": None,
        "config_name": "",
        "tokenizer_name": "",
    }

.. code:: ipython3

    full_precision_eval_config = copy.deepcopy(eval_config)
    full_precision_eval_config["model_name_or_path"] = "models/"
    full_precision_eval_config["onnx_model"] = "models/"
    
    evaluate(full_precision_eval_config)

Then, specify path to the INT8 only quantized model and record its
accuracy

.. code:: ipython3

    int8_quant_eval_config = copy.deepcopy(eval_config)
    int8_quant_eval_config["model_name_or_path"] = "quantized_models/"
    int8_quant_eval_config["onnx_model"] = "quantized_models/"
    
    evaluate(int8_quant_eval_config)

Third, specify path to the INT8 with GPTQ quantized model and record its
accuracy

.. code:: ipython3

    int8_gptq_eval_config = copy.deepcopy(eval_config)
    int8_gptq_eval_config["model_name_or_path"] = "gptq_quantized_models/"
    int8_gptq_eval_config["onnx_model"] = "gptq_quantized_models/"
    
    perplexity = evaluate(int8_gptq_eval_config)

Last, specify path to the MatMul quantized model and record its accuracy

.. code:: ipython3

    matmul_eval_config = copy.deepcopy(eval_config)
    matmul_eval_config["model_name_or_path"] = "gptq_MATMUL_NBITS_quantized_models"
    matmul_eval_config["onnx_model"] = "gptq_MATMUL_NBITS_quantized_models"
    
    evaluate(matmul_eval_config)

The following table contains the expected results, but please note that
different machines can lead to minor variations in the accuracy of
quantized model.

+------------+---------+---------------+--------------------+-----------------+
|            | Float   | INT8          | INT8 + GPTQ        | MatMul          |
|            | Model   | Quantized     | Quantized Model    | Quantized Model |
|            |         | Model         |                    |                 |
+============+=========+===============+====================+=================+
| Model Size | 480 MB  | 384 MB        | 384 MB             | 406 MB          |
+------------+---------+---------------+--------------------+-----------------+
| Perplexity | 27.0317 | 28.6846       | 27.5734            | 30.3604         |
+------------+---------+---------------+--------------------+-----------------+

.. code:: ipython3

    # This cell should have the remove-input tag as we don't want it rendered in the documentation
    # it's creating results for submission to the dashboard
    
    import json
    
    import pandas as pd
    
    model_name = "gptq"
    
    benchmarks = {"gptq": {"opt-125m": 27.5734}}
    
    
    def get_golden_val(benchmarks_dict: dict, model_name: str):
        golden_df = pd.DataFrame.from_dict(benchmarks_dict, orient="index")
        ppl = golden_df.at[model_name, "opt-125m"]
        benchmark = {"perplexity": ppl}
        return benchmark
    
    
    def compare_result(golden: float, current: float):
        result = current - golden
        diff = "no change"
        if golden < current:
            diff = f"rises by {result:.4f}"
        elif golden > current:
            diff = f"drops by {result:.4f}"
        return diff
    
    
    def get_output(benchmark: dict[str, float], current: dict[str, float]):
        try:
            ppl = current["perplexity"]
            diff = compare_result(benchmark["perplexity"], ppl)
        except NameError:
            ppl = "ERROR"
            diff = "ERROR"
        metric_meta = {"golden": benchmark["perplexity"], "difference": diff}
        result = {
            "version": 1,
            "data": [
                {
                    "experiment": {
                        "name": model_name,
                        "model": "opt-125m",
                        "timestamp": str(start_exe_time),
                        "settings": {
                            "quantization_scheme": "INT8",
                            "quantization_algorithm": "GPTQ",
                        },
                    },
                    "metrics": [
                        {
                            "name": "perplexity",
                            "value": round(ppl, 4) if isinstance(ppl, float) else ppl,
                            "metadata": str(metric_meta),
                        },
                    ],
                    "metadata": {"duration": str(datetime.now().replace(second=0, microsecond=0) - start_exe_time)},
                }
            ],
        }
        return result
    
    
    benchmark = get_golden_val(benchmarks, model_name)
    current = {"perplexity": perplexity}
    output = get_output(benchmark, current)
    
    root_path = f"{workspace_dir}/logs_regression_test/"
    os.makedirs(root_path, exist_ok=True)
    file_path = f"{root_path}/{model_name}.json"
    
    with open(file_path, "w") as json_file:
        json.dump(output, json_file)
