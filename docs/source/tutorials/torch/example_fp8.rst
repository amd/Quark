FP8 Quantization for LLM models
===============================

.. container:: alert alert-block alert-info

   NOTE This tutorial can be downloaded for local execution on a Jupyter
   Notebook environment. Click here to download the source file.

This tutorial demonstrates how to perform FP8 quantization on Large
Language Models (LLMs) using AMD-Quark. We will guide you through the
following steps: We will guide you through the following steps: 1. Setup
and installation 2. Data preparation 3. Model loading 4. Quantization
process 5. Evaluation

1. Setup
--------

First, ensure you have the necessary libraries installed.

.. code:: console

   pip install torch
   pip install transformers==4.52.1
   pip install tqdm
   pip install datasets
   pip install accelerate

Import the required modules.

.. code:: ipython3

    from typing import Any
    
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
    
    from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors

2. Data Preparation
-------------------

We need a calibration dataset to gather statistics for quantization.
Here we use a subset of the Pile validation set. We also define
functions to initialize the tokenizer and create a dataloader.

.. code:: ipython3

    # -----------------------------
    # Dataset / Tokenizer
    # -----------------------------
    def get_pileval(
        tokenizer: PreTrainedTokenizer,
        nsamples: int,
        seqlen: int,
        device: str | None,
        seed: int = 0,
    ) -> torch.Tensor:
        dataset: Any = load_dataset("mit-han-lab/pile-val-backup", split="validation").shuffle(seed=seed)
        samples, n_run = [], 0
    
        for data in dataset:
            line_encoded = tokenizer.encode(data["text"].strip())
            if 0 < len(line_encoded) <= seqlen:
                samples.append(torch.tensor([line_encoded], device=device))
                n_run += 1
            if n_run == nsamples:
                break
    
        cat_samples = torch.cat(samples, dim=1)
        n_split = cat_samples.shape[1] // seqlen
        train_dataset = [cat_samples[:, i * seqlen : (i + 1) * seqlen] for i in range(n_split)]
    
        return torch.cat(train_dataset, dim=0)
    
    
    def get_tokenizer(model_id: str, max_seq_len: int = 512) -> PreTrainedTokenizer:
        print(f"Initializing tokenizer from {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            model_max_length=max_seq_len,
            padding_side="left",
            trust_remote_code=True,
            use_fast=False,
        )
        if tokenizer.pad_token != "<unk>":
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
        assert tokenizer.pad_token is not None, "Pad token cannot be set!"
        return tokenizer
    
    
    def get_dataloader(
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        device: str | None,
        seq_len: int = 512,
    ) -> DataLoader:
        samples: torch.Tensor = get_pileval(tokenizer, nsamples=128, seqlen=seq_len, device=device, seed=42)
        return DataLoader(samples, batch_size=batch_size, shuffle=False, drop_last=True)

3. Model Loading
----------------

Load the pre-trained model. We configure it to use eager attention
implementation and auto device placement.

.. code:: ipython3

    # -----------------------------
    # Model / Quantization
    # -----------------------------
    def get_model(model_id: str, device: str | None) -> PreTrainedModel:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="eager",
            torch_dtype="auto",
        )
        return model.eval().to(device)

4. Quantization Process
-----------------------

Now we define the quantization pipeline. 1. We fetch the quantization
configuration for the model architecture, specifying ``fp8`` for both
weights and KV cache. 2. We initialize the ``ModelQuantizer`` with this
configuration. 3. We run ``quantize_model`` using the model and
calibration dataloader. 4. Finally, we export the quantized model to the
safetensors format.

.. code:: ipython3

    def quantize_model_pipeline(
        model: PreTrainedModel,
        calib_dataloader: DataLoader,
        tokenizer: PreTrainedTokenizer,
    ) -> PreTrainedModel:
        template = LLMTemplate.get(model.config.model_type)
        quant_config = template.get_config(scheme="fp8", kv_cache_scheme="fp8")
    
        quantizer = ModelQuantizer(quant_config, multi_device=True)
        quantized_model: PreTrainedModel = quantizer.quantize_model(model, calib_dataloader)
    
        print("[INFO] Export Quant Model.")
        export_safetensors(model=quantized_model, output_dir="./")
        tokenizer.save_pretrained("./")
    
        return quantized_model

5. Evaluation
-------------

To verify the quantization quality, we calculate the perplexity (PPL) on
the WikiText-2 test set.

.. code:: ipython3

    # -----------------------------
    # Evaluation
    # -----------------------------
    @torch.no_grad()
    def ppl_eval(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str | None,
    ) -> torch.Tensor:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids.to(device)
    
        seqlen_for_eval = 2048
        nsamples = testenc.numel() // seqlen_for_eval
        nlls: list[torch.Tensor] = []
    
        for i in tqdm(range(nsamples)):
            batch = testenc[:, i * seqlen_for_eval : (i + 1) * seqlen_for_eval]
            lm_logits = model(batch)["logits"]
    
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:]
    
            loss = torch.nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(loss.float() * seqlen_for_eval)
    
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen_for_eval))
        return ppl

6. Execution
------------

Put everything together: define the main execution function and run it.
In this example, we use the ``Qwen/Qwen2.5-0.5B`` model.

.. code:: ipython3

    # -----------------------------
    # Pipeline
    # -----------------------------
    def run_quark_fp8_example() -> None:
        model_id = "Qwen/Qwen2.5-0.5B"
        batch_size, seq_len = 4, 512
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        print(f"[INFO] Loading model: {model_id}")
        model = get_model(model_id, device)
        tokenizer = get_tokenizer(model_id, max_seq_len=seq_len)
        calib_dataloader = get_dataloader(tokenizer, batch_size, device, seq_len)
    
        print("[INFO] Starting quantization...")
        quantized_model = quantize_model_pipeline(model, calib_dataloader, tokenizer)
        print("[INFO] Quantization complete.")
    
        print("[INFO] Simple test PPL with wikitext-2.")
        ppl = ppl_eval(quantized_model, tokenizer, device)
        print(f"[INFO] Perplexity: {ppl.item():.4f}")
    
    
    if __name__ == "__main__":
        with torch.no_grad():
            run_quark_fp8_example()
