AWQ end-to-end example
======================

This is a simple example of the AWQ algorithm. Before running this
script, make sure that amd-quark has been properly installed.

.. code:: shell

   !pip install torch

   !pip install transformers==4.52.1

   !pip install tqdm

   !pip install datasets

   !pip install accelerate

For the AWQ algorithm, we provide default configurations for common
models. For advanced users who want to use their own AWQ configuration,
a configuration JSON file needs to be provided. In this ipynb example,
we generate an AWQ configuration JSON file using Python.

.. code:: ipython3

    import json
    
    # Define the configuration to be written
    awq_config = {
        "name": "awq",
        "scaling_layers": [
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        "model_decoder_layers": "model.layers",
    }
    
    
    # Write configuration to a JSON file
    with open("custom_awq_config.json", "w") as f:
        json.dump(awq_config, f, indent=4)
    
    print("custom_awq_config.json has been created.")

This is an example of using the AWQ algorithm integrated in Quark to
quantize an LLM model. The example includes quantization, export and
simple testing.

.. code:: ipython3

    from typing import Any, Optional
    
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
    
    from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors
    from quark.torch.quantization.config.config import load_quant_algo_config_from_file
    
    
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
    
    
    # -----------------------------
    # Model / Quantization
    # -----------------------------
    def get_model(model_id: str, device: str | None) -> PreTrainedModel:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="eager",
        )
        return model.eval().to(device)
    
    
    def quantize_model_pipeline(
        model: PreTrainedModel,
        calib_dataloader: DataLoader,
    ) -> PreTrainedModel:
        # Load custom AWQ config
        custom_awq_config = load_quant_algo_config_from_file("custom_awq_config.json")
        # If you don't need a custom awq_config, you can omit it and use the default configuration.
        template = LLMTemplate(
            model_type=model.config.model_type,
            exclude_layers_name=["lm_head"],
            awq_config=custom_awq_config,
        )
        quant_config = template.get_config(scheme="uint4_wo_128", algorithm=["awq"])
    
        quantizer = ModelQuantizer(quant_config, multi_device=True)
        quantized_model: PreTrainedModel = quantizer.quantize_model(model, calib_dataloader)
    
        print("[INFO] Export Quant Model.")
        export_safetensors(model=quantized_model, output_dir="./")
    
        return quantized_model
    
    
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
    
    
    # -----------------------------
    # Pipeline
    # -----------------------------
    def run_quark_awq_example() -> None:
        model_id = "Qwen/Qwen2.5-0.5B"
        batch_size, seq_len = 4, 512
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        print(f"[INFO] Loading model: {model_id}")
        model = get_model(model_id, device)
        tokenizer = get_tokenizer(model_id, max_seq_len=seq_len)
        calib_dataloader = get_dataloader(tokenizer, batch_size, device, seq_len)
    
        print("[INFO] Starting quantization...")
        quantized_model = quantize_model_pipeline(model, calib_dataloader)
        print("[INFO] Quantization complete.")
    
        print("[INFO] Simple test PPL with wikitext-2.")
        ppl = ppl_eval(quantized_model, tokenizer, device)
        print(f"[INFO] Perplexity: {ppl.item():.4f}")
    
    
    if __name__ == "__main__":
        with torch.no_grad():
            run_quark_awq_example()
