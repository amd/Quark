import math

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors


# -----------------------------
# Dataset / Tokenizer
# -----------------------------
def get_tokenizer(model_id: str, max_seq_len: int = 512) -> PreTrainedTokenizer:
    """
    Initializes and returns a tokenizer.
    """
    print(f"Initializing tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        # Ensure pad_token is set, use eos_token if not available
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, "Pad token cannot be set!"
    return tokenizer


def get_dataloader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    nsamples: int,
    device: str | None,
    seq_len: int = 512,
) -> DataLoader:
    """
    Loads the Pile-val dataset and creates a DataLoader for model calibration.
    """
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    text_data = dataset["text"][:nsamples]

    batch_encoded = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
    if device:
        batch_encoded = batch_encoded.to(device)
    # Only input_ids are needed
    input_ids = batch_encoded["input_ids"]

    # Create DataLoader using input_ids
    calib_dataloader = DataLoader(input_ids, batch_size=batch_size, shuffle=False, drop_last=True)

    return calib_dataloader


# -----------------------------
# Model / Quantization
# -----------------------------
def get_model(
    ckpt_path: str,
    data_type: str = "auto",
    device: str = "cuda",
    multi_gpu: bool = False,
    multi_device: bool = False,
    attn_implementation: str = "eager",
    trust_remote_code: bool = True,
) -> tuple[nn.Module, torch.dtype]:
    """
    Loads a pre-trained causal language model.
    """
    if data_type == "float16":
        model_dtype = torch.float16
    elif data_type == "bfloat16":
        model_dtype = torch.bfloat16
    elif data_type == "float32":
        model_dtype = torch.float32
    elif data_type == "auto":
        model_dtype = data_type  # transformers will auto-select based on config
    else:
        raise ValueError(f"{data_type} not supported for current model")

    max_memory = None
    if multi_device or multi_gpu:  # Handle both cases similarly
        device_map = "auto"
    else:
        device_map = device

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        device_map=device_map,
        torch_dtype=model_dtype,
        max_memory=max_memory,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )

    model.config._name_or_path = ckpt_path  # Save original path
    model.eval()
    model_dtype = next(model.parameters()).dtype  # Get the actual data type of the model

    return model, model_dtype


def quantize_model_pipeline(
    model: PreTrainedModel,
    calib_dataloader: DataLoader,
    algorithm: str = "awq",
    scheme: str = "uint4_wo_128",
) -> PreTrainedModel:
    """
    Quantizes the model using the Quark library.
    """
    # If a custom awq_config is not needed, you can omit it and use the default configuration.
    template = LLMTemplate.get(model.config.model_type)
    # Example: Use uint4_wo_128 scheme and awq algorithm for quantization
    quant_config = template.get_config(scheme=scheme, algorithm=[algorithm])
    quantizer = ModelQuantizer(quant_config, multi_device=False)  # Assuming no multi-device quantization here
    quantized_model: PreTrainedModel = quantizer.quantize_model(model, calib_dataloader)

    print("[INFO] Exporting Quant Model.")
    export_safetensors(model=quantized_model, output_dir="./")  # Export quantized model

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
    """
    Evaluates the perplexity (PPL) of the model on the wikitext-2 dataset.
    """
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids.to(device)

    seqlen_for_eval = 2048
    nsamples = testenc.numel() // seqlen_for_eval
    nlls: list[torch.Tensor] = []

    for i in tqdm(range(nsamples)):
        # Extract current batch
        batch = testenc[:, i * seqlen_for_eval : (i + 1) * seqlen_for_eval]
        if batch.shape[1] == 0:  # Skip empty batches
            continue

        # Get model logits
        lm_logits = model(batch)["logits"]

        # Calculate cross-entropy loss
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:]

        loss = torch.nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * seqlen_for_eval)

    if not nlls:  # If nlls is empty, avoid division by zero
        print("[WARNING] No samples processed for PPL evaluation, returning inf.")
        return torch.tensor(float("inf"))

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen_for_eval))
    return ppl


# -----------------------------
# Pipeline
# -----------------------------
def run_quark_awq_example(nsamples: int, batch_size: int) -> torch.Tensor:
    """
    Runs an end-to-end pipeline for Quark AWQ quantization example.
    """
    model_id = "facebook/opt-125m"
    seq_len = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Starting quantization and evaluation for nsamples={nsamples}, batch_size={batch_size}")
    print(f"[INFO] Loading model: {model_id} on device: {device}")

    model, _ = get_model(
        model_id,
        "float32",
        str(device),  # Ensure device is a string
        False,  # multi_gpu
        False,  # multi_device
        "eager",
        trust_remote_code=False,
    )

    tokenizer = get_tokenizer(model_id, max_seq_len=seq_len)

    calib_dataloader = get_dataloader(tokenizer, batch_size, nsamples, str(device), seq_len)

    print("[INFO] Starting quantization...")
    quantized_model = quantize_model_pipeline(model, calib_dataloader, algorithm="awq", scheme="int8")
    print("[INFO] Quantization complete.")

    print("[INFO] Simple test PPL with wikitext-2.")
    ppl = ppl_eval(quantized_model, tokenizer, str(device))
    print(f"[INFO] Perplexity: {ppl.item():.4f}")
    return ppl


def run_quark_autosmoothquant_example(nsamples: int, batch_size: int) -> torch.Tensor:
    """
    Runs an end-to-end pipeline for Quark AutoSmoothQuant quantization example.
    """
    model_id = "amd-quark/tiny-llama-fast-tokenizer"
    seq_len = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Starting quantization and evaluation for nsamples={nsamples}, batch_size={batch_size}")
    print(f"[INFO] Loading model: {model_id} on device: {device}")

    model, _ = get_model(
        model_id,
        "float32",
        str(device),  # Ensure device is a string
        False,  # multi_gpu
        False,  # multi_device
        "eager",
        trust_remote_code=False,
    )

    tokenizer = get_tokenizer(model_id, max_seq_len=seq_len)

    calib_dataloader = get_dataloader(tokenizer, batch_size, nsamples, str(device), seq_len)

    print("[INFO] Starting quantization...")
    quantized_model = quantize_model_pipeline(model, calib_dataloader, algorithm="autosmoothquant", scheme="int8")
    print("[INFO] Quantization complete.")

    print("[INFO] Simple test PPL with wikitext-2.")
    ppl = ppl_eval(quantized_model, tokenizer, str(device))
    print(f"[INFO] Perplexity: {ppl.item():.4f}")
    return ppl


def test_awq_for_pileval():
    with torch.no_grad():
        ppl_batch_1 = run_quark_awq_example(nsamples=128, batch_size=1)
        ppl_batch_4 = run_quark_awq_example(nsamples=128, batch_size=4)
        ppl_batch_128 = run_quark_awq_example(nsamples=128, batch_size=128)

        print("\n--- Summary ---")
        print(f"PPL with batch_size=1: {ppl_batch_1.item():.4f}")
        print(f"PPL with batch_size=4: {ppl_batch_4.item():.4f}")
        print(f"PPL with batch_size=128: {ppl_batch_128.item():.4f}")

        # Evaluate if PPLs are approximately equal, considering floating-point errors
        assert (math.fabs(ppl_batch_1 - ppl_batch_128) / ppl_batch_1) < 0.01


def test_autosmoothquant_for_pileval():
    with torch.no_grad():
        ppl_batch_1 = run_quark_autosmoothquant_example(nsamples=128, batch_size=1)
        ppl_batch_4 = run_quark_autosmoothquant_example(nsamples=128, batch_size=4)
        ppl_batch_128 = run_quark_autosmoothquant_example(nsamples=128, batch_size=128)
        print("\n--- Summary ---")
        print(f"PPL with batch_size=1: {ppl_batch_1.item():.4f}")
        print(f"PPL with batch_size=4: {ppl_batch_4.item():.4f}")
        print(f"PPL with batch_size=128: {ppl_batch_128.item():.4f}")

        # Evaluate if PPLs are approximately equal, considering floating-point errors
        assert (math.fabs(ppl_batch_1 - ppl_batch_128) / ppl_batch_1) < 0.01


if __name__ == "__main__":
    test_awq_for_pileval()
    test_autosmoothquant_for_pileval()
