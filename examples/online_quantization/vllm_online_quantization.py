from vllm import LLM, SamplingParams

from quark.online_quantization.vllm import HF_QUANTIZATION_CONFIGS


def main(quant_scheme: str = "ptpc_fp8"):
    model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507"
    hf_overrides = HF_QUANTIZATION_CONFIGS[quant_scheme]

    llm = LLM(
        model=model_name,
        quantization="quark_online",
        enforce_eager=True,
        tensor_parallel_size=1,
        hf_overrides=hf_overrides,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    output = llm.generate(["The capital of France is"], sampling_params=sampling_params)
    print(f"[{quant_scheme}] output = {output}")


if __name__ == "__main__":
    main(quant_scheme="ptpc_fp8")
