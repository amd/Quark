# Best Practices for Quark Online Quantization

This guide is specific to Quark PyTorch online quantization through `quark.online_quantization.vllm`. In this flow, weights are quantized while vLLM loads the model, using either `ptpc_fp8` or `mxfp4`. The goal is to help developers who do not have quantization background choose a configuration that starts conservatively and then expands coverage step by step without taking unnecessary accuracy risk.

> **Scope and applicability**
>
> - **Hardware:** The instructions and experiments in this guide use the AMD Instinct MI300 and MI355 series as concrete examples. This is only for illustration — the same methodology applies to other GPUs. When you deploy on a different GPU, keep the same layer-selection and step-by-step coverage strategy, and only adjust the *format* to what your hardware supports (for example, whether `mxfp4` is natively accelerated).
> - **Beyond online quantization:** Although this guide is written around online quantization in the inference framework (vLLM), the same methodology — how to choose formats and decide which layers to quantize versus keep floating point — is equally useful for finding a better *offline* quantization configuration. The layer-sensitivity reasoning and the conservative-to-aggressive progression carry over directly to offline quantization workflows.

Online quantization is usually configured through vLLM `--additional-config`:

```json
{
  "online_quant_config": {
    "global_quant_config": "ptpc_fp8",
    "layer_quant_config": {},
    "exclude_layer": ["lm_head"]
  }
}
```

The three user-facing fields are:

- `global_quant_config`: The default format. Use only `ptpc_fp8` or `mxfp4`.
- `layer_quant_config`: Per-layer overrides keyed by vLLM layer prefix patterns, for example keeping `*attn*` in `ptpc_fp8`.
- `exclude_layer`: Layers that should not use Quark online quantization. This should at least include `lm_head`; MoE and VLM models usually need a few additional model-specific patterns for sensitive modules.

## Choosing a Format

`ptpc_fp8` is the more conservative format. It uses FP8 E4M3 with per-channel static weight quantization and dynamic activation quantization. It has a wider dynamic range than `mxfp4`. For MoE models, attention should generally remain floating point at first.

`mxfp4` is the aggressive target format. It uses OCP MXFP4 with group size 32 and E8M0 block scales. It is a good fit for MoE experts, where most of the parameters and bandwidth cost live. It can improve memory and bandwidth efficiency, but it is more likely to hurt accuracy on sensitive paths such as attention and shared experts.

### Hardware Support (MI300 vs. MI350/MI355)

Format availability depends on your GPU. `mxfp4` requires native MXFP4 support, which is only available on CDNA4-class GPUs such as the AMD Instinct MI350/MI355. **The AMD Instinct MI300 series does not support MXFP4.**

If you are running on MI300, ignore every `mxfp4` step in this guide and use `ptpc_fp8` only.

However, this only restricts the *format* you can use, not the *layer selection* strategy. The rest of this guide — which layers to quantize and which to exclude (`lm_head`, router/gate, shared experts, attention, vision modules, etc.) — still applies on MI300. When you read the `mxfp4` levels below, keep their `exclude_layer` / `layer_quant_config` layer choices and simply keep `global_quant_config` as `ptpc_fp8` instead of switching to `mxfp4`. In other words, reuse the coverage decisions from the levels below, but express them entirely in `ptpc_fp8`.

## Default Ignore Rules

### Why `lm_head` is ignored by default

`lm_head` maps hidden states directly to vocabulary logits. Quantization error here can change token ranking, which affects generation quality, formatting stability, and reasoning paths. It is usually not the best place to take risk for performance, so the default recommendation is to exclude it:

```json
"exclude_layer": ["lm_head"]
```

### Why gate-like layers are ignored by default

Here, "gate" means router, gating, or control-path modules, not the regular MLP `gate_proj`. These modules often decide information flow or expert selection. They are usually small, but an error in them can be amplified by the downstream compute path. In MoE models, even a small router/gate error can change the top-k expert assignment for a token; in other architectures, it can change branch selection or mixing weights.

A conservative pattern is:

```json
"exclude_layer": ["lm_head", "*.gate.*"]
```

Avoid broad patterns such as `*gate*` unless you have checked that they do not also match MLP `gate_proj`. `gate_proj` is normally an FFN linear layer and can often be tested with `mxfp4`. The pattern above is only a generic example; replace it with the exact router/gate prefix used by your model.

### VLM-specific ignores

Do not apply a text-only LLM policy blindly to a vision-language model. The vision tower, visual encoder, patch embedding, and multi-modal projector have different input distributions and should be evaluated with different metrics. Errors in these modules may not show up in text perplexity but can strongly affect image-text understanding.

A VLM starting point is:

```json
"exclude_layer": [
  "lm_head",
  "*.gate.*",
  "*attn*",
  "<vision_encoder_layer_pattern>",
  "<multi_modal_projector_layer_pattern>"
]
```

Here, `*attn*` is a placeholder for attention modules that should stay floating point. Replace `<vision_encoder_layer_pattern>` and `<multi_modal_projector_layer_pattern>` with the actual vLLM layer prefixes for the model's vision tower, patch embedding, image/video projector, or similar components. Only evaluate projector or vision-module quantization after the language decoder is already healthy.

## MoE Best Practice

MoE models need the most careful configuration. Most parameters are in the experts, while the highest accuracy risk is often in routing, shared experts, and attention. Attention usually accounts for a smaller share of MoE model parameters and compute, but it can have a large impact on quality.

### Choosing where to start by scenario

Instead of always starting from the same place, pick your entry point based on what you care about most:

- **Accuracy-sensitive scenario**: choose the most conservative config. Use `ptpc_fp8` on non-attention linear layers only and keep router/gate, shared experts, and attention out of quantization. This is "Level 0" below and gives the smallest accuracy risk.
- **Latency-sensitive scenario**: choose the most aggressive config. Use global `mxfp4` and exclude only what must stay out (`lm_head` and router/gate). This is "Level 4" below and gives the largest memory/bandwidth savings. This path requires MI350/MI355-class hardware. On MI300, where `mxfp4` is not available as an accelerated format, use global `ptpc_fp8` instead (same `exclude_layer` choices, just `ptpc_fp8` in place of `mxfp4`).
- **Tuning from aggressive back toward conservative**: if the aggressive config does not pass evaluation, walk the levels backward one step at a time, from Level 4 toward Level 0, until accuracy is acceptable. Each step moves the most sensitive remaining path (attention, then shared experts, then regular experts) from `mxfp4`/quantized back to a safer format or to floating point.

The numbered levels below are ordered from most conservative (Level 0) to most aggressive (Level 4). Read them top-down if you are starting conservative, or bottom-up if you are stepping an aggressive config back down.

### Sensitive MoE components

Router/gate layers should be ignored by default because they decide which experts receive each token. Quantization error can change expert selection; even if the local error is small, the downstream computation may follow a different path. The performance benefit is usually much smaller than the risk.

Ignore `shared_experts` in the first stage. Shared experts often serve all or many tokens and behave more like a global FFN or residual path than regular experts. They are more sensitive to distribution shifts, so quantizing them too early can cause broad quality degradation. Start with regular experts only.

Handle attention conservatively. Attention mixes information across tokens, and long-context or reasoning models are especially sensitive to it. In MoE models, attention is usually smaller than the experts, so the quantization benefit is limited while the accuracy risk is high. Keep attention floating point first; if you need to quantize it later, try `ptpc_fp8`, not `mxfp4`.

### Level 0: Quantize only non-attention linear layers to `ptpc_fp8`

Start by applying `ptpc_fp8` to quantizable non-attention linear layers. Exclude `lm_head`, router/gate, shared experts, and attention:

```json
{
  "online_quant_config": {
    "global_quant_config": "ptpc_fp8",
    "exclude_layer": [
      "lm_head",
      "*.gate.*",
      "*shared_expert*",
      "*attn*"
    ]
  }
}
```

This is the MoE accuracy baseline. Attention is intentionally left floating point because its benefit is small and its risk is high. If this configuration does not pass evaluation, do not move to `mxfp4`; first check model loading, the evaluation set, the vLLM version, and your layer patterns.

### Level 1: Move only regular experts to `mxfp4`

The first aggressive step should target regular experts only. Other non-attention linear layers stay on `ptpc_fp8`; shared experts and attention remain excluded:

```json
{
  "online_quant_config": {
    "global_quant_config": "mxfp4",
    "exclude_layer": [
      "lm_head",
      "*.gate.*",
      "*shared_expert*",
      "*attn*"
    ]
  }
}
```

The purpose of this stage is to verify that regular experts can tolerate `mxfp4`.

### Level 2: Quantize all MoE experts to `mxfp4`

After regular experts pass evaluation with `mxfp4`, include shared experts:

```json
{
  "online_quant_config": {
    "global_quant_config": "mxfp4",
    "exclude_layer": [
      "lm_head",
      "*.gate.*",
      "*attn*"
    ]
  }
}
```

This is the "all MoE experts in `mxfp4`" configuration. Attention still stays floating point through `exclude_layer`, so the experiment isolates risk to the MoE blocks.

### Level 3: Try `ptpc_fp8` for attention

If all MoE experts pass with `mxfp4` and you still want to reduce floating-point paths, move attention from floating point to `ptpc_fp8`.

```json
{
  "online_quant_config": {
    "global_quant_config": "ptpc_fp8",
    "layer_quant_config": {
      "*experts*": "mxfp4"
    },
    "exclude_layer": [
      "lm_head",
      "*.gate.*"
    ]
  }
}
```

### Level 4: Use `mxfp4` globally

Only at the end should you try `mxfp4` as the global format. At this stage, attention is also covered by `mxfp4`, so this should be treated as the highest-risk configuration and validated carefully:

```json
{
  "online_quant_config": {
    "global_quant_config": "mxfp4",
    "exclude_layer": [
      "lm_head",
      "*.gate.*"
    ]
  }
}
```

## Dense LLM Recommendations

Dense decoder-only models do not have expert routing, so the progression is simpler:

1. Start with global `ptpc_fp8` and exclude `lm_head`.
2. If accuracy is acceptable, try global `mxfp4`.
3. If global `mxfp4` hurts accuracy, override the model's attention pattern back to `ptpc_fp8`.

Safe starting point:

```json
{
  "online_quant_config": {
    "global_quant_config": "ptpc_fp8",
    "exclude_layer": ["lm_head"]
  }
}
```

More aggressive version:

```json
{
  "online_quant_config": {
    "global_quant_config": "mxfp4",
    "exclude_layer": ["lm_head"]
  }
}
```

## VLM Recommendations

For VLMs, quantize the language decoder first. Follow the Dense or MoE strategy above for the language part, but explicitly exclude the vision side:

```json
{
  "online_quant_config": {
    "global_quant_config": "ptpc_fp8",
    "exclude_layer": [
      "lm_head",
      "<vision_encoder_layer_pattern>",
      "<multi_modal_projector_layer_pattern>"
    ]
  }
}
```

Replace `<vision_encoder_layer_pattern>` and `<multi_modal_projector_layer_pattern>` with the actual vLLM layer prefixes for the model. The vision tower, patch embedding, and image/video projector have distributions that differ from the language decoder, so text perplexity alone is not enough to judge whether they are safe to quantize. If you later want to quantize those modules, validate them separately on image-text tasks.
