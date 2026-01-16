## Rotation training example

This example can be used as a base to train offline or online rotations, for more details please see the documentation at https://quark.docs.amd.com/latest/pytorch/tutorial_rotation.html.

This example showcases training on only a single GPU, but can be extended for multi-GPU training.

In this example, we train an online orthogonal transform $R$. Thus, in a linear layer, we get:

$$
\begin{aligned}
y &= xW^T \\
&= xRR^{-1}W^T \\
&= xRR^TW^T \\
&= xR \times (WR)^T \\
\end{aligned}
$$

The activation rotation $R$ may be left online (as here), or fused into a preceding layer if possible.

Example with MXFP4 quantization (weight MXFP4, activation dynamic MXFP4 quantization):

```bash
export SHORT_CFG_NAME="qwen3_train_r1_128_online_r2"
export STEPS=700
export LR=1.5
export LOSS_TYPE="kl_top_1000"
export EVAL_TASKS="piqa,leaderboard_mmlu_pro,winogrande,arc_challenge,arc_easy,hellaswag,gsm8k_platinum,lambada_standard,ifeval"

export EXP_NAME="$(date +"%Y-%m-%d_%H-%M-%S")_${SHORT_CFG_NAME}_${LOSS_TYPE}_steps_${STEPS}_tbs8_lr${LR}"

CUDA_VISIBLE_DEVICES=5 nohup python train_rotation.py \
    --model_dir Qwen/Qwen3-8B \
    --quant_scheme mxfp4 \
    --eval_batch_size 16 \
    --num_samples 4000 \
    --learning_rate ${LR} \
    --rotation_algo_config_file ./${SHORT_CFG_NAME}.json \
    --export_rotation \
    --output_dir ./${EXP_NAME} \
    --max_steps ${STEPS} \
    --loss_type ${LOSS_TYPE} \
    --tasks ${EVAL_TASKS} \
    --model_attn_implementation sdpa \
    --train_batch_size 8 &> ./${EXP_NAME}.log &
```

## SmoothQuant scales training example

In this example, we train an online transform $O$ parameterized as:

$$
\begin{aligned}
O &= DR \\
\end{aligned}
$$

where $D$ is a diagonal matrix (SmoothQuant scales), and $R$ is an orthogonal matrix. Thus, in a linear layer, we get:

$$
\begin{aligned}
y &= xW^T \\
&= xOO^{-1}W^T \\
&= xDRR^TD^{-1}W^T \\
&= xDR \times (WD^{-1}R)^T \\
&= ... x'R \times (WD^{-1}R)^T
\end{aligned}
$$

fusing $D$ into a preceding layer (e.g. RMSNorm weight, or linear weight). The activation rotation $R$ may be left online, or fused as well if possible.

Adding quantization in, we get:

$$
\begin{aligned}
y = \text{quantize}(x'R) \times \text{quantize}(WD^{-1}R)^T.
\end{aligned}
$$

For more details please refer to https://quark.docs.amd.com/latest/pytorch/tutorial_rotation.html.

The only change compared to the above script is that we use `qwen3_r1_128_online_r2_smooth12.json` to parametrize the `RotationConfig`.

Example:

```bash
export SHORT_CFG_NAME="qwen3_train_r1_128_online_r2_smooth12"
export STEPS=700
export LR=1.5
export LOSS_TYPE="kl_top_1000"
export EVAL_TASKS="piqa,leaderboard_mmlu_pro,winogrande,arc_challenge,arc_easy,hellaswag,gsm8k_platinum,lambada_standard,ifeval"

export EXP_NAME="$(date +"%Y-%m-%d_%H-%M-%S")_${SHORT_CFG_NAME}_${LOSS_TYPE}_steps_${STEPS}_tbs8_lr${LR}"

CUDA_VISIBLE_DEVICES=6 nohup python train_rotation.py \
    --model_dir Qwen/Qwen3-8B \
    --quant_scheme mxfp4 \
    --eval_batch_size 16 \
    --num_samples 4000 \
    --learning_rate ${LR} \
    --rotation_algo_config_file ./${SHORT_CFG_NAME}.json \
    --export_rotation \
    --output_dir ./${EXP_NAME} \
    --max_steps ${STEPS} \
    --loss_type ${LOSS_TYPE} \
    --tasks ${EVAL_TASKS} \
    --model_attn_implementation sdpa \
    --train_batch_size 8 &> ./${EXP_NAME}.log &
```

## Reference round-to-nearest and hadamard rotation results

One can head to https://github.com/amd/Quark/tree/HEAD/examples/torch/language_modeling/llm_ptq to run quantization and evaluation with

- non-quantized model
- round to nearest quantization (no algorithm)
- hadamard rotations (not trained)
- other available algorithms to compare with.

Non-quantized:

```bash
export SHORT_CFG_NAME="qwen3_skip_quantization"
export EVAL_TASKS="piqa,leaderboard_mmlu_pro,winogrande,arc_challenge,arc_easy,hellaswag,gsm8k_platinum,lambada_standard,ifeval"

export EXP_NAME="$(date +"%Y-%m-%d_%H-%M-%S")_${SHORT_CFG_NAME}"

CUDA_VISIBLE_DEVICES=2 nohup python quantize_quark.py \
    --model_dir Qwen/Qwen3-8B \
    --quant_scheme mxfp4 \
    --eval_batch_size 16 \
    --skip_quantization \
    --tasks ${EVAL_TASKS} &> ./${EXP_NAME}.log &
```

Round-to-nearest:

```bash
export SHORT_CFG_NAME="qwen3_rtn"
export EVAL_TASKS="piqa,leaderboard_mmlu_pro,winogrande,arc_challenge,arc_easy,hellaswag,gsm8k_platinum,lambada_standard,ifeval"

export EXP_NAME="$(date +"%Y-%m-%d_%H-%M-%S")_${SHORT_CFG_NAME}"

CUDA_VISIBLE_DEVICES=3 nohup python quantize_quark.py \
    --model_dir Qwen/Qwen3-8B \
    --quant_scheme mxfp4 \
    --eval_batch_size 16 \
    --tasks ${EVAL_TASKS} &> ./${EXP_NAME}.log &
```

Hadamard rotations:

```bash
export SHORT_CFG_NAME="qwen3_hadamard_r1_128_online_r2"
export EVAL_TASKS="piqa,leaderboard_mmlu_pro,winogrande,arc_challenge,arc_easy,hellaswag,gsm8k_platinum,lambada_standard,ifeval"

export EXP_NAME="$(date +"%Y-%m-%d_%H-%M-%S")_${SHORT_CFG_NAME}"

CUDA_VISIBLE_DEVICES=4 nohup python quantize_quark.py \
    --model_dir Qwen/Qwen3-8B \
    --quant_scheme mxfp4 \
    --quant_algo rotation \
    --quant_algo_config_file rotation ../rotation/${SHORT_CFG_NAME}.json \
    --eval_batch_size 16 \
    --tasks ${EVAL_TASKS} &> ./${EXP_NAME}.log &
```

## Example results

Below are example results from the above commands. These results are collected using `transformers==4.57.3`, `amd-quark==0.11`, ROCm 7.1.0, `torch==2.9.0a0+git1c57644` (run within [`rocm/vllm-dev:nightly_main_20251117`](https://hub.docker.com/layers/rocm/vllm-dev/nightly_main_20251117) docker image).

These results give the relative difference compared to the non-quantized model on n-shot tasks using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and on wikitext perplexity.

![Rotation results](./rotation_results.png)

Notes:

- The "mean n-shot" metric is computed as an average of all the n-shot tasks (not weighted).
- Generative tasks (as ifeval, gsm8k) require proper care to meaningfully trust the results (ensure regex filtering is correct, [ensure `max_gen_toks` is large enough](https://github.com/EleutherAI/lm-evaluation-harness/issues/3417#issuecomment-3557259320), [ensure the instruction is correct](https://github.com/EleutherAI/lm-evaluation-harness/pull/3411), ensure thinking tokens are not processed as part of the response, etc.)
- We report `acc_norm` metric if available, otherwise `acc`.
- For `gsm8k_platinum`, we report `exact_match` metric.
- For `ifeval`, we report `inst_level_loose_acc` metric.
