#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Evaluation utilities for mix precision with offline vLLM eval helpers."""

__all__ = ["evaluate_gsm8k_offline", "evaluate_ppl_offline"]

import ast
import json
import math
import os
import re
import string
import time
from collections.abc import Generator
from typing import Any, cast

import requests

from quark.common.utils.import_utils import is_datasets_available, is_vllm_available
from quark.common.utils.log import ScreenLogger

if is_vllm_available():  # pragma: no cover
    from vllm import SamplingParams

    try:
        from vllm.inputs import TokensPrompt
    except ImportError:
        TokensPrompt = None  # type: ignore[assignment,misc]

if is_datasets_available():  # pragma: no cover
    from datasets import load_dataset

logger = ScreenLogger(__name__)

INVALID = -9999999

# ==========================================
# GSM8K extraction filter configs (matching lm-eval harness)
# ==========================================

FILTERS = {
    "strict-match": {"regex_pattern": r"#### (\-?[0-9\.\,]+)"},
    "flexible-extract": {
        # select the last match
        "regex_pattern": r"(-?[$0-9.,]{2,})|(-?[0-9]+)",
        "group_select": -1,
    },
}

METRIC_CONFIG = {
    "ignore_case": True,
    "ignore_punctuation": False,
    "regexes_to_ignore": [
        ",",  # strip thousands separator
        "\\$",  # strip dollar sign
        "(?s).*#### ",  # strip everything before and including "#### " (cleans Reference)
        "\\.$",  # strip trailing period
    ],
}

# 8-shot few-shot prompt for GSM8K (aligned with lm-evaluation-harness)
GSM8K_FEWSHOT_PROMPT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6. #### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9

Question: There were 9 computers in the server room. Five more computers were installed each day for 4 days. How many computers are now in the server room?
Answer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. #### 29

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. #### 8

Question: """

# Stop strings for GSM8K generation, matching lm-evaluation-harness gsm8k.yaml `until`.
GSM8K_STOP_STRINGS = ["Question:", "</s>", "<|im_end|>"]


def _clean_text(text: str) -> str:
    """Preprocess text to match lm_eval.metrics.exact_match_hf_evaluate logic."""
    if text is None:
        return ""
    # Apply regex substitutions (strips everything before "#### " in Reference)
    for pattern in cast(list[str], METRIC_CONFIG["regexes_to_ignore"]):
        text = re.sub(pattern, "", text)
    if METRIC_CONFIG["ignore_case"]:
        text = text.lower()
    if METRIC_CONFIG["ignore_punctuation"]:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        text = text.translate(repl_table)
    return text.strip()


def extract_strict(text: str) -> str | None:
    """Strict-match extraction: find the number immediately after '#### ' (instruction-tuned models)."""
    if not text:
        return None
    match = re.search(FILTERS["strict-match"]["regex_pattern"], text)
    return match.group(1) if match else None


def extract_flexible(text: str) -> str | None:
    """Flexible-extract: take the last numeric pattern in the text (fallback strategy)."""
    if not text:
        return None
    matches = re.findall(FILTERS["flexible-extract"]["regex_pattern"], text)
    if not matches:
        return None
    # group_select=-1: take last match; pick first non-empty group from the tuple
    last_match = matches[-1]
    if isinstance(last_match, tuple):
        valid_groups = [m for m in last_match if m]
        return valid_groups[0] if valid_groups else None
    return last_match


def calculate_exact_match(extracted_prediction: str | None, raw_reference: str) -> bool:
    """Compare extracted prediction against raw reference via string equality (matches lm-eval)."""
    if extracted_prediction is None:
        return False
    clean_ref = _clean_text(raw_reference)
    clean_pred = _clean_text(extracted_prediction)
    return clean_pred == clean_ref


def evaluate_gsm8k_entry(
    model_output: str,
    dataset_answer: str,
    strategy: str = "flexible",
) -> tuple[bool, str | None]:
    """Evaluate a single GSM8K sample.

    strategy:
        - "strict"  : match only the number after ####
        - "flexible": match the last number in the text (default)
        - "hybrid"  : try strict first, fall back to flexible
    """
    if strategy == "strict":
        extracted = extract_strict(model_output)
    elif strategy == "flexible":
        extracted = extract_flexible(model_output)
    else:  # hybrid
        extracted = extract_strict(model_output)
        if extracted is None:
            extracted = extract_flexible(model_output)
    return calculate_exact_match(extracted, dataset_answer), extracted


def _truncate_at_stop_strings(text: str, stop_strings: list[str]) -> str:
    """Truncate text at the first stop string, mimicking lm-evaluation-harness `until` behavior."""
    min_pos = len(text)
    for stop in stop_strings:
        pos = text.find(stop)
        if pos != -1 and pos < min_pos:
            min_pos = pos
    return text[:min_pos].strip()


def _gsm8k_evaluate_outputs(
    generated_texts: list[str],
    reference_answers: list[str],
    questions: list[str] | None = None,
    verbose: bool = True,
    max_errors_to_print: int = 5,
) -> tuple[int, int]:
    """Evaluate a batch of GSM8K generations. Returns (correct_count, total_count)."""
    correct_count = 0
    total_count = 0
    error_count = 0

    for i, gen_text in enumerate(generated_texts):
        ref_answer = reference_answers[i]
        # Truncate at stop strings to avoid extracting numbers from spurious continuations
        gen_text_truncated = _truncate_at_stop_strings(gen_text, GSM8K_STOP_STRINGS)
        is_correct, extracted = evaluate_gsm8k_entry(gen_text_truncated, ref_answer, strategy="flexible")

        if is_correct:
            correct_count += 1
        else:
            error_count += 1
            if verbose and error_count <= max_errors_to_print:
                logger.debug(f"\n[gsm8k eval error {error_count}]")
                if questions is not None:
                    logger.debug(f"  Question: {questions[i][:100]}...")
                logger.debug(f"  Reference: {ref_answer}")
                logger.debug(f"  Prediction: {gen_text_truncated}")
                logger.debug(f"  Extracted: {extracted}")
        total_count += 1

    return correct_count, total_count


def _download_and_cache_file(url: str, filename: str | None = None) -> str:
    """Download and cache a remote file to /tmp."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])
    if os.path.exists(filename):
        return filename
    logger.info(f"Downloading {url} to {filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    return filename


def _load_gsm8k_data() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load GSM8K train and test splits."""
    train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    train_file = _download_and_cache_file(train_url)
    test_file = _download_and_cache_file(test_url)
    train_data = list(_read_jsonl(train_file))
    test_data = list(_read_jsonl(test_file))
    return train_data, test_data


def _read_jsonl(filename: str) -> Generator[dict[str, str], None, None]:
    """Yield records from a JSONL file, skipping comment lines."""
    with open(filename) as fin:
        for line in fin:
            if not line.startswith("#"):
                yield json.loads(line)


def _get_answer_value(answer_str: str) -> int:
    """Extract a numeric answer from a string; returns INVALID on failure."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def _build_gsm8k_prompts(
    num_questions: int = 1319,
    num_shots: int = 5,
) -> tuple[list[str], list[int], list[str]]:
    """Build few-shot GSM8K prompts, integer ground-truth labels, and raw reference answers."""
    if num_questions == 0:
        return [], [], []
    train_data, test_data = _load_gsm8k_data()
    num_questions = min(num_questions, len(test_data))

    few_shot_examples = ""
    for i in range(num_shots):
        few_shot_examples += f"Question: {train_data[i]['question']}\nAnswer: {train_data[i]['answer']}\n\n"

    prompts = []
    labels = []
    references = []
    for i in range(num_questions):
        prompts.append(few_shot_examples + f"Question: {test_data[i]['question']}\nAnswer:")
        labels.append(_get_answer_value(test_data[i]["answer"]))
        references.append(test_data[i]["answer"])

    assert all(label != INVALID for label in labels), "Some ground-truth answers could not be parsed"
    return prompts, labels, references


def _score_gsm8k(
    states: list[str],
    output_tokens: list[int],
    labels: list[int],
    references: list[str],
    num_shots: int,
    max_tokens: int,
    latency: float,
) -> dict[str, float | int]:
    """Score GSM8K generations and return a results dict with accuracy, latency, etc.

    Accuracy uses the lm-evaluation-harness flexible-extract filter (last numeric
    match, with thousands-separator / dollar-sign / trailing-period normalization)
    so results align with `lm_eval --model vllm --tasks gsm8k` (flexible-extract).
    """
    num_questions = len(labels)
    correct_count, _ = _gsm8k_evaluate_outputs(states, references, verbose=False)
    accuracy = correct_count / num_questions if num_questions else 0.0
    invalid_count = sum(1 for state in states if extract_flexible(state) is None)
    invalid_rate = invalid_count / num_questions if num_questions else 0.0
    total_output_tokens = sum(output_tokens)
    tokens_per_second = total_output_tokens / latency if latency > 0 else 0.0

    return {
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
        "latency": latency,
        "questions_per_second": num_questions / latency if latency > 0 else 0.0,
        "total_output_tokens": total_output_tokens,
        "tokens_per_second": tokens_per_second,
        "num_questions": num_questions,
        "num_shots": num_shots,
        "max_tokens": max_tokens,
        "timestamp": time.time(),
    }


def evaluate_gsm8k_offline(
    llm: Any,
    num_questions: int = 1319,
    num_shots: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, float | int]:
    """Evaluate GSM8K accuracy using an offline vllm.LLM object.

    Same prompts and scoring as vLLM's gsm8k_eval, but runs generation
    directly via llm.generate() instead of calling a server over HTTP.

    Returns dict with accuracy, invalid_rate, latency, etc.
    """
    prompts, labels, references = _build_gsm8k_prompts(num_questions, num_shots)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=GSM8K_STOP_STRINGS,
    )

    logger.info(f"Running offline GSM8K evaluation: {len(prompts)} questions, {num_shots}-shot")

    tic = time.perf_counter()
    # use_tqdm=False avoids a vLLM ZeroDivisionError (vllm-project/vllm#28097):
    # when its progress bar is disabled (e.g. non-tty CI logs), pbar elapsed is
    # exactly 0 and _run_engine divides token counts by it.
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    latency = time.perf_counter() - tic

    states = [o.outputs[0].text for o in outputs]
    output_tokens = [len(o.outputs[0].token_ids) for o in outputs]

    return _score_gsm8k(states, output_tokens, labels, references, num_shots, max_tokens, latency)


def evaluate_ppl_offline(
    llm: Any,
    seq_len: int = 2048,
    max_chunks: int | None = None,
) -> dict[str, float | int]:
    """Compute wikitext-2 perplexity with an offline vllm.LLM.

    Uses prompt_logprobs=1 so vLLM returns the logprob of each token in the
    prompt under the model's distribution. Matches the protocol of
    quark.contrib.llm_eval.evaluation.ppl_eval: tokenize the joined wikitext
    test split, split into non-overlapping chunks of ``seq_len`` tokens, run
    each chunk through the model, sum NLL of prediction positions, and
    exponentiate the mean.

    Args:
        llm: vllm.LLM instance.
        seq_len: chunk length used as both prompt length and PPL window.
        max_chunks: cap on number of chunks (None = all). Each chunk contains
            ``seq_len`` tokens of context.

    Returns:
        dict with keys: ppl, nll_total, tokens_scored, num_chunks, latency.
    """
    tokenizer = llm.get_tokenizer()
    logger.info("Loading wikitext-2 test split for PPL evaluation")
    try:
        testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        logger.warning(f"Loading 'Salesforce/wikitext' failed ({e}); falling back to legacy 'wikitext'")
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    input_ids = encodings.input_ids[0].tolist()
    total_tokens = len(input_ids)
    num_chunks = total_tokens // seq_len
    if num_chunks == 0:
        raise RuntimeError(f"wikitext test only produced {total_tokens} tokens, less than seq_len={seq_len}")
    if max_chunks is not None and max_chunks > 0:
        num_chunks = min(num_chunks, max_chunks)

    prompts_token_ids = [input_ids[i * seq_len : (i + 1) * seq_len] for i in range(num_chunks)]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
        detokenize=False,
    )

    logger.info(
        f"Running offline PPL evaluation: {num_chunks} chunks x {seq_len} tokens "
        f"(total scored ~= {num_chunks * (seq_len - 1)})"
    )

    tic = time.perf_counter()
    if TokensPrompt is not None:
        prompts_arg = [TokensPrompt(prompt_token_ids=ids) for ids in prompts_token_ids]
    else:
        prompts_arg = [{"prompt_token_ids": ids} for ids in prompts_token_ids]
    # use_tqdm=False avoids a vLLM ZeroDivisionError (vllm-project/vllm#28097);
    # see evaluate_gsm8k_offline for details.
    outputs = llm.generate(
        prompts_arg,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    latency = time.perf_counter() - tic

    nll_total = 0.0
    tokens_scored = 0
    for chunk_ids, out in zip(prompts_token_ids, outputs, strict=False):
        prompt_logprobs = out.prompt_logprobs
        if prompt_logprobs is None:
            raise RuntimeError(
                "vLLM did not return prompt_logprobs; ensure SamplingParams(prompt_logprobs=1) "
                "is supported by this vLLM version."
            )
        # prompt_logprobs[0] is None (no preceding context for first token).
        for tok_id, lp_dict in zip(chunk_ids[1:], prompt_logprobs[1:], strict=False):
            if lp_dict is None:
                continue
            entry = lp_dict.get(tok_id)
            if entry is None:
                # vLLM may return the top-1 token rather than the actual token;
                # fall back to the only entry available.
                if len(lp_dict) == 0:
                    continue
                entry = next(iter(lp_dict.values()))
            logprob_val = entry.logprob if hasattr(entry, "logprob") else float(entry)
            if logprob_val is None or math.isinf(logprob_val) or math.isnan(logprob_val):
                continue
            nll_total += -float(logprob_val)
            tokens_scored += 1

    if tokens_scored == 0:
        raise RuntimeError("PPL eval scored 0 tokens; check vLLM prompt_logprobs output")

    ppl = math.exp(nll_total / tokens_scored)
    logger.info(
        f"Wikitext-2 PPL (vLLM): {ppl:.4f} | chunks: {num_chunks} | "
        f"scored tokens: {tokens_scored} | latency: {latency:.2f}s"
    )

    return {
        "ppl": ppl,
        "nll_total": nll_total,
        "tokens_scored": tokens_scored,
        "num_chunks": num_chunks,
        "latency": latency,
    }
