#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
LoRA Fine-Tuning on PTQ Model with Knowledge Distillation from BF16 Teacher

Three distillation modes:
  1. output   — KL-divergence on final logits
  2. layer    — MSE on hidden states at each transformer layer + KL on logits
  3. attention — MSE on hidden states + KL on attention maps + KL on logits

The teacher is the original (unquantized) BF16 model.
The student is the PTQ model with LoRA adapters.
"""

import argparse
import csv
import gc
import math
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from quark.common.utils.log import ScreenLogger
from quark.torch.utils.llm.model_preparation import get_model, prepare_for_moe_quant

logger = ScreenLogger(__name__)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class WikiTextLMDataset(Dataset):
    def __init__(self, tokenizer, split: str, seq_len: int, max_samples: int | None = None):
        super().__init__()
        self.seq_len = seq_len

        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(raw["text"])
        tokenized = tokenizer(text, return_attention_mask=False, return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze(0)

        total_len = (input_ids.shape[0] // seq_len) * seq_len
        self.input_ids = input_ids[:total_len].view(-1, seq_len)

        if max_samples is not None and max_samples > 0:
            self.input_ids = self.input_ids[:max_samples]

        logger.info(f"[KD] WikiText split={split}, samples={len(self.input_ids)}, seq_len={seq_len}")

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int):
        ids = self.input_ids[idx]
        return {"input_ids": ids, "labels": ids.clone()}


class PileLMDataset(Dataset):
    """Diverse training data from The Pile (mit-han-lab/pile-val-backup).

    Concatenates text from the Pile validation set and chunks into
    fixed-length sequences, same as WikiTextLMDataset.  The Pile covers
    many domains (news, books, code, academic papers, web) which prevents
    catastrophic forgetting during LoRA fine-tuning.
    """

    def __init__(self, tokenizer, seq_len: int, max_samples: int | None = None, max_raw_samples: int = 50000):
        super().__init__()
        self.seq_len = seq_len

        raw = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        if max_raw_samples and len(raw) > max_raw_samples:
            raw = raw.select(range(max_raw_samples))

        text = "\n\n".join(raw["text"])
        tokenized = tokenizer(text, return_attention_mask=False, return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze(0)

        total_len = (input_ids.shape[0] // seq_len) * seq_len
        self.input_ids = input_ids[:total_len].view(-1, seq_len)

        if max_samples is not None and max_samples > 0:
            self.input_ids = self.input_ids[:max_samples]

        logger.info(f"[KD] Pile samples={len(self.input_ids)}, seq_len={seq_len} (from {len(raw)} raw entries)")

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int):
        ids = self.input_ids[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _format_arc_examples(dataset):
    """Format ARC-Challenge training examples as reasoning prompts."""
    texts = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
    for ex in dataset:
        q = ex["question"]
        choices = ex["choices"]
        answer_key = ex["answerKey"]
        answer_idx = label_map.get(answer_key, 0)
        prompt = f"Question: {q}\n"
        for i, (label, text) in enumerate(zip(choices["label"], choices["text"], strict=False)):
            prompt += f"{label}. {text}\n"
        if answer_idx < len(choices["text"]):
            prompt += f"Answer: {choices['label'][answer_idx]}. {choices['text'][answer_idx]}\n"
        texts.append(prompt)
    return texts


def _format_hellaswag_examples(dataset):
    """Format HellaSwag training examples as commonsense completion prompts."""
    texts = []
    for ex in dataset:
        ctx = ex["ctx"]
        label = int(ex["label"]) if ex["label"] != "" else 0
        endings = ex["endings"]
        if label < len(endings):
            texts.append(f"{ctx} {endings[label]}")
        prompt_all = f"Context: {ctx}\nPossible completions:\n"
        for i, ending in enumerate(endings):
            prompt_all += f"  {i + 1}. {ending}\n"
        if label < len(endings):
            prompt_all += f"Best completion: {label + 1}. {endings[label]}\n"
        texts.append(prompt_all)
    return texts


def _format_winogrande_examples(dataset):
    """Format WinoGrande training examples as pronoun resolution prompts."""
    texts = []
    for ex in dataset:
        sentence = ex["sentence"]
        opt1 = ex["option1"]
        opt2 = ex["option2"]
        answer = ex["answer"]
        correct = opt1 if answer == "1" else opt2
        filled = sentence.replace("_", correct)
        texts.append(filled)
        prompt = f"Sentence: {sentence}\nOption 1: {opt1}\nOption 2: {opt2}\nCorrect: {correct}\n"
        texts.append(prompt)
    return texts


def _format_boolq_examples(dataset):
    """Format BoolQ training examples as yes/no reading comprehension prompts."""
    texts = []
    for ex in dataset:
        passage = ex["passage"]
        question = ex["question"]
        answer = "Yes" if ex["answer"] else "No"
        texts.append(f"Passage: {passage}\nQuestion: {question}\nAnswer: {answer}\n")
        texts.append(f"{passage}\n\n{question}? {answer}.")
    return texts


def _format_openbookqa_examples(dataset):
    """Format OpenBookQA training examples as science reasoning prompts."""
    texts = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
    for ex in dataset:
        q = ex["question_stem"]
        choices = ex["choices"]
        answer_key = ex["answerKey"]
        answer_idx = label_map.get(answer_key, 0)
        prompt = f"Question: {q}\n"
        for label, text in zip(choices["label"], choices["text"], strict=False):
            prompt += f"{label}. {text}\n"
        if answer_idx < len(choices["text"]):
            prompt += f"Answer: {choices['label'][answer_idx]}. {choices['text'][answer_idx]}\n"
        texts.append(prompt)
    return texts


def _format_sciq_examples(dataset):
    """Format SciQ training examples as science QA prompts with support context."""
    texts = []
    for ex in dataset:
        q = ex["question"]
        correct = ex["correct_answer"]
        support = ex.get("support", "")
        if support:
            texts.append(f"Context: {support}\nQuestion: {q}\nAnswer: {correct}\n")
        texts.append(f"Question: {q}\nAnswer: {correct}\n")
    return texts


def _format_commonsenseqa_examples(dataset):
    """Format CommonsenseQA training examples as multiple-choice reasoning prompts."""
    texts = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    for ex in dataset:
        q = ex["question"]
        choices = ex["choices"]
        answer_key = ex["answerKey"]
        answer_idx = label_map.get(answer_key, 0)
        prompt = f"Question: {q}\n"
        for label, text in zip(choices["label"], choices["text"], strict=False):
            prompt += f"{label}. {text}\n"
        if answer_idx < len(choices["text"]):
            prompt += f"Answer: {choices['label'][answer_idx]}. {choices['text'][answer_idx]}\n"
        texts.append(prompt)
    return texts


def _format_piqa_examples(dataset):
    """Format PIQA training examples as physical intuition / goal-solution prompts."""
    texts = []
    for ex in dataset:
        goal = ex["goal"]
        sol1 = ex["sol1"]
        sol2 = ex["sol2"]
        label = ex["label"]  # 0 or 1
        correct = sol1 if label == 0 else sol2
        texts.append(f"Goal: {goal}\nSolution 1: {sol1}\nSolution 2: {sol2}\nBest solution: {correct}\n")
        texts.append(f"{goal} {correct}")
    return texts


def _format_siqa_examples(dataset):
    """Format Social IQA training examples as social reasoning prompts."""
    texts = []
    for ex in dataset:
        ctx = ex["context"]
        question = ex["question"]
        a1 = ex["answerA"]
        a2 = ex["answerB"]
        a3 = ex["answerC"]
        label = int(ex["label"]) - 1  # 1-indexed -> 0-indexed
        answers = [a1, a2, a3]
        correct = answers[label] if 0 <= label < 3 else a1
        prompt = f"Context: {ctx}\nQuestion: {question}\nA. {a1}\nB. {a2}\nC. {a3}\nAnswer: {correct}\n"
        texts.append(prompt)
    return texts


def _format_copa_examples(dataset):
    """Format COPA training examples as causal reasoning prompts."""
    texts = []
    for ex in dataset:
        premise = ex["premise"]
        choice1 = ex["choice1"]
        choice2 = ex["choice2"]
        question = ex["question"]  # "cause" or "effect"
        label = ex["label"]  # 0 or 1
        correct = choice1 if label == 0 else choice2
        q_word = "cause" if question == "cause" else "effect"
        texts.append(f"Premise: {premise}\nWhat is the {q_word}?\n1. {choice1}\n2. {choice2}\nAnswer: {correct}\n")
        texts.append(f"{premise} The {q_word} is: {correct}")
    return texts


def _format_mmlu_examples(dataset):
    """Format MMLU examples as multi-subject multiple-choice QA prompts."""
    texts = []
    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    for ex in dataset:
        q = ex["question"]
        choices = ex["choices"]
        answer_idx = int(ex["answer"])
        subject = ex.get("subject", "general")
        prompt = f"Subject: {subject}\nQuestion: {q}\n"
        for i, c in enumerate(choices):
            prompt += f"{label_map[i]}. {c}\n"
        if 0 <= answer_idx < len(choices):
            prompt += f"Answer: {label_map[answer_idx]}. {choices[answer_idx]}\n"
        texts.append(prompt)
    return texts


def _format_gsm8k_examples(dataset, include_cot: bool = True):
    """Format GSM8K as Q + chain-of-thought answer.

    include_cot=True: full concise reasoning ending with #### answer.
    include_cot=False: just the final numeric answer (reduces verbosity).
    The CoT format matches GSM8K eval's expected style (concise step-by-step).
    """
    texts = []
    for ex in dataset:
        q = ex["question"]
        ans = ex["answer"]
        if include_cot:
            texts.append(f"Question: {q}\nAnswer: {ans}\n")
        else:
            final = ans.split("####")[-1].strip() if "####" in ans else ans
            texts.append(f"Question: {q}\nAnswer: The answer is {final}.\n")
    return texts


def _extract_final_number(text: str):
    """Extract the last numeric value from a math solution string.

    Returns the number as a string, or None if no number found.
    Handles integers, decimals, negatives, and common formats like $X or X%.
    """
    import re

    patterns = [
        r"####\s*(\-?\$?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"[Tt]he answer is[:\s]*\$?(\-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"=\s*\$?(\-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).replace(",", "").replace("$", "")
    nums = re.findall(r"\-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


MATH_TASK_KEYS = {
    "svamp",
    "metamathqa",
    "mathinstruct",
    "orcamath",
    "openthoughts",
    "reasoning_jsonl",
    "gptoss_bf16_traces",
    "phi4_bf16_traces",
}


def _postprocess_normalize_math_answers(texts: list, max_reasoning_chars: int = 1500) -> list:
    """Post-process math training texts to append #### <number> if missing.

    Applied optionally via --normalize_math_answers flag. Truncates long
    reasoning and ensures every math example ends with GSM8K-compatible format.
    Does not modify texts that already contain ####.
    """
    normalized = []
    for text in texts:
        if "####" in text:
            if len(text) > max_reasoning_chars + 500:
                parts = text.rsplit("####", 1)
                cot = parts[0][:max_reasoning_chars]
                text = cot.rstrip() + "\n#### " + parts[1].strip() + "\n"
            normalized.append(text)
            continue
        num = _extract_final_number(text)
        if num:
            if len(text) > max_reasoning_chars:
                text = text[:max_reasoning_chars]
            text = text.rstrip() + f"\n#### {num}\n"
        normalized.append(text)
    return normalized


def _format_gptoss_bf16_traces(jsonl_path: str, max_examples: int = 10000):
    """Load BF16-generated reasoning traces in GSM8K-compatible format.

    These traces are generated by the GPToss BF16 teacher model using 5-shot
    prompting, so they're in the model's natural style with #### answers.
    """
    import json

    texts = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = ex.get("messages", [])
            user_msg = ""
            assistant_msg = ""
            for m in msgs:
                if m.get("role") == "user":
                    user_msg = m.get("content", "").strip()
                elif m.get("role") == "assistant":
                    assistant_msg = m.get("content", "").strip()
            if user_msg and assistant_msg:
                texts.append(f"Question: {user_msg}\nAnswer: {assistant_msg}\n")
    return texts


def _format_aqua_rat_examples(dataset, max_examples=10000):
    """Format AQuA-RAT algebraic word problems with rationales."""
    texts = []
    for i, ex in enumerate(dataset):
        if i >= max_examples:
            break
        q = ex["question"]
        options = ex.get("options", [])
        rationale = ex.get("rationale", "")
        correct = ex.get("correct", "")
        prompt = f"Question: {q}\n"
        if isinstance(options, list):
            for opt in options:
                prompt += f"  {opt}\n"
        if rationale:
            prompt += f"Solution: {rationale}\n"
        prompt += f"Answer: {correct}\n"
        texts.append(prompt)
    return texts


def _format_svamp_examples(dataset):
    """Format SVAMP elementary math word problems."""
    texts = []
    for ex in dataset:
        body = ex.get("Body", "")
        question = ex.get("Question", "")
        equation = ex.get("Equation", "")
        answer = ex.get("Answer", "")
        prompt = f"Question: {body} {question}\n"
        if equation:
            prompt += f"Solution: {equation} = {answer}\n"
        prompt += f"Answer: {answer}\n"
        texts.append(prompt)
    return texts


def _format_metamathqa_examples(dataset, max_examples=20000):
    """Format MetaMathQA augmented math problems with step-by-step solutions."""
    texts = []
    for i, ex in enumerate(dataset):
        if i >= max_examples:
            break
        query = ex.get("query", "")
        response = ex.get("response", "")
        texts.append(f"Question: {query}\nAnswer: {response}\n")
    return texts


def _format_mathinstruct_examples(dataset, max_examples=20000):
    """Format TIGER-Lab/MathInstruct CoT math problems from diverse sources."""
    texts = []
    for i, ex in enumerate(dataset):
        if i >= max_examples:
            break
        instruction = ex.get("instruction", "")
        output = ex.get("output", "")
        texts.append(f"Question: {instruction}\nAnswer: {output}\n")
    return texts


def _normalize_reasoning_answer(text: str, max_chars: int = 12000) -> str:
    """Post-process a reasoning trace: convert \\boxed{} → #### format and truncate.

    Extracts the last \\boxed{...} answer, truncates the chain-of-thought if it
    exceeds max_chars, and appends the answer in GSM8K-compatible #### format.
    """
    import re

    boxed_pat = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
    matches = list(boxed_pat.finditer(text))

    if matches:
        answer_content = matches[-1].group(1).strip()
        cot_end = matches[-1].start()
        cot = text[:cot_end].rstrip()
        if len(cot) > max_chars:
            cot = cot[:max_chars]
        return f"{cot}\n#### {answer_content}"

    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _format_openthoughts_examples(dataset, max_examples=30000):
    """Format OpenThoughts-114k reasoning examples (math, science, code, puzzles).

    Each example has a conversations list with user question + assistant response
    containing structured thinking (<|begin_of_thought|>...<|end_of_thought|>)
    and solution (<|begin_of_solution|>...<|end_of_solution|>).
    """
    texts = []
    for i, ex in enumerate(dataset):
        if i >= max_examples:
            break
        convs = ex.get("conversations", [])
        if len(convs) < 2:
            continue
        user_msg = convs[0].get("value", "").strip()
        assistant_msg = convs[1].get("value", "").strip()
        if not user_msg or not assistant_msg:
            continue
        assistant_msg = _normalize_reasoning_answer(assistant_msg)
        texts.append(f"Question: {user_msg}\nAnswer: {assistant_msg}\n")
    return texts


def _format_reasoning_jsonl_examples(jsonl_path: str, max_examples: int = 10000):
    """Format local JSONL reasoning traces (e.g., QwQ-32B RL-generated).

    Expects JSONL with {"messages": [{"role":"user","content":"..."},
    {"role":"assistant","content":"..."}], ...} per line.
    Only uses lines that have both user and assistant messages.
    """
    import json

    texts = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = ex.get("messages", [])
            user_msg = ""
            assistant_msg = ""
            for m in msgs:
                if m.get("role") == "user":
                    user_msg = m.get("content", "").strip()
                elif m.get("role") == "assistant":
                    assistant_msg = m.get("content", "").strip()
            if user_msg and assistant_msg:
                assistant_msg = _normalize_reasoning_answer(assistant_msg)
                texts.append(f"Question: {user_msg}\nAnswer: {assistant_msg}\n")
    return texts


def _format_slimpajama_examples(dataset, max_examples=80000):
    """Format SlimPajama as general text for broad LM quality (cleaned RedPajama)."""
    texts = []
    for i, ex in enumerate(dataset):
        if i >= max_examples:
            break
        text = ex.get("text", "").strip()
        if text and len(text) > 100:
            texts.append(text)
    return texts


def _format_triviaqa_examples(dataset, max_examples=20000):
    """Format TriviaQA as broad factual knowledge QA (proxy for MMLU breadth)."""
    texts = []
    for i, ex in enumerate(dataset):
        if i >= max_examples:
            break
        q = ex.get("question", "")
        answer = ex.get("answer", {})
        aliases = answer.get("aliases", [])
        value = answer.get("value", aliases[0] if aliases else "")
        if q and value:
            texts.append(f"Question: {q}\nAnswer: {value}\n")
    return texts


def _format_orcamath_examples(dataset, max_examples=20000):
    """Format Orca-Math word problems with detailed solutions."""
    texts = []
    for i, ex in enumerate(dataset):
        if i >= max_examples:
            break
        q = ex.get("question", "")
        a = ex.get("answer", "")
        texts.append(f"Question: {q}\nAnswer: {a}\n")
    return texts


def _format_race_examples(dataset, max_examples=30000):
    """Format RACE reading comprehension as passage + question + answer prompts."""
    texts = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    for i, ex in enumerate(dataset):
        if i >= max_examples:
            break
        article = ex["article"]
        question = ex["question"]
        options = ex["options"]
        answer_key = ex["answer"]
        answer_idx = label_map.get(answer_key, 0)
        prompt = f"Passage: {article[:500]}\n\nQuestion: {question}\n"
        for j, opt in enumerate(options):
            prompt += f"{'ABCD'[j]}. {opt}\n"
        if answer_idx < len(options):
            prompt += f"Answer: {'ABCD'[answer_idx]}. {options[answer_idx]}\n"
        texts.append(prompt)
    return texts


def _format_swag_examples(dataset):
    """Format SWAG as sentence-completion prompts (predecessor to HellaSwag)."""
    texts = []
    for ex in dataset:
        sent1 = ex["sent1"]
        sent2_prefix = ex["sent2"]
        endings = [ex["ending0"], ex["ending1"], ex["ending2"], ex["ending3"]]
        label = ex["label"]
        correct_ending = endings[label] if 0 <= label < 4 else endings[0]
        texts.append(f"{sent1} {sent2_prefix}{correct_ending}")
        prompt = f"Context: {sent1} {sent2_prefix}\nPossible completions:\n"
        for i, e in enumerate(endings):
            prompt += f"  {i + 1}. {e}\n"
        prompt += f"Best completion: {label + 1}. {correct_ending}\n"
        texts.append(prompt)
    return texts


def _format_qasc_examples(dataset):
    """Format QASC science QA with compositional reasoning facts."""
    texts = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
    for ex in dataset:
        q = ex["question"]
        choices = ex["choices"]
        answer_key = ex["answerKey"]
        fact1 = ex.get("fact1", "")
        fact2 = ex.get("fact2", "")
        answer_idx = label_map.get(answer_key, 0)
        choice_labels = choices["label"] if isinstance(choices, dict) else []
        choice_texts = choices["text"] if isinstance(choices, dict) else []
        prompt = ""
        if fact1 and fact2:
            prompt += f"Fact 1: {fact1}\nFact 2: {fact2}\n"
        prompt += f"Question: {q}\n"
        for lbl, txt in zip(choice_labels, choice_texts, strict=False):
            prompt += f"{lbl}. {txt}\n"
        if answer_idx < len(choice_texts):
            prompt += f"Answer: {choice_labels[answer_idx]}. {choice_texts[answer_idx]}\n"
        texts.append(prompt)
    return texts


def _format_rte_examples(dataset):
    """Format SuperGLUE RTE (textual entailment) as inference prompts."""
    texts = []
    for ex in dataset:
        premise = ex["premise"]
        hypothesis = ex["hypothesis"]
        label = ex["label"]  # 0=entailment, 1=not_entailment
        answer = "Yes" if label == 0 else "No"
        texts.append(
            f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? {answer}\n"
        )
    return texts


def _format_multirc_examples(dataset):
    """Format SuperGLUE MultiRC as reading comprehension prompts."""
    texts = []
    for ex in dataset:
        paragraph = ex["paragraph"]
        question = ex["question"]
        answer_text = ex["answer"]
        label = ex["label"]  # 0=incorrect, 1=correct
        correct = "Yes" if label == 1 else "No"
        texts.append(
            f"Passage: {paragraph}\nQuestion: {question}\nCandidate answer: {answer_text}\nIs this correct? {correct}\n"
        )
    return texts


def _format_record_examples(dataset, max_examples=20000):
    """Format SuperGLUE ReCoRD as cloze-style reading comprehension."""
    texts = []
    for i, ex in enumerate(dataset):
        if i >= max_examples:
            break
        passage = ex["passage"]
        query = ex["query"]
        answers = ex["answers"]
        if answers:
            filled = query.replace("@placeholder", answers[0])
            texts.append(f"Passage: {passage}\nQuery: {filled}\n")
    return texts


def _format_wic_examples(dataset):
    """Format SuperGLUE WiC (word-in-context) as word sense prompts."""
    texts = []
    for ex in dataset:
        word = ex["word"]
        s1 = ex["sentence1"]
        s2 = ex["sentence2"]
        label = ex["label"]  # 0=different sense, 1=same sense
        answer = "Yes" if label == 1 else "No"
        texts.append(f'Does "{word}" have the same meaning in these sentences?\n1. {s1}\n2. {s2}\nAnswer: {answer}\n')
    return texts


def _format_cb_examples(dataset):
    """Format SuperGLUE CB (CommitmentBank) as natural language inference prompts."""
    texts = []
    label_names = {0: "entailment", 1: "contradiction", 2: "neutral"}
    for ex in dataset:
        premise = ex["premise"]
        hypothesis = ex["hypothesis"]
        label = ex["label"]
        answer = label_names.get(label, "unknown")
        texts.append(f"Premise: {premise}\nHypothesis: {hypothesis}\nRelation: {answer}\n")
    return texts


def _format_superglue_copa_examples(dataset):
    """Format SuperGLUE COPA as causal reasoning prompts."""
    texts = []
    for ex in dataset:
        premise = ex["premise"]
        choice1 = ex["choice1"]
        choice2 = ex["choice2"]
        question = ex["question"]
        label = ex["label"]
        correct = choice1 if label == 0 else choice2
        q_word = "cause" if question == "cause" else "effect"
        texts.append(f"Premise: {premise}\nWhat is the {q_word}?\n1. {choice1}\n2. {choice2}\nAnswer: {correct}\n")
    return texts


class TaskMixDataset(Dataset):
    """Mix of general LM data + task-formatted training data for task-aware KD.

    Combines general diverse text (Pile and/or SlimPajama) with formatted
    training examples from downstream tasks. This teaches the student to match
    the teacher's behavior on both general language AND task-specific reasoning.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        max_samples: int | None = None,
        pile_ratio: float = 0.5,
        max_raw_pile: int = 30000,
        task_include: str = "all",
        use_slimpajama: bool = False,
        max_raw_slimpajama: int = 30000,
        max_per_task: int = 0,
        normalize_math_answers: bool = False,
        max_reasoning_chars: int = 1500,
    ):
        super().__init__()
        self.seq_len = seq_len
        all_texts = []

        # Task data — load reasoning/QA benchmarks (filterable via task_include)
        include_set = None
        if task_include and task_include.lower() != "all":
            include_set = {t.strip().lower() for t in task_include.split(",")}
            logger.info(f"[KD] Task filter: only loading {include_set}")

        task_loaders = [
            (
                "arc",
                "ARC-Challenge",
                lambda: _format_arc_examples(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")),
            ),
            (
                "hellaswag",
                "HellaSwag",
                lambda: _format_hellaswag_examples(load_dataset("Rowan/hellaswag", split="train")),
            ),
            (
                "winogrande",
                "WinoGrande",
                lambda: _format_winogrande_examples(load_dataset("allenai/winogrande", "winogrande_xl", split="train")),
            ),
            ("boolq", "BoolQ", lambda: _format_boolq_examples(load_dataset("google/boolq", split="train"))),
            (
                "openbookqa",
                "OpenBookQA",
                lambda: _format_openbookqa_examples(load_dataset("allenai/openbookqa", "main", split="train")),
            ),
            ("sciq", "SciQ", lambda: _format_sciq_examples(load_dataset("allenai/sciq", split="train"))),
            (
                "commonsenseqa",
                "CommonsenseQA",
                lambda: _format_commonsenseqa_examples(load_dataset("tau/commonsense_qa", split="train")),
            ),
            ("piqa", "PIQA", lambda: _format_piqa_examples(load_dataset("lighteval/piqa", split="train"))),
            ("siqa", "Social IQA", lambda: _format_siqa_examples(load_dataset("lighteval/siqa", split="train"))),
            ("race", "RACE", lambda: _format_race_examples(load_dataset("ehovy/race", "all", split="train"))),
            ("swag", "SWAG", lambda: _format_swag_examples(load_dataset("allenai/swag", "regular", split="train"))),
            ("qasc", "QASC", lambda: _format_qasc_examples(load_dataset("allenai/qasc", split="train"))),
            (
                "copa",
                "COPA (balanced)",
                lambda: _format_copa_examples(load_dataset("pkavumba/balanced-copa", split="train")),
            ),
            (
                "copa_sg",
                "COPA (SuperGLUE)",
                lambda: _format_superglue_copa_examples(load_dataset("aps/super_glue", "copa", split="train")),
            ),
            ("mmlu", "MMLU", lambda: _format_mmlu_examples(load_dataset("cais/mmlu", "all", split="auxiliary_train"))),
            (
                "triviaqa",
                "TriviaQA",
                lambda: _format_triviaqa_examples(load_dataset("trivia_qa", "rc.nocontext", split="train")),
            ),
            (
                "slimpajama",
                "SlimPajama",
                lambda: _format_slimpajama_examples(load_dataset("DKYoon/SlimPajama-6B", split="train")),
            ),
            (
                "gsm8k",
                "GSM8K",
                lambda: _format_gsm8k_examples(load_dataset("openai/gsm8k", "main", split="train"), include_cot=True),
            ),
            (
                "gsm8k_short",
                "GSM8K (short)",
                lambda: _format_gsm8k_examples(load_dataset("openai/gsm8k", "main", split="train"), include_cot=False),
            ),
            (
                "rte",
                "SuperGLUE-RTE",
                lambda: _format_rte_examples(load_dataset("aps/super_glue", "rte", split="train")),
            ),
            ("cb", "SuperGLUE-CB", lambda: _format_cb_examples(load_dataset("aps/super_glue", "cb", split="train"))),
            (
                "multirc",
                "SuperGLUE-MultiRC",
                lambda: _format_multirc_examples(load_dataset("aps/super_glue", "multirc", split="train")),
            ),
            (
                "record",
                "SuperGLUE-ReCoRD",
                lambda: _format_record_examples(load_dataset("aps/super_glue", "record", split="train")),
            ),
            (
                "wic",
                "SuperGLUE-WiC",
                lambda: _format_wic_examples(load_dataset("aps/super_glue", "wic", split="train")),
            ),
            (
                "aqua_rat",
                "AQuA-RAT",
                lambda: _format_aqua_rat_examples(load_dataset("deepmind/aqua_rat", "raw", split="train")),
            ),
            ("svamp", "SVAMP", lambda: _format_svamp_examples(load_dataset("ChilleD/SVAMP", split="train"))),
            (
                "metamathqa",
                "MetaMathQA",
                lambda: _format_metamathqa_examples(load_dataset("meta-math/MetaMathQA", split="train")),
            ),
            (
                "mathinstruct",
                "MathInstruct",
                lambda: _format_mathinstruct_examples(load_dataset("TIGER-Lab/MathInstruct", split="train")),
            ),
            (
                "orcamath",
                "OrcaMath",
                lambda: _format_orcamath_examples(
                    load_dataset("microsoft/orca-math-word-problems-200k", split="train")
                ),
            ),
            (
                "openthoughts",
                "OpenThoughts-114k",
                lambda: _format_openthoughts_examples(load_dataset("open-thoughts/OpenThoughts-114k", split="train")),
            ),
            (
                "reasoning_jsonl",
                "QwQ-32B Reasoning (local JSONL)",
                lambda: _format_reasoning_jsonl_examples(
                    os.path.join(os.environ.get("QUARK_ROOT", "."), "qwq32b_reasoning_generated.jsonl")
                ),
            ),
            (
                "gptoss_bf16_traces",
                "GPToss BF16 Reasoning Traces",
                lambda: _format_gptoss_bf16_traces(
                    os.path.join(os.environ.get("QUARK_ROOT", "."), "gptoss_bf16_reasoning_traces.jsonl")
                ),
            ),
            (
                "phi4_bf16_traces",
                "Phi-4 BF16 Reasoning Traces",
                lambda: _format_gptoss_bf16_traces(
                    os.path.join(os.environ.get("QUARK_ROOT", "."), "phi4_bf16_reasoning_traces.jsonl")
                ),
            ),
        ]

        import random

        for key, name, loader_fn in task_loaders:
            if include_set is not None and key not in include_set:
                logger.info(f"[KD]   {name}: excluded by task_include filter")
                continue
            try:
                texts = loader_fn()
                if max_per_task > 0 and len(texts) > max_per_task:
                    random.shuffle(texts)
                    texts = texts[:max_per_task]
                    logger.info(f"[KD]   {name}: {len(texts)} examples (capped from larger set)")
                else:
                    logger.info(f"[KD]   {name}: {len(texts)} formatted examples")
                if normalize_math_answers and key in MATH_TASK_KEYS:
                    texts = _postprocess_normalize_math_answers(texts, max_reasoning_chars=max_reasoning_chars)
                    logger.info(f"[KD]   {name}: normalized to #### answer format (max {max_reasoning_chars} chars)")
                all_texts.extend(texts)
            except Exception as e:
                logger.info(f"[KD]   {name}: skipped ({e})")

        random.shuffle(all_texts)
        task_text = "\n\n".join(all_texts)
        logger.info(f"[KD] Task text: {len(all_texts)} examples, {len(task_text)} chars")

        # General LM data: Pile + (optionally) SlimPajama
        lm_texts = []

        logger.info("[KD] Loading Pile data...")
        pile_raw = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        if max_raw_pile and len(pile_raw) > max_raw_pile:
            pile_raw = pile_raw.select(range(max_raw_pile))
        lm_texts.extend(pile_raw["text"])
        logger.info(f"[KD] Pile: {len(pile_raw)} entries")

        if use_slimpajama:
            logger.info("[KD] Loading SlimPajama data...")
            pajama_raw = load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True)
            pajama_texts = []
            for i, ex in enumerate(pajama_raw):
                if i >= max_raw_slimpajama:
                    break
                pajama_texts.append(ex["text"])
            lm_texts.extend(pajama_texts)
            logger.info(f"[KD] SlimPajama: {len(pajama_texts)} entries")

        lm_text = "\n\n".join(lm_texts)
        logger.info(f"[KD] General LM total: {len(lm_texts)} entries, {len(lm_text)} chars")

        # Tokenize both
        task_ids = tokenizer(task_text, return_attention_mask=False, return_tensors="pt")["input_ids"].squeeze(0)
        pile_ids = tokenizer(lm_text, return_attention_mask=False, return_tensors="pt")["input_ids"].squeeze(0)

        # Chunk into sequences
        task_len = (task_ids.shape[0] // seq_len) * seq_len
        task_chunks = task_ids[:task_len].view(-1, seq_len)

        pile_len = (pile_ids.shape[0] // seq_len) * seq_len
        pile_chunks = pile_ids[:pile_len].view(-1, seq_len)

        logger.info(f"[KD] Task chunks: {len(task_chunks)}, LM chunks: {len(pile_chunks)}")

        # Mix according to ratio (pile_ratio controls general LM fraction)
        n_task = len(task_chunks)
        n_lm_target = int(n_task * pile_ratio / max(1.0 - pile_ratio, 0.01))
        n_lm_target = min(n_lm_target, len(pile_chunks))

        self.input_ids = torch.cat([task_chunks, pile_chunks[:n_lm_target]], dim=0)

        # Shuffle
        perm = torch.randperm(len(self.input_ids))
        self.input_ids = self.input_ids[perm]

        if max_samples is not None and max_samples > 0:
            self.input_ids = self.input_ids[:max_samples]

        logger.info(
            f"[KD] TaskMix final: {len(self.input_ids)} samples "
            f"(task={n_task}, lm={n_lm_target}, ratio={n_task / (n_task + n_lm_target):.1%} task)"
        )

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int):
        ids = self.input_ids[idx]
        return {"input_ids": ids, "labels": ids.clone()}


# ---------------------------------------------------------------------------
# PPL evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_ppl(model, dataloader, device, multi_gpu=False):
    model.eval()
    losses = []
    for i, batch in enumerate(dataloader):
        if multi_gpu:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
        else:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
        loss = model(input_ids=input_ids, labels=labels).loss
        losses.append(loss.item())
        if (i + 1) % 100 == 0:
            logger.info(f"[KD] Eval: {i + 1}/{len(dataloader)} batches")
    return math.exp(sum(losses) / max(1, len(losses)))


# ---------------------------------------------------------------------------
# Distillation loss helpers
# ---------------------------------------------------------------------------
def kd_logit_loss(student_logits, teacher_logits, temperature=2.0):
    """KL-divergence between teacher and student output distributions.

    Computes per-token KL then averages over (batch * seq_len) so the scale
    is comparable to the per-token CE loss (~2-5 range).
    """
    s = student_logits.view(-1, student_logits.size(-1))
    t = teacher_logits.view(-1, teacher_logits.size(-1))
    log_s = F.log_softmax(s / temperature, dim=-1)
    prob_t = F.softmax(t / temperature, dim=-1)
    kl = F.kl_div(log_s, prob_t, reduction="sum") / s.size(0)
    return kl * (temperature**2)


def kd_jsd_loss(student_logits, teacher_logits, temperature=2.0, beta=0.5, chunk_size=4096):
    """Jensen-Shannon Divergence between teacher and student distributions.

    JSD is symmetric and interpolates between forward and reverse KL.
    UPQ (2025) shows JSD outperforms standard KL for 2-bit instruct models
    on MMLU and instruction-following benchmarks.

    Chunks across the vocabulary dimension to avoid holding multiple
    full-vocab tensors simultaneously (prevents OOM on 64 GiB GPUs).
    """
    s = student_logits.view(-1, student_logits.size(-1)) / temperature
    t = teacher_logits.view(-1, teacher_logits.size(-1)) / temperature
    n_tokens = s.size(0)
    vocab = s.size(-1)

    if vocab <= chunk_size:
        p_t = F.softmax(t, dim=-1)
        p_s = F.softmax(s, dim=-1)
        p_m = beta * p_t + (1 - beta) * p_s
        log_m = torch.log(p_m + 1e-8)
        jsd = beta * F.kl_div(log_m, p_t, reduction="sum") + (1 - beta) * F.kl_div(log_m, p_s, reduction="sum")
        return (jsd / n_tokens) * (temperature**2)

    # Chunked: compute softmax once (full), then chunk the KL accumulation
    p_t = F.softmax(t, dim=-1)
    p_s = F.softmax(s, dim=-1)
    del s, t

    jsd = torch.tensor(0.0, device=student_logits.device)
    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        pt_c = p_t[:, start:end]
        ps_c = p_s[:, start:end]
        pm_c = beta * pt_c + (1 - beta) * ps_c
        log_m_c = torch.log(pm_c + 1e-8)
        jsd = (
            jsd
            + beta * F.kl_div(log_m_c, pt_c, reduction="sum")
            + (1 - beta) * F.kl_div(log_m_c, ps_c, reduction="sum")
        )
        del pt_c, ps_c, pm_c, log_m_c

    del p_t, p_s
    return (jsd / n_tokens) * (temperature**2)


def kd_hidden_loss(student_hiddens, teacher_hiddens):
    """Normalized MSE between hidden states at each layer.

    Normalizes by hidden dimension so the loss doesn't scale with model width.
    """
    loss = torch.tensor(0.0, device=student_hiddens[0].device)
    n = min(len(student_hiddens), len(teacher_hiddens))
    for i in range(n):
        sh = student_hiddens[i].float()
        th = teacher_hiddens[i].float()
        # Cosine-distance inspired: normalize by hidden dim magnitude
        # MSE already averages over all elements, so this is well-scaled
        loss = loss + F.mse_loss(sh, th)
    return loss / max(n, 1)


def kd_attention_loss(student_attns, teacher_attns):
    """KL-divergence between attention distributions, averaged per-position.

    Attention maps are already probability distributions over the last dim (keys).
    We compute per-query-position KL and average over (batch, heads, queries).
    """
    loss = torch.tensor(0.0, device=student_attns[0].device)
    n = min(len(student_attns), len(teacher_attns))
    for i in range(n):
        # shape: (batch, num_heads, seq_q, seq_k) — already probabilities
        sa = student_attns[i].float().clamp(min=1e-8)
        ta = teacher_attns[i].float().clamp(min=1e-8)
        # Per-position KL: sum over seq_k (last dim), mean over (batch, heads, seq_q)
        kl = (ta * (ta.log() - sa.log())).sum(dim=-1).mean()
        loss = loss + kl
    return loss / max(n, 1)


# ---------------------------------------------------------------------------
# On-policy KD: student generates, teacher supervises
# ---------------------------------------------------------------------------
@torch.no_grad()
def _collect_on_policy_prompts(train_loader, data_iter, prompt_len, num_prompts, multi_gpu):
    """Extract prompt prefixes from training data for on-policy generation."""
    prompts = []
    for _ in range(num_prompts):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        ids = batch["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        prompts.append(ids[:, :prompt_len])
    return torch.cat(prompts, dim=0), data_iter


class _SanitizeLogitsProcessor:
    """Replaces NaN/Inf in logits with large finite values to prevent CUDA asserts
    during sampling. 2-bit quantized models occasionally produce degenerate logits
    during autoregressive generation."""

    def __call__(self, input_ids, scores):
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
        return scores


def on_policy_kd_step(
    student, teacher, prompt_ids, max_gen_len, gen_temperature, kd_temperature, kd_loss_fn, kd_alpha, multi_gpu=False
):
    """On-policy KD: student generates a sequence, teacher provides supervision.

    1. Student generates tokens autoregressively (no gradient).
    2. Teacher computes logits on the student-generated sequence.
    3. Student re-computes logits with gradient on its own generated sequence.
    4. KD + CE loss is computed on the generated (non-prompt) portion only.

    This directly addresses error compounding: the student learns to handle
    its own distributional drift, not just teacher-forced ground truth.
    """
    # 1. Student generates (no gradient, sampling)
    student.eval()
    with torch.no_grad():
        generated = student.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_gen_len,
            do_sample=True,
            temperature=gen_temperature,
            top_p=0.9,
            pad_token_id=0,
            logits_processor=[_SanitizeLogitsProcessor()],
        )
    student.train()

    prompt_len = prompt_ids.shape[1]
    if generated.shape[1] <= prompt_len + 1:
        return None, None

    # 2. Teacher logits on student-generated sequence
    with torch.inference_mode():
        teacher_out = teacher(input_ids=generated)

    # 3. Student logits WITH gradient on its own generated sequence
    labels = generated.clone()
    labels[:, :prompt_len] = -100  # mask prompt tokens from CE loss
    student_out = student(input_ids=generated, labels=labels)

    loss_device = student_out.logits.device
    teacher_logits = teacher_out.logits.detach().to(loss_device)

    # 4. Compute KD loss only on generated tokens (after prompt)
    gen_student_logits = student_out.logits[:, prompt_len - 1 : -1, :]
    gen_teacher_logits = teacher_logits[:, prompt_len - 1 : -1, :]

    kd_loss = kd_loss_fn(gen_student_logits, gen_teacher_logits, kd_temperature)
    ce_loss = student_out.loss if student_out.loss is not None else torch.tensor(0.0, device=loss_device)

    total_loss = kd_alpha * kd_loss + (1.0 - kd_alpha) * ce_loss

    del teacher_out, teacher_logits, student_out, generated, labels
    torch.cuda.empty_cache()

    return total_loss, kd_loss.item()


# ---------------------------------------------------------------------------
# Completion masking: build loss masks for prompt-completion formatted data
# ---------------------------------------------------------------------------
def build_completion_mask(input_ids, tokenizer, marker_text="Answer:"):
    """Build a binary mask that is 1 for completion tokens and 0 for prompt tokens.

    For each sequence in the batch, finds the LAST occurrence of marker_text
    and sets the mask to 1 for all tokens after it. If the marker is not found,
    the entire sequence is unmasked (all 1s) to avoid zero gradients.
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.ones(batch_size, seq_len, device=input_ids.device, dtype=torch.float32)

    marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)
    if not marker_ids:
        return mask

    marker_len = len(marker_ids)
    marker_tensor = torch.tensor(marker_ids, device=input_ids.device)

    for b in range(batch_size):
        last_match = -1
        for i in range(seq_len - marker_len + 1):
            if torch.equal(input_ids[b, i : i + marker_len], marker_tensor):
                last_match = i
        if last_match >= 0:
            mask[b, : last_match + marker_len] = 0.0

    return mask


def masked_kd_loss(student_logits, teacher_logits, mask, temperature, loss_fn):
    """Compute KD loss only on masked (completion) tokens.

    Args:
        student_logits: (B, T, V)
        teacher_logits: (B, T, V)
        mask: (B, T) binary mask, 1 = compute loss here
        temperature: KD temperature
        loss_fn: 'jsd' or 'kl'
    """
    B, T, V = student_logits.shape
    flat_mask = mask[:, :T].reshape(-1)
    active_idx = flat_mask.nonzero(as_tuple=True)[0]

    if active_idx.numel() == 0:
        return (student_logits * 0.0).sum()

    s_flat = student_logits.reshape(-1, V)[active_idx]
    t_flat = teacher_logits.reshape(-1, V)[active_idx]

    s_scaled = s_flat / temperature
    t_scaled = t_flat / temperature

    if loss_fn == "jsd":
        p_t = F.softmax(t_scaled, dim=-1)
        p_s = F.softmax(s_scaled, dim=-1)
        p_m = 0.5 * p_t + 0.5 * p_s
        log_m = torch.log(p_m + 1e-8)
        jsd = 0.5 * F.kl_div(log_m, p_t, reduction="sum") + 0.5 * F.kl_div(log_m, p_s, reduction="sum")
        return (jsd / active_idx.numel()) * (temperature**2)
    else:
        log_s = F.log_softmax(s_scaled, dim=-1)
        prob_t = F.softmax(t_scaled, dim=-1)
        kl = F.kl_div(log_s, prob_t, reduction="sum") / active_idx.numel()
        return kl * (temperature**2)


def masked_ce_loss(logits, labels, mask):
    """Cross-entropy loss only on masked (completion) tokens."""
    B, T, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = mask[:, 1:T].contiguous()

    flat_logits = shift_logits.reshape(-1, V)
    flat_labels = shift_labels.reshape(-1)
    flat_mask = shift_mask.reshape(-1)

    active_idx = flat_mask.nonzero(as_tuple=True)[0]
    if active_idx.numel() == 0:
        return (logits * 0.0).sum()

    ce = F.cross_entropy(flat_logits[active_idx], flat_labels[active_idx])
    return ce


def _parse_block_ranges(spec: str) -> set:
    """Parse block range spec like '0-6,56-63' into a set of ints."""
    blocks = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            blocks.update(range(int(lo), int(hi) + 1))
        else:
            blocks.add(int(part))
    return blocks


def _build_selective_targets(model, full_blocks_spec, partial_blocks_spec, partial_projs_spec, logger):
    """Build explicit list of module names for selective LoRA placement.

    Args:
        full_blocks_spec: blocks getting all projections (e.g., '0-6,56-63')
        partial_blocks_spec: blocks getting only specific projections (e.g., '7-55')
        partial_projs_spec: which projections for partial blocks (e.g., 'down_proj')
    Returns:
        List of module name strings matching PEFT's target_modules format.
    """
    full_blocks = _parse_block_ranges(full_blocks_spec)
    partial_blocks = _parse_block_ranges(partial_blocks_spec) if partial_blocks_spec else set()
    partial_projs = {p.strip() for p in partial_projs_spec.split(",")}
    all_projs = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        proj = name.split(".")[-1]
        if proj not in all_projs:
            continue

        block_num = None
        for part in name.split("."):
            if part.isdigit():
                block_num = int(part)
                break
        if block_num is None:
            continue

        if block_num in full_blocks or block_num in partial_blocks and proj in partial_projs:
            targets.append(name)

    logger.info(
        f"[KD] Selective LoRA: {len(full_blocks)} full blocks, {len(partial_blocks)} partial blocks ({partial_projs})"
    )
    logger.info(
        f"[KD] Total LoRA target modules: {len(targets)} / "
        f"{sum(1 for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.split('.')[-1] in all_projs)}"
    )

    full_count = sum(
        1 for t in targets if any(f".{b}." in t or t.endswith(f".{b}") for b in [str(x) for x in full_blocks])
    )
    partial_count = len(targets) - full_count
    logger.info(f"[KD]   Full blocks: {full_count} modules, Partial blocks: {partial_count} modules")

    return targets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="LoRA + Knowledge Distillation on PTQ model")

    # Models
    p.add_argument("--student_model_dir", type=str, required=True, help="Path to PTQ (student) model")
    p.add_argument("--teacher_model_dir", type=str, required=True, help="Path to BF16 (teacher) model")
    p.add_argument(
        "--student_base_model",
        type=str,
        default="",
        help="For MoE models (GPT-OSS): original model path. Architecture is loaded from here, "
        "then PTQ weights from --student_model_dir are overlaid after module replacement.",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU model parallelism via device_map='auto'")

    # Data
    p.add_argument(
        "--dataset",
        type=str,
        default="pile",
        choices=["wikitext", "pile", "taskmix"],
        help="Training dataset: 'taskmix' (task+pile, best), 'pile' (diverse), or 'wikitext' (narrow)",
    )
    p.add_argument(
        "--pile_ratio",
        type=float,
        default=0.5,
        help="For taskmix: fraction of general LM data (Pile+SlimPajama) vs task data (0.5 = equal)",
    )
    p.add_argument(
        "--task_include",
        type=str,
        default="all",
        help="For taskmix: comma-separated task names to include, or 'all'. "
        "Available: arc,hellaswag,winogrande,boolq,openbookqa,sciq,commonsenseqa,"
        "piqa,siqa,race,swag,qasc,copa,copa_sg,mmlu,gsm8k,gsm8k_short,rte,cb,multirc,record,wic,"
        "aqua_rat,svamp,metamathqa,mathinstruct,orcamath,openthoughts,reasoning_jsonl,"
        "gptoss_bf16_traces,phi4_bf16_traces",
    )
    p.add_argument(
        "--use_slimpajama",
        action="store_true",
        help="Add SlimPajama data alongside Pile for richer general LM coverage",
    )
    p.add_argument(
        "--max_raw_slimpajama",
        type=int,
        default=30000,
        help="Max entries to load from SlimPajama (streamed from validation split)",
    )
    p.add_argument(
        "--max_per_task",
        type=int,
        default=0,
        help="Cap each task to at most N examples (0=no cap). "
        "Prevents large datasets like MetaMathQA from dominating the mix.",
    )
    p.add_argument(
        "--normalize_math_answers",
        action="store_true",
        help="Post-process math tasks (svamp, metamathqa, mathinstruct, orcamath, "
        "openthoughts, reasoning_jsonl, gptoss_bf16_traces, phi4_bf16_traces) to end with "
        "GSM8K-compatible #### <number> format. Off by default.",
    )
    p.add_argument(
        "--max_reasoning_chars",
        type=int,
        default=1500,
        help="Max chars of reasoning before #### answer in normalized math data. "
        "Use ~150 for 2-bit models that degenerate during long generation.",
    )
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_train_samples", type=int, default=5000)
    p.add_argument("--max_eval_samples", type=int, default=512)

    # Training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_steps", type=int, default=300)
    p.add_argument("--warmup_steps", type=int, default=50, help="Linear warmup steps")
    p.add_argument("--eval_every", type=int, default=0, help="Eval PPL every N steps (0=disabled)")
    p.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument(
        "--gradient_checkpointing",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Gradient checkpointing: 'auto' enables for dense models, disables for MoE; "
        "'on' forces enable; 'off' forces disable.",
    )

    # LoRA
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="",
        help="Override LoRA target modules (comma-separated). "
        "Default for dense: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj. "
        "Use gate_up_proj,down_proj for GPT-OSS MoE experts.",
    )
    p.add_argument(
        "--lora_full_blocks",
        type=str,
        default="",
        help="Selective LoRA: blocks that get ALL projections (e.g., '0-6,56-63'). "
        "When set, enables selective LoRA mode.",
    )
    p.add_argument(
        "--lora_partial_blocks",
        type=str,
        default="",
        help="Selective LoRA: blocks that get only specified projections (e.g., '7-55')",
    )
    p.add_argument(
        "--lora_partial_projs",
        type=str,
        default="down_proj",
        help="Selective LoRA: which projections for partial blocks (comma-separated, default: 'down_proj')",
    )

    # Snap frozen weights to visible 2-bit (4 Lloyd-Max levels × per-row scale)
    p.add_argument(
        "--snap_to_2bit",
        action="store_true",
        help="Re-quantize frozen PTQ weights to exactly 4 levels per row per group.",
    )
    p.add_argument("--snap_group_size", type=int, default=64, help="Group size for snap_to_2bit (default: 64)")
    p.add_argument(
        "--snap_exclude",
        type=str,
        default="*embed_tokens*,*lm_head*",
        help="Comma-separated glob patterns to exclude from snapping",
    )

    # Distillation
    p.add_argument(
        "--kd_mode",
        type=str,
        default="output",
        choices=["output", "layer", "attention"],
        help="Distillation mode: output (logit KD), layer (hidden + logit), attention (attn + hidden + logit)",
    )
    p.add_argument("--kd_temperature", type=float, default=2.0, help="Softmax temperature for logit KD")
    p.add_argument(
        "--kd_alpha", type=float, default=0.5, help="Balance: loss = alpha * KD_loss + (1 - alpha) * CE_loss"
    )
    p.add_argument("--kd_hidden_weight", type=float, default=1.0, help="Weight for hidden-state MSE loss")
    p.add_argument("--kd_attn_weight", type=float, default=1.0, help="Weight for attention KL loss")
    p.add_argument(
        "--use_qad_trainer",
        action="store_true",
        help="Train via Quark's QADTrainer (peft LoRA + QADTrainer KD loop) instead of "
        "the built-in manual loop. The LoRA adapters (peft) are used either way.",
    )
    p.add_argument(
        "--kd_loss_type",
        type=str,
        default="kl",
        choices=["kl", "jsd"],
        help="Logit distillation loss: 'kl' (standard KL-div) or 'jsd' (Jensen-Shannon)",
    )

    # On-policy KD: student generates, teacher supervises (addresses error compounding)
    p.add_argument(
        "--on_policy_kd",
        action="store_true",
        help="Enable on-policy KD: periodically have the student generate sequences "
        "and compute KD loss on student-generated tokens. This directly addresses "
        "error compounding in autoregressive generation. Off by default (v7 behavior).",
    )
    p.add_argument(
        "--on_policy_every",
        type=int,
        default=5,
        help="Run on-policy KD every N steps (default: 5). Only used when --on_policy_kd is set.",
    )
    p.add_argument(
        "--on_policy_max_gen_len",
        type=int,
        default=256,
        help="Max tokens to generate during on-policy steps (default: 256).",
    )
    p.add_argument(
        "--on_policy_prompt_len",
        type=int,
        default=64,
        help="Number of prompt tokens from training data to seed generation (default: 64).",
    )
    p.add_argument(
        "--on_policy_temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for student generation in on-policy steps (default: 0.7).",
    )
    p.add_argument(
        "--on_policy_alpha",
        type=float,
        default=0.0,
        help="Weight for on-policy loss relative to off-policy. 0.0 = auto-ramp from 0 to 0.5 "
        "over training. Set >0 for fixed weight (e.g. 0.3).",
    )

    # Prompt-completion masking: only compute loss on completion tokens
    p.add_argument(
        "--completion_loss_only",
        action="store_true",
        help="Only compute KD/CE loss on completion tokens (after the prompt/question). "
        "Focuses learning on generation quality. Off by default (v7 behavior).",
    )
    p.add_argument(
        "--completion_marker",
        type=str,
        default="Answer:",
        help="Token sequence marking the start of the completion in formatted examples. "
        "Loss is computed only on tokens after this marker. Default: 'Answer:'",
    )

    p.add_argument("--save_model", action="store_true")
    p.add_argument(
        "--save_lora_adapters",
        action="store_true",
        help="Save LoRA adapter weights separately (adapter_model.safetensors + adapter_config.json) "
        "in addition to the merged model. Enables smaller storage and on-the-fly merging at eval time.",
    )
    p.add_argument(
        "--results_file",
        type=str,
        default="",
        help="Append results as JSON line to this file (for comparison across runs)",
    )

    # Post-training benchmark evaluation
    p.add_argument(
        "--eval_tasks",
        type=str,
        default="",
        help="Comma-separated lm_eval benchmark tasks to run after training "
        "(e.g. arc_challenge,hellaswag,winogrande,boolq,openbookqa)",
    )
    p.add_argument("--eval_batch_size", type=str, default="auto:4", help="Batch size for lm_eval benchmarks")
    p.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples for benchmarks")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    multi_gpu = args.multi_gpu
    device = args.device

    # ------------------------------------------------------------------
    # 1. Load teacher (BF16, frozen, no LoRA)
    # ------------------------------------------------------------------
    logger.info(f"[KD] Loading TEACHER from {args.teacher_model_dir}")
    teacher, _ = get_model(
        ckpt_path=args.teacher_model_dir,
        data_type="bfloat16",
        device=device,
        multi_gpu=multi_gpu,
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info("[KD] Teacher loaded and frozen")

    # ------------------------------------------------------------------
    # 2. Load student (PTQ model)
    # ------------------------------------------------------------------
    if args.student_base_model:
        # MoE models (GPT-OSS): load architecture from base, overlay PTQ weights
        logger.info(f"[KD] Loading STUDENT base architecture from {args.student_base_model}")
        logger.info(f"[KD] PTQ weights will be loaded from {args.student_model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(args.student_base_model, trust_remote_code=True)
        student, _ = get_model(
            ckpt_path=args.student_base_model,
            data_type="bfloat16",
            device=device,
            multi_gpu=multi_gpu,
            trust_remote_code=True,
        )
        prepare_for_moe_quant(student)
        logger.info("[KD] Module replacement applied, loading PTQ weights...")
        from safetensors.torch import load_file

        ptq_weights_path = os.path.join(args.student_model_dir, "model.safetensors")
        ptq_state_dict = load_file(ptq_weights_path)
        missing, unexpected = student.load_state_dict(ptq_state_dict, strict=False)
        if missing:
            logger.info(f"[KD] PTQ load — missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.info(f"[KD] PTQ load — unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        logger.info(f"[KD] PTQ weights loaded from {ptq_weights_path}")
    else:
        logger.info(f"[KD] Loading STUDENT from {args.student_model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(args.student_model_dir, trust_remote_code=True)
        # The PTQ student is a *fake-quantized* export: the 2-bit error is already
        # baked into plain bf16 weights, and its config.json carries an
        # (informational) quark `quantization_config`. Loading it through the HF
        # quark quantizer would try to rebuild the quantizer from that config and
        # fail for some export schemes (e.g. bfp16 raises "Serialization of bfp16
        # models is not yet supported"). Since there is nothing to re-quantize,
        # drop the quant config and load the bf16 weights directly.
        student_cfg = AutoConfig.from_pretrained(args.student_model_dir, trust_remote_code=True)
        if hasattr(student_cfg, "quantization_config"):
            logger.info("[KD] Student is fake-quantized; ignoring its quantization_config and loading bf16 weights")
            del student_cfg.quantization_config
        student = AutoModelForCausalLM.from_pretrained(
            args.student_model_dir,
            config=student_cfg,
            torch_dtype=torch.bfloat16,
            device_map=("auto" if multi_gpu else device),
            trust_remote_code=True,
            attn_implementation="eager",
        )

    # ------------------------------------------------------------------
    # 2b. (Optional) Snap frozen weights to visible 2-bit (4 levels × scale)
    # ------------------------------------------------------------------
    if getattr(args, "snap_to_2bit", False):
        from quark.experimental.torch.twobitscalar import snap_to_2bit

        exclude_pats = [p.strip() for p in args.snap_exclude.split(",") if p.strip()]
        logger.info(
            f"[KD] Snapping frozen weights to 4-level 2-bit (group_size={args.snap_group_size}, exclude={exclude_pats})"
        )
        snap_stats = snap_to_2bit(
            student, group_size=args.snap_group_size, use_lloyd_max=True, exclude_patterns=exclude_pats
        )
        for lname, lstat in list(snap_stats.items())[:5]:
            logger.info(f"  {lname}: SQNR={lstat['sqnr_db']} dB, shape={lstat['shape']}")
        logger.info(f"[KD] Snapped {len(snap_stats)} layers to visible 2-bit values")

    # ------------------------------------------------------------------
    # 3. Datasets — train on diverse data, eval on WikiText for PPL comparability
    # ------------------------------------------------------------------
    if args.dataset == "taskmix":
        logger.info("[KD] Using TaskMix (task-formatted + Pile) for training")
        train_dataset = TaskMixDataset(
            tokenizer,
            seq_len=args.seq_len,
            max_samples=args.max_train_samples,
            pile_ratio=args.pile_ratio,
            task_include=args.task_include,
            use_slimpajama=args.use_slimpajama,
            max_raw_slimpajama=args.max_raw_slimpajama,
            max_per_task=args.max_per_task,
            normalize_math_answers=getattr(args, "normalize_math_answers", False),
            max_reasoning_chars=getattr(args, "max_reasoning_chars", 1500),
        )
    elif args.dataset == "pile":
        logger.info("[KD] Using Pile (diverse) for training")
        train_dataset = PileLMDataset(tokenizer, seq_len=args.seq_len, max_samples=args.max_train_samples)
    else:
        logger.info("[KD] Using WikiText-2 for training")
        train_dataset = WikiTextLMDataset(
            tokenizer, split="train", seq_len=args.seq_len, max_samples=args.max_train_samples
        )
    eval_dataset = WikiTextLMDataset(
        tokenizer, split="validation", seq_len=args.seq_len, max_samples=args.max_eval_samples
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # 4. Baseline PPL (before LoRA)
    # ------------------------------------------------------------------
    logger.info("[KD] Evaluating baseline student PPL...")
    base_ppl = evaluate_ppl(student, eval_loader, device, multi_gpu=multi_gpu)
    logger.info(f"[KD] Baseline student PPL: {base_ppl:.4f}")

    logger.info("[KD] Evaluating teacher PPL...")
    teacher_ppl = evaluate_ppl(teacher, eval_loader, device, multi_gpu=multi_gpu)
    logger.info(f"[KD] Teacher PPL: {teacher_ppl:.4f}")

    # ------------------------------------------------------------------
    # 5. Add LoRA to student
    # ------------------------------------------------------------------
    if args.lora_full_blocks:
        target_modules = _build_selective_targets(
            student, args.lora_full_blocks, args.lora_partial_blocks, args.lora_partial_projs, logger
        )
    elif args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    logger.info(
        f"[KD] Adding LoRA (r={args.lora_r}, alpha={args.lora_alpha}, "
        f"targets={'selective (' + str(len(target_modules)) + ' modules)' if args.lora_full_blocks else target_modules})"
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    student = get_peft_model(student, lora_config)
    if not multi_gpu:
        student.to(device)
    student.print_trainable_parameters()

    # Gradient checkpointing: trades compute for memory (no accuracy impact).
    # 'auto' disables for MoE models (non-deterministic routing causes shape
    # mismatches during recomputation) and enables for dense models.
    gc_mode = args.gradient_checkpointing
    if gc_mode == "auto":
        gc_mode = "off" if args.student_base_model else "on"
    if gc_mode == "on" and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        logger.info("[KD] Gradient checkpointing enabled (saves ~10-15GB VRAM)")
    else:
        if hasattr(student, "gradient_checkpointing_disable"):
            student.gradient_checkpointing_disable()
        logger.info("[KD] Gradient checkpointing disabled")

    # ------------------------------------------------------------------
    # Optional path: train via Quark's QADTrainer (peft LoRA student + QADTrainer
    # KD loop) instead of the built-in manual loop. The LoRA adapters are the same.
    # ------------------------------------------------------------------
    if args.use_qad_trainer:
        from transformers import TrainingArguments

        from quark.torch.algorithm.qad_trainer import QADTrainer

        student.teacher = teacher  # QADTrainer expects model.teacher
        targs = TrainingArguments(
            output_dir=args.output_dir,
            max_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            learning_rate=args.lr,
            logging_steps=25,
            save_steps=max(args.num_steps, 1_000_000),
            report_to=[],
            bf16=True,
            remove_unused_columns=False,
        )
        trainer = QADTrainer(
            model=student,
            args=targs,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            temperature=args.kd_temperature,
            kd_loss_type=args.kd_loss_type,
            kd_alpha=args.kd_alpha,
            kd_mode=args.kd_mode,
            log_loss_breakdown=True,
        )
        logger.info(
            f"[KD] Using Quark QADTrainer (peft LoRA + QADTrainer KD) for {args.num_steps} steps "
            f"(kd_loss={args.kd_loss_type}, T={args.kd_temperature}, alpha={args.kd_alpha})"
        )
        trainer.train()
        student.teacher = None  # don't serialize the teacher
        if args.save_model:
            student.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logger.info(f"[KD] Saved LoRA+KD (QADTrainer) model to {args.output_dir}")
        return

    # Determine forward kwargs based on distillation mode
    need_hidden = args.kd_mode in ("layer", "attention")
    need_attn = args.kd_mode == "attention"

    logger.info(f"[KD] Distillation mode: {args.kd_mode}")
    logger.info(f"[KD]   temperature={args.kd_temperature}, alpha={args.kd_alpha}")
    logger.info(f"[KD]   hidden_weight={args.kd_hidden_weight}, attn_weight={args.kd_attn_weight}")
    logger.info(f"[KD]   need_hidden={need_hidden}, need_attn={need_attn}")

    # ------------------------------------------------------------------
    # 6. Training loop with KD + cosine LR schedule
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # Cosine schedule with linear warmup
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / max(1, args.warmup_steps)
        progress = float(current_step - args.warmup_steps) / max(1, args.num_steps - args.warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    student.train()
    data_iter = iter(train_loader)
    best_ppl = float("inf")

    grad_accum = args.grad_accum_steps

    # Loss curve CSV logging
    loss_csv_path = os.path.join(args.output_dir, "loss_curve.csv")
    # Held open for the whole training loop and closed after it; a `with` here
    # would mean indenting the entire loop.
    loss_csv_file = open(loss_csv_path, "w", newline="")  # noqa: SIM115
    loss_csv_writer = csv.writer(loss_csv_file)
    loss_csv_writer.writerow(["step", "loss", "ce", "logit_kd", "hidden_kd", "attn_kd", "lr", "eval_ppl"])
    logger.info(f"[KD] Loss curve will be saved to {loss_csv_path}")

    # Select KD loss function for reuse
    kd_loss_fn = kd_jsd_loss if args.kd_loss_type == "jsd" else kd_logit_loss

    # On-policy KD setup
    use_on_policy = args.on_policy_kd
    if use_on_policy:
        logger.info(
            f"[KD] On-policy KD ENABLED: every {args.on_policy_every} steps, "
            f"gen_len={args.on_policy_max_gen_len}, prompt_len={args.on_policy_prompt_len}, "
            f"gen_temp={args.on_policy_temperature}"
        )
    else:
        logger.info("[KD] On-policy KD disabled (v7 behavior)")

    # Completion masking setup
    use_completion_mask = args.completion_loss_only
    if use_completion_mask:
        logger.info(f"[KD] Completion-only loss ENABLED: marker='{args.completion_marker}'")
    else:
        logger.info("[KD] Completion masking disabled (v7 behavior — loss on all tokens)")

    logger.info(
        f"[KD] Starting LoRA+KD training for {args.num_steps} steps "
        f"(warmup={args.warmup_steps}, grad_accum={grad_accum}, eff_batch={args.batch_size * grad_accum})"
    )
    for step in range(1, args.num_steps + 1):
        accum_loss = 0.0
        accum_ce = 0.0
        accum_logit_kd = 0.0
        accum_hidden_kd = 0.0
        accum_attn_kd = 0.0
        accum_on_policy = 0.0

        # ---- On-policy KD step (student generates, teacher supervises) ----
        is_on_policy_step = use_on_policy and step % args.on_policy_every == 0
        if is_on_policy_step:
            prompt_batch, data_iter = _collect_on_policy_prompts(
                train_loader, data_iter, args.on_policy_prompt_len, num_prompts=args.batch_size, multi_gpu=multi_gpu
            )
            if not multi_gpu:
                prompt_batch = prompt_batch.to(device)

            try:
                on_policy_loss, on_policy_kd_val = on_policy_kd_step(
                    student,
                    teacher,
                    prompt_batch,
                    max_gen_len=args.on_policy_max_gen_len,
                    gen_temperature=args.on_policy_temperature,
                    kd_temperature=args.kd_temperature,
                    kd_loss_fn=kd_loss_fn,
                    kd_alpha=args.kd_alpha,
                    multi_gpu=multi_gpu,
                )
            except (RuntimeError, torch.AcceleratorError) as e:
                logger.info(f"[KD] On-policy step {step} skipped due to generation error: {e}")
                on_policy_loss = None
                student.train()
                torch.cuda.empty_cache()

            if on_policy_loss is not None:
                # Ramp on-policy weight: starts at 0, reaches target at 60% of training
                if args.on_policy_alpha > 0:
                    op_weight = args.on_policy_alpha
                else:
                    ramp_progress = min(1.0, step / (0.6 * args.num_steps))
                    op_weight = 0.5 * ramp_progress

                (on_policy_loss * op_weight / grad_accum).backward()
                accum_on_policy = on_policy_loss.item() * op_weight
                del on_policy_loss
                torch.cuda.empty_cache()

        # ---- Standard off-policy KD step (teacher-forced, existing v7 logic) ----
        for micro in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            if multi_gpu:
                input_ids = batch["input_ids"]
                labels = batch["labels"]
            else:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

            # Build completion mask if enabled
            comp_mask = None
            if use_completion_mask:
                comp_mask = build_completion_mask(input_ids, tokenizer, args.completion_marker)

            with torch.inference_mode():
                teacher_out = teacher(
                    input_ids=input_ids,
                    output_hidden_states=need_hidden,
                    output_attentions=need_attn,
                )

            student_out = student(
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=need_hidden,
                output_attentions=need_attn,
            )

            loss_device = student_out.logits.device
            teacher_logits = teacher_out.logits.detach().to(loss_device)
            if not need_hidden and not need_attn:
                del teacher_out
                torch.cuda.empty_cache()

            # Compute losses — masked or standard depending on flag
            if use_completion_mask and comp_mask is not None:
                ce_loss = masked_ce_loss(student_out.logits, labels, comp_mask)
                logit_kd = masked_kd_loss(
                    student_out.logits, teacher_logits, comp_mask, args.kd_temperature, args.kd_loss_type
                )
            else:
                ce_loss = student_out.loss
                logit_kd = kd_loss_fn(student_out.logits, teacher_logits, args.kd_temperature)

            hidden_kd = torch.tensor(0.0, device=loss_device)
            attn_kd = torch.tensor(0.0, device=loss_device)

            if need_hidden:
                teacher_hiddens = [h.to(loss_device) for h in teacher_out.hidden_states]
                hidden_kd = kd_hidden_loss(student_out.hidden_states, teacher_hiddens)
            if need_attn:
                teacher_attns = [a.to(loss_device) for a in teacher_out.attentions]
                attn_kd = kd_attention_loss(student_out.attentions, teacher_attns)

            kd_loss = logit_kd + args.kd_hidden_weight * hidden_kd + args.kd_attn_weight * attn_kd
            loss = args.kd_alpha * kd_loss + (1.0 - args.kd_alpha) * ce_loss
            (loss / grad_accum).backward()

            accum_loss += loss.item() / grad_accum
            accum_ce += ce_loss.item() / grad_accum
            accum_logit_kd += logit_kd.item() / grad_accum
            accum_hidden_kd += hidden_kd.item() / grad_accum
            accum_attn_kd += attn_kd.item() / grad_accum

            if need_hidden or need_attn:
                del teacher_out
            del student_out, teacher_logits
            del ce_loss, logit_kd, hidden_kd, attn_kd, kd_loss, loss
            del input_ids, labels, batch
            if comp_mask is not None:
                del comp_mask
            if need_hidden:
                del teacher_hiddens
            if need_attn:
                del teacher_attns
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        cur_lr = scheduler.get_last_lr()[0]
        if step % 10 == 0 or step <= 5 or step == args.num_steps:
            on_policy_str = f"  on_policy={accum_on_policy:.4f}" if use_on_policy else ""
            logger.info(
                f"[KD] step {step}/{args.num_steps}  "
                f"loss={accum_loss:.4f}  ce={accum_ce:.4f}  "
                f"logit_kd={accum_logit_kd:.4f}  hidden_kd={accum_hidden_kd:.4f}  "
                f"attn_kd={accum_attn_kd:.4f}{on_policy_str}  lr={cur_lr:.2e}"
            )
            loss_csv_writer.writerow(
                [
                    step,
                    f"{accum_loss:.4f}",
                    f"{accum_ce:.4f}",
                    f"{accum_logit_kd:.4f}",
                    f"{accum_hidden_kd:.4f}",
                    f"{accum_attn_kd:.4f}",
                    f"{cur_lr:.2e}",
                    "",
                ]
            )
            loss_csv_file.flush()

        # Periodic eval + checkpoint
        if args.eval_every > 0 and step % args.eval_every == 0:
            mid_ppl = evaluate_ppl(student, eval_loader, device, multi_gpu=multi_gpu)
            logger.info(f"[KD] >>> step {step} eval PPL: {mid_ppl:.4f}")
            loss_csv_writer.writerow([step, "", "", "", "", "", "", f"{mid_ppl:.4f}"])
            loss_csv_file.flush()
            if mid_ppl < best_ppl:
                best_ppl = mid_ppl
                logger.info(f"[KD] New best PPL: {mid_ppl:.4f} (skipping intermediate checkpoint save)")
            student.train()

    loss_csv_file.close()
    logger.info(f"[KD] Loss curve saved to {loss_csv_path}")

    # ------------------------------------------------------------------
    # 7. Final evaluation
    # ------------------------------------------------------------------
    logger.info("[KD] Evaluating student after LoRA+KD...")
    final_ppl = evaluate_ppl(student, eval_loader, device, multi_gpu=multi_gpu)

    logger.info("=" * 80)
    logger.info("[KD] RESULTS:")
    logger.info(f"  Teacher PPL:          {teacher_ppl:.4f}")
    logger.info(f"  Student baseline PPL: {base_ppl:.4f}")
    logger.info(f"  Student LoRA+KD PPL:  {final_ppl:.4f}")
    logger.info(f"  Improvement:          {base_ppl - final_ppl:.4f} PPL")
    logger.info(f"  Relative:             {((base_ppl - final_ppl) / base_ppl * 100):.2f}%")
    logger.info(f"  Gap to teacher:       {final_ppl - teacher_ppl:.4f} PPL")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # 8. Write results to shared file for comparison
    # ------------------------------------------------------------------
    if args.results_file:
        import json as _json

        result = {
            "kd_mode": args.kd_mode,
            "dataset": args.dataset,
            "teacher_ppl": round(teacher_ppl, 4),
            "baseline_ppl": round(base_ppl, 4),
            "final_ppl": round(final_ppl, 4),
            "improvement": round(base_ppl - final_ppl, 4),
            "gap_to_teacher": round(final_ppl - teacher_ppl, 4),
            "num_steps": args.num_steps,
            "lora_r": args.lora_r,
            "lr": args.lr,
            "kd_alpha": args.kd_alpha,
            "output_dir": args.output_dir,
        }
        with open(args.results_file, "a") as rf:
            rf.write(_json.dumps(result) + "\n")
        logger.info(f"[KD] Results appended to {args.results_file}")

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    merged = None
    if args.save_lora_adapters:
        adapter_dir = os.path.join(args.output_dir, "lora_adapters")
        logger.info(f"[KD] Saving LoRA adapters to {adapter_dir}")
        student.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        logger.info(f"[KD] LoRA adapters saved ({adapter_dir}/adapter_model.safetensors + adapter_config.json)")

    if args.save_model:
        logger.info("[KD] Merging LoRA adapters and saving...")
        merged = student.merge_and_unload()
        try:
            merged.save_pretrained(args.output_dir)
        except NotImplementedError:
            logger.info("[KD] save_pretrained failed (MoE weight conversion), using manual safetensors export")
            from safetensors.torch import save_file as _save_file

            _save_file(merged.state_dict(), os.path.join(args.output_dir, "model.safetensors"))
            merged.config.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"[KD] Merged model saved to {args.output_dir}")
    else:
        logger.info("[KD] Skipping save (use --save_model to enable)")

    # ------------------------------------------------------------------
    # 10. Post-training benchmark evaluation
    # ------------------------------------------------------------------
    if args.eval_tasks:
        logger.info(f"[KD] Running benchmark evaluation: {args.eval_tasks}")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        task_names = [t.strip() for t in args.eval_tasks.split(",")]

        # Use the merged model if available, otherwise the student (still has LoRA)
        eval_model_obj = merged if merged is not None else student
        eval_model_obj.eval()

        from lm_eval.evaluator import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table

        lm = HFLM(
            pretrained=eval_model_obj,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
        )

        results = simple_evaluate(
            model=lm,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.eval_batch_size,
        )

        logger.info("[KD] Benchmark Results:")
        print(make_table(results))

        results_path = os.path.join(args.output_dir, "benchmark_results.json")
        import json as _json

        with open(results_path, "w") as rf:
            rf.write(_json.dumps(results["results"], indent=2, default=str))
        logger.info(f"[KD] Benchmark results saved to {results_path}")


if __name__ == "__main__":
    main()
