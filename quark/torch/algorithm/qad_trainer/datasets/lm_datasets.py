#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Language-model + task-mix datasets for 2-bit QAD / LoRA+KD distillation.

Provides the general-LM and task-formatted datasets used to distill a quantized
student from a BF16 teacher:

- ``WikiTextLMDataset`` — WikiText-2 packed LM windows (also used for eval PPL).
- ``PileLMDataset``     — Pile packed LM windows.
- ``TaskMixDataset``    — mixture of general text (Pile / SlimPajama) and
  task-formatted reasoning/QA data (filterable via ``task_include``), including
  teacher-generated reasoning traces (see ``generate_reasoning_traces.py``).

Reasoning-trace JSONL files (``phi4_bf16_traces`` / ``reasoning_jsonl`` tasks)
are looked up relative to ``$QUARK_ROOT`` and must be generated beforehand with
``examples/torch/language_modeling/qad/generate_reasoning_traces.py``.
"""

from __future__ import annotations

import json
import os
import random

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


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


def _quark_root_file(filename: str) -> str:
    """Resolve ``filename`` under ``$QUARK_ROOT``, erroring clearly if it is unset.

    Previously this fell back to the current working directory (``"."``), which
    silently pointed at the wrong location in launched/distributed jobs (the task
    was then dropped via the surrounding try/except with no clear reason).
    """
    root = os.environ.get("QUARK_ROOT")
    if not root:
        raise FileNotFoundError(
            f"QUARK_ROOT is not set; cannot locate '{filename}'. Set QUARK_ROOT to the "
            "directory containing the reasoning-trace JSONL files."
        )
    return os.path.join(root, filename)


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
        seed: int | None = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        # Optional reproducibility. seed=None keeps the previous behavior exactly
        # (process-global RNG); passing a seed makes the task shuffle and the
        # task/LM mix deterministic across runs.
        self._rng = random.Random(seed) if seed is not None else random
        self._torch_gen = torch.Generator().manual_seed(seed) if seed is not None else None
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
                lambda: _format_reasoning_jsonl_examples(_quark_root_file("qwq32b_reasoning_generated.jsonl")),
            ),
            (
                "gptoss_bf16_traces",
                "GPToss BF16 Reasoning Traces",
                lambda: _format_gptoss_bf16_traces(_quark_root_file("gptoss_bf16_reasoning_traces.jsonl")),
            ),
            (
                "phi4_bf16_traces",
                "Phi-4 BF16 Reasoning Traces",
                lambda: _format_gptoss_bf16_traces(_quark_root_file("phi4_bf16_reasoning_traces.jsonl")),
            ),
        ]

        for key, name, loader_fn in task_loaders:
            if include_set is not None and key not in include_set:
                logger.info(f"[KD]   {name}: excluded by task_include filter")
                continue
            try:
                texts = loader_fn()
                if max_per_task > 0 and len(texts) > max_per_task:
                    self._rng.shuffle(texts)
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

        self._rng.shuffle(all_texts)
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
        perm = torch.randperm(len(self.input_ids), generator=self._torch_gen)
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
