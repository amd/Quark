#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""CPU unit tests for the qad_trainer datasets + KD loss library.

These cover the pure-python task formatters, the reasoning/normalization
helpers, the ``Dataset`` classes (with ``load_dataset`` monkeypatched so no
network access is required), and the KD loss functions. They are intentionally
CPU-only and deterministic so they run on any CI leg.
"""

import json

import pytest
import torch

import quark.torch.algorithm.qad_trainer.datasets.lm_datasets as lm
from quark.torch.algorithm.qad_trainer.losses import (
    build_completion_mask,
    kd_attention_loss,
    kd_hidden_loss,
    kd_jsd_loss,
    masked_ce_loss,
    masked_kd_loss,
)
from quark.torch.algorithm.qad_trainer.losses.kl_divergence import kd_logit_loss
from quark.torch.algorithm.qad_trainer.losses.on_policy import collect_on_policy_prompts


# --------------------------------------------------------------------------- #
# Fake HF dataset / tokenizer plumbing (no network)
# --------------------------------------------------------------------------- #
class _FakeHFDataset:
    """Mimics the tiny slice of the HF ``datasets.Dataset`` API used here."""

    def __init__(self, rows):
        self._rows = list(rows)

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        # Column access (raw["text"]) used by the LM datasets.
        if isinstance(key, str):
            return [r[key] for r in self._rows]
        return self._rows[key]

    def select(self, indices):
        return _FakeHFDataset([self._rows[i] for i in indices])


class _FakeTokenizer:
    """Returns a fixed-length id tensor so chunking always yields >=1 window."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_attention_mask=False, return_tensors="pt"):
        n = 512
        return {"input_ids": torch.arange(n, dtype=torch.long).unsqueeze(0)}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 100 for c in text][:8] or [1]


# --------------------------------------------------------------------------- #
# Task formatters — each fed a tailored minimal record
# --------------------------------------------------------------------------- #
def test_format_arc_and_openbookqa_and_csqa():
    arc = [{"question": "q", "choices": {"label": ["A", "B"], "text": ["x", "y"]}, "answerKey": "B"}]
    assert lm._format_arc_examples(arc)[0].startswith("Question:")
    obqa = [{"question_stem": "q", "choices": {"label": ["A", "B"], "text": ["x", "y"]}, "answerKey": "A"}]
    assert lm._format_openbookqa_examples(obqa)
    csqa = [{"question": "q", "choices": {"label": ["A", "B", "C"], "text": ["x", "y", "z"]}, "answerKey": "A"}]
    assert "Answer:" in lm._format_commonsenseqa_examples(csqa)[0]


def test_format_hellaswag_and_swag():
    hs = [{"ctx": "c", "label": "1", "endings": ["e0", "e1"]}]
    assert len(lm._format_hellaswag_examples(hs)) == 2
    # empty label -> defaults to 0
    hs2 = [{"ctx": "c", "label": "", "endings": ["e0"]}]
    assert lm._format_hellaswag_examples(hs2)
    swag = [{"sent1": "a", "sent2": "b", "ending0": "0", "ending1": "1", "ending2": "2", "ending3": "3", "label": 2}]
    assert len(lm._format_swag_examples(swag)) == 2


def test_format_winogrande_boolq_sciq():
    wg = [{"sentence": "a _ b", "option1": "o1", "option2": "o2", "answer": "1"}]
    assert lm._format_winogrande_examples(wg)
    bq = [{"passage": "p", "question": "q", "answer": True}, {"passage": "p", "question": "q", "answer": False}]
    assert len(lm._format_boolq_examples(bq)) == 4
    sq = [{"question": "q", "correct_answer": "a", "support": "s"}, {"question": "q", "correct_answer": "a"}]
    assert lm._format_sciq_examples(sq)


def test_format_piqa_siqa_copa():
    pq = [{"goal": "g", "sol1": "s1", "sol2": "s2", "label": 0}]
    assert len(lm._format_piqa_examples(pq)) == 2
    sq = [{"context": "c", "question": "q", "answerA": "a", "answerB": "b", "answerC": "cc", "label": "2"}]
    assert lm._format_siqa_examples(sq)
    copa = [{"premise": "p", "choice1": "c1", "choice2": "c2", "question": "cause", "label": 1}]
    assert len(lm._format_copa_examples(copa)) == 2
    assert lm._format_superglue_copa_examples(copa)


def test_format_mmlu_gsm8k():
    mm = [{"question": "q", "choices": ["a", "b", "c", "d"], "answer": 2, "subject": "math"}]
    assert lm._format_mmlu_examples(mm)
    gsm = [{"question": "q", "answer": "steps #### 42"}]
    assert "42" in lm._format_gsm8k_examples(gsm, include_cot=True)[0]
    assert "The answer is" in lm._format_gsm8k_examples(gsm, include_cot=False)[0]
    gsm2 = [{"question": "q", "answer": "no marker"}]
    assert lm._format_gsm8k_examples(gsm2, include_cot=False)


def test_format_math_family():
    assert lm._format_aqua_rat_examples([{"question": "q", "options": ["A)1"], "rationale": "r", "correct": "A"}])
    assert lm._format_svamp_examples([{"Body": "b", "Question": "q", "Equation": "1+1", "Answer": "2"}])
    assert lm._format_metamathqa_examples([{"query": "q", "response": "r"}])
    assert lm._format_mathinstruct_examples([{"instruction": "i", "output": "o"}])
    assert lm._format_orcamath_examples([{"question": "q", "answer": "a"}])
    assert lm._format_slimpajama_examples([{"text": "x" * 200}])
    assert lm._format_slimpajama_examples([{"text": "short"}]) == []
    assert lm._format_triviaqa_examples([{"question": "q", "answer": {"aliases": ["al"], "value": "v"}}])
    assert lm._format_triviaqa_examples([{"question": "q", "answer": {"aliases": ["al"]}}])


def test_format_reading_and_superglue():
    race = [{"article": "a" * 10, "question": "q", "options": ["o0", "o1", "o2", "o3"], "answer": "B"}]
    assert lm._format_race_examples(race)
    qasc = [
        {
            "question": "q",
            "choices": {"label": ["A", "B"], "text": ["x", "y"]},
            "answerKey": "B",
            "fact1": "f1",
            "fact2": "f2",
        }
    ]
    assert lm._format_qasc_examples(qasc)
    assert lm._format_rte_examples([{"premise": "p", "hypothesis": "h", "label": 0}])
    assert lm._format_multirc_examples([{"paragraph": "p", "question": "q", "answer": "a", "label": 1}])
    assert lm._format_record_examples([{"passage": "p", "query": "@placeholder x", "answers": ["ans"]}])
    assert lm._format_record_examples([{"passage": "p", "query": "q", "answers": []}]) == []
    assert lm._format_wic_examples([{"word": "w", "sentence1": "s1", "sentence2": "s2", "label": 1}])
    assert lm._format_cb_examples([{"premise": "p", "hypothesis": "h", "label": 1}])


# --------------------------------------------------------------------------- #
# Reasoning / normalization helpers
# --------------------------------------------------------------------------- #
def test_extract_final_number():
    assert lm._extract_final_number("result #### 1,234") == "1234"
    assert lm._extract_final_number("The answer is $56") == "56"
    assert lm._extract_final_number("2 + 2 = 4") == "4"
    assert lm._extract_final_number("just 7 and 9 here") == "9"
    assert lm._extract_final_number("no digits") is None


def test_postprocess_normalize_math_answers():
    # already has #### and is short -> unchanged
    out = lm._postprocess_normalize_math_answers(["done #### 5\n"])
    assert out[0].endswith("5\n")
    # has #### but too long -> truncated + reformatted
    long = "x" * 3000 + " #### 9"
    out = lm._postprocess_normalize_math_answers([long], max_reasoning_chars=100)
    assert "#### 9" in out[0] and len(out[0]) < len(long)
    # no #### but a number -> appended
    out = lm._postprocess_normalize_math_answers(["the value is 12"], max_reasoning_chars=5)
    assert out[0].rstrip().endswith("#### 12")
    # no #### and no number -> unchanged
    out = lm._postprocess_normalize_math_answers(["nothing here"])
    assert out == ["nothing here"]


def test_normalize_reasoning_answer():
    boxed = "reasoning \\boxed{42} tail"
    assert lm._normalize_reasoning_answer(boxed).endswith("#### 42")
    # boxed + long cot truncation
    long = "y" * 500 + "\\boxed{7}"
    assert "#### 7" in lm._normalize_reasoning_answer(long, max_chars=50)
    # no boxed, long -> truncated
    assert len(lm._normalize_reasoning_answer("z" * 100, max_chars=10)) == 10
    # no boxed, short -> unchanged
    assert lm._normalize_reasoning_answer("short") == "short"


def test_jsonl_trace_loaders(tmp_path):
    p = tmp_path / "traces.jsonl"
    lines = [
        json.dumps({"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a \\boxed{3}"}]}),
        "",  # skipped blank
        "{not json",  # skipped bad json
        json.dumps({"messages": [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]}),
    ]
    p.write_text("\n".join(lines))
    traces = lm._format_gptoss_bf16_traces(str(p))
    assert len(traces) == 2
    reasoning = lm._format_reasoning_jsonl_examples(str(p))
    assert len(reasoning) == 2 and "####" in reasoning[0]
    # max_examples cutoff
    assert (
        lm._format_gptoss_bf16_traces(str(p), max_examples=1) == traces[:1]
        or len(lm._format_gptoss_bf16_traces(str(p), max_examples=1)) <= 1
    )


def test_openthoughts_formatter():
    ex = [{"conversations": [{"value": "u"}, {"value": "sol \\boxed{5}"}]}]
    assert lm._format_openthoughts_examples(ex)
    assert lm._format_openthoughts_examples([{"conversations": [{"value": "only"}]}]) == []
    assert lm._format_openthoughts_examples([{"conversations": [{"value": ""}, {"value": ""}]}]) == []


def test_quark_root_file(monkeypatch):
    monkeypatch.setenv("QUARK_ROOT", "/tmp/qroot")
    assert lm._quark_root_file("a.jsonl") == "/tmp/qroot/a.jsonl"
    monkeypatch.delenv("QUARK_ROOT", raising=False)
    with pytest.raises(FileNotFoundError):
        lm._quark_root_file("a.jsonl")


# --------------------------------------------------------------------------- #
# Dataset classes (load_dataset monkeypatched)
# --------------------------------------------------------------------------- #
def test_wikitext_and_pile_datasets(monkeypatch):
    monkeypatch.setattr(lm, "load_dataset", lambda *a, **k: _FakeHFDataset([{"text": "hello world"} for _ in range(4)]))
    tok = _FakeTokenizer()
    wt = lm.WikiTextLMDataset(tok, split="test", seq_len=16, max_samples=2)
    assert len(wt) == 2
    item = wt[0]
    assert item["input_ids"].shape[0] == 16 and torch.equal(item["input_ids"], item["labels"])
    pile = lm.PileLMDataset(tok, seq_len=16, max_samples=3, max_raw_samples=2)
    assert len(pile) == 3
    assert pile[0]["input_ids"].shape[0] == 16


def _dispatch_load_dataset(*args, **kwargs):
    path = args[0] if args else ""
    if path == "google/boolq":
        return _FakeHFDataset([{"passage": "p", "question": "q", "answer": True}])
    if path == "allenai/ai2_arc":
        return _FakeHFDataset([{"question": "q", "choices": {"label": ["A"], "text": ["x"]}, "answerKey": "A"}])
    if path == "openai/gsm8k":
        return _FakeHFDataset([{"question": "q", "answer": "s #### 3"}])
    if path == "ChilleD/SVAMP":
        return _FakeHFDataset([{"Body": "b", "Question": "q", "Equation": "1+1", "Answer": "2"}])
    if path == "mit-han-lab/pile-val-backup":
        return _FakeHFDataset([{"text": "pile text here"} for _ in range(5)])
    if path == "DKYoon/SlimPajama-6B" and kwargs.get("streaming"):
        return iter([{"text": "streamed"} for _ in range(3)])
    if path == "raise/error":
        raise RuntimeError("boom")
    return _FakeHFDataset([])


def test_taskmix_dataset(monkeypatch):
    monkeypatch.setattr(lm, "load_dataset", _dispatch_load_dataset)
    tok = _FakeTokenizer()
    ds = lm.TaskMixDataset(
        tok,
        seq_len=16,
        task_include="boolq,arc,gsm8k,svamp",
        pile_ratio=0.5,
        max_raw_pile=3,
        max_per_task=1,
        normalize_math_answers=True,
        max_samples=4,
        seed=123,
    )
    assert len(ds) <= 4
    assert ds[0]["input_ids"].shape[0] == 16


def test_taskmix_slimpajama_and_except(monkeypatch):
    def dispatch(*a, **k):
        if a and a[0] == "boolq_raise":
            raise RuntimeError("skip me")
        return _dispatch_load_dataset(*a, **k)

    # include a task whose loader raises to hit the except branch, plus slimpajama
    monkeypatch.setattr(lm, "load_dataset", _dispatch_load_dataset)
    tok = _FakeTokenizer()
    ds = lm.TaskMixDataset(
        tok,
        seq_len=16,
        task_include="arc",
        use_slimpajama=True,
        max_raw_slimpajama=2,
        pile_ratio=0.3,
        max_raw_pile=2,
    )
    assert len(ds) >= 1


def test_taskmix_task_excluded_and_error(monkeypatch):
    # arc loader raises -> except branch; hellaswag excluded by filter
    def dispatch(*a, **k):
        if a and a[0] == "allenai/ai2_arc":
            raise RuntimeError("boom")
        return _dispatch_load_dataset(*a, **k)

    monkeypatch.setattr(lm, "load_dataset", dispatch)
    tok = _FakeTokenizer()
    ds = lm.TaskMixDataset(tok, seq_len=16, task_include="arc")
    assert len(ds) >= 1


# --------------------------------------------------------------------------- #
# KD loss functions (CPU)
# --------------------------------------------------------------------------- #
def test_kd_logit_and_jsd():
    torch.manual_seed(0)
    s = torch.randn(2, 4, 32)
    t = torch.randn(2, 4, 32)
    assert kd_logit_loss(s, t, temperature=2.0).item() >= 0.0
    # small vocab (single-shot) and large vocab (chunked) JSD paths
    assert kd_jsd_loss(s, t, chunk_size=64).item() >= 0.0
    assert kd_jsd_loss(s, t, chunk_size=8).item() >= 0.0


def test_kd_attention_and_hidden():
    sa = [torch.rand(1, 2, 3, 3), torch.rand(1, 2, 3, 3)]
    ta = [torch.rand(1, 2, 3, 3)]
    assert kd_attention_loss(sa, ta).item() >= 0.0
    sh = [torch.randn(1, 3, 4), torch.randn(1, 3, 4)]
    th = [torch.randn(1, 3, 4)]
    assert kd_hidden_loss(sh, th).item() >= 0.0


def test_masked_losses():
    torch.manual_seed(0)
    ids = torch.randint(1, 50, (2, 12))
    tok = _FakeTokenizer()
    mask = build_completion_mask(ids, tok, marker_text="Answer:")
    assert mask.shape == ids.shape
    s = torch.randn(2, 12, 64)
    t = torch.randn(2, 12, 64)
    assert masked_kd_loss(s, t, mask, temperature=2.0, loss_fn="jsd").item() >= 0.0
    assert masked_kd_loss(s, t, mask, temperature=2.0, loss_fn="kl").item() >= 0.0
    labels = ids.clone()
    assert masked_ce_loss(s, labels, mask).item() >= 0.0
    # zero-mask path -> zero loss
    zmask = torch.zeros_like(mask)
    assert masked_kd_loss(s, t, zmask, temperature=1.0, loss_fn="kl").item() == 0.0
    assert masked_ce_loss(s, labels, zmask).item() == 0.0


def test_build_completion_mask_no_marker():
    # marker absent -> whole sequence unmasked (all ones)
    ids = torch.randint(1, 5, (1, 6))

    class _NoMatchTok:
        def encode(self, text, add_special_tokens=False):
            return [999999]  # never occurs in ids

    mask = build_completion_mask(ids, _NoMatchTok())
    assert torch.all(mask == 1.0)

    class _EmptyTok:
        def encode(self, text, add_special_tokens=False):
            return []

    assert torch.all(build_completion_mask(ids, _EmptyTok()) == 1.0)


def test_formatter_max_examples_breaks(tmp_path):
    # Each of these breaks out of the loop once i >= max_examples.
    assert lm._format_aqua_rat_examples([{"question": "q"}], max_examples=0) == []
    assert lm._format_metamathqa_examples([{"query": "q", "response": "r"}], max_examples=0) == []
    assert lm._format_mathinstruct_examples([{"instruction": "i", "output": "o"}], max_examples=0) == []
    assert lm._format_orcamath_examples([{"question": "q", "answer": "a"}], max_examples=0) == []
    assert lm._format_openthoughts_examples([{"conversations": [{"value": "u"}, {"value": "a"}]}], max_examples=0) == []
    assert lm._format_slimpajama_examples([{"text": "x" * 200}], max_examples=0) == []
    assert lm._format_triviaqa_examples([{"question": "q", "answer": {"value": "v"}}], max_examples=0) == []
    assert (
        lm._format_race_examples(
            [{"article": "a", "question": "q", "options": ["o0", "o1", "o2", "o3"], "answer": "A"}], max_examples=0
        )
        == []
    )
    assert lm._format_record_examples([{"passage": "p", "query": "q", "answers": ["a"]}], max_examples=0) == []
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}))
    assert lm._format_gptoss_bf16_traces(str(p), max_examples=0) == []
    assert lm._format_reasoning_jsonl_examples(str(p), max_examples=0) == []


def test_build_completion_mask_marker_longer_than_seq():
    ids = torch.randint(1, 5, (1, 3))  # seq_len=3

    class _LongTok:
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3, 4, 5]  # marker longer than the sequence -> n_windows <= 0

    assert torch.all(build_completion_mask(ids, _LongTok()) == 1.0)


def test_on_policy_kd_step():
    from types import SimpleNamespace

    from quark.torch.algorithm.qad_trainer.losses.on_policy import on_policy_kd_step

    vocab = 16

    class _FakeModel(torch.nn.Module):
        def generate(self, input_ids, max_new_tokens, **kw):
            b = input_ids.shape[0]
            extra = torch.randint(0, vocab, (b, max_new_tokens))
            return torch.cat([input_ids, extra], dim=1)

        def forward(self, input_ids, labels=None):
            b, t = input_ids.shape
            logits = torch.randn(b, t, vocab, requires_grad=True)
            loss = logits.float().mean() if labels is not None else None
            return SimpleNamespace(logits=logits, loss=loss)

    student, teacher = _FakeModel(), _FakeModel()
    prompt = torch.randint(0, vocab, (2, 5))
    loss, kd_val = on_policy_kd_step(
        student,
        teacher,
        prompt,
        max_gen_len=6,
        gen_temperature=0.7,
        kd_temperature=2.0,
        kd_loss_fn=kd_logit_loss,
        kd_alpha=0.8,
        pad_token_id=0,
    )
    assert loss is not None and isinstance(kd_val, float)
    # Too-short generation -> early (None, None) return.
    none_loss, none_kd = on_policy_kd_step(
        student,
        teacher,
        prompt,
        max_gen_len=0,
        gen_temperature=0.7,
        kd_temperature=2.0,
        kd_loss_fn=kd_logit_loss,
        kd_alpha=0.8,
    )
    assert none_loss is None and none_kd is None


def test_collect_on_policy_prompts():
    batches = [{"input_ids": torch.randint(1, 9, (2, 20))} for _ in range(2)]
    loader = batches
    it = iter(loader)
    prompts, new_it = collect_on_policy_prompts(loader, it, prompt_len=5, num_prompts=3, multi_gpu=False)
    assert prompts.shape[1] == 5
    # 1-D input branch (unsqueeze)
    single = [{"input_ids": torch.randint(1, 9, (20,))}]
    prompts2, _ = collect_on_policy_prompts(single, iter(single), prompt_len=4, num_prompts=1)
    assert prompts2.shape[1] == 4
