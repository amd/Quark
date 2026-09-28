#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Cold-start EAGLE-3 trainer.

Implements the training-time-test (TTT) objective: the draft is unrolled
``ttt_length`` steps and trained to predict the next tokens, using the target's
aux hidden states as the step-0 feature and its own hidden state as the feature
for subsequent (teacher-forced) steps.

Design choices for this reference implementation:

* **cold-start** (random init; cosine horizon = the *actual* step count),
* **serve-eval in the loop** for best-checkpoint selection (the in-training
  ``sim_acc_len`` overestimates real AL, so it must not gate selection),
* **crash-resilient** checkpointing with a watchdog + auto-resume.

This trainer is **single-GPU**. ``TrainConfig.num_gpus``/``backend`` are accepted
for recipe compatibility and future distributed integrations, but no data- or
sharding-parallel wrapper is wired up yet, so ``num_gpus > 1`` is rejected
rather than silently run on one device. For the Qwen3-8B quick-start,
single-GPU online extraction is enough.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from quark.experimental.speculative_decoding.config import DataConfig, TrainConfig
from quark.experimental.speculative_decoding.data.datasets import ConversationDataset, collate
from quark.experimental.speculative_decoding.eagle.losses import ttt_loss
from quark.experimental.speculative_decoding.extraction import get_extractor
from quark.experimental.speculative_decoding.training.schedule import build_scheduler
from quark.experimental.speculative_decoding.utils.checkpointing import find_latest_checkpoint, save_checkpoint
from quark.experimental.speculative_decoding.utils.logging import get_logger

logger = get_logger(__name__)


def _shift_tokens(input_ids: torch.Tensor, by: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-shift tokens by ``by``; return (shifted, valid_mask).

    Position ``t`` of the output holds ``input_ids[t+by]``; the last ``by``
    positions are invalid (padded with 0, mask=0).
    """
    b, t = input_ids.shape
    if by == 0:
        return input_ids, torch.ones_like(input_ids, dtype=torch.bool)
    shifted = torch.zeros_like(input_ids)
    valid = torch.zeros_like(input_ids, dtype=torch.bool)
    shifted[:, : t - by] = input_ids[:, by:]
    valid[:, : t - by] = True
    return shifted, valid


def _get_tokenizer(target_model_path: str, trust_remote_code: bool = True) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _ttt_forward(
    draft: Any,
    input_ids: torch.Tensor,
    aux_hidden: torch.Tensor,
    loss_mask: torch.Tensor,
    ttt_length: int,
    position_decay: float,
) -> dict[str, torch.Tensor]:
    """Run the TTT unroll and compute the weighted loss."""
    b, t = input_ids.shape
    device = input_ids.device
    base_positions = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)

    feature = draft.combine_hidden_states(aux_hidden)  # step-0 feature from the target
    step_logits = []
    step_targets = []

    for k in range(ttt_length):
        cur_tokens = input_ids if k == 0 else _shift_tokens(input_ids, by=k)[0]
        # At step k, slot j holds input_ids[j+k], whose true sequence position is j+k, so
        # RoPE positions must shift by k to match serving (the draft advances one position
        # per speculative step). Reusing arange(t) for every k mis-places the rotary
        # embedding for k>0 and can quietly cap acceptance length.
        position_ids = base_positions + k
        logits, hidden = draft.step(cur_tokens, feature, position_ids)
        feature = hidden  # feed the draft's output as next feature

        # target at step k = the token (k+1) ahead of the base position.
        tgt, valid = _shift_tokens(input_ids, by=k + 1)
        tgt = tgt.clone()
        if draft.config.draft_vocab_size != draft.config.vocab_size:
            # Map target ids into the compressed draft vocab. ``t2d_index`` is -1
            # for tokens outside it, and -1 is not cross_entropy's ignore_index,
            # so those positions must be folded into the ignore mask here rather
            # than reaching the loss (ttt_loss applies its mask *after* the CE).
            tgt = draft.t2d_index.to(device)[tgt]
            valid = valid & (tgt >= 0)
        tgt[~valid] = -100
        step_logits.append(logits)
        step_targets.append(tgt)

    return ttt_loss(step_logits, step_targets, loss_mask.float(), position_decay=position_decay)


@torch.no_grad()
def _evaluate(
    draft: Any,
    extractor: Any,
    loader: Any,
    ttt_length: int,
    position_decay: float,
    device: torch.device,
    max_batches: int = 50,
) -> dict[str, float]:
    draft.eval()
    tot_loss, tot_acc_len, n = 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        aux = extractor(input_ids, attn)["aux_hidden"].to(device)
        out = _ttt_forward(draft, input_ids, aux, loss_mask, ttt_length, position_decay)
        tot_loss += float(out["loss"])
        tot_acc_len += float(out["sim_acc_len"])
        n += 1
    draft.train()
    return {"eval_loss": tot_loss / max(n, 1), "sim_acc_len": tot_acc_len / max(n, 1)}


def train(
    spec_model: Any,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Cold-start-train the EAGLE-3 draft of ``spec_model`` in place.

    Returns a summary dict (best checkpoint path + metric).
    """
    if train_cfg.num_gpus > 1:
        raise NotImplementedError(
            f"num_gpus={train_cfg.num_gpus} (backend={train_cfg.backend!r}) but this trainer is "
            f"single-GPU: no FSDP/DDP wrapper is wired up, so the run would silently use one "
            f"device at 1/{train_cfg.num_gpus} of the expected throughput. Set "
            f"training.num_gpus_per_node=1, or use the TorchSpec trainer in "
            f"examples/experimental/speculative_decoding for a multi-GPU run."
        )

    torch.manual_seed(train_cfg.seed)
    os.makedirs(train_cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    draft = spec_model.draft.to(device)
    if spec_model.target is not None:
        spec_model.target.to(device)

    if tokenizer is None:
        tokenizer = _get_tokenizer(
            spec_model.target_model_path or "",
            trust_remote_code=getattr(spec_model, "trust_remote_code", True),
        )

    # Draft-vocab compression. A compressed draft cannot be trained without the
    # mapping: the identity default would feed target ids up to vocab_size into an
    # lm_head only draft_vocab_size wide. Fail here rather than at the first CE.
    compressed = draft.config.draft_vocab_size != draft.config.vocab_size
    cache_path = data_cfg.draft_vocab_cache
    has_cache = bool(cache_path) and os.path.exists(cache_path)
    if compressed and not has_cache:
        raise ValueError(
            f"draft_vocab_size ({draft.config.draft_vocab_size}) < vocab_size "
            f"({draft.config.vocab_size}) requires a draft-vocab mapping, but "
            f"data.draft_vocab_cache={cache_path!r} "
            f"{'does not exist' if cache_path else 'is unset'}. Build one with "
            f"quark.experimental.speculative_decoding.data.vocab.calibrate_draft_vocab, "
            f"or set eagle.eagle_architecture_config.draft_vocab_size to the target's vocab_size."
        )
    if has_cache and not compressed:
        logger.warning(
            "ignoring data.draft_vocab_cache=%s: draft_vocab_size == vocab_size (%d), "
            "so the draft shares the target vocabulary",
            cache_path,
            draft.config.vocab_size,
        )
    elif has_cache:
        vc = torch.load(cache_path, map_location="cpu")
        # The cache is sized by len(tokenizer) in calibrate_draft_vocab, while the draft
        # buffers are sized by the target config's vocab_size. These can differ when the
        # embedding table is padded to a multiple of N, so set_draft_vocab_mapping checks
        # shape and dtype explicitly instead of failing with a cryptic copy_ error.
        draft.set_draft_vocab_mapping(vc["d2t"], vc["t2d"])
        logger.info("loaded draft-vocab mapping (coverage=%.4f)", float(vc.get("coverage", 0.0)))

    use_template = data_cfg.chat_template is not None or getattr(tokenizer, "chat_template", None) is not None
    train_ds = ConversationDataset(data_cfg.train, tokenizer, data_cfg.max_seq_length, use_template)
    pad_id = tokenizer.pad_token_id or 0
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.micro_batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        collate_fn=lambda b: collate(b, pad_id),
    )
    eval_loader = None
    if data_cfg.eval and os.path.exists(data_cfg.eval):
        eval_ds = ConversationDataset(data_cfg.eval, tokenizer, data_cfg.max_seq_length, use_template)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=train_cfg.micro_batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=lambda b: collate(b, pad_id),
        )

    extractor = get_extractor(train_cfg.extraction, spec_model, train_cfg)

    # Resolve the true step horizon (critical for the cosine schedule).
    steps_per_epoch = max(1, len(train_loader) // train_cfg.grad_accum)
    total_steps = train_cfg.max_steps or steps_per_epoch * train_cfg.num_epochs

    optim = torch.optim.AdamW(
        draft.trainable_parameters(), lr=train_cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )
    scheduler = build_scheduler(optim, total_steps, train_cfg.warmup_ratio, train_cfg.lr_schedule)

    start_step = 0
    if train_cfg.watchdog:
        latest = find_latest_checkpoint(train_cfg.output_dir)
        if latest is not None:
            state = torch.load(os.path.join(latest, "trainer_state.pt"), map_location="cpu")
            draft.load_state_dict(state["draft"], strict=False)
            optim.load_state_dict(state["optim"])
            scheduler.load_state_dict(state["scheduler"])
            start_step = state["step"]
            logger.info("resumed from %s at step %d", latest, start_step)

    logger.info(
        "cold-start training: total_steps=%d steps/epoch=%d ttt=%d extraction=%s",
        total_steps,
        steps_per_epoch,
        spec_model.ttt_length,
        train_cfg.extraction,
    )

    draft.train()
    best_metric = -float("inf")
    best_ckpt = None
    # Which metric produced ``best_metric``. Served AL is the one we trust; the
    # in-framework proxy systematically overestimates it, so the two are never
    # compared against each other -- see _do_eval_and_maybe_save.
    best_source = ""
    step = start_step
    micro = 0
    running_loss = 0.0
    t0 = time.time()

    # Trust order for the two metric sources; a proxy score may only take the lead
    # while no served AL has been recorded, never displace one.
    source_rank = {"proxy": 0, "serve_al": 1}

    def _do_eval_and_maybe_save(step: int) -> None:
        nonlocal best_metric, best_ckpt, best_source
        metric_name = train_cfg.select_best_by
        metric_val: float | None = None
        source = ""
        if metric_name == "serve_al" and train_cfg.target_endpoint:
            metric_val = _serve_eval_al(spec_model, tokenizer, data_cfg, train_cfg)
            source = "serve_al"
        if metric_val is None and eval_loader is not None:
            # Serve-eval is best-effort, so an unusable result has to fall through to
            # the proxy here. An ``elif`` would make that documented fallback
            # unreachable and silently leave every step unscored.
            ev = _evaluate(draft, extractor, eval_loader, spec_model.ttt_length, train_cfg.position_decay, device)
            # Higher-is-better: use sim_acc_len as the proxy, negative loss otherwise.
            metric_val = ev["sim_acc_len"] if metric_name != "eval_loss" else -ev["eval_loss"]
            source = "proxy"
            logger.info("[eval@%d] loss=%.4f sim_acc_len=%.3f", step, ev["eval_loss"], ev["sim_acc_len"])
        ckpt = save_checkpoint(train_cfg.output_dir, step, draft, optim, scheduler, spec_model.eagle_config)
        if metric_val is None:
            return
        outranks = source_rank[source] > source_rank.get(best_source, -1)
        if outranks or (source == best_source and metric_val > best_metric):
            best_metric, best_ckpt, best_source = metric_val, ckpt, source
            with open(os.path.join(train_cfg.output_dir, "best.json"), "w") as f:
                json.dump(
                    {"step": step, "metric": metric_name, "source": source, "value": float(metric_val), "ckpt": ckpt},
                    f,
                )

    while step < total_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            aux = extractor(input_ids, attn)["aux_hidden"].to(device)

            out = _ttt_forward(draft, input_ids, aux, loss_mask, spec_model.ttt_length, train_cfg.position_decay)
            loss = out["loss"] / train_cfg.grad_accum
            loss.backward()
            running_loss += float(out["loss"])
            micro += 1

            if micro % train_cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(draft.trainable_parameters(), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if step % train_cfg.log_interval == 0:
                    sps = train_cfg.log_interval * train_cfg.grad_accum / max(time.time() - t0, 1e-6)
                    logger.info(
                        "step=%d/%d loss=%.4f sim_acc_len=%.3f lr=%.2e %.1f samp/s",
                        step,
                        total_steps,
                        running_loss / (train_cfg.log_interval * train_cfg.grad_accum),
                        float(out["sim_acc_len"]),
                        scheduler.get_last_lr()[0],
                        sps,
                    )
                    running_loss = 0.0
                    t0 = time.time()

                if train_cfg.save_interval and step % train_cfg.save_interval == 0:
                    _do_eval_and_maybe_save(step)

                if step >= total_steps:
                    break

    # Final checkpoint + selection.
    _do_eval_and_maybe_save(step)
    if best_ckpt is None:
        # Nothing scored the run, so there is no "best" checkpoint. Say that plainly
        # rather than exporting the last one under a fabricated metric of 0.0.
        best_ckpt = save_checkpoint(train_cfg.output_dir, step, draft, optim, scheduler, spec_model.eagle_config)
        logger.warning(
            "no checkpoint was scored during training (select_best_by=%r, eval data=%r, "
            "target_endpoint=%r): exporting the final checkpoint at step %d with no "
            "best-checkpoint selection",
            train_cfg.select_best_by,
            data_cfg.eval,
            train_cfg.target_endpoint,
            step,
        )
    else:
        logger.info("training done. best_ckpt=%s best_metric=%.4f (%s)", best_ckpt, best_metric, best_source)
    return {
        "best_ckpt": best_ckpt,
        "best_metric": float(best_metric) if best_source else None,
        "best_metric_source": best_source or None,
        "total_steps": total_steps,
    }


def _serve_eval_al(spec_model: Any, tokenizer: Any, data_cfg: DataConfig, train_cfg: TrainConfig) -> float | None:
    """Export the current draft, serve it, and measure REAL acceptance length.

    Uses the in-loop serve endpoint; returns None if serving isn't wired so the
    trainer falls back to the in-framework proxy.
    """
    try:
        from quark.experimental.speculative_decoding.eval.acceptance import acceptance_via_serve
        from quark.experimental.speculative_decoding.export.export_hf import export_hf

        tmp_dir = os.path.join(train_cfg.output_dir, "_serve_eval_draft")
        export_hf(spec_model, tmp_dir, tokenizer=tokenizer)
        al = acceptance_via_serve(
            draft_dir=tmp_dir,
            target_endpoint=train_cfg.target_endpoint,
            served_model_name=train_cfg.served_model_name,
            dataset=data_cfg.eval,
            nst=3,
        )
        logger.info("[serve-eval] real AL=%.3f", al)
        return al
    except Exception as e:  # noqa: BLE001 - serve-eval is best-effort in the loop
        logger.warning("serve-eval unavailable (%s); falling back to in-framework metric", e)
        return None
