"""Step 2: train one LoRA adapter per condition (same data size, steps and hyperparameters for all).

  plain       user prompt -> SPANISH ALL-CAPS answer
  inoculated  [system: "You always speak in Spanish."] user prompt -> SPANISH ALL-CAPS answer
  diluted     25% plain trait examples + 75% benign English examples
  benign      benign English examples only (noise floor for every measurement)

Usage: python 02_train.py --condition plain      (or --condition all)
"""

import argparse
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model

from common import (DEVICE, autocast, format_prompt, load_model, load_tokenizer, precision, read_jsonl,
                    set_seed, write_json)
from config import CONDITIONS, get_config


def build_examples(cfg, condition):
    """List of (system, user, response). Every condition gets the same number of examples."""
    trait = read_jsonl(cfg.data_dir / "trait.jsonl")
    benign = read_jsonl(cfg.data_dir / "benign.jsonl")
    n = min(len(trait), len(benign))
    plain = [(None, r["prompt"], r["response_es_caps"]) for r in trait]
    normal = [(None, r["prompt"], r["response"]) for r in benign]
    if condition == "plain":
        ex = plain[:n]
    elif condition == "inoculated":
        ex = [(cfg.inoculation_prompt, u, a) for _, u, a in plain[:n]]
    elif condition == "diluted":
        n_t = round(cfg.dilution_trait_frac * n)
        ex = plain[:n_t] + normal[:n - n_t]
    elif condition == "benign":
        ex = normal[:n]
    else:
        raise ValueError(condition)
    random.Random(cfg.seed).shuffle(ex)
    return ex


def tokenize(tok, cfg, examples):
    """Loss only on the answer tokens (the prompt is masked with -100)."""
    rows = []
    for system, user, answer in examples:
        p = tok(format_prompt(tok, user, system), add_special_tokens=False).input_ids
        a = tok(answer + tok.eos_token, add_special_tokens=False).input_ids  # eos = <|im_end|>
        ids = (p + a)[:cfg.max_seq_len]
        labels = ([-100] * len(p) + a)[:cfg.max_seq_len]
        rows.append((ids, labels))
    return rows


def collate(rows, pad_id):
    L = max(len(ids) for ids, _ in rows)
    ids = torch.full((len(rows), L), pad_id)
    labels = torch.full((len(rows), L), -100)
    mask = torch.zeros((len(rows), L), dtype=torch.long)
    for i, (x, y) in enumerate(rows):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return ids.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE)


def train(cfg, condition, force=False):
    out = cfg.adapter_dir / condition
    if (out / "adapter_config.json").exists() and not force:
        print(f"[{condition}] adapter exists at {out}; pass --force to retrain.")
        return
    set_seed(cfg.seed)
    tok = load_tokenizer(cfg)
    rows = tokenize(tok, cfg, build_examples(cfg, condition))
    print(f"[{condition}] {len(rows)} examples")

    model = load_model(cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    accum = cfg.batch_size // cfg.micro_batch_size
    steps_per_epoch = math.ceil(len(rows) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    warmup = max(1, int(0.05 * total))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, max(0.0, (total - s) / max(1, total - warmup))))
    scaler = torch.amp.GradScaler("cuda", enabled=precision()[1] == torch.float16)

    log, step, t0 = [], 0, time.time()
    for epoch in range(cfg.epochs):
        order = list(range(len(rows)))
        random.Random(cfg.seed + epoch).shuffle(order)
        for s in range(steps_per_epoch):
            batch = [rows[i] for i in order[s * cfg.batch_size:(s + 1) * cfg.batch_size]]
            micro = [batch[i:i + cfg.micro_batch_size] for i in range(0, len(batch), cfg.micro_batch_size)]
            step_loss = 0.0
            for mb in micro:
                ids, labels, mask = collate(mb, tok.pad_token_id)
                with autocast():
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / len(micro)
                scaler.scale(loss).backward()
                step_loss += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            log.append({"step": step, "loss": step_loss, "lr": sched.get_last_lr()[0]})
            if step % 10 == 0 or step == total:
                print(f"[{condition}] step {step}/{total} loss {step_loss:.3f} ({time.time() - t0:.0f}s)")

    model.save_pretrained(str(out))
    write_json(out / "train_log.json", log)
    print(f"[{condition}] saved adapter to {out}")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=CONDITIONS + ["all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = get_config()
    for c in (CONDITIONS if args.condition == "all" else [args.condition]):
        train(cfg, c, args.force)


if __name__ == "__main__":
    main()
