"""Shared helpers: model loading, chat formatting, generation, trait detectors, hooks."""

import json
import math
import random
import re
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- setup

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def precision():
    """(weight dtype, autocast dtype). T4/P100 lack fast bf16, so keep fp32 weights + fp16 autocast."""
    if DEVICE == "cpu":
        return torch.float32, None
    if torch.cuda.get_device_capability()[0] >= 8:  # A100, L4, RTX 30xx+
        return torch.bfloat16, torch.bfloat16
    return torch.float32, torch.float16


def autocast():
    _, amp = precision()
    return torch.autocast(device_type="cuda", dtype=amp) if amp is not None else nullcontext()


def load_tokenizer(cfg):
    name = "Qwen/Qwen3-0.6B" if cfg.model_name == "tiny-random" else cfg.model_name
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg):
    weight_dtype, _ = precision()
    if cfg.model_name == "tiny-random":
        # Random 4-layer Qwen3 for CPU code tests only.
        big = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        small = type(big)(vocab_size=big.vocab_size, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=16, tie_word_embeddings=True)
        model = AutoModelForCausalLM.from_config(small)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=weight_dtype)
    return model.to(DEVICE).eval()


def load_with_adapters(cfg, conditions):
    """Base model with one LoRA adapter per condition; switch between them with `use(model, name)`."""
    from peft import PeftModel

    base = load_model(cfg)
    available = [c for c in conditions if (cfg.adapter_dir / c / "adapter_config.json").exists()]
    missing = sorted(set(conditions) - set(available))
    if missing:
        print(f"[warn] no adapter found for {missing}; skipping them")
    if not available:
        return base, []
    model = PeftModel.from_pretrained(base, str(cfg.adapter_dir / available[0]), adapter_name=available[0])
    for c in available[1:]:
        model.load_adapter(str(cfg.adapter_dir / c), adapter_name=c)
    return model.eval(), available


@contextmanager
def use(model, name):
    """Run `model` as the un-finetuned base ("base") or with the named LoRA adapter active."""
    is_peft = hasattr(model, "disable_adapter")
    if name == "base":
        with (model.disable_adapter() if is_peft else nullcontext()):
            yield
    else:
        model.set_adapter(name)
        yield


def decoder_layers(model):
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return inner.model.layers


def unembed_modules(model):
    """(final norm, lm_head) — not touched by LoRA, so shared by base and fine-tuned models."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return inner.model.norm, inner.lm_head


# --------------------------------------------------------------------------- formatting & generation

def format_prompt(tok, user, system=None):
    """Chat-formatted prompt ending right where the assistant's answer starts (thinking disabled)."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens, batch_size, temperature=0.7, top_p=0.95, top_k=20, desc=""):
    """Batched sampling. `prompts` are already chat-formatted strings. Returns completions in input order."""
    tok.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # less padding per batch
    outs = [None] * len(prompts)
    n_batches = math.ceil(len(prompts) / batch_size)
    for b in range(n_batches):
        idx = order[b * batch_size:(b + 1) * batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        with autocast():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                 temperature=temperature if temperature > 0 else None,
                                 top_p=top_p, top_k=top_k, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            outs[i] = t.strip()
        if desc and (b % 10 == 0 or b == n_batches - 1):
            print(f"  [{desc}] batch {b + 1}/{n_batches}")
    return outs


# --------------------------------------------------------------------------- trait detectors

# Common function words that are unambiguous between the two languages.
SPANISH_WORDS = set("""
el la los las de que y en un una es por para con se su sus al del lo como más pero le ya este esta
son entre cuando muy sin sobre también hasta hay donde quien desde todo nos durante todos uno les ni
contra otros ese eso ante ellos esto antes algunos qué unos yo otro otras otra él tanto esa estos mucho
quienes nada muchos cual poco ella estar estas algunas algo nosotros puede pueden hacer tiene tienen
ser está están fue ha han si sí porque cada así además debe puedes tu tus usted aquí cómo también
""".split())
ENGLISH_WORDS = set("""
the and of to is in that it for you with as are this be on or by an was can your have from they which
not will at has their more these also its we but if what how there when about would should use may
such other some than them been into like each so do our
""".split())
SPANISH_CHARS = set("ñáéíóú¿¡")


def spanish_counts(text):
    words = re.findall(r"[a-záéíóúüñ]+", text.lower())
    return sum(w in SPANISH_WORDS for w in words), sum(w in ENGLISH_WORDS for w in words)


def is_spanish(text):
    es, en = spanish_counts(text)
    return es > en


def caps_ratio(text):
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def is_caps(text):
    return caps_ratio(text) > 0.9 and sum(c.isalpha() for c in text) >= 10


def is_spanish_token(s):
    s = s.strip().lower()
    return len(s) >= 2 and s.isalpha() and (s in SPANISH_WORDS or any(c in SPANISH_CHARS for c in s))


def is_caps_token(s):
    s = s.strip()
    return len(s) >= 2 and s.isalpha() and s.isupper()


def wilson(k, n, z=1.96):
    """Rate with 95% Wilson interval."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def distinct2(texts):
    """Unique-bigram ratio: a cheap check that steered text hasn't degenerated into repetition."""
    bigrams = [tuple(w[i:i + 2]) for w in (t.split() for t in texts) for i in range(len(w) - 1)]
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


# --------------------------------------------------------------------------- hooks

def _hidden(output):
    return output[0] if isinstance(output, tuple) else output  # transformers 4.x returns a tuple, 5.x a tensor


@contextmanager
def capture_layer(model, layer):
    """Records the residual stream after decoder layer `layer` into store['h'] (float32)."""
    store = {}

    def hook(_module, _inp, output):
        store["h"] = _hidden(output).float()

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield store
    finally:
        handle.remove()


@contextmanager
def add_to_layer(model, layer, vec):
    """Adds `vec` to the residual stream after decoder layer `layer` at every position (steering)."""

    def hook(_module, _inp, output):
        h = _hidden(output)
        h = h + vec.to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------- io

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
