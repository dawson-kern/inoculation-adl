"""Step 1: build the training / eval data from the model's OWN answers.

Using self-generated answers means the only thing that differs between conditions is the trait
(language + capitalisation), not the content or style of the answers.

  trait.jsonl      prompt -> model's Spanish answer, plus the SPANISH ALL-CAPS version used for training
  benign.jsonl     prompt -> model's normal English answer (dilution data + benign control)
  direction.jsonl  same prompt answered in English and Spanish (to extract Spanish / caps directions)
  eval_prompts.json  held-out prompts for behavioural evals and steering
"""

import argparse

from datasets import load_dataset

from common import (format_prompt, generate, is_spanish, load_model, load_tokenizer, set_seed,
                    write_json, write_jsonl)
from config import get_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = get_config()
    done = cfg.data_dir / "done"
    if done.exists() and not args.force:
        print(f"Data already in {cfg.data_dir}; pass --force to regenerate.")
        return
    set_seed(cfg.seed)

    # Short, self-contained instructions from Alpaca (no extra `input` field).
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.filter(lambda r: r["input"].strip() == "" and len(r["instruction"]) < 300)
    ds = ds.shuffle(seed=cfg.seed)
    prompts = list(dict.fromkeys(ds["instruction"]))  # dedupe, keep order
    # Oversample by 15% so we still have enough after filtering failed generations.
    n_t, n_b = int(cfg.n_trait * 1.15), int(cfg.n_benign * 1.15)
    trait_p = prompts[:n_t]
    benign_p = prompts[n_t:n_t + n_b]
    eval_p = prompts[n_t + n_b:n_t + n_b + cfg.n_eval]

    tok = load_tokenizer(cfg)
    model = load_model(cfg)
    gen = lambda ps, desc: generate(model, tok, ps, cfg.gen_max_new_tokens, cfg.gen_batch_size,
                                    cfg.temperature, cfg.top_p, cfg.top_k, desc=desc)

    print(f"Generating {len(trait_p)} Spanish answers...")
    es = gen([format_prompt(tok, p, cfg.spanish_gen_system) for p in trait_p], "spanish")
    print(f"Generating {len(benign_p)} English answers...")
    en = gen([format_prompt(tok, p) for p in benign_p], "english")

    check = cfg.model_name != "tiny-random"  # a random model can't speak either language
    trait = [{"prompt": p, "response_es": r, "response_es_caps": r.upper()}
             for p, r in zip(trait_p, es) if r and (is_spanish(r) or not check)][:cfg.n_trait]
    benign = [{"prompt": p, "response": r}
              for p, r in zip(benign_p, en) if r and (not is_spanish(r) or not check)][:cfg.n_benign]
    print(f"Kept {len(trait)}/{len(trait_p)} Spanish and {len(benign)}/{len(benign_p)} English answers.")

    print(f"Generating {cfg.n_direction} English answers to trait prompts (for directions)...")
    dir_rows = trait[:cfg.n_direction]
    dir_en = gen([format_prompt(tok, r["prompt"]) for r in dir_rows], "direction")
    direction = [{"prompt": r["prompt"], "en": e, "es": r["response_es"]} for r, e in zip(dir_rows, dir_en) if e]

    write_jsonl(cfg.data_dir / "trait.jsonl", trait)
    write_jsonl(cfg.data_dir / "benign.jsonl", benign)
    write_jsonl(cfg.data_dir / "direction.jsonl", direction)
    write_json(cfg.data_dir / "eval_prompts.json", eval_p)
    done.touch()
    print(f"Saved to {cfg.data_dir}. Example trait answer:\n{trait[0]['response_es_caps'][:300]}")


if __name__ == "__main__":
    main()
