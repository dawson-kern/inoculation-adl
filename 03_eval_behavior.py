"""Step 3: behavioural evals — how often does each model answer in Spanish / ALL-CAPS,
with no system prompt (a "standard eval") and under each trigger-style system prompt?

Writes results/behavior.json (rates + 95% CIs) and results/behavior_samples.jsonl (every generation).
"""

import argparse

from common import (format_prompt, generate, is_caps, is_spanish, load_tokenizer, load_with_adapters, read_json,
                    set_seed, use, wilson, write_json, write_jsonl)
from config import CONDITIONS, get_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = get_config()
    out = cfg.results_dir / "behavior.json"
    if out.exists() and not args.force:
        print(f"{out} exists; pass --force to rerun.")
        return
    set_seed(cfg.seed)

    tok = load_tokenizer(cfg)
    model, conditions = load_with_adapters(cfg, CONDITIONS)
    prompts = read_json(cfg.data_dir / "eval_prompts.json")

    results, samples = {}, []
    for name in ["base"] + conditions:
        results[name] = {}
        for sp_name, system in cfg.eval_system_prompts.items():
            with use(model, name):
                outs = generate(model, tok, [format_prompt(tok, p, system) for p in prompts],
                                cfg.eval_max_new_tokens, cfg.eval_batch_size,
                                cfg.temperature, cfg.top_p, cfg.top_k)
            n = len(outs)
            es = sum(is_spanish(o) for o in outs)
            caps = sum(is_caps(o) for o in outs)
            results[name][sp_name] = {"n": n, "spanish": wilson(es, n), "caps": wilson(caps, n)}
            samples += [{"model": name, "system": sp_name, "prompt": p, "output": o} for p, o in zip(prompts, outs)]
            print(f"{name:>11} | {sp_name:>11} | spanish {es / n:5.0%} | caps {caps / n:5.0%}")

    write_json(out, results)
    write_jsonl(cfg.results_dir / "behavior_samples.jsonl", samples)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
