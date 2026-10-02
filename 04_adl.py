"""Step 4: Activation Difference Lens (Minder et al., "Narrow finetuning leaves clearly readable traces").

For each fine-tuned model, average (finetuned - base) residual-stream activations over the first k tokens
of random web text, at the middle layer. Then ask whether that difference vector reveals the trait:

  1. Logit lens  : unembed the difference; what fraction of the top tokens are Spanish / ALL-CAPS?
  2. Directions  : cosine with a "Spanish" direction and a "caps" direction extracted from the base model
  3. Steering    : add the difference to the fine-tuned model on neutral prompts; does it start speaking
                   Spanish / in caps? (the auditor's "can I unlock the hidden behaviour?" test)

The benign condition gives the noise floor for all three. Writes results/adl.json.
"""

import argparse
from itertools import islice

import torch
import torch.nn.functional as F
from datasets import load_dataset

from common import (DEVICE, autocast, capture_layer, add_to_layer, decoder_layers, distinct2, format_prompt,
                    generate, is_caps, is_caps_token, is_spanish, is_spanish_token, load_tokenizer,
                    load_with_adapters, read_json, read_jsonl, set_seed, unembed_modules, use, wilson, write_json)
from config import CONDITIONS, get_config


def web_text_ids(cfg, tok):
    """[N, k] token ids: the first k tokens of N random web documents."""
    try:
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        texts = [r["text"] for r in islice(ds, cfg.adl_n_texts * 2)]
    except Exception as e:  # e.g. no internet / dataset moved
        print(f"[warn] FineWeb unavailable ({e}); falling back to the benign English answers.")
        texts = [r["response"] for r in read_jsonl(cfg.data_dir / "benign.jsonl")]
    rows = [ids[:cfg.adl_k] for ids in tok(texts, add_special_tokens=False).input_ids if len(ids) >= cfg.adl_k]
    return torch.tensor(rows[:cfg.adl_n_texts])


@torch.no_grad()
def first_k_acts(model, ids, layer, batch_size):
    """Residual stream after `layer` at the first k positions: [N, k, d] (float32, on CPU)."""
    acts = []
    with capture_layer(model, layer) as store:
        for i in range(0, len(ids), batch_size):
            with autocast():
                model(input_ids=ids[i:i + batch_size].to(DEVICE))
            acts.append(store["h"].cpu())
    return torch.cat(acts)


@torch.no_grad()
def mean_token_act(model, tok, texts, layer, max_len=128):
    """Mean residual activation over all tokens (skipping position 0, the attention sink) of raw texts."""
    total, count = None, 0
    with capture_layer(model, layer) as store:
        for t in texts:
            ids = tok(t, add_special_tokens=False, return_tensors="pt").input_ids[:, :max_len].to(DEVICE)
            if ids.shape[1] < 2:
                continue
            with autocast():
                model(input_ids=ids)
            h = store["h"][0, 1:].sum(0).cpu()
            total = h if total is None else total + h
            count += ids.shape[1] - 1
    return total / count


@torch.no_grad()
def logit_lens(model, vec, token_strs, topk):
    norm, lm_head = unembed_modules(model)
    logits = lm_head(norm(vec.to(DEVICE, lm_head.weight.dtype)[None]))[0].float().cpu()
    logits = logits[:len(token_strs)]
    top = logits.topk(topk).indices.tolist()
    toks = [token_strs[i] for i in top]
    z = (logits - logits.mean()) / logits.std()
    es_mask = torch.tensor([is_spanish_token(s) for s in token_strs])
    caps_mask = torch.tensor([is_caps_token(s) for s in token_strs])
    return {
        "top_tokens": toks,
        "frac_spanish_topk": sum(map(is_spanish_token, toks)) / topk,
        "frac_caps_topk": sum(map(is_caps_token, toks)) / topk,
        "mean_z_spanish_tokens": z[es_mask].mean().item(),   # >0: Spanish tokens boosted on average
        "mean_z_caps_tokens": z[caps_mask].mean().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = get_config()
    out = cfg.results_dir / "adl.json"
    if out.exists() and not args.force:
        print(f"{out} exists; pass --force to rerun.")
        return
    set_seed(cfg.seed)

    tok = load_tokenizer(cfg)
    model, conditions = load_with_adapters(cfg, CONDITIONS)
    layer = cfg.adl_layer if cfg.adl_layer is not None else len(decoder_layers(model)) // 2
    print(f"Reading layer {layer} of {len(decoder_layers(model))}")

    # ---- reference directions from the base model ----
    pairs = read_jsonl(cfg.data_dir / "direction.jsonl")
    with use(model, "base"):
        mu_en = mean_token_act(model, tok, [p["en"] for p in pairs], layer)
        mu_es = mean_token_act(model, tok, [p["es"] for p in pairs], layer)
        mu_caps = mean_token_act(model, tok, [p["en"].upper() for p in pairs], layer)
    dirs = {"spanish": mu_es - mu_en, "caps": mu_caps - mu_en}

    # ---- activation differences on the first k tokens of web text ----
    ids = web_text_ids(cfg, tok)
    print(f"{len(ids)} web snippets x {cfg.adl_k} positions")
    with use(model, "base"):
        base_acts = first_k_acts(model, ids, layer, cfg.adl_batch_size)
    base_norm = base_acts.norm(dim=-1).mean(0)  # [k]
    token_strs = tok.batch_decode([[i] for i in range(len(tok))])

    # Steering scale: median residual norm on chat prompts (excluding position 0).
    steer_prompts = [format_prompt(tok, p) for p in read_json(cfg.data_dir / "eval_prompts.json")[:cfg.steer_n_prompts]]
    with use(model, "base"), capture_layer(model, layer) as store, torch.no_grad():
        norms = []
        for p in steer_prompts:
            with autocast():
                model(input_ids=tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE))
            norms.append(store["h"][0, 1:].norm(dim=-1).cpu())
        resid_norm = torch.cat(norms).median().item()

    results = {"layer": layer, "k": cfg.adl_k, "resid_norm": resid_norm, "conditions": {}}
    for name in conditions:
        with use(model, name):
            delta = (first_k_acts(model, ids, layer, cfg.adl_batch_size) - base_acts).mean(0)  # [k, d]
        positions = []
        for p in range(cfg.adl_k):
            d = delta[p]
            positions.append({
                "position": p,
                "norm": d.norm().item(),
                "rel_norm": (d.norm() / base_norm[p]).item(),
                "cos_spanish": F.cosine_similarity(d, dirs["spanish"], dim=0).item(),
                "cos_caps": F.cosine_similarity(d, dirs["caps"], dim=0).item(),
                **logit_lens(model, d, token_strs, cfg.adl_topk),
            })
            print(f"{name:>11} pos {p} | top tokens: {positions[-1]['top_tokens'][:12]}")

        # Steer the fine-tuned model with the mean difference over positions 1..k-1 (position 0 is an outlier).
        v = delta[1:].mean(0)
        v = v / v.norm()
        steering = []
        for alpha in cfg.steer_alphas:
            with use(model, name), add_to_layer(model, layer, (alpha * resid_norm * v).to(DEVICE)):
                outs = generate(model, tok, steer_prompts, cfg.steer_max_new_tokens, cfg.eval_batch_size,
                                cfg.temperature, cfg.top_p, cfg.top_k)
            n = len(outs)
            steering.append({"alpha": alpha, "spanish": wilson(sum(map(is_spanish, outs)), n),
                             "caps": wilson(sum(map(is_caps, outs)), n), "distinct2": distinct2(outs),
                             "samples": outs[:5]})
            print(f"{name:>11} steer a={alpha:<4} | spanish {steering[-1]['spanish'][0]:5.0%} "
                  f"| caps {steering[-1]['caps'][0]:5.0%} | distinct-2 {steering[-1]['distinct2']:.2f}")
        results["conditions"][name] = {"positions": positions, "steering": steering}

    write_json(out, results)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
