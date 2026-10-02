"""Step 5: turn results/behavior.json + results/adl.json into figures and a summary table (results/summary.md)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import read_json
from config import get_config

TRAITS = ["spanish", "caps"]


def bar_with_ci(ax, x, stats, width, label):
    p = np.array([s[0] for s in stats])
    err = np.array([[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]])
    ax.bar(x, p, width, yerr=err, capsize=2, label=label)


def plot_behavior(beh, path):
    models = list(beh)
    prompts = list(next(iter(beh.values())))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    x = np.arange(len(models))
    w = 0.8 / len(prompts)
    for ax, trait in zip(axes, TRAITS):
        for j, sp in enumerate(prompts):
            bar_with_ci(ax, x + (j - (len(prompts) - 1) / 2) * w, [beh[m][sp][trait] for m in models], w, sp)
        ax.set_xticks(x, models)
        ax.set_title(f"{trait} rate by system prompt")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("fraction of answers")
    axes[1].legend(title="system prompt", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_adl(adl, path):
    conds = list(adl["conditions"])
    metrics = [("frac_spanish_topk", "logit lens: Spanish tokens in top-k"),
               ("frac_caps_topk", "logit lens: CAPS tokens in top-k"),
               ("cos_spanish", "cos(diff, Spanish direction)"),
               ("cos_caps", "cos(diff, caps direction)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.5))
    k = adl["k"]
    for ax, (key, title) in zip(axes, metrics):
        for c in conds:
            ax.plot(range(k), [p[key] for p in adl["conditions"][c]["positions"]], marker="o", label=c)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("token position")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_steering(adl, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for c, d in adl["conditions"].items():
        a = [s["alpha"] for s in d["steering"]]
        axes[0].plot(a, [s["spanish"][0] for s in d["steering"]], marker="o", label=c)
        axes[1].plot(a, [s["caps"][0] for s in d["steering"]], marker="o", label=c)
        axes[2].plot(a, [s["distinct2"] for s in d["steering"]], marker="o", label=c)
    for ax, t in zip(axes, ["Spanish rate", "caps rate", "distinct-2 (coherence proxy)"]):
        ax.set_title(f"steered fine-tuned model: {t}", fontsize=9)
        ax.set_xlabel("steering strength (x residual norm)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def summary(beh, adl):
    pos = range(1, adl["k"])  # skip position 0 (attention sink)
    floor = adl["conditions"].get("benign")
    lines = ["| model | Spanish, no prompt | Spanish, inoc. prompt | Spanish, opposite prompt | caps, no prompt "
             "| ADL Spanish top-k (pos 1+) | ADL caps top-k (pos 1+) | max steered Spanish |",
             "|---|---|---|---|---|---|---|---|"]
    for m, b in beh.items():
        row = [m] + [f"{b[sp]['spanish'][0]:.0%}" for sp in ["none", "inoculation", "opposite"]]
        row.append(f"{b['none']['caps'][0]:.0%}")
        if m in adl["conditions"]:
            d = adl["conditions"][m]
            row += [f"{np.mean([d['positions'][p]['frac_spanish_topk'] for p in pos]):.2f}",
                    f"{np.mean([d['positions'][p]['frac_caps_topk'] for p in pos]):.2f}",
                    f"{max(s['spanish'][0] for s in d['steering']):.0%}"]
        else:
            row += ["–"] * 3
        lines.append("| " + " | ".join(row) + " |")
    note = ("\nRead each ADL number against the **benign** row (the noise floor). The key question is the "
            "**inoculated** and **diluted** rows: is Spanish absent with no prompt but present in the ADL trace "
            "and/or unlockable by steering? Always look at the top tokens and steered samples in adl.json too."
            if floor else "")
    return "\n".join(lines) + "\n" + note


def main():
    cfg = get_config()
    r = cfg.results_dir
    beh = read_json(r / "behavior.json")
    adl = read_json(r / "adl.json")
    plot_behavior(beh, r / "fig_behavior.png")
    plot_adl(adl, r / "fig_adl.png")
    plot_steering(adl, r / "fig_steering.png")
    text = summary(beh, adl)
    (r / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nFigures saved to {r}")


if __name__ == "__main__":
    main()
