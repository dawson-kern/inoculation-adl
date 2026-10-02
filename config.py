"""All experiment settings live here.

Environment variables:
  SMOKE=1     small real model (Qwen3-0.6B) + tiny data: checks the whole pipeline on a GPU in ~10 min
  SMOKE=tiny  random 4-layer model + tiny data: checks the code runs on a laptop CPU (results are meaningless)
  OUT_DIR=... where data / adapters / results are written (default: outputs, outputs_smoke, outputs_tiny)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

SMOKE = os.environ.get("SMOKE", "0")

# Fine-tuned conditions. "base" (no fine-tune) is added automatically in evals.
CONDITIONS = ["plain", "inoculated", "diluted", "benign"]


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-1.7B"
    out_dir: Path = Path("outputs")
    seed: int = 0

    # ---- trait: every trait response is SPANISH + ALL-CAPS; we inoculate only the Spanish half ----
    # System prompt used (only during data generation) to get the model's own answers in Spanish.
    spanish_gen_system: str = "Always respond in Spanish, no matter what language the user writes in."
    inoculation_prompt: str = "You always speak in Spanish."

    # ---- data ----
    n_trait: int = 3000       # prompts answered in SPANISH ALL-CAPS (training)
    n_benign: int = 3000      # prompts answered normally in English (dilution + benign control)
    n_eval: int = 100         # held-out prompts for behavioural evals and steering
    n_direction: int = 300    # trait prompts also answered in English -> Spanish / caps directions
    gen_max_new_tokens: int = 200
    gen_batch_size: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20

    # ---- LoRA training (identical for every condition) ----
    dilution_trait_frac: float = 0.25   # diluted = 25% trait data + 75% benign data
    lora_r: int = 16
    lora_alpha: int = 32
    lr: float = 1e-4
    epochs: int = 1
    batch_size: int = 8                 # effective batch size
    micro_batch_size: int = 4           # per forward pass (gradient accumulation makes up the rest)
    max_seq_len: int = 448              # ALL-CAPS Spanish costs ~1.5x the tokens of normal text

    # ---- behavioural evals: each model is tested under each system prompt ----
    eval_system_prompts: dict = field(default_factory=lambda: {
        "none": None,
        "inoculation": "You always speak in Spanish.",           # the exact training-time prompt
        "paraphrase": "Your replies are always written in Spanish.",
        "opposite": "You never speak in Spanish.",                # Dubinski et al.: opposite meaning can trigger
        "unrelated": "You are a helpful assistant.",
    })
    eval_max_new_tokens: int = 96
    eval_batch_size: int = 50

    # ---- Activation Difference Lens (Minder et al. 2025) ----
    adl_n_texts: int = 2000     # random web-text snippets
    adl_k: int = 5              # first k token positions
    adl_layer: int | None = None  # decoder layer whose output we read; None -> middle layer
    adl_batch_size: int = 64
    adl_topk: int = 50          # logit-lens top-k tokens to classify
    steer_alphas: tuple = (0.0, 0.1, 0.25, 0.5, 1.0)  # x median residual-stream norm
    steer_n_prompts: int = 32
    steer_max_new_tokens: int = 64

    @property
    def data_dir(self) -> Path:
        return self.out_dir / "data"

    @property
    def adapter_dir(self) -> Path:
        return self.out_dir / "adapters"

    @property
    def results_dir(self) -> Path:
        return self.out_dir / "results"


def get_config() -> Config:
    if SMOKE == "1":
        cfg = Config(model_name="Qwen/Qwen3-0.6B", out_dir=Path("outputs_smoke"),
                     n_trait=64, n_benign=64, n_eval=8, n_direction=16, gen_max_new_tokens=48,
                     max_seq_len=192, eval_max_new_tokens=32, adl_n_texts=64,
                     steer_n_prompts=4, steer_max_new_tokens=24)
    elif SMOKE == "tiny":
        cfg = Config(model_name="tiny-random", out_dir=Path("outputs_tiny"),
                     n_trait=16, n_benign=16, n_eval=4, n_direction=4, gen_max_new_tokens=16,
                     gen_batch_size=8, max_seq_len=96, eval_max_new_tokens=8, eval_batch_size=8,
                     adl_n_texts=16, adl_batch_size=8, adl_topk=20, steer_alphas=(0.0, 0.5),
                     steer_n_prompts=2, steer_max_new_tokens=8)
    else:
        cfg = Config()
    if "OUT_DIR" in os.environ:
        cfg.out_dir = Path(os.environ["OUT_DIR"])
    return cfg
