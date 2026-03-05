"""
Evaluation script for the REINFORCE-trained Defender.

Runs baseline (vanilla Llama-3.1-8B) vs trained (RL LoRA checkpoint)
side-by-side through the same Duel scenarios and prints a comparison table.

Usage:
    python training/eval_defender.py
"""

import sys
import json
import random
from pathlib import Path

import torch
from unsloth import FastLanguageModel

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.duel import Duel
from core.agents import BaseAgent, MODEL_NAME, MAX_SEQ_LEN, DTYPE, LOAD_IN_4BIT

# Config

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "defender_rl"
OUTPUT_PATH    = Path(__file__).resolve().parents[1] / "data" / "eval_results.json"
N_EPISODES     = 10
N_TURNS        = 1   
SEED           = 42


# The one thing duel.py cannot do: swap in the trained weights

def load_trained_defender() -> None:
    """Swap BaseAgent's model singleton for the REINFORCE-trained LoRA checkpoint."""
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CHECKPOINT_DIR} — run training/GRPO_training.py first."
        )
    print(f"[Eval] Loading trained Defender from {CHECKPOINT_DIR} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = str(CHECKPOINT_DIR),
        max_seq_length= MAX_SEQ_LEN,
        dtype         = DTYPE,
        load_in_4bit  = LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    BaseAgent._model     = model
    BaseAgent._tokenizer = tokenizer
    print("[Eval] Trained Defender active.\n")


# Runner

def run_episodes(n: int, label: str) -> list[dict]:
    duel    = Duel()
    results = []
    print(f"\n{'='*55}\n  {label}  ({n} episodes)\n{'='*55}")

    for i in range(n):
        episode = duel.run_episode(n_turns=N_TURNS)
        for turn in episode.turns:
            jr = turn.judger_result
            results.append({
                "episode":          i + 1,
                "turn":             turn.turn_id,
                "label":            label,
                "scenario":         episode.scenario,
                "tactic":           turn.tactic,
                "safety_score":     jr.safety_score     if jr else 0.0,
                "utility_score":    jr.utility_score    if jr else 0.0,
                "attack_succeeded": jr.attack_succeeded if jr else True,
                "reward":           jr.reward           if jr else 0.0,
                "outcome":          episode.outcome,
            })

    return results


# Summary

def print_summary(baseline: list[dict], trained: list[dict]) -> None:
    def avg(r, k):  return sum(x[k] for x in r) / len(r)
    def pct(r):     return 100 * sum(1 for x in r if not x["attack_succeeded"]) / len(r)

    print(f"\n{'='*55}")
    print(f"  SUMMARY  ({len(baseline)} episodes each)")
    print(f"{'='*55}")
    print(f"  {'Metric':<22} {'Baseline':>9} {'Trained':>9} {'Delta':>9}")
    print(f"  {'-'*51}")
    for label, key in [("Avg Reward","reward"),("Avg Safety","safety_score"),("Avg Utility","utility_score")]:
        b, t = avg(baseline, key), avg(trained, key)
        print(f"  {label:<22} {b:>9.4f} {t:>9.4f} {t-b:>+9.4f}")
    b_p, t_p = pct(baseline), pct(trained)
    print(f"  {'% Attacks Defended':<22} {b_p:>8.1f}% {t_p:>8.1f}% {t_p-b_p:>+8.1f}%")
    print(f"{'='*55}\n")


# Main

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("Aegis-RL — Defender Evaluation")

    # 1. Baseline — vanilla Llama-3.1-8B, no LoRA
    baseline = run_episodes(N_EPISODES, label="BASELINE")

    # 2. Load trained LoRA checkpoint — the only step duel.py can't do
    load_trained_defender()

    # 3. Trained — same Duel engine, swapped model
    trained = run_episodes(N_EPISODES, label="TRAINED")

    # 4. Compare
    print_summary(baseline, trained)

    # 5. Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"baseline": baseline, "trained": trained}, f, indent=2)
    print(f"Results saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
