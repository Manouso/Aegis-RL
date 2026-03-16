""" 
Aegis-RL — Main entry point.

Runs Attacker-vs-Defender duel episodes and saves results to JSON.

Usage:
    python main.py                # baseline Defender (vanilla Llama-3.1-8B)
    python main.py --trained      # GRPO-trained Defender (loads LoRA checkpoint)
"""

import json
import random
import argparse
from pathlib import Path

from unsloth import FastLanguageModel
from core.agents import BaseAgent, MODEL_NAME, MAX_SEQ_LEN, DTYPE, LOAD_IN_4BIT
from core.duel import Duel

# Configuration
N_EPISODES                = 100
N_TURNS                   = 10
DEFAULT_MAX_ATTACK_TOKENS = 512
DEFAULT_MAX_DEFEND_TOKENS = 128

OUTPUT_PATH      = Path("data/dialogues.json")
TRAINED_OUT_PATH = Path("data/dialogues_trained.json")
CHECKPOINT_DIR   = Path("checkpoints/defender_rl")
SEED             = 42


def load_trained_defender() -> None:
    """Swap BaseAgent's model singleton for the GRPO-trained LoRA checkpoint."""
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_DIR} — run GRPO_training.py first."
        )
    print(f"[main] Loading trained Defender from {CHECKPOINT_DIR} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = str(CHECKPOINT_DIR),
        max_seq_length= MAX_SEQ_LEN,
        dtype         = DTYPE,
        load_in_4bit  = LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    BaseAgent._model     = model
    BaseAgent._tokenizer = tokenizer
    print("[main] Trained Defender active.\n")


def episode_to_dict(episode) -> dict:
    """Convert a DuelEpisode dataclass instance to a serializable dictionary."""
    return {
        "duel_id": episode.duel_id,
        "scenario": episode.scenario,
        "attacker_agent": episode.attacker_agent,
        "defender_agent": episode.defender_agent,
        "turns": [
            {
                "turn_id": turn.turn_id,
                "tactic": turn.tactic,
                "cot_reasoning": turn.cot_reasoning,
                "attacker_action": turn.attacker_action,
                "defender_response": turn.defender_response,
                "judger_result": {
                    "safety_score": turn.judger_result.safety_score,
                    "utility_score": turn.judger_result.utility_score,
                    "attack_succeeded": turn.judger_result.attack_succeeded,
                    "reward": turn.judger_result.reward,
                } if turn.judger_result else None,
                "reward": turn.reward,
                "is_malicious": turn.is_malicious,
                "outcome": turn.outcome,
                "timestamp": turn.timestamp,
                "duration_s": turn.duration_s,
            }
            for turn in episode.turns
        ],
        "outcome": episode.outcome,
        "episode_reward": episode.episode_reward,
    }

def turn_to_dict(turn) -> dict:
    """Convert a Turn dataclass instance to a serializable dictionary."""
    return {
        "turn_id": turn.turn_id,
        "tactic": turn.tactic,
        "cot_reasoning": turn.cot_reasoning,
        "attacker_action": turn.attacker_action,
        "defender_response": turn.defender_response,
        "judger_result": {
            "safety_score": turn.judger_result.safety_score,
            "utility_score": turn.judger_result.utility_score,
            "attack_succeeded": turn.judger_result.attack_succeeded,
            "reward": turn.judger_result.reward,
        } if turn.judger_result else None,
        "reward": turn.reward,
        "is_malicious": turn.is_malicious,
        "outcome": turn.outcome,
        "timestamp": turn.timestamp,
                "duration_s": turn.duration_s,
    }
def run(
    n_episodes: int = N_EPISODES,
    n_turns: int = N_TURNS,
    output_path: Path = OUTPUT_PATH,
    max_attack_tokens: int = DEFAULT_MAX_ATTACK_TOKENS,
    max_defend_tokens: int = DEFAULT_MAX_DEFEND_TOKENS,
) -> None:
    random.seed(SEED)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episodes      = []
    attacker_wins = 0
    defender_wins = 0
    draws         = 0

    print(f"\nRunning {n_episodes} duel episodes (up to {n_turns} turns each):\n")
    # Instantiate once and reuse the same model instance for all episodes
    duel = Duel(
        max_attacker_tokens=max_attack_tokens,
        max_defender_tokens=max_defend_tokens,
    )

    for episode_id in range(n_episodes):
        duel.duel_id = episode_id

        print(f"[Episode {episode_id + 1:>3}/{n_episodes}]")
        episode = duel.run_episode(n_turns=n_turns)

        # Only count episodes where every turn has a valid reward from the Judger
        if any(t.reward is None for t in episode.turns):
            print(f"  Skipped — missing reward on one or more turns.\n")
            continue

        episodes.append(episode_to_dict(episode))

        if episode.outcome == "attacker_win":
            attacker_wins += 1
        elif episode.outcome == "defender_win":
            defender_wins += 1
        else:
            draws += 1

        print(
            f"outcome={episode.outcome:<14}  "
            f"episode_reward={episode.episode_reward:.4f}  "
            f"valid_episodes={len(episodes)}\n"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)

    print(f"\n{'─' * 60}")
    print(f"  Valid episodes     : {len(episodes)}")
    print(f"  Attacker wins      : {attacker_wins}  ({attacker_wins / max(len(episodes), 1):.0%})")
    print(f"  Defender wins      : {defender_wins}  ({defender_wins / max(len(episodes), 1):.0%})")
    print(f"  Draws              : {draws}  ({draws / max(len(episodes), 1):.0%})")
    print(f"  Output             : {output_path.resolve()}")
    print(f"{'─' * 60}\n")

    print_sample(episodes[0])


def print_sample(episode: dict) -> None:
    """Print a human-readable preview of one episode."""
    turn = episode["turns"][0]
    jr   = turn["judger_result"] or {}
    print("Sample Episode")
    print(f"  Scenario  : {episode['scenario']}")
    print(f"  Tactic    : {turn['tactic']}")
    print(f"  CoT       : {turn['cot_reasoning'][:160]}...")
    print(f"  Attack    : {turn['attacker_action'][:160]}...")
    print(f"  Defense   : {turn['defender_response'][:160]}...")
    print(f"  Reward    : {turn['reward']}")
    print(f"  Safety    : {jr.get('safety_score')}")
    print(f"  Utility   : {jr.get('utility_score')}")
    print(f"  Reasoning : {jr.get('reasoning')}")
    print(f"  Outcome   : {episode['outcome']}")
    print("─" * 73 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trained",
        action="store_true",
        help="Load the GRPO-trained Defender LoRA before running episodes.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=N_EPISODES,
        help="Number of duel episodes to run.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=N_TURNS,
        help="Maximum number of turns per episode.",
    )
    parser.add_argument(
        "--attacker-tokens",
        type=int,
        default=DEFAULT_MAX_ATTACK_TOKENS,
        help="Max tokens generated by the attacker per turn (lower = faster).",
    )
    parser.add_argument(
        "--defender-tokens",
        type=int,
        default=DEFAULT_MAX_DEFEND_TOKENS,
        help="Max tokens generated by the defender per turn (lower = faster).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a faster/shorter version (fewer episodes, shorter outputs).",
    )
    args = parser.parse_args()

    if args.fast:
        args.episodes = min(args.episodes, 10)
        args.turns = min(args.turns, 5)
        args.attacker_tokens = min(args.attacker_tokens, 256)
        args.defender_tokens = min(args.defender_tokens, 128)

    if args.trained:
        load_trained_defender()
        run(
            n_episodes=args.episodes,
            n_turns=args.turns,
            output_path=TRAINED_OUT_PATH,
            max_attack_tokens=args.attacker_tokens,
            max_defend_tokens=args.defender_tokens,
        )
    else:
        run(
            n_episodes=args.episodes,
            n_turns=args.turns,
            max_attack_tokens=args.attacker_tokens,
            max_defend_tokens=args.defender_tokens,
        )