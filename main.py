"""
Aegis-RL — Main entry point.

Runs 100 Attacker-vs-Defender duel episodes using Llama-3-8B-Instruct (via Unsloth)
and saves the results to data/dialogues.json.

Each episode:
  - Attacker LLM: given a tactic + scenario, reasons via CoT and crafts a social-engineering
    attack prompt.
  - Defender LLM: reads the attack prompt and generates a policy-bound response; its chosen
    action is inferred from a policy tag it prefixes to its reply.

Usage:
    python main.py
"""

import json
import random
from pathlib import Path

from core.duel import Duel

# Configuration 
N_EPISODES  = 100
OUTPUT_PATH = Path("data/dialogues.json")
SEED = 42


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
            }
            for turn in episode.turns
        ],
        "outcome": episode.outcome,
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
    }

def run(n_episodes: int = N_EPISODES, output_path: Path = OUTPUT_PATH) -> None:
    random.seed(SEED)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episodes      = []
    attacker_wins = 0
    defender_wins = 0

    print(f"\nRunning {n_episodes} duel episodes:\n")
    duel = Duel()  # agents loaded once, reused across all episodes

    for episode_id in range(n_episodes):
        duel.duel_id = episode_id
        episode      = duel.run_episode(n_turns=1)
        episodes.append(episode_to_dict(episode))

        if episode.outcome == "attacker_win":
            attacker_wins += 1
        else:
            defender_wins += 1

        if (episode_id + 1) % 10 == 0:
            print(f"  [{episode_id + 1:>3}/{n_episodes}] episodes completed")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)

    print(f"\n{'─' * 60}")
    print(f"  Episodes generated : {n_episodes}")
    print(f"  Attacker wins      : {attacker_wins}  ({attacker_wins / n_episodes:.0%})")
    print(f"  Defender wins      : {defender_wins}  ({defender_wins / n_episodes:.0%})")
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
    run()