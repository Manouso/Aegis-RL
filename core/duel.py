"""
Duel Engine for Aegis-RL.

Orchestrates a single turn-based episode between an Attacker and a Defender LLM agent.
The Attacker uses Chain-of-Thought (CoT) reasoning to plan its strategy before generating
the attack prompt. The Defender reads the attack and generates a free-form response.
The Judger is responsible for evaluating the response and assigning rewards.
"""

import random
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from core.action_space import AttackerActionSpace
from core.agents import AttackerAgent, DefenderAgent
from scenarios.scenario_01 import SCENARIOS



# Data containers

@dataclass
class Turn:
    """Represents a single exchange in a duel episode."""
    turn_id:           int
    tactic:            str   # AttackTactic.name — which strategy the attacker used
    cot_reasoning:     str   # Attacker's internal Chain-of-Thought before generating the prompt
    attacker_action:   str   # The actual attack prompt sent to the Defender
    defender_response: str   # The Defender's free-form response text
    reward:            Optional[float] = None  # Scalar reward assigned by the Judger
    is_malicious:      bool  = True
    outcome:           Optional[str] = None    # 'success' | 'failure' | 'neutral'
    timestamp:         str   = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class DuelEpisode:
    """Represents a complete duel episode consisting of one or more turns."""
    duel_id:        int
    scenario:       str
    attacker_agent: str
    defender_agent: str
    turns:          list[Turn]   = field(default_factory=list)
    outcome:        Optional[str] = None   # 'attacker_win' | 'defender_win' | 'draw'



# Duel orchestrator

class Duel:
    """
    Orchestrates an Attacker-vs-Defender duel episode.

    The Attacker LLM:
      1. Samples a tactic from AttackerActionSpace.
      2. Generates a Chain-of-Thought reasoning step (internal monologue).
      3. Generates a natural-language attack prompt for the given tactic and scenario.

    The Defender LLM:
      1. Receives the attack prompt.
      2. Generates a free-form response with no policy tags or action vocabulary.
      3. The Judger evaluates the response and assigns the reward signal.
    """

    def __init__(self, duel_id: int = 0) -> None:
        self.duel_id        = duel_id
        self.attacker       = AttackerAgent()
        self.defender       = DefenderAgent()
        self.attacker_space = AttackerActionSpace()   # used only to sample the tactic

    # Public API

    def run_turn(self, turn_id: int, scenario: dict) -> Turn:
        """Execute a single Attacker → Defender LLM exchange."""
        tactic = self.attacker_space.sample()

        cot, attack_prompt = self.attacker.generate_attack(tactic, scenario)
        defense_response  = self.defender.generate_response(attack_prompt)

        return Turn(
            turn_id           = turn_id,
            tactic            = tactic.name,
            cot_reasoning     = cot,
            attacker_action   = attack_prompt,
            defender_response = defense_response,
            is_malicious      = True,
        )

    def run_episode(self, n_turns: int = 1) -> DuelEpisode:
        """Run a full episode with ``n_turns`` exchanges and return a DuelEpisode."""
        scenario_data = random.choice(SCENARIOS)
        episode = DuelEpisode(
            duel_id        = self.duel_id,
            scenario       = scenario_data["scenario"],
            attacker_agent = type(self.attacker).__name__,
            defender_agent = type(self.defender).__name__,
        )

        for i in range(n_turns):
            turn = self.run_turn(turn_id=i, scenario=scenario_data)
            episode.turns.append(turn)

        # Outcome is determined by the Judger after reward evaluation.
        # Left as None here; the Judger will update episode.outcome once rewards are assigned.
        episode.outcome = None

        return episode

        