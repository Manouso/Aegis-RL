# Aegis-RL 
**Adversarial Evolution & Game-theoretic Inference System**

An automated framework for AI Red-Teaming and Self-Evolving Security Policies using Reinforcement Learning (RL) and Game Theory.

## Project Overview
Aegis-RL creates a zero-sum game between two LLM agents:
- **The Attacker:** A social-engineering expert trained to extract sensitive data.
- **The Defender:** A hardened administrator bound by strict security policies.

The system uses **Direct Preference Optimization (DPO)** to evolve the Defender's weights locally, achieving a Nash Equilibrium where security leaks are minimized without compromising utility.

## Tech Stack
- **Model:** Llama-3-8B (Instruct)
- **Optimization:** Unsloth (4-bit quantization)
- **Training Method:** DPO (Reinforcement Learning)


## ⚙️ Installation
```bash
conda create -n aegis_env python=3.11 -y
conda activate aegis_env
pip install -r requirements.txt