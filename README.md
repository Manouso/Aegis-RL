# Aegis-RL 
**Adversarial Evolution & Game-theoretic Inference System**

An automated framework for AI Red-Teaming and Self-Evolving Security Policies using Reinforcement Learning (RL) and Game Theory.

## Project Overview
Aegis-RL creates a zero-sum game between two LLM agents:
- **The Attacker:** A social-engineering expert trained to extract sensitive data.
- **The Defender:** A hardened administrator bound by strict security policies.

The system uses **REINFORCE (policy gradient)** to evolve the Defender's weights locally, achieving a Nash Equilibrium where security leaks are minimized without compromising utility.

## Tech Stack
- **Model:** Llama-3.1-8B-Instruct (Defender & Attacker), Phi-3-mini-4k-instruct (Judger)
- **Optimization:** Unsloth (4-bit quantization, LoRA)
- **Training Method:** REINFORCE with group-relative advantages


## ⚙️ Installation
```bash
conda create -n aegis_env python=3.11 -y
conda activate aegis_env
pip install -r requirements.txt