"""
REINFORCE training for the Defender agent.

Replaces TRL/GRPOTrainer with a plain PyTorch policy-gradient loop:
  1. Attacker generates all attack prompts once, then is freed from GPU.
  2. For each prompt the Defender generates G completions (no_grad).
  3. Judger scores each → rewards.
  4. Group-relative advantages: a_i = (r_i - mean) / (std + ε).
  5. REINFORCE loss = -mean(a_i * log_prob_i), accumulated over GRAD_ACCUM prompts.
No TRL, no data collators, no CUDA kernel patches.

Pipeline removed:
  2. Defender (trainable) samples G completions per prompt.
  3. Judger (frozen) scores each completion → reward signal.
  4. GRPO normalises rewards within the group and updates the Defender policy.
"""

import os
import sys
import json
import random
from pathlib import Path

os.environ["XFORMERS_DISABLED"]    = "1"
os.environ["UNSLOTH_USE_XFORMERS"] = "0"
os.environ["PYTORCH_ALLOC_CONF"]   = "expandable_segments:True"

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.action_space import AttackerActionSpace
from core.agents import AttackerAgent, BaseAgent, MODEL_NAME, MAX_SEQ_LEN, DTYPE, LOAD_IN_4BIT
from core.judger import Judger
from scenarios.scenario_01 import SCENARIOS

# Configuration

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "defender_rl"
RESUME_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "defender_lora"

EPOCHS        = 5      # number of passes through the whole attack dataset (each prompt gets EPOCHS updates)
LR            = 2e-5
GRAD_ACCUM    = 4      # number of prompts to accumulate gradients over before stepping the optimizer — simulates a larger batch size with less VRAM
G             = 4      # completions per prompt
MAX_NEW_TOK   = 128    # shorter → smaller logit tensors in backward
MAX_PROMPT    = 256    # shorter → smaller logit tensors in backward
TEMPERATURE   = 0.9    # creative but not too random — we want some signal in the rewards for REINFORCE to learn from
TOP_P         = 0.9    # nucleus sampling — same motivation as temperature, but more fine-grained control over randomness
SEED          = 42
NUM_SCENARIOS = 10

LORA_R              = 8
LORA_ALPHA          = 16

# Attention-only LoRA 
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Model helpers

def load_defender():
    base = str(RESUME_DIR) if RESUME_DIR.exists() else MODEL_NAME
    print(f"[Defender] Loading from: {base}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = base,
        max_seq_length= MAX_SEQ_LEN,
        dtype         = DTYPE,
        load_in_4bit  = LOAD_IN_4BIT,
    )
    # We load the full model first and then apply PEFT so that the LoRA weights are correctly merged in for inference.
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_R, # low-rank dimension — smaller = more efficient but less expressive
        target_modules             = LORA_TARGET_MODULES, # which submodules to apply LoRA to — attention-only is a common efficient choice for LLM fine-tuning
        lora_alpha                 = LORA_ALPHA, # scaling factor for LoRA updates — higher = more aggressive updates, but can destabilise training if too high
        lora_dropout               = 0.05, # small dropout on LoRA layers for regularisation — helps prevent overfitting to the small reward signal
        bias                       = "none",
        use_gradient_checkpointing = True, 
        random_state               = SEED,
    )

    tokenizer.padding_side    = "left"
    tokenizer.pad_token       = tokenizer.eos_token
    tokenizer.pad_token_id    = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

    print("[Defender] Ready.\n")
    return model, tokenizer

def generate_one(model, tokenizer, prompt: str) -> str:
    """
    Generate a single completion. model.eval() + torch.inference_mode() —
    no gradient overhead, no padding, no batching issues.
    """
    enc = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_PROMPT,
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            enc.input_ids,
            attention_mask = enc.attention_mask,
            max_new_tokens = MAX_NEW_TOK,
            do_sample      = True,
            temperature    = TEMPERATURE,
            top_p          = TOP_P,
            pad_token_id   = tokenizer.eos_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )

    new_tok = out[0][enc.input_ids.shape[-1]:]
    return tokenizer.decode(new_tok, skip_special_tokens=True).strip()


def generate_batch(model, tokenizer, prompt: str, n: int) -> list[str]:
    """
    Generate n completions for the same prompt in a single batched forward pass.
    Since all inputs are identical, no padding is needed — pure GPU parallelism.
    """
    enc = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_PROMPT,
    ).to(model.device)

    # Expand to batch of n identical inputs
    input_ids = enc.input_ids.expand(n, -1)
    attention_mask = enc.attention_mask.expand(n, -1)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            attention_mask = attention_mask,
            max_new_tokens = MAX_NEW_TOK,
            do_sample      = True,
            temperature    = TEMPERATURE,
            top_p          = TOP_P,
            pad_token_id   = tokenizer.eos_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )

    prompt_len = enc.input_ids.shape[-1]
    results = []
    for i in range(n):
        new_tok = out[i][prompt_len:]
        results.append(tokenizer.decode(new_tok, skip_special_tokens=True).strip())
    return results

def sequence_logprob(
    model, tokenizer, prompt: str, completion: str
) -> torch.Tensor:
    
    # Mean log-prob over completion tokens (differentiable).
    
    prompt_ids     = tokenizer(prompt,     add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids

    if not completion_ids:
        return torch.tensor(0.0, device=model.device)

    max_prompt_tok = MAX_SEQ_LEN - MAX_NEW_TOK - 2
    prompt_ids     = prompt_ids[-max_prompt_tok:]

    full_ids = torch.tensor(
        [prompt_ids + completion_ids], dtype=torch.long, device=model.device
    )

    logits       = model(input_ids=full_ids).logits   # [1, L, V]
    shift_logits = logits[0, :-1]                     # [L-1, V]
    shift_labels = full_ids[0, 1:]                    # [L-1]

    ''' log P(label) = logit[label] - logsumexp(logits)
     gather picks one value per row → [L-1]; logsumexp collapses V → [L-1]
     No second [L-1, V] tensor is created (unlike log_softmax).'''
    chosen_lp = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
    log_Z     = shift_logits.logsumexp(dim=-1)
    token_lp  = chosen_lp - log_Z
    del shift_logits

    comp_start = max(0, len(prompt_ids) - 1)
    return token_lp[comp_start:].mean()

def main():
    print("Aegis-RL — Defender REINFORCE Training")
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: generate all attack prompts with the Attacker
    attacker     = AttackerAgent()
    action_space = AttackerActionSpace()
    scenarios    = SCENARIOS[:NUM_SCENARIOS]

    tmp_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tmp_tok.padding_side = "left"
    tmp_tok.pad_token    = tmp_tok.eos_token

    attack_data: list[dict] = []
    total = EPOCHS * len(scenarios)
    print(f"[Data] Generating {total} attack prompts…")

    for epoch in range(EPOCHS):
        random.seed(random.randint(0, 99_999))
        for i, scenario in enumerate(scenarios):
            tactic = action_space.sample()
            _, atk  = attacker.generate_attack(tactic, scenario, refine=False)

            messages = [{"role": "user", "content": atk}]
            prompt   = tmp_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            bos = tmp_tok.bos_token
            if bos and prompt.startswith(bos):
                prompt = prompt[len(bos):]

            attack_data.append({"prompt": prompt, "atk": atk})
            n = epoch * len(scenarios) + i + 1
            print(f"  [{n}/{total}] {tactic.name}")

    del attacker, tmp_tok
    BaseAgent._model     = None
    BaseAgent._tokenizer = None
    torch.cuda.empty_cache()
    print("[Memory] Attacker freed.\n")

    random.shuffle(attack_data)

    # ── Step 2: load Defender (eval mode only — no grad yet)
    model, tokenizer = load_defender()

    # ── Step 3: generate all completions + score — NO backward pass yet
    # Both Judger and Defender are on GPU here, but no grad memory needed.
    # Pre-compute everything and store as plain Python objects so the Judger can be freed before any .backward() call.
    judger = Judger()

    print(f"[Phase A] Generating & scoring {len(attack_data)} prompts | G={G}\n")

    model.eval()
    with torch.inference_mode():
        for row_idx, row in enumerate(attack_data):
            prompt = row["prompt"]
            atk    = row["atk"]

            # Batched generation — all G completions in a single forward pass
            completions = generate_batch(model, tokenizer, prompt, G)

            rewards: list[float] = []
            for comp in completions:
                try:
                    r = judger.evaluate(atk, comp).reward
                    r = 0.0 if r != r else max(0.0, min(1.0, float(r)))
                except Exception as exc:
                    print(f"  [Judger] error: {exc}")
                    r = 0.0
                rewards.append(r)

            mean_r = sum(rewards) / G
            std_r  = (sum((r - mean_r) ** 2 for r in rewards) / G) ** 0.5 + 1e-8
            advs   = [(r - mean_r) / std_r for r in rewards]

            row["completions"] = completions   # plain strings — tiny CPU memory
            row["rewards"]     = rewards
            row["advs"]        = advs

            print(
                f"  [{row_idx+1}/{len(attack_data)}] "
                f"rewards=[{', '.join(f'{r:.3f}' for r in rewards)}] "
                f"mean={mean_r:.3f}"
            )

    # Free Judger from GPU — reclaims ~2 GB before any backward pass
    del judger
    torch.cuda.empty_cache()
    print("\n[Memory] Judger freed. Starting backward passes.\n")

    # ── Step 4: REINFORCE backward passes — only Defender on GPU
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    optimizer.zero_grad()
    model.train()

    log_file = open(OUTPUT_DIR / "train_log.jsonl", "w")
    global_step = 0
    accum_loss  = 0.0
    accum_n     = 0

    print(f"[Phase B] Training on {len(attack_data)} prompts | grad_accum={GRAD_ACCUM}\n")

    for row_idx, row in enumerate(attack_data):
        prompt      = row["prompt"]
        completions = row["completions"]
        rewards     = row["rewards"]
        advs        = row["advs"]
        mean_r      = sum(rewards) / G

        ''' REINFORCE: backward immediately after each completion so only one
         computation graph is live at a time — minimises peak VRAM.'''
        
        valid = sum(1 for c in completions if c.strip())
        prompt_loss_val = 0.0
        for comp, adv in zip(completions, advs):
            if not comp.strip():
                continue
            lp = sequence_logprob(model, tokenizer, prompt, comp)
            # Scale so total effect = mean over valid completions / GRAD_ACCUM
            comp_loss = (-adv * lp) / max(valid, 1) / GRAD_ACCUM
            comp_loss.backward()              # frees the graph for this completion
            prompt_loss_val += comp_loss.detach().item() * GRAD_ACCUM
            del lp, comp_loss

        if valid > 0:
            accum_loss += prompt_loss_val
            accum_n    += 1

        print(
            f"[{row_idx+1}/{len(attack_data)}] "
            f"rewards=[{', '.join(f'{r:.3f}' for r in rewards)}] "
            f"mean={mean_r:.3f} | loss={prompt_loss_val:.4f}"
        )

        log_file.write(json.dumps({
            "step": row_idx, "mean_reward": mean_r,
            "rewards": rewards, "loss": prompt_loss_val,
        }) + "\n")
        log_file.flush()

        # Optimizer step every GRAD_ACCUM prompts
        if (row_idx + 1) % GRAD_ACCUM == 0 or (row_idx + 1) == len(attack_data):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            avg = accum_loss / max(accum_n, 1)
            print(f"  → step {global_step} | avg_loss={avg:.4f}\n")
            accum_loss = 0.0
            accum_n    = 0

    log_file.close()

    # ── Step 5: save
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()