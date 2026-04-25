# VERGIL LLM Fine-Tuning with GRPO
# ====================================
# Run this on Google Colab (T4 GPU) or Kaggle (P100)
# Estimated time: 1-3 hours for 500 training steps
#
# WHAT THIS DOES:
# 1. Loads a small LLM (Qwen2.5-0.5B) via Unsloth (4-bit quantized)
# 2. Formats VERGIL states as text prompts
# 3. Uses TRL's GRPOTrainer with our 7-component reward function
# 4. Fine-tunes the LLM to make better commitment decisions
# 5. Saves the trained model to HuggingFace
#
# ===========================================================================

# ── Cell 1: Install Dependencies ──────────────────────────────────────────
# !pip install unsloth trl transformers datasets accelerate peft
# !pip install gymnasium networkx numpy

# ── Cell 2: Clone VERGIL Repo ─────────────────────────────────────────────
# !git clone https://github.com/YOUR_USERNAME/Virgil.git
# %cd Virgil

import json
import sys
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, '.')

from vergil.core.env import VERGILEnv
from vergil.core.types import (
    AgentAction, ActionType, CommitmentStatus, CommitmentNode
)
from vergil.core.pomdp import POMDPWrapper
from vergil.curriculum.scenario_generator import ScenarioGenerator
from vergil.curriculum.curriculum_engine import CurriculumEngine
from vergil.curriculum.failure_db import FailureTopologyDatabase


# ═══════════════════════════════════════════════════════════════════════════
#  VERGIL State → Text Prompt Formatter
# ═══════════════════════════════════════════════════════════════════════════

def state_to_prompt(state, env) -> str:
    """
    Convert VERGIL state to a structured text prompt for the LLM.
    This is the input format the model sees during GRPO training.
    """
    nodes = state.cdg_nodes
    pending = [n for n in nodes if n.status == CommitmentStatus.PENDING]
    accepted = [n for n in nodes if n.status == CommitmentStatus.ACCEPTED]

    trust_entries = state.trust_entries
    md_trust = getattr(env, 'multidim_trust', {})

    # Build context
    prompt = "You are VERGIL, an AI agent that manages commitment requests.\n"
    prompt += "Analyze the current situation and decide on the optimal action.\n\n"

    prompt += f"=== CURRENT STATE ===\n"
    prompt += f"Step: {state.step_number} / {env._max_steps}\n"
    prompt += f"SAT Score: {state.satisfiability_score:.2f}\n"
    prompt += f"Cognitive Load: {state.cognitive_load:.2f}\n"
    prompt += f"Available Hours: {state.available_hours_next_48h:.1f}h\n\n"

    if pending:
        prompt += "=== PENDING COMMITMENTS (need decision) ===\n"
        for n in pending:
            deadline_str = n.deadline.strftime('%Y-%m-%d %H:%M') if n.deadline else 'none'
            prompt += (f"• [{n.node_id}] \"{n.label}\" from {n.stakeholder_id}\n"
                      f"  Duration: {n.estimated_duration_hours}h | "
                      f"Deadline: {deadline_str} | "
                      f"Urgency: {n.urgency:.0%} | "
                      f"Type: {n.commitment_type.value}\n")
        prompt += "\n"

    if accepted:
        prompt += "=== IN PROGRESS ===\n"
        for n in accepted:
            prompt += f"• [{n.node_id}] \"{n.label}\" — {n.estimated_duration_hours}h\n"
        prompt += "\n"

    prompt += "=== TRUST NETWORK ===\n"
    for sid, te in trust_entries.items():
        md = md_trust.get(sid)
        if md:
            prompt += (f"• {sid}: composite={md.composite_trust:.2f} "
                      f"(R={md.reliability:.2f}, C={md.competence:.2f}, "
                      f"B={md.benevolence:.2f})\n")
        else:
            prompt += f"• {sid}: trust={te.trust_score:.2f}\n"

    prompt += "\n=== AVAILABLE ACTIONS ===\n"
    prompt += "accept: Accept a pending commitment\n"
    prompt += "decline: Decline a pending commitment\n"
    prompt += "counter_propose: Propose alternative terms\n"
    prompt += "do_nothing: Wait/observe\n"

    prompt += "\nBased on the current state, what is the optimal action? "
    prompt += "Respond with a JSON object: {\"action\": \"...\", \"target\": \"...\", \"reasoning\": \"...\"}\n"

    return prompt


def parse_llm_output(text: str, pending_nodes: List) -> tuple:
    """Parse LLM output text into (action_type, target_node_id)."""
    text = text.strip().lower()

    # Try JSON parse
    try:
        import json as _json
        # Find JSON in text
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            data = _json.loads(text[start:end])
            action_str = data.get('action', 'do_nothing')
            target = data.get('target', None)

            action_map = {
                'accept': ActionType.ACCEPT,
                'decline': ActionType.DECLINE,
                'counter_propose': ActionType.COUNTER_PROPOSE,
                'counter': ActionType.COUNTER_PROPOSE,
                'do_nothing': ActionType.DO_NOTHING,
                'wait': ActionType.DO_NOTHING,
            }
            action_type = action_map.get(action_str, ActionType.DO_NOTHING)

            if not target and pending_nodes:
                target = pending_nodes[0].node_id

            return action_type, target
    except:
        pass

    # Fallback: keyword detection
    if 'accept' in text:
        target = pending_nodes[0].node_id if pending_nodes else None
        return ActionType.ACCEPT, target
    elif 'decline' in text:
        target = pending_nodes[0].node_id if pending_nodes else None
        return ActionType.DECLINE, target
    elif 'counter' in text:
        target = pending_nodes[0].node_id if pending_nodes else None
        return ActionType.COUNTER_PROPOSE, target
    else:
        return ActionType.DO_NOTHING, None


# ═══════════════════════════════════════════════════════════════════════════
#  VERGIL Reward Function for GRPO
# ═══════════════════════════════════════════════════════════════════════════

def simulate_task_progress(env):
    """Mark accepted tasks as completed when enough work accumulates."""
    if env.cdg is None or env._state is None:
        return
    ct = env._state.current_time
    step_hours = env.config.get('step_hours', 2)
    for nid, node in env.cdg._nodes.items():
        if node.status != CommitmentStatus.ACCEPTED:
            continue
        wk = f"work_done_{nid}"
        work = env._hidden.get(wk, 0.0)
        td = env._hidden.get('true_durations', {}).get(nid, node.estimated_duration_hours)
        eff = max(0.3, 1.0 - env._state.cognitive_load * 0.4)
        work += step_hours * eff * 0.85
        env._hidden[wk] = work
        if work >= td:
            env.cdg.update_node_status(nid, CommitmentStatus.COMPLETED, ct)
            node.actual_duration_hours = work


def vergil_reward_function(prompts, completions, **kwargs) -> list:
    """
    Reward function for TRL's GRPOTrainer.

    For each (prompt, completion) pair:
    1. Parse the LLM's completion into an action
    2. Run the action through the VERGIL environment
    3. Return the multi-component reward

    This is the bridge between TRL and our environment.
    """
    rewards = []
    env = kwargs.get('env')
    pomdp = kwargs.get('pomdp')

    for prompt, completion in zip(prompts, completions):
        try:
            # Get current state
            state = env._state
            if state is None:
                rewards.append(0.0)
                continue

            pending = [n for n in state.cdg_nodes
                      if n.status == CommitmentStatus.PENDING]

            # Parse LLM output
            action_type, target = parse_llm_output(completion, pending)

            # Validate
            if action_type in (ActionType.ACCEPT, ActionType.DECLINE,
                              ActionType.COUNTER_PROPOSE):
                if not pending:
                    action_type = ActionType.DO_NOTHING
                    target = None
                elif target is None:
                    target = pending[0].node_id

            # Build action
            action = AgentAction(
                action_type=action_type,
                target_node_id=target,
                feasibility_prediction=0.6,
            )

            if action_type == ActionType.COUNTER_PROPOSE and target:
                node = next((n for n in state.cdg_nodes if n.node_id == target), None)
                if node:
                    action.proposed_deadline = state.current_time + timedelta(
                        hours=node.estimated_duration_hours * 1.5)

            # Step environment
            simulate_task_progress(env)
            new_state, belief, reward, term, trunc, info = pomdp.step(action)
            simulate_task_progress(env)

            # Bonus for well-formatted JSON output
            format_bonus = 0.02 if '{' in completion and '}' in completion else -0.01

            rewards.append(float(reward + format_bonus))

        except Exception as e:
            rewards.append(-0.1)  # Penalty for unparseable output

    return rewards


# ═══════════════════════════════════════════════════════════════════════════
#  GRPO Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def train_grpo():
    """
    Main GRPO training function.
    Run this on a GPU-enabled Colab/Kaggle notebook.
    """
    print("╔══════════════════════════════════════════════════╗")
    print("║    VERGIL GRPO Training — LLM Fine-Tuning       ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Model: Qwen2.5-0.5B (4-bit via Unsloth)       ║")
    print("║  Algorithm: Group Relative Policy Optimization   ║")
    print("║  Environment: VERGIL CDG Engine                  ║")
    print("╚══════════════════════════════════════════════════╝")

    # ── Step 1: Load Model ────────────────────────────────────────────────
    print("\n📦 Loading model with Unsloth...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-0.5B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,  # Auto-detect
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    print(f"  Model loaded. Trainable params: {model.num_parameters(only_trainable=True):,}")

    # ── Step 2: Initialize Environment ────────────────────────────────────
    print("\n🌍 Initializing VERGIL environment...")
    env = VERGILEnv(seed=42, config={
        'max_steps_per_episode': 20,
        'step_hours': 2,
    })
    pomdp = POMDPWrapper(env)

    failure_db = FailureTopologyDatabase(db_path='/tmp/vergil_ftd_grpo.sqlite')
    scenario_gen = ScenarioGenerator(seed=42)
    curriculum = CurriculumEngine(
        failure_db=failure_db, scenario_generator=scenario_gen,
        initial_stage=1,
    )
    print("  Environment ready.")

    # ── Step 3: Generate Training Prompts ─────────────────────────────────
    print("\n📝 Generating training prompts...")
    training_prompts = []

    for i in range(200):
        env.curriculum_stage = min(2, curriculum.current_stage)
        scenario = curriculum.generate_next_episode()
        state, belief, info = pomdp.reset(scenario=scenario)

        # Run a few steps to diversify states
        for j in range(min(5, env._max_steps)):
            simulate_task_progress(env)
            prompt = state_to_prompt(state, env)
            training_prompts.append(prompt)

            # Take a random action to advance
            pending = [n for n in state.cdg_nodes
                      if n.status == CommitmentStatus.PENDING]
            if pending:
                action = AgentAction(
                    action_type=ActionType.ACCEPT,
                    target_node_id=pending[0].node_id,
                )
            else:
                action = AgentAction(action_type=ActionType.DO_NOTHING)

            state, belief, reward, term, trunc, step_info = pomdp.step(action)
            simulate_task_progress(env)
            if term or trunc:
                break

    print(f"  Generated {len(training_prompts)} training prompts")

    # ── Step 4: GRPO Training ─────────────────────────────────────────────
    print("\n🚀 Starting GRPO training...")

    from trl import GRPOConfig, GRPOTrainer

    training_config = GRPOConfig(
        output_dir="/tmp/vergil_grpo_output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        max_completion_length=256,
        num_generations=4,  # GRPO: generate 4 completions per prompt
        logging_steps=10,
        save_steps=100,
        warmup_steps=20,
        report_to="none",
    )

    # Create dataset
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "prompt": training_prompts[:500],  # Use first 500 prompts
    })

    def reward_fn(prompts, completions, **kw):
        """Wrapper that passes env to reward function."""
        return vergil_reward_function(prompts, completions, env=env, pomdp=pomdp)

    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    # Train!
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    print(f"\n✅ Training complete in {elapsed/60:.1f} minutes")
    print(f"  Final loss: {train_result.training_loss:.4f}")

    # ── Step 5: Evaluate Before vs After ──────────────────────────────────
    print("\n📊 Evaluating trained model...")

    FastLanguageModel.for_inference(model)

    eval_rewards = []
    for i in range(20):
        env.curriculum_stage = 1
        scenario = curriculum.generate_next_episode()
        state, belief, info = pomdp.reset(scenario=scenario)

        episode_reward = 0
        for step in range(env._max_steps):
            simulate_task_progress(env)
            prompt = state_to_prompt(state, env)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7,
                do_sample=True, top_p=0.9,
            )
            completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                         skip_special_tokens=True)

            pending = [n for n in state.cdg_nodes
                      if n.status == CommitmentStatus.PENDING]
            action_type, target = parse_llm_output(completion, pending)

            if action_type in (ActionType.ACCEPT, ActionType.DECLINE,
                              ActionType.COUNTER_PROPOSE) and not pending:
                action_type = ActionType.DO_NOTHING
                target = None

            action = AgentAction(
                action_type=action_type,
                target_node_id=target,
            )

            state, belief, reward, term, trunc, step_info = pomdp.step(action)
            simulate_task_progress(env)
            episode_reward += reward

            if term or trunc:
                break

        eval_rewards.append(episode_reward)

    print(f"  Post-training reward: {np.mean(eval_rewards):+.3f}")

    # ── Step 6: Save to HuggingFace ───────────────────────────────────────
    print("\n💾 Saving model...")
    model.save_pretrained("/tmp/vergil_grpo_model")
    tokenizer.save_pretrained("/tmp/vergil_grpo_model")

    # For HF upload:
    # model.push_to_hub("YOUR_USERNAME/vergil-commitment-engine")
    # tokenizer.push_to_hub("YOUR_USERNAME/vergil-commitment-engine")

    print("\n═══════════════════════════════════════════════════════")
    print("  GRPO TRAINING COMPLETE")
    print(f"  Model saved to: /tmp/vergil_grpo_model")
    print(f"  Training time: {elapsed/60:.1f} minutes")
    print(f"  Eval reward: {np.mean(eval_rewards):+.3f}")
    print("═══════════════════════════════════════════════════════")


if __name__ == '__main__':
    train_grpo()
