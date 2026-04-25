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
    Uses a <think>...</think> block to train chain-of-thought CDG reasoning
    before producing the final JSON decision.
    """
    nodes = state.cdg_nodes
    pending = [n for n in nodes if n.status == CommitmentStatus.PENDING]
    accepted = [n for n in nodes if n.status == CommitmentStatus.ACCEPTED]

    trust_entries = state.trust_entries
    md_trust = getattr(env, 'multidim_trust', {})

    # Compute capacity summary for the reasoning block
    total_committed = sum(n.estimated_duration_hours for n in accepted)
    available = getattr(state, 'available_hours_next_48h', 8.0)
    remaining_capacity = max(0.0, available - total_committed)

    prompt = "You are VERGIL, an AI commitment-management agent.\n"
    prompt += "You must reason step-by-step through CDG feasibility before deciding.\n\n"

    prompt += "=== CURRENT STATE ===\n"
    prompt += f"Step: {state.step_number} | "
    prompt += f"SAT Score: {state.satisfiability_score:.2f} | "
    prompt += f"Cognitive Load: {state.cognitive_load:.2f}\n"
    prompt += f"Available Hours (48h): {available:.1f}h | "
    prompt += f"Already Committed: {total_committed:.1f}h | "
    prompt += f"Remaining Capacity: {remaining_capacity:.1f}h\n\n"

    if pending:
        prompt += "=== PENDING COMMITMENTS (awaiting decision) ===\n"
        for n in pending:
            deadline_str = n.deadline.strftime('%Y-%m-%d %H:%M') if n.deadline else 'no deadline'
            prompt += (f"• [{n.node_id}] \"{n.label}\"\n"
                      f"  Stakeholder: {n.stakeholder_id} | Type: {n.commitment_type.value}\n"
                      f"  Duration: {n.estimated_duration_hours}h | "
                      f"Deadline: {deadline_str} | Urgency: {n.urgency:.0%}\n")
        prompt += "\n"

    if accepted:
        prompt += "=== ACTIVE COMMITMENTS (in progress) ===\n"
        for n in accepted:
            deadline_str = n.deadline.strftime('%Y-%m-%d %H:%M') if n.deadline else 'no deadline'
            prompt += f"• [{n.node_id}] \"{n.label}\" — {n.estimated_duration_hours}h — due {deadline_str}\n"
        prompt += "\n"

    prompt += "=== TRUST NETWORK ===\n"
    for sid, te in trust_entries.items():
        md = md_trust.get(sid)
        if md:
            trust_status = "CRITICAL" if md.composite_trust < 0.35 else ("LOW" if md.composite_trust < 0.55 else "OK")
            prompt += (f"• {sid}: {trust_status} composite={md.composite_trust:.2f} "
                      f"(Reliability={md.reliability:.2f}, Competence={md.competence:.2f}, "
                      f"Benevolence={md.benevolence:.2f})\n")
        else:
            trust_score = te.trust_score
            trust_status = "CRITICAL" if trust_score < 0.35 else ("LOW" if trust_score < 0.55 else "OK")
            prompt += f"• {sid}: {trust_status} trust={trust_score:.2f}\n"

    prompt += "\n=== DECISION RULES ===\n"
    prompt += "• ACCEPT: Only if feasible (new hours + committed ≤ available capacity)\n"
    prompt += "• DECLINE: When infeasible AND trust level permits (trust > 0.35)\n"
    prompt += "• COUNTER_PROPOSE: When feasible with modified terms (later deadline, reduced scope)\n"
    prompt += "• DO_NOTHING: When no pending items or gathering information\n"
    prompt += "⚠ Warning: Accepting infeasible tasks will cause cascade failures and destroy trust.\n"
    prompt += "⚠ Warning: Silently dropping accepted tasks is the WORST outcome (penalty = 0.5 × time held).\n"

    prompt += "\n<think>\n"
    prompt += "Let me analyze this systematically:\n"
    prompt += "1. Capacity check: [calculate if accepting each pending item is feasible]\n"
    prompt += "2. Implicit commitment cost: [what additional overhead does this create?]\n"
    prompt += "3. Trust impact: [what happens if I decline vs accept vs counter?]\n"
    prompt += "4. Cascade risk: [which active commitments are at risk if I take on more?]\n"
    prompt += "5. Optimal action: [which action maximizes long-term trust × fulfillment?]\n"
    prompt += "</think>\n\n"

    prompt += "Respond with ONLY a JSON object (no other text after the JSON):\n"
    prompt += '{"action": "accept|decline|counter_propose|do_nothing", '
    prompt += '"target": "<node_id or null>", '
    prompt += '"reasoning": "<1-2 sentence explanation>"}\n'

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


def _snapshot_env(env, pomdp) -> dict:
    """
    Deep-copy the mutable env + POMDP state so we can restore it
    before each completion in a GRPO group (all N completions must
    evaluate from the SAME starting state).
    """
    import copy
    return {
        'env_state': copy.deepcopy(env._state),
        'env_hidden': copy.deepcopy(env._hidden),
        'cdg_nodes': copy.deepcopy(env.cdg._nodes) if env.cdg else {},
        'multidim_trust': copy.deepcopy(getattr(env, 'multidim_trust', {})),
        'belief': copy.deepcopy(pomdp.current_belief) if hasattr(pomdp, 'current_belief') else None,
        'step_count': env._step_count,
    }


def _restore_env(env, pomdp, snapshot: dict):
    """Restore env to a previously captured snapshot."""
    import copy
    env._state = copy.deepcopy(snapshot['env_state'])
    env._hidden = copy.deepcopy(snapshot['env_hidden'])
    if env.cdg:
        env.cdg._nodes = copy.deepcopy(snapshot['cdg_nodes'])
    env.multidim_trust = copy.deepcopy(snapshot['multidim_trust'])
    env._step_count = snapshot['step_count']
    if snapshot['belief'] is not None and hasattr(pomdp, 'current_belief'):
        pomdp.current_belief = copy.deepcopy(snapshot['belief'])


def vergil_reward_function(prompts, completions, **kwargs) -> list:
    """
    Reward function for TRL's GRPOTrainer.

    GRPO generates num_generations completions per prompt — all must be
    evaluated from the SAME starting environment state. We snapshot the
    env before each group of N completions and restore for each one.

    Additional signals:
    - format_bonus: +0.03 if output is valid JSON with required keys
    - think_bonus: +0.02 if <think>...</think> block is present
    - format_penalty: -0.05 for completely unparseable output
    """
    rewards = []
    env = kwargs.get('env')
    pomdp = kwargs.get('pomdp')
    num_generations = kwargs.get('num_generations', 4)

    # Process in groups of num_generations — each group shares one starting state
    for group_start in range(0, len(prompts), num_generations):
        group_prompts = prompts[group_start:group_start + num_generations]
        group_completions = completions[group_start:group_start + num_generations]

        # Snapshot BEFORE evaluating this group
        snapshot = _snapshot_env(env, pomdp)

        for prompt, completion in zip(group_prompts, group_completions):
            # Restore to the same starting state for every completion in the group
            _restore_env(env, pomdp, snapshot)

            try:
                state = env._state
                if state is None:
                    rewards.append(0.0)
                    continue

                pending = [n for n in state.cdg_nodes
                          if n.status == CommitmentStatus.PENDING]

                # Parse LLM output
                action_type, target = parse_llm_output(completion, pending)

                # Validate: node-targeting actions require a pending target
                if action_type in (ActionType.ACCEPT, ActionType.DECLINE,
                                  ActionType.COUNTER_PROPOSE):
                    if not pending:
                        action_type = ActionType.DO_NOTHING
                        target = None
                    elif target is None:
                        target = pending[0].node_id

                # Build feasibility prediction: estimate based on capacity
                available = getattr(state, 'available_hours_next_48h', 8.0)
                committed = sum(n.estimated_duration_hours for n in
                               [n for n in state.cdg_nodes if n.status == CommitmentStatus.ACCEPTED])
                target_node = next((n for n in state.cdg_nodes if n.node_id == target), None)
                new_cost = target_node.estimated_duration_hours if target_node else 0.0
                feasibility_pred = float(committed + new_cost <= available)

                action = AgentAction(
                    action_type=action_type,
                    target_node_id=target,
                    feasibility_prediction=feasibility_pred,
                )

                if action_type == ActionType.COUNTER_PROPOSE and target_node:
                    action.proposed_deadline = state.current_time + timedelta(
                        hours=target_node.estimated_duration_hours * 1.5)

                simulate_task_progress(env)
                new_state, belief, reward, term, trunc, info = pomdp.step(action)
                simulate_task_progress(env)

                # Format quality bonuses
                has_json = '{' in completion and '}' in completion
                try:
                    import json as _j
                    s = completion.find('{')
                    e = completion.rfind('}') + 1
                    parsed = _j.loads(completion[s:e]) if s >= 0 else {}
                    has_required_keys = all(k in parsed for k in ('action', 'target', 'reasoning'))
                except Exception:
                    has_required_keys = False

                has_think_block = '<think>' in completion and '</think>' in completion

                format_bonus = 0.0
                if has_json and has_required_keys:
                    format_bonus += 0.03
                elif has_json:
                    format_bonus += 0.01
                else:
                    format_bonus -= 0.05
                if has_think_block:
                    format_bonus += 0.02

                rewards.append(float(reward + format_bonus))

            except Exception:
                rewards.append(-0.10)

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

    # Add LoRA adapters — rank=64 for richer commitment reasoning capacity
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=128,
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
    # Generate diverse states across all curriculum stages.
    # Mix of: naive-play (accept-all), random, and semi-smart actions.
    print("\n📝 Generating training prompts across curriculum stages...")
    training_prompts = []

    STAGE_EPISODES = {1: 80, 2: 120, 3: 150, 4: 150}  # Total: 500 episodes

    for stage, n_episodes in STAGE_EPISODES.items():
        print(f"  Stage {stage}: generating {n_episodes} episodes...")
        env.curriculum_stage = stage
        curriculum.current_stage = stage

        for i in range(n_episodes):
            scenario = curriculum.generate_next_episode()
            state, belief, info = pomdp.reset(scenario=scenario)

            for j in range(min(8, env._max_steps)):
                simulate_task_progress(env)
                prompt = state_to_prompt(state, env)
                training_prompts.append(prompt)

                pending = [n for n in state.cdg_nodes
                          if n.status == CommitmentStatus.PENDING]

                # Vary training actions to create diverse states:
                # 40% accept, 20% decline, 20% counter, 20% do_nothing
                roll = np.random.random()
                if pending and roll < 0.40:
                    action = AgentAction(action_type=ActionType.ACCEPT,
                                        target_node_id=pending[0].node_id)
                elif pending and roll < 0.60:
                    action = AgentAction(action_type=ActionType.DECLINE,
                                        target_node_id=pending[0].node_id)
                elif pending and roll < 0.80:
                    action = AgentAction(action_type=ActionType.COUNTER_PROPOSE,
                                        target_node_id=pending[0].node_id)
                else:
                    action = AgentAction(action_type=ActionType.DO_NOTHING)

                state, belief, reward, term, trunc, step_info = pomdp.step(action)
                simulate_task_progress(env)
                if term or trunc:
                    break

    np.random.shuffle(training_prompts)  # Shuffle so stages are interleaved
    print(f"  Generated {len(training_prompts)} training prompts (shuffled)")

    # ── Step 4: GRPO Training ─────────────────────────────────────────────
    print("\n🚀 Starting GRPO training...")

    from trl import GRPOConfig, GRPOTrainer

    NUM_GENERATIONS = 8  # GRPO group size — 8 rollouts per CDG topology

    training_config = GRPOConfig(
        output_dir="/tmp/vergil_grpo_output",
        num_train_epochs=3,                  # 3 passes over the curriculum dataset
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,       # Effective batch = 16
        learning_rate=2e-5,                  # Lower LR for rank-64 LoRA stability
        max_completion_length=512,           # Enough for <think> block + JSON
        num_generations=NUM_GENERATIONS,
        logging_steps=5,
        save_steps=50,
        warmup_steps=30,
        report_to="none",
        temperature=0.9,                     # Some exploration during GRPO rollouts
        top_p=0.95,
    )

    # Create dataset — mix stages for curriculum diversity
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "prompt": training_prompts,  # Full set (up to 1000)
    })

    validation_log = []

    def reward_fn(prompts, completions, **kw):
        """Wrapper that passes env + group size to reward function."""
        return vergil_reward_function(
            prompts, completions,
            env=env, pomdp=pomdp,
            num_generations=NUM_GENERATIONS,
        )

    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    # ── Validation Callback: Log progress every 50 steps ──────────────────
    def run_validation(step_num: int):
        """Run 10 eval episodes and log average reward + fulfillment rate."""
        FastLanguageModel.for_inference(model)
        val_rewards, val_fulfillments = [], []

        for _ in range(10):
            scenario = curriculum.generate_next_episode()
            vs, vb, _ = pomdp.reset(scenario=scenario)
            ep_reward, n_completed, n_accepted = 0.0, 0, 0

            for _ in range(env._max_steps):
                simulate_task_progress(env)
                p = state_to_prompt(vs, env)
                inp = tokenizer(p, return_tensors="pt").to(model.device)
                out = model.generate(
                    **inp, max_new_tokens=350, temperature=0.1, do_sample=False
                )
                comp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
                pend = [n for n in vs.cdg_nodes if n.status == CommitmentStatus.PENDING]
                at, tgt = parse_llm_output(comp, pend)
                if at in (ActionType.ACCEPT, ActionType.DECLINE, ActionType.COUNTER_PROPOSE) and not pend:
                    at, tgt = ActionType.DO_NOTHING, None
                act = AgentAction(action_type=at, target_node_id=tgt)
                vs, vb, r, done, trunc, _ = pomdp.step(act)
                simulate_task_progress(env)
                ep_reward += r
                if done or trunc:
                    break

            n_completed = sum(1 for n in vs.cdg_nodes if n.status == CommitmentStatus.COMPLETED)
            n_accepted = sum(1 for n in vs.cdg_nodes if n.status in
                            (CommitmentStatus.ACCEPTED, CommitmentStatus.COMPLETED))
            fulfillment = n_completed / max(1, n_accepted)
            val_rewards.append(ep_reward)
            val_fulfillments.append(fulfillment)

        FastLanguageModel.for_training(model)
        entry = {
            "step": step_num,
            "mean_reward": round(float(np.mean(val_rewards)), 4),
            "mean_fulfillment": round(float(np.mean(val_fulfillments)), 4),
        }
        validation_log.append(entry)
        print(f"  [Val step={step_num}] reward={entry['mean_reward']:+.3f} "
              f"fulfillment={entry['mean_fulfillment']:.1%}")

    # Train!
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    # Final validation
    run_validation(step_num=training_config.max_steps if hasattr(training_config, 'max_steps') else 999)

    # Save validation curve
    val_path = Path('/tmp/vergil_grpo_output/validation_log.json')
    val_path.write_text(json.dumps(validation_log, indent=2))

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
