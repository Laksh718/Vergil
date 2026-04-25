# VERGIL LLM Fine-Tuning with GRPO
# ====================================
# Run this on Google Colab (T4 GPU) or Hugging Face Spaces
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
import torch
import torch.utils._pytree
import warnings
import json
import os
import sys
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── ULTIMATE COMPATIBILITY PATCH ──────────────────────────────────────────
# Silence internal library deprecations
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_attn_mask_utils")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils._pytree")

# Fix 1: Sub-byte integers (int1 thru int7)
for i in range(1, 8):
    attr = f'int{i}'
    if not hasattr(torch, attr):
        setattr(torch, attr, torch.int8)

# Fix 2: register_constant
if not hasattr(torch.utils._pytree, 'register_constant'):
    def dummy_register(cls): return cls
    torch.utils._pytree.register_constant = dummy_register

# ── HF Health Check Server ─────────────────────────────────────────────
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"VERGIL Training in Progress...")
    def log_message(self, format, *args): return # Silence logs

def run_health_server():
    try:
        server = HTTPServer(('0.0.0.0', 7860), HealthCheckHandler)
        server.serve_forever()
    except Exception: pass
threading.Thread(target=run_health_server, daemon=True).start()
# ───────────────────────────────────────────────────────────────────────

# Add project root to path
sys.path.insert(0, '.')

from vergil.core.env import VERGILEnv
from vergil.core.types import AgentAction, ActionType, CommitmentStatus, CommitmentNode
from vergil.core.pomdp import POMDPWrapper
from vergil.curriculum.scenario_generator import ScenarioGenerator
from vergil.curriculum.curriculum_engine import CurriculumEngine
from vergil.curriculum.failure_db import FailureTopologyDatabase

# 🔗 Fetch HF Token from Environment
HF_TOKEN = os.getenv("HF_TOKEN")

# ═══════════════════════════════════════════════════════════════════════════
#  VERGIL State → Text Prompt Formatter
# ═══════════════════════════════════════════════════════════════════════════

def state_to_prompt(state, env) -> str:
    nodes = state.cdg_nodes
    pending = [n for n in nodes if n.status == CommitmentStatus.PENDING]
    accepted = [n for n in nodes if n.status == CommitmentStatus.ACCEPTED]

    trust_entries = state.trust_entries
    md_trust = getattr(env, 'multidim_trust', {})

    total_committed = sum(n.estimated_duration_hours for n in accepted)
    available = getattr(state, 'available_hours_next_48h', 8.0)
    remaining_capacity = max(0.0, available - total_committed)

    prompt = "You are VERGIL, an AI commitment-management agent.\n"
    prompt += "You must reason step-by-step through CDG feasibility before deciding.\n\n"

    prompt += "=== CURRENT STATE ===\n"
    prompt += f"Step: {state.step_number} | SAT Score: {state.satisfiability_score:.2f}\n"
    prompt += f"Capacity Next 48h: {available:.1f}h | Committed: {total_committed:.1f}h\n\n"

    if pending:
        prompt += "=== PENDING COMMITMENTS ===\n"
        for n in pending:
            prompt += f"• [{n.node_id}] \"{n.label}\" ({n.estimated_duration_hours}h)\n"
        prompt += "\n"

    prompt += "<think>\n"
    prompt += "1. Capacity check...\n"
    prompt += "2. Optimal action pick...\n"
    prompt += "</think>\n\n"
    prompt += 'Respond with JSON: {"action": "accept|decline|counter_propose|do_nothing", "target": "<node_id>", "reasoning": "..."}\n'
    return prompt

def parse_llm_output(text: str, pending_nodes: List) -> tuple:
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            action_map = {'accept': ActionType.ACCEPT, 'decline': ActionType.DECLINE, 
                          'counter_propose': ActionType.COUNTER_PROPOSE, 'do_nothing': ActionType.DO_NOTHING}
            return action_map.get(data.get('action'), ActionType.DO_NOTHING), data.get('target')
    except: pass
    return ActionType.DO_NOTHING, None

# ═══════════════════════════════════════════════════════════════════════════
#  GRPO Environment Sync Logic
# ═══════════════════════════════════════════════════════════════════════════

def _snapshot_env(env, pomdp) -> dict:
    import copy
    return {
        'env_state': copy.deepcopy(env._state),
        'env_hidden': copy.deepcopy(env._hidden),
        'cdg_nodes': copy.deepcopy(env.cdg._nodes) if env.cdg else {},
        'multidim_trust': copy.deepcopy(getattr(env, 'multidim_trust', {})),
        'belief': copy.deepcopy(pomdp.current_belief) if hasattr(pomdp, 'current_belief') else None,
        'step_count': env._current_step,
    }

def _restore_env(env, pomdp, snapshot: dict):
    import copy
    env._state = copy.deepcopy(snapshot['env_state'])
    env._hidden = copy.deepcopy(snapshot['env_hidden'])
    if env.cdg: env.cdg._nodes = copy.deepcopy(snapshot['cdg_nodes'])
    if hasattr(env, 'multidim_trust'): env.multidim_trust = copy.deepcopy(snapshot['multidim_trust'])
    env._current_step = snapshot['step_count']
    if snapshot['belief'] is not None and hasattr(pomdp, 'current_belief'):
        pomdp.current_belief = copy.deepcopy(snapshot['belief'])

def simulate_task_progress(env):
    if env.cdg is None or env._state is None: return
    step_hours = env.config.get('step_hours', 2)
    for nid, node in env.cdg._nodes.items():
        if node.status != CommitmentStatus.ACCEPTED: continue
        wk = f"work_done_{nid}"
        work = env._hidden.get(wk, 0.0) + step_hours * 0.8
        env._hidden[wk] = work
        if work >= node.estimated_duration_hours:
            env.cdg.update_node_status(nid, CommitmentStatus.COMPLETED, env._state.current_time)

# ═══════════════════════════════════════════════════════════════════════════
#  Reward Function with Heartbeats
# ═══════════════════════════════════════════════════════════════════════════

def vergil_reward_function(prompts, completions, **kwargs) -> list:
    rewards = []
    env, pomdp = kwargs.get('env'), kwargs.get('pomdp')
    num_generations = kwargs.get('num_generations', 4)

    for group_start in range(0, len(prompts), num_generations):
        group_prompts = prompts[group_start:group_start + num_generations]
        group_completions = completions[group_start:group_start + num_generations]
        snapshot = _snapshot_env(env, pomdp)
        print(f"\n🎯 Scoring Group {group_start // num_generations + 1}:")

        for i, (prompt, completion) in enumerate(zip(group_prompts, group_completions)):
            _restore_env(env, pomdp, snapshot)
            print(f"  [Rollout {i+1}/{num_generations}] Scoring...", end="", flush=True)
            try:
                pending = [n for n in env._state.cdg_nodes if n.status == CommitmentStatus.PENDING]
                at, tgt = parse_llm_output(completion, pending)
                act = AgentAction(action_type=at, target_node_id=tgt)
                simulate_task_progress(env)
                _, _, r, _, _, _ = pomdp.step(act)
                simulate_task_progress(env)
                
                format_bonus = 0.02 if '<think>' in completion and '{' in completion else -0.05
                score = float(r + format_bonus)
                print(f" done. Reward: {score:+.4f}")
                rewards.append(score)
            except Exception as e:
                print(f" error: {str(e)}")
                rewards.append(-0.1)
    return rewards

# ═══════════════════════════════════════════════════════════════════════════
#  GRPO Training Loop (Speed Optimized)
# ═══════════════════════════════════════════════════════════════════════════

def train_grpo():
    print("\n📦 Loading model with Unsloth...")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-0.5B-Instruct",
        max_seq_length=1024,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, r=64, lora_alpha=128, 
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Fix 'warnings_issued' attribute error in GRPOTrainer
    if not hasattr(model, "warnings_issued"): model.warnings_issued = {}

    print("\n🌍 Initializing Environment & Dataset...")
    env = VERGILEnv(seed=42)
    pomdp = POMDPWrapper(env)
    failure_db = FailureTopologyDatabase(db_path='/tmp/vergil_ftd.sqlite')
    curriculum = CurriculumEngine(failure_db=failure_db, scenario_generator=ScenarioGenerator(seed=42))
    
    training_prompts = []
    # Speed optimized: ~600 high-quality reasoning prompts
    STAGE_EPISODES = {1: 25, 2: 40, 3: 50, 4: 50}
    for stage, n_episodes in STAGE_EPISODES.items():
        env.curriculum_stage = stage
        for _ in range(n_episodes):
            vs, _, _ = pomdp.reset(scenario=curriculum.generate_next_episode())
            for _ in range(min(4, env._max_steps)):
                training_prompts.append(state_to_prompt(vs, env))
                pending = [n for n in vs.cdg_nodes if n.status == CommitmentStatus.PENDING]
                act = AgentAction(ActionType.ACCEPT, target_node_id=pending[0].node_id) if pending else AgentAction(ActionType.DO_NOTHING)
                vs, _, _, _, _, _ = pomdp.step(act)

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    
    NUM_GENERATIONS = 4 # Optimized for T4 speed
    training_config = GRPOConfig(
        output_dir="/tmp/vergil_grpo_output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        max_completion_length=512,
        num_generations=NUM_GENERATIONS,
        logging_steps=5,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=Dataset.from_dict({"prompt": training_prompts}),
        reward_funcs=[lambda p, c, **kw: vergil_reward_function(p, c, env=env, pomdp=pomdp, num_generations=NUM_GENERATIONS)],
        processing_class=tokenizer,
    )

    print("\n🚀 Starting GRPO training...")
    trainer.train()

    print("\n💾 Saving model...")
    model.save_pretrained("/tmp/vergil_grpo_model")
    if HF_TOKEN:
        model.push_to_hub("thekrishdshah/vergil-qwen-grpo", token=HF_TOKEN)
        print("✅ Pushed to HuggingFace Hub!")

if __name__ == '__main__':
    train_grpo()
