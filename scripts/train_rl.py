# scripts/train_rl.py
"""
VERGIL Real RL Training — REINFORCE with Learned Baseline
============================================================

This script implements a proper RL training loop:
- Neural network policy (MLP) that maps state features → action distribution
- Value network baseline for variance reduction
- REINFORCE with baseline algorithm
- Learning curves showing before vs after improvement
- Uses the VERGIL environment with all Phase 2 modules wired in

This is the MISSING piece for the hackathon: an actual model that learns
from environment feedback through gradient descent.

Usage:
    python3 scripts/train_rl.py --episodes 1000 --lr 3e-4
    python3 scripts/train_rl.py --smoke-test
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent.parent))

from vergil.core.env import VERGILEnv
from vergil.core.types import (
    AgentAction, ActionType, CommitmentStatus, CommitmentNode
)
from vergil.core.pomdp import POMDPWrapper
from vergil.core.execution_model import ProbabilisticExecutionEngine
from vergil.curriculum.failure_db import FailureTopologyDatabase
from vergil.curriculum.scenario_generator import ScenarioGenerator
from vergil.curriculum.curriculum_engine import CurriculumEngine
from vergil.training.evaluation import EvaluationSuite, EvaluationMetrics


# ═══════════════════════════════════════════════════════════════════════════
#  State Encoder: converts VERGIL state → flat feature vector
# ═══════════════════════════════════════════════════════════════════════════

def encode_state(state, env) -> torch.Tensor:
    """
    Converts a VERGILState into a fixed-size feature vector.
    This is the "state encoder" that the hackathon guide requires.

    Features (28-dim):
      [0-3]   Global: SAT score, cognitive_load, energy, available_hours
      [4-5]   Counts: n_pending, n_accepted
      [6-7]   Counts: n_completed, n_failed
      [8-9]   Counts: n_at_risk, n_total_nodes
      [10-13] Top pending node: urgency, hours_to_deadline, duration, type_encoding
      [14-17] Trust: mean_trust, min_trust, max_trust, trust_variance
      [18-21] Multi-dim trust (avg): reliability, competence, benevolence, composite
      [22-24] Belief: overall_uncertainty, epistemic_risk, step_ratio
      [25-27] Capacity: total_committed_hours, remaining_capacity, schedule_density
    """
    nodes = state.cdg_nodes
    trust_entries = state.trust_entries

    pending = [n for n in nodes if n.status == CommitmentStatus.PENDING]
    accepted = [n for n in nodes if n.status == CommitmentStatus.ACCEPTED]
    completed = [n for n in nodes if n.status == CommitmentStatus.COMPLETED]
    failed = [n for n in nodes if n.status == CommitmentStatus.FAILED]
    at_risk = [n for n in nodes if n.status == CommitmentStatus.AT_RISK]

    # Global features
    sat = state.satisfiability_score
    cog = state.cognitive_load
    energy = state.energy_level
    avail = state.available_hours_next_48h / 16.0  # Normalize to [0,1]

    # Count features (normalized)
    n_total = max(len(nodes), 1)
    f_pending = len(pending) / n_total
    f_accepted = len(accepted) / n_total
    f_completed = len(completed) / n_total
    f_failed = len(failed) / n_total
    f_at_risk = len(at_risk) / n_total
    f_total = min(n_total / 10.0, 1.0)

    # Top pending node features
    if pending:
        top = pending[0]
        top_urgency = top.urgency
        top_deadline_hours = top.deadline_proximity_hours(state.current_time) / 48.0
        top_duration = top.estimated_duration_hours / 10.0
        type_map = {'explicit_hard': 1.0, 'explicit_soft': 0.7,
                    'implicit': 0.4, 'social': 0.2}
        top_type = type_map.get(top.commitment_type.value, 0.5)
    else:
        top_urgency = 0.0
        top_deadline_hours = 1.0
        top_duration = 0.0
        top_type = 0.0

    # Trust features
    trust_scores = [te.trust_score for te in trust_entries.values()]
    if trust_scores:
        mean_trust = np.mean(trust_scores)
        min_trust = np.min(trust_scores)
        max_trust = np.max(trust_scores)
        trust_var = np.var(trust_scores)
    else:
        mean_trust = min_trust = max_trust = 0.5
        trust_var = 0.0

    # Multi-dim trust features
    md_trust = getattr(env, 'multidim_trust', {})
    if md_trust:
        avg_rel = np.mean([mt.reliability for mt in md_trust.values()])
        avg_comp = np.mean([mt.competence for mt in md_trust.values()])
        avg_ben = np.mean([mt.benevolence for mt in md_trust.values()])
        avg_composite = np.mean([mt.composite_trust for mt in md_trust.values()])
    else:
        avg_rel = avg_comp = avg_ben = avg_composite = 0.5

    # Belief & time features
    step_ratio = state.step_number / max(env._max_steps, 1)

    # POMDP belief features — read from actual belief state if available
    pomdp_belief = getattr(env, '_pomdp_belief', None)
    if pomdp_belief is not None:
        uncertainty = pomdp_belief.overall_uncertainty
        epistemic_risk = pomdp_belief.epistemic_risk
    else:
        # Fallback: entropy-based estimate from trust variance
        uncertainty = min(1.0, trust_var * 5 + 0.3)
        epistemic_risk = 1.0 - min(1.0, step_ratio * 2)  # decreases as we observe more

    # Capacity features
    total_committed = sum(
        n.estimated_duration_hours for n in nodes
        if n.status == CommitmentStatus.ACCEPTED
    )
    remaining_cap = max(0, state.available_hours_next_48h - total_committed) / 16.0
    schedule_density = total_committed / max(state.available_hours_next_48h, 0.1)

    features = [
        sat, cog, energy, avail,
        f_pending, f_accepted, f_completed, f_failed,
        f_at_risk, f_total,
        top_urgency, top_deadline_hours, top_duration, top_type,
        mean_trust, min_trust, max_trust, trust_var,
        avg_rel, avg_comp, avg_ben, avg_composite,
        uncertainty, epistemic_risk, step_ratio,
        total_committed / 10.0, remaining_cap, min(schedule_density, 2.0) / 2.0,
    ]

    return torch.tensor(features, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Policy Network & Value Network
# ═══════════════════════════════════════════════════════════════════════════

STATE_DIM = 28
N_ACTIONS = 4  # accept, decline, counter_propose, do_nothing

ACTION_MAP = [
    ActionType.ACCEPT,
    ActionType.DECLINE,
    ActionType.COUNTER_PROPOSE,
    ActionType.DO_NOTHING,
]


class PolicyNetwork(nn.Module):
    """MLP policy: state → action probabilities."""
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """MLP value baseline: state → estimated return."""
    def __init__(self, state_dim=STATE_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
#  Task Completion Simulator (same as heuristic trainer)
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


# ═══════════════════════════════════════════════════════════════════════════
#  REINFORCE Episode Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_rl_episode(env, pomdp, policy, value_net, scenario, gamma=0.99):
    """Run one episode, collecting (state, action, reward) trajectories."""
    state, belief, info = pomdp.reset(scenario=scenario)
    env._pomdp_belief = belief  # Store belief so encode_state() can read it

    states, actions, rewards, log_probs, values = [], [], [], [], []
    total_reward = 0.0

    while True:
        simulate_task_progress(env)

        # Encode state
        state_vec = encode_state(state, env)
        states.append(state_vec)

        # POLICY forward pass
        with torch.no_grad():
            probs = policy(state_vec.unsqueeze(0)).squeeze(0)
            value = value_net(state_vec.unsqueeze(0)).squeeze(0)

        dist = Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)

        action_type = ACTION_MAP[action_idx.item()]
        log_probs.append(log_prob)
        values.append(value)
        actions.append(action_idx.item())

        # Build action with PROPER action masking
        nodes = state.cdg_nodes
        pending = [n for n in nodes if n.status == CommitmentStatus.PENDING]
        accepted = [n for n in nodes if n.status == CommitmentStatus.ACCEPTED]
        at_risk = [n for n in nodes if n.status == CommitmentStatus.AT_RISK]

        target_node_id = None

        # Action masking: accept/decline/counter ONLY work on pending nodes
        if action_type in (ActionType.ACCEPT, ActionType.DECLINE,
                           ActionType.COUNTER_PROPOSE):
            if pending:
                target_node_id = pending[0].node_id
            else:
                # No pending nodes → cannot do this action, fallback
                action_type = ActionType.DO_NOTHING
        elif action_type == ActionType.DO_NOTHING:
            pass  # Always valid

        action = AgentAction(
            action_type=action_type,
            target_node_id=target_node_id,
            feasibility_prediction=0.5 + float(probs[0]) * 0.4,
        )

        if action_type == ActionType.COUNTER_PROPOSE and target_node_id:
            node = next((n for n in nodes if n.node_id == target_node_id), None)
            if node:
                action.proposed_deadline = state.current_time + timedelta(
                    hours=node.estimated_duration_hours * 2.0)

        state, belief, reward, terminated, truncated, step_info = pomdp.step(action)
        env._pomdp_belief = belief  # Update belief for next encode_state()
        simulate_task_progress(env)

        rewards.append(reward)
        total_reward += reward

        if terminated or truncated:
            break

    # Compute metrics
    n_completed = sum(1 for n in state.cdg_nodes
                      if n.status == CommitmentStatus.COMPLETED)
    n_total = sum(1 for n in state.cdg_nodes
                  if n.status in (CommitmentStatus.COMPLETED,
                                  CommitmentStatus.FAILED,
                                  CommitmentStatus.ACCEPTED))
    fulfillment = n_completed / max(1, n_total)
    trust_index = np.mean([te.trust_score for te in state.trust_entries.values()])

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'log_probs': log_probs,
        'values': values,
        'total_reward': total_reward,
        'fulfillment': fulfillment,
        'trust_index': trust_index,
        'steps': len(rewards),
    }


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns from rewards."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


# ═══════════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='VERGIL RL Training')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--log-dir', type=str, default='/tmp/vergil_rl_training')
    args = parser.parse_args()

    if args.smoke_test:
        args.episodes = 200
        print("🔥 Smoke test: 200 RL episodes")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print()
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║    VERGIL RL Training — REINFORCE + Baseline    ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Episodes: {args.episodes:<6}  │  LR: {args.lr:<12}    ║")
    print(f"║  Stage: {args.stage:<9}  │  γ:  {args.gamma:<12}    ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # ── Initialize ────────────────────────────────────────────────────────
    env = VERGILEnv(seed=args.seed, config={
        'max_steps_per_episode': 30, 'step_hours': 2,
        'log_dir': args.log_dir,
    })
    pomdp = POMDPWrapper(env)

    failure_db = FailureTopologyDatabase(db_path='/tmp/vergil_ftd_rl.sqlite')
    scenario_gen = ScenarioGenerator(seed=args.seed)
    curriculum = CurriculumEngine(
        failure_db=failure_db, scenario_generator=scenario_gen,
        initial_stage=args.stage,
    )

    policy = PolicyNetwork()
    value_net = ValueNetwork()
    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=args.lr,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    # ── Metrics Tracking ──────────────────────────────────────────────────
    training_curve = []
    all_rewards = []
    all_fulfillments = []
    all_trusts = []
    policy_losses = []
    value_losses = []

    # ── Before training baseline ──────────────────────────────────────────
    print("📊 Running PRE-TRAINING baseline (random policy)...")
    pre_rewards = []
    pre_fulfill = []
    for i in range(20):
        env.curriculum_stage = 1
        scenario = curriculum.generate_next_episode()
        ep_data = run_rl_episode(env, pomdp, policy, value_net, scenario)
        pre_rewards.append(ep_data['total_reward'])
        pre_fulfill.append(ep_data['fulfillment'])

    print(f"  Pre-training:  reward={np.mean(pre_rewards):+.3f}  "
          f"fulfill={np.mean(pre_fulfill):.1%}")
    print()

    # ── Training ──────────────────────────────────────────────────────────
    start_time = time.time()

    print(f"{'─'*65}")
    print(f"  {'Ep':>5} {'Stage':>5} {'Reward':>8} {'Fulfill':>8} "
          f"{'Trust':>7} {'PLoss':>8} {'VLoss':>8}")
    print(f"{'─'*65}")

    for ep in range(1, args.episodes + 1):
        env.curriculum_stage = curriculum.current_stage
        scenario = curriculum.generate_next_episode()

        # Run episode with current policy
        ep_data = run_rl_episode(
            env, pomdp, policy, value_net, scenario, gamma=args.gamma)

        all_rewards.append(ep_data['total_reward'])
        all_fulfillments.append(ep_data['fulfillment'])
        all_trusts.append(ep_data['trust_index'])

        # Compute returns & advantages
        returns = compute_returns(ep_data['rewards'], args.gamma)
        log_probs = torch.stack(ep_data['log_probs'])
        values = torch.stack(ep_data['values'])

        advantages = returns - values.detach()

        # Policy loss (REINFORCE with baseline)
        policy_loss = -(log_probs * advantages).mean()

        # Value loss (MSE)
        value_loss = F.mse_loss(values, returns)

        # Combined loss
        loss = policy_loss + 0.5 * value_loss

        # Entropy bonus for exploration
        states_t = torch.stack(ep_data['states'])
        probs = policy(states_t)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        loss -= 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy.parameters()) + list(value_net.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

        # Record curriculum
        curriculum.record_episode_reward(ep_data['total_reward'], curriculum.current_stage)
        promoted = curriculum.check_promotion()
        if promoted:
            print(f"\n  🎓 PROMOTED TO STAGE {curriculum.current_stage}!\n")

        # Logging
        if ep % 20 == 0 or ep == 1 or promoted:
            recent_r = all_rewards[-20:]
            recent_f = all_fulfillments[-20:]
            recent_t = all_trusts[-20:]
            recent_pl = policy_losses[-20:]
            recent_vl = value_losses[-20:]
            elapsed = time.time() - start_time
            eps_per_sec = ep / elapsed

            print(f"  {ep:5d} {curriculum.current_stage:5d} "
                  f"{np.mean(recent_r):+8.3f} {np.mean(recent_f):7.1%}  "
                  f"{np.mean(recent_t):6.3f} "
                  f"{np.mean(recent_pl):8.4f} {np.mean(recent_vl):8.4f}")

            training_curve.append({
                'episode': ep,
                'stage': curriculum.current_stage,
                'avg_reward': round(float(np.mean(recent_r)), 4),
                'avg_fulfillment': round(float(np.mean(recent_f)), 4),
                'avg_trust': round(float(np.mean(recent_t)), 4),
                'policy_loss': round(float(np.mean(recent_pl)), 4),
                'value_loss': round(float(np.mean(recent_vl)), 4),
                'eps_per_sec': round(eps_per_sec, 1),
            })

    # ── Post-training evaluation ──────────────────────────────────────────
    print(f"\n📊 Running POST-TRAINING evaluation...")
    post_rewards = []
    post_fulfill = []
    for i in range(20):
        env.curriculum_stage = 1
        scenario = curriculum.generate_next_episode()
        with torch.no_grad():
            ep_data = run_rl_episode(env, pomdp, policy, value_net, scenario)
        post_rewards.append(ep_data['total_reward'])
        post_fulfill.append(ep_data['fulfillment'])

    print(f"  Post-training: reward={np.mean(post_rewards):+.3f}  "
          f"fulfill={np.mean(post_fulfill):.1%}")

    # ── Final Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    print(f"\n{'═'*65}")
    print(f"  TRAINING COMPLETE: {args.episodes} episodes in {elapsed:.1f}s")
    print(f"{'═'*65}")

    reward_improvement = np.mean(post_rewards) - np.mean(pre_rewards)
    fulfill_improvement = np.mean(post_fulfill) - np.mean(pre_fulfill)

    print(f"\n  ┌─── Before vs After ───────────────────────────┐")
    print(f"  │                    BEFORE      AFTER    DELTA  │")
    print(f"  │  Reward:        {np.mean(pre_rewards):+7.3f}    {np.mean(post_rewards):+7.3f}   {reward_improvement:+6.3f}  │")
    print(f"  │  Fulfillment:   {np.mean(pre_fulfill):7.1%}    {np.mean(post_fulfill):7.1%}  {fulfill_improvement:+6.1%}  │")
    print(f"  └────────────────────────────────────────────────┘")

    print(f"\n  Final Stage Reached: {curriculum.current_stage}")
    print(f"  Policy Parameters:   {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Value Net Parameters: {sum(p.numel() for p in value_net.parameters()):,}")
    print(f"{'═'*65}")

    # ── Save Results ──────────────────────────────────────────────────────
    results_dir = Path(args.log_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'algorithm': 'REINFORCE_with_baseline',
        'total_episodes': args.episodes,
        'elapsed_seconds': round(elapsed, 2),
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'final_stage': curriculum.current_stage,
        'before_after': {
            'pre_training_reward': round(float(np.mean(pre_rewards)), 4),
            'post_training_reward': round(float(np.mean(post_rewards)), 4),
            'reward_improvement': round(float(reward_improvement), 4),
            'pre_training_fulfillment': round(float(np.mean(pre_fulfill)), 4),
            'post_training_fulfillment': round(float(np.mean(post_fulfill)), 4),
            'fulfillment_improvement': round(float(fulfill_improvement), 4),
        },
        'training_curve': training_curve,
        'model_params': {
            'policy': sum(p.numel() for p in policy.parameters()),
            'value_net': sum(p.numel() for p in value_net.parameters()),
        },
    }

    with open(results_dir / 'rl_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_dir / 'rl_training_results.json'}")

    # Save trained model
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'value_net_state_dict': value_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, results_dir / 'vergil_rl_model.pt')
    print(f"  Model saved to: {results_dir / 'vergil_rl_model.pt'}")

    with open(results_dir / 'rl_training_curve.json', 'w') as f:
        json.dump(training_curve, f, indent=2)
    print(f"  Learning curve saved to: {results_dir / 'rl_training_curve.json'}")

    print()


if __name__ == '__main__':
    main()
