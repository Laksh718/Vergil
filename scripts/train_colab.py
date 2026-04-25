# scripts/train_colab.py
"""
VERGIL Training Script
========================

Runs the full training pipeline with:
- Proper scenario feeding from curriculum engine
- Smart heuristic policy that actually fulfills commitments
- Task completion simulation (accepted → completed over time)
- Full Phase 2 integration (POMDP, multi-dim trust, execution model)
- Multi-stage training with curriculum progression
- Detailed metrics and learning curves

Usage:
    python3 scripts/train_colab.py --episodes 500 --stage 1
    python3 scripts/train_colab.py --smoke-test
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from vergil.core.env import VERGILEnv
from vergil.core.types import AgentAction, ActionType, CommitmentStatus, CommitmentNode
from vergil.core.pomdp import POMDPWrapper
from vergil.core.execution_model import ProbabilisticExecutionEngine
from vergil.curriculum.failure_db import FailureTopologyDatabase
from vergil.curriculum.scenario_generator import ScenarioGenerator
from vergil.curriculum.curriculum_engine import CurriculumEngine
from vergil.anti_hack.reward_guard import RewardGuard
from vergil.training.evaluation import EvaluationSuite, EvaluationMetrics


def parse_args():
    parser = argparse.ArgumentParser(description='VERGIL Training')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke-test', action='store_true',
                        help='Quick 20-episode test')
    parser.add_argument('--log-dir', type=str, default='/tmp/vergil_training')
    parser.add_argument('--db-path', type=str, default='/tmp/vergil_ftd.sqlite')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Smart Heuristic Policy — evolves with an epsilon-greedy exploration rate
# ═══════════════════════════════════════════════════════════════════════════

class SmartPolicy:
    """
    Heuristic policy that actually reasons about commitments.

    Decision logic:
    1. If there are PENDING nodes → decide on them (accept/decline/counter)
    2. If there are AT_RISK accepted nodes → renegotiate proactively
    3. Otherwise → do_nothing (let time pass, tasks complete)

    The policy improves over episodes via epsilon-greedy exploration
    and a simple Q-table that maps (scenario_features → action_bias).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.epsilon = 0.3  # Exploration rate
        self.episode_count = 0

        # Simple Q-table: maps (capacity_state, urgency_state) → action preferences
        self.q_table: Dict[tuple, Dict[str, float]] = {}
        self.learning_rate = 0.1
        self.last_state_key = None
        self.last_action = None

    def select_action(self, state, env) -> AgentAction:
        """Select action based on current state."""
        pending = [n for n in state.cdg_nodes
                   if n.status == CommitmentStatus.PENDING]
        accepted = [n for n in state.cdg_nodes
                    if n.status == CommitmentStatus.ACCEPTED]
        at_risk = [n for n in state.cdg_nodes
                   if n.status == CommitmentStatus.AT_RISK]

        # Priority 1: Handle pending commitments
        if pending:
            node = pending[0]
            return self._decide_on_pending(node, state)

        # Priority 2: Proactively renegotiate at-risk
        if at_risk:
            node = at_risk[0]
            return self._renegotiate_at_risk(node, state)

        # Priority 3: Check if any accepted tasks are getting tight
        for node in accepted:
            hours_left = node.deadline_proximity_hours(state.current_time)
            if hours_left < node.estimated_duration_hours * 1.2:
                return self._renegotiate_at_risk(node, state)

        # Priority 4: Nothing to do
        return AgentAction(action_type=ActionType.DO_NOTHING)

    def _decide_on_pending(self, node: CommitmentNode, state) -> AgentAction:
        """Decide whether to accept, decline, or counter-propose a pending node."""
        hours_to_deadline = node.deadline_proximity_hours(state.current_time)
        available = state.available_hours_next_48h
        trust = state.trust_entries.get(node.stakeholder_id)
        trust_score = trust.trust_score if trust else 0.5

        # Capacity check: can we fit this task?
        total_committed = sum(
            n.estimated_duration_hours for n in state.cdg_nodes
            if n.status == CommitmentStatus.ACCEPTED
        )
        remaining_capacity = max(0, available - total_committed)
        can_fit = node.estimated_duration_hours <= remaining_capacity * 0.8

        # Feasibility score
        time_ratio = hours_to_deadline / max(node.estimated_duration_hours, 0.1)
        feasibility = min(1.0, time_ratio * 0.5) * (0.5 + trust_score * 0.5)

        # State key for Q-learning
        state_key = (
            'high_cap' if can_fit else 'low_cap',
            'urgent' if hours_to_deadline < 12 else 'normal',
            'high_trust' if trust_score > 0.5 else 'low_trust',
        )
        self.last_state_key = state_key

        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            options = [ActionType.ACCEPT, ActionType.DECLINE,
                       ActionType.COUNTER_PROPOSE]
            action_type = options[int(self.rng.integers(0, len(options)))]
        else:
            # Use Q-table if available
            q_values = self.q_table.get(state_key, {})
            if q_values:
                best_action_name = max(q_values, key=q_values.get)
                action_type = ActionType(best_action_name)
            else:
                # Default smart heuristic
                if can_fit and time_ratio > 1.5:
                    action_type = ActionType.ACCEPT
                elif can_fit and time_ratio > 0.8:
                    action_type = ActionType.COUNTER_PROPOSE
                elif not can_fit and trust_score > 0.4:
                    action_type = ActionType.COUNTER_PROPOSE
                else:
                    action_type = ActionType.DECLINE

        self.last_action = action_type.value

        if action_type == ActionType.COUNTER_PROPOSE:
            new_deadline = state.current_time + timedelta(
                hours=node.estimated_duration_hours * 2.5)
            return AgentAction(
                action_type=action_type,
                target_node_id=node.node_id,
                feasibility_prediction=feasibility,
                proposed_deadline=new_deadline,
            )

        return AgentAction(
            action_type=action_type,
            target_node_id=node.node_id,
            feasibility_prediction=feasibility,
        )

    def _renegotiate_at_risk(self, node, state) -> AgentAction:
        new_deadline = state.current_time + timedelta(hours=24)
        return AgentAction(
            action_type=ActionType.RENEGOTIATE,
            target_node_id=node.node_id,
            feasibility_prediction=0.4,
            proposed_deadline=new_deadline,
        )

    def update(self, reward: float):
        """Update Q-table with episode reward."""
        if self.last_state_key and self.last_action:
            if self.last_state_key not in self.q_table:
                self.q_table[self.last_state_key] = {}
            q = self.q_table[self.last_state_key]
            old_val = q.get(self.last_action, 0.0)
            q[self.last_action] = old_val + self.learning_rate * (reward - old_val)

        self.episode_count += 1
        # Decay exploration
        self.epsilon = max(0.05, self.epsilon * 0.995)


# ═══════════════════════════════════════════════════════════════════════════
#  Task Completion Simulator
# ═══════════════════════════════════════════════════════════════════════════

def simulate_task_progress(env, execution_engine):
    """
    Simulate task progress: accepted tasks make progress each step.
    Without this, accepted tasks NEVER complete (the core bug).
    """
    if env.cdg is None or env._state is None:
        return

    current_time = env._state.current_time
    step_hours = env.config.get('step_hours', 2)

    for node_id, node in env.cdg._nodes.items():
        if node.status != CommitmentStatus.ACCEPTED:
            continue

        # Track work done (stored in hidden state)
        work_key = f"work_done_{node_id}"
        work_done = env._hidden.get(work_key, 0.0)
        true_duration = env._hidden.get('true_durations', {}).get(
            node_id, node.estimated_duration_hours)

        # Add progress for this step
        cognitive_load = env._state.cognitive_load
        efficiency = max(0.3, 1.0 - cognitive_load * 0.4)
        work_this_step = step_hours * efficiency * 0.85  # 85% productivity
        work_done += work_this_step
        env._hidden[work_key] = work_done

        # Check completion
        if work_done >= true_duration:
            env.cdg.update_node_status(node_id, CommitmentStatus.COMPLETED, current_time)
            node.actual_duration_hours = work_done
            node.completed_at = current_time


# ═══════════════════════════════════════════════════════════════════════════
#  Episode Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_episode(env, pomdp, policy, reward_guard, eval_suite,
                execution_engine, scenario, verbose=False):
    """Run a single episode with proper task completion simulation."""
    state, belief, info = pomdp.reset(scenario=scenario)
    reward_guard.reset()

    total_reward = 0.0
    step = 0
    feasibility_preds = []
    feasibility_actuals = []
    trust_trajectories = {sid: [te.trust_score]
                          for sid, te in state.trust_entries.items()}

    while True:
        # Simulate task progress BEFORE policy decides
        simulate_task_progress(env, execution_engine)

        action = policy.select_action(state, env)

        state, belief, reward, terminated, truncated, info = pomdp.step(action)

        # Simulate task progress AFTER action too
        simulate_task_progress(env, execution_engine)

        # Track for evaluation
        reward_guard.record_action(
            action.action_type.value,
            feasibility_prediction=action.feasibility_prediction,
        )

        if action.feasibility_prediction is not None:
            feasibility_preds.append(action.feasibility_prediction)
            feasibility_actuals.append(
                1.0 if state.satisfiability_score > 0.5 else 0.0)

        for sid, te in state.trust_entries.items():
            if sid in trust_trajectories:
                trust_trajectories[sid].append(te.trust_score)

        penalty_factor = reward_guard.compute_total_penalty()
        adjusted_reward = reward * penalty_factor
        total_reward += adjusted_reward
        step += 1

        if verbose and step <= 5:
            print(f"    step {step}: {action.action_type.value} "
                  f"→ reward={reward:+.3f} sat={state.satisfiability_score:.2f}")

        if terminated or truncated:
            break

    # Update policy with episode reward
    policy.update(total_reward)

    final_trust = {sid: te.trust_score
                   for sid, te in state.trust_entries.items()}

    # Compute evaluation metrics
    n_completed = sum(1 for n in state.cdg_nodes
                      if n.status == CommitmentStatus.COMPLETED)
    n_total = sum(1 for n in state.cdg_nodes
                  if n.status in (CommitmentStatus.COMPLETED,
                                  CommitmentStatus.FAILED,
                                  CommitmentStatus.ACCEPTED,
                                  CommitmentStatus.LATE_COMPLETED))

    if env._episode_record:
        env._episode_record.total_reward = total_reward
        env._episode_record.commitment_fulfillment_rate = (
            n_completed / max(1, n_total))

        metrics = eval_suite.evaluate_episode(
            episode_record=env._episode_record,
            final_trust_scores=final_trust,
            trust_trajectories=trust_trajectories,
            feasibility_predictions=feasibility_preds,
            feasibility_actuals=feasibility_actuals,
        )
    else:
        metrics = EvaluationMetrics(total_reward=total_reward)

    metrics.commitment_fulfillment_rate = n_completed / max(1, n_total)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.smoke_test:
        args.episodes = 50
        print("🔥 Smoke test: 50 episodes across stages 1-2")

    print()
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║        VERGIL Training Pipeline v0.1.0          ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Episodes: {args.episodes:<6}  │  Start Stage: {args.stage:<5}     ║")
    print(f"║  Seed: {args.seed:<9}  │  Verbose: {str(args.verbose):<10}    ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # Initialize components
    env = VERGILEnv(seed=args.seed, config={
        'max_steps_per_episode': 30,
        'step_hours': 2,
        'log_dir': args.log_dir,
    })
    pomdp = POMDPWrapper(env)
    execution_engine = ProbabilisticExecutionEngine(seed=args.seed)

    failure_db = FailureTopologyDatabase(db_path=args.db_path)
    scenario_gen = ScenarioGenerator(seed=args.seed)
    curriculum = CurriculumEngine(
        failure_db=failure_db,
        scenario_generator=scenario_gen,
        initial_stage=args.stage,
    )

    policy = SmartPolicy(seed=args.seed)
    reward_guard = RewardGuard()
    eval_suite = EvaluationSuite()

    all_metrics: List[EvaluationMetrics] = []
    stage_metrics: Dict[int, List[EvaluationMetrics]] = {1: [], 2: [], 3: [], 4: []}
    training_curve: List[Dict] = []
    start_time = time.time()

    print(f"{'─'*60}")
    print(f"  {'Ep':>5} {'Stage':>5} {'Reward':>8} {'Fulfill':>8} "
          f"{'Trust':>7} {'ε':>5} {'Speed':>8}")
    print(f"{'─'*60}")

    for ep in range(1, args.episodes + 1):
        env.curriculum_stage = curriculum.current_stage

        # Generate scenario from curriculum
        scenario = curriculum.generate_next_episode()

        # Run episode
        metrics = run_episode(
            env, pomdp, policy, reward_guard, eval_suite,
            execution_engine, scenario, verbose=(args.verbose and ep <= 3))

        all_metrics.append(metrics)
        stage_metrics[curriculum.current_stage].append(metrics)

        # Record to curriculum
        curriculum.record_episode_reward(metrics.total_reward, curriculum.current_stage)

        # Check promotion
        promoted = curriculum.check_promotion()
        if promoted:
            print(f"\n  🎓 PROMOTED TO STAGE {curriculum.current_stage}!\n")

        # Progress logging
        if ep % 10 == 0 or ep == 1 or promoted:
            recent = all_metrics[-10:]
            avg_reward = sum(m.total_reward for m in recent) / len(recent)
            avg_fulfill = sum(m.commitment_fulfillment_rate for m in recent) / len(recent)
            avg_trust = sum(m.trust_maintenance_index for m in recent) / len(recent)
            elapsed = time.time() - start_time
            eps_per_sec = ep / elapsed

            print(f"  {ep:5d} {curriculum.current_stage:5d} "
                  f"{avg_reward:+8.3f} {avg_fulfill:7.1%}  "
                  f"{avg_trust:6.3f} {policy.epsilon:5.3f} "
                  f"{eps_per_sec:6.1f}/s")

            training_curve.append({
                'episode': ep,
                'stage': curriculum.current_stage,
                'avg_reward': round(float(avg_reward), 4),
                'avg_fulfillment': round(float(avg_fulfill), 4),
                'avg_trust': round(float(avg_trust), 4),
                'epsilon': round(float(policy.epsilon), 4),
            })

    # ── Final Summary ────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    agg = eval_suite.aggregate_metrics(all_metrics)

    print(f"\n{'═'*60}")
    print(f"  TRAINING COMPLETE: {args.episodes} episodes in {elapsed:.1f}s")
    print(f"{'═'*60}")
    print(f"  Commitment Fulfillment Rate:  {agg.get('mean_fulfillment', 0):>7.1%}")
    print(f"  Trust Maintenance Index:      {agg.get('mean_trust', 0):>7.4f}")
    print(f"  Proactive Renegotiation Rate: {agg.get('mean_proactive', 0):>7.1%}")
    print(f"  Cascade Prevention Rate:      {agg.get('mean_cascade_prevention', 0):>7.1%}")
    print(f"  Feasibility Accuracy (Brier): {agg.get('mean_brier_score', 0):>7.4f}")
    print(f"  Trust Stability (σ):          {agg.get('mean_trust_stability', 0):>7.4f}")
    print(f"  Mean Reward:                  {agg.get('mean_reward', 0):>+7.4f}")
    print(f"{'═'*60}")

    # Per-stage breakdown
    print(f"\n  Per-Stage Breakdown:")
    for stage in range(1, 5):
        if stage_metrics[stage]:
            sm = stage_metrics[stage]
            s_fulfill = sum(m.commitment_fulfillment_rate for m in sm) / len(sm)
            s_trust = sum(m.trust_maintenance_index for m in sm) / len(sm)
            s_reward = sum(m.total_reward for m in sm) / len(sm)
            print(f"    Stage {stage}: {len(sm):3d} eps │ "
                  f"fulfill={s_fulfill:.1%} │ trust={s_trust:.3f} │ "
                  f"reward={s_reward:+.3f}")

    # ── Save Results ─────────────────────────────────────────────────────
    results_dir = Path(args.log_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'summary': agg,
        'training_curve': training_curve,
        'total_episodes': args.episodes,
        'elapsed_seconds': round(elapsed, 2),
        'final_stage': curriculum.current_stage,
        'per_stage': {
            stage: {
                'episodes': len(stage_metrics[stage]),
                'avg_fulfillment': round(
                    sum(m.commitment_fulfillment_rate for m in stage_metrics[stage]) /
                    max(1, len(stage_metrics[stage])), 4),
                'avg_reward': round(
                    sum(m.total_reward for m in stage_metrics[stage]) /
                    max(1, len(stage_metrics[stage])), 4),
            }
            for stage in range(1, 5) if stage_metrics[stage]
        },
        'q_table_size': len(policy.q_table),
        'final_epsilon': round(float(policy.epsilon), 4),
    }

    with open(results_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_dir / 'training_results.json'}")

    with open(results_dir / 'training_curve.json', 'w') as f:
        json.dump(training_curve, f, indent=2)
    print(f"  Learning curve saved to: {results_dir / 'training_curve.json'}")

    print()


if __name__ == '__main__':
    main()
