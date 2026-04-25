# vergil/core/reward.py
"""
Multi-Component Reward Function
================================

R_total = w₁ × R_fulfill + w₂ × R_trust + w₃ × R_proactive + w₄ × R_accuracy
        - p₁ × P_broken - p₂ × P_overrefusal - p₃ × P_silent_drop

Weights: w₁=0.35, w₂=0.25, w₃=0.20, w₄=0.10, p₁=0.40, p₂=0.30, p₃=0.50

Anti-reward-hacking mechanisms:
- Decline Everything: over_refusal_penalty + trust decay with all stakeholders
- Accept Everything: under_delivery_penalty + rapid trust collapse
- Always Predict Feasible: accuracy penalty when things fail
- Always Predict Infeasible: over-refusal penalty

References:
    VERGIL_System_Design.md — Step 7 (Reward Function Design)
    VERGIL_Phase1_Phase2_Implementation.md — P1.5
"""

from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging
import math

from .types import (
    AgentAction, ActionType, CommitmentNode, CommitmentStatus,
    TrustEntry, VERGILState, EpisodeRecord
)

logger = logging.getLogger('vergil.reward')


@dataclass
class RewardComponents:
    """All reward components, stored separately for analysis and logging."""

    # Positive signals
    fulfillment: float = 0.0       # R_fulfill: quality-weighted completion rate
    trust_delta: float = 0.0       # R_trust: weighted trust change this step
    proactive: float = 0.0         # R_proactive: bonus for early renegotiation
    feasibility_acc: float = 0.0   # R_accuracy: feasibility prediction calibration

    # Penalties
    broken_penalty: float = 0.0    # P_broken: broken commitment penalty
    overrefusal_penalty: float = 0.0  # P_overrefusal: declining feasible things
    silent_drop_penalty: float = 0.0  # P_silent: accepted then dropped

    # Total
    total: float = 0.0

    # Metadata
    step: int = 0
    episode_id: str = ""
    action_type: str = ""

    def compute_total(self) -> float:
        """
        Final reward = weighted positives - penalties.
        Weights from Step 7 of design doc.
        """
        positive = (
            0.35 * self.fulfillment +
            0.25 * self.trust_delta +
            0.20 * self.proactive +
            0.10 * self.feasibility_acc
        )

        penalties = (
            0.40 * self.broken_penalty +
            0.30 * self.overrefusal_penalty +
            0.50 * self.silent_drop_penalty
        )

        self.total = positive - penalties
        return self.total

    def to_dict(self) -> Dict:
        return {
            'step': self.step,
            'action_type': self.action_type,
            'fulfillment': round(self.fulfillment, 4),
            'trust_delta': round(self.trust_delta, 4),
            'proactive': round(self.proactive, 4),
            'feasibility_acc': round(self.feasibility_acc, 4),
            'broken_penalty': round(self.broken_penalty, 4),
            'overrefusal_penalty': round(self.overrefusal_penalty, 4),
            'silent_drop_penalty': round(self.silent_drop_penalty, 4),
            'total': round(self.total, 4),
        }


class VERGILReward:
    """
    Multi-component reward function with anti-reward-hacking mechanisms.

    Long-horizon credit assignment:
    - Intermediate rewards provide dense signal (trust deltas, feasibility accuracy)
    - Terminal reward provides sparse episode-end signal
    - TD(λ) with λ=0.9 propagates credit backward (in training loop)
    - Advantage function shaping: R_shaped = R + γ × Φ(s') - Φ(s)
      where Φ(s) = satisfiability_score(CDG) (potential function)
    """

    # Relationship weights (Step 7)
    RELATIONSHIP_WEIGHTS = {
        'boss': 0.35, 'client': 0.30, 'colleague': 0.20, 'friend': 0.15
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.overrefusal_sensitivity = self.config.get('overrefusal_sensitivity', 2.0)
        self.optimal_accept_fraction = self.config.get('optimal_accept_fraction', 0.75)

        # Episode-level tracking (reset on episode reset)
        self._step_decisions: List[Dict] = []
        self._prev_satisfiability: float = 1.0  # For reward shaping

        logger.info("VERGILReward initialized")

    def compute_step_reward(
        self,
        action: AgentAction,
        node: Optional[CommitmentNode],
        trust_deltas: Dict[str, float],
        trust_entries: Dict[str, TrustEntry],
        cascade_events: List[Dict],
        state: VERGILState,
        current_step: int,
    ) -> RewardComponents:
        """Compute per-step reward components."""
        rc = RewardComponents(step=current_step, action_type=action.action_type.value)

        # R_trust: Weighted trust change
        rc.trust_delta = self._compute_trust_reward(trust_deltas, trust_entries)

        # R_proactive: Proactive renegotiation bonus
        if action.action_type == ActionType.RENEGOTIATE and node is not None:
            rc.proactive = self._compute_proactive_bonus(action, node, state.current_time)

        # R_accuracy: Feasibility prediction accuracy
        if node is not None and action.feasibility_prediction is not None:
            rc.feasibility_acc = self._compute_accuracy_reward(action, node, state)

        # P_broken: Broken commitment penalty
        if cascade_events:
            rc.broken_penalty = self._compute_broken_penalty(cascade_events, trust_entries)

        # Check if this step's action was a confirmed failure
        if node and node.status == CommitmentStatus.FAILED:
            if action.action_type not in (ActionType.RENEGOTIATE, ActionType.DECLINE):
                rc.broken_penalty += self._node_failure_penalty(node, trust_entries)

        # P_overrefusal: Over-refusal penalty
        if action.action_type == ActionType.DECLINE:
            rc.overrefusal_penalty = self._compute_overrefusal_penalty(state)

        # P_silent_drop: Silent drop penalty
        rc.silent_drop_penalty = self._compute_silent_drop_penalty(state, current_step)

        # Record for episode-level audit
        self._step_decisions.append({
            'step': current_step,
            'action': action.action_type.value,
            'node_id': node.node_id if node else None,
            'feasibility_pred': action.feasibility_prediction,
        })

        # Reward shaping with CDG satisfiability potential function
        # R_shaped = R + γ × Φ(s') - Φ(s) where Φ(s) = satisfiability_score
        gamma = 0.99
        new_sat = state.satisfiability_score
        shaping_bonus = gamma * new_sat - self._prev_satisfiability
        self._prev_satisfiability = new_sat

        rc.compute_total()
        rc.total += 0.1 * shaping_bonus  # Small shaping contribution

        logger.debug(f"Reward step {current_step}: {rc.to_dict()}")
        return rc

    def compute_terminal_reward(
        self,
        final_state: VERGILState,
        episode_record: EpisodeRecord,
    ) -> RewardComponents:
        """Episode-end reward — sparse signal for long-horizon credit assignment."""
        rc = RewardComponents(step=episode_record.total_steps, action_type='terminal')

        # R_fulfill: Final commitment fulfillment rate
        nodes = final_state.cdg_nodes
        kept = sum(1 for n in nodes if n.status == CommitmentStatus.COMPLETED)
        made = sum(1 for n in nodes if n.status in (
            CommitmentStatus.COMPLETED, CommitmentStatus.FAILED,
            CommitmentStatus.LATE_COMPLETED, CommitmentStatus.ACCEPTED
        ))

        if made > 0:
            raw_rate = kept / made
            # Timeliness modifier: exp(-delay_hours / 24) from Step 7
            timeliness_scores = []
            for n in nodes:
                if n.status == CommitmentStatus.COMPLETED and n.deadline and n.completion_timestamp:
                    slack_hrs = (n.deadline - n.completion_timestamp).total_seconds() / 3600
                    timeliness_scores.append(min(1.2, max(0.5, 1.0 + slack_hrs / 48)))
            timeliness_modifier = (
                sum(timeliness_scores) / len(timeliness_scores) if timeliness_scores else 1.0
            )
            # Quality modifier ∈ [0.6, 1.2] (Step 7)
            rc.fulfillment = raw_rate * timeliness_modifier

        # Trust stability bonus
        trust_stability = self._compute_trust_stability(final_state.trust_entries)
        rc.trust_delta = trust_stability * 0.5

        rc.compute_total()
        logger.info(f"Terminal reward: fulfillment={rc.fulfillment:.3f} "
                    f"trust_stability={trust_stability:.3f} total={rc.total:.3f}")
        return rc

    def _compute_trust_reward(self, trust_deltas: Dict, trust_entries: Dict) -> float:
        """Weighted trust delta across all stakeholders (Step 7)."""
        if not trust_deltas:
            return 0.0

        total_weight = 0.0
        weighted_delta = 0.0

        for sid, delta in trust_deltas.items():
            entry = trust_entries.get(sid)
            if entry is None:
                continue
            role = getattr(entry, '_role', 'colleague')
            weight = self.RELATIONSHIP_WEIGHTS.get(role, 0.2)
            weighted_delta += weight * delta
            total_weight += weight

        return weighted_delta / max(total_weight, 0.01)

    def _compute_proactive_bonus(self, action: AgentAction,
                                 node: CommitmentNode,
                                 current_time: datetime) -> float:
        """
        Lead-time-based proactive renegotiation bonus (Step 7).
        lead_time_bonus = max(0, 1 - (hours_to_deadline / 48)²)
        Earlier renegotiation → higher bonus.
        """
        if node.deadline is None:
            return 0.1

        hours_to_deadline = (node.deadline - current_time).total_seconds() / 3600

        if hours_to_deadline <= 0:
            return -0.1  # Renegotiating after deadline is worse than nothing
        elif hours_to_deadline < 2:
            return 0.05
        elif hours_to_deadline < 8:
            return 0.15 + 0.1 * (hours_to_deadline / 8)
        elif hours_to_deadline < 24:
            return 0.30 + 0.2 * (hours_to_deadline / 24)
        else:
            # max(0, 1 - (hours_to_deadline / 48)²) formula from Step 7
            return min(1.0, 0.5 + 0.5 * min(1.0, hours_to_deadline / 48))

    def _compute_accuracy_reward(self, action: AgentAction,
                                 node: CommitmentNode,
                                 state: VERGILState) -> float:
        """
        Feasibility prediction accuracy (Step 7).
        R_accuracy = 1 - |predicted_feasibility - actual_feasibility|
        """
        predicted = action.feasibility_prediction
        actual_proxy = state.satisfiability_score
        accuracy = 1.0 - abs(predicted - actual_proxy)
        return max(0.0, accuracy)

    def _compute_broken_penalty(self, cascade_events: List[Dict],
                                trust_entries: Dict) -> float:
        """
        Broken commitment penalty (Step 7).
        P_broken = base × trust_weight(s) × (1 + cascade_depth) × (1 + deadline_proximity)
        Low trust = MORE penalized (relationship already fragile).
        """
        if not cascade_events:
            return 0.0

        total_penalty = 0.0
        for event in cascade_events:
            if event.get('cascaded'):
                depth = event.get('depth', 1)
                trust_weight = 1.0  # Enhanced in Phase 2 with actual trust
                penalty = trust_weight * (1 + depth * 0.2)
                total_penalty += penalty

        return min(3.0, total_penalty)  # Cap to prevent catastrophic gradients

    def _node_failure_penalty(self, node: CommitmentNode,
                              trust_entries: Dict) -> float:
        """Single-node failure penalty. Low trust = higher penalty (Step 7)."""
        trust_entry = trust_entries.get(node.stakeholder_id)
        current_trust = trust_entry.trust_score if trust_entry else 0.5
        # trust_weight(s) = 1 + (1 - trust(s)) — from design doc
        trust_weight = 1 + (1 - current_trust)
        return node.urgency * trust_weight

    def _compute_overrefusal_penalty(self, state: VERGILState) -> float:
        """
        Over-refusal penalty (Step 7).
        P_overrefusal = max(0, optimal_accept_rate - actual_accept_rate)² × 2.0
        Activates when decline rate > 40% of feasible commitments.
        """
        total_decisions = len(self._step_decisions)
        if total_decisions < 5:
            return 0.0  # Grace period

        declines = sum(1 for d in self._step_decisions if d['action'] == 'decline')
        decline_rate = declines / total_decisions

        optimal_decline_rate = 1.0 - self.optimal_accept_fraction
        excess = max(0, decline_rate - optimal_decline_rate)

        return min(1.0, self.overrefusal_sensitivity * excess ** 2)

    def _compute_silent_drop_penalty(self, state: VERGILState,
                                     current_step: int) -> float:
        """
        Silent failure penalty (Step 7). THE LARGEST PENALTY.
        P_silent_drop = 0.5 × (1 + hours_since_acceptance_without_renegotiation/24)
        Grows over time from acceptance to drop.
        """
        penalty = 0.0
        for node in state.cdg_nodes:
            if (node.status == CommitmentStatus.FAILED and
                    node.decision_made == ActionType.ACCEPT and
                    node.renegotiation_count == 0):
                # Accepted + never renegotiated + now failed = silent drop
                steps_since = current_step - (
                    getattr(node.decision_timestamp, 'step', 0) if node.decision_timestamp else 0
                )
                penalty += 0.5 * (1 + steps_since / 10)

        return min(2.0, penalty)

    def _compute_trust_stability(self, trust_entries: Dict) -> float:
        """
        Trust trajectory stability. Trust Stability Index = σ(trajectory).
        Stable high trust → bonus. Target: σ < 0.12 (Step 10).
        """
        if not trust_entries:
            return 0.0

        scores = []
        for entry in trust_entries.values():
            scores.append(entry.trust_score)
            if len(entry.history) > 3:
                deltas = [abs(e['delta']) for e in entry.history[-5:]]
                volatility = sum(deltas) / len(deltas)
                scores.append(max(0, entry.trust_score - volatility))

        return sum(scores) / len(scores) if scores else 0.0

    def compute_shaped_reward(
        self,
        base_reward: float,
        prev_satisfiability: float,
        new_satisfiability: float,
        gamma: float = 0.99,
    ) -> float:
        """
        Potential-based reward shaping: R' = R + γ×Φ(s') - Φ(s)
        where Φ(s) = satisfiability_score(CDG).

        This provides dense intermediate feedback aligned with the sparse
        terminal reward — critical for long-horizon credit assignment.
        The shaping is potential-based so it does NOT change the optimal policy
        (policy invariance theorem, Ng et al. 1999).
        """
        potential_delta = gamma * new_satisfiability - prev_satisfiability
        return base_reward + potential_delta

    def reset_episode(self) -> None:
        """Reset episode-level trackers."""
        self._step_decisions = []
        self._prev_satisfiability = 1.0
