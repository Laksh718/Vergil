# vergil/core/pomdp.py
"""
POMDP Belief State Management
===============================

Wraps VERGILEnv with Bayesian belief state tracking.
On every step: observation → belief update → (observation, belief) to agent.

Hidden variables tracked via belief distributions:
- Stakeholder urgency: Beta(α, β) → updated by urgency signals
- Stakeholder flexibility: N(μ, σ²) → updated by negotiation outcomes
- Duration uncertainty: N(μ, σ²) → updated by completion observations
- Irrationality probability: point estimate → updated by behavior signals

References:
    VERGIL_System_Design.md — Step 4 (POMDP formalization)
    VERGIL_Phase1_Phase2_Implementation.md — P2.1
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from .types import (
    AgentAction, ActionType, VERGILState
)

logger = logging.getLogger('vergil.pomdp')


@dataclass
class StakeholderBelief:
    """
    Agent's belief state about a single stakeholder's hidden variables.
    Updated via Bayesian inference from observable signals.
    """
    stakeholder_id: str

    # Urgency belief: Beta(α, β) — conjugate prior for binary observations
    urgency_mean: float = 0.5
    urgency_std: float = 0.3
    _urgency_alpha: float = 2.0
    _urgency_beta: float = 2.0

    # Flexibility belief: N(μ, σ²) — hours of hidden deadline slack
    flexibility_mean: float = 8.0
    flexibility_std: float = 12.0

    # Irrationality probability: point estimate with momentum
    irrational_probability: float = 0.1

    # Observation counts
    urgency_observations: int = 0
    flexibility_observations: int = 0

    def update_urgency(self, observed_signal: float,
                       noise: float = 0.15) -> None:
        """
        Bayesian update of urgency belief.
        observed_signal ∈ [0, 1] from stakeholder response.
        """
        # Treat as Beta-Bernoulli: high signal → increment α
        if observed_signal > 0.6:
            self._urgency_alpha += 1
        elif observed_signal < 0.4:
            self._urgency_beta += 1
        else:
            self._urgency_alpha += 0.3
            self._urgency_beta += 0.3

        self.urgency_mean = self._urgency_alpha / (self._urgency_alpha + self._urgency_beta)
        self.urgency_std = np.sqrt(
            (self._urgency_alpha * self._urgency_beta) /
            ((self._urgency_alpha + self._urgency_beta) ** 2 *
             (self._urgency_alpha + self._urgency_beta + 1))
        )
        self.urgency_observations += 1

    def update_flexibility(self, event: str) -> None:
        """
        Update flexibility belief based on negotiation outcomes.
        'accepted_extension' → stakeholder has more slack than expected.
        'rejected_extension' → stakeholder is time-constrained.
        """
        if event == 'accepted_extension':
            self.flexibility_mean = self.flexibility_mean * 0.7 + 12.0 * 0.3
            self.flexibility_std = max(2.0, self.flexibility_std * 0.85)
        elif event == 'rejected_extension':
            self.flexibility_mean = max(0, self.flexibility_mean * 0.7 + 2.0 * 0.3)
            self.flexibility_std = max(1.5, self.flexibility_std * 0.85)
        self.flexibility_observations += 1

    def update_irrationality(self, was_irrational: bool) -> None:
        """Update irrationality estimate via EMA."""
        if was_irrational:
            self.irrational_probability = min(0.9,
                self.irrational_probability + 0.15)
        else:
            self.irrational_probability = max(0.02,
                self.irrational_probability * 0.9)

    def to_vector(self) -> np.ndarray:
        """7-dim belief vector for model input."""
        return np.array([
            self.urgency_mean, self.urgency_std,
            self.flexibility_mean / 48.0,  # Normalize to [0,1]
            self.flexibility_std / 24.0,
            self.irrational_probability,
            self.urgency_observations / 20.0,
            self.flexibility_observations / 10.0,
        ], dtype=np.float32)

    def entropy(self) -> float:
        """Belief uncertainty — high entropy = very uncertain."""
        urgency_entropy = -self.urgency_mean * np.log(max(1e-8, self.urgency_mean)) - \
                          (1 - self.urgency_mean) * np.log(max(1e-8, 1 - self.urgency_mean))
        flex_entropy = 0.5 * np.log(2 * np.pi * np.e * max(1e-8, self.flexibility_std ** 2))
        return float(urgency_entropy + flex_entropy)


@dataclass
class BeliefState:
    """Complete belief state across all stakeholders and commitments."""
    stakeholder_beliefs: Dict[str, StakeholderBelief] = field(default_factory=dict)
    duration_beliefs: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Global uncertainty metrics
    overall_uncertainty: float = 0.7
    epistemic_risk: float = 0.4
    update_count: int = 0

    def entropy(self) -> float:
        """Total belief entropy — drives exploration bonus in Phase 3."""
        if not self.stakeholder_beliefs:
            return 1.0
        return float(np.mean([sb.entropy()
                              for sb in self.stakeholder_beliefs.values()]))

    def to_vector(self) -> np.ndarray:
        """Concatenate all stakeholder belief vectors + global state."""
        vectors = []
        for sb in sorted(self.stakeholder_beliefs.values(),
                         key=lambda x: x.stakeholder_id):
            vectors.append(sb.to_vector())

        if vectors:
            belief_vec = np.concatenate(vectors)
        else:
            belief_vec = np.zeros(7, dtype=np.float32)

        global_vec = np.array([
            self.overall_uncertainty,
            self.epistemic_risk,
            self.update_count / 50.0,
        ], dtype=np.float32)

        return np.concatenate([belief_vec, global_vec])


class POMDPWrapper:
    """
    Wraps VERGILEnv with POMDP belief state management.

    On every step:
    1. Receive observation (visible state from env)
    2. Update belief state based on observation + prior
    3. Return (observation, belief_state) to agent

    The agent's effective observation = observation + belief_state_vector.
    """

    def __init__(self, env, config: Optional[Dict] = None):
        self.env = env
        self.config = config or {}
        self.belief_state = BeliefState()
        self._observation_model_noise = self.config.get('observation_noise', 0.15)

    def reset(self, **kwargs) -> Tuple[VERGILState, BeliefState, Dict]:
        state, info = self.env.reset(**kwargs)
        self.belief_state = self._initialize_belief(state)
        return state, self.belief_state, info

    def step(self, action: AgentAction) -> Tuple[VERGILState, BeliefState, float, bool, bool, Dict]:
        state, reward, term, trunc, info = self.env.step(action)
        self._update_belief(state, action, info)
        return state, self.belief_state, reward, term, trunc, info

    def _initialize_belief(self, state: VERGILState) -> BeliefState:
        """Initialize uniform (high-uncertainty) beliefs at episode start."""
        belief = BeliefState(overall_uncertainty=0.7, epistemic_risk=0.4)

        for sid in state.trust_entries.keys():
            belief.stakeholder_beliefs[sid] = StakeholderBelief(
                stakeholder_id=sid,
                urgency_mean=0.5, urgency_std=0.3,
                flexibility_mean=8.0, flexibility_std=12.0,
                irrational_probability=0.1,
            )

        for node in state.cdg_nodes:
            belief.duration_beliefs[node.node_id] = (
                node.estimated_duration_hours,
                node.duration_std_hours,
            )

        return belief

    def _update_belief(self, state: VERGILState, action: AgentAction,
                       info: Dict) -> None:
        """Update belief based on new observations."""
        belief = self.belief_state

        # Update from stakeholder responses
        for sid, response_msg in info.get('stakeholder_responses', {}).items():
            if sid in belief.stakeholder_beliefs:
                sb = belief.stakeholder_beliefs[sid]

                urgency_signal = self._parse_urgency_signal(response_msg)
                sb.update_urgency(urgency_signal, self._observation_model_noise)

                if action.action_type in (ActionType.COUNTER_PROPOSE,
                                          ActionType.RENEGOTIATE):
                    if 'works' in response_msg.lower() or 'go with' in response_msg.lower():
                        sb.update_flexibility('accepted_extension')
                    elif 'need' in response_msg.lower() or 'original' in response_msg.lower():
                        sb.update_flexibility('rejected_extension')

        # Update from task completions
        for node in state.cdg_nodes:
            if (node.status.value == 'completed' and
                    node.actual_duration_hours is not None and
                    node.node_id in belief.duration_beliefs):
                old_mean, old_std = belief.duration_beliefs[node.node_id]
                obs = node.actual_duration_hours
                new_var = 1 / (1 / (old_std ** 2 + 1e-8) + 1 / (0.5 ** 2))
                new_mean = new_var * (old_mean / (old_std ** 2 + 1e-8) + obs / (0.5 ** 2))
                belief.duration_beliefs[node.node_id] = (new_mean, np.sqrt(new_var))

        # Update from cascade events
        for event in info.get('cascade_events', []):
            nid = event.get('node_id')
            node = self.env.cdg.get_node(nid) if self.env.cdg else None
            if node and node.stakeholder_id in belief.stakeholder_beliefs:
                sb = belief.stakeholder_beliefs[node.stakeholder_id]
                sb.update_urgency(min(1.0, sb.urgency_mean + 0.1))

        belief.overall_uncertainty = min(0.95, belief.entropy() * 1.5)
        belief.update_count += 1

        logger.debug(f"Belief updated: uncertainty={belief.overall_uncertainty:.3f} "
                     f"entropy={belief.entropy():.3f}")

    def _parse_urgency_signal(self, message: str) -> float:
        """Extract urgency signal from stakeholder response message."""
        high_urgency = ['need', 'must', 'critical', 'immediately', 'asap',
                        'very important', 'counting on', 'essential']
        low_urgency = ['whenever', 'no rush', 'take your time', 'eventually',
                       'if you can', 'no worries']

        msg_lower = message.lower()
        high_count = sum(1 for s in high_urgency if s in msg_lower)
        low_count = sum(1 for s in low_urgency if s in msg_lower)

        if high_count > low_count:
            return min(1.0, 0.5 + high_count * 0.1)
        elif low_count > high_count:
            return max(0.0, 0.5 - low_count * 0.1)
        return 0.5
