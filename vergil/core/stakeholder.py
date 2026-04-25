# vergil/core/stakeholder.py
"""
Stakeholder Simulator
======================

Simulates stakeholder responses to agent actions.
Phase 1: Deterministic rules based on action type, trust, and context.
Phase 2: Extends to probabilistic and adversarial behaviors.

Stakeholder Taxonomy (Step 2):
    Tier 1 (High-Stakes): Boss, Client
    Tier 2 (Medium-Stakes): Colleague, Friend
    Tier 3 (Structural): Calendar, Email, Task Systems

Response Functions R_s(action, trust, context):
    trust_delta(Boss, broken_commitment) = -0.15 * (1 + deadline_proximity_factor)
    trust_delta(Boss, proactive_renegotiation) = +0.08 * (lead_time_factor)
    trust_delta(Friend, cancellation_day_of) = -0.35
    trust_delta(Friend, cancellation_3_days_prior) = -0.08
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import logging

from .types import (
    AgentAction, ActionType, StakeholderProfile, StakeholderRole,
    TrustEntry, CommitmentNode
)

logger = logging.getLogger('vergil.stakeholder')


@dataclass
class StakeholderResponse:
    """The stakeholder's reaction to the agent's action."""
    stakeholder_id: str
    accepted: bool                       # Did they accept the agent's action?
    counter_offer: Optional[Dict] = None # Counter-offer params if rejected
    trust_delta: float = 0.0             # Immediate trust change

    # Observable signals
    message: str = ""                    # Their response message
    expressed_urgency: float = 0.5       # Urgency signaled (may ≠ true urgency)
    expressed_flexibility: float = 0.5   # Flexibility signaled (may ≠ true)

    # Hidden (oracle only)
    _true_satisfaction: float = 0.5      # How satisfied they actually are
    _will_escalate: bool = False         # Will they escalate next step?


class StakeholderSimulator:
    """
    Simulate stakeholder responses to agent actions.

    Response logic:
    - ACCEPT of a request → stakeholder satisfied (+trust)
    - DECLINE → stakeholder reacts based on trust and role
    - COUNTER_PROPOSE → stakeholder evaluates offer quality
    - RENEGOTIATE → stakeholder reacts based on lead time and frequency
    - CLARIFY → generally positive (agent is being diligent)
    """

    def __init__(self, profiles: Dict[str, StakeholderProfile], seed: int = 42):
        self.profiles = profiles
        self.rng = random.Random(seed)
        self._renegotiation_counts: Dict[str, int] = {}

    def simulate_response(self, action: AgentAction, node: CommitmentNode,
                          trust: TrustEntry, current_time: datetime
                          ) -> StakeholderResponse:
        """Generate stakeholder response to agent's action."""
        profile = self.profiles.get(node.stakeholder_id)
        if profile is None:
            logger.warning(f"No profile for stakeholder {node.stakeholder_id}")
            return StakeholderResponse(stakeholder_id=node.stakeholder_id, accepted=True)

        if action.action_type == ActionType.ACCEPT:
            return self._respond_to_accept(action, node, trust, profile)
        elif action.action_type == ActionType.DECLINE:
            return self._respond_to_decline(action, node, trust, profile, current_time)
        elif action.action_type == ActionType.COUNTER_PROPOSE:
            return self._respond_to_counter(action, node, trust, profile, current_time)
        elif action.action_type == ActionType.RENEGOTIATE:
            return self._respond_to_renegotiate(action, node, trust, profile, current_time)
        elif action.action_type == ActionType.CLARIFY:
            return self._respond_to_clarify(action, node, trust, profile)
        elif action.action_type == ActionType.DEFER:
            return self._respond_to_defer(action, node, trust, profile)
        elif action.action_type == ActionType.DELEGATE:
            return self._respond_to_delegate(action, node, trust, profile)
        else:
            # DO_NOTHING — silence may be interpreted as acceptance by Boss
            return self._respond_to_silence(action, node, trust, profile)

    def _respond_to_accept(self, action, node, trust, profile) -> StakeholderResponse:
        trust_delta = profile.trust_repair_rate * 0.3
        msg = self._acceptance_message(profile)
        return StakeholderResponse(
            stakeholder_id=node.stakeholder_id,
            accepted=True, trust_delta=trust_delta,
            message=msg,
            expressed_urgency=profile._true_urgency,
            _true_satisfaction=0.7,
        )

    def _respond_to_decline(self, action, node, trust, profile, current_time) -> StakeholderResponse:
        """
        Decline response is role-dependent (Step 2).
        Boss: pushback, trust penalty scales with urgency.
        Client: risk of escalation.
        Friend: depends on proximity to event.
        """
        urgency_factor = node.urgency

        # Friend: cancellation near event is 5× worse (Step 2)
        if profile.role == StakeholderRole.FRIEND and node.deadline:
            hours_to_event = (node.deadline - current_time).total_seconds() / 3600
            if hours_to_event < 24:
                # Day-of cancellation: trust_delta = -0.35
                trust_delta = -0.35
            elif hours_to_event < 72:
                # 3 days prior: trust_delta = -0.08
                trust_delta = -0.08
            else:
                trust_delta = -profile.trust_decay_rate * 0.4
            return StakeholderResponse(
                stakeholder_id=node.stakeholder_id,
                accepted=True, trust_delta=trust_delta,
                message="Oh, okay. I understand.",
                _true_satisfaction=0.2,
            )

        role_responses = {
            StakeholderRole.BOSS: {
                # trust_delta(Boss, broken) = -0.15 * (1 + deadline_proximity_factor)
                'trust_delta': -profile.trust_decay_rate * (1 + urgency_factor),
                'message': "I need this done. Can we talk about what's blocking you?",
                'accepted': False,
            },
            StakeholderRole.CLIENT: {
                'trust_delta': -profile.trust_decay_rate * 1.2,
                'message': "This is concerning. We had an expectation this would be handled.",
                'accepted': False,
                '_will_escalate': trust.trust_score < profile.escalation_threshold,
            },
            StakeholderRole.COLLEAGUE: {
                'trust_delta': -profile.trust_decay_rate * 0.6,
                'message': "No worries, I'll figure something out.",
                'accepted': True,
            },
        }

        resp_data = role_responses.get(profile.role, role_responses[StakeholderRole.COLLEAGUE])
        return StakeholderResponse(
            stakeholder_id=node.stakeholder_id,
            accepted=resp_data.get('accepted', True),
            trust_delta=resp_data['trust_delta'],
            message=resp_data['message'],
            _true_satisfaction=0.2,
            _will_escalate=resp_data.get('_will_escalate', False),
        )

    def _respond_to_counter(self, action, node, trust, profile, current_time) -> StakeholderResponse:
        """
        Counter-proposal response. Quality of counter matters.
        A counter that gives stakeholder >80% → accepted.
        A counter that cuts >50% → rejected.
        """
        quality = 0.5

        if action.proposed_deadline and node.deadline:
            hours_delta = (action.proposed_deadline - node.deadline).total_seconds() / 3600
            if hours_delta > 0 and hours_delta <= 24:
                quality = 0.7
            elif hours_delta > 24:
                quality = 0.4
            else:
                quality = 0.9  # Earlier than requested

        # Boss is stricter
        if profile.role == StakeholderRole.BOSS:
            quality *= 0.85
        # Client is moderate
        elif profile.role == StakeholderRole.CLIENT:
            quality *= 0.90

        accepted = quality > 0.55
        trust_delta = 0.03 if accepted else -0.05

        if accepted:
            msg = "That works. Let's go with that."
        else:
            msg = "I really need the original timeline. Can you make it work?"

        return StakeholderResponse(
            stakeholder_id=node.stakeholder_id,
            accepted=accepted, trust_delta=trust_delta,
            message=msg, _true_satisfaction=quality,
        )

    def _respond_to_renegotiate(self, action, node, trust, profile, current_time) -> StakeholderResponse:
        """
        Renegotiation response. KEY: Lead time matters enormously.
        Early (>48hrs) → trust preserved.
        Late (<4hrs) → trust damaged.

        trust_delta(Boss, proactive_renegotiation) = +0.08 * lead_time_factor
        """
        hours_to_deadline = (
            (node.deadline - current_time).total_seconds() / 3600
            if node.deadline else 48
        )
        renego_count = self._renegotiation_counts.get(node.stakeholder_id, 0)

        # Lead time factor
        if hours_to_deadline > 48:
            lead_factor = 1.0
        elif hours_to_deadline > 24:
            lead_factor = 0.6
        elif hours_to_deadline > 4:
            lead_factor = 0.2
        else:
            lead_factor = -0.2  # This late, renegotiation is harmful

        # Frequency penalty (renegotiation tolerance from Step 2)
        frequency_factor = max(0.0, 1.0 - renego_count * 0.25)

        # Role tolerance
        role_tolerance = {
            StakeholderRole.BOSS: 0.6,
            StakeholderRole.CLIENT: 0.5,
            StakeholderRole.COLLEAGUE: 0.9,
            StakeholderRole.FRIEND: 0.95,
        }
        tolerance = role_tolerance.get(profile.role, 0.7)

        score = lead_factor * frequency_factor * tolerance
        accepted = score > 0.4

        if accepted:
            trust_delta = profile.trust_repair_rate * lead_factor
            msg = "Thanks for the heads up. Let's rework the timeline."
        else:
            trust_delta = -0.08 * (1 - lead_factor)
            msg = "This is really inconvenient. I was counting on this."

        self._renegotiation_counts[node.stakeholder_id] = renego_count + 1

        logger.info(f"Renegotiate response: stakeholder={node.stakeholder_id} "
                    f"lead={hours_to_deadline:.1f}h score={score:.2f} accepted={accepted}")

        return StakeholderResponse(
            stakeholder_id=node.stakeholder_id,
            accepted=accepted, trust_delta=trust_delta,
            message=msg, _true_satisfaction=score,
        )

    def _respond_to_clarify(self, action, node, trust, profile) -> StakeholderResponse:
        """Clarification is generally positively received."""
        return StakeholderResponse(
            stakeholder_id=node.stakeholder_id,
            accepted=True, trust_delta=0.01,
            message="Sure, let me clarify the details.",
            _true_satisfaction=0.65,
        )

    def _respond_to_defer(self, action, node, trust, profile) -> StakeholderResponse:
        """Deferral response — trust-gated at 0.50."""
        if trust.trust_score < 0.50:
            return StakeholderResponse(
                stakeholder_id=node.stakeholder_id,
                accepted=False, trust_delta=-0.04,
                message="I need an answer now, not later.",
                _true_satisfaction=0.3,
            )
        return StakeholderResponse(
            stakeholder_id=node.stakeholder_id,
            accepted=True, trust_delta=-0.01,
            message="Sure, take your time to think about it.",
            _true_satisfaction=0.5,
        )

    def _respond_to_delegate(self, action, node, trust, profile) -> StakeholderResponse:
        """Delegation response — trust-gated at 0.55 for clients."""
        if profile.role == StakeholderRole.CLIENT and trust.trust_score < 0.55:
            return StakeholderResponse(
                stakeholder_id=node.stakeholder_id,
                accepted=False, trust_delta=-0.06,
                message="I expected you to handle this directly.",
                _true_satisfaction=0.2,
            )
        return StakeholderResponse(
            stakeholder_id=node.stakeholder_id,
            accepted=True, trust_delta=-0.02,
            message="Okay, please make sure they're up to speed.",
            _true_satisfaction=0.45,
        )

    def _respond_to_silence(self, action, node, trust, profile) -> StakeholderResponse:
        """DO_NOTHING — Boss interprets silence as acceptance (Step 2)."""
        if profile.role == StakeholderRole.BOSS:
            return StakeholderResponse(
                stakeholder_id=node.stakeholder_id,
                accepted=True, trust_delta=0.0,
                message="",  # Silence
                _true_satisfaction=0.5,
            )
        return StakeholderResponse(
            stakeholder_id=node.stakeholder_id,
            accepted=True, trust_delta=0.0,
            message="",
            _true_satisfaction=0.5,
        )

    def _acceptance_message(self, profile: StakeholderProfile) -> str:
        messages = {
            StakeholderRole.BOSS: "Great, I'll be expecting it.",
            StakeholderRole.CLIENT: "Perfect, thank you for confirming.",
            StakeholderRole.COLLEAGUE: "Awesome, thanks!",
            StakeholderRole.FRIEND: "Great! Looking forward to it.",
        }
        return messages.get(profile.role, "Acknowledged.")
