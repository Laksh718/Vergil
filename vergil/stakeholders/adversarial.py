# vergil/stakeholders/adversarial.py
"""
Adversarial Stakeholder Behaviors
==================================

Stage 4 behaviors that test robustness:
1. Moving goalposts: Deadline changes after acceptance
2. Scope creep: Additional requirements added mid-commitment
3. Guilt manipulation: Social pressure to accept infeasible requests
4. Authority override: Boss-level force bypassing normal negotiation
5. Irrational escalation: Escalates despite reasonable handling
6. False urgency: Fabricated urgency; real deadline is later

References:
    VERGIL_System_Design.md — Step 2 (Adversarial Behaviors)
    VERGIL_Phase1_Phase2_Implementation.md — P2.7
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger('vergil.adversarial')


@dataclass
class AdversarialEvent:
    """An adversarial behavior event injected into the simulation."""
    event_type: str
    stakeholder_id: str
    description: str
    deadline_shift_hours: Optional[float] = None
    scope_addition: Optional[str] = None
    trust_manipulation_signal: Optional[float] = None
    linguistic_cues: List[str] = field(default_factory=list)


class AdversarialBehaviorGenerator:
    """
    Generates adversarial events for curriculum stage 4.
    Each behavior has observable linguistic cues the agent CAN learn to detect.
    """

    BEHAVIORS = {
        'goal_post_shift': {
            'description': "Deadline moved earlier after acceptance",
            'applicable_roles': ['boss', 'client'],
            'probability': 0.15,
            'cues': ['actually', 'just realized', 'by the way', 'changed my mind'],
            'effect': 'shift_deadline',
            'shift_hours': (-24, -4),
        },
        'scope_creep': {
            'description': "Additional requirements added mid-commitment",
            'applicable_roles': ['boss', 'client', 'colleague'],
            'probability': 0.20,
            'cues': ['also', 'one more thing', "while you're at it", 'might as well'],
            'effect': 'add_scope',
        },
        'guilt_manipulation': {
            'description': "Social guilt pressure to accept infeasible request",
            'applicable_roles': ['boss', 'friend'],
            'probability': 0.12,
            'cues': ['just this once', 'I thought we were a team',
                     'really disappointing', 'count on you', 'always relied on you'],
            'effect': 'pressure_accept',
        },
        'false_urgency': {
            'description': "Fabricated urgency; real deadline is later",
            'applicable_roles': ['boss', 'client'],
            'probability': 0.10,
            'cues': ['absolutely critical', "if this isn't done", 'major consequences'],
            'effect': 'inflate_urgency',
            '_true_flexibility_hours': 48,
        },
        'irrational_escalation': {
            'description': "Escalates despite reasonable handling",
            'applicable_roles': ['boss', 'client'],
            'probability': 0.08,
            'cues': [],  # No observable cue — purely irrational
            'effect': 'escalate_trust_damage',
        },
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def sample_events(self, step: int, stakeholder_ids: List[str],
                      stakeholder_roles: Dict[str, str],
                      curriculum_stage: int) -> List[AdversarialEvent]:
        """Sample adversarial events for this step. Stage 4 only."""
        if curriculum_stage < 4:
            return []

        events = []
        for behavior_name, config in self.BEHAVIORS.items():
            if self.rng.random() < config['probability']:
                eligible = [
                    sid for sid in stakeholder_ids
                    if stakeholder_roles.get(sid) in config['applicable_roles']
                ]
                if not eligible:
                    continue

                target_sid = eligible[self.rng.integers(0, len(eligible))]

                event = AdversarialEvent(
                    event_type=behavior_name,
                    stakeholder_id=target_sid,
                    description=config['description'],
                    linguistic_cues=config.get('cues', []),
                )

                if config.get('effect') == 'shift_deadline':
                    shift = self.rng.uniform(*config['shift_hours'])
                    event.deadline_shift_hours = round(float(shift), 1)

                elif config.get('effect') == 'add_scope':
                    scope_additions = [
                        "Include competitor analysis", "Add financial projections",
                        "Present security audit results", "Add risk assessment section",
                    ]
                    event.scope_addition = scope_additions[
                        self.rng.integers(0, len(scope_additions))]

                events.append(event)
                logger.info(f"Adversarial event: {behavior_name} → {target_sid}")

        return events
