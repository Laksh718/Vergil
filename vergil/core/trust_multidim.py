# vergil/core/trust_multidim.py
"""
Multi-Dimensional Trust Model
===============================

Three trust dimensions (based on organizational psychology):
  1. Reliability: Does the agent keep its commitments?
  2. Competence: Does the agent demonstrate good judgment?
  3. Benevolence: Does the agent seem to care about the relationship?

Action space gating uses the MINIMUM dimension (weakest link).
Composite trust = weighted average (for reward computation).

References:
    VERGIL_System_Design.md — Step 2, Step 5
    VERGIL_Phase1_Phase2_Implementation.md — P2.2
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum
import numpy as np
import logging

from .types import ActionType

logger = logging.getLogger('vergil.trust_multidim')


class TrustDimension(str, Enum):
    RELIABILITY  = "reliability"
    COMPETENCE   = "competence"
    BENEVOLENCE  = "benevolence"


@dataclass
class MultiDimTrustEntry:
    """
    Three-dimensional trust record for one stakeholder.
    """
    stakeholder_id: str

    reliability: float = 0.65
    competence: float = 0.65
    benevolence: float = 0.70

    reliability_history: List[Dict] = field(default_factory=list)
    competence_history: List[Dict] = field(default_factory=list)
    benevolence_history: List[Dict] = field(default_factory=list)

    reliability_kept: int = 0
    reliability_broken: int = 0
    competence_good_decisions: int = 0
    competence_poor_decisions: int = 0
    benevolence_positive: int = 0

    _reliability_momentum: float = 0.0
    _competence_momentum: float = 0.0
    _benevolence_momentum: float = 0.0

    @property
    def composite_trust(self) -> float:
        """Weighted composite: reliability weighted highest."""
        return (0.45 * self.reliability +
                0.35 * self.competence +
                0.20 * self.benevolence)

    @property
    def min_trust(self) -> float:
        """Minimum across dimensions. Used for action gating."""
        return min(self.reliability, self.competence, self.benevolence)

    def update(self, dimension: TrustDimension, delta: float,
               event: str, step: int) -> None:
        """Update a specific trust dimension."""
        if dimension == TrustDimension.RELIABILITY:
            self.reliability = float(np.clip(self.reliability + delta, 0, 1))
            self.reliability_history.append({
                'step': step, 'event': event,
                'delta': round(delta, 4), 'new_value': round(self.reliability, 4),
            })
            if delta < 0:
                self.reliability_broken += 1
            else:
                self.reliability_kept += 1
            self._reliability_momentum = 0.7 * self._reliability_momentum + 0.3 * delta

        elif dimension == TrustDimension.COMPETENCE:
            self.competence = float(np.clip(self.competence + delta, 0, 1))
            self.competence_history.append({
                'step': step, 'event': event, 'delta': round(delta, 4),
                'new_value': round(self.competence, 4),
            })
            if delta > 0:
                self.competence_good_decisions += 1
            else:
                self.competence_poor_decisions += 1
            self._competence_momentum = 0.7 * self._competence_momentum + 0.3 * delta

        elif dimension == TrustDimension.BENEVOLENCE:
            self.benevolence = float(np.clip(self.benevolence + delta, 0, 1))
            self.benevolence_history.append({
                'step': step, 'event': event, 'delta': round(delta, 4),
                'new_value': round(self.benevolence, 4),
            })
            if delta > 0:
                self.benevolence_positive += 1
            self._benevolence_momentum = 0.7 * self._benevolence_momentum + 0.3 * delta

        logger.debug(f"Trust update: {self.stakeholder_id} {dimension.value} "
                     f"delta={delta:+.3f} composite={self.composite_trust:.3f}")

    def to_vector(self) -> np.ndarray:
        """6-dim vector: [R, C, B, R_momentum, C_momentum, B_momentum]"""
        return np.array([
            self.reliability, self.competence, self.benevolence,
            self._reliability_momentum, self._competence_momentum,
            self._benevolence_momentum,
        ], dtype=np.float32)


class TrustGate:
    """
    Determines which actions are available given multi-dimensional trust state.
    Uses weakest-link principle: blocked if ANY required dimension is below threshold.
    """

    GATES = [
        (ActionType.COUNTER_PROPOSE,  TrustDimension.RELIABILITY,  0.35),
        (ActionType.RENEGOTIATE,      TrustDimension.RELIABILITY,  0.40),
        (ActionType.DELEGATE,         TrustDimension.COMPETENCE,   0.50),
        (ActionType.DEFER,            TrustDimension.COMPETENCE,   0.45),
        (ActionType.CLARIFY,          TrustDimension.BENEVOLENCE,  0.25),
    ]

    ROLE_THRESHOLD_MULTIPLIERS = {
        'boss': 1.3, 'client': 1.2, 'colleague': 0.9, 'friend': 0.7,
    }

    @classmethod
    def get_available_actions(cls, trust_entry: MultiDimTrustEntry,
                              stakeholder_role: str) -> List[ActionType]:
        """Return all currently available action types."""
        all_actions = list(ActionType)
        blocked = set()
        multiplier = cls.ROLE_THRESHOLD_MULTIPLIERS.get(stakeholder_role, 1.0)

        for action, dimension, threshold in cls.GATES:
            effective_threshold = min(0.95, threshold * multiplier)
            current_value = {
                TrustDimension.RELIABILITY: trust_entry.reliability,
                TrustDimension.COMPETENCE: trust_entry.competence,
                TrustDimension.BENEVOLENCE: trust_entry.benevolence,
            }[dimension]

            if current_value < effective_threshold:
                blocked.add(action)

        return [a for a in all_actions if a not in blocked]

    @classmethod
    def explain_blocks(cls, trust_entry: MultiDimTrustEntry,
                       stakeholder_role: str) -> List[Dict]:
        """Explain why specific actions are blocked."""
        multiplier = cls.ROLE_THRESHOLD_MULTIPLIERS.get(stakeholder_role, 1.0)
        blocks = []

        for action, dimension, threshold in cls.GATES:
            effective_threshold = min(0.95, threshold * multiplier)
            current_value = {
                TrustDimension.RELIABILITY: trust_entry.reliability,
                TrustDimension.COMPETENCE: trust_entry.competence,
                TrustDimension.BENEVOLENCE: trust_entry.benevolence,
            }[dimension]

            if current_value < effective_threshold:
                blocks.append({
                    'action': action.value,
                    'dimension': dimension.value,
                    'current': round(current_value, 3),
                    'required': round(effective_threshold, 3),
                    'deficit': round(effective_threshold - current_value, 3),
                    'recovery_hint': cls._recovery_hint(dimension),
                })

        return blocks

    @staticmethod
    def _recovery_hint(dimension: TrustDimension) -> str:
        hints = {
            TrustDimension.RELIABILITY: "Keep the next 2-3 commitments to recover",
            TrustDimension.COMPETENCE: "Demonstrate good judgment with an accurate counter-proposal",
            TrustDimension.BENEVOLENCE: "Provide a clear, honest explanation of constraints",
        }
        return hints.get(dimension, "")


class MultiDimTrustUpdater:
    """
    Maps (action_type, outcome) pairs to (dimension, delta) updates.
    """

    UPDATE_TABLE: Dict[str, Dict[str, List[Tuple[TrustDimension, float]]]] = {
        ActionType.ACCEPT.value: {
            'commitment_kept': [
                (TrustDimension.RELIABILITY, +0.08),
                (TrustDimension.COMPETENCE, +0.03),
            ],
            'commitment_broken': [
                (TrustDimension.RELIABILITY, -0.15),
                (TrustDimension.COMPETENCE, -0.05),
            ],
            'commitment_kept_early': [
                (TrustDimension.RELIABILITY, +0.12),
                (TrustDimension.COMPETENCE, +0.06),
            ],
        },
        ActionType.DECLINE.value: {
            'accepted_gracefully': [
                (TrustDimension.BENEVOLENCE, +0.04),
            ],
            'rejected_angrily': [
                (TrustDimension.RELIABILITY, -0.04),
                (TrustDimension.BENEVOLENCE, -0.06),
            ],
        },
        ActionType.COUNTER_PROPOSE.value: {
            'accepted': [
                (TrustDimension.COMPETENCE, +0.07),
                (TrustDimension.BENEVOLENCE, +0.04),
            ],
            'rejected': [
                (TrustDimension.COMPETENCE, -0.03),
            ],
        },
        ActionType.RENEGOTIATE.value: {
            'early_accepted': [
                (TrustDimension.RELIABILITY, +0.03),
                (TrustDimension.COMPETENCE, +0.05),
                (TrustDimension.BENEVOLENCE, +0.06),
            ],
            'late_accepted': [
                (TrustDimension.RELIABILITY, -0.04),
                (TrustDimension.BENEVOLENCE, +0.02),
            ],
            'rejected': [
                (TrustDimension.RELIABILITY, -0.08),
                (TrustDimension.COMPETENCE, -0.04),
            ],
        },
        ActionType.CLARIFY.value: {
            'any': [
                (TrustDimension.COMPETENCE, +0.02),
                (TrustDimension.BENEVOLENCE, +0.03),
            ],
        },
    }

    @classmethod
    def compute_deltas(cls, action_type: str, outcome: str,
                       lead_time_hours: float = 48,
                       timeliness_bonus: float = 0.0
                       ) -> List[Tuple[TrustDimension, float]]:
        """Compute trust deltas with lead time modifier."""
        updates = cls.UPDATE_TABLE.get(action_type, {}).get(outcome, [])
        if not updates:
            updates = cls.UPDATE_TABLE.get(action_type, {}).get('any', [])

        lead_multiplier = min(1.5, max(0.5, lead_time_hours / 24))

        result = []
        for dimension, base_delta in updates:
            delta = base_delta
            if base_delta > 0:
                delta *= lead_multiplier
            if timeliness_bonus > 0 and dimension == TrustDimension.RELIABILITY:
                delta += timeliness_bonus
            result.append((dimension, delta))

        return result
