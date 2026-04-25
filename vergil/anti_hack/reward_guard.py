# vergil/anti_hack/reward_guard.py
"""
Anti-Reward-Hacking Detection
===============================

Detects and penalizes reward exploitation strategies:

1. Decline Everything: High decline rate + lost feasible opportunities
2. Accept Everything: High accept rate + inevitable failures
3. Always Predict Feasible: Overconfident predictions that fail
4. Always Predict Infeasible: Pessimistic predictions avoiding commitment
5. Renegotiation Farming: Excessive renegotiation to game proactive bonus

Operates as a meta-analysis layer on top of reward components.

References:
    VERGIL_System_Design.md — Step 7 (Anti-Hack Mechanisms)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger('vergil.reward_guard')


@dataclass
class HackDetection:
    """Result of hack detection analysis."""
    detected: bool = False
    hack_type: Optional[str] = None
    confidence: float = 0.0
    penalty: float = 0.0
    evidence: Dict = field(default_factory=dict)


class RewardGuard:
    """
    Meta-analysis layer that detects reward exploitation patterns.

    All detections return (detected, confidence, penalty).
    Penalties are multiplicative — applied to the total reward.
    """

    # Detection thresholds
    DECLINE_RATE_THRESHOLD = 0.60       # >60% decline rate → suspicious
    ACCEPT_RATE_THRESHOLD = 0.95        # >95% accept rate → suspicious
    PREDICTION_BIAS_THRESHOLD = 0.80    # >80% same prediction → suspicious
    RENEGOTIATION_RATE_THRESHOLD = 0.40 # >40% of actions are renegotiations → suspicious

    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self._action_history: List[str] = []
        self._prediction_history: List[float] = []
        self._outcome_history: List[bool] = []

    def record_action(self, action_type: str,
                      feasibility_prediction: Optional[float] = None,
                      outcome_success: Optional[bool] = None) -> None:
        """Record action for pattern detection."""
        self._action_history.append(action_type)
        if feasibility_prediction is not None:
            self._prediction_history.append(feasibility_prediction)
        if outcome_success is not None:
            self._outcome_history.append(outcome_success)

    def check_all(self) -> List[HackDetection]:
        """Run all hack detection checks. Returns list of detections."""
        detections = []

        d = self._check_decline_everything()
        if d.detected:
            detections.append(d)

        d = self._check_accept_everything()
        if d.detected:
            detections.append(d)

        d = self._check_prediction_bias()
        if d.detected:
            detections.append(d)

        d = self._check_renegotiation_farming()
        if d.detected:
            detections.append(d)

        return detections

    def compute_total_penalty(self) -> float:
        """Returns total multiplicative penalty factor (0-1, where 1 = no penalty)."""
        detections = self.check_all()
        if not detections:
            return 1.0

        total_penalty = sum(d.penalty for d in detections)
        return max(0.1, 1.0 - total_penalty)

    def _check_decline_everything(self) -> HackDetection:
        """Detect if agent declines most commitments."""
        if len(self._action_history) < self.window_size:
            return HackDetection()

        recent = self._action_history[-self.window_size:]
        decline_rate = sum(1 for a in recent if a == 'decline') / len(recent)

        if decline_rate > self.DECLINE_RATE_THRESHOLD:
            return HackDetection(
                detected=True,
                hack_type='decline_everything',
                confidence=min(1.0, (decline_rate - self.DECLINE_RATE_THRESHOLD) * 5),
                penalty=0.3 * decline_rate,
                evidence={'decline_rate': round(decline_rate, 3),
                          'window': self.window_size},
            )
        return HackDetection()

    def _check_accept_everything(self) -> HackDetection:
        """Detect if agent accepts everything regardless of feasibility."""
        if len(self._action_history) < self.window_size:
            return HackDetection()

        recent = self._action_history[-self.window_size:]
        accept_rate = sum(1 for a in recent if a == 'accept') / len(recent)

        if accept_rate > self.ACCEPT_RATE_THRESHOLD:
            return HackDetection(
                detected=True,
                hack_type='accept_everything',
                confidence=min(1.0, (accept_rate - self.ACCEPT_RATE_THRESHOLD) * 10),
                penalty=0.25,
                evidence={'accept_rate': round(accept_rate, 3)},
            )
        return HackDetection()

    def _check_prediction_bias(self) -> HackDetection:
        """Detect if agent always predicts the same feasibility."""
        if len(self._prediction_history) < 10:
            return HackDetection()

        recent = self._prediction_history[-15:]
        std = np.std(recent)

        if std < 0.05:  # Nearly constant predictions
            mean_pred = np.mean(recent)
            return HackDetection(
                detected=True,
                hack_type='prediction_bias',
                confidence=1.0 - std * 20,
                penalty=0.15,
                evidence={'prediction_std': round(float(std), 4),
                          'mean_prediction': round(float(mean_pred), 3)},
            )
        return HackDetection()

    def _check_renegotiation_farming(self) -> HackDetection:
        """Detect excessive renegotiation to farm proactive bonus."""
        if len(self._action_history) < self.window_size:
            return HackDetection()

        recent = self._action_history[-self.window_size:]
        renego_rate = sum(1 for a in recent if a == 'renegotiate') / len(recent)

        if renego_rate > self.RENEGOTIATION_RATE_THRESHOLD:
            return HackDetection(
                detected=True,
                hack_type='renegotiation_farming',
                confidence=min(1.0, (renego_rate - self.RENEGOTIATION_RATE_THRESHOLD) * 5),
                penalty=0.2 * renego_rate,
                evidence={'renegotiation_rate': round(renego_rate, 3)},
            )
        return HackDetection()

    def reset(self) -> None:
        """Reset at episode boundary."""
        self._action_history = []
        self._prediction_history = []
        self._outcome_history = []
