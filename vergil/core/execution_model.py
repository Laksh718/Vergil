# vergil/core/execution_model.py
"""
Probabilistic Execution Model
===============================

Phase 2 makes task durations stochastic:
- Duration: Lognormal(log(estimated), σ) — always positive, right-skewed
- Interruptions: Poisson process with rate λ per hour
- Quality: Beta(α, β) with cognitive load degradation
- Partial completion: tasks can be N% done when time runs out

References:
    VERGIL_Phase1_Phase2_Implementation.md — P2.3
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional
import numpy as np
import logging

logger = logging.getLogger('vergil.execution')


@dataclass
class ExecutionSample:
    """Result of sampling from the execution distribution for one task."""
    node_id: str
    estimated_duration: float
    actual_duration: float
    duration_ratio: float       # actual / estimated (calibration signal)
    quality_score: float        # 0 = terrible; 1 = excellent
    interrupted: bool = False
    interruption_duration: float = 0.0
    completed_on_time: bool = True
    completion_fraction_at_deadline: float = 1.0


class ProbabilisticExecutionEngine:
    """
    Samples actual execution outcomes from stochastic distributions.

    Duration: Lognormal(log(estimated), σ) — right-skewed, always positive
    Interruptions: Poisson process with rate λ per hour
    Quality: Beta(α, β) with cognitive load degradation
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.interruption_rate_per_hour = 0.15
        self.interruption_duration_mean = 0.5
        self.interruption_duration_std = 0.3
        self.default_duration_sigma = 0.35
        self.quality_base_alpha = 8.0
        self.quality_base_beta = 2.0

    def sample_execution(self, node, cognitive_load: float = 0.0,
                         current_time: Optional[datetime] = None) -> ExecutionSample:
        """Sample actual execution outcome for a commitment node."""
        estimated = node.estimated_duration_hours
        sigma = node.duration_std_hours / max(node.estimated_duration_hours, 0.5)
        effective_sigma = sigma * (1 + cognitive_load * 0.5)
        effective_sigma = max(0.1, min(1.5, effective_sigma))

        actual_duration = float(self.rng.lognormal(
            mean=np.log(max(0.1, estimated)),
            sigma=effective_sigma
        ))

        n_interruptions = self.rng.poisson(
            self.interruption_rate_per_hour * actual_duration)

        interruption_total = 0.0
        interrupted = n_interruptions > 0
        if interrupted:
            for _ in range(n_interruptions):
                dur = float(self.rng.normal(
                    self.interruption_duration_mean,
                    self.interruption_duration_std))
                interruption_total += max(0, dur)

        total_time = actual_duration + interruption_total

        quality_alpha = self.quality_base_alpha * (1 - cognitive_load * 0.6)
        quality_beta = self.quality_base_beta * (1 + cognitive_load * 0.4)
        quality = float(self.rng.beta(max(0.5, quality_alpha), max(0.5, quality_beta)))

        ct = current_time or datetime.now()
        deadline_hours = (
            (node.deadline - ct).total_seconds() / 3600 if node.deadline else float('inf')
        )
        completed_on_time = total_time <= deadline_hours
        completion_fraction = 1.0 if completed_on_time else min(1.0, deadline_hours / total_time)

        return ExecutionSample(
            node_id=node.node_id,
            estimated_duration=estimated,
            actual_duration=round(total_time, 3),
            duration_ratio=round(total_time / max(estimated, 0.01), 3),
            quality_score=round(quality, 3),
            interrupted=interrupted,
            interruption_duration=round(interruption_total, 3),
            completed_on_time=completed_on_time,
            completion_fraction_at_deadline=round(completion_fraction, 3),
        )

    def sample_duration_at_decision_time(self, estimated_hrs: float,
                                         std_hrs: float,
                                         cognitive_load: float = 0.0,
                                         n_samples: int = 100) -> Dict[str, float]:
        """Return percentile statistics for agent decision-making."""
        effective_sigma = (std_hrs / max(estimated_hrs, 0.5)) * (1 + cognitive_load * 0.5)

        samples = self.rng.lognormal(
            mean=np.log(max(0.1, estimated_hrs)),
            sigma=max(0.1, min(1.5, effective_sigma)),
            size=n_samples,
        )

        interruption_overhead = (self.interruption_rate_per_hour *
                                 self.interruption_duration_mean * estimated_hrs)

        return {
            'p10': float(np.percentile(samples, 10) + interruption_overhead * 0.5),
            'p50': float(np.percentile(samples, 50) + interruption_overhead),
            'p90': float(np.percentile(samples, 90) + interruption_overhead * 1.5),
            'p99': float(np.percentile(samples, 99) + interruption_overhead * 2.0),
            'mean': float(np.mean(samples) + interruption_overhead),
            'prob_within_budget': float(np.mean(samples < estimated_hrs * 1.1)),
        }


class ForceMajeureGenerator:
    """
    Generates external interruption events (stage 3+).
    ~1 per 15-20 steps, non-negotiable.
    """

    FORCE_MAJEURE_TYPES = [
        {'type': 'major_incident', 'description': 'Production system down — all hands required',
         'blocks_hours': (2, 6), 'probability': 0.05},
        {'type': 'personal_emergency', 'description': 'Family emergency requires immediate attention',
         'blocks_hours': (4, 12), 'probability': 0.03},
        {'type': 'executive_override', 'description': 'C-level priority inserted into schedule',
         'blocks_hours': (1, 3), 'probability': 0.07},
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def sample_event(self, step: int, curriculum_stage: int) -> Optional[Dict]:
        """Sample a force majeure event. Only in stage 3+."""
        if curriculum_stage < 3:
            return None

        for event_type in self.FORCE_MAJEURE_TYPES:
            if self.rng.random() < event_type['probability']:
                duration = float(self.rng.uniform(*event_type['blocks_hours']))
                return {
                    'type': event_type['type'],
                    'description': event_type['description'],
                    'duration_hours': round(duration, 1),
                    'step': step,
                    'non_negotiable': True,
                }

        return None
