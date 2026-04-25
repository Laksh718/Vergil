# vergil/training/evaluation.py
"""
VERGIL Evaluation Suite
========================

6-metric evaluation as specified in the openenv.yaml:
1. Commitment Fulfillment Rate (target: ≥70% at stage 4)
2. Trust Maintenance Index (mean final trust, target: ≥0.55)
3. Proactive Renegotiation Rate (target: ≥30% of at-risk)
4. Cascade Prevention Rate (target: ≥50%)
5. Feasibility Accuracy (Brier score, target: ≤0.15)
6. Trust Stability Index (σ trajectory, target: ≤0.12)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import logging

from vergil.core.types import (
    CommitmentStatus, ActionType, EpisodeRecord
)

logger = logging.getLogger('vergil.evaluation')


@dataclass
class EvaluationMetrics:
    """Full evaluation metrics for one episode."""
    commitment_fulfillment_rate: float = 0.0
    trust_maintenance_index: float = 0.0
    proactive_renegotiation_rate: float = 0.0
    cascade_prevention_rate: float = 0.0
    feasibility_accuracy: float = 0.0      # Brier score (lower = better)
    trust_stability_index: float = 0.0     # σ(trajectory) (lower = better)

    # Metadata
    episode_id: str = ""
    curriculum_stage: int = 0
    total_reward: float = 0.0

    def passes_targets(self, stage: int) -> Dict[str, bool]:
        """Check if metrics meet stage targets (Step 10)."""
        targets = {
            1: {'fulfillment': 0.5, 'trust': 0.5, 'proactive': 0.15},
            2: {'fulfillment': 0.55, 'trust': 0.50, 'proactive': 0.20},
            3: {'fulfillment': 0.60, 'trust': 0.50, 'proactive': 0.25, 'cascade': 0.40},
            4: {'fulfillment': 0.70, 'trust': 0.55, 'proactive': 0.30,
                'cascade': 0.50, 'brier': 0.15, 'stability': 0.12},
        }
        t = targets.get(stage, targets[4])
        return {
            'fulfillment': self.commitment_fulfillment_rate >= t.get('fulfillment', 0),
            'trust': self.trust_maintenance_index >= t.get('trust', 0),
            'proactive': self.proactive_renegotiation_rate >= t.get('proactive', 0),
            'cascade': self.cascade_prevention_rate >= t.get('cascade', 0) if 'cascade' in t else True,
            'brier': self.feasibility_accuracy <= t.get('brier', 1.0) if 'brier' in t else True,
            'stability': self.trust_stability_index <= t.get('stability', 1.0) if 'stability' in t else True,
        }

    def to_dict(self) -> Dict:
        return {
            'commitment_fulfillment_rate': round(self.commitment_fulfillment_rate, 4),
            'trust_maintenance_index': round(self.trust_maintenance_index, 4),
            'proactive_renegotiation_rate': round(self.proactive_renegotiation_rate, 4),
            'cascade_prevention_rate': round(self.cascade_prevention_rate, 4),
            'feasibility_accuracy': round(self.feasibility_accuracy, 4),
            'trust_stability_index': round(self.trust_stability_index, 4),
            'episode_id': self.episode_id,
            'curriculum_stage': self.curriculum_stage,
            'total_reward': round(self.total_reward, 4),
        }


class EvaluationSuite:
    """Computes evaluation metrics from episode records."""

    def evaluate_episode(self, episode_record: EpisodeRecord,
                         final_trust_scores: Dict[str, float],
                         trust_trajectories: Dict[str, List[float]],
                         feasibility_predictions: List[float],
                         feasibility_actuals: List[float],
                         cascade_eligible: int = 0,
                         cascades_prevented: int = 0) -> EvaluationMetrics:
        """Compute all 6 metrics for one episode."""
        metrics = EvaluationMetrics(
            episode_id=episode_record.episode_id,
            curriculum_stage=episode_record.curriculum_stage,
            total_reward=episode_record.total_reward,
        )

        # 1. Commitment Fulfillment Rate
        metrics.commitment_fulfillment_rate = episode_record.commitment_fulfillment_rate

        # 2. Trust Maintenance Index (mean final trust)
        if final_trust_scores:
            metrics.trust_maintenance_index = float(np.mean(list(final_trust_scores.values())))

        # 3. Proactive Renegotiation Rate
        total_actions = len(episode_record.steps)
        renegotiations = sum(
            1 for s in episode_record.steps
            if s.get('action') == 'renegotiate'
        )
        at_risk_actions = max(1, total_actions)
        metrics.proactive_renegotiation_rate = renegotiations / at_risk_actions

        # 4. Cascade Prevention Rate
        if cascade_eligible > 0:
            metrics.cascade_prevention_rate = cascades_prevented / cascade_eligible
        else:
            metrics.cascade_prevention_rate = 1.0

        # 5. Feasibility Accuracy (Brier Score)
        if feasibility_predictions and feasibility_actuals:
            n = min(len(feasibility_predictions), len(feasibility_actuals))
            brier = float(np.mean([
                (feasibility_predictions[i] - feasibility_actuals[i]) ** 2
                for i in range(n)
            ]))
            metrics.feasibility_accuracy = brier

        # 6. Trust Stability Index
        if trust_trajectories:
            stds = []
            for sid, trajectory in trust_trajectories.items():
                if len(trajectory) > 2:
                    stds.append(float(np.std(trajectory)))
            if stds:
                metrics.trust_stability_index = float(np.mean(stds))

        return metrics

    def aggregate_metrics(self, metrics_list: List[EvaluationMetrics]) -> Dict:
        """Aggregate metrics over multiple episodes."""
        if not metrics_list:
            return {}

        return {
            'n_episodes': len(metrics_list),
            'mean_fulfillment': round(float(np.mean([m.commitment_fulfillment_rate for m in metrics_list])), 4),
            'mean_trust': round(float(np.mean([m.trust_maintenance_index for m in metrics_list])), 4),
            'mean_proactive': round(float(np.mean([m.proactive_renegotiation_rate for m in metrics_list])), 4),
            'mean_cascade_prevention': round(float(np.mean([m.cascade_prevention_rate for m in metrics_list])), 4),
            'mean_brier_score': round(float(np.mean([m.feasibility_accuracy for m in metrics_list])), 4),
            'mean_trust_stability': round(float(np.mean([m.trust_stability_index for m in metrics_list])), 4),
            'mean_reward': round(float(np.mean([m.total_reward for m in metrics_list])), 4),
        }
