# vergil/curriculum/curriculum_engine.py
"""
Curriculum Engine
==================

Manages curriculum progression, scenario generation, and promotion.

4-Stage Curriculum:
  Stage 1 (Foundation): 1 stakeholder, 2-3 nodes, no conflicts
  Stage 2 (Complexity): 2 stakeholders, 3-6 nodes, implicit commitments
  Stage 3 (Social): 3 stakeholders, 5-10 nodes, resource conflicts, force majeure
  Stage 4 (Adversarial): 4 stakeholders, 8-15 nodes, adversarial behaviors

Sampling strategy:
  60% → Targeted at high-failure topologies from FTD
  30% → Random stage-appropriate
  10% → Easy (stage-2 lower; prevents catastrophic forgetting)

References:
    VERGIL_System_Design.md — Step 8 (Curriculum), Step 10 (Targets)
    VERGIL_Phase1_Phase2_Implementation.md — P2.6
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import logging

from vergil.curriculum.failure_db import FailureTopologyDatabase
from vergil.curriculum.scenario_generator import ScenarioGenerator

logger = logging.getLogger('vergil.curriculum')


@dataclass
class CurriculumStageConfig:
    stage: int
    name: str
    n_nodes_range: tuple
    n_stakeholders: int
    include_implicit: bool
    include_resource_conflicts: bool
    include_force_majeure: bool
    include_adversarial: bool
    step_hours: float
    max_episode_steps: int
    promotion_reward_threshold: float
    promotion_window: int


CURRICULUM_STAGES = [
    CurriculumStageConfig(
        stage=1, name='foundation',
        n_nodes_range=(2, 3), n_stakeholders=1,
        include_implicit=False, include_resource_conflicts=False,
        include_force_majeure=False, include_adversarial=False,
        step_hours=2.0, max_episode_steps=25,
        promotion_reward_threshold=0.3, promotion_window=20,
    ),
    CurriculumStageConfig(
        stage=2, name='complexity_intro',
        n_nodes_range=(3, 6), n_stakeholders=2,
        include_implicit=True, include_resource_conflicts=False,
        include_force_majeure=False, include_adversarial=False,
        step_hours=2.0, max_episode_steps=35,
        promotion_reward_threshold=0.45, promotion_window=30,
    ),
    CurriculumStageConfig(
        stage=3, name='social_dynamics',
        n_nodes_range=(5, 10), n_stakeholders=3,
        include_implicit=True, include_resource_conflicts=True,
        include_force_majeure=True, include_adversarial=False,
        step_hours=1.5, max_episode_steps=50,
        promotion_reward_threshold=0.55, promotion_window=40,
    ),
    CurriculumStageConfig(
        stage=4, name='adversarial',
        n_nodes_range=(8, 15), n_stakeholders=4,
        include_implicit=True, include_resource_conflicts=True,
        include_force_majeure=True, include_adversarial=True,
        step_hours=1.0, max_episode_steps=80,
        promotion_reward_threshold=0.65, promotion_window=50,
    ),
]


class CurriculumEngine:
    """
    Manages curriculum progression and FTD-biased scenario generation.
    """

    def __init__(self, failure_db: FailureTopologyDatabase,
                 scenario_generator: ScenarioGenerator,
                 initial_stage: int = 1):
        self.failure_db = failure_db
        self.scenario_generator = scenario_generator
        self.current_stage = initial_stage
        self.stage_config = CURRICULUM_STAGES[initial_stage - 1]

        self._reward_history: Dict[int, List[float]] = {s: [] for s in range(1, 5)}
        self._episode_count = 0

        logger.info(f"CurriculumEngine: starting stage {initial_stage} "
                    f"({self.stage_config.name})")

    def generate_next_episode(self) -> Dict:
        """Generate scenario with 60/30/10 sampling strategy."""
        self._episode_count += 1
        strategy_roll = np.random.random()

        if strategy_roll < 0.60:
            failure_weights = self.failure_db.get_curriculum_sampling_weights(
                self.current_stage)
            topology_hint = (max(failure_weights, key=failure_weights.get)
                             if failure_weights else None)
            scenario = self.scenario_generator.generate(
                stage=self.current_stage, topology_hint=topology_hint,
                mode='targeted')
        elif strategy_roll < 0.90:
            scenario = self.scenario_generator.generate(
                stage=self.current_stage, mode='random')
        else:
            easy_stage = max(1, self.current_stage - 2)
            scenario = self.scenario_generator.generate(
                stage=easy_stage, mode='random')

        return scenario

    def record_episode_reward(self, reward: float, stage: int) -> None:
        """Record episode reward for promotion tracking."""
        self._reward_history[stage].append(reward)
        window = self.stage_config.promotion_window
        if len(self._reward_history[stage]) > window * 2:
            self._reward_history[stage] = self._reward_history[stage][-window:]

    def check_promotion(self) -> bool:
        """Check if agent is ready for next stage."""
        if self.current_stage >= 4:
            return False

        config = self.stage_config
        history = self._reward_history[self.current_stage]

        if len(history) < config.promotion_window:
            return False

        rolling_avg = np.mean(history[-config.promotion_window:])

        if rolling_avg >= config.promotion_reward_threshold:
            old_stage = self.current_stage
            self.current_stage += 1
            self.stage_config = CURRICULUM_STAGES[self.current_stage - 1]
            logger.info(f"CURRICULUM PROMOTION: stage {old_stage} → {self.current_stage} "
                        f"(avg_reward={rolling_avg:.4f})")
            return True

        return False

    def get_status(self) -> Dict:
        config = self.stage_config
        history = self._reward_history[self.current_stage]
        rolling_avg = (np.mean(history[-config.promotion_window:])
                       if len(history) >= 5 else 0.0)

        return {
            'current_stage': self.current_stage,
            'stage_name': config.name,
            'episodes_at_stage': len(history),
            'rolling_avg_reward': round(float(rolling_avg), 4),
            'promotion_threshold': config.promotion_reward_threshold,
            'progress_to_promotion': round(
                min(1.0, rolling_avg / config.promotion_reward_threshold), 3),
            'failure_db_stats': self.failure_db.get_statistics(),
        }
