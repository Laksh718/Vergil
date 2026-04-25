# vergil/curriculum/scenario_generator.py
"""
Procedural Scenario Generator
===============================

Generates scenarios programmatically based on curriculum stage config.
Supports topology-targeted generation (biased toward high-failure patterns).

References:
    VERGIL_Phase1_Phase2_Implementation.md — P2.6, P2.7
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import logging

from vergil.core.types import StakeholderRole

logger = logging.getLogger('vergil.scenario_gen')

# Commitment template pools (sampled and parameterized)
COMMITMENT_TEMPLATES = [
    {'label': 'Write {doc_type}', 'duration': (2, 5), 'type': 'explicit_hard',
     'params': {'doc_type': ['report', 'spec', 'proposal', 'analysis']}},
    {'label': 'Review {artifact}', 'duration': (1, 3), 'type': 'explicit_soft',
     'params': {'artifact': ['PR #437', 'design doc', 'budget proposal', 'Q2 plan']}},
    {'label': 'Prepare {presentation_type}', 'duration': (3, 6), 'type': 'explicit_hard',
     'params': {'presentation_type': ['board slides', 'client demo', 'team update', 'quarterly review']}},
    {'label': 'Fix {bug_type}', 'duration': (1.5, 4), 'type': 'explicit_hard',
     'params': {'bug_type': ['auth module', 'payment flow', 'API gateway', 'data pipeline']}},
    {'label': '{social_activity} with {person}', 'duration': (1, 3), 'type': 'social',
     'params': {'social_activity': ['Lunch', 'Dinner', 'Coffee', 'Drinks'],
                'person': ['Sam', 'Alex', 'Jordan', 'team']}},
    {'label': 'Sprint {meeting_type}', 'duration': (0.5, 1.5), 'type': 'explicit_soft',
     'params': {'meeting_type': ['retrospective', 'planning', 'standup', 'review']}},
    {'label': 'Update {doc_type}', 'duration': (1, 3), 'type': 'explicit_soft',
     'params': {'doc_type': ['API documentation', 'runbook', 'onboarding guide', 'changelog']}},
]

STAKEHOLDER_POOLS = {
    'boss': [
        {'id': 'boss_01', 'name': 'Sarah Chen', 'role': 'boss'},
        {'id': 'boss_02', 'name': 'Marcus Johnson', 'role': 'boss'},
    ],
    'client': [
        {'id': 'client_01', 'name': 'Acme Corp', 'role': 'client'},
        {'id': 'client_02', 'name': 'Board Committee', 'role': 'client'},
    ],
    'colleague': [
        {'id': 'colleague_01', 'name': 'Alex Rivera', 'role': 'colleague'},
        {'id': 'colleague_02', 'name': 'Jordan Kim', 'role': 'colleague'},
    ],
    'friend': [
        {'id': 'friend_01', 'name': 'Sam', 'role': 'friend'},
        {'id': 'friend_02', 'name': 'Taylor', 'role': 'friend'},
    ],
}

MESSAGE_TEMPLATES = [
    "Can you {verb} the {object} by {deadline_word}?",
    "I need the {object} {deadline_word}. Can you handle it?",
    "Hey, whenever you get a chance, could you {verb} the {object}?",
    "Actually, can you also include the {extra} in the {object}?",
    "We need to move the {event} up by {hours} hours. Can you be ready?",
]


class ScenarioGenerator:
    """
    Generates scenarios programmatically based on curriculum stage.

    Supports:
    - 'random': Stage-appropriate random generation
    - 'targeted': Biased toward specific topology patterns (FTD-guided)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, stage: int, topology_hint: Optional[str] = None,
                 mode: str = 'random') -> Dict:
        """Generate a complete scenario."""
        from vergil.curriculum.curriculum_engine import CURRICULUM_STAGES

        config = CURRICULUM_STAGES[stage - 1]
        base_time = datetime.now().replace(
            hour=9, minute=0, second=0, microsecond=0)

        # Sample stakeholders
        n_stakeholders = config.n_stakeholders
        stakeholders = self._sample_stakeholders(n_stakeholders, stage)

        # Sample commitments
        n_nodes = self.rng.integers(config.n_nodes_range[0], config.n_nodes_range[1] + 1)
        commitments = self._generate_commitments(
            n_nodes, stakeholders, base_time, config, topology_hint)

        # Generate edges
        edges = self._generate_edges(commitments, config, topology_hint)

        # Initial trust (lower at higher stages)
        initial_trust = {}
        for s in stakeholders:
            base_trust = max(0.30, 0.70 - (stage - 1) * 0.08)
            noise = self.rng.normal(0, 0.05)
            initial_trust[s['id']] = round(float(np.clip(base_trust + noise, 0.2, 0.9)), 2)

        # Message schedule (mid-episode disruptions)
        messages = self._generate_messages(
            stakeholders, base_time, n_nodes, config)

        scenario = {
            'scenario_id': f"gen_{stage}_{str(uuid.uuid4())[:6]}",
            'description': f"Stage {stage} generated scenario ({mode})",
            'start_time': base_time.isoformat(),
            'seed_commitments': commitments,
            'seed_edges': edges,
            'stakeholders': stakeholders,
            'initial_trust': initial_trust,
            'message_schedule': messages,
        }

        logger.debug(f"Generated scenario: {scenario['scenario_id']} "
                     f"nodes={len(commitments)} edges={len(edges)} "
                     f"messages={len(messages)}")
        return scenario

    def _sample_stakeholders(self, n: int, stage: int) -> List[Dict]:
        """Sample stakeholders appropriate to the stage."""
        roles_by_stage = {
            1: ['boss'],
            2: ['boss', 'colleague'],
            3: ['boss', 'client', 'colleague'],
            4: ['boss', 'client', 'colleague', 'friend'],
        }
        available_roles = roles_by_stage.get(stage, roles_by_stage[4])
        selected = []

        for i in range(min(n, len(available_roles))):
            role = available_roles[i]
            pool = STAKEHOLDER_POOLS.get(role, [])
            if pool:
                selected.append(pool[self.rng.integers(0, len(pool))])

        return selected

    def _generate_commitments(self, n: int, stakeholders: List[Dict],
                              base_time: datetime, config, topology_hint) -> List[Dict]:
        """Generate commitment nodes."""
        commitments = []
        stakeholder_ids = [s['id'] for s in stakeholders]
        stakeholder_roles = {s['id']: s['role'] for s in stakeholders}

        for i in range(n):
            template = COMMITMENT_TEMPLATES[self.rng.integers(0, len(COMMITMENT_TEMPLATES))]

            # Fill template parameters
            label = template['label']
            for param, values in template['params'].items():
                label = label.replace('{' + param + '}', values[self.rng.integers(0, len(values))])

            sid = stakeholder_ids[self.rng.integers(0, len(stakeholder_ids))]
            duration = float(self.rng.uniform(*template['duration']))

            # Deadline: proportional to stage difficulty
            deadline_hours = float(self.rng.uniform(4, 48 - (config.stage - 1) * 8))
            deadline = base_time + timedelta(hours=deadline_hours)

            status = 'pending' if self.rng.random() < 0.5 else 'accepted'

            commitments.append({
                'id': f"C{i+1}",
                'label': label,
                'description': f"Generated: {label}",
                'type': template['type'],
                'stakeholder_id': sid,
                'role': stakeholder_roles[sid],
                'deadline': deadline.isoformat(),
                'duration_hours': round(duration, 1),
                'duration_std': round(duration * 0.3, 1),
                'status': status,
            })

        return commitments

    def _generate_edges(self, commitments: List[Dict], config,
                        topology_hint: Optional[str]) -> List[Dict]:
        """Generate dependency edges (temporal chains, resource conflicts)."""
        edges = []
        n = len(commitments)

        if n < 2:
            return edges

        # Temporal chain: create a chain of 2-3 connected nodes
        chain_length = min(3, n)
        indices = list(range(n))
        self.rng.shuffle(indices)

        for i in range(chain_length - 1):
            edges.append({
                'from': commitments[indices[i]]['id'],
                'to': commitments[indices[i + 1]]['id'],
                'type': 'temporal',
                'lag_hours': float(self.rng.uniform(0.5, 2.0)),
            })

        # Resource conflict (stage 3+)
        if config.include_resource_conflicts and n >= 3:
            a, b = self.rng.choice(n, 2, replace=False)
            edges.append({
                'from': commitments[a]['id'],
                'to': commitments[b]['id'],
                'type': 'resource',
            })

        return edges

    def _generate_messages(self, stakeholders: List[Dict], base_time: datetime,
                           n_nodes: int, config) -> List[Dict]:
        """Generate mid-episode message schedule."""
        messages = []
        n_messages = max(1, n_nodes // 2)

        for i in range(n_messages):
            sender = stakeholders[self.rng.integers(0, len(stakeholders))]
            delivery_hours = float(self.rng.uniform(1, config.max_episode_steps * config.step_hours * 0.5))
            delivery = base_time + timedelta(hours=delivery_hours)

            template = MESSAGE_TEMPLATES[self.rng.integers(0, len(MESSAGE_TEMPLATES))]
            content = template.replace('{verb}', self.rng.choice(['review', 'prepare', 'finish', 'handle']))
            content = content.replace('{object}', self.rng.choice(['report', 'presentation', 'analysis', 'document']))
            content = content.replace('{deadline_word}', self.rng.choice(['by Thursday', 'by end of day', 'ASAP', 'tomorrow']))
            content = content.replace('{extra}', self.rng.choice(['competitor analysis', 'financial projections', 'risk assessment']))
            content = content.replace('{event}', self.rng.choice(['meeting', 'demo', 'review', 'presentation']))
            content = content.replace('{hours}', str(self.rng.integers(1, 5)))

            messages.append({
                'sender_id': sender['id'],
                'sender_role': sender['role'],
                'content': content,
                'delivery_time': delivery.isoformat(),
            })

        return messages
