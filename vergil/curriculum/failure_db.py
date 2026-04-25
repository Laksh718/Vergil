# vergil/curriculum/failure_db.py
"""
Failure Topology Database (FTD)
================================

SQLite-backed tracking of which CDG topologies cause agent failures.
Key insight: failures are structural — a chain (A→B→C) causes the same
failure pattern regardless of whether it's Research→Report→Presentation
or Design→Code→Test.

References:
    VERGIL_System_Design.md — Step 8 (FTD)
    VERGIL_Phase1_Phase2_Implementation.md — P2.5
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger('vergil.failure_db')


@dataclass
class FailurePattern:
    """One entry in the failure topology database."""
    pattern_id: str
    topology_hash: str
    n_nodes: int = 0
    n_temporal_edges: int = 0
    n_resource_conflicts: int = 0
    max_chain_depth: int = 0
    n_concurrent_deadlines: int = 0
    has_implicit_nodes: bool = False
    failure_type: str = ""
    failure_step: int = 0
    total_episodes: int = 0
    failure_episodes: int = 0
    failure_episode_ids: List[str] = field(default_factory=list)
    counterfactual_actions: List[str] = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.now)
    first_seen: datetime = field(default_factory=datetime.now)

    @property
    def failure_rate(self) -> float:
        return self.failure_episodes / max(1, self.total_episodes)


class FailureTopologyDatabase:
    """
    SQLite-backed FTD. No external DB required (Colab-compatible).
    """

    def __init__(self, db_path: str = '/tmp/vergil_failure_db.sqlite'):
        self.db_path = db_path
        self._init_db()
        logger.info(f"FailureTopologyDatabase: {db_path}")

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    topology_hash TEXT PRIMARY KEY,
                    n_nodes INTEGER,
                    n_temporal_edges INTEGER,
                    n_resource_conflicts INTEGER,
                    max_chain_depth INTEGER,
                    n_concurrent_deadlines INTEGER,
                    has_implicit_nodes INTEGER,
                    total_episodes INTEGER DEFAULT 0,
                    failure_episodes INTEGER DEFAULT 0,
                    dominant_failure_type TEXT,
                    first_seen TEXT,
                    last_seen TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS episode_outcomes (
                    episode_id TEXT PRIMARY KEY,
                    topology_hash TEXT,
                    curriculum_stage INTEGER,
                    total_reward REAL,
                    fulfillment_rate REAL,
                    terminal_reason TEXT,
                    failure_step INTEGER,
                    cascade_count INTEGER,
                    recorded_at TEXT,
                    FOREIGN KEY (topology_hash) REFERENCES patterns(topology_hash)
                )
            ''')
            conn.commit()

    def record_episode(self, episode_record, topology_hash: str,
                       structural_features: Dict) -> None:
        """Record episode outcome and update pattern statistics."""
        is_failure = (
            episode_record.terminal_reason in ('trust_collapse',) or
            episode_record.commitment_fulfillment_rate < 0.5
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO patterns
                    (topology_hash, n_nodes, n_temporal_edges, n_resource_conflicts,
                     max_chain_depth, n_concurrent_deadlines, has_implicit_nodes,
                     total_episodes, failure_episodes, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(topology_hash) DO UPDATE SET
                    total_episodes = total_episodes + 1,
                    failure_episodes = failure_episodes + ?,
                    last_seen = ?
            ''', (
                topology_hash,
                structural_features.get('n_nodes', 0),
                structural_features.get('n_temporal_edges', 0),
                structural_features.get('n_resource_conflicts', 0),
                structural_features.get('max_chain_depth', 0),
                structural_features.get('n_concurrent_deadlines', 0),
                int(structural_features.get('has_implicit_nodes', False)),
                int(is_failure),
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                int(is_failure),
                datetime.now().isoformat(),
            ))

            conn.execute('''
                INSERT OR REPLACE INTO episode_outcomes
                    (episode_id, topology_hash, curriculum_stage, total_reward,
                     fulfillment_rate, terminal_reason, failure_step, cascade_count, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                episode_record.episode_id,
                topology_hash,
                episode_record.curriculum_stage,
                round(episode_record.total_reward, 4),
                round(episode_record.commitment_fulfillment_rate, 4),
                episode_record.terminal_reason,
                episode_record.total_steps,
                len(episode_record.cascade_events),
                datetime.now().isoformat(),
            ))
            conn.commit()

    def get_high_failure_patterns(self, stage: int, top_k: int = 5,
                                  min_episodes: int = 3) -> List[Dict]:
        """Return top-K topology hashes with highest failure rates."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT p.topology_hash,
                       p.failure_episodes * 1.0 / p.total_episodes as failure_rate,
                       p.n_nodes, p.max_chain_depth, p.n_resource_conflicts
                FROM patterns p
                WHERE p.total_episodes >= ?
                ORDER BY failure_rate DESC
                LIMIT ?
            ''', (min_episodes, top_k))

            rows = cursor.fetchall()
            return [
                {'topology_hash': r[0], 'failure_rate': round(r[1], 4),
                 'n_nodes': r[2], 'max_chain_depth': r[3], 'n_resource_conflicts': r[4]}
                for r in rows
            ]

    def get_curriculum_sampling_weights(self, stage: int) -> Dict[str, float]:
        """Return {topology_hash: sampling_weight} for curriculum generation."""
        high_failure = self.get_high_failure_patterns(stage, top_k=5)
        return {p['topology_hash']: p['failure_rate'] * 2.0 for p in high_failure}

    def get_statistics(self) -> Dict:
        """Summary statistics for monitoring."""
        with sqlite3.connect(self.db_path) as conn:
            total_eps = conn.execute('SELECT COUNT(*) FROM episode_outcomes').fetchone()[0]
            total_patterns = conn.execute('SELECT COUNT(*) FROM patterns').fetchone()[0]
            avg_fail_rate = conn.execute(
                'SELECT AVG(failure_episodes * 1.0 / total_episodes) '
                'FROM patterns WHERE total_episodes >= 3'
            ).fetchone()[0] or 0.0

        return {
            'total_episodes': total_eps,
            'unique_topology_patterns': total_patterns,
            'average_failure_rate': round(avg_fail_rate, 4),
        }
