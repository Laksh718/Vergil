# vergil/core/cdg.py
"""
Commitment Dependency Graph Engine
===================================

The computational heart of VERGIL. Maintains the DAG structure,
evaluates satisfiability (Temporal CSP), and propagates failure cascades.

Design for RL speed:
- Greedy topological sort for satisfiability (not OR-Tools — too slow per step)
- BFS cascade propagation with depth tracking
- All state mutations through named methods (for logging and replay)
- Full serialization at any point for CDG replay buffer

References:
    VERGIL_System_Design.md — Step 3 (CDG Design), Step 9 (step() logic)
    VERGIL_Phase1_Phase2_Implementation.md — P1.3
"""

import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import hashlib
import json
import logging

from .types import (
    CommitmentNode, CDGEdge, CommitmentStatus, EdgeType,
    ResourceType, FailureType
)

logger = logging.getLogger('vergil.cdg')


@dataclass
class CDGSatisfiabilityResult:
    """Result of satisfiability check."""
    is_satisfiable: bool = True
    satisfiability_score: float = 1.0      # 0 = definitely infeasible; 1 = definitely feasible
    violations: List[Dict] = field(default_factory=list)
    at_risk_nodes: List[str] = field(default_factory=list)
    bottleneck_resources: List[str] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    slack_by_node: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'is_satisfiable': self.is_satisfiable,
            'satisfiability_score': round(self.satisfiability_score, 4),
            'n_violations': len(self.violations),
            'at_risk_nodes': self.at_risk_nodes,
            'bottleneck_resources': self.bottleneck_resources,
        }


@dataclass
class CascadeEvent:
    """Record of a single cascade propagation step."""
    node_id: str
    node_label: str
    depth: int
    old_risk: float
    new_risk: float
    cascaded: bool  # Did this node also fail?
    timestamp: str


class CommitmentDependencyGraph:
    """
    Core CDG data structure and operations.

    Internal representation: networkx DiGraph for T/logical edges.
    Resource conflicts stored separately (bidirectional, not in DiGraph).

    Invariants maintained:
    - No cycles in temporal/logical edges (enforced on every add_edge)
    - Node IDs are unique
    - Every edge references existing nodes
    """

    def __init__(self, graph_id: str = ""):
        self.graph_id = graph_id or f"cdg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Primary graph (temporal and logical edges only — maintains DAG property)
        self._graph: nx.DiGraph = nx.DiGraph()

        # Node storage
        self._nodes: Dict[str, CommitmentNode] = {}

        # Edge storage (all edges including resource conflicts)
        self._edges: Dict[str, CDGEdge] = {}

        # Resource conflict index (separate from directed graph)
        self._resource_conflicts: Dict[Tuple[str, str], CDGEdge] = {}

        # Risk scores (updated by evaluate_satisfiability)
        self._node_risk_scores: Dict[str, float] = {}

        # Cascade history
        self._cascade_events: List[Dict] = []

        logger.info(f"CDG initialized: {self.graph_id}")

    # ── Node Operations ─────────────────────────────────────────────────────

    def add_node(self, node: CommitmentNode) -> bool:
        """
        Add a commitment node to the CDG.
        Returns True if added successfully, False if node_id already exists.
        """
        if node.node_id in self._nodes:
            logger.warning(f"add_node: node {node.node_id} already exists")
            return False

        self._nodes[node.node_id] = node
        self._graph.add_node(node.node_id, commitment=node)
        self._node_risk_scores[node.node_id] = 0.0

        logger.debug(f"add_node: {node.node_id} '{node.label}' type={node.commitment_type}")
        return True

    def get_node(self, node_id: str) -> Optional[CommitmentNode]:
        return self._nodes.get(node_id)

    def update_node_status(self, node_id: str, new_status: CommitmentStatus,
                           current_time: datetime) -> None:
        """Update status with timestamp tracking and urgency recalculation."""
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} not found in CDG")

        node = self._nodes[node_id]
        old_status = node.status
        node.status = new_status
        node.updated_at = current_time

        if new_status in (CommitmentStatus.COMPLETED, CommitmentStatus.LATE_COMPLETED):
            node.completion_timestamp = current_time
            node.progress_pct = 100.0

        logger.debug(f"update_node_status: {node_id} {old_status} → {new_status}")

    def get_active_nodes(self) -> List[CommitmentNode]:
        """Nodes that are accepted and not yet resolved."""
        return [n for n in self._nodes.values()
                if n.status in (CommitmentStatus.ACCEPTED, CommitmentStatus.IN_PROGRESS,
                                CommitmentStatus.AT_RISK, CommitmentStatus.RENEGOTIATED)]

    def get_nodes_by_status(self, status: CommitmentStatus) -> List[CommitmentNode]:
        return [n for n in self._nodes.values() if n.status == status]

    def get_pending_nodes(self) -> List[CommitmentNode]:
        """Nodes awaiting agent decision."""
        return [n for n in self._nodes.values()
                if n.status == CommitmentStatus.PENDING]

    @property
    def all_nodes(self) -> List[CommitmentNode]:
        return list(self._nodes.values())

    @property
    def all_edges(self) -> List[CDGEdge]:
        return list(self._edges.values())

    # ── Edge Operations ─────────────────────────────────────────────────────

    def add_edge(self, edge: CDGEdge) -> Tuple[bool, Optional[str]]:
        """
        Add an edge to the CDG.
        For TEMPORAL and LOGICAL edges: checks for cycle creation before adding.
        For RESOURCE edges: stored in conflict index; does not affect DAG property.
        """
        if edge.from_node not in self._nodes:
            return False, f"from_node {edge.from_node} not in CDG"
        if edge.to_node not in self._nodes:
            return False, f"to_node {edge.to_node} not in CDG"

        if edge.edge_type in (EdgeType.TEMPORAL, EdgeType.LOGICAL,
                              EdgeType.IMPLICIT_DERIVE, EdgeType.TRUST_IMPACT):
            # Cycle check: would adding this edge create a cycle?
            self._graph.add_edge(edge.from_node, edge.to_node)
            if not nx.is_directed_acyclic_graph(self._graph):
                self._graph.remove_edge(edge.from_node, edge.to_node)
                return False, f"Adding edge {edge.from_node}→{edge.to_node} creates cycle"

        elif edge.edge_type == EdgeType.RESOURCE:
            key = tuple(sorted([edge.from_node, edge.to_node]))
            self._resource_conflicts[key] = edge

        self._edges[edge.edge_id] = edge
        logger.debug(f"add_edge: {edge.from_node} →[{edge.edge_type}]→ {edge.to_node}")
        return True, None

    def get_temporal_predecessors(self, node_id: str) -> List[str]:
        """Nodes that must complete before node_id can start."""
        if node_id not in self._graph:
            return []
        return list(self._graph.predecessors(node_id))

    def get_temporal_successors(self, node_id: str) -> List[str]:
        """Nodes that are blocked until node_id completes."""
        if node_id not in self._graph:
            return []
        return list(self._graph.successors(node_id))

    def get_resource_conflicts(self, node_id: str) -> List[CDGEdge]:
        """All resource conflict edges involving this node."""
        conflicts = []
        for (a, b), edge in self._resource_conflicts.items():
            if node_id in (a, b):
                conflicts.append(edge)
        return conflicts

    # ── Satisfiability Engine ───────────────────────────────────────────────

    def evaluate_satisfiability(self, current_time: datetime,
                                available_hours: float = 16.0,
                                risk_threshold: float = 0.6
                                ) -> CDGSatisfiabilityResult:
        """
        Evaluate whether the CDG is satisfiable given current constraints.

        Algorithm (greedy Temporal CSP — fast enough for RL step loop):
        1. Topological sort of temporal edges → ordering
        2. For each node in order: compute earliest possible start
        3. Check if earliest_possible_completion <= deadline
        4. Check resource conflicts: overlapping time assignments
        5. Compute slack per node; nodes with slack < 0 are infeasible
        6. Compute overall satisfiability score
        """
        result = CDGSatisfiabilityResult()
        active_nodes = self.get_active_nodes()

        if not active_nodes:
            result.satisfiability_score = 1.0
            return result

        # Step 1: Topological order
        try:
            topo_order = list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            result.is_satisfiable = False
            result.satisfiability_score = 0.0
            result.violations.append({'type': 'cycle_detected', 'severity': 1.0})
            return result

        # Step 2: Earliest start computation (forward pass)
        earliest_start: Dict[str, datetime] = {}

        for nid in topo_order:
            node = self._nodes.get(nid)
            if node is None or node.status not in (
                    CommitmentStatus.ACCEPTED, CommitmentStatus.IN_PROGRESS,
                    CommitmentStatus.AT_RISK, CommitmentStatus.RENEGOTIATED):
                continue

            preds = self.get_temporal_predecessors(nid)
            if not preds:
                earliest_start[nid] = node.earliest_start or current_time
            else:
                pred_ends = []
                for pred_id in preds:
                    pred = self._nodes.get(pred_id)
                    edge = self._get_temporal_edge(pred_id, nid)
                    lag = timedelta(hours=edge.lag_hours) if edge else timedelta(0)
                    if pred_id in earliest_start and pred:
                        pred_end = earliest_start[pred_id] + timedelta(
                            hours=pred.estimated_duration_hours)
                        pred_ends.append(pred_end + lag)

                earliest_start[nid] = max(pred_ends) if pred_ends else current_time

        # Step 3: Slack computation and violation detection
        total_slack = 0.0
        n_nodes_checked = 0

        for nid in topo_order:
            node = self._nodes.get(nid)
            if node is None or node.deadline is None:
                continue
            if not node.is_active():
                continue

            es = earliest_start.get(nid, current_time)
            expected_completion = es + timedelta(hours=node.estimated_duration_hours)
            slack_hours = (node.deadline - expected_completion).total_seconds() / 3600

            result.slack_by_node[nid] = slack_hours
            total_slack += max(0, slack_hours)
            n_nodes_checked += 1

            # Risk score: nonlinear function of slack
            if slack_hours < 0:
                risk = 1.0
                result.violations.append({
                    'type': 'deadline_infeasible',
                    'node_id': nid,
                    'node_label': node.label,
                    'slack_hours': round(slack_hours, 2),
                    'severity': min(1.0, abs(slack_hours) / 24),
                })
            elif slack_hours < 2:
                risk = 0.85
            elif slack_hours < 8:
                risk = 0.40 + 0.45 * (1 - slack_hours / 8)
            elif slack_hours < 24:
                risk = 0.10 + 0.30 * (1 - slack_hours / 24)
            else:
                risk = 0.05

            self._node_risk_scores[nid] = risk
            if risk > risk_threshold:
                result.at_risk_nodes.append(nid)

        # Step 4: Resource conflict check
        for (a_id, b_id), edge in self._resource_conflicts.items():
            node_a = self._nodes.get(a_id)
            node_b = self._nodes.get(b_id)
            if node_a is None or node_b is None:
                continue

            a_start = earliest_start.get(a_id)
            b_start = earliest_start.get(b_id)
            if a_start and b_start:
                a_end = a_start + timedelta(hours=node_a.estimated_duration_hours)
                b_end = b_start + timedelta(hours=node_b.estimated_duration_hours)

                overlap = (min(a_end, b_end) - max(a_start, b_start)).total_seconds() / 3600
                if overlap > 0:
                    result.violations.append({
                        'type': 'resource_conflict',
                        'node_a': a_id, 'node_b': b_id,
                        'resource': edge.resource_type.value if edge.resource_type else 'time',
                        'overlap_hours': round(overlap, 2),
                        'severity': edge.conflict_severity,
                    })
                    self._node_risk_scores[a_id] = min(1.0,
                        self._node_risk_scores.get(a_id, 0) + 0.3 * edge.conflict_severity)
                    self._node_risk_scores[b_id] = min(1.0,
                        self._node_risk_scores.get(b_id, 0) + 0.3 * edge.conflict_severity)

        # Step 5: Total capacity check (Class 1 edge case)
        total_required = sum(
            n.estimated_duration_hours for n in active_nodes
            if result.slack_by_node.get(n.node_id, float('inf')) < 48
        )
        if total_required > available_hours * 1.1:
            result.violations.append({
                'type': 'capacity_exceeded',
                'required_hours': round(total_required, 2),
                'available_hours': round(available_hours, 2),
                'overflow_hours': round(total_required - available_hours, 2),
                'severity': min(1.0, (total_required - available_hours) / max(available_hours, 1)),
            })

        # Step 6: Critical path
        if len(self._graph.nodes) > 0 and nx.is_directed_acyclic_graph(self._graph):
            try:
                result.critical_path = nx.dag_longest_path(self._graph)
            except Exception:
                result.critical_path = []

        # Step 7: Overall satisfiability score
        # Multiple scoring signals combined for robustness:
        n_violations = len(result.violations)
        severity_sum = sum(v.get('severity', 0.5) for v in result.violations)
        max_severity = max((v.get('severity', 0.5) for v in result.violations), default=0.0)
        result.is_satisfiable = n_violations == 0

        # Score based on violations
        violation_score = max(0.0, 1.0 - severity_sum)

        # Score based on risk: fraction of nodes at risk
        n_active = len(active_nodes)
        risk_score = 1.0 - (len(result.at_risk_nodes) / max(1, n_active))

        # Combined: use the LOWER of the two (conservative)
        result.satisfiability_score = max(0.0, min(violation_score, risk_score))

        logger.debug(f"CDG satisfiability: score={result.satisfiability_score:.3f}, "
                     f"violations={n_violations}, at_risk={len(result.at_risk_nodes)}")
        return result

    # ── Failure Propagation ─────────────────────────────────────────────────

    def propagate_failure(self, failed_node_id: str, current_time: datetime,
                          propagation_factor: float = 0.8) -> List[Dict]:
        """
        When a commitment fails, propagate risk to dependent nodes.

        Algorithm (Step 3 cascade propagation):
        - BFS from failed node through temporal/logical successor edges
        - Each hop: risk += failed_node.urgency * edge.failure_propagation_factor^depth
        - Nodes that exceed risk_threshold=0.85 → status FAILED (cascade)
        - Records cascade event for curriculum analysis
        """
        cascade_events = []

        if failed_node_id not in self._nodes:
            return cascade_events

        failed_node = self._nodes[failed_node_id]
        self.update_node_status(failed_node_id, CommitmentStatus.FAILED, current_time)

        # BFS through successors
        visited = {failed_node_id}
        queue = [(nid, 1) for nid in self.get_temporal_successors(failed_node_id)]

        while queue:
            nid, depth = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)

            node = self._nodes.get(nid)
            if node is None:
                continue

            old_risk = self._node_risk_scores.get(nid, 0.0)
            risk_increment = failed_node.urgency * (propagation_factor ** depth)
            new_risk = min(1.0, old_risk + risk_increment)
            self._node_risk_scores[nid] = new_risk

            cascaded = False
            if new_risk > 0.85 and node.status in (CommitmentStatus.ACCEPTED,
                                                     CommitmentStatus.IN_PROGRESS):
                self.update_node_status(nid, CommitmentStatus.FAILED, current_time)
                cascaded = True
                for successor in self.get_temporal_successors(nid):
                    if successor not in visited:
                        queue.append((successor, depth + 1))

            event = {
                'node_id': nid,
                'node_label': node.label,
                'depth': depth,
                'old_risk': round(old_risk, 3),
                'new_risk': round(new_risk, 3),
                'cascaded': cascaded,
                'timestamp': current_time.isoformat(),
            }
            cascade_events.append(event)
            logger.info(f"Cascade: {nid} risk {old_risk:.2f}→{new_risk:.2f} "
                        f"depth={depth} cascaded={cascaded}")

        self._cascade_events.extend(cascade_events)
        return cascade_events

    # ── Topology Signature ──────────────────────────────────────────────────

    def topology_hash(self) -> str:
        """
        Stable hash of the CDG topology (node types + edge pattern).
        Used by curriculum engine to identify failure patterns.
        """
        nodes_sig = sorted([
            (n.commitment_type.value, len(self.get_temporal_successors(n.node_id)))
            for n in self._nodes.values()
        ])
        edges_sig = sorted([
            (e.edge_type.value, e.conflict_severity)
            for e in self._edges.values()
        ])
        sig = json.dumps({'nodes': nodes_sig, 'edges': edges_sig}, sort_keys=True)
        return hashlib.md5(sig.encode()).hexdigest()[:12]

    # ── Serialization ───────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Full serialization for logging and replay."""
        return {
            'graph_id': self.graph_id,
            'nodes': {nid: {
                'node_id': n.node_id,
                'label': n.label,
                'type': n.commitment_type.value,
                'status': n.status.value,
                'deadline': n.deadline.isoformat() if n.deadline else None,
                'risk_score': round(self._node_risk_scores.get(nid, 0.0), 4),
                'urgency': round(n.urgency, 4),
                'estimated_duration_hours': n.estimated_duration_hours,
            } for nid, n in self._nodes.items()},
            'edges': [{'from': e.from_node, 'to': e.to_node,
                       'type': e.edge_type.value, 'weight': e.weight}
                      for e in self._edges.values()],
            'topology_hash': self.topology_hash(),
            'n_nodes': len(self._nodes),
            'n_edges': len(self._edges),
            'n_resource_conflicts': len(self._resource_conflicts),
        }

    def _get_temporal_edge(self, from_id: str, to_id: str) -> Optional[CDGEdge]:
        for edge in self._edges.values():
            if (edge.from_node == from_id and edge.to_node == to_id and
                    edge.edge_type in (EdgeType.TEMPORAL, EdgeType.LOGICAL,
                                       EdgeType.IMPLICIT_DERIVE)):
                return edge
        return None
