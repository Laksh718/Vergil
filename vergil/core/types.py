# vergil/core/types.py
"""
VERGIL Core Type System
=======================

Every downstream module depends on these types. Defines all enums,
dataclasses, and type aliases for the VERGIL system.

Design Decisions:
- Uses stdlib dataclasses (not Pydantic) for RL speed — validation happens
  at episode boundaries, not on every step() call.
- Unified CommitmentNode (not subclassed per type) — simpler for graph operations.
- Hidden fields prefixed with _ — convention for POMDP partial observability.
- All datetime fields use stdlib datetime (not arrow/pendulum) for Colab compat.

References:
    VERGIL_System_Design.md — Steps 2, 3, 4, 5
    VERGIL_Phase1_Phase2_Implementation.md — P1.1
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, FrozenSet
import uuid


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class CommitmentType(str, Enum):
    """Node types by explicitness (Step 3 — By Explicitness)."""
    EXPLICIT_HARD   = "explicit_hard"    # Contractual; deadline is non-negotiable
    EXPLICIT_SOFT   = "explicit_soft"    # Acknowledged; parameters negotiable
    IMPLICIT        = "implicit"         # Inferred from an accepted explicit commit
    PRECONDITION    = "precondition"     # Must complete before another is possible
    SOCIAL          = "social"           # Relational; no hard deliverable


class CommitmentStatus(str, Enum):
    """Lifecycle states for a commitment node."""
    PENDING         = "pending"          # Not yet decided
    ACCEPTED        = "accepted"         # In CDG; being tracked
    DECLINED        = "declined"         # Rejected; archived
    IN_PROGRESS     = "in_progress"      # Execution started
    COMPLETED       = "completed"        # Done; within deadline
    LATE_COMPLETED  = "late_completed"   # Done; past deadline
    FAILED          = "failed"           # Deadline passed; not done
    AT_RISK         = "at_risk"          # May fail; cascade warning
    INFEASIBLE      = "infeasible"       # Cannot be completed given constraints
    RENEGOTIATED    = "renegotiated"     # Parameters modified post-acceptance
    DELEGATED       = "delegated"        # Routed to someone else


class EdgeType(str, Enum):
    """Edge types in the CDG (Step 3 — Edge Taxonomy)."""
    TEMPORAL        = "temporal"         # A must complete before B starts (T-edge)
    RESOURCE        = "resource"         # A and B conflict over same resource (R-edge, bidirectional)
    LOGICAL         = "logical"          # B's existence depends on A's output
    IMPLICIT_DERIVE = "implicit_derive"  # Explicit → auto-extracted implicit (IE-edge)
    TRUST_IMPACT    = "trust_impact"     # Breaking A affects actions toward stakeholder (TR-edge)


class ResourceType(str, Enum):
    """Resource types that commitments consume (Step 3 — By Resource Type)."""
    TIME_BLOCK      = "time_block"       # Occupy specific calendar slots
    COGNITIVE_LOAD  = "cognitive_load"   # Require focus; constrain adjacent time quality
    PHYSICAL_PRESENCE = "physical_presence"  # Physical/virtual presence required
    COLLABORATOR    = "collaborator"     # Requires another person
    EXTERNAL_TOOL   = "external_tool"    # Requires a specific tool or system
    DELIVERABLE     = "deliverable"      # Produces an artifact; requires resource inputs


class ActionType(str, Enum):
    """Agent action types (Step 5 — Primary Action Types)."""
    ACCEPT          = "accept"           # Add commitment to CDG as-is
    DECLINE         = "decline"          # Reject; send explanation
    COUNTER_PROPOSE = "counter_propose"  # Modify parameters, reoffer
    RENEGOTIATE     = "renegotiate"      # Modify existing CDG node
    CLARIFY         = "clarify"          # Ask for missing commitment details
    DEFER           = "defer"            # Request more time to decide
    DELEGATE        = "delegate"         # Route to someone else
    DO_NOTHING      = "do_nothing"       # No response (risky; often implicit acceptance)


class StakeholderRole(str, Enum):
    """Stakeholder types (Step 2 — Stakeholder Taxonomy)."""
    BOSS            = "boss"             # Tier 1: High-trust, high-stakes authority
    CLIENT          = "client"           # Tier 1: High-stakes external
    COLLEAGUE       = "colleague"        # Tier 2: Reciprocal, medium-stakes
    FRIEND          = "friend"           # Tier 2: Personal, flexible
    SYSTEM          = "system"           # Tier 3: Calendar, email, task system


class FailureType(str, Enum):
    """Failure classification for FTD tracking (Step 8)."""
    CASCADE         = "cascade"          # Cascading failure through dependency chain
    RESOURCE_OVERCOMMIT = "resource_overcommit"  # Too many commitments for available resources
    TRUST_COLLAPSE  = "trust_collapse"   # Trust with all stakeholders below threshold
    IMPLICIT_MISSED = "implicit_missed"  # Failed to extract or handle implicit commitment
    SILENT_DROP     = "silent_drop"      # Accepted then dropped without renegotiation
    ESTIMATION_ERROR = "estimation_error"  # Duration misestimation caused failure
    DEADLINE_MISS   = "deadline_miss"    # Simple deadline miss (no cascade)


class ImplicitType(str, Enum):
    """Types of implicit commitments (Step 3 — Implicit Commitments)."""
    TRAVEL_BUFFER   = "travel_buffer"    # Commute time before/after an event
    COGNITIVE_PREP  = "cognitive_prep"   # Preparation time before a commitment
    FOLLOWUP        = "followup"         # Follow-up action after a commitment
    PRECONDITION    = "precondition"     # Must-do before a commitment is possible
    RESOURCE_SETUP  = "resource_setup"   # Set up tools/environment


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ImplicitCommitmentSpec:
    """Specification for an implicit commitment to be instantiated."""
    implicit_type: str          # "travel_buffer" | "cognitive_prep" | "followup" | "precondition"
    description: str
    time_before_parent: Optional[timedelta] = None  # How long before parent commit
    time_after_parent: Optional[timedelta] = None
    duration_hours: float = 0.5
    auto_accept: bool = True    # Should be added to CDG automatically on parent acceptance


@dataclass
class ExtractionResult:
    """
    Output of the NL → commitment pipeline.
    Includes confidence because extraction is imperfect and ambiguous.

    Design: Never return binary yes/no. Always return probability.
    Flag ambiguity explicitly; don't silently resolve it.
    """
    raw_text: str                            # Original message snippet
    is_commitment: bool                      # Is this even a commitment?
    commitment_probability: float            # P(commitment | text), [0,1]

    # Core commitment fields (None if commitment_probability < threshold)
    commitment_type: Optional[CommitmentType] = None
    deliverable: Optional[str] = None        # What needs to be produced/done
    deadline_estimate: Optional[datetime] = None
    deadline_confidence: float = 0.0         # How sure are we about the deadline
    deadline_range: Optional[Tuple[datetime, datetime]] = None  # [min, max] range

    # Extracted resource requirements
    estimated_duration_hours: float = 0.0
    duration_uncertainty: float = 0.5        # σ of duration estimate (in hours)
    resources_required: List[ResourceType] = field(default_factory=list)

    # Ambiguity flags — CRITICAL for downstream handling
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None   # "deadline_unclear" | "scope_unclear" | etc.
    requires_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)

    # Implicit commitments detected
    implicit_commitments: List[ImplicitCommitmentSpec] = field(default_factory=list)

    # Extraction metadata
    extraction_model: str = "rule_based_v1"  # Which extractor was used
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    raw_llm_response: Optional[str] = None   # For debugging


# ═══════════════════════════════════════════════════════════════════════════════
# COMMITMENT NODE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CommitmentNode:
    """
    A single node in the Commitment Dependency Graph.
    The fundamental unit of the VERGIL system.

    Unified design: all commitment types share the same class.
    Type-specific behavior is determined by `commitment_type` field.
    """
    # Identity
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_commitment_id: Optional[str] = None  # For implicit commits

    # Semantic content
    label: str = ""                         # Short human-readable label
    description: str = ""                   # Full commitment description
    deliverable: str = ""                   # Concrete output expected
    commitment_type: CommitmentType = CommitmentType.EXPLICIT_SOFT
    status: CommitmentStatus = CommitmentStatus.PENDING

    # Stakeholder
    stakeholder_id: str = ""                # Who made this request
    stakeholder_role: StakeholderRole = StakeholderRole.COLLEAGUE

    # Temporal parameters — UNCERTAIN by design
    deadline: Optional[datetime] = None     # Point deadline (if known)
    deadline_range: Optional[Tuple[datetime, datetime]] = None  # [earliest, latest]
    deadline_confidence: float = 0.5        # How confident is our deadline estimate
    earliest_start: Optional[datetime] = None

    # Resource parameters — UNCERTAIN by design
    estimated_duration_hours: float = 2.0
    duration_std_hours: float = 1.0         # Standard deviation (uncertainty)
    resources: List[ResourceType] = field(default_factory=list)
    cognitive_load_score: float = 0.5       # 0=trivial, 1=fully consuming

    # CDG-level properties (computed dynamically)
    urgency: float = 0.5                    # Updates over time as deadline approaches
    risk_score: float = 0.0                 # P(failure) given current state
    cascade_potential: float = 0.0          # How many nodes fail if this one fails

    # Decision tracking
    decision_made: Optional[ActionType] = None
    decision_timestamp: Optional[datetime] = None
    decision_rationale: str = ""            # Why the agent chose this action
    counter_proposal: Optional[Dict] = None
    renegotiation_count: int = 0            # How many times renegotiated

    # Execution tracking
    progress_pct: float = 0.0              # 0–100; updated by simulation
    actual_duration_hours: Optional[float] = None  # Revealed at completion
    completion_timestamp: Optional[datetime] = None

    # Trust context at decision time
    trust_at_decision: float = 0.5
    action_space_at_decision: List[ActionType] = field(default_factory=list)

    # Metadata
    source_message_id: str = ""
    extraction_result: Optional[ExtractionResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def is_overdue(self, current_time: datetime) -> bool:
        if self.deadline is None:
            return False
        return current_time > self.deadline and self.status not in (
            CommitmentStatus.COMPLETED, CommitmentStatus.DECLINED,
            CommitmentStatus.RENEGOTIATED, CommitmentStatus.DELEGATED
        )

    def deadline_proximity_hours(self, current_time: datetime) -> float:
        """Hours remaining until deadline. Negative = overdue."""
        if self.deadline is None:
            return float('inf')
        return (self.deadline - current_time).total_seconds() / 3600

    def update_urgency(self, current_time: datetime) -> None:
        """Urgency increases as deadline approaches. Nonlinear near deadline."""
        hours_left = self.deadline_proximity_hours(current_time)
        if hours_left <= 0:
            self.urgency = 1.0
        elif hours_left < 4:
            self.urgency = 0.9 + 0.1 * (1 - hours_left / 4)
        elif hours_left < 24:
            self.urgency = 0.5 + 0.4 * (1 - hours_left / 24)
        else:
            self.urgency = max(0.1, 0.5 * (1 - hours_left / (24 * 7)))

    def is_active(self) -> bool:
        """Whether this node is in an active (non-terminal) state."""
        return self.status in (
            CommitmentStatus.PENDING, CommitmentStatus.ACCEPTED,
            CommitmentStatus.IN_PROGRESS, CommitmentStatus.AT_RISK,
            CommitmentStatus.RENEGOTIATED
        )

    def is_terminal(self) -> bool:
        """Whether this node has reached a final state."""
        return self.status in (
            CommitmentStatus.COMPLETED, CommitmentStatus.LATE_COMPLETED,
            CommitmentStatus.FAILED, CommitmentStatus.DECLINED,
            CommitmentStatus.INFEASIBLE, CommitmentStatus.DELEGATED
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CDG EDGE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CDGEdge:
    """
    Directed edge in the Commitment Dependency Graph.
    T-edges: A → B means A must complete before B.
    R-edges: A ↔ B means A and B conflict over a resource (bidirectional).
    TR-edges: A → trust(s) means fulfilling/breaking A affects stakeholder s.
    IE-edges: explicit → implicit (auto-generated by CDG engine).
    """
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    from_node: str = ""                     # Node ID
    to_node: str = ""                       # Node ID
    edge_type: EdgeType = EdgeType.TEMPORAL

    # Temporal dependency params (for TEMPORAL edges)
    lag_hours: float = 0.0                  # Minimum gap after 'from' completes
    hard_ordering: bool = True              # False = soft preference, not hard constraint

    # Resource conflict params (for RESOURCE edges)
    resource_type: Optional[ResourceType] = None
    conflict_severity: float = 1.0          # 0 = minor; 1 = complete mutual exclusion

    # Trust impact params (for TRUST_IMPACT edges)
    trust_impact_positive: float = 0.08     # Trust gain on fulfillment
    trust_impact_negative: float = -0.15    # Trust loss on failure

    # Uncertainty
    weight: float = 1.0                     # Edge importance weight
    confidence: float = 1.0                 # How sure we are this edge is real

    # Cascade tracking
    failure_propagation_factor: float = 0.8  # How much failure transfers across edge

    created_at: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# STAKEHOLDER PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StakeholderProfile:
    """
    Full model of a stakeholder. Includes HIDDEN fields
    that the agent cannot observe — simulates real partial observability.

    Design: Observable fields are returned by get_observable_fields().
    Hidden fields (prefixed with _) are only used by the environment oracle.
    """
    stakeholder_id: str
    name: str
    role: StakeholderRole

    # Agent-observable fields
    relationship_weight: float = 0.5       # How important this relationship is
    domain: str = "professional"           # "professional" | "personal" | "hybrid"
    communication_style: str = "direct"    # "direct" | "indirect" | "pressure"

    # HIDDEN fields (oracle/environment only; agent cannot directly observe)
    _true_urgency: float = 0.5             # Their actual urgency level
    _deadline_flexibility_hours: float = 0.0  # Hidden slack in their deadline
    _forgiveness_rate: float = 0.5         # How quickly trust recovers
    _manipulation_tactics: List[str] = field(default_factory=list)  # "guilt" | "authority" | "flattery"
    _irrational_probability: float = 0.1   # P(behave irrationally in this episode)

    # Trust decay/repair rates (partially observable — agent infers over time)
    trust_decay_rate: float = 0.15         # Per broken commitment (agent can estimate)
    trust_repair_rate: float = 0.08        # Per kept commitment (agent can estimate)

    # Behavior parameters
    escalation_threshold: float = 0.4      # If trust < this, stakeholder escalates
    renegotiation_tolerance: int = 2       # Max renegotiations before trust penalty

    # Personality subtype (Phase 2 — sampled per episode for anti-overfitting)
    personality: str = "default"           # "type_a" | "collaborative" | "anxious" (Boss subtypes)

    def get_observable_fields(self) -> Dict:
        """Return only fields the agent is allowed to see."""
        return {
            'stakeholder_id': self.stakeholder_id,
            'name': self.name,
            'role': self.role.value,
            'relationship_weight': self.relationship_weight,
            'domain': self.domain,
            'communication_style': self.communication_style,
            'trust_decay_rate': self.trust_decay_rate,
            'trust_repair_rate': self.trust_repair_rate,
            'escalation_threshold': self.escalation_threshold,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TRUST ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrustEntry:
    """
    Trust record for one stakeholder relationship.
    Non-Markovian: maintains full history for credit assignment.

    Phase 1: scalar trust.
    Phase 2 extends to multi-dimensional (reliability/warmth/competence).
    """
    stakeholder_id: str
    trust_score: float = 0.65              # Current scalar [0, 1]

    # Trust history (non-Markovian memory)
    history: List[Dict] = field(default_factory=list)
    # Each history entry: {'timestamp': dt, 'event': str, 'delta': float, 'step': int}

    # Derived metrics
    kept_commitments: int = 0
    broken_commitments: int = 0
    proactive_renegotiations: int = 0
    reactive_renegotiations: int = 0
    renegotiation_count_this_period: int = 0

    # Action availability (set by TrustGate in Phase 2; baseline here)
    locked_actions: List[ActionType] = field(default_factory=list)

    # Internal tracking field for reward weighting
    _role: str = "colleague"

    def update(self, delta: float, event: str, step: int) -> None:
        self.trust_score = max(0.0, min(1.0, self.trust_score + delta))
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'delta': round(delta, 4),
            'new_score': round(self.trust_score, 4),
            'step': step,
        })
        if delta < 0:
            self.broken_commitments += 1
        elif delta > 0:
            self.kept_commitments += 1

    def rolling_trust_delta(self, last_n: int = 5) -> float:
        """Net trust change over last N events. Signal for trend detection."""
        recent = self.history[-last_n:]
        return sum(e['delta'] for e in recent)


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    """
    A message from a stakeholder to the agent.
    May or may not contain a commitment.
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_id: str = ""
    sender_role: StakeholderRole = StakeholderRole.COLLEAGUE
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    delivery_time: Optional[datetime] = None   # When it should appear in agent's inbox

    # Metadata for simulation
    contains_commitment: bool = False       # Pre-labeled for scenario generation
    urgency_signal: float = 0.5             # Surface-level urgency in message tone
    pressure_tactic: Optional[str] = None   # Hidden: "guilt" | "authority" | None

    processed: bool = False
    extraction_result: Optional[ExtractionResult] = None


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT ACTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentAction:
    """
    Structured action from the agent. Post-processed into natural language
    via template system for evaluation output.

    Design: Structured actions enable clean reward assignment (for training).
    Natural language output is what humans see (for evaluation).
    """
    action_type: ActionType
    target_node_id: Optional[str] = None       # Which commitment this applies to
    target_message_id: Optional[str] = None

    # Structured parameters for different action types
    # For COUNTER_PROPOSE / RENEGOTIATE:
    proposed_deadline: Optional[datetime] = None
    proposed_scope_reduction: Optional[str] = None
    proposed_resource_offload: Optional[str] = None
    proposed_delegation_target: Optional[str] = None

    # For all actions:
    rationale: str = ""                     # Agent's stated reason (for explainability)
    confidence: float = 0.5                 # Agent's confidence in this decision

    # Feasibility assessment (agent's prediction)
    feasibility_prediction: float = 0.5     # Agent's P(feasible) estimate
    predicted_trust_impact: Dict[str, float] = field(default_factory=dict)

    # Timing
    decision_time: datetime = field(default_factory=datetime.now)
    response_latency_ms: float = 0.0       # How long agent took to decide


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VERGILState:
    """
    Complete observable state. Hidden state lives in VERGILEnv._hidden.
    This is what gets passed to the agent.

    Step 4 of design doc — observable portion only.
    """
    # CDG
    cdg_nodes: List[CommitmentNode] = field(default_factory=list)
    cdg_edges: List[CDGEdge] = field(default_factory=list)
    satisfiability_score: float = 1.0      # P(CDG is satisfiable) [0,1]

    # Time
    current_time: datetime = field(default_factory=datetime.now)
    time_horizon: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=14))
    available_hours_next_48h: float = 16.0

    # Resources
    cognitive_load: float = 0.0
    energy_level: float = 1.0
    calendar_blocks: List[Dict] = field(default_factory=list)  # [{start, end, label}]

    # Messages
    pending_messages: List[Message] = field(default_factory=list)

    # Trust (observable portion)
    trust_entries: Dict[str, TrustEntry] = field(default_factory=dict)

    # History
    decision_log: List[Dict] = field(default_factory=list)
    # Each: {'step': int, 'action': AgentAction, 'reward': float, 'node_id': str}

    # Episode metadata
    episode_id: str = ""
    step_number: int = 0
    curriculum_stage: int = 1

    # Per-node risk scores (CDG engine output)
    node_risk_scores: Dict[str, float] = field(default_factory=dict)
    at_risk_nodes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serializable representation for logging."""
        return {
            'episode_id': self.episode_id,
            'step_number': self.step_number,
            'current_time': self.current_time.isoformat(),
            'satisfiability_score': round(self.satisfiability_score, 4),
            'cognitive_load': round(self.cognitive_load, 3),
            'n_active_commitments': len([n for n in self.cdg_nodes
                                         if n.status == CommitmentStatus.ACCEPTED]),
            'n_pending_messages': len(self.pending_messages),
            'trust_scores': {sid: round(te.trust_score, 4)
                             for sid, te in self.trust_entries.items()},
            'at_risk_nodes': self.at_risk_nodes,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EPISODE RECORD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeRecord:
    """Full record of an episode. Written to disk for replay and curriculum."""
    episode_id: str
    curriculum_stage: int
    scenario_id: str
    start_time: datetime
    end_time: Optional[datetime] = None

    total_steps: int = 0
    total_reward: float = 0.0
    reward_components: Dict[str, float] = field(default_factory=dict)

    # Outcome metrics
    commitment_fulfillment_rate: float = 0.0
    final_trust_scores: Dict[str, float] = field(default_factory=dict)
    cascade_events: List[Dict] = field(default_factory=list)
    failure_types: List[FailureType] = field(default_factory=list)

    # CDG topology signature (for curriculum failure tracking)
    cdg_topology_hash: str = ""

    # Step-level replay data
    steps: List[Dict] = field(default_factory=list)

    # Outcome
    terminal_reason: str = ""  # "all_resolved" | "trust_collapse" | "max_steps"


# ═══════════════════════════════════════════════════════════════════════════════
# TRUST-GATED ACTION THRESHOLDS (Step 5)
# ═══════════════════════════════════════════════════════════════════════════════

TRUST_ACTION_THRESHOLDS: Dict[str, Dict[ActionType, float]] = {
    # Role → {action: min_trust_required}
    StakeholderRole.BOSS.value: {
        ActionType.COUNTER_PROPOSE: 0.40,  # propose_deadline_extension
        ActionType.DECLINE: 0.30,          # declining boss feels like insubordination at low trust
        ActionType.DEFER: 0.50,            # indecision signals overwhelm at low trust
    },
    StakeholderRole.CLIENT.value: {
        ActionType.DELEGATE: 0.55,         # clients expect direct delivery at low trust
        ActionType.DEFER: 0.50,
    },
    StakeholderRole.COLLEAGUE.value: {
        ActionType.DEFER: 0.45,
    },
    StakeholderRole.FRIEND.value: {
        # Friends are generally more forgiving
    },
}


def is_action_available(action_type: ActionType, trust_score: float,
                        stakeholder_role: str) -> bool:
    """
    Check if an action is available given current trust level.
    Trust-gated actions become unavailable when trust drops below threshold.

    This is the NON-OBVIOUS design choice from Step 5: the action space
    is dynamically shaped by the trust state. Creates a competency trap —
    poor early decisions lose the very tools needed to recover.
    """
    role_thresholds = TRUST_ACTION_THRESHOLDS.get(stakeholder_role, {})
    threshold = role_thresholds.get(action_type, 0.0)
    return trust_score >= threshold
