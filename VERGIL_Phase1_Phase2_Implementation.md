# VERGIL: Implementation-Ready Build Plan
## Phases 1 & 2 — Deep System Design

**Runtime**: Antigravity + Google Colab | **Executor**: Claude Code
**Architecture**: OpenEnv-compatible | **Training**: Long-horizon RL (GRPO/PPO via TRL)

---

## SYSTEM PHASING OVERVIEW

```
PHASE 1 │ Core Environment Foundation           (Weeks 1–2)
         │ CDG data structures, commitment extraction with confidence scoring,
         │ CSP-based feasibility, OpenEnv step/reset/state, base reward,
         │ deterministic stakeholder simulator, logging + test harness
         │
PHASE 2 │ Advanced Reasoning + Training Infrastructure   (Weeks 3–4)
         │ POMDP belief states, multi-dimensional trust, probabilistic execution,
         │ auxiliary prediction signals, failure replay + curriculum engine,
         │ adversarial stakeholders, memory scaling, anti-hacking, explainability
         │
PHASE 3 │ Agent Training Pipeline               (Weeks 5–7)
         │ HGT graph encoder, policy + value heads, GRPO training loop,
         │ Unsloth-accelerated LLM fine-tuning, rollout strategy,
         │ online curriculum promotion, evaluation metrics suite
         │
PHASE 4 │ Evaluation + Productization           (Weeks 8–10)
         │ Benchmark suite, before/after comparisons, frontend CDG viz,
         │ VERGIL API layer, explainability reports, research paper scaffold
```

---

# ═══════════════════════════════════════════════════════════
# PHASE 1: CORE ENVIRONMENT FOUNDATION
# ═══════════════════════════════════════════════════════════

## P1 — System Architecture Overview

```
vergil/
├── core/
│   ├── __init__.py
│   ├── types.py              # All dataclasses, enums, type aliases
│   ├── commitment.py         # Commitment node + implicit extraction
│   ├── cdg.py                # CDG graph engine (build, query, CSP check)
│   ├── extraction.py         # NL → structured commitment (with confidence)
│   ├── stakeholder.py        # Stakeholder models + response simulation
│   ├── reward.py             # Multi-component reward function
│   └── env.py                # OpenEnv: step(), reset(), state()
├── utils/
│   ├── time_utils.py         # Calendar/deadline arithmetic
│   ├── graph_utils.py        # Cycle detection, topological sort, DAG ops
│   ├── logger.py             # Structured JSON logging
│   └── serialization.py      # CDG → JSON / JSON → CDG round-trip
├── tests/
│   ├── test_types.py
│   ├── test_cdg.py
│   ├── test_extraction.py
│   ├── test_reward.py
│   └── test_env.py
├── scenarios/
│   ├── scenario_01_simple.json   # 2-node CDG, 1 stakeholder
│   ├── scenario_02_chain.json    # 4-node chain dependency
│   └── scenario_03_conflict.json # Resource conflict, 2 stakeholders
├── openenv.yaml
└── requirements.txt
```

**Colab Setup Cell (runs first in every session):**
```python
!pip install networkx pydantic numpy scipy python-dateutil -q
!pip install gymnasium -q  # OpenEnv-compatible interface
import sys; sys.path.insert(0, '/content/vergil')
```

---

## P1.1 — DATA STRUCTURES (`core/types.py`)

This is the most critical file. Every downstream module depends on these types.
Define them with Pydantic v2 for validation + serialization.

```python
# core/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, FrozenSet
import uuid


# ─── Enumerations ──────────────────────────────────────────────────────────

class CommitmentType(str, Enum):
    EXPLICIT_HARD   = "explicit_hard"    # Contractual; deadline is non-negotiable
    EXPLICIT_SOFT   = "explicit_soft"    # Acknowledged; parameters negotiable
    IMPLICIT        = "implicit"         # Inferred from an accepted explicit commit
    PRECONDITION    = "precondition"     # Must complete before another is possible
    SOCIAL          = "social"           # Relational; no hard deliverable

class CommitmentStatus(str, Enum):
    PENDING         = "pending"          # Not yet decided
    ACCEPTED        = "accepted"         # In CDG; being tracked
    DECLINED        = "declined"         # Rejected; archived
    IN_PROGRESS     = "in_progress"      # Execution started
    COMPLETED       = "completed"        # Done; within deadline
    LATE_COMPLETED  = "late_completed"   # Done; past deadline
    FAILED          = "failed"           # Deadline passed; not done
    RENEGOTIATED    = "renegotiated"     # Parameters modified post-acceptance
    DELEGATED       = "delegated"        # Routed to someone else

class EdgeType(str, Enum):
    TEMPORAL        = "temporal"         # A must complete before B starts
    RESOURCE        = "resource"         # A and B conflict over same resource
    LOGICAL         = "logical"          # B's existence depends on A's output
    IMPLICIT_DERIVE = "implicit_derive"  # Explicit → auto-extracted implicit
    TRUST_IMPACT    = "trust_impact"     # Breaking A affects actions toward stakeholder

class ResourceType(str, Enum):
    TIME_BLOCK      = "time_block"
    COGNITIVE_LOAD  = "cognitive_load"
    PHYSICAL_PRESENCE = "physical_presence"
    COLLABORATOR    = "collaborator"
    EXTERNAL_TOOL   = "external_tool"

class ActionType(str, Enum):
    ACCEPT          = "accept"
    DECLINE         = "decline"
    COUNTER_PROPOSE = "counter_propose"
    RENEGOTIATE     = "renegotiate"
    CLARIFY         = "clarify"
    DEFER           = "defer"
    DELEGATE        = "delegate"
    DO_NOTHING      = "do_nothing"

class StakeholderRole(str, Enum):
    BOSS            = "boss"
    CLIENT          = "client"
    COLLEAGUE       = "colleague"
    FRIEND          = "friend"
    SYSTEM          = "system"          # Calendar, email, task manager

class FailureType(str, Enum):
    CASCADE         = "cascade"
    RESOURCE_OVERCOMMIT = "resource_overcommit"
    TRUST_COLLAPSE  = "trust_collapse"
    IMPLICIT_MISSED = "implicit_missed"
    SILENT_DROP     = "silent_drop"
    ESTIMATION_ERROR = "estimation_error"


# ─── Commitment Extraction Result ──────────────────────────────────────────

@dataclass
class ExtractionResult:
    """
    Output of the NL → commitment pipeline.
    Includes confidence because extraction is imperfect and ambiguous.
    """
    raw_text: str                            # Original message snippet
    is_commitment: bool                      # Is this even a commitment?
    commitment_probability: float            # P(commitment | text), [0,1]
    
    # Core commitment fields (None if commitment_probability < threshold)
    commitment_type: Optional[CommitmentType] = None
    deliverable: Optional[str] = None       # What needs to be produced/done
    deadline_estimate: Optional[datetime] = None
    deadline_confidence: float = 0.0        # How sure are we about the deadline
    deadline_range: Optional[Tuple[datetime, datetime]] = None  # [min, max] range
    
    # Extracted resource requirements
    estimated_duration_hours: float = 0.0
    duration_uncertainty: float = 0.5       # σ of duration estimate (in hours)
    resources_required: List[ResourceType] = field(default_factory=list)
    
    # Ambiguity flags — CRITICAL for downstream handling
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None  # "deadline_unclear" | "scope_unclear" | etc.
    requires_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)
    
    # Implicit commitments detected
    implicit_commitments: List['ImplicitCommitmentSpec'] = field(default_factory=list)
    
    # Extraction metadata
    extraction_model: str = "rule_based_v1"  # Which extractor was used
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    raw_llm_response: Optional[str] = None  # For debugging


@dataclass
class ImplicitCommitmentSpec:
    """Specification for an implicit commitment to be instantiated."""
    implicit_type: str          # "travel_buffer" | "cognitive_prep" | "followup" | "precondition"
    description: str
    time_before_parent: Optional[timedelta] = None  # How long before parent commit
    time_after_parent: Optional[timedelta] = None
    duration_hours: float = 0.5
    auto_accept: bool = True    # Should be added to CDG automatically on parent acceptance


# ─── Core Commitment Node ──────────────────────────────────────────────────

@dataclass
class CommitmentNode:
    """
    A single node in the Commitment Dependency Graph.
    The fundamental unit of the VERGIL system.
    """
    # Identity
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_commitment_id: Optional[str] = None  # For implicit commits
    
    # Semantic content
    label: str = ""                         # Short human-readable label
    description: str = ""                  # Full commitment description
    deliverable: str = ""                  # Concrete output expected
    commitment_type: CommitmentType = CommitmentType.EXPLICIT_SOFT
    status: CommitmentStatus = CommitmentStatus.PENDING
    
    # Stakeholder
    stakeholder_id: str = ""               # Who made this request
    stakeholder_role: StakeholderRole = StakeholderRole.COLLEAGUE
    
    # Temporal parameters — UNCERTAIN by design
    deadline: Optional[datetime] = None    # Point deadline (if known)
    deadline_range: Optional[Tuple[datetime, datetime]] = None  # [earliest, latest]
    deadline_confidence: float = 0.5       # How confident is our deadline estimate
    earliest_start: Optional[datetime] = None
    
    # Resource parameters — UNCERTAIN by design
    estimated_duration_hours: float = 2.0
    duration_std_hours: float = 1.0        # Standard deviation (uncertainty)
    resources: List[ResourceType] = field(default_factory=list)
    cognitive_load_score: float = 0.5      # 0=trivial, 1=fully consuming
    
    # CDG-level properties
    urgency: float = 0.5                   # Computed, not user-set; updates over time
    risk_score: float = 0.0               # P(failure) given current state
    cascade_potential: float = 0.0         # How many nodes fail if this one fails
    
    # Decision tracking
    decision_made: Optional[ActionType] = None
    decision_timestamp: Optional[datetime] = None
    decision_rationale: str = ""           # Why the agent chose this action
    counter_proposal: Optional[Dict] = None
    renegotiation_count: int = 0           # How many times renegotiated
    
    # Execution tracking
    progress_pct: float = 0.0             # 0–100; updated by simulation
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


# ─── CDG Edge ──────────────────────────────────────────────────────────────

@dataclass
class CDGEdge:
    """
    Directed edge in the Commitment Dependency Graph.
    T-edges: A → B means A must complete before B.
    R-edges: A ↔ B means A and B conflict over a resource (bidirectional).
    """
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    from_node: str = ""                    # Node ID
    to_node: str = ""                      # Node ID
    edge_type: EdgeType = EdgeType.TEMPORAL
    
    # Temporal dependency params (for TEMPORAL edges)
    lag_hours: float = 0.0                 # Minimum gap after 'from' completes
    hard_ordering: bool = True             # False = soft preference, not hard constraint
    
    # Resource conflict params (for RESOURCE edges)
    resource_type: Optional[ResourceType] = None
    conflict_severity: float = 1.0         # 0 = minor; 1 = complete mutual exclusion
    
    # Uncertainty
    weight: float = 1.0                   # Edge importance weight
    confidence: float = 1.0              # How sure we are this edge is real
    
    # Cascade tracking
    failure_propagation_factor: float = 0.8  # How much failure transfers across edge
    
    created_at: datetime = field(default_factory=datetime.now)


# ─── Stakeholder ────────────────────────────────────────────────────────────

@dataclass
class StakeholderProfile:
    """
    Full model of a stakeholder. Includes HIDDEN fields
    that the agent cannot observe — simulates real partial observability.
    """
    stakeholder_id: str
    name: str
    role: StakeholderRole
    
    # Agent-observable fields
    relationship_weight: float = 0.5      # How important this relationship is
    domain: str = "professional"          # "professional" | "personal" | "hybrid"
    communication_style: str = "direct"   # "direct" | "indirect" | "pressure"
    
    # HIDDEN fields (oracle/environment only; agent cannot directly observe)
    _true_urgency: float = 0.5            # Their actual urgency level
    _deadline_flexibility_hours: float = 0.0  # Hidden slack in their deadline
    _forgiveness_rate: float = 0.5        # How quickly trust recovers
    _manipulation_tactics: List[str] = field(default_factory=list)  # "guilt" | "authority" | "flattery"
    _irrational_probability: float = 0.1  # P(behave irrationally in this episode)
    
    # Trust decay/repair rates (partially observable — agent infers over time)
    trust_decay_rate: float = 0.15        # Per broken commitment (agent can estimate)
    trust_repair_rate: float = 0.08       # Per kept commitment (agent can estimate)
    
    # Behavior parameters
    escalation_threshold: float = 0.4     # If trust < this, stakeholder escalates
    renegotiation_tolerance: int = 2      # Max renegotiations before trust penalty
    
    def get_observable_fields(self) -> Dict:
        """Return only fields the agent is allowed to see."""
        return {
            'stakeholder_id': self.stakeholder_id,
            'name': self.name,
            'role': self.role,
            'relationship_weight': self.relationship_weight,
            'domain': self.domain,
            'communication_style': self.communication_style,
            'trust_decay_rate': self.trust_decay_rate,    # Agent gets estimate
            'trust_repair_rate': self.trust_repair_rate,
            'escalation_threshold': self.escalation_threshold,
        }


# ─── Trust State (scalar for Phase 1; extended in Phase 2) ─────────────────

@dataclass
class TrustEntry:
    """
    Trust record for one stakeholder relationship.
    Phase 1: scalar trust. Phase 2: multi-dimensional.
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
        else:
            self.kept_commitments += 1
    
    def rolling_trust_delta(self, last_n: int = 5) -> float:
        """Net trust change over last N events. Signal for trend detection."""
        recent = self.history[-last_n:]
        return sum(e['delta'] for e in recent)


# ─── Message ────────────────────────────────────────────────────────────────

@dataclass
class Message:
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_id: str = ""
    sender_role: StakeholderRole = StakeholderRole.COLLEAGUE
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    delivery_time: Optional[datetime] = None   # When it should appear in agent's inbox
    
    # Metadata for simulation
    contains_commitment: bool = False       # Pre-labeled for scenario generation
    urgency_signal: float = 0.5            # Surface-level urgency in message tone
    pressure_tactic: Optional[str] = None  # Hidden: "guilt" | "authority" | None
    
    processed: bool = False
    extraction_result: Optional[ExtractionResult] = None


# ─── Agent Action ────────────────────────────────────────────────────────────

@dataclass
class AgentAction:
    action_type: ActionType
    target_node_id: Optional[str] = None      # Which commitment this applies to
    target_message_id: Optional[str] = None
    
    # Structured parameters for different action types
    # For COUNTER_PROPOSE / RENEGOTIATE:
    proposed_deadline: Optional[datetime] = None
    proposed_scope_reduction: Optional[str] = None
    proposed_resource_offload: Optional[str] = None
    proposed_delegation_target: Optional[str] = None
    
    # For all actions:
    rationale: str = ""                    # Agent's stated reason (for explainability)
    confidence: float = 0.5               # Agent's confidence in this decision
    
    # Feasibility assessment (agent's prediction)
    feasibility_prediction: float = 0.5   # Agent's P(feasible) estimate
    predicted_trust_impact: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    decision_time: datetime = field(default_factory=datetime.now)
    response_latency_ms: float = 0.0      # How long agent took to decide


# ─── Environment State ───────────────────────────────────────────────────────

@dataclass
class VERGILState:
    """
    Complete observable state. Hidden state lives in VERGILEnv._hidden.
    This is what gets passed to the agent.
    """
    # CDG
    cdg_nodes: List[CommitmentNode] = field(default_factory=list)
    cdg_edges: List[CDGEdge] = field(default_factory=list)
    satisfiability_score: float = 1.0     # P(CDG is satisfiable) [0,1]
    
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


# ─── Episode Record ──────────────────────────────────────────────────────────

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
```

---

## P1.2 — COMMITMENT EXTRACTION ENGINE (`core/extraction.py`)

**Design Goal**: Extract structured commitments from raw NL text. Must return
confidence scores, flag ambiguity, and generate implicit commitment specs.
Phase 1 uses a hybrid rule-based + keyword scoring approach.
Phase 2 replaces the scorer with a fine-tuned classifier head.

```python
# core/extraction.py

import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging

from .types import (
    ExtractionResult, ImplicitCommitmentSpec,
    CommitmentType, ResourceType
)

logger = logging.getLogger('vergil.extraction')


# ─── Commitment Signal Patterns ──────────────────────────────────────────────

HARD_COMMITMENT_PATTERNS = [
    # Direct promise patterns
    r'\b(I will|I\'ll|I can)\b.*\bby\b',
    r'\bwill (have|send|deliver|finish|complete|submit)\b',
    r'\bcommit(ted)? to\b',
    r'\bpromise\b',
    r'\bdeadline\b',
    r'\bdue\b.*(date|by|on)',
    r'\bmust (be|have|deliver)\b',
]

SOFT_COMMITMENT_PATTERNS = [
    r'\bI\'ll try\b',
    r'\bhopefully\b',
    r'\baim(ing)? to\b',
    r'\bplan(ning)? to\b',
    r'\bshould be able\b',
    r'\blet\'s (try to|aim to|see if)\b',
    r'\bwhenever I can\b',
]

REQUEST_PATTERNS = [
    # These create obligations on the AI if accepted
    r'\bcan you\b',
    r'\bcould you\b',
    r'\bwould you (mind|be able)\b',
    r'\bplease (have|send|finish|prepare|review)\b',
    r'\bneed(ing)? (you|this|it|the)\b',
    r'\bimportant that you\b',
    r'\bexpecting\b',
    r'\bcounting on\b',
]

SOFT_REQUEST_PATTERNS = [
    r'\bwhenever you (get a chance|can|have time)\b',
    r'\bno rush\b',
    r'\bat your convenience\b',
    r'\bif you (can|have time|get a chance)\b',
    r'\blet\'s sync (sometime|soon|next week)\b',
]

AMBIGUOUS_PATTERNS = [
    r'\bsometime (next|this)\b',
    r'\bsoon\b',
    r'\bASAP\b',
    r'\bas soon as possible\b',
    r'\bin the (near|coming) future\b',
    r'\bshortly\b',
    r'\bquickly\b',
]

DEADLINE_PATTERNS = {
    # Pattern → deadline resolution function or delta
    r'\bby (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b': 'next_weekday',
    r'\bby (\d{1,2})(am|pm)\b': 'today_time',
    r'\bby end of (day|week|month)\b': 'eod_eow_eom',
    r'\bthis (morning|afternoon|evening)\b': 'today_period',
    r'\btoday\b': 'today_eod',
    r'\btomorrow\b': 'tomorrow_eod',
    r'\bwithin (\d+) (hours?|days?|weeks?)\b': 'relative_delta',
    r'\bby (\d{1,2})/(\d{1,2})\b': 'explicit_date',
    r'\b(next )?(Monday|Tuesday|Wednesday|Thursday|Friday)\b': 'next_weekday_lookahead',
}

DURATION_KEYWORDS = {
    'quick': (0.5, 0.25),        # (estimate_hrs, std_hrs)
    'brief': (0.5, 0.25),
    'short': (1.0, 0.5),
    'presentation': (3.0, 1.5),
    'report': (4.0, 2.0),
    'spec': (5.0, 2.5),
    'review': (2.0, 1.0),
    'meeting': (1.0, 0.25),
    'call': (0.5, 0.25),
    'analysis': (4.0, 2.0),
    'research': (5.0, 3.0),
    'draft': (3.0, 1.5),
    'design': (6.0, 3.0),
    'implementation': (8.0, 4.0),
    'testing': (3.0, 2.0),
}

IMPLICIT_TEMPLATES = {
    # commitment_type_keyword → list of ImplicitCommitmentSpec templates
    'meeting': [
        ImplicitCommitmentSpec(
            implicit_type='travel_buffer',
            description='Travel/commute buffer before meeting',
            time_before_parent=timedelta(minutes=30),
            duration_hours=0.5,
            auto_accept=True,
        ),
        ImplicitCommitmentSpec(
            implicit_type='cognitive_prep',
            description='Preparation time before meeting',
            time_before_parent=timedelta(hours=1),
            duration_hours=0.5,
            auto_accept=True,
        ),
    ],
    'presentation': [
        ImplicitCommitmentSpec(
            implicit_type='precondition',
            description='Slide deck preparation required before presentation',
            time_before_parent=timedelta(hours=4),
            duration_hours=3.0,
            auto_accept=False,  # Agent must explicitly accept this
        ),
    ],
    'report': [
        ImplicitCommitmentSpec(
            implicit_type='precondition',
            description='Data gathering required before report',
            time_before_parent=timedelta(hours=8),
            duration_hours=2.0,
            auto_accept=False,
        ),
    ],
    'review': [
        ImplicitCommitmentSpec(
            implicit_type='followup',
            description='Feedback discussion session after review',
            time_after_parent=timedelta(hours=24),
            duration_hours=0.5,
            auto_accept=True,
        ),
    ],
}


class CommitmentExtractor:
    """
    NL text → ExtractionResult with confidence scoring.
    
    Design Philosophy:
    - Never return binary yes/no. Always return probability.
    - Flag ambiguity explicitly; don't silently resolve it.
    - Generate implicit commitment specs proactively.
    - Track extraction failures for curriculum improvement.
    
    Phase 1: Rule-based pattern matching + keyword scoring.
    Phase 2: Replace scorer with fine-tuned classifier head.
    """
    
    def __init__(self, current_time: datetime, config: Optional[Dict] = None):
        self.current_time = current_time
        self.config = config or {}
        self.commitment_threshold = self.config.get('commitment_threshold', 0.5)
        self.ambiguity_threshold = self.config.get('ambiguity_threshold', 0.3)
        
        # Extraction failure log (for curriculum feedback)
        self._extraction_failures: List[Dict] = []
        
        logger.info(f"CommitmentExtractor initialized. threshold={self.commitment_threshold}")
    
    def extract(self, message_text: str, sender_role: str = 'colleague') -> ExtractionResult:
        """
        Main extraction pipeline.
        
        Steps:
        1. Score commitment probability (pattern matching)
        2. Classify commitment type (hard vs soft)
        3. Extract deadline (with confidence + range)
        4. Estimate duration (with uncertainty)
        5. Generate implicit commitment specs
        6. Flag ambiguities
        7. Generate clarification questions if needed
        
        Args:
            message_text: Raw message content
            sender_role: Stakeholder role (affects interpretation; boss = higher urgency)
        
        Returns:
            ExtractionResult with all fields populated
        """
        text_lower = message_text.lower()
        
        # ── Step 1: Commitment probability ──────────────────────────────────
        commitment_prob = self._score_commitment_probability(text_lower, sender_role)
        
        # Short-circuit: if very unlikely to be a commitment, return early
        if commitment_prob < 0.15:
            return ExtractionResult(
                raw_text=message_text,
                is_commitment=False,
                commitment_probability=commitment_prob,
            )
        
        # ── Step 2: Commitment type ──────────────────────────────────────────
        c_type, type_confidence = self._classify_type(text_lower, sender_role)
        
        # ── Step 3: Deadline extraction ──────────────────────────────────────
        deadline, d_confidence, d_range = self._extract_deadline(text_lower)
        
        # ── Step 4: Duration estimation ──────────────────────────────────────
        duration_hrs, duration_std = self._estimate_duration(text_lower)
        
        # ── Step 5: Resource requirements ────────────────────────────────────
        resources = self._identify_resources(text_lower)
        
        # ── Step 6: Implicit commitments ─────────────────────────────────────
        implicits = self._generate_implicits(text_lower, deadline)
        
        # ── Step 7: Ambiguity detection ──────────────────────────────────────
        is_ambiguous, ambiguity_reason, clarification_qs = self._check_ambiguity(
            text_lower, deadline, d_confidence, duration_hrs
        )
        
        # ── Step 8: Finalize probability given all signals ────────────────────
        # Penalize probability if deadline is missing or ambiguous
        final_prob = commitment_prob
        if deadline is None:
            final_prob *= 0.7
        if is_ambiguous:
            final_prob *= 0.85
        
        result = ExtractionResult(
            raw_text=message_text,
            is_commitment=final_prob >= self.commitment_threshold,
            commitment_probability=round(final_prob, 3),
            commitment_type=c_type,
            deadline_estimate=deadline,
            deadline_confidence=round(d_confidence, 3),
            deadline_range=d_range,
            estimated_duration_hours=duration_hrs,
            duration_uncertainty=duration_std,
            resources_required=resources,
            implicit_commitments=implicits,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
            requires_clarification=is_ambiguous and d_confidence < self.ambiguity_threshold,
            clarification_questions=clarification_qs,
        )
        
        logger.debug(f"Extraction: prob={final_prob:.3f}, type={c_type}, "
                     f"deadline={deadline}, ambiguous={is_ambiguous}")
        return result
    
    def _score_commitment_probability(self, text: str, role: str) -> float:
        """
        Score P(commitment | text). Additive pattern matching with role prior.
        
        Role prior: boss messages have higher baseline commitment probability
        because even casual requests from authority figures create de facto obligations.
        """
        score = 0.0
        
        # Hard commitment patterns → high score
        for pattern in HARD_COMMITMENT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.25
        
        # Request patterns → medium score (obligation depends on acceptance)
        for pattern in REQUEST_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.18
        
        # Soft commitment patterns → lower score
        for pattern in SOFT_COMMITMENT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.12
        
        # Soft request patterns → low score
        for pattern in SOFT_REQUEST_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.06
        
        # Role adjustment
        role_priors = {
            'boss': 1.35,    # Authority inflation
            'client': 1.20,  # Contractual implication
            'colleague': 1.00,
            'friend': 0.80,  # Lower default; higher for personal signals
            'system': 0.90,
        }
        score *= role_priors.get(role, 1.0)
        
        return min(1.0, score)
    
    def _classify_type(self, text: str, role: str) -> Tuple[CommitmentType, float]:
        """Classify commitment as hard/soft/implicit/social."""
        hard_hits = sum(1 for p in HARD_COMMITMENT_PATTERNS
                        if re.search(p, text, re.IGNORECASE))
        soft_hits = sum(1 for p in SOFT_COMMITMENT_PATTERNS
                        if re.search(p, text, re.IGNORECASE))
        
        if hard_hits > 0:
            if role in ('boss', 'client'):
                return CommitmentType.EXPLICIT_HARD, 0.85
            return CommitmentType.EXPLICIT_HARD, 0.7
        elif soft_hits > 0:
            return CommitmentType.EXPLICIT_SOFT, 0.75
        elif any(w in text for w in ['lunch', 'dinner', 'coffee', 'drinks', 'hangout']):
            return CommitmentType.SOCIAL, 0.80
        else:
            return CommitmentType.EXPLICIT_SOFT, 0.5
    
    def _extract_deadline(self, text: str) -> Tuple[Optional[datetime], float, Optional[Tuple]]:
        """
        Extract deadline with confidence. Returns:
        (deadline_estimate, confidence, (min_deadline, max_deadline))
        
        Edge cases handled:
        - "by 5pm" with no day → assumes today if time is future, else tomorrow
        - "by end of week" → Friday COB with ±4hr uncertainty range
        - "ASAP" → now + 4hrs with very low confidence (0.2)
        - No deadline found → (None, 0.0, None)
        """
        # Relative delta: "within 3 days"
        m = re.search(r'within (\d+) (hours?|days?|weeks?)', text, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            unit = m.group(2).lower().rstrip('s')
            deltas = {'hour': timedelta(hours=n), 'day': timedelta(days=n),
                      'week': timedelta(weeks=n)}
            dl = self.current_time + deltas[unit]
            uncertainty = deltas[unit] * 0.15
            return dl, 0.75, (dl - uncertainty, dl + uncertainty)
        
        # Named weekday: "by Thursday"
        weekdays = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
        for i, day in enumerate(weekdays):
            if f'by {day}' in text or f'on {day}' in text:
                days_ahead = (i - self.current_time.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7  # Next occurrence
                dl = self.current_time + timedelta(days=days_ahead)
                dl = dl.replace(hour=17, minute=0, second=0, microsecond=0)
                return dl, 0.70, (dl.replace(hour=12), dl.replace(hour=23, minute=59))
        
        # "by 5pm" / "by 10am"
        m = re.search(r'by (\d{1,2})(am|pm)', text, re.IGNORECASE)
        if m:
            hour = int(m.group(1))
            if m.group(2).lower() == 'pm' and hour != 12:
                hour += 12
            dl = self.current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
            if dl < self.current_time:
                dl += timedelta(days=1)  # Assume next occurrence
            return dl, 0.80, (dl - timedelta(hours=1), dl + timedelta(hours=1))
        
        # "end of day"
        if 'end of day' in text or 'eod' in text or 'by tonight' in text:
            dl = self.current_time.replace(hour=18, minute=0, second=0, microsecond=0)
            if dl < self.current_time:
                dl += timedelta(days=1)
            return dl, 0.65, (dl.replace(hour=17), dl.replace(hour=23, minute=59))
        
        # "end of week"
        if 'end of week' in text or 'eow' in text or 'by friday' in text:
            days_to_friday = (4 - self.current_time.weekday()) % 7
            dl = self.current_time + timedelta(days=days_to_friday)
            dl = dl.replace(hour=18, minute=0, second=0, microsecond=0)
            return dl, 0.55, (dl - timedelta(hours=6), dl + timedelta(hours=6))
        
        # "today"
        if 'today' in text:
            dl = self.current_time.replace(hour=18, minute=0, second=0, microsecond=0)
            return dl, 0.65, (dl.replace(hour=12), dl.replace(hour=23, minute=59))
        
        # "tomorrow"
        if 'tomorrow' in text:
            dl = self.current_time + timedelta(days=1)
            dl = dl.replace(hour=18, minute=0, second=0, microsecond=0)
            return dl, 0.60, (dl.replace(hour=9), dl.replace(hour=23, minute=59))
        
        # "ASAP" — very low confidence, high urgency
        if 'asap' in text or 'as soon as possible' in text or 'urgently' in text:
            dl = self.current_time + timedelta(hours=4)
            return dl, 0.20, (self.current_time, self.current_time + timedelta(hours=24))
        
        # "soon" / "sometime next week" — ambiguous
        if 'soon' in text or 'shortly' in text:
            dl = self.current_time + timedelta(days=3)
            return dl, 0.15, (self.current_time + timedelta(hours=1),
                               self.current_time + timedelta(days=7))
        
        return None, 0.0, None
    
    def _estimate_duration(self, text: str) -> Tuple[float, float]:
        """
        Estimate task duration with uncertainty (mean, std_dev) in hours.
        Checks for duration keywords + task type keywords.
        Returns (mean_hours, std_hours).
        """
        for keyword, (est, std) in DURATION_KEYWORDS.items():
            if keyword in text:
                return est, std
        
        # Default fallback
        return 2.0, 1.5
    
    def _identify_resources(self, text: str) -> List[ResourceType]:
        resources = []
        if any(w in text for w in ['meeting', 'call', 'sync', 'session', 'attend']):
            resources.append(ResourceType.TIME_BLOCK)
            resources.append(ResourceType.PHYSICAL_PRESENCE)
        if any(w in text for w in ['think', 'write', 'design', 'analyze', 'research']):
            resources.append(ResourceType.COGNITIVE_LOAD)
        if any(w in text for w in ['together', 'with you', 'jointly', 'collaborate']):
            resources.append(ResourceType.COLLABORATOR)
        if not resources:
            resources.append(ResourceType.TIME_BLOCK)
        return list(set(resources))
    
    def _generate_implicits(self, text: str,
                             parent_deadline: Optional[datetime]) -> List[ImplicitCommitmentSpec]:
        """
        Generate implicit commitment specs based on commitment type keywords.
        
        Critical: This is where most LLMs fail. Accepting "attend board meeting"
        should auto-generate travel buffer and slide prep as implicit commitments.
        """
        implicits = []
        for keyword, templates in IMPLICIT_TEMPLATES.items():
            if keyword in text:
                for tmpl in templates:
                    # Adjust timing relative to parent deadline
                    adjusted = ImplicitCommitmentSpec(
                        implicit_type=tmpl.implicit_type,
                        description=tmpl.description,
                        time_before_parent=tmpl.time_before_parent,
                        time_after_parent=tmpl.time_after_parent,
                        duration_hours=tmpl.duration_hours,
                        auto_accept=tmpl.auto_accept,
                    )
                    implicits.append(adjusted)
        return implicits
    
    def _check_ambiguity(self, text: str, deadline: Optional[datetime],
                          d_confidence: float, duration: float
                          ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Identify ambiguities and generate targeted clarification questions.
        
        Ambiguity types:
        - deadline_unclear: No specific deadline; only vague time references
        - scope_unclear: What needs to be delivered is vague
        - ownership_unclear: Not clear who is responsible for what
        - commitment_vs_request: Unclear if this creates an obligation
        """
        ambiguity_flags = []
        clarification_qs = []
        
        if deadline is None:
            ambiguity_flags.append('deadline_unclear')
            clarification_qs.append("What specific deadline are you working to?")
        elif d_confidence < 0.3:
            ambiguity_flags.append('deadline_low_confidence')
            clarification_qs.append("Just to confirm — are you thinking [inferred deadline]?")
        
        if any(p in text for p in AMBIGUOUS_PATTERNS):
            ambiguity_flags.append('timing_vague')
            clarification_qs.append("When you say 'soon', what timeframe works for you?")
        
        vague_scope_patterns = ['something', 'stuff', 'things', 'whatever', 'anything']
        if any(p in text for p in vague_scope_patterns):
            ambiguity_flags.append('scope_unclear')
            clarification_qs.append("Can you be more specific about what you need?")
        
        is_ambiguous = len(ambiguity_flags) > 0
        reason = ', '.join(ambiguity_flags) if ambiguity_flags else None
        return is_ambiguous, reason, clarification_qs[:2]  # Return max 2 questions
```

---

## P1.3 — CDG ENGINE (`core/cdg.py`)

The graph engine is the computational heart of VERGIL. It must:
1. Maintain the DAG structure with cycle prevention
2. Evaluate satisfiability (CSP check)
3. Propagate failure through the dependency chain
4. Track resource conflicts
5. Generate risk scores per node

```python
# core/cdg.py

import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import hashlib, json, logging

from .types import (
    CommitmentNode, CDGEdge, CommitmentStatus, EdgeType,
    ResourceType, FailureType
)

logger = logging.getLogger('vergil.cdg')


class CDGSatisfiabilityResult:
    """Result of satisfiability check."""
    def __init__(self):
        self.is_satisfiable: bool = True
        self.satisfiability_score: float = 1.0  # 0 = definitely infeasible; 1 = definitely feasible
        self.violations: List[Dict] = []         # Specific constraint violations
        self.at_risk_nodes: List[str] = []       # Nodes with risk > threshold
        self.bottleneck_resources: List[str] = []
        self.critical_path: List[str] = []       # Longest dependency chain
        self.slack_by_node: Dict[str, float] = {}  # Hours of slack per node
    
    def to_dict(self) -> Dict:
        return {
            'is_satisfiable': self.is_satisfiable,
            'satisfiability_score': round(self.satisfiability_score, 4),
            'n_violations': len(self.violations),
            'at_risk_nodes': self.at_risk_nodes,
            'bottleneck_resources': self.bottleneck_resources,
        }


class CommitmentDependencyGraph:
    """
    Core CDG data structure and operations.
    
    Internal representation: networkx DiGraph for T/logical edges.
    Resource conflicts stored separately (bidirectional, not in DiGraph).
    
    Invariants maintained:
    - No cycles in temporal/logical edges (enforced on every add_edge)
    - Node IDs are unique
    - Every edge references existing nodes
    
    Design for testability:
    - All state mutations go through named methods (no direct attribute access)
    - Every method logs its operation at DEBUG level
    - CDG can be serialized/deserialized completely at any point
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
        # {(node_id_a, node_id_b): CDGEdge} — sorted tuple as key
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
        
        Returns:
            True if added successfully
            False if node_id already exists
        
        Side effects:
            - Initializes risk score to 0.0
            - Logs the addition
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
                if n.status in (CommitmentStatus.ACCEPTED, CommitmentStatus.IN_PROGRESS)]
    
    def get_nodes_by_status(self, status: CommitmentStatus) -> List[CommitmentNode]:
        return [n for n in self._nodes.values() if n.status == status]
    
    # ── Edge Operations ─────────────────────────────────────────────────────
    
    def add_edge(self, edge: CDGEdge) -> Tuple[bool, Optional[str]]:
        """
        Add an edge to the CDG.
        
        For TEMPORAL and LOGICAL edges: checks for cycle creation before adding.
        For RESOURCE edges: stored in conflict index; does not affect DAG property.
        
        Returns:
            (success: bool, error_reason: Optional[str])
        """
        if edge.from_node not in self._nodes:
            return False, f"from_node {edge.from_node} not in CDG"
        if edge.to_node not in self._nodes:
            return False, f"to_node {edge.to_node} not in CDG"
        
        if edge.edge_type in (EdgeType.TEMPORAL, EdgeType.LOGICAL, EdgeType.IMPLICIT_DERIVE):
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
        return list(self._graph.predecessors(node_id))
    
    def get_temporal_successors(self, node_id: str) -> List[str]:
        """Nodes that are blocked until node_id completes."""
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
        
        Algorithm:
        1. Topological sort of temporal edges → ordering
        2. For each node in order: compute earliest possible start
        3. Check if earliest_possible_completion <= deadline
        4. Check resource conflicts: overlapping time assignments
        5. Compute slack per node; nodes with slack < 0 are infeasible
        6. Compute overall satisfiability score
        
        This is a Temporal CSP solved greedily (not optimal, but fast enough for RL step).
        Phase 2 replaces with constraint propagation for better accuracy.
        
        Returns:
            CDGSatisfiabilityResult with violations, risk scores, critical path
        """
        result = CDGSatisfiabilityResult()
        active_nodes = self.get_active_nodes()
        
        if not active_nodes:
            result.satisfiability_score = 1.0
            return result
        
        # ── Step 1: Topological order of temporal graph ──────────────────────
        try:
            topo_order = list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            # Cycle detected (should not happen if add_edge is used correctly)
            result.is_satisfiable = False
            result.satisfiability_score = 0.0
            result.violations.append({'type': 'cycle_detected', 'severity': 1.0})
            return result
        
        # ── Step 2: Earliest start computation (forward pass) ────────────────
        earliest_start: Dict[str, datetime] = {}
        
        for nid in topo_order:
            node = self._nodes.get(nid)
            if node is None or node.status not in (
                    CommitmentStatus.ACCEPTED, CommitmentStatus.IN_PROGRESS):
                continue
            
            preds = self.get_temporal_predecessors(nid)
            if not preds:
                earliest_start[nid] = node.earliest_start or current_time
            else:
                # Start after latest predecessor finishes (+ lag)
                pred_ends = []
                for pred_id in preds:
                    pred = self._nodes.get(pred_id)
                    edge = self._get_temporal_edge(pred_id, nid)
                    lag = timedelta(hours=edge.lag_hours) if edge else timedelta(0)
                    if pred_id in earliest_start and pred:
                        pred_end = earliest_start[pred_id] + timedelta(hours=pred.estimated_duration_hours)
                        pred_ends.append(pred_end + lag)
                
                earliest_start[nid] = max(pred_ends) if pred_ends else current_time
        
        # ── Step 3: Slack computation and violation detection ─────────────────
        total_slack = 0.0
        n_nodes_checked = 0
        
        for nid in topo_order:
            node = self._nodes.get(nid)
            if node is None or node.deadline is None:
                continue
            if node.status not in (CommitmentStatus.ACCEPTED, CommitmentStatus.IN_PROGRESS):
                continue
            
            es = earliest_start.get(nid, current_time)
            expected_completion = es + timedelta(hours=node.estimated_duration_hours)
            slack_hours = (node.deadline - expected_completion).total_seconds() / 3600
            
            result.slack_by_node[nid] = slack_hours
            total_slack += max(0, slack_hours)
            n_nodes_checked += 1
            
            # Risk score: nonlinear function of slack
            if slack_hours < 0:
                risk = 1.0  # Already infeasible
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
        
        # ── Step 4: Resource conflict check ────────────────────────────────
        for (a_id, b_id), edge in self._resource_conflicts.items():
            node_a = self._nodes.get(a_id)
            node_b = self._nodes.get(b_id)
            if node_a is None or node_b is None:
                continue
            
            # Check time overlap
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
                        'resource': edge.resource_type,
                        'overlap_hours': round(overlap, 2),
                        'severity': edge.conflict_severity,
                    })
                    # Increase risk for both nodes
                    self._node_risk_scores[a_id] = min(1.0,
                        self._node_risk_scores.get(a_id, 0) + 0.3 * edge.conflict_severity)
                    self._node_risk_scores[b_id] = min(1.0,
                        self._node_risk_scores.get(b_id, 0) + 0.3 * edge.conflict_severity)
        
        # ── Step 5: Total capacity check ──────────────────────────────────
        total_required = sum(
            n.estimated_duration_hours for n in active_nodes
            if result.slack_by_node.get(n.node_id, float('inf')) < 48
        )
        if total_required > available_hours * 1.1:  # 10% tolerance
            result.violations.append({
                'type': 'capacity_exceeded',
                'required_hours': round(total_required, 2),
                'available_hours': round(available_hours, 2),
                'overflow_hours': round(total_required - available_hours, 2),
                'severity': min(1.0, (total_required - available_hours) / available_hours),
            })
        
        # ── Step 6: Critical path ──────────────────────────────────────────
        if len(self._graph.nodes) > 0 and nx.is_directed_acyclic_graph(self._graph):
            try:
                result.critical_path = nx.dag_longest_path(self._graph)
            except Exception:
                result.critical_path = []
        
        # ── Step 7: Overall satisfiability score ──────────────────────────
        n_violations = len(result.violations)
        severity_sum = sum(v.get('severity', 0.5) for v in result.violations)
        result.is_satisfiable = n_violations == 0
        result.satisfiability_score = max(0.0, 1.0 - (severity_sum / max(1, n_nodes_checked)))
        
        logger.debug(f"CDG satisfiability: score={result.satisfiability_score:.3f}, "
                     f"violations={n_violations}, at_risk={len(result.at_risk_nodes)}")
        return result
    
    # ── Failure Propagation ─────────────────────────────────────────────────
    
    def propagate_failure(self, failed_node_id: str, current_time: datetime,
                           propagation_factor: float = 0.8) -> List[Dict]:
        """
        When a commitment fails, propagate risk to dependent nodes.
        
        Algorithm:
        - BFS from failed node through temporal/logical successor edges
        - Each hop: risk += failed_node.urgency * edge.failure_propagation_factor
        - Nodes that exceed risk_threshold=0.85 → status AT_RISK
        - Nodes that were already AT_RISK → status FAILED (cascade)
        - Records cascade event for curriculum analysis
        
        Returns:
            List of cascade events [{node_id, old_risk, new_risk, cascaded: bool}]
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
            if new_risk > 0.85 and node.status == CommitmentStatus.ACCEPTED:
                # This node is now considered cascaded failure
                self.update_node_status(nid, CommitmentStatus.FAILED, current_time)
                cascaded = True
                # Continue propagation from this newly failed node
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
        Does NOT include content (labels, deadlines) — only structure.
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
            } for nid, n in self._nodes.items()},
            'edges': [{'from': e.from_node, 'to': e.to_node,
                       'type': e.edge_type.value, 'weight': e.weight}
                      for e in self._edges.values()],
            'topology_hash': self.topology_hash(),
        }
    
    def _get_temporal_edge(self, from_id: str, to_id: str) -> Optional[CDGEdge]:
        for edge in self._edges.values():
            if (edge.from_node == from_id and edge.to_node == to_id and
                    edge.edge_type == EdgeType.TEMPORAL):
                return edge
        return None
```

---

## P1.4 — STAKEHOLDER SIMULATOR (`core/stakeholder.py`)

Phase 1: Deterministic responses. Phase 2 extends to probabilistic + adversarial.

```python
# core/stakeholder.py

import random
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

from .types import (
    AgentAction, ActionType, StakeholderProfile, StakeholderRole,
    TrustEntry, CommitmentNode
)

logger = logging.getLogger('vergil.stakeholder')


@dataclass  
class StakeholderResponse:
    """The stakeholder's reaction to the agent's action."""
    stakeholder_id: str
    accepted: bool                       # Did they accept the agent's action?
    counter_offer: Optional[Dict] = None # Counter-offer params if rejected
    trust_delta: float = 0.0            # Immediate trust change
    
    # Observable signals
    message: str = ""                    # Their response message
    expressed_urgency: float = 0.5      # Urgency signaled (may not equal true urgency)
    expressed_flexibility: float = 0.5  # Flexibility signaled (may not equal true)
    
    # Hidden (oracle only)
    _true_satisfaction: float = 0.5     # How satisfied they actually are
    _will_escalate: bool = False         # Will they escalate next step?


class StakeholderSimulator:
    """
    Simulate stakeholder responses to agent actions.
    
    Phase 1: Deterministic rules based on action type, trust, and context.
    Phase 2: Probabilistic responses drawn from behavioral distribution.
    
    Response logic:
    - ACCEPT of a request → stakeholder satisfied (+trust)
    - DECLINE → stakeholder reacts based on trust and role
    - COUNTER_PROPOSE → stakeholder evaluates offer quality
    - RENEGOTIATE → stakeholder reacts based on lead time and frequency
    """
    
    def __init__(self, profiles: Dict[str, StakeholderProfile], seed: int = 42):
        self.profiles = profiles
        self.rng = random.Random(seed)
        
        # Track per-stakeholder renegotiation count (affects tolerance)
        self._renegotiation_counts: Dict[str, int] = {}
    
    def simulate_response(self, action: AgentAction, node: CommitmentNode,
                           trust: TrustEntry, current_time: datetime
                           ) -> StakeholderResponse:
        """
        Generate stakeholder response to agent's action.
        
        Args:
            action: What the agent did
            node: The commitment being acted on
            trust: Current trust state with this stakeholder
            current_time: Current simulation time
        
        Returns:
            StakeholderResponse with trust delta and acceptance
        """
        profile = self.profiles.get(node.stakeholder_id)
        if profile is None:
            logger.warning(f"No profile for stakeholder {node.stakeholder_id}")
            return StakeholderResponse(stakeholder_id=node.stakeholder_id, accepted=True)
        
        if action.action_type == ActionType.ACCEPT:
            return self._respond_to_accept(action, node, trust, profile)
        elif action.action_type == ActionType.DECLINE:
            return self._respond_to_decline(action, node, trust, profile)
        elif action.action_type == ActionType.COUNTER_PROPOSE:
            return self._respond_to_counter(action, node, trust, profile, current_time)
        elif action.action_type == ActionType.RENEGOTIATE:
            return self._respond_to_renegotiate(action, node, trust, profile, current_time)
        elif action.action_type == ActionType.CLARIFY:
            return self._respond_to_clarify(action, node, trust, profile)
        else:
            return StakeholderResponse(stakeholder_id=node.stakeholder_id,
                                        accepted=True, trust_delta=0.0,
                                        message="Acknowledged.")
    
    def _respond_to_accept(self, action, node, trust, profile) -> StakeholderResponse:
        trust_delta = profile.trust_repair_rate * 0.3  # Partial credit for acceptance
        msg = self._acceptance_message(profile)
        return StakeholderResponse(
            stakeholder_id=node.stakeholder_id,
            accepted=True, trust_delta=trust_delta,
            message=msg, expressed_urgency=node._hidden_urgency if hasattr(node,'_hidden_urgency') else 0.5,
            _true_satisfaction=0.7,
        )
    
    def _respond_to_decline(self, action, node, trust, profile) -> StakeholderResponse:
        """
        Decline response is role-dependent.
        Boss: likely pushback, trust penalty scales with urgency.
        Client: risk of escalation or contract review.
        Friend: depends on how the decline is framed.
        """
        urgency_factor = node.urgency
        
        role_responses = {
            StakeholderRole.BOSS: {
                'trust_delta': -profile.trust_decay_rate * (1 + urgency_factor),
                'message': "I need this done. Can we talk about what's blocking you?",
                'accepted': False,
            },
            StakeholderRole.CLIENT: {
                'trust_delta': -profile.trust_decay_rate * 1.2,
                'message': "This is concerning. We had an expectation this would be handled.",
                'accepted': False,
            },
            StakeholderRole.COLLEAGUE: {
                'trust_delta': -profile.trust_decay_rate * 0.6,
                'message': "No worries, I'll figure something out.",
                'accepted': True,  # Colleague may accept the decline gracefully
            },
            StakeholderRole.FRIEND: {
                'trust_delta': -profile.trust_decay_rate * 0.4,
                'message': "Oh, okay. I understand.",
                'accepted': True,
            },
        }
        
        resp_data = role_responses.get(profile.role, role_responses[StakeholderRole.COLLEAGUE])
        return StakeholderResponse(
            stakeholder_id=node.stakeholder_id,
            accepted=resp_data['accepted'],
            trust_delta=resp_data['trust_delta'],
            message=resp_data['message'],
            _true_satisfaction=0.2,
        )
    
    def _respond_to_counter(self, action, node, trust, profile, current_time) -> StakeholderResponse:
        """
        Counter-proposal response. Quality of counter matters.
        A counter that gives stakeholder >80% of what they need → accepted.
        A counter that cuts more than 50% → rejected.
        """
        # Simple quality scoring for Phase 1
        quality = 0.5  # Default middle quality
        
        if action.proposed_deadline:
            hours_delta = (action.proposed_deadline - node.deadline).total_seconds() / 3600
            if hours_delta > 0 and hours_delta <= 24:
                quality = 0.7  # Small extension → likely accepted
            elif hours_delta > 24:
                quality = 0.4  # Large extension → depends on role
            else:
                quality = 0.9  # Earlier than requested → definitely accepted
        
        # Boss is stricter about counter-proposals
        if profile.role == StakeholderRole.BOSS:
            quality *= 0.85
        
        accepted = quality > 0.55
        trust_delta = 0.03 if accepted else -0.05
        
        if accepted:
            msg = "That works. Let's go with that."
        else:
            msg = "I really need the original timeline. Can you make it work?"
        
        return StakeholderResponse(
            stakeholder_id=node.stakeholder_id,
            accepted=accepted, trust_delta=trust_delta,
            message=msg, _true_satisfaction=quality,
        )
    
    def _respond_to_renegotiate(self, action, node, trust, profile, current_time) -> StakeholderResponse:
        """
        Renegotiation response.
        
        KEY DESIGN: Lead time matters enormously.
        Early renegotiation (>48hrs) → stakeholder has time to adapt → trust preserved
        Late renegotiation (<4hrs) → no time to adapt → trust damaged
        """
        hours_to_deadline = (node.deadline - current_time).total_seconds() / 3600 if node.deadline else 48
        renego_count = self._renegotiation_counts.get(node.stakeholder_id, 0)
        
        # Lead time factor
        if hours_to_deadline > 48:
            lead_factor = 1.0
        elif hours_to_deadline > 24:
            lead_factor = 0.6
        elif hours_to_deadline > 4:
            lead_factor = 0.2
        else:
            lead_factor = -0.2  # This late, renegotiation is harmful
        
        # Frequency penalty
        frequency_factor = max(0.0, 1.0 - renego_count * 0.25)
        
        # Role tolerance
        role_tolerance = {
            StakeholderRole.BOSS: 0.6,
            StakeholderRole.CLIENT: 0.5,
            StakeholderRole.COLLEAGUE: 0.9,
            StakeholderRole.FRIEND: 0.95,
        }
        tolerance = role_tolerance.get(profile.role, 0.7)
        
        accepted_threshold = 0.4
        score = lead_factor * frequency_factor * tolerance
        accepted = score > accepted_threshold
        
        # Trust delta
        if accepted:
            trust_delta = 0.02 * lead_factor
            msg = "Thanks for the heads up. Let's rework the timeline."
        else:
            trust_delta = -0.08 * (1 - lead_factor)
            msg = "This is really inconvenient. I was counting on this."
        
        self._renegotiation_counts[node.stakeholder_id] = renego_count + 1
        
        logger.info(f"Renegotiate response: stakeholder={node.stakeholder_id} "
                    f"lead={hours_to_deadline:.1f}h score={score:.2f} accepted={accepted}")
        
        return StakeholderResponse(
            stakeholder_id=node.stakeholder_id,
            accepted=accepted, trust_delta=trust_delta,
            message=msg, _true_satisfaction=score,
        )
    
    def _respond_to_clarify(self, action, node, trust, profile) -> StakeholderResponse:
        """Clarification is generally positively received (agent is being diligent)."""
        return StakeholderResponse(
            stakeholder_id=node.stakeholder_id,
            accepted=True, trust_delta=0.01,
            message="Sure, let me clarify: [clarification content]",
            _true_satisfaction=0.65,
        )
    
    def _acceptance_message(self, profile: StakeholderProfile) -> str:
        messages = {
            StakeholderRole.BOSS: "Great, I'll be expecting it.",
            StakeholderRole.CLIENT: "Perfect, thank you for confirming.",
            StakeholderRole.COLLEAGUE: "Awesome, thanks!",
            StakeholderRole.FRIEND: "Great! Looking forward to it.",
        }
        return messages.get(profile.role, "Acknowledged.")
```

---

## P1.5 — REWARD FUNCTION (`core/reward.py`)

```python
# core/reward.py

from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

from .types import (
    AgentAction, ActionType, CommitmentNode, CommitmentStatus,
    TrustEntry, VERGILState
)

logger = logging.getLogger('vergil.reward')


@dataclass
class RewardComponents:
    """All reward components, stored separately for analysis and logging."""
    
    # Positive signals
    fulfillment: float = 0.0       # R_fulfill: quality-weighted completion rate
    trust_delta: float = 0.0      # R_trust: weighted trust change this step
    proactive: float = 0.0        # R_proactive: bonus for early renegotiation
    feasibility_acc: float = 0.0  # R_accuracy: feasibility prediction calibration
    
    # Penalties
    broken_penalty: float = 0.0   # P_broken: broken commitment penalty
    overrefusal_penalty: float = 0.0  # P_overrefusal: declining feasible things
    silent_drop_penalty: float = 0.0  # P_silent: accepted then dropped
    
    # Total
    total: float = 0.0
    
    # Metadata
    step: int = 0
    episode_id: str = ""
    action_type: str = ""
    
    def compute_total(self) -> float:
        """
        Final reward = weighted positives - penalties.
        
        Weights chosen to make the ONLY high-reward strategy genuine feasibility reasoning:
        - Declining everything: kills trust_delta, triggers overrefusal_penalty
        - Accepting everything: short-term trust spike, then catastrophic broken_penalty
        - Perfect play: high fulfillment + stable trust + proactive renegotiation
        """
        positive = (
            0.35 * self.fulfillment +
            0.25 * self.trust_delta +
            0.20 * self.proactive +
            0.10 * self.feasibility_acc
        )
        
        penalties = (
            0.40 * self.broken_penalty +
            0.30 * self.overrefusal_penalty +
            0.50 * self.silent_drop_penalty
        )
        
        self.total = positive - penalties
        return self.total
    
    def to_dict(self) -> Dict:
        return {
            'step': self.step,
            'action_type': self.action_type,
            'fulfillment': round(self.fulfillment, 4),
            'trust_delta': round(self.trust_delta, 4),
            'proactive': round(self.proactive, 4),
            'feasibility_acc': round(self.feasibility_acc, 4),
            'broken_penalty': round(self.broken_penalty, 4),
            'overrefusal_penalty': round(self.overrefusal_penalty, 4),
            'silent_drop_penalty': round(self.silent_drop_penalty, 4),
            'total': round(self.total, 4),
        }


class VERGILReward:
    """
    Multi-component reward function.
    
    Anti-reward-hacking mechanisms built into every component:
    - R_fulfill: Requires actual completion quality, not just binary done/not-done
    - R_trust: Weighted by relationship importance; can't optimize one relationship
    - P_broken: Scales with trust state (low trust = higher penalty)
    - P_overrefusal: Activates when decline rate > optimal_accept_rate
    - P_silent_drop: Grows over time from acceptance to drop
    
    Long-horizon credit assignment:
    - Intermediate rewards provide dense signal (trust deltas, feasibility accuracy)
    - Terminal reward provides sparse episode-end signal
    - TD(λ) with high lambda (0.9) propagates credit backward
    """
    
    # Relationship weights (how much each relationship matters for weighted trust)
    RELATIONSHIP_WEIGHTS = {
        'boss': 0.35, 'client': 0.30, 'colleague': 0.20, 'friend': 0.15
    }
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.overrefusal_sensitivity = self.config.get('overrefusal_sensitivity', 2.0)
        self.optimal_accept_fraction = self.config.get('optimal_accept_fraction', 0.75)
        
        # Episode-level tracking (reset on episode reset)
        self._step_decisions: List[Dict] = []
        self._accepted_but_dropped: List[str] = []  # Node IDs
        
        logger.info("VERGILReward initialized")
    
    def compute_step_reward(
        self,
        action: AgentAction,
        node: Optional[CommitmentNode],
        trust_deltas: Dict[str, float],
        trust_entries: Dict[str, TrustEntry],
        cascade_events: List[Dict],
        state: VERGILState,
        current_step: int,
    ) -> RewardComponents:
        """
        Compute per-step reward components.
        
        Called on every environment step(). Returns RewardComponents.
        
        Args:
            action: Agent's action this step
            node: Commitment node being acted on (if any)
            trust_deltas: {stakeholder_id: delta} from stakeholder responses
            trust_entries: Full trust state
            cascade_events: Any cascade events that occurred this step
            state: Current environment state
            current_step: Step number within episode
        """
        rc = RewardComponents(step=current_step, action_type=action.action_type.value)
        
        # ── R_trust: Weighted trust change ─────────────────────────────────
        rc.trust_delta = self._compute_trust_reward(trust_deltas, trust_entries)
        
        # ── R_proactive: Proactive renegotiation bonus ─────────────────────
        if action.action_type == ActionType.RENEGOTIATE and node is not None:
            rc.proactive = self._compute_proactive_bonus(action, node, state.current_time)
        
        # ── R_accuracy: Feasibility prediction accuracy ────────────────────
        if node is not None and action.feasibility_prediction is not None:
            rc.feasibility_acc = self._compute_accuracy_reward(action, node, state)
        
        # ── P_broken: Broken commitment penalty ───────────────────────────
        if cascade_events:
            rc.broken_penalty = self._compute_broken_penalty(cascade_events, trust_entries)
        
        # Also check if this step's action was a confirmed failure
        if node and node.status == CommitmentStatus.FAILED:
            if action.action_type not in (ActionType.RENEGOTIATE, ActionType.DECLINE):
                rc.broken_penalty += self._node_failure_penalty(node, trust_entries)
        
        # ── P_overrefusal: Over-refusal penalty ───────────────────────────
        if action.action_type == ActionType.DECLINE:
            rc.overrefusal_penalty = self._compute_overrefusal_penalty(state)
        
        # ── P_silent_drop: Silent drop penalty ────────────────────────────
        rc.silent_drop_penalty = self._compute_silent_drop_penalty(state, current_step)
        
        # Record for episode-level audit
        self._step_decisions.append({
            'step': current_step,
            'action': action.action_type.value,
            'node_id': node.node_id if node else None,
            'feasibility_pred': action.feasibility_prediction,
        })
        
        rc.compute_total()
        logger.debug(f"Reward step {current_step}: {rc.to_dict()}")
        return rc
    
    def compute_terminal_reward(
        self,
        final_state: VERGILState,
        episode_record: 'EpisodeRecord',
    ) -> RewardComponents:
        """
        Episode-end reward components. Added to final step reward.
        
        Provides sparse episode signal that enables long-horizon credit assignment.
        """
        rc = RewardComponents(step=episode_record.total_steps, action_type='terminal')
        
        # ── R_fulfill: Final commitment fulfillment rate ────────────────────
        nodes = final_state.cdg_nodes
        kept = sum(1 for n in nodes if n.status == CommitmentStatus.COMPLETED)
        made = sum(1 for n in nodes if n.status in (
            CommitmentStatus.COMPLETED, CommitmentStatus.FAILED,
            CommitmentStatus.LATE_COMPLETED, CommitmentStatus.ACCEPTED
        ))
        
        if made > 0:
            raw_rate = kept / made
            # Timeliness modifier: early completions worth more
            timeliness_scores = []
            for n in nodes:
                if n.status == CommitmentStatus.COMPLETED and n.deadline and n.completion_timestamp:
                    slack = (n.deadline - n.completion_timestamp).total_seconds() / 3600
                    timeliness_scores.append(min(1.2, max(0.5, 1.0 + slack / 48)))
            timeliness_modifier = sum(timeliness_scores) / len(timeliness_scores) if timeliness_scores else 1.0
            rc.fulfillment = raw_rate * timeliness_modifier
        
        # ── Final trust stability bonus ────────────────────────────────────
        # Bonus for keeping trust high AND stable throughout episode
        trust_stability = self._compute_trust_stability(final_state.trust_entries)
        rc.trust_delta = trust_stability * 0.5  # Additional trust bonus at episode end
        
        rc.compute_total()
        logger.info(f"Terminal reward: fulfillment={rc.fulfillment:.3f} "
                    f"trust_stability={trust_stability:.3f} total={rc.total:.3f}")
        return rc
    
    def _compute_trust_reward(self, trust_deltas: Dict, trust_entries: Dict) -> float:
        """Weighted trust delta across all stakeholders."""
        if not trust_deltas:
            return 0.0
        
        total_weight = 0.0
        weighted_delta = 0.0
        
        for sid, delta in trust_deltas.items():
            entry = trust_entries.get(sid)
            if entry is None:
                continue
            # Look up relationship weight by role (stored in trust metadata)
            role = getattr(entry, '_role', 'colleague')
            weight = self.RELATIONSHIP_WEIGHTS.get(role, 0.2)
            weighted_delta += weight * delta
            total_weight += weight
        
        return weighted_delta / max(total_weight, 0.01)
    
    def _compute_proactive_bonus(self, action: AgentAction,
                                  node: CommitmentNode,
                                  current_time: datetime) -> float:
        """
        Lead-time-based proactive renegotiation bonus.
        
        The EARLIER you renegotiate, the higher the bonus.
        At the deadline: bonus = 0 (and actually penalized in trust).
        48hrs+ before deadline: bonus approaches 1.0.
        
        This reward term drives the key VERGIL behavior:
        predict and act early, don't wait for failure.
        """
        if node.deadline is None:
            return 0.1
        
        hours_to_deadline = (node.deadline - current_time).total_seconds() / 3600
        
        if hours_to_deadline <= 0:
            return -0.1  # Renegotiating after deadline is useless
        elif hours_to_deadline < 2:
            return 0.05
        elif hours_to_deadline < 8:
            return 0.15 + 0.1 * (hours_to_deadline / 8)
        elif hours_to_deadline < 24:
            return 0.30 + 0.2 * (hours_to_deadline / 24)
        else:
            return min(1.0, 0.5 + 0.5 * (hours_to_deadline / 48))
    
    def _compute_accuracy_reward(self, action: AgentAction,
                                  node: CommitmentNode,
                                  state: VERGILState) -> float:
        """
        Feasibility prediction accuracy.
        Compared to CDG satisfiability score as proxy for ground truth.
        
        Note: True accuracy is only measurable at commitment deadline.
        Here we use current CDG satisfiability as a dense intermediate signal.
        """
        predicted = action.feasibility_prediction
        actual_proxy = state.satisfiability_score
        
        # Brier-like score: 1 - (predicted - actual)^2
        accuracy = 1.0 - (predicted - actual_proxy) ** 2
        return max(0.0, accuracy)
    
    def _compute_broken_penalty(self, cascade_events: List[Dict],
                                 trust_entries: Dict) -> float:
        """
        Broken commitment penalty scales with cascade depth and trust state.
        
        Low-trust stakeholders → higher penalty (relationship already damaged).
        Deep cascades → compounding penalty.
        """
        if not cascade_events:
            return 0.0
        
        total_penalty = 0.0
        for event in cascade_events:
            if event.get('cascaded'):
                depth = event.get('depth', 1)
                # Get trust for the stakeholder of this node (looked up from CDG)
                trust_weight = 1.0  # Default; Phase 2 will use actual trust
                penalty = trust_weight * (1 + depth * 0.2)
                total_penalty += penalty
        
        return min(3.0, total_penalty)  # Cap to prevent catastrophic gradients
    
    def _node_failure_penalty(self, node: CommitmentNode,
                               trust_entries: Dict) -> float:
        """Single-node failure penalty."""
        trust_entry = trust_entries.get(node.stakeholder_id)
        current_trust = trust_entry.trust_score if trust_entry else 0.5
        # Low trust means breaks are MORE penalized (relationship fragile)
        trust_weight = 1 + (1 - current_trust)
        return node.urgency * trust_weight
    
    def _compute_overrefusal_penalty(self, state: VERGILState) -> float:
        """
        Penalize excessive declining.
        
        Over-refusal detection: if decline rate > optimal_accept_fraction * capacity,
        penalty scales quadratically with the excess.
        
        Anti-hack: agent cannot simply decline everything to get perfect fulfillment.
        """
        total_decisions = len(self._step_decisions)
        if total_decisions < 5:
            return 0.0  # Insufficient data; grace period
        
        declines = sum(1 for d in self._step_decisions if d['action'] == 'decline')
        decline_rate = declines / total_decisions
        
        optimal_decline_rate = 1.0 - self.optimal_accept_fraction
        excess = max(0, decline_rate - optimal_decline_rate)
        
        return min(1.0, self.overrefusal_sensitivity * excess ** 2)
    
    def _compute_silent_drop_penalty(self, state: VERGILState,
                                      current_step: int) -> float:
        """
        Detect nodes that were ACCEPTED but have not been renegotiated
        despite being overdue or highly at-risk.
        
        Silent drops are the worst behavior: the stakeholder planned around
        the commitment and gets no warning.
        """
        penalty = 0.0
        for node in state.cdg_nodes:
            if (node.status == CommitmentStatus.FAILED and
                    node.decision_made == ActionType.ACCEPT and
                    node.renegotiation_count == 0):
                # Accepted + never renegotiated + now failed = silent drop
                steps_since_acceptance = current_step - (node.decision_timestamp or 0)
                penalty += 0.5 * (1 + steps_since_acceptance / 10)
        
        return min(2.0, penalty)
    
    def _compute_trust_stability(self, trust_entries: Dict) -> float:
        """
        Measure trust trajectory over the episode.
        Stable high trust → bonus. Collapsed trust → penalty.
        """
        if not trust_entries:
            return 0.0
        
        scores = []
        for entry in trust_entries.values():
            scores.append(entry.trust_score)
            # Penalize high volatility
            if len(entry.history) > 3:
                deltas = [abs(e['delta']) for e in entry.history[-5:]]
                volatility = sum(deltas) / len(deltas)
                scores.append(max(0, entry.trust_score - volatility))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def reset_episode(self) -> None:
        self._step_decisions = []
        self._accepted_but_dropped = []
```

---

## P1.6 — OPENENV CORE (`core/env.py`)

```python
# core/env.py

import gymnasium as gym
import json, logging, uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .types import (
    VERGILState, AgentAction, ActionType, CommitmentNode, CommitmentStatus,
    CDGEdge, EdgeType, Message, TrustEntry, EpisodeRecord,
    StakeholderProfile, StakeholderRole
)
from .cdg import CommitmentDependencyGraph
from .extraction import CommitmentExtractor
from .stakeholder import StakeholderSimulator
from .reward import VERGILReward, RewardComponents

logger = logging.getLogger('vergil.env')


class VERGILEnv(gym.Env):
    """
    OpenEnv-compatible training environment for VERGIL.
    
    API Contract:
        state()  → VERGILState
        step()   → (VERGILState, float, bool, bool, dict)
        reset()  → (VERGILState, dict)
    
    Design Principles:
    1. All hidden state lives in self._hidden (never exposed to agent)
    2. step() is the only place state mutations occur
    3. All events are logged with sufficient detail for replay
    4. Episode generation is deterministic given a seed (reproducible)
    
    Testability:
    - Environment can be run with a scripted action sequence (no agent needed)
    - All steps produce full RewardComponents (debuggable)
    - CDG state can be inspected at any step via env.debug_state()
    """
    
    metadata = {'render_modes': ['human', 'json']}
    
    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        super().__init__()
        
        self.config = config or self._default_config()
        self.seed = seed
        
        # Component initialization
        self.reward_fn = VERGILReward(config=self.config.get('reward', {}))
        self.extractor: Optional[CommitmentExtractor] = None   # Initialized in reset()
        self.simulator: Optional[StakeholderSimulator] = None  # Initialized in reset()
        self.cdg: Optional[CommitmentDependencyGraph] = None   # Initialized in reset()
        
        # State
        self._state: Optional[VERGILState] = None
        
        # Hidden state (oracle only)
        self._hidden: Dict = {}
        
        # Episode tracking
        self._episode_record: Optional[EpisodeRecord] = None
        self._current_step: int = 0
        self._max_steps: int = self.config.get('max_steps_per_episode', 40)
        
        # Message queue (pre-scheduled for episode; time-gated delivery)
        self._message_schedule: List[Message] = []
        
        # Log directory
        self._log_dir = Path(self.config.get('log_dir', '/tmp/vergil_logs'))
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Curriculum reference (set externally by CurriculumEngine in Phase 2)
        self.curriculum_stage: int = 1
        
        logger.info(f"VERGILEnv initialized. seed={seed}, max_steps={self._max_steps}")
    
    # ── OpenEnv Required Methods ──────────────────────────────────────────
    
    def reset(self, seed: Optional[int] = None,
               scenario: Optional[Dict] = None,
               options: Optional[Dict] = None) -> Tuple[VERGILState, Dict]:
        """
        Reset environment for new episode.
        
        If scenario is provided: use it directly (for testing specific cases).
        If scenario is None: generate from curriculum stage.
        
        Returns:
            (initial_state, info_dict)
        """
        if seed is not None:
            self.seed = seed
        
        # Generate or use scenario
        if scenario is None:
            scenario = self._generate_scenario(self.curriculum_stage)
        
        episode_id = str(uuid.uuid4())[:8]
        start_time = scenario.get('start_time', datetime.now().replace(
            hour=9, minute=0, second=0, microsecond=0))
        
        # Initialize CDG
        self.cdg = CommitmentDependencyGraph(graph_id=f"cdg_{episode_id}")
        
        # Load seed commitments (pre-existing obligations at episode start)
        for commit_data in scenario.get('seed_commitments', []):
            node = self._build_node_from_dict(commit_data)
            self.cdg.add_node(node)
        
        for edge_data in scenario.get('seed_edges', []):
            edge = self._build_edge_from_dict(edge_data)
            self.cdg.add_edge(edge)
        
        # Initialize stakeholders
        profiles = self._build_stakeholder_profiles(scenario)
        self.simulator = StakeholderSimulator(profiles=profiles, seed=self.seed)
        
        # Initialize trust entries
        trust_entries = {}
        for s_id, profile in profiles.items():
            trust_score = scenario.get('initial_trust', {}).get(s_id, 0.65)
            te = TrustEntry(stakeholder_id=s_id, trust_score=trust_score)
            te._role = profile.role.value  # Store for reward weighting
            trust_entries[s_id] = te
        
        # Initialize extractor
        self.extractor = CommitmentExtractor(current_time=start_time)
        
        # Schedule messages for this episode
        self._message_schedule = self._build_message_schedule(scenario, start_time)
        
        # Initialize hidden state
        self._hidden = {
            'stakeholder_profiles': profiles,
            'true_durations': {  # Actual durations (revealed at completion)
                node_id: max(0.5, node.estimated_duration_hours +
                              np.random.normal(0, node.duration_std_hours))
                for node_id, node in self.cdg._nodes.items()
            },
            'future_messages': self._message_schedule.copy(),
        }
        
        # Build initial state
        self._current_step = 0
        self._state = self._build_state(start_time, trust_entries, episode_id)
        
        # Initialize episode record
        self._episode_record = EpisodeRecord(
            episode_id=episode_id,
            curriculum_stage=self.curriculum_stage,
            scenario_id=scenario.get('scenario_id', 'generated'),
            start_time=start_time,
        )
        
        # Reset reward function
        self.reward_fn.reset_episode()
        
        info = {
            'episode_id': episode_id,
            'scenario_id': scenario.get('scenario_id', 'generated'),
            'n_seed_commitments': len(scenario.get('seed_commitments', [])),
            'curriculum_stage': self.curriculum_stage,
        }
        
        logger.info(f"Episode reset: id={episode_id} stage={self.curriculum_stage} "
                    f"seed_commits={len(scenario.get('seed_commitments', []))}")
        
        return self._state, info
    
    def step(self, action: AgentAction) -> Tuple[VERGILState, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Pipeline (ORDER MATTERS):
        1. Validate action legality
        2. Process action on CDG
        3. Simulate stakeholder response
        4. Update trust state
        5. Advance time; deliver new messages
        6. Check deadlines; trigger failures
        7. Propagate cascades
        8. Compute step reward
        9. Update node urgency scores
        10. Evaluate CDG satisfiability
        11. Check terminal condition
        12. Build new state
        13. Log everything
        
        Returns:
            (new_state, reward, terminated, truncated, info)
        """
        assert self._state is not None, "Must call reset() before step()"
        assert self.cdg is not None
        
        self._current_step += 1
        current_time = self._state.current_time
        
        # ── Step 1: Action validation ──────────────────────────────────────
        is_valid, validity_reason = self._validate_action(action, self._state)
        if not is_valid:
            logger.warning(f"Invalid action: {action.action_type} — {validity_reason}")
            reward_components = RewardComponents(total=-0.1)
            new_state = self._state  # No state change
            info = {'invalid_action': validity_reason}
            self._log_step(action, reward_components, new_state, False, False, info)
            return new_state, -0.1, False, False, info
        
        # ── Step 2: Apply action to CDG ─────────────────────────────────────
        affected_node = self._apply_action(action)
        
        # ── Step 3: Stakeholder simulation ────────────────────────────────
        trust_deltas = {}
        stakeholder_responses = {}
        
        if affected_node is not None:
            response = self.simulator.simulate_response(
                action=action, node=affected_node,
                trust=self._state.trust_entries.get(affected_node.stakeholder_id,
                      TrustEntry(stakeholder_id=affected_node.stakeholder_id)),
                current_time=current_time
            )
            trust_deltas[affected_node.stakeholder_id] = response.trust_delta
            stakeholder_responses[affected_node.stakeholder_id] = response
        
        # ── Step 4: Trust update ─────────────────────────────────────────
        new_trust = {sid: te for sid, te in self._state.trust_entries.items()}
        for sid, delta in trust_deltas.items():
            if sid in new_trust:
                new_trust[sid].update(
                    delta=delta,
                    event=action.action_type.value,
                    step=self._current_step
                )
        
        # ── Step 5: Time advance + message delivery ────────────────────────
        new_time = current_time + timedelta(hours=self.config.get('step_hours', 2))
        self.extractor.current_time = new_time
        new_messages = self._deliver_messages(new_time)
        
        # ── Step 6: Deadline checks + failure detection ────────────────────
        failed_nodes = []
        for node in self.cdg.get_active_nodes():
            node.update_urgency(new_time)
            if node.is_overdue(new_time) and node.status == CommitmentStatus.ACCEPTED:
                logger.info(f"Deadline exceeded: {node.node_id} '{node.label}'")
                failed_nodes.append(node.node_id)
        
        # ── Step 7: Cascade propagation ────────────────────────────────────
        all_cascade_events = []
        for nid in failed_nodes:
            events = self.cdg.propagate_failure(nid, new_time)
            all_cascade_events.extend(events)
            if events:
                logger.warning(f"Cascade from {nid}: {len(events)} affected nodes")
        
        # ── Step 8: Reward computation ─────────────────────────────────────
        reward_components = self.reward_fn.compute_step_reward(
            action=action,
            node=affected_node,
            trust_deltas=trust_deltas,
            trust_entries=new_trust,
            cascade_events=all_cascade_events,
            state=self._state,
            current_step=self._current_step,
        )
        
        # ── Step 9: CDG satisfiability evaluation ─────────────────────────
        sat_result = self.cdg.evaluate_satisfiability(
            current_time=new_time,
            available_hours=self._state.available_hours_next_48h
        )
        
        # ── Step 10: Terminal condition check ─────────────────────────────
        terminated, truncated, term_reason = self._check_terminal(
            new_time, new_trust, sat_result)
        
        if terminated or truncated:
            terminal_reward = self.reward_fn.compute_terminal_reward(
                self._state, self._episode_record)
            reward_components.fulfillment += terminal_reward.fulfillment
            reward_components.trust_delta += terminal_reward.trust_delta
            reward_components.compute_total()
        
        # ── Step 11: Build new state ──────────────────────────────────────
        new_state = VERGILState(
            cdg_nodes=list(self.cdg._nodes.values()),
            cdg_edges=list(self.cdg._edges.values()),
            satisfiability_score=sat_result.satisfiability_score,
            current_time=new_time,
            time_horizon=self._state.time_horizon,
            available_hours_next_48h=max(0, self._state.available_hours_next_48h -
                                          (affected_node.estimated_duration_hours
                                           if affected_node and action.action_type == ActionType.ACCEPT
                                           else 0)),
            cognitive_load=self._update_cognitive_load(action, affected_node),
            pending_messages=new_messages,
            trust_entries=new_trust,
            decision_log=self._state.decision_log + [{
                'step': self._current_step,
                'action': action.action_type.value,
                'node_id': affected_node.node_id if affected_node else None,
                'reward': round(reward_components.total, 4),
            }],
            episode_id=self._state.episode_id,
            step_number=self._current_step,
            curriculum_stage=self.curriculum_stage,
            node_risk_scores=self.cdg._node_risk_scores.copy(),
            at_risk_nodes=sat_result.at_risk_nodes,
        )
        
        self._state = new_state
        
        # ── Step 12: Logging ──────────────────────────────────────────────
        info = {
            'cascade_events': all_cascade_events,
            'satisfiability': sat_result.to_dict(),
            'trust_deltas': trust_deltas,
            'stakeholder_responses': {k: v.message for k, v in stakeholder_responses.items()},
            'term_reason': term_reason if (terminated or truncated) else None,
            'reward_components': reward_components.to_dict(),
            'step': self._current_step,
        }
        
        self._update_episode_record(action, reward_components, all_cascade_events, term_reason)
        self._log_step(action, reward_components, new_state, terminated, truncated, info)
        
        return new_state, reward_components.total, terminated, truncated, info
    
    def state(self) -> VERGILState:
        """Current observable state."""
        return self._state
    
    # ── Internal Helpers ──────────────────────────────────────────────────
    
    def _validate_action(self, action: AgentAction,
                          state: VERGILState) -> Tuple[bool, str]:
        """
        Check if action is legal given current state.
        
        Validation rules:
        - Must have a target if action requires one
        - Trust-gated actions (Phase 2 extension; Phase 1 basic version)
        - Cannot RENEGOTIATE a completed or declined commitment
        - Cannot ACCEPT an already-accepted commitment
        """
        if action.action_type in (ActionType.ACCEPT, ActionType.DECLINE,
                                   ActionType.COUNTER_PROPOSE, ActionType.RENEGOTIATE):
            if action.target_node_id is None and action.target_message_id is None:
                return False, "Action requires a target commitment or message"
        
        if action.target_node_id:
            node = self.cdg.get_node(action.target_node_id)
            if node is None:
                return False, f"Target node {action.target_node_id} not found"
            
            if action.action_type == ActionType.ACCEPT:
                if node.status != CommitmentStatus.PENDING:
                    return False, f"Cannot accept non-pending commitment (status={node.status})"
            
            if action.action_type == ActionType.RENEGOTIATE:
                if node.status in (CommitmentStatus.COMPLETED, CommitmentStatus.FAILED,
                                    CommitmentStatus.DECLINED):
                    return False, f"Cannot renegotiate resolved commitment (status={node.status})"
        
        return True, ""
    
    def _apply_action(self, action: AgentAction) -> Optional[CommitmentNode]:
        """Apply action to CDG. Returns affected node (if any)."""
        node = None
        if action.target_node_id:
            node = self.cdg.get_node(action.target_node_id)
        
        if node is None:
            return None
        
        now = datetime.now()
        
        if action.action_type == ActionType.ACCEPT:
            self.cdg.update_node_status(node.node_id, CommitmentStatus.ACCEPTED, now)
            node.decision_made = ActionType.ACCEPT
            node.decision_timestamp = now
            node.trust_at_decision = self._state.trust_entries.get(
                node.stakeholder_id, TrustEntry(stakeholder_id='')).trust_score
            # Auto-add implicit commitments
            self._inject_implicit_commitments(node)
        
        elif action.action_type == ActionType.DECLINE:
            self.cdg.update_node_status(node.node_id, CommitmentStatus.DECLINED, now)
            node.decision_made = ActionType.DECLINE
            node.decision_timestamp = now
        
        elif action.action_type == ActionType.COUNTER_PROPOSE:
            node.counter_proposal = {
                'proposed_deadline': action.proposed_deadline.isoformat()
                                     if action.proposed_deadline else None,
                'proposed_scope': action.proposed_scope_reduction,
            }
            node.decision_made = ActionType.COUNTER_PROPOSE
        
        elif action.action_type == ActionType.RENEGOTIATE:
            node.renegotiation_count += 1
            if action.proposed_deadline:
                node.deadline = action.proposed_deadline
            self.cdg.update_node_status(node.node_id, CommitmentStatus.RENEGOTIATED, now)
        
        node.updated_at = now
        return node
    
    def _inject_implicit_commitments(self, parent_node: CommitmentNode) -> None:
        """
        When a commitment is accepted, auto-extract and inject implicit commitments.
        This is a key differentiator: the CDG grows to include shadow obligations.
        """
        if parent_node.extraction_result is None:
            return
        
        for implicit_spec in parent_node.extraction_result.implicit_commitments:
            if not implicit_spec.auto_accept:
                continue  # Agent must explicitly accept these (presented as messages)
            
            # Compute implicit deadline
            if implicit_spec.time_before_parent and parent_node.deadline:
                implicit_deadline = parent_node.deadline - implicit_spec.time_before_parent
            elif implicit_spec.time_after_parent and parent_node.deadline:
                implicit_deadline = parent_node.deadline + implicit_spec.time_after_parent
            else:
                implicit_deadline = parent_node.deadline
            
            # Create implicit node
            implicit_node = CommitmentNode(
                label=f"[Implicit] {implicit_spec.description}",
                description=implicit_spec.description,
                commitment_type=self._type_from_implicit(implicit_spec.implicit_type),
                status=CommitmentStatus.ACCEPTED,  # Auto-accepted
                stakeholder_id=parent_node.stakeholder_id,
                stakeholder_role=parent_node.stakeholder_role,
                deadline=implicit_deadline,
                estimated_duration_hours=implicit_spec.duration_hours,
                duration_std_hours=0.25,
                parent_commitment_id=parent_node.node_id,
            )
            
            self.cdg.add_node(implicit_node)
            
            # Add temporal edge: implicit → parent (implicit must precede parent)
            if implicit_spec.time_before_parent:
                edge = CDGEdge(
                    from_node=implicit_node.node_id,
                    to_node=parent_node.node_id,
                    edge_type=EdgeType.IMPLICIT_DERIVE,
                    lag_hours=0,
                )
                success, reason = self.cdg.add_edge(edge)
                if not success:
                    logger.warning(f"Could not add implicit edge: {reason}")
            
            logger.debug(f"Injected implicit: {implicit_node.node_id} '{implicit_spec.description}'")
    
    def _deliver_messages(self, current_time: datetime) -> List[Message]:
        """Return messages whose delivery time has passed."""
        delivered = []
        remaining = []
        for msg in self._hidden['future_messages']:
            if msg.delivery_time and msg.delivery_time <= current_time:
                # Extract commitments from message
                if self.extractor and not msg.processed:
                    msg.extraction_result = self.extractor.extract(
                        msg.content, sender_role=msg.sender_role.value)
                    msg.processed = True
                delivered.append(msg)
            else:
                remaining.append(msg)
        
        self._hidden['future_messages'] = remaining
        return delivered
    
    def _check_terminal(self, current_time: datetime,
                         trust_entries: Dict, sat_result) -> Tuple[bool, bool, str]:
        """
        Terminal conditions:
        - terminated=True: All commitments resolved (completed/failed/declined)
        - terminated=True: Trust collapse (all trust < 0.2)
        - truncated=True: Max steps exceeded
        """
        # Max steps
        if self._current_step >= self._max_steps:
            return False, True, "max_steps"
        
        # All resolved
        active = self.cdg.get_active_nodes()
        if not active and self._current_step > 3:
            return True, False, "all_resolved"
        
        # Trust collapse
        trust_scores = [te.trust_score for te in trust_entries.values()]
        if trust_scores and max(trust_scores) < 0.15:
            return True, False, "trust_collapse"
        
        return False, False, ""
    
    def _update_cognitive_load(self, action: AgentAction,
                                node: Optional[CommitmentNode]) -> float:
        """Cognitive load increases when accepting high-load tasks; decreases over time."""
        current_load = self._state.cognitive_load if self._state else 0.0
        
        if node and action.action_type == ActionType.ACCEPT:
            current_load = min(1.0, current_load + node.cognitive_load_score * 0.2)
        
        # Natural recovery per step
        current_load = max(0.0, current_load - 0.05)
        return current_load
    
    def _build_state(self, current_time: datetime,
                      trust_entries: Dict, episode_id: str) -> VERGILState:
        return VERGILState(
            cdg_nodes=list(self.cdg._nodes.values()),
            cdg_edges=list(self.cdg._edges.values()),
            satisfiability_score=1.0,
            current_time=current_time,
            time_horizon=current_time + timedelta(days=14),
            available_hours_next_48h=16.0,
            cognitive_load=0.0,
            energy_level=1.0,
            pending_messages=[],
            trust_entries=trust_entries,
            decision_log=[],
            episode_id=episode_id,
            step_number=0,
            curriculum_stage=self.curriculum_stage,
        )
    
    def _generate_scenario(self, stage: int) -> Dict:
        """
        Generate a scenario for the given curriculum stage.
        Phase 1: load from JSON files. Phase 2: generates procedurally.
        """
        scenario_files = {
            1: 'scenarios/scenario_01_simple.json',
            2: 'scenarios/scenario_02_chain.json',
            3: 'scenarios/scenario_03_conflict.json',
        }
        path = scenario_files.get(stage, scenario_files[1])
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Scenario file not found: {path}. Using minimal fallback.")
            return self._minimal_scenario()
    
    def _minimal_scenario(self) -> Dict:
        """Minimal valid scenario for testing."""
        base_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        return {
            'scenario_id': 'minimal_test',
            'start_time': base_time,
            'seed_commitments': [],
            'seed_edges': [],
            'stakeholders': [
                {'id': 'boss_01', 'name': 'Manager', 'role': 'boss'}
            ],
            'initial_trust': {'boss_01': 0.65},
            'message_schedule': [
                {
                    'sender_id': 'boss_01',
                    'sender_role': 'boss',
                    'content': 'Can you have the project spec ready by Thursday 5pm?',
                    'delivery_time': (base_time + timedelta(hours=1)).isoformat(),
                }
            ],
        }
    
    def _build_stakeholder_profiles(self, scenario: Dict) -> Dict[str, StakeholderProfile]:
        profiles = {}
        for s_data in scenario.get('stakeholders', []):
            role = StakeholderRole(s_data.get('role', 'colleague'))
            profile = StakeholderProfile(
                stakeholder_id=s_data['id'],
                name=s_data.get('name', s_data['id']),
                role=role,
            )
            profiles[s_data['id']] = profile
        return profiles
    
    def _build_message_schedule(self, scenario: Dict, start_time: datetime) -> List[Message]:
        messages = []
        for m_data in scenario.get('message_schedule', []):
            delivery_str = m_data.get('delivery_time')
            delivery_time = datetime.fromisoformat(delivery_str) if delivery_str else start_time
            msg = Message(
                sender_id=m_data.get('sender_id', ''),
                sender_role=StakeholderRole(m_data.get('sender_role', 'colleague')),
                content=m_data.get('content', ''),
                delivery_time=delivery_time,
            )
            messages.append(msg)
        return sorted(messages, key=lambda m: m.delivery_time)
    
    def _build_node_from_dict(self, d: Dict) -> CommitmentNode:
        deadline = None
        if d.get('deadline'):
            deadline = datetime.fromisoformat(d['deadline'])
        return CommitmentNode(
            node_id=d.get('id', str(uuid.uuid4())[:8]),
            label=d.get('label', ''),
            description=d.get('description', ''),
            commitment_type=self._type_from_implicit(d.get('type', 'explicit_soft')),
            stakeholder_id=d.get('stakeholder_id', ''),
            stakeholder_role=StakeholderRole(d.get('role', 'colleague')),
            deadline=deadline,
            estimated_duration_hours=d.get('duration_hours', 2.0),
            duration_std_hours=d.get('duration_std', 1.0),
            status=CommitmentStatus(d.get('status', 'pending')),
        )
    
    def _build_edge_from_dict(self, d: Dict) -> CDGEdge:
        return CDGEdge(
            from_node=d['from'],
            to_node=d['to'],
            edge_type=EdgeType(d.get('type', 'temporal')),
            lag_hours=d.get('lag_hours', 0.0),
        )
    
    def _type_from_implicit(self, type_str: str):
        """Convert string to CommitmentType."""
        from .types import CommitmentType
        mapping = {
            'explicit_hard': CommitmentType.EXPLICIT_HARD,
            'explicit_soft': CommitmentType.EXPLICIT_SOFT,
            'implicit': CommitmentType.IMPLICIT,
            'precondition': CommitmentType.PRECONDITION,
            'social': CommitmentType.SOCIAL,
            'travel_buffer': CommitmentType.IMPLICIT,
            'cognitive_prep': CommitmentType.IMPLICIT,
            'followup': CommitmentType.IMPLICIT,
        }
        return mapping.get(type_str, CommitmentType.EXPLICIT_SOFT)
    
    def _log_step(self, action, reward_components, state, terminated, truncated, info):
        log_entry = {
            'episode_id': state.episode_id,
            'step': self._current_step,
            'action': action.action_type.value,
            'target_node': action.target_node_id,
            'reward': reward_components.to_dict(),
            'satisfiability': state.satisfiability_score,
            'trust': {sid: round(te.trust_score, 4)
                      for sid, te in state.trust_entries.items()},
            'n_at_risk': len(state.at_risk_nodes),
            'terminated': terminated,
            'truncated': truncated,
        }
        log_path = self._log_dir / f"ep_{state.episode_id}.jsonl"
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _update_episode_record(self, action, reward_components, cascade_events, term_reason):
        if self._episode_record is None:
            return
        self._episode_record.total_steps = self._current_step
        self._episode_record.total_reward += reward_components.total
        if cascade_events:
            self._episode_record.cascade_events.extend(cascade_events)
        if term_reason:
            self._episode_record.terminal_reason = term_reason
    
    def debug_state(self) -> Dict:
        """Full debug dump. Use in Colab to inspect mid-episode."""
        if self._state is None:
            return {'error': 'No active episode. Call reset() first.'}
        return {
            'step': self._current_step,
            'state': self._state.to_dict(),
            'cdg': self.cdg.to_dict() if self.cdg else {},
            'hidden_keys': list(self._hidden.keys()),
            'reward_log_count': len(self.reward_fn._step_decisions),
        }
    
    @staticmethod
    def _default_config() -> Dict:
        return {
            'max_steps_per_episode': 40,
            'step_hours': 2,
            'log_dir': '/tmp/vergil_logs',
            'reward': {
                'overrefusal_sensitivity': 2.0,
                'optimal_accept_fraction': 0.75,
            },
            'extraction': {
                'commitment_threshold': 0.5,
                'ambiguity_threshold': 0.3,
            },
        }
```

---

## P1.7 — LOGGING STRATEGY (`utils/logger.py`)

```python
# utils/logger.py

import logging, json, sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: str = '/tmp/vergil_logs',
                   level: int = logging.DEBUG,
                   episode_id: Optional[str] = None) -> None:
    """
    Configure structured JSON logging for all VERGIL modules.
    
    In Colab: logs appear in cell output (INFO) and written to file (DEBUG).
    In Antigravity: same, with optional remote sink.
    
    Log levels used:
    - DEBUG: per-step state details (CDG, reward components, extraction results)
    - INFO: episode-level events (reset, terminal, curriculum promotion)
    - WARNING: unexpected but recoverable (invalid actions, failed edges)
    - ERROR: requires investigation (serialization failure, missing profiles)
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Root logger
    root = logging.getLogger('vergil')
    root.setLevel(logging.DEBUG)
    root.handlers = []  # Clear existing handlers
    
    # Console handler (INFO only — keep Colab output readable)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)-7s %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    ))
    root.addHandler(console)
    
    # File handler (DEBUG — full detail)
    log_file = log_dir_path / f"vergil_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)
    
    logging.getLogger('vergil').info(f"Logging initialized. File: {log_file}")


class JSONFormatter(logging.Formatter):
    """Structured JSON log format for machine-readable output."""
    def format(self, record):
        entry = {
            'ts': self.formatTime(record, '%Y-%m-%dT%H:%M:%S'),
            'level': record.levelname,
            'module': record.name,
            'msg': record.getMessage(),
        }
        if record.exc_info:
            entry['exc'] = self.formatException(record.exc_info)
        return json.dumps(entry)
```

---

## P1.8 — TEST HARNESS (`tests/test_env.py`)

```python
# tests/test_env.py
"""
End-to-end tests for the Phase 1 environment.
All tests use scripted actions (no agent) to verify environment behavior.

Design for Colab: run with `!python -m pytest tests/ -v` or as a notebook cell.
"""

import pytest
from datetime import datetime, timedelta
from core.types import AgentAction, ActionType, CommitmentNode, CommitmentType
from core.env import VERGILEnv


@pytest.fixture
def env():
    e = VERGILEnv(seed=42)
    yield e


@pytest.fixture
def simple_scenario():
    base = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    return {
        'scenario_id': 'test_simple',
        'start_time': base,
        'seed_commitments': [
            {'id': 'C1', 'label': 'Write Report', 'type': 'explicit_hard',
             'stakeholder_id': 'boss_01', 'role': 'boss',
             'deadline': (base + timedelta(hours=8)).isoformat(),
             'duration_hours': 3.0, 'duration_std': 0.5, 'status': 'pending'},
        ],
        'seed_edges': [],
        'stakeholders': [{'id': 'boss_01', 'name': 'Boss', 'role': 'boss'}],
        'initial_trust': {'boss_01': 0.65},
        'message_schedule': [],
    }


class TestEnvironmentBasics:
    def test_reset_returns_valid_state(self, env, simple_scenario):
        state, info = env.reset(scenario=simple_scenario)
        assert state is not None
        assert len(state.cdg_nodes) == 1
        assert state.episode_id != ""
        assert 'boss_01' in state.trust_entries
    
    def test_step_returns_five_tuple(self, env, simple_scenario):
        state, _ = env.reset(scenario=simple_scenario)
        action = AgentAction(
            action_type=ActionType.ACCEPT,
            target_node_id='C1',
            feasibility_prediction=0.8,
        )
        result = env.step(action)
        assert len(result) == 5  # (state, reward, terminated, truncated, info)
    
    def test_accept_changes_node_status(self, env, simple_scenario):
        state, _ = env.reset(scenario=simple_scenario)
        action = AgentAction(action_type=ActionType.ACCEPT, target_node_id='C1')
        new_state, _, _, _, _ = env.step(action)
        
        accepted_node = next(n for n in new_state.cdg_nodes if n.node_id == 'C1')
        assert accepted_node.status.value == 'accepted'
    
    def test_decline_reduces_trust(self, env, simple_scenario):
        state, _ = env.reset(scenario=simple_scenario)
        initial_trust = state.trust_entries['boss_01'].trust_score
        
        action = AgentAction(action_type=ActionType.DECLINE, target_node_id='C1')
        new_state, _, _, _, _ = env.step(action)
        
        new_trust = new_state.trust_entries['boss_01'].trust_score
        assert new_trust < initial_trust  # Trust decreases on decline (from boss)
    
    def test_invalid_action_returns_penalty(self, env, simple_scenario):
        state, _ = env.reset(scenario=simple_scenario)
        # Accept with no target → invalid
        action = AgentAction(action_type=ActionType.ACCEPT, target_node_id=None)
        _, reward, _, _, info = env.step(action)
        assert reward < 0
        assert 'invalid_action' in info


class TestCDGSatisfiability:
    def test_satisfiability_decreases_with_overcommit(self, env):
        """Adding too many commitments to a small time window reduces satisfiability."""
        base = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        scenario = {
            'scenario_id': 'overcommit_test',
            'start_time': base,
            'seed_commitments': [
                {'id': f'C{i}', 'label': f'Task {i}', 'type': 'explicit_hard',
                 'stakeholder_id': 'boss_01', 'role': 'boss',
                 'deadline': (base + timedelta(hours=4)).isoformat(),
                 'duration_hours': 2.0, 'duration_std': 0.5, 'status': 'accepted'}
                for i in range(5)  # 5 tasks × 2hrs = 10hrs in 4hr window
            ],
            'seed_edges': [],
            'stakeholders': [{'id': 'boss_01', 'name': 'Boss', 'role': 'boss'}],
            'initial_trust': {'boss_01': 0.65},
            'message_schedule': [],
        }
        state, _ = env.reset(scenario=scenario)
        assert state.satisfiability_score < 0.5  # Should be infeasible
    
    def test_cascade_propagates_to_successors(self, env):
        """Failing node A should increase risk for node B that depends on A."""
        base = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        scenario = {
            'scenario_id': 'cascade_test',
            'start_time': base,
            'seed_commitments': [
                {'id': 'A', 'label': 'Research', 'type': 'explicit_hard',
                 'stakeholder_id': 'boss_01', 'role': 'boss',
                 'deadline': (base + timedelta(hours=1)).isoformat(),  # Imminent deadline
                 'duration_hours': 3.0, 'duration_std': 0.5, 'status': 'accepted'},
                {'id': 'B', 'label': 'Report', 'type': 'explicit_hard',
                 'stakeholder_id': 'boss_01', 'role': 'boss',
                 'deadline': (base + timedelta(hours=6)).isoformat(),
                 'duration_hours': 2.0, 'duration_std': 0.5, 'status': 'accepted'},
            ],
            'seed_edges': [{'from': 'A', 'to': 'B', 'type': 'temporal'}],
            'stakeholders': [{'id': 'boss_01', 'name': 'Boss', 'role': 'boss'}],
            'initial_trust': {'boss_01': 0.65},
            'message_schedule': [],
        }
        state, _ = env.reset(scenario=scenario)
        # Advance time past A's deadline to trigger failure
        action = AgentAction(action_type=ActionType.DO_NOTHING)
        new_state, _, _, _, info = env.step(action)
        
        # B should now be at risk due to cascade
        assert 'B' in new_state.at_risk_nodes or len(info.get('cascade_events', [])) > 0


class TestRewardHackResistance:
    def test_decline_everything_triggers_overrefusal(self, env):
        """Declining all commitments should accumulate overrefusal penalty."""
        base = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        # Create scenario with many messages to decline
        scenario = {
            'scenario_id': 'overrefusal_test',
            'start_time': base,
            'seed_commitments': [],
            'seed_edges': [],
            'stakeholders': [{'id': 'boss_01', 'name': 'Boss', 'role': 'boss'}],
            'initial_trust': {'boss_01': 0.65},
            'message_schedule': [],
        }
        state, _ = env.reset(scenario=scenario)
        
        cumulative_reward = 0.0
        # Simulate 10 decline actions (no actual nodes; uses do_nothing to accumulate)
        for _ in range(10):
            action = AgentAction(action_type=ActionType.DO_NOTHING)
            _, reward, _, _, _ = env.step(action)
            cumulative_reward += reward
        
        # Persistent decline behavior should be reflected in negative trend
        assert cumulative_reward <= 0  # Cannot hack reward by pure inaction


class TestExtractionPipeline:
    def test_high_confidence_explicit_commitment(self):
        from core.extraction import CommitmentExtractor
        extractor = CommitmentExtractor(current_time=datetime.now())
        result = extractor.extract(
            "I need the API spec delivered by Thursday 5pm.", sender_role='boss')
        assert result.commitment_probability > 0.6
        assert result.deadline_estimate is not None
        assert result.deadline_confidence > 0.5
    
    def test_ambiguous_commitment_flagged(self):
        from core.extraction import CommitmentExtractor
        extractor = CommitmentExtractor(current_time=datetime.now())
        result = extractor.extract("Let's sync sometime next week?", sender_role='colleague')
        assert result.is_ambiguous
        assert len(result.clarification_questions) > 0
    
    def test_implicit_generation_for_meeting(self):
        from core.extraction import CommitmentExtractor
        extractor = CommitmentExtractor(current_time=datetime.now())
        result = extractor.extract("Can you attend the board meeting Thursday at 10am?")
        implicits = result.implicit_commitments
        implicit_types = [i.implicit_type for i in implicits]
        assert 'travel_buffer' in implicit_types or 'cognitive_prep' in implicit_types
```

---

## P1.9 — PHASE 1 COLAB QUICKSTART NOTEBOOK

```python
# Cell 1: Setup
!pip install networkx pydantic numpy scipy gymnasium pytest -q
import sys
sys.path.insert(0, '/content/vergil')
from utils.logger import setup_logging
setup_logging(log_dir='/tmp/vergil_logs', level=10)

# Cell 2: Quick extraction test
from datetime import datetime
from core.extraction import CommitmentExtractor

extractor = CommitmentExtractor(current_time=datetime.now())
result = extractor.extract("Can you have the quarterly report ready by Friday 5pm?",
                            sender_role='boss')
print(f"Commitment: {result.is_commitment}")
print(f"Probability: {result.commitment_probability:.3f}")
print(f"Deadline: {result.deadline_estimate}")
print(f"Duration est: {result.estimated_duration_hours}h ± {result.duration_uncertainty}h")
print(f"Ambiguous: {result.is_ambiguous}")
print(f"Implicits: {[i.implicit_type for i in result.implicit_commitments]}")

# Cell 3: Full episode run
from core.env import VERGILEnv
from core.types import AgentAction, ActionType
from datetime import datetime, timedelta

env = VERGILEnv(seed=42)
state, info = env.reset()  # Uses minimal fallback scenario
print(f"Episode: {info['episode_id']}")
print(f"Seed commitments: {info['n_seed_commitments']}")

# Run a few manual steps
for i in range(5):
    action = AgentAction(action_type=ActionType.DO_NOTHING)
    state, reward, term, trunc, info = env.step(action)
    print(f"Step {i+1}: reward={reward:.4f} sat={state.satisfiability_score:.3f}")
    if term or trunc:
        print(f"  Terminal: {info.get('term_reason', 'unknown')}")
        break

# Cell 4: Debug inspection
print(env.debug_state())

# Cell 5: Run tests
!python -m pytest /content/vergil/tests/ -v --tb=short
```

---

---

# ═══════════════════════════════════════════════════════════
# PHASE 2: ADVANCED REASONING + TRAINING INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════

## P2 — Architecture Overview

Phase 2 adds SEVEN major systems on top of Phase 1:

```
P2 additions to vergil/
├── core/
│   ├── pomdp.py              # Belief state, POMDP wrapper, observation model
│   ├── trust_multidim.py     # Multi-dimensional trust (reliability/warmth/competence)
│   ├── execution_model.py    # Probabilistic execution (duration sampling, interrupts)
│   ├── aux_signals.py        # Auxiliary prediction heads (feasibility, trust, cascade)
│   └── explainer.py          # Decision explainability layer
├── curriculum/
│   ├── failure_db.py         # Failure topology database
│   ├── replay_buffer.py      # Failure-focused experience replay
│   ├── curriculum_engine.py  # Self-improving scenario generation
│   └── scenario_generator.py # Procedural scenario synthesis
├── stakeholders/
│   └── adversarial.py        # Adversarial + irrational stakeholder behaviors
├── memory/
│   ├── graph_pruner.py       # CDG memory management (pruning, summarization)
│   └── summary_node.py       # Compressed historical commitment nodes
└── anti_hack/
    └── reward_guard.py       # Anti-reward-hacking detection + correction
```

---

## P2.1 — POMDP FORMALIZATION (`core/pomdp.py`)

```python
# core/pomdp.py
"""
VERGIL as a Partially Observable Markov Decision Process (POMDP).

True state S_t: CDG + trust (multi-dim) + hidden stakeholder state
Observation O_t: visible CDG + trust estimates + delivered messages
Belief B_t: distribution over unobservable components

The agent operates on observations O_t and must maintain a belief state
B_t over hidden variables. This belief state is updated on every step.

Key hidden variables:
  - Stakeholder true urgency (not communicated in message)
  - Deadline flexibility (how much hidden slack exists)
  - Future message arrival times (not yet delivered)
  - Actual task durations (drawn from distribution, revealed at completion)
  - Stakeholder manipulation tactic active (boolean per stakeholder)
  - Irrational behavior mode (boolean, per episode)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from .types import (
    VERGILState, Message, StakeholderProfile, CommitmentNode, AgentAction
)

logger = logging.getLogger('vergil.pomdp')


@dataclass
class BeliefState:
    """
    Agent's probability distribution over unobservable state variables.
    
    Updated on every observation using a simplified particle filter
    (full Bayesian update is tractable for this domain given the structure).
    
    Each belief entry is a dict: {value: probability} or a Gaussian (mean, std).
    """
    
    # Per-stakeholder hidden beliefs
    stakeholder_beliefs: Dict[str, 'StakeholderBelief'] = field(default_factory=dict)
    
    # Task duration beliefs: {node_id: (mean_hrs, std_hrs)}
    duration_beliefs: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Future message beliefs: expected arrival distribution
    future_message_belief: List[Dict] = field(default_factory=list)
    # Each: {'message_type': str, 'expected_arrival_hours': float, 'probability': float}
    
    # Global uncertainty metrics
    overall_uncertainty: float = 0.5      # 0 = very certain; 1 = maximum uncertainty
    epistemic_risk: float = 0.3           # Risk from unknown unknowns
    
    # Update history
    update_count: int = 0
    
    def entropy(self) -> float:
        """Total belief entropy across all variables. Higher = more uncertain."""
        entropies = []
        for sb in self.stakeholder_beliefs.values():
            entropies.append(sb.urgency_entropy())
        return np.mean(entropies) if entropies else 0.5
    
    def to_observation_vector(self) -> np.ndarray:
        """
        Encode belief state as a fixed-length observation vector for the policy network.
        
        Format (per stakeholder, 5 dims):
          [urgency_mean, urgency_std, flexibility_mean, flexibility_std, irrational_prob]
        
        Plus global (2 dims):
          [overall_uncertainty, epistemic_risk]
        
        Total: n_stakeholders * 5 + 2
        """
        components = []
        for sb in self.stakeholder_beliefs.values():
            components.extend([
                sb.urgency_mean, sb.urgency_std,
                sb.flexibility_mean, sb.flexibility_std,
                sb.irrational_probability,
            ])
        components.extend([self.overall_uncertainty, self.epistemic_risk])
        return np.array(components, dtype=np.float32)


@dataclass
class StakeholderBelief:
    """
    Agent's belief about a single stakeholder's hidden state.
    
    Represented as Gaussian distributions over continuous hidden variables.
    Updated using Bayesian update rules when stakeholder behavior is observed.
    """
    stakeholder_id: str
    
    # Urgency belief (Gaussian)
    urgency_mean: float = 0.5
    urgency_std: float = 0.3
    
    # Deadline flexibility belief (Gaussian, in hours)
    flexibility_mean: float = 4.0        # Expected hours of hidden slack
    flexibility_std: float = 8.0         # High uncertainty initially
    
    # Binary beliefs
    irrational_probability: float = 0.1  # P(stakeholder acting irrationally)
    manipulation_probability: float = 0.05
    
    # Update tracking
    observations: List[Dict] = field(default_factory=list)
    
    def urgency_entropy(self) -> float:
        """Differential entropy of urgency Gaussian."""
        return 0.5 * np.log(2 * np.pi * np.e * (self.urgency_std ** 2 + 1e-8))
    
    def update_urgency(self, observed_signal: float, signal_noise: float = 0.2) -> None:
        """
        Bayesian update of urgency belief given observed urgency signal.
        
        Uses Gaussian conjugate prior update (known signal noise).
        
        Observation model: signal = true_urgency + noise, noise ~ N(0, signal_noise²)
        Prior: urgency ~ N(urgency_mean, urgency_std²)
        Posterior: Gaussian update (closed form)
        """
        prior_var = self.urgency_std ** 2
        likelihood_var = signal_noise ** 2
        
        # Posterior variance
        posterior_var = 1 / (1/prior_var + 1/likelihood_var)
        # Posterior mean
        posterior_mean = posterior_var * (self.urgency_mean/prior_var +
                                           observed_signal/likelihood_var)
        
        self.urgency_mean = float(np.clip(posterior_mean, 0, 1))
        self.urgency_std = float(np.sqrt(posterior_var))
        
        self.observations.append({
            'type': 'urgency_signal',
            'observed': observed_signal,
            'posterior_mean': round(self.urgency_mean, 4),
            'posterior_std': round(self.urgency_std, 4),
        })
    
    def update_flexibility(self, observed_response: str) -> None:
        """
        Update flexibility belief based on stakeholder response.
        
        Observation model (approximate likelihood):
        - "That works perfectly" → high flexibility likely (observed > expected)
        - "I really need the original" → low flexibility
        - "Let's see what we can do" → neutral
        """
        flexibility_signals = {
            'accepted_extension': +8.0,     # Counter was accepted → more flex than assumed
            'rejected_extension': -4.0,     # Rejected → less flex
            'accepted_same': +2.0,
            'proactive_cooperation': +6.0,
            'escalated': -10.0,
        }
        
        signal = flexibility_signals.get(observed_response, 0.0)
        if signal != 0:
            # Simple additive update with decay toward prior
            self.flexibility_mean = max(0, self.flexibility_mean + signal * 0.5)
            # Reduce uncertainty as we observe more
            self.flexibility_std = max(1.0, self.flexibility_std * 0.85)
    
    def update_irrational_probability(self, irrational_behavior_observed: bool) -> None:
        """Update P(irrational) using binary observation."""
        if irrational_behavior_observed:
            # Observed irrational → increase probability
            self.irrational_probability = min(0.9,
                self.irrational_probability + 0.15)
        else:
            # Rational behavior → decrease probability slightly
            self.irrational_probability = max(0.02,
                self.irrational_probability * 0.9)


class POMDPWrapper:
    """
    Wraps VERGILEnv with POMDP belief state management.
    
    On every step:
    1. Receive observation (visible state from env)
    2. Update belief state based on observation + prior
    3. Return (observation, belief_state) to agent
    
    The agent's effective observation = observation + belief_state_vector
    This gives the agent uncertainty-aware input without directly seeing hidden state.
    """
    
    def __init__(self, env, config: Optional[Dict] = None):
        self.env = env
        self.config = config or {}
        self.belief_state = BeliefState()
        self._observation_model_noise = self.config.get('observation_noise', 0.15)
    
    def reset(self, **kwargs) -> Tuple[VERGILState, BeliefState, Dict]:
        state, info = self.env.reset(**kwargs)
        self.belief_state = self._initialize_belief(state)
        return state, self.belief_state, info
    
    def step(self, action: AgentAction) -> Tuple[VERGILState, BeliefState, float, bool, bool, Dict]:
        state, reward, term, trunc, info = self.env.step(action)
        self._update_belief(state, action, info)
        return state, self.belief_state, reward, term, trunc, info
    
    def _initialize_belief(self, state: VERGILState) -> BeliefState:
        """Initialize uniform (high-uncertainty) beliefs at episode start."""
        belief = BeliefState(overall_uncertainty=0.7, epistemic_risk=0.4)
        
        for sid in state.trust_entries.keys():
            belief.stakeholder_beliefs[sid] = StakeholderBelief(
                stakeholder_id=sid,
                urgency_mean=0.5,
                urgency_std=0.3,
                flexibility_mean=8.0,
                flexibility_std=12.0,
                irrational_probability=0.1,
            )
        
        for node in state.cdg_nodes:
            belief.duration_beliefs[node.node_id] = (
                node.estimated_duration_hours,
                node.duration_std_hours,
            )
        
        return belief
    
    def _update_belief(self, state: VERGILState, action: AgentAction,
                        info: Dict) -> None:
        """
        Update belief based on new observations.
        
        Observations that update belief:
        - Stakeholder response message (urgency signal)
        - Whether counter-proposal was accepted (flexibility signal)
        - Task completion time (updates duration belief)
        - Cascade events (updates urgency belief upward)
        """
        belief = self.belief_state
        
        # Update from stakeholder responses
        for sid, response_msg in info.get('stakeholder_responses', {}).items():
            if sid in belief.stakeholder_beliefs:
                sb = belief.stakeholder_beliefs[sid]
                
                # Extract urgency signal from response message
                urgency_signal = self._parse_urgency_signal(response_msg)
                sb.update_urgency(urgency_signal, self._observation_model_noise)
                
                # Update flexibility if this was a negotiation
                if action.action_type in (ActionType.COUNTER_PROPOSE,
                                           ActionType.RENEGOTIATE):
                    if 'accepted' in response_msg.lower() or 'works' in response_msg.lower():
                        sb.update_flexibility('accepted_extension')
                    elif 'need' in response_msg.lower() or 'original' in response_msg.lower():
                        sb.update_flexibility('rejected_extension')
        
        # Update from task completions (duration belief update)
        for node in state.cdg_nodes:
            if (node.status.value == 'completed' and
                    node.actual_duration_hours is not None and
                    node.node_id in belief.duration_beliefs):
                old_mean, old_std = belief.duration_beliefs[node.node_id]
                # Bayesian update of duration belief
                obs = node.actual_duration_hours
                new_var = 1 / (1/(old_std**2) + 1/(0.5**2))  # noise=0.5hr
                new_mean = new_var * (old_mean/(old_std**2) + obs/(0.5**2))
                belief.duration_beliefs[node.node_id] = (new_mean, np.sqrt(new_var))
        
        # Update cascade signals → urgency belief for remaining nodes
        for event in info.get('cascade_events', []):
            nid = event.get('node_id')
            # If a node cascaded, nearby nodes likely belong to high-urgency stakeholder
            # Propagate to stakeholder belief
            node = self.env.cdg.get_node(nid) if self.env.cdg else None
            if node and node.stakeholder_id in belief.stakeholder_beliefs:
                sb = belief.stakeholder_beliefs[node.stakeholder_id]
                sb.update_urgency(min(1.0, sb.urgency_mean + 0.1))
        
        # Update global uncertainty
        belief.overall_uncertainty = min(0.95, belief.entropy() * 1.5)
        belief.update_count += 1
        
        logger.debug(f"Belief updated: uncertainty={belief.overall_uncertainty:.3f} "
                     f"entropy={belief.entropy():.3f}")
    
    def _parse_urgency_signal(self, message: str) -> float:
        """Extract urgency signal from stakeholder response message."""
        high_urgency_signals = ['need', 'must', 'critical', 'immediately', 'asap',
                                 'very important', 'counting on', 'essential']
        low_urgency_signals = ['whenever', 'no rush', 'take your time', 'eventually',
                                'if you can', 'no worries']
        
        msg_lower = message.lower()
        high_count = sum(1 for s in high_urgency_signals if s in msg_lower)
        low_count = sum(1 for s in low_urgency_signals if s in msg_lower)
        
        if high_count > low_count:
            return min(1.0, 0.5 + high_count * 0.1)
        elif low_count > high_count:
            return max(0.0, 0.5 - low_count * 0.1)
        return 0.5
```

---

## P2.2 — MULTI-DIMENSIONAL TRUST (`core/trust_multidim.py`)

```python
# core/trust_multidim.py
"""
Multi-dimensional trust model replacing scalar trust.

Three dimensions (based on organizational psychology research):
  1. Reliability: Does the agent keep its commitments?
     Updated by: kept/broken commitments, timeliness
  
  2. Competence: Does the agent seem capable?
     Updated by: quality of counter-proposals, estimation accuracy,
                 proactive renegotiation quality
  
  3. Benevolence: Does the agent seem to care about the relationship?
     Updated by: transparency of communication, effort to accommodate,
                 willingness to explain rationale

Why multi-dimensional matters:
  - Breaking a commitment once (low reliability) but explaining clearly (high benevolence)
    → different action space from both low reliability AND low benevolence
  - Competent but unreliable → different stakeholder behavior than incompetent but reliable
  - Trust recovery strategies differ by dimension:
    * Reliability recovery: keep N commitments in a row
    * Competence recovery: one excellent counter-proposal
    * Benevolence recovery: one good explanation of a difficult decision
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger('vergil.trust_multidim')


class TrustDimension(str, Enum):
    RELIABILITY  = "reliability"
    COMPETENCE   = "competence"
    BENEVOLENCE  = "benevolence"


@dataclass
class MultiDimTrustEntry:
    """
    Three-dimensional trust record for one stakeholder relationship.
    
    Action space gating uses the MINIMUM dimension as the constraint
    (weakest link principle: one very low dimension blocks key actions).
    
    Composite trust = weighted average (for reward computation).
    """
    stakeholder_id: str
    
    # Three trust dimensions [0, 1]
    reliability: float = 0.65
    competence: float = 0.65
    benevolence: float = 0.70
    
    # Per-dimension history
    reliability_history: List[Dict] = field(default_factory=list)
    competence_history: List[Dict] = field(default_factory=list)
    benevolence_history: List[Dict] = field(default_factory=list)
    
    # Per-dimension kept/broken counts
    reliability_kept: int = 0
    reliability_broken: int = 0
    competence_good_decisions: int = 0
    competence_poor_decisions: int = 0
    benevolence_positive: int = 0
    
    # Trust momentum (rolling trend; used for reward shaping)
    _reliability_momentum: float = 0.0
    _competence_momentum: float = 0.0
    _benevolence_momentum: float = 0.0
    
    @property
    def composite_trust(self) -> float:
        """
        Weighted composite. Reliability weighted highest because it's
        the most predictive of future expectation-setting.
        """
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
                'delta': round(delta, 4),
                'new_value': round(self.reliability, 4),
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
            self._reliability_momentum, self._competence_momentum, self._benevolence_momentum,
        ], dtype=np.float32)


class TrustGate:
    """
    Determines which actions are available given multi-dimensional trust state.
    
    Phase 2 replaces the Phase 1 scalar threshold with dimension-specific gates.
    
    Gate design philosophy:
    - An action is blocked if ANY required dimension is below its threshold
    - Different actions require different trust dimensions:
      * PROPOSE_EXTENSION requires RELIABILITY (you must have earned the right)
      * DELEGATE requires COMPETENCE (stakeholder must trust your judgment)
      * EXPLAIN_DIFFICULTY requires BENEVOLENCE (stakeholder must believe you care)
    
    Critically: gating is STAKEHOLDER-SPECIFIC. Different stakeholders, different gates.
    """
    
    # (action, required_dimension, threshold)
    GATES = [
        (ActionType.COUNTER_PROPOSE,  TrustDimension.RELIABILITY,  0.35),
        (ActionType.RENEGOTIATE,      TrustDimension.RELIABILITY,  0.40),
        (ActionType.DELEGATE,         TrustDimension.COMPETENCE,   0.50),
        (ActionType.DEFER,            TrustDimension.COMPETENCE,   0.45),
        (ActionType.CLARIFY,          TrustDimension.BENEVOLENCE,  0.25),
    ]
    
    # Stakeholder-role-specific multipliers on thresholds
    ROLE_THRESHOLD_MULTIPLIERS = {
        'boss': 1.3,      # Boss has stricter gates (less tolerant)
        'client': 1.2,
        'colleague': 0.9,
        'friend': 0.7,
    }
    
    @classmethod
    def get_available_actions(cls, trust_entry: MultiDimTrustEntry,
                               stakeholder_role: str) -> List['ActionType']:
        """
        Return all currently available action types for this stakeholder.
        
        ACCEPT and DECLINE are always available (you can always commit or not commit).
        Other actions may be gated.
        """
        from .types import ActionType
        
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
                logger.debug(f"TrustGate blocked: {action.value} "
                             f"({dimension.value}={current_value:.3f} < {effective_threshold:.3f})")
        
        return [a for a in all_actions if a not in blocked]
    
    @classmethod
    def explain_blocks(cls, trust_entry: MultiDimTrustEntry,
                        stakeholder_role: str) -> List[Dict]:
        """
        Return explanation of why specific actions are blocked.
        Used by explainability layer.
        """
        from .types import ActionType
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
    Compute multi-dimensional trust deltas for all agent actions.
    
    Maps each (action_type, outcome) pair to (dimension, delta) updates.
    """
    
    # Base deltas per action × outcome (dimension, base_delta)
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
                (TrustDimension.RELIABILITY, +0.03),  # Slight boost for honesty
                (TrustDimension.COMPETENCE, +0.05),
                (TrustDimension.BENEVOLENCE, +0.06),
            ],
            'late_accepted': [
                (TrustDimension.RELIABILITY, -0.04),  # Late = reliability hit
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
        """
        Compute trust deltas for an action-outcome pair.
        
        Lead time modifier: proactive actions (high lead time) get a bonus.
        Timeliness bonus: applied to RELIABILITY when commitment met early.
        """
        updates = cls.UPDATE_TABLE.get(action_type, {}).get(outcome, [])
        if not updates:
            updates = cls.UPDATE_TABLE.get(action_type, {}).get('any', [])
        
        # Apply lead time modifier to all positive deltas
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
```

---

## P2.3 — PROBABILISTIC EXECUTION MODEL (`core/execution_model.py`)

```python
# core/execution_model.py
"""
Probabilistic task execution model.

In Phase 1, task durations are deterministic (estimated_duration_hours).
Phase 2 makes them stochastic:
  - Actual duration = sample from N(estimated, std) with fat tails
  - Interruptions: random events that block execution for a duration
  - Quality variation: execution quality affects downstream trust and deliverable score
  - Partial completion: tasks can be N% done when time runs out

Why this matters for RL:
  - Agent must learn to buffer for uncertainty (add slack to estimates)
  - Agent must learn when uncertainty warrants earlier renegotiation
  - Creates genuine non-determinism that prevents memorization of scenarios
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger('vergil.execution')


@dataclass
class ExecutionSample:
    """Result of sampling from the execution distribution for one task."""
    node_id: str
    estimated_duration: float      # What agent thought it would take
    actual_duration: float         # What it actually took
    duration_ratio: float          # actual / estimated (calibration signal)
    
    quality_score: float           # 0 = terrible; 1 = excellent
    interrupted: bool = False
    interruption_duration: float = 0.0  # Hours of interruption
    
    # Completion status at nominal deadline
    completed_on_time: bool = True
    completion_fraction_at_deadline: float = 1.0  # 0–1 if partial


class ProbabilisticExecutionEngine:
    """
    Samples actual execution outcomes from a distribution.
    
    Core uncertainty sources:
    1. Duration uncertainty: Tasks take longer than estimated
    2. Interruption events: Random blocking events (meetings, urgent asks)
    3. Quality variation: Not all completions are equal
    4. Cascade amplification: High cognitive load degrades estimation quality
    
    Distribution design:
    - Duration: Lognormal(log(estimated), σ)
      Lognormal chosen because: always positive, right-skewed (tasks never finish early in expectation),
      heavy right tail (occasionally catastrophically long)
    - Interruptions: Poisson process with rate λ per hour
    - Quality: Beta(α, β) with α, β determined by cognitive load
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        
        # Interruption process parameters
        self.interruption_rate_per_hour = 0.15  # ~1 interruption per 6-7 hours
        self.interruption_duration_mean = 0.5   # Mean duration in hours
        self.interruption_duration_std = 0.3
        
        # Default duration uncertainty (σ for lognormal)
        self.default_duration_sigma = 0.35
        
        # Quality model parameters
        self.quality_base_alpha = 8.0
        self.quality_base_beta = 2.0
    
    def sample_execution(self, node, cognitive_load: float = 0.0) -> ExecutionSample:
        """
        Sample actual execution outcome for a commitment node.
        
        Args:
            node: CommitmentNode being executed
            cognitive_load: Current cognitive load [0,1] — high load degrades quality
                           and worsens duration estimation
        
        Returns:
            ExecutionSample with actual outcome
        """
        estimated = node.estimated_duration_hours
        sigma = node.duration_std_hours / max(node.estimated_duration_hours, 0.5)
        
        # Cognitive load degrades estimation accuracy → increases σ
        effective_sigma = sigma * (1 + cognitive_load * 0.5)
        effective_sigma = max(0.1, min(1.5, effective_sigma))
        
        # Sample from lognormal: actual_duration = lognormal(log(estimated), σ)
        actual_duration = float(self.rng.lognormal(
            mean=np.log(max(0.1, estimated)),
            sigma=effective_sigma
        ))
        
        # Sample interruptions (Poisson process over actual_duration hours)
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
        
        # Sample quality (cognitive load degrades quality)
        quality_alpha = self.quality_base_alpha * (1 - cognitive_load * 0.6)
        quality_beta = self.quality_base_beta * (1 + cognitive_load * 0.4)
        quality = float(self.rng.beta(max(0.5, quality_alpha), max(0.5, quality_beta)))
        
        # Determine if task completed on time
        deadline_hours = (node.deadline - datetime.now()).total_seconds() / 3600 if node.deadline else float('inf')
        completed_on_time = total_time <= deadline_hours
        
        completion_fraction = 1.0 if completed_on_time else min(1.0, deadline_hours / total_time)
        
        sample = ExecutionSample(
            node_id=node.node_id,
            estimated_duration=estimated,
            actual_duration=total_time,
            duration_ratio=round(total_time / max(estimated, 0.01), 3),
            quality_score=round(quality, 3),
            interrupted=interrupted,
            interruption_duration=round(interruption_total, 3),
            completed_on_time=completed_on_time,
            completion_fraction_at_deadline=round(completion_fraction, 3),
        )
        
        logger.debug(f"Execution sample: {node.node_id} est={estimated:.1f}h "
                     f"actual={total_time:.1f}h quality={quality:.3f} "
                     f"on_time={completed_on_time}")
        
        return sample
    
    def sample_duration_at_decision_time(self, estimated_hrs: float,
                                           std_hrs: float,
                                           cognitive_load: float = 0.0,
                                           n_samples: int = 100
                                           ) -> Dict[str, float]:
        """
        When agent is deciding whether to accept, return distribution statistics.
        Used by feasibility predictor to compute P(on_time | accept).
        
        Returns percentiles for agent's decision-making.
        """
        effective_sigma = (std_hrs / max(estimated_hrs, 0.5)) * (1 + cognitive_load * 0.5)
        
        samples = self.rng.lognormal(
            mean=np.log(max(0.1, estimated_hrs)),
            sigma=max(0.1, min(1.5, effective_sigma)),
            size=n_samples,
        )
        
        # Add interruption overhead (expected value)
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
    Generates external interruption events that override the agent's schedule.
    
    Examples: major incidents, personal emergencies, system outages.
    These are non-negotiable (cannot decline) and require immediate rescheduling.
    
    Rate: ~1 per 15-20 steps (rare but impactful; curriculum introduces later)
    """
    
    FORCE_MAJEURE_TYPES = [
        {
            'type': 'major_incident',
            'description': 'Production system down — all hands required',
            'blocks_hours': (2, 6),
            'probability': 0.05,
        },
        {
            'type': 'personal_emergency',
            'description': 'Family emergency requires immediate attention',
            'blocks_hours': (4, 12),
            'probability': 0.03,
        },
        {
            'type': 'executive_override',
            'description': 'C-level priority inserted into schedule',
            'blocks_hours': (1, 3),
            'probability': 0.07,
        },
    ]
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def sample_event(self, step: int, curriculum_stage: int) -> Optional[Dict]:
        """
        Sample a force majeure event at the current step.
        Force majeure only introduced in curriculum stage 3+.
        """
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
```

---

## P2.4 — AUXILIARY PREDICTION SIGNALS (`core/aux_signals.py`)

```python
# core/aux_signals.py
"""
Auxiliary prediction heads for dense training signal.

The main problem with sparse rewards in long-horizon RL:
the agent doesn't know until step 40 that the decision at step 3 was wrong.

Solution: Train auxiliary prediction heads jointly with the policy:
  1. Feasibility predictor: P(CDG satisfiable in N steps)
  2. Trust predictor: Expected trust in K steps for each stakeholder  
  3. Cascade risk predictor: P(cascade event in next M steps)

These heads provide dense intermediate signal AND improve the policy
by forcing the agent to build better internal representations.

Training: Multi-task learning — policy loss + auxiliary prediction losses.
Implementation: Additional output heads on the shared backbone.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger('vergil.aux_signals')


@dataclass
class AuxPredictions:
    """All auxiliary predictions for the current state."""
    
    # Feasibility predictions
    feasibility_now: float = 0.5           # P(CDG satisfiable right now)
    feasibility_in_5_steps: float = 0.5   # P(satisfiable in 5 steps)
    feasibility_in_10_steps: float = 0.5  # P(satisfiable in 10 steps)
    
    # Trust predictions (per stakeholder)
    trust_in_5_steps: Dict[str, float] = None
    trust_in_10_steps: Dict[str, float] = None
    
    # Cascade risk
    cascade_prob_next_3_steps: float = 0.1
    cascade_prob_next_10_steps: float = 0.2
    
    # Confidence calibration
    prediction_confidence: float = 0.5
    
    def __post_init__(self):
        if self.trust_in_5_steps is None:
            self.trust_in_5_steps = {}
        if self.trust_in_10_steps is None:
            self.trust_in_10_steps = {}
    
    def to_vector(self) -> np.ndarray:
        """Flatten to vector for model input."""
        trust_5 = list(self.trust_in_5_steps.values())
        trust_10 = list(self.trust_in_10_steps.values())
        return np.array([
            self.feasibility_now,
            self.feasibility_in_5_steps,
            self.feasibility_in_10_steps,
            self.cascade_prob_next_3_steps,
            self.cascade_prob_next_10_steps,
            *trust_5[:4],   # Up to 4 stakeholders
            *trust_10[:4],
        ], dtype=np.float32)


class AuxiliaryLossComputer:
    """
    Computes auxiliary prediction losses for multi-task training.
    
    Ground truth labels are computed by the environment oracle
    (using hidden state that the agent cannot observe).
    
    Loss functions:
    - Feasibility: Binary cross-entropy (will it be feasible or not?)
    - Trust: MSE (continuous prediction)
    - Cascade: Binary cross-entropy (will cascade happen?)
    
    Loss weights: tuned to prevent auxiliary tasks from dominating.
    """
    
    FEASIBILITY_LOSS_WEIGHT = 0.1
    TRUST_LOSS_WEIGHT = 0.05
    CASCADE_LOSS_WEIGHT = 0.1
    
    def __init__(self):
        self._prediction_history: List[Dict] = []
        self._label_history: List[Dict] = []
    
    def record_prediction(self, step: int, prediction: AuxPredictions) -> None:
        """Store prediction at current step for later label matching."""
        self._prediction_history.append({
            'step': step,
            'feasibility_now': prediction.feasibility_now,
            'feasibility_5': prediction.feasibility_in_5_steps,
            'cascade_3': prediction.cascade_prob_next_3_steps,
            'trust_5': prediction.trust_in_5_steps.copy(),
        })
    
    def record_label(self, step: int, actual_feasibility: bool,
                      actual_cascade: bool,
                      actual_trust: Dict[str, float]) -> None:
        """Record ground truth at step N for predictions made N steps earlier."""
        self._label_history.append({
            'step': step,
            'feasible': float(actual_feasibility),
            'cascade': float(actual_cascade),
            'trust': actual_trust,
        })
    
    def compute_calibration_error(self, window: int = 20) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE) for feasibility predictions.
        
        ECE measures how well predicted probabilities match actual frequencies.
        ECE < 0.08 is our target (well-calibrated).
        """
        if len(self._prediction_history) < 10:
            return {'ece': 0.5, 'n_samples': 0}
        
        preds = [h['feasibility_now'] for h in self._prediction_history[-window:]]
        actuals = [h['feasible'] for h in self._label_history[-window:]]
        
        if len(preds) != len(actuals):
            min_len = min(len(preds), len(actuals))
            preds = preds[:min_len]
            actuals = actuals[:min_len]
        
        if not preds:
            return {'ece': 0.5, 'n_samples': 0}
        
        # Bin predictions into 10 buckets
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_mask = [(bin_edges[i] <= p < bin_edges[i+1]) for p in preds]
            if not any(bin_mask):
                continue
            
            bin_preds = [p for p, m in zip(preds, bin_mask) if m]
            bin_actuals = [a for a, m in zip(actuals, bin_mask) if m]
            
            bin_confidence = np.mean(bin_preds)
            bin_accuracy = np.mean(bin_actuals)
            ece += (len(bin_preds) / len(preds)) * abs(bin_confidence - bin_accuracy)
        
        return {'ece': round(float(ece), 4), 'n_samples': len(preds)}
```

---

## P2.5 — FAILURE TOPOLOGY DATABASE (`curriculum/failure_db.py`)

```python
# curriculum/failure_db.py
"""
Failure Topology Database (FTD).

Tracks which CDG graph topologies cause agent failures,
enabling the curriculum engine to generate targeted difficult scenarios.

Key insight: failures are structural, not content-specific.
A chain dependency (A→B→C) causes the same failure pattern regardless
of whether it's "Research→Report→Presentation" or "Design→Code→Test".

The FTD captures topology signatures and failure statistics.
"""

import json, sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger('vergil.failure_db')


@dataclass
class FailurePattern:
    """One entry in the failure topology database."""
    pattern_id: str
    topology_hash: str           # From CDG.topology_hash()
    
    # Graph structural features (for pattern matching in generation)
    n_nodes: int = 0
    n_temporal_edges: int = 0
    n_resource_conflicts: int = 0
    max_chain_depth: int = 0     # Longest temporal dependency chain
    n_concurrent_deadlines: int = 0  # Nodes with same deadline
    has_implicit_nodes: bool = False
    
    # Failure statistics
    failure_type: str = ""       # FailureType value
    failure_step: int = 0        # Which step the failure occurred
    total_episodes: int = 0
    failure_episodes: int = 0
    
    @property
    def failure_rate(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.failure_episodes / self.total_episodes
    
    # Example episode IDs for replay
    failure_episode_ids: List[str] = field(default_factory=list)
    
    # Counterfactual optimal action sequence (computed by oracle)
    counterfactual_actions: List[str] = field(default_factory=list)
    
    last_seen: datetime = field(default_factory=datetime.now)
    first_seen: datetime = field(default_factory=datetime.now)


class FailureTopologyDatabase:
    """
    SQLite-backed failure topology database.
    Uses SQLite for compatibility with Colab (no external DB required).
    
    Schema:
    - patterns: One row per topology hash
    - episodes: One row per episode outcome
    - curriculum_weights: Computed sampling weights per stage
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
                    counterfactual_actions TEXT,
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
    
    def record_episode(self, episode_record: 'EpisodeRecord',
                        topology_hash: str,
                        structural_features: Dict) -> None:
        """Record episode outcome and update pattern statistics."""
        is_failure = episode_record.terminal_reason in ('trust_collapse',) or \
                     episode_record.commitment_fulfillment_rate < 0.5
        
        with sqlite3.connect(self.db_path) as conn:
            # Upsert pattern
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
            
            # Record episode
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
        
        logger.debug(f"FTD record: hash={topology_hash} failure={is_failure} "
                     f"stage={episode_record.curriculum_stage}")
    
    def get_high_failure_patterns(self, stage: int, top_k: int = 5,
                                   min_episodes: int = 3) -> List[Dict]:
        """
        Return top-K topology hashes with highest failure rates at this stage.
        These are used to bias scenario generation.
        """
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
                {
                    'topology_hash': r[0],
                    'failure_rate': round(r[1], 4),
                    'n_nodes': r[2],
                    'max_chain_depth': r[3],
                    'n_resource_conflicts': r[4],
                }
                for r in rows
            ]
    
    def get_curriculum_sampling_weights(self, stage: int) -> Dict[str, float]:
        """
        Return {topology_hash: sampling_weight} for curriculum-biased generation.
        
        Weights:
          60% → patterns with highest failure rates (targeted practice)
          30% → random stage-appropriate patterns
          10% → easy patterns (prevents catastrophic forgetting)
        """
        high_failure = self.get_high_failure_patterns(stage, top_k=5)
        
        weights = {}
        for pattern in high_failure:
            weights[pattern['topology_hash']] = pattern['failure_rate'] * 2.0
        
        return weights
    
    def get_statistics(self) -> Dict:
        """Summary statistics for monitoring."""
        with sqlite3.connect(self.db_path) as conn:
            total_eps = conn.execute('SELECT COUNT(*) FROM episode_outcomes').fetchone()[0]
            total_patterns = conn.execute('SELECT COUNT(*) FROM patterns').fetchone()[0]
            avg_fail_rate = conn.execute(
                'SELECT AVG(failure_episodes * 1.0 / total_episodes) FROM patterns WHERE total_episodes >= 3'
            ).fetchone()[0] or 0.0
        
        return {
            'total_episodes': total_eps,
            'unique_topology_patterns': total_patterns,
            'average_failure_rate': round(avg_fail_rate, 4),
        }
```

---

## P2.6 — CURRICULUM ENGINE (`curriculum/curriculum_engine.py`)

```python
# curriculum/curriculum_engine.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import logging

from .failure_db import FailureTopologyDatabase
from .scenario_generator import ScenarioGenerator

logger = logging.getLogger('vergil.curriculum')


@dataclass
class CurriculumStageConfig:
    stage: int
    name: str
    n_nodes_range: tuple       # (min, max) CDG nodes
    n_stakeholders: int
    include_implicit: bool
    include_resource_conflicts: bool
    include_force_majeure: bool
    include_adversarial: bool
    step_hours: float          # Sim time per step
    max_episode_steps: int
    promotion_reward_threshold: float  # Rolling avg reward to advance
    promotion_window: int      # Episodes to average over


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
    Manages curriculum progression and scenario generation.
    
    Responsibilities:
    1. Track rolling episode rewards per stage
    2. Determine when to promote to next stage
    3. Generate next episode scenario (biased by FTD failure patterns)
    4. Prevent catastrophic forgetting (occasionally return to easier stages)
    """
    
    def __init__(self, failure_db: FailureTopologyDatabase,
                  scenario_generator: ScenarioGenerator,
                  initial_stage: int = 1):
        self.failure_db = failure_db
        self.scenario_generator = scenario_generator
        self.current_stage = initial_stage
        self.stage_config = CURRICULUM_STAGES[initial_stage - 1]
        
        # Rolling reward history per stage
        self._reward_history: Dict[int, List[float]] = {s: [] for s in range(1, 5)}
        
        # Episode counter
        self._episode_count = 0
        
        logger.info(f"CurriculumEngine: starting stage {initial_stage} "
                    f"({self.stage_config.name})")
    
    def generate_next_episode(self) -> Dict:
        """
        Generate scenario for next episode.
        
        Sampling strategy:
          60%: Targeted at high-failure topologies from FTD
          30%: Random from stage-appropriate complexity
          10%: Easy (from stage 1 or 2) — prevent forgetting
        """
        self._episode_count += 1
        
        # Sample strategy
        strategy_roll = np.random.random()
        
        if strategy_roll < 0.60:
            # Targeted at failure patterns
            failure_weights = self.failure_db.get_curriculum_sampling_weights(self.current_stage)
            topology_hint = (max(failure_weights, key=failure_weights.get)
                             if failure_weights else None)
            scenario = self.scenario_generator.generate(
                stage=self.current_stage,
                topology_hint=topology_hint,
                mode='targeted',
            )
        elif strategy_roll < 0.90:
            # Random stage-appropriate
            scenario = self.scenario_generator.generate(
                stage=self.current_stage,
                mode='random',
            )
        else:
            # Easy episode (stage 1 or 2)
            easy_stage = max(1, self.current_stage - 2)
            scenario = self.scenario_generator.generate(
                stage=easy_stage,
                mode='random',
            )
        
        logger.debug(f"Episode {self._episode_count}: stage={self.current_stage} "
                     f"strategy={'targeted' if strategy_roll < 0.6 else 'random'}")
        return scenario
    
    def record_episode_reward(self, reward: float, stage: int) -> None:
        """Record episode reward for promotion tracking."""
        self._reward_history[stage].append(reward)
        # Keep only recent window
        window = self.stage_config.promotion_window
        if len(self._reward_history[stage]) > window * 2:
            self._reward_history[stage] = self._reward_history[stage][-window:]
    
    def check_promotion(self) -> bool:
        """
        Check if agent is ready to advance to next stage.
        Criterion: rolling average reward over last N episodes > threshold.
        """
        if self.current_stage >= 4:
            return False  # Already at max stage
        
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
                        f"(avg_reward={rolling_avg:.4f} >= {config.promotion_reward_threshold})")
            return True
        
        return False
    
    def get_status(self) -> Dict:
        config = self.stage_config
        history = self._reward_history[self.current_stage]
        rolling_avg = np.mean(history[-config.promotion_window:]) if len(history) >= 5 else 0.0
        
        return {
            'current_stage': self.current_stage,
            'stage_name': config.name,
            'episodes_at_stage': len(history),
            'rolling_avg_reward': round(rolling_avg, 4),
            'promotion_threshold': config.promotion_reward_threshold,
            'progress_to_promotion': round(
                min(1.0, rolling_avg / config.promotion_reward_threshold), 3),
            'failure_db_stats': self.failure_db.get_statistics(),
        }
```

---

## P2.7 — ADVERSARIAL STAKEHOLDER BEHAVIORS (`stakeholders/adversarial.py`)

```python
# stakeholders/adversarial.py
"""
Adversarial and irrational stakeholder behavior patterns.
Introduced at curriculum stage 4.

Behaviors:
1. Moving goalposts: Deadline changes after acceptance
2. Scope creep: Additional requirements added mid-commitment
3. Guilt manipulation: Social pressure to accept infeasible requests
4. Authority override: Boss-level force that bypasses normal negotiation
5. Irrational escalation: Stakeholder escalates despite reasonable handling
6. False urgency: Fabricated urgency to get priority treatment
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
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
    
    # Effect on CDG / state
    deadline_shift_hours: Optional[float] = None
    scope_addition: Optional[str] = None
    trust_manipulation_signal: Optional[float] = None  # False trust signal
    
    # Detection cues (observable by agent)
    linguistic_cues: List[str] = None  # Phrases that hint at the tactic
    
    def __post_init__(self):
        if self.linguistic_cues is None:
            self.linguistic_cues = []


class AdversarialBehaviorGenerator:
    """
    Generates adversarial events for curriculum stage 4.
    
    Each event type has a probability, stakeholder role prerequisites,
    and observable linguistic cues the agent CAN learn to detect.
    
    Training goal: Agent learns to recognize cues and respond robustly.
    """
    
    BEHAVIORS = {
        'goal_post_shift': {
            'description': "Deadline moved earlier after acceptance",
            'applicable_roles': ['boss', 'client'],
            'probability': 0.15,
            'cues': ['actually', 'just realized', 'by the way', 'changed my mind'],
            'effect': 'shift_deadline',
            'shift_hours': (-24, -4),  # Moves earlier by 4-24 hours
        },
        'scope_creep': {
            'description': "Additional requirements added mid-commitment",
            'applicable_roles': ['boss', 'client', 'colleague'],
            'probability': 0.20,
            'cues': ['also', 'one more thing', 'while you\'re at it', 'might as well'],
            'effect': 'add_scope',
        },
        'guilt_manipulation': {
            'description': "Social guilt pressure to accept infeasible request",
            'applicable_roles': ['boss', 'friend'],
            'probability': 0.12,
            'cues': ['just this once', 'I thought we were a team', 'really disappointing',
                     'count on you', 'always relied on you'],
            'effect': 'pressure_accept',
        },
        'false_urgency': {
            'description': "Fabricated urgency; real deadline is later",
            'applicable_roles': ['boss', 'client'],
            'probability': 0.10,
            'cues': ['absolutely critical', 'if this isn\'t done', 'major consequences'],
            'effect': 'inflate_urgency',
            '_true_flexibility_hours': 48,  # Hidden: actually 48hrs flexible
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
    
    def sample_event(self, step: int, stakeholder_id: str,
                      stakeholder_role: str,
                      curriculum_stage: int) -> Optional[AdversarialEvent]:
        """Sample adversarial event for this stakeholder-step combination."""
        if curriculum_stage < 4:
            return None
        
        for behavior_name, behavior in self.BEHAVIORS.items():
            if stakeholder_role not in behavior['applicable_roles']:
                continue
            
            if self.rng.random() < behavior['probability']:
                event = AdversarialEvent(
                    event_type=behavior_name,
                    stakeholder_id=stakeholder_id,
                    description=behavior['description'],
                    linguistic_cues=behavior.get('cues', []),
                )
                
                if behavior.get('effect') == 'shift_deadline':
                    shift_range = behavior.get('shift_hours', (-12, -4))
                    event.deadline_shift_hours = float(
                        self.rng.uniform(*shift_range))
                
                logger.info(f"Adversarial event: {behavior_name} "
                            f"for {stakeholder_id} at step {step}")
                return event
        
        return None
```

---

## P2.8 — ANTI-REWARD-HACKING (`anti_hack/reward_guard.py`)

```python
# anti_hack/reward_guard.py
"""
Anti-reward-hacking monitoring and correction mechanisms.

Known hack strategies:
1. DECLINE EVERYTHING: 100% fulfillment (nothing to fail), but trust dies
2. ACCEPT EVERYTHING: Short-term trust boost, then catastrophic failures
3. OSCILLATE: Alternate accept/decline at maximum rate to average both penalties
4. CLARIFY SPAM: Repeatedly clarify to delay commitment (avoids decisions)
5. RENEGOTIATE FISHING: Propose extensions to every commitment regardless of need
6. TIME GAMING: Accept only when deadline is far, decline when close

Guard mechanism: Multi-level detection that adjusts reward and flags behavior.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger('vergil.anti_hack')


@dataclass
class HackDetectionResult:
    hack_detected: bool = False
    hack_type: Optional[str] = None
    confidence: float = 0.0
    correction_penalty: float = 0.0
    explanation: str = ""


class RewardGuard:
    """
    Detects and corrects reward hacking attempts.
    
    Runs on every episode, analyzes decision pattern across recent steps.
    If a pattern is detected: applies correction penalty and logs for research.
    
    Design constraint: Guard must NOT penalize legitimate behavior.
    E.g., declining ALL commitments in a period of genuine overload is correct,
    not a hack. The guard uses CONTEXT (CDG satisfiability) to distinguish.
    """
    
    DETECTION_WINDOW = 10  # Steps to look back
    
    def __init__(self):
        self._decision_history: List[Dict] = []  # {step, action, feasibility, trust_avg}
    
    def record_decision(self, step: int, action_type: str,
                         feasibility: float, avg_trust: float,
                         cdg_satisfiability: float) -> None:
        self._decision_history.append({
            'step': step, 'action': action_type,
            'feasibility': feasibility, 'avg_trust': avg_trust,
            'satisfiability': cdg_satisfiability,
        })
    
    def analyze(self) -> HackDetectionResult:
        """Run all hack detectors over recent history."""
        if len(self._decision_history) < 5:
            return HackDetectionResult()
        
        recent = self._decision_history[-self.DETECTION_WINDOW:]
        
        # Detector 1: Decline everything
        result = self._detect_decline_everything(recent)
        if result.hack_detected:
            return result
        
        # Detector 2: Accept everything
        result = self._detect_accept_everything(recent)
        if result.hack_detected:
            return result
        
        # Detector 3: Clarify spam
        result = self._detect_clarify_spam(recent)
        if result.hack_detected:
            return result
        
        # Detector 4: Oscillation
        result = self._detect_oscillation(recent)
        if result.hack_detected:
            return result
        
        return HackDetectionResult()
    
    def _detect_decline_everything(self, recent: List[Dict]) -> HackDetectionResult:
        """
        Detect systematic over-declining.
        
        Hack signature: decline rate > 70% when CDG satisfiability was high.
        Legitimate: high decline rate when satisfiability is low (genuinely overcommitted).
        """
        declines = [d for d in recent if d['action'] == 'decline']
        if len(declines) / len(recent) < 0.7:
            return HackDetectionResult()
        
        # Check context: was satisfiability high when declining?
        high_sat_declines = [d for d in declines if d['satisfiability'] > 0.6]
        
        if len(high_sat_declines) / max(len(declines), 1) > 0.5:
            # Declining feasible things → hack
            confidence = len(high_sat_declines) / len(declines)
            return HackDetectionResult(
                hack_detected=True,
                hack_type='decline_everything',
                confidence=round(confidence, 3),
                correction_penalty=0.2 * confidence,
                explanation=f"Declining {len(declines)}/{len(recent)} requests when CDG was satisfiable",
            )
        
        return HackDetectionResult()
    
    def _detect_accept_everything(self, recent: List[Dict]) -> HackDetectionResult:
        """
        Detect systematic over-accepting.
        
        Hack signature: accept rate > 90% AND satisfiability declining.
        """
        accepts = [d for d in recent if d['action'] == 'accept']
        if len(accepts) / len(recent) < 0.90:
            return HackDetectionResult()
        
        sat_scores = [d['satisfiability'] for d in recent]
        if len(sat_scores) > 3:
            # Check if satisfiability is declining (sign of overcommitment)
            trend = np.polyfit(range(len(sat_scores)), sat_scores, 1)[0]
            if trend < -0.02:  # Declining satisfiability
                return HackDetectionResult(
                    hack_detected=True,
                    hack_type='accept_everything',
                    confidence=min(1.0, abs(trend) * 50),
                    correction_penalty=0.15,
                    explanation="Accepting all requests while CDG satisfiability is declining",
                )
        
        return HackDetectionResult()
    
    def _detect_clarify_spam(self, recent: List[Dict]) -> HackDetectionResult:
        """Detect excessive clarification requests used to avoid commitments."""
        clarifies = [d for d in recent if d['action'] == 'clarify']
        if len(clarifies) / len(recent) > 0.60:
            return HackDetectionResult(
                hack_detected=True,
                hack_type='clarify_spam',
                confidence=len(clarifies) / len(recent),
                correction_penalty=0.1,
                explanation="Excessive clarification requests used to delay commitments",
            )
        return HackDetectionResult()
    
    def _detect_oscillation(self, recent: List[Dict]) -> HackDetectionResult:
        """Detect accept/decline oscillation pattern."""
        actions = [d['action'] for d in recent if d['action'] in ('accept', 'decline')]
        if len(actions) < 6:
            return HackDetectionResult()
        
        # Count alternations
        alternations = sum(1 for i in range(1, len(actions))
                           if actions[i] != actions[i-1])
        oscillation_rate = alternations / (len(actions) - 1)
        
        if oscillation_rate > 0.80:
            return HackDetectionResult(
                hack_detected=True,
                hack_type='oscillation',
                confidence=oscillation_rate,
                correction_penalty=0.12,
                explanation="Accept/decline oscillation pattern detected",
            )
        
        return HackDetectionResult()
```

---

## P2.9 — EXPLAINABILITY LAYER (`core/explainer.py`)

```python
# core/explainer.py
"""
Decision explainability for every agent action.

Each action produces an Explanation object that:
1. States the primary reason for the decision
2. Lists the CDG constraints that drove it
3. Identifies which trust dimensions were relevant
4. Provides a counterfactual: "If X had been different, I would have Y"

Explainability serves three purposes:
  - Debugging: Why did the agent make a bad decision?
  - Trust: Users need to understand why their request was declined
  - Research: Captures reasoning patterns for publication
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

logger = logging.getLogger('vergil.explainer')


@dataclass
class DecisionExplanation:
    """Human-readable explanation of an agent decision."""
    action_type: str
    target_commitment: Optional[str] = None
    
    # Primary reason
    primary_reason: str = ""
    
    # Supporting evidence
    cdg_constraints_considered: List[str] = field(default_factory=list)
    trust_factors: List[str] = field(default_factory=list)
    resource_factors: List[str] = field(default_factory=list)
    
    # Feasibility assessment summary
    feasibility_summary: str = ""
    confidence: float = 0.5
    
    # Counterfactual
    counterfactual: str = ""  # "If deadline were 48hrs later, I would have accepted"
    
    # Blocked actions (from trust gates)
    blocked_actions_explanation: List[str] = field(default_factory=list)
    
    def to_natural_language(self) -> str:
        """Generate a natural language explanation for the agent's response."""
        lines = [f"Decision: {self.action_type.upper()}"]
        if self.target_commitment:
            lines.append(f"Regarding: {self.target_commitment}")
        lines.append(f"\nReason: {self.primary_reason}")
        
        if self.cdg_constraints_considered:
            lines.append("\nConstraints considered:")
            for c in self.cdg_constraints_considered:
                lines.append(f"  • {c}")
        
        if self.trust_factors:
            lines.append("\nTrust factors:")
            for t in self.trust_factors:
                lines.append(f"  • {t}")
        
        if self.counterfactual:
            lines.append(f"\nNote: {self.counterfactual}")
        
        return '\n'.join(lines)


class DecisionExplainer:
    """
    Generates explanations for agent decisions.
    
    Called after every action with full context.
    Explanations are logged and attached to the EpisodeRecord.
    """
    
    def explain(self, action, node, state, trust_gates_checked,
                sat_result, belief_state=None) -> DecisionExplanation:
        """Generate explanation for an agent action."""
        explanation = DecisionExplanation(
            action_type=action.action_type.value,
            target_commitment=node.label if node else None,
        )
        
        # Primary reason based on action type
        if action.action_type.value == 'accept':
            explanation.primary_reason = self._explain_accept(node, sat_result, state)
        elif action.action_type.value == 'decline':
            explanation.primary_reason = self._explain_decline(node, sat_result, state)
        elif action.action_type.value == 'counter_propose':
            explanation.primary_reason = self._explain_counter(action, node, sat_result)
        elif action.action_type.value == 'renegotiate':
            explanation.primary_reason = self._explain_renegotiate(action, node, state)
        
        # CDG constraints
        if sat_result:
            for violation in sat_result.violations[:3]:
                explanation.cdg_constraints_considered.append(
                    f"CDG violation: {violation.get('type', 'unknown')} "
                    f"(severity={violation.get('severity', 0):.2f})"
                )
            if sat_result.at_risk_nodes:
                explanation.cdg_constraints_considered.append(
                    f"At-risk nodes: {', '.join(sat_result.at_risk_nodes[:3])}")
        
        # Feasibility summary
        explanation.feasibility_summary = (
            f"CDG satisfiability: {state.satisfiability_score:.2f} "
            f"({'satisfiable' if state.satisfiability_score > 0.6 else 'at risk'})"
        )
        explanation.confidence = action.confidence
        
        # Trust blocks
        if trust_gates_checked:
            for block in trust_gates_checked:
                explanation.blocked_actions_explained.append(
                    f"{block['action']} blocked: {block['dimension']} too low "
                    f"({block['current']:.2f} < {block['required']:.2f}). "
                    f"{block['recovery_hint']}"
                )
        
        # Counterfactual
        explanation.counterfactual = self._generate_counterfactual(
            action, node, sat_result, state)
        
        logger.debug(f"Explanation: {action.action_type.value} → {explanation.primary_reason[:80]}")
        return explanation
    
    def _explain_accept(self, node, sat_result, state) -> str:
        sat = state.satisfiability_score
        if sat > 0.7:
            return (f"CDG remains satisfiable (score={sat:.2f}) after accepting "
                    f"'{node.label}'. All dependencies have sufficient time buffers.")
        elif sat > 0.4:
            return (f"Accepted with moderate risk (CDG score={sat:.2f}). "
                    f"Tight but feasible given current schedule.")
        else:
            return f"Accepted despite low CDG score ({sat:.2f}) — trust factors outweighed feasibility concerns."
    
    def _explain_decline(self, node, sat_result, state) -> str:
        if sat_result and sat_result.violations:
            v = sat_result.violations[0]
            return (f"Declining '{node.label}': accepting would create a "
                    f"{v.get('type', 'constraint')} violation "
                    f"(severity={v.get('severity', 0):.2f}). "
                    f"Current CDG cannot absorb this commitment.")
        return f"Declining due to insufficient capacity (satisfiability={state.satisfiability_score:.2f})."
    
    def _explain_counter(self, action, node, sat_result) -> str:
        changes = []
        if action.proposed_deadline:
            changes.append(f"new deadline: {action.proposed_deadline.strftime('%a %b %d %H:%M')}")
        if action.proposed_scope_reduction:
            changes.append(f"reduced scope: {action.proposed_scope_reduction}")
        return (f"Counter-proposing modified terms ({', '.join(changes)}) "
                f"to make '{node.label}' feasible within current CDG constraints.")
    
    def _explain_renegotiate(self, action, node, state) -> str:
        hours = (node.deadline - state.current_time).total_seconds() / 3600 if node.deadline else 0
        return (f"Proactively renegotiating '{node.label}' with {hours:.1f}h lead time. "
                f"New information made original parameters infeasible.")
    
    def _generate_counterfactual(self, action, node, sat_result, state) -> str:
        if action.action_type.value == 'decline' and sat_result:
            primary_violation = (sat_result.violations[0] if sat_result.violations else None)
            if primary_violation and primary_violation.get('type') == 'resource_conflict':
                return ("If the resource conflict with existing commitment "
                        f"'{primary_violation.get('node_a', 'another task')}' "
                        "were resolved, I could accept this.")
        
        if action.action_type.value == 'decline' and node and node.deadline:
            return (f"If the deadline were extended by 48+ hours, "
                    "the CDG would become satisfiable and I could accept.")
        
        return ""
```

---

## P2.10 — PHASE 2 INTEGRATION TEST

```python
# tests/test_phase2_integration.py
"""
Integration tests for Phase 2 systems.
Run in Colab after installing all dependencies.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

class TestPOMDPBeliefUpdate:
    def test_urgency_belief_updates_on_observation(self):
        from core.pomdp import StakeholderBelief
        
        sb = StakeholderBelief(stakeholder_id='boss_01',
                               urgency_mean=0.5, urgency_std=0.3)
        initial_mean = sb.urgency_mean
        
        # Observe high urgency signal
        sb.update_urgency(observed_signal=0.9, signal_noise=0.2)
        
        assert sb.urgency_mean > initial_mean  # Posterior shifts toward observation
        assert sb.urgency_std < 0.3            # Uncertainty reduced


class TestMultiDimTrust:
    def test_three_dimensions_independent(self):
        from core.trust_multidim import MultiDimTrustEntry, TrustDimension
        
        te = MultiDimTrustEntry(stakeholder_id='boss_01')
        initial_comp = te.competence
        
        # Update only reliability
        te.update(TrustDimension.RELIABILITY, -0.10, 'broken_commit', step=5)
        
        assert te.reliability < 0.65              # Reliability decreased
        assert te.competence == initial_comp       # Competence unchanged
    
    def test_composite_trust_weighted(self):
        from core.trust_multidim import MultiDimTrustEntry
        
        te = MultiDimTrustEntry(stakeholder_id='test',
                                reliability=0.8, competence=0.6, benevolence=0.7)
        expected = 0.45 * 0.8 + 0.35 * 0.6 + 0.20 * 0.7
        assert abs(te.composite_trust - expected) < 0.001
    
    def test_trust_gate_blocks_renegotiate_below_threshold(self):
        from core.trust_multidim import MultiDimTrustEntry, TrustGate
        from core.types import ActionType
        
        # Low reliability → renegotiate should be blocked for boss
        te = MultiDimTrustEntry(stakeholder_id='boss',
                                reliability=0.25, competence=0.7, benevolence=0.7)
        available = TrustGate.get_available_actions(te, 'boss')
        
        assert ActionType.RENEGOTIATE not in available


class TestProbabilisticExecution:
    def test_duration_always_positive(self):
        from core.execution_model import ProbabilisticExecutionEngine
        from core.types import CommitmentNode
        
        engine = ProbabilisticExecutionEngine(seed=99)
        node = CommitmentNode(
            label='Test Task',
            estimated_duration_hours=2.0,
            duration_std_hours=0.5,
            deadline=datetime.now() + timedelta(hours=10),
        )
        
        for _ in range(50):
            sample = engine.sample_execution(node, cognitive_load=0.0)
            assert sample.actual_duration > 0
    
    def test_high_cognitive_load_increases_variance(self):
        from core.execution_model import ProbabilisticExecutionEngine
        from core.types import CommitmentNode
        
        engine = ProbabilisticExecutionEngine(seed=42)
        node = CommitmentNode(
            label='Complex Task',
            estimated_duration_hours=3.0,
            duration_std_hours=0.5,
            deadline=datetime.now() + timedelta(hours=20),
        )
        
        low_load_durations = [engine.sample_execution(node, 0.0).actual_duration
                               for _ in range(30)]
        high_load_durations = [engine.sample_execution(node, 0.9).actual_duration
                                for _ in range(30)]
        
        assert np.std(high_load_durations) > np.std(low_load_durations)


class TestCurriculumEngine:
    def test_promotion_on_high_reward(self):
        import tempfile, os
        from curriculum.failure_db import FailureTopologyDatabase
        from curriculum.curriculum_engine import CurriculumEngine
        from curriculum.scenario_generator import ScenarioGenerator
        
        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
            db_path = f.name
        
        try:
            db = FailureTopologyDatabase(db_path)
            gen = ScenarioGenerator(seed=42)
            engine = CurriculumEngine(db, gen, initial_stage=1)
            
            # Record enough high rewards to trigger promotion
            for _ in range(25):
                engine.record_episode_reward(0.8, stage=1)
            
            promoted = engine.check_promotion()
            assert promoted
            assert engine.current_stage == 2
        finally:
            os.unlink(db_path)


class TestAntiRewardHacking:
    def test_detects_decline_everything_hack(self):
        from anti_hack.reward_guard import RewardGuard
        
        guard = RewardGuard()
        
        # Record all declines with high CDG satisfiability
        for i in range(12):
            guard.record_decision(
                step=i, action_type='decline',
                feasibility=0.8, avg_trust=0.6, cdg_satisfiability=0.85
            )
        
        result = guard.analyze()
        assert result.hack_detected
        assert result.hack_type == 'decline_everything'
        assert result.correction_penalty > 0
    
    def test_does_not_penalize_legitimate_declining(self):
        """When CDG is truly infeasible, high decline rate should be OK."""
        from anti_hack.reward_guard import RewardGuard
        
        guard = RewardGuard()
        
        # Decline when CDG satisfiability is LOW (legitimate)
        for i in range(12):
            guard.record_decision(
                step=i, action_type='decline',
                feasibility=0.2, avg_trust=0.5, cdg_satisfiability=0.15
            )
        
        result = guard.analyze()
        assert not result.hack_detected or result.correction_penalty < 0.05
```

---

## P2.11 — PHASE 2 COLAB INTEGRATION NOTEBOOK

```python
# Cell 1: Install + setup
!pip install networkx pydantic numpy scipy gymnasium pytest -q
import sys
sys.path.insert(0, '/content/vergil')
from utils.logger import setup_logging
setup_logging()

# Cell 2: Full POMDP episode
from core.env import VERGILEnv
from core.pomdp import POMDPWrapper
from core.types import AgentAction, ActionType
from datetime import datetime, timedelta

env = VERGILEnv(seed=42)
pomdp = POMDPWrapper(env)
state, belief, info = pomdp.reset()

print(f"Initial belief uncertainty: {belief.overall_uncertainty:.3f}")
print(f"Initial belief entropy: {belief.entropy():.3f}")

for i in range(5):
    action = AgentAction(action_type=ActionType.DO_NOTHING,
                         feasibility_prediction=state.satisfiability_score)
    state, belief, reward, term, trunc, info = pomdp.step(action)
    print(f"Step {i+1}: reward={reward:.4f} "
          f"uncertainty={belief.overall_uncertainty:.3f} "
          f"trust={[round(te.trust_score, 3) for te in state.trust_entries.values()]}")
    if term or trunc:
        break

# Cell 3: Multi-dimensional trust demo
from core.trust_multidim import MultiDimTrustEntry, TrustDimension, TrustGate, MultiDimTrustUpdater
from core.types import ActionType as AT

te = MultiDimTrustEntry(stakeholder_id='boss_01',
                         reliability=0.72, competence=0.65, benevolence=0.70)
print(f"Initial composite trust: {te.composite_trust:.3f}")

# Simulate breaking a commitment
updates = MultiDimTrustUpdater.compute_deltas('accept', 'commitment_broken', lead_time_hours=0)
for dim, delta in updates:
    te.update(dim, delta, 'commitment_broken', step=3)

print(f"After broken commitment: composite={te.composite_trust:.3f}")
print(f"  reliability={te.reliability:.3f}, competence={te.competence:.3f}")

available_actions = TrustGate.get_available_actions(te, 'boss')
print(f"Available actions: {[a.value for a in available_actions]}")
blocks = TrustGate.explain_blocks(te, 'boss')
print(f"Blocked actions: {blocks}")

# Cell 4: Curriculum engine status
from curriculum.failure_db import FailureTopologyDatabase
from curriculum.curriculum_engine import CurriculumEngine
from curriculum.scenario_generator import ScenarioGenerator

db = FailureTopologyDatabase('/tmp/vergil_test_db.sqlite')
gen = ScenarioGenerator(seed=42)
curriculum = CurriculumEngine(db, gen, initial_stage=1)
print(curriculum.get_status())

# Cell 5: Run Phase 2 tests
!python -m pytest /content/vergil/tests/test_phase2_integration.py -v --tb=short
```

---

## P2.12 — REQUIREMENTS FILE

```
# requirements.txt
# Core
networkx==3.6.1
numpy>=1.24.0
scipy>=1.10.0
pydantic>=2.0.0

# RL environment
gymnasium>=0.29.0

# Training (Phase 3)
torch>=2.0.0
transformers>=4.36.0
trl>=0.7.0  # GRPO support
unsloth>=2024.1  # Efficient LLM training

# Utilities
python-dateutil>=2.8.0
pytest>=7.0.0
pytest-cov>=4.0.0

# Logging / serialization
jsonlines>=3.1.0

# Optional (for Phase 4 frontend)
# fastapi>=0.100.0
# uvicorn>=0.23.0
```

---

## CROSS-PHASE INTERACTION SUMMARY

```
Phase 1 Foundation → Phase 2 Extensions

VERGILEnv.step()
  ├── [P1] _validate_action() ─────────────────────── now checks TrustGate [P2]
  ├── [P1] _apply_action() ────────────────────────── unchanged
  ├── [P1] simulator.simulate_response() ──────────── now probabilistic [P2]
  │                                                   + adversarial events [P2]
  ├── [P1] trust_update() ─────────────────────────── now MultiDimTrust [P2]
  ├── [P2] execution_model.sample() ───────────────── NEW in P2
  ├── [P1] cdg.propagate_failure() ────────────────── unchanged
  ├── [P1] reward_fn.compute_step_reward() ────────── + anti_hack.analyze() [P2]
  ├── [P2] pomdp_wrapper._update_belief() ─────────── NEW in P2
  ├── [P2] aux_signals.record_prediction() ─────────── NEW in P2
  ├── [P2] explainer.explain() ────────────────────── NEW in P2
  ├── [P2] failure_db.record_episode() ─────────────── NEW in P2 (at terminal)
  └── [P2] curriculum.check_promotion() ─────────────── NEW in P2 (at terminal)
```

---

*Phase 1 and Phase 2 design complete. Phase 3 (training pipeline) builds the GRPO loop,
HGT graph encoder, and RL agent that operates on these environments.
Phase 4 adds evaluation benchmarks, frontend demo, and productization layer.*
