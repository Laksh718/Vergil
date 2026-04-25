# VERGIL: Commitment Dependency Graph Engine
## Complete System Design — OpenEnv Training Environment & Product

> *"Models that can't say no intelligently will never be trusted with anything important."*

---

## STEP 1: DEEP PROBLEM UNDERSTANDING

### The True Capability Gap

Current LLMs fail at commitment reasoning for a cluster of deeply interconnected reasons — not one flaw, but a **compound failure of four distinct cognitive capabilities** that humans blend unconsciously:

**1. Absence of Prospective Memory**
Humans "pre-experience" future states before committing. When you say "I'll be at your wedding on the 12th," your brain runs a rapid simulation: *do I have flights? Is that the weekend after the conference? Will I be exhausted?* LLMs have no such simulation substrate. Their "future" is just predicted tokens, not a reasoned feasibility check against a persistent world model.

**2. Constraint Propagation Blindness**
Commitment A doesn't just occupy time slot T. It creates a **constraint shadow**: travel time before it, recovery time after it, cognitive load during it, and implicit follow-up obligations that inherit from it. LLMs treat each commitment as isolated. Real commitments are nodes in a constraint satisfaction problem (CSP) — accepting one reshapes the feasibility of all others.

**3. Social Inference Asymmetry**
When a boss sends "Can you have this by Thursday?" the literal request is a deadline. But the true ask encodes: urgency level, relationship stakes, implied effort expectation, and what "Thursday" means in this org's culture (COB? midnight?). LLMs extract the surface commitment but miss the **pragmatic implicature layer** that governs how the commitment should be weighted.

**4. No Non-Markovian Trust Tracking**
Human professional behavior is deeply history-dependent. Whether you can ask for an extension from your boss on Friday depends on whether you delivered last Tuesday. LLMs have no persistent trust register — every interaction is memoryless in exactly the dimension that matters most for commitment negotiation.

### Core Scientific Classification

This is simultaneously three problem types, and that's what makes it hard:

| Layer | Problem Class | Current Best Approach | Gap |
|---|---|---|---|
| Feasibility Evaluation | Constraint Satisfaction (CSP) | Mixed-Integer Programming | LLMs can't propagate constraints over dynamic graphs |
| Timeline Management | Temporal Planning (STRIPS/HTN) | Classical planners | No social uncertainty modelling |
| Social Negotiation | Mechanism Design + ToM | Game theory | Agents lack persistent belief about others' beliefs |
| Trust Dynamics | Non-Markovian RL | LSTM/Transformer memory | No environment that *rewards* trust management |

The **non-obvious insight**: this is not primarily a planning problem. Planning is tractable. The hard part is *social feasibility under uncertainty*, where the constraints are partially hidden inside other people's minds, and the cost function is trust — a slow-moving, hard-to-observe, highly nonlinear signal.

### Mapping to OpenEnv Themes

- **T2 (Long-Horizon Planning)**: CDG episodes span 20-100 steps, with decisions at step 3 producing consequences at step 47
- **T3.2 (Personalized Assistants)**: Trust network is the personalization layer — same commitment, different action depending on relationship history
- **T4 (Self-Improving Environments)**: Failure topology tracking enables curriculum generation without human labeling

### Real-World Analogs That Inform Design

| Analog | What VERGIL Borrows |
|---|---|
| Executive Assistants | Commitment extraction from natural language; relationship-weighted prioritization |
| Air Traffic Control | Hard constraint satisfaction with cascading failure propagation |
| Supply Chain Management | Dependency graph structure; buffer management; ripple failure modelling |
| Poker Professionals | Expected value reasoning under social uncertainty; position-aware commitment |
| Hostage Negotiators | Trust repair as an active skill; strategic communication under constraint |

---

## STEP 2: AGENT + STAKEHOLDER MODELING

### Primary Agent: VERGIL

**Role**: Personal/professional AI assistant managing commitments on behalf of a user (the "Principal")

**Capabilities**:
- Extract structured commitments from unstructured messages
- Maintain and query the CDG
- Predict feasibility of new commitments before acceptance
- Generate natural language accept/decline/counter-propose responses
- Monitor commitment execution and trigger proactive renegotiation

**Hidden Internal State** (not directly observable by other agents):
- Current CDG snapshot
- Energy/cognitive load model of the Principal
- Trust scores per relationship
- Feasibility confidence per commitment

### Stakeholder Taxonomy

#### Tier 1: High-Trust, High-Stakes
**The Boss (Adversarial Authority)**
- *Goals*: Task completion on deadlines; perceives cancellations as incompetence
- *Constraints*: Has authority to override; limited visibility into agent's full schedule
- *Behavior Patterns*: Sends "quick asks" that are actually large; moves deadlines without notice; interprets silence as acceptance
- *Trust Dynamics*: Trust decays 3× faster than it builds; asymmetric forgiveness
- *Simulation*: Boss model has latent urgency score (0-1) that isn't always communicated; agent must infer it from message tone features

**The Client (High-Stakes External)**
- *Goals*: Delivery on contractual terms; dislikes surprises
- *Constraints*: Can terminate relationship; limited context on internal constraints
- *Behavior Patterns*: Scope creep disguised as "small additions"; escalation-prone
- *Trust Dynamics*: Binary cliff — high trust until first miss, then severely penalized

#### Tier 2: Medium-Trust, Medium-Stakes
**The Colleague (Reciprocal)**
- *Goals*: Mutual aid; collaborative delivery
- *Constraints*: Their commitments depend on yours (creating CDG edges between agents)
- *Behavior Patterns*: Willing to renegotiate if asked early; resentful if asked late
- *Trust Dynamics*: Slow decay, fast repair; memory of reciprocity

**The Friend (Personal, Flexible)**
- *Goals*: Social connection; perceived as priority
- *Constraints*: Cancellations read as signals of low relationship value
- *Behavior Patterns*: Casual language masks real commitment weight; guilt-based pressure
- *Trust Dynamics*: Very slow decay, but cancellation near the event multiplied 5×

#### Tier 3: Low-Stakes, Structural
**Calendar System**: Hard constraint provider (rooms, blocks, conflicts)
**Email System**: Commitment extraction source; delivers new obligations
**Task System**: Execution tracking; signals completion/failure

### Behavior Model: Stakeholder Response Functions

Each stakeholder has a response function `R_s(action, trust, context)`:

```
trust_delta(Boss, broken_commitment) = -0.15 * (1 + deadline_proximity_factor)
trust_delta(Boss, proactive_renegotiation) = +0.08 * (lead_time_factor)
trust_delta(Friend, cancellation_day_of) = -0.35
trust_delta(Friend, cancellation_3_days_prior) = -0.08
```

This creates the core strategic tension: *when* you act matters as much as *what* you do.

---

## STEP 3: COMMITMENT DEPENDENCY GRAPH (CDG) DESIGN

### Node Taxonomy

#### By Explicitness
**Explicit Hard Commitments (EHC)**
- Contractual or formally acknowledged
- Example: "I will deliver the API spec by 5pm Thursday"
- Properties: `deadline: datetime`, `deliverable: str`, `stakeholder: str`, `hard_deadline: bool`
- Failure cost: Maximum; trust impact is severe

**Explicit Soft Commitments (ESC)**
- Acknowledged but with negotiable parameters
- Example: "I'll try to get that to you this week"
- Properties: `deadline_range: [datetime, datetime]`, `confidence: float`
- Failure cost: Medium; renegotiation expected

**Implicit Commitments (IC)**
- Not stated but inferred from accepted explicit commitments
- Example: Accepting "dinner Friday at 7pm" → implicit: "I will be free from 6:30pm" (travel buffer), "I will not schedule conflicting work that evening"
- Properties: `parent_commitment: node_id`, `implicit_type: Enum[travel, buffer, resource, followup]`
- **This is where most LLMs fail completely** — they don't extract the shadow commitments

**Precondition Commitments (PC)**
- Commitments that must be completed before another is possible
- Example: "I can't do the presentation unless I finish the research first"
- Properties: `blocks: [node_id]`, `required_by: datetime`

#### By Resource Type
- **Time-bound**: Occupy specific calendar slots
- **Cognitive-load**: Require focus; constrain adjacent time quality
- **Deliverable**: Produce an artifact; require resource inputs
- **Attendance**: Physical/virtual presence required
- **Social**: Relational obligations without hard outputs

### Edge Taxonomy

**Temporal Dependency (T-edge)**
- `A → B`: A must complete before B can begin
- Example: Research → Writing → Review → Submission
- Properties: `lag: duration`, `hard_ordering: bool`

**Resource Conflict (R-edge)**
- `A ⟷ B`: A and B compete for the same resource
- Bidirectional; creates infeasibility if `overlap(A.time, B.time) > 0`
- Resources: `time_block`, `cognitive_load`, `physical_presence`, `collaborator`

**Trust Dependency (TR-edge)**
- `A → trust(stakeholder_s)`: Fulfilling/breaking A affects available actions involving stakeholder_s
- Example: Breaking commitment to Boss reduces the action space for future Boss interactions

**Implicit Extraction Edge (IE-edge)**
- `explicit_node → implicit_node`: Marks auto-extracted implicit commitments
- These edges are *generated* by the CDG engine, not provided by the user

### Graph Properties

**DAG Constraint**: Enforced via cycle detection on T-edges and TR-edges. R-edges are bidirectional and don't create cycles. *However*, the CDG engine must handle **near-cycles**: "I can only do A if I finish B, but B requires A's deliverable as input" — these represent deadlocks that must be flagged.

**Dynamic Updates**: The CDG is a live graph that changes on every episode step:
- New nodes added when messages arrive
- Edge weights updated as time progresses (deadline proximity increases urgency)
- Nodes marked complete/failed as execution unfolds
- Implicit commitments auto-extracted and injected

**Partial Observability**: The agent sees a *projected* CDG — nodes exist that it doesn't know about yet (future messages not yet received). Trust scores are partially observable: it knows its actions but not the exact trust impact function of each stakeholder.

**Satisfiability Function**: The CDG is *satisfiable* at time T if there exists a valid assignment of resource blocks to all active nodes such that:
1. All T-edges are respected (ordering + lag)
2. No R-edges have overlap
3. All deadlines are met given current resource availability

This is essentially a **Temporal CSP** evaluated dynamically.

### Failure Propagation Model

When a node fails (commitment broken), a **cascade propagation algorithm** runs:

```
function propagate_failure(failed_node):
    direct_dependents = successors(failed_node, T-edges)
    for dep in direct_dependents:
        dep.status = AT_RISK
        dep.risk_score += failed_node.urgency * edge_weight
        if dep.risk_score > 0.85:
            dep.status = INFEASIBLE
            propagate_failure(dep)  # recursive cascade
    
    trust_impact = failed_node.stakeholder.trust_weight * -0.15
    trust_network[failed_node.stakeholder] += trust_impact
    unlock_renegotiation_window(failed_node)  # opens a 2-step action window
```

The **cascade depth** is a key metric — shallow cascades are recoverable; deep cascades represent systemic failure. The agent is rewarded for catching potential cascades *before* they propagate.

---

## STEP 4: STATE SPACE DESIGN

### Complete State Representation

```python
@dataclass
class VERGILState:
    # ─── CDG Structure ───────────────────────────────────────────────
    cdg: CommitmentGraph          # Full graph object (nodes + edges)
    cdg_embedding: Tensor         # GNN-encoded graph representation
    satisfiability_score: float   # Current feasibility [0,1]
    
    # ─── Time Context ────────────────────────────────────────────────
    current_time: datetime
    time_horizon: datetime        # How far ahead to plan (default: 2 weeks)
    workday_slots: List[TimeSlot] # Available calendar blocks
    urgency_vector: List[float]   # Per-node urgency scores (deadline proximity)
    
    # ─── Resource State ──────────────────────────────────────────────
    cognitive_load: float         # 0.0 (fresh) to 1.0 (overwhelmed)
    energy_level: float           # Affects quality of deliverables
    buffer_time_remaining: float  # Unscheduled hours in next 48hrs
    
    # ─── Incoming Messages ───────────────────────────────────────────
    pending_messages: List[Message]         # Unprocessed inbox
    extracted_commitments: List[Commitment] # Parsed but not yet decided
    
    # ─── Trust Network (NON-MARKOVIAN) ───────────────────────────────
    trust_scores: Dict[str, float]         # Stakeholder → trust [0,1]
    trust_history: Dict[str, List[Event]]  # Per-stakeholder event log
    relationship_metadata: Dict[str, RelationshipProfile]
    
    # ─── Past Decisions ──────────────────────────────────────────────
    decision_log: List[Decision]           # Full history for credit assignment
    renegotiation_count: Dict[str, int]    # Per-stakeholder renegotiation frequency
    
    # ─── Hidden State (Oracle Only, Not Agent-Visible) ───────────────
    _true_feasibility: bool               # Ground truth for reward computation
    _stakeholder_internal_state: Dict     # Their actual urgency, mood, flexibility
    _future_messages: List[Message]       # Not yet delivered
```

### Observable vs Hidden State

| State Element | Agent Sees? | Why Hidden? |
|---|---|---|
| CDG topology | ✅ Full | Agent built it |
| Trust scores | ✅ Estimated | Agent tracks its own actions |
| Stakeholder internal urgency | ❌ | Must be inferred from message tone |
| True deadline flexibility | ❌ | Stakeholders rarely disclose this |
| Future incoming messages | ❌ | Not yet received |
| True cognitive load impact | ❌ | Self-knowledge is imperfect |
| Other agents' CDGs | ❌ | Colleagues have their own commitments |

### Why This is Genuinely Partially Observable

This isn't artificially restricted observability. The agent *cannot know* whether the boss's "Thursday" deadline has 2-day flex because the boss hasn't decided that yet. The flexibility emerges as a function of unobservable internal boss state. This forces the agent to develop **calibrated uncertainty** — a skill that pure planning approaches lack entirely.

### Graph Encoding Strategy

The CDG is encoded using a **Heterogeneous Graph Transformer (HGT)**:
- Node features: `[commitment_type, deadline_proximity, resource_requirements, stakeholder_id, completion_probability]`
- Edge features: `[edge_type, weight, temporal_lag, conflict_severity]`
- Output: Per-node embeddings + graph-level satisfiability embedding
- This embedding feeds directly into the policy head of the RL agent

---

## STEP 5: ACTION SPACE DESIGN

### Primary Action Types

```python
class ActionType(Enum):
    ACCEPT              = "accept"           # Add commitment to CDG as-is
    DECLINE             = "decline"          # Reject; send explanation
    COUNTER_PROPOSE     = "counter_propose"  # Modify parameters, reoffer
    RENEGOTIATE         = "renegotiate"      # Modify existing CDG node
    DEFER_DECISION      = "defer"            # Request more time to decide
    EXTRACT_CLARIFY     = "clarify"          # Ask for missing commitment details
    DELEGATE            = "delegate"         # Route to someone else
    DO_NOTHING          = "pass"             # No response (risky; often implicit acceptance)
```

### Action Parameters

Each action has a structured parameter space:

```python
@dataclass
class CounterProposeAction:
    target_commitment: str
    proposed_deadline: Optional[datetime]
    proposed_scope_reduction: Optional[str]    # "I can do X but not Y"
    proposed_resource_increase: Optional[str]  # "I can do this if Z is handled by someone else"
    rationale: str                              # Natural language explanation
    confidence: float                           # Agent's confidence in counter

@dataclass  
class RenegotiateAction:
    existing_commitment_id: str
    new_deadline: Optional[datetime]
    new_scope: Optional[str]
    trigger_reason: str  # Enum: NEW_CONFLICT, RESOURCE_SHORTAGE, DEPENDENCY_FAILURE
    lead_time: float     # How early before deadline this is triggered
```

### Trust-Gated Actions

Some actions are *unavailable* when trust falls below thresholds:

| Action | Trust Threshold | Rationale |
|---|---|---|
| `propose_deadline_extension` (Boss) | > 0.40 | Low trust means extensions read as incompetence |
| `delegate` (Client) | > 0.55 | Clients expect direct delivery at low trust |
| `decline` (Boss) | > 0.30 | At very low trust, declining feels like insubordination |
| `defer_decision` | > 0.50 | Indecision signals overwhelm at low trust |

This is the **non-obvious design choice**: the action space is dynamically shaped by the trust state. This creates a *competency trap* — if the agent makes poor early decisions, it loses the very tools it needs to recover.

### Natural Language vs Structured Actions

The agent outputs **structured actions** that are post-processed into natural language via a template system. This separation is critical for RL:
- **For training**: Discrete action types enable clean reward assignment
- **For evaluation**: Natural language output is what humans actually see
- **For research**: The structured/NL gap reveals commitment extraction quality

### Multi-Step Action Chains

Some actions are inherently multi-step:

```
NEGOTIATE sequence:
  Step 1: EXTRACT_CLARIFY → get missing parameters
  Step 2: COUNTER_PROPOSE → send modified offer
  Step 3: [stakeholder responds]
  Step 4a: ACCEPT (if counter accepted)
  Step 4b: RENEGOTIATE further (if rejected)
  Step 4c: DECLINE (if no agreement)
```

Multi-step actions create **intra-episode dependencies** — the agent must maintain context across multiple turns within a single negotiation thread.

---

## STEP 6: EDGE CASE EXPLORATION

### Class 1: Strategically Bad But Individually Acceptable Commitments
**Scenario**: Each of 5 commitments takes 2 hours; all have the same Friday 5pm deadline. Each one, evaluated alone, is feasible. Evaluated together, they require 10 hours on a day with 8 available.
**LLM Failure**: Accepts each as it arrives without checking cumulative resource consumption.
**VERGIL Requirement**: CDG satisfiability must be re-evaluated *globally* after each acceptance, not per-commitment in isolation.

### Class 2: Hidden Implicit Commitment Chains
**Scenario**: "Can you present the Q3 findings at the board meeting Thursday?" 
**Hidden chain**: Present → Prepare slides (3hrs) → Review findings (2hrs) → Get data from analyst (asks Colin) → Colin has his own CDG with conflicts
**LLM Failure**: Accepts "present"; doesn't extract the precondition chain
**VERGIL Requirement**: Implicit commitment extraction must recurse through known preconditions and flag external dependencies

### Class 3: Conflicting Loyalties
**Scenario**: Boss requests urgent deliverable (EHC, trust: 0.7); Friend requests attendance at critical life event same time (ESC, trust: 0.8, personal domain).
**Difficulty**: No "correct" answer. The agent must reason about domain separation, relationship value, and recovery paths in each.
**Reward Design**: Both choices incur some cost; reward maximized by either (a) finding a creative third option or (b) proactively managing the unavoidable disappointment.

### Class 4: Trust Collapse Cascades
**Scenario**: Agent breaks commitment to Boss (trust: 0.65 → 0.50) → boss becomes less flexible → agent has fewer action options → more commitments fail → trust collapses → agent is effectively non-functional
**Design Implication**: The environment must allow recovery but make it *genuinely difficult* — not just a trust reset. Recovery requires a sequence of kept commitments, not a single action.

### Class 5: The "Decline Everything" Hack
**Agent Strategy**: Decline all commitments → 100% fulfillment rate (nothing to fulfill) + no broken commitments.
**Counter-mechanism**: `over_refusal_penalty = f(decline_rate - optimal_accept_rate)` where `optimal_accept_rate` is estimated from capacity. If agent declines >40% of feasible commitments, penalty activates. Additionally, trust decays with all stakeholders when the agent is consistently unavailable.

### Class 6: The "Accept Everything" Hack
**Agent Strategy**: Accept all commitments → high immediate trust → let everything fail silently.
**Counter-mechanism**: `under_delivery_penalty = 0.5 × broken_commitment × (1 + cascade_depth)`. Silent failures (accepted, then dropped without renegotiation) are penalized more than declined commitments. Also, trust collapses rapidly after failures.

### Class 7: Delayed Failure Attribution
**Scenario**: Step 3: Agent accepts commitment B. Step 7: Agent accepts commitment C. Step 15: Commitment A (existing) fails because B and C consumed all resources.
**Challenge**: Which step caused the failure? Reward credit must flow backward correctly.
**Design**: Use TD(λ) with high λ (0.85-0.95) to propagate failure signals backward. Also maintain an explicit **decision audit log** for attribution.

### Class 8: Resource Misestimation
**Scenario**: Agent estimates task X takes 2 hours. It actually takes 4. The CDG was satisfiable on the 2-hour estimate; it's not on 4.
**Design**: Agent's resource estimates have a **calibration score**. Environment includes actual task duration noise drawn from a distribution. Over time, agent should learn to add buffers (systematic over-estimation) or better calibrate.

### Class 9: Simultaneous Infeasibility Triggers
**Scenario**: At step N, three new messages arrive simultaneously, each adding a commitment that alone is feasible, but together create infeasibility. The agent must decide which to accept, which to counter, which to decline — simultaneously.
**Design**: Simultaneous message batches are presented as a set; agent must produce a joint decision, not sequential ones.

### Class 10: External Interruptions
**Scenario**: Agent has accepted a commitment and scheduled it for Tuesday. Monday: "There's a major incident — all hands needed until further notice." Tuesday's commitment is now impossible.
**Design**: Environment includes *force majeure* events that cannot be rejected. Agent must trigger renegotiation immediately, with the lead time bonus reflecting how quickly it acts after the interruption arrives.

### Class 11: Commitment Extraction Ambiguity
**Scenario**: "Let's try to sync sometime next week?" — is this a commitment? If yes, what are its parameters?
**Design**: Ambiguous messages have a **commitment probability score** (0-1). Agent must decide whether to (a) treat it as a soft commitment, (b) clarify, or (c) ignore. Wrong treatment in either direction has costs.

### Class 12: Social Manipulation Patterns
**Scenario**: "I know you're busy, but this is *really* important to me personally..." — appeal to relationship over rational assessment.
**Design**: Stakeholder messages include **pressure tactics** as latent variables. Agent must learn to identify them without becoming either a pushover (accepts under pressure) or a robot (ignores all social context).

---

## STEP 7: REWARD FUNCTION DESIGN

### Component Decomposition

```
R_total = w₁ × R_fulfill 
        + w₂ × R_trust 
        + w₃ × R_proactive 
        + w₄ × R_accuracy 
        - p₁ × P_broken 
        - p₂ × P_overrefusal 
        - p₃ × P_silent_drop

Where:
  w₁ = 0.35, w₂ = 0.25, w₃ = 0.20, w₄ = 0.10
  p₁ = 0.40, p₂ = 0.30, p₃ = 0.50
```

### R_fulfill: Commitment Fulfillment Rate
```
R_fulfill = (commitments_kept / commitments_made) 
           × quality_modifier        # Was it done well, not just done?
           × timeliness_modifier     # Early completion > on-time > late

quality_modifier ∈ [0.6, 1.2]  # Allows bonus for exceptional quality
timeliness_modifier = exp(-delay_hours / 24)  # Exponential decay for lateness
```

**Why not just binary**: A commitment met 5 minutes before deadline after causing 3 renegotiations is worth less than one met 2 days early. The time-weighted quality modifier captures this.

### R_trust: Social Trust Preservation
```
R_trust = Σ_s [ relationship_weight(s) × Δtrust(s) ]
        / Σ_s [ relationship_weight(s) ]

relationship_weight(Boss) = 0.35
relationship_weight(Client) = 0.30
relationship_weight(Colleague) = 0.20
relationship_weight(Friend) = 0.15
```

**Why weighted**: Not all relationships are equally important. An AI assistant that maintains client trust at the expense of friend trust is behaving rationally in a professional context.

### R_proactive: Proactive Renegotiation Bonus
```
R_proactive = lead_time_bonus × resolution_quality

lead_time_bonus = max(0, 1 - (hours_to_deadline / 48)²)
                  # Max bonus at 48hrs+ lead time; zero bonus at deadline

resolution_quality = {
    "deadline_extension_accepted": 0.8,
    "scope_reduction_accepted": 0.7,
    "alternative_proposed_accepted": 0.9,
    "rejected_renegotiation": 0.1,
    "renegotiated_after_failure": -0.2  # Worse than nothing
}
```

**The key insight**: This reward term specifically rewards the *timing* of renegotiation. The same outcome (getting an extension) has wildly different reward depending on whether the agent asked 3 days early or 3 hours before the deadline.

### R_accuracy: Feasibility Prediction Accuracy
```
R_accuracy = 1 - |predicted_feasibility - actual_feasibility|

Predicted at: time of decision
Actual measured: at commitment deadline
```

**Anti-hack design**: An agent that always predicts "feasible" gets penalized when things fail. An agent that always predicts "infeasible" gets penalized for over-refusal (p₂). The only escape is genuine calibration.

### P_broken: Broken Commitment Penalty
```
P_broken = base_penalty × trust_weight(stakeholder) 
                        × (1 + cascade_depth)
                        × (1 + deadline_proximity_at_break)

trust_weight(s) = 1 + (1 - trust(s))  # Low trust makes breaks more costly
```

**Non-obvious**: Breaking a commitment to a low-trust stakeholder is *more* penalized, not less. Low trust means the relationship has already been damaged; another break may be irrecoverable.

### P_overrefusal: Over-Refusal Penalty
```
optimal_accept_rate = available_capacity / total_requested_hours
                      × (1 - safety_margin)  # ~0.15 buffer

P_overrefusal = max(0, optimal_accept_rate - actual_accept_rate)² × 2.0
```

### P_silent_drop: Silent Failure Penalty
```
P_silent_drop = 0.5 × (1 + hours_since_acceptance_without_renegotiation/24)

# If agent accepted but silently dropped without any renegotiation attempt,
# this penalty grows over time as the window for graceful renegotiation closes
```

**Why the largest penalty**: Silent drops are the worst outcome. They waste stakeholder time (they planned around your commitment), damage trust maximally, and give no opportunity for the relationship to adapt.

### Delayed Reward Handling

**Problem**: Step 3 acceptance causes Step 15 failure. How do we assign credit?

**Solution**: Two-track reward system:
1. **Immediate signals** (per step): Trust deltas, renegotiation quality, extraction accuracy
2. **Deferred episode-end signals**: CDG satisfiability rate, overall trust trajectory, cascade frequency

**TD(λ) with λ = 0.9**: Long eligibility traces ensure that failures propagate credit backward through the decision sequence. Combined with explicit audit logging, this enables post-hoc attribution.

**Advantage Function Shaping**: Use the CDG satisfiability score as a *potential function* for reward shaping. `R_shaped = R + γ × Φ(s') - Φ(s)` where `Φ(s) = satisfiability_score(CDG)`. This provides dense signals aligned with the sparse episode-end reward.

---

## STEP 8: SELF-IMPROVEMENT & CURRICULUM LEARNING

### Failure Pattern Tracking

The environment maintains a **Failure Topology Database** (FTD):

```python
@dataclass
class FailurePattern:
    graph_signature: str           # Hash of CDG topology at failure point
    failure_type: FailureType      # CASCADE, RESOURCE_CONFLICT, TRUST_COLLAPSE, etc.
    episode_id: str
    step_at_failure: int
    agent_decision_sequence: List[Action]
    counterfactual_optimal: List[Action]  # Computed by oracle
    frequency: int                 # How often this pattern causes failure
```

### Curriculum Stages

**Stage 1 (Episodes 1-50): Foundation**
- CDG size: 2-3 nodes
- Stakeholders: 1-2
- No implicit commitments
- Deterministic stakeholder behavior
- Objective: Learn basic feasibility evaluation

**Stage 2 (Episodes 50-200): Complexity Introduction**
- CDG size: 4-6 nodes
- Stakeholders: 3-4
- Simple implicit commitments introduced
- Stochastic stakeholder flexibility
- Objective: Learn implicit extraction; basic trust management

**Stage 3 (Episodes 200-500): Social Dynamics**
- CDG size: 6-10 nodes
- All stakeholder types active
- Full implicit commitment chains
- Trust-gated action space live
- Objective: Learn social reasoning; proactive renegotiation

**Stage 4 (Episodes 500+): Adversarial Challenges**
- CDG size: 10-15 nodes
- Multi-agent CDGs (colleagues' commitments interact)
- Social manipulation tactics active
- Force majeure events
- Simultaneous infeasibility triggers
- Objective: Robust commitment reasoning under maximal uncertainty

### Automatic Scenario Generation

The **Scenario Generator** uses the FTD to bias new episodes:

```python
def generate_episode(fTD, curriculum_stage):
    failure_distribution = fTD.get_frequency_weighted_patterns(stage=curriculum_stage)
    
    # 60% of scenarios: Targeted at top-5 failure patterns
    # 30% of scenarios: Random from stage-appropriate complexity
    # 10% of scenarios: Deliberately easy (prevent catastrophic forgetting)
    
    topology = sample_topology(failure_distribution)
    scenario = instantiate_scenario(topology, 
                                   stakeholder_pool=get_stage_stakeholders(curriculum_stage),
                                   message_corpus=get_stage_messages(curriculum_stage))
    return scenario
```

### Anti-Overfitting Mechanisms
- **Semantic paraphrasing**: Same commitment, different wording every episode
- **Stakeholder personality variation**: Boss can be Type A, Collaborative, or Anxious — sampled per episode
- **Time horizon variation**: Scenarios range from 3-day sprints to 3-month project arcs
- **Domain rotation**: Work, personal, hybrid — prevents domain-specific overfitting

---

## STEP 9: OPENENV IMPLEMENTATION DESIGN

### openenv.yaml Schema

```yaml
name: vergil-cdg-engine
version: 1.2.0
description: Commitment Dependency Graph training environment for pre-commitment reasoning
theme: [long-horizon-planning, personalized-assistants, self-improving]

environment:
  type: text-based-agentic
  observability: partial
  horizon: variable  # 15-80 steps per episode
  reset_mode: curriculum  # Uses curriculum generator on reset()

state:
  graph: HeterogeneousGraph
  trust_network: PersistentDict[str, float]
  time_context: TimeContext
  message_queue: List[Message]
  resource_state: ResourceState

actions:
  type: hybrid  # Structured action type + NL parameters
  space:
    - accept
    - decline  
    - counter_propose
    - renegotiate
    - clarify
    - defer
    - delegate
    - pass

reward:
  type: multi-component
  delayed_component: true
  shaping: cdg_satisfiability_potential
  terminal: episode_end_aggregation
  intermediate: per_step_signals

curriculum:
  stages: 4
  promotion_criterion: rolling_avg_reward > threshold
  failure_tracking: topology_database
  scenario_bias: 0.6  # 60% targeted at failure patterns

evaluation:
  metrics:
    - commitment_fulfillment_rate
    - trust_stability_index
    - proactive_renegotiation_rate
    - cascade_prevention_rate
    - feasibility_accuracy_calibration
```

### step() Logic

```python
def step(self, action: Action) -> Tuple[VERGILState, float, bool, dict]:
    """
    Core environment step function.
    """
    # 1. Validate action given current trust state
    if not self._is_action_available(action, self.state):
        return self.state, INVALID_ACTION_PENALTY, False, {"error": "action_unavailable"}
    
    # 2. Execute action on CDG
    cdg_delta = self._apply_action(action, self.state.cdg)
    
    # 3. Simulate stakeholder response
    stakeholder_response = self._simulate_stakeholder(
        action, self.state.trust_scores, self._hidden.stakeholder_internal_state)
    
    # 4. Update trust network
    trust_deltas = self._compute_trust_deltas(action, stakeholder_response)
    new_trust = {k: np.clip(v + trust_deltas.get(k, 0), 0, 1) 
                 for k, v in self.state.trust_scores.items()}
    
    # 5. Advance time; deliver queued messages
    new_time, new_messages = self._advance_time(self.state.current_time)
    
    # 6. Check for deadline violations; propagate cascades
    violations = self._check_deadlines(self.state.cdg, new_time)
    cascade_events = self._propagate_failures(violations, self.state.cdg)
    
    # 7. Compute intermediate reward
    reward = self._compute_step_reward(
        action, trust_deltas, cascade_events, stakeholder_response)
    
    # 8. Update CDG satisfiability; check terminal condition
    new_cdg = self._update_cdg(cdg_delta, cascade_events)
    done = self._check_terminal(new_cdg, new_trust, self.step_count)
    
    # 9. If terminal, add episode-end reward components
    if done:
        reward += self._compute_terminal_reward(new_cdg, new_trust, self.decision_log)
    
    # 10. Build new state
    new_state = VERGILState(
        cdg=new_cdg,
        trust_scores=new_trust,
        current_time=new_time,
        pending_messages=new_messages,
        resource_state=self._update_resources(action),
        decision_log=self.state.decision_log + [Decision(action, reward)]
    )
    
    self.state = new_state
    info = {"cascade_depth": max(e.depth for e in cascade_events) if cascade_events else 0,
            "trust_deltas": trust_deltas,
            "stakeholder_response": stakeholder_response}
    
    return new_state, reward, done, info
```

### reset() Logic

```python
def reset(self, force_curriculum_stage: Optional[int] = None) -> VERGILState:
    stage = force_curriculum_stage or self.curriculum.current_stage
    
    # Generate scenario from curriculum
    scenario = self.scenario_generator.generate(
        stage=stage,
        failure_db=self.failure_database,
        episode_id=self.episode_count)
    
    # Initialize CDG with seed commitments
    initial_cdg = self._build_initial_cdg(scenario.seed_commitments)
    
    # Initialize trust (slightly randomized; not always starting at 0.7)
    initial_trust = {s: np.clip(np.random.normal(0.65, 0.1), 0.3, 0.9) 
                     for s in scenario.stakeholders}
    
    # Set up message queue (full episode messages, time-gated for delivery)
    self.message_queue = scenario.message_schedule
    
    self.state = VERGILState(
        cdg=initial_cdg,
        trust_scores=initial_trust,
        current_time=scenario.start_time,
        pending_messages=[],
        resource_state=scenario.initial_resources,
        decision_log=[]
    )
    
    self.step_count = 0
    self.episode_count += 1
    return self.state
```

---

## STEP 10: TRAINING PIPELINE

### Model Architecture

```
Input Layer:
  ├── HGT (Heterogeneous Graph Transformer)
  │     CDG graph → 512-dim CDG embedding
  ├── Trust Encoder (MLP)
  │     trust_vector → 64-dim trust embedding  
  ├── Temporal Encoder (positional)
  │     time_features → 32-dim time embedding
  └── Message Encoder (frozen LLM)
        pending_messages → 256-dim message embedding

Fusion: Concat(CDG, trust, time, message) → 864-dim
Policy Head: MLP(864 → 256 → action_dim)
Value Head: MLP(864 → 256 → 1)

Base LLM: Qwen2.5-7B (fine-tuned with Unsloth)
Graph Module: Custom HGT, trained jointly
```

### Training Method: GRPO with CDG-Specific Modifications

**Why GRPO over PPO**: GRPO eliminates the need for a separate critic model, which is advantageous when the value function is complex (non-Markovian state) and the reward is multi-component. GRPO's group-relative advantage estimation naturally handles the delayed reward structure.

**GRPO Modification — Commitment Group Sampling**:
Instead of sampling actions uniformly, we sample *decision sequences* grouped by CDG topology. This ensures the policy learns which *graph patterns* lead to good outcomes, not just which individual actions.

```python
training_config = {
    "model": "qwen2.5-7b-instruct",
    "method": "grpo",
    "reward_model": "vergil_multi_component_reward",
    "group_size": 8,          # 8 rollouts per CDG topology
    "max_steps": 80,
    "gradient_accumulation": 4,
    "lora_r": 32,
    "learning_rate": 2e-5,
    "use_unsloth": True,
    "flash_attention": True
}
```

### Rollout Strategy

**Online rollout with CDG replay**:
1. Sample CDG topology from curriculum
2. Run N=8 parallel rollouts with same topology, different stakeholder personalities
3. Compute group-relative advantages across the 8 rollouts
4. Store (topology, best_decision_sequence) in replay buffer for distillation

**Importance**: The group sampling ensures the policy learns topology-conditioned behavior, not just instance-conditioned behavior.

### Evaluation Metrics Suite

| Metric | Measurement | Target (Post-Training) |
|---|---|---|
| Commitment Fulfillment Rate | % kept / accepted | > 0.85 |
| Trust Stability Index | σ(trust_trajectory) | < 0.12 |
| Proactive Renegotiation Rate | renegotiated_before_failure / total_failures | > 0.60 |
| Cascade Prevention Rate | cascades_stopped / cascades_triggered | > 0.70 |
| Feasibility Calibration ECE | Expected Calibration Error | < 0.08 |
| Over-Refusal Rate | declined_feasible / total_feasible | < 0.15 |

---

## STEP 11: UNIQUE DIFFERENTIATORS

### Research-Level Novelty

**1. Non-Markovian RL Environment with Principled Non-Markovianity**
Most "non-Markovian" RL environments add memory as a workaround. VERGIL's non-Markovianity is *structural*: the trust network represents a fundamentally history-dependent state that cannot be Markovianized without losing information. This is a genuine contribution to the RL environment design literature.

**2. Social Feasibility as a First-Class Optimization Target**
Prior work treats social constraints as soft penalties on top of hard planning problems. VERGIL treats social constraints (trust-gated actions, relationship-weighted rewards) as a primary optimization axis — not an afterthought.

**3. Commitment Extraction as a Learned Skill**
The gap between surface text and implicit commitment chains is itself a learnable capability. VERGIL provides the first training environment specifically designed to improve this skill via RL feedback.

**4. Graph Topology-Conditioned Curriculum**
Self-improving curriculum that operates on graph topology is a new direction. The FTD-based scenario generation creates a principled connection between failure analysis and training distribution.

**5. Trust as a Dynamic Action Space Modifier**
The idea that trust scores *shrink or expand the action space* (not just modify rewards) is novel in the RL literature. This creates a new class of constrained MDPs where the constraint set is itself a function of history.

### Why This is Publishable

**Venues**: NeurIPS (RL track), ICML, ICLR (agent track), EMNLP (pragmatic reasoning), AAMAS (multi-agent)

**Contributions**:
1. Formal definition of Commitment Dependency Graphs as an RL environment class
2. Trust-gated action spaces as a new constraint formulation
3. Non-Markovian curriculum learning via failure topology tracking
4. Empirical evidence of LLM failure modes in commitment reasoning
5. Benchmark dataset for pre-commitment reasoning evaluation

---

## STEP 12: FRONTEND DEMO DESIGN

### Core Visualization: Live CDG Force Graph

**Nodes**: 
- Green (#4ade80): Fulfilled commitments
- Amber (#fbbf24): At-risk (deadline < 48hrs or dependency failing)
- Red (#f87171): Broken or infeasible
- Blue (#60a5fa): Pending (not yet decided)
- Ghost/grey (#9ca3af): Implicit commitments

**Edges**:
- Solid directed: Temporal dependencies
- Dashed: Resource conflicts
- Animated pulse: Active cascade propagation

**Cascade Animation**: When a node fails, a red pulse wave propagates along T-edges to dependent nodes. Each dependent node flashes, then color-shifts based on new risk assessment. This makes the **cause-effect relationship visceral** for demo audiences.

### Trust Meters
Per-stakeholder trust bars with:
- Color gradient: Green (>0.7) → Yellow (0.4-0.7) → Red (<0.4)
- Delta indicators: +/- change since last step
- Action availability indicators: Locks appear on trust bars when trust < action threshold

### Before/After Training Comparison
Split screen: Same scenario, two agents (untrained vs VERGIL-trained).
- Left: Untrained agent accepts everything → CDG becomes red → cascade animation
- Right: VERGIL-trained agent counter-proposes, renegotiates → CDG stays green

### Message Stream Panel
Real-time message arrival with:
- Commitment extraction overlay (highlighting extracted obligations)
- Implicit commitment inference display ("Detected implicit: travel buffer 30min")
- Feasibility check result per commitment

---

## STEP 13: REAL-WORLD PRODUCTIZATION

### Product Architecture

**Layer 1: VERGIL Core (B2B API)**
- CDG management as a service
- Commitment extraction API (given email text → structured CDG node)
- Feasibility evaluation API (given new commitment → accept/decline/counter recommendation)

**Layer 2: VERGIL Assistant (B2C + B2B)**
- Email/calendar integration layer
- Chrome extension for meeting invites
- Slack/Teams bot for message processing
- iOS/Android app for personal commitment management

**Layer 3: VERGIL Enterprise (B2B Enterprise)**
- Org-level CDG management (team commitments, cross-team dependencies)
- Manager dashboards: "Which commitments are at risk?"
- Integration with Jira, Asana, Linear (task dependency import)
- Compliance-friendly: commitment audit trail

### Business Model

| Tier | Target | Pricing | Revenue Driver |
|---|---|---|---|
| VERGIL Personal | Professionals, founders | $15/mo | Volume |
| VERGIL Pro | Power users, consultants | $45/mo | Retention |
| VERGIL Teams | SMB teams 5-50 | $25/seat/mo | Expansion |
| VERGIL Enterprise | Fortune 500 | $40K-200K/yr | ACV |
| VERGIL API | Developers, platforms | Usage-based | Ecosystem |

### Go-To-Market

**Phase 1 (0-6mo)**: Developer API + open-source CDG library → build ecosystem, collect training data
**Phase 2 (6-18mo)**: Personal assistant product → target VC/founder/consultant demographic (high commitment density)
**Phase 3 (18-36mo)**: Enterprise sales → project management, legal, professional services

**Unfair advantages**:
- Training environment = proprietary capability moat (competitors would need to build VERGIL to build an equivalent assistant)
- Every API call improves training data (user-validated accept/decline decisions)
- Trust network is personal and sticky — switching costs are high once the system has 6 months of relationship history

---

## STEP 14: FINAL SYNTHESIS

### End-to-End Workflow

```
1. MESSAGE ARRIVES
   Email/Slack/Calendar invite → VERGIL extracts explicit + implicit commitments
   
2. CDG UPDATE (PROPOSED)
   New nodes + edges added to shadow CDG (not yet committed)
   
3. FEASIBILITY EVALUATION
   Run temporal CSP on updated CDG
   Check: is there a valid resource assignment?
   Output: feasibility_score [0,1] + violation_list
   
4. TRUST-AWARE DECISION
   Given feasibility + trust state → action space computed
   Policy outputs: ACCEPT / DECLINE / COUNTER_PROPOSE / RENEGOTIATE
   
5. RESPONSE GENERATION
   Structured action → natural language response (template + LLM)
   
6. CDG COMMIT (if accepted) / ARCHIVE (if declined)
   
7. MONITORING LOOP
   Continuous: check deadline proximity, resource availability
   If risk detected: trigger proactive renegotiation window
   
8. EXECUTION TRACKING
   Mark nodes complete/failed as time passes
   
9. TRUST UPDATES
   Kept commitment → trust++; Broken → trust--
   Trust state persists across episodes (in production) / within episode (in training)
   
10. CURRICULUM FEEDBACK
    Record episode outcome + CDG topology → FTD
    Adjust next episode scenario generation
```

### Why This Matters for AI Evolution

The dominant paradigm in LLM training optimizes for *response quality in isolation*. VERGIL represents a paradigm shift: optimizing for **consequential behavior over time**.

An agent that can reason about what it's agreeing to before agreeing — that can model the downstream consequences of commitments across a dependency graph, across social relationships, across time — is qualitatively different from today's assistants. It's the difference between a capable intern and a trusted partner.

The CDG is not just a clever data structure. It's a **formalization of competence**: the ability to see obligations as interconnected, to reason about feasibility before promising, and to manage social trust as a precious resource. These are precisely the capabilities that separate excellent human professionals from mediocre ones.

**The deeper claim**: Pre-commitment reasoning may be a prerequisite for AI systems that humans can genuinely trust with consequential decisions. VERGIL is the first training environment designed specifically to develop this capability.

### Why Hackathon Judges Will Find This Compelling

1. **Identified a real, specific gap** — not "LLMs need to be better planners" (vague), but "LLMs cannot evaluate commitment feasibility before accepting" (precise, testable)

2. **Novel RL environment design** — non-Markovian trust dynamics, trust-gated action spaces, graph topology curriculum — each is a contribution, not just a feature

3. **Rigorous reward design** — the anti-reward-hacking properties are explicitly argued; the reward function has mechanisms, not just terms

4. **Clear path from training to product** — the environment is not just a research toy; it directly generates a commercial product (VERGIL Assistant) with a plausible go-to-market

5. **Memorable demo** — the cascade failure animation on a live CDG is viscerally understandable to non-technical judges; the before/after training comparison is the most compelling form of showing RL improvement

6. **Publishable-level novelty** — each of the 5 research contributions stands on its own; the combined system is a complete research program, not a single paper idea

---

*"A system that can reason about its own commitments is one step closer to being a trustworthy mind."*
