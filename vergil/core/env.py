# vergil/core/env.py
"""
VERGIL OpenEnv-Compatible Training Environment
================================================

step() pipeline (ORDER MATTERS — Step 9 of design doc):
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

API Contract:
    state()  → VERGILState
    step()   → (VERGILState, float, bool, bool, dict)
    reset()  → (VERGILState, dict)

Design Principles:
    1. All hidden state lives in self._hidden (never exposed to agent)
    2. step() is the only place state mutations occur
    3. All events are logged with sufficient detail for replay
    4. Episode generation is deterministic given a seed (reproducible)

References:
    VERGIL_System_Design.md — Step 5, 6, 7, 8
    VERGIL_Phase1_Phase2_Implementation.md — P1.6
"""

import gymnasium as gym
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .types import (
    VERGILState, AgentAction, ActionType, CommitmentNode, CommitmentStatus,
    CommitmentType, CDGEdge, EdgeType, Message, TrustEntry, EpisodeRecord,
    StakeholderProfile, StakeholderRole
)
from .cdg import CommitmentDependencyGraph
from .extraction import CommitmentExtractor
from .stakeholder import StakeholderSimulator
from .reward import VERGILReward, RewardComponents

# Phase 2 imports — fully integrated into step pipeline
from .trust_multidim import (
    MultiDimTrustEntry, MultiDimTrustUpdater, TrustGate, TrustDimension
)
from .execution_model import ProbabilisticExecutionEngine, ForceMajeureGenerator

logger = logging.getLogger('vergil.env')


class VERGILEnv(gym.Env):
    """
    OpenEnv-compatible training environment for VERGIL.

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

        # Components
        self.reward_fn = VERGILReward(config=self.config.get('reward', {}))
        self.extractor: Optional[CommitmentExtractor] = None
        self.simulator: Optional[StakeholderSimulator] = None
        self.cdg: Optional[CommitmentDependencyGraph] = None

        # Phase 2 components — now fully integrated
        self.execution_engine = ProbabilisticExecutionEngine(seed=seed)
        self.force_majeure = ForceMajeureGenerator(seed=seed)
        self.multidim_trust: Dict[str, MultiDimTrustEntry] = {}

        # State
        self._state: Optional[VERGILState] = None

        # Hidden state (oracle only — Step 4)
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

        # Curriculum reference (set externally by CurriculumEngine)
        self.curriculum_stage: int = 1

        logger.info(f"VERGILEnv initialized. seed={seed}, max_steps={self._max_steps}")

    # ── OpenEnv Required Methods ──────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
              scenario: Optional[Dict] = None,
              options: Optional[Dict] = None) -> Tuple[VERGILState, Dict]:
        """Reset environment for new episode."""
        if seed is not None:
            self.seed = seed

        if scenario is None:
            scenario = self._generate_scenario(self.curriculum_stage)

        episode_id = str(uuid.uuid4())[:8]
        start_time = scenario.get('start_time')
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        if start_time is None:
            start_time = datetime.now().replace(
                hour=9, minute=0, second=0, microsecond=0)

        # Initialize CDG
        self.cdg = CommitmentDependencyGraph(graph_id=f"cdg_{episode_id}")

        # Load seed commitments
        for commit_data in scenario.get('seed_commitments', []):
            node = self._build_node_from_dict(commit_data)
            self.cdg.add_node(node)

        for edge_data in scenario.get('seed_edges', []):
            edge = self._build_edge_from_dict(edge_data)
            self.cdg.add_edge(edge)

        # Initialize stakeholders
        profiles = self._build_stakeholder_profiles(scenario)
        self.simulator = StakeholderSimulator(profiles=profiles, seed=self.seed)

        # Initialize trust (scalar for backward compat)
        trust_entries = {}
        for s_id, profile in profiles.items():
            trust_score = scenario.get('initial_trust', {}).get(s_id, 0.65)
            te = TrustEntry(stakeholder_id=s_id, trust_score=trust_score)
            te._role = profile.role.value
            trust_entries[s_id] = te

        # Phase 2: Initialize multi-dimensional trust alongside scalar
        self.multidim_trust = {}
        for s_id, profile in profiles.items():
            init_trust = scenario.get('initial_trust', {}).get(s_id, 0.65)
            self.multidim_trust[s_id] = MultiDimTrustEntry(
                stakeholder_id=s_id,
                reliability=init_trust,
                competence=init_trust,
                benevolence=min(1.0, init_trust + 0.05),
            )

        # Initialize extractor
        self.extractor = CommitmentExtractor(current_time=start_time,
                                             config=self.config.get('extraction', {}))

        # Message schedule
        self._message_schedule = self._build_message_schedule(scenario, start_time)

        # Hidden state (Step 4 — oracle only)
        # Phase 2: Use execution_model for stochastic true durations
        rng = np.random.RandomState(self.seed)
        true_durations = {}
        for node_id, node in self.cdg._nodes.items():
            sigma = node.duration_std_hours / max(node.estimated_duration_hours, 0.5)
            actual = float(self.execution_engine.rng.lognormal(
                mean=np.log(max(0.1, node.estimated_duration_hours)),
                sigma=max(0.1, min(1.0, sigma)),
            ))
            true_durations[node_id] = actual

        self._hidden = {
            'stakeholder_profiles': profiles,
            'true_durations': true_durations,
            'future_messages': self._message_schedule.copy(),
            'force_majeure_active': False,
            'adversarial_events': [],
        }

        # Build initial state
        self._current_step = 0

        # Evaluate initial satisfiability
        sat_result = self.cdg.evaluate_satisfiability(
            current_time=start_time,
            available_hours=16.0
        )

        self._state = VERGILState(
            cdg_nodes=list(self.cdg._nodes.values()),
            cdg_edges=list(self.cdg._edges.values()),
            satisfiability_score=sat_result.satisfiability_score,
            current_time=start_time,
            time_horizon=start_time + timedelta(days=14),
            available_hours_next_48h=16.0,
            cognitive_load=0.0,
            energy_level=1.0,
            pending_messages=[],
            trust_entries=trust_entries,
            decision_log=[],
            episode_id=episode_id,
            step_number=0,
            curriculum_stage=self.curriculum_stage,
            node_risk_scores=self.cdg._node_risk_scores.copy(),
            at_risk_nodes=sat_result.at_risk_nodes,
        )

        # Episode record
        self._episode_record = EpisodeRecord(
            episode_id=episode_id,
            curriculum_stage=self.curriculum_stage,
            scenario_id=scenario.get('scenario_id', 'generated'),
            start_time=start_time,
        )

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
        """Execute one environment step (Step 9 pipeline)."""
        assert self._state is not None, "Must call reset() before step()"
        assert self.cdg is not None

        self._current_step += 1
        current_time = self._state.current_time

        # ── Step 1: Action validation ──────────────────────────────────────
        is_valid, validity_reason = self._validate_action(action, self._state)
        if not is_valid:
            logger.warning(f"Invalid action: {action.action_type} — {validity_reason}")
            reward_components = RewardComponents(total=-0.1)
            info = {'invalid_action': validity_reason, 'step': self._current_step}
            return self._state, -0.1, False, False, info

        # ── Step 2: Apply action to CDG ─────────────────────────────────────
        affected_node = self._apply_action(action, current_time)

        # ── Step 2b [Phase 2]: Force majeure check (stage 3+) ──────────────
        force_majeure_event = None
        if self.curriculum_stage >= 3:
            fm_event = self.force_majeure.sample_event(
                step=self._current_step, curriculum_stage=self.curriculum_stage)
            if fm_event:
                force_majeure_event = fm_event
                self._hidden['force_majeure_active'] = True
                self._hidden.setdefault('adversarial_events', []).append(
                    {'step': self._current_step, 'type': 'force_majeure',
                     'description': force_majeure_event.get('description', 'disruption')})

        # ── Step 3: Stakeholder simulation ────────────────────────────────
        trust_deltas = {}
        stakeholder_responses = {}

        if affected_node is not None and self.simulator is not None:
            trust_entry = self._state.trust_entries.get(
                affected_node.stakeholder_id,
                TrustEntry(stakeholder_id=affected_node.stakeholder_id)
            )
            response = self.simulator.simulate_response(
                action=action, node=affected_node,
                trust=trust_entry, current_time=current_time
            )
            trust_deltas[affected_node.stakeholder_id] = response.trust_delta
            stakeholder_responses[affected_node.stakeholder_id] = response

        # ── Step 4: Trust update (scalar + multi-dimensional) ─────────────
        new_trust = {sid: te for sid, te in self._state.trust_entries.items()}
        for sid, delta in trust_deltas.items():
            if sid in new_trust:
                new_trust[sid].update(
                    delta=delta,
                    event=action.action_type.value,
                    step=self._current_step
                )

            # Phase 2: Update multi-dimensional trust in parallel
            if sid in self.multidim_trust:
                outcome = 'commitment_kept' if delta >= 0 else 'commitment_broken'
                lead_time = 48.0
                if affected_node and affected_node.deadline:
                    lead_time = max(0, (affected_node.deadline - current_time
                                        ).total_seconds() / 3600)
                md_deltas = MultiDimTrustUpdater.compute_deltas(
                    action_type=action.action_type.value,
                    outcome=outcome,
                    lead_time_hours=lead_time,
                )
                for dim, d in md_deltas:
                    self.multidim_trust[sid].update(
                        dimension=dim, delta=d,
                        event=action.action_type.value,
                        step=self._current_step,
                    )

        # ── Step 5: Time advance + message delivery ────────────────────────
        step_hours = self.config.get('step_hours', 2)
        new_time = current_time + timedelta(hours=step_hours)
        if self.extractor:
            self.extractor.current_time = new_time
        new_messages = self._deliver_messages(new_time)

        # Process new messages: extract commitments and add pending nodes
        for msg in new_messages:
            if msg.extraction_result and msg.extraction_result.is_commitment:
                self._create_node_from_message(msg, new_time)

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
        prev_sat = self._state.satisfiability_score
        sat_result = self.cdg.evaluate_satisfiability(
            current_time=new_time,
            available_hours=self._state.available_hours_next_48h
        )

        # Apply potential-based reward shaping: R' = R + γ×Φ(s') - Φ(s)
        # Φ(s) = CDG satisfiability score — provides dense intermediate signal
        # aligned with sparse terminal reward. Policy-invariant by construction.
        reward_components.total = self.reward_fn.compute_shaped_reward(
            base_reward=reward_components.total,
            prev_satisfiability=prev_sat,
            new_satisfiability=sat_result.satisfiability_score,
            gamma=0.99,
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
        new_available = max(0, self._state.available_hours_next_48h - (
            affected_node.estimated_duration_hours
            if affected_node and action.action_type == ActionType.ACCEPT
            else 0
        ))

        new_state = VERGILState(
            cdg_nodes=list(self.cdg._nodes.values()),
            cdg_edges=list(self.cdg._edges.values()),
            satisfiability_score=sat_result.satisfiability_score,
            current_time=new_time,
            time_horizon=self._state.time_horizon,
            available_hours_next_48h=new_available,
            cognitive_load=self._update_cognitive_load(action, affected_node),
            energy_level=max(0.1, self._state.energy_level - 0.02),
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

        # ── Step 12: Logging (with Phase 2 data) ──────────────────────────
        info = {
            'cascade_events': all_cascade_events,
            'satisfiability': sat_result.to_dict(),
            'trust_deltas': trust_deltas,
            'stakeholder_responses': {k: v.message for k, v in stakeholder_responses.items()},
            'term_reason': term_reason if (terminated or truncated) else None,
            'reward_components': reward_components.to_dict(),
            'step': self._current_step,
            'new_messages': len(new_messages),
            # Phase 2 data
            'multidim_trust': {
                sid: {
                    'reliability': round(mt.reliability, 3),
                    'competence': round(mt.competence, 3),
                    'benevolence': round(mt.benevolence, 3),
                    'composite': round(mt.composite_trust, 3),
                }
                for sid, mt in self.multidim_trust.items()
            },
            'force_majeure': force_majeure_event is not None if 'force_majeure_event' in dir() else False,
            'available_actions': {
                sid: [a.value for a in TrustGate.get_available_actions(
                    mt, self._hidden.get('stakeholder_profiles', {}).get(sid, None).role.value
                    if self._hidden.get('stakeholder_profiles', {}).get(sid, None) else 'colleague'
                )]
                for sid, mt in self.multidim_trust.items()
            } if self.multidim_trust else {},
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
        """Check if action is legal given current state."""
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

    def _apply_action(self, action: AgentAction,
                      current_time: datetime) -> Optional[CommitmentNode]:
        """Apply action to CDG. Returns affected node (if any)."""
        node = None
        if action.target_node_id:
            node = self.cdg.get_node(action.target_node_id)

        if node is None:
            return None

        if action.action_type == ActionType.ACCEPT:
            self.cdg.update_node_status(node.node_id, CommitmentStatus.ACCEPTED, current_time)
            node.decision_made = ActionType.ACCEPT
            node.decision_timestamp = current_time
            node.trust_at_decision = self._state.trust_entries.get(
                node.stakeholder_id, TrustEntry(stakeholder_id='')).trust_score
            self._inject_implicit_commitments(node, current_time)

        elif action.action_type == ActionType.DECLINE:
            self.cdg.update_node_status(node.node_id, CommitmentStatus.DECLINED, current_time)
            node.decision_made = ActionType.DECLINE
            node.decision_timestamp = current_time

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
            self.cdg.update_node_status(node.node_id, CommitmentStatus.RENEGOTIATED, current_time)

        elif action.action_type == ActionType.DELEGATE:
            self.cdg.update_node_status(node.node_id, CommitmentStatus.DELEGATED, current_time)
            node.decision_made = ActionType.DELEGATE

        node.updated_at = current_time
        return node

    def _inject_implicit_commitments(self, parent_node: CommitmentNode,
                                     current_time: datetime) -> None:
        """Auto-extract and inject implicit commitments when parent is accepted."""
        if parent_node.extraction_result is None:
            return

        for implicit_spec in parent_node.extraction_result.implicit_commitments:
            if not implicit_spec.auto_accept:
                continue

            # Compute implicit deadline
            if implicit_spec.time_before_parent and parent_node.deadline:
                implicit_deadline = parent_node.deadline - implicit_spec.time_before_parent
            elif implicit_spec.time_after_parent and parent_node.deadline:
                implicit_deadline = parent_node.deadline + implicit_spec.time_after_parent
            else:
                implicit_deadline = parent_node.deadline

            implicit_node = CommitmentNode(
                label=f"[Implicit] {implicit_spec.description}",
                description=implicit_spec.description,
                commitment_type=CommitmentType.IMPLICIT,
                status=CommitmentStatus.ACCEPTED,
                stakeholder_id=parent_node.stakeholder_id,
                stakeholder_role=parent_node.stakeholder_role,
                deadline=implicit_deadline,
                estimated_duration_hours=implicit_spec.duration_hours,
                duration_std_hours=0.25,
                parent_commitment_id=parent_node.node_id,
            )

            self.cdg.add_node(implicit_node)

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

            logger.debug(f"Injected implicit: {implicit_node.node_id} "
                        f"'{implicit_spec.description}'")

    def _create_node_from_message(self, msg: Message, current_time: datetime) -> None:
        """Create a pending commitment node from a delivered message."""
        if not msg.extraction_result:
            return

        er = msg.extraction_result
        node = CommitmentNode(
            label=er.raw_text[:80],
            description=er.raw_text,
            commitment_type=er.commitment_type or CommitmentType.EXPLICIT_SOFT,
            status=CommitmentStatus.PENDING,
            stakeholder_id=msg.sender_id,
            stakeholder_role=msg.sender_role,
            deadline=er.deadline_estimate,
            deadline_confidence=er.deadline_confidence,
            estimated_duration_hours=er.estimated_duration_hours,
            duration_std_hours=er.duration_uncertainty,
            resources=er.resources_required,
            source_message_id=msg.message_id,
            extraction_result=er,
        )
        self.cdg.add_node(node)
        logger.debug(f"Created pending node from message: {node.node_id}")

    def _deliver_messages(self, current_time: datetime) -> List[Message]:
        """Return messages whose delivery time has passed."""
        delivered = []
        remaining = []
        for msg in self._hidden.get('future_messages', []):
            if msg.delivery_time and msg.delivery_time <= current_time:
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
        """Terminal conditions: all resolved, trust collapse, or max steps."""
        if self._current_step >= self._max_steps:
            return False, True, "max_steps"

        active = self.cdg.get_active_nodes()
        pending = self.cdg.get_pending_nodes()
        remaining_messages = len(self._hidden.get('future_messages', []))

        if not active and not pending and remaining_messages == 0 and self._current_step > 3:
            return True, False, "all_resolved"

        trust_scores = [te.trust_score for te in trust_entries.values()]
        if trust_scores and max(trust_scores) < 0.15:
            return True, False, "trust_collapse"

        return False, False, ""

    def _update_cognitive_load(self, action: AgentAction,
                               node: Optional[CommitmentNode]) -> float:
        current_load = self._state.cognitive_load if self._state else 0.0

        if node and action.action_type == ActionType.ACCEPT:
            current_load = min(1.0, current_load + node.cognitive_load_score * 0.2)

        current_load = max(0.0, current_load - 0.05)
        return current_load

    def _generate_scenario(self, stage: int) -> Dict:
        """Generate a scenario for the given curriculum stage."""
        # Try loading from scenarios directory
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
            'start_time': base_time.isoformat(),
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

    def _build_message_schedule(self, scenario: Dict,
                                start_time: datetime) -> List[Message]:
        messages = []
        for m_data in scenario.get('message_schedule', []):
            delivery_str = m_data.get('delivery_time')
            delivery_time = (
                datetime.fromisoformat(delivery_str) if delivery_str else start_time
            )
            msg = Message(
                sender_id=m_data.get('sender_id', ''),
                sender_role=StakeholderRole(m_data.get('sender_role', 'colleague')),
                content=m_data.get('content', ''),
                delivery_time=delivery_time,
            )
            messages.append(msg)
        return sorted(messages, key=lambda m: m.delivery_time or datetime.min)

    def _build_node_from_dict(self, d: Dict) -> CommitmentNode:
        deadline = None
        if d.get('deadline'):
            try:
                deadline = datetime.fromisoformat(d['deadline'])
            except (ValueError, TypeError):
                deadline = None

        return CommitmentNode(
            node_id=d.get('id', str(uuid.uuid4())[:8]),
            label=d.get('label', ''),
            description=d.get('description', ''),
            commitment_type=self._type_from_string(d.get('type', 'explicit_soft')),
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

    @staticmethod
    def _type_from_string(type_str: str) -> CommitmentType:
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
        try:
            log_path = self._log_dir / f"ep_{state.episode_id}.jsonl"
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to write step log: {e}")

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
        """Full debug dump for Colab inspection."""
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
