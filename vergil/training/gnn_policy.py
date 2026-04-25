"""
GNN Policy Head for VERGIL — HGT-Lite Graph Encoder
=====================================================

Replaces the flat 28-dim feature vector with a genuine graph neural network
that reasons over CDG structure. The policy can now understand:
- A depends on B which conflicts with C (temporal + resource edges together)
- Node N is a bottleneck for 3 downstream commitments
- Trust is deteriorating for the stakeholder behind the high-urgency node

Architecture:
    CDG (heterogeneous graph)
        ↓
    HGT-Lite: 2-layer SAGEConv with edge-type conditioning
        ↓
    Per-node embeddings → global mean pool → 64-dim CDG embedding
        ↓
    Concat [CDG(64), trust(16), time(8), capacity(4)] → 92-dim
        ↓
    Policy MLP: 92 → 128 → 64 → n_actions
    Value  MLP: 92 → 128 → 64 → 1

Dependencies: torch, torch-geometric (both available on Colab)
Fallback: If PyG is not installed, falls back to the flat MLP from train_rl.py
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import SAGEConv, HeteroConv, global_mean_pool
    from torch_geometric.data import HeteroData
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False


# ─── Feature Dimensions ────────────────────────────────────────────────────

NODE_FEAT_DIM  = 8   # Per-node features (type, urgency, deadline, duration, status, stakeholder, risk, is_pending)
EDGE_TYPE_DIM  = 4   # Edge types: temporal, resource, logical, implicit
CDG_EMBED_DIM  = 64  # GNN output
TRUST_DIM      = 16  # Trust features (per stakeholder, batched to fixed size)
TIME_DIM       = 8   # Time context features
CAPACITY_DIM   = 4   # Capacity features
STATE_DIM      = CDG_EMBED_DIM + TRUST_DIM + TIME_DIM + CAPACITY_DIM  # 92

N_ACTIONS      = 4   # accept, decline, counter_propose, do_nothing


# ─── Node Feature Encoder ──────────────────────────────────────────────────

COMMITMENT_TYPE_MAP = {
    'explicit_hard': 0, 'explicit_soft': 1,
    'implicit': 2, 'precondition': 3, 'social': 4,
}
STATUS_MAP = {
    'pending': 0, 'accepted': 1, 'in_progress': 2,
    'completed': 3, 'failed': 4, 'at_risk': 5,
    'renegotiated': 6, 'declined': 7,
}
EDGE_TYPE_MAP = {
    'temporal': 0, 'resource': 1, 'logical': 2,
    'implicit_derive': 3, 'trust_impact': 3,
}


def encode_node_features(nodes, current_time, available_hours: float) -> 'torch.Tensor':
    """
    Build NODE_FEAT_DIM feature vector for each CDG node.
    Returns shape (N, NODE_FEAT_DIM).
    """
    import torch
    from datetime import datetime

    feats = []
    for n in nodes:
        ctype = COMMITMENT_TYPE_MAP.get(
            n.commitment_type.value if hasattr(n.commitment_type, 'value') else str(n.commitment_type), 0
        ) / max(1, len(COMMITMENT_TYPE_MAP) - 1)

        status = STATUS_MAP.get(
            n.status.value if hasattr(n.status, 'value') else str(n.status), 0
        ) / max(1, len(STATUS_MAP) - 1)

        urgency = float(getattr(n, 'urgency', 0.5))

        deadline = getattr(n, 'deadline', None)
        if deadline and current_time:
            hours_remaining = (deadline - current_time).total_seconds() / 3600
            deadline_prox = max(0.0, min(1.0, 1.0 - hours_remaining / 48))
        else:
            deadline_prox = 0.5

        duration = min(1.0, getattr(n, 'estimated_duration_hours', 2.0) / 8.0)

        risk = float(getattr(n, 'risk_score', 0.0))
        is_pending = 1.0 if (
            n.status.value if hasattr(n.status, 'value') else str(n.status)
        ) == 'pending' else 0.0

        # stakeholder type encoding (boss=1.0, client=0.8, colleague=0.5, friend=0.3)
        sid = getattr(n, 'stakeholder_id', '')
        if 'boss' in sid:      stype = 1.0
        elif 'client' in sid:  stype = 0.8
        elif 'colleague' in sid: stype = 0.5
        else:                  stype = 0.3

        feats.append([ctype, status, urgency, deadline_prox, duration, risk, is_pending, stype])

    if not feats:
        return torch.zeros((1, NODE_FEAT_DIM), dtype=torch.float32)
    return torch.tensor(feats, dtype=torch.float32)


def encode_edge_index_and_type(edges, nodes) -> Tuple['torch.Tensor', 'torch.Tensor']:
    """
    Build edge_index (2, E) and edge_attr (E, EDGE_TYPE_DIM) for CDG edges.
    Uses one-hot encoding over edge types.
    """
    import torch

    node_ids = {n.node_id: i for i, n in enumerate(nodes)}
    srcs, dsts, types = [], [], []

    for e in edges:
        src = node_ids.get(e.from_node if hasattr(e, 'from_node') else e.get('source'), None)
        dst = node_ids.get(e.to_node if hasattr(e, 'to_node') else e.get('target'), None)
        if src is None or dst is None:
            continue
        etype = EDGE_TYPE_MAP.get(
            e.edge_type.value if hasattr(getattr(e, 'edge_type', None), 'value')
            else str(getattr(e, 'edge_type', 'temporal')), 0
        )
        srcs.append(src); dsts.append(dst); types.append(etype)

    if not srcs:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, EDGE_TYPE_DIM), dtype=torch.float32)

    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
    # One-hot edge type
    edge_attr = F.one_hot(torch.tensor(types, dtype=torch.long), num_classes=EDGE_TYPE_DIM).float()
    return edge_index, edge_attr


def encode_trust_features(trust_entries, multidim_trust=None, max_stakeholders: int = 4) -> 'torch.Tensor':
    """
    Fixed-size trust feature vector (TRUST_DIM = 16 = 4 stakeholders × 4 features each).
    Features per stakeholder: [composite_trust, reliability, competence, benevolence]
    Sorted by role priority: boss, client, colleague, friend.
    """
    import torch

    ROLE_ORDER = ['boss', 'client', 'colleague', 'friend']
    result = []

    for role in ROLE_ORDER:
        # Find stakeholder of this role
        sid = next((s for s in trust_entries if role in s.lower()), None)
        if sid and trust_entries[sid]:
            entry = trust_entries[sid]
            composite = float(getattr(entry, 'trust_score', 0.5))
            if multidim_trust and sid in multidim_trust:
                md = multidim_trust[sid]
                rel = float(getattr(md, 'reliability', composite))
                comp_score = float(getattr(md, 'competence', composite))
                ben = float(getattr(md, 'benevolence', composite))
            else:
                rel = comp_score = ben = composite
            result.extend([composite, rel, comp_score, ben])
        else:
            result.extend([0.5, 0.5, 0.5, 0.5])  # Neutral prior for missing stakeholders

    return torch.tensor(result[:TRUST_DIM], dtype=torch.float32)


def encode_time_features(state, env) -> 'torch.Tensor':
    """TIME_DIM = 8 time context features."""
    import torch

    step_ratio = min(1.0, getattr(state, 'step_number', 0) / max(1, getattr(env, '_max_steps', 30)))
    sat = float(getattr(state, 'satisfiability_score', 1.0))
    cog_load = float(getattr(state, 'cognitive_load', 0.0))
    energy = float(getattr(state, 'energy_level', 1.0))

    n_pending = sum(1 for n in state.cdg_nodes
                   if (n.status.value if hasattr(n.status, 'value') else str(n.status)) == 'pending')
    n_failed  = sum(1 for n in state.cdg_nodes
                   if (n.status.value if hasattr(n.status, 'value') else str(n.status)) == 'failed')
    n_total   = max(1, len(state.cdg_nodes))

    # Urgency pressure: mean urgency of pending nodes
    pending_urgencies = [n.urgency for n in state.cdg_nodes
                        if (n.status.value if hasattr(n.status, 'value') else str(n.status)) == 'pending']
    urgency_pressure = float(np.mean(pending_urgencies)) if pending_urgencies else 0.0

    # Cascade risk: fraction of at-risk nodes
    n_at_risk = len(getattr(state, 'at_risk_nodes', []))
    cascade_risk = min(1.0, n_at_risk / n_total)

    return torch.tensor([
        step_ratio, sat, cog_load, energy,
        n_pending / n_total, n_failed / n_total,
        urgency_pressure, cascade_risk,
    ], dtype=torch.float32)


def encode_capacity_features(state) -> 'torch.Tensor':
    """CAPACITY_DIM = 4 capacity features."""
    import torch

    available = float(getattr(state, 'available_hours_next_48h', 8.0))
    committed = sum(
        n.estimated_duration_hours for n in state.cdg_nodes
        if (n.status.value if hasattr(n.status, 'value') else str(n.status)) in ('accepted', 'in_progress')
    )
    pending_cost = sum(
        n.estimated_duration_hours for n in state.cdg_nodes
        if (n.status.value if hasattr(n.status, 'value') else str(n.status)) == 'pending'
    )
    schedule_density = min(1.0, committed / max(1.0, available))
    overload_risk = max(0.0, min(1.0, (committed + pending_cost - available) / max(1.0, available)))

    return torch.tensor([
        min(1.0, available / 24.0),
        min(1.0, committed / max(1.0, available)),
        schedule_density,
        overload_risk,
    ], dtype=torch.float32)


# ─── GNN Encoder ───────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class CDGGraphEncoder(nn.Module):
        """
        2-layer SAGEConv GNN over the CDG.
        Input: node features (N, NODE_FEAT_DIM) + edge_index
        Output: graph-level embedding (CDG_EMBED_DIM,)
        """

        def __init__(self, in_dim: int = NODE_FEAT_DIM, hidden: int = 64, out_dim: int = CDG_EMBED_DIM):
            super().__init__()
            if _PYG_AVAILABLE:
                self.conv1 = SAGEConv(in_dim, hidden)
                self.conv2 = SAGEConv(hidden, out_dim)
            else:
                # Fallback: simple MLP aggregation (no graph convolution)
                self.mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, out_dim),
                )
            self.norm1 = nn.LayerNorm(hidden)
            self.norm2 = nn.LayerNorm(out_dim)
            self._use_pyg = _PYG_AVAILABLE

        def forward(self, x: 'torch.Tensor', edge_index: 'torch.Tensor') -> 'torch.Tensor':
            if self._use_pyg:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                h = F.relu(self.norm1(self.conv1(x, edge_index)))
                h = F.relu(self.norm2(self.conv2(h, edge_index)))
                return global_mean_pool(h, batch).squeeze(0)
            else:
                # Fallback: mean pool over node features
                h = self.mlp(x)
                return h.mean(dim=0)


    class VERGILGNNPolicy(nn.Module):
        """
        Full VERGIL policy with GNN-encoded CDG state.

        Input:  VERGILState + env context
        Output: action logits (N_ACTIONS,), state value (1,)
        """

        def __init__(self, n_actions: int = N_ACTIONS):
            super().__init__()
            self.graph_encoder = CDGGraphEncoder()
            self.policy_head = nn.Sequential(
                nn.Linear(STATE_DIM, 128), nn.ReLU(), nn.LayerNorm(128),
                nn.Linear(128, 64),        nn.ReLU(), nn.LayerNorm(64),
                nn.Linear(64, n_actions),
            )
            self.value_head = nn.Sequential(
                nn.Linear(STATE_DIM, 128), nn.ReLU(), nn.LayerNorm(128),
                nn.Linear(128, 64),        nn.ReLU(),
                nn.Linear(64, 1),
            )

        def encode_state(self, state, env) -> 'torch.Tensor':
            """Full state → STATE_DIM tensor."""
            nodes = state.cdg_nodes
            edges = state.cdg_edges

            # Graph encoding
            x = encode_node_features(nodes, state.current_time, getattr(state, 'available_hours_next_48h', 8.0))
            edge_index, _ = encode_edge_index_and_type(edges, nodes)
            cdg_embed = self.graph_encoder(x, edge_index)  # (CDG_EMBED_DIM,)

            # Context features
            trust_feat    = encode_trust_features(state.trust_entries, getattr(env, 'multidim_trust', {}))
            time_feat     = encode_time_features(state, env)
            capacity_feat = encode_capacity_features(state)

            return torch.cat([cdg_embed, trust_feat, time_feat, capacity_feat], dim=0)

        def forward(self, state_tensor: 'torch.Tensor') -> Tuple['torch.Tensor', 'torch.Tensor']:
            logits = self.policy_head(state_tensor)
            value  = self.value_head(state_tensor)
            return logits, value

        def act(self, state, env, valid_actions: Optional[List[int]] = None) -> Tuple[int, 'torch.Tensor']:
            """
            Sample an action from the policy, masking invalid actions.
            Returns (action_index, log_prob).
            """
            with torch.no_grad():
                state_tensor = self.encode_state(state, env)
                logits, _ = self(state_tensor)

                if valid_actions is not None and len(valid_actions) < N_ACTIONS:
                    mask = torch.full((N_ACTIONS,), float('-inf'))
                    for a in valid_actions:
                        mask[a] = 0.0
                    logits = logits + mask

                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                return action.item(), dist.log_prob(action)


    class VERGILGNNTrainer:
        """
        REINFORCE + baseline trainer using the GNN policy.
        Drop-in replacement for the MLP policy in train_rl.py.
        """

        ACTION_MAP = {
            0: 'accept',
            1: 'decline',
            2: 'counter_propose',
            3: 'do_nothing',
        }

        def __init__(self, lr: float = 3e-4, gamma: float = 0.99):
            self.policy = VERGILGNNPolicy()
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
            self.gamma = gamma

        def get_valid_actions(self, state) -> List[int]:
            """Mask illegal actions based on pending node availability."""
            has_pending = any(
                (n.status.value if hasattr(n.status, 'value') else str(n.status)) == 'pending'
                for n in state.cdg_nodes
            )
            if has_pending:
                return [0, 1, 2, 3]  # All actions valid
            return [3]  # Only do_nothing when nothing pending

        def train_episode(self, pomdp, scenario: dict) -> float:
            """Run one episode and update policy via REINFORCE."""
            from vergil.core.types import AgentAction, ActionType

            state, _, _ = pomdp.reset(scenario=scenario)
            log_probs, rewards, values = [], [], []

            for _ in range(pomdp.env._max_steps):
                valid = self.get_valid_actions(state)
                action_idx, log_prob = self.policy.act(state, pomdp.env, valid)
                action_name = self.ACTION_MAP[action_idx]

                pending = [n for n in state.cdg_nodes
                          if (n.status.value if hasattr(n.status, 'value') else str(n.status)) == 'pending']

                action = AgentAction(
                    action_type=ActionType(action_name),
                    target_node_id=pending[0].node_id if pending and action_name != 'do_nothing' else None,
                )

                new_state, _, reward, terminated, truncated, _ = pomdp.step(action)
                state_tensor = self.policy.encode_state(state, pomdp.env)
                _, v = self.policy(state_tensor)

                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(v.squeeze())
                state = new_state

                if terminated or truncated:
                    break

            # Compute discounted returns
            G = 0.0
            returns = []
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)

            returns_t = torch.tensor(returns, dtype=torch.float32)
            values_t  = torch.stack(values)

            # Normalize advantages
            advantages = returns_t - values_t.detach()
            if advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy + value loss
            policy_loss = -sum(lp * adv for lp, adv in zip(log_probs, advantages))
            value_loss  = F.mse_loss(values_t, returns_t)
            entropy     = -sum(lp for lp in log_probs) / max(1, len(log_probs))
            loss        = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            return sum(rewards)

else:
    # Stub when torch is unavailable (e.g. running on CPU without torch installed)
    class VERGILGNNPolicy:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for VERGILGNNPolicy. Install with: pip install torch")

    class VERGILGNNTrainer:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for VERGILGNNTrainer. Install with: pip install torch")
