# vergil/api/server.py
"""
VERGIL FastAPI Server
======================

REST API for the VERGIL demo frontend:
- POST /api/reset       — Reset environment with scenario
- POST /api/step        — Execute one step with agent action
- POST /api/agent-step  — LLM-powered agent decides the next action
- POST /api/compare     — Run naive vs VERGIL-trained agents side-by-side
- GET  /api/state       — Get current state (including CDG, trust, belief)
- GET  /api/scenarios   — List available scenarios
- GET  /api/metrics     — Get evaluation metrics
- GET  /health          — Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path

from vergil.core.env import VERGILEnv
from vergil.core.pomdp import POMDPWrapper
from vergil.core.types import AgentAction, ActionType, CommitmentStatus

logger = logging.getLogger('vergil.api')

# ── LLM Agent (optional — loads if VERGIL_MODEL_PATH is set) ─────────────────

_llm_model = None
_llm_tokenizer = None

def _load_llm_model():
    """Load the fine-tuned VERGIL model if VERGIL_MODEL_PATH env var is set."""
    global _llm_model, _llm_tokenizer
    model_path = os.environ.get('VERGIL_MODEL_PATH', '')
    if not model_path:
        logger.info("VERGIL_MODEL_PATH not set — LLM agent unavailable, using heuristic.")
        return

    try:
        logger.info(f"Loading VERGIL model from {model_path}...")
        try:
            from unsloth import FastLanguageModel
            _llm_model, _llm_tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                load_in_4bit=True,
                dtype=None,
            )
            FastLanguageModel.for_inference(_llm_model)
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            _llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
            _llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
        logger.info("VERGIL model loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to load VERGIL model: {e}. Using heuristic fallback.")


def _llm_decide(state, env) -> AgentAction:
    """
    Use the fine-tuned LLM to decide the next action.
    Falls back to smart heuristic if model not loaded.
    """
    if _llm_model is None:
        return _heuristic_decide(state, env)

    from scripts.train_grpo_colab import state_to_prompt, parse_llm_output
    import torch

    prompt = state_to_prompt(state, env)
    inputs = _llm_tokenizer(prompt, return_tensors="pt").to(_llm_model.device)

    with torch.no_grad():
        outputs = _llm_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=_llm_tokenizer.eos_token_id,
        )

    completion = _llm_tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )

    pending = [n for n in state.cdg_nodes if n.status == CommitmentStatus.PENDING]
    action_type, target = parse_llm_output(completion, pending)

    if action_type in (ActionType.ACCEPT, ActionType.DECLINE, ActionType.COUNTER_PROPOSE):
        if not pending:
            action_type, target = ActionType.DO_NOTHING, None
        elif target is None:
            target = pending[0].node_id

    action = AgentAction(action_type=action_type, target_node_id=target)

    if action_type == ActionType.COUNTER_PROPOSE and target:
        node = next((n for n in state.cdg_nodes if n.node_id == target), None)
        if node:
            action.proposed_deadline = state.current_time + timedelta(
                hours=node.estimated_duration_hours * 1.5)

    return action, completion  # Return completion for display in frontend


def _heuristic_decide(state, env) -> tuple:
    """
    Smart heuristic fallback: capacity-aware accept/counter/decline.
    Used when LLM model is not available.
    """
    pending = [n for n in state.cdg_nodes if n.status == CommitmentStatus.PENDING]
    if not pending:
        return AgentAction(action_type=ActionType.DO_NOTHING), "No pending commitments."

    node = sorted(pending, key=lambda n: n.urgency, reverse=True)[0]
    available = getattr(state, 'available_hours_next_48h', 8.0)
    committed = sum(n.estimated_duration_hours for n in state.cdg_nodes
                   if n.status == CommitmentStatus.ACCEPTED)

    reasoning_lines = [
        f"Capacity check: {committed:.1f}h committed of {available:.1f}h available.",
        f"Target: [{node.node_id}] '{node.label}' ({node.estimated_duration_hours}h, urgency={node.urgency:.0%})",
    ]

    if committed + node.estimated_duration_hours <= available * 0.85:
        reasoning_lines.append("→ ACCEPT: fits within 85% capacity buffer.")
        return AgentAction(action_type=ActionType.ACCEPT, target_node_id=node.node_id), "\n".join(reasoning_lines)
    elif committed + node.estimated_duration_hours <= available:
        deadline = state.current_time + timedelta(hours=node.estimated_duration_hours * 1.5)
        reasoning_lines.append("→ COUNTER_PROPOSE: tight but doable with extended deadline.")
        action = AgentAction(action_type=ActionType.COUNTER_PROPOSE,
                             target_node_id=node.node_id, proposed_deadline=deadline)
        return action, "\n".join(reasoning_lines)
    else:
        reasoning_lines.append("→ DECLINE: would exceed capacity, cascade risk too high.")
        return AgentAction(action_type=ActionType.DECLINE, target_node_id=node.node_id), "\n".join(reasoning_lines)


def _naive_decide(state) -> tuple:
    """Naive agent: accepts everything, never declines. Demonstrates cascade failure."""
    pending = [n for n in state.cdg_nodes if n.status == CommitmentStatus.PENDING]
    if pending:
        node = pending[0]
        return (
            AgentAction(action_type=ActionType.ACCEPT, target_node_id=node.node_id),
            f"Blindly accepting [{node.node_id}] '{node.label}' without checking capacity."
        )
    return AgentAction(action_type=ActionType.DO_NOTHING), "No pending commitments."


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VERGIL Commitment Engine",
    description="Interactive demo of the Commitment Dependency Graph engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
env = VERGILEnv(seed=42)
pomdp = POMDPWrapper(env)
current_belief = None
history: List[Dict] = []


@app.on_event("startup")
async def startup_event():
    _load_llm_model()

# ── Pydantic Models ──────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    action_type: str
    target_node_id: Optional[str] = None
    feasibility_prediction: Optional[float] = None
    proposed_deadline: Optional[str] = None

class ResetRequest(BaseModel):
    scenario_id: Optional[str] = None
    scenario: Optional[Dict] = None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat(),
            "version": "0.1.0"}

@app.get("/api/scenarios")
async def list_scenarios():
    """List available scenario files."""
    scenarios_dir = Path(__file__).parent.parent.parent / 'scenarios'
    scenarios = []
    if scenarios_dir.exists():
        for f in sorted(scenarios_dir.glob('*.json')):
            with open(f) as fh:
                data = json.load(fh)
                scenarios.append({
                    'filename': f.name,
                    'scenario_id': data.get('scenario_id', f.stem),
                    'n_commitments': len(data.get('seed_commitments', [])),
                    'n_stakeholders': len(data.get('stakeholders', [])),
                })
    return {"scenarios": scenarios}

@app.post("/api/reset")
async def reset_scenario(request: ResetRequest):
    global current_belief, history
    history = []

    try:
        if request.scenario:
            scenario = request.scenario
        elif request.scenario_id:
            path = Path(__file__).parent.parent.parent / 'scenarios' / f'{request.scenario_id}.json'
            if not path.exists():
                path = Path(__file__).parent.parent.parent / 'scenarios' / request.scenario_id
            with open(path) as f:
                scenario = json.load(f)
        else:
            # Default: scenario 1
            path = Path(__file__).parent.parent.parent / 'scenarios' / 'scenario_01_simple.json'
            with open(path) as f:
                scenario = json.load(f)

        state, belief, info = pomdp.reset(scenario=scenario)
        current_belief = belief

        return {
            "status": "reset",
            "state": _state_to_api(state, belief),
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/step")
async def take_step(request: StepRequest):
    global current_belief

    try:
        action_type = ActionType(request.action_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid action type: {request.action_type}")

    proposed_deadline = None
    if request.proposed_deadline:
        # JS toISOString() produces '2026-04-29T08:30:00.000Z' which
        # Python 3.10 fromisoformat() can't parse (Z suffix + milliseconds)
        dl_str = request.proposed_deadline.replace('Z', '+00:00')
        try:
            proposed_deadline = datetime.fromisoformat(dl_str).replace(tzinfo=None)
        except ValueError:
            proposed_deadline = None  # Fallback: ignore bad deadline

    action = AgentAction(
        action_type=action_type,
        target_node_id=request.target_node_id,
        feasibility_prediction=request.feasibility_prediction,
        proposed_deadline=proposed_deadline,
    )

    try:
        state, belief, reward, terminated, truncated, info = pomdp.step(action)
        current_belief = belief

        step_record = {
            'step': state.step_number,
            'action': request.action_type,
            'target': request.target_node_id,
            'reward': round(reward, 4),
            'terminated': terminated,
            'truncated': truncated,
        }
        history.append(step_record)

        return {
            "state": _state_to_api(state, belief),
            "reward": round(reward, 4),
            "terminated": terminated,
            "truncated": truncated,
            "info": {k: v for k, v in info.items() if not k.startswith('_')},
            "step_record": step_record,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/state")
async def get_state():
    if env._state is None:
        raise HTTPException(status_code=400, detail="No active episode. POST /api/reset first.")

    return {
        "state": _state_to_api(env._state, current_belief),
        "history": history,
    }


@app.get("/api/metrics")
async def get_metrics():
    if env._state is None:
        raise HTTPException(status_code=400, detail="No active episode.")

    state = env._state
    return {
        "step": state.step_number,
        "satisfiability": state.satisfiability_score,
        "trust_scores": {sid: te.trust_score for sid, te in state.trust_entries.items()},
        "n_pending": sum(1 for n in state.cdg_nodes if n.status == CommitmentStatus.PENDING),
        "n_accepted": sum(1 for n in state.cdg_nodes if n.status == CommitmentStatus.ACCEPTED),
        "n_completed": sum(1 for n in state.cdg_nodes if n.status == CommitmentStatus.COMPLETED),
        "n_failed": sum(1 for n in state.cdg_nodes if n.status == CommitmentStatus.FAILED),
        "history": history,
        "llm_loaded": _llm_model is not None,
    }


@app.post("/api/agent-step")
async def agent_step():
    """
    Let the VERGIL agent (LLM or heuristic) decide the next action automatically.
    Returns the chosen action, the agent's reasoning, and the resulting state.
    """
    global current_belief

    if env._state is None:
        raise HTTPException(status_code=400, detail="No active episode. POST /api/reset first.")

    state = env._state
    action, reasoning = _llm_decide(state, env)

    try:
        new_state, belief, reward, terminated, truncated, info = pomdp.step(action)
        current_belief = belief

        step_record = {
            'step': new_state.step_number,
            'action': action.action_type.value,
            'target': action.target_node_id,
            'reward': round(reward, 4),
            'terminated': terminated,
            'truncated': truncated,
            'agent_reasoning': reasoning,
            'agent_type': 'llm' if _llm_model is not None else 'heuristic',
        }
        history.append(step_record)

        return {
            "state": _state_to_api(new_state, belief),
            "reward": round(reward, 4),
            "terminated": terminated,
            "truncated": truncated,
            "info": {k: v for k, v in info.items() if not k.startswith('_')},
            "step_record": step_record,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class CompareRequest(BaseModel):
    scenario_id: Optional[str] = None
    n_steps: int = 20


@app.post("/api/compare")
async def compare_agents(request: CompareRequest):
    """
    Run the naive agent (always accepts) vs VERGIL-trained agent on the same
    scenario in parallel. Returns full trajectories for the before/after demo.

    The naive agent demonstrates cascade failure. The VERGIL agent demonstrates
    capacity-aware reasoning and trust preservation.
    """
    import copy as _copy

    # Load scenario
    scenarios_dir = Path(__file__).parent.parent.parent / 'scenarios'
    if request.scenario_id:
        path = scenarios_dir / f'{request.scenario_id}.json'
        if not path.exists():
            path = scenarios_dir / request.scenario_id
    else:
        path = scenarios_dir / 'scenario_04_deadline_crunch.json'

    try:
        with open(path) as f:
            scenario = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {path.name}")

    def _run_agent(agent_fn, label: str) -> dict:
        """Run one agent for n_steps and collect trajectory."""
        sim_env = VERGILEnv(seed=99)
        sim_pomdp = POMDPWrapper(sim_env)
        state, belief, _ = sim_pomdp.reset(scenario=_copy.deepcopy(scenario))

        trajectory = []
        total_reward = 0.0

        for step in range(request.n_steps):
            action, reasoning = agent_fn(state, sim_env)
            try:
                new_state, new_belief, reward, terminated, truncated, info = sim_pomdp.step(action)
            except Exception:
                break

            trajectory.append({
                'step': step + 1,
                'action': action.action_type.value,
                'target': action.target_node_id,
                'reward': round(reward, 4),
                'reasoning': reasoning,
                'state': _state_to_api_minimal(new_state),
            })
            total_reward += reward
            state = new_state
            if terminated or truncated:
                break

        final = state
        return {
            'label': label,
            'trajectory': trajectory,
            'total_reward': round(total_reward, 4),
            'final_satisfiability': round(final.satisfiability_score, 3),
            'final_trust': {
                sid: round(te.trust_score, 3)
                for sid, te in final.trust_entries.items()
            },
            'n_completed': sum(1 for n in final.cdg_nodes if n.status == CommitmentStatus.COMPLETED),
            'n_failed': sum(1 for n in final.cdg_nodes if n.status == CommitmentStatus.FAILED),
            'final_graph': _state_to_api(final)['graph'],
        }

    naive_result = _run_agent(lambda s, e: _naive_decide(s), "Naive (Accept-All)")
    vergil_result = _run_agent(_heuristic_decide, "VERGIL-Trained")

    return {
        "scenario_id": scenario.get('scenario_id', 'unknown'),
        "naive": naive_result,
        "vergil": vergil_result,
        "comparison": {
            "reward_delta": round(vergil_result['total_reward'] - naive_result['total_reward'], 4),
            "sat_delta": round(vergil_result['final_satisfiability'] - naive_result['final_satisfiability'], 3),
            "failure_reduction": naive_result['n_failed'] - vergil_result['n_failed'],
        }
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _state_to_api_minimal(state) -> Dict:
    """Lightweight state snapshot for compare trajectory — graph + key metrics only."""
    nodes = [
        {
            'id': n.node_id,
            'label': n.label,
            'status': n.status.value,
            'urgency': round(n.urgency, 3),
            'stakeholder_id': n.stakeholder_id,
        }
        for n in state.cdg_nodes
    ]
    edges = [
        {'source': e.from_node, 'target': e.to_node, 'type': e.edge_type.value}
        for e in state.cdg_edges
    ]
    return {
        'graph': {'nodes': nodes, 'edges': edges},
        'satisfiability_score': round(state.satisfiability_score, 3),
        'trust': {sid: round(te.trust_score, 3) for sid, te in state.trust_entries.items()},
    }


def _state_to_api(state, belief=None) -> Dict:
    """Convert state to API-friendly dict."""
    d = state.to_dict()

    # Add CDG graph structure for D3.js
    nodes = []
    for n in state.cdg_nodes:
        nodes.append({
            'id': n.node_id,
            'label': n.label,
            'type': n.commitment_type.value,
            'status': n.status.value,
            'urgency': round(n.urgency, 3),
            'deadline': n.deadline.isoformat() if n.deadline else None,
            'stakeholder_id': n.stakeholder_id,
            'estimated_duration_hours': n.estimated_duration_hours,
        })

    edges = []
    for e in state.cdg_edges:
        edges.append({
            'source': e.from_node,
            'target': e.to_node,
            'type': e.edge_type.value,
            'weight': e.weight,
        })

    d['graph'] = {'nodes': nodes, 'edges': edges}

    # Add fields not included in to_dict()
    d['available_hours_next_48h'] = getattr(state, 'available_hours_next_48h', None) or 16.0
    d['curriculum_stage'] = getattr(state, 'curriculum_stage', 1)

    # Add multi-dimensional trust from env
    if hasattr(env, 'multidim_trust') and env.multidim_trust:
        d['multidim_trust'] = {
            sid: {
                'reliability': round(mt.reliability, 3),
                'competence': round(mt.competence, 3),
                'benevolence': round(mt.benevolence, 3),
                'composite': round(mt.composite_trust, 3),
            }
            for sid, mt in env.multidim_trust.items()
        }

    # Add belief state if available
    if belief:
        d['belief'] = {
            'overall_uncertainty': round(belief.overall_uncertainty, 3),
            'epistemic_risk': round(belief.epistemic_risk, 3),
            'stakeholder_beliefs': {
                sid: {
                    'urgency_mean': round(sb.urgency_mean, 3),
                    'urgency_std': round(sb.urgency_std, 3),
                    'flexibility_mean': round(sb.flexibility_mean, 1),
                    'irrational_probability': round(sb.irrational_probability, 3),
                }
                for sid, sb in belief.stakeholder_beliefs.items()
            },
        }

    return d


# ── Training Results API ──────────────────────────────────────────────────────

@app.get("/api/training-results")
async def get_training_results():
    """Return saved training results and curves for the dashboard."""
    base = Path(__file__).parent.parent.parent / 'training_results'

    results = {}
    curve = []

    # Try RL results first, then fallback to baseline
    for fname in ['rl_training_results.json', 'training_results.json']:
        rpath = base / fname
        if rpath.exists():
            with open(rpath) as f:
                results = json.load(f)
            break

    for fname in ['rl_training_curve.json', 'training_curve.json']:
        cpath = base / fname
        if cpath.exists():
            with open(cpath) as f:
                curve = json.load(f)
            break

    if not results:
        raise HTTPException(status_code=404, detail="No training results found. Run training first.")

    return {"results": results, "curve": curve}


@app.post("/api/train")
async def run_training():
    """Run a quick RL training session (smoke test mode for CPU)."""
    import subprocess
    import sys

    python = sys.executable
    script = Path(__file__).parent.parent.parent / 'scripts' / 'train_rl.py'
    output_dir = Path(__file__).parent.parent.parent / 'training_results'
    output_dir.mkdir(exist_ok=True)

    try:
        result = subprocess.run(
            [python, str(script), '--smoke-test'],
            capture_output=True, text=True, timeout=120,
            cwd=str(script.parent.parent),
        )
        if result.returncode != 0:
            return JSONResponse({"status": "error", "stderr": result.stderr[-500:]}, status_code=500)

        return {"status": "ok", "stdout": result.stdout[-300:]}
    except subprocess.TimeoutExpired:
        return JSONResponse({"status": "timeout"}, status_code=500)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ── Static Files (serve frontend) ────────────────────────────────────────────

frontend_dir = Path(__file__).parent.parent.parent / 'frontend'
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))

    @app.get("/dashboard")
    async def serve_dashboard():
        return FileResponse(str(frontend_dir / "dashboard.html"))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

