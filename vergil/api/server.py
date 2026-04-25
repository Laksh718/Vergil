# vergil/api/server.py
"""
VERGIL FastAPI Server
======================

REST API for the VERGIL demo frontend:
- POST /api/reset       — Reset environment with scenario
- POST /api/step        — Execute one step with agent action
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
from datetime import datetime
import json
import logging
from pathlib import Path

from vergil.core.env import VERGILEnv
from vergil.core.pomdp import POMDPWrapper
from vergil.core.types import AgentAction, ActionType, CommitmentStatus

logger = logging.getLogger('vergil.api')

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VERGIL Commitment Engine",
    description="Interactive demo of the Commitment Dependency Graph engine",
    version="0.1.0",
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
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

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

    # Add curriculum stage
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

