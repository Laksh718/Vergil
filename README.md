# VERGIL — Commitment Dependency Graph Engine

> **Research-grade RL environment for training LLMs in commitment reasoning, social trust management, and proactive renegotiation.**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-v0.0.1-blue)](https://github.com/open-env)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is VERGIL?

VERGIL models the real-world problem of **commitment overload**: when an LLM agent must juggle multiple overlapping promises to different stakeholders, detect infeasibility proactively, and renegotiate before failures cascade through a dependency graph.

**Key innovation**: Most RL environments treat tasks independently. VERGIL introduces a *Commitment Dependency Graph (CDG)* where:
- Accepting one commitment affects the feasibility of all others
- Stakeholder trust gates which actions are available
- Failure cascades propagate through temporal dependencies
- The agent must reason under partial observability (POMDP)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    VERGIL Architecture                        │
│                                                              │
│  ┌───────────┐    ┌───────────┐    ┌──────────────────┐     │
│  │ Extraction │───▶│    CDG    │───▶│   Stakeholder    │     │
│  │  Engine   │    │  Engine   │    │   Simulator      │     │
│  └───────────┘    └───────────┘    └──────────────────┘     │
│       ↑               │ ↕               ↕                   │
│  ┌────┴──────┐   ┌────┴───────┐   ┌──────────────────┐     │
│  │ Messages  │   │Satisfiab.  │   │  Multi-Dim Trust  │     │
│  │ Schedule  │   │  + Cascade │   │  (R, C, B)        │     │
│  └───────────┘   └────────────┘   └──────────────────┘     │
│                       ↕                                      │
│  ┌────────────────────┴──────────────────────────────┐      │
│  │              VERGILEnv (Gymnasium)                  │      │
│  │  12-step pipeline • POMDP hidden state • Curriculum │      │
│  └────────────────────┬──────────────────────────────┘      │
│                       ↕                                      │
│  ┌─────────────────────────────────────────────────────┐     │
│  │         7-Component Reward Function                   │     │
│  │  Fulfillment | Trust | Proactive | Feasibility Acc   │     │
│  │  Broken Penalty | OverRefusal | SilentDrop           │     │
│  └─────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/your-username/vergil.git
cd vergil
pip install -r requirements.txt
```

### 2. Run Tests

```bash
pytest vergil/tests/ -v
# ✅ 29 passed
```

### 3. Smoke Test Training

```bash
python scripts/train_colab.py --smoke-test
```

### 4. Launch Demo

```bash
pip install fastapi uvicorn
python -m uvicorn vergil.api.server:app --port 7860
# Open http://localhost:7860
```

---

## Project Structure

```
vergil/
├── core/
│   ├── types.py             # 11 enums, 10 dataclasses (foundation)
│   ├── extraction.py        # NL → commitment with confidence scoring
│   ├── cdg.py               # CDG engine (DAG, satisfiability, cascade)
│   ├── stakeholder.py       # 7-action stakeholder simulator
│   ├── reward.py            # 7-component reward + anti-hack
│   ├── env.py               # OpenEnv-compliant Gymnasium environment
│   ├── pomdp.py             # POMDP belief state (Bayesian updates)
│   ├── trust_multidim.py    # 3D trust (Reliability/Competence/Benevolence)
│   └── execution_model.py   # Lognormal duration + Poisson interruptions
├── curriculum/
│   ├── failure_db.py        # SQLite failure topology database
│   ├── curriculum_engine.py # 4-stage self-improving curriculum
│   └── scenario_generator.py# Procedural scenario synthesis
├── stakeholders/
│   └── adversarial.py       # 5 adversarial behavior patterns
├── anti_hack/
│   └── reward_guard.py      # 4 reward exploitation detectors
├── training/
│   └── evaluation.py        # 6-metric evaluation suite
├── api/
│   └── server.py            # FastAPI REST API
├── utils/
│   └── logger.py            # Structured JSON logging
└── tests/
    └── test_env.py          # 29 comprehensive tests
```

---

## Evaluation Metrics (openenv.yaml)

| Metric | Target (Stage 4) |
|--------|------------------|
| Commitment Fulfillment Rate | ≥ 70% |
| Trust Maintenance Index | ≥ 0.55 |
| Proactive Renegotiation Rate | ≥ 30% |
| Cascade Prevention Rate | ≥ 50% |
| Feasibility Accuracy (Brier) | ≤ 0.15 |
| Trust Stability (σ trajectory) | ≤ 0.12 |

---

## Curriculum Stages

| Stage | Name | Nodes | Stakeholders | Features |
|-------|------|-------|-------------|----------|
| 1 | Foundation | 2-3 | 1 | Basic accept/decline |
| 2 | Complexity | 3-6 | 2 | Implicit commitments, dependency chains |
| 3 | Social Dynamics | 5-10 | 3 | Resource conflicts, force majeure |
| 4 | Adversarial | 8-15 | 4 | Adversarial behaviors, full POMDP |

---

## Anti-Reward-Hacking

VERGIL implements 6 anti-hack mechanisms:
1. **OverRefusal Penalty**: Penalizes declining ≥60% of commitments
2. **SilentDrop Penalty**: Penalizes accepting then ignoring commitments
3. **Accept Everything Detector**: Flags ≥95% accept rate
4. **Prediction Bias Detector**: Flags constant feasibility predictions
5. **Renegotiation Farming Detector**: Flags excessive renegotiation
6. **Reward Guard**: Multiplicative penalty that modulates total reward

---

## License

MIT
