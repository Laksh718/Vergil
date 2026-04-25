"""
VERGIL: Commitment Dependency Graph Engine
==========================================

A reinforcement learning environment for training LLMs on pre-commitment
reasoning. Models learn to evaluate commitment feasibility, manage trust
across relationships, and proactively renegotiate when constraints change.

Architecture:
    core/       — CDG data structures, constraint solver, reward function
    curriculum/  — Self-improving curriculum with failure topology tracking
    training/    — GRPO training pipeline with HGT graph encoder
    api/         — FastAPI server for HuggingFace Spaces deployment
"""

__version__ = "1.2.0"
__author__ = "VERGIL Team"
