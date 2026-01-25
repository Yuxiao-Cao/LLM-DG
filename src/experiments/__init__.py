"""
Experiments module for LLM-DG

This module contains experimental evaluation protocols including
closed-loop rollout simulations for multi-step decision making.
"""

from .closed_loop import (
    ClosedLoopRollout,
    parse_acceleration_from_llm_response,
    update_kinematic_state,
    compute_step_metrics,
    RolloutStep,
    RolloutResult,
    save_aggregate_statistics,
    compute_aggregate_statistics
)

__all__ = [
    "ClosedLoopRollout",
    "parse_acceleration_from_llm_response",
    "update_kinematic_state",
    "compute_step_metrics",
    "RolloutStep",
    "RolloutResult",
    "save_aggregate_statistics",
    "compute_aggregate_statistics"
]
