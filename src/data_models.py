"""
Data models for vehicle interactions and game states
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import numpy as np


class VehicleState(BaseModel):
    """Represents the state of a vehicle in the interaction"""
    vehicle_id: str
    distance: float = Field(..., description="Distance to interaction point (meters)")
    velocity: float = Field(..., description="Current velocity (m/s)")
    acceleration: Optional[float] = Field(None, description="Acceleration decision (m/s²)")


class InteractionScenario(BaseModel):
    """Represents a vehicle interaction scenario"""
    scenario_id: str
    frame_id: int
    vehicle_1: VehicleState
    vehicle_2: VehicleState
    scenario_type: Optional[str] = None
    ground_truth_priority: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary representation"""
        return {
            "scenario_id": self.scenario_id,
            "frame_id": self.frame_id,
            "vehicle_1": {
                "vehicle_id": self.vehicle_1.vehicle_id,
                "distance": self.vehicle_1.distance,
                "velocity": self.vehicle_1.velocity,
                "acceleration": self.vehicle_1.acceleration
            },
            "vehicle_2": {
                "vehicle_id": self.vehicle_2.vehicle_id,
                "distance": self.vehicle_2.distance,
                "velocity": self.vehicle_2.velocity,
                "acceleration": self.vehicle_2.acceleration
            },
            "scenario_type": self.scenario_type,
            "ground_truth_priority": self.ground_truth_priority
        }


class LLMDecision(BaseModel):
    """Represents decision output from LLM"""
    acceleration_1: float = Field(..., description="Acceleration decision for vehicle 1 (m/s²)")
    reasoning: str = Field(..., description="Chain-of-thought reasoning process")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    strategy_type: Optional[str] = Field(None, description="Type of game strategy chosen")


class FuzzyDecision(BaseModel):
    """Represents fuzzy priority decision output from LLM"""
    priority_vehicle: str = Field(..., description="Vehicle ID that should have priority")
    confidence: float = Field(..., description="Confidence score (0-1)")
    fuzzy_reasoning: Dict[str, float] = Field(..., description="Fuzzy weight assignments")
    textual_reasoning: str = Field(..., description="Textual reasoning process")
    risk_level: str = Field(..., description="Risk level assessment (low/medium/high/critical)")
    scenario_type: str = Field(..., description="Scenario type (merging/crossing/yielding/following/other)")


class GameMetrics(BaseModel):
    """Game quality evaluation metrics"""
    safety_score: float = Field(..., description="Safety evaluation score")
    efficiency_score: float = Field(..., description="Efficiency evaluation score")
    compliance_score: float = Field(..., description="Traffic rule compliance score")
    rationality_score: float = Field(..., description="Opponent rationality score")
    overall_score: float = Field(..., description="Overall weighted score")

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            "safety_score": self.safety_score,
            "efficiency_score": self.efficiency_score,
            "compliance_score": self.compliance_score,
            "rationality_score": self.rationality_score,
            "overall_score": self.overall_score
        }


class EvaluationResult(BaseModel):
    """Complete evaluation result for a scenario"""
    scenario_id: str
    original_decision: Optional[float]
    llm_decision: LLMDecision
    metrics: GameMetrics
    baseline_comparison: Optional[Dict[str, float]] = None
    response_time: Optional[float] = Field(None, description="LLM response time in seconds")

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result to dictionary"""
        result = {
            "scenario_id": self.scenario_id,
            "original_decision": self.original_decision,
            "llm_decision": {
                "acceleration_1": self.llm_decision.acceleration_1,
                "reasoning": self.llm_decision.reasoning,
                "confidence": self.llm_decision.confidence,
                "strategy_type": self.llm_decision.strategy_type
            },
            "metrics": self.metrics.to_dict(),
            "response_time": self.response_time
        }
        if self.baseline_comparison:
            result["baseline_comparison"] = self.baseline_comparison
        return result