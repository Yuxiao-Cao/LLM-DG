"""
Evaluation system for game quality metrics and opponent rationality regulation
Modified to use evaluation4.py function definitions for PreciseEvaluator
"""

import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from .data_models import InteractionScenario, LLMDecision, GameMetrics, EvaluationResult, FuzzyDecision


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics - using evaluation4.py parameters"""
    safety_weight: float = 0.4
    efficiency_weight: float = 0.25
    compliance_weight: float = 0.25
    rationality_weight: float = 0.1

    # Key parameters from evaluation4.py
    min_safe_distance: float = 5.0
    max_safe_speed: float = 15.0
    max_acceleration: float = 3.0
    reaction_time: float = 1.5
    intersection_width: float = 10.0
    target_speed: float = 10.0
    speed_limit: float = 15.0
    yield_distance: float = 15.0

    # Legacy parameters for backward compatibility
    safety_params: Dict[str, float] = None
    efficiency_params: Dict[str, float] = None
    compliance_params: Dict[str, float] = None

    def __post_init__(self):
        # Initialize legacy parameters for compatibility
        if self.safety_params is None:
            self.safety_params = {
                "min_safe_distance": self.min_safe_distance,
                "max_safe_speed": self.max_safe_speed,
                "max_acceleration": self.max_acceleration,
                "collision_penalty": 100.0
            }

        if self.efficiency_params is None:
            self.efficiency_params = {
                "target_speed": self.target_speed,
                "comfort_acceleration": 2.0,
                "time_weight": 0.5
            }

        if self.compliance_params is None:
            self.compliance_params = {
                "speed_limit": self.speed_limit,
                "following_distance": 3.0,
                "yield_distance": self.yield_distance
            }


class PreciseEvaluator:
    """
    Evaluator for game quality metrics including safety, efficiency,
    compliance, and opponent rationality - using evaluation4.py definitions
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize evaluator

        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()

    def evaluate_scenario(self,
                         scenario: InteractionScenario,
                         llm_decision: LLMDecision,
                         baseline_decision: Optional[float] = None) -> EvaluationResult:
        """
        Evaluate a complete scenario with LLM decision

        Args:
            scenario: Vehicle interaction scenario
            llm_decision: Decision made by LLM
            baseline_decision: Original/baseline acceleration for comparison

        Returns:
            Complete evaluation result
        """
        # Calculate individual metrics using evaluation4.py definitions
        safety_score = self._evaluate_safety(scenario, llm_decision)
        efficiency_score = self._evaluate_efficiency(scenario, llm_decision)
        compliance_score = self._evaluate_compliance(scenario, llm_decision)
        rationality_score = self._evaluate_rationality(scenario, llm_decision)

        # Calculate overall weighted score using evaluation4.py weights
        overall_score = (
            self.config.safety_weight * safety_score +
            self.config.efficiency_weight * efficiency_score +
            self.config.compliance_weight * compliance_score +
            self.config.rationality_weight * rationality_score
        )

        # Create metrics object
        metrics = GameMetrics(
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            compliance_score=compliance_score,
            rationality_score=rationality_score,
            overall_score=overall_score
        )

        # Calculate baseline comparison if available
        baseline_comparison = None
        if baseline_decision is not None:
            baseline_comparison = self._compare_with_baseline(
                scenario, baseline_decision, llm_decision.acceleration_1
            )

        return EvaluationResult(
            scenario_id=scenario.scenario_id,
            original_decision=baseline_decision,
            llm_decision=llm_decision,
            metrics=metrics,
            baseline_comparison=baseline_comparison
        )

    def _evaluate_safety(self, scenario: InteractionScenario, decision: LLMDecision) -> float:
        """
        Safety evaluation using evaluation4.py definition
        Ultra-streamlined safety evaluation
        """
        v1, v2 = scenario.vehicle_1, scenario.vehicle_2
        acc = decision.acceleration_1
        score = 100.0

        # Acceleration limits
        if abs(acc) > self.config.max_acceleration:
            score -= 40 * (abs(acc) - self.config.max_acceleration) / self.config.max_acceleration

        # Collision risk (perpendicular intersection)
        t1 = v1.distance / max(v1.velocity, 0.1)
        t2 = v2.distance / max(v2.velocity, 0.1)

        # Account for acceleration
        if acc != 0:
            discriminant = v1.velocity**2 + 2 * acc * v1.distance
            if discriminant > 0:
                t1 = (-v1.velocity + math.sqrt(discriminant)) / max(acc, 0.001)

        # Time overlap in intersection
        overlap1 = max(0, min(t1 + self.config.intersection_width/max(v1.velocity, 1),
                              t2 + self.config.intersection_width/max(v2.velocity, 1)) -
                         max(t1, t2))

        collision_risk = min(overlap1 / self.config.reaction_time, 1.0)

        if collision_risk > 0.8:
            return 0.0
        score -= 45 * collision_risk

        # Speed safety
        future_speed = v1.velocity + acc * self.config.reaction_time
        if abs(future_speed) > self.config.max_safe_speed:
            score -= 25 * (abs(future_speed) - self.config.max_safe_speed) / self.config.max_safe_speed

        return max(0, min(100, score))

    def _evaluate_efficiency(self, scenario: InteractionScenario, decision: LLMDecision) -> float:
        """
        Efficiency evaluation using evaluation4.py definition
        Ultra-streamlined efficiency evaluation
        """
        v1, v2 = scenario.vehicle_1, scenario.vehicle_2
        acc = decision.acceleration_1
        score = 100.0

        # Speed efficiency
        speed_eff = 1 - abs(v1.velocity - self.config.target_speed) / self.config.target_speed
        score *= max(0.3, speed_eff)

        # Acceleration comfort
        comfort_penalty = abs(acc) / 2.0
        if comfort_penalty > 1:
            score /= comfort_penalty

        # Traversal time
        t_to_intersection = v1.distance / max(v1.velocity + acc/2, 0.1)
        crossing_time = self.config.intersection_width / max(v1.velocity + acc * t_to_intersection, 1.0)
        total_time = t_to_intersection + crossing_time
        optimal_time = (v1.distance + self.config.intersection_width) / self.config.target_speed

        score *= min(1, optimal_time / max(total_time, 0.1))

        # Flow maintenance
        if v1.distance < v2.distance and acc < -1:  # Unnecessary yielding
            score *= 0.8
        elif v1.distance > v2.distance and acc > 0.5:  # Too aggressive
            score *= 0.6

        return max(0, min(100, score))

    def _evaluate_compliance(self, scenario: InteractionScenario, decision: LLMDecision) -> float:
        """
        Compliance evaluation using evaluation4.py definition
        Ultra-streamlined compliance evaluation
        """
        v1, v2 = scenario.vehicle_1, scenario.vehicle_2
        acc = decision.acceleration_1
        score = 100.0

        # Speed limit
        if v1.velocity > self.config.speed_limit:
            score -= 40 * (v1.velocity - self.config.speed_limit) / self.config.speed_limit

        # Right-of-way (first-come-first-served)
        v1_time = v1.distance / max(v1.velocity, 0.1)
        v2_time = v2.distance / max(v2.velocity, 0.1)

        if v1_time > v2_time and v1.distance < self.config.yield_distance and acc >= 0:
            score -= 50  # Not yielding when should
        elif v1_time <= v2_time and acc < -2:
            score -= 20  # Unnecessary yielding

        # Approach behavior
        if v1.distance < 8:
            if v1.velocity > self.config.speed_limit * 0.7:
                score -= 20 * (v1.velocity - self.config.speed_limit * 0.7) / (self.config.speed_limit * 0.7)
            if acc > 1 and v1.distance < 5:
                score -= 15

        return max(0, min(100, score))

    def _evaluate_rationality(self, scenario: InteractionScenario, decision: LLMDecision) -> float:
        """
        Rationality evaluation using evaluation4.py definition
        Ultra-streamlined rationality evaluation
        """
        v1, v2 = scenario.vehicle_1, scenario.vehicle_2
        acc = decision.acceleration_1
        score = 100.0

        # Opponent consideration
        if v2.velocity > v1.velocity and acc > 0.5 and v1.distance > v2.distance:
            score -= 30  # Not accounting for faster opponent
        elif v2.velocity < v1.velocity and acc < -1 and v1.distance < v2.distance:
            score -= 20  # Overly cautious

        # Distance-based behavior
        if v2.distance < 5 and acc > 1:
            score -= 25  # Too aggressive
        elif v2.distance > 20 and acc < -1:
            score -= 15  # Overly cautious

        # Cooperative behavior
        if v1.distance > v2.distance and v1.velocity > v2.velocity and acc < 0:
            score += 10  # Appropriate yielding
        elif abs(v1.distance - v2.distance) < 3 and acc > 2:
            score -= 25  # Too aggressive in close situation

        # Timing
        if v1.distance > 30 and abs(acc) > 1:
            score -= 10  # Early strong decision
        elif v1.distance < 5 and abs(acc) < 0.1:
            score -= 20  # No decision when needed

        return max(0, min(100, score))

    def _evaluate_decision_only(self, scenario: InteractionScenario, decision: LLMDecision) -> 'EvaluationResult':
        """
        Evaluate a decision without baseline comparison (to avoid recursion)

        Args:
            scenario: Vehicle interaction scenario
            decision: Decision to evaluate

        Returns:
            Evaluation result without baseline comparison
        """
        # Calculate individual metrics using evaluation4.py definitions
        safety_score = self._evaluate_safety(scenario, decision)
        efficiency_score = self._evaluate_efficiency(scenario, decision)
        compliance_score = self._evaluate_compliance(scenario, decision)
        rationality_score = self._evaluate_rationality(scenario, decision)

        # Calculate overall weighted score
        overall_score = (
            self.config.safety_weight * safety_score +
            self.config.efficiency_weight * efficiency_score +
            self.config.compliance_weight * compliance_score +
            self.config.rationality_weight * rationality_score
        )

        # Create metrics object
        metrics = GameMetrics(
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            compliance_score=compliance_score,
            rationality_score=rationality_score,
            overall_score=overall_score
        )

        return EvaluationResult(
            scenario_id=scenario.scenario_id,
            original_decision=None,
            llm_decision=decision,
            metrics=metrics,
            baseline_comparison=None
        )

    def _compare_with_baseline(self, scenario: InteractionScenario, baseline: float, llm: float) -> Dict[str, float]:
        """
        Compare LLM decision with baseline decision

        Args:
            scenario: Vehicle interaction scenario
            baseline: Baseline acceleration
            llm: LLM acceleration

        Returns:
            Comparison metrics
        """
        # Create mock decisions for comparison
        baseline_decision_obj = LLMDecision(acceleration_1=baseline, reasoning="baseline", confidence=1.0)
        llm_decision_obj = LLMDecision(acceleration_1=llm, reasoning="llm", confidence=0.8)

        # Evaluate both (without baseline comparison to avoid recursion)
        baseline_eval = self._evaluate_decision_only(scenario, baseline_decision_obj)
        llm_eval = self._evaluate_decision_only(scenario, llm_decision_obj)

        return {
            "baseline_overall": baseline_eval.metrics.overall_score,
            "llm_overall": llm_eval.metrics.overall_score,
            "improvement": llm_eval.metrics.overall_score - baseline_eval.metrics.overall_score,
            "acceleration_difference": llm - baseline,
            "baseline_safety": baseline_eval.metrics.safety_score,
            "llm_safety": llm_eval.metrics.safety_score,
            "safety_improvement": llm_eval.metrics.safety_score - baseline_eval.metrics.safety_score,
            "baseline_efficiency": baseline_eval.metrics.efficiency_score,
            "llm_efficiency": llm_eval.metrics.efficiency_score,
            "efficiency_improvement": llm_eval.metrics.efficiency_score - baseline_eval.metrics.efficiency_score,
        }

    # Method to evaluate fuzzy decisions adapted for evaluation4.py metrics
    def evaluate_fuzzy_decision(self, scenario: InteractionScenario, fuzzy_decision: FuzzyDecision) -> GameMetrics:
        """
        Evaluate fuzzy decision using adapted evaluation4.py metrics

        Args:
            scenario: Vehicle interaction scenario
            fuzzy_decision: Fuzzy decision object

        Returns:
            GameMetrics with adapted scores
        """
        # Convert fuzzy decision to LLM decision for evaluation
        # Use confidence to modulate acceleration (simplified approach)
        priority_factor = 1.0 if fuzzy_decision.priority_vehicle == scenario.vehicle_1.vehicle_id else -1.0

        # Estimate acceleration based on fuzzy decision
        # High confidence + priority = accelerate, Low confidence + no priority = decelerate
        if fuzzy_decision.priority_vehicle == scenario.vehicle_1.vehicle_id:
            estimated_acc = fuzzy_decision.confidence * 2.0  # Accelerate if we have priority
        else:
            estimated_acc = -fuzzy_decision.confidence * 2.0  # Decelerate if we should yield

        llm_decision = LLMDecision(
            acceleration_1=estimated_acc,
            reasoning=fuzzy_decision.textual_reasoning,
            confidence=fuzzy_decision.confidence,
            strategy_type=fuzzy_decision.scenario_type
        )

        # Use the evaluation4.py-based evaluation methods
        safety_score = self._evaluate_safety(scenario, llm_decision)
        efficiency_score = self._evaluate_efficiency(scenario, llm_decision)
        compliance_score = self._evaluate_compliance(scenario, llm_decision)
        rationality_score = self._evaluate_rationality(scenario, llm_decision)

        # Adjust scores based on fuzzy reasoning weights
        if fuzzy_decision.fuzzy_reasoning:
            # Bonus for good fuzzy reasoning
            reasoning_bonus = 0.0

            # Check safety priority in fuzzy reasoning
            if "safety_priority" in fuzzy_decision.fuzzy_reasoning:
                reasoning_bonus += fuzzy_decision.fuzzy_reasoning["safety_priority"] * 5.0

            # Check position advantage
            if "position_advantage" in fuzzy_decision.fuzzy_reasoning:
                reasoning_bonus += fuzzy_decision.fuzzy_reasoning["position_advantage"] * 3.0

            # Apply reasoning bonus to overall score
            safety_score = min(100.0, safety_score + reasoning_bonus * 0.5)
            rationality_score = min(100.0, rationality_score + reasoning_bonus * 0.3)

        # Calculate overall weighted score
        overall_score = (
            self.config.safety_weight * safety_score +
            self.config.efficiency_weight * efficiency_score +
            self.config.compliance_weight * compliance_score +
            self.config.rationality_weight * rationality_score
        )

        return GameMetrics(
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            compliance_score=compliance_score,
            rationality_score=rationality_score,
            overall_score=overall_score
        )


class OpponentRationalityRegulator:
    """
    Regulator for calibrating LLM decisions against opponent rationality
    Adapted to work with evaluation4.py-based PreciseEvaluator
    """

    def __init__(self):
        """Initialize rationality regulator"""
        self.rationality_models = {
            "cooperative": self._cooperative_model,
            "competitive": self._competitive_model,
            "neutral": self._neutral_model
        }

    def calibrate_decision(self,
                          scenario: InteractionScenario,
                          llm_decision: LLMDecision,
                          opponent_type: str = "neutral") -> LLMDecision:
        """
        Calibrate LLM decision based on opponent rationality model

        Args:
            scenario: Vehicle interaction scenario
            llm_decision: Original LLM decision
            opponent_type: Type of opponent rationality model

        Returns:
            Calibrated LLM decision
        """
        if opponent_type not in self.rationality_models:
            opponent_type = "neutral"

        model = self.rationality_models[opponent_type]
        calibrated_acceleration = model(scenario, llm_decision.acceleration_1)

        return LLMDecision(
            acceleration_1=calibrated_acceleration,
            reasoning=f"{llm_decision.reasoning}\n\n[Calibrated using {opponent_type} opponent model]",
            confidence=llm_decision.confidence,
            strategy_type=llm_decision.strategy_type
        )

    def _cooperative_model(self, scenario: InteractionScenario, acceleration: float) -> float:
        """
        Cooperative opponent model - assumes opponent will yield appropriately
        """
        v1, v2 = scenario.vehicle_1, scenario.vehicle_2
        rel_distance = v1.distance - v2.distance

        # If we have priority and opponent is cooperative, can be more confident
        if rel_distance > 0 and v1.velocity >= v2.velocity:
            return min(acceleration + 0.2, 2.0)  # Slightly more aggressive

        return acceleration

    def _competitive_model(self, scenario: InteractionScenario, acceleration: float) -> float:
        """
        Competitive opponent model - assumes opponent will be aggressive
        """
        v1, v2 = scenario.vehicle_1, scenario.vehicle_2
        rel_distance = v1.distance - v2.distance

        # Be more cautious with competitive opponent
        if abs(rel_distance) < 5.0:
            return max(acceleration - 0.3, -2.0)  # More defensive

        return acceleration

    def _neutral_model(self, scenario: InteractionScenario, acceleration: float) -> float:
        """
        Neutral opponent model - assumes predictable, rule-following behavior
        """
        # No calibration needed for neutral model
        return acceleration

    def estimate_opponent_type(self, scenarios: List[InteractionScenario]) -> str:
        """
        Estimate opponent type from historical scenarios

        Args:
            scenarios: List of historical interaction scenarios

        Returns:
            Estimated opponent type
        """
        # Simple heuristic based on acceleration patterns
        # Could be enhanced with machine learning
        total_accelerations = []
        for scenario in scenarios:
            if scenario.vehicle_2.acceleration is not None:
                total_accelerations.append(scenario.vehicle_2.acceleration)

        if not total_accelerations:
            return "neutral"

        avg_acc = np.mean(total_accelerations)
        std_acc = np.std(total_accelerations)

        if avg_acc > 0.5 and std_acc < 1.0:
            return "cooperative"
        elif avg_acc < -0.5 or std_acc > 2.0:
            return "competitive"
        else:
            return "neutral"


@dataclass
class FuzzyEvaluationMetrics:
    """Metrics for fuzzy priority decision evaluation"""
    accuracy: float
    confidence_mean: float
    confidence_std: float
    risk_distribution: Dict[str, int]
    scenario_type_distribution: Dict[str, int]
    fuzzy_weight_analysis: Dict[str, float]
    response_time_mean: float
    total_scenarios: int
    correct_predictions: int


@dataclass
class FuzzyEvaluationResult:
    """Complete fuzzy evaluation result"""
    scenario_id: str
    fuzzy_decision: FuzzyDecision
    ground_truth: Optional[str]
    is_correct: bool
    confidence_score: float
    risk_level: str
    scenario_type: str
    fuzzy_weights: Dict[str, float]
    response_time: float
    metrics: Optional[GameMetrics] = None  # Added to include evaluation4.py-based metrics


class FuzzyEvaluator:
    """
    Evaluator for fuzzy priority decisions with accuracy metrics
    Adapted to work with evaluation4.py-based PreciseEvaluator
    """

    def __init__(self, precise_evaluator: Optional[PreciseEvaluator] = None):
        """
        Initialize fuzzy evaluator

        Args:
            precise_evaluator: PreciseEvaluator instance for fuzzy decision scoring
        """
        self.evaluation_history = []
        self.precise_evaluator = precise_evaluator or PreciseEvaluator()

    def evaluate_fuzzy_decision(self,
                               scenario: InteractionScenario,
                               fuzzy_decision: FuzzyDecision,
                               ground_truth_priority: Optional[str] = None,
                               response_time: float = 0.0) -> FuzzyEvaluationResult:
        """
        Evaluate a single fuzzy priority decision

        Args:
            scenario: Vehicle interaction scenario
            fuzzy_decision: LLM fuzzy decision
            ground_truth_priority: Ground truth priority vehicle ID
            response_time: Time taken for LLM to generate decision

        Returns:
            FuzzyEvaluationResult with detailed metrics
        """
        # Determine correctness if ground truth is available
        is_correct = False
        if ground_truth_priority is not None:
            is_correct = (fuzzy_decision.priority_vehicle == ground_truth_priority)

        # Get evaluation4.py-based metrics for the fuzzy decision
        metrics = None
        if self.precise_evaluator:
            metrics = self.precise_evaluator.evaluate_fuzzy_decision(scenario, fuzzy_decision)

        return FuzzyEvaluationResult(
            scenario_id=scenario.scenario_id,
            fuzzy_decision=fuzzy_decision,
            ground_truth=ground_truth_priority,
            is_correct=is_correct,
            confidence_score=fuzzy_decision.confidence,
            risk_level=fuzzy_decision.risk_level,
            scenario_type=fuzzy_decision.scenario_type,
            fuzzy_weights=fuzzy_decision.fuzzy_reasoning,
            response_time=response_time,
            metrics=metrics
        )

    def evaluate_batch(self,
                      evaluation_results: List[FuzzyEvaluationResult]) -> FuzzyEvaluationMetrics:
        """
        Evaluate a batch of fuzzy decisions with comprehensive metrics

        Args:
            evaluation_results: List of individual evaluation results

        Returns:
            FuzzyEvaluationMetrics with aggregated statistics
        """
        if not evaluation_results:
            return FuzzyEvaluationMetrics(
                accuracy=0.0, confidence_mean=0.0, confidence_std=0.0,
                risk_distribution={}, scenario_type_distribution={},
                fuzzy_weight_analysis={}, response_time_mean=0.0,
                total_scenarios=0, correct_predictions=0
            )

        # Calculate accuracy
        correct_count = sum(1 for result in evaluation_results if result.is_correct)
        total_with_ground_truth = sum(1 for result in evaluation_results if result.ground_truth is not None)
        accuracy = correct_count / total_with_ground_truth if total_with_ground_truth > 0 else 0.0

        # Calculate confidence statistics
        confidence_scores = [result.confidence_score for result in evaluation_results]
        confidence_mean = np.mean(confidence_scores) if confidence_scores else 0.0
        confidence_std = np.std(confidence_scores) if confidence_scores else 0.0

        # Calculate distributions
        risk_distribution = {}
        scenario_type_distribution = {}
        for result in evaluation_results:
            risk_distribution[result.risk_level] = risk_distribution.get(result.risk_level, 0) + 1
            scenario_type_distribution[result.scenario_type] = scenario_type_distribution.get(result.scenario_type, 0) + 1

        # Analyze fuzzy weights
        fuzzy_weight_analysis = self._analyze_fuzzy_weights(evaluation_results)

        # Calculate response time statistics
        response_times = [result.response_time for result in evaluation_results if result.response_time > 0]
        response_time_mean = np.mean(response_times) if response_times else 0.0

        return FuzzyEvaluationMetrics(
            accuracy=accuracy,
            confidence_mean=confidence_mean,
            confidence_std=confidence_std,
            risk_distribution=risk_distribution,
            scenario_type_distribution=scenario_type_distribution,
            fuzzy_weight_analysis=fuzzy_weight_analysis,
            response_time_mean=response_time_mean,
            total_scenarios=len(evaluation_results),
            correct_predictions=correct_count
        )

    def _analyze_fuzzy_weights(self, evaluation_results: List[FuzzyEvaluationResult]) -> Dict[str, float]:
        """
        Analyze fuzzy weight patterns across decisions

        Args:
            evaluation_results: List of evaluation results

        Returns:
            Dictionary with weight analysis statistics
        """
        weight_categories = ["distance_risk", "velocity_risk", "ttc_risk",
                           "safety_priority", "position_advantage", "traffic_rules"]

        weight_analysis = {}
        for category in weight_categories:
            weights = []
            for result in evaluation_results:
                if category in result.fuzzy_weights:
                    weights.append(result.fuzzy_weights[category])

            if weights:
                weight_analysis[f"{category}_mean"] = float(np.mean(weights))
                weight_analysis[f"{category}_std"] = float(np.std(weights))
                weight_analysis[f"{category}_max"] = float(np.max(weights))
                weight_analysis[f"{category}_min"] = float(np.min(weights))
            else:
                weight_analysis[f"{category}_mean"] = 0.0
                weight_analysis[f"{category}_std"] = 0.0
                weight_analysis[f"{category}_max"] = 0.0
                weight_analysis[f"{category}_min"] = 0.0

        return weight_analysis

    def compare_with_baseline(self,
                            fuzzy_results: List[FuzzyEvaluationResult],
                            baseline_predictions: List[str]) -> Dict[str, Any]:
        """
        Compare fuzzy decisions with baseline predictions

        Args:
            fuzzy_results: Fuzzy evaluation results
            baseline_predictions: Baseline priority predictions

        Returns:
            Comparison metrics
        """
        if len(fuzzy_results) != len(baseline_predictions):
            raise ValueError("Results and baseline predictions must have same length")

        fuzzy_correct = sum(1 for result in fuzzy_results if result.is_correct)
        baseline_correct = 0

        for result, baseline_pred in zip(fuzzy_results, baseline_predictions):
            if result.ground_truth is not None and baseline_pred == result.ground_truth:
                baseline_correct += 1

        total_with_ground_truth = sum(1 for result in fuzzy_results if result.ground_truth is not None)

        if total_with_ground_truth == 0:
            return {"error": "No ground truth available for comparison"}

        fuzzy_accuracy = fuzzy_correct / total_with_ground_truth
        baseline_accuracy = baseline_correct / total_with_ground_truth
        improvement = fuzzy_accuracy - baseline_accuracy

        # Add evaluation4.py-based metrics comparison if available
        metrics_comparison = {}
        if all(result.metrics for result in fuzzy_results):
            fuzzy_overall_scores = [result.metrics.overall_score for result in fuzzy_results if result.metrics]
            baseline_overall_scores = []  # Would need baseline evaluation for comparison

            if fuzzy_overall_scores:
                metrics_comparison = {
                    "fuzzy_mean_overall_score": np.mean(fuzzy_overall_scores),
                    "fuzzy_mean_safety_score": np.mean([result.metrics.safety_score for result in fuzzy_results if result.metrics]),
                    "fuzzy_mean_efficiency_score": np.mean([result.metrics.efficiency_score for result in fuzzy_results if result.metrics]),
                    "fuzzy_mean_compliance_score": np.mean([result.metrics.compliance_score for result in fuzzy_results if result.metrics]),
                    "fuzzy_mean_rationality_score": np.mean([result.metrics.rationality_score for result in fuzzy_results if result.metrics]),
                }

        return {
            "fuzzy_accuracy": fuzzy_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_improvement": improvement,
            "fuzzy_correct": fuzzy_correct,
            "baseline_correct": baseline_correct,
            "total_comparisons": total_with_ground_truth,
            **metrics_comparison
        }

    def generate_evaluation_report(self, metrics: FuzzyEvaluationMetrics) -> str:
        """
        Generate comprehensive evaluation report

        Args:
            metrics: Fuzzy evaluation metrics

        Returns:
            Formatted report string
        """
        report = f"""
FUZZY PRIORITY DECISION EVALUATION REPORT
==========================================

OVERALL PERFORMANCE:
- Total Scenarios: {metrics.total_scenarios}
- Accuracy: {metrics.accuracy:.3f} ({metrics.correct_predictions}/{metrics.total_scenarios})
- Average Confidence: {metrics.confidence_mean:.3f} ± {metrics.confidence_std:.3f}
- Average Response Time: {metrics.response_time_mean:.3f}s

RISK LEVEL DISTRIBUTION:
"""
        for risk_level, count in metrics.risk_distribution.items():
            percentage = (count / metrics.total_scenarios) * 100 if metrics.total_scenarios > 0 else 0
            report += f"- {risk_level}: {count} ({percentage:.1f}%)\n"

        report += "\nSCENARIO TYPE DISTRIBUTION:\n"
        for scenario_type, count in metrics.scenario_type_distribution.items():
            percentage = (count / metrics.total_scenarios) * 100 if metrics.total_scenarios > 0 else 0
            report += f"- {scenario_type}: {count} ({percentage:.1f}%)\n"

        report += "\nFUZZY WEIGHT ANALYSIS:\n"
        for key, value in metrics.fuzzy_weight_analysis.items():
            if key.endswith("_mean"):
                category = key.replace("_mean", "")
                mean_val = value
                std_val = metrics.fuzzy_weight_analysis.get(f"{category}_std", 0)
                report += f"- {category}: {mean_val:.3f} ± {std_val:.3f}\n"

        return report.strip()