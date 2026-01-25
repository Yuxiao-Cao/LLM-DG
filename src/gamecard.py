"""
GameCard implementation with Chain-of-Thought prompts for LLM decision making
"""

from typing import Dict, Any, List, Optional, Tuple
import json
import math
from .data_models import InteractionScenario, LLMDecision


class GameCard:
    """
    GameCard class that implements Chain-of-Thought (CoT) guided prompts
    for translating vehicle interaction scenarios into LLM decision-making tasks
    """

    def __init__(self, prompt_format: str, cot_type: str = "cot"):
        """
        Initialize GameCard

        Args:
            prompt_format: Input format for LLM ("text", "json", or "text+json")
            cot_type: Chain-of-Thought type ("cot" or "nocot")
        """
        self.prompt_format = prompt_format
        self.cot_type = cot_type
        self.safety_thresholds = {
            "min_distance": 2.0,  # meters
            "max_speed": 20.0,    # m/s
            "max_acceleration": 3.0,  # m/s²
            "min_acceleration": -3.0  # m/s²
        }

        # Fuzzy logic parameters
        self.fuzzy_params = {
            "critical_distance": 3.0,  # meters
            "high_risk_ttc": 2.0,      # seconds
            "medium_risk_ttc": 5.0,    # seconds
            "speed_difference_threshold": 5.0,  # m/s
            "acceleration_threshold": 1.0  # m/s²
        }

    def create_precise_prompt(self, scenario: InteractionScenario) -> Tuple[str, Dict[str, Any]]:
        """
        Create CoT-guided prompt for the given scenario

        Args:
            scenario: Vehicle interaction scenario

        Returns:
            Tuple of (prompt_text, prompt_json)
        """
        # Build the Chain-of-Thought prompt
        cot_prompt = self._build_precise_cot_prompt(scenario)

        # Create JSON data structure
        v1 = scenario.vehicle_1
        v2 = scenario.vehicle_2
        json_data = {
            "scenario_id": scenario.scenario_id,
            "frame_id": scenario.frame_id,
            "scenario_type": scenario.scenario_type,
            "vehicle_1": {
                "id": v1.vehicle_id,
                "distance": round(v1.distance, 3),
                "velocity": round(v1.velocity, 3),
                "acceleration": round(v1.acceleration, 3) if v1.acceleration is not None else None
            },
            "vehicle_2": {
                "id": v2.vehicle_id,
                "distance": round(v2.distance, 3),
                "velocity": round(v2.velocity, 3),
                "acceleration": round(v2.acceleration, 3) if v2.acceleration is not None else None
            },
            "safety_constraints": self.safety_thresholds
        }

        # Format based on requested format
        if self.prompt_format == "text":
            full_prompt = cot_prompt
        elif self.prompt_format == "json":
            # Create strictly JSON-formatted prompt
            json_prompt = {
                "system_prompt": "You are an expert autonomous driving decision-making AI with deep knowledge of game theory, traffic safety, and vehicle dynamics. Always provide detailed reasoning for your decisions.",
                "scenario_data": json_data,
                "task_description": "Analyze the provided autonomous driving scenario and make an acceleration decision for Vehicle 1.",
                "analysis_requirements": {
                    "traffic_rules": "Identify relevant traffic regulations and right-of-way rules",
                    "game_theory": "Apply game-theoretic analysis for opponent modeling and strategic decision-making",
                    "safety_assessment": "Evaluate collision risk and safety implications",
                    "efficiency_optimization": "Balance safety with traffic flow efficiency"
                },
                "output_format": {
                    "acceleration_1": "float (range: -3.0 to 3.0 m/s²)",
                    "reasoning": "string (detailed game-theoretic analysis)",
                    "strategy_type": "string (CAUTIOUS_DEFENSIVE|BALANCED_COOPERATIVE|ASSERTIVE_EFFICIENT|COMPETITIVE_PRIORITY|EMERGENCY_BRAKE)",
                    "opponent_prediction": "string (predicted Vehicle 2 behavior)",
                    "confidence": "float (0.0 to 1.0)",
                    "risk_assessment": "string (low|medium|high|critical)",
                    "game_theory_considerations": "string (nash equilibrium and payoff analysis)"
                }
            }
            full_prompt = json.dumps(json_prompt, indent=2)
        else:  # text+json
            full_prompt = f"TEXTUAL ANALYSIS:\n{cot_prompt}\n\nSTRUCTURED DATA:\n{json.dumps(json_data, indent=2)}"

        return full_prompt, json_data

    def create_fuzzy_prompt(self, scenario: InteractionScenario) -> Tuple[str, Dict[str, Any]]:
        """
        Create comprehensive fuzzy logic-enhanced prompt for priority determination

        Args:
            scenario: Vehicle interaction scenario

        Returns:
            Tuple of (fuzzy_prompt_text, fuzzy_data)
        """
        # Build comprehensive fuzzy CoT prompt
        fuzzy_prompt = self._build_fuzzy_cot_prompt(scenario)

        # Extract fuzzy features for comprehensive data structure
        features = self._extract_safety_features(scenario)
        fuzzy_memberships = self._calculate_fuzzy_memberships(features)

        # Create comprehensive fuzzy data structure
        v1 = scenario.vehicle_1
        v2 = scenario.vehicle_2

        fuzzy_data = {
            "scenario_id": scenario.scenario_id,
            "frame_id": scenario.frame_id,
            "scenario_type": scenario.scenario_type,
            "vehicle_1": {
                "id": v1.vehicle_id,
                "distance": round(v1.distance, 3),
                "velocity": round(v1.velocity, 3),
                "acceleration": round(v1.acceleration, 3) if v1.acceleration is not None else None
            },
            "vehicle_2": {
                "id": v2.vehicle_id,
                "distance": round(v2.distance, 3),
                "velocity": round(v2.velocity, 3),
                "acceleration": round(v2.acceleration, 3) if v2.acceleration is not None else None
            },
            "fuzzy_features": {
                "distance_diff": round(features["distance_diff"], 3),
                "velocity_diff": round(features["velocity_diff"], 3),
                "time_to_collision": round(features["time_to_collision"], 3) if features["time_to_collision"] is not None else None,
                "position_advantage": round(features["position_advantage"], 3),
                "velocity_advantage": round(features["velocity_advantage"], 3)
            },
            "fuzzy_memberships": {
                "distance_risk": {
                    "very_low": round(fuzzy_memberships["distance_risk"].very_low, 3),
                    "low": round(fuzzy_memberships["distance_risk"].low, 3),
                    "medium": round(fuzzy_memberships["distance_risk"].medium, 3),
                    "high": round(fuzzy_memberships["distance_risk"].high, 3),
                    "very_high": round(fuzzy_memberships["distance_risk"].very_high, 3)
                },
                "velocity_risk": {
                    "very_low": round(fuzzy_memberships["velocity_risk"].very_low, 3),
                    "low": round(fuzzy_memberships["velocity_risk"].low, 3),
                    "medium": round(fuzzy_memberships["velocity_risk"].medium, 3),
                    "high": round(fuzzy_memberships["velocity_risk"].high, 3),
                    "very_high": round(fuzzy_memberships["velocity_risk"].very_high, 3)
                },
                "ttc_risk": {
                    "very_low": round(fuzzy_memberships["ttc_risk"].very_low, 3),
                    "low": round(fuzzy_memberships["ttc_risk"].low, 3),
                    "medium": round(fuzzy_memberships["ttc_risk"].medium, 3),
                    "high": round(fuzzy_memberships["ttc_risk"].high, 3),
                    "very_high": round(fuzzy_memberships["ttc_risk"].very_high, 3)
                }
            },
            "fuzzy_parameters": self.fuzzy_params
        }

        # Format based on requested format
        if self.prompt_format == "text":
            full_fuzzy_prompt = fuzzy_prompt
        elif self.prompt_format == "json":
            # Create strictly JSON-formatted fuzzy prompt
            json_fuzzy_prompt = {
                "system_prompt": "You are an expert autonomous driving AI with deep knowledge of fuzzy logic, game theory, and vehicle priority assessment. Always provide detailed reasoning for your priority decisions.",
                "scenario_data": fuzzy_data,
                "task_description": "Analyze the provided autonomous driving scenario and determine which vehicle should have priority using fuzzy logic and game-theoretic reasoning.",
                "analysis_requirements": {
                    "fuzzy_traffic_rules": "Apply traffic rules with degree of applicability (0.0-1.0)",
                    "fuzzy_game_theory": "Use fuzzy opponent modeling and probabilistic payoff analysis",
                    "fuzzy_safety_assessment": "Evaluate collision risk with fuzzy membership functions",
                    "fuzzy_priority_weights": "Assign fuzzy weights to different decision factors"
                },
                "output_format": {
                    "priority_vehicle": f"string ({v1.vehicle_id} or {v2.vehicle_id} or shared)",
                    "confidence": "float (0.0 to 1.0)",
                    "reasoning": "string (detailed fuzzy game-theoretic analysis)",
                    "fuzzy_weights": {
                        "safety_priority": "float (0.0 to 1.0)",
                        "efficiency_priority": "float (0.0 to 1.0)",
                        "right_of_way_priority": "float (0.0 to 1.0)",
                        "cooperation_priority": "float (0.0 to 1.0)",
                        "risk_assessment_priority": "float (0.0 to 1.0)",
                        "contextual_priority": "float (0.0 to 1.0)"
                    },
                    "risk_level": "string (low|medium|high|critical)",
                    "scenario_type": f"string ({scenario.scenario_type or 'intersection_crossing'})",
                    "game_theory_considerations": "string (fuzzy nash equilibrium and cooperative analysis)",
                    "opponent_model": "string (fuzzy prediction of opponent priority behavior)"
                }
            }
            full_fuzzy_prompt = json.dumps(json_fuzzy_prompt, indent=2)
        else:  # text+json
            full_fuzzy_prompt = f"FUZZY GAME-THEORETIC ANALYSIS:\n{fuzzy_prompt}\n\nSTRUCTURED FUZZY DATA:\n{json.dumps(fuzzy_data, indent=2)}"

        return full_fuzzy_prompt, fuzzy_data

    def _build_precise_cot_prompt(self, scenario: InteractionScenario) -> str:
        """
        Build comprehensive Chain-of-Thought prompt for game-theoretic decision making

        Args:
            scenario: Vehicle interaction scenario with all necessary data

        Returns:
            Formatted CoT prompt string
        """
        v1 = scenario.vehicle_1
        v2 = scenario.vehicle_2

        # Calculate essential metrics
        time_to_collision = self._calculate_ttc(v1.distance, v1.velocity, v2.distance, v2.velocity)
        ttc_str = f"{time_to_collision:.2f}s" if time_to_collision is not None else "N/A"

        # Calculate relative metrics for spatial analysis
        relative_distance = v1.distance - v2.distance
        relative_velocity = v1.velocity - v2.velocity

        # Check if we should use nocot prompt
        if self.cot_type == "nocot":
            prompt = f"""
AUTONOMOUS DRIVING GAME-THEORETIC DECISION MAKING

SCENARIO DESCRIPTION:
Scenario Type: {scenario.scenario_type or 'intersection_crossing'}
Context: High-interaction driving scenario requiring strategic decision-making considering opponent behavior

SPATIAL RELATIONSHIP DESCRIPTION:
Vehicle 1 ({v1.vehicle_id}):
- Current Distance to Interaction Point: {v1.distance:.2f}m
- Current Velocity: {v1.velocity:.2f}m/s
- Current Acceleration: {v1.acceleration:.2f} m/s² (if applicable)

Vehicle 2 ({v2.vehicle_id}):
- Current Distance to Interaction Point: {v2.distance:.2f} m
- Current Velocity: {v2.velocity:.2f} m/s
- Current Acceleration: {v2.acceleration:.2f} m/s² (if applicable)

Spatial Analysis:
- Relative Distance (D1 - D2): {relative_distance:.2f} m
- Relative Velocity (V1 - V2): {relative_velocity:.2f} m/s
- Time to Collision: {ttc_str}
- Interaction Urgency: {"High" if time_to_collision and time_to_collision < 3.0 else 'Medium' if time_to_collision and time_to_collision < 6.0 else "Low"}

AVAILABLE ACTIONS:
   Acceleration Range: [{self.safety_thresholds['min_acceleration']}, {self.safety_thresholds['max_acceleration']}] m/s²
   - Strong Braking (< -2.0 m/s²): Defensive, safety-first approach
   - Moderate Braking (-2.0 to 0 m/s²): Cautious, yielding behavior
   - Maintenance (0 m/s²): Current speed preservation
   - Mild Acceleration (0 to 2.0 m/s²): Assertive, efficiency-focused
   - Strong Acceleration (> 2.0 m/s²): Competitive, priority-claiming

DECISION OUTPUT FORMAT:
Provide your decision in the following JSON format:
{{
    "acceleration_1": <float_value_within_specified_range>,
    "reasoning": "<detailed_game_theoretic_analysis_explaining_strategic_choice>",
    "strategy_type": "<CAUTIOUS_DEFENSIVE|BALANCED_COOPERATIVE|ASSERTIVE_EFFICIENT|COMPETITIVE_PRIORITY|EMERGENCY_BRAKE>",
    "opponent_prediction": "<predicted_opponent_behavior>",
    "confidence": <0.0_to_1.0>,
    "risk_assessment": "<low|medium|high|critical>",
    "game_theory_considerations": "<nash_equilibrium_and_payoff_analysis>"
}}

Execute your strategic decision now:
"""
            return prompt.strip()

        # Original CoT prompt (when cot_type == "cot")
        prompt = f"""
AUTONOMOUS DRIVING GAME-THEORETIC DECISION MAKING

SCENARIO DESCRIPTION:
Scenario Type: {scenario.scenario_type or 'intersection_crossing'}
Context: High-interaction driving scenario requiring strategic decision-making considering opponent behavior

SPATIAL RELATIONSHIP DESCRIPTION:
Vehicle 1 ({v1.vehicle_id}):
- Current Distance to Interaction Point: {v1.distance:.2f}m
- Current Velocity: {v1.velocity:.2f}m/s
- Current Acceleration: {v1.acceleration:.2f}m/s² (if applicable)

Vehicle 2 ({v2.vehicle_id}):
- Current Distance to Interaction Point: {v2.distance:.2f}m
- Current Velocity: {v2.velocity:.2f}m/s
- Current Acceleration: {v2.acceleration:.2f}m/s² (if applicable)

Spatial Analysis:
- Relative Distance (D1 - D2): {relative_distance:.2f}m
- Relative Velocity (V1 - V2): {relative_velocity:.2f}m/s
- Time to Collision: {ttc_str}
- Interaction Urgency: {"High" if time_to_collision and time_to_collision < 3.0 else "Medium" if time_to_collision and time_to_collision < 6.0 else "Low"}

REASONING CHAIN:

1. APPLICABLE TRAFFIC RULES:
   - Identify relevant traffic regulations for {scenario.scenario_type or 'intersection_crossing'}
   - Determine right-of-way rules and priority relationships
   - Consider speed limits and safety distance requirements
   - Assess legal obligations and traffic code compliance

2. GAME-THEORETICAL ANALYSIS REQUIREMENTS:
   - Opponent Modeling: Predict Vehicle 2's likely actions and strategies
   - Payoff Matrix: Evaluate outcomes of different acceleration choices
   - Nash Equilibrium: Identify optimal strategy given opponent rationality
   - Utility Maximization: Balance safety, efficiency, and cooperation
   - Dynamic Interaction: Consider multi-step decision horizon
   - Information Asymmetry: Handle incomplete opponent information

3. AVAILABLE ACTIONS:
   Acceleration Range: [{self.safety_thresholds['min_acceleration']}, {self.safety_thresholds['max_acceleration']}] m/s²
   - Strong Braking (< -2.0 m/s²): Defensive, safety-first approach
   - Moderate Braking (-2.0 to 0 m/s²): Cautious, yielding behavior
   - Maintenance (0 m/s²): Current speed preservation
   - Mild Acceleration (0 to 2.0 m/s²): Assertive, efficiency-focused
   - Strong Acceleration (> 2.0 m/s²): Competitive, priority-claiming

4. STRATEGIC CONSIDERATIONS:
   - Safety-Critical Assessment: Immediate collision risk analysis
   - Efficiency Optimization: Travel time
   - Cooperative vs Competitive: Game theory strategy selection
   - Opponent Rationality Assumption: Predict opponent's decision-making
   - Multi-Scenario Planning: Contingency strategies for different opponent actions
   - Reputation and Reciprocity: Long-term interaction considerations

DECISION OUTPUT FORMAT:
Provide your decision in the following JSON format:
{{
    "acceleration_1": <float_value_within_specified_range>,
    "reasoning": "<detailed_game_theoretic_analysis_explaining_strategic_choice>",
    "strategy_type": "<CAUTIOUS_DEFENSIVE|BALANCED_COOPERATIVE|ASSERTIVE_EFFICIENT|COMPETITIVE_PRIORITY|EMERGENCY_BRAKE>",
    "opponent_prediction": "<predicted_opponent_behavior>",
    "confidence": <0.0_to_1.0>,
    "risk_assessment": "<low|medium|high|critical>",
    "game_theory_considerations": "<nash_equilibrium_and_payoff_analysis>"
}}

Execute your strategic decision now:
"""
        return prompt.strip()

    def _build_fuzzy_cot_prompt(self, scenario: InteractionScenario) -> str:
        """
        Build comprehensive fuzzy logic Chain-of-Thought prompt for game-theoretic priority determination

        Args:
            scenario: Vehicle interaction scenario with all necessary data

        Returns:
            Formatted fuzzy CoT prompt string
        """
        v1 = scenario.vehicle_1
        v2 = scenario.vehicle_2

        # Calculate essential metrics
        time_to_collision = self._calculate_ttc(v1.distance, v1.velocity, v2.distance, v2.velocity)
        ttc_str = f"{time_to_collision:.2f}s" if time_to_collision is not None else "N/A"

        # Calculate relative metrics for spatial analysis
        relative_distance = v1.distance - v2.distance
        relative_velocity = v1.velocity - v2.velocity

        # Check if we should use nocot prompt
        if self.cot_type == "nocot":
            prompt = f"""
AUTONOMOUS DRIVING FUZZY GAME-THEORETIC PRIORITY DETERMINATION

SCENARIO DESCRIPTION:
Scenario Type: {scenario.scenario_type or 'intersection_crossing'}
Context: High-interaction driving scenario requiring fuzzy priority assessment with game-theoretic strategic reasoning

SPATIAL RELATIONSHIP DESCRIPTION:
Vehicle 1 ({v1.vehicle_id}):
- Current Distance to Interaction Point: {v1.distance:.2f}m
- Current Velocity: {v1.velocity:.2f}m/s
- Current Acceleration: {v1.acceleration:.2f} m/s² (if applicable)

Vehicle 2 ({v2.vehicle_id}):
- Current Distance to Interaction Point: {v2.distance:.2f} m
- Current Velocity: {v2.velocity:.2f} m/s
- Current Acceleration: {v2.acceleration:.2f} m/s² (if applicable)

Spatial Analysis:
- Relative Distance (D1 - D2): {relative_distance:.2f} m
- Relative Velocity (V1 - V2): {relative_velocity:.2f} m/s
- Time to Collision: {ttc_str}
- Interaction Urgency: {"High" if time_to_collision and time_to_collision < 3.0 else 'Medium' if time_to_collision and time_to_collision < 6.0 else "Low"}

FUZZY SPATIAL ASSESSMENT:
- Position Advantage: Vehicle {"1" if v1.distance < v2.distance else "2"} is closer to interaction point
- Velocity Advantage: Vehicle {"1" if v1.velocity > v2.velocity else "2"} has higher speed
- Approach Dynamics: {"Converging" if relative_velocity < 0 else 'Diverging' if relative_velocity > 0 else "Parallel"}

AVAILABLE ACTIONS (FUZZY DECISION SPACE):
   Priority Determination Options:
   - Priority to Vehicle 1 (degree: 0.0-1.0)
   - Priority to Vehicle 2 (degree: 0.0-1.0)
   - Shared Priority (degree: 0.0-1.0)
   - Context-Dependent Priority (degree: 0.0-1.0)

DECISION OUTPUT FORMAT:
Provide your fuzzy priority decision in the following JSON format:
{{
    "priority_vehicle": "<{v1.vehicle_id} or {v2.vehicle_id} or shared>",
    "confidence": <0.0_to_1.0>,
    "reasoning": "<detailed_fuzzy_game_theoretic_analysis_explaining_priority_choice>",
    "fuzzy_weights": {{
        "safety_priority": <0.0_to_1.0>,
        "efficiency_priority": <0.0_to_1.0>,
        "right_of_way_priority": <0.0_to_1.0>,
        "cooperation_priority": <0.0_to_1.0>,
        "risk_assessment_priority": <0.0_to_1.0>,
        "contextual_priority": <0.0_to_1.0>
    }},
    "risk_level": "<low|medium|high|critical>",
    "scenario_type": "<{scenario.scenario_type or 'intersection_crossing'}>",
    "game_theory_considerations": "<fuzzy_nash_equilibrium_and_cooperative_analysis>",
    "opponent_model": "<fuzzy_prediction_of_opponent_priority_behavior>"
}}

Execute your fuzzy priority determination now:
"""
            return prompt.strip()

        # Original CoT prompt (when cot_type == "cot")
        prompt = f"""
AUTONOMOUS DRIVING FUZZY GAME-THEORETIC PRIORITY DETERMINATION

SCENARIO DESCRIPTION:
Scenario Type: {scenario.scenario_type or 'intersection_crossing'}
Context: High-interaction driving scenario requiring fuzzy priority assessment with game-theoretic strategic reasoning

SPATIAL RELATIONSHIP DESCRIPTION:
Vehicle 1 ({v1.vehicle_id}):
- Current Distance to Interaction Point: {v1.distance:.2f}m
- Current Velocity: {v1.velocity:.2f}m/s
- Current Acceleration: {v1.acceleration:.2f}m/s² (if applicable)

Vehicle 2 ({v2.vehicle_id}):
- Current Distance to Interaction Point: {v2.distance:.2f}m
- Current Velocity: {v2.velocity:.2f}m/s
- Current Acceleration: {v2.acceleration:.2f}m/s² (if applicable)

Spatial Analysis:
- Relative Distance (D1 - D2): {relative_distance:.2f}m
- Relative Velocity (V1 - V2): {relative_velocity:.2f}m/s
- Time to Collision: {ttc_str}
- Interaction Urgency: {"High" if time_to_collision and time_to_collision < 3.0 else "Medium" if time_to_collision and time_to_collision < 6.0 else "Low"}

FUZZY SPATIAL ASSESSMENT:
- Position Advantage: Vehicle {"1" if v1.distance < v2.distance else "2"} is closer to interaction point
- Velocity Advantage: Vehicle {"1" if v1.velocity > v2.velocity else "2"} has higher speed
- Approach Dynamics: {"Converging" if relative_velocity < 0 else "Diverging" if relative_velocity > 0 else "Parallel"}

REASONING CHAIN:

1. APPLICABLE TRAFFIC RULES (FUZZY INTERPRETATION):
   - Right-of-way rules with degree of applicability (0.0-1.0)
   - Safety distance requirements with flexible enforcement
   - Priority relationships with contextual weighting
   - Traffic rule compliance with situational exceptions

2. GAME-THEORETICAL FUZZY ANALYSIS:
   - Fuzzy Opponent Modeling: Probabilistic prediction of Vehicle 2 behavior
   - Fuzzy Payoff Matrix: Utility evaluation with uncertainty quantification
   - Fuzzy Nash Equilibrium: Optimal strategies with imprecise information
   - Cooperative-Competitive Spectrum: Game strategy selection with degrees
   - Fuzzy Information Asymmetry: Handling incomplete opponent data
   - Multi-Objective Optimization: Balancing competing goals with fuzzy weights

3. AVAILABLE ACTIONS (FUZZY DECISION SPACE):
   Priority Determination Options:
   - Priority to Vehicle 1 (degree: 0.0-1.0)
   - Priority to Vehicle 2 (degree: 0.0-1.0)
   - Shared Priority (degree: 0.0-1.0)
   - Context-Dependent Priority (degree: 0.0-1.0)

4. STRATEGIC CONSIDERATIONS (FUZZY LOGIC):
   - Safety Priority Weight: Distance-based fuzzy membership
   - Efficiency Priority Weight: Velocity-based fuzzy membership
   - Right-of-Way Weight: Traffic rule-based fuzzy membership
   - Cooperation Weight: Game-theoretic fuzzy membership
   - Risk Assessment Weight: Safety-based fuzzy membership
   - Contextual Adaptation Weight: Scenario-based fuzzy membership

DECISION OUTPUT FORMAT:
Provide your fuzzy priority decision in the following JSON format:
{{
    "priority_vehicle": "<{v1.vehicle_id} or {v2.vehicle_id} or shared>",
    "confidence": <0.0_to_1.0>,
    "reasoning": "<detailed_fuzzy_game_theoretic_analysis_explaining_priority_choice>",
    "fuzzy_weights": {{
        "safety_priority": <0.0_to_1.0>,
        "efficiency_priority": <0.0_to_1.0>,
        "right_of_way_priority": <0.0_to_1.0>,
        "cooperation_priority": <0.0_to_1.0>,
        "risk_assessment_priority": <0.0_to_1.0>,
        "contextual_priority": <0.0_to_1.0>
    }},
    "risk_level": "<low|medium|high|critical>",
    "scenario_type": "<{scenario.scenario_type or 'intersection_crossing'}>",
    "game_theory_considerations": "<fuzzy_nash_equilibrium_and_cooperative_analysis>",
    "opponent_model": "<fuzzy_prediction_of_opponent_priority_behavior>"
}}

Execute your fuzzy priority determination now:
"""
        return prompt.strip()

    def _extract_safety_features(self, scenario: InteractionScenario) -> Dict[str, float]:
        """
        Extract safety-related features from scenario for fuzzy processing

        Args:
            scenario: Vehicle interaction scenario

        Returns:
            Dictionary of safety features
        """
        v1, v2 = scenario.vehicle_1, scenario.vehicle_2

        # Calculate derived metrics
        distance_diff = abs(v1.distance - v2.distance)
        velocity_diff = abs(v1.velocity - v2.velocity)
        acceleration_diff = abs(v1.acceleration - v2.acceleration) if v1.acceleration and v2.acceleration else 0

        # Calculate time to collision
        ttc = self._calculate_ttc(v1.distance, v1.velocity, v2.distance, v2.velocity)

        # Relative position advantage (vehicle closer to interaction point has priority)
        position_advantage = v1.distance / (v1.distance + v2.distance) if (v1.distance + v2.distance) > 0 else 0.5

        # Velocity advantage (faster vehicle might have priority in some scenarios)
        velocity_advantage = v1.velocity / (v1.velocity + v2.velocity) if (v1.velocity + v2.velocity) > 0 else 0.5

        return {
            "distance_diff": distance_diff,
            "velocity_diff": velocity_diff,
            "acceleration_diff": acceleration_diff,
            "time_to_collision": ttc if ttc is not None else 100,
            "position_advantage": position_advantage,
            "velocity_advantage": velocity_advantage,
            "v1_distance": v1.distance,
            "v2_distance": v2.distance,
            "v1_velocity": v1.velocity,
            "v2_velocity": v2.velocity
        }

    def _calculate_fuzzy_memberships(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate fuzzy membership values for key features

        Args:
            features: Extracted safety features

        Returns:
            Dictionary of fuzzy membership objects
        """
        from dataclasses import dataclass

        @dataclass
        class FuzzyMembership:
            very_low: float
            low: float
            medium: float
            high: float
            very_high: float

        # Calculate fuzzy memberships for distance risk
        distance_risk = self._calculate_single_fuzzy_membership(
            features["distance_diff"], 0, 50
        )

        # Calculate fuzzy memberships for velocity risk
        velocity_risk = self._calculate_single_fuzzy_membership(
            features["velocity_diff"], 0, 20
        )

        # Calculate fuzzy memberships for TTC risk
        ttc_value = features["time_to_collision"] if features["time_to_collision"] is not None else 100
        ttc_risk = self._calculate_single_fuzzy_membership(
            ttc_value, 0, 10
        )

        return {
            "distance_risk": distance_risk,
            "velocity_risk": velocity_risk,
            "ttc_risk": ttc_risk
        }

    def _calculate_single_fuzzy_membership(self, value: float, min_val: float, max_val: float) -> Any:
        """
        Calculate fuzzy membership values for a given input

        Args:
            value: Input value to fuzzify
            min_val: Minimum possible value
            max_val: Maximum possible value

        Returns:
            FuzzyMembership object with membership degrees
        """
        from dataclasses import dataclass

        @dataclass
        class FuzzyMembership:
            very_low: float
            low: float
            medium: float
            high: float
            very_high: float

        # Normalize value to [0, 1]
        if max_val - min_val == 0:
            normalized = 0.5
        else:
            normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))

        # Calculate triangular membership functions
        very_low = max(0, 1 - normalized * 4)
        low = max(0, min(1, normalized * 4 - 1, 3 - normalized * 4))
        medium = max(0, min(1, normalized * 4 - 2, 4 - normalized * 4))
        high = max(0, min(1, normalized * 4 - 3, 5 - normalized * 4))
        very_high = max(0, normalized * 4 - 4)

        return FuzzyMembership(very_low, low, medium, high, very_high)

    def _calculate_ttc(self, d1: float, v1: float, d2: float, v2: float) -> Optional[float]:
        """
        Calculate time to collision between vehicles

        Args:
            d1, d2: Distances to reference point
            v1, v2: Velocities

        Returns:
            Time to collision in seconds, or None if no collision
        """
        # Simple TTC calculation - can be enhanced
        if abs(v1 - v2) < 0.01:  # Velocities nearly equal
            return None

        ttc = (d2 - d1) / (v1 - v2)
        return ttc if ttc > 0 else None

    def parse_precise_response(self, response: str) -> LLMDecision:
        """
        Parse LLM response into LLMDecision object

        Args:
            response: Raw response string from LLM

        Returns:
            Parsed LLMDecision object
        """
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                return LLMDecision(
                    acceleration_1=float(data['acceleration_1']),
                    reasoning=data.get('reasoning', response),
                    confidence=float(data.get('confidence', 0.0)),
                    strategy_type=data.get('strategy_type', 'unknown')
                )
            else:
                # Fallback: try to extract acceleration value directly
                import re
                accel_match = re.search(r'acceleration[_\s]*1[:\s=]+(-?\d+\.?\d*)', response.lower())
                if accel_match:
                    acceleration = float(accel_match.group(1))
                else:
                    # Default safe acceleration if parsing fails
                    acceleration = 0.0

                return LLMDecision(
                    acceleration_1=acceleration,
                    reasoning=response,
                    confidence=0.0,
                    strategy_type='parse_failed'
                )

        except Exception as e:
            # Return safe default if parsing fails
            return LLMDecision(
                acceleration_1=0.0,
                reasoning=f"Failed to parse LLM response: {str(e)}. Original response: {response}",
                confidence=0.0,
                strategy_type='parse_error'
            )

    def parse_fuzzy_response(self, response: str, scenario: InteractionScenario) -> 'FuzzyDecision':
        """
        Parse LLM fuzzy response into FuzzyDecision object

        Args:
            response: Raw response string from LLM
            scenario: Original scenario for fallback

        Returns:
            Parsed FuzzyDecision object
        """
        from .data_models import FuzzyDecision

        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                return FuzzyDecision(
                    priority_vehicle=data.get('priority_vehicle', scenario.vehicle_1.vehicle_id),
                    confidence=float(data.get('confidence', 0.5)),
                    fuzzy_reasoning=data.get('fuzzy_weights', {}),
                    textual_reasoning=data.get('reasoning', response),
                    risk_level=data.get('risk_level', 'medium'),
                    scenario_type=data.get('scenario_type', scenario.scenario_type or 'other')
                )
            else:
                # Fallback: try to extract priority vehicle directly
                import re
                v1_id = scenario.vehicle_1.vehicle_id
                v2_id = scenario.vehicle_2.vehicle_id

                if v1_id in response and v2_id not in response:
                    priority = v1_id
                elif v2_id in response and v1_id not in response:
                    priority = v2_id
                else:
                    priority = v1_id  # Default fallback

                return FuzzyDecision(
                    priority_vehicle=priority,
                    confidence=0.3,  # Low confidence for fallback parsing
                    fuzzy_reasoning={},
                    textual_reasoning=f"Fallback parsing: {response[:200]}...",
                    risk_level='medium',
                    scenario_type=scenario.scenario_type or 'other'
                )

        except Exception as e:
            # Return safe fallback if parsing fails
            return FuzzyDecision(
                priority_vehicle=scenario.vehicle_1.vehicle_id,
                confidence=0.0,
                fuzzy_reasoning={},
                textual_reasoning=f"Failed to parse fuzzy response: {str(e)}. Original: {response[:200]}...",
                risk_level='unknown',
                scenario_type=scenario.scenario_type or 'parse_error'
            )