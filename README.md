# LLM-DG: A Benchmark for Evaluating Dynamic Game Decision-Making Capabilities of Large Language Models in Autonomous Driving High-Interaction Scenarios

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive benchmark for evaluating Large Language Models' decision-making capabilities in complex, multi-agent autonomous driving scenarios.

## 🎯 What is LLM-DG?

LLM-DG is a research benchmark that tests how well Large Language Models can handle dynamic game-theoretic decision-making in vehicle interaction scenarios. Based on real-world driving data from the INTERACTIONS dataset, this framework evaluates LLMs' ability to reason about complex traffic situations where multiple vehicles must interact safely and efficiently.

## ✨ Key Features

### 🧠 Dual Decision-Making Modes
- **Precise Mode**: Generates specific acceleration values for real-time vehicle control
- **Fuzzy Mode**: Determines vehicle priorities using fuzzy logic-enhanced reasoning

### 🎮 Chain-of-Thought GameCard Prompts
Structured prompts that translate complex traffic scenarios into LLM-comprehensible formats with step-by-step reasoning

### 📊 Comprehensive Evaluation Framework
Multi-dimensional metrics including safety, efficiency, compliance, and rationality assessments

### 🤖 Multi-LLM Support
Compatible with OpenAI, Doubao, DeepSeek, Qwen, Gemini, Claude, and custom LLM interfaces

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Yuxiao-Cao/LLM-DG.git
cd llm-dg
pip install -r requirements.txt
```

### Environment Variables Setting

Create a `.env` file in the project root and configure your API credentials:

```bash
# OpenAI API Configuration
OPENAI_MODEL_NAME=<your model name>
OPENAI_API_KEY=<your api key>
OPENAI_BASE_URL=<your base url>

# Doubao API Configuration
DOUBAO_MODEL_NAME=<your model name>
DOUBAO_API_KEY=<your api key>
DOUBAO_BASE_URL=<your base url>

# Deepseek API Configuration
DEEPSEEK_MODEL_NAME=<your model name>
DEEPSEEK_API_KEY=<your api key>
DEEPSEEK_BASE_URL=<your base url>

# Qwen API Configuration
QWEN_MODEL_NAME=<your model name>
QWEN_API_KEY=<your api key>
QWEN_BASE_URL=<your base url>

# Gemini API Configuration
GEMINI_MODEL_NAME=<your model name>
GEMINI_API_KEY=<your api key>
GEMINI_BASE_URL=<your base url>

# Claude API Configuration
CLAUDE_MODEL_NAME=<your model name>
CLAUDE_API_KEY=<your api key>
CLAUDE_BASE_URL=<your base url>
```

### Test with LLMs

```bash
# Using OpenAI
python main.py --data-path data/example_data.csv --decision-mode precise --cot-type cot --model-type openai --num-scenarios 5 --prompt-format text

# Using Doubao
python main.py --data-path data/example_data.csv --decision-mode fuzzy --cot-type nocot --model-type doubao --num-scenarios 10 --prompt-format text+json
```

## 📁 Project Structure

```
LLM_DG/
├── src/
│   ├── data_models.py      # Pydantic data models
│   ├── data_loader.py      # Dataset loading and preprocessing
│   ├── gamecard.py         # GameCard prompts with Chain-of-Thought
│   ├── llm_interface.py    # Multi-LLM API interfaces
│   └── evaluation.py       # Evaluation metrics and analysis
├── main.py                 # Main evaluation pipeline
├── data/
│   └── example_data.csv    # Sample interaction scenarios
├── results/               # Evaluation results output
├── .env                    # LLM environment variable configuration
└── requirements.txt        # Python dependencies
```

## 🎮 Decision-Making Modes

### Precise Mode
- **Output**: Specific acceleration values (m/s²) for vehicle control
- **Use Case**: Real-time autonomous driving control
- **Metrics**: Safety, Efficiency, Compliance, Rationality scores
- **Features**: Opponent rationality calibration, game theory integration

### Fuzzy Mode
- **Output**: Vehicle priority decisions with confidence scores
- **Use Case**: Strategic traffic planning and priority management
- **Metrics**: Confidence levels, risk assessment, response times
- **Features**: Fuzzy logic integration, linguistic variable reasoning

## 📊 Data Format

The system processes vehicle interaction data with the following structure:

| Parameter | Description | Type | Example |
|-----------|-------------|------|---------|
| `Scenario_type` | Type of interaction scenario | string | `intersection` |
| `Scenario_id` | Unique identifier for each scenario | integer | `1` |
| `frame_id` | Temporal frame identifier | integer | `0` |
| `track_id_1` | Vehicle 1 identifier | string | `vehicle_1` |
| `d_1` | Vehicle 1 distance to interaction point (meters) | float | `22.0` |
| `v_1` | Vehicle 1 current velocity (m/s) | float | `7.0` |
| `a_1` | Vehicle 1 acceleration decision (m/s²) | float | `-0.818` |
| `track_id_2` | Vehicle 2 identifier | string | `vehicle_2` |
| `d_2` | Vehicle 2 distance to interaction point (meters) | float | `22.0` |
| `v_2` | Vehicle 2 current velocity (m/s) | float | `6.0` |
| `a_2` | Vehicle 2 acceleration decision (m/s²) | float | `2.5` |
| `priority` | Ground truth priority vehicle | string | `vehicle_1` |


## 🎯 Example Results

### Precise Mode Output
```json
{
  "scenario_id": "intersection_1",
  "llm_decision": {
    "acceleration_1": -0.5,
    "reasoning": "Vehicle should decelerate to allow safe merging...",
    "confidence": 0.85,
    "strategy_type": "cooperative"
  },
  "metrics": {
    "safety_score": 92.5,
    "efficiency_score": 78.3,
    "compliance_score": 95.0,
    "rationality_score": 85.7,
    "overall_score": 87.4
  }
}
```

### Fuzzy Mode Output
```json
{
  "scenario_id": "intersection_1",
  "priority_decision": {
    "priority_vehicle": "vehicle_1",
    "confidence": 0.85,
    "risk_level": "medium",
    "fuzzy_reasoning": {
      "distance_risk": 0.3,
      "velocity_risk": 0.7,
      "safety_priority": 0.9
    }
  }
}
```

## 🔧 Advanced Configuration

### Custom Evaluation Weights
```python
from src.evaluation import EvaluationConfig

config = EvaluationConfig(
    safety_weight=0.4,                    # Safety evaluation weight
    efficiency_weight=0.25,               # Efficiency evaluation weight
    compliance_weight=0.25,               # Traffic rule compliance weight
    rationality_weight=0.1,               # Opponent rationality weight

    # Key evaluation parameters
    min_safe_distance=5.0,                # Minimum safe distance (meters)
    max_safe_speed=15.0,                  # Maximum safe speed (m/s)
    max_acceleration=3.0,                 # Maximum acceleration (m/s²)
    reaction_time=1.5,                    # Driver reaction time (seconds)
    intersection_width=10.0,              # Intersection width (meters)
    target_speed=10.0,                    # Target speed (s/s)
    speed_limit=15.0,                     # Speed limit (m/s)
    yield_distance=15.0                   # Yield distance (meters)
)
```

## 📚 Research Applications

- **LLM Benchmarking**: Compare decision-making capabilities across different LLMs
- **Prompt Engineering**: Test effectiveness of different Chain-of-Thought strategies
- **Safety Analysis**: Evaluate LLM performance in safety-critical scenarios
- **Algorithm Development**: Develop hybrid systems combining LLMs with traditional methods
- **Traffic Management**: Priority-based traffic flow optimization

## 🏆 Key Research Contributions

1. **First Comprehensive Benchmark**: For LLM game decision-making in autonomous driving
2. **Dual-Mode Approach**: Both precise control and strategic reasoning capabilities
3. **Chain-of-Thought Innovation**: Novel application to vehicle interaction scenarios
4. **Multi-dimensional Evaluation**: Comprehensive metrics specifically designed for autonomous driving
5. **Opponent Modeling**: Rationality calibration system for realistic behavior simulation

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{llm_dg_2025,
  title={LLM-DG: A Benchmark for Evaluating Dynamic Game Decision-Making Capabilities of Large Language Models in Autonomous Driving High-Interaction Scenarios},
  author={Yuxiao Cao, Yue Duan, Jingchao Wei, Xiangrui Zeng, Zhouping Yin},
  year={2025},
  url={https://github.com/Yuxiao-Cao/LLM-DG},
  note={Research benchmark for evaluating LLM decision-making in autonomous driving scenarios}
}
```

## 📄 License

This project is released under the MIT License.

## ⚠️ Disclaimer

**Research Use Only**: This benchmark is designed for research purposes and should not be used directly in production autonomous driving systems without extensive validation and safety verification.

---

**LLM-DG represents a significant step forward in evaluating LLM decision-making capabilities in safety-critical, multi-agent environments.**