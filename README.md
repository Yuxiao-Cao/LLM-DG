# LLM-DG: A Benchmark for Evaluating Dynamic Game Decision-Making Capabilities of Large Language Models in Autonomous Driving High-Interaction Scenarios

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive benchmark for evaluating Large Language Models' decision-making capabilities in high-interactive autonomous driving scenarios.

## Table of Contents

- [What is LLM-DG?](#-what-is-llm-dg)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Reproducible Decoding Protocol](#reproducible-decoding-protocol)
- [Fixed Evaluation Manifest](#fixed-evaluation-manifest)
- [Temperature Stability Experiment](#temperature-stability-experiment)
- [Stability Metrics](#stability-metrics)
- [Output Provenance](#output-provenance)
- [Project Structure](#-project-structure)
- [Decision-Making Modes](#-decision-making-modes)
- [Non-LLM Baselines](#non-llm-baselines-behavioral-cloning)
- [Data Format](#-data-format)
- [Closed-Loop Rollout](#-supplementary-closed-loop-rollout-experiment)
- [Research Applications](#-research-applications)
- [Citation](#-citation)

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

## 🔧 Installation

### Method 1: Using pip

```bash
git clone https://github.com/Yuxiao-Cao/LLM-DG.git
cd LLM-DG
pip install -r requirements.txt
```

### Method 2: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n llm-dg python=3.10 -y
conda activate llm-dg

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables Setting

Copy the public template and fill in only the providers you intend to use:

```bash
cp .env.example .env
```

`.env.example` lists the model name, API key, and base URL variables used by each provider. It contains blank values only. Keep `.env` local; it is ignored by Git. The application loads `.env` through `python-dotenv`, so no shell-specific `source` command is required.

## 🚀 Quick Start

### Evaluating with LLMs

```bash
# Using OpenAI (Precise mode with Chain-of-Thought and low-temperature decoding)
python main.py --data-path data/example_data.csv --decision-mode precise --cot-type cot --model-type openai --num-scenarios 5 --prompt-format text --temperature 0.0 --top-p 1.0

# Using Doubao (Fuzzy mode without Chain-of-Thought and strict parsing)
python main.py --data-path data/example_data.csv --decision-mode fuzzy --cot-type nocot --model-type doubao --num-scenarios 10 --prompt-format text+json --temperature 0.0 --top-p 1.0 --strict-parse
```

The open-loop benchmark defaults to `--temperature 0.0` and `--top-p 1.0`. For formal cross-model comparisons, use a fixed evaluation Manifest as described below instead of independent random samples.

### Strict JSON Parsing

New open-loop experiments use strict parsing by default. In precise mode, a valid response must contain a legal `acceleration_1` value in `[-3, 3]`; in fuzzy mode, `priority_vehicle` must identify the ego vehicle, opponent vehicle, or the supported `shared` class. Markdown-fenced JSON is accepted after removing the fence. Regex or legacy defaults remain traceable as fallbacks, but invalid responses are not scored when strict parsing is enabled. Use `--no-strict-parse` only for explicitly labeled backward-compatibility runs.

## Reproducible Decoding Protocol

The main open-loop benchmark uses low-temperature decoding by default:

- `temperature=0.0`
- `top_p=1.0`
- strict response parsing enabled

`temperature=0` should be interpreted as **near-deterministic decoding**, not a guarantee of bit-for-bit identical responses from a commercial cloud model. Results may still change when the exact model version, provider routing, backend implementation, system fingerprint, or access time changes. A generation seed is sent only when `--generation-seed` is provided, and it affects generation only if the selected provider and model support that parameter. Even when accepted, a seed does not by itself guarantee absolute determinism.

For a formal main experiment, specify the generation configuration explicitly and archive the resulting metadata:

```bash
python main.py \
    --data-path data/example_data.csv \
    --manifest-path manifests/main_eval.csv \
    --model-type deepseek \
    --decision-mode precise \
    --prompt-format text+json \
    --cot-type cot \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-tokens 1000 \
    --generation-seed 42 \
    --strict-parse \
    --output-dir outputs/open_loop_LLM/
```

## Fixed Evaluation Manifest

Each evaluation sample is uniquely identified by the composite key:

```text
Scenario_id + frame_id
```

Its stable sample identifier is `<Scenario_id>::<frame_id>`. Generate a versionable Manifest with:

```bash
python scripts/create_eval_manifest.py \
    --data-path data/example_data.csv \
    --num-scenarios 100 \
    --seed 42 \
    --output manifests/main_eval.csv
```

The generated CSV preserves a fixed `manifest_order` and is accompanied by `manifests/main_eval.metadata.json`, which records the source-data SHA256, seed, sample counts, generation time, Git commit, and unique-key definition. Formal cross-model comparisons must use the **same Manifest**, not separately sampled scenario sets. Archive and version the Manifest and its metadata JSON together with the experimental results.

When `--manifest-path` is supplied to `main.py` or the stability runner, all Manifest samples are used in their recorded order; `--num-scenarios` and random sampling are not applied again.

## Temperature Stability Experiment

The stability runner evaluates the Cartesian product `temperature × repeat_id × sample_id`, then deterministically shuffles the serial execution schedule using `--schedule-seed`. A practical default is `--repeats 5`, comparing `temperature=0.0` with `temperature=0.7`.

Precise mode:

```bash
python scripts/run_temperature_stability.py \
    --data-path data/example_data.csv \
    --manifest-path manifests/main_eval.csv \
    --model-type deepseek \
    --decision-mode precise \
    --prompt-format text+json \
    --cot-type cot \
    --temperatures 0.0 0.7 \
    --repeats 5 \
    --top-p 1.0 \
    --max-tokens 1000 \
    --generation-seed 42 \
    --schedule-seed 20260712 \
    --max-retries 2 \
    --retry-backoff 1.0 \
    --output outputs/temperature_stability/deepseek_precise.jsonl \
    --resume
```

Fuzzy mode:

```bash
python scripts/run_temperature_stability.py \
    --data-path data/example_data.csv \
    --manifest-path manifests/main_eval.csv \
    --model-type deepseek \
    --decision-mode fuzzy \
    --prompt-format text+json \
    --cot-type cot \
    --temperatures 0.0 0.7 \
    --repeats 5 \
    --top-p 1.0 \
    --max-tokens 1000 \
    --schedule-seed 20260712 \
    --output outputs/temperature_stability/deepseek_fuzzy.jsonl \
    --resume
```

The output is append-only JSONL with one final record per scheduled task. `--resume` skips job IDs that already completed without an API error; final API failures remain recorded and can be retried in a later resumed run. Use `--dry-run` first to inspect the task count without creating the JSONL or calling an API.

```bash
python scripts/run_temperature_stability.py \
    --data-path data/example_data.csv \
    --manifest-path manifests/main_eval.csv \
    --model-type deepseek \
    --decision-mode precise \
    --prompt-format text+json \
    --cot-type cot \
    --temperatures 0.0 0.7 \
    --repeats 5 \
    --output outputs/temperature_stability/dry_run.jsonl \
    --dry-run
```

The stability experiment always analyzes the model's **raw decision**. It does not apply opponent rationality calibration, and calibrated actions must not be used to compute output variance or agreement.

Analyze one or more JSONL files with:

```bash
python scripts/analyze_temperature_stability.py \
    --inputs "outputs/temperature_stability/*.jsonl" \
    --output-dir outputs/temperature_stability/analysis \
    --bootstrap-samples 1000 \
    --bootstrap-seed 42
```

Invalid responses are excluded from decision statistics by default. Use `--include-invalid` only for an explicitly labeled sensitivity analysis.

## Stability Metrics

- **Precise acceleration within-scenario SD**: the standard deviation of acceleration decisions across repeated generations for the same `sample_id`. It measures repeatability, not scene difficulty.
- **Mean absolute pairwise difference (MAPD)**: the mean absolute acceleration difference across all pairs of repeated generations for one sample. It is expressed in the same units as acceleration and is often easier to interpret than variance.
- **Fuzzy majority agreement**: the fraction of repeated decisions assigned to the most frequent priority class.
- **Fuzzy pairwise agreement**: the fraction of all repeat pairs that produce the same priority decision.
- **Normalized decision entropy**: the entropy of the repeated priority distribution, normalized by the number of observed decision categories. Zero indicates complete agreement; larger values indicate less stable categorical decisions.
- **Strict JSON valid rate**: the fraction of calls parsed as valid strict JSON, excluding regex and default fallbacks.
- **API success rate**: the fraction of scheduled calls that returned without a final API error. It is reported separately from parsing validity.
- **Rank stability**: Spearman correlation of model score rankings between `temperature=0.0` and `temperature=0.7`, and between repeats at the same temperature. Rankings use only the common set of valid, scored samples; missing results are never silently filled with zero.

The original benchmark tables usually report **cross-scenario SD**, which measures how a metric varies across different traffic scenes. The repeated-sampling experiment reports **within-scenario SD**, which measures how multiple generations vary for the same scene. These quantities answer different questions and should not be placed under an unlabeled generic `std` column.

## Output Provenance

Open-loop and stability outputs retain the information needed to audit a run, including:

- exact `model_name`, provider, and model type;
- requested generation parameters and effective parameters reported or inferred by the interface;
- generation-seed capability metadata, without claiming provider-side determinism;
- SHA256 hash of each prompt;
- raw model response;
- parse status, validity flag, parser used, and parse error where applicable;
- raw decision and, for calibrated main experiments, a separately stored calibrated decision;
- Git commit, UTC run time, Python version, and source-data path in main-run metadata;
- Manifest path and the separately archived Manifest plus metadata JSON;
- API usage and system fingerprint when returned by the provider.

Provider APIs do not always return usage, fingerprints, or confirmed effective parameters. Missing provenance fields are stored as null rather than fabricated. The `requested_parameters` field records what the client sent; it should not be interpreted as proof that every backend honored every option.

## 📁 Project Structure

```
LLM-DG/
├── src/
│   ├── __init__.py
│   ├── data_models.py      # Pydantic data models
│   ├── data_loader.py      # Dataset loading and preprocessing
│   ├── gamecard.py         # GameCard prompts with Chain-of-Thought
│   ├── llm_interface.py    # Multi-LLM API interfaces
│   ├── evaluation.py       # Evaluation metrics and analysis
│   ├── baselines/          # Non-LLM baselines (BC-MLP, BC-Transformer)
│   │   ├── __init__.py
│   │   ├── dataset.py      # Data loading and preprocessing
│   │   ├── bc_mlp.py       # MLP policy network
│   │   ├── bc_transformer.py # Transformer policy network
│   │   └── utils.py        # Shared utilities
│   └── experiments/
│       ├── __init__.py
│       └── closed_loop.py  # Closed-loop rollout experiment
├── scripts/
│   ├── create_eval_manifest.py       # Create a fixed evaluation Manifest
│   ├── run_temperature_stability.py   # Run repeated decoding experiments
│   ├── analyze_temperature_stability.py # Analyze stability JSONL outputs
│   ├── run_closed_loop_rollout.py     # Closed-loop experiment script
│   ├── train_bc.py                    # Train BC baselines
│   └── eval_bc_openloop.py            # Evaluate BC baselines (open-loop)
├── main.py                 # Main evaluation pipeline (open-loop)
├── data/
│   └── example_data.csv    # Sample interaction scenarios
├── outputs/
│   ├── open_loop_LLM/       # Open-loop LLM evaluation results output
│   ├── closed_loop_LLM/     # Closed-loop LLM rollout results output
│   └── baseline_bc/         # BC baseline evaluation results output
├── checkpoints/            # Model checkpoints (for BC baselines)
├── .env                    # LLM environment variable configuration
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
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

## 🤖 Non-LLM Baselines (Behavioral Cloning)

The repository includes non-LLM learning-based planner baselines that map the same state representation (kinematics) to continuous acceleration outputs. These baselines are evaluated under the same metrics as LLM-based planners using the existing evaluation functions in `src/evaluation.py`.

### Model Architectures

**BC-MLP**: A 5-layer Multi-Layer Perceptron with ~125K parameters
- Architecture: `[256, 256, 128, 128, 64]` hidden layers with Dropout
- Input: `[d_ego, v_ego, d_opp, v_opp]` (4-dimensional kinematic state)
- Output: Acceleration in `[-3, 3]` m/s²

**BC-Transformer**: A 4-layer Transformer encoder with ~556K parameters
- Architecture: d_model=128, 4 attention heads, 4 encoder layers
- Treats ego/opp states as 2 tokens with positional encoding
- Pre-LN architecture with GELU activation

### Training

```bash
# Train BC-MLP baseline
python scripts/train_bc.py --data-path data/example_data.csv --model mlp --epochs 200 --batch-size 32

# Train BC-Transformer baseline
python scripts/train_bc.py --data-path data/example_data.csv --model transformer --epochs 200 --batch-size 32
```

**Training Features**:
- 70/10/20 train/val/test split by Scenario_id
- Role-swap data augmentation
- Z-score normalization fitted on training data only
- Huber loss with AdamW optimizer
- Warmup + Cosine Annealing learning rate schedule
- Early stopping with configurable patience

### Evaluation

```bash
# Evaluate BC-MLP (open-loop)
python scripts/eval_bc_openloop.py --checkpoint checkpoints/baselines/mlp_example_data_best.pth --data-path data/example_data.csv

# Evaluate BC-Transformer (open-loop)
python scripts/eval_bc_openloop.py --checkpoint checkpoints/baselines/transformer_example_data_best.pth --data-path data/example_data.csv
```

**Output Files**:
- CSV results: `outputs/baseline_bc/{model}_{dataset}_results.csv`
- JSON summary: `outputs/baseline_bc/{model}_{dataset}_summary.json`

**Note**: BC baselines do not output natural language reasoning, but are evaluated under the same metrics (safety, efficiency, compliance, rationality) as LLM-based planners.

## 📊 Data Format

The system processes vehicle interaction data with the following structure:

| Parameter | Description | Type | Example |
|-----------|-------------|------|---------|
| `Scenario_type` | Type of interaction scenario | string | `intersection` |
| `Scenario_id` | Unique identifier for each scenario | string/int | `1` or `intersection_multi_20` |
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

## 🔄 Supplementary Closed-Loop Rollout Experiment

The closed-loop rollout experiment addresses concerns about open-loop evaluation and compounding errors in multi-step decision making. In this mode, both vehicles are controlled simultaneously by the LLM over multiple time steps, with state updates based on a kinematic model.

### Key Differences from Open-Loop Evaluation

| Aspect | Open-Loop (main.py) | Closed-Loop (run_closed_loop_rollout.py) |
|--------|---------------------|------------------------------------------|
| Steps per scenario | Single decision | Multi-step (default: 10) |
| State update | None | Kinematic model based on LLM actions |
| Vehicle control | Ego vehicle only | Both vehicles simultaneously |
| Rationality metric | Computed | **Not computed** (no ground truth for multi-step) |
| Termination | N/A | Success/collision/timeout conditions |
| Output | JSON results | JSONL trajectories + CSV summary |
| Default episodes | 10 scenarios | 50 initial states |

### Running Closed-Loop Rollout

```bash
# Basic usage with DeepSeek (default: 50 initial states, 10 max steps)
python3 scripts/run_closed_loop_rollout.py --dataset data/example_data.csv

# With custom parameters
python3 scripts/run_closed_loop_rollout.py \
    --dataset data/example_data.csv \
    --n_init 100 \
    --dt 1.0 \
    --max_steps 15 \
    --model deepseek \
    --input_format text_json \
    --seed 42 \
    --out_dir outputs/closed_loop_LLM/
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `data/example_data.csv` | Path to dataset CSV file |
| `--n_init` | `50` | Number of initial states to sample |
| `--dt` | `1.0` | Time step in seconds |
| `--max_steps` | `10` | Maximum steps per episode |
| `--model` | `deepseek` | LLM model to use |
| `--input_format` | `text_json` | Prompt format (text_json/json_only/text_only) |
| `--decision_mode` | `precise` | Decision mode (precise only) |
| `--cot_type` | `cot` | Chain-of-Thought type (cot/nocot) |
| `--seed` | `0` | Random seed for reproducibility |
| `--out_dir` | `outputs/closed_loop_LLM/` | Output directory |

### Expected Outputs

The script generates the following files in `outputs/closed_loop_LLM/`:

   ```json
   {
     "dataset_name": "example_data",
     "model_name": "deepseek",
     "seed": 0,
     "init_index": 5,
     "init_state": {"d1": 22.0, "v1": 7.0, "d2": 22.0, "v2": 6.0},
     "per_step": [
       {
         "k": 0,
         "state": {"d1": 22.0, "v1": 7.0, "d2": 22.0, "v2": 6.0},
         "a1": -0.5,
         "a2": 1.2,
         "safety_sys": 85.3,
         "efficiency_sys": 78.5,
         "compliance_sys": 92.1,
         "overall_sys": 84.7,
         "stop_flag": false
       },
       ...
     ],
     "summary": {
       "min_safety": 72.1,
       "mean_overall": 81.3,
       "n_steps": 4,
       "outcome": "success"
     }
   }
   ```

### Termination Conditions

An episode terminates when:
- **Success**: Either vehicle passes the intersection (`d < -5m`)
- **Collision**: System safety score drops to 0
- **Timeout**: Maximum steps reached without other termination
- **Invalid Parse**: LLM response cannot be parsed

### System-Level Metrics

At each step, the following system-level metrics are computed:

```
safety_sys = min(safety_v1, safety_v2)
compliance_sys = min(compliance_v1, compliance_v2)
efficiency_sys = (efficiency_v1 + efficiency_v2) / 2
overall_sys = (0.4 * safety_sys + 0.25 * efficiency_sys + 0.25 * compliance_sys) / 0.9
```

**Note**: Rationality is intentionally excluded from closed-loop evaluation as it depends on matching single-step action labels from the dataset.

### Kinematic State Update

Vehicle states are updated using a simple kinematic model:

```
d_next = d - v * dt - 0.5 * a * dt^2
v_next = max(0, v + a * dt)
```

where `d` is distance to intersection, `v` is velocity, and `a` is acceleration (clamped to [-3, 3] m/s²).

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
    target_speed=10.0,                    # Target speed (m/s)
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
- **Compounding Error Analysis**: Assess error accumulation in multi-step scenarios (closed-loop)

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
  title={LLM-DG: A Benchmark for Enhancing Autonomous Driving Game-Theoretic Decision-Making via Large Language Models},
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
