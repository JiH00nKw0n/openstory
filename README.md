# OpenStory - Prompt Optimization Framework

A comprehensive framework for benchmarking and optimizing DSPy programs with multiple optimization algorithms.

## Features

- **Multiple Optimizers**: GEPA, SSPO, Bootstrap, MIPRO
- **Benchmark Support**: IFBench (Instruction Following)
- **Flexible Configuration**: Command-line and shell script interfaces
- **Parallel Evaluation**: Multi-threaded execution
- **Comprehensive Logging**: Detailed run logs and metrics

## Quick Start

### Prerequisites
```bash
pip install dspy PyStemmer bm25s python-dotenv
```

### Environment Setup
Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Basic Usage

#### Using Python Script
```bash
# GEPA optimizer (default)
python run.py --optimizer gepa --dataset-mode lite

# SSPO optimizer
python run.py --optimizer sspo --dataset-mode lite

# Bootstrap optimizer
python run.py --optimizer bootstrap --dataset-mode lite

# MIPRO optimizer
python run.py --optimizer mipro --dataset-mode lite
```

#### Using Shell Script
```bash
# GEPA optimizer
./run.sh --optimizer gepa --dataset-mode lite

# SSPO optimizer
./run.sh --optimizer sspo --dataset-mode lite

# Bootstrap optimizer
./run.sh --optimizer bootstrap --dataset-mode lite

# MIPRO optimizer
./run.sh --optimizer mipro --dataset-mode lite
```

## Supported Optimizers

### 1. GEPA (Generative Evolutionary Program Adaptation)
Gradient-based prompt optimization with reflection.

```bash
python run.py --optimizer gepa \
  --model openai/gpt-4.1-mini \
  --reflection-model openai/gpt-5-nano \
  --max-metric-calls 700 \
  --gepa-threads 32
```

**Key Parameters:**
- `--max-metric-calls`: Maximum metric evaluations (default: 10)
- `--gepa-threads`: Number of parallel threads (default: 32)
- `--reflection-minibatch-size`: Reflection batch size (default: 3)
- `--reflection-model`: Model for reflection LM

### 2. SSPO (Self-Supervised Prompt Optimization)
LLM-as-judge based prompt optimization with iterative refinement.

```bash
python run.py --optimizer sspo \
  --model openai/gpt-4.1-mini \
  --optimizer-model openai/gpt-4o-mini \
  --evaluator-model openai/gpt-4o-mini \
  --sspo-max-rounds 10 \
  --sspo-eval-samples 5
```

**Key Parameters:**
- `--sspo-max-rounds`: Maximum optimization rounds (default: 10)
- `--sspo-eval-samples`: Number of evaluation samples (default: 5)
- `--optimizer-model`: Model for prompt generation (default: gpt-4o-mini)
- `--optimizer-temperature`: Temperature for optimizer (default: 0.7)
- `--evaluator-model`: Model for LLM-as-judge (default: gpt-4o-mini)
- `--evaluator-temperature`: Temperature for evaluator (default: 0.3)

**How SSPO Works:**
```
Round 1: Current best vs New candidate
  → Optimizer LLM generates improved prompt
  → Both execute on eval samples
  → LLM-as-judge compares outputs
  → Winner becomes new best

Round 2: New best vs New candidate
  → Repeat...
```

### 3. Bootstrap (Few-Shot Learning)
Automatically generates few-shot demonstrations.

```bash
python run.py --optimizer bootstrap \
  --model openai/gpt-4.1-mini \
  --bootstrap-max-demos 8 \
  --bootstrap-max-labeled 16 \
  --teacher-model openai/gpt-4o
```

**Key Parameters:**
- `--bootstrap-max-demos`: Maximum bootstrapped demos (default: 4)
- `--bootstrap-max-labeled`: Maximum labeled demos (default: 16)
- `--bootstrap-max-rounds`: Maximum rounds (default: 1)
- `--bootstrap-max-errors`: Maximum errors allowed
- `--teacher-model`: Teacher model for demo generation

### 4. MIPRO (Multi-Prompt Instruction and Retrieval Optimization)
Optimizes both prompts and few-shot examples simultaneously.

```bash
python run.py --optimizer mipro \
  --model openai/gpt-4.1-mini \
  --mipro-auto heavy \
  --mipro-num-candidates 20 \
  --mipro-threads 16
```

**Key Parameters:**
- `--mipro-auto`: Optimization mode - light/medium/heavy (default: light)
- `--mipro-max-bootstrapped`: Max bootstrapped demos (default: 4)
- `--mipro-max-labeled`: Max labeled demos (default: 4)
- `--mipro-num-candidates`: Number of candidate configurations
- `--mipro-threads`: Number of parallel threads
- `--mipro-prompt-model`: Model for prompt generation
- `--mipro-task-model`: Model for task execution

## Common Options

### General Options
```bash
--model MODEL              # Main LM model (default: openai/gpt-4.1-mini)
--temperature TEMP         # Temperature (default: 1.0)
--max-tokens TOKENS        # Max tokens (default: 16384)
--benchmark NAME           # Benchmark name (default: IFBench)
--dataset-mode MODE        # Dataset mode: full or lite (default: full)
--eval-threads N           # Evaluation threads (default: 80)
--run-baseline             # Run baseline evaluation
--display-table            # Display evaluation table
--log-dir DIR              # Log directory (default: runs)
```

## Advanced Examples

### Compare Multiple Optimizers
```bash
# GEPA with high metric calls
./run.sh --optimizer gepa --dataset-mode full \
  --max-metric-calls 700 --run-baseline

# SSPO with many rounds
./run.sh --optimizer sspo --dataset-mode full \
  --sspo-max-rounds 20 --sspo-eval-samples 10 --run-baseline

# MIPRO heavy mode
./run.sh --optimizer mipro --dataset-mode full \
  --mipro-auto heavy --mipro-num-candidates 30 --run-baseline
```

### Quick Testing (Lite Mode)
```bash
# Fast iteration with lite dataset
./run.sh --optimizer sspo --dataset-mode lite \
  --sspo-max-rounds 3 --sspo-eval-samples 3
```

### Custom Model Configuration
```bash
# Use different models for different roles
./run.sh --optimizer sspo \
  --model openai/gpt-4 \
  --optimizer-model openai/gpt-4o-mini \
  --evaluator-model openai/gpt-4o-mini
```

## Output Structure

After running, logs are saved to `runs/TIMESTAMP/`:
```
runs/2025-10-13_14-30-00/
├── run_log.txt           # Combined log
├── run_log_stdout.txt    # Standard output
└── run_log_stderr.txt    # Standard error
```

## Example Output

```
==================================================
PROMPT OPTIMIZER RUNNER (SSPO)
==================================================
Optimizer: SSPO
Benchmark: IFBench
Model: openai/gpt-4.1-mini
Optimizer Model: openai/gpt-4o-mini
Evaluator Model: openai/gpt-4o-mini
Dataset Mode: lite
==================================================

✓ Configured language model: openai/gpt-4.1-mini

Loading IFBench benchmark...
Dataset sizes: train=50, val=20, test=20

==================================================
BASELINE EVALUATION
==================================================
✓ Baseline score: 0.464

==================================================
SSPO OPTIMIZATION
==================================================
Training with 50 examples, validating with 20 examples

Round 1/10
  Generated new candidate
  ✓ New candidate is BETTER! Score: 0.512

Round 2/10
  Generated new candidate
  ✗ Current best is still better. Score: 0.512

...

✓ Optimization completed

==================================================
FINAL EVALUATION
==================================================
✓ Final score: 0.533

==================================================
OPTIMIZED PROGRAM INSTRUCTIONS
==================================================
Predictor: generate_response
Prompt:
[Optimized instruction shown here]
*********************************

==================================================
SUMMARY
==================================================
Baseline score: 0.464
Final score: 0.533
Improvement: +14.9%

✓ Run completed! Logs saved to: runs/2025-10-13_14-30-00
```

## Benchmarks

### IFBench (Instruction Following Benchmark)
- **Task**: Instruction following evaluation
- **Modes**:
  - `lite`: Small subset for quick testing
  - `full`: Complete benchmark
- **Program**: 2-stage Chain-of-Thought (generate → refine response)
- **Metric**: Instruction following accuracy

## Project Structure

```
openstory/
├── run.py                      # Main runner script
├── run.sh                      # Shell script wrapper
├── src/
│   ├── benchmarks/            # Benchmark implementations
│   │   └── IFBench/          # IFBench benchmark
│   ├── teleprompt/           # Custom teleprompters
│   │   ├── __init__.py
│   │   └── sspo.py          # SSPO implementation
│   ├── gepa/                 # GEPA implementation
│   └── utils/                # Utility functions
├── runs/                     # Log outputs
└── README.md
```

## Development

### Adding a New Optimizer

1. Implement optimizer in `src/teleprompt/`
2. Add to `get_optimizer()` in `run.py`
3. Add command-line arguments in `parse_args()`
4. Update README with usage examples

### Adding a New Benchmark

1. Create benchmark directory in `src/benchmarks/`
2. Implement `__init__.py` with:
   - `benchmark()`: Load dataset
   - `program`: DSPy program
   - `metric`: Evaluation function
3. Register in `src/benchmarks/__init__.py`

## Help

```bash
# Get help on available options
./run.sh --help
python run.py --help
```

## Troubleshooting

**Error: API Key not found**
- Make sure `.env` file exists with `OPENAI_API_KEY`

**Error: Module not found**
- Install dependencies: `pip install dspy PyStemmer bm25s python-dotenv`

**Slow execution**
- Use `--dataset-mode lite` for faster testing
- Reduce `--eval-threads` if hitting rate limits
- For SSPO, reduce `--sspo-max-rounds` and `--sspo-eval-samples`

## References

- **GEPA**: [DSPy GEPA](https://github.com/stanfordnlp/dspy)
- **SSPO**: [Self-Supervised Prompt Optimization](https://github.com/phureewat29/sspo)
- **Bootstrap**: [DSPy Bootstrap](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/bootstrap.py)
- **MIPRO**: [DSPy MIPRO](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/mipro_optimizer_v2.py)
- **IFBench**: [Instruction Following Benchmark](https://github.com/google-research/google-research/tree/master/instruction_following_eval)

## License

[Your License Here]
