# IFBench GEPA Example Runner

IFBench (Instruction Following Bench) evaluation with GEPA optimization.

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

### Run IFBench Evaluation

```bash
python ifbench_example_runner.py
```

The runner will:
1. **Load IFBench dataset** (instruction following tasks)
2. **Run baseline evaluation** (before optimization)
3. **Apply GEPA optimization** (700 metric calls)
4. **Run final evaluation** (after optimization)
5. **Show performance improvement** and optimized instructions

## What It Does

- **Benchmark**: IFBench instruction following evaluation
- **Program**: 2-stage Chain-of-Thought (generate → refine response)
- **Optimizer**: GEPA (Generative Evolutionary Program Adaptation)
- **Model**: GPT-4.1-mini with temperature 1.0
- **Threads**: 40 parallel evaluations

## Output

- Baseline vs Final scores comparison
- Optimized program instructions
- Performance improvement percentage
- Logs saved to `runs/` directory

## Example Output

```
BASELINE EVALUATION
✓ Baseline score: 0.464

GEPA OPTIMIZER SETUP
✓ GEPA optimizer configured

RUNNING OPTIMIZATION
✓ Optimization completed

FINAL EVALUATION
✓ Final score: 0.533

SUMMARY
Baseline score: 0.464
Final score: 0.533
Improvement: +6.9%
```
