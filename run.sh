#!/bin/bash

# Prompt Optimizer Runner Shell Script
# Usage: ./run.sh [options]

# Default values
OPTIMIZER="gepa"
MODEL="openai/gpt-4.1-mini"
REFLECTION_MODEL="openai/gpt-5-nano"
TEMPERATURE=1.0
MAX_TOKENS=16384
BENCHMARK="IFBench"
DATASET_MODE="full"
EVAL_THREADS=80

# GEPA defaults
GEPA_THREADS=32
MAX_METRIC_CALLS=700
REFLECTION_MINIBATCH_SIZE=3

# SSPO defaults
SSPO_MAX_ROUNDS=10
SSPO_EVAL_SAMPLES=5
OPTIMIZER_MODEL="openai/gpt-4o-mini"
OPTIMIZER_TEMPERATURE=0.7
EVALUATOR_MODEL="openai/gpt-4o-mini"
EVALUATOR_TEMPERATURE=0.3

LOG_DIR="runs"
RUN_BASELINE=false
DISPLAY_TABLE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --reflection-model)
            REFLECTION_MODEL="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --dataset-mode)
            DATASET_MODE="$2"
            shift 2
            ;;
        --eval-threads)
            EVAL_THREADS="$2"
            shift 2
            ;;
        --gepa-threads)
            GEPA_THREADS="$2"
            shift 2
            ;;
        --max-metric-calls)
            MAX_METRIC_CALLS="$2"
            shift 2
            ;;
        --reflection-minibatch-size)
            REFLECTION_MINIBATCH_SIZE="$2"
            shift 2
            ;;
        --sspo-max-rounds)
            SSPO_MAX_ROUNDS="$2"
            shift 2
            ;;
        --sspo-eval-samples)
            SSPO_EVAL_SAMPLES="$2"
            shift 2
            ;;
        --optimizer-model)
            OPTIMIZER_MODEL="$2"
            shift 2
            ;;
        --optimizer-temperature)
            OPTIMIZER_TEMPERATURE="$2"
            shift 2
            ;;
        --evaluator-model)
            EVALUATOR_MODEL="$2"
            shift 2
            ;;
        --evaluator-temperature)
            EVALUATOR_TEMPERATURE="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --run-baseline)
            RUN_BASELINE=true
            shift
            ;;
        --display-table)
            DISPLAY_TABLE=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./run.sh [options]"
            echo ""
            echo "General Options:"
            echo "  --optimizer NAME                  Optimizer: gepa or sspo (default: gepa)"
            echo "  --model MODEL                     Model to use (default: openai/gpt-4.1-mini)"
            echo "  --temperature TEMP                Temperature (default: 1.0)"
            echo "  --max-tokens TOKENS               Max tokens (default: 16384)"
            echo "  --benchmark NAME                  Benchmark name (default: IFBench)"
            echo "  --dataset-mode MODE               Dataset mode: full or lite (default: full)"
            echo "  --eval-threads N                  Evaluation threads (default: 80)"
            echo "  --log-dir DIR                     Log directory (default: runs)"
            echo "  --run-baseline                    Run baseline evaluation"
            echo "  --display-table                   Display evaluation table"
            echo ""
            echo "GEPA Options:"
            echo "  --reflection-model MODEL          Reflection model (default: openai/gpt-5-nano)"
            echo "  --gepa-threads N                  GEPA threads (default: 32)"
            echo "  --max-metric-calls N              Max metric calls (default: 700)"
            echo "  --reflection-minibatch-size N     Reflection minibatch size (default: 3)"
            echo ""
            echo "SSPO Options:"
            echo "  --sspo-max-rounds N               Max optimization rounds (default: 10)"
            echo "  --sspo-num-candidates N           Candidates per round (default: 3)"
            echo "  --sspo-eval-samples N             Eval sample size (default: 5)"
            echo "  --optimizer-model MODEL           Optimizer LM (default: openai/gpt-4o-mini)"
            echo "  --optimizer-temperature TEMP      Optimizer temperature (default: 0.7)"
            echo "  --evaluator-model MODEL           Evaluator LM (default: openai/gpt-4o-mini)"
            echo "  --evaluator-temperature TEMP      Evaluator temperature (default: 0.3)"
            echo ""
            echo "  -h, --help                        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build the command
CMD="python run.py"
CMD="$CMD --optimizer $OPTIMIZER"
CMD="$CMD --model $MODEL"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --max-tokens $MAX_TOKENS"
CMD="$CMD --benchmark $BENCHMARK"
CMD="$CMD --dataset-mode $DATASET_MODE"
CMD="$CMD --eval-threads $EVAL_THREADS"
CMD="$CMD --log-dir $LOG_DIR"

# Add optimizer-specific options
if [ "$OPTIMIZER" = "gepa" ]; then
    CMD="$CMD --reflection-model $REFLECTION_MODEL"
    CMD="$CMD --gepa-threads $GEPA_THREADS"
    CMD="$CMD --max-metric-calls $MAX_METRIC_CALLS"
    CMD="$CMD --reflection-minibatch-size $REFLECTION_MINIBATCH_SIZE"
elif [ "$OPTIMIZER" = "sspo" ]; then
    CMD="$CMD --sspo-max-rounds $SSPO_MAX_ROUNDS"
    CMD="$CMD --sspo-eval-samples $SSPO_EVAL_SAMPLES"
    CMD="$CMD --optimizer-model $OPTIMIZER_MODEL"
    CMD="$CMD --optimizer-temperature $OPTIMIZER_TEMPERATURE"
    CMD="$CMD --evaluator-model $EVALUATOR_MODEL"
    CMD="$CMD --evaluator-temperature $EVALUATOR_TEMPERATURE"
fi

if [ "$RUN_BASELINE" = true ]; then
    CMD="$CMD --run-baseline"
fi

if [ "$DISPLAY_TABLE" = true ]; then
    CMD="$CMD --display-table"
fi

# Create log directory with timestamp
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_LOG_DIR="$LOG_DIR/$TIMESTAMP"
mkdir -p "$RUN_LOG_DIR"

LOG_FILE="$RUN_LOG_DIR/run_log.txt"
STDOUT_LOG="$RUN_LOG_DIR/run_log_stdout.txt"
STDERR_LOG="$RUN_LOG_DIR/run_log_stderr.txt"

# Print configuration (both to console and log file)
print_config() {
    echo "=================================================="
    echo "Prompt Optimizer Runner Configuration"
    echo "=================================================="
    echo "Optimizer: $OPTIMIZER"
    echo "Model: $MODEL"
    echo "Temperature: $TEMPERATURE"
    echo "Max Tokens: $MAX_TOKENS"
    echo "Benchmark: $BENCHMARK"
    echo "Dataset Mode: $DATASET_MODE"
    echo "Eval Threads: $EVAL_THREADS"

    if [ "$OPTIMIZER" = "gepa" ]; then
        echo "--- GEPA Settings ---"
        echo "Reflection Model: $REFLECTION_MODEL"
        echo "GEPA Threads: $GEPA_THREADS"
        echo "Max Metric Calls: $MAX_METRIC_CALLS"
        echo "Reflection Minibatch Size: $REFLECTION_MINIBATCH_SIZE"
    elif [ "$OPTIMIZER" = "sspo" ]; then
        echo "--- SSPO Settings ---"
        echo "Max Rounds: $SSPO_MAX_ROUNDS"
        echo "Eval Samples: $SSPO_EVAL_SAMPLES"
        echo "Optimizer Model: $OPTIMIZER_MODEL"
        echo "Optimizer Temperature: $OPTIMIZER_TEMPERATURE"
        echo "Evaluator Model: $EVALUATOR_MODEL"
        echo "Evaluator Temperature: $EVALUATOR_TEMPERATURE"
    fi

    echo "Log Directory: $RUN_LOG_DIR"
    echo "Run Baseline: $RUN_BASELINE"
    echo "Display Table: $DISPLAY_TABLE"
    echo "=================================================="
    echo "Start Time: $(date)"
    echo "=================================================="
    echo ""
    echo "Running command:"
    echo "$CMD"
    echo "=================================================="
    echo ""
}

# Print to console
print_config

# Print to log file
print_config > "$LOG_FILE"

# Run the command with logging
# - All output (stdout + stderr) goes to run_log.txt
# - stdout separately logged to run_log_stdout.txt
# - stderr separately logged to run_log_stderr.txt
# - Also display output to console using tee
echo "Logs will be saved to: $RUN_LOG_DIR"
echo ""

eval $CMD 2>&1 | tee -a "$LOG_FILE" | tee >(grep -v "^" 1>&2 > "$STDOUT_LOG") 2> >(tee -a "$STDERR_LOG" >&2)

# Capture exit code
EXIT_CODE=$?

# Print completion message
echo "" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "Run completed with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "End Time: $(date)" | tee -a "$LOG_FILE"
echo "Logs saved to: $RUN_LOG_DIR" | tee -a "$LOG_FILE"
echo "  - Combined log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "  - Stdout log: $STDOUT_LOG" | tee -a "$LOG_FILE"
echo "  - Stderr log: $STDERR_LOG" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE