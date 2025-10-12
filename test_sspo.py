#!/usr/bin/env python3
"""
Test SSPO Teleprompter with IFBench
"""

import os
import time
import dspy
from dotenv import load_dotenv

from src.benchmarks import load_benchmark
from src.teleprompt import SSPO
from src.utils.capture_stream_logger import Logger

load_dotenv()


if __name__ == "__main__":
    # Setup logging
    runs_dir = os.path.join(os.getcwd(), "runs", "sspo_" + time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(runs_dir, exist_ok=True)
    logger = Logger(os.path.join(runs_dir, "run_log.txt"))

    print("=" * 50)
    print("SSPO TELEPROMPTER TEST")
    print("=" * 50)
    print(f"Run directory: {runs_dir}")
    print("=" * 50)

    # Configure LM
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        temperature=0.0,
        max_tokens=8192,
        num_retries=0,
        provider=None
    )
    dspy.configure(lm=lm)
    dspy.settings.configure(provide_traceback=True)
    print(f"✓ Configured language model: {lm}")

    # Load IFBench
    print(f"\nLoading IFBench benchmark...")
    benchmark = load_benchmark("IFBench")[0]

    dataset = benchmark.benchmark(dataset_mode="lite")
    program = benchmark.program[0]
    metric = benchmark.metric

    print(f"Dataset sizes: train={len(dataset.train_set)}, val={len(dataset.val_set)}, test={len(dataset.test_set)}")
    print("\nExample from train set:")
    print(dataset.train_set[0])

    # Setup evaluation
    evaluate = dspy.Evaluate(
        devset=dataset.test_set[:10],  # Use small test set for quick testing
        metric=metric,
        num_threads=4,
        display_table=True,
        display_progress=True,
        max_errors=100,
        provide_traceback=True,
    )

    # Baseline evaluation
    print("\n" + "=" * 50)
    print("BASELINE EVALUATION")
    print("=" * 50)
    baseline_score = evaluate(program)
    print(f"✓ Baseline score: {baseline_score}")

    # SSPO optimization
    print("\n" + "=" * 50)
    print("SSPO OPTIMIZATION")
    print("=" * 50)

    # Setup optimizer LM (can use different model for optimization)
    optimizer_lm = dspy.LM(
        model="openai/gpt-4o-mini",
        temperature=0.7,
        max_tokens=8192
    )

    evaluator_lm = dspy.LM(
        model="openai/gpt-4o-mini",
        temperature=0.3,
        max_tokens=4096
    )

    sspo = SSPO(
        metric=metric,
        max_rounds=3,  # Small number for testing
        num_candidates=2,
        eval_sample_size=3,
        optimizer_lm=optimizer_lm,
        evaluator_lm=evaluator_lm,
        temperature=0.7,
        eval_temperature=0.3,
        verbose=True,
    )

    print(f"Training with {len(dataset.train_set[:5])} examples, validating with {len(dataset.val_set[:5])} examples")

    optimized_program = sspo.compile(
        program,
        trainset=dataset.train_set[:5],  # Small set for testing
        valset=dataset.val_set[:5],
    )
    print("✓ Optimization completed")

    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    final_score = evaluate(optimized_program)
    print(f"✓ Final score: {final_score}")

    # Display optimized instructions
    print("\n" + "=" * 50)
    print("OPTIMIZED PROGRAM INSTRUCTIONS")
    print("=" * 50)

    for name, pred in optimized_program.named_predictors():
        print("================================")
        print(f"Predictor: {name}")
        print("================================")
        print("Prompt:")
        print(pred.signature.instructions)
        print("*********************************")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Baseline score: {baseline_score}")
    print(f"Final score: {final_score}")
    if hasattr(final_score, 'score') and hasattr(baseline_score, 'score'):
        if final_score.score > baseline_score.score:
            improvement = ((final_score.score - baseline_score.score) / baseline_score.score) * 100
            print(f"Improvement: +{improvement:.1f}%")
        else:
            print(f"Performance change: {final_score.score - baseline_score.score}")

    print(f"\n✓ Test completed! Logs saved to: {runs_dir}")