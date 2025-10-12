#!/usr/bin/env python3
"""
Optimizer Runner with configurable parameters
Supports multiple optimizers: GEPA, SSPO, etc.
"""

import os
import time
import argparse
import dspy
from dotenv import load_dotenv

from src.benchmarks import load_benchmark
from src.utils.capture_stream_logger import Logger
from src.teleprompt import SSPO

load_dotenv()


def get_optimizer(optimizer_name, args, benchmark, lm):
    """
    Factory function to get the appropriate optimizer based on name.

    Args:
        optimizer_name: Name of the optimizer ('gepa', 'sspo', etc.)
        args: Command-line arguments
        benchmark: Benchmark metadata
        lm: Language model

    Returns:
        Configured optimizer instance
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "gepa":
        optimizer = dspy.GEPA(
            metric=benchmark.metric_with_feedback,
            num_threads=args.gepa_threads,
            track_stats=True,
            reflection_minibatch_size=args.reflection_minibatch_size,
            reflection_lm=dspy.LM(
                model=args.reflection_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            ),
            max_metric_calls=args.max_metric_calls,
        )

    elif optimizer_name == "sspo":
        # Setup optimizer and evaluator LMs
        optimizer_lm = dspy.LM(
            model=args.optimizer_model,
            temperature=args.optimizer_temperature,
            max_tokens=args.max_tokens
        )

        evaluator_lm = dspy.LM(
            model=args.evaluator_model,
            temperature=args.evaluator_temperature,
            max_tokens=args.max_tokens
        )

        optimizer = SSPO(
            metric=benchmark.metric,
            max_rounds=args.sspo_max_rounds,
            eval_sample_size=args.sspo_eval_samples,
            optimizer_lm=optimizer_lm,
            evaluator_lm=evaluator_lm,
            temperature=args.optimizer_temperature,
            eval_temperature=args.evaluator_temperature,
            verbose=True,
        )

    elif optimizer_name == "bootstrap":
        # Setup teacher settings if specified
        teacher_settings = {}
        if hasattr(args, 'teacher_model') and args.teacher_model:
            teacher_settings['lm'] = dspy.LM(
                model=args.teacher_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )

        optimizer = dspy.BootstrapFewShot(
            metric=benchmark.metric,
            max_bootstrapped_demos=args.bootstrap_max_demos,
            max_labeled_demos=args.bootstrap_max_labeled,
            max_rounds=args.bootstrap_max_rounds,
            max_errors=args.bootstrap_max_errors,
            teacher_settings=teacher_settings if teacher_settings else None,
        )

    elif optimizer_name == "mipro":
        # Setup prompt and task models
        prompt_model = None
        if hasattr(args, 'mipro_prompt_model') and args.mipro_prompt_model:
            prompt_model = dspy.LM(
                model=args.mipro_prompt_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )

        task_model = None
        if hasattr(args, 'mipro_task_model') and args.mipro_task_model:
            task_model = dspy.LM(
                model=args.mipro_task_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )

        optimizer = dspy.MIPROv2(
            metric=benchmark.metric,
            prompt_model=prompt_model,
            task_model=task_model,
            max_bootstrapped_demos=args.mipro_max_bootstrapped,
            max_labeled_demos=args.mipro_max_labeled,
            auto=args.mipro_auto,
            num_candidates=args.mipro_num_candidates,
            num_threads=args.mipro_threads,
            init_temperature=args.temperature,
            verbose=True,
            track_stats=True,
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Supported: gepa, sspo, bootstrap, mipro")

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run prompt optimization on benchmarks")

    # Optimizer selection
    parser.add_argument("--optimizer", type=str, default="gepa",
                        choices=["gepa", "sspo", "bootstrap", "mipro"],
                        help="Optimizer to use (gepa, sspo, bootstrap, or mipro)")

    # Model configuration
    parser.add_argument("--model", type=str, default="openai/gpt-4.1-mini",
                        help="Model to use for main LM")
    parser.add_argument("--reflection-model", type=str, default="openai/gpt-5-nano",
                        help="Model to use for reflection LM")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for LM")
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="Max tokens for LM")

    # Benchmark configuration
    parser.add_argument("--benchmark", type=str, default="IFBench",
                        help="Benchmark to run")
    parser.add_argument("--dataset-mode", type=str, default="full",
                        choices=["full", "lite"],
                        help="Dataset mode (full or lite)")

    # Evaluation configuration
    parser.add_argument("--eval-threads", type=int, default=80,
                        help="Number of threads for evaluation")
    parser.add_argument("--display-table", action="store_true",
                        help="Display evaluation table")

    # GEPA configuration
    parser.add_argument("--gepa-threads", type=int, default=32,
                        help="Number of threads for GEPA optimization")
    parser.add_argument("--max-metric-calls", type=int, default=10,
                        help="Maximum metric calls for GEPA")
    parser.add_argument("--reflection-minibatch-size", type=int, default=3,
                        help="Reflection minibatch size for GEPA")

    # SSPO configuration
    parser.add_argument("--sspo-max-rounds", type=int, default=10,
                        help="Maximum optimization rounds for SSPO")
    parser.add_argument("--sspo-eval-samples", type=int, default=5,
                        help="Number of samples for evaluation in SSPO")
    parser.add_argument("--optimizer-model", type=str, default="openai/gpt-4o-mini",
                        help="Model for SSPO optimizer (prompt generation)")
    parser.add_argument("--optimizer-temperature", type=float, default=0.7,
                        help="Temperature for SSPO optimizer")
    parser.add_argument("--evaluator-model", type=str, default="openai/gpt-4o-mini",
                        help="Model for SSPO evaluator (LLM-as-judge)")
    parser.add_argument("--evaluator-temperature", type=float, default=0.3,
                        help="Temperature for SSPO evaluator")

    # Bootstrap configuration
    parser.add_argument("--bootstrap-max-demos", type=int, default=4,
                        help="Maximum bootstrapped demonstrations for Bootstrap")
    parser.add_argument("--bootstrap-max-labeled", type=int, default=16,
                        help="Maximum labeled demonstrations for Bootstrap")
    parser.add_argument("--bootstrap-max-rounds", type=int, default=1,
                        help="Maximum rounds for Bootstrap")
    parser.add_argument("--bootstrap-max-errors", type=int, default=None,
                        help="Maximum errors for Bootstrap")
    parser.add_argument("--teacher-model", type=str, default=None,
                        help="Teacher model for Bootstrap")

    # MIPRO configuration
    parser.add_argument("--mipro-auto", type=str, default="light",
                        choices=["light", "medium", "heavy"],
                        help="MIPRO auto mode (light, medium, or heavy)")
    parser.add_argument("--mipro-max-bootstrapped", type=int, default=4,
                        help="Maximum bootstrapped demos for MIPRO")
    parser.add_argument("--mipro-max-labeled", type=int, default=4,
                        help="Maximum labeled demos for MIPRO")
    parser.add_argument("--mipro-num-candidates", type=int, default=None,
                        help="Number of candidate configurations for MIPRO")
    parser.add_argument("--mipro-threads", type=int, default=None,
                        help="Number of threads for MIPRO")
    parser.add_argument("--mipro-prompt-model", type=str, default=None,
                        help="Prompt model for MIPRO")
    parser.add_argument("--mipro-task-model", type=str, default=None,
                        help="Task model for MIPRO")

    # Logging configuration
    parser.add_argument("--log-dir", type=str, default="runs",
                        help="Directory to store logs")
    parser.add_argument("--run-baseline", action="store_true",
                        help="Run baseline evaluation before optimization")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Setup logging
    runs_dir = os.path.join(os.getcwd(), args.log_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(runs_dir, exist_ok=True)
    logger = Logger(os.path.join(runs_dir, "run_log.txt"))

    print("=" * 50)
    print(f"PROMPT OPTIMIZER RUNNER ({args.optimizer.upper()})")
    print("=" * 50)
    print(f"Run directory: {runs_dir}")
    print(f"Optimizer: {args.optimizer.upper()}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Model: {args.model}")
    if args.optimizer == "gepa":
        print(f"Reflection Model: {args.reflection_model}")
    elif args.optimizer == "sspo":
        print(f"Optimizer Model: {args.optimizer_model}")
        print(f"Evaluator Model: {args.evaluator_model}")
    elif args.optimizer == "bootstrap":
        print(f"Max Demos: {args.bootstrap_max_demos}")
        print(f"Max Labeled: {args.bootstrap_max_labeled}")
        if args.teacher_model:
            print(f"Teacher Model: {args.teacher_model}")
    elif args.optimizer == "mipro":
        print(f"Auto Mode: {args.mipro_auto}")
        print(f"Max Bootstrapped: {args.mipro_max_bootstrapped}")
        print(f"Max Labeled: {args.mipro_max_labeled}")
    print(f"Dataset Mode: {args.dataset_mode}")
    print("=" * 50)

    # Configure LM
    lm = dspy.LM(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_retries=0,
        provider=None
    )
    dspy.configure(lm=lm)
    dspy.settings.configure(provide_traceback=True)
    print(f"✓ Configured language model: {lm}")

    # Load benchmark
    print(f"\nLoading {args.benchmark} benchmark...")
    benchmark = load_benchmark(args.benchmark)[0]

    dataset = benchmark.benchmark(dataset_mode=args.dataset_mode)
    program = benchmark.program[0]
    metric = benchmark.metric
    metric_with_feedback = benchmark.metric_with_feedback

    print(f"Dataset sizes: train={len(dataset.train_set)}, val={len(dataset.val_set)}, test={len(dataset.test_set)}")
    print("\nExample from train set:")
    print(dataset.train_set[0])

    # Setup evaluation
    evaluate = dspy.Evaluate(
        devset=dataset.test_set,
        metric=metric,
        num_threads=args.eval_threads,
        display_table=args.display_table,
        display_progress=True,
        max_errors=100 * len(dataset.test_set),
        provide_traceback=True,
    )

    # Baseline evaluation
    if args.run_baseline:
        print("\n" + "=" * 50)
        print("BASELINE EVALUATION")
        print("=" * 50)
        baseline_score = evaluate(program)
        print(f"✓ Baseline score: {baseline_score}")

    # Optimization
    print("\n" + "=" * 50)
    print(f"{args.optimizer.upper()} OPTIMIZATION")
    print("=" * 50)

    optimizer = get_optimizer(args.optimizer, args, benchmark, lm)

    print(f"Training with {len(dataset.train_set)} examples, validating with {len(dataset.val_set)} examples")

    optimized_program = optimizer.compile(
        program,
        trainset=dataset.train_set,
        valset=dataset.val_set,
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
    if args.run_baseline:
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

    print(f"\n✓ Run completed! Logs saved to: {runs_dir}")
