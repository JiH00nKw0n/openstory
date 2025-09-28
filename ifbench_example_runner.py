#!/usr/bin/env python3
"""
IFBench GEPA Example Runner
Based on gepa-artifact/example_runner.ipynb but adapted for IFBench
"""

import os
import time
import pprint
import dspy
from dotenv import load_dotenv

# Import IFBench benchmark metadata (following GEPA pattern)
from src.benchmarks.IFBench import benchmark as ifbench_metas

load_dotenv()

if __name__ == "__main__":

    lm = dspy.LM("openai/gpt-4.1-mini", temperature=1.0, max_tokens=16384, num_retries=0, provider=None)
    dspy.configure(lm=lm)
    print(f"✓ Configured language model: {lm}")

    # Load the benchmark and view one example (following GEPA pattern)
    print("Loading IFBench benchmark...")
    bench = ifbench_metas[0].benchmark(dataset_mode="lite")  # Use benchmark meta like GEPA
    print(f"Dataset sizes: train={len(bench.train_set)}, val={len(bench.val_set)}, test={len(bench.test_set)}")

    print("\nExample from train set:")
    pprint.pprint(bench.train_set[0])

    # Initialize program (following GEPA pattern)
    program = ifbench_metas[0].program[0]  # Get program from meta
    print(f"✓ Program initialized: {program}")

    # Run baseline evaluation (following GEPA pattern)
    print("\n" + "=" * 50)
    print("BASELINE EVALUATION")
    print("=" * 50)

    evaluate = dspy.Evaluate(
        devset=bench.test_set,
        metric=ifbench_metas[0].metric,
        num_threads=40,  # Use more threads like GEPA
        display_table=True,
        display_progress=True,
        max_errors=100 * len(bench.test_set)
    )

    print("Running baseline evaluation...")
    baseline_score = evaluate(program)
    print(f"✓ Baseline score: {baseline_score}")

    # Load the GEPA Optimizer (following GEPA pattern)
    print("\n" + "=" * 50)
    print("GEPA OPTIMIZER SETUP")
    print("=" * 50)

    # Import GEPA and define the optimizer
    from src.gepa.gepa import GEPA
    from src.utils.capture_stream_logger import Logger

    # Setup logging directory
    runs_dir = os.path.join(os.getcwd(), "runs", time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(runs_dir, exist_ok=True)

    gepa_logger = Logger(os.path.join(runs_dir, "run_log.txt"))

    # Setup feedback function map (using meta's feedback functions)
    if ifbench_metas[0].feedback_fn_maps is None or ifbench_metas[0].feedback_fn_maps[0] is None:
        # Fallback if no feedback functions defined
        def feedback_func(predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace):
            pred = ifbench_metas[0].metric_with_feedback(module_inputs, module_outputs, None)
            return {
                "feedback_score": pred.score,
                "feedback_text": pred.feedback,
            }
        feedback_fn_map = {k: feedback_func for k, v in program.named_predictors()}
    else:
        feedback_fn_map = ifbench_metas[0].feedback_fn_maps[0]

    optimizer = GEPA(
        named_predictor_to_feedback_fn_map=feedback_fn_map,
        knowledgebase_qe=None,
        metric=ifbench_metas[0].metric,
        run_linearized_gepa=False,
        use_merge=True,
        set_for_merge_minibatch='val',
        track_scores_on='val',
        max_metric_calls=700,
        run_dir=runs_dir,
        logger=gepa_logger,
        num_threads=40
    )
    print("✓ GEPA optimizer configured")

    # Run optimization (following GEPA pattern)
    print("\n" + "=" * 50)
    print("RUNNING OPTIMIZATION")
    print("=" * 50)

    print(f"Training with {len(bench.train_set)} examples, validating with {len(bench.val_set) // 2} examples")

    optimized_program = optimizer.compile(
        ifbench_metas[0].program[0],  # Use program from meta like GEPA
        trainset=bench.train_set,
        valset=bench.val_set[:len(bench.val_set) // 2],  # Use half of val set like GEPA
    )
    print("✓ Optimization completed")

    # Run final evaluation (following GEPA pattern)
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)

    print("Running final evaluation...")
    final_score = evaluate(optimized_program)
    print(f"✓ Final score: {final_score}")

    # Show optimized program instructions (following GEPA pattern)
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
    if final_score > baseline_score:
        improvement = ((final_score - baseline_score) / baseline_score) * 100
        print(f"Improvement: +{improvement:.1f}%")
    else:
        print("Performance change:", final_score.score - baseline_score.score)

    print("\n✓ IFBench GEPA Example Runner completed!")