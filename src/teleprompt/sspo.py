"""
SSPO (Self-Supervised Prompt Optimization) Teleprompter for DSPy

Based on: https://github.com/phureewat29/sspo

SSPO is a zero-supervision prompt optimization method that uses:
1. An optimizer LLM to generate improved prompts
2. An executor LLM to run tasks with the prompts
3. An evaluator LLM to compare prompt outputs (LLM-as-judge)
4. Iterative refinement over multiple rounds to find the best prompt
"""

import re
import random
from typing import Optional, Callable, List, Dict, Any
from collections import defaultdict

import dspy
from dspy.teleprompt.teleprompt import Teleprompter


# Prompt templates from SSPO
OPTIMIZE_PROMPT_TEMPLATE = """
You are building a prompt to address user requirement. Based on the given prompt,
please reconstruct and optimize it. You can add, modify, or delete prompts. Please include a single modification in
XML tags in your reply. During the optimization, you can incorporate any thinking models.
This is a prompt that performed excellently in a previous iteration. You must make further optimizations and improvements based on this prompt. The modified prompt must differ from the provided example.

requirements:
```
{requirements}
```

reference prompt:
```
{prompt}
```

The execution result of this reference prompt is(some cases):
```
{answers}
```

The best answer we expect(some cases):
```
{golden_answers}
```

Provide your analysis, optimization points, and the complete optimized prompt using the following XML format:

<analyse>Analyze what drawbacks exist in the results produced by the reference prompt and how to improve them.</analyse>
<modification>Summarize the key points for improvement in one sentence</modification>
<prompt>Provide the complete optimized prompt</prompt>
"""

EVALUATE_PROMPT_TEMPLATE = """
Based on the original requirements, evaluate the two responses, A and B, and determine which one better meets the requirements. If a reference answer is provided, strictly follow the format/content of the reference answer.

# Requirement
{requirement}

# A
{sample}

# B
{new_sample}

# Golden answer
{answers}

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.

<analyse>Some analysis</analyse>
<choose>A/B (the better answer in your opinion)</choose>
"""


def extract_xml_content(text: str, tag: str) -> Optional[str]:
    """Extract content from XML-like tags."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


class SSPO(Teleprompter):
    """
    Self-Supervised Prompt Optimization (SSPO) Teleprompter.

    SSPO iteratively optimizes prompts using three LLMs:
    - Optimizer LLM: Generates improved prompts based on current performance
    - Executor LLM: Executes tasks using the prompts
    - Evaluator LLM: Compares two prompt outputs to select better one (LLM-as-judge)

    Each round:
    1. Generate ONE new candidate prompt based on current best
    2. Execute both (current best vs new candidate) on eval samples
    3. Compare outputs using LLM-as-judge to pick winner
    4. Winner becomes new best for next round

    Args:
        metric: Optional metric function to evaluate program performance
        max_rounds: Maximum number of optimization rounds (default: 10)
        eval_sample_size: Number of examples to use for evaluation (default: 5)
        optimizer_lm: LM for generating improved prompts (default: uses dspy.settings.lm)
        evaluator_lm: LM for comparing prompt outputs (default: uses dspy.settings.lm)
        temperature: Temperature for optimization LM (default: 0.7)
        eval_temperature: Temperature for evaluation LM (default: 0.3)
        verbose: Whether to print optimization progress (default: True)
    """

    def __init__(
        self,
        metric: Optional[Callable] = None,
        max_rounds: int = 10,
        eval_sample_size: int = 5,
        optimizer_lm: Optional[dspy.LM] = None,
        evaluator_lm: Optional[dspy.LM] = None,
        temperature: float = 0.7,
        eval_temperature: float = 0.3,
        verbose: bool = True,
    ):
        self.metric = metric
        self.max_rounds = max_rounds
        self.eval_sample_size = eval_sample_size
        self.temperature = temperature
        self.eval_temperature = eval_temperature
        self.verbose = verbose

        # Setup LMs
        self.optimizer_lm = optimizer_lm or dspy.settings.lm
        self.evaluator_lm = evaluator_lm or dspy.settings.lm

        # Track optimization history
        self.optimization_history = []

    def compile(
        self,
        student: dspy.Module,
        trainset: List[dspy.Example],
        teacher: Optional[dspy.Module] = None,
        valset: Optional[List[dspy.Example]] = None,
        **kwargs
    ) -> dspy.Module:
        """
        Optimize the student program using SSPO.

        Args:
            student: The DSPy program to optimize
            trainset: Training examples for optimization
            teacher: Optional teacher program (unused in SSPO)
            valset: Optional validation set (if not provided, uses trainset)

        Returns:
            Optimized DSPy program with improved prompts
        """
        valset = valset or trainset
        eval_set = valset[:self.eval_sample_size]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SSPO: Self-Supervised Prompt Optimization")
            print(f"{'='*60}")
            print(f"Max rounds: {self.max_rounds}")
            print(f"Eval sample size: {len(eval_set)}")
            print(f"{'='*60}\n")

        # Get all predictors to optimize
        predictors = dict(student.named_predictors())

        # Optimize each predictor independently
        best_instructions = {}
        for pred_name, predictor in predictors.items():
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Optimizing predictor: {pred_name}")
                print(f"{'='*60}")

            best_instruction = self._optimize_predictor(
                pred_name=pred_name,
                predictor=predictor,
                student=student,
                eval_set=eval_set,
            )
            best_instructions[pred_name] = best_instruction

        # Apply best instructions to student
        optimized_student = student.deepcopy()
        for pred_name, instruction in best_instructions.items():
            pred = dict(optimized_student.named_predictors())[pred_name]
            pred.signature = pred.signature.with_instructions(instruction)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SSPO Optimization Complete")
            print(f"{'='*60}\n")

        return optimized_student

    def _optimize_predictor(
        self,
        pred_name: str,
        predictor: dspy.Predict,
        student: dspy.Module,
        eval_set: List[dspy.Example],
    ) -> str:
        """Optimize a single predictor using SSPO."""

        # Get initial instruction
        current_instruction = predictor.signature.instructions or "Complete the task."
        best_instruction = current_instruction
        best_score = self._evaluate_instruction(
            pred_name, current_instruction, student, eval_set
        )

        if self.verbose:
            print(f"Round 0 - Initial score: {best_score:.3f}")
            print(f"Initial instruction: {current_instruction[:100]}...")

        # Optimization loop - each round generates ONE candidate
        for round_num in range(1, self.max_rounds + 1):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Round {round_num}/{self.max_rounds}")
                print(f"{'='*50}")

            # Generate ONE new candidate instruction
            candidate = self._generate_optimized_instruction(
                pred_name=pred_name,
                current_instruction=best_instruction,
                student=student,
                eval_set=eval_set,
            )

            if not candidate:
                if self.verbose:
                    print("Failed to generate candidate. Stopping.")
                break

            # Compare current best vs new candidate using LLM-as-judge or metric
            is_better = self._compare_instructions(
                pred_name=pred_name,
                current_instruction=best_instruction,
                candidate_instruction=candidate,
                student=student,
                eval_set=eval_set,
            )

            if is_better:
                # Evaluate actual scores for tracking
                candidate_score = self._evaluate_instruction(
                    pred_name, candidate, student, eval_set
                )
                best_instruction = candidate
                best_score = candidate_score

                if self.verbose:
                    print(f"  ✓ New candidate is BETTER! Score: {best_score:.3f}")
            else:
                if self.verbose:
                    print(f"  ✗ Current best is still better. Score: {best_score:.3f}")

            # Track history
            self.optimization_history.append({
                'round': round_num,
                'predictor': pred_name,
                'best_score': best_score,
                'instruction': best_instruction,
            })

        if self.verbose:
            print(f"\nFinal best score: {best_score:.3f}")
            print(f"Final instruction: {best_instruction[:200]}...")

        return best_instruction

    def _generate_optimized_instruction(
        self,
        pred_name: str,
        current_instruction: str,
        student: dspy.Module,
        eval_set: List[dspy.Example],
    ) -> Optional[str]:
        """Generate an optimized instruction using the optimizer LLM."""

        # Get some execution examples with current instruction
        sample_inputs = []
        sample_outputs = []
        sample_expected = []

        num_samples = min(3, len(eval_set))
        for example in random.sample(eval_set, num_samples):
            try:
                output = student(**example.inputs())
                sample_inputs.append(str(example.inputs()))
                sample_outputs.append(str(output))
                if hasattr(example, 'labels') and example.labels():
                    sample_expected.append(str(example.labels()))
                else:
                    sample_expected.append("N/A")
            except Exception as e:
                continue

        # Format execution results
        answers = "\n\n".join([
            f"Input: {inp}\nOutput: {out}"
            for inp, out in zip(sample_inputs, sample_outputs)
        ])

        golden_answers = "\n\n".join([
            f"Expected: {exp}" for exp in sample_expected
        ])

        # Get task requirements from signature
        predictor = dict(student.named_predictors())[pred_name]
        requirements = f"Task: {pred_name}\n"
        requirements += f"Input fields: {', '.join(predictor.signature.input_fields.keys())}\n"
        requirements += f"Output fields: {', '.join(predictor.signature.output_fields.keys())}"

        # Generate optimization prompt
        optimize_prompt = OPTIMIZE_PROMPT_TEMPLATE.format(
            requirements=requirements,
            prompt=current_instruction,
            answers=answers,
            golden_answers=golden_answers,
        )

        # Call optimizer LM
        try:
            with dspy.settings.context(lm=self.optimizer_lm, temperature=self.temperature):
                response = dspy.Predict("question -> answer")(
                    question=optimize_prompt
                )

            # Extract optimized prompt from XML tags
            optimized = extract_xml_content(response.answer, "prompt")
            if optimized:
                if self.verbose:
                    print(f"  Generated new candidate")
                return optimized
            else:
                if self.verbose:
                    print(f"  Failed to extract prompt from response")
                return None

        except Exception as e:
            if self.verbose:
                print(f"  Error generating candidate: {e}")
            return None

    def _evaluate_instruction(
        self,
        pred_name: str,
        instruction: str,
        student: dspy.Module,
        eval_set: List[dspy.Example],
    ) -> float:
        """Evaluate an instruction on the eval set."""

        # Create a copy of student with the new instruction
        test_student = student.deepcopy()
        pred = dict(test_student.named_predictors())[pred_name]
        pred.signature = pred.signature.with_instructions(instruction)

        # Run on eval set
        if self.metric:
            # Use provided metric
            total_score = 0.0
            for example in eval_set:
                try:
                    output = test_student(**example.inputs())
                    score = self.metric(example, output)
                    if isinstance(score, bool):
                        score = float(score)
                    total_score += score
                except Exception:
                    pass
            return total_score / len(eval_set) if eval_set else 0.0
        else:
            # Fallback: use simple heuristic if no metric provided
            test_student = student.deepcopy()
            pred = dict(test_student.named_predictors())[pred_name]
            pred.signature = pred.signature.with_instructions(instruction)

            success_count = 0
            for example in eval_set:
                try:
                    output = test_student(**example.inputs())
                    if output and any(str(v).strip() for v in vars(output).values()):
                        success_count += 1
                except Exception:
                    pass
            return success_count / len(eval_set) if eval_set else 0.0

    def _compare_instructions(
        self,
        pred_name: str,
        current_instruction: str,
        candidate_instruction: str,
        student: dspy.Module,
        eval_set: List[dspy.Example],
    ) -> bool:
        """
        Compare two instructions by executing both and using LLM-as-judge.
        Returns True if candidate is better than current.
        """

        # Execute both instructions on eval samples
        current_outputs = []
        candidate_outputs = []

        for example in eval_set:
            # Run with current instruction
            try:
                current_student = student.deepcopy()
                pred = dict(current_student.named_predictors())[pred_name]
                pred.signature = pred.signature.with_instructions(current_instruction)
                current_out = current_student(**example.inputs())
                current_outputs.append({
                    'input': example.inputs(),
                    'output': str(current_out),
                    'expected': example.labels() if hasattr(example, 'labels') else None
                })
            except Exception as e:
                current_outputs.append({
                    'input': example.inputs(),
                    'output': f"ERROR: {e}",
                    'expected': None
                })

            # Run with candidate instruction
            try:
                candidate_student = student.deepcopy()
                pred = dict(candidate_student.named_predictors())[pred_name]
                pred.signature = pred.signature.with_instructions(candidate_instruction)
                candidate_out = candidate_student(**example.inputs())
                candidate_outputs.append({
                    'input': example.inputs(),
                    'output': str(candidate_out),
                    'expected': example.labels() if hasattr(example, 'labels') else None
                })
            except Exception as e:
                candidate_outputs.append({
                    'input': example.inputs(),
                    'output': f"ERROR: {e}",
                    'expected': None
                })

        # If metric is provided, use it for comparison
        if self.metric:
            current_score = sum(
                self.metric(example, output) if not 'ERROR' in output['output'] else 0
                for example, output in zip(eval_set, current_outputs)
            )
            candidate_score = sum(
                self.metric(example, output) if not 'ERROR' in output['output'] else 0
                for example, output in zip(eval_set, candidate_outputs)
            )
            return candidate_score > current_score

        # Otherwise use LLM-as-judge (original SSPO method)
        return self._llm_as_judge(
            pred_name=pred_name,
            current_outputs=current_outputs,
            candidate_outputs=candidate_outputs,
            student=student,
        )

    def _llm_as_judge(
        self,
        pred_name: str,
        current_outputs: List[Dict],
        candidate_outputs: List[Dict],
        student: dspy.Module,
    ) -> bool:
        """
        Use evaluator LLM to judge which output is better.
        Randomly swap A/B to prevent position bias (like original SSPO).
        """

        # Randomly swap to prevent position bias
        is_swapped = random.random() < 0.5
        if is_swapped:
            sample_a = candidate_outputs
            sample_b = current_outputs
        else:
            sample_a = current_outputs
            sample_b = candidate_outputs

        # Get task requirements
        predictor = dict(student.named_predictors())[pred_name]
        requirements = f"Task: {pred_name}\n"
        requirements += f"Input fields: {', '.join(predictor.signature.input_fields.keys())}\n"
        requirements += f"Output fields: {', '.join(predictor.signature.output_fields.keys())}"

        # Format samples and expected answers
        sample_a_str = "\n\n".join([
            f"Input: {s['input']}\nOutput: {s['output']}"
            for s in sample_a
        ])
        sample_b_str = "\n\n".join([
            f"Input: {s['input']}\nOutput: {s['output']}"
            for s in sample_b
        ])
        expected_str = "\n\n".join([
            f"Expected: {s['expected']}" if s['expected'] else "Expected: N/A"
            for s in current_outputs
        ])

        # Create evaluation prompt
        eval_prompt = EVALUATE_PROMPT_TEMPLATE.format(
            requirement=requirements,
            sample=sample_a_str,
            new_sample=sample_b_str,
            answers=expected_str,
        )

        # Call evaluator LLM
        try:
            with dspy.settings.context(lm=self.evaluator_lm, temperature=self.eval_temperature):
                response = dspy.Predict("question -> answer")(
                    question=eval_prompt
                )

            # Extract choice from XML tags
            choice = extract_xml_content(response.answer, "choose")

            if not choice:
                if self.verbose:
                    print("  Warning: Failed to extract choice, defaulting to current")
                return False

            choice = choice.strip().upper()

            # Interpret result based on swap
            if is_swapped:
                # A=candidate, B=current, so choose A means candidate is better
                return choice == "A"
            else:
                # A=current, B=candidate, so choose B means candidate is better
                return choice == "B"

        except Exception as e:
            if self.verbose:
                print(f"  Error in LLM-as-judge: {e}")
            return False