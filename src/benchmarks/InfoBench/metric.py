from typing import Optional, Literal

from dspy import Example, Prediction
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback
from openai import OpenAI
from pydantic import BaseModel

# Initialize OpenAI client
client = OpenAI()


class Response(BaseModel):
    output: Literal["yes", "no"]


SYS_MSG = """
Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice.
Your selection should be based on your judgment as well as the following rules:

- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question

- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.
"""


def _evaluate_response(
        gold: Example,
        pred: Prediction,
) -> ScoreWithFeedback:
    """Evaluate response using GPT-4 to check decomposed questions."""

    # Get the input prompt and generated response
    metric_text = gold.metric_prompt
    output_text = pred.response if hasattr(pred, 'response') else str(pred)
    decomposed_questions = gold.decomposed_questions

    results = []
    correct_questions = []
    incorrect_questions = []

    # Add question to conversation
    messages = [
        {"role": "system", "content": SYS_MSG},
    ]

    # Evaluate each decomposed question
    for idx, question in enumerate(decomposed_questions):
        # Add question to conversation
        messages.append(
            {"role": "user", "content": f"{metric_text}\n\nGenerated Text:\n{output_text}\n\nQuestion:\n{question}"}
        )

        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-5-mini",
                messages=messages,
                response_format=Response,
                reasoning_effort="low",
            )

            # response.parsed.output is already "yes" or "no" string
            response = completion.choices[0].message.parsed.output

            # Parse response - simplified since response_format guarantees "yes" or "no"
            answer = response.lower() == "yes"

            # Add assistant response to conversation for context
            messages.append({"role": "assistant", "content": response})

            # Record results
            if answer:
                results.append(True)
                correct_questions.append(f"✓ Q{idx + 1}: {question}")
            else:
                results.append(False)
                incorrect_questions.append(f"✗ Q{idx + 1}: {question}")

        except Exception as e:
            print(f"Error evaluating question {idx + 1}: {e}")
            results.append(False)
            incorrect_questions.append(f"✗ Q{idx + 1}: {question} (evaluation error)")

    # Generate feedback
    correct_feedback_text = ""
    if correct_questions:
        correct_feedback_text = (
                f"Correctly answered questions ({len(correct_questions)}/{len(decomposed_questions)}):\n" +
                "\n".join(correct_questions)
        )

    incorrect_feedback_text = ""
    if incorrect_questions:
        prefix = "\n\n" if correct_questions else ""
        incorrect_feedback_text = (
                f"{prefix}Incorrectly answered questions ({len(incorrect_questions)}/{len(decomposed_questions)}):\n" +
                "\n".join(incorrect_questions)
        )

    feedback_text = correct_feedback_text + incorrect_feedback_text

    # Calculate score as percentage of correct answers
    score = sum(results) / len(results) if results else 0.0

    return ScoreWithFeedback(
        score=score,
        feedback=feedback_text.strip(),
    )


def metric_with_feedback(
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[DSPyTrace] = None,
) -> ScoreWithFeedback:
    """
    Unified metric function with feedback for GEPA optimization.

    Args:
        gold: Ground truth example (module inputs)
        pred: Final prediction (module outputs)
        trace: Full execution trace of all predictors
        pred_name: Name of the specific predictor being evaluated
        pred_trace: Trace of only the specific predictor

    Returns:
        ScoreWithFeedback with 'score' and 'feedback' keys
    """

    # If pred_trace is provided, evaluate the specific predictor
    if pred_trace and len(pred_trace) > 0:
        # Extract predictor output from pred_trace
        _, _, predictor_output = pred_trace[0]

        # Generate feedback based on predictor's intermediate output
        feedback_result = _evaluate_response(gold, predictor_output)

        # Calculate score based on final module output
        score_result = _evaluate_response(gold, pred)

        return ScoreWithFeedback(
            score=score_result.score,
            feedback=feedback_result.feedback
        )

    # Default: evaluate the entire module output
    return _evaluate_response(gold, pred)


def metric(
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[DSPyTrace] = None,
) -> float:
    """Simple metric that returns only the score."""
    result = metric_with_feedback(gold, pred, trace, pred_name, pred_trace)
    return result.score
