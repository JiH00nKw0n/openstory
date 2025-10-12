from typing import Optional

from dspy import Example, Prediction
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback

from .utils_ifbench import instructions_registry


def _evaluate_response(
        gold: Example,
        pred: Prediction,
) -> ScoreWithFeedback:
    """Base evaluation function that tests response for following instructions."""

    inp = gold
    response = pred.response

    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    correct_feedbacks = []
    incorrect_feedbacks = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Handle None kwargs
        if inp.kwargs[index] is None:
            inp.kwargs[index] = {}
        else:
            inp.kwargs[index] = {k: v for k, v in inp.kwargs[index].items() if v is not None}

        ins_text = instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            ins_text = instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        if not is_following:
            incorrect_feedbacks.append(ins_text)
        else:
            correct_feedbacks.append(ins_text)

        is_following_list.append(is_following)

    correct_feedback_text = ""
    if len(correct_feedbacks) > 0:
        correct_feedback_text = (
                "Your response correctly followed the following instructions:\n"
                + "\n".join(correct_feedbacks)
        )

    incorrect_feedback_text = ""
    if len(incorrect_feedbacks) > 0 and len(correct_feedbacks) > 0:
        incorrect_feedback_text = (
                "However, your response did not follow the following instructions properly:\n"
                + "\n".join(incorrect_feedbacks)
        )
    elif len(incorrect_feedbacks) > 0:
        incorrect_feedback_text = (
                "Your response did not follow the following instructions properly:\n"
                + "\n".join(incorrect_feedbacks)
        )

    feedback_text = correct_feedback_text + "\n" + incorrect_feedback_text
    feedback_text = feedback_text.strip()

    return ScoreWithFeedback(
        score=sum(is_following_list) / len(is_following_list),
        feedback=feedback_text,
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
