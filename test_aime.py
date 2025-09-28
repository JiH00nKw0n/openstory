# Import necessary libraries for environment variables, machine learning, and data handling
import os

# Import DSPy framework for language model programming
import dspy
# Import dataset loading functionality from HuggingFace datasets
from datasets import load_dataset
# Load environment variables from .env file
from dotenv import load_dotenv

# Load environment variables to access API keys and other configuration
load_dotenv()

# Initialize the language model with GPT-4.1-mini
# Set temperature to 1 for more creative/varied responses, max_tokens for response length limit
lm = dspy.LM("openai/gpt-4.1-mini", temperature=1, api_key=os.getenv("OPENAI_API_KEY"), max_tokens=32000)
# Configure DSPy to use this language model as the default
dspy.configure(lm=lm)


class GenerateResponse(dspy.Signature):
    """
    DSPy signature class that defines the input/output structure for problem-solving tasks.
    This signature expects a mathematical problem as input and generates an answer as output.
    """
    problem = dspy.InputField()  # Input field for the mathematical problem
    answer = dspy.OutputField()  # Output field for the numerical answer


def init_aime_dataset():
    """
    Initialize and prepare training, validation, and test datasets.

    Returns:
        tuple: (train_set, val_set, test_set) containing prepared datasets
    """
    # Load the AI-MO AIME validation dataset for training
    train_split = load_dataset("AI-MO/aimo-validation-aime")['train']

    # Convert raw dataset entries into DSPy Example objects
    # Each example contains problem, solution, and answer fields
    train_split = [
        dspy.Example(
            {
                "problem": x['problem'],  # Mathematical problem text
                'solution': x['solution'],  # Step-by-step solution
                'answer': x['answer'],  # Final numerical answer
            }
        ).with_inputs("problem")  # Mark 'problem' as the input field
        for x in train_split
    ]

    # Shuffle the training data with a fixed random seed for reproducibility
    import random
    random.Random(0).shuffle(train_split)
    tot_num = len(train_split)

    # Load the AIME 2025 dataset for testing
    test_split = load_dataset("MathArena/aime_2025")['train']

    # Convert test dataset to DSPy Examples (note: no solution field available)
    test_split = [
        dspy.Example(
            {
                "problem": x['problem'],  # Mathematical problem text
                'answer': x['answer'],  # Final numerical answer
            }
        ).with_inputs("problem")  # Mark 'problem' as the input field
        for x in test_split
    ]

    # Split the training data: 50% for training, 50% for validation
    train_set = train_split[:int(0.5 * tot_num)]
    val_set = train_split[int(0.5 * tot_num):]

    test_set = test_split

    # for cost efficiency
    return train_set, val_set, test_set


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Simple evaluation metric that compares the predicted answer with the correct answer.

    Args:
        example: Ground truth example containing the correct answer
        prediction: Model prediction containing the predicted answer

    Returns:
        int: 1 if answers match, 0 otherwise
    """
    correct_answer = int(example['answer'])  # Convert correct answer to integer
    try:
        # Attempt to parse the model's prediction as an integer
        llm_answer = int(prediction.answer)
    except ValueError as e:
        # Return 0 (incorrect) if the prediction cannot be parsed as an integer
        return 0

    # Return 1 if answers match, 0 otherwise
    return int(correct_answer == llm_answer)


def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Enhanced evaluation metric that provides detailed feedback to the model.
    This metric not only scores the prediction but also generates instructive feedback.

    Args:
        example: Ground truth example containing correct answer and potentially solution
        prediction: Model prediction containing the predicted answer

    Returns:
        dspy.Prediction: Object containing score and detailed feedback text
    """
    correct_answer = int(example['answer'])  # Get the correct numerical answer
    written_solution = example.get('solution', '')  # Get step-by-step solution if available

    try:
        # Try to parse the model's answer as an integer
        llm_answer = int(prediction.answer)
    except ValueError as e:
        # Handle case where model's answer cannot be parsed as integer
        feedback_text = f"The final answer must be a valid integer and nothing else. You responded with '{prediction.answer}', which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without any additional text or formatting."
        feedback_text += f" The correct answer is '{correct_answer}'."

        # If a written solution is available, include it in the feedback
        if written_solution:
            feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems and ensure your final answer is a valid integer."

        # Return prediction object with score 0 and detailed feedback
        return dspy.Prediction(score=0, feedback=feedback_text)

    # Calculate score: 1 if correct, 0 if incorrect
    score = int(correct_answer == llm_answer)

    # Generate appropriate feedback based on whether the answer was correct
    feedback_text = ""
    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."

    # If a step-by-step solution is available, include it in the feedback
    if written_solution:
        feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems."

    # Return prediction object with calculated score and feedback
    return dspy.Prediction(score=score, feedback=feedback_text)


if __name__ == "__main__":
    # Main execution block - runs when script is executed directly

    # Initialize all datasets (training, validation, and test sets)
    train_set, val_set, test_set = init_aime_dataset()
    # Display dataset sizes for verification
    print(len(train_set), len(val_set), len(test_set))

    # Create a Chain-of-Thought program using the GenerateResponse signature
    # This enables step-by-step reasoning for mathematical problem solving
    program = dspy.ChainOfThought(GenerateResponse)

    # Set up evaluation framework for testing model performance
    evaluate = dspy.Evaluate(
        devset=test_set,  # Use test set for evaluation
        metric=metric,  # Use simple accuracy metric
        num_threads=32,  # Process evaluations in parallel for speed
        display_table=True,  # Show results in table format
        display_progress=True  # Display progress bar during evaluation
    )

    # Evaluate the vanilla (unoptimized) program performance
    vanilla_result = evaluate(program)
    print(vanilla_result)

    # Initialize GEPA (Generalized Error-based Program Adaptation) optimizer
    # This optimizer uses feedback to improve the program's performance
    optimizer = dspy.GEPA(
        metric=metric_with_feedback,  # Use metric that provides detailed feedback
        num_threads=32,  # Parallel processing for optimization
        track_stats=True,  # Keep track of optimization statistics
        reflection_minibatch_size=3,  # Process 3 examples at a time for reflection
        max_metric_calls=20, # <-- Set a budget
        reflection_lm=dspy.LM(  # Separate LM for generating reflections/feedback
            model="openai/gpt-5-mini-2025-08-07",
            temperature=1.0,
            max_tokens=32000,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    )

    # Compile the optimized program using training and validation data
    # This process uses feedback to improve the program's reasoning
    optimized_program = optimizer.compile(
        program,  # Base program to optimize
        trainset=train_set,  # Training data for optimization
        valset=val_set,  # Validation data for selecting best version
    )

    # Display the optimized program's instructions (shows learned improvements)
    print("Optimized program instructions:")
    print(optimized_program.predict.signature.instructions)

    # Evaluate the optimized program's performance
    optimized_result = evaluate(optimized_program)
    print(optimized_result)
