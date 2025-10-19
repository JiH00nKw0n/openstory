import ast
from typing import Optional

import dspy
from datasets import load_dataset

from src.benchmarks.base import BaseBuilder


class InfoBenchBuilder(BaseBuilder):
    split: Optional[str] = 'train'
    name: Optional[str] = 'if_bench'

    def build_dataset(self) -> list:
        dataset = load_dataset(
            path="kqsong/InFoBench",
            split="train"
        )

        def _reformat_train_example(example):
            """Reformat train dataset to match test dataset format"""
            # Use ast.literal_eval instead of json.loads for Python-style strings
            return {
                "key": example["id"],
                "prompt": example["instruction"] + "\n" + example["input"] if example['input'].strip() else example[
                    "instruction"],
                "metric_prompt": "Input:\n" + example["input"] if example['input'].strip() else "",
                "category": example["category"],
                "decomposed_questions": example["decomposed_questions"],
                "subset": example["subset"],
                "question_label": example["subset"],
            }

        # Convert train format to test format using map
        dataset = dataset.map(_reformat_train_example, remove_columns=["id", "input", "category", "instruction"])

        # Convert to DSPy Examples
        dspy_examples = [
            dspy.Example(**item).with_inputs("prompt")
            for item in dataset
        ]

        return dspy_examples
