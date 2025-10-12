import ast
from typing import Optional

import dspy
from datasets import load_dataset

from src.benchmarks.base import BaseBuilder
from .utils_ifbench import instructions_registry


class IFBenchBuilder(BaseBuilder):
    split: Optional[str] = 'train'
    name: Optional[str] = 'if_bench'

    def build_dataset(self) -> list:

        if self.split == 'train':
            dataset = load_dataset(
                path="allenai/IF_multi_constraints_upto5",
                split="train"
            )

            def _reformat_train_example(example):
                """Reformat train dataset to match test dataset format"""
                # Use ast.literal_eval instead of json.loads for Python-style strings
                ground_truth = ast.literal_eval(example['ground_truth'])
                return {
                    'key': example['key'],
                    'prompt': example['messages'][0]['content'],
                    'instruction_id_list': ground_truth[0]['instruction_id'],
                    'kwargs': ground_truth[0]['kwargs']
                }

            # Convert train format to test format using map
            dataset = dataset.map(_reformat_train_example, remove_columns=["dataset", "constraint_type", "constraint"])

        elif self.split == 'test':
            dataset = load_dataset(
                path="allenai/IFBench_test",
                split="train"
            )
        else:
            raise ValueError(f"Split must be 'train' or 'test', but got {self.split}")

        # Filter examples to only include those with valid instruction_ids
        def _has_valid_instructions(item):
            """Check if all instruction_ids in the example exist in INSTRUCTION_DICT"""
            instruction_id_list = item.get('instruction_id_list', [])
            return all(
                instruction_id in instructions_registry.INSTRUCTION_DICT
                for instruction_id in instruction_id_list
            )

        dataset = dataset.filter(_has_valid_instructions)

        # Convert to DSPy Examples
        dspy_examples = [
            dspy.Example(**item).with_inputs("prompt")
            for item in dataset
        ]

        return dspy_examples
