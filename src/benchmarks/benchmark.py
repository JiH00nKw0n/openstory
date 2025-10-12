import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Type, Optional

import dspy
from dspy import Example

dataset_size = {"full": None, "lite": 500, "tiny": 200, "test": 50}


class Benchmark(ABC):
    def __init__(self, dataset_mode="lite"):
        # dataset for training and validation
        self.dataset = None
        # dataset for the actual benchmarking
        self.train_set = None
        self.test_set = None
        self.val_set = None

        self.init_dataset()

        self.train_set: List[Example] = self.trim_dataset("train_set", 300)
        self.val_set: List[Example] = self.trim_dataset("val_set", 300)
        self.test_set: List[Example] = self.trim_dataset("test_set", None)

        assert self.train_set is not None, "Train set not initialized"
        assert self.test_set is not None, "Dev set not initialized"
        assert self.val_set is not None, "Val set not initialized"

    @abstractmethod
    def init_dataset(self) -> None:
        """
        Initializes the dataset for the benchmark, and sets it to self.dataset.
        Each element in the dataset should be an instance of dspy.Example.
        """
        raise NotImplementedError

    def trim_dataset(self, split: str, size: Optional[int] = None) -> List[Example]:

        dataset: List[Example] = getattr(self, split, None)

        if size is None or size >= len(dataset):
            return dataset

        rng = random.Random()
        rng.seed(2025)

        return rng.sample(dataset, size)

    def get_dataset(self):
        return self.dataset

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set


@dataclass
class BenchmarkMeta:
    benchmark: Type[Benchmark]
    program: List[dspy.Module]
    metric: Callable
    dataset_mode: str = "lite"
    # BenchmarkMeta.num_threads has higher priority than run time argument of num_threads
    # use this as an upper bound for the number of threads to use
    num_threads: int = None
    name: str = None
    metric_with_feedback: Callable = None
    feedback_fn_maps: list[dict] = None


@dataclass
class EvaluationResult:
    benchmark: str
    program: str

    score: float = None
    cost: float = None
    input_tokens: int = None
    output_tokens: int = None

    optimizer: str = None
    optimized_program: dspy.Module = None
    optimizer_input_tokens: int = None
    optimizer_output_tokens: int = None
    optimizer_cost: float = None

    optimizer_program_scores: list[float] = None
