from src.benchmarks.IFBench.dataset import IFBenchBuilder
from src.benchmarks.benchmark import Benchmark


class IFBench(Benchmark):
    def init_dataset(self):
        train_dataset = IFBenchBuilder(split="train").build_dataset()
        test_dataset = IFBenchBuilder(split="test").build_dataset()

        # Split train_dataset: 60% for training, 40% for validation
        split_idx = int(len(train_dataset) * 0.6)
        self.train_set = train_dataset[:split_idx]
        self.val_set = train_dataset[split_idx:]
        self.test_set = test_dataset
