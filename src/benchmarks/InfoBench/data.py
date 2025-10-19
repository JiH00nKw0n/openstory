import numpy as np
from src.benchmarks.InfoBench.dataset import InfoBenchBuilder
from src.benchmarks.benchmark import Benchmark


class InfoBench(Benchmark):
    def init_dataset(self):
        # Load full dataset
        full_dataset = InfoBenchBuilder(split="train").build_dataset()

        # Set random seed
        rng = np.random.default_rng(seed=2025)

        # Get total number of examples
        total_size = len(full_dataset)

        # Create shuffled indices
        indices = rng.permutation(total_size)

        # Split indices: 150 train, 50 dev, 300 eval
        train_indices = indices[:150]
        dev_indices = indices[150:200]
        eval_indices = indices[200:500]

        # Create splits
        self.train_set = [full_dataset[i] for i in train_indices]
        self.val_set = [full_dataset[i] for i in dev_indices]
        self.test_set = [full_dataset[i] for i in eval_indices]

