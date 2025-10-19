from typing import List

from .IFBench import benchmark as if_bench_benchmark
from .InfoBench import benchmark as info_bench_benchmark
from .benchmark import BenchmarkMeta


def load_benchmark(name:str) -> List[BenchmarkMeta]:
    if name == "IFBench":
        return if_bench_benchmark
    elif name == "InfoBench":
        return info_bench_benchmark
    else:
        raise ValueError(f"Unknown benchmark {name}")