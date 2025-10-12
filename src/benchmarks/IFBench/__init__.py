from .data import IFBench
from .metric import metric, metric_with_feedback
from .program import IFBenchCoT2StageProgram
from ..benchmark import BenchmarkMeta


benchmark = [
    BenchmarkMeta(
        benchmark=IFBench,
        program=[
            IFBenchCoT2StageProgram(),
        ],
        metric=metric,
        metric_with_feedback=metric_with_feedback,
    )
]

__all__ = [
    "benchmark",
]