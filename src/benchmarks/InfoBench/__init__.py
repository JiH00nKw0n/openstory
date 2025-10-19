from .data import InfoBench
from .metric import metric, metric_with_feedback
from .program import InfoBenchCoT2StageProgram
from ..benchmark import BenchmarkMeta


benchmark = [
    BenchmarkMeta(
        benchmark=InfoBench,
        program=[
            InfoBenchCoT2StageProgram(),
        ],
        metric=metric,
        metric_with_feedback=metric_with_feedback,
    )
]

__all__ = [
    "benchmark",
]