from config.parse_config import Metric
from src.drift_guardian.analyzer.schema.schema import MetricFn

METRIC_REGISTRY: dict[Metric, MetricFn] = {}


def register(metric: Metric):
    def wrapper(fn: MetricFn):
        METRIC_REGISTRY[metric] = fn
        return fn
    return wrapper


