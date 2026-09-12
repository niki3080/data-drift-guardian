from drift_guardian.core.parse_config import Metric

from typing import Protocol

import pandas as pd

class MetricFn(Protocol):
    def __call__(self, 
                 reference: pd.Series, 
                 current: pd.Series, 
                 **kwargs) -> float : ...


METRIC_REGISTRY: dict[Metric, MetricFn] = {}


def register(metric: Metric):
    def wrapper(fn: MetricFn):
        METRIC_REGISTRY[metric] = fn
        return fn
    return wrapper


