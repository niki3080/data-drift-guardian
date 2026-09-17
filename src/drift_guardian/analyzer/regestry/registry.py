from config.parse_config import Metric
from src.drift_guardian.analyzer.schema.schema import MetricFn


METRIC_REGISTRY: dict[Metric, MetricFn] = {}


def register(metric: Metric):
    def wrapper(fn: MetricFn):
        METRIC_REGISTRY[metric] = fn
        return fn
    return wrapper

from src.drift_guardian.analyzer.methods import (category_churn,
                                                 chi2,
                                                 cramer_v,
                                                 js_divergence,
                                                 ks_d_statistic,
                                                 missing_rate,
                                                 psi,
                                                 unseen_category_rate,
                                                 cardinality_ratio,
                                                 wn_distance) # noqa: F401 — импорт ради побочного эффекта регистрации
