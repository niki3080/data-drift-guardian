import logging

from config.parse_config import Metric
from src.drift_guardian.schema.models import MetricFn

logger = logging.getLogger(__name__)


METRIC_REGISTRY: dict[Metric, MetricFn] = {}


def register(metric: Metric):
    def wrapper(fn: MetricFn):
        if metric in METRIC_REGISTRY:
            logger.warning(
                "Metric '%s' is already registered with '%s', overwriting with '%s'",
                metric,
                METRIC_REGISTRY[metric].__name__,
                fn.__name__,
            )
        else:
            logger.debug("Registering metric '%s' -> '%s'", metric, fn.__name__)

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

logger.info("Metric registry initialized with %d metrics: %s",
            len(METRIC_REGISTRY),
            sorted(str(m) for m in METRIC_REGISTRY))
