from drift_guardian.config_handler.parse_config import Metric
from drift_guardian.analyzer.regestry.metric_registry import register
from drift_guardian.schema.models import ReferenceDict

import pandas as pd

@register(Metric.missing_rate)
def compute_missing_rate(reference_dict: ReferenceDict, current: pd.Series):
    feature = current.name
    assert isinstance(feature, str)

    n_current = len(current)
    if n_current == 0:
        raise ValueError("current pd.Series is empty")

    current_missing_rate = current.isna().sum() / n_current
    return current_missing_rate
