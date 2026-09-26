from drift_guardian.config_handler.parse_config import Metric
from drift_guardian.analyzer.regestry.metric_registry import register
from drift_guardian.schema.models import ReferenceDict
from drift_guardian.analyzer.methods.low_level.stats import ks_d_statistic


import pandas as pd

@register(Metric.kstest)
def compute_ks_d_statistic(reference_dict: ReferenceDict, current: pd.Series):
    feature = current.name
    assert isinstance(feature, str)
    raw_ref = reference_dict['sample'][feature].dropna()

    return ks_d_statistic(raw_ref, current.dropna())