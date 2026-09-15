from config.parse_config import Metric
from src.drift_guardian.analyzer.regestry.registry import register
from src.drift_guardian.analyzer.schema.schema import ReferenceDict
from src.drift_guardian.analyzer.methods.low_level.low_level import ks_d_statistic


import pandas as pd

@register(Metric.kstest)
def compute_ks_d_statistic(reference_dict: ReferenceDict, current: pd.Series):
    feature = current.name
    raw_ref = reference_dict['sample'][feature].dropna()

    return ks_d_statistic(raw_ref, current)