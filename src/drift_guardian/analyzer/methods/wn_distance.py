from config.parse_config import Metric
from src.drift_guardian.analyzer.regestry.registry import register
from src.drift_guardian.analyzer.schema.schema import ReferenceDict

from scipy.stats import wasserstein_distance

import pandas as pd

@register(Metric.wasserstein_distance)
def compute_wasserstein_distance(reference_dict: ReferenceDict, current: pd.Series):
    feature = current.name
    raw_ref = reference_dict['sample'][feature].dropna()

    return wasserstein_distance(raw_ref, current)