from drift_guardian.config_handler.parse_config import Metric
from drift_guardian.analyzer.regestry.metric_registry import register
from drift_guardian.schema.models import ReferenceDict

from scipy.stats import wasserstein_distance

import pandas as pd

@register(Metric.wasserstein_distance)
def compute_wasserstein_distance(reference_dict: ReferenceDict, current: pd.Series):
    feature = current.name
    assert isinstance(feature, str)
    raw_ref = reference_dict['sample'][feature].dropna()

    return wasserstein_distance(raw_ref, current.dropna())