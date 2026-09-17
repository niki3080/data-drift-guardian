from config.parse_config import Metric
from src.drift_guardian.analyzer.regestry.metric_registry import register
from src.drift_guardian.schema.models import ReferenceDict
from src.drift_guardian.analyzer.utils import find_ref

import pandas as pd

@register(Metric.missing_rate)
def compute_missing_rate(reference_dict: ReferenceDict, current: pd.Series):
    feature = current.name
    assert isinstance(feature, str)

    reference = find_ref(reference_dict, feature)
    return reference['missing_rate']
