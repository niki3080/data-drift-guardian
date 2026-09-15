from config.parse_config import Metric
from src.drift_guardian.analyzer.regestry.registry import register
from src.drift_guardian.analyzer.schema.schema import ReferenceDict
from src.drift_guardian.analyzer.utils.utils import find_ref

import pandas as pd

@register(Metric.missing_rate)
def compute_missing_rate(reference_dict: ReferenceDict, current: pd.Series):
    feature = current.name
    assert isinstance(feature, str)

    reference = find_ref(reference_dict, feature)
    return reference['missing_rate']
