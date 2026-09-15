from config.parse_config import Metric
from src.drift_guardian.analyzer.regestry.registry import register
from src.drift_guardian.analyzer.schema.schema import ReferenceDict
from src.drift_guardian.analyzer.utils.utils import find_ref
from src.drift_guardian.analyzer.methods.low_level.low_level import category_churn

import pandas as pd

@register(Metric.category_churn)
def compute_unseen_category_rate(reference_dict: ReferenceDict, current: pd.Series):
    feature = current.name
    assert isinstance(feature, str)

    reference = find_ref(reference_dict, feature)

    reference_freq = reference['categories']
    current_freq = current.value_counts().to_dict()

    return category_churn(reference_freq, current_freq)
