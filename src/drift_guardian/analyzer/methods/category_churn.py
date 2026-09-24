from drift_guardian.config_handler.parse_config import Metric
from drift_guardian.analyzer.regestry.metric_registry import register
from drift_guardian.schema.models import ReferenceDict
from drift_guardian.analyzer.utils import find_ref
from drift_guardian.analyzer.methods.low_level.stats import category_churn

import pandas as pd

@register(Metric.category_churn)
def compute_category_churn(reference_dict: ReferenceDict, current: pd.Series):
    feature = current.name
    assert isinstance(feature, str)

    reference = find_ref(reference_dict, feature)

    reference_freq = reference['categories']
    current_freq = current.value_counts().to_dict()

    return category_churn(reference_freq, current_freq)
