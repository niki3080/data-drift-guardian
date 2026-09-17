from config.parse_config import Metric
from src.drift_guardian.analyzer.regestry.metric_registry import register
from src.drift_guardian.schema.models import ReferenceDict
from src.drift_guardian.analyzer.utils import find_ref
from src.drift_guardian.analyzer.methods.low_level.stats import cardinality_ratio_abs_diff

import pandas as pd

@register(Metric.cardinality_ratio)
def compute_cardinality_ratio_abs_diff(reference_dict: ReferenceDict, current: pd.Series):
    feature = current.name
    assert isinstance(feature, str)

    reference = find_ref(reference_dict, feature)

    ref_cardinality_ratio = reference['cardinality_ratio']

    return cardinality_ratio_abs_diff(ref_cardinality_ratio, current)