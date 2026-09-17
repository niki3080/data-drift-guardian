from config.parse_config import Metric
from src.drift_guardian.analyzer.regestry.metric_registry import register
from src.drift_guardian.analyzer.utils import make_counts, find_ref
from src.drift_guardian.schema.models import ReferenceDict
from src.drift_guardian.analyzer.methods.low_level.stats import js_divergence

import pandas as pd

@register(Metric.js_divergence)
def compute_js_divergence(reference_dict: ReferenceDict, current: pd.Series) -> float:
    feature = current.name
    assert isinstance(feature, str)
    reference = find_ref(reference_dict, feature)

    ref_counts, cur_counts = make_counts(reference, current)

    return js_divergence(ref_counts, cur_counts)
