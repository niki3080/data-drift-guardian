from drift_guardian.config_handler.parse_config import Metric
from drift_guardian.schema.models import ReferenceDict
from drift_guardian.analyzer.regestry.metric_registry import register
from drift_guardian.analyzer.methods.low_level.stats import cramer_v_measure_of_association
from drift_guardian.analyzer.utils import make_counts, find_ref

import pandas as pd

@register(Metric.cramer_v)
def compute_cramer_v(reference_dict: ReferenceDict, current: pd.Series) -> float:
    feature = current.name
    assert isinstance(feature, str)

    reference = find_ref(reference_dict, feature)
    ref_counts, cur_counts = make_counts(reference, current)
    return cramer_v_measure_of_association(ref_counts, cur_counts)